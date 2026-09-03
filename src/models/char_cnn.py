"""
src/models/char_cnn.py
───────────────────────
Kiến trúc CharCNN backbone cho DGA detection.

Kiến trúc (Section IV-B của paper):
    Input: domain string → one-hot encoded, padded/truncated to MAX_LEN=64
    3 nhánh conv song song: kernel {3,4,5} × 256 filters
    → ReLU → global max-pool
    → concat → 768-dim
    → FC(768→256, ReLU, Dropout 0.3)
    → FC(256→128, ReLU, Dropout 0.3)
    → embedding z ∈ ℝ¹²⁸  (dùng cho ADD/MMD)
    → classifier FC(128→1)  (dùng cho binary prediction)

Backbone được ĐÓNG BĂNG sau khi pretrain trên D01.
Chỉ LoRA adapter mới được cập nhật trong DRC-CL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── Vocabulary ────────────────────────────────────────────────────────────────
VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789-.")  # 38 ký tự
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}     # 1-based, 0 = padding
VOCAB_SIZE = len(VOCAB) + 1  # +1 cho padding
MAX_LEN = 64                 # độ dài domain (ký tự), pad/truncate


def domain_to_tensor(domain: str, max_len: int = MAX_LEN) -> torch.Tensor:
    """Chuyển chuỗi domain thành tensor one-hot index, shape (max_len,)."""
    domain = domain.lower().strip()
    indices = [CHAR2IDX.get(c, 0) for c in domain[:max_len]]
    # Padding bên phải
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def domains_to_batch(domains: list[str],
                     max_len: int = MAX_LEN,
                     device: str = "cpu") -> torch.Tensor:
    """Chuyển list domains thành batch tensor, shape (N, max_len)."""
    tensors = [domain_to_tensor(d, max_len) for d in domains]
    return torch.stack(tensors).to(device)


# ── Model ─────────────────────────────────────────────────────────────────────
class CharCNN(nn.Module):
    """
    Character-level CNN backbone.
    forward() trả về logit (scalar) cho binary classification.
    embed() trả về embedding z ∈ ℝ¹²⁸ để dùng cho MMD/ADD.
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 64,
                 num_filters: int = 256,
                 kernel_sizes: tuple[int, ...] = (3, 4, 5),
                 fc_dims: tuple[int, int] = (256, 128),
                 dropout: float = 0.3,
                 max_len: int = MAX_LEN):
        super().__init__()
        self.max_len = max_len

        # Character embedding (học được, nhẹ hơn one-hot)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 3 nhánh conv song song
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        # Fully connected layers
        conv_out_dim = num_filters * len(kernel_sizes)  # 256 × 3 = 768
        self.fc1 = nn.Linear(conv_out_dim, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Binary classifier head
        self.classifier = nn.Linear(fc_dims[1], 1)

        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            nn.init.zeros_(conv.bias)
        for fc in [self.fc1, self.fc2, self.classifier]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, max_len) LongTensor
        trả về: (batch, 128) embedding
        """
        # Embedding: (batch, max_len, embed_dim)
        emb = self.embedding(x)
        # Conv cần (batch, embed_dim, seq_len)
        emb = emb.permute(0, 2, 1)

        # Mỗi nhánh: conv → relu → global max pool
        pooled = []
        for conv in self.convs:
            c = self.relu(conv(emb))           # (batch, num_filters, L)
            c = c.max(dim=2).values            # (batch, num_filters)
            pooled.append(c)

        # Concat 3 nhánh → (batch, 768)
        out = torch.cat(pooled, dim=1)

        # FC layers
        out = self.dropout(self.relu(self.fc1(out)))   # (batch, 256)
        out = self.dropout(self.relu(self.fc2(out)))   # (batch, 128)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, max_len) LongTensor
        trả về: (batch,) logits cho binary classification
        """
        z = self._extract_features(x)
        return self.classifier(z).squeeze(1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trả về embedding z ∈ ℝ¹²⁸ (dùng cho MMD/ADD).
        x: (batch, max_len) LongTensor
        trả về: (batch, 128) FloatTensor
        """
        return self._extract_features(x)

    @torch.no_grad()
    def embed(self, domains: list[str],
              device: str = "cpu",
              batch_size: int = 512) -> np.ndarray:
        """
        Tiện ích: nhận list domain strings, trả về numpy array (N, 128).
        Dùng trong step4_annotate_drift để tính MMD².
        """
        self.eval()
        # Đảm bảo model và input cùng device
        model_device = next(self.parameters()).device
        all_embs = []
        for i in range(0, len(domains), batch_size):
            batch = domains_to_batch(domains[i:i + batch_size]).to(model_device)
            emb = self.get_embeddings(batch)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    @torch.no_grad()
    def predict_proba(self, domains: list[str],
                      device: str = "cpu",
                      batch_size: int = 512) -> np.ndarray:
        """
        Trả về xác suất DGA cho mỗi domain, numpy array (N,).
        """
        self.eval()
        model_device = next(self.parameters()).device
        all_probs = []
        for i in range(0, len(domains), batch_size):
            batch  = domains_to_batch(domains[i:i + batch_size]).to(model_device)
            logits = self.forward(batch)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs)

    # ── Lưu / tải checkpoint ──────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Lưu toàn bộ model state + config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "vocab_size":   self.embedding.num_embeddings,
                "embed_dim":    self.embedding.embedding_dim,
                "num_filters":  self.convs[0].out_channels,
                "kernel_sizes": tuple(c.kernel_size[0] for c in self.convs),
                "fc_dims":      (self.fc1.out_features, self.fc2.out_features),
                "dropout":      self.dropout.p,
                "max_len":      self.max_len,
            }
        }, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "CharCNN":
        """Tải model từ checkpoint."""
        ckpt   = torch.load(path, map_location=map_location, weights_only=False)
        model  = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def freeze_backbone(self) -> None:
        """Đóng băng toàn bộ tham số — gọi sau khi pretrain xong."""
        for param in self.parameters():
            param.requires_grad = False

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
