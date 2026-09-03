"""
src/models/lora_adapter.py
───────────────────────────
LoRA (Low-Rank Adaptation) adapter cho CharCNN backbone.

Theo Section IV-B của paper:
    - Backbone CharCNN bị ĐÓNG BĂNG hoàn toàn
    - Chỉ thêm ma trận B, A hạng thấp vào 6 lớp Linear của FC blocks
    - Forward: h = W₀x + BAx  (W₀ đóng băng, B và A trainable)
    - Khởi tạo: A ~ N(0, σ²),  B = 0  → ban đầu không ảnh hưởng output
    - Số tham số: 2 × 6 × (128×8 + 8×256) = 36,864  (4.7% full fine-tune)

Ngân hàng adapter (AdapterBank):
    - Lưu checkpoint adapter sau mỗi sudden drift event
    - Hỗ trợ khôi phục adapter gần nhất khi recurring drift
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.char_cnn import CharCNN


# ── LoRA Linear Layer ─────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    Thay thế nn.Linear với LoRA: h = W₀x + (B @ A)x
    W₀ đóng băng, B và A trainable.
    """

    def __init__(self,
                 linear: nn.Linear,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.rank         = rank
        self.alpha        = alpha
        self.scaling      = alpha / rank   # = 16/8 = 2.0

        # Đóng băng weight gốc
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = nn.Parameter(linear.bias.data.clone(),   requires_grad=False) \
                      if linear.bias is not None else None

        # LoRA matrices: A ~ N(0, 1/√r),  B = 0
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path
        result = F.linear(x, self.weight, self.bias)
        # LoRA path: x → dropout → A → B → scale
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}")


# ── CharCNN với LoRA ──────────────────────────────────────────────────────────
class CharCNNWithLoRA(nn.Module):
    """
    Wraps CharCNN backbone với LoRA adapters trên các lớp FC.

    Backbone hoàn toàn đóng băng.
    Chỉ lora_A, lora_B trong fc1, fc2, classifier được train.

    Cách dùng:
        backbone = CharCNN.load("backbone_d01.pt")
        model    = CharCNNWithLoRA(backbone, rank=8, alpha=16)
        # Chỉ optimizer trên model.lora_parameters()
    """

    def __init__(self,
                 backbone: CharCNN,
                 rank: int = 8,
                 alpha: float = 16.0,
                 lora_dropout: float = 0.0):
        super().__init__()
        self.rank  = rank
        self.alpha = alpha

        # Copy toàn bộ backbone, đóng băng hết
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Thay 3 lớp FC bằng LoRALinear (= 6 ma trận trainable: fc1, fc2, classifier)
        self.backbone.fc1        = LoRALinear(backbone.fc1,        rank, alpha, lora_dropout)
        self.backbone.fc2        = LoRALinear(backbone.fc2,        rank, alpha, lora_dropout)
        self.backbone.classifier = LoRALinear(backbone.classifier, rank, alpha, lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_embeddings(x)

    def embed(self, domains: list[str], device: str = "cpu",
              batch_size: int = 512):
        return self.backbone.embed(domains, device, batch_size)

    def predict_proba(self, domains: list[str], device: str = "cpu",
                      batch_size: int = 512):
        return self.backbone.predict_proba(domains, device, batch_size)

    # ── Tham số ──────────────────────────────────────────────────────────────
    def lora_parameters(self) -> list[nn.Parameter]:
        """Chỉ trả về tham số LoRA (A, B) — dùng cho optimizer."""
        params = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                params.extend([module.lora_A, module.lora_B])
        return params

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_only = sum(p.numel() for p in self.lora_parameters())
        return {
            "total":     total,
            "trainable": trainable,        # = lora_only
            "frozen":    total - trainable,
            "lora_only": lora_only,
        }

    # ── Lưu / tải adapter ────────────────────────────────────────────────────
    def save_adapter(self, path: str | Path) -> None:
        """Chỉ lưu trọng số LoRA (nhỏ gọn ~144 KB)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: param.data
            for name, param in self.named_parameters()
            if param.requires_grad   # chỉ lora_A, lora_B
        }
        torch.save({"lora_state": state, "rank": self.rank, "alpha": self.alpha}, path)

    def load_adapter(self, path: str | Path) -> None:
        """Nạp trọng số LoRA từ file vào model hiện tại."""
        path = Path(path)
        ckpt = torch.load(path, map_location=next(self.parameters()).device,
                          weights_only=False)
        lora_state = ckpt["lora_state"]
        current    = dict(self.named_parameters())
        for name, data in lora_state.items():
            if name in current and current[name].requires_grad:
                current[name].data.copy_(data)

    def reset_adapter(self) -> None:
        """Reset adapter về khởi tạo (B=0, A~N) — dùng khi bắt đầu episode mới."""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)

    @classmethod
    def from_checkpoint(cls,
                        backbone_path: str | Path,
                        adapter_path:  Optional[str | Path] = None,
                        rank: int = 8,
                        alpha: float = 16.0,
                        map_location: str = "cpu") -> "CharCNNWithLoRA":
        """Tải backbone + tùy chọn adapter từ file."""
        backbone = CharCNN.load(backbone_path, map_location=map_location)
        model    = cls(backbone, rank=rank, alpha=alpha)
        if adapter_path is not None:
            model.load_adapter(adapter_path)
        return model


# ── Adapter Bank ──────────────────────────────────────────────────────────────
class AdapterBank:
    """
    Ngân hàng adapter: lưu checkpoint sau mỗi sudden drift event,
    hỗ trợ khôi phục adapter gần nhất khi recurring drift.

    Cấu trúc thư mục:
        results/checkpoints/adapters/
            adapter_ep001.pt   ← episode 1 (D01→D02 sudden)
            adapter_ep002.pt
            ...
        results/checkpoints/adapters/bank_index.pt  ← metadata
    """

    def __init__(self, bank_dir: str | Path):
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.index: list[dict] = []   # [{episode, window_id, centroid, path}]
        self._load_index()

    def _index_path(self) -> Path:
        return self.bank_dir / "bank_index.pt"

    def _load_index(self) -> None:
        p = self._index_path()
        if p.exists():
            self.index = torch.load(p, map_location="cpu", weights_only=False)

    def _save_index(self) -> None:
        torch.save(self.index, self._index_path())

    def save(self, model: CharCNNWithLoRA,
             window_id: str,
             centroid:  torch.Tensor) -> Path:
        """Lưu adapter hiện tại vào bank."""
        episode   = len(self.index) + 1
        ckpt_path = self.bank_dir / f"adapter_ep{episode:03d}.pt"
        model.save_adapter(ckpt_path)
        self.index.append({
            "episode":   episode,
            "window_id": window_id,
            "centroid":  centroid.cpu(),
            "path":      str(ckpt_path),
        })
        self._save_index()
        return ckpt_path

    def find_closest(self,
                     query_centroid: torch.Tensor) -> Optional[dict]:
        """
        Tìm adapter có centroid cosine-similarity cao nhất với query.
        Trả về None nếu bank rỗng.
        """
        if not self.index:
            return None
        query = F.normalize(query_centroid.cpu().unsqueeze(0), dim=1)
        best_sim, best_entry = -1.0, None
        for entry in self.index:
            stored = F.normalize(entry["centroid"].unsqueeze(0), dim=1)
            sim    = (query @ stored.T).item()
            if sim > best_sim:
                best_sim   = sim
                best_entry = entry
        return best_entry

    def restore(self, model: CharCNNWithLoRA,
                query_centroid: torch.Tensor,
                tau: float = 0.85) -> tuple[bool, float]:
        """
        Khôi phục adapter gần nhất nếu similarity > tau.
        Trả về (restored: bool, best_similarity: float).
        """
        entry = self.find_closest(query_centroid)
        if entry is None:
            return False, 0.0
        query  = F.normalize(query_centroid.cpu().unsqueeze(0), dim=1)
        stored = F.normalize(entry["centroid"].unsqueeze(0), dim=1)
        sim    = (query @ stored.T).item()
        if sim >= tau:
            model.load_adapter(entry["path"])
            return True, sim
        return False, sim

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"AdapterBank(size={len(self)}, dir={self.bank_dir})"
