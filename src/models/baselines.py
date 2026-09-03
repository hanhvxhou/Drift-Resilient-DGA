"""
src/models/baselines.py
────────────────────────
Triển khai các baseline cho Table IV của paper:

  1. Static-CNN      : train trên D01, không bao giờ update (đã có trong evaluate_cl)
  2. SW-Retrain      : huấn luyện lại từ đầu chỉ trên D_t (sliding window)
  3. EWC-only        : full fine-tune + EWC, không có replay buffer
  4. iCaRL           : exemplar-based replay + knowledge distillation
  5. GDumb           : greedy buffer fill + retrain on buffer mỗi step

Tất cả dùng cùng CharCNN backbone architecture, đánh giá theo
prequential protocol (test-then-train) trên 24 cửa sổ.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit as sigmoid_stable

from src.models.char_cnn import CharCNN, domain_to_tensor, domains_to_batch
from src.utils.common import get_logger, get_window_ids, load_config


# ── Shared Dataset ────────────────────────────────────────────────────────────
class DomainDataset(Dataset):
    def __init__(self, domains: list[str], labels: list[int]):
        self.domains = domains
        self.labels  = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):  return len(self.domains)
    def __getitem__(self, idx):
        return domain_to_tensor(self.domains[idx]), self.labels[idx]


# ── Shared evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def eval_window(model: nn.Module, df: pd.DataFrame,
                device: str, batch_size: int = 512) -> dict:
    model.eval()
    domains = df["domain"].tolist()
    labels  = np.array(df["label"].tolist())
    all_logits = []
    for i in range(0, len(domains), batch_size):
        x = domains_to_batch(domains[i:i+batch_size]).to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    probs = sigmoid_stable(logits_np)
    preds = (probs >= 0.5).astype(int)
    return {
        "f1":     f1_score(labels, preds, zero_division=0),
        "auc":    roc_auc_score(labels, probs),
        "logits": logits_np,
    }


def train_model(model: nn.Module, domains: list[str], labels: list[int],
                device: str, epochs: int = 5, lr: float = 1e-3,
                batch_size: int = 512, ewc_penalty_fn=None) -> float:
    """Train model trên data, trả về final loss."""
    ds     = DomainDataset(domains, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler("cuda") if device == "cuda" else None
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    logits = model(x)
                    loss   = criterion(logits, y)
                    if ewc_penalty_fn:
                        loss = loss + ewc_penalty_fn()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss   = criterion(logits, y)
                if ewc_penalty_fn:
                    loss = loss + ewc_penalty_fn()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            last_loss = loss.item()
    return last_loss


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 1: SW-Retrain (Sliding Window Retrain)
# ══════════════════════════════════════════════════════════════════════════════
def run_sw_retrain(cfg: dict, backbone_path: Path, device: str = "cuda") -> list[dict]:
    """
    Sliding Window Retrain: tại mỗi step t, model đã train trên D_{t-1}
    được evaluate trên D_t (prequential), rồi retrain chỉ trên D_t.
    Model chỉ giữ kiến thức từ cửa sổ gần nhất.
    """
    logger     = get_logger("baseline_sw_retrain")
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)
    epochs     = cfg.get("training", {}).get("update_epochs", 10)
    lr         = cfg.get("training", {}).get("lr", 1e-3)
    batch_size = cfg.get("training", {}).get("batch_size", 512)
    results    = []

    # Bắt đầu với pretrained backbone (đã train trên D01)
    model = CharCNN.load(backbone_path, map_location=device).to(device)

    for t, win_id in enumerate(window_ids):
        t0   = time.time()
        df_t = pd.read_csv(bench_dir / f"{win_id}.csv")

        # 1. EVALUATE trên D_t với model hiện tại (prequential)
        metrics = eval_window(model, df_t, device)

        # 2. Retrain: reset backbone rồi fine-tune CHỈ trên D_t
        if t > 0:
            model = CharCNN.load(backbone_path, map_location=device).to(device)
            train_model(model, df_t["domain"].tolist(), df_t["label"].tolist(),
                        device, epochs=epochs, lr=lr, batch_size=batch_size)

        elapsed = time.time() - t0
        q_label = df_t["quarter_label"].iloc[0]
        logger.info(f"  {win_id} ({q_label}): F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  ({elapsed:.1f}s)")

        results.append({
            "method": "SW-Retrain", "window_id": win_id,
            "quarter_label": q_label,
            "f1": metrics["f1"], "auc": metrics["auc"],
            "elapsed_s": elapsed,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 2: EWC-only (full fine-tune + EWC, no replay)
# ══════════════════════════════════════════════════════════════════════════════
class SimpleEWC:
    """EWC đơn giản trên toàn bộ model parameters."""
    def __init__(self, lam: float = 0.4):
        self.lam = lam
        self.fisher: dict[str, torch.Tensor] = {}
        self.theta_star: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module, loader: DataLoader, device: str):
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        fisher_acc = {}
        count = 0
        for x, y in loader:
            if count > 1024: break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if n not in fisher_acc:
                        fisher_acc[n] = torch.zeros_like(p.data)
                    fisher_acc[n] += p.grad.data.pow(2)
            count += len(y)
        nb = max(count / loader.batch_size, 1)
        self.fisher = {k: v / nb for k, v in fisher_acc.items()}
        self.theta_star = {n: p.data.clone() for n, p in model.named_parameters()
                          if p.requires_grad}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n].to(p.device) * (p - self.theta_star[n].to(p.device)).pow(2)).sum()
        return self.lam * loss


def run_ewc_only(cfg: dict, backbone_path: Path, device: str = "cuda") -> list[dict]:
    """
    Full fine-tune CharCNN + EWC regularization, không replay buffer.
    """
    logger     = get_logger("baseline_ewc_only")
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)
    epochs     = cfg.get("training", {}).get("update_epochs", 5)
    lr         = cfg.get("training", {}).get("lr", 5e-4)
    batch_size = cfg.get("training", {}).get("batch_size", 512)
    lam        = cfg.get("ewc", {}).get("lambda", 0.4)
    results    = []

    model = CharCNN.load(backbone_path, map_location=device).to(device)
    ewc   = SimpleEWC(lam=lam)

    for t, win_id in enumerate(window_ids):
        t0   = time.time()
        df_t = pd.read_csv(bench_dir / f"{win_id}.csv")

        # Evaluate (prequential)
        metrics = eval_window(model, df_t, device)

        # Train + EWC
        if t > 0:
            train_model(model, df_t["domain"].tolist(), df_t["label"].tolist(),
                        device, epochs=epochs, lr=lr, batch_size=batch_size,
                        ewc_penalty_fn=lambda: ewc.penalty(model))

        # Update Fisher after training
        ds = DomainDataset(df_t["domain"].tolist()[:2000], df_t["label"].tolist()[:2000])
        ewc.update(model, DataLoader(ds, batch_size=batch_size, shuffle=True), device)

        elapsed = time.time() - t0
        q_label = df_t["quarter_label"].iloc[0]
        logger.info(f"  {win_id} ({q_label}): F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  ({elapsed:.1f}s)")

        results.append({
            "method": "EWC-only", "window_id": win_id,
            "quarter_label": q_label,
            "f1": metrics["f1"], "auc": metrics["auc"],
            "elapsed_s": elapsed,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 3: iCaRL (Incremental Classifier and Representation Learning)
# ══════════════════════════════════════════════════════════════════════════════
def run_icarl(cfg: dict, backbone_path: Path, device: str = "cuda") -> list[dict]:
    """
    iCaRL: class-balanced exemplar memory + knowledge distillation.
    Simplified: binary classification → buffer cân bằng DGA/benign
    + distillation loss từ model cũ.
    """
    logger     = get_logger("baseline_icarl")
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)
    epochs     = cfg.get("training", {}).get("update_epochs", 5)
    lr         = cfg.get("training", {}).get("lr", 5e-4)
    batch_size = cfg.get("training", {}).get("batch_size", 512)
    buf_size   = cfg.get("ser", {}).get("capacity", 5000)
    results    = []

    model     = CharCNN.load(backbone_path, map_location=device).to(device)
    old_model = None

    # Exemplar buffer: class-balanced
    buf_domains: list[str] = []
    buf_labels:  list[int] = []

    for t, win_id in enumerate(window_ids):
        t0   = time.time()
        df_t = pd.read_csv(bench_dir / f"{win_id}.csv")
        domains_t = df_t["domain"].tolist()
        labels_t  = df_t["label"].tolist()

        # Evaluate (prequential)
        metrics = eval_window(model, df_t, device)

        # Train with distillation
        if t > 0:
            # Mix buffer + new data
            mix_d = domains_t + buf_domains
            mix_l = labels_t  + buf_labels
            ds    = DomainDataset(mix_d, mix_l)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            scaler    = GradScaler("cuda") if device == "cuda" else None

            model.train()
            for _ in range(epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()

                    if scaler:
                        with autocast("cuda"):
                            logits = model(x)
                            ce_loss = criterion(logits, y)
                            # Knowledge distillation
                            dist_loss = torch.tensor(0.0, device=device)
                            if old_model is not None:
                                with torch.no_grad():
                                    old_logits = old_model(x)
                                dist_loss = TF.mse_loss(logits, old_logits)
                            loss = ce_loss + 0.5 * dist_loss
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(x)
                        ce_loss = criterion(logits, y)
                        dist_loss = torch.tensor(0.0, device=device)
                        if old_model is not None:
                            with torch.no_grad():
                                old_logits = old_model(x)
                            dist_loss = TF.mse_loss(logits, old_logits)
                        loss = ce_loss + 0.5 * dist_loss
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

        # Save old model for distillation
        old_model = copy.deepcopy(model)
        old_model.eval()

        # Update exemplar buffer (class-balanced, nearest-mean)
        rng  = np.random.default_rng(cfg["random_seed"] + t)
        half = buf_size // 2
        # Sample from current window
        dga_idx    = [i for i, l in enumerate(labels_t) if l == 1]
        benign_idx = [i for i, l in enumerate(labels_t) if l == 0]
        n_dga    = min(half, len(dga_idx))
        n_benign = min(half, len(benign_idx))
        sel_dga    = rng.choice(dga_idx,    n_dga,    replace=False).tolist()
        sel_benign = rng.choice(benign_idx, n_benign, replace=False).tolist()
        sel = sel_dga + sel_benign
        buf_domains = [domains_t[i] for i in sel]
        buf_labels  = [labels_t[i]  for i in sel]

        elapsed = time.time() - t0
        q_label = df_t["quarter_label"].iloc[0]
        logger.info(f"  {win_id} ({q_label}): F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  buf={len(buf_domains)}  ({elapsed:.1f}s)")

        results.append({
            "method": "iCaRL", "window_id": win_id,
            "quarter_label": q_label,
            "f1": metrics["f1"], "auc": metrics["auc"],
            "elapsed_s": elapsed,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 4: GDumb (Greedy buffer + retrain from scratch)
# ══════════════════════════════════════════════════════════════════════════════
def run_gdumb(cfg: dict, backbone_path: Path, device: str = "cuda") -> list[dict]:
    """
    GDumb: greedily populate a balanced buffer, retrain from pretrained
    weights on buffer at each step. Evaluate BEFORE retrain (prequential).
    """
    logger     = get_logger("baseline_gdumb")
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)
    epochs     = cfg.get("training", {}).get("update_epochs", 10)
    lr         = cfg.get("training", {}).get("lr", 1e-3)
    batch_size = cfg.get("training", {}).get("batch_size", 512)
    buf_size   = cfg.get("ser", {}).get("capacity", 5000)
    results    = []

    # Model hiện tại (bắt đầu = pretrained trên D01)
    model = CharCNN.load(backbone_path, map_location=device).to(device)

    # GDumb buffer
    buf_domains: list[str] = []
    buf_labels:  list[int] = []

    for t, win_id in enumerate(window_ids):
        t0   = time.time()
        df_t = pd.read_csv(bench_dir / f"{win_id}.csv")
        domains_t = df_t["domain"].tolist()
        labels_t  = df_t["label"].tolist()

        # 1. EVALUATE trên D_t với model hiện tại (prequential)
        metrics = eval_window(model, df_t, device)

        # 2. Greedy buffer update: thêm D_t, giữ cân bằng class
        rng  = np.random.default_rng(cfg["random_seed"] + t)
        half = buf_size // 2
        all_d = buf_domains + domains_t
        all_l = buf_labels  + labels_t
        dga_idx    = [i for i, l in enumerate(all_l) if l == 1]
        benign_idx = [i for i, l in enumerate(all_l) if l == 0]
        sel_dga    = rng.choice(dga_idx,    min(half, len(dga_idx)),    replace=False).tolist()
        sel_benign = rng.choice(benign_idx, min(half, len(benign_idx)), replace=False).tolist()
        sel = sel_dga + sel_benign
        buf_domains = [all_d[i] for i in sel]
        buf_labels  = [all_l[i] for i in sel]

        # 3. Retrain from pretrained weights trên buffer
        if t > 0 and buf_domains:
            model = CharCNN.load(backbone_path, map_location=device).to(device)
            train_model(model, buf_domains, buf_labels,
                        device, epochs=epochs, lr=lr, batch_size=batch_size)

        elapsed = time.time() - t0
        q_label = df_t["quarter_label"].iloc[0]
        logger.info(f"  {win_id} ({q_label}): F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  buf={len(buf_domains)}  ({elapsed:.1f}s)")

        results.append({
            "method": "GDumb", "window_id": win_id,
            "quarter_label": q_label,
            "f1": metrics["f1"], "auc": metrics["auc"],
            "elapsed_s": elapsed,
        })

    return results
