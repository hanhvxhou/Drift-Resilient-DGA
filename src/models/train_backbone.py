"""
src/models/train_backbone.py
──────────────────────────────
Huấn luyện CharCNN backbone trên D01 (cửa sổ đầu tiên).

Quy trình:
    1. Load D01.csv → train/val split (80/20, stratified)
    2. Huấn luyện với AdamW + CosineAnnealingLR + AMP FP16
    3. Early stopping theo val_loss (patience=5)
    4. Lưu best checkpoint → results/checkpoints/backbone_d01.pt
    5. In classification report trên val set

Sau khi train xong, backbone bị ĐÓNG BĂNG và chỉ
LoRA adapter mới được update trong DRC-CL.

Usage:
    python -m src.models.train_backbone
    python -m src.models.train_backbone --window D01 --epochs 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit as sigmoid_stable
from sklearn.metrics import (classification_report, roc_auc_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.models.char_cnn import CharCNN, domains_to_batch, MAX_LEN
from src.utils.common import get_logger, load_config


# ── Dataset ───────────────────────────────────────────────────────────────────
class DomainDataset(Dataset):
    def __init__(self, domains: list[str], labels: list[int], max_len: int = MAX_LEN):
        self.domains = domains
        self.labels  = torch.tensor(labels, dtype=torch.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        from src.models.char_cnn import domain_to_tensor
        x = domain_to_tensor(self.domains[idx], self.max_len)
        y = self.labels[idx]
        return x, y


# ── Training helpers ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(x)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast("cuda"):
            logits = model(x)
            loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = sigmoid_stable(logits)        # numerically stable sigmoid
    preds  = (probs >= 0.5).astype(int)

    return {
        "loss":     total_loss / len(loader.dataset),
        "f1":       f1_score(labels, preds, zero_division=0),
        "auc":      roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "probs":    probs,
        "labels":   labels,
        "preds":    preds,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run(cfg: dict,
        window_id: str = "D01",
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 5,
        val_ratio: float = 0.2) -> Path:

    log_dir  = Path(cfg["paths"]["results"]) / "logs"
    ckpt_dir = Path(cfg["paths"]["results"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger   = get_logger("train_backbone", log_dir=log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = Path(cfg["paths"]["benchmark_dir"]) / f"{window_id}.csv"
    logger.info(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    logger.info(f"  Tổng: {len(df):,}  |  DGA: {df.label.sum():,}  |  Benign: {(df.label==0).sum():,}")

    # Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        df["domain"].tolist(), df["label"].tolist(),
        test_size=val_ratio, random_state=cfg["random_seed"], stratify=df["label"]
    )
    logger.info(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    train_ds = DomainDataset(X_train, y_train)
    val_ds   = DomainDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=(device == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=(device == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CharCNN().to(device)
    params = model.count_parameters()
    logger.info(f"Model: {params['total']:,} tham số")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler    = GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"\n{'─'*55}")
    logger.info(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} {'ValF1':>8} {'ValAUC':>8} {'LR':>10}")
    logger.info('─'*55)

    best_val_loss = float("inf")
    best_ckpt     = ckpt_dir / f"backbone_{window_id.lower()}.pt"
    patience_cnt  = 0

    for epoch in range(1, epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_f1   = val_metrics["f1"]
        val_auc  = val_metrics["auc"]
        cur_lr   = scheduler.get_last_lr()[0]
        elapsed  = time.time() - t0

        logger.info(
            f"{epoch:>6} {train_loss:>10.4f} {val_loss:>10.4f} "
            f"{val_f1:>8.4f} {val_auc:>8.4f} {cur_lr:>10.2e}  ({elapsed:.1f}s)"
        )

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            model.save(best_ckpt)
            logger.info(f"         ✓ Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"\nEarly stopping tại epoch {epoch} (patience={patience})")
                break

    # ── Final evaluation ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info("Đánh giá cuối trên validation set (best checkpoint):")
    logger.info('='*55)
    best_model = CharCNN.load(best_ckpt, map_location=device)
    best_model = best_model.to(device)
    final = evaluate(best_model, val_dl, criterion, device)

    logger.info(f"  Val Loss : {final['loss']:.4f}")
    logger.info(f"  Val F1   : {final['f1']:.4f}")
    logger.info(f"  Val AUC  : {final['auc']:.4f}")
    logger.info(f"  Val Acc  : {final['accuracy']:.4f}")
    logger.info("\nClassification Report:")
    report = classification_report(
        final["labels"], final["preds"],
        target_names=["Benign", "DGA"], digits=4
    )
    for line in report.splitlines():
        logger.info("  " + line)

    logger.info(f"\nCheckpoint → {best_ckpt}")
    logger.info("Backbone training complete ✓")
    logger.info("Tiếp theo: chạy step4 với --backbone để tính drift labels thực tế")
    logger.info(f"  python -m src.data.step4_annotate_drift --backbone {best_ckpt}")

    return best_ckpt


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện CharCNN backbone")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--window",     default="D01",
                        help="Window ID dùng để train (mặc định: D01)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg,
        window_id  = args.window,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        patience   = args.patience)
