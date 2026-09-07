"""
src/models/run_joint_offline.py
───────────────────────────────
E4 — Joint / offline all-data upper bound (Reviewer #4, Concern #1).

SW-Retrain is only a recency comparator (resets and fine-tunes on the current
window). Reviewer #4 asked for a true all-data ceiling: a model trained on ALL
windows at once, to show how far DRC-CL sits from the ideal.

Design (confirmed):
  - Definition: OFFLINE all-data (ĐN-B). Train ONE CharCNN per seed on the union
    of D01..D24 train splits, then evaluate it on every window's test set and
    average -> "AA-F1 if all data were available from the start". This is the
    correct ceiling to compare against DRC-CL's AA-F1 (mean of the diagonal).
    We do NOT retrain per step (that answers a question the reviewer did not ask
    and is far more expensive).
  - Backbone: CharCNN only (DistilBERT joint on 1.9M samples is prohibitive;
    DistilBERT+FT at 100% params already serves as its practical upper ref).
  - Seeds: 10 (for consistency with the CharCNN experiments).
  - Early stopping on a held-out validation split (patience=5).

Reuses CharCNN / DomainDataset / train_one_epoch / evaluate / build_accuracy_row
from the existing code — no new model or metric logic.

Usage
-----
    python -m src.models.run_joint_offline \
        --seeds 42 123 456 789 2024 3141 5926 5358 9793 2384

Outputs (results/multi_seed/)
    joint_offline_raw.csv   — one row per seed (AA-F1 ceiling + per-metric)
    joint_offline_agg.csv   — mean +/- std (ddof=1) + gap vs DRC-CL
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.common import get_logger, load_config, get_window_ids
from src.models.char_cnn import CharCNN
from src.models.train_backbone import DomainDataset, train_one_epoch, evaluate
from src.models.cl_metrics import build_accuracy_row

DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 3141, 5926, 5358, 9793, 2384]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_joint(cfg, seed, split_dir, window_ids, device, logger,
                epochs=30, patience=5, batch_size=512, lr=1e-3, val_ratio=0.1):
    """Train ONE CharCNN on the union of all train splits, with early stopping."""
    # Load & concatenate all training splits
    dfs = [pd.read_csv(split_dir / f"{w}_train.csv") for w in window_ids]
    allX = pd.concat(dfs, ignore_index=True)
    logger.info(f"    Joint training set: {len(allX):,} samples "
                f"(union of {len(window_ids)} windows)")

    Xtr, Xval, ytr, yval = train_test_split(
        allX["domain"].tolist(), allX["label"].tolist(),
        test_size=val_ratio, random_state=seed, stratify=allX["label"].tolist())

    tr_dl = DataLoader(DomainDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=(device == "cuda"))
    va_dl = DataLoader(DomainDataset(Xval, yval), batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=(device == "cuda"))

    model = CharCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val = float("inf"); best_state = None; pc = 0
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, tr_dl, opt, crit, scaler, device)
        vm = evaluate(model, va_dl, crit, device)
        sched.step()
        logger.info(f"      epoch {ep:2d}: train_loss={tr_loss:.4f}  "
                    f"val_loss={vm['loss']:.4f}  val_f1={vm.get('f1', float('nan')):.4f}")
        if vm["loss"] < best_val:
            best_val = vm["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                logger.info(f"      early stop at epoch {ep} (patience {patience})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run(cfg, seeds, device="cuda", resume=False, epochs=30, patience=5):
    out_dir = Path(cfg["paths"]["results"]); ms = out_dir / "multi_seed"
    ms.mkdir(parents=True, exist_ok=True)
    logger = get_logger("joint_offline", log_dir=out_dir / "logs")
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    window_ids = get_window_ids(cfg)

    raw_path = ms / "joint_offline_raw.csv"
    rows = []
    done = set()
    if resume and raw_path.exists():
        prev = pd.read_csv(raw_path); rows = prev.to_dict("records")
        done = set(int(r["seed"]) for r in rows)
        logger.info(f"  Resume: {len(done)} seeds present")

    logger.info("=" * 70)
    logger.info(f" E4 — JOINT / OFFLINE ALL-DATA CEILING (CharCNN, {len(seeds)} seeds)")
    logger.info("=" * 70)

    for seed in seeds:
        if seed in done:
            logger.info(f"  seed {seed}: cached, skip"); continue
        logger.info(f"\n  -- seed {seed} --")
        set_all_seeds(seed)
        model = train_joint(cfg, seed, split_dir, window_ids, device, logger,
                            epochs=epochs, patience=patience)

        # Evaluate the single all-data model on EVERY window's test set.
        # up_to_t = last index => build_accuracy_row scores all 24 test sets.
        model.eval()
        row_scores = build_accuracy_row(model, split_dir, window_ids,
                                        up_to_t=len(window_ids) - 1, device=device)
        f1s = [row_scores[w]["f1"] for w in window_ids if w in row_scores]
        aucs = [row_scores[w].get("auc", np.nan) for w in window_ids if w in row_scores]
        aa_f1 = float(np.mean(f1s))
        rows.append({"seed": seed, "aa_f1": round(aa_f1, 4),
                     "aa_auc": round(float(np.nanmean(aucs)), 4),
                     "f1_first": round(f1s[0], 4), "f1_last": round(f1s[-1], 4)})
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        logger.info(f"    Joint ceiling AA-F1 (seed {seed}) = {aa_f1:.4f}")

    # aggregate + gap vs DRC-CL
    raw = pd.DataFrame(rows)
    m = float(raw["aa_f1"].mean()); s = float(raw["aa_f1"].std(ddof=1)) if len(raw) > 1 else 0.0
    DRCCL_AA = 0.9457  # DRC-CL(CharCNN) AA-F1 from run_multi_seed
    agg = pd.DataFrame([{
        "config": "Joint/Offline (all-data ceiling)",
        "n_seeds": len(raw),
        "aa_f1_mean": round(m, 4), "aa_f1_std": round(s, 4),
        "drc_cl_aa_f1": DRCCL_AA,
        "gap_drc_cl_to_ceiling": round(DRCCL_AA - m, 4),
    }])
    agg.to_csv(ms / "joint_offline_agg.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info(f"  Joint/Offline ceiling AA-F1 = {m:.4f} +/- {s:.4f} (ddof=1, n={len(raw)})")
    logger.info(f"  DRC-CL (CharCNN) AA-F1      = {DRCCL_AA:.4f}")
    logger.info(f"  Gap (DRC-CL - ceiling)     = {DRCCL_AA - m:+.4f}")
    logger.info("  Interpretation: a small negative gap means DRC-CL, updating only")
    logger.info("  2.8% of parameters online, is close to the all-data offline ceiling.")
    logger.info(f"\n  Raw -> {raw_path}")
    logger.info(f"  Agg -> {ms / 'joint_offline_agg.csv'}")
    logger.info("  E4 complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E4: joint/offline all-data ceiling")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(load_config(args.config), seeds=args.seeds, device=args.device,
        resume=args.resume, epochs=args.epochs, patience=args.patience)
