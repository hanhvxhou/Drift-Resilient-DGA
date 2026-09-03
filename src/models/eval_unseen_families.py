"""
src/models/eval_unseen_families.py
───────────────────────────────────
Leave-K-Families-Out: danh gia kha nang tong quat hoa
voi gia dinh DGA hoan toan moi (chua tung thay khi train).

Thiet ke:
  1. Chon K families lam "unseen" (loai khoi train, giu trong test)
  2. Chay DRC-CL tren du lieu da loai
  3. Danh gia model tren:
     - Seen families (train co)  -> do stability
     - Unseen families (train khong co) -> do generalization
  4. Lap lai voi nhieu bo families khac nhau

Usage:
    python -m src.models.eval_unseen_families
    python -m src.models.eval_unseen_families --n-folds 3
"""

from __future__ import annotations

import argparse, time, copy, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from scipy.special import expit as sigmoid_stable

from src.models.char_cnn import CharCNN, domain_to_tensor, domains_to_batch
from src.models.lora_adapter import CharCNNWithLoRA
from src.models.cl_metrics import AccuracyMatrix
from src.detect.add_detector import ADDDetector, extract_embeddings
from src.utils.common import get_logger, load_config, get_window_ids
from src.utils.dga_taxonomy import WORD_BASED_FAMILIES


class DomainDataset(Dataset):
    def __init__(self, domains, labels):
        self.domains = domains
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.domains)
    def __getitem__(self, idx):
        return domain_to_tensor(self.domains[idx]), self.labels[idx]


def select_holdout_families(split_dir, window_ids, n_holdout=5, seed=42):
    """
    Chon families de hold out:
    - Xuat hien trong nhieu windows (de co du test data)
    - Uu tien families xuat hien o windows SAU (mo phong "new family arrival")
    - Mix ca char-based va word-based
    """
    rng = np.random.default_rng(seed)

    # Count families across all windows
    family_windows = {}  # family -> set of window_ids where it appears
    family_counts = {}   # family -> total sample count

    for win_id in window_ids:
        train_path = split_dir / f"{win_id}_train.csv"
        if not train_path.exists():
            continue
        df = pd.read_csv(train_path)
        dga = df[df["label"] == 1]
        for fam, count in dga["family"].value_counts().items():
            if fam == "benign":
                continue
            if fam not in family_windows:
                family_windows[fam] = set()
                family_counts[fam] = 0
            family_windows[fam].add(win_id)
            family_counts[fam] += count

    # Filter: families present in >= 6 windows and >= 1000 total samples
    candidates = [f for f in family_windows
                  if len(family_windows[f]) >= 6 and family_counts[f] >= 1000]

    # Sort by first appearance (later = more "new")
    def first_appearance(fam):
        wins = sorted(family_windows[fam])
        return window_ids.index(wins[0]) if wins[0] in window_ids else 0

    candidates.sort(key=first_appearance, reverse=True)

    # Select mix of char-based and word-based
    selected = []
    word_candidates = [f for f in candidates if f in WORD_BASED_FAMILIES]
    char_candidates = [f for f in candidates if f not in WORD_BASED_FAMILIES]

    # Take word-based first (fewer options)
    for f in word_candidates[:min(2, n_holdout)]:
        selected.append(f)

    # Fill rest with char-based
    remaining = n_holdout - len(selected)
    for f in char_candidates:
        if len(selected) >= n_holdout:
            break
        selected.append(f)

    return selected, family_windows, family_counts


def create_filtered_data(split_dir, window_ids, holdout_families):
    """
    Tao du lieu da loai holdout families khoi TRAIN, giu nguyen trong TEST.
    Returns: {win_id: {train_df_filtered, test_df_original}}
    """
    data = {}
    for win_id in window_ids:
        train_path = split_dir / f"{win_id}_train.csv"
        test_path = split_dir / f"{win_id}_test.csv"
        if not train_path.exists():
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Remove holdout families from TRAIN only
        mask_keep = ~((train_df["label"] == 1) & (train_df["family"].isin(holdout_families)))
        train_filtered = train_df[mask_keep].reset_index(drop=True)

        data[win_id] = {
            "train": train_filtered,
            "test": test_df,  # keep ALL families in test
        }

    return data


@torch.no_grad()
def eval_by_family_group(model, test_df, device, holdout_families, batch_size=512):
    """Evaluate model separately on seen vs unseen families."""
    model.eval()
    domains = test_df["domain"].tolist()
    labels = np.array(test_df["label"].tolist())

    all_logits = []
    for i in range(0, len(domains), batch_size):
        x = domains_to_batch(domains[i:i+batch_size]).to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    probs = sigmoid_stable(logits_np)
    preds = (probs >= 0.5).astype(int)

    # Overall F1
    f1_all = f1_score(labels, preds, zero_division=0)

    # Seen families: DGA not in holdout + all benign
    mask_seen = ~((test_df["label"] == 1) & (test_df["family"].isin(holdout_families)))
    if mask_seen.sum() > 0 and test_df.loc[mask_seen, "label"].nunique() >= 2:
        seen_l = labels[mask_seen.values[:len(labels)]]
        seen_p = preds[mask_seen.values[:len(preds)]]
        n = min(len(seen_l), len(seen_p))
        f1_seen = f1_score(seen_l[:n], seen_p[:n], zero_division=0)
    else:
        f1_seen = float("nan")

    # Unseen families: holdout DGA + all benign
    mask_unseen_dga = (test_df["label"] == 1) & (test_df["family"].isin(holdout_families))
    mask_unseen = mask_unseen_dga | (test_df["label"] == 0)
    if mask_unseen.sum() > 0 and mask_unseen_dga.sum() > 0:
        unseen_l = labels[mask_unseen.values[:len(labels)]]
        unseen_p = preds[mask_unseen.values[:len(preds)]]
        n = min(len(unseen_l), len(unseen_p))
        f1_unseen = f1_score(unseen_l[:n], unseen_p[:n], zero_division=0)
        n_unseen_dga = int(mask_unseen_dga.sum())
    else:
        f1_unseen = float("nan")
        n_unseen_dga = 0

    return {"f1_all": f1_all, "f1_seen": f1_seen, "f1_unseen": f1_unseen,
            "n_unseen_dga": n_unseen_dga}


def run_drc_cl_filtered(cfg, backbone_path, device, filtered_data, window_ids,
                        holdout_families, logger):
    """Run DRC-CL on filtered data, evaluate on full test (including unseen)."""
    rank = cfg.get("lora", {}).get("rank", 8)
    alpha = cfg.get("lora", {}).get("alpha", 16.0)
    epochs = cfg.get("training", {}).get("update_epochs", 5)
    lr = cfg.get("training", {}).get("lr", 5e-4)
    bs = cfg.get("training", {}).get("batch_size", 512)
    mu = cfg.get("training", {}).get("mix_ratio", 0.3)
    lam = cfg.get("ewc", {}).get("lambda", 0.4)
    seed = cfg["random_seed"]
    rng = np.random.default_rng(seed)

    # Init model
    model = CharCNNWithLoRA.from_checkpoint(
        backbone_path, rank=rank, alpha=alpha, map_location=device).to(device)

    # Simple buffer
    buf_d, buf_l = [], []
    buf_cap = cfg.get("ser", {}).get("capacity", 5000)

    # ADD
    add = ADDDetector(max_no_update=4)

    # EWC
    from src.models.cl_experiment import EWCReg
    ewc = EWCReg(lam=lam)

    per_window = []

    for t, win_id in enumerate(window_ids):
        if win_id not in filtered_data:
            continue

        train_df = filtered_data[win_id]["train"]
        test_df = filtered_data[win_id]["test"]
        train_d = train_df["domain"].tolist()
        train_l = train_df["label"].tolist()

        if t == 0:
            # Pretrain on D01 (filtered)
            ds = DomainDataset(train_d, train_l)
            loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
            opt = torch.optim.Adam(model.lora_parameters(), lr=lr)
            crit = nn.BCEWithLogitsLoss()
            scaler = GradScaler("cuda") if device == "cuda" else None
            model.train()
            for _ in range(epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    if scaler:
                        with autocast("cuda"):
                            loss = crit(model(x), y)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss = crit(model(x), y)
                        loss.backward()
                        opt.step()

            # Init ADD + EWC
            embs = extract_embeddings(model, train_d, device=device, max_n=5000)
            add.calibrate(embs)
            ds0 = DomainDataset(train_d[:1000], train_l[:1000])
            ewc.update(model, DataLoader(ds0, batch_size=bs, shuffle=True), device)

            # Init buffer
            n_add = min(buf_cap, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            buf_d = [train_d[i] for i in idx]
            buf_l = [train_l[i] for i in idx]
        else:
            # Detect drift
            embs = extract_embeddings(model, train_d, device=device, max_n=5000)
            event = add.detect(embs)

            if event.needs_update:
                # Mix buffer + new data
                n_buf = int(len(train_d) * mu / max(1-mu, 0.01))
                if buf_d and n_buf > 0:
                    b_idx = rng.choice(len(buf_d), min(n_buf, len(buf_d)), replace=False)
                    mix_d = train_d + [buf_d[i] for i in b_idx]
                    mix_l = train_l + [buf_l[i] for i in b_idx]
                else:
                    mix_d, mix_l = train_d, train_l

                ds = DomainDataset(mix_d, mix_l)
                loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
                opt = torch.optim.Adam(model.lora_parameters(), lr=lr)
                crit = nn.BCEWithLogitsLoss()
                scaler = GradScaler("cuda") if device == "cuda" else None
                model.train()
                for _ in range(epochs):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        if scaler:
                            with autocast("cuda"):
                                loss = crit(model(x), y) + ewc.penalty(model)
                            scaler.scale(loss).backward()
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss = crit(model(x), y) + ewc.penalty(model)
                            loss.backward()
                            opt.step()

                # Update EWC
                ds_f = DomainDataset(train_d[:512], train_l[:512])
                ewc.update(model, DataLoader(ds_f, batch_size=bs, shuffle=True), device)

            # Update buffer
            n_add = min(buf_cap, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            new_d = [train_d[i] for i in idx]
            new_l = [train_l[i] for i in idx]
            buf_d = (buf_d + new_d)[-buf_cap:]
            buf_l = (buf_l + new_l)[-buf_cap:]

        # Evaluate on FULL test (including unseen families)
        metrics = eval_by_family_group(model, test_df, device, holdout_families)
        per_window.append({"window_id": win_id, **metrics})

    return per_window


def run(cfg, n_folds=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("unseen_families", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    backbone_path = out_dir / "checkpoints" / "backbone_d01.pt"
    window_ids = get_window_ids(cfg)

    cfg.setdefault("lora", {"rank": 8, "alpha": 16.0})
    cfg.setdefault("ser", {"capacity": 5000, "beta": 0.92, "min_k": 50})
    cfg.setdefault("ewc", {"lambda": 0.4})
    cfg.setdefault("training", {"lr": 5e-4, "update_epochs": 5,
                                "batch_size": 512, "mix_ratio": 0.3})

    logger.info("=" * 65)
    logger.info(" LEAVE-K-FAMILIES-OUT EVALUATION")
    logger.info("=" * 65)
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Folds: {n_folds}")

    # Select holdout families
    all_holdout, fam_wins, fam_counts = select_holdout_families(
        split_dir, window_ids, n_holdout=5, seed=42
    )
    logger.info(f"\n  Candidate holdout families ({len(all_holdout)}):")
    for f in all_holdout:
        ftype = "word" if f in WORD_BASED_FAMILIES else "char"
        logger.info(f"    {f:<20s} type={ftype}  windows={len(fam_wins[f])}  samples={fam_counts[f]:,}")

    # Run multiple folds with different holdout sets
    all_fold_results = []
    rng = np.random.default_rng(42)

    for fold in range(n_folds):
        # Rotate holdout families for each fold
        fold_seed = 42 + fold * 100
        rng_fold = np.random.default_rng(fold_seed)

        holdout, _, _ = select_holdout_families(
            split_dir, window_ids, n_holdout=5, seed=fold_seed
        )

        logger.info(f"\n{'─'*65}")
        logger.info(f"  Fold {fold+1}/{n_folds}: holdout = {holdout}")
        logger.info(f"{'─'*65}")

        # Create filtered data
        filtered = create_filtered_data(split_dir, window_ids, set(holdout))

        # Log train sizes
        for win_id in window_ids[:3]:
            if win_id in filtered:
                orig = pd.read_csv(split_dir / f"{win_id}_train.csv")
                filt = filtered[win_id]["train"]
                removed = len(orig) - len(filt)
                logger.info(f"    {win_id}: {len(orig)} → {len(filt)} (removed {removed} holdout DGA)")

        # Run DRC-CL on filtered data
        t0 = time.time()
        pw = run_drc_cl_filtered(cfg, backbone_path, device, filtered, window_ids,
                                 set(holdout), logger)
        elapsed = time.time() - t0

        # Aggregate
        f1_all = np.mean([w["f1_all"] for w in pw])
        f1_seen = np.nanmean([w["f1_seen"] for w in pw])
        f1_unseen_vals = [w["f1_unseen"] for w in pw if not np.isnan(w["f1_unseen"])]
        f1_unseen = np.mean(f1_unseen_vals) if f1_unseen_vals else float("nan")

        fold_result = {
            "fold": fold + 1,
            "holdout": ", ".join(holdout),
            "f1_all": round(f1_all, 4),
            "f1_seen": round(f1_seen, 4),
            "f1_unseen": round(f1_unseen, 4) if not np.isnan(f1_unseen) else None,
            "time_s": round(elapsed, 1),
        }
        all_fold_results.append(fold_result)

        logger.info(f"\n    Fold {fold+1}: F1-All={f1_all:.4f}  F1-Seen={f1_seen:.4f}  "
                    f"F1-Unseen={f1_unseen:.4f}  ({elapsed:.1f}s)")

        # Per-window detail for last fold
        if fold == n_folds - 1:
            logger.info(f"\n    Per-window detail (Fold {fold+1}):")
            for w in pw:
                logger.info(f"      {w['window_id']}: All={w['f1_all']:.4f}  "
                            f"Seen={w['f1_seen']:.4f}  "
                            f"Unseen={w['f1_unseen']:.4f}  "
                            f"(n_unseen_dga={w['n_unseen_dga']})")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info(" LEAVE-K-FAMILIES-OUT RESULTS")
    logger.info(f"{'='*65}")
    logger.info(f"  {'Fold':<8} {'Holdout Families':<40} {'F1-All':>8} {'F1-Seen':>8} {'F1-Unseen':>10}")
    logger.info(f"  {'-'*75}")
    for r in all_fold_results:
        unseen_str = f"{r['f1_unseen']:.4f}" if r['f1_unseen'] is not None else "N/A"
        logger.info(f"  {r['fold']:<8} {r['holdout']:<40} {r['f1_all']:>8.4f} {r['f1_seen']:>8.4f} {unseen_str:>10}")
    logger.info(f"  {'-'*75}")

    # Mean across folds
    mean_all = np.mean([r["f1_all"] for r in all_fold_results])
    mean_seen = np.mean([r["f1_seen"] for r in all_fold_results])
    unseen_vals = [r["f1_unseen"] for r in all_fold_results if r["f1_unseen"] is not None]
    mean_unseen = np.mean(unseen_vals) if unseen_vals else float("nan")

    logger.info(f"  {'Mean':<8} {'':40} {mean_all:>8.4f} {mean_seen:>8.4f} {mean_unseen:>10.4f}")

    logger.info(f"\n  KEY FINDING:")
    if not np.isnan(mean_unseen):
        gap = mean_seen - mean_unseen
        logger.info(f"  F1 gap (Seen - Unseen) = {gap:.4f}")
        if gap < 0.05:
            logger.info(f"  DRC-CL generalizes well to unseen families (gap < 5pp)")
        elif gap < 0.15:
            logger.info(f"  DRC-CL shows moderate generalization gap")
        else:
            logger.info(f"  DRC-CL struggles with unseen families (gap >= 15pp)")

    # Save
    pd.DataFrame(all_fold_results).to_csv(out_dir / "unseen_families_results.csv", index=False)
    logger.info(f"\n  Saved: {out_dir / 'unseen_families_results.csv'}")
    logger.info("  Leave-K-Families-Out evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-K-Families-Out evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-folds", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, n_folds=args.n_folds)
