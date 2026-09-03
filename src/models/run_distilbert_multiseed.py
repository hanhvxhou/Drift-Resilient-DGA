"""
src/models/run_distilbert_multiseed.py
───────────────────────────────────────
Multi-seed cho DRC-CL (DistilBERT).
Chay 5 seeds, bao cao mean +/- std.

Usage:
    python -m src.models.run_distilbert_multiseed
    python -m src.models.run_distilbert_multiseed --seeds 42 123 456 789 2024
"""

from __future__ import annotations
import argparse, time, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from src.utils.common import get_logger, load_config, get_window_ids


DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def run_one_seed(cfg, seed, device, logger, split_dir, window_ids):
    """Run DRC-CL DistilBERT for one seed. Returns metrics dict."""
    from transformers import DistilBertTokenizer
    from src.models.drc_cl_distilbert import (
        DistilBERTWithLoRA, TokenDataset, SERBuffer, EWCReg,
        SimpleDriftDetector, eval_on_test, build_row,
        extract_cls_embeddings
    )
    from src.models.cl_metrics import AccuracyMatrix
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from torch.amp import GradScaler, autocast

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    T = len(window_ids)
    epochs = 3
    lr = 2e-5
    bs = 64
    mu = cfg.get("training", {}).get("mix_ratio", 0.3)
    lam = cfg.get("ewc", {}).get("lambda", 0.4)
    buf_cap = cfg.get("ser", {}).get("capacity", 5000)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERTWithLoRA(lora_r=8, lora_alpha=16).to(device)
    ser = SERBuffer(capacity=buf_cap, seed=seed)
    ewc = EWCReg(lam=lam)
    add = SimpleDriftDetector()
    matrix = AccuracyMatrix(window_ids)

    for t in range(T):
        win_id = window_ids[t]
        train_df = pd.read_csv(split_dir / f"{win_id}_train.csv")
        train_d = train_df["domain"].tolist()
        train_l = train_df["label"].tolist()

        if t == 0:
            # Pretrain
            ds = TokenDataset(train_d, train_l, tokenizer)
            loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
            opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.01)
            crit = nn.BCEWithLogitsLoss()
            scaler = GradScaler("cuda") if device == "cuda" else None
            model.train()
            for _ in range(epochs):
                for ids, mask, y in loader:
                    ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                    opt.zero_grad()
                    if scaler:
                        with autocast("cuda"):
                            loss = crit(model(ids, mask), y)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss = crit(model(ids, mask), y)
                        loss.backward()
                        opt.step()
            # Init components
            embs = extract_cls_embeddings(model, train_d, tokenizer, device, bs)
            add.calibrate(embs)
            ds0 = TokenDataset(train_d[:1000], train_l[:1000], tokenizer)
            ewc.update(model, DataLoader(ds0, batch_size=bs, shuffle=True), device)
            n_add = min(buf_cap, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            ser.add_batch([train_d[i] for i in idx], [train_l[i] for i in idx])
        else:
            embs = extract_cls_embeddings(model, train_d, tokenizer, device, bs)
            drift, should_update = add.detect(embs)
            if should_update:
                n_buf = int(len(train_d) * mu / max(1-mu, 0.01))
                b_d, b_l = ser.sample(n_buf)
                mix_d, mix_l = train_d + b_d, train_l + b_l
                ds = TokenDataset(mix_d, mix_l, tokenizer)
                loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
                opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
                crit = nn.BCEWithLogitsLoss()
                scaler = GradScaler("cuda") if device == "cuda" else None
                model.train()
                for _ in range(epochs):
                    for ids, mask, y in loader:
                        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                        opt.zero_grad()
                        if scaler:
                            with autocast("cuda"):
                                loss = crit(model(ids, mask), y) + ewc.penalty(model)
                            scaler.scale(loss).backward()
                            scaler.unscale_(opt)
                            nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss = crit(model(ids, mask), y) + ewc.penalty(model)
                            loss.backward()
                            opt.step()
                ds_f = TokenDataset(train_d[:512], train_l[:512], tokenizer)
                ewc.update(model, DataLoader(ds_f, batch_size=bs, shuffle=True), device)
            n_add = min(buf_cap, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            ser.add_batch([train_d[i] for i in idx], [train_l[i] for i in idx])

        row = build_row(model, tokenizer, split_dir, window_ids, t, device, bs)
        matrix.add_row(t, row)
        f1_t = row.get(win_id, {}).get("f1", 0)
        logger.info(f"      {win_id}: F1={f1_t:.4f}")

    metrics = matrix.compute_metrics()
    return metrics


def run(cfg, seeds=None):
    if seeds is None:
        seeds = DEFAULT_SEEDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("distilbert_multiseed", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    window_ids = get_window_ids(cfg)

    cfg.setdefault("ser", {"capacity": 5000})
    cfg.setdefault("ewc", {"lambda": 0.4})
    cfg.setdefault("training", {"mix_ratio": 0.3})

    logger.info("=" * 60)
    logger.info(f" DRC-CL (DistilBERT) MULTI-SEED — {len(seeds)} seeds")
    logger.info("=" * 60)

    all_metrics = []
    for i, seed in enumerate(seeds):
        logger.info(f"\n  Seed {seed} ({i+1}/{len(seeds)})")
        t0 = time.time()
        m = run_one_seed(cfg, seed, device, logger, split_dir, window_ids)
        elapsed = time.time() - t0
        m["seed"] = seed
        m["time_s"] = round(elapsed, 1)
        all_metrics.append(m)
        logger.info(f"    AA-F1={m['aa_f1']:.4f}  Forg={m['forgetting']:.4f}  ({elapsed:.1f}s)")

    # Aggregate
    keys = ["aa_f1", "bwt", "forgetting", "degrad", "aa_f1_char", "aa_f1_word"]
    logger.info(f"\n{'='*60}")
    logger.info(f" DRC-CL (DistilBERT) — {len(seeds)} seeds")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Metric':<16} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    logger.info(f"  {'-'*56}")

    agg = {"method": "DRC-CL (DistilBERT)"}
    for k in keys:
        vals = [m.get(k, float("nan")) for m in all_metrics]
        vals = [v for v in vals if v == v]  # remove nan
        if vals:
            agg[f"{k}_mean"] = round(np.mean(vals), 4)
            agg[f"{k}_std"] = round(np.std(vals), 4)
            logger.info(f"  {k:<16} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} {min(vals):>10.4f} {max(vals):>10.4f}")

    # Save
    pd.DataFrame(all_metrics).to_csv(out_dir / "distilbert_multiseed_raw.csv", index=False)
    with open(out_dir / "distilbert_multiseed_agg.json", "w") as f:
        json.dump(agg, f, indent=2)
    logger.info(f"\n  Raw    → {out_dir / 'distilbert_multiseed_raw.csv'}")
    logger.info(f"  Agg    → {out_dir / 'distilbert_multiseed_agg.json'}")
    logger.info("  Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    run(load_config(args.config), seeds=args.seeds)
