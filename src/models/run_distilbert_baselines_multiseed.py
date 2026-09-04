"""
src/models/run_distilbert_baselines_multiseed.py
────────────────────────────────────────────────
E1 — Multi-seed runner for the two DistilBERT BASELINES:
       • DistilBERT (Static)      — train on D01, freeze, evaluate 24 windows
       • DistilBERT + Fine-tune   — full fine-tune each window

Why this script
---------------
Reviewers #1, #2, and #4 noted that these two baselines were reported with a
SINGLE seed while everything else uses multiple seeds. `distilbert_baseline.run()`
does not set a seed and overwrites results/distilbert_results.csv on each call.
This wrapper:
  1. sets ALL RNG seeds deterministically before each run,
  2. calls the existing distilbert_baseline.run() unchanged,
  3. reads back the distilbert_results.csv it writes,
  4. tags each row with its seed and accumulates,
  5. writes per-seed raw + aggregated mean±std (ddof=1) files.

It reuses the existing, reviewed baseline code — no metric logic is duplicated.

Usage
-----
    # default 5 seeds (matches DistilBERT DRC-CL budget)
    python -m src.models.run_distilbert_baselines_multiseed

    # explicit seeds / resume
    python -m src.models.run_distilbert_baselines_multiseed --seeds 42 123 456 789 2024
    python -m src.models.run_distilbert_baselines_multiseed --resume

Outputs (under results/multi_seed/)
    distilbert_baselines_raw.csv   — one row per (method, seed)
    distilbert_baselines_agg.csv   — mean ± std (ddof=1) per method
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
METRIC_COLS = ["aa_f1", "bwt", "forgetting", "degrad"]


def set_all_seeds(seed: int) -> None:
    """Make a run reproducible: python, numpy, torch (CPU+CUDA), cudnn."""
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic cudnn (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg: dict, seeds: list[int], resume: bool = False,
        epochs: int = 3, lr: float = 2e-5, batch_size: int = 64) -> None:
    from src.models import distilbert_baseline

    out_dir = Path(cfg["paths"]["results"])
    ms_dir = out_dir / "multi_seed"
    ms_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("distilbert_bl_multiseed", log_dir=out_dir / "logs")

    raw_path = ms_dir / "distilbert_baselines_raw.csv"
    agg_path = ms_dir / "distilbert_baselines_agg.csv"

    # resume: load rows already computed
    done = {}
    rows: list[dict] = []
    if resume and raw_path.exists():
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        for r in rows:
            done.setdefault(int(r["seed"]), set()).add(r["method"])
        logger.info(f"  Resume: {len(rows)} rows already present")

    logger.info("=" * 68)
    logger.info(f" E1 — DistilBERT BASELINES, multi-seed ({len(seeds)} seeds)")
    logger.info(f"  Seeds: {seeds}")
    logger.info("=" * 68)

    tmp_csv = out_dir / "distilbert_results.csv"  # what run() writes

    for seed in seeds:
        if resume and done.get(seed, set()) >= {"DistilBERT (Static)", "DistilBERT + Fine-tune"}:
            logger.info(f"  seed {seed}: already done, skipping")
            continue

        logger.info(f"\n  ── seed {seed} ──")
        set_all_seeds(seed)

        # run BOTH baselines for this seed (only=None → static + finetune)
        if tmp_csv.exists():
            tmp_csv.unlink()  # ensure we read fresh output
        distilbert_baseline.run(cfg, only=None, epochs=epochs, lr=lr,
                                batch_size=batch_size)

        if not tmp_csv.exists():
            logger.error(f"  seed {seed}: distilbert_results.csv not produced — aborting")
            raise RuntimeError("distilbert_baseline.run did not write results")

        res = pd.read_csv(tmp_csv)
        for _, r in res.iterrows():
            row = {"method": r["method"], "seed": seed}
            for c in METRIC_COLS:
                row[c] = r[c] if c in res.columns else np.nan
            rows.append(row)
            logger.info(f"    {r['method']:<24} AA-F1={r.get('aa_f1', float('nan')):.4f}  "
                        f"Forg={r.get('forgetting', float('nan')):+.4f}")

        # write raw after every seed (crash-safe)
        pd.DataFrame(rows).to_csv(raw_path, index=False)

    # ── aggregate: mean ± std (ddof=1) ──
    raw = pd.DataFrame(rows)
    agg_rows = []
    for method, g in raw.groupby("method"):
        row = {"method": method, "n_seeds": len(g)}
        for c in METRIC_COLS:
            vals = g[c].dropna().values
            if len(vals) == 0:
                row[f"{c}_mean"] = row[f"{c}_std"] = np.nan
                row[f"{c}_str"] = "N/A"
                continue
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{c}_mean"] = round(m, 4)
            row[f"{c}_std"] = round(sd, 4)
            row[f"{c}_str"] = f"{m:.4f} +/- {sd:.4f}"
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(agg_path, index=False)

    logger.info("\n" + "=" * 68)
    logger.info("  AGGREGATED (mean +/- std, ddof=1)")
    logger.info("=" * 68)
    for _, r in agg.iterrows():
        logger.info(f"  {r['method']:<24} n={r['n_seeds']}  "
                    f"AA-F1={r['aa_f1_str']}  Forg={r['forgetting_str']}")
    logger.info(f"\n  Raw → {raw_path}")
    logger.info(f"  Agg → {agg_path}")
    logger.info("  E1 complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E1: DistilBERT baselines, multi-seed")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--resume", action="store_true",
                    help="skip (method, seed) rows already in the raw CSV")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    args = ap.parse_args()
    run(load_config(args.config), seeds=args.seeds, resume=args.resume,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
