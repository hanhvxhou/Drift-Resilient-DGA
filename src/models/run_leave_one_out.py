"""
src/models/run_leave_one_out.py
───────────────────────────────
E2 — Leave-one-out ablation for DRC-CL (Reviewers #3-Q2 and #4-C2).

The cumulative Table V cannot separate a component's independent contribution
when components interact. This runs the complementary leave-one-out ablation:
start from FULL DRC-CL and remove ONE component at a time.

CONSISTENCY: this script calls cl_experiment.run_method() — the SAME function
that produced Table IV and the run_multi_seed results — with the component
flags toggled. It therefore uses the identical train/test split protocol
(train on {win}_train.csv, evaluate the 24x24 matrix on {win}_test.csv) and the
identical metric definitions (AccuracyMatrix.compute_metrics: AA-F1, BWT, FWT,
Forgetting). The leave-one-out numbers are directly comparable to the rest of
the paper, and FULL (DRC-CL) here must match DRC-CL in run_multi_seed.

Configurations (all keep LoRA; LoRA cannot be removed since it is the only
trainable pathway — the never-updated Static-CNN already bounds the no-LoRA
case):
    FULL (DRC-CL)  use_lora, use_ser, use_ewc, use_add
    FULL - SER     use_lora,          use_ewc, use_add
    FULL - EWC     use_lora, use_ser,          use_add
    FULL - ADD     use_lora, use_ser, use_ewc            (update every window)

Usage
-----
    python -m src.models.run_leave_one_out \
        --seeds 42 123 456 789 2024 3141 5926 5358 9793 2384

Outputs (results/multi_seed/)
    leave_one_out_raw.csv   — one row per (config, seed)
    leave_one_out_agg.csv   — mean +/- std (ddof=1) per config, Delta vs FULL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, get_window_ids
from src.models.run_multi_seed import train_backbone_with_seed

DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 3141, 5926, 5358, 9793, 2384]
METRICS = ["aa_f1", "bwt", "fwt", "forgetting", "degrad"]

# (label, flags for run_method) — all keep use_lora=True
CONFIGS = [
    ("FULL (DRC-CL)", dict(use_lora=True, use_ser=True,  use_ewc=True,  use_add=True)),
    ("FULL - SER",    dict(use_lora=True, use_ser=False, use_ewc=True,  use_add=True)),
    ("FULL - EWC",    dict(use_lora=True, use_ser=True,  use_ewc=False, use_add=True)),
    ("FULL - ADD",    dict(use_lora=True, use_ser=True,  use_ewc=True,  use_add=False,
                          update_every=True)),
]


def run(cfg: dict, seeds: list[int], device: str = "cuda", resume: bool = False) -> None:
    from src.models.cl_experiment import run_method

    out_dir = Path(cfg["paths"]["results"])
    ms_dir = out_dir / "multi_seed"
    ms_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("leave_one_out", log_dir=out_dir / "logs")

    raw_path = ms_dir / "leave_one_out_raw.csv"
    agg_path = ms_dir / "leave_one_out_agg.csv"
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    window_ids = get_window_ids(cfg)

    rows: list[dict] = []
    done: dict[int, set] = {}
    if resume and raw_path.exists():
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        for r in rows:
            done.setdefault(int(r["seed"]), set()).add(r["config"])
        logger.info(f"  Resume: {len(rows)} rows present")

    logger.info("=" * 70)
    logger.info(f" E2 — LEAVE-ONE-OUT ABLATION ({len(seeds)} seeds)")
    logger.info(" Uses cl_experiment.run_method (same path as Table IV)")
    logger.info("=" * 70)

    for seed in seeds:
        need = [c for c, _ in CONFIGS if c not in done.get(seed, set())]
        if not need:
            logger.info(f"  seed {seed}: all configs cached, skip")
            continue

        cfg_seed = {**cfg, "random_seed": seed}
        backbone_path = train_backbone_with_seed(cfg_seed, seed, device, logger)
        common = dict(cfg=cfg_seed, backbone_path=backbone_path, device=device,
                      logger=logger, split_dir=split_dir, window_ids=window_ids)

        for label, flags in CONFIGS:
            if label in done.get(seed, set()):
                logger.info(f"  seed {seed} · {label}: cached, skip")
                continue
            logger.info(f"\n  -- seed {seed} : {label} --")
            result = run_method(method_name=label, **common, **flags)
            m = result["metrics"]
            row = {"config": label, "seed": seed}
            for k in METRICS:
                row[k] = m.get(k, np.nan)
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)  # crash-safe
            logger.info(f"     AA-F1={m['aa_f1']:.4f}  Forgetting={m['forgetting']:.4f}  "
                        f"BWT={m['bwt']:+.4f}  FWT={m.get('fwt', float('nan')):+.4f}")

    # aggregate mean +/- std (ddof=1) + delta vs FULL
    raw = pd.DataFrame(rows)
    agg = []
    means = {}
    for label, _ in CONFIGS:
        g = raw[raw["config"] == label]
        r = {"config": label, "n_seeds": len(g)}
        for k in METRICS:
            v = g[k].dropna().values
            mean = float(np.mean(v)) if len(v) else np.nan
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            r[f"{k}_mean"] = round(mean, 4)
            r[f"{k}_std"] = round(sd, 4)
        means[label] = r
        agg.append(r)
    full = means["FULL (DRC-CL)"]
    for r in agg:
        r["d_forgetting_vs_full"] = round(r["forgetting_mean"] - full["forgetting_mean"], 4)
        r["d_aa_f1_vs_full"] = round(r["aa_f1_mean"] - full["aa_f1_mean"], 4)
    pd.DataFrame(agg).to_csv(agg_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("  LEAVE-ONE-OUT RESULTS (mean +/- std, ddof=1)")
    logger.info("=" * 70)
    for r in agg:
        logger.info(f"  {r['config']:<16} "
                    f"AA-F1={r['aa_f1_mean']:.4f}+/-{r['aa_f1_std']:.4f}  "
                    f"Forg={r['forgetting_mean']:.4f}+/-{r['forgetting_std']:.4f}  "
                    f"dForg={r['d_forgetting_vs_full']:+.4f}")
    logger.info("\n  A POSITIVE dForg means removing that component makes forgetting")
    logger.info("  WORSE, i.e. the component genuinely helps.")
    logger.info(f"\n  Raw -> {raw_path}")
    logger.info(f"  Agg -> {agg_path}")
    logger.info("  CHECK: FULL (DRC-CL) here should match DRC-CL in run_multi_seed.")
    logger.info("  E2 complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E2: leave-one-out ablation (via run_method)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(load_config(args.config), seeds=args.seeds, device=args.device, resume=args.resume)
