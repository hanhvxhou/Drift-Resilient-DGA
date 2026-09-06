"""
src/models/run_leave_one_out.py
───────────────────────────────
E2 — Leave-one-out ablation for DRC-CL.

Reviewers #3 (Q2) and #4 (Concern #2) asked whether each protection layer is
truly necessary or merely overlapping. The existing Table V is a *cumulative*
ablation (LoRA → +SER → +EWC → +ADD), which cannot separate a component's
independent contribution when components interact. This script runs the
complementary *leave-one-out* ablation: start from the FULL DRC-CL and remove
ONE component at a time, so the drop attributable to each component is direct.

Reuses run_variant() from run_ablation.py unchanged — no new metric logic.

Note: LoRA cannot be "left out": the model is CharCNNWithLoRA and LoRA is the
only trainable pathway (freezing the backbone leaves nothing to update). The
never-updated Static-CNN baseline already bounds the no-LoRA case, so we run
leave-one-out over the three optional layers: SER, EWC, ADD.

Configurations (all keep LoRA):
    FULL              LoRA + SER + EWC + ADD          (= DRC-CL)
    FULL − SER        LoRA +       EWC + ADD
    FULL − EWC        LoRA + SER +       ADD
    FULL − ADD        LoRA + SER + EWC                (= update every window)

Usage
-----
    python -m src.models.run_leave_one_out \
        --backbone results/checkpoints/backbone_d01.pt \
        --seeds 42 123 456 789 2024 3141 5926 5358 9793 2384

Outputs (results/multi_seed/)
    leave_one_out_raw.csv   — one row per (config, seed)
    leave_one_out_agg.csv   — mean ± std (ddof=1) per config, with Δ vs FULL
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config

DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 3141, 5926, 5358, 9793, 2384]
METRICS = ["aa_f1", "bwt", "forgetting", "degrad"]

# (label, use_ser, use_ewc, use_add)
CONFIGS = [
    ("FULL (DRC-CL)",  True,  True,  True),
    ("FULL - SER",     False, True,  True),
    ("FULL - EWC",     True,  False, True),
    ("FULL - ADD",     True,  True,  False),
]


def set_all_seeds(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg: dict, backbone_path: Path, seeds: list[int],
        device: str = "cuda", resume: bool = False) -> None:
    from src.models.run_ablation import run_variant

    out_dir = Path(cfg["paths"]["results"])
    ms_dir = out_dir / "multi_seed"
    ms_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("leave_one_out", log_dir=out_dir / "logs")

    raw_path = ms_dir / "leave_one_out_raw.csv"
    agg_path = ms_dir / "leave_one_out_agg.csv"

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
    logger.info("=" * 70)

    for seed in seeds:
        cfg_s = cfg.copy()
        cfg_s["random_seed"] = seed
        for label, use_ser, use_ewc, use_add in CONFIGS:
            if resume and label in done.get(seed, set()):
                logger.info(f"  seed {seed} · {label}: cached, skip")
                continue
            logger.info(f"\n  ── seed {seed} · {label} ──")
            set_all_seeds(seed)
            res = run_variant(label, cfg_s, backbone_path, device,
                              use_ser=use_ser, use_ewc=use_ewc, use_add=use_add,
                              logger=logger)
            row = {"config": label, "seed": seed}
            for m in METRICS:
                row[m] = res.get(m, np.nan)
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)  # crash-safe

    # ── aggregate mean ± std (ddof=1) + Δ vs FULL ──
    raw = pd.DataFrame(rows)
    agg = []
    means = {}
    for label, *_ in CONFIGS:
        g = raw[raw["config"] == label]
        r = {"config": label, "n_seeds": len(g)}
        for m in METRICS:
            v = g[m].dropna().values
            mean = float(np.mean(v)) if len(v) else np.nan
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            r[f"{m}_mean"] = round(mean, 4)
            r[f"{m}_std"] = round(sd, 4)
        means[label] = r
        agg.append(r)
    # delta forgetting vs FULL (how much WORSE without this component)
    full = means["FULL (DRC-CL)"]
    for r in agg:
        r["d_forgetting_vs_full"] = round(r["forgetting_mean"] - full["forgetting_mean"], 4)
        r["d_aa_f1_vs_full"] = round(r["aa_f1_mean"] - full["aa_f1_mean"], 4)
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(agg_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("  LEAVE-ONE-OUT RESULTS (mean ± std, ddof=1)")
    logger.info("=" * 70)
    logger.info(f"  {'Config':<16}{'AA-F1':>16}{'Forgetting':>16}{'ΔForg vs FULL':>16}")
    for r in agg:
        logger.info(f"  {r['config']:<16}"
                    f"{r['aa_f1_mean']:.4f}±{r['aa_f1_std']:.4f}   "
                    f"{r['forgetting_mean']:.4f}±{r['forgetting_std']:.4f}   "
                    f"{r['d_forgetting_vs_full']:+.4f}")
    logger.info(f"\n  Raw → {raw_path}")
    logger.info(f"  Agg → {agg_path}")
    logger.info("  Interpretation: a POSITIVE ΔForg means removing that component")
    logger.info("  makes forgetting WORSE, i.e. the component genuinely helps.")
    logger.info("  E2 complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E2: leave-one-out ablation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", required=True,
                    help="path to backbone_d01.pt (frozen CharCNN)")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(load_config(args.config), backbone_path=Path(args.backbone),
        seeds=args.seeds, device=args.device, resume=args.resume)
