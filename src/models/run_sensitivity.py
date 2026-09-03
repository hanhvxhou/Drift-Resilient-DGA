"""
src/models/run_sensitivity.py
──────────────────────────────
Hyperparameter sensitivity analysis for Table VIII.
Grid search on W1-W12, report AA-F1 and Forgetting.

Parameters tested:
  - LoRA rank: {4, 8, 16}
  - EWC lambda: {0.0, 0.1, 0.4, 1.0}
  - Buffer size: {500, 1000, 5000, 10000}

Usage:
    python -m src.models.run_sensitivity
    python -m src.models.run_sensitivity --param rank
    python -m src.models.run_sensitivity --param lambda
    python -m src.models.run_sensitivity --param buffer
"""

from __future__ import annotations
import argparse, time, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.models.cl_experiment import run_method
from src.models.cl_metrics import AccuracyMatrix, print_metrics_table
from src.utils.common import get_logger, load_config, get_window_ids


def run_sensitivity_experiment(cfg, backbone_path, device, logger,
                               param_name, param_values, default_cfg):
    """Run DRC-CL with different values of one parameter on W1-W12."""
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    window_ids = get_window_ids(cfg)[:12]  # W1-W12 only

    results = []
    for val in param_values:
        # Build config with this parameter value
        test_cfg = copy.deepcopy(default_cfg)
        if param_name == "rank":
            test_cfg["lora"]["rank"] = val
        elif param_name == "lambda":
            test_cfg["ewc"]["lambda"] = val
        elif param_name == "buffer":
            test_cfg["ser"]["capacity"] = val

        label = f"{param_name}={val}"
        logger.info(f"\n  Running DRC-CL with {label} (W1-W12)...")

        t0 = time.time()
        result = run_method(
            method_name=f"DRC-CL ({label})",
            cfg=test_cfg, backbone_path=backbone_path, device=device,
            logger=logger, split_dir=split_dir, window_ids=window_ids,
            use_lora=True, use_ser=True, use_ewc=(test_cfg["ewc"]["lambda"] > 0),
            use_add=True
        )
        elapsed = time.time() - t0
        m = result["metrics"]
        m["param"] = param_name
        m["value"] = val
        results.append(m)
        logger.info(f"    {label}: AA-F1={m['aa_f1']:.4f}  Forg={m['forgetting']:.4f}  ({elapsed:.1f}s)")

    return results


def run(cfg, param=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("sensitivity", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["results"])
    backbone_path = out_dir / "checkpoints" / "backbone_d01.pt"

    # Defaults
    cfg.setdefault("lora", {"rank": 8, "alpha": 16.0})
    cfg.setdefault("ser", {"capacity": 5000, "beta": 0.92, "min_k": 50})
    cfg.setdefault("ewc", {"lambda": 0.4})
    cfg.setdefault("training", {"lr": 5e-4, "update_epochs": 5,
                                "batch_size": 512, "mix_ratio": 0.3})

    logger.info("=" * 60)
    logger.info(" HYPERPARAMETER SENSITIVITY (W1-W12)")
    logger.info("=" * 60)
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    all_results = []
    default_cfg = copy.deepcopy(cfg)

    EXPERIMENTS = {
        "rank":   [4, 8, 16],
        "lambda": [0.0, 0.1, 0.4, 1.0],
        "buffer": [500, 1000, 5000, 10000],
    }

    params_to_run = [param] if param else list(EXPERIMENTS.keys())

    for p in params_to_run:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Parameter: {p} — values: {EXPERIMENTS[p]}")
        logger.info(f"{'─'*60}")

        results = run_sensitivity_experiment(
            cfg, backbone_path, device, logger,
            p, EXPERIMENTS[p], default_cfg
        )
        all_results.extend(results)

    # ── Print Table VIII ──────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(" TABLE VIII: Hyperparameter Sensitivity (W1-W12)")
    logger.info(f"{'='*70}")

    for p in params_to_run:
        p_results = [r for r in all_results if r.get("param") == p]
        logger.info(f"\n  {p.upper()}")
        logger.info(f"  {'Value':<12} {'AA-F1':>8} {'BWT':>8} {'Forg.':>8} {'Degrad.':>8}")
        logger.info(f"  {'-'*48}")
        for r in p_results:
            marker = " <-- default" if (
                (p == "rank" and r["value"] == 8) or
                (p == "lambda" and r["value"] == 0.4) or
                (p == "buffer" and r["value"] == 5000)
            ) else ""
            logger.info(f"  {str(r['value']):<12} {r['aa_f1']:>8.4f} {r['bwt']:>+8.4f} {r['forgetting']:>8.4f} {r['degrad']:>+8.4f}{marker}")

    # Save
    save_path = out_dir / "sensitivity_results.csv"
    pd.DataFrame(all_results).to_csv(save_path, index=False)
    logger.info(f"\n  Saved: {save_path}")
    logger.info("  Sensitivity analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity")
    parser.add_argument("--config", default=None)
    parser.add_argument("--param", default=None, choices=["rank", "lambda", "buffer"],
                        help="Test only this parameter (default: all)")
    args = parser.parse_args()
    run(load_config(args.config), param=args.param)
