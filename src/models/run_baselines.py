"""
src/models/run_baselines.py
────────────────────────────
Chạy tất cả baselines cho Table IV và so sánh với DRC-CL.

Baselines:
  1. Static-CNN     (đã chạy bởi drc_cl_runner → load results)
  2. SW-Retrain     (sliding window, full retrain mỗi step)
  3. EWC-only       (full fine-tune + EWC, no replay)
  4. iCaRL          (exemplar replay + knowledge distillation)
  5. GDumb          (greedy buffer + retrain from scratch)

Output:
  results/all_baselines_per_window.csv
  results/table_iv_comparison.csv    ← Table IV cho paper
  results/f1_over_time_all.csv       ← Figure 2 cho paper

Usage:
    # Chạy tất cả baselines:
    python -m src.models.run_baselines

    # Chạy chỉ 1 baseline:
    python -m src.models.run_baselines --only ewc-only

    # Bỏ qua baseline đã chạy rồi:
    python -m src.models.run_baselines --skip sw-retrain
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.utils.common import get_logger, load_config, get_window_ids
from src.models.baselines import (
    run_sw_retrain, run_ewc_only, run_icarl, run_gdumb
)
from src.models.evaluate_cl import (
    evaluate_static_cnn, compute_summary,
    print_comparison_table, save_f1_over_time
)


BASELINE_REGISTRY = {
    "static-cnn": ("Static-CNN",  None),        # handled separately
    "sw-retrain": ("SW-Retrain",  run_sw_retrain),
    "ewc-only":   ("EWC-only",    run_ewc_only),
    "icarl":      ("iCaRL",       run_icarl),
    "gdumb":      ("GDumb",       run_gdumb),
}


def run(cfg: dict,
        backbone_path: Path,
        only: str | None = None,
        skip: list[str]  | None = None,
        device: str = "cuda") -> None:

    import torch
    device = device if torch.cuda.is_available() else "cpu"

    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("run_baselines", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["results"])
    bench_dir = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)

    skip = skip or []

    logger.info("=" * 60)
    logger.info(" BASELINE EVALUATION — Table IV")
    logger.info("=" * 60)
    logger.info(f"  Device   : {device}")
    if device == "cuda":
        logger.info(f"  GPU      : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Backbone : {backbone_path}")
    logger.info(f"  Windows  : {len(window_ids)}")
    logger.info("")

    all_results: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    # ── Load DRC-CL results nếu đã chạy ─────────────────────────────────────
    drc_path = out_dir / "drc_cl_per_window.csv"
    if drc_path.exists():
        drc_pw = pd.read_csv(drc_path).to_dict("records")
        all_results["DRC-CL"] = drc_pw
        summaries.append(compute_summary(drc_pw, "DRC-CL"))
        logger.info(f"  Loaded DRC-CL results ({len(drc_pw)} windows)")

    # ── Run baselines ─────────────────────────────────────────────────────────
    baselines_to_run = list(BASELINE_REGISTRY.keys())
    if only:
        baselines_to_run = [only]

    for key in baselines_to_run:
        if key in skip:
            logger.info(f"\n  [{key}] SKIPPED (--skip)")
            continue

        name, run_fn = BASELINE_REGISTRY[key]

        logger.info(f"\n{'─'*60}")
        logger.info(f"  Running: {name}")
        logger.info(f"{'─'*60}")

        t0 = time.time()

        if key == "static-cnn":
            pw = evaluate_static_cnn(backbone_path, bench_dir, window_ids, device)
        else:
            # Merge DRC-CL config defaults cho baselines
            bl_cfg = cfg.copy()
            bl_cfg.setdefault("training", {})
            bl_cfg["training"].setdefault("update_epochs", 5)
            bl_cfg["training"].setdefault("lr", 5e-4)
            bl_cfg["training"].setdefault("batch_size", 512)
            bl_cfg.setdefault("ewc", {})
            bl_cfg["ewc"].setdefault("lambda", 0.4)
            bl_cfg.setdefault("ser", {})
            bl_cfg["ser"].setdefault("capacity", 5000)
            pw = run_fn(bl_cfg, backbone_path, device)

        elapsed = time.time() - t0
        logger.info(f"  {name} done in {elapsed:.1f}s")

        all_results[name] = pw
        s = compute_summary(pw, name)
        summaries.append(s)

        # Lưu per-window
        csv_name = f"baseline_{key.replace('-','_')}_per_window.csv"
        pd.DataFrame(pw).to_csv(out_dir / csv_name, index=False)
        logger.info(f"  Saved: {csv_name}")

    # ── Comparison Table (Table IV) ───────────────────────────────────────────
    logger.info(f"\n{'═'*60}")

    # Sắp xếp theo thứ tự paper
    order = ["Static-CNN", "SW-Retrain", "EWC-only", "iCaRL", "GDumb", "DRC-CL"]
    summaries_sorted = sorted(summaries,
                              key=lambda s: order.index(s["method"])
                              if s["method"] in order else 99)

    print_comparison_table(summaries_sorted, logger)

    # Lưu Table IV
    table_path = out_dir / "table_iv_comparison.csv"
    pd.DataFrame(summaries_sorted).to_csv(table_path, index=False)
    logger.info(f"\n  Table IV → {table_path}")

    # Lưu F1-over-time cho Figure 2
    f1t_path = out_dir / "f1_over_time_all.csv"
    save_f1_over_time(all_results, f1t_path)
    logger.info(f"  Figure 2 data → {f1t_path}")

    # Lưu all baselines combined
    all_pw = []
    for name, pw in all_results.items():
        for r in pw:
            r["method"] = name
            all_pw.append(r)
    all_path = out_dir / "all_baselines_per_window.csv"
    pd.DataFrame(all_pw).to_csv(all_path, index=False)
    logger.info(f"  All baselines → {all_path}")

    logger.info(f"\n{'═'*60}")
    logger.info(" BASELINE EVALUATION COMPLETE ✓")
    logger.info(f"{'═'*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all baselines for Table IV comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config",   default=None)
    parser.add_argument("--backbone", default=None,
                        help="Path to backbone_d01.pt")
    parser.add_argument("--only",     default=None,
                        choices=list(BASELINE_REGISTRY.keys()),
                        help="Run only this baseline")
    parser.add_argument("--skip",     nargs="*", default=[],
                        choices=list(BASELINE_REGISTRY.keys()),
                        help="Skip these baselines")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Thêm defaults cho baselines
    cfg.setdefault("lora", {"rank": 8, "alpha": 16.0})
    cfg.setdefault("ser",  {"capacity": 5000, "beta": 0.92, "min_k": 50})
    cfg.setdefault("ewc",  {"lambda": 0.4})
    cfg.setdefault("training", {"lr": 5e-4, "update_epochs": 5,
                                "batch_size": 512, "mix_ratio": 0.3})

    if args.backbone:
        backbone_path = Path(args.backbone)
    else:
        backbone_path = Path(cfg["paths"]["results"]) / "checkpoints" / "backbone_d01.pt"
    if not backbone_path.exists():
        print(f"ERROR: Backbone not found at {backbone_path}")
        exit(1)

    run(cfg, backbone_path, only=args.only, skip=args.skip)
