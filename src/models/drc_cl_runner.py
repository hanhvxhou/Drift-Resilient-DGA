"""
src/models/drc_cl_runner.py
────────────────────────────
Script chạy toàn bộ DRC-CL framework + đánh giá + so sánh với Static-CNN.

Luồng:
    1. Load backbone đã pretrain (backbone_d01.pt)
    2. Chạy DRC-CL prequential loop qua 24 cửa sổ
    3. Đánh giá Static-CNN baseline (cùng backbone, không update)
    4. In bảng so sánh IEEE
    5. Lưu tất cả kết quả (CSV + JSON)

Usage:
    # Chạy với config mặc định:
    python -m src.models.drc_cl_runner

    # Tùy chỉnh hyperparameters:
    python -m src.models.drc_cl_runner --rank 8 --lambda-ewc 0.4 --mix-ratio 0.3

    # Chạy với seed khác (reproducibility):
    python -m src.models.drc_cl_runner --seed 123
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.utils.common import get_logger, load_config
from src.models.drc_cl import DRCCL
from src.models.evaluate_cl import (
    evaluate_static_cnn, compute_summary,
    print_comparison_table, save_f1_over_time,
    get_window_ids
)


def build_full_config(base_cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI args vào config, thêm các key cần thiết cho DRC-CL."""
    cfg = base_cfg.copy()

    # Đảm bảo các section tồn tại
    cfg.setdefault("lora",     {})
    cfg.setdefault("ser",      {})
    cfg.setdefault("ewc",      {})
    cfg.setdefault("training", {})

    # Override từ CLI (nếu có)
    if args.rank       is not None: cfg["lora"]["rank"]           = args.rank
    if args.alpha      is not None: cfg["lora"]["alpha"]          = args.alpha
    if args.lambda_ewc is not None: cfg["ewc"]["lambda"]          = args.lambda_ewc
    if args.buffer     is not None: cfg["ser"]["capacity"]         = args.buffer
    if args.beta       is not None: cfg["ser"]["beta"]             = args.beta
    if args.mix_ratio  is not None: cfg["training"]["mix_ratio"]   = args.mix_ratio
    if args.lr         is not None: cfg["training"]["lr"]           = args.lr
    if args.epochs     is not None: cfg["training"]["update_epochs"]= args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"]   = args.batch_size
    if args.seed       is not None: cfg["random_seed"]              = args.seed

    # Defaults nếu chưa có
    cfg["lora"].setdefault("rank",   8)
    cfg["lora"].setdefault("alpha",  16.0)
    cfg["ser"].setdefault("capacity", 5000)
    cfg["ser"].setdefault("beta",     0.92)
    cfg["ser"].setdefault("min_k",    50)
    cfg["ewc"].setdefault("lambda",   0.4)
    cfg["training"].setdefault("lr",            5e-4)
    cfg["training"].setdefault("update_epochs", 5)
    cfg["training"].setdefault("batch_size",    512)
    cfg["training"].setdefault("mix_ratio",     0.3)

    return cfg


def run(cfg: dict, backbone_path: Path, skip_static: bool = False) -> None:
    import torch

    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("drc_cl_runner", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["results"])
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info(" DRC-CL RUNNER")
    logger.info("=" * 60)
    logger.info(f"  Device     : {device}")
    if device == "cuda":
        logger.info(f"  GPU        : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Backbone   : {backbone_path}")
    logger.info(f"  LoRA rank  : {cfg['lora']['rank']}")
    logger.info(f"  LoRA alpha : {cfg['lora']['alpha']}")
    logger.info(f"  EWC lambda : {cfg['ewc']['lambda']}")
    logger.info(f"  SER buffer : {cfg['ser']['capacity']}")
    logger.info(f"  SER beta   : {cfg['ser']['beta']}")
    logger.info(f"  Mix ratio  : {cfg['training']['mix_ratio']}")
    logger.info(f"  Update LR  : {cfg['training']['lr']}")
    logger.info(f"  Update ep  : {cfg['training']['update_epochs']}")
    logger.info(f"  Batch size : {cfg['training']['batch_size']}")
    logger.info(f"  Seed       : {cfg['random_seed']}")
    logger.info("")

    t_total = time.time()

    # ==================================================================
    # Phase 1: DRC-CL
    # ==================================================================
    logger.info("-" * 60)
    logger.info("PHASE 1: DRC-CL Training (Prequential)")
    logger.info("-" * 60)

    drc = DRCCL(backbone_path=backbone_path, cfg=cfg, device=device)
    drc_results = drc.run(save_results=True)

    # ==================================================================
    # Phase 2: Static-CNN Baseline
    # ==================================================================
    results_by_method = {"DRC-CL": drc_results["per_window"]}
    summaries = [compute_summary(drc_results["per_window"], "DRC-CL")]

    if not skip_static:
        logger.info("")
        logger.info("-" * 60)
        logger.info("PHASE 2: Static-CNN Baseline")
        logger.info("-" * 60)

        bench_dir  = Path(cfg["paths"]["benchmark_dir"])
        window_ids = get_window_ids(cfg)

        static_pw  = evaluate_static_cnn(
            backbone_path, bench_dir, window_ids, device
        )
        static_csv = out_dir / "static_cnn_per_window.csv"
        pd.DataFrame(static_pw).to_csv(static_csv, index=False)
        logger.info(f"  Static-CNN results saved to {static_csv}")

        s_static = compute_summary(static_pw, "Static-CNN")
        summaries.append(s_static)
        results_by_method["Static-CNN"] = static_pw

    # ==================================================================
    # Phase 3: Comparison
    # ==================================================================
    logger.info("")
    logger.info("-" * 60)
    logger.info("PHASE 3: Comparison")
    logger.info("-" * 60)

    print_comparison_table(summaries, logger)

    # Lưu comparison
    comp_path = out_dir / "comparison_table.csv"
    pd.DataFrame(summaries).to_csv(comp_path, index=False)
    logger.info(f"  Comparison table saved to {comp_path}")

    # Lưu F1-over-time
    f1t_path = out_dir / "f1_over_time.csv"
    save_f1_over_time(results_by_method, f1t_path)
    logger.info(f"  F1-over-time saved to {f1t_path}")

    # Lưu full config dùng trong lần chạy này
    config_record = {
        "lora":     cfg["lora"],
        "ser":      cfg["ser"],
        "ewc":      cfg["ewc"],
        "training": cfg["training"],
        "seed":     cfg["random_seed"],
        "device":   device,
    }
    config_path = out_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=2)
    logger.info(f"  Run config saved to {config_path}")

    elapsed = time.time() - t_total
    logger.info(f"\n{'='*60}")
    logger.info(f" TOTAL TIME: {elapsed:.1f}s")
    logger.info(f"{'='*60}")
    logger.info("\n  Output files:")
    logger.info(f"    results/drc_cl_per_window.csv   <- per-window F1, AUC, drift")
    logger.info(f"    results/drc_cl_summary.json     <- AA-F1, BWT, FWT")
    if not skip_static:
        logger.info(f"    results/static_cnn_per_window.csv")
    logger.info(f"    results/comparison_table.csv    <- comparison table")
    logger.info(f"    results/f1_over_time.csv        <- data for Figure 2")
    logger.info(f"    results/run_config.json         <- hyperparams logged")
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    1. Open f1_over_time.csv in Excel/matplotlib for Figure 2")
    logger.info("    2. Add baselines (iCaRL, GDumb) for full Table IV")
    logger.info("    3. Run ablation: --rank 4 / --lambda-ewc 0 / --buffer 500")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DRC-CL framework + evaluation + comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run:
  python -m src.models.drc_cl_runner

  # Custom LoRA rank and EWC:
  python -m src.models.drc_cl_runner --rank 16 --lambda-ewc 0.6

  # Ablation: no EWC:
  python -m src.models.drc_cl_runner --lambda-ewc 0

  # Ablation: small buffer:
  python -m src.models.drc_cl_runner --buffer 500

  # Quick run (skip Static-CNN evaluation):
  python -m src.models.drc_cl_runner --skip-static
        """
    )
    parser.add_argument("--config",     default=None)
    parser.add_argument("--backbone",   default=None,
                        help="Path to backbone_d01.pt (auto-detect if empty)")

    # LoRA
    parser.add_argument("--rank",       type=int,   default=None, help="LoRA rank (default: 8)")
    parser.add_argument("--alpha",      type=float, default=None, help="LoRA alpha (default: 16)")

    # EWC
    parser.add_argument("--lambda-ewc", type=float, default=None, help="EWC lambda (default: 0.4)")

    # SER
    parser.add_argument("--buffer",     type=int,   default=None, help="SER buffer size (default: 5000)")
    parser.add_argument("--beta",       type=float, default=None, help="SER diversity beta (default: 0.92)")
    parser.add_argument("--mix-ratio",  type=float, default=None, help="Buffer mix ratio mu (default: 0.3)")

    # Training
    parser.add_argument("--lr",         type=float, default=None, help="Update LR (default: 5e-4)")
    parser.add_argument("--epochs",     type=int,   default=None, help="Update epochs per event (default: 5)")
    parser.add_argument("--batch-size", type=int,   default=None, help="Batch size (default: 512)")
    parser.add_argument("--seed",       type=int,   default=None, help="Random seed (default: 42)")

    # Flags
    parser.add_argument("--skip-static", action="store_true",
                        help="Skip Static-CNN baseline evaluation")

    args = parser.parse_args()

    # Load config
    base_cfg = load_config(args.config)
    cfg      = build_full_config(base_cfg, args)

    # Auto-detect backbone
    if args.backbone:
        backbone_path = Path(args.backbone)
    else:
        backbone_path = Path(cfg["paths"]["results"]) / "checkpoints" / "backbone_d01.pt"
    if not backbone_path.exists():
        print(f"ERROR: Backbone not found at {backbone_path}")
        print("Run first: python -m src.models.train_backbone")
        exit(1)

    run(cfg, backbone_path, skip_static=args.skip_static)
