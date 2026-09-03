"""
src/data/run_pipeline.py
─────────────────────────
Master script: chạy toàn bộ pipeline chuẩn bị dữ liệu theo thứ tự.

Bước:
  0. Tải dữ liệu benign (Tranco)         → data/raw/benign/tranco_YYYY.csv
  1. Gộp DGArchive CSVs                  → data/interim/dgarchive_merged.parquet
  2. Xây dựng 24 cửa sổ quý DGA         → data/processed/windows/
  3. Ghép benign vào cửa sổ             → data/processed/benchmark/D01.csv…D24.csv
  4. Gán nhãn trượt khái niệm           → data/processed/benchmark/drift_labels.json
  5. Báo cáo tính toàn vẹn              → data/processed/benchmark/integrity_report.txt

Usage (từ thư mục gốc project):
    # Toàn bộ pipeline (tải Tranco + stub drift):
    python -m src.data.run_pipeline --stub

    # Bỏ qua bước 0 (đã có file benign):
    python -m src.data.run_pipeline --stub --start-from 1

    # Có backbone để tính MMD²:
    python -m src.data.run_pipeline --backbone results/checkpoints/backbone_d01.pt

    # Chỉ chạy từ bước 3:
    python -m src.data.run_pipeline --stub --start-from 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.utils.common import get_logger, load_config
from src.data import (
    step0_download_benign,
    step1_merge_dgarchive,
    step2_build_dga_windows,
    step3_merge_benign,
    step4_annotate_drift,
    step5_integrity_report,
)

STEPS = [
    (0, "Tải dữ liệu benign (Tranco)",     step0_download_benign.run),
    (1, "Gộp DGArchive CSVs → Parquet",    step1_merge_dgarchive.run),
    (2, "Xây dựng 24 cửa sổ DGA",          step2_build_dga_windows.run),
    (3, "Ghép benign vào cửa sổ",          step3_merge_benign.run),
    (4, "Gán nhãn trượt khái niệm",        None),
    (5, "Báo cáo tính toàn vẹn",           step5_integrity_report.run),
]


def run(cfg: dict, start_from: int, backbone: str | None, stub: bool) -> None:
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("pipeline", log_dir=log_dir)

    logger.info("=" * 60)
    logger.info(" DRC-CL Data Preparation Pipeline")
    logger.info("=" * 60)

    t_total = time.time()

    for step_num, step_name, step_fn in STEPS:
        if step_num < start_from:
            logger.info(f"\n[Bước {step_num}] {step_name} — BỎ QUA (--start-from {start_from})")
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(f"[Bước {step_num}/{len(STEPS)-1}] {step_name}")
        logger.info(f"{'─'*60}")
        t0 = time.time()

        try:
            if step_num == 4:
                if stub or backbone is None:
                    step4_annotate_drift.run_stub(cfg)
                else:
                    step4_annotate_drift.run_with_backbone(cfg, backbone)
            else:
                step_fn(cfg)
        except Exception as exc:
            logger.error(f"Bước {step_num} THẤT BẠI: {exc}", exc_info=True)
            sys.exit(1)

        elapsed = time.time() - t0
        logger.info(f"[Bước {step_num}] Hoàn thành trong {elapsed:.1f}s")

    total = time.time() - t_total
    logger.info(f"\n{'='*60}")
    logger.info(f" Pipeline hoàn thành trong {total:.1f}s  ✓")
    logger.info(f"{'='*60}")
    logger.info("\nOutput files:")
    logger.info(f"  Benign raw  : {cfg['paths']['benign_raw']}/tranco_YYYY.csv")
    logger.info(f"  DGA parquet : {cfg['paths']['interim']}/dgarchive_merged.parquet")
    logger.info(f"  Windows     : {cfg['paths']['windows_dir']}/D01_dga.csv … D24_dga.csv")
    logger.info(f"  Benchmark   : {cfg['paths']['benchmark_dir']}/D01.csv … D24.csv")
    logger.info(f"  Drift labels: {cfg['paths']['benchmark_dir']}/drift_labels.json")
    logger.info(f"  QA report   : {cfg['paths']['benchmark_dir']}/integrity_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRC-CL Data Preparation Pipeline")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--backbone",   default=None,
                        help="Path to pretrained CharCNN .pt file for drift annotation")
    parser.add_argument("--stub",       action="store_true",
                        help="Dùng stub drift labels (không cần backbone)")
    parser.add_argument("--start-from", type=int, default=0, dest="start_from",
                        help="Bắt đầu từ bước số N (0-5)")
    args = parser.parse_args()

    if args.start_from not in range(0, 6):
        parser.error("--start-from phải từ 0 đến 5")

    cfg = load_config(args.config)
    run(cfg, start_from=args.start_from, backbone=args.backbone, stub=args.stub)
