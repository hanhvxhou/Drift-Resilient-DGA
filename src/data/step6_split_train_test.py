"""
src/data/step6_split_train_test.py
────────────────────────────────────
Step 6: Chia moi cua so benchmark thanh train/test co dinh.

Output:  data/processed/benchmark/splits/D01_train.csv ... D24_test.csv

Fix: families co < 5 samples duoc gom vao nhom "rare_dga" de stratify khong loi.

Usage:
    python -m src.data.step6_split_train_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.common import get_logger, load_config, get_window_ids


def _make_stratify_col(df: pd.DataFrame, min_count: int = 5) -> pd.Series:
    """
    Tao cot stratify: families co < min_count samples duoc gom vao 'rare_dga'.
    Benign giu nguyen.
    """
    family_counts = df["family"].value_counts()
    rare_families = set(family_counts[family_counts < min_count].index) - {"benign"}

    strat = df["family"].copy()
    strat[strat.isin(rare_families)] = "rare_dga"
    return strat


def run(cfg: dict, test_ratio: float = 0.2) -> None:
    log_dir   = Path(cfg["paths"]["results"]) / "logs"
    logger    = get_logger("step6_split", log_dir=log_dir)
    bench_dir = Path(cfg["paths"]["benchmark_dir"])
    split_dir = bench_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    window_ids = get_window_ids(cfg)
    seed       = cfg["random_seed"]

    logger.info("=" * 60)
    logger.info(" Step 6: Fixed Train/Test Split")
    logger.info(f" test_ratio={test_ratio}, seed={seed}")
    logger.info("=" * 60)

    stats = []
    for win_id in window_ids:
        path = bench_dir / f"{win_id}.csv"
        if not path.exists():
            logger.warning(f"  {win_id}: file not found, skipping")
            continue

        df = pd.read_csv(path)

        # Stratify col: gom rare families (< 5 samples) de tranh loi
        strat_col = _make_stratify_col(df, min_count=5)

        train_df, test_df = train_test_split(
            df, test_size=test_ratio,
            stratify=strat_col,
            random_state=seed
        )
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        # Verify no overlap
        overlap = set(train_df["domain"]) & set(test_df["domain"])
        assert len(overlap) == 0, f"{win_id}: {len(overlap)} domains in both!"

        train_path = split_dir / f"{win_id}_train.csv"
        test_path  = split_dir / f"{win_id}_test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        q_label = df["quarter_label"].iloc[0]
        n_fam_train = train_df[train_df["label"] == 1]["family"].nunique()
        n_fam_test  = test_df[test_df["label"] == 1]["family"].nunique()

        # Count rare families gom lai
        n_rare = int((strat_col == "rare_dga").sum())

        logger.info(
            f"  {win_id} ({q_label}): "
            f"train={len(train_df):>6,}  test={len(test_df):>5,}  "
            f"families train={n_fam_train}  test={n_fam_test}"
            + (f"  (rare grouped: {n_rare})" if n_rare > 0 else "")
        )

        stats.append({
            "window_id": win_id, "quarter_label": q_label,
            "n_train": len(train_df), "n_test": len(test_df),
            "n_dga_train": int((train_df["label"] == 1).sum()),
            "n_dga_test":  int((test_df["label"] == 1).sum()),
            "families_train": n_fam_train, "families_test": n_fam_test,
            "n_rare_grouped": n_rare,
        })

    stats_path = split_dir / "split_stats.csv"
    pd.DataFrame(stats).to_csv(stats_path, index=False)
    logger.info(f"\n  Stats  -> {stats_path}")
    logger.info(f"  Splits -> {split_dir}/")
    logger.info("  Step 6 complete")
    logger.info("\n  IMPORTANT: Test sets are FIXED. Never use *_test.csv for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Fixed train/test split")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()
    run(load_config(args.config), test_ratio=args.test_ratio)
