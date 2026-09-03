"""
src/data/step2_build_dga_windows.py
────────────────────────────────────
Step 2: From data/interim/dgarchive_merged.parquet, produce
        data/processed/windows/D01.csv … D24.csv  (DGA side only).

Logic (Section V.A.1 of the paper):
  1. Load merged Parquet.
  2. Drop families whose total sample count (over 2018-2023) < min_family_samples.
  3. For each quarterly window D_t:
       a. Collect all DGA rows with quarter_id == D_t.
       b. Compute per-family proportions within that quarter.
       c. Sample up to max_per_window rows stratified by family proportion.
       d. Add label = 1, shuffle, write D_t.csv.
  4. Write data/processed/windows/window_stats.csv (per-window summary).

Output columns per window CSV:
    domain, label, family, quarter_id, quarter_label

Usage (from project root):
    python -m src.data.step2_build_dga_windows
    python -m src.data.step2_build_dga_windows --config configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, get_path, load_config, quarter_id, quarter_label, get_window_ids

# ── Constants ─────────────────────────────────────────────────────────────────
INTERIM_FILE  = "dgarchive_merged.parquet"
STATS_FILE    = "window_stats.csv"


# ── Stratified sampler ────────────────────────────────────────────────────────
def stratified_sample(df: pd.DataFrame,
                      group_col: str,
                      max_total: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample up to max_total rows from df, preserving group proportions.
    If df already has <= max_total rows, return all rows.
    Within each group, sample is drawn without replacement.
    """
    if len(df) <= max_total:
        return df.copy()

    # Compute per-group target sizes (proportional, integer allocation)
    counts  = df[group_col].value_counts()          # family → count in this window
    total   = counts.sum()
    targets = (counts / total * max_total).astype(int)

    # Distribute any remaining slots due to rounding
    remainder = max_total - targets.sum()
    if remainder > 0:
        # Add 1 to the families with largest fractional parts
        fracs = (counts / total * max_total) - targets
        top   = fracs.nlargest(int(remainder)).index
        targets[top] += 1

    sampled_parts = []
    for family, n in targets.items():
        if n <= 0:
            continue
        pool = df.loc[df[group_col] == family]
        n    = min(n, len(pool))
        sampled_parts.append(pool.sample(n=n, random_state=int(rng.integers(1 << 31))))

    return pd.concat(sampled_parts, ignore_index=True)


def run(cfg: dict) -> None:
    log_dir    = Path(cfg["paths"]["results"]) / "logs"
    logger     = get_logger("step2_windows", log_dir=log_dir)

    interim_dir = Path(cfg["paths"]["interim"])
    windows_dir = get_path(cfg, "windows_dir")

    parquet_file = interim_dir / INTERIM_FILE
    if not parquet_file.exists():
        raise FileNotFoundError(
            f"{parquet_file} not found. Run step1_merge_dgarchive.py first."
        )

    min_samples  = cfg["dga"]["min_family_samples"]
    max_per_win  = cfg["dga"]["max_per_window"]
    dga_label    = cfg["dga"]["label"]
    seed         = cfg["random_seed"]
    rng          = np.random.default_rng(seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading {parquet_file} …")
    df = pd.read_parquet(parquet_file,
                         columns=["domain", "family", "quarter_id", "quarter_label"])
    logger.info(f"  Loaded {len(df):,} rows, {df['family'].nunique()} families")

    # ── Filter rare families ──────────────────────────────────────────────────
    family_counts = df["family"].value_counts()
    valid_families = family_counts[family_counts >= min_samples].index
    dropped = family_counts[family_counts < min_samples]
    if not dropped.empty:
        logger.info(f"  Dropping {len(dropped)} families with < {min_samples} samples:")
        for fam, cnt in dropped.items():
            logger.info(f"    {fam:35s}  {cnt:>8,}")
    df = df[df["family"].isin(valid_families)].copy()
    logger.info(f"  After filter: {len(df):,} rows, {df['family'].nunique()} families")

    # ── Build window ids từ config (dynamic) ──────────────────────────────────
    window_ids = get_window_ids(cfg)

    stats_rows = []

    for win_id in window_ids:
        subset = df[df["quarter_id"] == win_id]
        n_before = len(subset)
        families_active = subset["family"].nunique()

        if n_before == 0:
            logger.warning(f"  {win_id}: 0 DGA rows — window will be empty (check date range)")
            continue

        # Stratified sample
        sampled = stratified_sample(subset, group_col="family",
                                    max_total=max_per_win, rng=rng)
        sampled = sampled.sample(frac=1, random_state=int(rng.integers(1 << 31)))  # shuffle
        sampled = sampled.reset_index(drop=True)
        sampled["label"] = dga_label

        # Output columns
        out_cols = ["domain", "label", "family", "quarter_id", "quarter_label"]
        sampled  = sampled[out_cols]

        out_path = windows_dir / f"{win_id}_dga.csv"
        sampled.to_csv(out_path, index=False)

        q_label = sampled["quarter_label"].iloc[0]
        logger.info(
            f"  {win_id} ({q_label}):  "
            f"{families_active:3d} families  |  "
            f"raw={n_before:>8,}  sampled={len(sampled):>6,}  "
            f"→ {out_path.name}"
        )

        stats_rows.append({
            "window_id":       win_id,
            "quarter_label":   q_label,
            "families_active": families_active,
            "raw_dga_count":   n_before,
            "sampled_dga":     len(sampled),
            "output_file":     out_path.name,
        })

    # ── Write window stats ────────────────────────────────────────────────────
    stats_df   = pd.DataFrame(stats_rows)
    stats_path = windows_dir / STATS_FILE
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"\nWindow stats → {stats_path}")
    logger.info("Step 2 complete ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Build 24 DGA quarterly windows")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config))
