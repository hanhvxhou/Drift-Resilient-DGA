"""
src/data/step1_merge_dgarchive.py
──────────────────────────────────
Step 1: Read all *_dga.csv files from data/raw/dgarchive/,
        parse timestamps, filter to 2018-01-01…2023-12-31,
        add 'family' and 'quarter' columns, write to a single
        Parquet file at data/interim/dgarchive_merged.parquet.

Runtime (typical): ~2–5 min for 60M+ domains depending on disk speed.

Usage (from project root):
    python -m src.data.step1_merge_dgarchive
    python -m src.data.step1_merge_dgarchive --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from src.utils.common import get_logger, get_path, load_config, quarter_id, quarter_label

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_FILE = "dgarchive_merged.parquet"
CHUNK_SIZE  = 200_000        # rows per CSV chunk → controls peak RAM
DATE_COL    = "valid_from"   # timestamp column used for windowing


def parse_family_name(csv_path: Path) -> str:
    """
    Extract family name from filename.
    'blackhole_dga.csv'  →  'blackhole'
    'gameover_p2p_dga.csv'  →  'gameover_p2p'
    Pattern: remove trailing '_dga' before extension.
    """
    stem = csv_path.stem                        # e.g. 'blackhole_dga'
    family = re.sub(r"_dga$", "", stem)         # e.g. 'blackhole'
    return family.lower()


def assign_quarter(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Add 'year', 'quarter', 'quarter_id', 'quarter_label' columns from a datetime column."""
    dt = df[date_col]
    df = df.copy()
    df["year"]          = dt.dt.year
    df["quarter"]       = dt.dt.quarter
    df["quarter_id"]    = df.apply(lambda r: quarter_id(r["year"], r["quarter"]), axis=1)
    df["quarter_label"] = df.apply(lambda r: quarter_label(r["year"], r["quarter"]), axis=1)
    return df


def process_one_file(csv_path: Path,
                     start: pd.Timestamp,
                     end: pd.Timestamp,
                     logger) -> pd.DataFrame | None:
    """
    Read a single *_dga.csv in chunks, filter to [start, end],
    return a DataFrame with columns: domain, family, valid_from,
    year, quarter, quarter_id, quarter_label.
    Returns None if the file contributes 0 rows after filtering.
    """
    family = parse_family_name(csv_path)
    chunks = []

    try:
        reader = pd.read_csv(
            csv_path,
            usecols=["domain", DATE_COL],   # only columns we need
            parse_dates=[DATE_COL],
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8",
            quotechar='"',
            on_bad_lines="skip",
        )
        for chunk in reader:
            # Filter to target date range
            mask = (chunk[DATE_COL] >= start) & (chunk[DATE_COL] <= end)
            chunk = chunk.loc[mask]
            if chunk.empty:
                continue
            chunk = chunk.drop_duplicates(subset="domain")
            chunks.append(chunk)

    except Exception as exc:
        logger.warning(f"  Skipped {csv_path.name}: {exc}")
        return None

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="domain")
    df["family"] = family
    df = assign_quarter(df)
    return df[["domain", "family", DATE_COL, "year", "quarter", "quarter_id", "quarter_label"]]


def run(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("step1_merge", log_dir=log_dir)

    raw_dir     = Path(cfg["paths"]["dgarchive_raw"])
    interim_dir = get_path(cfg, "interim")
    out_file    = interim_dir / OUTPUT_FILE

    start = pd.Timestamp(cfg["temporal"]["start"])
    end   = pd.Timestamp(cfg["temporal"]["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    csv_files = sorted(raw_dir.glob("*_dga.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_dga.csv files found in {raw_dir}. "
            "Place DGArchive CSV files there and re-run."
        )

    logger.info(f"Found {len(csv_files)} DGA family files in {raw_dir}")
    logger.info(f"Filtering to {start.date()} → {end.date()}")

    all_frames = []
    t0 = time.time()

    for i, csv_path in enumerate(csv_files, 1):
        family = parse_family_name(csv_path)
        logger.info(f"  [{i:3d}/{len(csv_files)}] {family:30s}  ← {csv_path.name}")
        df = process_one_file(csv_path, start, end, logger)
        if df is None or df.empty:
            logger.info(f"          → 0 rows in target range, skipped")
            continue
        logger.info(f"          → {len(df):>10,} rows")
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No data remained after filtering. Check date range and raw files.")

    logger.info("Concatenating all families …")
    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="domain")

    elapsed = time.time() - t0
    logger.info(f"Merged total: {len(merged):,} domains across {merged['family'].nunique()} families")
    logger.info(f"              date range: {merged[DATE_COL].min().date()} → {merged[DATE_COL].max().date()}")
    logger.info(f"              elapsed: {elapsed:.1f}s")

    logger.info(f"Writing Parquet → {out_file}")
    merged.to_parquet(out_file, index=False, compression="snappy")
    logger.info("Step 1 complete ✓")

    # Quick family summary
    summary = (merged.groupby("family")["domain"]
               .count()
               .rename("count")
               .sort_values(ascending=False))
    summary_path = interim_dir / "family_counts.csv"
    summary.to_csv(summary_path)
    logger.info(f"Family counts written → {summary_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Merge DGArchive CSVs to Parquet")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    run(cfg)
