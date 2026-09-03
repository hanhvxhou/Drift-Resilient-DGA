"""
src/data/step3_merge_benign.py
──────────────────────────────
Step 3: Load benign domain lists, filter cross-contamination,
        và merge với DGA windows → benchmark/D01.csv … D24.csv.

VẤN ĐỀ CỐT LÕI: Tranco/Alexa không có timestamp từng tên miền.
Điều này tạo ra 3 rủi ro:
  (A) Temporal leak: cùng tên miền benign có thể xuất hiện ở nhiều cửa sổ.
  (B) Distributional mismatch: tên miền 2023 bị gán vào cửa sổ 2018.
  (C) Không phản ánh drift của không gian benign theo thời gian.

Giải pháp được triển khai:
─────────────────────────────────────────────────────────────────────
  STRATEGY A (mặc định — "annual_partition"):
      Mỗi năm dùng snapshot Tranco/Alexa của năm đó.
      Pool của năm đó được PHÂN VÙNG (partition, không lặp) thành 4 phần,
      mỗi phần gán cho 1 quý. Đảm bảo không temporal leak trong năm.
      Các năm khác nhau có thể trùng tên miền → cross-year overlap được
      kiểm tra và loại bỏ theo cờ verify_cross_year_overlap.

  STRATEGY B ("single_snapshot"):
      Dùng MỘT snapshot duy nhất (thường Tranco mới nhất).
      Pool được PHÂN VÙNG ngẫu nhiên thành 24 phần không chồng nhau.
      Mỗi phần gán cho 1 cửa sổ. Hoàn toàn không có temporal leak.
      Hạn chế: không phản ánh drift benign qua các năm.

Cả hai chiến lược đều đảm bảo: KHÔNG có tên miền benign nào
xuất hiện ở hơn 1 cửa sổ (integrity check tuyệt đối).

Cấu hình trong config.yaml:
    benign:
      strategy: "annual_partition"   # hoặc "single_snapshot"
      single_snapshot_file: null     # dùng cho strategy B

Usage:
    python -m src.data.step3_merge_benign
    python -m src.data.step3_merge_benign --config configs/config.yaml
    python -m src.data.step3_merge_benign --strategy single_snapshot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, get_path, load_config, quarter_id, quarter_label, get_window_ids, year_of_window, get_base_year

BENIGN_LABEL  = 0
BENIGN_FAMILY = "benign"


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path, logger) -> pd.DataFrame | None:
    """Load a benign CSV with at least a 'domain' column (or first column)."""
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8", on_bad_lines="skip")
        # If no 'domain' column, assume first column is the domain list
        if "domain" not in df.columns:
            df = df.rename(columns={df.columns[0]: "domain"})
        df["domain"] = df["domain"].astype(str).str.strip().str.lower()
        df = df[["domain"]].drop_duplicates().dropna()
        logger.info(f"    Loaded {len(df):>8,} domains ← {path.name}")
        return df
    except Exception as exc:
        logger.warning(f"    Could not load {path.name}: {exc}")
        return None


def _load_for_year(benign_dir: Path, year: int, logger) -> pd.DataFrame | None:
    """Try tranco_{year}.csv then alexa_{year}.csv."""
    for name in [f"tranco_{year}.csv", f"alexa_{year}.csv"]:
        path = benign_dir / name
        if path.exists():
            return _load_csv(path, logger)
    logger.warning(f"    No benign file for year {year} — windows of {year} will have 0 benign rows.")
    return None


def _remove_dga_overlap(df: pd.DataFrame, dga_domains: set[str], logger,
                        label: str = "") -> pd.DataFrame:
    """Remove any domains that appear in the DGA set."""
    before = len(df)
    df = df[~df["domain"].isin(dga_domains)].copy()
    removed = before - len(df)
    if removed:
        logger.info(f"    {label}Cross-contamination removed: {removed:,} domains")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A: annual_partition
# ─────────────────────────────────────────────────────────────────────────────

def strategy_annual_partition(
    cfg: dict,
    window_ids: list[str],
    dga_domains: set[str],
    benign_dir: Path,
    rng: np.random.Generator,
    logger,
) -> dict[str, pd.DataFrame]:
    """
    Cho mỗi năm: load snapshot năm đó → shuffle → chia đều thành 4 phần không chồng nhau
    → mỗi phần = 1 quý.  Cross-year overlap được loại bỏ.

    Đảm bảo:
      ✓ Không temporal leak trong cùng một năm (partition, không sample)
      ✓ Cross-year overlap được xử lý bằng cờ verify_cross_year_overlap
    """
    logger.info("  Strategy: annual_partition")
    max_bw       = cfg["benign"]["max_per_window"]
    verify_cy    = cfg["benign"].get("verify_cross_year_overlap", True)
    result: dict[str, pd.DataFrame] = {}
    all_assigned: set[str] = set()   # tracks domains already assigned (for cross-year)

    base = get_base_year(cfg)
    end_year = year_of_window(window_ids[-1], cfg)
    for year in range(base, end_year + 1):
        # Windows for this year: e.g. year=2018 → D01,D02,D03,D04
        year_windows = [w for w in window_ids
                        if year_of_window(w, cfg) == year]
        if not year_windows:
            continue

        pool = _load_for_year(benign_dir, year, logger)
        if pool is None or pool.empty:
            for w in year_windows:
                result[w] = pd.DataFrame(columns=["domain"])
            continue

        # Remove DGA overlap
        pool = _remove_dga_overlap(pool, dga_domains, logger, label=f"{year} ")

        # Remove cross-year overlap (domains already assigned to earlier years)
        if verify_cy and all_assigned:
            before = len(pool)
            pool = pool[~pool["domain"].isin(all_assigned)].copy()
            removed = before - len(pool)
            if removed:
                logger.info(f"    {year}: Cross-year overlap removed: {removed:,} domains")

        # Shuffle deterministically
        pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

        n_quarters = len(year_windows)
        # Partition pool into n_quarters disjoint slices
        # Each slice has at most max_bw rows
        total_available = len(pool)
        slice_size = min(max_bw, total_available // n_quarters)

        for i, win_id in enumerate(year_windows):
            start = i * slice_size
            end   = start + slice_size
            part  = pool.iloc[start:end].copy()
            result[win_id] = part
            all_assigned.update(part["domain"].tolist())
            logger.info(
                f"    {win_id}: {len(part):>6,} benign domains "
                f"(slice {i+1}/{n_quarters} of {year} pool, "
                f"pool size={total_available:,}, slice_size={slice_size:,})"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B: single_snapshot
# ─────────────────────────────────────────────────────────────────────────────

def strategy_single_snapshot(
    cfg: dict,
    window_ids: list[str],
    dga_domains: set[str],
    benign_dir: Path,
    rng: np.random.Generator,
    logger,
) -> dict[str, pd.DataFrame]:
    """
    Dùng MỘT file benign duy nhất (hoặc tự động chọn file mới nhất).
    Pool được partition thành đúng 24 phần không chồng nhau.

    Đảm bảo:
      ✓ Tuyệt đối không temporal leak (partition cứng)
    Hạn chế:
      ✗ Không phản ánh drift của không gian benign theo năm
    """
    logger.info("  Strategy: single_snapshot")
    max_bw  = cfg["benign"]["max_per_window"]

    # Chọn file snapshot
    snapshot_file = cfg["benign"].get("single_snapshot_file")
    if snapshot_file:
        pool = _load_csv(benign_dir / snapshot_file, logger)
    else:
        # Tự động: ưu tiên Tranco mới nhất
        candidates = sorted(benign_dir.glob("tranco_*.csv"), reverse=True)
        if not candidates:
            candidates = sorted(benign_dir.glob("alexa_*.csv"), reverse=True)
        if not candidates:
            logger.error("No benign snapshot file found in " + str(benign_dir))
            return {w: pd.DataFrame(columns=["domain"]) for w in window_ids}
        pool = _load_csv(candidates[0], logger)
        logger.info(f"    Auto-selected snapshot: {candidates[0].name}")

    if pool is None or pool.empty:
        return {w: pd.DataFrame(columns=["domain"]) for w in window_ids}

    # Remove DGA overlap
    pool = _remove_dga_overlap(pool, dga_domains, logger)

    # Shuffle
    pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

    n_windows  = len(window_ids)
    slice_size = min(max_bw, len(pool) // n_windows)

    if slice_size == 0:
        logger.warning(
            f"    Pool too small ({len(pool):,}) to supply {n_windows} windows "
            f"× {max_bw} domains. Reduce max_per_window or use a larger snapshot."
        )

    result: dict[str, pd.DataFrame] = {}
    for i, win_id in enumerate(window_ids):
        start = i * slice_size
        end   = start + slice_size
        result[win_id] = pool.iloc[start:end].copy()
        logger.info(f"    {win_id}: {len(result[win_id]):>6,} benign domains (slice {i+1}/{n_windows})")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict, strategy_override: str | None = None) -> None:
    log_dir      = Path(cfg["paths"]["results"]) / "logs"
    logger       = get_logger("step3_benign", log_dir=log_dir)

    windows_dir   = Path(cfg["paths"]["windows_dir"])
    benign_dir    = Path(cfg["paths"]["benign_raw"])
    benchmark_dir = get_path(cfg, "benchmark_dir")

    seed = cfg["random_seed"]
    rng  = np.random.default_rng(seed)

    strategy = strategy_override or cfg["benign"].get("strategy", "annual_partition")
    logger.info(f"Benign strategy: {strategy}")
    logger.info(
        "NOTE: Tranco/Alexa do not carry per-domain timestamps.\n"
        f"         Using '{strategy}' to assign benign domains to windows\n"
        "         without temporal leak."
    )

    # Window ids
    window_ids = get_window_ids(cfg)

    # Pre-load all DGA domains for cross-contamination check
    logger.info("Pre-loading DGA domains for cross-contamination check …")
    dga_domains: set[str] = set()
    for win_id in window_ids:
        dga_file = windows_dir / f"{win_id}_dga.csv"
        if dga_file.exists():
            tmp = pd.read_csv(dga_file, usecols=["domain"])
            dga_domains.update(tmp["domain"].str.lower().tolist())
    logger.info(f"  Total DGA domains loaded: {len(dga_domains):,}")

    # Dispatch strategy
    if strategy == "annual_partition":
        benign_by_window = strategy_annual_partition(
            cfg, window_ids, dga_domains, benign_dir, rng, logger)
    elif strategy == "single_snapshot":
        benign_by_window = strategy_single_snapshot(
            cfg, window_ids, dga_domains, benign_dir, rng, logger)
    else:
        raise ValueError(f"Unknown benign strategy: '{strategy}'. "
                         "Choose 'annual_partition' or 'single_snapshot'.")

    # ── Merge DGA + benign, write benchmark files ─────────────────────────────
    logger.info("\nMerging DGA + benign windows …")
    stats_rows = []

    for win_id in window_ids:
        dga_file = windows_dir / f"{win_id}_dga.csv"
        if not dga_file.exists():
            logger.warning(f"  {win_id}: DGA window file missing, skipping.")
            continue

        dga_df   = pd.read_csv(dga_file)
        q_label  = dga_df["quarter_label"].iloc[0] if len(dga_df) else "?"

        benign_pool = benign_by_window.get(win_id, pd.DataFrame(columns=["domain"]))
        n_benign = len(benign_pool)

        if n_benign > 0:
            benign_pool = benign_pool.copy()
            benign_pool["label"]         = BENIGN_LABEL
            benign_pool["family"]        = BENIGN_FAMILY
            benign_pool["quarter_id"]    = win_id
            benign_pool["quarter_label"] = q_label

        combined = pd.concat([dga_df, benign_pool], ignore_index=True)
        combined = combined.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        combined = combined.reset_index(drop=True)
        combined = combined[["domain", "label", "family", "quarter_id", "quarter_label"]]

        out_path = benchmark_dir / f"{win_id}.csv"
        combined.to_csv(out_path, index=False)

        n_dga   = len(dga_df)
        n_total = len(combined)
        ratio   = n_benign / n_total * 100 if n_total else 0
        logger.info(
            f"  {win_id} ({q_label}): DGA={n_dga:>6,}  benign={n_benign:>6,}  "
            f"total={n_total:>7,}  benign%={ratio:4.1f}%  → {out_path.name}"
        )
        stats_rows.append({
            "window_id":     win_id,
            "quarter_label": q_label,
            "strategy":      strategy,
            "n_dga":         n_dga,
            "n_benign":      n_benign,
            "n_total":       n_total,
            "output_file":   out_path.name,
        })

    # ── Final no-duplicate check across benchmark files ───────────────────────
    logger.info("\nVerifying: no benign domain appears in more than 1 window …")
    seen_benign: set[str] = set()
    violations = 0
    for win_id in window_ids:
        path = benchmark_dir / f"{win_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        benign_here = set(df.loc[df["label"] == 0, "domain"].str.lower())
        overlap = benign_here & seen_benign
        if overlap:
            logger.error(f"  {win_id}: {len(overlap)} benign domains already in earlier window! "
                         f"Sample: {list(overlap)[:3]}")
            violations += len(overlap)
        seen_benign.update(benign_here)
    if violations == 0:
        logger.info("  ✓ No benign temporal leak detected")
    else:
        logger.error(f"  ✗ {violations} benign temporal leak violations found")

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(stats_rows)
    stats_path = benchmark_dir / "benchmark_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"\nBenchmark stats → {stats_path}")
    logger.info("Step 3 complete ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Merge benign data into benchmark windows")
    parser.add_argument("--config",   default=None)
    parser.add_argument("--strategy", default=None,
                        choices=["annual_partition", "single_snapshot"],
                        help="Override benign strategy from config")
    args = parser.parse_args()
    run(load_config(args.config), strategy_override=args.strategy)
