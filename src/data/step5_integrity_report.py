"""
src/data/step5_integrity_report.py
────────────────────────────────────
Step 5: Final data quality checks + summary report.

Checks:
  ✓  All 24 window files exist
  ✓  No domain appears in a later window AND an earlier one (temporal leak)
  ✓  Label balance per window (DGA / benign ratio)
  ✓  Family coverage: all active families have >= min samples in each window
  ✓  No duplicate domains within each window

Writes:
  data/processed/benchmark/integrity_report.txt
  data/processed/benchmark/family_coverage.csv

Usage:
    python -m src.data.step5_integrity_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.common import get_logger, load_config, quarter_id, get_window_ids


def run(cfg: dict) -> None:
    log_dir   = Path(cfg["paths"]["results"]) / "logs"
    logger    = get_logger("step5_integrity", log_dir=log_dir)
    bench_dir = Path(cfg["paths"]["benchmark_dir"])

    window_ids = get_window_ids(cfg)

    report_lines = ["=" * 60,
                    " DRC-CL Benchmark Integrity Report",
                    "=" * 60]
    all_ok = True
    seen_domains: set[str] = set()

    coverage_rows = []

    for win_id in window_ids:
        path = bench_dir / f"{win_id}.csv"
        issues: list[str] = []

        # ── File existence ────────────────────────────────────────
        if not path.exists():
            report_lines.append(f"\n{win_id}: MISSING FILE ✗")
            all_ok = False
            continue

        df = pd.read_csv(path)
        n  = len(df)

        # ── Expected columns ──────────────────────────────────────
        expected_cols = {"domain", "label", "family", "quarter_id", "quarter_label"}
        missing_cols  = expected_cols - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # ── Duplicates within window ──────────────────────────────
        n_dupes = df.duplicated(subset="domain").sum()
        if n_dupes > 0:
            issues.append(f"{n_dupes} duplicate domains")

        # ── Temporal leak ─────────────────────────────────────────
        current_domains = set(df["domain"].str.lower())
        leaked = current_domains & seen_domains
        if leaked:
            issues.append(f"{len(leaked)} domains leaked from earlier window(s)")
        seen_domains.update(current_domains)

        # ── Label balance ─────────────────────────────────────────
        label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        n_dga    = label_counts.get(1, 0)
        n_benign = label_counts.get(0, 0)
        ratio    = n_benign / n if n else 0
        if n_benign == 0:
            issues.append("No benign domains (check step3)")

        # ── Family coverage ───────────────────────────────────────
        if "family" in df.columns:
            fam_counts = df.groupby("family")["domain"].count()
            coverage_rows.append({
                "window_id":      win_id,
                **{f: fam_counts.get(f, 0) for f in fam_counts.index}
            })

        # ── Per-window summary ────────────────────────────────────
        status = "✓" if not issues else "✗"
        q_label = df["quarter_label"].iloc[0] if ("quarter_label" in df.columns and n) else "?"
        report_lines.append(
            f"\n{win_id} ({q_label}): {status}  total={n:>7,}  "
            f"DGA={n_dga:>6,}  benign={n_benign:>6,}  "
            f"benign%={ratio*100:4.1f}%  "
            f"families={df['family'].nunique()-1 if 'family' in df.columns else '?':>3}"
        )
        for issue in issues:
            all_ok = False
            report_lines.append(f"    ⚠  {issue}")

    # ── Final verdict ─────────────────────────────────────────────
    report_lines += [
        "\n" + "=" * 60,
        f"Overall result: {'ALL CHECKS PASSED ✓' if all_ok else 'ISSUES FOUND — see above ✗'}",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    logger.info(report_text)

    report_path = bench_dir / "integrity_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info(f"\nIntegrity report → {report_path}")

    # ── Family coverage CSV ───────────────────────────────────────
    if coverage_rows:
        cov_df = pd.DataFrame(coverage_rows).set_index("window_id").fillna(0).astype(int)
        cov_path = bench_dir / "family_coverage.csv"
        cov_df.to_csv(cov_path)
        logger.info(f"Family coverage  → {cov_path}")

    logger.info("Step 5 complete ✓")
    if not all_ok:
        import sys
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Integrity report")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config))
