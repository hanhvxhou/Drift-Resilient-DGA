"""
src/models/compute_statistics.py
────────────────────────────────
R1.3 (statistics) — confidence intervals, effect sizes, and multiple-comparison
correction, computed from the existing per-seed results. No GPU / no retraining.

Reviewer #1 Concern #3 asked for: confidence intervals, effect sizes, and
corrected significance tests. This script adds, for CharCNN methods (10 seeds):
  1. 95% confidence interval for AA-F1 (t-interval, small-sample correct)
  2. Cliff's delta effect size vs DRC-CL (non-parametric, paired by seed)
  3. Wilcoxon signed-rank p-values vs DRC-CL, with Holm correction across the
     family of comparisons (controls family-wise error rate)

Data: results/multi_seed/all_seeds_raw.json (9 CharCNN methods x 10 seeds).
DER++/AGEM (5 seeds, results/extra_baselines_results.csv) are reported with CI
only and excluded from p<0.01 significance claims, since the exact Wilcoxon test
cannot reach p<0.01 with 5 paired observations (p_min = 2/2^5 = 0.0625).

Usage
-----
    python -m src.models.compute_statistics
Outputs (results/multi_seed/)
    statistics_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REFERENCE = "DRC-CL"
METRIC = "aa_f1"


def t_ci95(x):
    """95% t-confidence interval for the mean (correct for small n)."""
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, m, m
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, df=n - 1)
    return m, m - h, m + h


def cliffs_delta(a, b):
    """Cliff's delta effect size between paired samples a (method) and b (ref).
    Positive => a tends to be larger than b. |d|: <0.147 negligible,
    <0.33 small, <0.474 medium, else large (Romano et al.)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    d = (gt - lt) / (len(a) * len(b))
    ad = abs(d)
    mag = ("negligible" if ad < 0.147 else "small" if ad < 0.33
           else "medium" if ad < 0.474 else "large")
    return d, mag


def holm_correction(pairs):
    """pairs: list of (name, p). Returns list of (name, p, p_holm, reject@0.05)."""
    order = sorted(pairs, key=lambda t: t[1])
    m = len(order)
    out = []
    prev = 0.0
    for i, (name, p) in enumerate(order):
        p_adj = min(1.0, (m - i) * p)
        p_adj = max(p_adj, prev)  # enforce monotonicity
        prev = p_adj
        out.append((name, p, p_adj, p_adj < 0.05))
    return out


def run(cfg=None):
    results_dir = Path("results/multi_seed")
    J = json.loads((results_dir / "all_seeds_raw.json").read_text())
    seeds = sorted(J.keys(), key=lambda s: int(s))
    methods = [m for m in J[seeds[0]].keys()]

    # per-method AA-F1 across seeds, paired by seed
    vals = {m: np.array([J[s][m][METRIC] for s in seeds]) for m in methods}

    # ── 1. 95% CI per method ──
    rows = []
    for m in methods:
        mean, lo, hi = t_ci95(vals[m])
        rows.append({"method": m, "n": len(vals[m]),
                     "aa_f1_mean": round(mean, 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                     "ci95_halfwidth": round((hi - lo) / 2, 4)})

    # ── 2. Cliff's delta + 3. Wilcoxon vs DRC-CL ──
    ref = vals[REFERENCE]
    wilcox = []
    for m in methods:
        if m == REFERENCE:
            continue
        d, mag = cliffs_delta(vals[m], ref)
        # exact Wilcoxon signed-rank, paired by seed
        try:
            stat, p = stats.wilcoxon(vals[m], ref, alternative="two-sided",
                                     zero_method="wilcox", mode="exact")
        except Exception:
            stat, p = np.nan, np.nan
        wilcox.append((m, p, d, mag))

    # Holm correction across the family of comparisons vs DRC-CL
    holm = holm_correction([(m, p) for m, p, _, _ in wilcox])
    holm_map = {name: (p, padj, rej) for name, p, padj, rej in holm}

    for r in rows:
        m = r["method"]
        if m == REFERENCE:
            r["cliffs_delta_vs_DRC"] = 0.0
            r["effect_mag"] = "-"
            r["wilcoxon_p"] = "-"
            r["holm_p"] = "-"
            r["sig_holm_0.05"] = "-"
        else:
            d, mag = next((d, mg) for mm, _, d, mg in wilcox if mm == m)
            p, padj, rej = holm_map[m]
            r["cliffs_delta_vs_DRC"] = round(d, 3)
            r["effect_mag"] = mag
            r["wilcoxon_p"] = round(p, 4) if not np.isnan(p) else "nan"
            r["holm_p"] = round(padj, 4) if not np.isnan(padj) else "nan"
            r["sig_holm_0.05"] = bool(rej)

    df = pd.DataFrame(rows)
    out = results_dir / "statistics_summary.csv"
    df.to_csv(out, index=False)

    # console summary
    print("=" * 92)
    print(f"  STATISTICS (CharCNN, {len(seeds)} seeds) — 95% CI, Cliff's delta & Wilcoxon vs {REFERENCE}")
    print("=" * 92)
    print(f"  {'method':<24}{'AA-F1':>9}{'95% CI':>20}{'Cliff d':>9}{'Holm p':>9}{'sig':>6}")
    print("  " + "-" * 88)
    for r in rows:
        ci = f"[{r['ci95_low']:.4f},{r['ci95_high']:.4f}]"
        sig = "" if r["method"] == REFERENCE else ("yes" if r["sig_holm_0.05"] else "no")
        cd = "-" if r["method"] == REFERENCE else f"{r['cliffs_delta_vs_DRC']:+.2f}"
        hp = "-" if r["method"] == REFERENCE else f"{r['holm_p']}"
        print(f"  {r['method']:<24}{r['aa_f1_mean']:>9.4f}{ci:>20}{cd:>9}{hp:>9}{sig:>6}")
    print(f"\n  Saved -> {out}")
    print("  Note: Holm controls family-wise error across all comparisons vs DRC-CL.")
    print("  DER++/AGEM (5 seeds) are reported elsewhere with CI only; the exact")
    print("  Wilcoxon test cannot reach p<0.01 with 5 pairs (p_min=0.0625).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.parse_args()
    run()
