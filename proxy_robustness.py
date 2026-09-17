"""
proxy_robustness.py  —  R1.5 (proxy 2018 robustness, KIỂU 2)

Robustness check for the 2018 benign proxy: recompute the CL metrics on the
2019-onward sub-benchmark (windows D05-D24, which use REAL Tranco snapshots)
and compare to the full D01-D24 result. If they agree closely, the proxy used
for D01-D04 does not distort the conclusions.

This reads the EXISTING 24x24 accuracy matrices — no GPU / no retraining.
The 2019+ metrics are computed from the sub-matrix A[4:,4:].

Place at project root. Reads results/*_accuracy_matrix.csv.

Usage
-----
    python proxy_robustness.py
Outputs: results/proxy_robustness.csv
"""
from __future__ import annotations
import glob, os, re
from pathlib import Path
import numpy as np, pandas as pd


def cl_metrics(A):
    """AA-F1, BWT, Forgetting from an accuracy matrix (lower-tri filled)."""
    T = A.shape[0]
    diag = [A[i, i] for i in range(T) if not np.isnan(A[i, i])]
    aa = float(np.mean(diag)) if diag else np.nan
    # BWT: mean_{i<T-1} A[T-1,i] - A[i,i]
    bwt_v = [A[T-1, i] - A[i, i] for i in range(T-1)
             if not np.isnan(A[T-1, i]) and not np.isnan(A[i, i])]
    bwt = float(np.mean(bwt_v)) if bwt_v else np.nan
    # Forgetting: mean_{i<T-1} (max_{j>=i} A[j,i]) - A[T-1,i]
    fg = []
    for i in range(T-1):
        col = [A[j, i] for j in range(i, T) if not np.isnan(A[j, i])]
        if col:
            fg.append(max(col) - col[-1])
    forg = float(np.mean(fg)) if fg else np.nan
    return aa, bwt, forg


def run(results_dir="results", start=4, out="results/proxy_robustness.csv"):
    files = sorted(glob.glob(os.path.join(results_dir, "*_accuracy_matrix.csv")))
    if not files:
        print(f"  No *_accuracy_matrix.csv in {results_dir}")
        return
    rows = []
    for f in files:
        method = re.sub(r"_accuracy_matrix\.csv$", "", os.path.basename(f))
        A = pd.read_csv(f, index_col=0).values.astype(float)
        aa_f, bwt_f, fg_f = cl_metrics(A)               # full D01-D24
        aa_s, bwt_s, fg_s = cl_metrics(A[start:, start:])  # D05-D24 (2019+)
        rows.append({
            "method": method,
            "aa_f1_full": round(aa_f, 4), "aa_f1_2019on": round(aa_s, 4),
            "d_aa_f1": round(aa_s - aa_f, 4),
            "forg_full": round(fg_f, 4), "forg_2019on": round(fg_s, 4),
            "d_forg": round(fg_s - fg_f, 4),
        })
    df = pd.DataFrame(rows).sort_values("method")
    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("=" * 78)
    print("  PROXY ROBUSTNESS: full (D01-D24) vs 2019-onward (D05-D24, real benign)")
    print("=" * 78)
    print(f"  {'method':<26}{'AA-F1 full':>11}{'AA-F1 2019+':>13}{'ΔAA-F1':>9}{'Δforg':>9}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['method']:<26}{r['aa_f1_full']:>11.4f}{r['aa_f1_2019on']:>13.4f}"
              f"{r['d_aa_f1']:>+9.4f}{r['d_forg']:>+9.4f}")
    max_d = max(abs(r["d_aa_f1"]) for r in rows)
    print(f"\n  Max |ΔAA-F1| across methods = {max_d:.4f}")
    print(f"  Small deltas => the 2018 benign proxy does not distort conclusions.")
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--start", type=int, default=4, help="first window index for 2019+ (D05=index 4)")
    ap.add_argument("--out", default="results/proxy_robustness.csv")
    a = ap.parse_args()
    run(a.results_dir, a.start, a.out)
