"""
src/models/gen_figures.py
──────────────────────────
Tao Figure 2: F1-over-time cho tat ca methods tren 24 windows.
Doc diagonal cua accuracy matrix files.

Usage:
    python -m src.models.gen_figures
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def run(results_dir=None):
    if results_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(results_dir)

    # Method -> file prefix mapping
    methods = {
        "Static-CNN":           "static-cnn",
        "EWC-only":             "ewc-only",
        "iCaRL":                "icarl",
        "SW-Retrain":           "sw-retrain",
        "DRC-CL (CharCNN)":     "drc-cl",
        "DRC-CL (DistilBERT)":  "drc_cl_distilbert",
    }

    # Colors and styles
    styles = {
        "Static-CNN":           {"color": "#888780", "ls": "--",  "lw": 1.5, "marker": ""},
        "EWC-only":             {"color": "#534AB7", "ls": "-",   "lw": 2,   "marker": "s"},
        "iCaRL":                {"color": "#1D9E75", "ls": "-",   "lw": 1.5, "marker": "^"},
        "SW-Retrain":           {"color": "#D85A30", "ls": "-.",  "lw": 1.5, "marker": ""},
        "DRC-CL (CharCNN)":     {"color": "#185FA5", "ls": "-",   "lw": 2.5, "marker": "o"},
        "DRC-CL (DistilBERT)":  {"color": "#993556", "ls": "-",   "lw": 2.5, "marker": "D"},
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for method, prefix in methods.items():
        # Try to find accuracy matrix file
        candidates = [
            results_dir / f"{prefix}_accuracy_matrix.csv",
            results_dir / f"{prefix.replace('-','_')}_accuracy_matrix.csv",
            results_dir / f"{prefix.replace(' ','_').lower()}_accuracy_matrix.csv",
        ]
        matrix_path = None
        for c in candidates:
            if c.exists():
                matrix_path = c
                break

        if matrix_path is None:
            print(f"  WARNING: Matrix not found for {method}, tried: {[str(c) for c in candidates]}")
            continue

        df = pd.read_csv(matrix_path, index_col=0)
        # Diagonal = F1 at time of learning
        diag = [df.iloc[i, i] for i in range(min(df.shape))]
        windows = list(range(1, len(diag) + 1))

        s = styles.get(method, {"color": "gray", "ls": "-", "lw": 1})
        ax.plot(windows, diag, label=method,
                color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
                marker=s.get("marker", ""), markersize=4, markevery=2)

    ax.set_xlabel("Window (quarterly, 2018Q1 – 2023Q4)", fontsize=12)
    ax.set_ylabel("F1 Score (diagonal of accuracy matrix)", fontsize=12)
    ax.set_title("Figure 2: Detection performance over 24 temporal windows", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_xlim(1, 24)
    ax.set_ylim(0.80, 1.00)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3)

    # Add year markers
    for yr, x in [(2019, 5), (2020, 9), (2021, 13), (2022, 17), (2023, 21)]:
        ax.axvline(x=x-0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.text(x+0.5, 0.805, str(yr), fontsize=8, color="gray", alpha=0.7)

    plt.tight_layout()
    out_path = results_dir / "figure2_f1_over_time.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "figure2_f1_over_time.pdf", bbox_inches="tight")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {results_dir / 'figure2_f1_over_time.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    run(args.results_dir)
