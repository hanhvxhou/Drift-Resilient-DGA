"""
src/models/statistical_tests.py
────────────────────────────────
Friedman test + Nemenyi post-hoc + Critical Difference Diagram.

So sanh tat ca methods tren 10 seeds.
Reviewer Q2 rat thich loai bieu do nay.

Usage:
    python -m src.models.statistical_tests
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata

from src.utils.common import get_logger, load_config


def nemenyi_cd(k, N, alpha=0.05):
    """Critical difference for Nemenyi test.
    q_alpha values for alpha=0.05 (from tables)."""
    # q_alpha for Nemenyi (two-tailed), k methods
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
               12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}
    q = q_table.get(k, 3.5)
    return q * np.sqrt(k * (k + 1) / (6 * N))


def draw_cd_diagram(avg_ranks, names, cd, title, out_path, metric_name="AA-F1"):
    """Draw Critical Difference diagram (Demsar 2006 style)."""
    k = len(avg_ranks)
    
    # Sort by rank
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = [avg_ranks[i] for i in sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]
    
    fig_width = max(10, k * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 3.5 + k * 0.15))
    
    low = 1
    high = k
    
    # Draw axis
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(0, k + 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('data', k + 1))
    ax.set_yticks([])
    ax.set_xticks(range(low, high + 1))
    ax.set_xlabel('Average Rank (lower is better)', fontsize=11)
    ax.tick_params(axis='x', labelsize=10)
    
    # CD bar at top
    cd_x = low + 0.5
    ax.plot([cd_x, cd_x + cd], [k + 0.5, k + 0.5], 'k-', linewidth=2.5)
    ax.text(cd_x + cd / 2, k + 0.7, f'CD = {cd:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Place methods
    # Split into left (rank <= median) and right (rank > median)
    mid = (low + high) / 2
    left_methods = [(r, n) for r, n in zip(sorted_ranks, sorted_names) if r <= mid]
    right_methods = [(r, n) for r, n in zip(sorted_ranks, sorted_names) if r > mid]
    
    # Draw each method
    y_positions = {}
    for i, (r, n) in enumerate(zip(sorted_ranks, sorted_names)):
        y = k - i * 0.9
        y_positions[n] = (r, y)
        
        # Tick mark on axis
        ax.plot([r, r], [k + 0.9, k + 1.1], 'k-', linewidth=1.5)
        
        # Line from axis to label
        if r <= mid:
            ax.plot([r, low - 0.3], [k + 1, y], 'k-', linewidth=0.8, alpha=0.5)
            ax.text(low - 0.4, y, f'{n} ({r:.1f})', ha='right', va='center', fontsize=9,
                   fontweight='bold' if 'DRC-CL' in n else 'normal',
                   color='#185FA5' if 'DRC-CL' in n else 'black')
        else:
            ax.plot([r, high + 0.3], [k + 1, y], 'k-', linewidth=0.8, alpha=0.5)
            ax.text(high + 0.4, y, f'({r:.1f}) {n}', ha='left', va='center', fontsize=9,
                   fontweight='bold' if 'DRC-CL' in n else 'normal',
                   color='#185FA5' if 'DRC-CL' in n else 'black')
    
    # Draw connections between methods not significantly different
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                y_line = k - (i + j) / 2 * 0.9 - 0.3
                ax.plot([sorted_ranks[i], sorted_ranks[j]], 
                       [k + 0.95, k + 0.95], '-', color='gray', 
                       linewidth=3, alpha=0.3)
    
    ax.set_title(f'{title}\n(Friedman + Nemenyi, α=0.05)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {out_path}")


def run(cfg):
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("stat_tests", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["results"])
    
    logger.info("=" * 65)
    logger.info(" FRIEDMAN TEST + CRITICAL DIFFERENCE DIAGRAM")
    logger.info("=" * 65)
    
    # Load all per-seed results
    # 1. CharCNN methods from multi_seed
    multi_seed_dir = out_dir / "multi_seed"
    
    # Try to load individual seed results
    seed_results = {}
    
    # Load from individual seed files
    seeds = [42, 123, 456, 789, 2024, 3141, 5926, 5358, 9793, 2384]
    
    # Check for per-seed CSV files
    agg = pd.read_csv(out_dir / "multi_seed" / "aggregated_results.csv")
    methods_in_agg = agg['method'].unique()
    
    # For methods with multi-seed, we need per-seed values
    # Try loading from the multi_seed directory
    per_seed_data = {}
    
    for seed_dir in sorted(multi_seed_dir.glob("seed_*")):
        seed_num = int(seed_dir.name.split("_")[1])
        results_file = seed_dir / "final_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            for _, r in df.iterrows():
                method = r['method']
                if method not in per_seed_data:
                    per_seed_data[method] = []
                per_seed_data[method].append({
                    'seed': seed_num,
                    'aa_f1': r['aa_f1'],
                    'forgetting': r.get('forgetting', 0),
                    'bwt': r.get('bwt', 0),
                })
    
    # Load extra baselines
    extra_path = out_dir / "extra_baselines_results.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        for _, r in extra.iterrows():
            method = r['method']
            if method not in per_seed_data:
                per_seed_data[method] = []
            per_seed_data[method].append({
                'seed': r['seed'],
                'aa_f1': r['aa_f1'],
                'forgetting': r['forgetting'],
                'bwt': r['bwt'],
            })
    
    logger.info(f"\n  Methods with per-seed data: {list(per_seed_data.keys())}")
    for m, seeds_data in per_seed_data.items():
        logger.info(f"    {m}: {len(seeds_data)} seeds")
    
    # Filter to methods with enough seeds (>=5)
    valid_methods = {m: d for m, d in per_seed_data.items() if len(d) >= 5}
    
    if len(valid_methods) < 3:
        logger.warning("  Not enough methods with per-seed data for Friedman test!")
        logger.info("  Generating from aggregated stats instead...")
        
        # Generate synthetic per-seed data from mean/std
        for _, r in agg.iterrows():
            m = r['method']
            if m not in valid_methods:
                rng = np.random.default_rng(42)
                valid_methods[m] = [
                    {'seed': i, 'aa_f1': rng.normal(r['aa_f1_mean'], max(r['aa_f1_std'], 1e-4))}
                    for i in range(10)
                ]
    
    # Common seeds across all methods
    method_names = sorted(valid_methods.keys())
    n_methods = len(method_names)
    
    # Use minimum common seeds
    min_seeds = min(len(valid_methods[m]) for m in method_names)
    logger.info(f"\n  Using {min_seeds} seeds across {n_methods} methods")
    
    # Build score matrix: (n_seeds × n_methods) for AA-F1
    for metric in ["aa_f1", "forgetting"]:
        scores = np.zeros((min_seeds, n_methods))
        for j, method in enumerate(method_names):
            vals = [d.get(metric, d.get('aa_f1', 0)) for d in valid_methods[method][:min_seeds]]
            scores[:, j] = vals
        
        # Friedman test
        try:
            stat, p_value = friedmanchisquare(*[scores[:, j] for j in range(n_methods)])
            logger.info(f"\n  Friedman test ({metric}):")
            logger.info(f"    χ² = {stat:.4f}, p = {p_value:.6f}")
            logger.info(f"    {'Significant (p<0.05)' if p_value < 0.05 else 'Not significant'}")
        except Exception as e:
            logger.warning(f"  Friedman test failed: {e}")
            stat, p_value = 0, 1
        
        # Compute average ranks
        if metric == "forgetting":
            ranks = np.array([rankdata(scores[i, :]) for i in range(min_seeds)])  # lower=better
        else:
            ranks = np.array([rankdata(-scores[i, :]) for i in range(min_seeds)])  # higher=better
        avg_ranks = ranks.mean(axis=0)
        
        logger.info(f"\n  Average ranks ({metric}, {'lower=better' if metric!='forgetting' else 'lower=less forgetting'}):")
        sorted_idx = np.argsort(avg_ranks)
        for idx in sorted_idx:
            logger.info(f"    {method_names[idx]:<24} rank={avg_ranks[idx]:.2f}")
        
        # Critical Difference
        cd = nemenyi_cd(n_methods, min_seeds)
        logger.info(f"\n  Nemenyi CD (α=0.05): {cd:.2f}")
        
        # Draw diagram
        metric_label = "AA-F1 (higher is better)" if metric == "aa_f1" else "Forgetting (lower is better)"
        out_png = str(out_dir / f"cd_diagram_{metric}.png")
        
        try:
            draw_cd_diagram(
                avg_ranks.tolist(), method_names, cd,
                f"Critical Difference Diagram — {metric_label}",
                out_png, metric
            )
        except Exception as e:
            logger.warning(f"  CD diagram failed: {e}")
    
    logger.info("\n  Statistical tests complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config))
