"""
src/models/drift_analysis.py
──────────────────────────────
Buoc 4: t-SNE/UMAP embedding drift visualization
Buoc 5: Error analysis per-family

Usage:
    python -m src.models.drift_analysis
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from scipy.special import expit as sigmoid_stable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.char_cnn import CharCNN, domains_to_batch
from src.detect.add_detector import extract_embeddings
from src.utils.common import get_logger, load_config, get_window_ids
from src.utils.dga_taxonomy import WORD_BASED_FAMILIES


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("drift_analysis", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    backbone_path = out_dir / "checkpoints" / "backbone_d01.pt"
    window_ids = get_window_ids(cfg)

    logger.info("=" * 65)
    logger.info(" DRIFT ANALYSIS: t-SNE + Per-Family Error Analysis")
    logger.info("=" * 65)

    model = CharCNN.load(backbone_path, map_location=device).to(device)
    model.eval()

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: t-SNE Embedding Drift Visualization
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n  PART 1: t-SNE Embedding Drift")
    
    # Select 4 representative windows: D01 (start), D07 (pre-drift), D08 (post-drift), D24 (end)
    vis_windows = ["D01", "D07", "D08", "D24"]
    vis_labels = ["D01 (2018Q1)", "D07 (2019Q3, pre-drift)", "D08 (2019Q4, post-drift)", "D24 (2023Q4)"]
    vis_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    
    all_embs = []
    all_win_labels = []
    all_class_labels = []  # DGA vs benign
    
    for win_id in vis_windows:
        test_path = split_dir / f"{win_id}_test.csv"
        if not test_path.exists():
            logger.warning(f"  {win_id} not found, skipping")
            continue
        
        df = pd.read_csv(test_path)
        # Sample 1000 per window for t-SNE speed
        n_sample = min(1000, len(df))
        df_sample = df.sample(n=n_sample, random_state=42)
        
        embs = extract_embeddings(model, df_sample["domain"].tolist(), device=device, max_n=n_sample)
        all_embs.append(embs)
        all_win_labels.extend([win_id] * len(embs))
        all_class_labels.extend(df_sample["label"].tolist())
    
    if all_embs:
        X = np.vstack(all_embs)
        logger.info(f"  Total embeddings: {X.shape}")
        
        # t-SNE
        logger.info("  Running t-SNE (perplexity=30)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        X_2d = tsne.fit_transform(X)
        
        # Plot: color by window
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # (a) Color by temporal window
        ax = axes[0]
        offset = 0
        for i, (win_id, label, color) in enumerate(zip(vis_windows, vis_labels, vis_colors)):
            mask = np.array(all_win_labels) == win_id
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=8, alpha=0.5, label=label)
        ax.legend(fontsize=9, markerscale=3)
        ax.set_title("(a) Embedding drift across temporal windows", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        
        # (b) Color by DGA/benign
        ax = axes[1]
        class_arr = np.array(all_class_labels)
        ax.scatter(X_2d[class_arr==0, 0], X_2d[class_arr==0, 1], c='#4CAF50', s=8, alpha=0.4, label='Benign')
        ax.scatter(X_2d[class_arr==1, 0], X_2d[class_arr==1, 1], c='#F44336', s=8, alpha=0.4, label='DGA')
        ax.legend(fontsize=9, markerscale=3)
        ax.set_title("(b) DGA vs Benign distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        
        plt.suptitle("Figure 5: t-SNE visualization of embedding space drift", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        tsne_path = out_dir / "figure5_tsne_drift.png"
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.savefig(out_dir / "figure5_tsne_drift.pdf", bbox_inches='tight')
        logger.info(f"  Saved: {tsne_path}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Per-Family Error Analysis
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n  PART 2: Per-Family Error Analysis (Static-CNN on D24)")
    
    # Evaluate static model on D24 test, analyze per-family
    test_d24 = pd.read_csv(split_dir / "D24_test.csv")
    domains = test_d24["domain"].tolist()
    labels = np.array(test_d24["label"].tolist())
    families = test_d24["family"].tolist()
    
    # Get predictions
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(domains), 512):
            x = domains_to_batch(domains[i:i+512]).to(device)
            all_logits.append(model(x).cpu().numpy())
    probs = sigmoid_stable(np.concatenate(all_logits))
    preds = (probs >= 0.5).astype(int)
    
    # Per-family F1
    family_results = []
    unique_fams = sorted(set(families))
    
    for fam in unique_fams:
        if fam == "benign":
            continue
        mask = np.array([f == fam for f in families])
        n = mask.sum()
        if n < 5:
            continue
        
        # For this family: true label=1 (all DGA), check if predicted correctly
        correct = (preds[mask] == 1).sum()
        recall = correct / n  # recall = detection rate for this family
        ftype = "word" if fam in WORD_BASED_FAMILIES else "char"
        
        family_results.append({
            "family": fam, "type": ftype, "n_samples": int(n),
            "recall": round(float(recall), 4),
            "missed": int(n - correct),
        })
    
    family_df = pd.DataFrame(family_results).sort_values("recall")
    
    # Top 10 worst families
    logger.info("\n  Top 10 WORST detected families (Static-CNN, D24):")
    logger.info(f"  {'Family':<24} {'Type':<6} {'N':>6} {'Recall':>8} {'Missed':>8}")
    logger.info(f"  {'-'*54}")
    for _, r in family_df.head(10).iterrows():
        logger.info(f"  {r['family']:<24} {r['type']:<6} {r['n_samples']:>6} {r['recall']:>8.4f} {r['missed']:>8}")
    
    # Top 10 best families
    logger.info("\n  Top 10 BEST detected families:")
    for _, r in family_df.tail(10).iterrows():
        logger.info(f"  {r['family']:<24} {r['type']:<6} {r['n_samples']:>6} {r['recall']:>8.4f} {r['missed']:>8}")
    
    # Summary by type
    logger.info("\n  Summary by DGA type:")
    for dtype in ["char", "word"]:
        sub = family_df[family_df['type'] == dtype]
        if len(sub) > 0:
            avg_recall = sub['recall'].mean()
            n_fam = len(sub)
            n_total = sub['n_samples'].sum()
            logger.info(f"  {dtype:<6} {n_fam} families, {n_total} samples, avg_recall={avg_recall:.4f}")
    
    # ── Plot per-family analysis ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # (a) Worst 15 families bar chart
    ax = axes[0]
    worst15 = family_df.head(15)
    colors = ['#F44336' if t == 'word' else '#2196F3' for t in worst15['type']]
    bars = ax.barh(range(len(worst15)), worst15['recall'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(worst15)))
    ax.set_yticklabels(worst15['family'], fontsize=9)
    ax.set_xlabel('Detection Recall', fontsize=11)
    ax.set_title('(a) 15 Worst-Detected Families\n(Static-CNN on D24)', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color='#F44336'), Patch(color='#2196F3')], 
             ['Word-based', 'Char-based'], fontsize=9)
    ax.invert_yaxis()
    
    # (b) Recall distribution by type
    ax = axes[1]
    char_recalls = family_df[family_df['type']=='char']['recall']
    word_recalls = family_df[family_df['type']=='word']['recall']
    
    bp = ax.boxplot([char_recalls, word_recalls], tick_labels=['Character-based', 'Word-based'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][1].set_facecolor('#F44336')
    for b in bp['boxes']:
        b.set_alpha(0.6)
    ax.set_ylabel('Detection Recall', fontsize=11)
    ax.set_title('(b) Recall Distribution by DGA Type\n(Static-CNN on D24)', fontsize=12, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle("Figure 6: Per-Family Error Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    err_path = out_dir / "figure6_error_analysis.png"
    plt.savefig(err_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / "figure6_error_analysis.pdf", bbox_inches='tight')
    logger.info(f"\n  Saved: {err_path}")
    
    # Save data
    family_df.to_csv(out_dir / "per_family_analysis.csv", index=False)
    logger.info(f"  Saved: {out_dir / 'per_family_analysis.csv'}")
    logger.info("\n  Drift analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config))
