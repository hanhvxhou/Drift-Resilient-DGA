"""
src/models/gen_roc_pr.py
─────────────────────────
Tao ROC curve va Precision-Recall curve cho cac phuong phap chinh.
Danh gia tren D24_test (cua so cuoi cung) sau khi model da hoc 24 windows.

Usage:
    python -m src.models.gen_roc_pr
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit as sigmoid_stable
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.char_cnn import CharCNN, domains_to_batch
from src.utils.common import get_logger, load_config, get_window_ids


@torch.no_grad()
def get_probs_cnn(model, domains, device, batch_size=512):
    model.eval()
    all_logits = []
    for i in range(0, len(domains), batch_size):
        x = domains_to_batch(domains[i:i+batch_size]).to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
    return sigmoid_stable(np.concatenate(all_logits))


@torch.no_grad()
def get_probs_distilbert(model, domains, tokenizer, device, batch_size=64):
    from torch.amp import autocast
    model.eval()
    all_logits = []
    for i in range(0, len(domains), batch_size):
        enc = tokenizer(domains[i:i+batch_size], padding="max_length",
                        truncation=True, max_length=64, return_tensors="pt")
        with autocast("cuda", enabled=(device=="cuda")):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_logits.append(logits.cpu().numpy())
    return sigmoid_stable(np.concatenate(all_logits))


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("roc_pr", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    window_ids = get_window_ids(cfg)

    # Load D24 test
    test_df = pd.read_csv(split_dir / f"{window_ids[-1]}_test.csv")
    domains = test_df["domain"].tolist()
    labels = np.array(test_df["label"].tolist())

    logger.info("=" * 60)
    logger.info(" ROC & PRECISION-RECALL CURVES (D24 test)")
    logger.info("=" * 60)

    methods = {}

    # 1. Static-CNN (backbone only, no updates)
    bp = out_dir / "checkpoints" / "backbone_d01.pt"
    if bp.exists():
        logger.info("  Loading Static-CNN...")
        model_static = CharCNN.load(bp, map_location=device).to(device)
        methods["Static-CNN"] = get_probs_cnn(model_static, domains, device)
        del model_static

    # 2. DRC-CL (CharCNN) — need to find the final model
    # Look for latest accuracy matrix to confirm it ran
    drc_matrix = out_dir / "drc-cl_accuracy_matrix.csv"
    if not drc_matrix.exists():
        drc_matrix = out_dir / "drc_cl_accuracy_matrix.csv"

    # 3. Try loading DistilBERT models
    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # DistilBERT Static
        db_ckpt = out_dir / "checkpoints" / "distilbert_d01.pt"
        if db_ckpt.exists():
            logger.info("  Loading DistilBERT Static...")
            from src.models.distilbert_baseline import DistilBERTClassifier
            model_db = DistilBERTClassifier().to(device)
            model_db.load_weights(db_ckpt, device)
            methods["DistilBERT Static"] = get_probs_distilbert(model_db, domains, tokenizer, device)
            del model_db
    except Exception as e:
        logger.warning(f"  DistilBERT load failed: {e}")

    if not methods:
        logger.warning("  No models found. Run experiments first.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────
    colors = {
        "Static-CNN": "#888780",
        "DRC-CL (CharCNN)": "#185FA5",
        "EWC-only": "#534AB7",
        "DistilBERT Static": "#D85A30",
        "DistilBERT+FT": "#993556",
        "DRC-CL (DistilBERT)": "#1D9E75",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    ax_roc = axes[0]
    ax_roc.plot([0,1], [0,1], 'k--', lw=0.5, alpha=0.3, label='Random')
    for name, probs in methods.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        c = colors.get(name, "gray")
        ax_roc.plot(fpr, tpr, color=c, lw=2, label=f'{name} (AUC={roc_auc:.4f})')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('(a) ROC Curve (D24 test)', fontsize=13, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.grid(True, alpha=0.2)

    # Precision-Recall
    ax_pr = axes[1]
    for name, probs in methods.items():
        prec, rec, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        c = colors.get(name, "gray")
        ax_pr.plot(rec, prec, color=c, lw=2, label=f'{name} (AP={ap:.4f})')
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('(b) Precision-Recall Curve (D24 test)', fontsize=13, fontweight='bold')
    ax_pr.legend(loc='lower left', fontsize=9)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save
    out_png = out_dir / "figure3_roc_pr_curves.png"
    out_pdf = out_dir / "figure3_roc_pr_curves.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    logger.info(f"\n  Saved: {out_png}")
    logger.info(f"  Saved: {out_pdf}")

    # Print AUC / AP table
    logger.info(f"\n  {'Method':<24} {'AUC-ROC':>10} {'AP':>10} {'F1':>10}")
    logger.info(f"  {'-'*54}")
    for name, probs in methods.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(labels, probs)
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        logger.info(f"  {name:<24} {roc_auc:>10.4f} {ap:>10.4f} {f1:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config))
