"""
src/models/evaluate_cl.py
──────────────────────────
Đánh giá DRC-CL theo chuẩn Continual Learning:
    - BWT (Backward Transfer): đo quên thảm họa
    - FWT (Forward Transfer): đo tổng quát hóa
    - AA-F1, AA-AUC: hiệu suất trung bình qua 24 cửa sổ
    - ADD-F1: độ chính xác phát hiện drift
    - Update latency: thời gian cập nhật mỗi sự kiện

Có thể chạy độc lập sau khi DRC-CL đã chạy xong,
hoặc so sánh với baseline Static-CNN, SW-Retrain.

Usage:
    # Đánh giá DRC-CL (sau khi chạy drc_cl_runner.py):
    python -m src.models.evaluate_cl --results results/drc_cl_per_window.csv

    # So sánh nhiều phương pháp:
    python -m src.models.evaluate_cl --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from scipy.special import expit as sigmoid_stable
import torch
from torch.utils.data import DataLoader

from src.models.char_cnn import CharCNN, domains_to_batch
from src.models.lora_adapter import CharCNNWithLoRA
from src.utils.common import get_logger, get_window_ids, load_config


# ── Metric computations ───────────────────────────────────────────────────────
def compute_bwt(a_matrix: dict[tuple, float], T: int) -> float:
    """
    BWT = (1/(T-1)) Σ_{i=1}^{T-1} [a(T, i) - a(i, i)]
    a_matrix: dict (train_t, eval_s) → f1
    Cần đã đánh giá tất cả windows sau khi train xong window cuối.
    """
    vals = []
    for i in range(T - 1):
        key_ii = (i, i)
        key_Ti = (T - 1, i)
        if key_ii in a_matrix and key_Ti in a_matrix:
            vals.append(a_matrix[key_Ti] - a_matrix[key_ii])
    return float(np.mean(vals)) if vals else 0.0


def compute_fwt(a_matrix: dict[tuple, float],
                b_vector: dict[int, float],
                T: int) -> float:
    """
    FWT = (1/(T-1)) Σ_{i=1}^{T-1} [a(i-1, i) - b_i]
    b_vector: zero-shot F1 của model khởi tạo ngẫu nhiên (≈ 0.5 cho balanced data)
    """
    vals = []
    for i in range(1, T):
        key = (i - 1, i)
        if key in a_matrix and i in b_vector:
            vals.append(a_matrix[key] - b_vector[i])
    return float(np.mean(vals)) if vals else 0.0


# ── Full a-matrix evaluation (chính xác hơn BWT approximation) ──────────────
@torch.no_grad()
def build_a_matrix(model: CharCNNWithLoRA,
                   bench_dir: Path,
                   window_ids: list[str],
                   device: str = "cuda",
                   batch_size: int = 512) -> dict[tuple, float]:
    """
    Đánh giá model hiện tại trên tất cả windows để lấy hàng cuối của a-matrix.
    Gọi sau khi model đã train xong toàn bộ chuỗi.
    """
    model.eval()
    a = {}
    T = len(window_ids)
    for s, win_id in enumerate(window_ids):
        df = pd.read_csv(bench_dir / f"{win_id}.csv")
        domains = df["domain"].tolist()
        labels  = np.array(df["label"].tolist())
        all_logits = []
        for i in range(0, len(domains), batch_size):
            x = domains_to_batch(domains[i:i+batch_size]).to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
        logits_np = np.concatenate(all_logits)
        probs  = sigmoid_stable(logits_np)
        preds  = (probs >= 0.5).astype(int)
        f1     = f1_score(labels, preds, zero_division=0)
        a[(T - 1, s)] = f1
    return a


# ── Static-CNN baseline ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_static_cnn(backbone_path: Path,
                        bench_dir:     Path,
                        window_ids:    list[str],
                        device:        str = "cuda",
                        batch_size:    int = 512) -> list[dict]:
    """
    Đánh giá Static-CNN: train 1 lần trên D01, không bao giờ update.
    """
    model = CharCNN.load(backbone_path, map_location=device).to(device)
    model.eval()
    results = []
    for win_id in window_ids:
        df      = pd.read_csv(bench_dir / f"{win_id}.csv")
        domains = df["domain"].tolist()
        labels  = np.array(df["label"].tolist())
        all_logits = []
        for i in range(0, len(domains), batch_size):
            x = domains_to_batch(domains[i:i+batch_size]).to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
        logits_np = np.concatenate(all_logits)
        probs = sigmoid_stable(logits_np)
        preds = (probs >= 0.5).astype(int)
        results.append({
            "method":       "Static-CNN",
            "window_id":    win_id,
            "quarter_label": df["quarter_label"].iloc[0],
            "f1":   f1_score(labels, preds, zero_division=0),
            "auc":  roc_auc_score(labels, probs),
        })
    return results


# ── Summary report ────────────────────────────────────────────────────────────
def compute_summary(per_window: list[dict], method: str = "DRC-CL") -> dict:
    """Tính AA-F1, AA-AUC, BWT xấp xỉ từ per-window results."""
    f1s  = [r["f1"]  for r in per_window]
    aucs = [r["auc"] for r in per_window]
    T    = len(f1s)

    aa_f1  = float(np.mean(f1s))
    aa_auc = float(np.mean(aucs))

    # BWT: f1 cuối - f1 đầu cho mỗi window (approximation)
    bwt = float(f1s[-1] - np.mean(f1s[:-1])) if T > 1 else 0.0

    # Degradation: giảm F1 từ đầu đến cuối
    degradation = float(f1s[0] - f1s[-1]) if T > 1 else 0.0

    return {
        "method":      method,
        "aa_f1":       round(aa_f1,  4),
        "aa_auc":      round(aa_auc, 4),
        "bwt":         round(bwt,    4),
        "f1_first":    round(f1s[0], 4),
        "f1_last":     round(f1s[-1],4),
        "f1_min":      round(min(f1s),4),
        "f1_max":      round(max(f1s),4),
        "degradation": round(degradation, 4),
        "n_windows":   T,
    }


def print_comparison_table(summaries: list[dict], logger) -> None:
    """In bảng so sánh theo format IEEE."""
    headers = ["Method", "AA-F1", "AA-AUC", "BWT", "F1-First", "F1-Last", "Degrad."]
    rows = []
    for s in summaries:
        rows.append([
            s["method"],
            f"{s['aa_f1']:.4f}",
            f"{s['aa_auc']:.4f}",
            f"{s['bwt']:+.4f}",
            f"{s['f1_first']:.4f}",
            f"{s['f1_last']:.4f}",
            f"{s['degradation']:+.4f}",
        ])

    col_w = [max(len(h), max(len(r[i]) for r in rows)) + 2
             for i, h in enumerate(headers)]
    line  = "─" * sum(col_w)

    logger.info(f"\n{'='*60}")
    logger.info(" COMPARISON TABLE (IEEE format)")
    logger.info(f"{'='*60}")
    logger.info(line)
    logger.info("".join(h.ljust(col_w[i]) for i, h in enumerate(headers)))
    logger.info(line)
    for row in rows:
        logger.info("".join(v.ljust(col_w[i]) for i, v in enumerate(row)))
    logger.info(line)


# ── Per-window plot helper ─────────────────────────────────────────────────────
def save_f1_over_time(results_by_method: dict[str, list[dict]],
                      out_path: Path) -> None:
    """
    Lưu CSV so sánh F1 theo cửa sổ của nhiều phương pháp.
    Dễ import vào Excel/matplotlib để vẽ Figure 2 trong paper.
    """
    rows = []
    for method, per_window in results_by_method.items():
        for r in per_window:
            rows.append({
                "method":       method,
                "window_id":    r["window_id"],
                "quarter_label": r.get("quarter_label", ""),
                "f1":           r["f1"],
                "auc":          r.get("auc", 0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


# ── Main runner ───────────────────────────────────────────────────────────────
def run(cfg: dict,
        drc_cl_results_path: Optional[str] = None,
        compare_static: bool = True) -> None:

    log_dir   = Path(cfg["paths"]["results"]) / "logs"
    logger    = get_logger("evaluate_cl", log_dir=log_dir)
    bench_dir = Path(cfg["paths"]["benchmark_dir"])
    ckpt_dir  = Path(cfg["paths"]["results"]) / "checkpoints"
    out_dir   = Path(cfg["paths"]["results"])
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    window_ids = get_window_ids(cfg)

    summaries = []
    results_by_method = {}

    # ── Load DRC-CL per-window results ────────────────────────────────────────
    drc_path = drc_cl_results_path or str(out_dir / "drc_cl_per_window.csv")
    if Path(drc_path).exists():
        drc_pw  = pd.read_csv(drc_path).to_dict("records")
        s_drc   = compute_summary(drc_pw, "DRC-CL")
        summaries.append(s_drc)
        results_by_method["DRC-CL"] = drc_pw
        logger.info(f"Loaded DRC-CL results: {len(drc_pw)} windows")
    else:
        logger.warning(f"DRC-CL results not found at {drc_path}")
        logger.warning("Chạy drc_cl_runner.py trước.")

    # ── Static-CNN baseline ────────────────────────────────────────────────────
    backbone_path = ckpt_dir / "backbone_d01.pt"
    if compare_static and backbone_path.exists():
        logger.info("Đánh giá Static-CNN baseline ...")
        static_pw = evaluate_static_cnn(backbone_path, bench_dir, window_ids, device)
        s_static  = compute_summary(static_pw, "Static-CNN")
        summaries.append(s_static)
        results_by_method["Static-CNN"] = static_pw

        # Lưu static results
        pd.DataFrame(static_pw).to_csv(out_dir / "static_cnn_per_window.csv", index=False)
        logger.info(f"Static-CNN AA-F1={s_static['aa_f1']:.4f}  "
                    f"Degradation={s_static['degradation']:+.4f}")

    # ── In bảng so sánh ───────────────────────────────────────────────────────
    if summaries:
        print_comparison_table(summaries, logger)

        # Lưu comparison CSV
        comp_path = out_dir / "comparison_table.csv"
        pd.DataFrame(summaries).to_csv(comp_path, index=False)
        logger.info(f"\nComparison table → {comp_path}")

    # ── Lưu F1-over-time CSV ──────────────────────────────────────────────────
    if results_by_method:
        f1t_path = out_dir / "f1_over_time.csv"
        save_f1_over_time(results_by_method, f1t_path)
        logger.info(f"F1-over-time CSV  → {f1t_path}")
        logger.info("(Import vào Excel/matplotlib để vẽ Figure 2 trong paper)")

    logger.info("\nevaluate_cl.py complete ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────
from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DRC-CL continual learning metrics")
    parser.add_argument("--config",  default=None)
    parser.add_argument("--results", default=None,
                        help="Path to drc_cl_per_window.csv")
    parser.add_argument("--no-static", action="store_true",
                        help="Bỏ qua Static-CNN baseline")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    run(cfg,
        drc_cl_results_path = args.results,
        compare_static      = not args.no_static)
