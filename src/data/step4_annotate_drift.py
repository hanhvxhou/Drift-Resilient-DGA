"""
src/data/step4_annotate_drift.py
─────────────────────────────────
Step 4 (optional – requires a pretrained char-CNN backbone):
    For each of the 23 inter-window boundaries (D01→D02, D02→D03, …),
    compute the embedding-space MMD² and assign a drift label:
        sudden   / gradual / recurring / none

    Output: data/processed/benchmark/drift_labels.json

    If no backbone is available, a stub version is written with
    drift_type = "UNLABELED" for all boundaries so downstream
    code doesn't break.

Usage:
    # With backbone (requires src/models/char_cnn.py to exist):
    python -m src.data.step4_annotate_drift --backbone results/checkpoints/backbone_d01.pt

    # Without backbone (stub output):
    python -m src.data.step4_annotate_drift --stub
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, quarter_id, quarter_label, get_window_ids

# ── MMD² (biased estimator, Gaussian RBF kernel) ──────────────────────────────
def rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel cross-matrix K(X, Y)."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dist_sq = XX + YY.T - 2 * X @ Y.T
    return np.exp(-dist_sq / (2 * sigma ** 2))


def mmd2_biased(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Biased MMD² estimator with median-heuristic bandwidth.
    X, Y: (n, d) float arrays of embeddings.
    """
    # Median heuristic: sigma = median pairwise distance on reference X
    from scipy.spatial.distance import pdist
    all_pts = np.vstack([X, Y])
    dists   = pdist(all_pts, metric="euclidean")
    sigma   = float(np.median(dists)) if len(dists) else 1.0
    if sigma == 0:
        sigma = 1.0

    Kxx = rbf_kernel_matrix(X, X, sigma)
    Kxy = rbf_kernel_matrix(X, Y, sigma)
    Kyy = rbf_kernel_matrix(Y, Y, sigma)

    n, m = len(X), len(Y)
    return (Kxx.mean() - 2 * Kxy.mean() + Kyy.mean())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


# ── Main ──────────────────────────────────────────────────────────────────────
def run_stub(cfg: dict) -> None:
    """Write drift_labels.json with UNLABELED entries (no backbone required)."""
    log_dir   = Path(cfg["paths"]["results"]) / "logs"
    logger    = get_logger("step4_drift_stub", log_dir=log_dir)
    bench_dir = Path(cfg["paths"]["benchmark_dir"])

    window_ids = get_window_ids(cfg)
    boundaries = []
    for i in range(len(window_ids) - 1):
        boundaries.append({
            "boundary_id":  f"{window_ids[i]}_to_{window_ids[i+1]}",
            "window_from":  window_ids[i],
            "window_to":    window_ids[i + 1],
            "mmd2":         None,
            "cosine_sim_to_history": None,
            "drift_type":   "UNLABELED",
            "note":         "Run step4 with --backbone to compute real labels",
        })

    out_path = bench_dir / "drift_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"boundaries": boundaries,
                   "thresholds": {"delta1": None, "delta2": None, "tau": None, "k": None}},
                  f, indent=2)
    logger.info(f"Stub drift_labels.json → {out_path}  ({len(boundaries)} boundaries)")


def run_with_backbone(cfg: dict, backbone_path: str) -> None:
    """Compute real MMD² values and assign drift labels."""
    import torch
    from src.models.char_cnn import CharCNN   # noqa: imported lazily

    log_dir   = Path(cfg["paths"]["results"]) / "logs"
    logger    = get_logger("step4_drift", log_dir=log_dir)
    bench_dir = Path(cfg["paths"]["benchmark_dir"])

    n_embed  = cfg["drift"]["embedding_samples_per_window"]
    tau      = cfg["drift"]["tau"]
    k        = cfg["drift"]["k"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading backbone from {backbone_path} on {device}")
    model = CharCNN.load(backbone_path, map_location=device)
    model.eval()

    window_ids = get_window_ids(cfg)

    # Compute centroid embeddings per window
    logger.info("Computing per-window embeddings …")
    centroids: dict[str, np.ndarray] = {}
    embeddings_full: dict[str, np.ndarray] = {}

    for win_id in window_ids:
        path = bench_dir / f"{win_id}.csv"
        if not path.exists():
            logger.warning(f"  {win_id}.csv missing, skipping")
            continue
        df = pd.read_csv(path, usecols=["domain"]).sample(
            n=min(n_embed, len(pd.read_csv(path))),
            random_state=cfg["random_seed"]
        )
        domains = df["domain"].tolist()
        with torch.no_grad():
            emb = model.embed(domains, device=device)   # (n, d) numpy
        embeddings_full[win_id] = emb
        centroids[win_id]       = emb.mean(axis=0)
        logger.info(f"  {win_id}: {len(domains)} domains embedded, centroid shape {centroids[win_id].shape}")

    # Calibrate thresholds on D01 (held-out 20% split)
    d01_emb  = embeddings_full.get("D01", np.zeros((1, 128)))
    n        = len(d01_emb)
    split    = int(n * 0.8)
    ref_emb  = d01_emb[:split]
    val_emb  = d01_emb[split:]
    baseline_mmd2 = mmd2_biased(ref_emb, val_emb)
    delta2 = baseline_mmd2 * 2.0
    delta1 = baseline_mmd2 * 5.0
    logger.info(f"Calibrated thresholds: baseline_mmd2={baseline_mmd2:.6f}  "
                f"delta2={delta2:.6f}  delta1={delta1:.6f}  tau={tau}  k={k}")

    # Assign drift labels
    mmd2_series: list[float] = []
    boundaries  = []
    history_centroids: list[np.ndarray] = []

    for i in range(1, len(window_ids)):
        prev_id = window_ids[i - 1]
        curr_id = window_ids[i]

        if prev_id not in embeddings_full or curr_id not in embeddings_full:
            boundaries.append({
                "boundary_id": f"{prev_id}_to_{curr_id}",
                "window_from": prev_id, "window_to": curr_id,
                "mmd2": None, "cosine_sim_to_history": None,
                "drift_type": "UNKNOWN",
            })
            continue

        m2 = mmd2_biased(embeddings_full[prev_id], embeddings_full[curr_id])
        mmd2_series.append(m2)

        # Check recurring: cosine similarity with any historical centroid
        max_cos = max(
            (cosine_similarity(centroids[curr_id], h) for h in history_centroids),
            default=0.0
        )

        # Drift classification
        if m2 < delta2:
            drift_type = "none"
        elif max_cos >= tau:
            drift_type = "recurring"
        elif m2 >= delta1:
            drift_type = "sudden"
        elif len(mmd2_series) >= k and all(v >= delta2 for v in mmd2_series[-k:]):
            drift_type = "gradual"
        else:
            drift_type = "none"

        boundaries.append({
            "boundary_id":           f"{prev_id}_to_{curr_id}",
            "window_from":           prev_id,
            "window_to":             curr_id,
            "mmd2":                  round(m2, 8),
            "cosine_sim_to_history": round(max_cos, 6),
            "drift_type":            drift_type,
        })
        logger.info(f"  {prev_id}→{curr_id}: mmd2={m2:.6f}  cos_hist={max_cos:.3f}  → {drift_type}")

        # After labeling, archive this centroid for future recurring checks
        if drift_type in ("sudden", "none"):
            history_centroids.append(centroids[prev_id])

    def _to_py(obj):
        """Chuyển numpy types sang Python native để JSON serialize."""
        import numpy as np
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)):     return int(obj)
        return obj

    # Convert toàn bộ boundaries
    boundaries_clean = [
        {k: _to_py(v) for k, v in b.items()} for b in boundaries
    ]

    out_path = bench_dir / "drift_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "boundaries": boundaries_clean,
            "thresholds": {
                "delta1": float(round(delta1, 8)),
                "delta2": float(round(delta2, 8)),
                "tau":    float(tau),
                "k":      int(k),
            },
        }, f, indent=2)

    # Summary
    from collections import Counter
    counts = Counter(b["drift_type"] for b in boundaries)
    logger.info(f"\nDrift label summary: {dict(counts)}")
    logger.info(f"drift_labels.json → {out_path}")
    logger.info("Step 4 complete ✓")

    # Write thresholds back to config for reference
    cfg["drift"]["delta1"] = round(delta1, 8)
    cfg["drift"]["delta2"] = round(delta2, 8)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Annotate drift labels")
    parser.add_argument("--config",   default=None, help="Path to config.yaml")
    parser.add_argument("--backbone", default=None,
                        help="Path to pretrained CharCNN checkpoint (.pt)")
    parser.add_argument("--stub",     action="store_true",
                        help="Write stub drift_labels.json without backbone")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    if args.stub or args.backbone is None:
        run_stub(cfg)
    else:
        run_with_backbone(cfg, args.backbone)
