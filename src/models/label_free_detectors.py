"""
src/models/label_free_detectors.py
──────────────────────────────────
E6 — Label-free (unsupervised) drift detectors on the embedding space, added
to answer Reviewer #1 Concern #2:
  (a) label-free comparators for ADD, and
  (b) fixed / sliding / hybrid reference windows.

All detectors work ONLY on embeddings (no labels), using the SAME
bootstrap-percentile calibration as ADD so the comparison is fair: on D01,
split embeddings into random halves n_bootstrap times, take the 95th percentile
of the null two-sample statistic as the drift threshold.

Detectors:
  - ADD (MMD2)         with reference = fixed / sliding / hybrid
  - KS-test            two-sample Kolmogorov-Smirnov on pooled 1-D projection
  - Energy distance    two-sample energy statistic

Reference modes:
  - fixed  : compare each window to D01 (original ADD behaviour)
  - sliding: compare each window to the PREVIOUS window
  - hybrid : drift if EITHER (vs D01) OR (vs previous window) exceeds threshold
             — catches both long-horizon cumulative drift and abrupt shifts

Returns, per detector, a list[bool] of length T-1 (one decision per transition),
directly comparable with compute_ground_truth / compute_detection_metrics in
compare_drift_detectors.py.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from src.detect.add_detector import mmd2_biased, extract_embeddings


# ── two-sample statistics (all label-free) ────────────────────────────────────
def _mmd2(X, Y):
    return mmd2_biased(X, Y)

def _ks(X, Y):
    """Max KS statistic over embedding dims (pooled), a scalar distance."""
    d = min(X.shape[1], 32)  # cap dims for speed; KS per dim then take max
    stats = [ks_2samp(X[:, j], Y[:, j]).statistic for j in range(d)]
    return float(np.max(stats))

def _energy(X, Y):
    """Two-sample energy distance (Szekely-Rizzo), label-free."""
    def _pdist_mean(A, B):
        # mean pairwise Euclidean distance between rows of A and B
        # subsample for speed if large
        na, nb = len(A), len(B)
        ia = np.random.choice(na, min(na, 500), replace=False)
        ib = np.random.choice(nb, min(nb, 500), replace=False)
        A2, B2 = A[ia], B[ib]
        d = np.sqrt(((A2[:, None, :] - B2[None, :, :]) ** 2).sum(-1))
        return d.mean()
    xy = _pdist_mean(X, Y)
    xx = _pdist_mean(X, X)
    yy = _pdist_mean(Y, Y)
    return float(2 * xy - xx - yy)


STATS = {"MMD2": _mmd2, "KS": _ks, "Energy": _energy}


def _calibrate_threshold(ref_embs, stat_fn, n_bootstrap=50, pct=95, seed=42):
    """Bootstrap-percentile null threshold, same scheme as ADD.calibrate."""
    rng = np.random.default_rng(seed)
    n = len(ref_embs)
    null = []
    for _ in range(n_bootstrap):
        idx = rng.permutation(n)
        half = n // 2
        X = ref_embs[idx[:half]]
        Y = ref_embs[idx[half:2 * half]]
        null.append(stat_fn(X, Y))
    return float(np.percentile(null, pct))


def run_label_free(model, split_dir, window_ids, device,
                   stat_name="MMD2", ref_mode="fixed",
                   max_n=5000, seed=42):
    """
    Run one label-free detector over all transitions.
    Returns list[bool] length T-1 (drift decision per transition).
    """
    import pandas as pd
    from pathlib import Path
    split_dir = Path(split_dir)
    stat_fn = STATS[stat_name]

    # embeddings for every window (train split), cached in a list
    embs = []
    for w in window_ids:
        df = pd.read_csv(split_dir / f"{w}_train.csv")
        embs.append(extract_embeddings(model, df["domain"].tolist(),
                                        device=device, max_n=max_n))

    # calibrate threshold on D01 (same for all reference modes)
    thr = _calibrate_threshold(embs[0], stat_fn, seed=seed)

    detections = []
    for i in range(1, len(window_ids)):
        cur = embs[i]
        if ref_mode == "fixed":
            drift = stat_fn(cur, embs[0]) >= thr
        elif ref_mode == "sliding":
            drift = stat_fn(cur, embs[i - 1]) >= thr
        elif ref_mode == "hybrid":
            drift = (stat_fn(cur, embs[0]) >= thr) or (stat_fn(cur, embs[i - 1]) >= thr)
        else:
            raise ValueError(f"unknown ref_mode {ref_mode}")
        detections.append(bool(drift))
    return detections


# Configurations added to Table X (all label-free / unsupervised)
LABEL_FREE_CONFIGS = [
    ("ADD-MMD2 (fixed ref)",   "MMD2",   "fixed"),
    ("ADD-MMD2 (sliding ref)", "MMD2",   "sliding"),
    ("ADD-MMD2 (hybrid ref)",  "MMD2",   "hybrid"),
    ("KS-test (sliding ref)",  "KS",     "sliding"),
    ("Energy dist (sliding)",  "Energy", "sliding"),
]
