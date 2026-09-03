"""
src/detect/add_detector.py — Fixed version
───────────────────────────
Adaptive Drift Detector with:
  - Bootstrap percentile calibration (stable across dims)
  - max_no_update safeguard (force update after N consecutive none)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist


def mmd2_biased(X: np.ndarray, Y: np.ndarray) -> float:
    """MMD2 with median-heuristic bandwidth."""
    all_pts = np.vstack([X, Y])
    dists = cdist(all_pts, all_pts, metric="euclidean")
    idx = np.triu_indices(len(all_pts), k=1)
    sigma = float(np.median(dists[idx]))
    if sigma < 1e-10:
        sigma = 1.0
    def rbf(A, B):
        return np.exp(-cdist(A, B, "sqeuclidean") / (2 * sigma ** 2))
    return float(rbf(X,X).mean() - 2*rbf(X,Y).mean() + rbf(Y,Y).mean())


@dataclass
class DriftEvent:
    drift_type: str
    mmd2: float
    window_idx: int
    forced: bool = False

    @property
    def is_drift(self): return self.drift_type != "none"

    @property
    def needs_update(self): return self.drift_type != "none" or self.forced


class ADDDetector:
    def __init__(self, delta1=None, delta2=None, tau=0.85, k=3,
                 n_ref=5000, max_no_update=4):
        self.delta1 = delta1
        self.delta2 = delta2
        self.tau = tau
        self.k = k
        self.n_ref = n_ref
        self.max_no_update = max_no_update  # safeguard

        self.ref_embeddings = None
        self.window_idx = 0
        self.consecutive_no_update = 0      # counter for safeguard
        self.mmd2_history = []
        self.calibrated = False
        self.event_log = []

    def calibrate(self, ref_embeddings: np.ndarray, n_bootstrap: int = 50) -> None:
        """
        Bootstrap percentile calibration (stable across dimensions).
        Split ref into random halves n_bootstrap times, compute MMD2 distribution,
        set delta2 = 95th percentile, delta1 = 99th percentile.
        """
        rng = np.random.default_rng(42)
        n = len(ref_embeddings)
        mmd2_null = []

        for _ in range(n_bootstrap):
            perm = rng.permutation(n)
            half = n // 2
            X = ref_embeddings[perm[:half]]
            Y = ref_embeddings[perm[half:half*2]]
            m2 = mmd2_biased(X, Y)
            mmd2_null.append(m2)

        mmd2_null = np.array(mmd2_null)
        self.delta2 = float(np.percentile(mmd2_null, 95))
        self.delta1 = float(np.percentile(mmd2_null, 99))

        # Ensure minimum thresholds (avoid zero)
        if self.delta2 < 1e-8:
            self.delta2 = 1e-6
        if self.delta1 < self.delta2 * 1.5:
            self.delta1 = self.delta2 * 3.0

        self.ref_embeddings = ref_embeddings
        self.calibrated = True

    def set_reference(self, embeddings: np.ndarray):
        self.ref_embeddings = embeddings

    def detect(self, curr_embeddings: np.ndarray) -> DriftEvent:
        if not self.calibrated or self.ref_embeddings is None:
            raise RuntimeError("Call calibrate() first")

        self.window_idx += 1
        m2 = mmd2_biased(curr_embeddings, self.ref_embeddings)
        self.mmd2_history.append(m2)

        # Classify drift
        if m2 >= self.delta1:
            drift_type = "sudden"
        elif m2 >= self.delta2:
            # Check if sustained (gradual)
            if (len(self.mmd2_history) >= self.k and
                    all(v >= self.delta2 for v in self.mmd2_history[-self.k:])):
                drift_type = "gradual"
            else:
                drift_type = "moderate"  # above delta2 but not sustained
        else:
            drift_type = "none"

        # Safeguard: force update after max_no_update consecutive "none"
        forced = False
        if drift_type == "none":
            self.consecutive_no_update += 1
            if self.consecutive_no_update >= self.max_no_update:
                forced = True
                self.consecutive_no_update = 0
        else:
            self.consecutive_no_update = 0

        event = DriftEvent(
            drift_type=drift_type, mmd2=m2,
            window_idx=self.window_idx, forced=forced
        )
        self.event_log.append(event)

        # Update reference after drift
        self.ref_embeddings = curr_embeddings

        return event

    def summary(self):
        from collections import Counter
        counts = Counter(e.drift_type for e in self.event_log)
        n_forced = sum(1 for e in self.event_log if e.forced)
        return {"drift_counts": dict(counts), "forced_updates": n_forced,
                "delta1": self.delta1, "delta2": self.delta2}

    @classmethod
    def from_config(cls, cfg: dict) -> "ADDDetector":
        drift_cfg = cfg.get("drift", {})
        return cls(
            delta1=drift_cfg.get("delta1"),
            delta2=drift_cfg.get("delta2"),
            tau=drift_cfg.get("tau", 0.85),
            k=drift_cfg.get("k", 3),
            n_ref=drift_cfg.get("embedding_samples_per_window", 5000),
            max_no_update=drift_cfg.get("max_no_update", 4),
        )


def extract_embeddings(model, domains, device="cuda", batch_size=512, max_n=5000):
    """Extract embeddings from CharCNN model."""
    import torch
    if len(domains) > max_n:
        idx = np.random.choice(len(domains), max_n, replace=False)
        domains = [domains[i] for i in idx]
    model.eval()
    return model.embed(domains, device=device, batch_size=batch_size)
