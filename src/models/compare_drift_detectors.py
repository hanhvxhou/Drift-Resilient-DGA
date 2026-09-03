"""
src/models/compare_drift_detectors.py
──────────────────────────────────────
So sanh ADD voi cac drift detectors khac:
  - ADD (MMD2, unsupervised, embedding-space)
  - ADWIN (supervised, error-rate)
  - DDM (supervised, error-rate)
  - EDDM (supervised, error-rate)
  - Page-Hinkley (supervised, error-rate)
  - KSWIN (supervised, error-rate window)

Ground truth: drift khi F1 giam > 2pp so voi cua so truoc.

Yeu cau: pip install river

Usage:
    python -m src.models.compare_drift_detectors
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.special import expit as sigmoid_stable

from src.models.char_cnn import CharCNN, domains_to_batch
from src.detect.add_detector import ADDDetector, extract_embeddings
from src.utils.common import get_logger, load_config, get_window_ids


# ── Evaluate model per window to get error stream ─────────────────────────────
@torch.no_grad()
def get_prediction_stream(model, split_dir, window_ids, device, batch_size=512):
    """
    Run model on all windows, return per-sample predictions and labels.
    Returns: list of dicts {window_id, domains, labels, preds, errors, f1}
    """
    model.eval()
    results = []
    for win_id in window_ids:
        test_path = split_dir / f"{win_id}_test.csv"
        if not test_path.exists():
            continue
        df = pd.read_csv(test_path)
        domains = df["domain"].tolist()
        labels = np.array(df["label"].tolist())

        all_logits = []
        for i in range(0, len(domains), batch_size):
            x = domains_to_batch(domains[i:i+batch_size]).to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())

        logits_np = np.concatenate(all_logits)
        probs = sigmoid_stable(logits_np)
        preds = (probs >= 0.5).astype(int)
        errors = (preds != labels).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)

        results.append({
            "window_id": win_id, "labels": labels, "preds": preds,
            "errors": errors, "f1": f1, "n_errors": int(errors.sum()),
        })
    return results


# ── Define ground truth: drift when F1 drops > threshold ──────────────────────
def compute_ground_truth(stream_results, threshold=0.02):
    """
    Drift = True when F1 drops by more than threshold vs previous window.
    Returns: list of bool (length T-1), one per transition.
    """
    gt = []
    for i in range(1, len(stream_results)):
        f1_prev = stream_results[i-1]["f1"]
        f1_curr = stream_results[i]["f1"]
        drop = f1_prev - f1_curr
        gt.append(drop > threshold)
    return gt


# ── Run supervised drift detectors (ADWIN, DDM, etc.) ─────────────────────────
def run_supervised_detector(detector_name, stream_results):
    """
    Feed error stream to a supervised drift detector from river.
    Returns: list of bool (drift detected at each transition).
    """
    from river import drift

    detector_map = {
        "ADWIN": lambda: drift.ADWIN(delta=0.002),
        "DDM": lambda: drift.DDM(min_num_instances=100),
        "EDDM": lambda: drift.EDDM(),
        "PageHinkley": lambda: drift.PageHinkley(delta=0.005, threshold=50),
    }

    if detector_name not in detector_map:
        return [False] * (len(stream_results) - 1)

    det = detector_map[detector_name]()
    detections = []

    for i in range(1, len(stream_results)):
        # Feed all errors from this window to detector
        window_errors = stream_results[i]["errors"]
        drift_in_window = False

        for err in window_errors:
            det.update(err)
            if det.drift_detected:
                drift_in_window = True
                # Reset detector after detection
                det = detector_map[detector_name]()
                break

        detections.append(drift_in_window)

    return detections


# ── Run KSWIN (Kolmogorov-Smirnov Windowing) ─────────────────────────────────
def run_kswin(stream_results):
    """KSWIN: uses KS test on sliding windows of predictions."""
    from river import drift
    det = drift.KSWIN(alpha=0.005, window_size=300, stat_size=100)
    detections = []

    for i in range(1, len(stream_results)):
        window_errors = stream_results[i]["errors"]
        drift_in_window = False

        for err in window_errors:
            det.update(err)
            if det.drift_detected:
                drift_in_window = True
                det = drift.KSWIN(alpha=0.005, window_size=300, stat_size=100)
                break

        detections.append(drift_in_window)

    return detections


# ── Run ADD (our method, unsupervised) ────────────────────────────────────────
def run_add_detector(model, split_dir, window_ids, device):
    """Run ADD on embedding space. Returns: list of bool (drift detected)."""
    add = ADDDetector(max_no_update=4)
    detections = []

    # Calibrate on D01
    train_d01 = pd.read_csv(split_dir / f"{window_ids[0]}_train.csv")
    ref_embs = extract_embeddings(model, train_d01["domain"].tolist(),
                                   device=device, max_n=5000)
    add.calibrate(ref_embs)

    for i in range(1, len(window_ids)):
        train_df = pd.read_csv(split_dir / f"{window_ids[i]}_train.csv")
        embs = extract_embeddings(model, train_df["domain"].tolist(),
                                   device=device, max_n=5000)
        event = add.detect(embs)
        detections.append(event.needs_update)

    return detections


# ── Compute detection metrics ─────────────────────────────────────────────────
def compute_detection_metrics(gt, pred):
    """Compute P, R, F1 for drift detection."""
    gt_arr = np.array(gt, dtype=int)
    pred_arr = np.array(pred, dtype=int)
    if gt_arr.sum() == 0:
        return {"precision": 0, "recall": 0, "f1": 0,
                "n_gt": 0, "n_pred": int(pred_arr.sum())}
    p = precision_score(gt_arr, pred_arr, zero_division=0)
    r = recall_score(gt_arr, pred_arr, zero_division=0)
    f = f1_score(gt_arr, pred_arr, zero_division=0)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "n_gt": int(gt_arr.sum()), "n_pred": int(pred_arr.sum())}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("drift_compare", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    backbone_path = out_dir / "checkpoints" / "backbone_d01.pt"
    window_ids = get_window_ids(cfg)

    logger.info("=" * 65)
    logger.info(" DRIFT DETECTOR COMPARISON")
    logger.info("=" * 65)
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Windows: {len(window_ids)}")

    # Load model
    model = CharCNN.load(backbone_path, map_location=device).to(device)

    # Get prediction stream (static model, never updated)
    logger.info("\n  Computing prediction stream on static model...")
    stream = get_prediction_stream(model, split_dir, window_ids, device)

    # F1 per window
    logger.info("\n  F1 per window (static model):")
    for s in stream:
        logger.info(f"    {s['window_id']}: F1={s['f1']:.4f}  errors={s['n_errors']}/20000")

    # Ground truth: drift when F1 drops > 2%
    gt_2pp = compute_ground_truth(stream, threshold=0.02)
    gt_5pp = compute_ground_truth(stream, threshold=0.05)
    n_drift_2 = sum(gt_2pp)
    n_drift_5 = sum(gt_5pp)
    logger.info(f"\n  Ground truth: {n_drift_2} drifts (>2pp), {n_drift_5} drifts (>5pp)")

    # ── Run all detectors ─────────────────────────────────────────────────
    all_results = []

    # ADD (unsupervised)
    logger.info("\n  Running ADD (unsupervised, MMD2)...")
    t0 = time.time()
    add_pred = run_add_detector(model, split_dir, window_ids, device)
    t_add = time.time() - t0
    m_add = compute_detection_metrics(gt_2pp, add_pred)
    m_add["method"] = "ADD (ours)"
    m_add["type"] = "unsupervised"
    m_add["time_s"] = round(t_add, 2)
    all_results.append(m_add)
    logger.info(f"    P={m_add['precision']:.3f}  R={m_add['recall']:.3f}  "
                f"F1={m_add['f1']:.3f}  detected={m_add['n_pred']}/{len(gt_2pp)}")

    # Supervised detectors
    supervised_detectors = ["ADWIN", "DDM", "EDDM", "PageHinkley"]

    for det_name in supervised_detectors:
        logger.info(f"\n  Running {det_name} (supervised, error-rate)...")
        t0 = time.time()
        try:
            pred = run_supervised_detector(det_name, stream)
            t_det = time.time() - t0
            m = compute_detection_metrics(gt_2pp, pred)
            m["method"] = det_name
            m["type"] = "supervised"
            m["time_s"] = round(t_det, 2)
            all_results.append(m)
            logger.info(f"    P={m['precision']:.3f}  R={m['recall']:.3f}  "
                        f"F1={m['f1']:.3f}  detected={m['n_pred']}/{len(gt_2pp)}")
        except Exception as e:
            logger.warning(f"    {det_name} failed: {e}")
            all_results.append({"method": det_name, "type": "supervised",
                                "f1": 0, "error": str(e)})

    # KSWIN
    logger.info("\n  Running KSWIN (supervised, KS test)...")
    t0 = time.time()
    try:
        kswin_pred = run_kswin(stream)
        t_ks = time.time() - t0
        m_ks = compute_detection_metrics(gt_2pp, kswin_pred)
        m_ks["method"] = "KSWIN"
        m_ks["type"] = "supervised"
        m_ks["time_s"] = round(t_ks, 2)
        all_results.append(m_ks)
        logger.info(f"    P={m_ks['precision']:.3f}  R={m_ks['recall']:.3f}  "
                    f"F1={m_ks['f1']:.3f}  detected={m_ks['n_pred']}/{len(gt_2pp)}")
    except Exception as e:
        logger.warning(f"    KSWIN failed: {e}")

    # ── Print comparison table ────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(" DRIFT DETECTION COMPARISON (ground truth: F1 drop > 2pp)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Method':<16} {'Type':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Detected':>10} {'Time':>8}")
    logger.info(f"  {'-'*70}")
    for r in all_results:
        if 'error' in r:
            logger.info(f"  {r['method']:<16} {r['type']:<14} {'ERROR':>10}")
        else:
            logger.info(f"  {r['method']:<16} {r['type']:<14} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                        f"{r['f1']:>8.3f} {r['n_pred']:>10} {r['time_s']:>7.2f}s")
    logger.info(f"  {'-'*70}")
    logger.info(f"  Ground truth: {n_drift_2} drift events in {len(gt_2pp)} transitions")

    # ── Key advantage: ADD is unsupervised ────────────────────────────────
    logger.info(f"\n  KEY FINDING:")
    logger.info(f"  ADD operates UNSUPERVISED (on embeddings, no labels needed).")
    logger.info(f"  All other methods require SUPERVISED feedback (prediction errors).")
    logger.info(f"  In DGA detection, labels have weeks/months delay [12].")
    logger.info(f"  => ADD is the only viable option for real-time deployment.")

    # ── Detail: which windows each method detected ────────────────────────
    logger.info(f"\n  Detection detail per transition:")
    logger.info(f"  {'Transition':<12} {'GT':>4} {'ADD':>5} {'ADWIN':>7} {'DDM':>5} {'EDDM':>6} {'PH':>5} {'KSWIN':>7}")
    logger.info(f"  {'-'*55}")

    all_preds = {"ADD": add_pred}
    for det_name in supervised_detectors:
        try:
            all_preds[det_name] = run_supervised_detector(det_name, stream)
        except:
            all_preds[det_name] = [False] * len(gt_2pp)
    try:
        all_preds["KSWIN"] = run_kswin(stream)
    except:
        all_preds["KSWIN"] = [False] * len(gt_2pp)

    for i in range(len(gt_2pp)):
        w1 = window_ids[i]
        w2 = window_ids[i+1]
        gt_str = "DRIFT" if gt_2pp[i] else "—"
        vals = [("✓" if all_preds.get(m, [False]*len(gt_2pp))[i] else "—")
                for m in ["ADD", "ADWIN", "DDM", "EDDM", "PageHinkley", "KSWIN"]]
        logger.info(f"  {w1}→{w2}  {gt_str:>5} {vals[0]:>5} {vals[1]:>7} {vals[2]:>5} {vals[3]:>6} {vals[4]:>5} {vals[5]:>7}")

    # Save
    pd.DataFrame(all_results).to_csv(out_dir / "drift_detector_comparison.csv", index=False)
    logger.info(f"\n  Saved: {out_dir / 'drift_detector_comparison.csv'}")
    logger.info("  Drift detector comparison complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare drift detection methods")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)
