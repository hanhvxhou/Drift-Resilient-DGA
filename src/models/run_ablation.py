"""
src/models/run_ablation.py
───────────────────────────
Ablation study xây tăng dần + BWT đúng chuẩn + F1-OldFamilies.

Bảng ablation:
  Row 1: CNN (Static)              — không update
  Row 2: CNN + LoRA Update         — LoRA, update mọi cửa sổ
  Row 3: CNN + LoRA + SER          — thêm replay buffer
  Row 4: CNN + LoRA + SER + EWC    — thêm EWC regularization
  Row 5: DRC-CL (full)             — thêm ADD drift detector

Metrics:
  - AA-F1        : trung bình F1 qua 24 cửa sổ (accuracy)
  - BWT (real)   : evaluate model CUỐI trên TẤT CẢ 24 cửa sổ (forgetting)
  - Degrad.      : F1_first - F1_last (long-term stability)
  - F1-Old       : F1 trên DGA families có trong D01, evaluate tại D24 (memory retention)

Usage:
    python -m src.models.run_ablation
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit as sigmoid_stable

from src.models.char_cnn import CharCNN, domain_to_tensor, domains_to_batch
from src.models.lora_adapter import CharCNNWithLoRA
from src.models.drc_cl import SERBuffer, EWCRegularizer, DomainDataset
from src.detect.add_detector import ADDDetector, extract_embeddings
from src.utils.common import get_logger, get_window_ids, load_config
from src.utils.dga_taxonomy import split_by_dga_type, WORD_BASED_FAMILIES


# ── Evaluation helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def eval_window(model, df: pd.DataFrame, device: str, batch_size: int = 512) -> dict:
    model.eval()
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
    return {"f1": f1_score(labels, preds, zero_division=0),
            "auc": roc_auc_score(labels, probs)}


@torch.no_grad()
def eval_by_dga_type(model, df: pd.DataFrame, device: str,
                     batch_size: int = 512) -> dict:
    """
    F1 tách theo DGA type: char-based vs word-based.
    Mỗi subset = DGA của loại đó + toàn bộ benign → tính F1 binary.
    """
    model.eval()
    df_char, df_word = split_by_dga_type(df)
    result = {}

    for subset_name, subset_df in [("f1_char", df_char), ("f1_word", df_word)]:
        if len(subset_df) == 0 or subset_df["label"].nunique() < 2:
            result[subset_name] = float("nan")
            continue
        domains = subset_df["domain"].tolist()
        labels  = np.array(subset_df["label"].tolist())
        all_logits = []
        for i in range(0, len(domains), batch_size):
            x = domains_to_batch(domains[i:i+batch_size]).to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
        logits_np = np.concatenate(all_logits)
        probs = sigmoid_stable(logits_np)
        preds = (probs >= 0.5).astype(int)
        result[subset_name] = f1_score(labels, preds, zero_division=0)

    return result


@torch.no_grad()
def eval_old_families(model, df: pd.DataFrame, old_families: set,
                      device: str, batch_size: int = 512) -> float:
    """
    F1 chỉ trên các DGA families có trong D01.
    Đánh giá: model có còn nhớ các gia đình cũ không?
    """
    model.eval()
    # Lọc: chỉ giữ domain DGA thuộc old families + tất cả benign
    mask_old_dga = (df["family"].isin(old_families)) & (df["label"] == 1)
    mask_benign  = df["label"] == 0
    subset = df[mask_old_dga | mask_benign].copy()

    if len(subset) == 0 or subset["label"].nunique() < 2:
        return float("nan")

    domains = subset["domain"].tolist()
    labels  = np.array(subset["label"].tolist())
    all_logits = []
    for i in range(0, len(domains), batch_size):
        x = domains_to_batch(domains[i:i+batch_size]).to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    probs = sigmoid_stable(logits_np)
    preds = (probs >= 0.5).astype(int)
    return f1_score(labels, preds, zero_division=0)


@torch.no_grad()
def compute_real_bwt(model, bench_dir: Path, window_ids: list[str],
                     a_diag: list[float], device: str,
                     batch_size: int = 512) -> tuple[float, list[float]]:
    """
    BWT chuẩn:
      1. Model CUỐI evaluate trên TẤT CẢ windows → a[T][i]
      2. BWT = mean(a[T][i] - a[i][i]) cho i = 0..T-2

    Returns: (bwt_value, list_of_a_T_i)
    """
    model.eval()
    T = len(window_ids)
    a_final = []  # a[T-1][i] for all i

    for i, win_id in enumerate(window_ids):
        df = pd.read_csv(bench_dir / f"{win_id}.csv")
        metrics = eval_window(model, df, device, batch_size)
        a_final.append(metrics["f1"])

    # BWT = mean(a[T][i] - a[i][i]) for i = 0..T-2
    bwt_vals = []
    for i in range(T - 1):
        bwt_vals.append(a_final[i] - a_diag[i])

    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    return bwt, a_final


# ── LoRA update helper ────────────────────────────────────────────────────────
def update_lora(model: CharCNNWithLoRA, domains: list[str], labels: list[int],
                device: str, epochs: int = 5, lr: float = 5e-4,
                batch_size: int = 512, ewc_penalty_fn=None) -> float:
    ds     = DomainDataset(domains, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.lora_parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler("cuda") if device == "cuda" else None
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    logits = model(x)
                    loss   = criterion(logits, y)
                    if ewc_penalty_fn:
                        loss = loss + ewc_penalty_fn()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.lora_parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss   = criterion(logits, y)
                if ewc_penalty_fn:
                    loss = loss + ewc_penalty_fn()
                loss.backward()
                nn.utils.clip_grad_norm_(model.lora_parameters(), 1.0)
                optimizer.step()
            last_loss = loss.item()
    return last_loss


# ── Generic ablation variant runner ───────────────────────────────────────────
def run_variant(name: str,
                cfg: dict,
                backbone_path: Path,
                device: str,
                use_ser: bool = False,
                use_ewc: bool = False,
                use_add: bool = False,
                logger = None) -> dict:
    """
    Chạy 1 variant qua 24 cửa sổ, trả về full metrics.
    """
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)
    epochs     = cfg.get("training", {}).get("update_epochs", 5)
    lr         = cfg.get("training", {}).get("lr", 5e-4)
    batch_size = cfg.get("training", {}).get("batch_size", 512)
    rank       = cfg.get("lora", {}).get("rank", 8)
    alpha      = cfg.get("lora", {}).get("alpha", 16.0)
    buf_size   = cfg.get("ser", {}).get("capacity", 5000)
    lam        = cfg.get("ewc", {}).get("lambda", 0.4)
    mu         = cfg.get("training", {}).get("mix_ratio", 0.3)
    seed       = cfg["random_seed"]
    rng        = np.random.default_rng(seed)
    T          = len(window_ids)

    logger.info(f"  Components: LoRA=✓  SER={'✓' if use_ser else '✗'}  "
                f"EWC={'✓' if use_ewc else '✗'}  ADD={'✓' if use_add else '✗'}")

    # Init model
    model = CharCNNWithLoRA.from_checkpoint(
        backbone_path, rank=rank, alpha=alpha, map_location=device
    ).to(device)

    # Init components
    ser = SERBuffer(capacity=buf_size, seed=seed) if use_ser else None
    ewc = EWCRegularizer(lam=lam) if use_ewc else None
    add = None

    if use_add:
        add = ADDDetector.from_config(cfg)

    # Init EWC Fisher on D01
    if ewc:
        df0 = pd.read_csv(bench_dir / f"{window_ids[0]}.csv")
        ds0 = DomainDataset(df0["domain"].tolist()[:2000], df0["label"].tolist()[:2000])
        ewc.update_fisher(model, DataLoader(ds0, batch_size=batch_size, shuffle=True), device)

    # Init ADD calibration on D01
    if add:
        df0 = pd.read_csv(bench_dir / f"{window_ids[0]}.csv")
        ref_embs = extract_embeddings(model, df0["domain"].tolist(), device=device, max_n=5000)
        add.calibrate(ref_embs)

    # Get old families from D01 (for F1-Old metric)
    df0 = pd.read_csv(bench_dir / f"{window_ids[0]}.csv")
    old_families = set(df0.loc[df0["label"] == 1, "family"].unique())

    # ── Prequential loop ──────────────────────────────────────────────────────
    a_diag     = []   # a[i][i]: F1 on D_i before training on D_i
    per_window = []

    for t, win_id in enumerate(window_ids):
        t0     = time.time()
        df_t   = pd.read_csv(bench_dir / f"{win_id}.csv")
        domains_t  = df_t["domain"].tolist()
        labels_t   = df_t["label"].tolist()
        families_t = df_t["family"].tolist()

        # 1. EVALUATE before update (prequential)
        metrics = eval_window(model, df_t, device)
        a_diag.append(metrics["f1"])

        # 2. Decide whether to update
        should_update = True
        drift_type    = "always"

        if use_add and add and t > 0:
            curr_embs = extract_embeddings(model, domains_t, device=device, max_n=5000)
            event     = add.detect(curr_embs)
            drift_type = event.drift_type
            should_update = event.needs_update
            # Update ADD reference
            if drift_type in ("none", "sudden"):
                add.archive_centroid()
            add.set_reference(curr_embs)

        # 3. UPDATE if needed
        if t > 0 and should_update:
            # Build training data
            train_d = domains_t
            train_l = labels_t

            if use_ser and ser:
                n_buf = int(len(domains_t) * mu / (1 - mu))
                b_dom, b_lab, _ = ser.sample(n_buf, rng)
                train_d = domains_t + b_dom
                train_l = labels_t  + b_lab

            ewc_fn = (lambda: ewc.penalty(model)) if ewc else None
            update_lora(model, train_d, train_l, device,
                        epochs=epochs, lr=lr, batch_size=batch_size,
                        ewc_penalty_fn=ewc_fn)

            # Update EWC Fisher after training
            if ewc:
                if use_ser and ser and len(ser) > 0:
                    fb_d, fb_l, _ = ser.sample(min(1024, len(ser)), rng)
                else:
                    fb_d, fb_l = domains_t[:1024], labels_t[:1024]
                fisher_ds = DomainDataset(fb_d, fb_l)
                ewc.update_fisher(model, DataLoader(fisher_ds, batch_size=batch_size,
                                                     shuffle=True), device)

        # 4. Update SER buffer
        if use_ser and ser:
            embs = extract_embeddings(model, domains_t, device=device, max_n=5000)
            idx  = rng.choice(len(domains_t), min(5000, len(domains_t)), replace=False)
            ser.add_batch([domains_t[i] for i in idx], [labels_t[i] for i in idx],
                          [families_t[i] for i in idx], embs[:len(idx)])

        elapsed = time.time() - t0
        q_label = df_t["quarter_label"].iloc[0]
        upd_str = f"drift={drift_type}" if use_add else "update=always"
        logger.info(f"    {win_id} ({q_label}): F1={metrics['f1']:.4f}  {upd_str}  ({elapsed:.1f}s)")

        per_window.append({
            "method": name, "window_id": win_id, "quarter_label": q_label,
            "f1": metrics["f1"], "auc": metrics["auc"],
        })

    # ── Post-training: compute REAL BWT ───────────────────────────────────────
    logger.info(f"    Computing real BWT (evaluating final model on all {T} windows)...")
    bwt, a_final = compute_real_bwt(model, bench_dir, window_ids, a_diag, device)
    logger.info(f"    Real BWT = {bwt:+.4f}")

    # ── F1-OldFamilies: final model on D24, filtered to D01 families ──────────
    df_last = pd.read_csv(bench_dir / f"{window_ids[-1]}.csv")
    f1_old  = eval_old_families(model, df_last, old_families, device)
    logger.info(f"    F1-OldFamilies (D01 families @ D24) = {f1_old:.4f}")

    # F1 by DGA type (char-based vs word-based) on last window
    type_metrics = eval_by_dga_type(model, df_last, device)
    f1_char = type_metrics.get("f1_char", float("nan"))
    f1_word = type_metrics.get("f1_word", float("nan"))
    logger.info(f"    F1-CharDGA = {f1_char:.4f}   F1-WordDGA = {f1_word:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    f1s    = [w["f1"]  for w in per_window]
    aucs   = [w["auc"] for w in per_window]
    aa_f1  = float(np.mean(f1s))
    aa_auc = float(np.mean(aucs))
    degrad = f1s[0] - f1s[-1]

    return {
        "method":    name,
        "aa_f1":     round(aa_f1,  4),
        "aa_auc":    round(aa_auc, 4),
        "bwt":       round(bwt,    4),
        "degrad":    round(degrad, 4),
        "f1_old":    round(f1_old, 4),
        "f1_char":   round(f1_char, 4) if not np.isnan(f1_char) else None,
        "f1_word":   round(f1_word, 4) if not np.isnan(f1_word) else None,
        "f1_first":  round(f1s[0], 4),
        "f1_last":   round(f1s[-1],4),
        "per_window": per_window,
        "a_diag":    a_diag,
        "a_final":   a_final,
    }


# ── Static-CNN (no update) ────────────────────────────────────────────────────
def run_static(cfg: dict, backbone_path: Path, device: str, logger) -> dict:
    """Static-CNN: evaluate pretrained backbone on all windows, no update."""
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)

    model = CharCNN.load(backbone_path, map_location=device).to(device)

    df0 = pd.read_csv(bench_dir / f"{window_ids[0]}.csv")
    old_families = set(df0.loc[df0["label"] == 1, "family"].unique())

    per_window = []
    a_diag     = []
    for win_id in window_ids:
        df = pd.read_csv(bench_dir / f"{win_id}.csv")
        m  = eval_window(model, df, device)
        a_diag.append(m["f1"])
        per_window.append({"method": "CNN (Static)", "window_id": win_id,
                           "quarter_label": df["quarter_label"].iloc[0],
                           "f1": m["f1"], "auc": m["auc"]})
        logger.info(f"    {win_id}: F1={m['f1']:.4f}")

    # BWT: model never changes → a[T][i] = a[i][i] → BWT = 0 by definition
    # But let's compute real BWT properly (should be 0)
    bwt, a_final = compute_real_bwt(model, bench_dir, window_ids, a_diag, device)

    df_last = pd.read_csv(bench_dir / f"{window_ids[-1]}.csv")
    f1_old  = eval_old_families(model, df_last, old_families, device)
    type_m  = eval_by_dga_type(model, df_last, device)
    f1_char = type_m.get("f1_char", float("nan"))
    f1_word = type_m.get("f1_word", float("nan"))

    f1s = [w["f1"] for w in per_window]
    return {
        "method": "CNN (Static)", "aa_f1": round(np.mean(f1s), 4),
        "aa_auc": round(np.mean([w["auc"] for w in per_window]), 4),
        "bwt": round(bwt, 4), "degrad": round(f1s[0] - f1s[-1], 4),
        "f1_old": round(f1_old, 4),
        "f1_char": round(f1_char, 4) if not np.isnan(f1_char) else None,
        "f1_word": round(f1_word, 4) if not np.isnan(f1_word) else None,
        "f1_first": round(f1s[0], 4), "f1_last": round(f1s[-1], 4),
        "per_window": per_window, "a_diag": a_diag, "a_final": a_final,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg: dict, backbone_path: Path) -> None:
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("run_ablation", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["results"])

    logger.info("=" * 65)
    logger.info(" ABLATION STUDY — Incremental + Real BWT + F1-OldFamilies")
    logger.info("=" * 65)
    logger.info(f"  Device: {device}")
    if device == "cuda":
        logger.info(f"  GPU   : {torch.cuda.get_device_name(0)}")
    logger.info("")

    results  = []
    t_total  = time.time()

    # ── Row 1: CNN (Static) ───────────────────────────────────────────────────
    logger.info(f"{'─'*65}")
    logger.info("Row 1/5: CNN (Static) — no update")
    logger.info(f"{'─'*65}")
    r1 = run_static(cfg, backbone_path, device, logger)
    results.append(r1)

    # ── Row 2: CNN + LoRA Update ──────────────────────────────────────────────
    logger.info(f"\n{'─'*65}")
    logger.info("Row 2/5: CNN + LoRA Update (no SER, no EWC, no ADD)")
    logger.info(f"{'─'*65}")
    r2 = run_variant("CNN + LoRA Update", cfg, backbone_path, device,
                     use_ser=False, use_ewc=False, use_add=False, logger=logger)
    results.append(r2)

    # ── Row 3: CNN + LoRA + SER ───────────────────────────────────────────────
    logger.info(f"\n{'─'*65}")
    logger.info("Row 3/5: CNN + LoRA + SER (no EWC, no ADD)")
    logger.info(f"{'─'*65}")
    r3 = run_variant("CNN + LoRA + SER", cfg, backbone_path, device,
                     use_ser=True, use_ewc=False, use_add=False, logger=logger)
    results.append(r3)

    # ── Row 4: CNN + LoRA + SER + EWC ─────────────────────────────────────────
    logger.info(f"\n{'─'*65}")
    logger.info("Row 4/5: CNN + LoRA + SER + EWC (no ADD)")
    logger.info(f"{'─'*65}")
    r4 = run_variant("CNN + LoRA + SER + EWC", cfg, backbone_path, device,
                     use_ser=True, use_ewc=True, use_add=False, logger=logger)
    results.append(r4)

    # ── Row 5: DRC-CL (full) ─────────────────────────────────────────────────
    logger.info(f"\n{'─'*65}")
    logger.info("Row 5/5: DRC-CL (full) = LoRA + SER + EWC + ADD")
    logger.info(f"{'─'*65}")
    r5 = run_variant("DRC-CL (full)", cfg, backbone_path, device,
                     use_ser=True, use_ewc=True, use_add=True, logger=logger)
    results.append(r5)

    total_time = time.time() - t_total

    # ── Print full ablation table ─────────────────────────────────────────────
    logger.info(f"\n{'═'*80}")
    logger.info(" ABLATION TABLE (Table V for paper)")
    logger.info(f"{'═'*80}")

    header = f"{'Method':<26} {'AA-F1':>7} {'AA-AUC':>7} {'BWT':>8} {'Degrad.':>8} {'F1-Old':>7} {'F1-Char':>7} {'F1-Word':>7} {'F1-1st':>7} {'F1-Last':>7}"
    sep    = "─" * 100
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for r in results:
        f1c = f"{r['f1_char']:>7.4f}" if r.get('f1_char') is not None else "    N/A"
        f1w = f"{r['f1_word']:>7.4f}" if r.get('f1_word') is not None else "    N/A"
        logger.info(
            f"{r['method']:<26} {r['aa_f1']:>7.4f} {r['aa_auc']:>7.4f} "
            f"{r['bwt']:>+8.4f} {r['degrad']:>+8.4f} {r['f1_old']:>7.4f} "
            f"{f1c} {f1w} "
            f"{r['f1_first']:>7.4f} {r['f1_last']:>7.4f}"
        )
    logger.info(sep)

    # ── Component contribution ────────────────────────────────────────────────
    logger.info(f"\n  Component contribution:")
    logger.info(f"  {'':40s} {'ΔAA-F1':>9} {'ΔBWT':>9} {'ΔF1-Old':>9}")
    logger.info(f"  {'─'*70}")
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        d_f1  = curr["aa_f1"]  - prev["aa_f1"]
        d_bwt = curr["bwt"]    - prev["bwt"]
        d_old = curr["f1_old"] - prev["f1_old"]
        arrow_f1  = "↑" if d_f1  > 0.0005 else "↓" if d_f1  < -0.0005 else "→"
        arrow_bwt = "↑" if d_bwt > 0.0005 else "↓" if d_bwt < -0.0005 else "→"
        arrow_old = "↑" if d_old > 0.0005 else "↓" if d_old < -0.0005 else "→"
        logger.info(
            f"  {prev['method']:<20s} → {curr['method']:<20s}"
            f"{arrow_f1} {d_f1:>+8.4f} {arrow_bwt} {d_bwt:>+8.4f} {arrow_old} {d_old:>+8.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    # Summary table
    summary_rows = [{k: v for k, v in r.items()
                     if k not in ("per_window", "a_diag", "a_final")}
                    for r in results]
    table_path = out_dir / "ablation_table.csv"
    pd.DataFrame(summary_rows).to_csv(table_path, index=False)
    logger.info(f"\n  Ablation Table     → {table_path}")

    # Per-window combined
    all_pw = []
    for r in results:
        for w in r["per_window"]:
            all_pw.append(w)
    pw_path = out_dir / "ablation_per_window.csv"
    pd.DataFrame(all_pw).to_csv(pw_path, index=False)
    logger.info(f"  Per-window data    → {pw_path}")

    # BWT detail: a_diag and a_final for each variant
    bwt_detail = []
    for r in results:
        for i, win_id in enumerate(get_window_ids(cfg)):
            bwt_detail.append({
                "method":  r["method"],
                "window":  win_id,
                "a_diag":  r["a_diag"][i]  if i < len(r["a_diag"])  else None,
                "a_final": r["a_final"][i] if i < len(r["a_final"]) else None,
            })
    bwt_path = out_dir / "ablation_bwt_detail.csv"
    pd.DataFrame(bwt_detail).to_csv(bwt_path, index=False)
    logger.info(f"  BWT detail         → {bwt_path}")

    logger.info(f"\n  Total time: {total_time:.1f}s")
    logger.info("  Ablation study complete ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study with real BWT + F1-OldFamilies")
    parser.add_argument("--config",   default=None)
    parser.add_argument("--backbone", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("lora",     {"rank": 8, "alpha": 16.0})
    cfg.setdefault("ser",      {"capacity": 5000, "beta": 0.92, "min_k": 50})
    cfg.setdefault("ewc",      {"lambda": 0.4})
    cfg.setdefault("training", {"lr": 5e-4, "update_epochs": 5,
                                "batch_size": 512, "mix_ratio": 0.3})

    if args.backbone:
        backbone_path = Path(args.backbone)
    else:
        backbone_path = Path(cfg["paths"]["results"]) / "checkpoints" / "backbone_d01.pt"
    if not backbone_path.exists():
        print(f"ERROR: Backbone not found at {backbone_path}")
        exit(1)

    run(cfg, backbone_path)
