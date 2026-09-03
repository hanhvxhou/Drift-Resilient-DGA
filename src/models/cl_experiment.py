"""
src/models/cl_experiment.py
────────────────────────────
Unified CL experiment runner with proper train/test protocol.

Protocol (chuẩn CL):
    For each window t = 0..T-1:
        1. Load D_t_train, D_t_test
        2. Evaluate model on D_t_test BEFORE training (prequential → for FWT)
        3. Update model on D_t_train (method-specific)
        4. After update: evaluate on ALL test sets W1_test..Wt_test
           → Fill row t of accuracy matrix
        5. Update buffer/Fisher (method-specific)

    Result: 24×24 accuracy matrix → AA, BWT, FWT, Forgetting

Works with: CharCNN, CharCNNWithLoRA, DistilBERT — any model with forward().

Usage:
    python -m src.models.cl_experiment
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from scipy.special import expit as sigmoid_stable

from src.models.char_cnn import CharCNN, domain_to_tensor, domains_to_batch, MAX_LEN
from src.models.lora_adapter import CharCNNWithLoRA, AdapterBank
from src.models.cl_metrics import AccuracyMatrix, build_accuracy_row, print_metrics_table, print_per_type_table
from src.detect.add_detector import ADDDetector, extract_embeddings
from src.utils.common import get_logger, get_window_ids, load_config
from src.utils.dga_taxonomy import WORD_BASED_FAMILIES


# ── Dataset ───────────────────────────────────────────────────────────────────
class DomainDataset(Dataset):
    def __init__(self, domains, labels):
        self.domains = domains
        self.labels  = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):  return len(self.domains)
    def __getitem__(self, idx):
        return domain_to_tensor(self.domains[idx]), self.labels[idx]


# ── SER Buffer (simplified, same logic) ───────────────────────────────────────
class SERBuffer:
    def __init__(self, capacity=5000, seed=42):
        self.capacity = capacity
        self.half     = capacity // 2
        self.rng      = np.random.default_rng(seed)
        self._buf_0   = []  # benign: list of (domain, label, family)
        self._buf_1   = []  # DGA
        self._n_seen  = 0

    def add_batch(self, domains, labels, families):
        for d, l, f in zip(domains, labels, families):
            self._n_seen += 1
            buf = self._buf_1 if l == 1 else self._buf_0
            item = (d, l, f)
            if len(buf) < self.half:
                buf.append(item)
            elif self.rng.random() < self.half / self._n_seen:
                idx = self.rng.integers(0, len(buf))
                buf[idx] = item

    def sample(self, n):
        all_items = self._buf_0 + self._buf_1
        if not all_items: return [], [], []
        n = min(n, len(all_items))
        idx = self.rng.choice(len(all_items), n, replace=False)
        sampled = [all_items[i] for i in idx]
        return [s[0] for s in sampled], [s[1] for s in sampled], [s[2] for s in sampled]

    def __len__(self): return len(self._buf_0) + len(self._buf_1)


# ── EWC Regularizer ───────────────────────────────────────────────────────────
class EWCReg:
    def __init__(self, lam=0.4):
        self.lam = lam
        self.fisher = {}
        self.theta_star = {}

    def update(self, model, loader, device):
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        fisher_acc = {}
        count = 0
        for x, y in loader:
            if count > 1024: break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if n not in fisher_acc:
                        fisher_acc[n] = torch.zeros_like(p.data)
                    fisher_acc[n] += p.grad.data.pow(2)
            count += len(y)
        nb = max(count / loader.batch_size, 1)
        self.fisher = {k: v / nb for k, v in fisher_acc.items()}
        self.theta_star = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        if not self.fisher: return torch.tensor(0.0)
        dev = next(model.parameters()).device
        loss = torch.tensor(0.0, device=dev)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n].to(dev) * (p - self.theta_star[n].to(dev)).pow(2)).sum()
        return self.lam * loss


# ── Generic train function ────────────────────────────────────────────────────
def train_on_data(model, domains, labels, device, epochs=5, lr=5e-4,
                  batch_size=512, ewc_fn=None, lora_only=False):
    ds     = DomainDataset(domains, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    params = model.lora_parameters() if (lora_only and hasattr(model, 'lora_parameters')) else \
             [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda") if device == "cuda" else None
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                    if ewc_fn: loss = loss + ewc_fn()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                if ewc_fn: loss = loss + ewc_fn()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()


# ══════════════════════════════════════════════════════════════════════════════
# Method definitions
# ══════════════════════════════════════════════════════════════════════════════

def run_method(method_name: str, cfg: dict, backbone_path: Path,
               device: str, logger, split_dir: Path, window_ids: list[str],
               # Method flags
               use_lora: bool = True,
               update_every: bool = True,
               use_ser: bool = False,
               use_ewc: bool = False,
               use_add: bool = False,
               # Baselines
               is_static: bool = False,
               is_sw_retrain: bool = False,
               is_icarl: bool = False,
               is_gdumb: bool = False,
               is_ewc_only_fulltune: bool = False,
               ) -> dict:
    """
    Unified CL experiment runner.
    Returns: {method, metrics_dict, accuracy_matrix, per_type}
    """
    T = len(window_ids)
    epochs  = cfg.get("training", {}).get("update_epochs", 5)
    lr      = cfg.get("training", {}).get("lr", 5e-4)
    bs      = cfg.get("training", {}).get("batch_size", 512)
    rank    = cfg.get("lora", {}).get("rank", 8)
    alpha   = cfg.get("lora", {}).get("alpha", 16.0)
    buf_cap = cfg.get("ser", {}).get("capacity", 5000)
    lam     = cfg.get("ewc", {}).get("lambda", 0.4)
    mu      = cfg.get("training", {}).get("mix_ratio", 0.3)
    seed    = cfg["random_seed"]
    rng     = np.random.default_rng(seed)

    # ── Init model ────────────────────────────────────────────────────────────
    if use_lora and not is_static and not is_sw_retrain and not is_gdumb and not is_ewc_only_fulltune and not is_icarl:
        model = CharCNNWithLoRA.from_checkpoint(backbone_path, rank=rank, alpha=alpha,
                                                 map_location=device).to(device)
        lora_only = True
    else:
        model = CharCNN.load(backbone_path, map_location=device).to(device)
        lora_only = False

    # ── Init components ───────────────────────────────────────────────────────
    ser = SERBuffer(capacity=buf_cap, seed=seed) if use_ser else None
    ewc = EWCReg(lam=lam) if use_ewc else None
    ewc_ft = EWCReg(lam=lam) if is_ewc_only_fulltune else None  # full-tune EWC
    add_det = ADDDetector.from_config(cfg) if use_add else None

    old_model = None  # for iCaRL distillation
    gdumb_buf_d, gdumb_buf_l = [], []  # for GDumb

    # ── Accuracy matrix ───────────────────────────────────────────────────────
    matrix = AccuracyMatrix(window_ids)

    logger.info(f"  Running: {method_name}")
    t_total = time.time()

    for t in range(T):
        win_id = window_ids[t]
        t0 = time.time()

        train_df = pd.read_csv(split_dir / f"{win_id}_train.csv")
        train_d  = train_df["domain"].tolist()
        train_l  = train_df["label"].tolist()
        train_f  = train_df["family"].tolist()

        # ── Step 3: Update model (method-specific) ────────────────────────────
        if t > 0 and not is_static:

            if is_sw_retrain:
                # Reset to pretrained, fine-tune on D_t_train only
                model = CharCNN.load(backbone_path, map_location=device).to(device)
                train_on_data(model, train_d, train_l, device, epochs=10, lr=1e-3, batch_size=bs)

            elif is_gdumb:
                # Greedy buffer update
                half = buf_cap // 2
                all_d = gdumb_buf_d + train_d
                all_l = gdumb_buf_l + train_l
                dga_i   = [i for i, l in enumerate(all_l) if l == 1]
                ben_i   = [i for i, l in enumerate(all_l) if l == 0]
                sel = rng.choice(dga_i, min(half, len(dga_i)), replace=False).tolist() + \
                      rng.choice(ben_i, min(half, len(ben_i)), replace=False).tolist()
                gdumb_buf_d = [all_d[i] for i in sel]
                gdumb_buf_l = [all_l[i] for i in sel]
                # Reset and retrain on buffer
                model = CharCNN.load(backbone_path, map_location=device).to(device)
                if gdumb_buf_d:
                    train_on_data(model, gdumb_buf_d, gdumb_buf_l, device, epochs=10, lr=1e-3, batch_size=bs)

            elif is_icarl:
                # iCaRL: replay + distillation
                icl_d = train_d + (gdumb_buf_d if gdumb_buf_d else [])
                icl_l = train_l + (gdumb_buf_l if gdumb_buf_l else [])
                # Train with distillation
                ds = DomainDataset(icl_d, icl_l)
                loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()
                scaler = GradScaler("cuda") if device == "cuda" else None
                model.train()
                for _ in range(epochs):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        if scaler:
                            with autocast("cuda"):
                                logits = model(x)
                                loss = criterion(logits, y)
                                if old_model is not None:
                                    with torch.no_grad():
                                        old_logits = old_model(x)
                                    loss = loss + 0.5 * TF.mse_loss(logits, old_logits)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            logits = model(x)
                            loss = criterion(logits, y)
                            if old_model is not None:
                                with torch.no_grad():
                                    old_logits = old_model(x)
                                loss = loss + 0.5 * TF.mse_loss(logits, old_logits)
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                old_model = copy.deepcopy(model).eval()
                # Update exemplar buffer
                half = buf_cap // 2
                dga_i   = [i for i, l in enumerate(train_l) if l == 1]
                ben_i   = [i for i, l in enumerate(train_l) if l == 0]
                sel = rng.choice(dga_i, min(half, len(dga_i)), replace=False).tolist() + \
                      rng.choice(ben_i, min(half, len(ben_i)), replace=False).tolist()
                gdumb_buf_d = [train_d[i] for i in sel]
                gdumb_buf_l = [train_l[i] for i in sel]

            elif is_ewc_only_fulltune:
                # Full fine-tune + EWC
                train_on_data(model, train_d, train_l, device, epochs=epochs, lr=lr,
                              batch_size=bs, ewc_fn=lambda: ewc_ft.penalty(model))
                ds = DomainDataset(train_d[:2000], train_l[:2000])
                ewc_ft.update(model, DataLoader(ds, batch_size=bs, shuffle=True), device)

            else:
                # LoRA-based methods (DRC-CL variants)
                should_update = True
                if use_add and add_det:
                    embs = extract_embeddings(model, train_d, device=device, max_n=5000)
                    if t == 1:  # calibrate on first transition
                        ref_embs = extract_embeddings(model,
                            pd.read_csv(split_dir / f"{window_ids[0]}_train.csv")["domain"].tolist(),
                            device=device, max_n=5000)
                        add_det.calibrate(ref_embs)
                    event = add_det.detect(embs)
                    should_update = event.needs_update  # includes forced safeguard

                if should_update:
                    upd_d, upd_l = train_d, train_l
                    if use_ser and ser:
                        n_buf = int(len(train_d) * mu / max(1 - mu, 0.01))
                        b_d, b_l, _ = ser.sample(n_buf)
                        upd_d = train_d + b_d
                        upd_l = train_l + b_l

                    ewc_fn = (lambda: ewc.penalty(model)) if use_ewc and ewc else None
                    train_on_data(model, upd_d, upd_l, device, epochs=epochs, lr=lr,
                                  batch_size=bs, ewc_fn=ewc_fn, lora_only=lora_only)

                    if use_ewc and ewc:
                        if use_ser and ser and len(ser) > 0:
                            fb_d, fb_l, _ = ser.sample(min(1024, len(ser)))
                        else:
                            fb_d, fb_l = train_d[:1024], train_l[:1024]
                        ds = DomainDataset(fb_d, fb_l)
                        ewc.update(model, DataLoader(ds, batch_size=bs, shuffle=True), device)

                # Update SER buffer (always, from TRAIN split only)
                if use_ser and ser:
                    n_add = min(5000, len(train_d))
                    idx = rng.choice(len(train_d), n_add, replace=False)
                    ser.add_batch([train_d[i] for i in idx], [train_l[i] for i in idx],
                                 [train_f[i] for i in idx])

        elif t == 0 and not is_static:
            # Init EWC Fisher on D01 train
            if use_ewc and ewc:
                ds0 = DomainDataset(train_d[:2000], train_l[:2000])
                ewc.update(model, DataLoader(ds0, batch_size=bs, shuffle=True), device)
            if is_ewc_only_fulltune and ewc_ft:
                ds0 = DomainDataset(train_d[:2000], train_l[:2000])
                ewc_ft.update(model, DataLoader(ds0, batch_size=bs, shuffle=True), device)
            # Init ADD
            if use_add and add_det:
                ref_embs = extract_embeddings(model, train_d, device=device, max_n=5000)
                add_det.calibrate(ref_embs)
            # Init SER
            if use_ser and ser:
                n_add = min(5000, len(train_d))
                idx = rng.choice(len(train_d), n_add, replace=False)
                ser.add_batch([train_d[i] for i in idx], [train_l[i] for i in idx],
                              [train_f[i] for i in idx])
            # Init iCaRL/GDumb buffers
            if is_icarl or is_gdumb:
                half = buf_cap // 2
                dga_i = [i for i, l in enumerate(train_l) if l == 1]
                ben_i = [i for i, l in enumerate(train_l) if l == 0]
                sel = rng.choice(dga_i, min(half, len(dga_i)), replace=False).tolist() + \
                      rng.choice(ben_i, min(half, len(ben_i)), replace=False).tolist()
                gdumb_buf_d = [train_d[i] for i in sel]
                gdumb_buf_l = [train_l[i] for i in sel]
                if is_icarl:
                    old_model = copy.deepcopy(model).eval()

        # ── Step 4: Evaluate on ALL test sets W1..Wt → fill matrix row t ──────
        row = build_accuracy_row(model, split_dir, window_ids, up_to_t=t, device=device)
        matrix.add_row(t, row)

        elapsed = time.time() - t0
        f1_t = row.get(win_id, {}).get("f1", 0)
        q_label = train_df["quarter_label"].iloc[0]
        logger.info(f"    {win_id} ({q_label}): F1={f1_t:.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - t_total
    metrics = matrix.compute_metrics()
    metrics["method"] = method_name
    metrics["time_s"] = round(total_time, 1)
    per_type = matrix.compute_per_type_metrics()

    logger.info(f"    Done: AA-F1={metrics['aa_f1']:.4f}  BWT={metrics['bwt']:+.4f}  "
                f"Forg={metrics['forgetting']:+.4f}  ({total_time:.1f}s)")

    return {"method": method_name, "metrics": metrics, "matrix": matrix, "per_type": per_type}


# ══════════════════════════════════════════════════════════════════════════════
# Main: Run ALL experiments
# ══════════════════════════════════════════════════════════════════════════════
def run_all(cfg: dict, backbone_path: Path,
            skip: list[str] | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir    = Path(cfg["paths"]["results"]) / "logs"
    logger     = get_logger("cl_experiment", log_dir=log_dir)
    out_dir    = Path(cfg["paths"]["results"])
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    split_dir  = bench_dir / "splits"
    window_ids = get_window_ids(cfg)
    skip = skip or []

    logger.info("=" * 70)
    logger.info(" FULL CL EXPERIMENT — Accuracy Matrix Protocol")
    logger.info(f" {len(window_ids)} windows, train/test split, device={device}")
    logger.info("=" * 70)

    # Ensure defaults
    cfg.setdefault("lora",     {"rank": 8, "alpha": 16.0})
    cfg.setdefault("ser",      {"capacity": 5000, "beta": 0.92, "min_k": 50})
    cfg.setdefault("ewc",      {"lambda": 0.4})
    cfg.setdefault("training", {"lr": 5e-4, "update_epochs": 5, "batch_size": 512, "mix_ratio": 0.3})

    common = dict(cfg=cfg, backbone_path=backbone_path, device=device,
                  logger=logger, split_dir=split_dir, window_ids=window_ids)

    EXPERIMENTS = [
        # Table IV baselines
        ("Static-CNN",        dict(is_static=True, use_lora=False)),
        ("SW-Retrain",        dict(is_sw_retrain=True, use_lora=False)),
        ("EWC-only",          dict(is_ewc_only_fulltune=True, use_lora=False)),
        ("iCaRL",             dict(is_icarl=True, use_lora=False)),
        ("GDumb",             dict(is_gdumb=True, use_lora=False)),
        # Table V ablation (incremental)
        ("CNN + LoRA Update", dict(use_lora=True, update_every=True)),
        ("CNN + LoRA + SER",  dict(use_lora=True, update_every=True, use_ser=True)),
        ("CNN + LoRA + SER + EWC", dict(use_lora=True, update_every=True, use_ser=True, use_ewc=True)),
        # DRC-CL (full)
        ("DRC-CL",            dict(use_lora=True, use_ser=True, use_ewc=True, use_add=True)),
    ]

    all_results = []
    all_per_type = {}

    for name, kwargs in EXPERIMENTS:
        if name in skip:
            logger.info(f"\n  [{name}] SKIPPED")
            continue
        logger.info(f"\n{'─'*70}")
        result = run_method(method_name=name, **common, **kwargs)
        all_results.append(result["metrics"])
        all_per_type[name] = result["per_type"]
        # Save matrix
        result["matrix"].save(out_dir, prefix=name.replace(" ", "_").replace("+", "").lower())

    # ── Print Tables ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*70}")
    logger.info(" TABLE IV + TABLE V COMBINED")
    logger.info(f"{'═'*70}")
    print_metrics_table(all_results, logger)
    print_per_type_table(all_per_type, logger)

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_results).to_csv(out_dir / "final_results.csv", index=False)
    with open(out_dir / "final_per_type.json", "w") as f:
        # Convert any nan to None for JSON
        clean = {}
        for method, pt in all_per_type.items():
            clean[method] = {}
            for dtype, vals in pt.items():
                clean[method][dtype] = {k: (v if v == v else None) for k, v in vals.items()}
        json.dump(clean, f, indent=2)

    logger.info(f"\n  Results  → {out_dir / 'final_results.csv'}")
    logger.info(f"  Per-type → {out_dir / 'final_per_type.json'}")
    logger.info(f"  Matrices → {out_dir}/*_accuracy_matrix.csv")
    logger.info("\n  EXPERIMENT COMPLETE ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full CL experiment with accuracy matrix")
    parser.add_argument("--config",   default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--skip",     nargs="*", default=[],
                        help="Skip these methods by name")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bp  = Path(args.backbone) if args.backbone else Path(cfg["paths"]["results"]) / "checkpoints" / "backbone_d01.pt"
    if not bp.exists():
        print(f"ERROR: Backbone not found at {bp}")
        exit(1)
    run_all(cfg, bp, skip=args.skip)
