"""
src/models/run_all_experiments.py
──────────────────────────────────
Master runner: chạy TẤT CẢ methods với accuracy matrix protocol chuẩn.

Methods:
  Table IV:  Static-CNN, SW-Retrain, EWC-only, iCaRL, GDumb, DRC-CL
  Table V:   CNN, +LoRA, +SER, +EWC, +ADD (ablation)

Protocol:
  1. Fixed train/test split (step6)
  2. Train chỉ trên *_train.csv
  3. Evaluate trên *_test.csv → accuracy matrix 24×24
  4. AA, BWT, FWT, Forgetting từ matrix
  5. F1-Old, F1-Char, F1-Word trên test set cuối

Usage:
    # Chạy tất cả (Table IV + V):
    python -m src.models.run_all_experiments

    # Chỉ Table IV (baselines + DRC-CL):
    python -m src.models.run_all_experiments --table iv

    # Chỉ Table V (ablation):
    python -m src.models.run_all_experiments --table v

    # Chỉ 1 method:
    python -m src.models.run_all_experiments --only drc-cl
    python -m src.models.run_all_experiments --only static-cnn
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.models.char_cnn import CharCNN, domain_to_tensor, domains_to_batch
from src.models.lora_adapter import CharCNNWithLoRA, AdapterBank
from src.models.drc_cl import SERBuffer, EWCRegularizer, DomainDataset
from src.detect.add_detector import ADDDetector, extract_embeddings
from src.models.cl_experiment import run_cl_experiment, evaluate_f1, print_full_table
from src.utils.common import get_logger, get_window_ids, load_config


# ══════════════════════════════════════════════════════════════════════════════
# Update functions for each method
# ══════════════════════════════════════════════════════════════════════════════

def _train_on_data(model, domains, labels, device, epochs=5, lr=1e-3,
                   batch_size=512, ewc_fn=None, params_fn=None):
    """Shared training loop."""
    ds     = DomainDataset(domains, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    params = params_fn() if params_fn else [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler("cuda") if device == "cuda" else None
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


# ── Static-CNN: no update ─────────────────────────────────────────────────────
def make_static_cnn(backbone_path, device, cfg):
    model = CharCNN.load(backbone_path, map_location=device).to(device)
    def update_fn(model, train_df, t, cfg):
        pass  # never update
    return model, update_fn


# ── SW-Retrain: reset + retrain on D_t only ───────────────────────────────────
def make_sw_retrain(backbone_path, device, cfg):
    model = CharCNN.load(backbone_path, map_location=device).to(device)
    _base_state = CharCNN.load(backbone_path, map_location="cpu").state_dict()

    def update_fn(model, train_df, t, cfg):
        if t == 0: return
        epochs = cfg.get("training", {}).get("update_epochs", 10)
        lr     = cfg.get("training", {}).get("lr", 1e-3)
        bs     = cfg.get("training", {}).get("batch_size", 512)
        model.load_state_dict(copy.deepcopy(_base_state))
        model.to(device)
        _train_on_data(model, train_df["domain"].tolist(), train_df["label"].tolist(),
                       device, epochs=epochs, lr=lr, batch_size=bs)
    return model, update_fn


# ── EWC-only: full fine-tune + EWC ────────────────────────────────────────────
def make_ewc_only(backbone_path, device, cfg):
    model = CharCNN.load(backbone_path, map_location=device).to(device)
    lam   = cfg.get("ewc", {}).get("lambda", 0.4)
    ewc   = [None]  # mutable container

    class SimpleEWC:
        def __init__(self, lam):
            self.lam = lam
            self.fisher, self.theta_star = {}, {}
        def update(self, model, loader, device):
            model.train()
            criterion = nn.BCEWithLogitsLoss()
            fa = {}; count = 0
            for x, y in loader:
                if count > 1024: break
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                criterion(model(x), y).backward()
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fa.setdefault(n, torch.zeros_like(p.data))
                        fa[n] += p.grad.data.pow(2)
                count += len(y)
            nb = max(count / loader.batch_size, 1)
            self.fisher = {k: v/nb for k,v in fa.items()}
            self.theta_star = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        def penalty(self, model):
            if not self.fisher: return torch.tensor(0.0)
            loss = torch.tensor(0.0, device=next(model.parameters()).device)
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.fisher:
                    loss += (self.fisher[n].to(p.device) * (p - self.theta_star[n].to(p.device)).pow(2)).sum()
            return self.lam * loss

    ewc[0] = SimpleEWC(lam)

    def update_fn(model, train_df, t, cfg):
        epochs = cfg.get("training", {}).get("update_epochs", 5)
        lr     = cfg.get("training", {}).get("lr", 5e-4)
        bs     = cfg.get("training", {}).get("batch_size", 512)
        doms, labs = train_df["domain"].tolist(), train_df["label"].tolist()
        if t > 0:
            _train_on_data(model, doms, labs, device, epochs=epochs, lr=lr,
                           batch_size=bs, ewc_fn=lambda: ewc[0].penalty(model))
        ds = DomainDataset(doms[:2000], labs[:2000])
        ewc[0].update(model, DataLoader(ds, batch_size=bs, shuffle=True), device)

    return model, update_fn


# ── iCaRL: exemplar replay + distillation ──────────────────────────────────────
def make_icarl(backbone_path, device, cfg):
    import torch.nn.functional as TF
    model     = CharCNN.load(backbone_path, map_location=device).to(device)
    old_model = [None]
    buf       = [[], []]  # [domains, labels]
    buf_size  = cfg.get("ser", {}).get("capacity", 5000)
    seed      = cfg["random_seed"]

    def update_fn(model, train_df, t, cfg):
        epochs = cfg.get("training", {}).get("update_epochs", 5)
        lr     = cfg.get("training", {}).get("lr", 5e-4)
        bs     = cfg.get("training", {}).get("batch_size", 512)
        doms   = train_df["domain"].tolist()
        labs   = train_df["label"].tolist()

        if t > 0:
            mix_d = doms + buf[0]
            mix_l = labs  + buf[1]
            ds     = DomainDataset(mix_d, mix_l)
            loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            scaler    = GradScaler("cuda") if device == "cuda" else None
            model.train()
            for _ in range(epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    if scaler:
                        with autocast("cuda"):
                            logits = model(x)
                            loss = criterion(logits, y)
                            if old_model[0] is not None:
                                with torch.no_grad():
                                    old_logits = old_model[0](x)
                                loss = loss + 0.5 * TF.mse_loss(logits, old_logits)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)
                        if old_model[0] is not None:
                            with torch.no_grad():
                                old_logits = old_model[0](x)
                            loss = loss + 0.5 * TF.mse_loss(logits, old_logits)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

        old_model[0] = copy.deepcopy(model); old_model[0].eval()
        # Update buffer (class-balanced)
        rng  = np.random.default_rng(seed + t)
        half = buf_size // 2
        dga_idx = [i for i, l in enumerate(labs) if l == 1]
        ben_idx = [i for i, l in enumerate(labs) if l == 0]
        sel = (rng.choice(dga_idx, min(half, len(dga_idx)), replace=False).tolist() +
               rng.choice(ben_idx, min(half, len(ben_idx)), replace=False).tolist())
        buf[0] = [doms[i] for i in sel]
        buf[1] = [labs[i]  for i in sel]

    return model, update_fn


# ── GDumb: greedy buffer + retrain from scratch ───────────────────────────────
def make_gdumb(backbone_path, device, cfg):
    model      = CharCNN.load(backbone_path, map_location=device).to(device)
    base_state = CharCNN.load(backbone_path, map_location="cpu").state_dict()
    buf        = [[], []]
    buf_size   = cfg.get("ser", {}).get("capacity", 5000)
    seed       = cfg["random_seed"]

    def update_fn(model, train_df, t, cfg):
        epochs = cfg.get("training", {}).get("update_epochs", 10)
        lr     = cfg.get("training", {}).get("lr", 1e-3)
        bs     = cfg.get("training", {}).get("batch_size", 512)
        doms   = train_df["domain"].tolist()
        labs   = train_df["label"].tolist()
        rng    = np.random.default_rng(seed + t)
        half   = buf_size // 2
        all_d  = buf[0] + doms
        all_l  = buf[1] + labs
        dga_i  = [i for i, l in enumerate(all_l) if l == 1]
        ben_i  = [i for i, l in enumerate(all_l) if l == 0]
        sel    = (rng.choice(dga_i, min(half, len(dga_i)), replace=False).tolist() +
                  rng.choice(ben_i, min(half, len(ben_i)), replace=False).tolist())
        buf[0] = [all_d[i] for i in sel]
        buf[1] = [all_l[i] for i in sel]
        if t > 0 and buf[0]:
            model.load_state_dict(copy.deepcopy(base_state))
            model.to(device)
            _train_on_data(model, buf[0], buf[1], device, epochs=epochs, lr=lr, batch_size=bs)

    return model, update_fn


# ── DRC-CL Ablation variants ─────────────────────────────────────────────────
def make_lora_variant(backbone_path, device, cfg,
                      use_ser=False, use_ewc=False, use_add=False):
    """Factory for DRC-CL và ablation variants."""
    rank   = cfg.get("lora", {}).get("rank", 8)
    alpha  = cfg.get("lora", {}).get("alpha", 16.0)
    model  = CharCNNWithLoRA.from_checkpoint(
        backbone_path, rank=rank, alpha=alpha, map_location=device
    ).to(device)

    buf_size = cfg.get("ser", {}).get("capacity", 5000)
    lam      = cfg.get("ewc", {}).get("lambda", 0.4)
    mu       = cfg.get("training", {}).get("mix_ratio", 0.3)
    seed     = cfg["random_seed"]

    ser = SERBuffer(capacity=buf_size, seed=seed) if use_ser else None
    ewc = EWCRegularizer(lam=lam) if use_ewc else None
    add = [None]
    rng = np.random.default_rng(seed)

    def update_fn(model, train_df, t, cfg):
        epochs = cfg.get("training", {}).get("update_epochs", 5)
        lr     = cfg.get("training", {}).get("lr", 5e-4)
        bs     = cfg.get("training", {}).get("batch_size", 512)
        doms   = train_df["domain"].tolist()
        labs   = train_df["label"].tolist()
        fams   = train_df["family"].tolist()

        # Init ADD on W1
        if use_add and t == 0:
            add[0] = ADDDetector.from_config(cfg)
            embs = extract_embeddings(model, doms, device=device, max_n=5000)
            add[0].calibrate(embs)

        # Init EWC Fisher on W1
        if ewc and t == 0:
            ds0 = DomainDataset(doms[:2000], labs[:2000])
            ewc.update_fisher(model, DataLoader(ds0, batch_size=bs, shuffle=True), device)

        if t == 0:
            # Fill SER buffer from W1
            if ser:
                embs = extract_embeddings(model, doms, device=device, max_n=5000)
                idx  = rng.choice(len(doms), min(5000, len(doms)), replace=False)
                ser.add_batch([doms[i] for i in idx], [labs[i] for i in idx],
                              [fams[i] for i in idx], embs[:len(idx)])
            return

        # Drift detection
        should_update = True
        if use_add and add[0]:
            embs  = extract_embeddings(model, doms, device=device, max_n=5000)
            event = add[0].detect(embs)
            should_update = event.needs_update
            if event.drift_type in ("none", "sudden"):
                add[0].archive_centroid()
            add[0].set_reference(embs)

        if should_update:
            train_d, train_l = doms, labs
            if ser:
                n_buf = int(len(doms) * mu / (1 - mu))
                b_d, b_l, _ = ser.sample(n_buf, rng)
                train_d = doms + b_d
                train_l = labs  + b_l

            ewc_fn = (lambda: ewc.penalty(model)) if ewc else None
            _train_on_data(model, train_d, train_l, device, epochs=epochs, lr=lr,
                           batch_size=bs, ewc_fn=ewc_fn,
                           params_fn=model.lora_parameters)

            if ewc:
                if ser and len(ser) > 0:
                    fb_d, fb_l, _ = ser.sample(min(1024, len(ser)), rng)
                else:
                    fb_d, fb_l = doms[:1024], labs[:1024]
                ds = DomainDataset(fb_d, fb_l)
                ewc.update_fisher(model, DataLoader(ds, batch_size=bs, shuffle=True), device)

        # Update SER buffer
        if ser:
            embs = extract_embeddings(model, doms, device=device, max_n=5000)
            idx  = rng.choice(len(doms), min(5000, len(doms)), replace=False)
            ser.add_batch([doms[i] for i in idx], [labs[i] for i in idx],
                          [fams[i] for i in idx], embs[:len(idx)])

    return model, update_fn


# ══════════════════════════════════════════════════════════════════════════════
# Method registry
# ══════════════════════════════════════════════════════════════════════════════
TABLE_IV_METHODS = [
    ("Static-CNN",  make_static_cnn),
    ("SW-Retrain",  make_sw_retrain),
    ("EWC-only",    make_ewc_only),
    ("iCaRL",       make_icarl),
    ("GDumb",       make_gdumb),
    ("DRC-CL",      lambda bp, d, c: make_lora_variant(bp, d, c,
                        use_ser=True, use_ewc=True, use_add=True)),
]

TABLE_V_METHODS = [
    ("CNN (Static)",            make_static_cnn),
    ("CNN + LoRA Update",       lambda bp, d, c: make_lora_variant(bp, d, c,
                                    use_ser=False, use_ewc=False, use_add=False)),
    ("CNN + LoRA + SER",        lambda bp, d, c: make_lora_variant(bp, d, c,
                                    use_ser=True, use_ewc=False, use_add=False)),
    ("CNN + LoRA + SER + EWC",  lambda bp, d, c: make_lora_variant(bp, d, c,
                                    use_ser=True, use_ewc=True, use_add=False)),
    ("DRC-CL (full)",           lambda bp, d, c: make_lora_variant(bp, d, c,
                                    use_ser=True, use_ewc=True, use_add=True)),
]


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg: dict, backbone_path: Path,
        table: str = "all", only: str | None = None) -> None:

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("run_all_experiments", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["results"])

    logger.info("=" * 65)
    logger.info(" CONTINUAL LEARNING EXPERIMENTS — Full Accuracy Matrix")
    logger.info("=" * 65)
    logger.info(f"  Device   : {device}")
    if device == "cuda":
        logger.info(f"  GPU      : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Backbone : {backbone_path}")
    logger.info(f"  Windows  : {len(get_window_ids(cfg))}")
    logger.info(f"  Table    : {table}")
    logger.info("")

    # Verify train/test splits exist
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    window_ids = get_window_ids(cfg)
    test_file  = bench_dir / f"{window_ids[0]}_test.csv"
    if not test_file.exists():
        logger.error("Train/test splits not found! Run step6 first:")
        logger.error("  python -m src.data.step6_split_train_test")
        return

    all_metrics = []
    t_total     = time.time()

    # ── Table IV ──────────────────────────────────────────────────────────────
    if table in ("all", "iv"):
        methods = TABLE_IV_METHODS
        if only:
            methods = [(n, f) for n, f in methods if n.lower().replace(" ", "-") == only]

        logger.info(f"\n{'═'*65}")
        logger.info(" TABLE IV — Method Comparison")
        logger.info(f"{'═'*65}")

        for name, factory_fn in methods:
            logger.info(f"\n{'─'*65}")
            logger.info(f"  {name}")
            logger.info(f"{'─'*65}")

            model, update_fn = factory_fn(backbone_path, device, cfg)
            result = run_cl_experiment(
                method_name=name, model=model, cfg=cfg, device=device,
                update_fn=update_fn, logger=logger,
            )

            # Save matrix
            matrix_path = out_dir / f"matrix_{name.lower().replace(' ', '_').replace('+', '')}.csv"
            result["matrix"].save(str(matrix_path))
            logger.info(f"    Matrix → {matrix_path}")

            all_metrics.append(result["metrics"])

        # Print Table IV
        if all_metrics:
            print_full_table(all_metrics, logger)
            pd.DataFrame(all_metrics).to_csv(out_dir / "table_iv_matrix.csv", index=False)

    # ── Table V (Ablation) ────────────────────────────────────────────────────
    ablation_metrics = []
    if table in ("all", "v"):
        methods = TABLE_V_METHODS
        if only:
            methods = [(n, f) for n, f in methods if only in n.lower()]

        logger.info(f"\n{'═'*65}")
        logger.info(" TABLE V — Ablation Study")
        logger.info(f"{'═'*65}")

        for name, factory_fn in methods:
            logger.info(f"\n{'─'*65}")
            logger.info(f"  {name}")
            logger.info(f"{'─'*65}")

            model, update_fn = factory_fn(backbone_path, device, cfg)
            result = run_cl_experiment(
                method_name=name, model=model, cfg=cfg, device=device,
                update_fn=update_fn, logger=logger,
            )

            matrix_path = out_dir / f"matrix_ablation_{name.lower().replace(' ', '_').replace('+', '')}.csv"
            result["matrix"].save(str(matrix_path))

            ablation_metrics.append(result["metrics"])

        if ablation_metrics:
            print_full_table(ablation_metrics, logger)
            pd.DataFrame(ablation_metrics).to_csv(out_dir / "table_v_ablation_matrix.csv", index=False)

            # Component contribution
            logger.info(f"\n  Component contribution:")
            logger.info(f"  {'':40s} {'ΔAA-F1':>9} {'ΔBWT':>9} {'ΔForget':>9} {'ΔF1-Old':>9}")
            logger.info(f"  {'─'*80}")
            for i in range(1, len(ablation_metrics)):
                p = ablation_metrics[i-1]
                c = ablation_metrics[i]
                df1 = c["aa_f1"] - p["aa_f1"]
                dbwt = c["bwt"] - p["bwt"]
                dfgt = c["forgetting"] - p["forgetting"]
                dold = (c.get("f1_old",0) or 0) - (p.get("f1_old",0) or 0)
                logger.info(
                    f"  {p['method']:<20s} → {c['method']:<20s}"
                    f"{df1:>+9.4f} {dbwt:>+9.4f} {dfgt:>+9.4f} {dold:>+9.4f}"
                )

    total_time = time.time() - t_total
    logger.info(f"\n{'═'*65}")
    logger.info(f" TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"{'═'*65}")
    logger.info(f"\n  Output files:")
    logger.info(f"    results/table_iv_matrix.csv         ← Table IV")
    logger.info(f"    results/table_v_ablation_matrix.csv ← Table V")
    logger.info(f"    results/matrix_*.csv                ← accuracy matrices")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all CL experiments with accuracy matrix")
    parser.add_argument("--config",   default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--table",    default="all", choices=["all", "iv", "v"])
    parser.add_argument("--only",     default=None,
                        help="Run only this method (e.g. 'drc-cl', 'static-cnn')")
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

    run(cfg, backbone_path, table=args.table, only=args.only)
