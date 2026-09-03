"""
src/models/cl_baselines_extra.py
─────────────────────────────────
DER++ va AGEM baselines cho so sanh voi DRC-CL.

DER++ (Dark Experience Replay++):
  - Luu (domain, label, logit) trong buffer
  - Loss = L_CE(new) + α·MSE(logit_replay, stored_logit) + β·L_CE(replay)
  - Ref: Buzzega et al., "Dark Experience for General CL," NeurIPS 2020

AGEM (Averaged Gradient Episodic Memory):
  - Chieu gradient hien tai len vung kha thi cua gradient memory
  - Ref: Chaudhry et al., "Efficient Lifelong Learning with A-GEM," ICLR 2019

Usage:
    python -m src.models.cl_baselines_extra
    python -m src.models.cl_baselines_extra --method derpp
    python -m src.models.cl_baselines_extra --method agem
"""

from __future__ import annotations
import argparse, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from scipy.special import expit as sigmoid_stable

from src.models.char_cnn import CharCNN, domain_to_tensor, domains_to_batch
from src.models.cl_metrics import AccuracyMatrix
from src.utils.common import get_logger, load_config, get_window_ids


class DomainDataset(Dataset):
    def __init__(self, domains, labels):
        self.domains = domains
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.domains)
    def __getitem__(self, idx):
        return domain_to_tensor(self.domains[idx]), self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# DER++ Buffer: stores (domain, label, logit)
# ══════════════════════════════════════════════════════════════════════════════
class DERBuffer:
    """Buffer storing (domain, label, logit) for Dark Experience Replay."""
    
    def __init__(self, capacity=5000, seed=42):
        self.capacity = capacity
        self.domains = []
        self.labels = []
        self.logits = []
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
    
    def add_batch(self, domains, labels, logits):
        """Add samples with reservoir sampling, storing logits."""
        for d, l, lg in zip(domains, labels, logits):
            self.n_seen += 1
            if len(self.domains) < self.capacity:
                self.domains.append(d)
                self.labels.append(l)
                self.logits.append(lg)
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.capacity:
                    self.domains[j] = d
                    self.labels[j] = l
                    self.logits[j] = lg
    
    def sample(self, n):
        """Sample n items from buffer."""
        if len(self.domains) == 0:
            return [], [], []
        n = min(n, len(self.domains))
        idx = self.rng.choice(len(self.domains), n, replace=False)
        return ([self.domains[i] for i in idx],
                [self.labels[i] for i in idx],
                [self.logits[i] for i in idx])


# ══════════════════════════════════════════════════════════════════════════════
# DER++ Method
# ══════════════════════════════════════════════════════════════════════════════
def run_derpp(cfg, seed, device, logger, split_dir, window_ids, backbone_path):
    """
    DER++ (Dark Experience Replay++).
    Loss = L_CE(new_data) + α·MSE(replay_logit, stored_logit) + β·L_CE(replay)
    α=0.5, β=0.5 (default from Buzzega et al.)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    T = len(window_ids)
    epochs = 5
    lr = 5e-4
    bs = 512
    alpha_der = 0.5  # weight for logit distillation
    beta_der = 0.5   # weight for replay CE
    buf_cap = 5000
    
    model = CharCNN.load(backbone_path, map_location=device).to(device)
    buffer = DERBuffer(capacity=buf_cap, seed=seed)
    matrix = AccuracyMatrix(window_ids)
    
    for t in range(T):
        win_id = window_ids[t]
        train_df = pd.read_csv(split_dir / f"{win_id}_train.csv")
        train_d = train_df["domain"].tolist()
        train_l = train_df["label"].tolist()
        
        # Train
        ds = DomainDataset(train_d, train_l)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        scaler = GradScaler("cuda") if device == "cuda" else None
        
        model.train()
        for _ in range(epochs):
            for x_new, y_new in loader:
                x_new, y_new = x_new.to(device), y_new.to(device)
                optimizer.zero_grad()
                
                if scaler:
                    with autocast("cuda"):
                        logits_new = model(x_new)
                        loss = criterion(logits_new, y_new)
                        
                        # Replay component
                        if len(buffer.domains) >= bs:
                            buf_d, buf_l, buf_lg = buffer.sample(bs)
                            x_buf = domains_to_batch(buf_d).to(device)
                            y_buf = torch.tensor(buf_l, dtype=torch.float32).to(device)
                            stored_lg = torch.tensor(buf_lg, dtype=torch.float32).to(device)
                            
                            logits_buf = model(x_buf)
                            # DER++ loss: distillation + CE on buffer
                            loss += alpha_der * mse(logits_buf, stored_lg)
                            loss += beta_der * criterion(logits_buf, y_buf)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits_new = model(x_new)
                    loss = criterion(logits_new, y_new)
                    if len(buffer.domains) >= bs:
                        buf_d, buf_l, buf_lg = buffer.sample(bs)
                        x_buf = domains_to_batch(buf_d).to(device)
                        y_buf = torch.tensor(buf_l, dtype=torch.float32).to(device)
                        stored_lg = torch.tensor(buf_lg, dtype=torch.float32).to(device)
                        logits_buf = model(x_buf)
                        loss += alpha_der * mse(logits_buf, stored_lg)
                        loss += beta_der * criterion(logits_buf, y_buf)
                    loss.backward()
                    optimizer.step()
        
        # Store current logits in buffer
        model.eval()
        with torch.no_grad():
            store_n = min(buf_cap, len(train_d))
            rng = np.random.default_rng(seed + t)
            idx = rng.choice(len(train_d), store_n, replace=False)
            store_d = [train_d[i] for i in idx]
            store_l = [train_l[i] for i in idx]
            store_logits = []
            for i in range(0, len(store_d), bs):
                x = domains_to_batch(store_d[i:i+bs]).to(device)
                lg = model(x).cpu().numpy()
                store_logits.extend(lg.tolist())
            buffer.add_batch(store_d, store_l, store_logits)
        
        # Evaluate
        row = eval_all_windows(model, split_dir, window_ids, t, device, bs)
        matrix.add_row(t, row)
    
    return matrix.compute_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# AGEM Method
# ══════════════════════════════════════════════════════════════════════════════
def run_agem(cfg, seed, device, logger, split_dir, window_ids, backbone_path):
    """
    A-GEM (Averaged Gradient Episodic Memory).
    If gradient on current task conflicts with gradient on memory,
    project current gradient to not increase memory loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    T = len(window_ids)
    epochs = 5
    lr = 5e-4
    bs = 512
    buf_cap = 5000
    rng = np.random.default_rng(seed)
    
    model = CharCNN.load(backbone_path, map_location=device).to(device)
    
    # Episodic memory buffer (simple reservoir)
    mem_d, mem_l = [], []
    matrix = AccuracyMatrix(window_ids)
    
    for t in range(T):
        win_id = window_ids[t]
        train_df = pd.read_csv(split_dir / f"{win_id}_train.csv")
        train_d = train_df["domain"].tolist()
        train_l = train_df["label"].tolist()
        
        ds = DomainDataset(train_d, train_l)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        for _ in range(epochs):
            for x_new, y_new in loader:
                x_new, y_new = x_new.to(device), y_new.to(device)
                
                # Compute gradient on current batch
                optimizer.zero_grad()
                loss_new = criterion(model(x_new), y_new)
                loss_new.backward()
                
                # Store current gradient
                grad_new = []
                for p in model.parameters():
                    if p.grad is not None:
                        grad_new.append(p.grad.data.clone().flatten())
                    else:
                        grad_new.append(torch.zeros(p.numel(), device=device))
                grad_new = torch.cat(grad_new)
                
                # AGEM projection (if we have memory)
                if len(mem_d) >= bs:
                    # Compute gradient on memory
                    optimizer.zero_grad()
                    m_idx = rng.choice(len(mem_d), min(bs, len(mem_d)), replace=False)
                    m_doms = [mem_d[i] for i in m_idx]
                    m_labs = [mem_l[i] for i in m_idx]
                    x_mem = domains_to_batch(m_doms).to(device)
                    y_mem = torch.tensor(m_labs, dtype=torch.float32).to(device)
                    
                    loss_mem = criterion(model(x_mem), y_mem)
                    loss_mem.backward()
                    
                    grad_mem = []
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_mem.append(p.grad.data.clone().flatten())
                        else:
                            grad_mem.append(torch.zeros(p.numel(), device=device))
                    grad_mem = torch.cat(grad_mem)
                    
                    # Project if conflict: dot(g_new, g_mem) < 0
                    dot = torch.dot(grad_new, grad_mem)
                    if dot < 0:
                        # Project: g_proj = g_new - (dot / ||g_mem||²) * g_mem
                        ref_mag = torch.dot(grad_mem, grad_mem)
                        if ref_mag > 1e-10:
                            grad_new = grad_new - (dot / ref_mag) * grad_mem
                
                # Apply (projected) gradient
                optimizer.zero_grad()
                offset = 0
                for p in model.parameters():
                    numel = p.numel()
                    if p.requires_grad:
                        p.grad = grad_new[offset:offset+numel].reshape(p.shape)
                    offset += numel
                optimizer.step()
        
        # Update memory (reservoir sampling)
        n_add = min(buf_cap, len(train_d))
        idx = rng.choice(len(train_d), n_add, replace=False)
        new_d = [train_d[i] for i in idx]
        new_l = [train_l[i] for i in idx]
        mem_d = (mem_d + new_d)[-buf_cap:]
        mem_l = (mem_l + new_l)[-buf_cap:]
        
        # Evaluate
        row = eval_all_windows(model, split_dir, window_ids, t, device, bs)
        matrix.add_row(t, row)
    
    return matrix.compute_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# Shared evaluation
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_all_windows(model, split_dir, window_ids, t, device, bs=512):
    """Evaluate model on all windows up to t."""
    model.eval()
    row = {}
    for j in range(t + 1):
        test_df = pd.read_csv(split_dir / f"{window_ids[j]}_test.csv")
        doms = test_df["domain"].tolist()
        labs = np.array(test_df["label"].tolist())
        all_logits = []
        for i in range(0, len(doms), bs):
            x = domains_to_batch(doms[i:i+bs]).to(device)
            all_logits.append(model(x).cpu().numpy())
        preds = (sigmoid_stable(np.concatenate(all_logits)) >= 0.5).astype(int)
        row[window_ids[j]] = {"f1": f1_score(labs, preds, zero_division=0)}
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg, method=None, n_seeds=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("cl_extra", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    backbone_path = out_dir / "checkpoints" / "backbone_d01.pt"
    window_ids = get_window_ids(cfg)
    
    seeds = [42, 123, 456, 789, 2024, 3141, 5926, 5358, 9793, 2384][:n_seeds]
    methods_to_run = [method] if method else ["derpp", "agem"]
    
    logger.info("=" * 65)
    logger.info(f" EXTRA CL BASELINES: {', '.join(m.upper() for m in methods_to_run)}")
    logger.info("=" * 65)
    logger.info(f"  Seeds: {len(seeds)}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    all_results = {}
    
    for meth in methods_to_run:
        logger.info(f"\n{'─'*65}")
        logger.info(f"  Method: {meth.upper()}")
        logger.info(f"{'─'*65}")
        
        results = []
        for i, seed in enumerate(seeds):
            logger.info(f"\n  Seed {seed} ({i+1}/{len(seeds)})...")
            t0 = time.time()
            
            if meth == "derpp":
                m = run_derpp(cfg, seed, device, logger, split_dir, window_ids, backbone_path)
            elif meth == "agem":
                m = run_agem(cfg, seed, device, logger, split_dir, window_ids, backbone_path)
            
            elapsed = time.time() - t0
            m["seed"] = seed
            results.append(m)
            logger.info(f"    AA-F1={m['aa_f1']:.4f}  Forg={m['forgetting']:.4f}  "
                        f"BWT={m['bwt']:+.4f}  ({elapsed:.1f}s)")
        
        # Aggregate
        all_results[meth] = results
        
        logger.info(f"\n  {meth.upper()} — {len(seeds)} seeds:")
        for metric in ["aa_f1", "bwt", "forgetting", "degrad"]:
            vals = [r[metric] for r in results]
            logger.info(f"    {metric:<12} {np.mean(vals):>8.4f} ± {np.std(vals):.4f}")
    
    # ── Comparison with existing methods ──────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info(" COMPARISON TABLE (all methods)")
    logger.info(f"{'='*65}")
    
    # Load existing results
    agg_path = out_dir / "multi_seed" / "aggregated_results.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        logger.info(f"  {'Method':<24} {'AA-F1':>12} {'Forg.':>12} {'BWT':>12} {'Params':>10}")
        logger.info(f"  {'-'*70}")
        
        for _, r in agg.iterrows():
            if r['method'] in ['Static-CNN', 'EWC-only', 'iCaRL', 'GDumb', 'DRC-CL']:
                logger.info(f"  {r['method']:<24} {r['aa_f1_mean']:>8.4f}±{r['aa_f1_std']:.4f} "
                           f"{r['forgetting_mean']:>8.4f}±{r['forgetting_std']:.4f} "
                           f"{r['bwt_mean']:>+8.4f} {'429K' if r['method']!='DRC-CL' else '12K':>10}")
    
    for meth, results in all_results.items():
        vals_aa = [r["aa_f1"] for r in results]
        vals_fg = [r["forgetting"] for r in results]
        vals_bw = [r["bwt"] for r in results]
        logger.info(f"  {meth.upper():<24} {np.mean(vals_aa):>8.4f}±{np.std(vals_aa):.4f} "
                    f"{np.mean(vals_fg):>8.4f}±{np.std(vals_fg):.4f} "
                    f"{np.mean(vals_bw):>+8.4f} {'429K':>10}")
    
    # Save
    save_data = {}
    for meth, results in all_results.items():
        save_data[meth] = [{
            "seed": r["seed"], "aa_f1": r["aa_f1"], "bwt": r["bwt"],
            "forgetting": r["forgetting"], "degrad": r["degrad"]
        } for r in results]
    
    save_path = out_dir / "extra_baselines_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    
    # Also save as CSV
    rows = []
    for meth, results in all_results.items():
        for r in results:
            rows.append({"method": meth.upper(), "seed": r["seed"],
                        "aa_f1": r["aa_f1"], "bwt": r["bwt"],
                        "forgetting": r["forgetting"], "degrad": r["degrad"]})
    pd.DataFrame(rows).to_csv(out_dir / "extra_baselines_results.csv", index=False)
    
    logger.info(f"\n  Saved: {save_path}")
    logger.info(f"  Saved: {out_dir / 'extra_baselines_results.csv'}")
    logger.info("  Extra baselines complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DER++ and AGEM baselines")
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", default=None, choices=["derpp", "agem"],
                        help="Run only this method (default: both)")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    run(load_config(args.config), method=args.method, n_seeds=args.seeds)
