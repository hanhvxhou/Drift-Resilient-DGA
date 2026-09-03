"""
src/models/adaptive_replay.py
──────────────────────────────
Adaptive Replay Weight: μ_t thich ung theo drift magnitude.

Novelty:
  - Fixed replay: μ = 0.3 (constant)
  - Adaptive replay: μ_t = f(m_t, δ₁, δ₂)
    * Drift nhe (m_t ~ δ₂): μ cao → replay nhieu, bao toan kien thuc
    * Drift manh (m_t >> δ₁): μ thap → uu tien du lieu moi
    * No drift (forced update): μ = μ_max → toi da bao toan

  Formula:
    μ_t = μ_min + (μ_max − μ_min) · exp(−α · max(0, m_t − δ₂) / (δ₁ − δ₂))

  where:
    μ_min = 0.1  (minimum replay ratio, for sudden drift)
    μ_max = 0.5  (maximum replay ratio, for mild/no drift)
    α = 2.0      (decay rate)

Usage:
    python -m src.models.adaptive_replay
    python -m src.models.adaptive_replay --compare
"""

from __future__ import annotations
import argparse, time, copy, json
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
from src.models.lora_adapter import CharCNNWithLoRA
from src.models.cl_metrics import AccuracyMatrix
from src.detect.add_detector import ADDDetector, extract_embeddings
from src.utils.common import get_logger, load_config, get_window_ids


class AdaptiveReplayWeight:
    """
    Adaptive replay weight μ_t based on drift magnitude.
    
    Core novelty: creates a feedback loop between drift detection (ADD)
    and knowledge preservation (SER), modulating the stability-plasticity
    trade-off based on empirical distributional distance.
    
    μ_t = μ_min + (μ_max − μ_min) · exp(−α · z_t)
    
    where z_t = max(0, m_t − δ₂) / (δ₁ − δ₂) is the normalized drift intensity.
    
    Behavior:
      - z_t = 0 (no drift / forced): μ_t = μ_max → maximum replay
      - z_t = 1 (δ₁ boundary):       μ_t ≈ μ_min + 0.14·(μ_max−μ_min)
      - z_t >> 1 (extreme drift):     μ_t → μ_min → prioritize new data
    """
    
    def __init__(self, mu_min=0.1, mu_max=0.5, alpha=2.0):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.alpha = alpha
        self.history = []  # log of (window, m_t, mu_t, drift_type)
    
    def compute(self, m_t, delta1, delta2, drift_type="gradual"):
        """Compute adaptive μ_t given drift magnitude and thresholds."""
        if drift_type == "forced" or m_t <= delta2:
            # Forced update or no significant drift: maximum replay
            mu_t = self.mu_max
        else:
            # Normalize drift intensity
            z_t = max(0, m_t - delta2) / max(delta1 - delta2, 1e-10)
            mu_t = self.mu_min + (self.mu_max - self.mu_min) * np.exp(-self.alpha * z_t)
        
        self.history.append({
            "m_t": float(m_t), "mu_t": float(mu_t),
            "drift_type": drift_type,
            "delta1": float(delta1), "delta2": float(delta2)
        })
        return mu_t
    
    def get_summary(self):
        """Return summary statistics of adaptive μ."""
        if not self.history:
            return {}
        mus = [h["mu_t"] for h in self.history]
        return {
            "mu_mean": round(np.mean(mus), 4),
            "mu_std": round(np.std(mus), 4),
            "mu_min_actual": round(min(mus), 4),
            "mu_max_actual": round(max(mus), 4),
            "n_updates": len(mus),
        }


class DomainDataset(Dataset):
    def __init__(self, domains, labels):
        self.domains = domains
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.domains)
    def __getitem__(self, idx):
        return domain_to_tensor(self.domains[idx]), self.labels[idx]


class EWCReg:
    """Lightweight EWC for adapter parameters."""
    def __init__(self, lam=0.4):
        self.lam = lam
        self.fisher = {}
        self.optpar = {}
    
    def update(self, model, loader, device):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        crit = nn.BCEWithLogitsLoss()
        n_samples = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            n_samples += len(y)
        for n in fisher:
            fisher[n] /= max(n_samples, 1)
        self.fisher = {n: f.clone() for n, f in fisher.items()}
        self.optpar = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.optpar[n]).pow(2)).sum()
        return self.lam * loss


def run_drc_cl(cfg, seed, device, logger, split_dir, window_ids,
               backbone_path, adaptive=False):
    """Run DRC-CL with fixed or adaptive replay weight."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    rank = cfg.get("lora", {}).get("rank", 8)
    alpha_lora = cfg.get("lora", {}).get("alpha", 16.0)
    epochs = cfg.get("training", {}).get("update_epochs", 5)
    lr = cfg.get("training", {}).get("lr", 5e-4)
    bs = cfg.get("training", {}).get("batch_size", 512)
    mu_fixed = cfg.get("training", {}).get("mix_ratio", 0.3)
    lam = cfg.get("ewc", {}).get("lambda", 0.4)
    buf_cap = cfg.get("ser", {}).get("capacity", 5000)
    
    T = len(window_ids)
    model = CharCNNWithLoRA.from_checkpoint(
        backbone_path, rank=rank, alpha=alpha_lora, map_location=device).to(device)
    
    add = ADDDetector(max_no_update=4)
    ewc = EWCReg(lam=lam)
    arw = AdaptiveReplayWeight(mu_min=0.1, mu_max=0.5, alpha=2.0) if adaptive else None
    
    buf_d, buf_l = [], []
    matrix = AccuracyMatrix(window_ids)
    mu_log = []
    
    for t in range(T):
        win_id = window_ids[t]
        train_df = pd.read_csv(split_dir / f"{win_id}_train.csv")
        train_d = train_df["domain"].tolist()
        train_l = train_df["label"].tolist()
        
        if t == 0:
            # Pretrain
            ds = DomainDataset(train_d, train_l)
            loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
            opt = torch.optim.Adam(model.lora_parameters(), lr=lr)
            crit = nn.BCEWithLogitsLoss()
            scaler = GradScaler("cuda") if device == "cuda" else None
            model.train()
            for _ in range(epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    if scaler:
                        with autocast("cuda"):
                            loss = crit(model(x), y)
                        scaler.scale(loss).backward()
                        scaler.step(opt); scaler.update()
                    else:
                        loss = crit(model(x), y)
                        loss.backward(); opt.step()
            
            embs = extract_embeddings(model, train_d, device=device, max_n=5000)
            add.calibrate(embs)
            ds0 = DomainDataset(train_d[:1000], train_l[:1000])
            ewc.update(model, DataLoader(ds0, batch_size=bs, shuffle=True), device)
            
            n_add = min(buf_cap, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            buf_d = [train_d[i] for i in idx]
            buf_l = [train_l[i] for i in idx]
        else:
            embs = extract_embeddings(model, train_d, device=device, max_n=5000)
            event = add.detect(embs)
            
            if event.needs_update:
                # Compute μ_t
                if adaptive and arw:
                    mu_t = arw.compute(
                        event.mmd2, add.delta1, add.delta2, event.drift_type
                    )
                else:
                    mu_t = mu_fixed
                
                mu_log.append({"window": win_id, "mu": mu_t, 
                              "drift": event.drift_type, "mmd2": event.mmd2})
                
                # Mix buffer + new data with μ_t
                # Key: μ_t controls the ACTUAL ratio in the training batch,
                # not just the number of buffer samples requested.
                # Total batch size = |D_t|. Sample (1-μ)×|D_t| from D_t, μ×|D_t| from buffer.
                total_size = len(train_d)
                n_from_buf = int(total_size * mu_t)
                n_from_new = total_size - n_from_buf
                
                # Subsample new data
                if n_from_new < len(train_d):
                    new_idx = rng.choice(len(train_d), n_from_new, replace=False)
                    sub_d = [train_d[i] for i in new_idx]
                    sub_l = [train_l[i] for i in new_idx]
                else:
                    sub_d, sub_l = train_d, train_l
                
                # Sample from buffer (with replacement if needed)
                if buf_d and n_from_buf > 0:
                    replace = n_from_buf > len(buf_d)
                    b_idx = rng.choice(len(buf_d), min(n_from_buf, len(buf_d)), replace=replace)
                    buf_sample_d = [buf_d[i] for i in b_idx]
                    buf_sample_l = [buf_l[i] for i in b_idx]
                    mix_d = sub_d + buf_sample_d
                    mix_l = sub_l + buf_sample_l
                else:
                    mix_d, mix_l = sub_d, sub_l
                
                ds = DomainDataset(mix_d, mix_l)
                loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
                opt = torch.optim.Adam(model.lora_parameters(), lr=lr)
                crit = nn.BCEWithLogitsLoss()
                scaler = GradScaler("cuda") if device == "cuda" else None
                model.train()
                for _ in range(epochs):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        if scaler:
                            with autocast("cuda"):
                                loss = crit(model(x), y) + ewc.penalty(model)
                            scaler.scale(loss).backward()
                            scaler.step(opt); scaler.update()
                        else:
                            loss = crit(model(x), y) + ewc.penalty(model)
                            loss.backward(); opt.step()
                
                ds_f = DomainDataset(train_d[:512], train_l[:512])
                ewc.update(model, DataLoader(ds_f, batch_size=bs, shuffle=True), device)
            
            # Update buffer
            n_add = min(buf_cap, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            new_d = [train_d[i] for i in idx]
            new_l = [train_l[i] for i in idx]
            buf_d = (buf_d + new_d)[-buf_cap:]
            buf_l = (buf_l + new_l)[-buf_cap:]
        
        # Evaluate
        model.eval()
        row = {}
        with torch.no_grad():
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
        matrix.add_row(t, row)
    
    metrics = matrix.compute_metrics()
    metrics["mu_log"] = mu_log
    if arw:
        metrics["arw_summary"] = arw.get_summary()
    
    return metrics


def run(cfg, n_seeds=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("adaptive_replay", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    backbone_path = out_dir / "checkpoints" / "backbone_d01.pt"
    window_ids = get_window_ids(cfg)
    
    cfg.setdefault("lora", {"rank": 8, "alpha": 16.0})
    cfg.setdefault("ser", {"capacity": 5000, "beta": 0.92, "min_k": 50})
    cfg.setdefault("ewc", {"lambda": 0.4})
    cfg.setdefault("training", {"lr": 5e-4, "update_epochs": 5,
                                "batch_size": 512, "mix_ratio": 0.3})
    
    seeds = [42, 123, 456, 789, 2024][:n_seeds]
    
    logger.info("=" * 65)
    logger.info(" ADAPTIVE REPLAY WEIGHT EXPERIMENT")
    logger.info("=" * 65)
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Formula: μ_t = μ_min + (μ_max − μ_min) · exp(−α · z_t)")
    logger.info(f"  μ_min=0.1, μ_max=0.5, α=2.0")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    results = {"fixed": [], "adaptive": []}
    
    for mode in ["fixed", "adaptive"]:
        adaptive = (mode == "adaptive")
        logger.info(f"\n{'─'*65}")
        logger.info(f"  Mode: {'ADAPTIVE μ_t' if adaptive else 'FIXED μ=0.3'}")
        logger.info(f"{'─'*65}")
        
        for i, seed in enumerate(seeds):
            logger.info(f"\n  Seed {seed} ({i+1}/{len(seeds)})...")
            t0 = time.time()
            
            m = run_drc_cl(cfg, seed, device, logger, split_dir, window_ids,
                          backbone_path, adaptive=adaptive)
            
            elapsed = time.time() - t0
            m["seed"] = seed
            m["mode"] = mode
            results[mode].append(m)
            
            arw_info = ""
            if adaptive and "arw_summary" in m:
                s = m["arw_summary"]
                arw_info = f"  μ_mean={s['mu_mean']:.3f} [{s['mu_min_actual']:.3f}–{s['mu_max_actual']:.3f}]"
            
            logger.info(f"    AA-F1={m['aa_f1']:.4f}  Forg={m['forgetting']:.4f}  "
                        f"BWT={m['bwt']:+.4f}{arw_info}  ({elapsed:.1f}s)")
    
    # ── Print comparison ──────────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info(" COMPARISON: Fixed μ vs Adaptive μ")
    logger.info(f"{'='*65}")
    
    for metric in ["aa_f1", "forgetting", "bwt", "degrad"]:
        fixed_vals = [r[metric] for r in results["fixed"]]
        adapt_vals = [r[metric] for r in results["adaptive"]]
        
        f_mean, f_std = np.mean(fixed_vals), np.std(fixed_vals)
        a_mean, a_std = np.mean(adapt_vals), np.std(adapt_vals)
        diff = a_mean - f_mean
        
        better = "adaptive" if (metric == "forgetting" and diff < 0) or \
                               (metric != "forgetting" and diff > 0) else "fixed"
        marker = "★" if better == "adaptive" else ""
        
        logger.info(f"  {metric:<12} Fixed: {f_mean:>8.4f}±{f_std:.4f}  "
                    f"Adaptive: {a_mean:>8.4f}±{a_std:.4f}  "
                    f"Δ={diff:>+8.4f} {marker}")
    
    # Wilcoxon test
    from scipy.stats import wilcoxon
    for metric in ["aa_f1", "forgetting"]:
        fixed_vals = [r[metric] for r in results["fixed"]]
        adapt_vals = [r[metric] for r in results["adaptive"]]
        try:
            stat, p = wilcoxon(fixed_vals, adapt_vals)
            sig = "p<0.05 *" if p < 0.05 else f"p={p:.4f} ns"
            logger.info(f"  Wilcoxon {metric}: {sig}")
        except:
            logger.info(f"  Wilcoxon {metric}: cannot compute (identical?)")
    
    # Log adaptive μ behavior
    logger.info(f"\n  Adaptive μ behavior (last seed):")
    last_adaptive = results["adaptive"][-1]
    if "mu_log" in last_adaptive:
        for entry in last_adaptive["mu_log"]:
            logger.info(f"    {entry['window']}: drift={entry['drift']:<8} "
                        f"MMD²={entry['mmd2']:.4f}  μ={entry['mu']:.3f}")
    
    if "arw_summary" in last_adaptive:
        s = last_adaptive["arw_summary"]
        logger.info(f"\n  μ summary: mean={s['mu_mean']:.3f}  "
                    f"range=[{s['mu_min_actual']:.3f}, {s['mu_max_actual']:.3f}]  "
                    f"updates={s['n_updates']}")
    
    # Save
    save_data = {
        "fixed": [{"seed": r["seed"], "aa_f1": r["aa_f1"], "bwt": r["bwt"],
                   "forgetting": r["forgetting"], "degrad": r["degrad"]}
                  for r in results["fixed"]],
        "adaptive": [{"seed": r["seed"], "aa_f1": r["aa_f1"], "bwt": r["bwt"],
                      "forgetting": r["forgetting"], "degrad": r["degrad"],
                      "arw_summary": r.get("arw_summary", {}),
                      "mu_log": r.get("mu_log", [])}
                     for r in results["adaptive"]],
    }
    
    save_path = out_dir / "adaptive_replay_comparison.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"\n  Saved: {save_path}")
    logger.info("  Adaptive replay experiment complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Replay Weight")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    run(load_config(args.config), n_seeds=args.seeds)
