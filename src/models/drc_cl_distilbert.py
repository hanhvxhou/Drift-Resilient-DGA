"""
src/models/drc_cl_distilbert.py
────────────────────────────────
DRC-CL with DistilBERT backbone + PEFT-LoRA.
Chung minh framework architecture-agnostic.

Components (giong CharCNN version):
  - DistilBERT backbone (66M params, DONG BANG)
  - PEFT LoRA adapters (r=8, ~300K trainable params)
  - SER buffer (M=5000)
  - EWC regularization (lambda=0.4)
  - ADD drift detector (MMD2 on [CLS] embeddings)

Protocol: accuracy matrix, train on _train.csv, eval on _test.csv

Requirements:
    pip install transformers peft

Usage:
    python -m src.models.drc_cl_distilbert
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
from sklearn.metrics import f1_score
from scipy.special import expit as sigmoid_stable

from src.models.cl_metrics import AccuracyMatrix, print_metrics_table, print_per_type_table
from src.utils.common import get_logger, get_window_ids, load_config
from src.utils.dga_taxonomy import split_by_dga_type


# ── Model ─────────────────────────────────────────────────────────────────────
class DistilBERTWithLoRA(nn.Module):
    """DistilBERT + PEFT LoRA for binary DGA classification."""

    def __init__(self, model_name="distilbert-base-uncased",
                 lora_r=8, lora_alpha=16, dropout=0.3):
        super().__init__()
        from transformers import DistilBertModel
        from peft import get_peft_model, LoraConfig, TaskType

        base = DistilBertModel.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules=["q_lin", "v_lin"],  # attention Q, V matrices
            bias="none",
        )
        self.bert = get_peft_model(base, lora_config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base.config.hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # Freeze everything except LoRA + classifier
        for name, param in self.bert.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls).squeeze(1)

    def get_embeddings(self, input_ids, attention_mask):
        """[CLS] embedding for ADD drift detection."""
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable, "pct": round(trainable/total*100, 2)}

    def save_adapter(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        torch.save(state, path)

    def load_adapter(self, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        current = dict(self.named_parameters())
        for name, data in state.items():
            if name in current and current[name].requires_grad:
                current[name].data.copy_(data)


# ── Dataset ───────────────────────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, domains, labels, tokenizer, max_len=64):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.enc = tokenizer(domains, padding="max_length", truncation=True,
                             max_length=max_len, return_tensors="pt")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.enc["input_ids"][i], self.enc["attention_mask"][i], self.labels[i]


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_on_test(model, test_df, tokenizer, device, batch_size=64):
    model.eval()
    domains = test_df["domain"].tolist()
    labels = np.array(test_df["label"].tolist())
    all_logits = []
    for i in range(0, len(domains), batch_size):
        enc = tokenizer(domains[i:i+batch_size], padding="max_length",
                        truncation=True, max_length=64, return_tensors="pt")
        with autocast("cuda", enabled=(device=="cuda")):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_logits.append(logits.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    probs = sigmoid_stable(logits_np)
    preds = (probs >= 0.5).astype(int)
    from sklearn.metrics import roc_auc_score
    result = {"f1": f1_score(labels, preds, zero_division=0),
              "auc": roc_auc_score(labels, probs) if labels.sum()>0 and (1-labels).sum()>0 else 0.0}
    df_char, df_word = split_by_dga_type(test_df)
    for sname, sdf in [("f1_char", df_char), ("f1_word", df_word)]:
        if len(sdf) < 10 or sdf["label"].nunique() < 2:
            result[sname] = float("nan"); continue
        sub_idx = test_df.index.isin(sdf.index)
        sub_p = probs[sub_idx[:len(probs)]]
        sub_l = np.array(sdf["label"].tolist())
        n = min(len(sub_p), len(sub_l))
        result[sname] = f1_score(sub_l[:n], (sub_p[:n]>=0.5).astype(int), zero_division=0)
    return result


def build_row(model, tokenizer, split_dir, window_ids, up_to_t, device, bs=64):
    row = {}
    for s in range(up_to_t + 1):
        test_path = split_dir / f"{window_ids[s]}_test.csv"
        if not test_path.exists(): continue
        row[window_ids[s]] = eval_on_test(model, pd.read_csv(test_path), tokenizer, device, bs)
    return row


# ── Embedding extraction for ADD ──────────────────────────────────────────────
@torch.no_grad()
def extract_cls_embeddings(model, domains, tokenizer, device, batch_size=64, max_n=5000):
    model.eval()
    if len(domains) > max_n:
        idx = np.random.choice(len(domains), max_n, replace=False)
        domains = [domains[i] for i in idx]
    all_embs = []
    for i in range(0, len(domains), batch_size):
        enc = tokenizer(domains[i:i+batch_size], padding="max_length",
                        truncation=True, max_length=64, return_tensors="pt")
        emb = model.get_embeddings(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


# ── SER Buffer (simplified) ──────────────────────────────────────────────────
class SERBuffer:
    def __init__(self, capacity=5000, seed=42):
        self.capacity = capacity
        self.half = capacity // 2
        self.rng = np.random.default_rng(seed)
        self._buf_0, self._buf_1 = [], []
        self._n = 0

    def add_batch(self, domains, labels):
        for d, l in zip(domains, labels):
            self._n += 1
            buf = self._buf_1 if l == 1 else self._buf_0
            if len(buf) < self.half:
                buf.append((d, l))
            elif self.rng.random() < self.half / self._n:
                buf[self.rng.integers(len(buf))] = (d, l)

    def sample(self, n):
        all_items = self._buf_0 + self._buf_1
        if not all_items: return [], []
        n = min(n, len(all_items))
        idx = self.rng.choice(len(all_items), n, replace=False)
        return [all_items[i][0] for i in idx], [all_items[i][1] for i in idx]

    def __len__(self): return len(self._buf_0) + len(self._buf_1)


# ── EWC ───────────────────────────────────────────────────────────────────────
class EWCReg:
    def __init__(self, lam=0.4):
        self.lam = lam
        self.fisher, self.theta_star = {}, {}

    def update(self, model, loader, device):
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        fisher_acc = {}
        count = 0
        for ids, mask, y in loader:
            if count > 512: break
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            model.zero_grad()
            loss = criterion(model(ids, mask), y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if n not in fisher_acc:
                        fisher_acc[n] = torch.zeros_like(p.data)
                    fisher_acc[n] += p.grad.data.pow(2)
            count += len(y)
        nb = max(count / loader.batch_size, 1)
        self.fisher = {k: v/nb for k, v in fisher_acc.items()}
        self.theta_star = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        if not self.fisher: return torch.tensor(0.0)
        dev = next(model.parameters()).device
        loss = torch.tensor(0.0, device=dev)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n].to(dev) * (p - self.theta_star[n].to(dev)).pow(2)).sum()
        return self.lam * loss


# ── ADD (MMD2 on CLS embeddings) ──────────────────────────────────────────────
class SimpleDriftDetector:
    """ADD with bootstrap percentile calibration + max_no_update safeguard."""
    def __init__(self, max_no_update=4):
        self.ref_emb = None
        self.delta1 = None
        self.delta2 = None
        self.max_no_update = max_no_update
        self.consecutive_none = 0

    def _mmd2(self, X, Y):
        from scipy.spatial.distance import cdist
        all_pts = np.vstack([X, Y])
        dists = cdist(all_pts, all_pts)
        sigma = float(np.median(dists[np.triu_indices(len(all_pts), k=1)]))
        if sigma < 1e-10: sigma = 1.0
        def rbf(A, B):
            return np.exp(-cdist(A, B, "sqeuclidean") / (2*sigma**2))
        return float(rbf(X,X).mean() - 2*rbf(X,Y).mean() + rbf(Y,Y).mean())

    def calibrate(self, ref_embeddings, n_bootstrap=50):
        """Bootstrap percentile calibration — stable across dimensions."""
        rng = np.random.default_rng(42)
        n = len(ref_embeddings)
        mmd2_null = []
        for _ in range(n_bootstrap):
            perm = rng.permutation(n)
            half = n // 2
            m2 = self._mmd2(ref_embeddings[perm[:half]], ref_embeddings[perm[half:half*2]])
            mmd2_null.append(m2)
        mmd2_null = np.array(mmd2_null)
        self.delta2 = float(np.percentile(mmd2_null, 95))
        self.delta1 = float(np.percentile(mmd2_null, 99))
        if self.delta2 < 1e-8: self.delta2 = 1e-6
        if self.delta1 < self.delta2 * 1.5: self.delta1 = self.delta2 * 3.0
        self.ref_emb = ref_embeddings

    def detect(self, curr_emb):
        """Returns (drift_type, needs_update)."""
        if self.ref_emb is None: return "none", False
        m2 = self._mmd2(self.ref_emb, curr_emb)
        self.ref_emb = curr_emb
        if m2 >= self.delta1:
            drift = "sudden"
        elif m2 >= self.delta2:
            drift = "drift"
        else:
            drift = "none"
        # Safeguard
        forced = False
        if drift == "none":
            self.consecutive_none += 1
            if self.consecutive_none >= self.max_no_update:
                forced = True
                self.consecutive_none = 0
        else:
            self.consecutive_none = 0
        needs_update = (drift != "none") or forced
        return drift, needs_update


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg: dict, lora_r=8, lora_alpha=16, epochs=3, lr=2e-5, batch_size=64):
    from transformers import DistilBertTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("drc_cl_distilbert", log_dir=log_dir)
    split_dir  = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir    = Path(cfg["paths"]["results"])
    window_ids = get_window_ids(cfg)
    T = len(window_ids)
    seed = cfg["random_seed"]
    mu   = cfg.get("training", {}).get("mix_ratio", 0.3)
    lam  = cfg.get("ewc", {}).get("lambda", 0.4)
    buf_cap = cfg.get("ser", {}).get("capacity", 5000)

    logger.info("=" * 65)
    logger.info(" DRC-CL (DistilBERT + PEFT-LoRA)")
    logger.info("=" * 65)
    logger.info(f"  Device: {device}, LoRA r={lora_r}, alpha={lora_alpha}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    rng = np.random.default_rng(seed)

    # Init model
    model = DistilBERTWithLoRA(lora_r=lora_r, lora_alpha=lora_alpha).to(device)
    params = model.count_parameters()
    logger.info(f"  Params: total={params['total']:,}  trainable={params['trainable']:,} ({params['pct']}%)")

    # Init components
    ser = SERBuffer(capacity=buf_cap, seed=seed)
    ewc = EWCReg(lam=lam)
    add = SimpleDriftDetector()

    matrix = AccuracyMatrix(window_ids)
    t_total = time.time()

    for t in range(T):
        win_id = window_ids[t]
        tw = time.time()
        train_df = pd.read_csv(split_dir / f"{win_id}_train.csv")
        train_d = train_df["domain"].tolist()
        train_l = train_df["label"].tolist()

        # ── Train on D01 (first window) or update (subsequent) ────────────
        if t == 0:
            # Pretrain on D01
            logger.info("  Pretraining on D01_train ...")
            ds = TokenDataset(train_d, train_l, tokenizer)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
            optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()
            scaler = GradScaler("cuda") if device == "cuda" else None
            model.train()
            for _ in range(epochs):
                for ids, mask, y in loader:
                    ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                    optimizer.zero_grad()
                    if scaler:
                        with autocast("cuda"):
                            loss = criterion(model(ids, mask), y)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = criterion(model(ids, mask), y)
                        loss.backward()
                        optimizer.step()

            # Init ADD + EWC + SER
            embs = extract_cls_embeddings(model, train_d, tokenizer, device, batch_size)
            add.calibrate(embs)
            ds0 = TokenDataset(train_d[:1000], train_l[:1000], tokenizer)
            ewc.update(model, DataLoader(ds0, batch_size=batch_size, shuffle=True), device)
            n_add = min(5000, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            ser.add_batch([train_d[i] for i in idx], [train_l[i] for i in idx])

        else:
            # Detect drift
            embs = extract_cls_embeddings(model, train_d, tokenizer, device, batch_size)
            drift, should_update = add.detect(embs)

            if should_update:
                # Build D_mix = mu*Buffer + (1-mu)*D_t_train
                n_buf = int(len(train_d) * mu / max(1-mu, 0.01))
                b_d, b_l = ser.sample(n_buf)
                mix_d = train_d + b_d
                mix_l = train_l + b_l

                ds = TokenDataset(mix_d, mix_l, tokenizer)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
                optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()
                scaler = GradScaler("cuda") if device == "cuda" else None
                model.train()
                for _ in range(epochs):
                    for ids, mask, y in loader:
                        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                        optimizer.zero_grad()
                        if scaler:
                            with autocast("cuda"):
                                loss = criterion(model(ids, mask), y) + ewc.penalty(model)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss = criterion(model(ids, mask), y) + ewc.penalty(model)
                            loss.backward()
                            optimizer.step()

                # Update EWC Fisher
                if len(ser) > 0:
                    fb_d, fb_l = ser.sample(min(512, len(ser)))
                else:
                    fb_d, fb_l = train_d[:512], train_l[:512]
                ds_f = TokenDataset(fb_d, fb_l, tokenizer)
                ewc.update(model, DataLoader(ds_f, batch_size=batch_size, shuffle=True), device)

            # Update SER buffer
            n_add = min(5000, len(train_d))
            idx = rng.choice(len(train_d), n_add, replace=False)
            ser.add_batch([train_d[i] for i in idx], [train_l[i] for i in idx])

        # ── Evaluate on all test sets W1..Wt ──────────────────────────────
        row = build_row(model, tokenizer, split_dir, window_ids, t, device, batch_size)
        matrix.add_row(t, row)

        f1_t = row.get(win_id, {}).get("f1", 0)
        q_label = train_df["quarter_label"].iloc[0]
        elapsed = time.time() - tw
        drift_str = f"drift={drift}" + (" [forced]" if t > 0 and should_update and drift == "none" else "") if t > 0 else "init"
        logger.info(f"    {win_id} ({q_label}): F1={f1_t:.4f}  {drift_str}  ({elapsed:.1f}s)")

    total_time = time.time() - t_total
    metrics = matrix.compute_metrics()
    metrics["method"] = "DRC-CL (DistilBERT)"
    metrics["time_s"] = round(total_time, 1)
    per_type = matrix.compute_per_type_metrics()

    # Print results
    logger.info(f"\n{'='*65}")
    logger.info(" DRC-CL (DistilBERT) RESULTS")
    logger.info(f"{'='*65}")
    print_metrics_table([metrics], logger)
    print_per_type_table({"DRC-CL (DistilBERT)": per_type}, logger)
    logger.info(f"  Trainable params: {params['trainable']:,} ({params['pct']}%)")
    logger.info(f"  Total time: {total_time:.1f}s")

    # Save
    matrix.save(out_dir, prefix="drc_cl_distilbert")
    pd.DataFrame([metrics]).to_csv(out_dir / "drc_cl_distilbert_results.csv", index=False)
    logger.info(f"\n  Results → {out_dir / 'drc_cl_distilbert_results.csv'}")
    logger.info("  DRC-CL (DistilBERT) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRC-CL with DistilBERT + PEFT-LoRA")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--lora-r",     type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("ser", {"capacity": 5000})
    cfg.setdefault("ewc", {"lambda": 0.4})
    cfg.setdefault("training", {"mix_ratio": 0.3})
    run(cfg, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
