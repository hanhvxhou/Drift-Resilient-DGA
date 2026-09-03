"""
src/models/distilbert_baseline.py
──────────────────────────────────
DistilBERT baselines voi accuracy matrix protocol:
  1. DistilBERT (Static)     — train D01, freeze, evaluate 24 windows
  2. DistilBERT + Fine-tune  — update moi cua so, prequential

Protocol (chuan CL):
  - Train tren *_train.csv, evaluate tren *_test.csv
  - Full 24x24 accuracy matrix
  - BWT, FWT, Forgetting, F1-Char, F1-Word

Usage:
    pip install transformers
    python -m src.models.distilbert_baseline
    python -m src.models.distilbert_baseline --only static
    python -m src.models.distilbert_baseline --only finetune
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
from sklearn.metrics import f1_score
from scipy.special import expit as sigmoid_stable

from src.models.cl_metrics import AccuracyMatrix, build_accuracy_row, print_metrics_table, print_per_type_table
from src.utils.common import get_logger, get_window_ids, load_config


# ── Model ─────────────────────────────────────────────────────────────────────
class DistilBERTClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", dropout=0.3):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls).squeeze(1)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))


# ── Dataset ───────────────────────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, domains, labels, tokenizer, max_len=64):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.enc = tokenizer(domains, padding="max_length", truncation=True,
                             max_length=max_len, return_tensors="pt")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.enc["input_ids"][i], self.enc["attention_mask"][i], self.labels[i]


# ── Evaluate (compatible with AccuracyMatrix) ─────────────────────────────────
@torch.no_grad()
def eval_distilbert_on_test(model, test_df, tokenizer, device, batch_size=64):
    """Evaluate DistilBERT, return dict compatible with AccuracyMatrix."""
    from src.utils.dga_taxonomy import split_by_dga_type
    model.eval()
    domains = test_df["domain"].tolist()
    labels = np.array(test_df["label"].tolist())

    all_logits = []
    for i in range(0, len(domains), batch_size):
        batch_d = domains[i:i+batch_size]
        enc = tokenizer(batch_d, padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt")
        with autocast("cuda", enabled=(device == "cuda")):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_logits.append(logits.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    probs = sigmoid_stable(logits_np)
    preds = (probs >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score
    result = {
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if labels.sum() > 0 and (1-labels).sum() > 0 else 0.0,
    }

    df_char, df_word = split_by_dga_type(test_df)
    for sname, sdf in [("f1_char", df_char), ("f1_word", df_word)]:
        if len(sdf) < 10 or sdf["label"].nunique() < 2:
            result[sname] = float("nan"); continue
        sub_idx = test_df.index.isin(sdf.index)
        sub_p = probs[sub_idx[:len(probs)]]
        sub_l = np.array(sdf["label"].tolist())
        n = min(len(sub_p), len(sub_l))
        result[sname] = f1_score(sub_l[:n], (sub_p[:n] >= 0.5).astype(int), zero_division=0)

    return result


def build_distilbert_row(model, tokenizer, split_dir, window_ids, up_to_t, device, batch_size=64):
    """Build one row of accuracy matrix for DistilBERT."""
    row = {}
    for s in range(up_to_t + 1):
        test_path = split_dir / f"{window_ids[s]}_test.csv"
        if not test_path.exists(): continue
        test_df = pd.read_csv(test_path)
        row[window_ids[s]] = eval_distilbert_on_test(model, test_df, tokenizer, device, batch_size)
    return row


# ── Train ─────────────────────────────────────────────────────────────────────
def train_distilbert(model, domains, labels, tokenizer, device,
                     epochs=3, lr=2e-5, batch_size=64):
    ds = TokenDataset(domains, labels, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda") if device == "cuda" else None
    model.train()
    for _ in range(epochs):
        for input_ids, att_mask, y in loader:
            input_ids, att_mask, y = input_ids.to(device), att_mask.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    loss = criterion(model(input_ids, att_mask), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(input_ids, att_mask), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg: dict, only=None, epochs=3, lr=2e-5, batch_size=64):
    from transformers import DistilBertTokenizer

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir    = Path(cfg["paths"]["results"]) / "logs"
    logger     = get_logger("distilbert_baseline", log_dir=log_dir)
    bench_dir  = Path(cfg["paths"]["benchmark_dir"])
    split_dir  = bench_dir / "splits"
    out_dir    = Path(cfg["paths"]["results"])
    ckpt_dir   = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    window_ids = get_window_ids(cfg)
    T          = len(window_ids)

    logger.info("=" * 65)
    logger.info(" DistilBERT BASELINES (Accuracy Matrix Protocol)")
    logger.info("=" * 65)
    logger.info(f"  Device: {device}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    all_results = []
    all_per_type = {}

    # ══════════════════════════════════════════════════════════════════════
    # 1. DistilBERT (Static)
    # ══════════════════════════════════════════════════════════════════════
    if only is None or only == "static":
        logger.info(f"\n{'─'*65}")
        logger.info("  DistilBERT (Static) — train D01, no update")
        logger.info(f"{'─'*65}")

        t0 = time.time()
        model = DistilBERTClassifier().to(device)

        # Train on D01 train split
        df0_train = pd.read_csv(split_dir / f"{window_ids[0]}_train.csv")
        logger.info("  Training on D01_train ...")
        train_distilbert(model, df0_train["domain"].tolist(), df0_train["label"].tolist(),
                         tokenizer, device, epochs=epochs, lr=lr, batch_size=batch_size)
        model.save(ckpt_dir / "distilbert_d01.pt")

        # Build accuracy matrix (model never changes → all rows identical)
        matrix = AccuracyMatrix(window_ids)
        for t in range(T):
            row = build_distilbert_row(model, tokenizer, split_dir, window_ids, t, device, batch_size)
            matrix.add_row(t, row)
            f1_t = row.get(window_ids[t], {}).get("f1", 0)
            logger.info(f"    {window_ids[t]}: F1={f1_t:.4f}")

        metrics = matrix.compute_metrics()
        metrics["method"] = "DistilBERT (Static)"
        metrics["time_s"] = round(time.time() - t0, 1)
        per_type = matrix.compute_per_type_metrics()

        all_results.append(metrics)
        all_per_type["DistilBERT (Static)"] = per_type
        matrix.save(out_dir, prefix="distilbert_static")

        logger.info(f"\n  Static: AA-F1={metrics['aa_f1']:.4f}  BWT={metrics['bwt']:+.4f}  "
                    f"Forg={metrics['forgetting']:+.4f}  ({metrics['time_s']}s)")

    # ══════════════════════════════════════════════════════════════════════
    # 2. DistilBERT + Fine-tune
    # ══════════════════════════════════════════════════════════════════════
    if only is None or only == "finetune":
        logger.info(f"\n{'─'*65}")
        logger.info("  DistilBERT + Fine-tune — update every window")
        logger.info(f"{'─'*65}")

        t0 = time.time()
        model_ft = DistilBERTClassifier().to(device)

        # Load D01 checkpoint if available
        d01_ckpt = ckpt_dir / "distilbert_d01.pt"
        if d01_ckpt.exists():
            model_ft.load_weights(d01_ckpt, device)
            logger.info(f"  Loaded D01 checkpoint")
        else:
            df0_train = pd.read_csv(split_dir / f"{window_ids[0]}_train.csv")
            train_distilbert(model_ft, df0_train["domain"].tolist(), df0_train["label"].tolist(),
                             tokenizer, device, epochs=epochs, lr=lr, batch_size=batch_size)

        matrix = AccuracyMatrix(window_ids)

        for t in range(T):
            win_id = window_ids[t]
            tw = time.time()

            train_df = pd.read_csv(split_dir / f"{win_id}_train.csv")

            # Train on D_t (skip D01 — already trained)
            if t > 0:
                train_distilbert(model_ft, train_df["domain"].tolist(), train_df["label"].tolist(),
                                 tokenizer, device, epochs=epochs, lr=lr, batch_size=batch_size)

            # Evaluate on ALL test sets W1..Wt
            row = build_distilbert_row(model_ft, tokenizer, split_dir, window_ids, t, device, batch_size)
            matrix.add_row(t, row)

            f1_t = row.get(win_id, {}).get("f1", 0)
            q_label = train_df["quarter_label"].iloc[0]
            elapsed = time.time() - tw
            logger.info(f"    {win_id} ({q_label}): F1={f1_t:.4f}  ({elapsed:.1f}s)")

        metrics = matrix.compute_metrics()
        metrics["method"] = "DistilBERT + Fine-tune"
        metrics["time_s"] = round(time.time() - t0, 1)
        per_type = matrix.compute_per_type_metrics()

        all_results.append(metrics)
        all_per_type["DistilBERT + Fine-tune"] = per_type
        matrix.save(out_dir, prefix="distilbert_finetune")

        logger.info(f"\n  Fine-tune: AA-F1={metrics['aa_f1']:.4f}  BWT={metrics['bwt']:+.4f}  "
                    f"Forg={metrics['forgetting']:+.4f}  ({metrics['time_s']}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        logger.info(f"\n{'='*65}")
        logger.info(" DistilBERT RESULTS")
        logger.info(f"{'='*65}")
        print_metrics_table(all_results, logger)
        print_per_type_table(all_per_type, logger)

        pd.DataFrame(all_results).to_csv(out_dir / "distilbert_results.csv", index=False)
        logger.info(f"\n  Results → {out_dir / 'distilbert_results.csv'}")
        logger.info("  DistilBERT baselines complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT baselines (accuracy matrix protocol)")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--only",       default=None, choices=["static", "finetune"])
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int,   default=64)
    args = parser.parse_args()
    run(load_config(args.config), only=args.only, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size)
