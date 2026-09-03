"""
src/models/measure_cost.py
───────────────────────────
Do chi phi tinh toan cho Table IX:
  - Training time per window
  - Inference time per 20K domains
  - GPU memory peak
  - Parameters updated
  - FLOPs estimation

Usage:
    python -m src.models.measure_cost
"""

from __future__ import annotations
import argparse, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast

from src.utils.common import get_logger, load_config, get_window_ids


def measure_inference(model, domains, device, tokenizer=None,
                      is_transformer=False, batch_size=512, n_repeats=3):
    """Measure inference time on domains, average over n_repeats."""
    from src.models.char_cnn import domains_to_batch
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            for i in range(0, len(domains), batch_size):
                batch_d = domains[i:i+batch_size]
                if is_transformer and tokenizer:
                    enc = tokenizer(batch_d, padding="max_length", truncation=True,
                                    max_length=64, return_tensors="pt")
                    with autocast("cuda", enabled=(device=="cuda")):
                        model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                else:
                    x = domains_to_batch(batch_d).to(device)
                    model(x)
            torch.cuda.synchronize() if device == "cuda" else None
            times.append(time.perf_counter() - t0)
    return {"mean_s": round(np.mean(times), 3), "std_s": round(np.std(times), 3)}


def measure_gpu_memory(model, domains, device, tokenizer=None,
                       is_transformer=False, batch_size=512):
    """Measure peak GPU memory during forward pass."""
    if device != "cuda":
        return {"peak_mb": 0}
    from src.models.char_cnn import domains_to_batch
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        for i in range(0, min(len(domains), batch_size*4), batch_size):
            batch_d = domains[i:i+batch_size]
            if is_transformer and tokenizer:
                enc = tokenizer(batch_d, padding="max_length", truncation=True,
                                max_length=64, return_tensors="pt")
                with autocast("cuda"):
                    model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            else:
                x = domains_to_batch(batch_d).to(device)
                model(x)
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    return {"peak_mb": round(peak, 1)}


def measure_training_time(model, domains, labels, device, epochs=5,
                          lr=5e-4, batch_size=512, lora_only=False):
    """Measure time for one training episode."""
    from src.models.char_cnn import domain_to_tensor
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __init__(self, doms, labs):
            self.x = [domain_to_tensor(d) for d in doms]
            self.y = torch.tensor(labs, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.x[i], self.y[i]

    ds = DS(domains[:10000], labels[:10000])  # measure on 10K for speed
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    params = model.lora_parameters() if (lora_only and hasattr(model, 'lora_parameters')) else \
             [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    from torch.amp import GradScaler
    scaler = GradScaler("cuda") if device == "cuda" else None

    model.train()
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0
    return {"train_time_s": round(elapsed, 2), "epochs": epochs, "n_samples": len(ds)}


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger = get_logger("measure_cost", log_dir=log_dir)
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    out_dir = Path(cfg["paths"]["results"])
    backbone_path = out_dir / "checkpoints" / "backbone_d01.pt"

    logger.info("=" * 60)
    logger.info(" COMPUTATIONAL COST MEASUREMENT")
    logger.info("=" * 60)
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # Load test data
    test_df = pd.read_csv(split_dir / "D12_test.csv")  # use middle window
    domains = test_df["domain"].tolist()
    labels = test_df["label"].tolist()

    results = []

    # ── 1. Static-CNN ─────────────────────────────────────────────────────
    logger.info("\n  Measuring: Static-CNN")
    from src.models.char_cnn import CharCNN
    model = CharCNN.load(backbone_path, map_location=device).to(device)
    params_total = sum(p.numel() for p in model.parameters())
    inf = measure_inference(model, domains, device)
    mem = measure_gpu_memory(model, domains, device)
    results.append({"method": "Static-CNN", "params_total": params_total,
                    "params_updated": 0, "pct_updated": "—",
                    "inference_s": inf["mean_s"], "train_time_s": "—",
                    "gpu_peak_mb": mem["peak_mb"]})
    logger.info(f"    Inference: {inf['mean_s']}s / {len(domains)} domains")
    logger.info(f"    GPU peak: {mem['peak_mb']} MB")

    # ── 2. EWC-only (full fine-tune) ──────────────────────────────────────
    logger.info("\n  Measuring: EWC-only")
    model_ewc = CharCNN.load(backbone_path, map_location=device).to(device)
    trainable = sum(p.numel() for p in model_ewc.parameters() if p.requires_grad)
    tr = measure_training_time(model_ewc, domains, labels, device, epochs=5)
    inf = measure_inference(model_ewc, domains, device)
    mem = measure_gpu_memory(model_ewc, domains, device)
    results.append({"method": "EWC-only", "params_total": params_total,
                    "params_updated": trainable, "pct_updated": "100%",
                    "inference_s": inf["mean_s"], "train_time_s": tr["train_time_s"],
                    "gpu_peak_mb": mem["peak_mb"]})
    logger.info(f"    Train: {tr['train_time_s']}s ({tr['epochs']} ep, {tr['n_samples']} samples)")
    logger.info(f"    Inference: {inf['mean_s']}s")
    del model_ewc; torch.cuda.empty_cache()

    # ── 3. DRC-CL (CharCNN + LoRA) ───────────────────────────────────────
    logger.info("\n  Measuring: DRC-CL (CharCNN)")
    from src.models.lora_adapter import CharCNNWithLoRA
    model_lora = CharCNNWithLoRA.from_checkpoint(
        backbone_path, rank=8, alpha=16, map_location=device).to(device)
    p_info = model_lora.count_parameters()
    tr = measure_training_time(model_lora, domains, labels, device, epochs=5, lora_only=True)
    inf = measure_inference(model_lora, domains, device)
    mem = measure_gpu_memory(model_lora, domains, device)
    results.append({"method": "DRC-CL (CharCNN)", "params_total": p_info["total"],
                    "params_updated": p_info["trainable"], "pct_updated": f"{p_info['trainable']/p_info['total']*100:.1f}%",
                    "inference_s": inf["mean_s"], "train_time_s": tr["train_time_s"],
                    "gpu_peak_mb": mem["peak_mb"]})
    logger.info(f"    Train: {tr['train_time_s']}s")
    logger.info(f"    Params: {p_info['trainable']:,} / {p_info['total']:,} ({p_info['trainable']/p_info['total']*100:.1f}%)")
    del model_lora; torch.cuda.empty_cache()

    # ── 4. DRC-CL (DistilBERT + LoRA) ────────────────────────────────────
    try:
        logger.info("\n  Measuring: DRC-CL (DistilBERT)")
        from src.models.drc_cl_distilbert import DistilBERTWithLoRA
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model_db = DistilBERTWithLoRA(lora_r=8, lora_alpha=16).to(device)
        p_db = model_db.count_parameters()
        inf = measure_inference(model_db, domains[:2000], device, tokenizer, True, batch_size=64)
        # Scale inference to 20K
        inf_20k = round(inf["mean_s"] * len(domains) / 2000, 2)
        mem = measure_gpu_memory(model_db, domains[:1000], device, tokenizer, True, batch_size=64)
        # Actual training time measurement (3 epochs, 10K samples, LoRA only)
        from torch.utils.data import DataLoader, Dataset
        class TokDS2(Dataset):
            def __init__(self, doms, labs, tok):
                self.labels = torch.tensor(labs, dtype=torch.float32)
                self.enc = tok(doms, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            def __len__(self): return len(self.labels)
            def __getitem__(self, i): return self.enc["input_ids"][i], self.enc["attention_mask"][i], self.labels[i]

        ds_db = TokDS2(domains[:10000], labels[:10000], tokenizer)
        loader_db = DataLoader(ds_db, batch_size=64, shuffle=True, num_workers=0)
        opt_db = torch.optim.AdamW(model_db.trainable_parameters(), lr=2e-5)
        crit_db = nn.BCEWithLogitsLoss()
        from torch.amp import GradScaler
        scaler_db = GradScaler("cuda") if device == "cuda" else None
        model_db.train()
        torch.cuda.synchronize() if device == "cuda" else None
        t0_db = time.perf_counter()
        for _ in range(3):
            for ids, mask, y in loader_db:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                opt_db.zero_grad()
                if scaler_db:
                    with autocast("cuda"):
                        loss = crit_db(model_db(ids, mask), y)
                    scaler_db.scale(loss).backward()
                    scaler_db.step(opt_db)
                    scaler_db.update()
                else:
                    loss = crit_db(model_db(ids, mask), y)
                    loss.backward()
                    opt_db.step()
        torch.cuda.synchronize() if device == "cuda" else None
        train_db = round(time.perf_counter() - t0_db, 2)

        results.append({"method": "DRC-CL (DistilBERT)", "params_total": p_db["total"],
                        "params_updated": p_db["trainable"], "pct_updated": f"{p_db['pct']}%",
                        "inference_s": inf_20k, "train_time_s": train_db,
                        "gpu_peak_mb": mem["peak_mb"]})
        logger.info(f"    Train (3ep, 10K LoRA): {train_db}s")
        logger.info(f"    Inference (scaled to 20K): {inf_20k}s")
        logger.info(f"    Params: {p_db['trainable']:,} / {p_db['total']:,} ({p_db['pct']}%)")
        del model_db; torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"    DistilBERT measurement failed: {e}")

    # ── 5. DistilBERT + FT ───────────────────────────────────────────────
    try:
        logger.info("\n  Measuring: DistilBERT + FT")
        # Robust import: try both class names
        try:
            from src.models.distilbert_baseline import DistilBERTClassifier as DBClass
        except ImportError:
            from src.models.distilbert_baseline import DistilBERTDomainClassifier as DBClass

        model_ft = DBClass().to(device)

        # Robust count_parameters
        if hasattr(model_ft, 'count_parameters'):
            p_ft = model_ft.count_parameters()
            p_total = p_ft.get("total", sum(p.numel() for p in model_ft.parameters()))
            p_train = p_ft.get("trainable", sum(p.numel() for p in model_ft.parameters() if p.requires_grad))
        else:
            p_total = sum(p.numel() for p in model_ft.parameters())
            p_train = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)

        # Inference measurement
        inf = measure_inference(model_ft, domains[:2000], device, tokenizer, True, batch_size=64)
        inf_20k = round(inf["mean_s"] * len(domains) / 2000, 2)

        # GPU memory
        mem = measure_gpu_memory(model_ft, domains[:1000], device, tokenizer, True, batch_size=64)

        # Training time: actual measurement on 10K domains
        from src.models.char_cnn import domain_to_tensor
        from torch.utils.data import DataLoader, Dataset
        class TokDS(Dataset):
            def __init__(self, doms, labs, tok):
                self.labels = torch.tensor(labs, dtype=torch.float32)
                self.enc = tok(doms, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            def __len__(self): return len(self.labels)
            def __getitem__(self, i): return self.enc["input_ids"][i], self.enc["attention_mask"][i], self.labels[i]

        ds_ft = TokDS(domains[:10000], labels[:10000], tokenizer)
        loader_ft = DataLoader(ds_ft, batch_size=64, shuffle=True, num_workers=0)
        optimizer_ft = torch.optim.AdamW(model_ft.parameters(), lr=2e-5)
        criterion_ft = nn.BCEWithLogitsLoss()
        from torch.amp import GradScaler
        scaler_ft = GradScaler("cuda") if device == "cuda" else None

        model_ft.train()
        torch.cuda.synchronize() if device == "cuda" else None
        t0_ft = time.perf_counter()
        for _ in range(3):  # 3 epochs like distilbert_baseline
            for ids, mask, y in loader_ft:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                optimizer_ft.zero_grad()
                if scaler_ft:
                    with autocast("cuda"):
                        loss = criterion_ft(model_ft(ids, mask), y)
                    scaler_ft.scale(loss).backward()
                    scaler_ft.step(optimizer_ft)
                    scaler_ft.update()
                else:
                    loss = criterion_ft(model_ft(ids, mask), y)
                    loss.backward()
                    optimizer_ft.step()
        torch.cuda.synchronize() if device == "cuda" else None
        train_time_ft = round(time.perf_counter() - t0_ft, 2)

        results.append({"method": "DistilBERT + FT", "params_total": p_total,
                        "params_updated": p_train, "pct_updated": "100%",
                        "inference_s": inf_20k, "train_time_s": train_time_ft,
                        "gpu_peak_mb": mem["peak_mb"]})
        logger.info(f"    Train (3ep, 10K): {train_time_ft}s")
        logger.info(f"    Inference (scaled to 20K): {inf_20k}s")
        logger.info(f"    Params: {p_train:,} / {p_total:,}")
        del model_ft; torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        logger.warning(f"    DistilBERT+FT measurement failed: {e}")
        traceback.print_exc()

    # ── Print Table IX ────────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info(" TABLE IX: Computational Cost")
    logger.info(f"{'='*80}")
    logger.info(f"{'Method':<24} {'Params Total':>12} {'Updated':>10} {'%':>6} {'Train/win':>10} {'Infer/20K':>10} {'GPU MB':>8}")
    logger.info("-" * 80)
    for r in results:
        logger.info(f"{r['method']:<24} {r['params_total']:>12,} {r['params_updated']:>10} {r['pct_updated']:>6} {str(r['train_time_s']):>10}s {r['inference_s']:>10}s {r['gpu_peak_mb']:>8}")
    logger.info("-" * 80)

    # Save
    pd.DataFrame(results).to_csv(out_dir / "computational_cost.csv", index=False)
    logger.info(f"\n  Saved: {out_dir / 'computational_cost.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config))
