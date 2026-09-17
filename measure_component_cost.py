"""
measure_component_cost.py — B5 / R1.4 (per-component cost, measured)

Reviewer #1 Concern #4 asks for measured runtime/memory of EACH component, not
just the total. This times the individual operations of one update step on a
real window, averaged over repeats, and reports GPU peak memory.

Components timed (all on the frozen-backbone + 12K-adapter setup):
  - ADD.detect   : one MMD² drift test on window embeddings (label-free)
  - SER.sample   : reservoir sampling of the replay batch
  - EWC.penalty  : Fisher-diagonal penalty term over the 12K adapter
  - extract_embeddings : embedding pass used by ADD/SER (reported separately)

Run at project root:
    python measure_component_cost.py --window D12 --repeats 20
Outputs results/component_cost.csv
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch

from src.utils.common import load_config, get_logger
from src.models.char_cnn import CharCNN
from src.models.lora_adapter import CharCNNWithLoRA
from src.models.drc_cl import SERBuffer, EWCRegularizer, DomainDataset
from src.detect.add_detector import ADDDetector, extract_embeddings
from torch.utils.data import DataLoader


def _time(fn, repeats, warmup=3):
    for _ in range(warmup): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0  # ms


def run(cfg, window="D12", repeats=20, device="cuda"):
    logger = get_logger("component_cost", log_dir=Path(cfg["paths"]["results"]) / "logs")
    bench = Path(cfg["paths"]["benchmark_dir"])
    df = pd.read_csv(bench / f"{window}.csv")
    domains = df["domain"].tolist(); labels = df["label"].tolist()
    logger.info(f"  Window {window}: {len(domains):,} samples, repeats={repeats}")

    # build a DRC-CL-style model (frozen backbone + LoRA adapter)
    backbone = CharCNN().to(device)
    ckpt = Path(cfg["paths"]["results"]) / "checkpoints" / "backbone_d01.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=device)
        backbone.load_state_dict(sd if not isinstance(sd, dict) or "state_dict" not in sd else sd["state_dict"], strict=False)
    model = CharCNNWithLoRA(backbone, rank=8, alpha=16.0).to(device)
    model.eval()

    rows = []
    def rec(name, ms, note=""):
        rows.append({"component": name, "time_ms_per_window": round(ms, 3), "note": note})
        logger.info(f"    {name:<22} {ms:8.3f} ms   {note}")

    # embeddings (shared prerequisite for ADD & SER)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    ms_emb = _time(lambda: extract_embeddings(model, domains, device=device, max_n=5000), repeats)
    rec("extract_embeddings", ms_emb, "shared by ADD & SER")
    embs = extract_embeddings(model, domains, device=device, max_n=5000)

    # ADD.detect (needs a calibrated detector)
    add = ADDDetector.from_config(cfg)
    add.calibrate(embs)
    ms_add = _time(lambda: add.detect(embs), repeats)
    rec("ADD.detect", ms_add, "MMD² drift test, label-free")

    # SER.sample
    ser = SERBuffer(capacity=5000)
    fams = df["family"].tolist() if "family" in df.columns else ["-"] * len(domains)
    idx = np.random.choice(len(domains), min(5000, len(domains)), replace=False)
    ser.add_batch([domains[i] for i in idx], [labels[i] for i in idx],
                  [fams[i] for i in idx], embs[:len(idx)])
    rng = np.random.default_rng(42)
    ms_ser = _time(lambda: ser.sample(1500, rng), repeats)
    rec("SER.sample", ms_ser, "reservoir sampling; buffer ≈ 80 KB")

    # EWC.penalty (needs a fisher; build once)
    ewc = EWCRegularizer(lam=0.4)
    fl = DataLoader(DomainDataset(domains[:1024], labels[:1024]), batch_size=512)
    ewc.update_fisher(model, fl, device)
    ms_ewc = _time(lambda: ewc.penalty(model), repeats)
    rec("EWC.penalty", ms_ewc, "Fisher-diagonal over 12K adapter params")

    gpu_mb = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else float("nan")
    logger.info(f"\n  GPU peak during component ops: {gpu_mb:.1f} MB")
    logger.info(f"  (Total per-update cost incl. LoRA training is in Table IX: 0.41 s/window, 92.9 MB)")

    out = Path(cfg["paths"]["results"]) / "component_cost.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"  Saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--window", default="D12")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(load_config(a.config), a.window, a.repeats, a.device)
