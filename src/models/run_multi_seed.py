"""
src/models/run_multi_seed.py
──────────────────────────────
Multi-seed experiment: chay tat ca methods voi nhieu seed,
bao cao mean +/- std, Wilcoxon signed-rank test.

Train/test split CO DINH (seed=42 tu step6).
Chi model training thay doi seed (weight init, shuffle, buffer sampling).

Protocol:
  For each seed:
    1. Re-train backbone on D01_train
    2. Run all methods (cl_experiment)
    3. Collect metrics
  Aggregate: mean +/- std, Wilcoxon p-values

Usage:
    # 10 seeds (recommended for p<0.01):
    python -m src.models.run_multi_seed --seeds 42 123 456 789 2024 3141 5926 5358 9793 2384

    # 5 seeds (faster):
    python -m src.models.run_multi_seed --seeds 42 123 456 789 2024

    # Resume after interruption (skip completed seeds):
    python -m src.models.run_multi_seed --seeds 42 123 456 789 2024 --resume
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, get_window_ids


DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 3141, 5926, 5358, 9793, 2384]


def train_backbone_with_seed(cfg: dict, seed: int, device: str, logger) -> Path:
    """Train CharCNN backbone on D01_train with given seed. Return checkpoint path."""
    import torch
    from src.models.char_cnn import CharCNN
    from src.models.train_backbone import run as train_run

    # Override seed in config
    cfg_copy = cfg.copy()
    cfg_copy["random_seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = Path(cfg["paths"]["results"]) / "multi_seed" / f"seed_{seed}" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "backbone_d01.pt"

    if ckpt_path.exists():
        logger.info(f"    Backbone seed={seed} already exists, skipping training")
        return ckpt_path

    # Train backbone
    logger.info(f"    Training backbone with seed={seed} ...")
    # Use the split directory for D01 train data
    split_dir = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    train_df  = pd.read_csv(split_dir / "D01_train.csv")

    from src.models.char_cnn import CharCNN, domains_to_batch
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    from torch.amp import GradScaler, autocast
    from sklearn.model_selection import train_test_split

    class DS(Dataset):
        def __init__(self, domains, labels):
            from src.models.char_cnn import domain_to_tensor
            self.x = [domain_to_tensor(d) for d in domains]
            self.y = torch.tensor(labels, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.x[i], self.y[i]

    # Split D01_train into train/val for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_df["domain"].tolist(), train_df["label"].tolist(),
        test_size=0.2, random_state=seed, stratify=train_df["label"]
    )

    model = CharCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda") if device == "cuda" else None

    train_dl = DataLoader(DS(X_tr, y_tr), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(DS(X_val, y_val), batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    best_val_loss = float("inf")
    patience_cnt  = 0

    for epoch in range(1, 31):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(y)
        val_loss /= len(val_dl.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            model.save(ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= 5:
                break

    logger.info(f"    Backbone trained: {epoch} epochs, val_loss={best_val_loss:.4f}")
    return ckpt_path


def run_one_seed(cfg: dict, seed: int, device: str, logger,
                 methods_to_run: list[str] | None = None) -> dict:
    """Run all methods for one seed. Return {method: metrics_dict}."""
    from src.models.cl_experiment import run_method
    import torch

    cfg_copy = cfg.copy()
    cfg_copy["random_seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    split_dir  = Path(cfg["paths"]["benchmark_dir"]) / "splits"
    window_ids = get_window_ids(cfg)
    seed_dir   = Path(cfg["paths"]["results"]) / "multi_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Result file for this seed
    result_file = seed_dir / "seed_results.json"
    if result_file.exists():
        existing = json.loads(result_file.read_text())
    else:
        existing = {}

    # Train backbone
    backbone_path = train_backbone_with_seed(cfg_copy, seed, device, logger)

    common = dict(cfg=cfg_copy, backbone_path=backbone_path, device=device,
                  logger=logger, split_dir=split_dir, window_ids=window_ids)

    EXPERIMENTS = [
        ("Static-CNN",             dict(is_static=True, use_lora=False)),
        ("SW-Retrain",             dict(is_sw_retrain=True, use_lora=False)),
        ("EWC-only",               dict(is_ewc_only_fulltune=True, use_lora=False)),
        ("iCaRL",                  dict(is_icarl=True, use_lora=False)),
        ("GDumb",                  dict(is_gdumb=True, use_lora=False)),
        ("CNN + LoRA Update",      dict(use_lora=True, update_every=True)),
        ("CNN + LoRA + SER",       dict(use_lora=True, update_every=True, use_ser=True)),
        ("CNN + LoRA + SER + EWC", dict(use_lora=True, update_every=True, use_ser=True, use_ewc=True)),
        ("DRC-CL",                 dict(use_lora=True, use_ser=True, use_ewc=True, use_add=True)),
    ]

    seed_results = {}
    for name, kwargs in EXPERIMENTS:
        if methods_to_run and name not in methods_to_run:
            continue
        if name in existing:
            logger.info(f"      {name}: loaded from cache")
            seed_results[name] = existing[name]
            continue

        logger.info(f"      Running {name} ...")
        t0 = time.time()
        result = run_method(method_name=name, **common, **kwargs)
        elapsed = time.time() - t0
        metrics = result["metrics"]
        metrics["time_s"] = round(elapsed, 1)
        seed_results[name] = metrics
        logger.info(f"      {name}: AA-F1={metrics['aa_f1']:.4f}  BWT={metrics['bwt']:+.4f}  "
                    f"Forg={metrics['forgetting']:+.4f}  ({elapsed:.1f}s)")

        # Save incrementally (resume support)
        existing[name] = metrics
        result_file.write_text(json.dumps(existing, indent=2))

    return seed_results


def aggregate_results(all_seeds: dict[int, dict]) -> pd.DataFrame:
    """Aggregate multi-seed results into mean +/- std table."""
    rows = []
    methods = list(next(iter(all_seeds.values())).keys())
    metrics_keys = ["aa_f1", "bwt", "forgetting", "degrad", "f1_old", "aa_f1_char", "aa_f1_word"]

    for method in methods:
        values = {k: [] for k in metrics_keys}
        for seed, seed_results in all_seeds.items():
            if method not in seed_results:
                continue
            m = seed_results[method]
            for k in metrics_keys:
                v = m.get(k)
                if v is not None and v == v:  # not NaN
                    values[k].append(v)

        row = {"method": method}
        for k in metrics_keys:
            vals = values[k]
            if vals:
                mean = np.mean(vals)
                sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[f"{k}_mean"] = round(mean, 4)
                row[f"{k}_std"] = round(sd, 4)
                row[f"{k}_str"] = f"{mean:.4f} +/- {sd:.4f}"
            else:
                row[f"{k}_mean"] = None
                row[f"{k}_std"]  = None
                row[f"{k}_str"]  = "N/A"
        rows.append(row)

    return pd.DataFrame(rows)


def wilcoxon_tests(all_seeds: dict[int, dict], reference: str = "DRC-CL") -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests vs reference method."""
    from scipy.stats import wilcoxon

    seeds = sorted(all_seeds.keys())
    methods = list(next(iter(all_seeds.values())).keys())

    ref_f1s = [all_seeds[s][reference]["aa_f1"] for s in seeds if reference in all_seeds[s]]

    rows = []
    for method in methods:
        if method == reference:
            continue
        method_f1s = [all_seeds[s][method]["aa_f1"] for s in seeds if method in all_seeds[s]]

        n = min(len(ref_f1s), len(method_f1s))
        if n < 5:
            rows.append({"method": method, "vs": reference,
                         "p_value": None, "significant": "N/A (n<5)"})
            continue

        try:
            stat, p = wilcoxon(ref_f1s[:n], method_f1s[:n])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            rows.append({"method": method, "vs": reference,
                         "p_value": round(p, 6), "stat": round(stat, 4),
                         "significant": sig,
                         "ref_mean": round(np.mean(ref_f1s[:n]), 4),
                         "method_mean": round(np.mean(method_f1s[:n]), 4)})
        except Exception as e:
            rows.append({"method": method, "vs": reference,
                         "p_value": None, "significant": f"error: {e}"})

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg: dict, seeds: list[int], resume: bool = False):
    import torch
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("multi_seed", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["results"]) / "multi_seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure defaults
    cfg.setdefault("lora",     {"rank": 8, "alpha": 16.0})
    cfg.setdefault("ser",      {"capacity": 5000, "beta": 0.92, "min_k": 50})
    cfg.setdefault("ewc",      {"lambda": 0.4})
    cfg.setdefault("training", {"lr": 5e-4, "update_epochs": 5, "batch_size": 512, "mix_ratio": 0.3})

    logger.info("=" * 65)
    logger.info(f" MULTI-SEED EXPERIMENT — {len(seeds)} seeds")
    logger.info("=" * 65)
    logger.info(f"  Seeds  : {seeds}")
    logger.info(f"  Device : {device}")
    if device == "cuda":
        logger.info(f"  GPU    : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Resume : {resume}")
    logger.info("")

    all_seeds = {}
    t_total = time.time()

    for i, seed in enumerate(seeds):
        logger.info(f"\n{'='*65}")
        logger.info(f"  SEED {seed} ({i+1}/{len(seeds)})")
        logger.info(f"{'='*65}")

        t_seed = time.time()
        seed_results = run_one_seed(cfg, seed, device, logger)
        all_seeds[seed] = seed_results
        elapsed = time.time() - t_seed

        logger.info(f"\n  Seed {seed} done in {elapsed:.1f}s")

        # Save progress after each seed
        progress = {str(s): r for s, r in all_seeds.items()}
        (out_dir / "all_seeds_raw.json").write_text(json.dumps(progress, indent=2))

    total_time = time.time() - t_total

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info(" AGGREGATED RESULTS (mean +/- std)")
    logger.info(f"{'='*65}")

    agg_df = aggregate_results(all_seeds)

    # Print table
    sep = "-" * 95
    logger.info(sep)
    logger.info(f"{'Method':<28} {'AA-F1':>16} {'BWT':>16} {'Forg.':>16} {'Degrad.':>16}")
    logger.info(sep)
    for _, r in agg_df.iterrows():
        logger.info(f"{r['method']:<28} {r['aa_f1_str']:>16} {r['bwt_str']:>16} "
                    f"{r['forgetting_str']:>16} {r['degrad_str']:>16}")
    logger.info(sep)

    # ── Wilcoxon tests ────────────────────────────────────────────────────────
    if len(seeds) >= 5:
        logger.info(f"\n  Wilcoxon signed-rank tests (vs DRC-CL):")
        wilcox_df = wilcoxon_tests(all_seeds, reference="DRC-CL")
        for _, r in wilcox_df.iterrows():
            logger.info(f"    DRC-CL vs {r['method']:<24s} p={r['p_value']:<10} {r['significant']}")

        wilcox_path = out_dir / "wilcoxon_tests.csv"
        wilcox_df.to_csv(wilcox_path, index=False)
        logger.info(f"\n  Wilcoxon tests → {wilcox_path}")

    # ── Save ──────────────────────────────────────────────────────────────────
    agg_path = out_dir / "aggregated_results.csv"
    agg_df.to_csv(agg_path, index=False)
    logger.info(f"  Aggregated    → {agg_path}")
    logger.info(f"\n  Total time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
    logger.info("  Multi-seed experiment complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed CL experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 10 seeds (p<0.01 possible):
  python -m src.models.run_multi_seed

  # 5 seeds (faster):
  python -m src.models.run_multi_seed --seeds 42 123 456 789 2024

  # Resume after interruption:
  python -m src.models.run_multi_seed --resume
        """
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds",  nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--resume", action="store_true",
                        help="Skip seeds/methods already completed")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, seeds=args.seeds, resume=args.resume)
