"""
src/data/step7_generate_hashes.py
─────────────────────────────────
Generate reproducibility artifacts for the DRC-CL benchmark:

  1. SHA-256 hash of every benchmark / split CSV  ->  *.sha256 (one per file)
  2. A single manifest                            ->  benchmark_manifest.csv
     (filename, rows, DGA rows, benign rows, n_families, sha256)

Why this exists
---------------
DGArchive's license forbids redistributing the raw domains, and the Tranco
snapshots are large. Instead we publish content hashes, so anyone who obtains
the same raw sources and reruns run_pipeline can byte-verify that their rebuilt
benchmark matches the one used in the paper. The pipeline is deterministic
(fixed seed + split config in configs/), so the exact train/test splits are
reproducible without publishing any domain list.

Run AFTER the benchmark + splits exist:
    python -m src.data.step6_split_train_test
    python -m src.data.step7_generate_hashes

Verify a rebuilt benchmark later:
    python -m src.data.step7_generate_hashes --verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from src.utils.common import get_logger, load_config


CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 so large CSVs don't load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _summarise(csv_path: Path) -> dict:
    """Row / class / family counts for the manifest (label, family optional)."""
    df = pd.read_csv(csv_path)
    out = {"rows": len(df)}
    if "label" in df.columns:
        out["dga_rows"] = int((df["label"] == 1).sum())
        out["benign_rows"] = int((df["label"] == 0).sum())
    else:
        out["dga_rows"] = out["benign_rows"] = ""
    if "family" in df.columns:
        out["n_families"] = int(df.loc[df.get("label", 1) == 1, "family"].nunique())
    else:
        out["n_families"] = ""
    return out


def _iter_targets(bench_dir: Path, split_dir: Path):
    """All CSVs we hash: final windows D01..D24 + their train/test splits."""
    for p in sorted(bench_dir.glob("D*.csv")):
        yield p
    if split_dir.exists():
        for p in sorted(split_dir.glob("D*_*.csv")):
            yield p


def generate(cfg: dict, logger) -> None:
    bench_dir = Path(cfg["paths"]["benchmark_dir"])
    split_dir = bench_dir / "splits"
    if not bench_dir.exists():
        raise FileNotFoundError(
            f"{bench_dir} not found. Run the pipeline first: "
            f"python -m src.data.run_pipeline"
        )

    manifest_rows = []
    n = 0
    for csv_path in _iter_targets(bench_dir, split_dir):
        digest = sha256_of(csv_path)
        # write sidecar <file>.sha256 in the standard "  <hash>  <name>" form
        side = csv_path.with_suffix(csv_path.suffix + ".sha256")
        side.write_text(f"{digest}  {csv_path.name}\n", encoding="utf-8")
        row = {"file": str(csv_path.relative_to(bench_dir)), "sha256": digest}
        row.update(_summarise(csv_path))
        manifest_rows.append(row)
        n += 1
        logger.info(f"  hashed {csv_path.name}  {digest[:12]}…")

    # single manifest
    cols = ["file", "rows", "dga_rows", "benign_rows", "n_families", "sha256"]
    man = pd.DataFrame(manifest_rows)[cols]
    man_path = bench_dir / "benchmark_manifest.csv"
    man.to_csv(man_path, index=False)
    logger.info(f"\n  Wrote {man_path}  ({n} files hashed)")

    # NOTE: We intentionally do NOT emit a per-domain split index.
    # The pipeline is deterministic (fixed seed + split config in configs/),
    # so anyone can rebuild identical train/test splits by rerunning
    # run_pipeline and then verify them against the per-file SHA-256 hashes
    # in benchmark_manifest.csv. Publishing a 2.4M-row domain-level index
    # (even hashed) is unnecessary for reproducibility and needlessly bloats
    # the repository, so it is omitted.


def verify(cfg: dict, logger) -> int:
    """Recompute hashes and compare against benchmark_manifest.csv."""
    bench_dir = Path(cfg["paths"]["benchmark_dir"])
    man_path = bench_dir / "benchmark_manifest.csv"
    if not man_path.exists():
        logger.error(f"  No manifest at {man_path}. Run without --verify first.")
        return 2
    man = pd.read_csv(man_path)
    ok = miss = bad = 0
    for _, r in man.iterrows():
        p = bench_dir / r["file"]
        if not p.exists():
            logger.error(f"  MISSING  {r['file']}")
            miss += 1
            continue
        if sha256_of(p) == r["sha256"]:
            ok += 1
        else:
            logger.error(f"  MISMATCH {r['file']}")
            bad += 1
    logger.info(f"\n  verify: {ok} ok, {bad} mismatch, {miss} missing")
    return 0 if (bad == 0 and miss == 0) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against the saved manifest")
    args = ap.parse_args()
    cfg = load_config(args.config)
    logger = get_logger("gen_hashes", log_dir=Path(cfg["paths"]["results"]) / "logs")
    if args.verify:
        sys.exit(verify(cfg, logger))
    generate(cfg, logger)
    logger.info("  Reproducibility artifacts complete.")
