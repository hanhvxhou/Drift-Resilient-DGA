"""
measure_tranco_overlap.py  —  R1.5 (proxy 2018 quantification)

Quantifies how stable the benign (Tranco) domain distribution is year-over-year.
If consecutive-year Tranco snapshots overlap heavily, then using the earliest
available snapshot (March 2019) as a proxy for 2018 benign traffic changes the
benign distribution only marginally — directly quantifying the proxy's impact.

Place this at the project root (where data/raw/benign/tranco_YYYY.csv live).

Usage
-----
    python measure_tranco_overlap.py
    python measure_tranco_overlap.py --dir data/raw/benign --topk 50000

Reads tranco_2019.csv .. tranco_2023.csv (single column: domain).
Prints and saves results/tranco_overlap.csv:
    year_a, year_b, jaccard, overlap_of_a_in_b (%), n_a, n_b
"""
from __future__ import annotations
import argparse, glob, os, re
from pathlib import Path
import pandas as pd


def load_domains(path, topk=None):
    df = pd.read_csv(path)
    col = "domain" if "domain" in df.columns else df.columns[0]
    doms = df[col].astype(str).str.strip().str.lower()
    if topk:
        doms = doms.head(topk)
    return set(doms)


def run(dirpath="data/raw/benign", topk=50000, out="results/tranco_overlap.csv"):
    files = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "tranco_*.csv"))):
        m = re.search(r"tranco_(\d{4})", os.path.basename(p))
        if m:
            files[int(m.group(1))] = p
    years = sorted(files)
    if len(years) < 2:
        print(f"  Need >=2 tranco_YYYY.csv in {dirpath}; found: {years}")
        return

    print(f"  Loading {len(years)} snapshots (top {topk}): {years}")
    sets = {y: load_domains(files[y], topk) for y in years}

    rows = []
    # consecutive-year overlaps (the relevant quantity for a 1-year proxy)
    for a, b in zip(years[:-1], years[1:]):
        inter = len(sets[a] & sets[b])
        union = len(sets[a] | sets[b])
        jacc = inter / union if union else 0
        ov_a = inter / len(sets[a]) * 100 if sets[a] else 0
        rows.append({"year_a": a, "year_b": b,
                     "jaccard": round(jacc, 4),
                     "overlap_a_in_b_pct": round(ov_a, 2),
                     "n_a": len(sets[a]), "n_b": len(sets[b])})
        print(f"    {a} vs {b}: Jaccard={jacc:.3f}, {ov_a:.1f}% of {a} still in {b}")

    # earliest vs each later year (proxy is earliest→2018)
    e = years[0]
    for b in years[1:]:
        inter = len(sets[e] & sets[b]); ov = inter / len(sets[e]) * 100
        print(f"    {e} vs {b}: {ov:.1f}% of {e} still present in {b}")

    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    avg = sum(r["overlap_a_in_b_pct"] for r in rows) / len(rows)
    print(f"\n  Mean consecutive-year overlap: {avg:.1f}%")
    print(f"  -> Higher overlap = benign distribution more stable = proxy impact smaller.")
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/benign")
    ap.add_argument("--topk", type=int, default=50000,
                    help="use top-K domains (match max_per_window in config)")
    ap.add_argument("--out", default="results/tranco_overlap.csv")
    args = ap.parse_args()
    run(args.dir, args.topk, args.out)
