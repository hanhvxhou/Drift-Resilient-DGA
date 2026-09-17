"""
check_benign_source.py — Xác định D01-D04 dùng benign 2018 thật hay proxy 2019.

So các benign domain (label==0) trong D01..D04 đã build với tranco_2018.csv và
tranco_2019.csv. Nguồn nào phủ (overlap) benign của benchmark cao hơn → đó là
nguồn thực sự đã dùng khi build.

Chạy ở gốc project. Đọc:
  data/processed/benchmark/D01.csv .. D04.csv   (cột: domain,label)
  data/raw/benign/tranco_2018.csv, tranco_2019.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

BENCH = Path("data/processed/benchmark")
BENIGN = Path("data/raw/benign")

def load_set(p, col_guess="domain"):
    df = pd.read_csv(p)
    c = col_guess if col_guess in df.columns else df.columns[0]
    return set(df[c].astype(str).str.strip().str.lower())

def main():
    t2018 = load_set(BENIGN / "tranco_2018.csv")
    t2019 = load_set(BENIGN / "tranco_2019.csv")
    print(f"  tranco_2018: {len(t2018):,} domains")
    print(f"  tranco_2019: {len(t2019):,} domains\n")

    for win in ["D01", "D02", "D03", "D04"]:
        f = BENCH / f"{win}.csv"
        if not f.exists():
            print(f"  {win}.csv not found"); continue
        df = pd.read_csv(f)
        benign = set(df[df["label"] == 0]["domain"].astype(str).str.strip().str.lower())
        n = len(benign)
        in18 = len(benign & t2018) / n * 100 if n else 0
        in19 = len(benign & t2019) / n * 100 if n else 0
        src = "2018 (real)" if in18 > in19 + 5 else ("2019 (proxy)" if in19 > in18 + 5 else "ambiguous")
        print(f"  {win}: {n:,} benign  |  in tranco_2018: {in18:5.1f}%  in tranco_2019: {in19:5.1f}%  -> {src}")

    print("\n  Rule: the source with clearly higher coverage (>5pp) is the one used.")
    print("  If both ~equal, the two snapshots overlap too much to tell (then it")
    print("  does not matter — the benign distribution is effectively the same).")

if __name__ == "__main__":
    main()
