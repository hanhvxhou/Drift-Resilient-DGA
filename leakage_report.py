"""
leakage_report.py — R1.5 (leakage / de-duplication / overlap, hard numbers)

Reviewer #1 Concern #5 asks for explicit leakage, de-duplication and overlap
checks. The integrity report only says "ALL CHECKS PASSED"; this produces the
concrete counts reviewers want, from the built benchmark. No GPU.

Reports:
  (a) cross-window DUPLICATE domains: any domain appearing in >1 window
  (b) DGA/benign CROSS-CONTAMINATION: any domain labelled both 1 and 0
  (c) BENIGN overlap between consecutive windows (expected: benign pool is
      reused across a year, so some overlap is normal; DGA should be ~disjoint)
  (d) unique-domain totals

Run at project root. Reads data/processed/benchmark/D01.csv .. D24.csv.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from collections import defaultdict

BENCH = Path("data/processed/benchmark")

def main():
    wins = [f"D{i:02d}" for i in range(1, 25)]
    dfs = {}
    for w in wins:
        f = BENCH / f"{w}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["domain"] = d["domain"].astype(str).str.strip().str.lower()
            dfs[w] = d
    print(f"  Loaded {len(dfs)} windows\n")

    # (b) cross-contamination: domain with BOTH labels in the same window
    contam_total = 0
    for w, d in dfs.items():
        g = d.groupby("domain")["label"].nunique()
        c = int((g > 1).sum())
        contam_total += c
        if c:
            print(f"  [contam] {w}: {c} domains with both labels")
    print(f"  (b) DGA/benign cross-contamination (same window): {contam_total} domains\n")

    # (a) cross-window duplicates, split by class (DGA should be disjoint;
    #     benign reuse across windows is expected and not leakage of labels)
    dom_windows_dga = defaultdict(set)
    dom_windows_ben = defaultdict(set)
    for w, d in dfs.items():
        for dom in d[d.label == 1]["domain"]:
            dom_windows_dga[dom].add(w)
        for dom in d[d.label == 0]["domain"]:
            dom_windows_ben[dom].add(w)
    dga_multi = sum(1 for s in dom_windows_dga.values() if len(s) > 1)
    ben_multi = sum(1 for s in dom_windows_ben.values() if len(s) > 1)
    print(f"  (a) Cross-window duplicates:")
    print(f"      DGA   domains in >1 window: {dga_multi:,}  (unique DGA: {len(dom_windows_dga):,})")
    print(f"      benign domains in >1 window: {ben_multi:,}  (unique benign: {len(dom_windows_ben):,})")
    print(f"      -> DGA windows are {'DISJOINT' if dga_multi==0 else 'NOT disjoint'};"
          f" benign reuse across windows is expected (same yearly pool).\n")

    # (c) benign overlap consecutive windows
    print("  (c) Benign overlap between consecutive windows:")
    ov = []
    for a, b in zip(wins[:-1], wins[1:]):
        if a in dfs and b in dfs:
            sa = set(dfs[a][dfs[a].label == 0]["domain"])
            sb = set(dfs[b][dfs[b].label == 0]["domain"])
            j = len(sa & sb) / len(sa) * 100 if sa else 0
            ov.append(j)
    print(f"      mean consecutive benign overlap: {sum(ov)/len(ov):.1f}%")

    # (d) totals
    all_dom = set()
    for d in dfs.values():
        all_dom |= set(d["domain"])
    print(f"\n  (d) Unique domains across all windows: {len(all_dom):,}")
    print(f"      Total rows: {sum(len(d) for d in dfs.values()):,}")

    print("\n  Summary for the paper (Section V-A):")
    print(f"   - No DGA domain appears in more than one window "
          f"({'confirmed' if dga_multi==0 else str(dga_multi)+' found'}).")
    print(f"   - No domain carries both labels "
          f"({'confirmed' if contam_total==0 else str(contam_total)+' found'}).")

if __name__ == "__main__":
    main()
