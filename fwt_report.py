"""Aggregate FWT across all seeds. Run: python fwt_report.py"""
import json, numpy as np
d = json.load(open("results/multi_seed/all_seeds_raw.json"))
seeds = list(d.keys())
methods = ["Static-CNN","SW-Retrain","EWC-only","iCaRL","GDumb",
           "CNN + LoRA Update","CNN + LoRA + SER","CNN + LoRA + SER + EWC","DRC-CL"]
print(f"FWT — Forward Transfer (mean +/- std, ddof=1, {len(seeds)} seeds)")
print("-" * 58)
for m in methods:
    vals = [d[s][m]["fwt"] for s in seeds if m in d[s] and "fwt" in d[s][m]]
    if vals:
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        print(f"  {m:<24} {mean:+.4f} +/- {sd:.4f}  (n={len(vals)})")
