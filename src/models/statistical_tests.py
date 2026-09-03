"""
src/models/statistical_tests.py
────────────────────────────────
Friedman test + Nemenyi post-hoc + Critical Difference Diagram.

So sanh tat ca methods tren cac seed THAT (khong bao gio sinh du lieu gia).

Usage:
    python -m src.models.statistical_tests
    python -m src.models.statistical_tests --no-figures
    python -m src.models.statistical_tests --keep-static-in-forgetting

────────────────────────────────────────────────────────────────────────────
LICH SU SUA LOI  (2026-07-15) — ban cu da sinh ra 2 hinh CD SAI HOAN TOAN.
Neu tai chay ban cu, loi se lap lai. Cac loi da sua:

 [1] SAI TEN FILE — loi goc keo theo tat ca.
     Cu : results_file = seed_dir / "final_results.csv"     <- KHONG TON TAI
     run_multi_seed.py ghi ra  seed_dir / "seed_results.json"  va
                               multi_seed/all_seeds_raw.json
     ("final_results.csv" co that, nhung nam o results/ chu khong o seed_*/)
     Hau qua: per_seed_data chi nhan duoc 2 method (DERPP, AGEM tu
     extra_baselines_results.csv) -> len(valid_methods)=2 < 3 -> kich hoat [2].

 [2] SINH DU LIEU GIA — nghiem trong nhat.
     Cu : if len(valid_methods) < 3:
              rng = np.random.default_rng(42)
              valid_methods[m] = [{'seed': i,
                                   'aa_f1': rng.normal(mean, std)} ...]
     -> 9/11 method trong ca hai hinh la SO BIA.
     -> rng re-seed(42) TRONG vong lap => moi method rut cung chuoi nhieu
        => hang trung binh ra so nguyen, p-value dep gia tao.
     Nay : RAISE RuntimeError. Tha de chuong trinh chet con hon bia so.

 [3] FALLBACK NGAM forgetting -> aa_f1.
     Cu : vals = [d.get(metric, d.get('aa_f1', 0)) for d in ...]
     Dict synthetic thieu key 'forgetting' -> tra ve aa_f1 (~0.96) thay vi
     forgetting (~0.003), lech 1000 lan -> xep hang Forgetting bi dao lon
     (iCaRL 0.0011 -> hang 9.8; EWC-only -> hang 11.0; GDumb 0.012 -> hang 4.0).
     Nay : truy cap truc tiep d[metric] -> KeyError neu thieu.

 [4] SEED KHONG KHOP KHOI — vi pham gia dinh cua Friedman.
     Cu : sorted(glob("seed_*")) tra ve thu tu TU DIEN
          (seed_123, seed_2024, seed_2384, seed_3141, seed_42, ...)
          roi [:min_seeds] cat theo thu tu nap.
     -> hang i cua ma tran KHONG cung mot seed giua cac method.
     Nay : ghep cap tuong minh bang dict {seed: value} + lay giao cac seed chung.

 [5] CROSSBAR VE DE LEN NHAU.
     Cu : y_line = k - (i+j)/2*0.9 - 0.3     # tinh xong roi VUT DI
          ax.plot([...], [k+0.95, k+0.95])   # dung hang so -> moi thanh de nhau
     -> alpha=0.3 chong lop, nhin ra 1 thanh dam duy nhat.
     Nay : tinh cac clique cuc dai, moi clique mot lane y rieng.

 [6] CROSSBAR: '< cd' phai la '<= cd'.

 [7] COT PHAI XEP SAI THU TU -> duong noi cat cheo nhau.
     Nay : hang xau nhat len tren cung (quy uoc Demsar 2006).

 [8] q_table.get(k, 3.5) — k>15 tra ve 3.5 vo can cu, khong canh bao.
     Nay : raise KeyError.

 [9] Khong canh bao khi N < k (Demsar khuyen nghi N >> k).
     N=5, k=11 -> CD=6.75 (phinh gap doi). Nay : canh bao ro.

[10] Static-CNN co Forgetting = 0.0 theo dinh nghia (khong cap nhat thi khong
     the quen) -> luon thang giai "it quen nhat" mot cach thoai hoa.
     Nay : loai khoi bang xep hang Forgetting (--keep-static-in-forgetting de giu).

[11] Them assertion tong hang = k(k+1)/2 de bat loi tinh hang.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, rankdata

from src.utils.common import get_logger, load_config


# ── Cau hinh ────────────────────────────────────────────────────────────────

# Nemenyi q_alpha, alpha=0.05 (two-tailed, df vo han)
Q_TABLE_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
               12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
               17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}

METRICS = ["aa_f1", "forgetting"]
LOWER_IS_BETTER = {"forgetting"}

# Khong cap nhat => Forgetting = 0 theo dinh nghia => thang thoai hoa. Xem [10].
FORGETTING_EXCLUDE = {"Static-CNN", "DistilBERT Static"}

MIN_SEEDS = 5          # so seed toi thieu de mot method duoc dua vao test
MIN_METHODS = 3        # so method toi thieu de chay Friedman


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """CD = q_alpha * sqrt( k(k+1) / (6N) ).  Xem [8]: khong co default am tham."""
    if alpha != 0.05:
        raise ValueError(f"chi co bang q cho alpha=0.05, nhan duoc {alpha}")
    if k not in Q_TABLE_005:
        raise KeyError(
            f"khong co q_alpha cho k={k} methods (bang chi den k=20). "
            f"Bo sung bang q hoac giam so method."
        )
    return Q_TABLE_005[k] * math.sqrt(k * (k + 1) / (6.0 * n))


# ── Nap du lieu per-seed ────────────────────────────────────────────────────

def load_per_seed(out_dir: Path, logger) -> dict[str, dict[int, dict]]:
    """
    Tra ve {method: {seed: metrics_dict}}.

    Nguon (theo thu tu uu tien) — xem [1]:
      1. results/multi_seed/all_seeds_raw.json      {seed: {method: metrics}}
      2. results/multi_seed/seed_*/seed_results.json  {method: metrics}
      3. results/extra_baselines_results.csv          cot: method,seed,aa_f1,...
    """
    per_seed: dict[str, dict[int, dict]] = {}
    multi_seed_dir = out_dir / "multi_seed"

    def add(method: str, seed: int, metrics: dict):
        per_seed.setdefault(method, {})[int(seed)] = metrics

    # -- 1. all_seeds_raw.json (day du nhat) --
    raw_path = multi_seed_dir / "all_seeds_raw.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        for seed_key, methods in raw.items():
            for method, metrics in methods.items():
                add(method, int(seed_key), metrics)
        logger.info(f"  Nap {raw_path.name}: {len(raw)} seeds")
    else:
        logger.warning(f"  Khong thay {raw_path}")

    # -- 2. seed_*/seed_results.json (du phong) --
    if not per_seed and multi_seed_dir.exists():
        seed_dirs = sorted(multi_seed_dir.glob("seed_*"),
                           key=lambda p: int(p.name.split("_")[1]))  # so, khong phai tu dien
        for seed_dir in seed_dirs:
            seed_num = int(seed_dir.name.split("_")[1])
            f = seed_dir / "seed_results.json"      # <- KHONG phai final_results.csv
            if not f.exists():
                continue
            for method, metrics in json.loads(f.read_text()).items():
                add(method, seed_num, metrics)
        if per_seed:
            logger.info(f"  Nap tu {len(seed_dirs)} thu muc seed_*/seed_results.json")

    # -- 3. extra baselines --
    extra_path = out_dir / "extra_baselines_results.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        for _, r in extra.iterrows():
            add(r["method"], int(r["seed"]), r.to_dict())
        logger.info(f"  Nap {extra_path.name}: {len(extra)} dong")

    return per_seed


def build_matrix(per_seed: dict[str, dict[int, dict]],
                 methods: list[str],
                 metric: str) -> tuple[np.ndarray, list[int]]:
    """
    Ghep cap TUONG MINH theo seed — xem [4].
    Tra ve (scores[n_blocks, n_methods], seeds_da_dung).
    """
    common = set(per_seed[methods[0]].keys())
    for m in methods[1:]:
        common &= set(per_seed[m].keys())
    seeds = sorted(common)
    if len(seeds) < 2:
        raise RuntimeError(f"chi co {len(seeds)} seed chung giua {len(methods)} method")

    scores = np.empty((len(seeds), len(methods)), dtype=float)
    for j, m in enumerate(methods):
        for i, s in enumerate(seeds):
            d = per_seed[m][s]
            if metric not in d:                       # xem [3]: khong fallback
                raise KeyError(
                    f"method {m!r}, seed {s}: thieu metric {metric!r}. "
                    f"Cac key co: {sorted(d.keys())}"
                )
            scores[i, j] = float(d[metric])
    return scores, seeds


# ── Ve CD diagram ───────────────────────────────────────────────────────────

def maximal_cliques(sorted_ranks: list[float], cd: float, tol: float = 1e-9):
    """Cac nhom lien tiep co do trai hang <= CD, chi giu nhom cuc dai. Xem [5][6]."""
    k = len(sorted_ranks)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and sorted_ranks[j + 1] - sorted_ranks[i] <= cd + tol:  # <= chu khong <
            j += 1
        if j > i:
            cliques.append((i, j))
    out = []
    for c in cliques:
        if any(c2 != c and c2[0] <= c[0] and c[1] <= c2[1] for c2 in cliques):
            continue
        if c not in out:
            out.append(c)
    return out


def draw_cd_diagram(avg_ranks, names, cd, title, subtitle, out_path,
                    highlight=("DRC-CL",), width=11.0, textspace=2.6,
                    row=0.42, dpi=300):
    """Critical Difference diagram theo quy uoc Demsar (2006)."""
    order = np.argsort(avg_ranks)
    vals = [float(avg_ranks[i]) for i in order]
    labels = [names[i] for i in order]
    k = len(vals)
    lo, hi = 1, k
    scale = width - 2 * textspace

    def xpos(r):
        return textspace + scale * (r - lo) / (hi - lo)

    y_ruler, y_axis, y_bar0, bar_step = -0.95, 0.0, 0.34, 0.20
    cliques = maximal_cliques(vals, cd)
    y_rows0 = y_bar0 + bar_step * max(len(cliques), 1) + 0.34
    n_left = int(math.ceil(k / 2.0))
    height = y_rows0 + row * n_left + 1.05

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(height - 0.55, y_ruler - 0.75)
    ax.axis("off")

    def line(pts, **kw):
        ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts], **kw))

    # truc hang
    line([(xpos(lo), y_axis), (xpos(hi), y_axis)], color="#222222", lw=1.6)
    for r in range(lo, hi + 1):
        line([(xpos(r), y_axis), (xpos(r), y_axis - 0.09)], color="#222222", lw=1.2)
        ax.text(xpos(r), y_axis - 0.15, str(r), ha="center", va="bottom",
                fontsize=9, color="#222222")

    # thuoc CD — lane rieng, co nap 2 dau
    cd_x0, cd_x1 = xpos(lo), xpos(min(lo + cd, hi))
    line([(cd_x0, y_ruler), (cd_x1, y_ruler)], color="#222222", lw=2.4)
    for x in (cd_x0, cd_x1):
        line([(x, y_ruler - 0.09), (x, y_ruler + 0.09)], color="#222222", lw=2.0)
    ax.text((cd_x0 + cd_x1) / 2, y_ruler - 0.14, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=10, fontweight="bold", color="#222222")

    # crossbars — moi clique mot lane rieng, xem [5]
    for idx, (i, j) in enumerate(cliques):
        y = y_bar0 + idx * bar_step
        line([(xpos(vals[i]) - 0.035, y), (xpos(vals[j]) + 0.035, y)],
             color="#111111", lw=4.2, solid_capstyle="round")

    hl = set(highlight)

    def draw_label(i, side, depth):
        y = y_rows0 + depth * row
        x = xpos(vals[i])
        bold = any(h in labels[i] for h in hl)
        col = "#1A56A8" if bold else "#333333"
        if side == "left":
            line([(x, y_axis), (x, y), (textspace - 0.14, y)],
                 color=col if bold else "#888888", lw=1.6 if bold else 1.0)
            ax.text(textspace - 0.24, y, f"{labels[i]}  ({vals[i]:.1f})", ha="right",
                    va="center", fontsize=9.5, color=col,
                    fontweight="bold" if bold else "normal")
        else:
            line([(x, y_axis), (x, y), (textspace + scale + 0.14, y)],
                 color=col if bold else "#888888", lw=1.6 if bold else 1.0)
            ax.text(textspace + scale + 0.24, y, f"({vals[i]:.1f})  {labels[i]}",
                    ha="left", va="center", fontsize=9.5, color=col,
                    fontweight="bold" if bold else "normal")

    for i in range(n_left):
        draw_label(i, "left", i)
    for i in range(n_left, k):
        draw_label(i, "right", k - 1 - i)      # hang xau nhat len tren, xem [7]

    ax.text(width / 2, y_ruler - 0.62, title, ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(width / 2, y_ruler - 0.40, subtitle, ha="center", va="bottom",
                fontsize=9.5, color="#444444")
    ax.text(width / 2, height - 0.62, "Average rank  (lower = better)",
            ha="center", va="top", fontsize=9.5, color="#444444")
    ax.text(width / 2, height - 0.32,
            "Thick bars join methods whose average ranks are NOT significantly "
            "different (spread <= CD).", ha="center", va="top",
            fontsize=8.2, color="#666666")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return cliques


# ── Chuong trinh chinh ──────────────────────────────────────────────────────

def run(cfg: dict, make_figures: bool = True, keep_static_in_forgetting: bool = False):
    out_dir = Path(cfg["paths"]["results"])
    logger = get_logger("stat_tests", log_dir=out_dir / "logs")

    logger.info("=" * 68)
    logger.info(" FRIEDMAN TEST + CRITICAL DIFFERENCE DIAGRAM")
    logger.info("=" * 68)

    per_seed = load_per_seed(out_dir, logger)
    if not per_seed:
        raise RuntimeError(
            "Khong nap duoc du lieu per-seed nao.\n"
            "  Can co: results/multi_seed/all_seeds_raw.json\n"
            "      hoac: results/multi_seed/seed_*/seed_results.json\n"
            "  Chay truoc: python -m src.models.run_multi_seed"
        )

    logger.info("\n  So seed per-seed THAT cua tung method:")
    for m in sorted(per_seed):
        seeds = sorted(per_seed[m])
        logger.info(f"    {m:<24} {len(seeds):>2} seeds  {seeds}")

    all_results = []
    for metric in METRICS:
        logger.info("\n" + "-" * 68)
        logger.info(f"  METRIC: {metric}")
        logger.info("-" * 68)

        methods = sorted(m for m, d in per_seed.items() if len(d) >= MIN_SEEDS)
        if metric == "forgetting" and not keep_static_in_forgetting:
            dropped = [m for m in methods if m in FORGETTING_EXCLUDE]
            methods = [m for m in methods if m not in FORGETTING_EXCLUDE]
            if dropped:
                logger.info(f"  Loai khoi bang xep hang Forgetting (xem [10]): {dropped}")

        # [2] KHONG BAO GIO sinh du lieu gia — tha chet con hon bia so.
        if len(methods) < MIN_METHODS:
            raise RuntimeError(
                f"Chi co {len(methods)} method du >= {MIN_SEEDS} seed "
                f"(can >= {MIN_METHODS}) cho metric {metric!r}.\n"
                f"  Method tim thay: {sorted(per_seed)}\n"
                f"  Ban CU se sinh du lieu gia bang rng.normal() o day — da xoa.\n"
                f"  Hay chay lai run_multi_seed.py cho du seed."
            )

        scores, seeds = build_matrix(per_seed, methods, metric)
        n_blocks, k = scores.shape
        logger.info(f"  Blocks (seed) = {n_blocks}  {seeds}")
        logger.info(f"  Methods       = {k}")

        # [12] Method it seed keo N cua CA NHOM xuong (giao cac seed chung).
        n_max = max(len(per_seed[m]) for m in methods)
        if n_blocks < n_max:
            bottleneck = sorted(m for m in methods if len(per_seed[m]) == n_blocks)
            logger.warning(
                f"  CANH BAO: N bi cat tu {n_max} xuong {n_blocks} vi cac method "
                f"chi co {n_blocks} seed: {bottleneck}. "
                f"Dieu nay VUT BO {n_max - n_blocks} seed cua cac method khac va "
                f"lam CD phinh to. Cach xu ly: (a) chay them seed cho "
                f"{bottleneck}, hoac (b) loai chung khoi test de dat N={n_max} "
                f"(CD se giam dang ke)."
            )

        # [9] Demsar khuyen nghi N >> k
        if n_blocks < k:
            logger.warning(
                f"  CANH BAO: N={n_blocks} < k={k}. Friedman thieu luc kiem dinh; "
                f"CD se phinh to va ket luan 'khong khac biet' co the chi la "
                f"artifact. Demsar (2006) khuyen nghi N >> k."
            )
        p_min = 2.0 / (2 ** n_blocks)
        if p_min > 0.05:
            logger.warning(
                f"  CANH BAO: voi N={n_blocks}, p nho nhat co the dat = 2/2^N = "
                f"{p_min:.4f} > 0.05. Khong the co y nghia thong ke du khac biet "
                f"lon den dau."
            )

        try:
            stat, p_value = friedmanchisquare(*[scores[:, j] for j in range(k)])
            logger.info(f"\n  Friedman: chi2 = {stat:.4f}, p = {p_value:.6f}  "
                        f"({'significant' if p_value < 0.05 else 'not significant'})")
        except Exception as e:
            logger.warning(f"  Friedman that bai: {e}")
            stat, p_value = float("nan"), float("nan")

        sign = 1 if metric in LOWER_IS_BETTER else -1
        ranks = np.array([rankdata(sign * scores[i, :]) for i in range(n_blocks)])
        avg_ranks = ranks.mean(axis=0)

        # [11] tong hang phai = k(k+1)/2
        expected = k * (k + 1) / 2.0
        assert abs(avg_ranks.sum() - expected) < 1e-6, \
            f"tong hang = {avg_ranks.sum():.4f}, phai = {expected:.4f} — loi tinh hang!"

        logger.info(f"\n  Hang trung binh ({metric}, lower = better):")
        for idx in np.argsort(avg_ranks):
            logger.info(f"    {methods[idx]:<24} rank = {avg_ranks[idx]:5.2f}   "
                        f"mean = {scores[:, idx].mean():.4f} "
                        f"+/- {scores[:, idx].std(ddof=1):.4f}")

        cd = nemenyi_cd(k, n_blocks)
        logger.info(f"\n  Nemenyi CD (alpha=0.05, k={k}, N={n_blocks}) = {cd:.3f}")

        sig = [(methods[a], methods[b], abs(avg_ranks[a] - avg_ranks[b]))
               for a in range(k) for b in range(a + 1, k)
               if abs(avg_ranks[a] - avg_ranks[b]) > cd]
        logger.info(f"  Cap khac biet CO y nghia: {len(sig)}")
        for a, b, d in sorted(sig, key=lambda t: -t[2]):
            logger.info(f"    {a:<22} vs {b:<22} delta = {d:.2f}")
        if not sig:
            logger.info("    (khong cap nao) — LUU Y: khong bac bo != tuong duong. "
                        "Muon ket luan tuong duong phai dung TOST.")

        for a, b, d in sig:
            all_results.append({"metric": metric, "method_a": a, "method_b": b,
                                "rank_diff": round(d, 3), "cd": round(cd, 3)})

        if make_figures:
            label = ("AA-F1 (higher is better)" if metric == "aa_f1"
                     else "Forgetting (lower is better)")
            out_png = out_dir / f"cd_diagram_{metric}.png"
            cliques = draw_cd_diagram(
                avg_ranks.tolist(), methods, cd,
                f"Critical Difference Diagram — {label}",
                f"Friedman + Nemenyi post-hoc, alpha = 0.05  |  k = {k}, N = {n_blocks}",
                out_png,
            )
            logger.info(f"  Da luu: {out_png}  ({len(cliques)} crossbar)")

    if all_results:
        p = out_dir / "nemenyi_significant_pairs.csv"
        pd.DataFrame(all_results).to_csv(p, index=False)
        logger.info(f"\n  Da luu: {p}")

    logger.info("\n  Statistical tests complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-figures", action="store_true",
                        help="chi chay test, khong ve hinh")
    parser.add_argument("--keep-static-in-forgetting", action="store_true",
                        help="giu Static-CNN trong bang xep hang Forgetting (xem [10])")
    args = parser.parse_args()
    run(load_config(args.config),
        make_figures=not args.no_figures,
        keep_static_in_forgetting=args.keep_static_in_forgetting)
