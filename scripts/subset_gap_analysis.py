#!/usr/bin/env python
"""
subset_gap_analysis.py — why didn't the search find a better subset? (Step 193, Phase 4.2)

Step 192 left an open question: on the cells where a searched subset appears to beat the
fixed GOOD_5/GOOD_6, why doesn't any label-free selector convert that into a real win?
There are two very different answers and they call for opposite follow-ups:

    "the search failed"     — a better subset exists and our selectors miss it
                              -> keep building selectors
    "there was nothing"     — the apparent win is winner's curse and does not replicate
                              -> stop building selectors

The split-half oracle separates them directly. A subset is chosen greedily on half A and
then scored on the held-out half B:

    optimism_gap = greedy_halfA - greedy_halfB     how much of the win was illusion
    honest_gain  = greedy_halfB - good5_halfB      what survives out of sample

`honest_gain` is the only quantity that matters for the thesis claim. This script pairs it
per cell with the covariates that could explain it (n, class balance, K, pool size,
anti-oriented feature count) and reports which one actually tracks the gaps.

Usage:
    python scripts/subset_gap_analysis.py
Writes: results/advisor_inscope/subset_gap_analysis.csv
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_cells import INSCOPE, GROUP

BENCH = os.path.join(REPO, "results", "selector_bench")
SPLIT = os.path.join(BENCH, "splithalf_oracle_c46_inscope_summary.csv")
ORIENT = os.path.join(BENCH, "inscope_feature_orientation.csv")
REFMAC = os.path.join(BENCH, "reference_macros__c46.csv")
SCORES = os.path.join(REPO, "results", "repgrid", "scores_lsml_upcr.csv")
OUT = os.path.join(REPO, "results", "advisor_inscope", "subset_gap_analysis.csv")


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def spearman(x, y):
    """Rank correlation on the pairs where both are finite (no scipy dependency)."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
             and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 4:
        return float("nan"), 0
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan"), len(pairs)
    return float(np.corrcoef(ra, rb)[0, 1]), len(pairs)


def main():
    # covariates ----------------------------------------------------------------------
    pos_rate, K_good6, p_pool = {}, {}, {}
    for r in read(SCORES):
        if r["cell"] in INSCOPE and r["cell"] not in pos_rate:
            pos_rate[r["cell"]] = fnum(r.get("acc"))
    for r in read(REFMAC):
        if r.get("cell") in INSCOPE:
            p_pool[r["cell"]] = fnum(r.get("p_pool"))
            if r.get("variant") == "ref.GOOD_6":
                K_good6[r["cell"]] = fnum(r.get("K"))

    n_anti = {}
    for r in read(ORIENT):
        c = r.get("cell")
        if c not in INSCOPE:
            continue
        a = fnum(r.get("oriented_auroc"))
        if a is None:
            continue
        n_anti[c] = n_anti.get(c, 0) + (1 if a < 0.5 else 0)

    rows = []
    for r in read(SPLIT):
        c = r.get("cell")
        if c not in INSCOPE:
            continue
        gA, gB = fnum(r.get("greedy_halfA")), fnum(r.get("greedy_halfB"))
        g5B = fnum(r.get("good5_halfB"))
        rows.append(dict(
            cell=c, group=GROUP.get(c), n=fnum(r.get("n")),
            greedy_halfA=gA, greedy_halfB=gB, good5_halfB=g5B,
            optimism_gap=(gA - gB) if (gA is not None and gB is not None) else None,
            honest_gain=(gB - g5B) if (gB is not None and g5B is not None) else None,
            insample_gain=(gA - g5B) if (gA is not None and g5B is not None) else None,
            fulloracle_halfB=fnum(r.get("fulloracle_halfB")),
            pos_rate=pos_rate.get(c), K_good6=K_good6.get(c),
            p_pool=p_pool.get(c), n_anti_oriented=n_anti.get(c),
        ))

    rows.sort(key=lambda d: (d["honest_gain"] is None, d["honest_gain"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── report ────────────────────────────────────────────────────────────────────────
    print("Split-half oracle: subset chosen greedily on half A, scored on held-out half B.")
    print("  insample_gain = greedy_halfA - good5_halfB   (what a naive search would claim)")
    print("  honest_gain   = greedy_halfB - good5_halfB   (what actually survives)")
    print("  optimism_gap  = greedy_halfA - greedy_halfB  (the winner's curse)\n")
    print(f"{'cell':<32} {'grp':>4} {'n':>6} {'insample':>9} {'honest':>8} "
          f"{'optimism':>9} {'pos':>6} {'K':>3} {'anti':>5}")
    print("-" * 96)
    for d in rows:
        f2 = lambda v, w=8, p=4: (f"{v:>{w}.{p}f}" if v is not None and np.isfinite(v)
                                  else f"{'--':>{w}}")
        print(f"{d['cell']:<32} {d['group'] or '?':>4} {int(d['n'] or 0):>6} "
              f"{f2(d['insample_gain'], 9)} {f2(d['honest_gain'])} {f2(d['optimism_gap'], 9)} "
              f"{f2(d['pos_rate'], 6, 3)} {int(d['K_good6'] or 0):>3} "
              f"{int(d['n_anti_oriented'] or 0):>5}")
    print("-" * 96)

    ins = [d["insample_gain"] for d in rows if d["insample_gain"] is not None]
    hon = [d["honest_gain"] for d in rows if d["honest_gain"] is not None]
    opt = [d["optimism_gap"] for d in rows if d["optimism_gap"] is not None]
    n_win = sum(1 for v in hon if v > 0)
    print(f"\nmean in-sample gain over GOOD_5 : {np.mean(ins):+.4f}   "
          f"(cells apparently better: {sum(1 for v in ins if v > 0)}/{len(ins)})")
    print(f"mean HONEST gain over GOOD_5    : {np.mean(hon):+.4f}   "
          f"(cells actually better: {n_win}/{len(hon)})")
    print(f"mean winner's-curse optimism    : {np.mean(opt):+.4f}   "
          f"max {np.max(opt):+.4f} on "
          f"{max(rows, key=lambda d: d['optimism_gap'] or -9)['cell']}")
    print(f"=> {100*np.mean(opt)/max(np.mean(ins), 1e-9):.0f}% of the apparent gain is illusion")

    print("\nwhich covariate explains the outcome? (Spearman rho over the 25 cells)")
    print(f"{'covariate':<20} {'vs optimism_gap':>18} {'vs honest_gain':>16}")
    print("-" * 58)
    for cov in ("n", "pos_rate", "K_good6", "p_pool", "n_anti_oriented"):
        xs = [d[cov] for d in rows]
        r1, n1 = spearman(xs, [d["optimism_gap"] for d in rows])
        r2, n2 = spearman(xs, [d["honest_gain"] for d in rows])
        print(f"{cov:<20} {r1:>13.3f} (n={n1:>2}) {r2:>11.3f} (n={n2:>2})")

    missing = [d["cell"] for d in rows if d["fulloracle_halfB"] is None]
    print(f"\ncells with no full-oracle column: {missing}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
