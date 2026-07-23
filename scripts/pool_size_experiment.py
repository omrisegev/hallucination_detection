#!/usr/bin/env python
"""
pool_size_experiment.py — would shrinking the 30-view pool raise performance? (Step 193f)

Omri's question: if we drop back to ~20 views, does the label-free selector get better?
The intuition is that a smaller search space means less selection noise, which Step 193
identified as the binding constraint (65% of apparent selection gain is winner's curse).

This tests it directly: build nested pools of decreasing size, run the trace-based
Gated-Laplacian selector on each, and report macro AUROC over the 25 in-scope cells.

RANKING CRITERION — informativeness, i.e. |AUROC - 0.5|, NOT raw AUROC. Continuous L-SML
assigns negative weights, so a view at 0.28 is as usable as one at 0.72; both are strong.
A view at 0.50 is the useless one. Ranking by raw AUROC would drop the strongest inverted
views (epr_spilled 0.277 -> 0.723 flipped) and keep genuinely dead ones (pe_mean 0.4994).

CAVEAT: the ranking is computed from per-cell labelled AUROC aggregated over all 25 cells,
so a pool derived this way carries corpus-level label information — the same asymmetry
GOOD_6 has. These numbers are an IN-SAMPLE upper bound on what pruning can buy. A real
claim needs the leave-one-cell-out protocol (results/subset_sweep/loco.csv machinery).

Usage:
    python scripts/pool_size_experiment.py [--sizes 30,24,20,16,12]
Writes: results/advisor_inscope/pool_size_experiment.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.subset_sweep import (
    iter_cells, prepare_cell, CANONICAL_POOL, GOOD_6,
)
from spectral_utils.selector_bench import eval_subset_flex, UnlabeledCell
from spectral_utils.selectors import get_selector
from inscope_cells import INSCOPE, GROUP

BENCH = os.path.join(REPO, "results", "selector_bench")
OUT = os.path.join(REPO, "results", "advisor_inscope", "pool_size_experiment.csv")
DD = os.path.join(REPO, "local_cache")
VARIANT = "a2.dufs"


def informativeness_ranking():
    """Views ordered most -> least informative by mean |AUROC - 0.5| across the 25 cells."""
    agg = {}
    path = os.path.join(BENCH, "inscope_feature_orientation.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["cell"] not in INSCOPE:
                continue
            try:
                a = float(r["oriented_auroc"])
            except (TypeError, ValueError):
                continue
            agg.setdefault(r["feature"], []).append(abs(a - 0.5))
    score = {k: float(np.mean(v)) for k, v in agg.items() if v}
    return sorted(score, key=lambda k: -score[k]), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="30,24,20,16,12")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    ranked, score = informativeness_ranking()
    print("Views ranked by informativeness = mean |AUROC - 0.5| over the 25 in-scope cells")
    print("(this is the criterion L-SML actually cares about; sign is handled by the fusion)\n")
    for i, k in enumerate(ranked, 1):
        mark = ""
        if i > min(sizes):
            mark = "  <- dropped first"
        print(f"  {i:>2}. {k:<24}{score[k]:.4f}{mark}")

    # cells, prepared once per pool
    raw = []
    for domain, ck, fd, labels in iter_cells(
            DD, domains=None, cells=None,
            derived_views_pkl=os.path.join(DD, "derived_views.pkl"),
            trace_cells_pkl=os.path.join(DD, "trace_cells.pkl")):
        if ck in INSCOPE:
            raw.append((domain, ck, fd, labels))
    print(f"\ncells loaded: {len(raw)}")

    sel_fn = get_selector("a2_groupfs")
    rows = []
    for p in sizes:
        pool = [f for f in CANONICAL_POOL if f in set(ranked[:p])]
        aur_sel, aur_g6, sizes_sel = [], [], []
        for domain, ck, fd, labels in raw:
            ctx = prepare_cell(domain, ck, fd, labels, feature_pool=pool)
            if ctx is None:
                continue
            # fixed GOOD_6 (pool-invariant sanity check)
            g6cols = sorted(ctx.pool.index(f) for f in GOOD_6 if f in ctx.pool)
            if len(g6cols) >= 3:
                aur_g6.append(eval_subset_flex(ctx, g6cols)["auroc"])
            # the selector
            cell = UnlabeledCell.from_context(ctx)
            rng = np.random.default_rng(0)
            try:
                out = sel_fn(cell, rng)
            except Exception as e:
                print(f"   [{ck}] selector failed: {type(e).__name__}: {e}")
                continue
            d = next((x for x in out if x["variant"] == VARIANT), None)
            if d is None or len(d["cols"]) < 3:
                continue
            r = eval_subset_flex(ctx, list(d["cols"]))
            if np.isfinite(r["auroc"]):
                aur_sel.append(r["auroc"])
                sizes_sel.append(len(d["cols"]))
        rows.append(dict(pool_size=p, n_cells=len(aur_sel),
                         selector_macro=float(np.mean(aur_sel)) if aur_sel else None,
                         good6_macro=float(np.mean(aur_g6)) if aur_g6 else None,
                         mean_selected=float(np.mean(sizes_sel)) if sizes_sel else None,
                         dropped="|".join(ranked[p:])))
        print(f"  pool {p:>2} views -> selector {rows[-1]['selector_macro']:.4f} "
              f"| GOOD_6 {rows[-1]['good6_macro']:.4f} "
              f"| mean picked {rows[-1]['mean_selected']:.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'pool':>6}{'selector':>11}{'vs 30-view':>12}{'GOOD_6':>10}{'picked':>9}")
    print("-" * 50)
    base = next((r["selector_macro"] for r in rows if r["pool_size"] == max(sizes)), None)
    for r in rows:
        d = (100 * (r["selector_macro"] - base)) if base else float("nan")
        print(f"{r['pool_size']:>6}{r['selector_macro']:>11.4f}{d:>+12.2f}"
              f"{r['good6_macro']:>10.4f}{r['mean_selected']:>9.1f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
