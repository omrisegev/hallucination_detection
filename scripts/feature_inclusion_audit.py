#!/usr/bin/env python
"""
feature_inclusion_audit.py — which views are NEVER worth picking? (Step 193e)

Omri's pool-pruning question, answered from the sweeps we already have rather than from a
new one: for every view, does it ever appear in a cell's best subsets, and what does it cost
to drop it?

Source: results/subset_sweep/<domain>__<cell>.npz — the exhaustive Step-153 enumeration of
every subset of the H16 pool (sizes 3-16, 65,399 subsets per cell), with each subset's AUROC
already stored. So the "was it ever chosen" question needs no new compute.

SCOPE LIMIT: these enumerations cover the **16-view H16 pool**, not the 30-view wide pool.
So this prunes within H16 and cannot say anything about the 14 energy/logprob views. The
30-view answer needs the bounded sweep queued for next session.

Three statistics per view, from weakest to strongest evidence:
  1. top-N inclusion   — how often it appears in each cell's N best subsets.
  2. best-subset hit   — is it in the single best subset for that cell.
  3. LOVO cost         — best AUROC achievable WITHOUT it, vs the cell's overall best.
                         This is the decisive one: a view worth keeping is one whose removal
                         costs something on at least some cell.

A view is a drop candidate only if it is (near-)absent from top-N everywhere AND its LOVO
cost is ~0 everywhere. Individual AUROC is deliberately NOT used as a criterion: L-SML
assigns negative weights, so an anti-oriented view can still carry information.

Usage:
    python scripts/feature_inclusion_audit.py [--top 100]
Writes: results/advisor_inscope/feature_inclusion_audit.csv
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

from spectral_utils.subset_sweep import CANONICAL_POOL, H16
from inscope_cells import INSCOPE, GROUP

SWEEP = os.path.join(REPO, "results", "subset_sweep")
OUT = os.path.join(REPO, "results", "advisor_inscope", "feature_inclusion_audit.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100,
                    help="how many best subsets per cell count as 'the best subsets'")
    ap.add_argument("--pool", choices=["h16", "c46"], default="h16",
                    help="h16 = Step-153 16-view enumerations (results/subset_sweep); "
                         "c46 = Step-194 30-view sizes-3-5 enumerations "
                         "(results/subset_sweep_c46)")
    ap.add_argument("--npz-dir", default=None,
                    help="override the npz directory (default depends on --pool)")
    args = ap.parse_args()

    global SWEEP, OUT
    if args.pool == "c46":
        from spectral_utils.feature_utils import FEAT_NAMES
        from spectral_utils.subset_sweep import REPGRID_VIEWS
        pool = list(FEAT_NAMES) + REPGRID_VIEWS          # the 30-view manifest pool
        SWEEP = args.npz_dir or os.path.join(REPO, "results", "subset_sweep_c46")
        OUT = OUT.replace(".csv", "_c46.csv")
    else:
        pool = list(H16)
        if args.npz_dir:
            SWEEP = args.npz_dir
    bit_of = {f: CANONICAL_POOL.index(f) for f in pool}

    per_cell = {}
    for c in INSCOPE:
        path = os.path.join(SWEEP, f"repgrid__{c}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        mask, auroc = z["mask"], z["auroc"].astype(float)
        ok = np.isfinite(auroc)
        mask, auroc = mask[ok], auroc[ok]
        if len(auroc) == 0:
            continue
        order = np.argsort(-auroc)
        topN = order[:args.top]
        best_mask = int(mask[order[0]])
        best_auc = float(auroc[order[0]])

        stats = {}
        for fname in pool:
            b = np.uint64(1) << np.uint64(bit_of[fname])
            has = (mask & b) != 0
            in_top = int(has[topN].sum())
            without = auroc[~has]
            lovo = float(best_auc - without.max()) if without.size else float("nan")
            stats[fname] = dict(in_top=in_top,
                                in_best=bool(best_mask & int(b)),
                                lovo_cost=lovo)
        per_cell[c] = dict(stats=stats, best_auc=best_auc, n=len(auroc))

    if not per_cell:
        print("no enumerations found for the in-scope cells"); return

    rows = []
    for fname in pool:
        it = [per_cell[c]["stats"][fname]["in_top"] for c in per_cell]
        ib = [per_cell[c]["stats"][fname]["in_best"] for c in per_cell]
        lc = [per_cell[c]["stats"][fname]["lovo_cost"] for c in per_cell]
        lc = [x for x in lc if np.isfinite(x)]
        rows.append(dict(
            feature=fname,
            cells=len(per_cell),
            mean_top_inclusion=round(float(np.mean(it)) / args.top, 4),
            cells_absent_from_top=int(sum(1 for x in it if x == 0)),
            cells_in_best_subset=int(sum(ib)),
            mean_lovo_cost=round(float(np.mean(lc)), 5) if lc else None,
            max_lovo_cost=round(float(np.max(lc)), 5) if lc else None,
            cells_lovo_zero=int(sum(1 for x in lc if x <= 1e-6)),
        ))

    rows.sort(key=lambda d: (d["mean_top_inclusion"], d["mean_lovo_cost"] or 0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"Exhaustive H16 enumerations found for {len(per_cell)} of {len(INSCOPE)} "
          f"in-scope cells ({rows[0]['cells']} used).")
    print(f"'top' = each cell's {args.top} best subsets by AUROC.\n")
    print(f"{'feature':<22}{'in top-N':>10}{'absent':>8}{'in best':>9}"
          f"{'LOVO mean':>11}{'LOVO max':>10}{'LOVO=0':>8}  verdict")
    print("-" * 96)
    for d in rows:
        drop = (d["cells_absent_from_top"] == d["cells"]
                and (d["max_lovo_cost"] or 0) <= 1e-6)
        weak = d["mean_top_inclusion"] < 0.10 and (d["max_lovo_cost"] or 0) < 0.005
        verdict = ("DROP" if drop else ("weak - candidate" if weak else "keep"))
        print(f"{d['feature']:<22}{100*d['mean_top_inclusion']:>9.1f}%"
              f"{d['cells_absent_from_top']:>8}{d['cells_in_best_subset']:>9}"
              f"{100*(d['mean_lovo_cost'] or 0):>10.2f}"
              f"{100*(d['max_lovo_cost'] or 0):>10.2f}{d['cells_lovo_zero']:>8}  {verdict}")
    print("-" * 96)
    print("in top-N = mean share of that cell's best subsets containing the view")
    print("absent   = cells where it appears in NONE of the top-N")
    print("LOVO     = AUROC lost by banning the view outright (pp); 0 = fully redundant")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
