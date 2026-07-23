#!/usr/bin/env python
"""
anchor_sweep_inscope.py — is `epr` the right label-free anchor? (Step 193, Phase 2)

L-SML's fused score carries an arbitrary global sign (a leading-eigenvector coin flip,
Step 148). We resolve it label-free by correlating the fused score against one oriented
"anchor" view and flipping if they disagree. `epr` has been the anchor since the start
(ANCHOR_PRIORITY in spectral_utils/subset_sweep.py); this sweep asks whether that choice
is load-bearing, and whether any other view does better.

The key structural fact this exploits: **the fusion does not depend on the anchor.** Only
the final sign does. So we fuse GOOD_6 once per cell and then re-orient the SAME fused
score against each candidate anchor. Any AUROC difference between anchors is therefore
purely a sign decision — a per-cell binary — never a different model.

That also means the right way to read the output is NOT "which anchor has the highest
macro AUROC". An anchor's job is to be *stably right about the sign*, so the columns that
matter are `n_wrong_sign` (times the anchor flipped the score away from the better
orientation) and `agree_with_epr`.

Usage:
    python scripts/anchor_sweep_inscope.py
Writes: results/advisor_inscope/anchor_sweep.csv
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

from sklearn.metrics import roc_auc_score

from spectral_utils.subset_sweep import (
    iter_cells, prepare_cell, CANONICAL_POOL, GOOD_6,
)
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.streaming_utils import anchor_orient
from inscope_cells import INSCOPE, GROUP

# `epr` first = the incumbent. The rest are the wide-pool views that outrank epr on mean
# oriented AUROC in the Step-192 orientation audit, plus cusum_max (the Step-184 alt).
CANDIDATES = [
    "epr", "topk_tail_mass", "mean_logprob_entropy", "renyi_entropy_2",
    "cusum_max", "rpdi", "stft_max_high_power", "spectral_entropy", "low_band_power",
]

OUT = os.path.join(REPO, "results", "advisor_inscope", "anchor_sweep.csv")
DD = os.path.join(REPO, "local_cache")


def main():
    rows = []
    for domain, ck, fd, labels in iter_cells(
            DD, domains=None, cells=None,
            derived_views_pkl=os.path.join(DD, "derived_views.pkl"),
            trace_cells_pkl=os.path.join(DD, "trace_cells.pkl")):
        if ck not in INSCOPE:
            continue
        ctx = prepare_cell(domain, ck, fd, labels, feature_pool=CANONICAL_POOL)
        if ctx is None:
            continue
        cols = [ctx.pool.index(f) for f in GOOD_6 if f in ctx.pool]
        if len(cols) < 3:
            print(f"[skip] {ck}: GOOD_6 has <3 members in pool")
            continue
        cols = sorted(cols)

        # fuse ONCE — the anchor plays no part in this
        fused, meta = lsml_continuous(*[ctx.V[:, j] for j in cols], method="residual")
        y = np.asarray(ctx.labels, dtype=int)
        if np.std(fused) < 1e-12:
            continue
        auc_pos = float(roc_auc_score(y, fused))
        best = max(auc_pos, 1.0 - auc_pos)          # the better of the two orientations

        for a in CANDIDATES:
            if a not in ctx.pool:
                rows.append(dict(cell=ck, group=GROUP.get(ck), anchor=a,
                                 present=False, auroc=None, flipped=None,
                                 wrong_sign=None, K=int(meta["K"]), best_possible=best))
                continue
            oriented, flipped = anchor_orient(fused, ctx.V[:, ctx.pool.index(a)])
            auc = float(roc_auc_score(y, oriented))
            rows.append(dict(cell=ck, group=GROUP.get(ck), anchor=a, present=True,
                             auroc=round(auc, 4), flipped=bool(flipped),
                             wrong_sign=bool(auc < 0.5 - 1e-12 or auc < best - 1e-12),
                             K=int(meta["K"]), best_possible=round(best, 4)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── summary ───────────────────────────────────────────────────────────────────────
    epr = {r["cell"]: r for r in rows if r["anchor"] == "epr"}
    print(f"\nGOOD_6 fused once per cell; only the sign decision differs between anchors.\n")
    print(f"{'anchor':<22} {'cells':>6} {'macro':>8} {'QA':>8} {'math':>8} "
          f"{'wrong_sign':>11} {'agree_epr':>10}")
    print("-" * 80)
    for a in CANDIDATES:
        sel = [r for r in rows if r["anchor"] == a and r["present"]]
        if not sel:
            print(f"{a:<22} {'0':>6}  (absent from every cell)")
            continue
        macro = np.mean([r["auroc"] for r in sel])
        qa = [r["auroc"] for r in sel if r["group"] == "QA"]
        mt = [r["auroc"] for r in sel if r["group"] == "math"]
        wrong = sum(1 for r in sel if r["wrong_sign"])
        agree = sum(1 for r in sel
                    if epr.get(r["cell"]) and epr[r["cell"]]["present"]
                    and epr[r["cell"]]["flipped"] == r["flipped"])
        print(f"{a:<22} {len(sel):>6} {macro:>8.4f} "
              f"{(np.mean(qa) if qa else float('nan')):>8.4f} "
              f"{(np.mean(mt) if mt else float('nan')):>8.4f} "
              f"{wrong:>11} {agree:>7}/{len(sel)}")

    print(f"\nceiling if the sign were always resolved correctly: "
          f"{np.mean([r['best_possible'] for r in rows if r['anchor'] == 'epr']):.4f}")
    bad = [r["cell"] for r in rows if r["anchor"] == "epr" and r["present"] and r["wrong_sign"]]
    print(f"cells where the epr anchor picks the WRONG sign: {bad or 'none'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
