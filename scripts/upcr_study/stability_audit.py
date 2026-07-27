"""
stability_audit.py — how much of each published number is decided by the data,
and how much by floating-point rounding?

WHY THIS EXISTS
---------------
`reproduction_audit.py` found 91 of 169 bench rows no longer reproducing bit for
bit, and the drift was concentrated in one place: **size-4 subsets**, with
per-cell swings up to 53pp. Chasing one of them down (`ref.consensus_4` on
`lapeigvals_gsm8k_phi35`, 0.6833 -> 0.7802) gave the explanation, and it is not a
bug in either version:

    score-matrix magnitude          0.156
    max |vectorised - loop| at m=4  2.5e-16      (pure float rounding)
    K=3 partition, loop             [0,1,2,0] -> Eq.14 residual 0.6018
    K=3 partition, vectorised       [0,1,0,2] -> Eq.14 residual 0.3927

The residual grid then prefers K=3 in one case and K=2 in the other, and the
fused AUROC moves 9.7pp. Perturbing the score matrix by a *relative* 1e-16 —
i.e. by nothing at all — already produces two different K=3 partitions across
random draws; by 1e-8 it produces four.

So at small m, L-SML's group assignment is not determined by the data. Which
partition you get is decided in the last bits, the residual grid amplifies that
into a different K, and K changes the fused score materially. Step 203's
vectorisation did not cause this; it perturbed at 1e-16 and that was enough to
expose it. Reverting would not fix it either — it would just re-pick one
arbitrary side of the tie.

WHAT THIS SCRIPT MEASURES
-------------------------
For every (variant, pool) row, re-fuses that row's stored subsets with the
feature matrix jittered by a relative 1e-10 — far below any measurement
precision, far above float noise — across several seeds, and reports how far the
AUROC moves:

    spread_pp        mean over cells of (max - min) AUROC across seeds
    max_spread_pp    the worst single cell
    frac_unstable    share of cells whose spread exceeds 0.5pp
    macro_spread_pp  spread of the row's MACRO across seeds (what the table quotes)

A row with spread ~0 is a measurement. A row with spread of several pp is one
draw from a distribution, and its rank on the leaderboard is not meaningful at
that resolution.

Out: results/upcr_study/00_reproduction_audit/stability_audit.csv (+ summary)
"""
import os
import sys
import csv
import glob
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                   # noqa: E402

from spectral_utils.fusion_utils import lsml_continuous, upcr_fuse   # noqa: E402
from spectral_utils.streaming_utils import anchor_orient             # noqa: E402

OUT = S.outdir("00_reproduction_audit")

REL_JITTER = 1e-10      # 100x above double rounding, 1e8 x below any real precision
N_SEEDS = 5
UNSTABLE_PP = 0.5       # a cell counts as unstable if its AUROC moves this much
GROUPS_OVERRIDE = {"a2.select+groups", "a2.groups@good5"}
from reproduction_audit import is_k_override        # noqa: E402


def fuse_auroc(cell, cols, fusion, k_override, rng=None):
    V = cell["V"]
    if rng is not None:
        V = V * (1.0 + REL_JITTER * rng.standard_normal(V.shape))
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return float("nan")
    if fusion == "upcr":
        w, _, _ = upcr_fuse(V[:, cols].T)
        fused = V[:, cols] @ w
    else:
        kw = {}
        if k_override is not None:
            m = len(cols)
            kw["K_range"] = [max(2, min(int(k_override), max(2, min(m - 1, 8))))]
        fused, _ = lsml_continuous(*[V[:, j] for j in cols], method="residual", **kw)
    oriented, _ = anchor_orient(np.asarray(fused, float), cell["anchor"])
    if np.std(oriented) < 1e-12:
        return float("nan")
    return float(roc_auc_score(cell["labels"], oriented))


def main():
    cells = S.load()
    S.validity_check(cells)
    inscope = set(cells)

    by = {}
    for f in sorted(glob.glob(os.path.join(S.REPO, "results", "selector_bench", "*.csv"))):
        b = os.path.basename(f)
        if "__" not in b:
            continue
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
        except Exception:
            continue
        if not rows or "variant" not in rows[0]:
            continue
        for r in rows:
            if r.get("cell") in inscope and r.get("chosen"):
                by.setdefault((r["variant"], r.get("pool_mode", "")), []).append(r)

    print(f"{len(by)} (variant, pool) pairs x {N_SEEDS} seeds, "
          f"relative jitter {REL_JITTER:.0e}")

    out_rows, t0 = [], time.time()
    for n, (key, rows) in enumerate(sorted(by.items()), 1):
        variant, pool = key
        if variant in GROUPS_OVERRIDE:
            continue
        k_rule = is_k_override(variant)
        per_cell, macros = [], [[] for _ in range(N_SEEDS)]
        sizes = []
        for r in rows:
            cell = cells[r["cell"]]
            names = [x for x in r["chosen"].split("|") if x]
            cols = [cell["pool"].index(x) for x in names if x in cell["pool"]]
            sizes.append(len(cols))
            fusion = r.get("fusion") or "lsml"
            kov = int(r["K"]) if (k_rule and r.get("K") and int(r["K"]) > 0) else None
            vals = [fuse_auroc(cell, cols, fusion, kov,
                               np.random.default_rng([7, s, n]))
                    for s in range(N_SEEDS)]
            for s, v in enumerate(vals):
                macros[s].append(v)
            v = np.array(vals, float)
            per_cell.append(float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any()
                            else float("nan"))

        pc = np.array(per_cell, float)
        mac = np.array([np.nanmean(m) for m in macros], float)
        out_rows.append({
            "variant": variant, "pool": pool, "cells": len(rows),
            "size_mean": round(float(np.mean(sizes)), 2),
            "frac_size3": round(float(np.mean(np.array(sizes) == 3)), 3),
            "frac_size4": round(float(np.mean(np.array(sizes) == 4)), 3),
            "spread_pp": round(float(np.nanmean(pc)) * 100, 4),
            "median_spread_pp": round(float(np.nanmedian(pc)) * 100, 4),
            "max_spread_pp": round(float(np.nanmax(pc)) * 100, 4),
            "frac_unstable": round(float(np.nanmean(pc > UNSTABLE_PP / 100)), 3),
            "macro_spread_pp": round(float(mac.max() - mac.min()) * 100, 4),
            "macro_mean": round(float(mac.mean()), 6),
        })
        if n % 20 == 0:
            print(f"  {n}/{len(by)}  ({time.time()-t0:.0f}s)")

    out_rows.sort(key=lambda r: -r["macro_spread_pp"])
    S.save_csv(os.path.join(OUT, "stability_audit.csv"), out_rows)

    def band(lo, hi):
        return [r for r in out_rows if lo <= r["macro_spread_pp"] < hi]

    summary = {
        "rel_jitter": REL_JITTER, "n_seeds": N_SEEDS, "n_rows": len(out_rows),
        "macro_spread_pp_median": float(np.median([r["macro_spread_pp"] for r in out_rows])),
        "rows_macro_stable_under_0.1pp": len(band(0, 0.1)),
        "rows_macro_0.1_to_0.5pp": len(band(0.1, 0.5)),
        "rows_macro_over_0.5pp": len(band(0.5, 1e9)),
        "worst": [f"{r['variant']} [{r['pool']}] macro moves {r['macro_spread_pp']:.2f}pp "
                  f"(mean cell {r['spread_pp']:.2f}pp, worst cell "
                  f"{r['max_spread_pp']:.2f}pp, mean size {r['size_mean']})"
                  for r in out_rows[:15]],
        "elapsed_s": round(time.time() - t0, 1),
    }
    S.save_json(os.path.join(OUT, "stability_summary.json"), summary)

    print(f"\nmacro spread under a {REL_JITTER:.0e} relative jitter:")
    print(f"  < 0.1pp : {summary['rows_macro_stable_under_0.1pp']}")
    print(f"  0.1-0.5 : {summary['rows_macro_0.1_to_0.5pp']}")
    print(f"  > 0.5pp : {summary['rows_macro_over_0.5pp']}")
    print("\nleast determined rows:")
    for line in summary["worst"]:
        print("   " + line)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
