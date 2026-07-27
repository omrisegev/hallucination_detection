"""
reproduction_audit.py — does every published number still come out of today's code?

WHY THIS EXISTS
---------------
Before the comparison page goes to advisors, every row on it has to be a number
today's checkout actually produces. It was not: `a1.relres_greedy` is published at
0.6952 and re-scores to 0.6899 — a 0.53pp drift with per-cell differences up to
3.8pp. Bisecting fusion_utils located it at commit a7e8741 (Step 203), whose
`_score_matrix_lsml` vectorisation is documented as "identical output to the
original quadruple loop". It is not, in one specific place:

  * At m = 3 the Eq.15 sum   s_ij = SUM_{k != i,j} SUM_{l != i,j,k} |...|
    is EMPTY — with three features, k is forced to the third index and then no l
    remains. The old quadruple loop returned exactly 0.0. The vectorised form
    computes it as a difference of large partial sums and returns 2.8e-17.
  * Spectral clustering of an all-zero similarity is arbitrary either way, so that
    float noise flips the group assignment ([0,0,1] -> [1,0,1] on the cell below),
    which changes the residual (0.00297 -> 0.05284) and the fused score.

So at THREE features the L-SML group assignment is not determined by the data at
all. That is not a regression to revert — both answers are equally meaningless —
but it does mean every size-3 bench row is noise-valued, and size-3 is exactly
where the residual-guided selectors converge (a1.relres_greedy: 46/51 cells;
a3.cae_k3: 51/51).

WHAT THIS SCRIPT DOES
---------------------
Replays every selector-bench row through today's fusion code on the 25 in-scope
cells and reports, per (variant, pool):

  macro_published  what the bench CSV / scoreboard says
  macro_today      the same stored subsets re-scored now
  drift            today - published, and the worst single cell
  frac_size3       fraction of cells where the subset has 3 features, i.e. where
                   the group assignment is decided by float noise
  macro_eigen / macro_complete
                   the same subsets fused at the other two loading scales, so the
                   scale sensitivity of every row is a measured number rather than
                   a caveat

Replay fidelity mirrors `selector_bench.eval_subset_flex`:
  * `fusion=upcr` rows go through upcr_fuse (a1's router arms),
  * `+K_` variants are K-overridden by construction, so the stored K is re-applied
    with the same clamp,
  * a2's two `groups`-override variants cannot be replayed (the assignment is not
    stored in the CSV) and are reported as NOT REPLAYABLE rather than guessed.

Out: results/upcr_study/reproduction_audit.csv  (+ summary.json)
"""
import os
import sys
import csv
import glob
import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                   # noqa: E402

from spectral_utils.fusion_utils import lsml_continuous, upcr_fuse   # noqa: E402
from spectral_utils.streaming_utils import anchor_orient             # noqa: E402

OUT = S.outdir("00_reproduction_audit")

# a2's clustering-swap arms pass an explicit group assignment that the CSV does
# not store; replaying them with the default search would be a different method.
GROUPS_OVERRIDE = {"a2.select+groups", "a2.groups@good5"}

# Variants whose selector emits an explicit K (grep `'K':` in spectral_utils/
# selectors/). For these the forced K is part of the METHOD, so the replay must
# re-apply the stored K; letting the default search pick K instead measures a
# different algorithm. Most are named `*+K_*`, but a4.intrinsic_k_ah is not —
# matching on the name alone left it mis-replayed and showed up as the audit's
# only "unexplained" drift.
K_OVERRIDE_VARIANTS = {"a4.intrinsic_k_ah"}


def is_k_override(variant):
    return "+K_" in variant or variant in K_OVERRIDE_VARIANTS


def replay(cell, cols, fusion, k_override, scale):
    """One subset scored the way selector_bench.eval_subset_flex scores it."""
    V = cell["V"]
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return float("nan")
    if fusion == "upcr":
        if scale != "unit":            # U-PCR has no L-SML loading scale
            return float("nan")
        w, _, _ = upcr_fuse(V[:, cols].T)
        fused = V[:, cols] @ w
    else:
        kw = {}
        if k_override is not None:
            m = len(cols)
            kw["K_range"] = [max(2, min(int(k_override), max(2, min(m - 1, 8))))]
        fused, _ = lsml_continuous(*[V[:, j] for j in cols],
                                   method="residual", loading_scale=scale, **kw)
    oriented, _ = anchor_orient(np.asarray(fused, float), cell["anchor"])
    if np.std(oriented) < 1e-12:
        return float("nan")
    return float(roc_auc_score(cell["labels"], oriented))


def load_rows(inscope):
    """All bench rows for in-scope cells, keyed by (variant, pool_mode)."""
    by = {}
    src = {}
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
        fam = b.split("__")[0]
        mtime = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f)))
        for r in rows:
            if r.get("cell") not in inscope or not r.get("chosen"):
                continue
            key = (r["variant"], r.get("pool_mode", ""))
            by.setdefault(key, []).append(r)
            src[key] = (fam, mtime)
    return by, src


def main():
    cells = S.load()
    S.validity_check(cells)
    inscope = set(cells)
    by, src = load_rows(inscope)
    print(f"{len(by)} (variant, pool) pairs over {len(inscope)} in-scope cells")

    out_rows = []
    t0 = time.time()
    for n, (key, rows) in enumerate(sorted(by.items()), 1):
        variant, pool = key
        fam, mtime = src[key]
        assert len(rows) <= 25, f"in-scope filter leaked: {key} -> {len(rows)}"

        pub, now, eig, comp, sizes, bad_names = [], [], [], [], [], 0
        replayable = variant not in GROUPS_OVERRIDE
        k_rule = is_k_override(variant)

        for r in rows:
            cell = cells[r["cell"]]
            names = [x for x in r["chosen"].split("|") if x]
            cols = [cell["pool"].index(x) for x in names if x in cell["pool"]]
            if len(cols) != len(names):
                bad_names += 1
            sizes.append(len(cols))
            pub.append(float(r["auroc"]) if r.get("auroc") else float("nan"))
            if not replayable:
                now.append(float("nan")); eig.append(float("nan")); comp.append(float("nan"))
                continue
            fusion = r.get("fusion") or "lsml"
            kov = int(r["K"]) if (k_rule and r.get("K") and int(r["K"]) > 0) else None
            now.append(replay(cell, cols, fusion, kov, "unit"))
            eig.append(replay(cell, cols, fusion, kov, "eigen"))
            comp.append(replay(cell, cols, fusion, kov, "complete"))

        pub, now = np.array(pub, float), np.array(now, float)
        d = now - pub
        fin = np.isfinite(d)
        out_rows.append({
            "variant": variant, "pool": pool, "family": fam, "bench_csv_date": mtime,
            "cells": len(rows),
            "macro_published": round(float(np.nanmean(pub)), 6),
            "macro_today": round(float(np.nanmean(now)), 6) if replayable else None,
            "drift_pp": round(float(np.nanmean(d)) * 100, 4) if replayable else None,
            "max_cell_drift_pp": (round(float(np.nanmax(np.abs(d[fin]))) * 100, 4)
                                  if replayable and fin.any() else None),
            "cells_differing": int((np.abs(d[fin]) > 1e-9).sum()) if replayable else None,
            "reproduces": (None if not replayable else
                           bool(fin.any() and np.nanmax(np.abs(d[fin])) <= 1e-9)),
            "macro_eigen": round(float(np.nanmean(eig)), 6) if replayable else None,
            "macro_complete": round(float(np.nanmean(comp)), 6) if replayable else None,
            "frac_size3": round(float(np.mean(np.array(sizes) == 3)), 3),
            "size_mean": round(float(np.mean(sizes)), 2),
            "replay_note": ("groups override not stored in CSV — NOT REPLAYABLE"
                            if not replayable else
                            ("K-rule variant: stored K re-applied" if k_rule else "")),
            "chosen_names_missing_from_pool": bad_names,
        })
        if n % 20 == 0:
            print(f"  {n}/{len(by)}  ({time.time()-t0:.0f}s)")

    out_rows.sort(key=lambda r: (-(abs(r["drift_pp"]) if r["drift_pp"] is not None else -1)))
    S.save_csv(os.path.join(OUT, "reproduction_audit.csv"), out_rows)

    ok = [r for r in out_rows if r["reproduces"] is True]
    bad = [r for r in out_rows if r["reproduces"] is False]
    na = [r for r in out_rows if r["reproduces"] is None]
    summary = {
        "n_variant_pool_pairs": len(out_rows),
        "reproduces_exactly": len(ok),
        "drifts": len(bad),
        "not_replayable": len(na),
        "worst_drift_pp": bad[0]["drift_pp"] if bad else 0.0,
        "worst_variant": f"{bad[0]['variant']} [{bad[0]['pool']}]" if bad else None,
        "drifting_variants": [f"{r['variant']} [{r['pool']}] "
                              f"{r['macro_published']:.4f} -> {r['macro_today']:.4f} "
                              f"({r['drift_pp']:+.2f}pp)" for r in bad],
        "size3_heavy": [f"{r['variant']} [{r['pool']}] {r['frac_size3']:.0%}"
                        for r in out_rows if r["frac_size3"] >= 0.5],
        "elapsed_s": round(time.time() - t0, 1),
    }
    S.save_json(os.path.join(OUT, "summary.json"), summary)

    print(f"\nreproduces exactly : {len(ok)}")
    print(f"DRIFTS             : {len(bad)}")
    print(f"not replayable     : {len(na)}")
    for r in bad[:25]:
        print(f"   {r['variant']:<26} [{r['pool']:<5}] {r['macro_published']:.4f} -> "
              f"{r['macro_today']:.4f}  ({r['drift_pp']:+.2f}pp, worst cell "
              f"{r['max_cell_drift_pp']:.2f}pp, size3 {r['frac_size3']:.0%}, "
              f"csv {r['bench_csv_date']})")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
