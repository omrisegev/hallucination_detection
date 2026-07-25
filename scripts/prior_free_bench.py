#!/usr/bin/env python
"""
prior_free_bench.py — Phase 5: integrated prior-free L-SML benchmark + decision gate.

WHY THIS WAS REWRITTEN (Step 201, defects 1, 2, 8)
--------------------------------------------------
The Step-200 version compared arms that were not comparable:
  * the GOOD_6 arm went through `lsml_continuous_pipeline` (which z-scores each
    view internally) while the a7 arm fused **raw, un-z-scored** columns — on the
    first cell the column-std ratio is 9.3e+08, so the covariance was dominated by
    a single high-variance column;
  * the anchor fell back to `feat_names[0]` — an arbitrary dict key — with
    `ALL_SIGNS` defaulting to +1, which is how the "anchored" arm produced 2 cells
    *below* 0.5 (min 0.3558) when the canonical path never goes sub-0.5;
  * the pool was `fd.keys()` rather than `CANONICAL_POOL`, so it differed from
    every other script in the repo;
  * `wilcoxon` was imported but never called, so results were reported as
    win/loss counts with no p-value.

This version loads cells through `compare_anchor_quality.load_all_inscope_cells()`
(which runs `prepare_cell` -> z-scored V on `CANONICAL_POOL`, with the cell's own
resolved anchor) and scores every arm through one function that mirrors
`spectral_utils/repgrid_scoring.score_subset`.

ARMS
----
  good6            reference subset, canonical path            [validity anchor]
  a7_anchored      a7.iter_consensus selection + anchor_orient (uses the epr anchor,
                   so it is NOT prior-free — labelled accordingly)
  a7_prior_free    a7.iter_consensus selection + distributional_orient
                   (zero hand-picked input: no seed subset, no anchor, no fixed K)

VALIDITY ANCHOR
---------------
The `good6` arm MUST reproduce macro 0.7594. If it does not, the data being loaded
is not the data the existing numbers came from and every downstream conclusion is
void (SPEC_gap_ladder.md §8). The script asserts this and refuses to report.

Usage:  python scripts/prior_free_bench.py
Writes: results/advisor_inscope/prior_free_bench_results.csv
        results/advisor_inscope/prior_free_bench_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.fusion_utils import lsml_continuous, boot_auc     # noqa: E402
from spectral_utils.streaming_utils import anchor_orient              # noqa: E402
from spectral_utils.subset_sweep import GOOD_6                        # noqa: E402
from spectral_utils.orientation import distributional_orient          # noqa: E402
from spectral_utils.selector_bench import UnlabeledCell               # noqa: E402
from spectral_utils.selectors.a7_iter_consensus import (              # noqa: E402
    a7_iter_consensus)
from inscope_cells import GROUP                                       # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")
RES_CSV = os.path.join(OUT_DIR, "prior_free_bench_results.csv")
SUM_CSV = os.path.join(OUT_DIR, "prior_free_bench_summary.csv")

GOOD6_EXPECTED = 0.7594
GOOD6_TOL = 0.002
SEED = 20260725


def fuse(V, cols):
    """L-SML continuous fusion of the given columns of an already-z-scored V."""
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return None
    fused, _ = lsml_continuous(*[V[:, c] for c in cols])
    return np.asarray(fused, dtype=float)


def score_anchored(V, cols, anchor, y):
    """Canonical scoring: fuse, orient label-free against the cell's anchor, raw
    AUROC + bootstrap CI. Mirrors repgrid_scoring.score_subset."""
    f = fuse(V, cols)
    if f is None:
        return np.nan, np.nan, np.nan
    s, _ = anchor_orient(f, np.asarray(anchor, dtype=float))
    auc, lo, hi = boot_auc(y.astype(int), s)     # NOTE: (labels, scores)
    return float(auc), float(lo), float(hi)


def score_prior_free(V, cols, y):
    """Fully prior-free: same fusion, global sign from the distributional rule."""
    f = fuse(V, cols)
    if f is None:
        return np.nan
    s, _ = distributional_orient(f)
    return float(roc_auc_score(y.astype(int), s))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from compare_anchor_quality import load_all_inscope_cells
    cells = load_all_inscope_cells()

    rows = []
    for ck, cd in sorted(cells.items()):
        u = cd["unlabeled"]
        V = np.asarray(u.V, dtype=np.float64)
        y = np.asarray(cd["labels"], dtype=int)
        pool = list(u.pool)
        anchor = np.asarray(u.anchor, dtype=np.float64)

        g6_cols = [pool.index(f) for f in GOOD_6 if f in pool]
        a_g6, lo_g6, hi_g6 = score_anchored(V, g6_cols, anchor, y)

        # a7 sees only the UnlabeledCell — the real one, not a local shadow class.
        rng = np.random.default_rng(SEED)
        res = a7_iter_consensus(u, rng)[0]
        cols = list(res["cols"])
        a_anch, lo_a, hi_a = score_anchored(V, cols, anchor, y)
        a_pf = score_prior_free(V, cols, y)

        rows.append({
            "cell": ck, "group": GROUP.get(ck, "?"), "n": int(len(y)),
            "p_pool": int(V.shape[1]), "pos_rate": round(float(y.mean()), 4),
            "good6": a_g6, "good6_lo": lo_g6, "good6_hi": hi_g6,
            "a7_anchored": a_anch, "a7_anchored_lo": lo_a, "a7_anchored_hi": hi_a,
            "a7_prior_free": a_pf,
            "k_selected": int(len(cols)),
            "a7_fallback": bool(res.get("fallback", False)),
            "k_star": res.get("diag", {}).get("k_star"),
        })
        print(f"  {ck:34s} good6 {a_g6:.4f} | a7_anch {a_anch:.4f} | "
              f"a7_pf {a_pf:.4f} | k={len(cols)}"
              f"{'  [FALLBACK]' if res.get('fallback') else ''}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES_CSV, index=False)

    # ---- validity anchor ---------------------------------------------------
    macro_g6 = float(df["good6"].mean())
    ok = abs(macro_g6 - GOOD6_EXPECTED) <= GOOD6_TOL
    print(f"\n{'='*78}\nVALIDITY: GOOD_6 macro = {macro_g6:.4f} "
          f"(expected {GOOD6_EXPECTED} +/- {GOOD6_TOL}) -> "
          f"{'PASS' if ok else 'FAIL'}\n{'='*78}")
    if not ok:
        print("FAIL — the loaded data is not the data the canonical numbers came "
              "from. Refusing to report arm comparisons (SPEC_gap_ladder §8).")
        sys.exit(1)

    n_fb = int(df["a7_fallback"].sum())
    print(f"a7 fell back to the full pool on {n_fb}/{len(df)} cells")
    print(f"k_selected distribution: {df['k_selected'].value_counts().to_dict()}")

    # ---- arm comparison ----------------------------------------------------
    print(f"\n{'arm':16s} {'macro':>8s} {'QA':>8s} {'math':>8s} {'<0.5':>5s} "
          f"{'vs GOOD_6':>11s} {'W/L':>8s} {'p':>9s}")
    print("-" * 78)
    summary = []
    base = df["good6"].to_numpy()
    for arm in ("good6", "a7_anchored", "a7_prior_free"):
        v = df[arm].to_numpy()
        qa = df.loc[df["group"] == "QA", arm].to_numpy()
        mt = df.loc[df["group"] == "math", arm].to_numpy()
        d = v - base
        try:
            p = float(wilcoxon(v, base).pvalue) if np.any(d != 0) else float("nan")
        except Exception:
            p = float("nan")
        print(f"{arm:16s} {v.mean():8.4f} {qa.mean():8.4f} {mt.mean():8.4f} "
              f"{int((v<0.5).sum()):5d} {d.mean()*100:+10.2f}pp "
              f"{int((d>0).sum()):3d}/{int((d<0).sum()):<3d} {p:9.4f}")
        summary.append({"arm": arm, "n_cells": len(v),
                        "macro_all": round(float(v.mean()), 4),
                        "macro_qa": round(float(qa.mean()), 4),
                        "macro_math": round(float(mt.mean()), 4),
                        "cells_below_0.5": int((v < 0.5).sum()),
                        "delta_vs_good6": round(float(d.mean()), 4),
                        "wins": int((d > 0).sum()),
                        "losses": int((d < 0).sum()),
                        "wilcoxon_p": round(p, 5) if np.isfinite(p) else None})
    pd.DataFrame(summary).to_csv(SUM_CSV, index=False)

    gate = df["a7_prior_free"].mean() >= GOOD6_EXPECTED
    print(f"\nDECISION GATE (prior-free macro >= GOOD_6 {GOOD6_EXPECTED} with zero "
          f"hand-picked input): {'PASS' if gate else 'FAIL'}")
    print("Note: sub-1pp deltas are not findings — 25 cells resolve ~>=1pp.")
    print(f"\nwrote {RES_CSV}\nwrote {SUM_CSV}")


if __name__ == "__main__":
    main()
