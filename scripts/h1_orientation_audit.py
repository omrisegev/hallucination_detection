#!/usr/bin/env python
"""
h1_orientation_audit.py — H1 (Extension H): can orientation be made prior-free?

Extension H proposed replacing two hand-picked orientation priors:
  (i)  ALL_SIGNS  — 42 hand-derived per-feature polarities (subset_sweep.py)
  (ii) epr anchor — one hand-picked feature fixing the fused score's global ±1
with structure-only estimators (Z2 synchronization for relative signs; a
distributional tiebreaker — "hallucination is the minority class" — for the
global bit).

This script measures whether either replacement is needed or possible. It runs
three tests on the 25 in-scope cells:

  TEST 1 (gauge)   Is lsml_continuous invariant to input column signs?
                   For each cell, fuse under ALL_SIGNS, raw, Z2, and 20 random
                   sign vectors; compare to the ALL_SIGNS fusion up to a global
                   flip. Algebraically this MUST hold — flipping columns is
                   X -> XD with D = diag(±1), so cov(XD) = D cov(X) D has
                   eigenvector Dv and (XD)(Dv) = X D^2 v = Xv — and
                   detect_dependent_groups scores pairs on |correlation|. The
                   test confirms the implementation honours the algebra.

  TEST 2 (bit)     Conditions holding fusion FIXED and varying only orientation:
                     A_allsigns_anchor   ALL_SIGNS + anchor(epr)   status quo
                     B_z2_anchor         Z2 signs   + anchor(epr)
                     C_z2_skew           Z2 signs   + skew rule     0 priors
                     D_raw_anchor        raw signs  + anchor(epr)
                     E_z2_oracle         Z2 signs   + ORACLE bit    ceiling
                   E − A is the headroom in the one global bit; C is the
                   prior-free candidate Extension H named.

  TEST 3 (premise) Per-cell positive rate, which is the premise the skew /
                   class-imbalance tiebreaker rests on.

Labels are used ONLY for scoring and for the E ceiling — never inside a rule.

Usage:  python scripts/h1_orientation_audit.py
Writes: results/advisor_inscope/h1_orientation_audit.csv
        results/advisor_inscope/h1_orientation_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew, wilcoxon
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.fusion_utils import lsml_continuous, zscore   # noqa: E402
from spectral_utils.streaming_utils import anchor_orient          # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS, GOOD_6         # noqa: E402
# The SHIPPED H1 estimators are what this audit must test — do not re-implement
# them here, or the audit stops describing the code that would deploy.
from spectral_utils.orientation import (                          # noqa: E402
    z2_sign_recovery, distributional_orient)
from inscope_cells import GROUP                                   # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")
N_RANDOM_SIGNS = 20
SEED = 20260725

CONDITIONS = ["A_allsigns_anchor", "B_z2_anchor", "C_z2_skew",
              "D_raw_anchor", "E_z2_oracle"]


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def _fuse(Vsub):
    fused, meta = lsml_continuous(*[Vsub[:, j] for j in range(Vsub.shape[1])])
    return zscore(np.asarray(fused, dtype=float)), meta


def _equal_up_to_flip(a, b, tol=1e-9):
    d = min(float(np.max(np.abs(a - b))), float(np.max(np.abs(a + b))))
    return d < tol, d


# ---------------------------------------------------------------------------

def load_cells():
    from compare_anchor_quality import load_all_inscope_cells
    return load_all_inscope_cells()


def run_cell(ck, cd, cols, rng):
    V = np.asarray(cd["V"], dtype=float)
    y = np.asarray(cd["labels"], dtype=int)
    pool = list(cd["unlabeled"].pool)
    anchor = np.asarray(cd["anchor"], dtype=float)

    sp = np.array([ALL_SIGNS.get(pool[j], 1) for j in cols], dtype=float)
    V_signed = V[:, cols]              # prepare_cell already applied ALL_SIGNS
    V_raw = V_signed * sp              # undo it -> raw, unsigned
    s_z2 = z2_sign_recovery(V_raw)
    V_z2 = V_raw * s_z2[None, :]

    f_all, meta = _fuse(V_signed)
    f_z2, _ = _fuse(V_z2)
    f_raw, _ = _fuse(V_raw)

    # TEST 1 — gauge invariance under arbitrary input sign vectors
    n_ok, n_tot, worst = 0, 0, 0.0
    trials = [np.ones(len(cols)), sp, s_z2]
    trials += [rng.choice([-1.0, 1.0], size=len(cols))
               for _ in range(N_RANDOM_SIGNS)]
    for s in trials:
        f_s, _ = _fuse(V_raw * s[None, :])
        ok, dev = _equal_up_to_flip(f_all, f_s)
        n_ok += int(ok)
        n_tot += 1
        worst = max(worst, dev)

    # TEST 2 — orientation conditions, fusion held fixed
    def auc(s):
        return float(roc_auc_score(y, s))

    a_all, _ = anchor_orient(f_all, anchor)
    b_z2, _ = anchor_orient(f_z2, anchor)
    d_raw, _ = anchor_orient(f_raw, anchor)
    c_skew, _ = distributional_orient(f_z2)
    auc_z2 = auc(f_z2)

    # how often does the label-free ALL_SIGNS polarity match the oracle one?
    aucs = np.array([roc_auc_score(y, V_signed[:, j])
                     for j in range(V_signed.shape[1])])

    return {
        "cell": ck, "group": GROUP.get(ck, "?"), "n": int(len(y)),
        "p_used": int(len(cols)), "pos_rate": round(float(y.mean()), 4),
        "gauge_ok": int(n_ok), "gauge_trials": int(n_tot),
        "gauge_worst_dev": worst,
        "allsigns_frac_wrong": round(float((aucs < 0.5).mean()), 4),
        "skew_fused": round(float(skew(f_z2)), 4),
        "lsml_K": int(meta.get("K", 0)),
        "A_allsigns_anchor": round(auc(a_all), 4),
        "B_z2_anchor": round(auc(b_z2), 4),
        "C_z2_skew": round(auc(c_skew), 4),
        "D_raw_anchor": round(auc(d_raw), 4),
        "E_z2_oracle": round(max(auc_z2, 1.0 - auc_z2), 4),
    }


def summarize(df, fset):
    out = []
    A = df["A_allsigns_anchor"].to_numpy()
    for c in CONDITIONS:
        v = df[c].to_numpy()
        qa = df.loc[df["group"] == "QA", c].to_numpy()
        mt = df.loc[df["group"] == "math", c].to_numpy()
        d = v - A
        try:
            p = float(wilcoxon(v, A).pvalue) if np.any(d != 0) else float("nan")
        except Exception:
            p = float("nan")
        out.append({
            "fset": fset, "condition": c, "n_cells": int(len(v)),
            "macro_all": round(float(v.mean()), 4),
            "macro_qa": round(float(qa.mean()), 4) if len(qa) else None,
            "macro_math": round(float(mt.mean()), 4) if len(mt) else None,
            "cells_below_0.5": int((v < 0.5).sum()),
            "delta_vs_A": round(float(d.mean()), 4),
            "wins_vs_A": int((d > 0).sum()), "losses_vs_A": int((d < 0).sum()),
            "wilcoxon_p_vs_A": round(p, 5) if np.isfinite(p) else None,
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = load_cells()
    rng = np.random.default_rng(SEED)

    fsets = {
        "GOOD_6": lambda pool: [pool.index(f) for f in GOOD_6 if f in pool],
        "FULL": lambda pool: list(range(len(pool))),
    }

    rows, summary = [], []
    for fset, pick in fsets.items():
        print(f"\n=== {fset} ===", flush=True)
        frows = []
        for ck, cd in sorted(cells.items()):
            cols = pick(list(cd["unlabeled"].pool))
            if len(cols) < 3:
                print(f"  {ck}: only {len(cols)} usable cols — skipped")
                continue
            r = run_cell(ck, cd, cols, rng)
            r["fset"] = fset
            frows.append(r)
            print(f"  {ck:34s} A {r['A_allsigns_anchor']:.3f}  "
                  f"C_skew {r['C_z2_skew']:.3f}  E_oracle {r['E_z2_oracle']:.3f}"
                  f"  gauge {r['gauge_ok']}/{r['gauge_trials']}", flush=True)
        df = pd.DataFrame(frows)
        rows.extend(frows)
        summary.extend(summarize(df, fset))

    df_all = pd.DataFrame(rows)
    df_sum = pd.DataFrame(summary)
    p1 = os.path.join(OUT_DIR, "h1_orientation_audit.csv")
    p2 = os.path.join(OUT_DIR, "h1_orientation_summary.csv")
    df_all.to_csv(p1, index=False)
    df_sum.to_csv(p2, index=False)

    print("\n" + "=" * 78)
    print("TEST 1 — gauge invariance of lsml_continuous to input column signs")
    print("=" * 78)
    ok, tot = int(df_all["gauge_ok"].sum()), int(df_all["gauge_trials"].sum())
    print(f"  {ok}/{tot} sign vectors reproduce the fused score up to a global "
          f"flip (worst deviation {df_all['gauge_worst_dev'].max():.3e})")
    print("  => per-feature input signs are a GAUGE SYMMETRY of L-SML."
          if ok == tot else "  => NOT invariant — investigate.")

    print("\n" + "=" * 78)
    print("TEST 2 — orientation conditions (fusion held fixed)")
    print("=" * 78)
    print(df_sum.to_string(index=False))

    print("\n" + "=" * 78)
    print("TEST 3 — premise of the class-imbalance / skew tiebreaker")
    print("=" * 78)
    pr = df_all[df_all["fset"] == "FULL"]["pos_rate"]
    print(f"  pos_rate: min {pr.min():.3f}  median {pr.median():.3f}  "
          f"max {pr.max():.3f}")
    print(f"  cells with pos_rate > 0.5: {int((pr > 0.5).sum())}/{len(pr)}")
    print(f"  median ALL_SIGNS wrong-fraction: "
          f"{df_all[df_all['fset']=='FULL']['allsigns_frac_wrong'].median():.3f}")
    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
