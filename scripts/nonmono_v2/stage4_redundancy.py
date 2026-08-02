#!/usr/bin/env python
"""
stage4_redundancy.py — is the recovered signal NEW, or does the fusion already have it?

WHY THIS IS THE DECISIVE MEASUREMENT
------------------------------------
The unit test established that folding a view recovers ~73% of its non-monotone
headroom: `sciq_llama8b/pe_mean` goes 0.434 -> 0.699 on its own. If the fused
score then does not move, there are exactly two explanations, and they imply
OPPOSITE next steps:

  REDUNDANCY   the other ~29 views already carry that information, so a repaired
               view adds nothing conditional on them. Then no per-view reshaping
               can ever help, the pool is the binding constraint, and the effort
               belongs in better views or a non-linear fusion.

  MECHANISM    the information is genuinely new, but the pipeline cannot route it
               -- U-PCR excludes it, DUFS does not select it, or the weight it
               gets is too small. Then the transform line is alive and the work
               is in the selector, not the transform.

A marginal AUROC gain cannot tell those apart, because it conditions on nothing.
This does, by conditioning on the fusion itself:

    s_-j   = the DEPLOYED U-PCR fused score computed on the pool WITHOUT view j
             (not a hand-rolled average -- the actual thing that would be shipped)
    A_alone = cross-fitted AUROC of s_-j
    A_mono  = cross-fitted AUROC of a logistic on [iso(s_-j), iso_dir(x_j)]
    A_free  = cross-fitted AUROC of a logistic on [iso(s_-j), binmap_K(x_j)]

`iso` on both arms puts the fusion and the monotone reading of the view on the
same footing, so the ONLY difference between A_mono and A_free is whether view j
is allowed a non-monotone reading. Therefore

    conditional_shape_gain = A_free - A_mono

is the value of repairing view j GIVEN everything else the fusion already knows,
and

    marginal_shape_gain = headroom (the unit-test quantity)

is its value in isolation. Marginal +12pp with conditional ~0 IS the proof of
redundancy; conditional ~= marginal is the proof that the loss is mechanical.

Everything is cross-fitted on the same folds with the sign resolved on TRAIN, so
no test label is used to orient anything.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (OUT, GROUP, INSCOPE, load_cells_cached, iso_fit,     # noqa: E402
                    bin_fit_apply, upcr_score, n_min)
from unit_test_transforms import cv_auc, folds, SEEDS                    # noqa: E402

SEL_JSON = os.path.join(OUT, "transform_selection.json")
KBIN = 10


def _stack_auc(y, cols_tr, cols_te, ytr, yte):
    """Logistic on the stacked pair, fitted on train, AUROC on test.

    `class_weight='balanced'` per SUPERVISED_ORACLE_CORRECTION.md -- several of
    these cells are heavily imbalanced and an unweighted fit collapses toward the
    majority class, which would understate BOTH arms but not equally."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return np.nan
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    try:
        lr.fit(cols_tr, ytr)
        return float(roc_auc_score(yte, lr.predict_proba(cols_te)[:, 1]))
    except Exception:                                # noqa: BLE001
        return np.nan


def conditional_gain(x, s, y, k=KBIN, seed_list=SEEDS):
    """(A_alone, A_mono, A_free) — all cross-fitted on identical folds."""
    x, s, y = np.asarray(x, float), np.asarray(s, float), np.asarray(y, int)
    alone, mono, free = [], [], []
    for sd in seed_list:
        pa, pm, pf = [], [], []
        for tr, te in folds(y, sd):
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            sgn = 1.0 if roc_auc_score(y[tr], s[tr]) >= 0.5 else -1.0
            pa.append(roc_auc_score(y[te], sgn * s[te]))
            i_s = iso_fit(s[tr], y[tr])
            i_x = iso_fit(x[tr], y[tr])
            if i_s is None or i_x is None:
                continue
            fs_tr, fs_te = i_s.predict(s[tr]), i_s.predict(s[te])
            fx_tr, fx_te = i_x.predict(x[tr]), i_x.predict(x[te])
            bx_te, _ = bin_fit_apply(x[tr], y[tr], x[te], k)
            bx_tr, _ = bin_fit_apply(x[tr], y[tr], x[tr], k)
            if bx_te is None or bx_tr is None:
                continue
            pm.append(_stack_auc(y, np.c_[fs_tr, fx_tr], np.c_[fs_te, fx_te],
                                 y[tr], y[te]))
            pf.append(_stack_auc(y, np.c_[fs_tr, bx_tr], np.c_[fs_te, bx_te],
                                 y[tr], y[te]))
        for acc, per in ((alone, pa), (mono, pm), (free, pf)):
            if per and np.isfinite(per).any():
                acc.append(float(np.nanmean(per)))
    f = lambda v: float(np.mean(v)) if v else np.nan          # noqa: E731
    return f(alone), f(mono), f(free)


def run_cell(payload):
    ck, cell, feats = payload
    V, pool, y, anchor = cell["V"], cell["pool"], cell["labels"], cell["anchor"]
    out = []
    for f, headroom in feats:
        if f not in pool:
            continue
        j = pool.index(f)
        keep = [i for i in range(V.shape[1]) if i != j]
        if len(keep) < 3:
            continue
        s, _, _ = upcr_score(V[:, keep], [pool[i] for i in keep], anchor)
        a_alone, a_mono, a_free = conditional_gain(V[:, j], s, y)
        out.append(dict(
            cell=ck, domain=GROUP.get(ck, "?"), feature=f, n_min=n_min(y),
            headroom_pp=round(float(headroom), 2),
            sv_auc=round(cv_auc(V[:, j], y), 4),
            a_alone=round(a_alone, 4), a_mono=round(a_mono, 4),
            a_free=round(a_free, 4),
            cond_view_gain_pp=round(100 * (a_mono - a_alone), 3),
            cond_shape_gain_pp=round(100 * (a_free - a_mono), 3),
            absorbed_frac=(round(1 - (a_free - a_mono) * 100 / headroom, 3)
                           if headroom > 0 else float("nan"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-headroom", type=float, default=3.0)
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()

    cells = load_cells_cached()
    with open(SEL_JSON, encoding="utf-8") as fh:
        panels = json.load(fh)["panels"]
    by_cell = {}
    for p in panels:
        if p["is_candidate"]:
            by_cell.setdefault(p["cell"], []).append((p["feature"], p["headroom_pp"]))
    keys = args.cells or list(INSCOPE)
    payloads = [(ck, cells[ck], by_cell[ck]) for ck in keys if ck in by_cell]
    print(f"{sum(len(v) for v in by_cell.values())} candidate views "
          f"on {len(payloads)} cells\n")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_cell, payloads):
            rows += r
            if r:
                print(f"  done {r[0]['cell']}", flush=True)

    rows.sort(key=lambda r: -r["headroom_pp"])
    with open(os.path.join(OUT, "redundancy.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*106}")
    print("R2 — CONDITIONAL SHAPE GAIN.  What is repairing view j worth GIVEN the "
          "fusion of the other views?")
    print(f"{'='*106}")
    print(f"{'cell':<30}{'feature':<18}{'marg':>7}{'fusion':>8}{'+mono':>7}"
          f"{'+free':>7}{'cond':>7}{'absorbed':>10}")
    print("-" * 106)
    for r in rows:
        print(f"{r['cell'][:29]:<30}{r['feature'][:17]:<18}"
              f"{r['headroom_pp']:>+7.1f}{r['a_alone']:>8.3f}"
              f"{r['cond_view_gain_pp']:>+7.2f}"
              f"{100*(r['a_free']-r['a_alone']):>+7.2f}"
              f"{r['cond_shape_gain_pp']:>+7.2f}"
              f"{r['absorbed_frac']:>10.2f}")

    marg = np.array([r["headroom_pp"] for r in rows], float)
    cond = np.array([r["cond_shape_gain_pp"] for r in rows], float)
    m = np.isfinite(marg) & np.isfinite(cond)
    print("-" * 106)
    print(f"  mean marginal shape gain (isolation)  {marg[m].mean():+.2f}pp")
    print(f"  mean conditional shape gain (in fusion){cond[m].mean():+7.2f}pp")
    print(f"  -> absorbed by the other views        "
          f"{100*(1 - cond[m].mean()/marg[m].mean()):.0f}%")
    print(f"  views with conditional gain > +1pp    "
          f"{int((cond[m] > 1).sum())}/{int(m.sum())}")
    for g in ("QA", "math"):
        sub = [r for r in rows if r["domain"] == g and np.isfinite(r["cond_shape_gain_pp"])]
        if sub:
            print(f"    {g:<5} marginal {np.mean([r['headroom_pp'] for r in sub]):+.2f}pp"
                  f"  -> conditional {np.mean([r['cond_shape_gain_pp'] for r in sub]):+.2f}pp")
    print(f"\nwrote {os.path.join(OUT, 'redundancy.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
