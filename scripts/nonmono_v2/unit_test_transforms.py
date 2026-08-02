#!/usr/bin/env python
"""
unit_test_transforms.py — THE UNIT TEST. One feature, one transform, one AUROC.

THE POINT, AND WHY THIS IS THE RIGHT FIRST EXPERIMENT
-----------------------------------------------------
AUROC depends only on the RANKING of a score, so it is invariant under any
monotone transform. Therefore a feature's oriented single-view AUROC IS the
ceiling over every monotone reading of it, and:

    any transform that raises single-view AUROC is necessarily non-monotone.

That makes this a clean one-number test with no fusion, no selector, no gates.
If a transform cannot beat the raw feature's own AUROC, it is useless and the
ensemble question never arises. If it can, we know exactly how much is on the
table before spending anything on the ensemble.

WHAT IS MEASURED, PER (cell, feature)
-------------------------------------
  auc_mono      cross-fitted AUROC with the sign chosen on TRAIN.
                The monotone ceiling. Every candidate is scored against this.
  auc_ceiling   cross-fitted unconstrained bin map = what a transform fitted on
                THIS cell's labels achieves. The empirical target.
  auc_<T>       each candidate transform, same folds, sign chosen on train.

Everything is cross-fitted on identical folds and the orientation is resolved on
TRAIN, never on test — otherwise `max(a, 1-a)` silently rewards noise (the defect
that inflated Step 214's numbers by up to 12.6pp on a 6-positive cell).

THE CANDIDATES
--------------
  identity      baseline (= auc_mono by construction; printed as a check)
  dist_median   |x - median|          Gemini's, symmetric, median-centred
  squared       x^2                   symmetric, mean-centred (x is z-scored)
  abs_rank      |Phi^-1(u)|           symmetric, median-centred in rank space
  mode_centre   |u - r0|, r0 = KDE mode percentile        LABEL-FREE per cell
  best_centre   |u - r0*|, r0* swept on a grid            LABEL-FITTED -> the
                best this FAMILY can do, and where the centre should sit. Not
                deployable; it is the diagnostic that separates "wrong family"
                from "right family, wrong centre".
  loco_centre   |u - r0|, r0 fitted on the OTHER cells    deployable
  loco_binmap   bin map of u fitted on the OTHER cells    deployable, and the
                strongest form of the idea

`u` is the within-cell percentile rank, so every transform is scale-free and the
LOCO variants are comparable across cells.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (OUT, GROUP, INSCOPE, load_cells_cached, pct,        # noqa: E402
                    kde_modes, n_min, bin_fit_apply, MIN_NMIN)

SHAPE_TEST = os.path.join(os.path.dirname(OUT), "nonmono_transform", "shape_test.csv")
RECURRENT = ["rpdi", "pe_mean", "cusum_shift_idx", "hurst_exponent"]
CGRID = np.round(np.arange(0.05, 0.96, 0.05), 2)     # centre grid for |u - r0|
N_FOLDS, SEEDS = 5, (0, 1, 2)


def folds(y, seed):
    return list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=42 + seed).split(np.zeros(len(y)), y))


def cv_auc(v, y, seed_list=SEEDS):
    """Cross-fitted AUROC with the SIGN CHOSEN ON TRAIN.

    Not `max(a, 1-a)` on the full sample: that resolves the sign using the test
    labels, which can only inflate and fires hardest on noise."""
    v = np.asarray(v, float)
    y = np.asarray(y, int)
    if np.std(v) < 1e-12:
        return 0.5
    out = []
    for s in seed_list:
        per = []
        for tr, te in folds(y, s):
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            if np.std(v[tr]) < 1e-12 or np.std(v[te]) < 1e-12:
                continue
            sgn = 1.0 if roc_auc_score(y[tr], v[tr]) >= 0.5 else -1.0
            per.append(roc_auc_score(y[te], sgn * v[te]))
        if per:
            out.append(float(np.mean(per)))
    return float(np.mean(out)) if out else float("nan")


def cv_binmap_auc(x, y, k=10, seed_list=SEEDS):
    """Cross-fitted unconstrained bin map — the label-fitted reshaping ceiling."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    out = []
    for s in seed_list:
        per = []
        for tr, te in folds(y, s):
            if len(np.unique(y[te])) < 2:
                continue
            u, _ = bin_fit_apply(x[tr], y[tr], x[te], k)
            if u is None or np.std(u) < 1e-12:
                continue
            per.append(roc_auc_score(y[te], u))
        if per:
            out.append(float(np.mean(per)))
    return float(np.mean(out)) if out else float("nan")


# ── the transform family ──────────────────────────────────────────────────────
def t_identity(x, u, **kw):
    return x


def t_dist_median(x, u, **kw):
    return np.abs(x - np.median(x))


def t_squared(x, u, **kw):
    return x ** 2


def t_abs_rank(x, u, **kw):
    return np.abs(norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)))


def t_centre(x, u, r0=0.5, **kw):
    return np.abs(u - r0)


def centre_curve(u, y, grid=CGRID):
    """Cross-fitted AUROC of |u - r0| at every r0 on the grid."""
    return np.array([cv_auc(np.abs(u - r0), y) for r0 in grid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cells = load_cells_cached(use_cache=not args.no_cache)

    flagged = set()
    if os.path.exists(SHAPE_TEST):
        with open(SHAPE_TEST, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["exceeds_null"] == "1":
                    flagged.add((r["cell"], r["feature"]))

    # candidate set: every flagged pair with usable power, plus the four recurrent
    # views everywhere they are live (LOCO needs several cells per feature to pool).
    pairs = []
    for ck in INSCOPE:
        cell = cells[ck]
        if n_min(cell["labels"]) < MIN_NMIN:
            continue
        for f in cell["pool"]:
            if (ck, f) in flagged or f in RECURRENT:
                pairs.append((ck, f))
    print(f"{len(pairs)} (cell, feature) pairs under test "
          f"[{len(flagged)} flagged; {len(RECURRENT)} recurrent views everywhere]\n")

    # ── pass 1: per-pair curves ──────────────────────────────────────────────
    store = {}
    for ck, f in pairs:
        cell = cells[ck]
        j = cell["pool"].index(f)
        x, y = cell["V"][:, j], cell["labels"]
        u = pct(x)
        store[(ck, f)] = dict(x=x, y=y, u=u, curve=centre_curve(u, y))
        print(f"  swept {ck:<32}{f}", flush=True)

    # ── pass 2: LOCO fits, then score every candidate ────────────────────────
    rows = []
    for (ck, f), d in store.items():
        x, y, u, curve = d["x"], d["y"], d["u"], d["curve"]
        others = [(c2, f2) for (c2, f2) in store if f2 == f and c2 != ck]

        auc_mono = cv_auc(x, y)
        auc_ceil = cv_binmap_auc(x, y)
        _, _, mode_pct = kde_modes(x)

        cand = {
            "identity": t_identity(x, u),
            "dist_median": t_dist_median(x, u),
            "squared": t_squared(x, u),
            "abs_rank": t_abs_rank(x, u),
        }
        if np.isfinite(mode_pct):
            cand["mode_centre"] = t_centre(x, u, r0=float(mode_pct))

        # LABEL-FITTED family ceiling: best r0 on this cell (not deployable)
        best_i = int(np.nanargmax(curve))
        auc_best_centre, r0_best = float(curve[best_i]), float(CGRID[best_i])

        # LOCO centre: argmax of the MEAN centre-curve over the other cells
        r0_loco = float("nan")
        if others:
            m = np.nanmean(np.vstack([store[o]["curve"] for o in others]), axis=0)
            r0_loco = float(CGRID[int(np.nanargmax(m))])
            cand["loco_centre"] = t_centre(x, u, r0=r0_loco)

        # LOCO bin map: pooled (percentile, label) from the other cells
        if others:
            uu = np.concatenate([store[o]["u"] for o in others])
            yy = np.concatenate([store[o]["y"] for o in others])
            mp, _ = bin_fit_apply(uu, yy, u, 10)
            if mp is not None:
                cand["loco_binmap"] = mp

        rec = dict(cell=ck, domain=GROUP.get(ck, "?"), feature=f,
                   flagged=int((ck, f) in flagged), n=len(y), n_min=n_min(y),
                   auc_mono=round(auc_mono, 4), auc_ceiling=round(auc_ceil, 4),
                   headroom_pp=round((auc_ceil - auc_mono) * 100, 2),
                   kde_mode_pct=(round(float(mode_pct), 3)
                                 if np.isfinite(mode_pct) else ""),
                   r0_best=r0_best, auc_best_centre=round(auc_best_centre, 4),
                   r0_loco=(r0_loco if np.isfinite(r0_loco) else ""),
                   n_loco_cells=len(others))
        for name, v in cand.items():
            a = cv_auc(v, y)
            rec[f"auc_{name}"] = round(a, 4)
            rec[f"d_{name}_pp"] = round((a - auc_mono) * 100, 2)
        rows.append(rec)

    # ── report ───────────────────────────────────────────────────────────────
    cols = ["dist_median", "squared", "abs_rank", "mode_centre",
            "loco_centre", "loco_binmap", "best_centre"]
    rows.sort(key=lambda r: -r["headroom_pp"])

    print(f"\n{'='*136}")
    print("SINGLE-VIEW AUROC. auc_mono = the monotone ceiling (any monotone "
          "transform gives exactly this). Deltas are vs auc_mono, in points.")
    print(f"{'='*136}")
    hdr = (f"{'cell':<30}{'feature':<18}{'n_min':>6}{'mono':>7}{'ceil':>7}{'hd':>6}"
           + "".join(f"{c[:9]:>10}" for c in cols))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['cell'][:29]:<30}{r['feature'][:17]:<18}{r['n_min']:>6}"
                f"{r['auc_mono']:>7.3f}{r['auc_ceiling']:>7.3f}{r['headroom_pp']:>6.1f}")
        for c in cols:
            k = f"d_{c}_pp"
            line += f"{r[k]:>+10.1f}" if k in r else f"{'-':>10}"
        print(line)

    print(f"\n{'='*136}\nSUMMARY — mean delta vs the monotone ceiling, in points"
          f"\n{'='*136}")
    fl = [r for r in rows if r["flagged"]]
    for name, grp in (("all pairs", rows), ("flagged only", fl)):
        print(f"\n  {name}  (n={len(grp)})")
        print(f"    {'ceiling (label-fitted binmap)':<34}"
              f"{np.mean([r['headroom_pp'] for r in grp]):>+8.2f}pp"
              f"   best {max(r['headroom_pp'] for r in grp):>+6.1f}pp")
        for c in cols:
            k = f"d_{c}_pp"
            v = [r[k] for r in grp if k in r]
            if not v:
                continue
            wins = sum(1 for q in v if q > 0)
            print(f"    {c:<34}{np.mean(v):>+8.2f}pp   best {max(v):>+6.1f}pp"
                  f"   improves {wins}/{len(v)}")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "unit_test_transforms.csv")
    keys = sorted({k for r in rows for k in r})
    lead = ["cell", "domain", "feature", "flagged", "n", "n_min", "auc_mono",
            "auc_ceiling", "headroom_pp"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=lead + [k for k in keys if k not in lead])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {p}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
