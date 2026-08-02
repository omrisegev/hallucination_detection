#!/usr/bin/env python
"""
transform_selection.py — pick the transform for each candidate, and show the work.

WHAT THIS DECIDES
-----------------
For every candidate (cell, feature): which transform, chosen under a stated rule,
with the evidence needed to justify it visually — the class-conditional densities
of the feature before and after, for EVERY option considered.

WHY CLASS-CONDITIONAL DENSITIES ARE THE RIGHT PICTURE. A linear fusion can only
use a view whose two class densities are SHIFTED versions of each other: that is
what "monotone in P(correct)" means geometrically. A non-monotone view has two
densities that overlap in location but differ in SPREAD — the correct answers
concentrated in the middle, the hallucinations pushed to both tails (or the
reverse). No monotone reading can separate those, which is exactly why AUROC sits
at ~0.5. Folding the axis converts a spread difference into a location
difference, which is what these transforms do and why they work.

THE MENU
--------
  identity        baseline; its AUROC is the monotone ceiling for the view
  squared         x^2                     symmetric, mean-centred (x is z-scored)
  dist_median     |x - median|            symmetric, median-centred
  abs_rank        |Phi^-1(u)|             symmetric, median-centred in rank space
  mode_centre     |u - mode|              label-free centre from the KDE mode
  best_centre     |u - c*|, c* swept      LABEL-FITTED -> the family's own ceiling
                                          and the diagnostic for "wrong centre"
  loco_centre     |u - c|, c from other cells
  hinge           asymmetric two-piece around c: (u-c)+ - a*(c-u)+, a swept
                  -> covers the W / off-centre shapes the symmetric family cannot
  consensus_map   bin map of x against a PSEUDO-LABEL built from the other views
                  -> FULLY LABEL-FREE reshaping; no answer key anywhere
  loco_binmap     bin map of u pooled over the other cells

THE SELECTION PROBLEM IS THE REAL ONE. Applying a fold indiscriminately costs
-5.7pp on views that are already monotone, so "which views" must be decided
without labels. Step 217 showed the marginal two-peak measures cannot do it
(Spearman +0.014 with the KDE peak count). The candidate here is the CONSENSUS
DETECTOR: build a pseudo-label from the other views, and ask whether view j bends
against it. That is label-free by construction; whether it works is measured.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import norm, spearmanr, gaussian_kde
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (OUT, GROUP, INSCOPE, load_cells_cached, pct, kde_modes,   # noqa: E402
                    n_min, bin_fit_apply, d1_shape_gain, MIN_NMIN)
from unit_test_transforms import cv_auc, cv_binmap_auc, folds                  # noqa: E402

SHAPE_TEST = os.path.join(os.path.dirname(OUT), "nonmono_transform", "shape_test.csv")
RECURRENT = ["rpdi", "pe_mean", "cusum_shift_idx", "hurst_exponent"]
CGRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
AGRID = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
DENS_GRID = 72
MIN_GAIN_PP = 2.0        # a transform must beat the raw view by this to be adopted


# ── transforms ────────────────────────────────────────────────────────────────
def t_squared(x, u, **kw):
    return x ** 2


def t_dist_median(x, u, **kw):
    return np.abs(x - np.median(x))


def t_abs_rank(x, u, **kw):
    return np.abs(norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)))


def t_centre(x, u, c=0.5, **kw):
    return np.abs(u - c)


def t_hinge(x, u, c=0.5, a=1.0, **kw):
    """Asymmetric two-piece fold. a=1 is the symmetric |u-c|; a=-1 is the identity
    (monotone); intermediate values tilt one arm against the other, which is what
    an off-centre inverted-U or a W needs and what the symmetric family cannot do."""
    return np.maximum(u - c, 0.0) - a * np.maximum(c - u, 0.0)


# ── the label-free consensus pseudo-label ─────────────────────────────────────
def consensus_score(V, j):
    """Mean of the other (already sign-oriented, z-scored) views — a label-free
    stand-in for the answer key. Deliberately the simple average rather than the
    deployed fusion: it does not depend on which arm we ship, and it cannot import
    the selector's own decisions about view j."""
    keep = [k for k in range(V.shape[1]) if k != j]
    if len(keep) < 3:
        return None
    return V[:, keep].mean(axis=1)


def consensus_pseudo_labels(s, frac=0.5):
    """Binarise the consensus at its median. Crude on purpose: a sharper cut would
    just make the pseudo-label a proxy for the consensus's own confidence."""
    thr = np.quantile(s, frac)
    return (s > thr).astype(int)


def cv_consensus_map(x, s_pseudo, y_unused=None, k=10, n_folds=5):
    """Reshape x by its bin map against the PSEUDO-label — no answer key anywhere.
    Cross-fitted on the pseudo-label so the map never sees the rows it scores."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for tr, te in folds(s_pseudo, 0):
        m, _ = bin_fit_apply(x[tr], s_pseudo[tr], x[te], k)
        if m is not None:
            out[te] = m
    if not np.isfinite(out).all():
        med = np.nanmedian(out)
        out = np.where(np.isfinite(out), out, med if np.isfinite(med) else 0.0)
    return out


# ── densities for the visual ──────────────────────────────────────────────────
def class_densities(v, y, grid_n=DENS_GRID):
    """Class-conditional densities on a shared grid, clipped to [p1,p99] so a
    single outlier cannot flatten the picture. Returns None if degenerate."""
    v = np.asarray(v, float)
    y = np.asarray(y, int)
    m = np.isfinite(v)
    v, y = v[m], y[m]
    if len(v) < 40 or np.std(v) < 1e-12:
        return None
    lo, hi = np.percentile(v, [1, 99])
    if hi - lo < 1e-12:
        lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return None
    g = np.linspace(lo, hi, grid_n)
    out = {}
    for lab, key in ((1, "correct"), (0, "halluc")):
        vv = v[y == lab]
        if len(vv) < 8:
            return None
        if np.std(vv) < 1e-9 or len(np.unique(vv)) < 5:
            d, _ = np.histogram(np.clip(vv, lo, hi), bins=grid_n, range=(lo, hi))
            d = d.astype(float)
            d /= max(d.sum() * (g[1] - g[0]), 1e-12)
        else:
            try:
                d = gaussian_kde(vv)(g)
            except Exception:                          # noqa: BLE001
                d, _ = np.histogram(np.clip(vv, lo, hi), bins=grid_n, range=(lo, hi))
                d = d.astype(float)
                d /= max(d.sum() * (g[1] - g[0]), 1e-12)
        out[key] = d
    mx = max(out["correct"].max(), out["halluc"].max(), 1e-12)
    return dict(grid=[round(float(z), 5) for z in g],
                correct=[round(float(z / mx), 4) for z in out["correct"]],
                halluc=[round(float(z / mx), 4) for z in out["halluc"]],
                n_correct=int((y == 1).sum()), n_halluc=int((y == 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--min-headroom", type=float, default=3.0,
                    help="pairs below this measured headroom are controls, not candidates")
    args = ap.parse_args()

    cells = load_cells_cached(use_cache=not args.no_cache)
    flagged = set()
    if os.path.exists(SHAPE_TEST):
        with open(SHAPE_TEST, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["exceeds_null"] == "1":
                    flagged.add((r["cell"], r["feature"]))

    pairs = []
    for ck in INSCOPE:
        cell = cells[ck]
        if n_min(cell["labels"]) < MIN_NMIN:
            continue
        for f in cell["pool"]:
            if (ck, f) in flagged or f in RECURRENT:
                pairs.append((ck, f))
    print(f"{len(pairs)} pairs under test\n")

    # ── pass 1: per-pair sweeps + the label-free consensus detector ───────────
    store = {}
    for ck, f in pairs:
        cell = cells[ck]
        j = cell["pool"].index(f)
        x, y, V = cell["V"][:, j], cell["labels"], cell["V"]
        u = pct(x)
        cons = consensus_score(V, j)
        pseudo = consensus_pseudo_labels(cons) if cons is not None else None

        curve = np.array([cv_auc(t_centre(x, u, c=c), y) for c in CGRID])
        # label-free twin of the same sweep, scored against the pseudo-label
        curve_pl = (np.array([cv_auc(t_centre(x, u, c=c), pseudo) for c in CGRID])
                    if pseudo is not None else None)
        cons_gain = (d1_shape_gain(x, pseudo, seeds=(0, 1))[0]
                     if pseudo is not None else float("nan"))

        store[(ck, f)] = dict(x=x, y=y, u=u, curve=curve, curve_pl=curve_pl,
                              pseudo=pseudo, cons_gain=cons_gain, j=j)
        print(f"  swept {ck:<32}{f}", flush=True)

    # ── pass 2: build every candidate, score it, pick one ────────────────────
    rows, panels = [], []
    for (ck, f), d in store.items():
        x, y, u, curve = d["x"], d["y"], d["u"], d["curve"]
        pseudo, curve_pl = d["pseudo"], d["curve_pl"]
        others = [(c2, f2) for (c2, f2) in store if f2 == f and c2 != ck]

        auc_mono = cv_auc(x, y)
        auc_ceil = cv_binmap_auc(x, y)
        headroom = (auc_ceil - auc_mono) * 100
        _, _, mode_pct = kde_modes(x)

        cand = {"identity": (x, {})}
        cand["squared"] = (t_squared(x, u), {})
        cand["dist_median"] = (t_dist_median(x, u), {})
        cand["abs_rank"] = (t_abs_rank(x, u), {})
        if np.isfinite(mode_pct):
            cand["mode_centre"] = (t_centre(x, u, c=float(mode_pct)),
                                   {"c": round(float(mode_pct), 3)})

        # label-fitted family ceiling
        bi = int(np.nanargmax(curve))
        cand["best_centre"] = (t_centre(x, u, c=float(CGRID[bi])),
                               {"c": float(CGRID[bi]), "fitted_on": "this cell's labels"})

        # label-FREE centre: argmax of the same sweep scored on the pseudo-label
        if curve_pl is not None and np.isfinite(curve_pl).any():
            cpl = float(CGRID[int(np.nanargmax(curve_pl))])
            cand["consensus_centre"] = (t_centre(x, u, c=cpl),
                                        {"c": cpl, "fitted_on": "consensus pseudo-label"})

        if others:
            m = np.nanmean(np.vstack([store[o]["curve"] for o in others]), axis=0)
            cl = float(CGRID[int(np.nanargmax(m))])
            cand["loco_centre"] = (t_centre(x, u, c=cl),
                                   {"c": cl, "fitted_on": f"{len(others)} other cells"})
            uu = np.concatenate([store[o]["u"] for o in others])
            yy = np.concatenate([store[o]["y"] for o in others])
            mp, _ = bin_fit_apply(uu, yy, u, 10)
            if mp is not None:
                cand["loco_binmap"] = (mp, {"fitted_on": f"{len(others)} other cells"})

        # asymmetric hinge, centre+asymmetry swept on this cell's labels (ceiling)
        best_h, best_v, best_p = -np.inf, None, None
        for c in CGRID:
            for a in AGRID:
                v = t_hinge(x, u, c=float(c), a=float(a))
                s = cv_auc(v, y)
                if np.isfinite(s) and s > best_h:
                    best_h, best_v, best_p = s, v, {"c": float(c), "a": float(a)}
        if best_v is not None:
            best_p["fitted_on"] = "this cell's labels"
            cand["hinge"] = (best_v, best_p)

        if pseudo is not None:
            cand["consensus_map"] = (cv_consensus_map(x, pseudo),
                                     {"fitted_on": "consensus pseudo-label"})

        # score everything on identical folds
        opts = []
        for name, (v, prm) in cand.items():
            a = cv_auc(v, y)
            opts.append(dict(name=name, auc=round(float(a), 4),
                             delta_pp=round(float((a - auc_mono) * 100), 2),
                             params=prm, label_free=name not in
                             ("best_centre", "hinge", "loco_centre", "loco_binmap")))

        # ── the selection rule, stated before looking ────────────────────────
        # Deployable = uses no labels from ANY cell. Among those, take the best,
        # and only adopt it if it clears MIN_GAIN_PP over the raw view.
        free = [o for o in opts if o["label_free"] and o["name"] != "identity"]
        best_free = max(free, key=lambda o: o["auc"]) if free else None
        chosen, why = "identity", "no label-free option cleared the bar"
        if best_free and best_free["delta_pp"] >= MIN_GAIN_PP:
            chosen = best_free["name"]
            why = (f"best label-free option; +{best_free['delta_pp']:.1f}pp over the raw "
                   f"view, which is {100*best_free['delta_pp']/headroom:.0f}% of the "
                   f"{headroom:.1f}pp available") if headroom > 0 else \
                  f"best label-free option; +{best_free['delta_pp']:.1f}pp"
        elif headroom < args.min_headroom:
            why = (f"only {headroom:.1f}pp of headroom exists — this view is already "
                   f"read correctly, and folding it costs "
                   f"{min(o['delta_pp'] for o in opts):.1f}pp at worst")

        rec = dict(cell=ck, domain=GROUP.get(ck, "?"), feature=f,
                   flagged=int((ck, f) in flagged), n=len(y), n_min=n_min(y),
                   base_rate=round(float(np.mean(y)), 4),
                   auc_mono=round(auc_mono, 4), auc_ceiling=round(auc_ceil, 4),
                   headroom_pp=round(headroom, 2),
                   cons_gain=(round(float(d["cons_gain"]), 4)
                              if np.isfinite(d["cons_gain"]) else ""),
                   is_candidate=int(headroom >= args.min_headroom),
                   chosen=chosen, why=why,
                   options={o["name"]: o for o in opts})
        rows.append(rec)

        panels.append(dict(
            **{k: rec[k] for k in ("cell", "domain", "feature", "flagged", "n", "n_min",
                                   "base_rate", "auc_mono", "auc_ceiling", "headroom_pp",
                                   "cons_gain", "is_candidate", "chosen", "why")},
            options=[dict(o, dens=class_densities(cand[o["name"]][0], y))
                     for o in sorted(opts, key=lambda o: -o["auc"])],
            centre_sweep=dict(grid=[float(c) for c in CGRID],
                              auc=[round(float(z), 4) for z in curve],
                              auc_pseudo=([round(float(z), 4) for z in curve_pl]
                                          if curve_pl is not None else None)),
        ))

    # ── does the LABEL-FREE detector find the right views? ───────────────────
    cg = np.array([float(r["cons_gain"]) if r["cons_gain"] != "" else np.nan for r in rows])
    hd = np.array([r["headroom_pp"] for r in rows])
    m = np.isfinite(cg) & np.isfinite(hd)
    rho, pv = spearmanr(cg[m], hd[m])
    print(f"\n{'='*104}\nTHE LABEL-FREE SELECTOR — does the consensus detector find the "
          f"views with real headroom?\n{'='*104}")
    print(f"  Spearman(consensus shape-gain, true headroom) = {rho:+.3f}  "
          f"p = {pv:.2e}   n = {int(m.sum())}")
    for thr in (0.005, 0.01, 0.02, 0.03):
        fl = [r for r in rows if r["cons_gain"] != "" and float(r["cons_gain"]) > thr]
        if not fl:
            continue
        tp = sum(1 for r in fl if r["headroom_pp"] >= args.min_headroom)
        base = sum(1 for r in rows if r["headroom_pp"] >= args.min_headroom) / len(rows)
        print(f"    flag at cons_gain > {thr:<5} -> {len(fl):>3} views, "
              f"precision {tp/len(fl):.3f}  (base rate {base:.3f})  "
              f"recall {tp/max(sum(1 for r in rows if r['headroom_pp']>=args.min_headroom),1):.3f}")

    cands = [r for r in rows if r["is_candidate"]]
    ctrls = [r for r in rows if not r["is_candidate"]]
    print(f"\n{'='*104}\nSELECTION — {len(cands)} candidates "
          f"(headroom >= {args.min_headroom}pp), {len(ctrls)} controls\n{'='*104}")
    print(f"{'cell':<30}{'feature':<18}{'hd':>6}{'mono':>7}{'chosen':>18}{'auc':>7}{'gain':>7}")
    print("-" * 104)
    adopted = 0
    for r in sorted(cands, key=lambda r: -r["headroom_pp"]):
        o = r["options"][r["chosen"]]
        adopted += r["chosen"] != "identity"
        print(f"{r['cell'][:29]:<30}{r['feature'][:17]:<18}{r['headroom_pp']:>6.1f}"
              f"{r['auc_mono']:>7.3f}{r['chosen']:>18}{o['auc']:>7.3f}{o['delta_pp']:>+7.1f}")
    print(f"\n  adopted a transform on {adopted}/{len(cands)} candidates")
    ctrl_bad = [r for r in ctrls if r["chosen"] != "identity"]
    print(f"  of the {len(ctrls)} controls, {len(ctrl_bad)} would also be transformed "
          f"(these are the false positives a label-free rule must avoid)")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "transform_selection.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(dict(min_headroom=args.min_headroom, min_gain_pp=MIN_GAIN_PP,
                       selector=dict(rho=float(rho), p=float(pv), n=int(m.sum())),
                       panels=panels), fh, separators=(",", ":"))
    print(f"\nsaved {p}  ({os.path.getsize(p)/1024:.0f} KB, {len(panels)} panels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
