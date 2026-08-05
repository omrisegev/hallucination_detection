#!/usr/bin/env python
"""
build_data.py — per-cell evidence for the Jul-2026 action-item pages.

Emits ONE json per cell with everything the pages need, so the renderer can be
iterated on without paying the ~4 min cell load again. Nothing is aggregated
here: every number is per cell, and macros are only ever computed downstream for
context.

Per cell we record, for every view in the pool:
  its AUROC as-is and under the oracle sign, its cross-fitted non-monotone gain,
  which subsets contain it, its L-SML group and effective fusion weight, and a
  class-conditional histogram so the view can actually be LOOKED at.

And for every subset (GOOD_6 / the deployed DUFS choice / the label-chosen
oracle-5 / the full pool): the fused AUROC under the deployed anchor rule, the
fused AUROC under a TRUE-LABEL anchor, whether the anchor flipped, the grouping,
the Eq.-14 residual, and the fused-score histogram by class.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import INSCOPE, QA_CELLS                             # noqa: E402
from inscope_bench_common import load_cells, assert_good6, good6_cols   # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous                 # noqa: E402
from spectral_utils.streaming_utils import anchor_orient                # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS, GOOD_6               # noqa: E402
from spectral_utils.upcr import upcr_fit                                # noqa: E402

OUT = os.path.join(REPO, "results", "action_items_jul2026", "_data")

FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

WEAK = ["losnet_hotpotqa_mistral7b", "inside_coqa_llama7b",
        "seiclr_triviaqa_opt30b", "truthfulqa_llama8b",
        "internalstates_gsm8k_qwen25_7b", "noise_gsm8k_phi3mini",
        "trace_math500_qwenmath15b_k10", "ars_gsm8k_r1distill8b",
        "lapeigvals_gsm8k_llama3b"]

NBINS = 26
CLIP = 2.8


def hist_by_class(x, y, lo=-CLIP, hi=CLIP, nbins=NBINS):
    """Class-conditional histogram of a z-scored view, as densities so the two
    classes are comparable when one is 6x larger than the other."""
    x = np.clip(np.asarray(x, dtype=float), lo, hi)
    edges = np.linspace(lo, hi, nbins + 1)
    out = {}
    for key, mask in (("pos", y == 1), ("neg", y == 0)):
        h, _ = np.histogram(x[mask], bins=edges)
        s = h.sum()
        out[key] = (h / s).round(5).tolist() if s else [0.0] * nbins
    out["edges"] = edges.round(3).tolist()
    return out


def binned_auc_cv(y, x, n_bins=10, n_folds=5, seed=0):
    """Cross-fitted AUROC of a bin-mean predictor: how much of the view's signal
    survives WITHOUT assuming it is monotone in the label."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if len(np.unique(x)) < n_bins or len(np.unique(y)) < 2:
        return None
    oof = np.full(len(x), np.nan)
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x):
        e = np.unique(np.quantile(x[tr], np.linspace(0, 1, n_bins + 1)[1:-1]))
        if len(e) == 0:
            continue
        btr, bte = np.digitize(x[tr], e), np.digitize(x[te], e)
        gm = y[tr].mean()
        rate = np.array([y[tr][btr == b].mean() if (btr == b).any() else gm
                         for b in range(len(e) + 1)])
        oof[te] = rate[bte]
    ok = np.isfinite(oof)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return None
    a = roc_auc_score(y[ok], oof[ok])
    return float(max(a, 1 - a))


def eff_weights(meta, m):
    """Per-view effective linear weight of the L-SML fusion.

    fused = sum_g cross_w[g] * (X[:, idx_g] @ w_g), so view idx_g[i] carries
    cross_w[g] * w_g[i]. Reported normalised to sum |w| = 1 so cells compare.
    """
    w = np.zeros(m)
    cw = np.asarray(meta["cross_weights"], dtype=float).ravel()
    for g, (idx, wg) in enumerate(meta["group_weights"]):
        c = cw[g] if g < len(cw) else (cw[0] if len(cw) else 1.0)
        w[np.asarray(idx, dtype=int)] = c * np.asarray(wg, dtype=float)
    s = np.abs(w).sum()
    return (w / s if s else w)


def score_subset(cell, cols, y, osign):
    """Fuse a column subset the deployed way, and record what the deployed
    global-sign rule cost against a true-label anchor.

    Also records the two simple-average rungs the ladder needs: averaging the
    views when told their relative signs, and when not told. The gap between
    them is the sign information L-SML then has to recover on its own.
    """
    V = cell["V"]
    cols = sorted(set(int(c) for c in cols))
    fused, meta = lsml_continuous(*[V[:, c] for c in cols])
    s, flipped = anchor_orient(np.asarray(fused, float), cell["anchor"])
    a_dep = float(roc_auc_score(y, s))
    a_true = max(a_dep, 1.0 - a_dep)          # what a TRUE-LABEL anchor would give
    w = eff_weights(meta, len(cols))

    Vs = V[:, cols]
    a_or = roc_auc_score(y, (Vs * osign[cols]).mean(axis=1))
    a_no = roc_auc_score(y, Vs.mean(axis=1))
    return dict(
        _avg_oracle=round(float(max(a_or, 1 - a_or)), 4),
        _avg_none=round(float(max(a_no, 1 - a_no)), 4),
        cols=cols, feats=[cell["pool"][c] for c in cols],
        auroc_deployed=round(a_dep, 4),
        auroc_true_anchor=round(a_true, 4),
        anchor_cost_pp=round((a_dep - a_true) * 100, 3),
        anchor_flipped=bool(flipped),
        K=int(meta["K"]), groups=[int(v) for v in meta["c"]],
        residual=round(float(meta["residual"]), 3),
        degenerate=bool(meta.get("degenerate", False)),
        weights=[round(float(v), 4) for v in w],
        hist=hist_by_class((s - s.mean()) / (s.std() or 1), y),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = load_cells()
    ok, macro = assert_good6(cells, verbose=True)
    if not ok:
        raise SystemExit(f"validity gate failed: {macro:.4f} != 0.7594")
    print(f"GATE GOOD_6 macro {macro:.4f} PASS\n")

    import csv
    with open(os.path.join(REPO, "results/selector_bench/a2_groupfs__c46.csv"),
              newline="", encoding="utf-8") as f:
        dufs = {r["cell"]: r for r in csv.DictReader(f)
                if r["variant"] == "a2.dufs_pf"}
    with open(os.path.join(REPO,
              "results/advisor_inscope/cell_oracle_vs_chosen.csv"),
              newline="", encoding="utf-8") as f:
        orac = {r["cell"]: r for r in csv.DictReader(f)}

    index = []
    for ck in INSCOPE:
        cell = cells[ck]
        y = np.asarray(cell["labels"], dtype=int)
        V, pool = cell["V"], cell["pool"]
        p = len(pool)
        u = cell["unlabeled"]

        auc_raw = np.array([roc_auc_score(y, V[:, j]) for j in range(p)])
        auc_or = np.maximum(auc_raw, 1 - auc_raw)
        osign = np.where(auc_raw >= 0.5, 1, -1)

        g6 = good6_cols(cell)
        dsel = [pool.index(f) for f in dufs[ck]["chosen"].split("|") if f in pool]
        o5 = [pool.index(f) for f in orac[ck]["oracle_feats"].split("|")
              if f in pool]

        subs = {}
        for nm, cols in (("GOOD_6", g6), ("deployed", dsel),
                         ("oracle5", o5), ("FULL", list(range(p)))):
            if len(set(cols)) >= 3:
                subs[nm] = score_subset(cell, cols, y, osign)

        # U-PCR: which views survive Algorithm 1's exclusion, on this cell
        hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], float)
        V_un = V * hand
        der = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
        der[der == 0] = 1.0
        F = (V_un * der).T
        ures = upcr_fit(F, **FIT)
        s_up, up_flip = anchor_orient(ures.w @ F, cell["anchor"])
        a_up = float(roc_auc_score(y, s_up))

        # ── the U-PCR ladder, on the FULL pool ──────────────────────────────
        # U-PCR is not handed a subset: it takes the whole pool and excludes
        # internally, so its ladder runs over all p views. Unlike L-SML — which
        # is sign-gauge invariant and recovers polarity implicitly — U-PCR
        # ESTIMATES each view's polarity explicitly as sign(rho_hat). That step
        # can succeed or fail on its own, so it gets its own rung.
        osign_un = (np.array([ALL_SIGNS.get(f, +1) for f in pool], float)
                    * osign)                      # oracle polarity of V_un
        pol_agree = float((der == osign_un).mean())

        a_or_f = roc_auc_score(y, (V * osign).mean(axis=1))
        a_no_f = roc_auc_score(y, V.mean(axis=1))
        a_der_f = roc_auc_score(y, (V_un * der).mean(axis=1))
        u_r1 = float(auc_or.max())
        u_r2 = float(max(a_or_f, 1 - a_or_f))     # average, oracle signs
        u_r3 = float(max(a_no_f, 1 - a_no_f))     # average, no signs
        u_r3b = float(max(a_der_f, 1 - a_der_f))  # average, sign(rho) signs
        u_r4 = float(max(a_up, 1 - a_up))         # U-PCR weights, oracle global
        cost_u = (u_r2 - u_r3) * 100
        upcr_ladder = dict(
            r1=round(u_r1, 4), r2=round(u_r2, 4), r3=round(u_r3, 4),
            r3_derived=round(u_r3b, 4), r4=round(u_r4, 4), r5=round(a_up, 4),
            polarity_agreement=round(pol_agree, 4),
            n_polarity_wrong=int((der != osign_un).sum()),
            recovery=(round(((u_r4 - u_r3) * 100) / cost_u, 3)
                      if cost_u > 0.5 else None),
            sign_step_recovery=(round(((u_r3b - u_r3) * 100) / cost_u, 3)
                                if cost_u > 0.5 else None),
        )

        anc = np.asarray(cell["anchor"], float)
        a_anc = float(roc_auc_score(y, anc))

        views = []
        for j, f in enumerate(pool):
            views.append(dict(
                name=f, auc_raw=round(float(auc_raw[j]), 4),
                auc_oriented=round(float(auc_or[j]), 4),
                oracle_sign=int(osign[j]),
                hand_sign=int(ALL_SIGNS.get(f, +1)),
                nonmono=binned_auc_cv(y, V[:, j]),
                in_good6=f in GOOD_6, in_deployed=j in dsel, in_oracle5=j in o5,
                kept_upcr=bool(ures.keep[j]),
                upcr_weight=round(float(ures.w[j]), 5),
                hist=hist_by_class(V[:, j], y),
            ))
        for v in views:
            v["nonmono_gain"] = (None if v["nonmono"] is None
                                 else round(v["nonmono"] - v["auc_oriented"], 4))

        rec = dict(
            cell=ck, group="QA" if ck in QA_CELLS else "math",
            weak=ck in WEAK, n=int(len(y)), n_pos=int(y.sum()),
            pos_rate=round(float(y.mean()), 4), p_pool=p, pool=pool,
            anchor_name=u.anchor_name,
            anchor_auc_raw=round(a_anc, 4),
            anchor_auc_oriented=round(max(a_anc, 1 - a_anc), 4),
            anchor_hist=hist_by_class((anc - anc.mean()) / (anc.std() or 1), y),
            upcr=dict(auroc=round(a_up, 4),
                      auroc_true_anchor=round(max(a_up, 1 - a_up), 4),
                      flipped=bool(up_flip), kept=int(ures.keep.sum()),
                      abstained=bool(ures.abstained),
                      simple_avg=bool(ures.used_simple_average),
                      ncomp=int(ures.n_components_used),
                      ladder=upcr_ladder,
                      derived_signs=[int(v) for v in der],
                      oracle_signs_unoriented=[int(v) for v in osign_un]),
            views=views, subsets=subs,
        )
        with open(os.path.join(OUT, f"{ck}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)
        index.append(dict(cell=ck, group=rec["group"], weak=rec["weak"],
                          n=rec["n"], pos_rate=rec["pos_rate"],
                          anchor_name=rec["anchor_name"],
                          anchor_auc=rec["anchor_auc_oriented"],
                          best_view=max(views, key=lambda v: v["auc_oriented"])["name"],
                          best_view_auc=round(float(auc_or.max()), 4),
                          good6=subs["GOOD_6"]["auroc_deployed"],
                          deployed=subs["deployed"]["auroc_deployed"],
                          upcr=rec["upcr"]["auroc"],
                          oracle5=subs["oracle5"]["auroc_deployed"]
                          if "oracle5" in subs else None))
        print(f"  {ck:32s} anchor {rec['anchor_name']:>18s} {a_anc:.3f}  "
              f"dep {subs['deployed']['auroc_deployed']:.4f}  "
              f"g6 {subs['GOOD_6']['auroc_deployed']:.4f}  upcr {a_up:.4f}",
              flush=True)

    with open(os.path.join(OUT, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(dict(good6_macro=macro, cells=index), f, indent=1)
    print(f"\nwrote {len(index)} cell files to {OUT}")


if __name__ == "__main__":
    main()
