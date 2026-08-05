#!/usr/bin/env python
"""
failure_deepdive.py — why do we fail where we fail? (Jul-30 advisor action item 1)

DIAGNOSIS ONLY. This script names mechanisms; it does not test repairs. Candidate
fixes are pre-registered in the HISTORY Step-210 write-up and tested in a later
step, so the diagnosis cannot be tuned to make a fix look good.

Nine cells were pinned as "failing" (HISTORY Step 209). All 25 in-scope cells are
measured — a diagnostic that only looks at failing cells cannot say what is
DIFFERENT about them.

THE CONFOUND THIS SCRIPT EXISTS TO AVOID
----------------------------------------
The nine weak cells are exactly the nine lowest `anchor_auc` in the grid, and
Spearman(anchor_auc, deployed AUROC) ~ +0.97. That looks like an orientation
story and is not one: `epr` is itself a pooled feature, so a weak anchor just
means every view is weak on that cell. Gate 4 below re-measures
Spearman(anchor_auc, best_single_view) on freshly loaded data; if it does not
come back ~+0.97 the framing in Step 209 is wrong and must be rewritten.

The quantity that survives the confound is FUSION MINUS BEST SINGLE VIEW.

FOUR PANELS (the advisors' four questions)
------------------------------------------
B1  what features were selected      -> percell.csv  (survivor sets, Jaccard vs
                                        GOOD_6 and vs the label-chosen oracle-5,
                                        and which STRONG views were excluded)
B2  did any feature behave           -> perfeature.csv (per cell x view AUROC and
    differently there                   cross-fitted non-monotonicity, each as a
                                        deviation from that view's own across-cell
                                        median)
B3  what the residual process did    -> residual_curves.csv (full per-K Eq.14
                                        curve, chosen K, relative gap to the
                                        runner-up, degeneracy flag, eigengap
                                        counterfactual)
B4  which stage loses the AUROC      -> percell.csv (the ladder, r1..r6)

Pure CPU, offline against local_cache/. Run in the background.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import INSCOPE, QA_CELLS                             # noqa: E402
from inscope_bench_common import load_cells, assert_good6, good6_cols   # noqa: E402
from spectral_utils.fusion_utils import (                              # noqa: E402
    lsml_continuous, detect_dependent_groups,
)
from spectral_utils.streaming_utils import anchor_orient                # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                       # noqa: E402
from spectral_utils.upcr import upcr_fit                                # noqa: E402

OUTDIR = os.path.join(REPO, "results", "failure_deepdive")

# exp06's fitted configuration, verbatim from labelfree_standing_report.py.
# This script must not silently re-tune the arm it is diagnosing.
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

# The nine, per HISTORY Step 209. Eight named by Omri off labelfree_standing.html,
# plus truthfulqa (rank 4 of 25 on every weakness measure, interleaved with them).
WEAK = [
    "losnet_hotpotqa_mistral7b", "inside_coqa_llama7b", "seiclr_triviaqa_opt30b",
    "truthfulqa_llama8b", "internalstates_gsm8k_qwen25_7b", "noise_gsm8k_phi3mini",
    "trace_math500_qwenmath15b_k10", "ars_gsm8k_r1distill8b",
    "lapeigvals_gsm8k_llama3b",
]

STRONG_VIEW_PCTL = 75      # "strong" = oracle-oriented AUROC at/above this pctl
N_BINS = 10                # quantile bins for the non-monotonicity probe
N_FOLDS = 5


# ── small helpers ────────────────────────────────────────────────────────────
def oriented_auc(y, x):
    """AUROC under the ORACLE per-view sign: max(a, 1-a). Never used for a
    deployed number — only as the ceiling a label-free path is measured against."""
    a = roc_auc_score(y, x)
    return max(a, 1.0 - a)


def binned_auc_cv(y, x, n_bins=N_BINS, n_folds=N_FOLDS, seed=0):
    """Cross-fitted AUROC of a piecewise-constant (bin-mean) predictor.

    Measures how much of the view's signal is available WITHOUT assuming a
    monotone relationship to the label. Cross-fitted because the in-sample
    version is optimistic by construction — bin means fitted and evaluated on
    the same rows will beat any monotone score even on pure noise.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(np.unique(x)) < n_bins or len(np.unique(y)) < 2:
        return np.nan
    oof = np.full(len(x), np.nan)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(x):
        edges = np.quantile(x[tr], np.linspace(0, 1, n_bins + 1)[1:-1])
        edges = np.unique(edges)
        if len(edges) == 0:
            continue
        b_tr, b_te = np.digitize(x[tr], edges), np.digitize(x[te], edges)
        gm = y[tr].mean()
        rate = np.array([y[tr][b_tr == b].mean() if (b_tr == b).any() else gm
                         for b in range(len(edges) + 1)])
        oof[te] = rate[b_te]
    ok = np.isfinite(oof)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return np.nan
    return max(roc_auc_score(y[ok], oof[ok]), 1.0 - roc_auc_score(y[ok], oof[ok]))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else np.nan


def lsml_anchor(cell, cols, groups=None):
    """The DEPLOYED path: L-SML over the columns, global sign from the anchor."""
    V = cell["V"]
    fused, meta = lsml_continuous(*[V[:, c] for c in sorted(set(cols))],
                                  groups=groups)
    s, flipped = anchor_orient(np.asarray(fused, dtype=float), cell["anchor"])
    return s, meta, flipped


def upcr_rho(cell):
    """U-PCR with polarity from sign(rho-hat), global sign from the anchor.

    Verbatim in structure from labelfree_standing_report.upcr_rho_oriented:
    `prepare_cell` has already applied ALL_SIGNS, so V must be multiplied by the
    hand signs to RECOVER the unoriented feature before rho derives polarity
    itself. Getting this backwards silently scores the incumbent arm."""
    V, pool = cell["V"], cell["pool"]
    hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], dtype=float)
    V_un = V * hand
    derived = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V_un * derived).T
    res = upcr_fit(F, **FIT)
    s, _ = anchor_orient(res.w @ F, cell["anchor"])
    return s, res


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cells = load_cells()

    # GATE 1 — validity anchor. A fail means the loaded data is not the data the
    # published numbers came from, and every row below would be void.
    ok, macro = assert_good6(cells, verbose=True)
    if not ok:
        raise SystemExit(f"GATE 1 FAILED: GOOD_6 macro {macro:.4f} != 0.7594. "
                         "Refusing to diagnose off data that is not ours.")
    print(f"GATE 1 PASS: GOOD_6 macro {macro:.4f}\n")

    bench = read_csv(os.path.join(REPO, "results/selector_bench/a2_groupfs__c46.csv"))
    dufs_rec = {r["cell"]: r for r in bench if r["variant"] == "a2.dufs_pf"}
    upcr_rec = {r["cell"]: r for r in read_csv(
        os.path.join(REPO, "results/upcr_study/06_orientation/per_cell.csv"))}
    oracle_rec = {r["cell"]: r for r in read_csv(
        os.path.join(REPO, "results/advisor_inscope/cell_oracle_vs_chosen.csv"))}

    percell, perfeat, curves, drift, ladder_viol = [], [], [], [], []

    for ck in INSCOPE:
        cell = cells[ck]
        y = np.asarray(cell["labels"], dtype=int)
        V, pool, anchor = cell["V"], cell["pool"], cell["anchor"]
        p = len(pool)

        # ── B2: per-view behaviour ───────────────────────────────────────────
        auc_raw = np.array([roc_auc_score(y, V[:, j]) for j in range(p)])
        auc_or = np.maximum(auc_raw, 1.0 - auc_raw)
        osign = np.where(auc_raw >= 0.5, 1.0, -1.0)
        nonmono = np.array([binned_auc_cv(y, V[:, j]) for j in range(p)]) - auc_or

        for j, f in enumerate(pool):
            perfeat.append(dict(cell=ck, group="QA" if ck in QA_CELLS else "math",
                                weak=ck in WEAK, feature=f,
                                auc_raw=round(float(auc_raw[j]), 4),
                                auc_oriented=round(float(auc_or[j]), 4),
                                oracle_sign=int(osign[j]),
                                nonmono_gain_cv=round(float(nonmono[j]), 4)))

        # ── subsets to walk ──────────────────────────────────────────────────
        g6 = good6_cols(cell)
        chosen = [f for f in dufs_rec[ck]["chosen"].split("|") if f in pool]
        dufs = [pool.index(f) for f in chosen]
        subsets = {"GOOD_6": g6, "dufs_pf": dufs, "FULL": list(range(p))}

        # ── B3: the residual process, per subset ─────────────────────────────
        res_info = {}
        for sname, cols in subsets.items():
            cols = sorted(set(cols))
            if len(cols) < 3:
                continue
            views = [V[:, c] for c in cols]
            K, c_as, resid, _s, curve, diag = detect_dependent_groups(
                views, return_curve=True, return_diag=True)
            rs = sorted(r for _K, r, _c in curve)
            runner = rs[1] if len(rs) > 1 else np.nan
            gap_rel = ((runner - resid) / abs(resid)
                       if np.isfinite(runner) and resid else np.nan)
            for kk, rr, _cc in curve:
                curves.append(dict(cell=ck, subset=sname, K=int(kk),
                                   residual=round(float(rr), 6),
                                   chosen=bool(kk == K)))
            # eigengap counterfactual: does a different K rule change the answer?
            Ke, ce, *_ = detect_dependent_groups(views, method="eigengap")
            s_eig, _, _ = lsml_anchor(cell, cols, groups=ce)
            res_info[sname] = dict(
                K=int(K), residual=float(resid), gap_rel=float(gap_rel),
                degenerate=bool(diag.get("degenerate", False)),
                m=len(cols), K_eig=int(Ke),
                auc_eig=float(roc_auc_score(y, s_eig)))

        # ── B4: the ladder, on the deployed (dufs_pf) subset ─────────────────
        cols = sorted(set(dufs))
        Vs = V[:, cols]
        r1 = float(auc_or[cols].max())
        r2 = oriented_auc(y, (Vs * osign[cols]).mean(axis=1))   # oracle rel. signs
        r3 = oriented_auc(y, Vs.mean(axis=1))                   # no rel. signs
        s_lsml, meta6, flipped = lsml_anchor(cell, cols)
        r5 = float(roc_auc_score(y, s_lsml))                    # DEPLOYED
        r4 = max(r5, 1.0 - r5)                                  # oracle global sign
        r6 = float(oracle_rec[ck]["oracle_auroc"])

        # r5 <= r4 by construction (the anchor can only match or lose to the
        # oracle global bit). A violation is a bug in the ladder, not a finding.
        if r5 > r4 + 1e-12:
            ladder_viol.append(f"{ck}: r5 {r5:.4f} > r4 {r4:.4f}")

        # ── B1: what was selected ────────────────────────────────────────────
        s_up, up = upcr_rho(cell)
        a_up = float(roc_auc_score(y, s_up))
        kept = [pool[j] for j in range(p) if up.keep[j]]
        oracle_feats = oracle_rec[ck]["oracle_feats"].split("|")
        g6_names = [pool[j] for j in g6]

        thr = np.percentile(auc_or, STRONG_VIEW_PCTL)
        strong = {pool[j] for j in range(p) if auc_or[j] >= thr}
        excluded_strong = sorted(strong - set(kept))
        dropped_strong = sorted(strong - set(chosen))

        # ── reproduction gate (GATE 2) ───────────────────────────────────────
        for nm, got, want in (("a2.dufs_pf", r5, float(dufs_rec[ck]["auroc"])),
                              ("upcr+sign(rho)", a_up,
                               float(upcr_rec[ck]["auroc_rho_anchor"]))):
            if abs(got - want) > 5e-4:
                drift.append(f"{ck}/{nm}: {got:.4f} vs recorded {want:.4f}")

        a_g6 = float(roc_auc_score(y, lsml_anchor(cell, g6)[0]))
        rd = res_info.get("dufs_pf", {})
        percell.append(dict(
            cell=ck, group="QA" if ck in QA_CELLS else "math", weak=ck in WEAK,
            n=int(len(y)), pos_rate=round(float(y.mean()), 4), p_pool=p,
            anchor_auc=round(float(oriented_auc(y, anchor)), 4),
            best_single=round(float(auc_or.max()), 4),
            best_single_name=pool[int(np.argmax(auc_or))],
            median_single=round(float(np.median(auc_or)), 4),
            # ladder
            r1_best_single_in_subset=round(r1, 4),
            r2_avg_oracle_signs=round(r2, 4),
            r3_avg_no_signs=round(r3, 4),
            r4_lsml_oracle_global=round(r4, 4),
            r5_lsml_anchor_DEPLOYED=round(r5, 4),
            r6_oracle5_ceiling=round(r6, 4),
            d_fuse_vs_best=round((r2 - r1) * 100, 2),
            d_signs=round((r3 - r2) * 100, 2),
            d_lsml_vs_avg=round((r4 - r3) * 100, 2),
            d_global_sign=round((r5 - r4) * 100, 2),
            d_headroom=round((r6 - r5) * 100, 2),
            # arms
            dufs_auroc=round(r5, 4), dufs_size=len(chosen),
            upcr_auroc=round(a_up, 4), upcr_size=int(up.keep.sum()),
            good6_auroc=round(a_g6, 4),
            upcr_minus_best=round((a_up - float(auc_or.max())) * 100, 2),
            dufs_minus_best=round((r5 - float(auc_or.max())) * 100, 2),
            good6_minus_best=round((a_g6 - float(auc_or.max())) * 100, 2),
            # residual process on the deployed subset
            K=rd.get("K"), residual=round(rd.get("residual", np.nan), 3),
            residual_gap_rel=round(rd.get("gap_rel", np.nan), 5),
            degenerate=rd.get("degenerate"), K_eigengap=rd.get("K_eig"),
            auc_eigengap=round(rd.get("auc_eig", np.nan), 4),
            d_eigengap=round((rd.get("auc_eig", np.nan) - r5) * 100, 2),
            # U-PCR internals
            upcr_abstained=bool(up.abstained),
            upcr_simple_avg=bool(up.used_simple_average),
            upcr_ncomp=int(up.n_components_used),
            upcr_lambda2_frac=round(float(up.lambda2_frac), 4),
            upcr_g2_at_ceiling=bool(up.g2_at_ceiling),
            # selection overlap
            jac_dufs_oracle=round(jaccard(chosen, oracle_feats), 3),
            jac_upcr_oracle=round(jaccard(kept, oracle_feats), 3),
            jac_dufs_good6=round(jaccard(chosen, g6_names), 3),
            n_strong_excluded_upcr=len(excluded_strong),
            n_strong_dropped_dufs=len(dropped_strong),
            strong_excluded_upcr="|".join(excluded_strong),
            strong_dropped_dufs="|".join(dropped_strong),
            nonmono_mean=round(float(np.nanmean(nonmono)), 4),
            nonmono_p90=round(float(np.nanpercentile(nonmono, 90)), 4),
            dufs_chosen="|".join(chosen), upcr_kept="|".join(kept),
        ))
        print(f"  {ck:32s} dufs {r5:.4f}  upcr {a_up:.4f}  g6 {a_g6:.4f}  "
              f"best1 {auc_or.max():.4f}  K={rd.get('K')}", flush=True)

    # ── per-view deviation from its own across-cell median (B2) ──────────────
    by_feat = {}
    for r in perfeat:
        by_feat.setdefault(r["feature"], []).append(r)
    for f, rows in by_feat.items():
        med_a = float(np.nanmedian([r["auc_oriented"] for r in rows]))
        med_n = float(np.nanmedian([r["nonmono_gain_cv"] for r in rows]))
        for r in rows:
            r["auc_dev_from_own_median"] = round(r["auc_oriented"] - med_a, 4)
            r["nonmono_dev_from_own_median"] = round(
                r["nonmono_gain_cv"] - med_n, 4)
            r["n_cells_for_feature"] = len(rows)

    # ── gates 3-5 ────────────────────────────────────────────────────────────
    print()
    print(f"GATE 2 reproduction (<5e-4): "
          f"{'PASS' if not drift else 'FAIL — ' + '; '.join(drift)}")
    print(f"GATE 3 ladder r5<=r4: "
          f"{'PASS' if not ladder_viol else 'FAIL — ' + '; '.join(ladder_viol)}")

    aa = np.array([r["anchor_auc"] for r in percell])
    bs = np.array([r["best_single"] for r in percell])
    dep = np.array([r["dufs_auroc"] for r in percell])
    rho_bs = stats.spearmanr(aa, bs)
    rho_dep = stats.spearmanr(aa, dep)
    print(f"GATE 4 confound re-check: Spearman(anchor_auc, best_single) = "
          f"{rho_bs.statistic:+.3f} (p={rho_bs.pvalue:.2g})  "
          f"{'PASS' if rho_bs.statistic > 0.9 else 'FAIL — the Step-209 framing is wrong'}")
    print(f"         Spearman(anchor_auc, deployed)    = "
          f"{rho_dep.statistic:+.3f} (p={rho_dep.pvalue:.2g})")

    small = [r for r in percell if (r["K"] is not None and r["p_pool"] and
                                    r.get("dufs_size", 99) <= 4)]
    print(f"GATE 5 staleness: {len(small)} cell(s) with a deployed subset of size "
          f"<=4 (K/residual recomputed here, never joined)")

    # ── headline contrast ────────────────────────────────────────────────────
    w = np.array([r["weak"] for r in percell])
    for arm in ("upcr_minus_best", "dufs_minus_best", "good6_minus_best"):
        v = np.array([r[arm] for r in percell], dtype=float)
        print(f"  {arm:20s} weak {v[w].mean():+6.2f}pp   healthy {v[~w].mean():+6.2f}pp")

    # ── write ────────────────────────────────────────────────────────────────
    def dump(rows, name, fields=None):
        path = os.path.join(OUTDIR, name)
        fields = fields or list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        print(f"  wrote {path} ({len(rows)} rows)")

    print()
    dump(percell, "percell.csv")
    dump(perfeat, "perfeature.csv")
    dump(curves, "residual_curves.csv")
    with open(os.path.join(OUTDIR, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(dict(good6_macro=macro, drift=drift, ladder_viol=ladder_viol,
                       spearman_anchor_best_single=rho_bs.statistic,
                       spearman_anchor_deployed=rho_dep.statistic), f, indent=2)


if __name__ == "__main__":
    main()
