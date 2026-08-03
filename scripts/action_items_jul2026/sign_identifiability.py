#!/usr/bin/env python
"""
sign_identifiability.py — M0 / run R6. ESTABLISHES TARGET T4.

WHY THIS EXISTS
---------------
Sign-recovery failure keeps recurring on the weak cells and the problematic views,
and both previous attempts at it went nowhere because both treated the sign as a
quantity to be ESTIMATED BETTER. Step 213 refuted Z2 synchronisation without ever
asking how often the incumbent estimator is actually wrong ON THE VIEWS THAT MATTER.

WHAT IS AND IS NOT NEW HERE. The macro recovery rate is NOT new — it reproduces exp06's
`mean_polarity_agreement_empirical` bit-identically (0.9414116189259868), which is the
right cross-check to run and the wrong thing to claim as a discovery. What is new is the
DECOMPOSITION: Gate P on this arm, the bootstrap stability statistic, and above all the
split by Step 218's non-monotone flag, which is where the 94% macro turns out to be
hiding a 66% sub-population.

The hypothesis under test, and it comes from Step 218 rather than the sign literature:
THE DEPLOYED ORIENTER IS sign(rho_hat), A LINEAR COVARIANCE STATISTIC. On a
non-monotone view Cov(f, y) ~ 0 however much rank information the view carries, so
sign(rho_hat) reads noise and returns a coin flip. Step 218.1 found exactly this bug
inside its own instrument (nonmono_shape_test.py:109 picked the isotonic direction by
Pearson corrcoef; "on a U-shape that correlation is ~0 and its sign is coin-flip
noise") and fixed it with a union-of-cones MLE. The deployed orienter has the identical
failure mode and has not been fixed.

WHAT IS MEASURED — four things, in this order
---------------------------------------------
  GATE P   Is the ARM OF RECORD even sign-sensitive? L-SML is sign-gauge invariant
           (max |d AUROC| = 0.00e+00 under a random +-1 vector, Step 213; 24 cells here since
           Step 216),
           so half of Step 213's repair was invalid by construction. The arm here is
           U-PCR + sign(rho), which RE-DERIVES the polarity from the flipped data — so
           it may absorb the flip and be invariant too. That has never been checked.
           IT IS CHECKED FIRST, because everything downstream depends on it.

  M0-a     STABILITY, label-free. Bootstrap over rows (B replicates); per view, the
           fraction of replicates agreeing with the point estimate's sign. Each
           replicate's single global +-1 gauge is resolved AGAINST THE POINT ESTIMATE,
           never against the oracle — otherwise labels leak into a label-free statistic.

  M0-b     CORRECTNESS, oracle. Does sign(rho_hat) match the sign the labels give?
           Scored after resolving the cell's ONE global bit, exactly as exp06 does:
           charging that bit to all 30 views understates agreement (it read `epr` at 4%
           before the fix). The folded statistic max(a, 1-a) cannot go below 0.5, so
           0.5 is NOT its chance level — the null is Monte-Carloed (exp06 defect D5).

  M0-c     Is disagreement CONCENTRATED on the non-monotone views, or uniform on the
           weak cells? The two pre-registered readings:
             concentrated -> the hypothesis holds, S1/S2/S3 are the right responses
             ~50% uniform -> the sign is not recoverable there by any label-free
                             method, and the honest deliverable is that ceiling
           Joined against Step 218's answer key, results/nonmono_v2/unit_test_transforms.csv.

  S3-det   Bonus, and the cheapest path to the 0.654 trigger in the plan: score
           "unstable sign" as a DETECTOR of true non-monotone headroom on the same 99
           labelled views, against the incumbent consensus detector's precision 0.562.

Run:  python scripts/action_items_jul2026/sign_identifiability.py [--boot 500]
Out:  results/action_items_jul2026/_data/sign_identifiability_{views,cells}.csv
      results/action_items_jul2026/_data/sign_identifiability.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, os.path.join(REPO, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_bench_common import load_cells, assert_good6            # noqa: E402
from spectral_utils.streaming_utils import anchor_orient             # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                    # noqa: E402
from spectral_utils.upcr import upcr_fit                             # noqa: E402
from test_sign_repair import WEAK                                    # noqa: E402

OUT = os.path.join(REPO, "results", "action_items_jul2026", "_data")
ANSWER_KEY = os.path.join(REPO, "results", "nonmono_v2", "unit_test_transforms.csv")

FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

HEADROOM_CANDIDATE_PP = 3.0     # Step 218's candidate threshold
INCUMBENT_PRECISION = 0.562     # consensus detector, results/nonmono_v2
INCUMBENT_BASE_RATE = 0.384
TRIGGER_PRECISION = 0.654       # Step 218 §218.6.3: where `squared` becomes correct


def raw_views(cell):
    """Un-oriented z-scored views. prepare_cell applied ALL_SIGNS and
    zscore(s*x) == s*zscore(x) for s = +-1, so multiplying by the hand signs recovers
    the raw feature (exp06_orientation.py:76-81)."""
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    return np.asarray(cell["V"], dtype=float) * hand, hand


def derived_polarity(V_raw):
    pol = np.sign(upcr_fit(V_raw.T, **FIT).rho_hat_full)
    pol[pol == 0] = 1.0
    return pol


def score_derived(cell, V_raw, pol):
    """Full arm-of-record score: apply polarity, fit, anchor-orient, raw AUROC."""
    F = (V_raw * pol).T
    res = upcr_fit(F, **FIT)
    if not np.isfinite(res.w).all():
        return float("nan")
    s, _ = anchor_orient(res.w @ F, cell["anchor"])
    return float(roc_auc_score(cell["labels"], s))


# ---------------------------------------------------------------------------
# GATE P — is the arm of record sign-sensitive at all?
# ---------------------------------------------------------------------------
def gate_p(cells, n_draws=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ck, cell in cells.items():
        V_raw, _ = raw_views(cell)
        m = V_raw.shape[1]
        base = score_derived(cell, V_raw, derived_polarity(V_raw))
        deltas = []
        for _ in range(n_draws):
            s = rng.choice([-1.0, 1.0], size=m)
            Vs = V_raw * s
            deltas.append(score_derived(cell, Vs, derived_polarity(Vs)) - base)
        deltas = np.array(deltas, float)
        rows.append({"cell": ck, "auroc_base": base,
                     "max_abs_delta": float(np.nanmax(np.abs(deltas))),
                     "mean_abs_delta": float(np.nanmean(np.abs(deltas))),
                     "n_draws": n_draws})
        print(f"  {ck[:34]:34s} base {base:.4f}  max|d AUROC| under random +-1: "
              f"{rows[-1]['max_abs_delta']:.2e}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# M0 — stability and correctness, per view
# ---------------------------------------------------------------------------
def m0(cells, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    view_rows, cell_rows = [], []
    for ck, cell in cells.items():
        V_raw, _ = raw_views(cell)
        y = cell["labels"]
        n, m = V_raw.shape

        # oracle sign per view, from the labels
        oracle = np.array([1.0 if roc_auc_score(y, V_raw[:, j]) >= 0.5 else -1.0
                           for j in range(m)])
        orc_auc = np.array([max(a, 1 - a) for a in
                            (roc_auc_score(y, V_raw[:, j]) for j in range(m))])

        rho_full = upcr_fit(V_raw.T, **FIT).rho_hat_full
        pol = np.sign(rho_full)
        pol[pol == 0] = 1.0
        raw_agree = float(np.mean(pol == oracle))
        global_flip = raw_agree < 0.5
        pol_res = -pol if global_flip else pol           # resolve the ONE global bit
        agree = max(raw_agree, 1.0 - raw_agree)

        # bootstrap: gauge of each replicate resolved against the POINT ESTIMATE
        # (label-free), never against the oracle
        agree_cnt = np.zeros(m)
        n_ok = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            # NO label-based filter here. `derived_polarity` never sees y, and dropping
            # replicates on a label criterion would make the acceptance of each
            # replicate label-dependent inside a statistic this docstring calls
            # label-free. Degenerate-column replicates are still dropped (that is a
            # property of V, not of y).
            Vb = V_raw[idx]
            sd = Vb.std(axis=0)
            if np.any(sd < 1e-12):
                continue
            try:
                pb = derived_polarity((Vb - Vb.mean(axis=0)) / sd)
            except Exception:
                continue
            if np.mean(pb == pol_res) < 0.5:
                pb = -pb
            agree_cnt += (pb == pol_res)
            n_ok += 1
        stability = agree_cnt / max(n_ok, 1)

        for j, f in enumerate(cell["pool"]):
            view_rows.append({
                "cell": ck, "feature": f, "n": int(n),
                "weak_cell": ck in WEAK,
                "stability": float(stability[j]),
                "sign_correct": int(pol_res[j] == oracle[j]),
                "oracle_single_auroc": float(orc_auc[j]),
                "abs_rho": float(abs(rho_full[j])),
            })

        cell_rows.append({
            "cell": ck, "m": int(m), "n": int(n),
            "weak_cell": ck in WEAK,
            "sign_recovery": float(np.mean(pol_res == oracle)),
            "folded_agreement": agree,
            "global_flip_applied": bool(global_flip),
            "mean_stability": float(np.mean(stability)),
            "n_boot_used": int(n_ok),
            "auroc": score_derived(cell, V_raw, pol),
        })
        print(f"  {ck[:34]:34s} recovery {cell_rows[-1]['sign_recovery']*100:5.1f}%  "
              f"mean stability {np.mean(stability)*100:5.1f}%  "
              f"{'WEAK' if ck in WEAK else ''}", flush=True)
    return view_rows, cell_rows


def folded_null(cell_rows, n_mc=20000, seed=0):
    """max(a, 1-a) can never fall below 0.5, so 0.5 is not its chance level."""
    rng = np.random.default_rng(seed)
    ms = [r["m"] for r in cell_rows]
    draws = [np.mean([max(v, 1 - v) for v in
                      (rng.binomial(m, 0.5) / m for m in ms)]) for _ in range(n_mc)]
    return np.asarray(draws)


# ---------------------------------------------------------------------------
def _top3_share(cand):
    from collections import Counter
    c = Counter(r["cell"] for r in cand).most_common(3)
    return float(sum(v for _, v in c) / max(len(cand), 1))


def _cell_perm_p(lab, n_perm=20000, seed=0):
    """Cell-clustered null for the candidate-vs-control recovery gap.

    A plain Fisher test on the 99 views treats them as independent. They are not: the
    candidates are concentrated (18 of 38 sit in 3 cells) and every view inside a cell
    shares that cell's single global +-1 resolution, so the effective n is nearer the
    number of cells than 99 (Step 204 defect D7). This shuffles the candidate/control
    label WITHIN each cell, which holds each cell's candidate count and each cell's
    recovery pattern fixed and destroys only the view-level pairing.
    """
    rng = np.random.default_rng(seed)
    by_cell = {}
    for r in lab:
        by_cell.setdefault(r["cell"], []).append(r)
    def gap(assign):
        c = [v for k, rs in assign.items() for v in rs[0]]
        d = [v for k, rs in assign.items() for v in rs[1]]
        if not c or not d:
            return float("nan")
        return float(np.mean(c) - np.mean(d))
    obs_assign = {k: ([r["sign_correct"] for r in rs if r["is_candidate"] == 1],
                      [r["sign_correct"] for r in rs if r["is_candidate"] == 0])
                  for k, rs in by_cell.items()}
    obs = gap(obs_assign)
    if not np.isfinite(obs):
        return float("nan")
    hits = 0
    for _ in range(n_perm):
        a = {}
        for k, rs in by_cell.items():
            k_cand = sum(1 for r in rs if r["is_candidate"] == 1)
            vals = [r["sign_correct"] for r in rs]
            rng.shuffle(vals)
            a[k] = (vals[:k_cand], vals[k_cand:])
        g = gap(a)
        if np.isfinite(g) and g <= obs:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def load_answer_key():
    if not os.path.exists(ANSWER_KEY):
        return {}
    out = {}
    for r in csv.DictReader(open(ANSWER_KEY, encoding="utf-8")):
        try:
            out[(r["cell"], r["feature"])] = float(r["headroom_pp"])
        except (KeyError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--gatep-draws", type=int, default=25)
    ap.add_argument("--force", action="store_true",
                    help="measure T4 even if GATE P fails (it would be uninterpretable)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cells = load_cells()
    ok, macro = assert_good6(cells)
    if not ok:
        raise SystemExit(f"VALIDITY FAILED: GOOD_6 macro {macro:.4f}")

    print("\n=== GATE P — is U-PCR + sign(rho) sign-sensitive? ===")
    gp = gate_p(cells, n_draws=args.gatep_draws)
    gp_max = float(np.nanmax([r["max_abs_delta"] for r in gp]))
    n_inert = int(sum(1 for r in gp if r["max_abs_delta"] < 1e-12))
    print(f"\n  max |d AUROC| over all cells and draws: {gp_max:.3e}")
    print(f"  cells that are EXACTLY sign-invariant: {n_inert}/{len(gp)}")
    if gp_max < 1e-12 and not args.force:
        raise SystemExit(
            "GATE P FAILED: max |d AUROC| = 0 under a random +-1 vector, so the arm of "
            "record absorbs input sign flips and every sign proposal is a no-op on it, "
            "exactly as for L-SML (Step 213). Refusing to report T4 numbers that could "
            "not bind. Re-run with --force to measure anyway.")

    print(f"\n=== M0 — sign stability and correctness (B = {args.boot}) ===")
    view_rows, cell_rows = m0(cells, n_boot=args.boot)

    key = load_answer_key()
    for r in view_rows:
        hd = key.get((r["cell"], r["feature"]))
        r["headroom_pp"] = hd if hd is not None else ""
        r["is_candidate"] = ("" if hd is None
                             else int(hd >= HEADROOM_CANDIDATE_PP))

    # ---- M0-b: folded null -------------------------------------------------
    null = folded_null(cell_rows)
    obs = float(np.mean([r["folded_agreement"] for r in cell_rows]))
    p_null = float(np.mean(null >= obs))

    # ---- M0-c: is disagreement concentrated on the non-monotone views? -----
    lab = [r for r in view_rows if r["is_candidate"] != ""]
    cand = [r for r in lab if r["is_candidate"] == 1]
    ctrl = [r for r in lab if r["is_candidate"] == 0]
    conc = {}
    if cand and ctrl:
        conc = {
            "n_candidate": len(cand), "n_control": len(ctrl),
            "recovery_candidate": float(np.mean([r["sign_correct"] for r in cand])),
            "recovery_control": float(np.mean([r["sign_correct"] for r in ctrl])),
            "stability_candidate": float(np.mean([r["stability"] for r in cand])),
            "stability_control": float(np.mean([r["stability"] for r in ctrl])),
            "mannwhitney_stability_p": float(stats.mannwhitneyu(
                [r["stability"] for r in cand],
                [r["stability"] for r in ctrl], alternative="less").pvalue),
            "cell_permutation_p": _cell_perm_p(lab),
            "n_cells_contributing_candidates": len({r["cell"] for r in cand}),
            "top3_cells_share_of_candidates": _top3_share(cand),
            "fisher_recovery_p_PSEUDOREPLICATED": float(stats.fisher_exact([
                [sum(r["sign_correct"] for r in cand),
                 len(cand) - sum(r["sign_correct"] for r in cand)],
                [sum(r["sign_correct"] for r in ctrl),
                 len(ctrl) - sum(r["sign_correct"] for r in ctrl)]])[1]),
        }

    # ---- weak cells: is recovery at chance? --------------------------------
    weak = [r for r in cell_rows if r["weak_cell"]]
    healthy = [r for r in cell_rows if not r["weak_cell"]]
    n_correct_weak = int(sum(round(r["sign_recovery"] * r["m"]) for r in weak))
    n_views_weak = int(sum(r["m"] for r in weak))
    weak_block = {
        "n_cells": len(weak),
        "recovery_weak": float(np.mean([r["sign_recovery"] for r in weak])),
        "recovery_healthy": float(np.mean([r["sign_recovery"] for r in healthy])),
        "stability_weak": float(np.mean([r["mean_stability"] for r in weak])),
        "stability_healthy": float(np.mean([r["mean_stability"] for r in healthy])),
        "n_correct_weak": n_correct_weak, "n_views_weak": n_views_weak,
        "mannwhitney_weak_vs_healthy_p": float(stats.mannwhitneyu(
            [r["sign_recovery"] for r in weak],
            [r["sign_recovery"] for r in healthy]).pvalue) if weak and healthy
        else float("nan"),
        "REMOVED_binom_p_weak_vs_half": (
            "withdrawn by the Phase-1 audit. `sign_recovery` is the FOLDED statistic "
            "max(a, 1-a) (pol_res is flipped by the oracle comparison), so it can never "
            "fall below 0.5 and testing it against 0.5 is meaningless; the correct null "
            "is `folded_null_mean` = 0.575, already in this JSON. It also pooled 225 "
            "views from 8 cells into one Bernoulli n (Step 204 D7). It read p=5.7e-50 "
            "for a comparison that is directionally moot anyway (weak 0.953 > healthy "
            "0.936). The unit here is the CELL, so the cell-level Mann-Whitney above "
            "replaces it."),
    }

    # ---- S3: instability as a detector of non-monotone headroom (T2) -------
    s3 = {}
    if lab:
        score = np.array([1.0 - r["stability"] for r in lab])   # unstable = high
        truth = np.array([r["is_candidate"] for r in lab])
        if 0 < truth.sum() < len(truth) and score.std() > 1e-12:
            best_prec, best_thr, best_k = 0.0, float("nan"), 0
            for thr in np.unique(score):
                sel = score >= thr
                if sel.sum() >= 5:
                    prec = float(truth[sel].mean())
                    if prec > best_prec:
                        best_prec, best_thr, best_k = prec, float(thr), int(sel.sum())
            s3 = {
                "n_views": int(len(truth)), "base_rate": float(truth.mean()),
                "auroc_instability_vs_candidate": float(roc_auc_score(truth, score)),
                "spearman_vs_headroom": float(stats.spearmanr(
                    score, [float(r["headroom_pp"]) for r in lab]).statistic),
                "best_precision": best_prec, "at_threshold": best_thr,
                "n_selected": best_k,
                "incumbent_precision": INCUMBENT_PRECISION,
                "trigger_precision": TRIGGER_PRECISION,
                "beats_incumbent": bool(best_prec > INCUMBENT_PRECISION),
                "clears_trigger": bool(best_prec > TRIGGER_PRECISION),
            }

    summary = {
        "n_cells": len(cells), "n_boot": args.boot,
        "gate_p": {"max_abs_delta": gp_max, "n_exactly_invariant": n_inert,
                   "n_cells": len(gp), "n_draws": args.gatep_draws,
                   "arm_is_sign_sensitive": bool(gp_max > 1e-12)},
        "m0_recovery_macro": float(np.mean([r["sign_recovery"] for r in cell_rows])),
        "m0_stability_macro": float(np.mean([r["mean_stability"] for r in cell_rows])),
        "folded_agreement_observed": obs,
        "folded_null_mean": float(null.mean()),
        "folded_null_ci": [float(np.quantile(null, .025)),
                           float(np.quantile(null, .975))],
        "folded_p_vs_null": p_null,
        "concentration_on_nonmonotone": conc,
        "weak_vs_healthy": weak_block,
        "s3_instability_detector": s3,
    }

    with open(os.path.join(OUT, "sign_identifiability_views.csv"), "w",
              newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(view_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(view_rows)
    with open(os.path.join(OUT, "sign_identifiability_cells.csv"), "w",
              newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(cell_rows)
    with open(os.path.join(OUT, "sign_identifiability.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print("\n" + "=" * 72)
    print("T4 — SIGN IDENTIFIABILITY, THE INCUMBENT NUMBER")
    print("=" * 72)
    print(f"  sign(rho) recovers the oracle sign on "
          f"{summary['m0_recovery_macro']*100:.1f}% of views (macro over cells)")
    print(f"  bootstrap self-stability                 "
          f"{summary['m0_stability_macro']*100:.1f}%")
    print(f"  folded agreement {obs*100:.1f}% vs CHANCE "
          f"{null.mean()*100:.1f}% [{np.quantile(null,.025)*100:.1f}, "
          f"{np.quantile(null,.975)*100:.1f}]  p={p_null:.4f}")
    print(f"  weak cells {weak_block['recovery_weak']*100:.1f}% vs healthy "
          f"{weak_block['recovery_healthy']*100:.1f}%  "
          f"(cell-level Mann-Whitney p="
          f"{weak_block['mannwhitney_weak_vs_healthy_p']:.3g})")
    if conc:
        print(f"  non-monotone views: recovery {conc['recovery_candidate']*100:.1f}% "
              f"(n={conc['n_candidate']}) vs control "
              f"{conc['recovery_control']*100:.1f}% (n={conc['n_control']}), "
              f"cell-clustered permutation p={conc['cell_permutation_p']:.3g}")
        print(f"  stability            {conc['stability_candidate']*100:.1f}% vs "
              f"{conc['stability_control']*100:.1f}%, "
              f"Mann-Whitney p={conc['mannwhitney_stability_p']:.3g}")
    if s3:
        print(f"  S3 detector: precision {s3['best_precision']:.3f} vs incumbent "
              f"{INCUMBENT_PRECISION} (base rate {s3['base_rate']:.3f}), "
              f"trigger {TRIGGER_PRECISION} -> "
              f"{'CLEARS' if s3['clears_trigger'] else 'does not clear'}")
    print(f"\nWrote -> {OUT}")


if __name__ == "__main__":
    main()
