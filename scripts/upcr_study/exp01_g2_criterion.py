"""
exp01_g2_criterion.py — Phase B1. Reproduce the paper's Figure 1 on our data.

THE DECISIVE EXPERIMENT, AND IT GATES EVERYTHING AFTER IT
---------------------------------------------------------
U-PCR cannot observe how good a weight vector is, so it picks `g2` (the shared
signal) by a computable proxy: Eq. 20's projection residual RES(q). The paper
validates that proxy in its Figure 1 by drawing the unobservable MSE curve beside
RES(q) and showing the residual's minimum lands where the error is low.

We have never done this. Two things can be wrong and they need separating:

  (a) the search RANGE is wrong. `var_y` is hardcoded to 0.25 while the covariance
      of z-scored features has diagonal EXACTLY 1.0, so the grid only ever explored
      the bottom quarter of the meaningful interval. If RES(q) is still falling at
      the ceiling, the reported g2 is a grid artifact, not an estimate.
  (b) the CRITERION is wrong for us. If argmin RES sits nowhere near argmax AUROC,
      widening the range fixes nothing and the whole g2 machinery is inert.

So we sweep q well past var_y, plot RES(q) and AUROC(q) together, and mark the
current ceiling, the ratio-1 ceiling, both extrema, and the paper's alternative
estimator lambda1/m (Eq. 18).

Labels are used ONLY to draw the AUROC curve — never to fit anything. That is the
same contract the paper's own Figure 1 has with its MSE curve.

Run:  python scripts/upcr_study/exp01_g2_criterion.py
Out:  results/upcr_study/01_g2_criterion/{curves.npz,per_cell.csv,summary.json,index.html}
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import (                                   # noqa: E402
    additive_design, solve_additive, upcr_fit,
)

# q is swept in units of mean(diag C), which for z-scored features is 1.0.
# 0.25 = the legacy ceiling, 1.0 = the paper's "accurate regressor" calibration.
Q_MAX_RATIO = 4.0
N_GRID = 400
LEGACY_RATIO = 0.25


def sweep_cell(cell, loss="l2", lambda2_threshold=0.1):
    """RES(q) and AUROC(q) on one cell, over a range that runs well past var_y.

    Two AUROC curves, deliberately:

      k=1     provably FLAT. Eq. 21 gives `w = (v1.rho / lam1) * v1`, so the
              direction is v1 whatever q is — q only rescales the score, and AUROC
              ignores positive scale. Plotted to make that visible rather than
              leaving it as an assertion.
      k=auto  the DEPLOYED path. The 2-component rule fires whenever
              lambda2 > 0.1*Trace(C), which Step 142 measured on 28/29 cells. Here
              q genuinely moves the answer: it slides weight between v1 and v2,
              because rho(q) tilts toward the all-ones direction and v1 is close to
              it while v2 is orthogonal. So q is really a 2-component-vs-1-component
              shrinkage dial, and the legacy 0.25 ceiling pins it to one end.

    Two residual curves too: k=1 is Eq. 20 as written, k=auto is what our code has
    been doing (deviation #7).

    WHAT THIS FIGURE DOES *NOT* MEASURE (Step 205, and it matters)
    -------------------------------------------------------------
    Everything here is fitted on the FULL feature pool, with no exclusion step.
    That is the right object for the paper's Figure 1, but it is NOT the g2 the
    pipeline returns. `upcr_fit` runs Algorithm 1's exclusion, keeps 8-16 of the
    ~28 features, and then RECALCULATES rho and g2 on the survivors — and it is
    that second fit whose g2 is reported and whose rho feeds Eq. 21.

    The two fits answer the range question differently, and both are right:

        pre-exclusion  (this figure)   g2/var_y 0.05-0.31,  at ceiling  0/25
        post-exclusion (upcr_fit)      g2/var_y 1.0000,     at ceiling 24/25

    On the smaller surviving block the normalised residual falls monotonically
    across [0, var_y], so the argmin sits on the grid edge. So "the range never
    binds" is true of the curve drawn below and FALSE of the deployed estimate.
    The practical conclusion survives either way: un-pinning it (scale_ratio
    0.25 -> 1.0, difficulty gate off) is -0.28pp, 12W/13L — a wash.
    `post_exclusion_at_ceiling` below records this per cell so the distinction
    cannot be lost again.
    """
    F = cell["V"].T
    m, n = F.shape
    C = (F @ F.T) / n
    diag_mean = float(np.mean(np.diag(C)))

    A, pairs = additive_design(m)
    b = np.array([C[i, j] for i, j in pairs], dtype=float)
    rho0 = solve_additive(A, b, loss=loss)          # Eq. 15 once (Eq. 16 shift)

    evals, evecs = eigh(C)
    lam1, lam2 = float(evals[-1]), float(evals[-2])
    v1, v2 = evecs[:, -1], evecs[:, -2]
    trace_C = float(np.trace(C))
    lambda2_frac = lam2 / (trace_C + 1e-12)
    k_auto = 2 if lambda2_frac > lambda2_threshold else 1

    qs = np.linspace(0.0, Q_MAX_RATIO * diag_mean, N_GRID)
    res1 = np.empty(N_GRID)
    res_auto = np.empty(N_GRID)
    auc1 = np.empty(N_GRID)
    auc_auto = np.empty(N_GRID)
    B1 = v1[:, None]
    Ba = evecs[:, -k_auto:]
    for i, q in enumerate(qs):
        rho = rho0 + 0.5 * q
        p1 = B1 @ (B1.T @ rho)
        pa = Ba @ (Ba.T @ rho)
        nrm = np.linalg.norm(rho) + 1e-12
        res1[i] = np.linalg.norm(rho - p1) / nrm
        res_auto[i] = np.linalg.norm(rho - pa) / nrm
        w1 = (v1 @ rho) / (lam1 + 1e-12) * v1
        auc1[i] = S.auroc_from_score(cell, w1 @ F)
        wa = w1 if k_auto == 1 else w1 + (v2 @ rho) / (lam2 + 1e-12) * v2
        auc_auto[i] = S.auroc_from_score(cell, wa @ F)

    # the DEPLOYED estimate, for contrast with the curve above
    dep = upcr_fit(F, scale_ratio=LEGACY_RATIO)

    return {
        "ratios": qs / diag_mean, "res": res1, "res_auto": res_auto,
        "auc": auc_auto, "auc_k1": auc1, "k_auto": k_auto,
        "lambda2_frac": lambda2_frac,
        "diag_mean": diag_mean, "lam1_over_m": lam1 / m / diag_mean,
        "g2_pre_exclusion_frac": float(dep.meta["g2_full_pool"] / dep.var_y),
        "g2_post_exclusion_frac": float(dep.g2_frac_of_var_y),
        "post_exclusion_at_ceiling": bool(dep.g2_at_ceiling),
        "n_kept": int(dep.meta["n_kept"]), "m_pool": int(dep.meta["m"]),
    }


def main():
    out = S.outdir("01_g2_criterion")
    cells = S.load()
    S.validity_check(cells)

    curves, rows = {}, []
    for ck, cell in cells.items():
        c = sweep_cell(cell)
        curves[ck] = c
        # `res` is Eq.20 as written (k=1); `res_auto` is what our code actually
        # does (k follows the weight count, = 2 on 24/25 cells). Report BOTH — the
        # deployed number is res_auto, and using res here understated the regret
        # as 0.76pp when the deployed criterion gives 1.21pp. Found in review.
        r, a, rat = c["res_auto"], c["auc"], c["ratios"]
        r_paper = c["res"]

        in_legacy = rat <= LEGACY_RATIO
        argmin_legacy = float(rat[in_legacy][int(np.argmin(r[in_legacy]))])
        argmin_full = float(rat[int(np.argmin(r))])
        argmin_paper = float(rat[int(np.argmin(r_paper))])
        argmax_auc = float(rat[int(np.nanargmax(a))])
        # is the legacy answer just the grid's edge?
        pinned = argmin_legacy >= LEGACY_RATIO * 0.98

        finite = np.isfinite(a)
        rho_ra = (spearmanr(r[finite], a[finite]).correlation
                  if finite.sum() > 10 else float("nan"))
        # monotonicity IN q — the statistic the write-up actually claimed. The
        # earlier version used |Spearman(RES, AUROC)| > 0.99, which measures
        # something else entirely and ignores sign (3 cells sit at +0.9999, i.e.
        # the criterion points the WRONG way there).
        da = np.diff(a[finite])
        mono_in_q = bool(np.all(da <= 1e-12) or np.all(da >= -1e-12))

        rows.append({
            "cell": ck,
            "argmin_RES_within_legacy_range": argmin_legacy,
            "legacy_pinned_at_ceiling": pinned,
            "argmin_RES_full_range": argmin_full,
            "argmin_RES_paper_k1": argmin_paper,
            "argmax_AUROC": argmax_auc,
            "auroc_at_legacy_pick": float(a[in_legacy][int(np.argmin(r[in_legacy]))]),
            "auroc_at_full_pick": float(a[int(np.argmin(r))]),
            "auroc_best_possible": float(np.nanmax(a)),
            "auroc_spread_over_q": float(np.nanmax(a) - np.nanmin(a)),
            "auroc_spread_k1": float(np.nanmax(c["auc_k1"]) - np.nanmin(c["auc_k1"])),
            "k_auto": int(c["k_auto"]),
            "lambda2_frac": float(c["lambda2_frac"]),
            "lambda1_over_m": float(c["lam1_over_m"]),
            "spearman_RES_vs_AUROC": float(rho_ra),
            "auroc_monotone_in_q": mono_in_q,
            "criterion_points_wrong_way": bool(rho_ra > 0),
            # the deployed estimate — a different fit from the curve above
            "g2_pre_exclusion_frac_of_var_y": c["g2_pre_exclusion_frac"],
            "g2_post_exclusion_frac_of_var_y": c["g2_post_exclusion_frac"],
            "post_exclusion_at_ceiling": c["post_exclusion_at_ceiling"],
            "n_kept_after_exclusion": c["n_kept"], "m_pool": c["m_pool"],
        })

    S.save_csv(os.path.join(out, "per_cell.csv"), rows)
    np.savez_compressed(
        os.path.join(out, "curves.npz"),
        **{f"{k}__{f}": curves[k][f] for k in curves
           for f in ("ratios", "res", "res_auto", "auc", "auc_k1")})

    n_pinned = sum(1 for r in rows if r["legacy_pinned_at_ceiling"])
    spread = np.array([r["auroc_spread_over_q"] for r in rows])
    d_legacy = np.array([r["auroc_best_possible"] - r["auroc_at_legacy_pick"]
                         for r in rows])
    d_full = np.array([r["auroc_best_possible"] - r["auroc_at_full_pick"]
                       for r in rows])
    corr = np.array([r["spearman_RES_vs_AUROC"] for r in rows])
    wl = S.wl_wilcoxon([r["auroc_at_legacy_pick"] for r in rows],
                       [r["auroc_at_full_pick"] for r in rows])

    summary = {
        "n_cells": len(rows),
        "n_legacy_pinned_at_ceiling": int(n_pinned),
        "mean_auroc_spread_over_q": float(spread.mean()),
        "max_auroc_spread_over_q": float(spread.max()),
        "mean_regret_legacy_range": float(d_legacy.mean()),
        "mean_regret_full_range": float(d_full.mean()),
        "mean_spearman_RES_vs_AUROC": float(np.nanmean(corr)),
        "n_cells_criterion_correct_direction": int((corr < 0).sum()),
        "widen_range_effect": wl,
    }

    print("\n" + "=" * 72)
    print("B1 RESULTS")
    print("=" * 72)
    print(f"  legacy grid pinned at its ceiling in "
          f"{n_pinned}/{len(rows)} cells")
    print(f"  AUROC spread across the whole q range: mean "
          f"{spread.mean()*100:.2f}pp, max {spread.max()*100:.2f}pp")
    print(f"  regret vs best-possible-q: legacy range "
          f"{d_legacy.mean()*100:.2f}pp, full range {d_full.mean()*100:.2f}pp")
    print(f"  widening the range: {S.fmt_wl(wl)}")
    print(f"  Spearman(RES, AUROC): mean {np.nanmean(corr):+.3f} "
          f"(want NEGATIVE - low residual should mean high AUROC); "
          f"correct direction in {(corr < 0).sum()}/{len(rows)} cells")

    n_moved = sum(1 for r in rows
                  if abs(r["argmin_RES_within_legacy_range"]
                         - r["argmin_RES_full_range"]) > 1e-9)
    # the deployed g2 is a POST-EXCLUSION refit, and it is a different story
    n_post_pinned = sum(1 for r in rows if r["post_exclusion_at_ceiling"])
    summary["n_post_exclusion_pinned_at_ceiling"] = int(n_post_pinned)
    summary["g2_post_exclusion_frac_of_var_y_median"] = float(
        np.median([r["g2_post_exclusion_frac_of_var_y"] for r in rows]))
    summary["g2_pre_exclusion_frac_of_var_y_median"] = float(
        np.median([r["g2_pre_exclusion_frac_of_var_y"] for r in rows]))
    summary["lambda2_frac_median"] = float(
        np.median([r["lambda2_frac"] for r in rows]))
    monotone = int(sum(1 for r in rows if r["auroc_monotone_in_q"]))
    wrong_way = int(sum(1 for r in rows if r["criterion_points_wrong_way"]))
    summary["n_cells_criterion_points_wrong_way"] = wrong_way
    at_zero = sum(1 for r in rows if r["argmax_AUROC"] <= 1e-9)
    summary["n_argmin_moved_when_range_widened"] = int(n_moved)
    summary["n_cells_auroc_monotone_in_q"] = monotone
    summary["n_cells_auroc_best_at_q_zero"] = at_zero

    if n_moved == 0:
        verdict = (
            f"THE RANGE DOES NOT BIND ON THE PRE-EXCLUSION FIT - WHICH IS THE ONLY "
            f"FIT THIS FIGURE DRAWS. On the full pool the residual minimum sits at "
            f"q~0.01-0.08 in all {len(rows)} cells, far below the legacy 0.25 "
            f"ceiling, and widening the range 16x moves the chosen q in "
            f"{n_moved}/{len(rows)} cells. BUT THE DEPLOYED g2 IS A DIFFERENT FIT: "
            f"upcr_fit excludes weak experts and RECALCULATES rho and g2 on the "
            f"{np.mean([r['n_kept_after_exclusion'] for r in rows]):.0f} survivors "
            f"of ~{np.mean([r['m_pool'] for r in rows]):.0f}, and there g2 sits "
            f"exactly on the ceiling in {n_post_pinned}/{len(rows)} cells "
            f"(g2/var_y median {np.median([r['g2_post_exclusion_frac_of_var_y'] for r in rows]):.4f}). "
            f"So an unqualified 'the range never binds' is WRONG - it binds on the "
            f"estimate the pipeline actually returns, and Step 204's B1 is narrowed "
            f"to the pre-exclusion fit accordingly (Step 205). The PRACTICAL "
            f"conclusion survives: un-pinning it (scale_ratio 0.25 -> 1.0 with the "
            f"difficulty gate off) is -0.28pp, 12W/13L, a wash, so var_y=0.25 still "
            f"costs ~nothing. And g2 is NOT the component-count dial that was "
            f"suspected: auto_components keys off lambda2_frac > lambda2_threshold, "
            f"computed before g2 and independently of it (median lambda2_frac "
            f"{np.median([r['lambda2_frac'] for r in rows]):.4f} against a threshold "
            f"of 0.1 - that threshold is the real dial, see exp07). g2 IS a large "
            f"lever ({spread.mean()*100:.2f}pp mean AUROC swing, "
            f"{spread.max()*100:.2f}pp max) and the criterion mostly points the "
            f"right way (wrong direction in {wrong_way}/{len(rows)} cells), but it "
            f"leaves {d_legacy.mean()*100:.2f}pp on the table against the oracle q. "
            f"AUROC is strictly monotone in q in only {monotone}/{len(rows)} cells "
            f"(an earlier version of this script reported {len(rows)-1}/{len(rows)} "
            f"by computing monotonicity in RES rather than in q), so the curve does "
            f"have interior structure - it is simply structure this criterion does "
            f"not exploit.")
    elif np.nanmean(corr) < -0.2:
        verdict = ("CRITERION WORKS and the range matters - widen it to ratio 1.0 "
                   "and keep the g2 arms in Phase C.")
    else:
        verdict = ("CRITERION DOES NOT TRACK PERFORMANCE - argmin RES is not near "
                   "argmax AUROC, so widening the range fixes nothing. The g2 "
                   "machinery is inert for us; Phase D is where the effort belongs.")
    print(f"\n  VERDICT: {verdict}")
    summary["verdict"] = verdict
    S.save_json(os.path.join(out, "summary.json"), summary)
    render(out, rows, curves, summary)
    print(f"\nWrote -> {out}")


def render(out, rows, curves, summary):
    body = []
    body.append("<h2>What this figure is</h2>")
    body.append(
        "<p>The paper picks <code>g2</code> — the shared-signal level — by "
        "minimising a computable residual, and validates that choice in its "
        "Figure 1 by drawing the unobservable error curve beside it. This is that "
        "figure on our data, with AUROC standing in for the paper's MSE. "
        "<b>Labels draw the AUROC curve only; nothing is fitted with them.</b> "
        "The x-axis is <code>q</code> in units of the covariance diagonal, so "
        "<b>0.25 is the ceiling our code has been using</b> and 1.0 is the "
        "paper's calibration.</p>")

    # a handful of representative cells, full detail
    order = sorted(rows, key=lambda r: -r["auroc_spread_over_q"])
    for r in order[:6]:
        ck = r["cell"]
        c = curves[ck]
        rr, aa, rat = c["res"], c["auc"], c["ratios"]
        step = max(1, len(rat) // 120)
        body.append(f"<h3>{S.plain_cell(ck)}</h3>")
        body.append(S.line_chart(
            [("projection residual (computable, Eq. 20)", rat[::step], rr[::step]),
             ("AUROC, deployed 2-component path", rat[::step], aa[::step]),
             ("AUROC, 1 component (flat by construction)",
              rat[::step], c["auc_k1"][::step])],
            "q  /  mean(diag C)", "value",
            hlines=[("legacy ceiling 0.25", float(rr[np.searchsorted(rat, 0.25)])),
                    ("ratio 1.0", float(rr[np.searchsorted(rat, 1.0)]))],
            xticks=[0, 0.25, 1, 2, 3, 4]))
        body.append(
            f"<p class='note'>residual minimum at q={r['argmin_RES_full_range']:.2f}"
            f" &middot; AUROC maximum at q={r['argmax_AUROC']:.2f}"
            f" &middot; AUROC spread over the whole range "
            f"{r['auroc_spread_over_q']*100:.2f}pp</p>")

    body.append("<h2>Does the criterion point where performance is?</h2>")
    body.append(S.scatter_chart(
        [r["argmin_RES_full_range"] for r in rows],
        [r["argmax_AUROC"] for r in rows],
        "q at the residual minimum (what the method picks)",
        "q at the AUROC maximum (what it should pick)",
        diagonal=True))
    body.append(
        "<p class='note'>If the criterion worked, these would lie on the diagonal."
        "</p>")

    body.append("<h2>How much does g2 move the answer at all?</h2>")
    body.append(S.bar_chart(
        [S.plain_cell(r["cell"]) for r in order],
        [r["auroc_spread_over_q"] for r in order],
        "AUROC spread across the entire q range (pp)", value_fmt="{:.4f}"))

    body.append("<h2>Per cell</h2>")
    body.append(S.html_table(
        ["test set", "argmin RES (legacy range)", "pinned at ceiling?",
         "argmin RES (full)", "argmax AUROC", "AUROC spread", "Spearman(RES,AUROC)"],
        [[S.plain_cell(r["cell"]),
          f"{r['argmin_RES_within_legacy_range']:.3f}",
          "YES" if r["legacy_pinned_at_ceiling"] else "no",
          f"{r['argmin_RES_full_range']:.3f}",
          f"{r['argmax_AUROC']:.3f}",
          f"{r['auroc_spread_over_q']*100:.2f}pp",
          f"{r['spearman_RES_vs_AUROC']:+.3f}"] for r in order],
        numeric_cols=(1, 3, 4, 5, 6)))

    S.write_page(
        os.path.join(out, "index.html"),
        "Does the g2 criterion work on our data?",
        "U-PCR study, Phase B1 — the paper's Figure 1, reproduced",
        [f"On the <b>pre-exclusion</b> fit drawn below, the legacy search range was "
         f"pinned at its own ceiling in "
         f"<b>{summary['n_legacy_pinned_at_ceiling']}/{summary['n_cells']}</b> cells "
         f"&mdash; but the g2 the pipeline actually returns is a <b>post-exclusion "
         f"refit</b>, and that one sits on the ceiling in "
         f"<b>{summary['n_post_exclusion_pinned_at_ceiling']}/{summary['n_cells']}</b>. "
         f"Un-pinning it is a wash (&minus;0.28pp, 12W/13L), so the range costs "
         f"~nothing either way.",
         f"Across the full range, g2 moves AUROC by "
         f"<b>{summary['mean_auroc_spread_over_q']*100:.2f}pp</b> on average "
         f"(max {summary['max_auroc_spread_over_q']*100:.2f}pp).",
         f"Spearman(residual, AUROC) = "
         f"<b>{summary['mean_spearman_RES_vs_AUROC']:+.3f}</b>; it should be "
         f"negative, and is in "
         f"{summary['n_cells_criterion_correct_direction']}/{summary['n_cells']} cells.",
         f"<b>{summary['verdict']}</b>"],
        "".join(body))


if __name__ == "__main__":
    main()
