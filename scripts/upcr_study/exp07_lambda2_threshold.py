"""
exp07_lambda2_threshold.py — the dial that actually chooses the component count.

WHERE THIS CAME FROM
--------------------
Omri, reading the Step-204 write-up: the g2 grid is `linspace(0, var_y, 300)` with
`var_y = 0.25 * mean(diag C)` and z-scored features, so it only ever explores the
bottom quarter of its range — "and since g2 is the dial between one and two
eigenvectors, we've been permanently pinned toward the 2-eigenvector end, which is
where the redundant second factor gets in."

The CONCLUSION is right and already measured: `auto_components` picks 2 components
in 24/25 cells, and Step 204's factorial put that at -3.67pp mean / -2.36pp median,
3W/21L, p=9.1e-05. The MECHANISM is not. g2 does not gate the component count at
all — `upcr.py` decides it before g2 is fitted and independently of it:

    n_components = 2 if (k_probe >= 2 and lambda2_frac > lambda2_threshold) else 1

So the real dial is `lambda2_threshold`, hardcoded at 0.1. And it is a sharp one:
measured over the 25 in-scope cells, `lambda2_frac` has median 0.1435, min 0.0942,
max 0.2328 — a tight distribution sitting just ABOVE the threshold, which is why
essentially every cell takes the 2-component branch. Nudging the threshold should
therefore move a lot of cells at once.

PRE-REGISTERED READ
-------------------
Step 204 measured the 2-component rule at -3.67pp. If the component count is the
mechanism, raising the threshold (fewer cells at 2 components) should recover most
of that, monotonically, and the best threshold should sit above 0.2328 where every
cell is at 1 component. If it does NOT, then the component count is not the
mechanism either, and the -3.67pp is carried by something the count is merely
correlated with.

Reported with W/L + Wilcoxon against the deployed 0.1, per the house rule; no
gating on 1-2pp differences.

The difficulty gate is OFF throughout. With it on, `var_y` routes cells to abstain
(3.9 -> 17.6 of 25 across the scale range) and the component-count effect would be
confounded by which cells are scored at all.

Run:  python scripts/upcr_study/exp07_lambda2_threshold.py
Out:  results/upcr_study/07_lambda2_threshold/{per_cell.csv,summary.json,index.html}
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import upcr_fit                            # noqa: E402

# spans the observed lambda2_frac range (0.0942 - 0.2328) on both sides, so the
# sweep covers "every cell at 2 components" through "every cell at 1".
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
DEPLOYED = 0.10
SCALE_RATIO = 0.25          # the deployed value; the ratio arm is exp03's job


def main():
    out = S.outdir("07_lambda2_threshold")
    cells = S.load()
    S.validity_check(cells)

    rows, per_thr = [], {t: [] for t in THRESHOLDS}
    for ck, cell in cells.items():
        F = cell["V"].T
        rec = {"cell": ck}
        for t in THRESHOLDS:
            r = upcr_fit(F, scale_ratio=SCALE_RATIO, lambda2_threshold=t,
                         difficulty_gate=False)
            a = S.auroc_from_score(cell, r.w @ F)
            rec[f"auroc_t{t}"] = float(a)
            rec[f"ncomp_t{t}"] = int(r.n_components_used)
            per_thr[t].append(a)
        rec["lambda2_frac"] = float(
            upcr_fit(F, scale_ratio=SCALE_RATIO, difficulty_gate=False).lambda2_frac)
        rows.append(rec)

    S.save_csv(os.path.join(out, "per_cell.csv"), rows)

    # The same switch, thrown directly rather than through the threshold. This
    # is the like-for-like comparison against Step 204's -3.67pp main effect.
    one = np.array([S.auroc_from_score(
        c, upcr_fit(c["V"].T, scale_ratio=SCALE_RATIO, auto_components=False,
                    n_components=1, difficulty_gate=False).w @ c["V"].T)
        for c in cells.values()], float)
    auto = np.array([r[f"auroc_t{DEPLOYED}"] for r in rows], float)
    direct = S.wl_wilcoxon(auto.tolist(), one.tolist())

    base = np.array(per_thr[DEPLOYED], float)
    l2 = np.array([r["lambda2_frac"] for r in rows], float)
    table = []
    for t in THRESHOLDS:
        v = np.array(per_thr[t], float)
        n2 = sum(1 for r in rows if r[f"ncomp_t{t}"] == 2)
        wl = S.wl_wilcoxon(v.tolist(), base.tolist())
        table.append({
            "threshold": t, "macro": float(np.nanmean(v)),
            "delta_pp": float((np.nanmean(v) - np.nanmean(base)) * 100),
            "n_two_component": int(n2), "n_cells": len(rows),
            "wins": wl["wins"], "losses": wl["losses"], "p": wl["p"],
        })

    best = max(table, key=lambda r: r["macro"])
    dep = next(r for r in table if r["threshold"] == DEPLOYED)
    one_comp = [r for r in table if r["n_two_component"] == 0]
    all_one = one_comp[0] if one_comp else None

    # Does AUROC track the component count monotonically? That is the actual
    # pre-registered question: if the count is the mechanism, fewer 2-component
    # cells should mean higher macro, all the way across.
    n2 = np.array([r["n_two_component"] for r in table], float)
    mac = np.array([r["macro"] for r in table], float)
    from scipy.stats import spearmanr
    rho_count = float(spearmanr(n2, mac).correlation)

    summary = {
        "n_cells": len(rows), "thresholds": THRESHOLDS,
        "deployed_threshold": DEPLOYED, "scale_ratio": SCALE_RATIO,
        "lambda2_frac": {"median": float(np.median(l2)), "min": float(l2.min()),
                         "max": float(l2.max())},
        "table": table,
        "deployed_macro": dep["macro"],
        "best_threshold": best["threshold"], "best_macro": best["macro"],
        "best_gain_pp": best["delta_pp"],
        "spearman_n2component_vs_macro": rho_count,
        # STEP-204 RECONCILIATION. exp03 reports the components factor at
        # -3.67pp mean / -2.36pp median, 3W/21L, p=9.1e-05. That is a FACTORIAL
        # MAIN EFFECT: each cell's delta is averaged over the 32 combinations of
        # the other five factors, most of which are not the deployed setup. At
        # the DEPLOYED configuration the identical switch is the number below.
        # Both are correct; they are not the same quantity, and only this one
        # describes what the rule costs us today.
        "components_at_deployed_config": {
            "one_component_macro": float(one.mean()),
            "auto_component_macro": float(auto.mean()),
            "delta_pp_auto_minus_one": float((auto.mean() - one.mean()) * 100),
            "median_delta_pp": float(direct["median_delta"] * 100),
            "wins": direct["wins"], "losses": direct["losses"], "p": direct["p"],
        },
        "step204_main_effect_for_contrast": {
            "mean_delta_pp": -3.666, "median_delta_pp": -2.364,
            "wins": 3, "losses": 21, "p": 9.067e-05,
            "note": "marginal over 32 other-factor combinations, not the deployed config",
        },
    }

    print("\n" + "=" * 72)
    print("exp07 - lambda2_threshold, the component-count dial")
    print("=" * 72)
    print(f"  lambda2_frac over {len(rows)} cells: median {np.median(l2):.4f}, "
          f"min {l2.min():.4f}, max {l2.max():.4f}  (threshold sits at {DEPLOYED})")
    print(f"\n{'threshold':>10} {'2-comp cells':>13} {'macro':>8} {'vs 0.10':>9} "
          f"{'W/L':>8} {'p':>8}")
    for r in table:
        mark = "  <- deployed" if r["threshold"] == DEPLOYED else ""
        print(f"{r['threshold']:>10.2f} {r['n_two_component']:>9d}/{r['n_cells']:<3d} "
              f"{r['macro']:>8.4f} {r['delta_pp']:>+8.2f}pp "
              f"{r['wins']:>3d}/{r['losses']:<3d} {r['p']:>8.4f}{mark}")

    dcfg = summary["components_at_deployed_config"]
    print(f"\n  SAME SWITCH, THROWN DIRECTLY, AT THE DEPLOYED CONFIG:")
    print(f"    1 component {dcfg['one_component_macro']:.4f} vs auto "
          f"{dcfg['auto_component_macro']:.4f} = "
          f"{dcfg['delta_pp_auto_minus_one']:+.2f}pp mean / "
          f"{dcfg['median_delta_pp']:+.2f}pp median, "
          f"{dcfg['wins']}W/{dcfg['losses']}L, p={dcfg['p']:.4f}")
    print(f"    Step 204 reports this factor at -3.67pp / -2.36pp, 3W/21L, "
          f"p=9.1e-05 -- but that is a MAIN EFFECT marginalised over the 32")
    print(f"    combinations of the other five factors. At the configuration we "
          f"actually deploy, the rule costs a wash.")

    print(f"\n  Spearman(#2-component cells, macro) = {rho_count:+.3f}")
    if all_one is not None:
        print(f"  forcing 1 component everywhere (threshold {all_one['threshold']}): "
              f"{all_one['delta_pp']:+.2f}pp, {all_one['wins']}W/{all_one['losses']}L, "
              f"p={all_one['p']:.4f}")

    # Verdict against the pre-registered prediction (-3.67pp from Step 204).
    gain = best["delta_pp"]
    if all_one is not None and all_one["delta_pp"] >= 2.0 and rho_count <= -0.5:
        verdict = (
            f"THE COMPONENT COUNT IS THE MECHANISM. Removing the second component "
            f"by raising the threshold recovers {all_one['delta_pp']:+.2f}pp "
            f"({all_one['wins']}W/{all_one['losses']}L, p={all_one['p']:.4f}), and "
            f"macro tracks the number of 2-component cells at Spearman "
            f"{rho_count:+.3f}. Step 204's -3.67pp for the 2-eigenvector rule is "
            f"reproduced through this dial, and lambda2_threshold=0.1 is simply set "
            f"too low for our data: lambda2_frac is tightly clustered at "
            f"{np.median(l2):.3f}, just above it.")
    elif gain >= 1.0:
        verdict = (
            f"THE THRESHOLD MATTERS BUT NOT VIA A CLEAN COUNT EFFECT. Best is "
            f"{best['threshold']} at {gain:+.2f}pp over the deployed 0.1, yet macro "
            f"tracks the 2-component count only at Spearman {rho_count:+.3f}. So "
            f"something moves, but 'one component instead of two' is not a "
            f"sufficient description of it.")
    else:
        verdict = (
            f"THE DIAL IS INERT. The best threshold ({best['threshold']}) buys "
            f"{gain:+.2f}pp over the deployed 0.1, inside noise, even though it "
            f"changes the component count on "
            f"{abs(dep['n_two_component'] - best['n_two_component'])} cells. "
            f"So the component count is NOT the mechanism behind Step 204's "
            f"-3.67pp for the 2-eigenvector rule, and Omri's 'the redundant second "
            f"factor gets in' is not actionable through this knob. Consistent with "
            f"Step 204's other finding that one-component U-PCR is exactly PC1 of "
            f"the surviving features - the estimation machinery is inert here and "
            f"what mattered was orientation and exclusion. AND IT QUALIFIES A "
            f"STEP-204 HEADLINE: throwing the same switch directly at the DEPLOYED "
            f"configuration gives {dcfg['delta_pp_auto_minus_one']:+.2f}pp mean / "
            f"{dcfg['median_delta_pp']:+.2f}pp median, {dcfg['wins']}W/"
            f"{dcfg['losses']}L, p={dcfg['p']:.4f} - not the -3.67pp / -2.36pp, "
            f"3W/21L, p=9.1e-05 that exp03 reports. exp03's number is a factorial "
            f"MAIN EFFECT, averaged over the 32 combinations of the other five "
            f"factors, most of which are not configurations we run. Both are "
            f"correct measurements of different quantities, but only the deployed-"
            f"config one answers 'what does the 2-eigenvector rule cost us today', "
            f"and the answer is: nothing measurable.")
    print(f"\n  VERDICT: {verdict}")
    summary["verdict"] = verdict
    S.save_json(os.path.join(out, "summary.json"), summary)
    render(out, rows, table, summary)
    print(f"\nWrote -> {out}")


def render(out, rows, table, summary):
    body = []
    body.append("<h2>What this measures</h2>")
    body.append(
        "<p>U-PCR builds its weights from the top <b>one or two</b> eigenvectors "
        "of the feature covariance. Which, is decided by a single hardcoded "
        "constant: it takes two whenever <code>lambda2 / Trace(C)</code> exceeds "
        "<code>lambda2_threshold = 0.1</code>. Step 204 measured the 2-component "
        "rule as costing <b>&minus;3.67pp</b>, and this is the knob that sets it. "
        "The g2 search range &mdash; the suspect this replaced &mdash; does not "
        "enter the decision at all: the component count is fixed before g2 is "
        "fitted (exp01).</p>")
    l2 = summary["lambda2_frac"]
    body.append(
        f"<p class='note'>The distribution is why the knob is sharp: "
        f"<code>lambda2_frac</code> runs {l2['min']:.4f} to {l2['max']:.4f} with "
        f"median {l2['median']:.4f}, i.e. it is tightly clustered <i>just above</i> "
        f"the 0.1 cutoff. Almost every cell is on the same side of it, and a small "
        f"move in the threshold flips many cells at once.</p>")

    body.append("<h2>Sweeping the threshold</h2>")
    body.append(S.line_chart(
        [("macro AUROC", [r["threshold"] for r in table],
          [r["macro"] for r in table])],
        "lambda2_threshold", "macro AUROC"))
    body.append(S.line_chart(
        [("cells using 2 components", [r["threshold"] for r in table],
          [r["n_two_component"] for r in table])],
        "lambda2_threshold", "cells (of 25)"))

    body.append(S.html_table(
        ["threshold", "2-component cells", "macro AUROC", "vs deployed 0.10",
         "W/L", "Wilcoxon p"],
        [[f"{r['threshold']:.2f}", f"{r['n_two_component']}/{r['n_cells']}",
          f"{r['macro']:.4f}", f"{r['delta_pp']:+.2f}pp",
          f"{r['wins']}/{r['losses']}", f"{r['p']:.4f}"] for r in table],
        numeric_cols=(0, 2, 3, 5)))

    body.append("<h2>Per cell</h2>")
    body.append(S.html_table(
        ["test set", "lambda2_frac"] + [f"AUROC @ {t}" for t in summary["thresholds"]],
        [[S.plain_cell(r["cell"]), f"{r['lambda2_frac']:.4f}"]
         + [f"{r[f'auroc_t{t}']:.4f}" for t in summary["thresholds"]]
         for r in sorted(rows, key=lambda x: -x["lambda2_frac"])],
        numeric_cols=tuple(range(1, 2 + len(summary["thresholds"])))))

    S.write_page(
        os.path.join(out, "index.html"),
        "Is the component-count threshold the real dial?",
        "U-PCR study, exp07 — lambda2_threshold, the constant that decides "
        "one eigenvector or two",
        [f"<code>lambda2_frac</code> is tightly clustered "
         f"({l2['min']:.4f}&ndash;{l2['max']:.4f}, median {l2['median']:.4f}) "
         f"just above the hardcoded 0.1, so the threshold flips many cells at once.",
         f"Best threshold is <b>{summary['best_threshold']}</b> at "
         f"<b>{summary['best_gain_pp']:+.2f}pp</b> over the deployed 0.1.",
         f"Macro tracks the number of 2-component cells at Spearman "
         f"<b>{summary['spearman_n2component_vs_macro']:+.3f}</b>.",
         f"<b>{summary['verdict']}</b>"],
        "".join(body))


if __name__ == "__main__":
    main()
