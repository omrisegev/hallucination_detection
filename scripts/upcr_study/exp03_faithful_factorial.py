"""
exp03_faithful_factorial.py — Phase C. Every paper-faithful flag, measured.

WHY THIS EXISTS
---------------
Omri, this session: "I prefer rerunning and getting the answers with the algorithm
fixed and loyal to the paper. Measurements taken before that are not reliable."

That is right, and it applies to three results I had wrongly treated as settled:

  R2 (2 eigenvectors)  Step 142 measured it, but with the g2 search capped at 0.25
                       — and g2 IS the dial between 1 and 2 components, so the
                       measurement was confounded by the thing it was testing.
  R4 (absolute loss)   Step 203's robust-IRLS was a DIFFERENT estimator on the
                       L-SML path (loadings from the off-block covariance), not
                       U-PCR's Eq. 15 pair-fit. Never actually measured here.
  scale ratio          var_y / mean(diag C), pinned at 0.25 with no justification.

FACTORS (2^6 = 64 configurations x 25 test sets)
------------------------------------------------
  scale_ratio        0.25 (legacy)      vs 1.0 (the paper's calibration)
  loss               squared (legacy)   vs absolute (Remark 1)          <- R4
  g2_projection_k    1 (Eq. 20)         vs match-the-weights (legacy)
  components         1                  vs auto 2 when lam2 > 0.1 Trace <- R2
  exclusion          on (Algorithm 1)   vs off
  difficulty gate    off (legacy)       vs on (Algorithm 1 STOP)

REPORTING
---------
MAIN EFFECTS, with wins/losses and Wilcoxon — never best-of-64. 64 configs x 25
cells against ~1.5pp of headroom is a textbook winner's-curse setup (Step 193).
And per Omri: do not gate on 1-2pp differences.

Phase B1 pre-registered a caveat that matters for reading the scale_ratio row: the
g2 argmin does not move when the range widens (0/25 cells), so scale_ratio cannot
act through the g2 pick. If it moves anything, it acts through the EXCLUSION
thresholds — making it a feature-selection knob, not a weight-estimation fix. The
per-config record of how many features survived is what distinguishes those.

Run:  python scripts/upcr_study/exp03_faithful_factorial.py
Out:  results/upcr_study/03_faithful_factorial/{per_config.csv,main_effects.csv,
      summary.json,index.html}
"""
import os
import sys
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import upcr_fit                            # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous             # noqa: E402

FACTORS = {
    "scale_ratio": [0.25, 1.0],
    "loss": ["l2", "l1"],
    "g2_projection_k": [1, "match"],
    "components": ["one", "auto"],
    "exclusion": [True, False],
    "difficulty_gate": [False, True],
}
LEGACY = {"scale_ratio": 0.25, "loss": "l2", "g2_projection_k": "match",
          "components": "auto", "exclusion": True, "difficulty_gate": False}
FAITHFUL = {"scale_ratio": 1.0, "loss": "l1", "g2_projection_k": 1,
            "components": "auto", "exclusion": True, "difficulty_gate": True}


def run_config(cell, cfg):
    F = cell["V"].T
    res = upcr_fit(
        F,
        scale_ratio=cfg["scale_ratio"],
        loss=cfg["loss"],
        g2_projection_k=cfg["g2_projection_k"],
        n_components=1,
        auto_components=(cfg["components"] == "auto"),
        exclusion=cfg["exclusion"],
        difficulty_gate=cfg["difficulty_gate"],
        on_abstain="mean",
        simple_avg_fallback=True,
        recompute_after_exclusion=True,
    )
    return S.auroc_from_score(cell, res.w @ F), res


def main():
    out = S.outdir("03_faithful_factorial")
    cells = S.load()
    S.validity_check(cells)

    keys = list(FACTORS)
    configs = [dict(zip(keys, v)) for v in itertools.product(*FACTORS.values())]
    print(f"{len(configs)} configurations x {len(cells)} test sets")

    # References at EVERY loading scale, with GOOD_6 and all-30 computed at the
    # SAME scale as each other. The earlier version took all-30 at 'complete' and
    # GOOD_6 via `good6_score`, which pins the default 'unit' — two scales in one
    # table. Found in review.
    refs = {sc: S.reference_macros(cells, sc) for sc in S.SCALES}

    rows = []
    per_cell_auroc = {}
    cell_keys = list(cells)
    for ci, cfg in enumerate(configs):
        aurocs, kept, abstains = [], [], 0
        for ck, cell in cells.items():
            a, res = run_config(cell, cfg)
            aurocs.append(a)
            kept.append(res.meta["n_kept"] / res.meta["m"])
            abstains += int(res.abstained)
        key = "|".join(f"{k}={cfg[k]}" for k in keys)
        per_cell_auroc[key] = aurocs
        rows.append({**{k: str(cfg[k]) for k in keys},
                     "config": key,
                     "macro_auroc": float(np.nanmean(aurocs)),
                     "mean_frac_features_kept": float(np.mean(kept)),
                     "n_cells_abstained": abstains})
        if (ci + 1) % 8 == 0:
            print(f"  {ci+1}/{len(configs)} configs done", flush=True)

    S.save_csv(os.path.join(out, "per_config.csv"), rows)

    # ---- main effects: paired across configs that differ only in this factor ----
    eff_rows = []
    for k in keys:
        lo, hi = FACTORS[k]
        a_all, b_all, grp = [], [], []
        for cfg in configs:
            if cfg[k] != lo:
                continue
            other = dict(cfg)
            other[k] = hi
            ka = "|".join(f"{kk}={cfg[kk]}" for kk in keys)
            kb = "|".join(f"{kk}={other[kk]}" for kk in keys)
            a_all.extend(per_cell_auroc[ka])
            b_all.extend(per_cell_auroc[kb])
            grp.extend(cell_keys)
        # groups=cell -> average each cell's delta over the 32 other-factor
        # combinations FIRST, then test over 25 independent test sets. Pooling
        # them gave n=800 and p-values like 1e-44. Found in review.
        st = S.wl_wilcoxon(a_all, b_all, groups=grp)
        eff_rows.append({"factor": k, "from": str(lo), "to": str(hi),
                         "mean_delta_pp": st["mean_delta"] * 100,
                         "median_delta_pp": st["median_delta"] * 100,
                         "wins": st["wins"], "losses": st["losses"],
                         "n_test_sets": st["n"], "wilcoxon_p": st["p"]})

    S.save_csv(os.path.join(out, "main_effects.csv"), eff_rows)

    lk = "|".join(f"{k}={LEGACY[k]}" for k in keys)
    fk = "|".join(f"{k}={FAITHFUL[k]}" for k in keys)
    macro = {r["config"]: r["macro_auroc"] for r in rows}
    summary = {
        "n_configs": len(configs), "n_cells": len(cells),
        "macro_legacy_config": macro.get(lk),
        "macro_faithful_config": macro.get(fk),
        "macro_best_config": max(macro.values()),
        "macro_worst_config": min(macro.values()),
        "span": max(macro.values()) - min(macro.values()),
        "faithful_vs_legacy": S.wl_wilcoxon(per_cell_auroc[lk],
                                            per_cell_auroc[fk],
                                            groups=cell_keys),
        "references_by_scale": {sc: {"good6": refs[sc]["good6"],
                                     "lsml_all30": refs[sc]["lsml_all30"]}
                                for sc in S.SCALES},
        "main_effects": eff_rows,
    }

    print("\n" + "=" * 72)
    print("PHASE C RESULTS - main effects (not best-of-64)")
    print("=" * 72)
    print(f"{'factor':<18} {'from':>8} -> {'to':<8} {'mean':>9} {'median':>9} "
          f"{'W/L':>8} {'p':>8}")
    for e in sorted(eff_rows, key=lambda r: -abs(r["mean_delta_pp"])):
        print(f"{e['factor']:<18} {e['from']:>8} -> {e['to']:<8} "
              f"{e['mean_delta_pp']:+8.2f}pp {e['median_delta_pp']:+8.2f}pp "
              f"{e['wins']:3d}/{e['losses']:<4d} {e['wilcoxon_p']:8.4f}")
    print(f"\n  legacy config      {summary['macro_legacy_config']:.4f}")
    print(f"  fully faithful     {summary['macro_faithful_config']:.4f}  "
          f"({S.fmt_wl(summary['faithful_vs_legacy'])})")
    print(f"  span over 64       {summary['macro_worst_config']:.4f} .. "
          f"{summary['macro_best_config']:.4f}  ({summary['span']*100:.2f}pp)")
    for sc in S.SCALES:
        r = summary["references_by_scale"][sc]
        print(f"  [{sc:>8}] L-SML all 30 {r['lsml_all30']:.4f}   "
              f"GOOD_6 {r['good6']:.4f}")

    S.save_json(os.path.join(out, "summary.json"), summary)
    render(out, rows, eff_rows, summary, keys)
    print(f"\nWrote -> {out}")


def render(out, rows, eff_rows, summary, keys):
    body = []
    body.append("<h2>Why every prior U-PCR number was re-run</h2>")
    body.append(
        "<p>Three results were treated as settled and should not have been. The "
        "2-eigenvector measurement was taken with the g2 search capped at a "
        "quarter of its range — and g2 <em>is</em> the dial between one and two "
        "eigenvectors, so the test was confounded by the bug it was meant to "
        "evaluate. The 'robust loss hurts' result came from a different estimator "
        "on the L-SML path, not U-PCR's pair-fit. And the scale ratio had never "
        "been varied at all. All three are measured here on the faithful "
        "implementation.</p>")

    body.append("<h2>Main effects</h2>")
    order = sorted(eff_rows, key=lambda r: -abs(r["mean_delta_pp"]))
    body.append(S.bar_chart(
        [f"{e['factor']}: {e['from']} → {e['to']}" for e in order],
        [e["mean_delta_pp"] / 100 for e in order],
        "mean AUROC change (paired over all other factors)",
        hlines=[("no effect", 0.0)], value_fmt="{:+.4f}"))
    body.append(S.html_table(
        ["factor", "from", "to", "mean change", "median change",
         "wins/losses", "Wilcoxon p (n=25 test sets)"],
        [[e["factor"], e["from"], e["to"], f"{e['mean_delta_pp']:+.2f}pp",
          f"{e['median_delta_pp']:+.2f}pp",
          f"{e['wins']}/{e['losses']}", f"{e['wilcoxon_p']:.4f}"] for e in order],
        numeric_cols=(3, 4, 6)))
    body.append(
        "<p class='note'>Each factor's effect is averaged within a test set over "
        "the 32 combinations of the other five factors before testing, so the "
        "Wilcoxon runs over 25 independent test sets. An earlier version pooled "
        "all 800 rows and produced p-values as small as 1e-44; those were "
        "pseudo-replication and must not be quoted. Note how many effects have a "
        "median of exactly 0.00pp — the means are tail-driven.</p>")
    body.append(
        "<p class='note'>Read the scale-ratio row with Phase B1 in hand: the g2 "
        "choice does not move when the range widens (0/25 cells), so this factor "
        "cannot be acting through g2. If it moves anything it acts through the "
        "exclusion thresholds — a feature-selection effect, not a weight-"
        "estimation one.</p>")

    body.append("<h2>Reference points at all three loading scales</h2>")
    body.append(S.html_table(
        ["loading scale", "GOOD_6", "continuous L-SML, all 30"],
        [[sc, f"{summary['references_by_scale'][sc]['good6']:.4f}",
          f"{summary['references_by_scale'][sc]['lsml_all30']:.4f}"]
         for sc in S.SCALES], numeric_cols=(1, 2)))
    body.append(
        "<p class='note'>Reported three ways rather than chosen. The spec "
        "pre-registered <code>eigen</code>; <code>complete</code> is exact on the "
        "unit checks that <code>eigen</code> fails, but it is also the only scale "
        "under which the clustered variant runs on every cell — the convenient "
        "answer — so no single scale is presented as the number.</p>")

    body.append("<h2>Where the configurations land</h2>")
    top = sorted(rows, key=lambda r: -r["macro_auroc"])[:12]
    body.append(S.html_table(
        ["configuration", "macro AUROC", "features kept", "cells abstained"],
        [[r["config"].replace("|", " &middot; "), f"{r['macro_auroc']:.4f}",
          f"{r['mean_frac_features_kept']*100:.0f}%", r["n_cells_abstained"]]
         for r in top],
        numeric_cols=(1, 2, 3)))
    body.append(
        f"<p class='note'>Best of 64 is shown for completeness only. With 64 "
        f"configurations over 25 test sets and roughly 1.5pp of headroom, the top "
        f"of this table is a winner's-curse artifact — the main-effects table "
        f"above is the result. Whole span: "
        f"{summary['span']*100:.2f}pp.</p>")

    S.write_page(
        os.path.join(out, "index.html"),
        "Every paper-faithful flag, measured",
        "U-PCR study, Phase C — R2, R4 and the scale ratio, re-run on the fixed "
        "implementation",
        [f"Legacy configuration <b>{summary['macro_legacy_config']:.4f}</b> vs "
         f"fully faithful <b>{summary['macro_faithful_config']:.4f}</b> "
         f"({S.fmt_wl(summary['faithful_vs_legacy'])}).",
         f"All 64 configurations span only "
         f"<b>{summary['span']*100:.2f}pp</b> "
         f"({summary['macro_worst_config']:.4f} to "
         f"{summary['macro_best_config']:.4f}).",
         f"Reference points (both sides at the SAME loading scale): GOOD_6 "
         + " / ".join(f"{sc} {summary['references_by_scale'][sc]['good6']:.4f}"
                      for sc in S.SCALES) + ".",
         "Main effects are the result; the best-of-64 row is not."],
        "".join(body))


if __name__ == "__main__":
    main()
