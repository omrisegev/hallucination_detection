"""
exp05_cluster_variant.py — Phase D. The clustered U-PCR variant.

Gated on Phase 0 (the loading-scale fix, so K is trustworthy and every cell sits
at K>=3) and on Phase B2's kill-switch (same-cluster pairs fit 2.03x worse in
25/25 cells, so the premise holds).

ARMS, all label-free at runtime
-------------------------------
  upcr.all      faithful U-PCR, additive system on every pair (the incumbent)
  upcr.cross    additive system on CROSS-CLUSTER pairs only  <- the contribution
  upcr.hier     two-level: U-PCR inside each cluster, then across the clusters
  lsml          continuous L-SML, the deployed detector, for reference
  good6         the project's standing reference subset, for reference

MECHANISM GATE BEFORE PERFORMANCE GATE
--------------------------------------
Step 203 diagnosed the weight-estimation gap as a DISAGREEMENT about which features
matter, not a calibration error: rank agreement +0.186 against the supervised
weights, and only 1 of the top 5 features shared. If dropping the corrupted
equations is the right fix, those two numbers must move first. They are the gate.
An AUROC win without them would be luck, and this project has been burned by
best-of-N before (Step 193).

  gate 1  rank agreement (Spearman of |w| vs the supervised |w|) must beat +0.186
  gate 2  top-5 overlap must beat 1/5

Reported with wins/losses and Wilcoxon, never an average alone (Omri, Step 203),
and never gated on a 1-2pp difference.

Run:  python scripts/upcr_study/exp05_cluster_variant.py
Out:  results/upcr_study/05_cluster_variant/{per_cell.csv,summary.json,index.html}
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import upcr_fit                            # noqa: E402
from spectral_utils.upcr_clustered import (                         # noqa: E402
    upcr_clustered_fit, upcr_hierarchical_fit, discover_clusters,
    check_identifiability,
)
from spectral_utils.fusion_utils import lsml_continuous             # noqa: E402

# Step 203 baselines the mechanism gate must clear.
GATE_RANK_AGREEMENT = 0.186
GATE_TOP5_OVERLAP = 1

FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=1.0)


def supervised_weights(V, y):
    """Trust levels a model learns straight from answer keys. Mirrors
    pruning_study/exp04_weight_diagnostic.learned_weights exactly."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0))
    pipe.fit(V, y)
    return pipe.named_steps["logisticregression"].coef_.ravel()


def agreement(w_guess, w_learn):
    """The two Step-203 diagnostics, computed the same way they were there."""
    rank = spearmanr(np.abs(w_learn), np.abs(w_guess)).statistic
    top_l = set(np.argsort(-np.abs(w_learn))[:5])
    top_g = set(np.argsort(-np.abs(w_guess))[:5])
    return float(rank), int(len(top_l & top_g))


def main():
    out = S.outdir("05_cluster_variant")
    cells = S.load()
    S.validity_check(cells)

    rows = []
    for scale in S.SCALES:
      for ck, cell in cells.items():
        V, y = cell["V"], cell["labels"]
        F = V.T
        m = F.shape[0]
        rec = {"cell": ck, "scale": scale, "m": m}

        K, c, _ = discover_clusters(F, loading_scale=scale)
        ident = check_identifiability(c, m)
        rec.update({"K": ident["K"], "cluster_sizes": str(ident["cluster_sizes"]),
                    "n_cross_pairs": ident["n_cross_pairs"],
                    "identifiable": ident["ok"], "ident_reason": ident["reason"]})

        w_learn = supervised_weights(V, y)

        # --- arm: all pairs (incumbent) ---
        r_all = upcr_fit(F, **FIT)
        rec["auroc_upcr_all"] = S.auroc_from_score(cell, r_all.w @ F)
        ra, ta = agreement(r_all.w, w_learn)
        rec["rank_agreement_all"], rec["top5_overlap_all"] = ra, ta

        # --- arm: cross-cluster pairs only (the contribution) ---
        if ident["ok"]:
            r_x, _ = upcr_clustered_fit(F, groups=c, **FIT)
            rec["auroc_upcr_cross"] = S.auroc_from_score(cell, r_x.w @ F)
            rx, tx = agreement(r_x.w, w_learn)
            rec["rank_agreement_cross"], rec["top5_overlap_cross"] = rx, tx

            r_h, _ = upcr_hierarchical_fit(F, groups=c, **FIT)
            rec["auroc_upcr_hier"] = S.auroc_from_score(cell, r_h.w @ F)
            rh, th = agreement(r_h.w, w_learn)
            rec["rank_agreement_hier"], rec["top5_overlap_hier"] = rh, th
            rec["n_clusters_averaged"] = r_h.meta["n_clusters_averaged"]
        else:
            for k in ("auroc_upcr_cross", "rank_agreement_cross",
                      "top5_overlap_cross", "auroc_upcr_hier",
                      "rank_agreement_hier", "top5_overlap_hier"):
                rec[k] = float("nan")

        # --- references ---
        fused, _ = lsml_continuous(*[V[:, i] for i in range(m)],
                                   compute_score_matrix=False,
                                   loading_scale=scale)
        rec["auroc_lsml_all30"] = S.auroc_from_score(cell, fused)
        from inscope_bench_common import good6_cols
        # GOOD_6 at the SAME scale, not via good6_score which pins 'unit'
        rec["auroc_good6"] = S.score_cols_at_scale(cell, good6_cols(cell), scale)

        rec["pairs_degraded"] = bool(
            r_x.meta.get("pairs_degraded_after_exclusion", False)
            if ident["ok"] else False)
        rows.append(rec)
        print(f"  [{scale:>8}] {ck[:28]:28s} K={rec['K']} | upcr.all "
              f"{rec['auroc_upcr_all']:.4f}  cross {rec['auroc_upcr_cross']:.4f}  "
              f"hier {rec['auroc_upcr_hier']:.4f} | rank "
              f"{rec['rank_agreement_all']:+.3f}->{rec['rank_agreement_cross']:+.3f} "
              f"top5 {rec['top5_overlap_all']}->{rec['top5_overlap_cross']}",
              flush=True)

    S.save_csv(os.path.join(out, "per_cell.csv"), rows)

    summary = {"n_cells": len(cells), "by_scale": {}}
    for scale in S.SCALES:
        sub = [r for r in rows if r["scale"] == scale]
        cellk = [r["cell"] for r in sub]

        def sarr(k, _sub=sub):
            return np.array([r[k] for r in _sub], dtype=float)

        sc_rec = {"n_identifiable": int(sum(1 for r in sub if r["identifiable"])),
                  "n_pairs_degraded": int(sum(1 for r in sub
                                              if r.get("pairs_degraded"))),
                  "cell_keys": cellk}
        for a in ("upcr_all", "upcr_cross", "upcr_hier", "lsml_all30", "good6"):
            sc_rec[f"macro_{a}"] = float(np.nanmean(sarr(f"auroc_{a}")))
        for a in ("all", "cross", "hier"):
            sc_rec[f"median_rank_agreement_{a}"] = float(
                np.nanmedian(sarr(f"rank_agreement_{a}")))
            sc_rec[f"median_top5_{a}"] = float(
                np.nanmedian(sarr(f"top5_overlap_{a}")))
        sc_rec["cross_vs_all"] = S.wl_wilcoxon(sarr("auroc_upcr_all"),
                                               sarr("auroc_upcr_cross"),
                                               groups=cellk)
        sc_rec["rank_shift"] = S.wl_wilcoxon(sarr("rank_agreement_all"),
                                             sarr("rank_agreement_cross"),
                                             groups=cellk)
        sc_rec["gate1_rank_passed"] = bool(
            sc_rec["median_rank_agreement_cross"] > GATE_RANK_AGREEMENT)
        sc_rec["gate2_top5_passed"] = bool(
            sc_rec["median_top5_cross"] > GATE_TOP5_OVERLAP)
        summary["by_scale"][scale] = sc_rec

    rows_c = [r for r in rows if r["scale"] == "complete"]

    def arr(k):
        return np.array([r[k] for r in rows_c], dtype=float)

    summary["n_identifiable"] = summary["by_scale"]["complete"]["n_identifiable"]
    for a in ("upcr_all", "upcr_cross", "upcr_hier", "lsml_all30", "good6"):
        summary[f"macro_{a}"] = summary["by_scale"]["complete"][f"macro_{a}"]

    # ---- mechanism gate ----
    mech = {}
    for a in ("all", "cross", "hier"):
        mech[a] = {
            "median_rank_agreement": float(np.nanmedian(arr(f"rank_agreement_{a}"))),
            "median_top5_overlap": float(np.nanmedian(arr(f"top5_overlap_{a}"))),
            "mean_top5_overlap": float(np.nanmean(arr(f"top5_overlap_{a}"))),
        }
    mech["gate1_rank_passed"] = bool(
        mech["cross"]["median_rank_agreement"] > GATE_RANK_AGREEMENT)
    mech["gate2_top5_passed"] = bool(
        mech["cross"]["median_top5_overlap"] > GATE_TOP5_OVERLAP)
    ck_c = [r["cell"] for r in rows_c]
    mech["rank_shift_cross_vs_all"] = S.wl_wilcoxon(
        arr("rank_agreement_all"), arr("rank_agreement_cross"), groups=ck_c)
    mech["top5_shift_cross_vs_all"] = S.wl_wilcoxon(
        arr("top5_overlap_all"), arr("top5_overlap_cross"), groups=ck_c)
    summary["mechanism"] = mech

    # ---- performance ----
    perf = {
        "cross_vs_all": S.wl_wilcoxon(arr("auroc_upcr_all"),
                                      arr("auroc_upcr_cross"), groups=ck_c),
        "hier_vs_all": S.wl_wilcoxon(arr("auroc_upcr_all"),
                                     arr("auroc_upcr_hier"), groups=ck_c),
        "cross_vs_lsml": S.wl_wilcoxon(arr("auroc_lsml_all30"),
                                       arr("auroc_upcr_cross"), groups=ck_c),
        "cross_vs_good6": S.wl_wilcoxon(arr("auroc_good6"),
                                        arr("auroc_upcr_cross"), groups=ck_c),
    }
    summary["performance"] = perf

    print("\n" + "=" * 72)
    print("PHASE D RESULTS")
    print("=" * 72)
    print("\n  ALL THREE LOADING SCALES (no single scale is 'the' answer)")
    print(f"{'scale':>9} {'upcr.all':>9} {'upcr.cross':>11} {'upcr.hier':>10} "
          f"{'L-SML 30':>9} {'GOOD_6':>8} {'ident':>8} {'gates':>7}")
    for sc in S.SCALES:
        b = summary["by_scale"][sc]
        gates = ("PASS" if (b["gate1_rank_passed"] and b["gate2_top5_passed"])
                 else "FAIL")
        print(f"{sc:>9} {b['macro_upcr_all']:9.4f} {b['macro_upcr_cross']:11.4f} "
              f"{b['macro_upcr_hier']:10.4f} {b['macro_lsml_all30']:9.4f} "
              f"{b['macro_good6']:8.4f} {b['n_identifiable']:5d}/{len(cells)} "
              f"{gates:>7}")
    print(f"\n  headline block below = 'complete' scale; identifiable "
          f"{summary['n_identifiable']}/{len(cells)}")
    print("\n  macro AUROC")
    for a in ("upcr_all", "upcr_cross", "upcr_hier", "lsml_all30", "good6"):
        print(f"    {a:<14} {summary[f'macro_{a}']:.4f}")
    print("\n  MECHANISM GATE (must clear before any performance claim)")
    print(f"    gate 1  rank agreement  {mech['all']['median_rank_agreement']:+.3f} "
          f"(all pairs) -> {mech['cross']['median_rank_agreement']:+.3f} (cross only)"
          f"   bar {GATE_RANK_AGREEMENT:+.3f} -> "
          f"{'PASS' if mech['gate1_rank_passed'] else 'FAIL'}")
    print(f"    gate 2  top-5 overlap   {mech['all']['median_top5_overlap']:.1f}/5 "
          f"-> {mech['cross']['median_top5_overlap']:.1f}/5"
          f"   bar >{GATE_TOP5_OVERLAP}/5 -> "
          f"{'PASS' if mech['gate2_top5_passed'] else 'FAIL'}")
    print(f"    rank shift  {S.fmt_wl(mech['rank_shift_cross_vs_all'])}")
    print(f"    top5 shift  {S.fmt_wl(mech['top5_shift_cross_vs_all'])}")
    print("\n  PERFORMANCE")
    for k, v in perf.items():
        print(f"    {k:<18} {S.fmt_wl(v)}")

    both = mech["gate1_rank_passed"] and mech["gate2_top5_passed"]
    verdict = (
        "MECHANISM GATE PASSES - dropping same-cluster equations moves the "
        "estimator toward the supervised one, which is the effect the variant was "
        "built to produce. Performance can now be read as evidence rather than "
        "coincidence."
        if both else
        "MECHANISM GATE FAILS - the variant does not move the estimator toward "
        "the supervised weights, so any AUROC difference is not evidence for the "
        "proposed mechanism. Report the failure; do not go hunting for a win.")
    print(f"\n  VERDICT: {verdict}")
    summary["verdict"] = verdict

    S.save_json(os.path.join(out, "summary.json"), summary)
    render(out, rows, summary)
    print(f"\nWrote -> {out}")


def render(out, rows, summary):
    mech, perf = summary["mechanism"], summary["performance"]
    body = []
    body.append("<h2>What the variant changes</h2>")
    body.append(
        "<p>U-PCR turns 'which features should I trust?' into one equation per "
        "feature pair, assuming features fail independently. Ours do not: near-"
        "duplicates have correlations up to 0.996, and the model can only explain "
        "such a pair by inflating <em>both</em> features' scores. The variant keeps "
        "only the equations from pairs in <b>different</b> clusters, where the "
        "assumption still holds — Phase B2 measured those as fitting "
        "<b>2.03&times; better</b> in 25/25 cells. Nothing else in the method "
        "changes, so a difference is attributable.</p>")

    body.append("<h2>Mechanism gate — does it move the estimator the right way?</h2>")
    body.append(S.html_table(
        ["diagnostic", "all pairs", "cross-cluster only", "bar", "verdict"],
        [["rank agreement with the supervised weights (median)",
          f"{mech['all']['median_rank_agreement']:+.3f}",
          f"{mech['cross']['median_rank_agreement']:+.3f}",
          f">{GATE_RANK_AGREEMENT:+.3f}",
          "PASS" if mech["gate1_rank_passed"] else "FAIL"],
         ["top-5 features shared with the supervised model (median)",
          f"{mech['all']['median_top5_overlap']:.1f}/5",
          f"{mech['cross']['median_top5_overlap']:.1f}/5",
          f">{GATE_TOP5_OVERLAP}/5",
          "PASS" if mech["gate2_top5_passed"] else "FAIL"]],
        numeric_cols=(1, 2)))
    body.append(f"<p class='note'>Paired shifts: rank "
                f"{S.fmt_wl(mech['rank_shift_cross_vs_all'])}; top-5 "
                f"{S.fmt_wl(mech['top5_shift_cross_vs_all'])}.</p>")

    order = sorted(rows, key=lambda r: -(r["auroc_upcr_cross"]
                                         - r["auroc_upcr_all"]
                                         if np.isfinite(r["auroc_upcr_cross"]) else -9))
    body.append("<h2>Per-cell change from dropping the corrupted equations</h2>")
    body.append(S.bar_chart(
        [S.plain_cell(r["cell"]) for r in order],
        [r["auroc_upcr_cross"] - r["auroc_upcr_all"] for r in order],
        "AUROC change, cross-cluster-only minus all-pairs",
        hlines=[("no change", 0.0)], value_fmt="{:+.4f}"))

    body.append("<h2>Where each arm lands</h2>")
    body.append(S.bar_chart(
        ["U-PCR, cross-cluster only", "U-PCR, all pairs", "U-PCR, hierarchical",
         "continuous L-SML, all 30", "GOOD_6 reference"],
        [summary["macro_upcr_cross"], summary["macro_upcr_all"],
         summary["macro_upcr_hier"], summary["macro_lsml_all30"],
         summary["macro_good6"]],
        "macro AUROC over 25 test sets"))

    body.append("<h2>Per cell</h2>")
    body.append(S.html_table(
        ["test set", "K", "cluster sizes", "cross pairs", "U-PCR all",
         "U-PCR cross", "U-PCR hier", "L-SML all-30", "GOOD_6"],
        [[S.plain_cell(r["cell"]), r["K"], r["cluster_sizes"], r["n_cross_pairs"],
          f"{r['auroc_upcr_all']:.4f}", f"{r['auroc_upcr_cross']:.4f}",
          f"{r['auroc_upcr_hier']:.4f}", f"{r['auroc_lsml_all30']:.4f}",
          f"{r['auroc_good6']:.4f}"] for r in order],
        numeric_cols=(1, 3, 4, 5, 6, 7, 8)))

    S.write_page(
        os.path.join(out, "index.html"),
        "U-PCR for dependent features",
        "U-PCR study, Phase D — our extension: fit the additive system on "
        "cross-cluster pairs only",
        [f"Mechanism gate 1 (rank agreement): "
         f"{mech['all']['median_rank_agreement']:+.3f} &rarr; "
         f"<b>{mech['cross']['median_rank_agreement']:+.3f}</b> "
         f"({'PASS' if mech['gate1_rank_passed'] else 'FAIL'} vs the "
         f"{GATE_RANK_AGREEMENT:+.3f} bar).",
         f"Mechanism gate 2 (top-5 overlap): "
         f"{mech['all']['median_top5_overlap']:.1f}/5 &rarr; "
         f"<b>{mech['cross']['median_top5_overlap']:.1f}/5</b> "
         f"({'PASS' if mech['gate2_top5_passed'] else 'FAIL'}).",
         f"Macro AUROC: cross-cluster <b>{summary['macro_upcr_cross']:.4f}</b> vs "
         f"all-pairs {summary['macro_upcr_all']:.4f} "
         f"({S.fmt_wl(perf['cross_vs_all'])}).",
         f"All {summary['n_identifiable']}/{summary['n_cells']} cells satisfy the "
         f"K&nbsp;&ge;&nbsp;3 identifiability floor.",
         f"<b>{summary['verdict']}</b>"],
        "".join(body))


if __name__ == "__main__":
    main()
