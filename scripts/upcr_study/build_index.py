"""
build_index.py — one browsable front page for the U-PCR study.

Reads each experiment's summary.json (no recomputation) and renders
results/upcr_study/index.html.

NOTE ON MARKUP: `html_table` ESCAPES its cells, so any HTML passed into it renders
as literal `&lt;b&gt;`. Rows here are plain text; emphasis goes in the surrounding
prose instead. (Both this and a stale hardcoded "the 2 cells" string in the Phase E
row were review findings.)

Run:  python scripts/upcr_study/build_index.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

PAGES = [
    ("01_g2_criterion", "Does the g2 criterion work on our data?",
     "Phase B1 — the paper's Figure 1, reproduced"),
    ("03_faithful_factorial", "Every paper-faithful flag, measured",
     "Phase C — R2, R4 and the scale ratio, re-run on fixed code"),
    ("04_violation_map", "Does clustering find the violation, or just correlation?",
     "Phase B2 — redesigned after review exposed a confound"),
    ("05_cluster_variant", "U-PCR for dependent features",
     "Phase D — our extension: cross-cluster pairs only"),
    ("06_orientation", "Orientation without an anchor",
     "Phase E — what structure can and cannot supply"),
    ("07_lambda2_threshold", "Is the component-count threshold the real dial?",
     "Step 205 — the constant that actually chooses one eigenvector or two"),
]


def load(name):
    p = os.path.join(S.OUT_ROOT, name, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(S.OUT_ROOT, exist_ok=True)
    s = {n: load(n) for n, _, _ in PAGES}

    body = []
    body.append("<h2>What this study was for, and what happened</h2>")
    body.append(
        "<p>Two review documents claimed our U-PCR implementation had drifted from "
        "the paper it cites. All seven claimed deviations checked out. The "
        "implementation was made faithful — each deviation behind its own flag — "
        "and every one measured. Prior U-PCR numbers were re-run rather than "
        "carried over, because two of them had been measured with the very bug "
        "they were testing.</p>")
    body.append(
        "<p><b>An independent adversarial review of this study then found 17 "
        "defects of its own</b>, several of which changed a stated conclusion. "
        "The numbers below are post-correction. Where a claim was withdrawn, it "
        "is named rather than quietly dropped.</p>")

    body.append("<h2>Headline results</h2>")
    rows = []

    b1 = s.get("01_g2_criterion")
    if b1:
        rows.append([
            "B1", "Is the g2 search range the problem?",
            f"It binds, but it costs nothing. On the PRE-exclusion fit the chosen "
            f"g2 does not move in "
            f"{b1['n_cells'] - b1['n_argmin_moved_when_range_widened']}/"
            f"{b1['n_cells']} cells when the range is widened 16x. The g2 the "
            f"pipeline returns is a POST-exclusion refit, though, and that one sits "
            f"on the ceiling in "
            f"{b1.get('n_post_exclusion_pinned_at_ceiling', '?')}/{b1['n_cells']} "
            f"cells - so the earlier unqualified 'never binds' was narrowed in "
            f"Step 205. Un-pinning it is a wash (-0.28pp, 12W/13L). g2 is still a "
            f"large lever ({b1['mean_auroc_spread_over_q']*100:.1f}pp mean AUROC "
            f"swing) and the criterion leaves "
            f"{b1['mean_regret_legacy_range']*100:.2f}pp on the table, pointing the "
            f"wrong way in "
            f"{b1['n_cells_criterion_points_wrong_way']}/{b1['n_cells']} cells. It "
            f"is NOT the component-count dial - that is lambda2_threshold (exp07)."])

    c = s.get("03_faithful_factorial")
    if c:
        eff = {e["factor"]: e for e in c["main_effects"]}
        cm = eff["components"]
        lo = eff["loss"]
        rows.append([
            "C", "R2 — does the 2-eigenvector rule help?",
            f"No, it hurts: {cm['mean_delta_pp']:+.2f}pp mean / "
            f"{cm['median_delta_pp']:+.2f}pp median, {cm['wins']}W/{cm['losses']}L "
            f"over {cm['n_test_sets']} test sets, p={cm['wilcoxon_p']:.4f}. The "
            f"earlier +0.5pp was measured with the g2 range capped, i.e. "
            f"confounded by the dial it was testing."])
        rows.append([
            "C", "R4 — does the absolute loss help?",
            f"No reliable effect: {lo['mean_delta_pp']:+.2f}pp mean but "
            f"{lo['median_delta_pp']:+.2f}pp median, {lo['wins']}W/{lo['losses']}L, "
            f"p={lo['wilcoxon_p']:.2f}. The mean is tail-driven. Never actually "
            f"measured before — the prior number was a different estimator on the "
            f"L-SML path."])
        fv = c["faithful_vs_legacy"]
        refs = c["references_by_scale"]
        rows.append([
            "C", "Is being faithful better?",
            f"No, but weakly: fully faithful {c['macro_faithful_config']:.4f} vs "
            f"legacy {c['macro_legacy_config']:.4f} — "
            f"{fv['mean_delta']*100:+.2f}pp mean but only "
            f"{fv['median_delta']*100:+.2f}pp median, "
            f"{fv['wins']}W/{fv['losses']}L, p={fv['p']:.3f}. No configuration of "
            f"64 beats GOOD_6 ("
            + ", ".join(f"{k} {refs[k]['good6']:.4f}" for k in S.SCALES) + ")."])

    b2 = s.get("04_violation_map")
    if b2:
        bs = b2["by_scale"]["complete"]
        rows.append([
            "B2", "Does clustering isolate the assumption violation?",
            f"WITHDRAWN. The original 2.03x looked decisive but was a confound: "
            f"fit error is essentially pair correlation "
            f"(Spearman {b2['mean_spearman_fiterr_vs_abscorr']:.3f}). Matched on "
            f"correlation decile the ratio collapses to "
            f"{bs['mean_ratio_decile_matched']:.3f}, a random partition gives "
            f"{bs['mean_ratio_random_partition']:.3f}, and magnitude-only "
            f"clustering separates it BETTER "
            f"({bs['mean_ratio_magnitude_clustering']:.2f}x) than L-SML."])

    d = s.get("05_cluster_variant")
    if d:
        bs = d.get("by_scale", {}).get("complete", {})
        cv = bs.get("cross_vs_all", d.get("performance", {}).get("cross_vs_all", {}))
        rows.append([
            "D", "Does dropping the corrupted equations help?",
            f"No. Both mechanism gates fail — rank agreement "
            f"{bs.get('median_rank_agreement_all', 0):+.3f} -> "
            f"{bs.get('median_rank_agreement_cross', 0):+.3f} against a +0.186 bar, "
            f"top-5 overlap unchanged — and performance drops "
            f"{cv.get('mean_delta', 0)*100:+.2f}pp mean / "
            f"{cv.get('median_delta', 0)*100:+.2f}pp median, "
            f"{cv.get('wins', 0)}W/{cv.get('losses', 0)}L, p={cv.get('p', 0):.3f}."])

    e = s.get("06_orientation")
    if e:
        rows.append([
            "E", "Can structure replace the hand-tuned polarities?",
            f"Yes for the 42 per-feature signs: deriving them from the estimated "
            f"correlations scores {e['macro_rho_anchor']:.4f} vs "
            f"{e['macro_hand_anchor']:.4f} "
            f"({e['rho_anchor_vs_hand']['mean_delta']*100:+.2f}pp, "
            f"{e['rho_anchor_vs_hand']['wins']}W/"
            f"{e['rho_anchor_vs_hand']['losses']}L, "
            f"p={e['rho_anchor_vs_hand']['p']:.3f})."])
        rows.append([
            "E", "Can we drop the anchor too?",
            f"No, and provably not: a global flip of every feature leaves the "
            f"estimate bit-identical, so the global direction is not recoverable "
            f"from the covariance structure at all. The rule that tried was wrong "
            f"in {e['n_cells']-e['n_cells_global_rule_matches_anchor']}/"
            f"{e['n_cells']} cells."])
        rows.append([
            "E", "How many hand signs are wrong?",
            f"{e['n_features_mis_signed']} of {e['n_features_audited']} pool "
            f"features have oriented AUROC below 0.5 — not the 3 first reported, "
            f"which came from inspecting only a top-6 list. Structure recovers the "
            f"EMPIRICAL direction on "
            f"{e['mean_polarity_agreement_empirical']*100:.1f}% of features "
            f"(p={e['p_empirical_vs_null']:.3f}); agreement with our DECLARED "
            f"signs is {e['mean_polarity_agreement']*100:.1f}%, at the "
            f"{e['polarity_null_mean']*100:.1f}% chance level for this statistic "
            f"(p={e['p_hand_vs_null']:.2f}) and therefore not a result."])

    g = s.get("07_lambda2_threshold")
    if g:
        d = g["components_at_deployed_config"]
        rows.append([
            "205", "Is the 2-eigenvector rule what hurts?",
            f"No. lambda2_frac is tightly clustered ({g['lambda2_frac']['min']:.3f}"
            f"-{g['lambda2_frac']['max']:.3f}, median "
            f"{g['lambda2_frac']['median']:.3f}) just above the hardcoded 0.1, so "
            f"the threshold flips 24/25 cells at once - yet removing the second "
            f"component everywhere buys only "
            f"{g['best_gain_pp']:+.2f}pp. AND the -3.67pp that Phase C reports for "
            f"this factor is a factorial MAIN EFFECT, marginalised over 32 "
            f"other-factor combinations: at the DEPLOYED configuration the same "
            f"switch is {d['delta_pp_auto_minus_one']:+.2f}pp mean / "
            f"{d['median_delta_pp']:+.2f}pp median, {d['wins']}W/{d['losses']}L, "
            f"p={d['p']:.3f}. Do not quote a main effect as a deployed cost."])

    body.append(S.html_table(["phase", "question", "answer"], rows))

    body.append("<h2>The honest headline</h2>")
    body.append(
        "<p>Every faithful flag hurts or does nothing, the clustered variant "
        "fails, the anchor-free rung is impossible, and no configuration reaches "
        "the existing reference subset. One-component U-PCR is <em>exactly</em> "
        "the first principal component of the surviving features, so the entire "
        "rho / g2 / projection-residual apparatus enters only through which "
        "features it excludes. <b>U-PCR's estimation machinery is inert on our "
        "data; what mattered was feature orientation and feature exclusion.</b> "
        "The orientation result in Phase E is the finding worth carrying "
        "forward.</p>")

    body.append("<h2>Experiment pages</h2><ul>")
    for name, title, sub in PAGES:
        if os.path.exists(os.path.join(S.OUT_ROOT, name, "index.html")):
            body.append(f'<li><a href="{name}/index.html">{title}</a> '
                        f'&mdash; <span style="color:var(--mut)">{sub}</span></li>')
    body.append("</ul>")

    body.append("<h2>The loading scale is reported three ways, never chosen</h2>")
    body.append(
        "<p>All of this sits on top of a correction to the L-SML residual, which "
        "returned a unit-length loading vector where the theory requires the "
        "loadings to reproduce the covariance — so a perfectly clustered block "
        "scored as an increasingly <em>bad</em> fit the tighter it got. "
        "<code>SPEC_residual_scaling_fix.md</code> pre-registered one fix "
        "(<code>eigen</code>); it fails that spec's own unit check and drops 6 of "
        "25 test sets to 2 clusters. A completion-based estimator "
        "(<code>complete</code>) is exact on the unit checks — but it is also the "
        "only scale under which the clustered variant runs everywhere, i.e. the "
        "convenient answer. Rather than adjudicate that, every scale-dependent "
        "number here is reported under all three. See "
        "<code>results/residual_scaling/</code>.</p>")

    body.append("<h2>What was withdrawn or dropped</h2>")
    body.append(
        "<ul>"
        "<li><b>B2's premise</b> — the 2.03x same-vs-cross gap is a confound; it "
        "does not survive matching on correlation magnitude.</li>"
        "<li><b>\"Three features are mis-signed\"</b> — fifteen are.</li>"
        "<li><b>\"Structure recovers 56% of our hand polarities\"</b> — that is "
        "chance for this statistic.</li>"
        "<li><b>A standalone scale-ratio sweep</b> — Phase B1 removed its premise; "
        "it survives only as one factor in the Phase C factorial, where it acts "
        "through the abstention threshold rather than through g2.</li>"
        "<li><b>All pooled p-values</b> — main effects are now tested over 25 "
        "independent test sets, not 800 pooled rows.</li>"
        "</ul>")

    S.write_page(
        os.path.join(S.OUT_ROOT, "index.html"),
        "U-PCR: faithful implementation, and a clustered variant",
        "Making the implementation match its paper, then testing whether "
        "clustering fixes what dependence breaks — corrected after adversarial "
        "review",
        ["Every claimed deviation from the paper was real; none of them helps.",
         "Two previously-settled results reverse on the fixed implementation.",
         "The clustered variant's premise was a confound, and the variant fails.",
         "The g2 search range binds on the deployed estimate but costs nothing; "
         "the component-count threshold is the real dial.",
         "The finding that survives is about feature orientation, not U-PCR."],
        "".join(body))
    print(f"Wrote -> {os.path.join(S.OUT_ROOT, 'index.html')}")


if __name__ == "__main__":
    main()
