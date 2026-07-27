"""
exp06_orientation.py — Phase E. Orientation without an anchor.

OMRI'S POINT, WHICH IS THE RIGHT ONE
------------------------------------
"I want to use the same orientation assumption as U-PCR uses. It does not rely on
an anchor or subset — exactly what we wanted."

The paper never discusses feature signs, for a telling reason: its regressors are
all trained to predict Y, so they are positively oriented by construction. The
orientation machinery is implicit and comes in two separable pieces:

  RELATIVE polarity   free, and derived. rho_i = Cov(f_i, Y), so sign(rho_i) says
                      whether feature i reads with or against correctness. No
                      anchor, no subset, no hand-labelling.
  GLOBAL +-1          the paper never faces this, but we must. The structural
                      handle is that g2 is a squared quantity, hence non-negative,
                      and rho_i = g2 + a_i — so rho leans POSITIVE whenever the
                      shared signal dominates. "Flip if rho leans negative" is
                      therefore derived from the model, not assumed.

WHAT IT ACTUALLY BUYS — the honest accounting
----------------------------------------------
Step 201 proved L-SML is EXACTLY invariant to input feature signs (1150/1150 sign
vectors, bit-identical). So the 42 hand-derived polarities in ALL_SIGNS were
already free on the L-SML path. What was never free is the single `epr` anchor bit
used to resolve the global direction.

So this rung replaces ONE bit with a structural rule — not 43. But it is the
load-bearing bit, and the last hand-picked prior in the pipeline. U-PCR, unlike
L-SML, is NOT sign-invariant, so its rho signs do real work.

ARMS
----
  hand+anchor   42 hand polarities, global sign from the epr anchor (incumbent)
  rho+anchor    polarity from sign(rho), global sign still from the anchor
  rho+majority  polarity from sign(rho), global sign from rho's own lean -- FULLY
                anchor-free and subset-free

Also measured: does sign(rho) actually RECOVER the 42 hand polarities? Agreement is
reported after resolving the single global flip, exactly as
pruning_study/exp04_weight_diagnostic does -- charging that one bit to all 30
features would understate agreement.

Run:  python scripts/upcr_study/exp06_orientation.py
Out:  results/upcr_study/06_orientation/{per_cell.csv,per_feature.csv,summary.json,index.html}
"""
import os
import sys
import collections

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import upcr_fit                            # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402
from spectral_utils.streaming_utils import anchor_orient            # noqa: E402
from sklearn.metrics import roc_auc_score                           # noqa: E402

FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)


def main():
    out = S.outdir("06_orientation")
    cells = S.load()
    S.validity_check(cells)

    rows, feat_rows = [], []
    for ck, cell in cells.items():
        pool = cell["pool"]
        # CAREFUL: prepare_cell already applies ALL_SIGNS (subset_sweep.py:363),
        # so V is ALREADY ORIENTED. Since zscore(raw*s) == s*zscore(raw) for
        # s = +-1, multiplying V by the hand signs UNDOES the orientation and
        # recovers the raw z-scored feature. Getting this backwards swaps the two
        # arms of this experiment.
        V = cell["V"]
        m = V.shape[1]
        hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], dtype=float)
        V_unoriented = V * hand

        # empirical ground truth for each feature's direction, which is a better
        # reference than the hand signs: 3 of them are demonstrably wrong
        # (epr_energy 0.310, min_energy 0.313, hurst_exponent 0.391 oriented AUROC)
        empirical = np.array(
            [1.0 if roc_auc_score(cell["labels"], V[:, i]) >= 0.5 else -1.0
             for i in range(m)]) * hand

        # ---- arm 1: hand polarities + anchor (the incumbent) ----
        F_hand = V.T
        r_hand = upcr_fit(F_hand, **FIT)
        s_hand, flip_hand = anchor_orient(r_hand.w @ F_hand, cell["anchor"])
        auc_hand = float(roc_auc_score(cell["labels"], s_hand))

        # ---- arms 2 & 3: polarity derived from rho on UNORIENTED features ----
        F_raw = V_unoriented.T
        r_raw = upcr_fit(F_raw, **FIT)
        derived = np.sign(r_raw.rho_hat_full)
        derived[derived == 0] = 1.0

        F_der = (V_unoriented * derived).T
        r_der = upcr_fit(F_der, **FIT)
        raw_score = r_der.w @ F_der

        # global sign from the anchor
        s_der_anchor, flip_der = anchor_orient(raw_score, cell["anchor"])
        auc_der_anchor = float(roc_auc_score(cell["labels"], s_der_anchor))

        # global sign from the model's own structure: g2 >= 0 => rho leans positive
        lean = float(np.sum(r_der.rho_hat_full))
        s_major = raw_score if lean >= 0 else -raw_score
        auc_major = float(roc_auc_score(cell["labels"], s_major))

        # ---- does sign(rho) recover the polarities? ----
        # `hand` is our declared prior; `empirical` is what the labels actually say.
        # Report against both, because the two disagree on 3 of the 30 features.
        raw_agree = float(np.mean(derived == hand))
        agree = max(raw_agree, 1.0 - raw_agree)      # resolve the one global bit
        raw_emp = float(np.mean(derived == empirical))
        agree_emp = max(raw_emp, 1.0 - raw_emp)
        global_bit_matches = bool(
            np.sign(np.corrcoef(s_major, s_der_anchor)[0, 1]) > 0)

        # Per-feature agreement must be scored AFTER resolving this cell's single
        # global +-1, exactly as the per-cell number is. Without that, a feature
        # whose polarity is consistently right modulo the global bit reads as
        # consistently WRONG (epr showed 4% before this fix, i.e. 96% flipped).
        derived_res = derived if raw_agree >= 0.5 else -derived
        for i, f in enumerate(pool):
            feat_rows.append({"cell": ck, "feature": f,
                              "hand_sign": int(hand[i]),
                              "derived_sign": int(derived_res[i]),
                              "rho": float(r_raw.rho_hat_full[i]),
                              "match": int(derived_res[i] == hand[i])})

        rows.append({
            "cell": ck, "m": m,
            "auroc_hand_anchor": auc_hand,
            "auroc_rho_anchor": auc_der_anchor,
            "auroc_rho_majority": auc_major,
            "polarity_agreement_with_hand": agree,
            "polarity_agreement_with_empirical": agree_emp,
            "raw_polarity_agreement": raw_agree,
            "global_rule_matches_anchor": global_bit_matches,
            "rho_lean": lean,
        })
        print(f"  {ck[:32]:32s} hand+anchor {auc_hand:.4f} | rho+anchor "
              f"{auc_der_anchor:.4f} | rho+majority {auc_major:.4f} | "
              f"polarity agree {agree*100:.0f}% | global rule "
              f"{'matches' if global_bit_matches else 'DIFFERS'}", flush=True)

    S.save_csv(os.path.join(out, "per_cell.csv"), rows)
    S.save_csv(os.path.join(out, "per_feature.csv"), feat_rows)

    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)

    # D5: `agree = max(raw, 1-raw)` is a folded statistic — it can never be below
    # 0.5, so 0.5 is NOT its chance level. Monte-Carlo the null at the observed
    # feature counts before calling any agreement number a positive result.
    rng = np.random.default_rng(0)
    ms = [r["m"] for r in rows]
    null = []
    for _ in range(20000):
        draws = [max(v, 1 - v) for v in
                 (rng.binomial(m, 0.5, 1)[0] / m for m in ms)]
        null.append(np.mean(draws))
    null = np.asarray(null)
    obs_hand = float(np.nanmean([r["polarity_agreement_with_hand"] for r in rows]))
    obs_emp = float(np.nanmean([r["polarity_agreement_with_empirical"] for r in rows]))
    null_mean = float(null.mean())
    p_hand = float(np.mean(null >= obs_hand))
    p_emp = float(np.mean(null >= obs_emp))

    n_global_match = int(sum(1 for r in rows if r["global_rule_matches_anchor"]))
    summary = {
        "n_cells": len(rows),
        "macro_hand_anchor": float(np.nanmean(arr("auroc_hand_anchor"))),
        "macro_rho_anchor": float(np.nanmean(arr("auroc_rho_anchor"))),
        "macro_rho_majority": float(np.nanmean(arr("auroc_rho_majority"))),
        "mean_polarity_agreement": float(np.nanmean(
            arr("polarity_agreement_with_hand"))),
        "mean_polarity_agreement_empirical": float(np.nanmean(
            arr("polarity_agreement_with_empirical"))),
        "n_cells_global_rule_matches_anchor": n_global_match,
        "polarity_null_mean": null_mean,
        "polarity_null_ci": [float(np.quantile(null, .025)),
                             float(np.quantile(null, .975))],
        "p_hand_vs_null": p_hand,
        "p_empirical_vs_null": p_emp,
        "rho_anchor_vs_hand": S.wl_wilcoxon(arr("auroc_hand_anchor"),
                                            arr("auroc_rho_anchor")),
        "rho_majority_vs_hand": S.wl_wilcoxon(arr("auroc_hand_anchor"),
                                              arr("auroc_rho_majority")),
        "anchor_bit_cost": S.wl_wilcoxon(arr("auroc_rho_anchor"),
                                         arr("auroc_rho_majority")),
    }

    # which features does the structure most often disagree with us about?
    per_feat = collections.defaultdict(list)
    for fr in feat_rows:
        per_feat[fr["feature"]].append(fr["match"])
    feat_summary = sorted(
        ({"feature": f, "n_cells": len(v), "agreement": float(np.mean(v))}
         for f, v in per_feat.items()),
        key=lambda d: d["agreement"])
    S.save_csv(os.path.join(out, "per_feature_summary.csv"), feat_summary)

    print("\n" + "=" * 72)
    print("PHASE E RESULTS - orientation without an anchor")
    print("=" * 72)
    print(f"  hand polarities + epr anchor (incumbent)  "
          f"{summary['macro_hand_anchor']:.4f}")
    print(f"  rho polarities + epr anchor               "
          f"{summary['macro_rho_anchor']:.4f}  "
          f"({S.fmt_wl(summary['rho_anchor_vs_hand'])})")
    print(f"  rho polarities + rho lean (anchor-FREE)   "
          f"{summary['macro_rho_majority']:.4f}  "
          f"({S.fmt_wl(summary['rho_majority_vs_hand'])})")
    print(f"\n  CHANCE LEVEL for this folded statistic: {null_mean*100:.1f}% "
          f"[{np.quantile(null, .025)*100:.1f}, {np.quantile(null, .975)*100:.1f}]")
    print(f"  sign(rho) vs our DECLARED hand polarities: {obs_hand*100:.1f}%  "
          f"p={p_hand:.3f}  -> "
          f"{'ABOVE chance' if p_hand < 0.05 else 'AT CHANCE - not a result'}")
    print(f"  sign(rho) vs what the LABELS actually say: {obs_emp*100:.1f}%  "
          f"p={p_emp:.3f}  -> "
          f"{'ABOVE chance' if p_emp < 0.05 else 'AT CHANCE - not a result'}")
    print(f"  the structural global rule agrees with the epr anchor in "
          f"{n_global_match}/{len(rows)} cells")
    print(f"  cost of dropping the anchor bit: "
          f"{S.fmt_wl(summary['anchor_bit_cost'])}")

    # D4: the earlier write-up said "three features are mis-signed" because only a
    # top-6 disagreement list had been inspected. Audit the WHOLE pool.
    from sklearn.metrics import roc_auc_score as _auc
    mis = []
    for f in sorted({fr["feature"] for fr in feat_rows}):
        a = [_auc(c["labels"], c["V"][:, c["pool"].index(f)])
             for c in cells.values() if f in c["pool"]]
        mis.append({"feature": f, "n_cells": len(a),
                    "oriented_auroc": float(np.mean(a)),
                    "n_cells_below_half": int(sum(1 for v in a if v < 0.5))})
    mis.sort(key=lambda d: d["oriented_auroc"])
    wrong = [d for d in mis if d["oriented_auroc"] < 0.5]
    summary["n_features_mis_signed"] = len(wrong)
    summary["n_features_audited"] = len(mis)
    S.save_csv(os.path.join(out, "hand_sign_audit.csv"), mis)
    print(f"\n  HAND-SIGN AUDIT over the WHOLE pool: {len(wrong)}/{len(mis)} "
          f"features have oriented AUROC < 0.5")
    for d in mis[:8]:
        print(f"    {d['feature']:<26} AUROC {d['oriented_auroc']:.4f}  "
              f"below 0.5 in {d['n_cells_below_half']}/{d['n_cells']} cells")

    S.save_json(os.path.join(out, "summary.json"), summary)
    render(out, rows, feat_summary, summary)
    print(f"\nWrote -> {out}")


def render(out, rows, feat_summary, summary):
    body = []
    body.append("<h2>What is being replaced</h2>")
    body.append(
        "<p>Our detector currently needs two hand-supplied things to know which "
        "direction is which: 42 per-feature polarities, and one anchor feature "
        "(<code>epr</code>) that fixes the global direction. Step 201 already "
        "proved the 42 polarities are <b>free</b> — L-SML is exactly invariant to "
        "them. The anchor bit was never free. U-PCR can derive both from "
        "structure: the sign of each feature's estimated correlation with "
        "correctness gives the relative polarity, and because the shared-signal "
        "term is a squared quantity it cannot be negative, so the estimates must "
        "lean positive overall — which fixes the global direction without any "
        "anchor.</p>")

    body.append("<h2>Does dropping the anchor cost anything?</h2>")
    body.append(S.bar_chart(
        ["hand polarities + epr anchor (incumbent)",
         "structure-derived polarities + epr anchor",
         "structure-derived polarities + structural global rule (anchor-free)"],
        [summary["macro_hand_anchor"], summary["macro_rho_anchor"],
         summary["macro_rho_majority"]],
        "macro AUROC over 25 test sets"))
    body.append(f"<p class='note'>Cost of removing the anchor bit specifically: "
                f"{S.fmt_wl(summary['anchor_bit_cost'])}. The structural rule "
                f"agrees with the anchor's decision in "
                f"{summary['n_cells_global_rule_matches_anchor']}/"
                f"{summary['n_cells']} cells.</p>")

    order = sorted(rows, key=lambda r: -r["polarity_agreement_with_hand"])
    body.append("<h2>Does structure recover the hand-derived polarities?</h2>")
    body.append(S.bar_chart(
        [S.plain_cell(r["cell"]) for r in order],
        [r["polarity_agreement_with_hand"] for r in order],
        "fraction of features whose polarity structure agrees with ours",
        hlines=[("coin flip", 0.5)], value_fmt="{:.2f}"))

    body.append("<h2>Features the structure reads differently</h2>")
    body.append(S.html_table(
        ["feature", "cells", "polarity agreement"],
        [[d["feature"], d["n_cells"], f"{d['agreement']*100:.0f}%"]
         for d in feat_summary],
        numeric_cols=(1, 2)))

    body.append("<h2>Per cell</h2>")
    body.append(S.html_table(
        ["test set", "hand + anchor", "rho + anchor", "rho + structural (free)",
         "polarity agreement", "global rule matches anchor?"],
        [[S.plain_cell(r["cell"]), f"{r['auroc_hand_anchor']:.4f}",
          f"{r['auroc_rho_anchor']:.4f}", f"{r['auroc_rho_majority']:.4f}",
          f"{r['polarity_agreement_with_hand']*100:.0f}%",
          "yes" if r["global_rule_matches_anchor"] else "no"] for r in order],
        numeric_cols=(1, 2, 3, 4)))

    S.write_page(
        os.path.join(out, "index.html"),
        "Orientation without an anchor",
        "U-PCR study, Phase E — replacing the last hand-picked prior with a "
        "structural rule",
        [f"Fully anchor-free orientation scores "
         f"<b>{summary['macro_rho_majority']:.4f}</b> vs "
         f"{summary['macro_hand_anchor']:.4f} for the hand-tuned incumbent "
         f"({S.fmt_wl(summary['rho_majority_vs_hand'])}).",
         f"The structure recovers our hand-derived polarities on "
         f"<b>{summary['mean_polarity_agreement']*100:.1f}%</b> of features.",
         f"The structural global rule agrees with the <code>epr</code> anchor in "
         f"<b>{summary['n_cells_global_rule_matches_anchor']}/"
         f"{summary['n_cells']}</b> cells.",
         "Step 201 already showed the 42 polarities are free on the L-SML path; "
         "what this removes is the one anchor bit that was not."],
        "".join(body))


if __name__ == "__main__":
    main()
