# PRM-anchored view selection: stage 1 pre-registration, exhaustive subset search (2026-10-25)

Omri, 2026-10-25 (Hebrew, verbatim): "כן. תכתוב רישום מוקדם עם התכנון המעודכן: עוגן PRM ומשקלות אי-שליליים, SML
בתוך קבוצה ו-DS-EM בין הקבוצות (HEM כגרסה משנית), משקלות בינאריים מופעלים על ערכים רציפים, ומינימום של 3 קבוצות בכל
תא. תמדוד זמן על מדגם ואז תריץ הלילה."

Translation: write the pre-registration with the updated plan:
- PRM anchor and non-negative weights;
- SML within a group, DS-EM between groups, HEM as a secondary variant;
- binary-estimated weights applied to continuous values;
- a minimum of 3 groups in every cell.

Time it on a sample, then run it overnight.

**Goal (Omri):** one telemetry subset S, shared by all five PRMs. Fused with the PRM by label-free L-SML, refitted from
scratch in every benchmark × PRM cell, it beats both the PRM alone and the plain mean of the same views in all 10
cells (5 PRMs × PRMBench and Socratic).

The subset is shared; the fitted weights are not. Background: stage 0 (`results/prm_view_selection_v1/stage0/`,
Step 478).

## Data, views, representation (as in stage 0, amendment 2)

- **Cells.** 10: PRMBench (non-control answers in the metric) and Socratic, × qwen_prm_7b, qwen_prm_72b, pure_prm,
  rlhflow_prm, math_shepherd.
- **Views.** The PRM error score (STATUS_META reference scale) and the 16 N16 telemetry channels.
- **Orientation.** Each channel is multiplied by the sign of its correlation with the PRM on answer-z values, over the
  pool. This makes the PRM the anchor. Label-free.
- **Continuous value.** The answer-z of every oriented view.
- **Binary vote.** Per-answer GAP on the answer-z (top-k = +1 = error). A view constant within an answer abstains
  (all −1).
- **Estimation pool.** `keep` steps of answers with more than one step. The fit is pooled across answers within one
  cell: transductive and label-free.

## The method (fitted from scratch in each cell for each subset S; no label enters)

1. **Groups.**
   - Telemetry groups of S are found from the vote covariance R_S over the pool, with exactly the logic of
     `fusion_utils.detect_dependent_groups` (Jaffe, Fetaya and Nadler 2016, Algorithm 1): residual method, default
     K_range 2..min(|S|, 9)−1, exact enumeration at |S| ≤ 4, `unit` loading scale.
   - The PRM is its own group (the CHAIN structure).
2. **Minimum structure.** S is evaluated only if every one of the 10 cells finds K ≥ 3 telemetry groups. The between
   level is then the PRM plus at least 3 virtual views (4 or more), where the rank-one fit has degrees of freedom.
   - Smaller structures are skipped and counted. Since K ≤ |S|−1, this means |S| ≥ 4.
3. **Within a group: SML** (Parisi et al. 2014).
   - Weights: the leading eigenvector of the zero-diagonal covariance of the group's votes.
   - Sign: the sum of the weights is made positive. The views are already oriented by the PRM, so a negative entry is
     a model violation, not an inverted view (Jaffe 2016 Lemma 4).
   - Negative entries are clipped to 0. A singleton group gets weight 1.
   - The group's virtual vote is ξ_g = sign(V_g w_g). A tie (0) takes the PRM's vote (the anchor).
4. **Between groups: two-coin Dawid-Skene EM** on [PRM vote, ξ_1, …, ξ_K], over the pool.
   - Run on compressed vote patterns; initialized from the majority vote; at most 500 iterations; absolute tolerance
     1e-9 on the log-likelihood.
   - Label switching is resolved by the anchor (PRM ψ + η > 1).
   - Weights are the ML log-odds w = log(ψη / ((1−ψ)(1−η))), clipped at 0.
   - Jaffe 2016 states that each stage is a Dawid-Skene model solvable by any DS method. Step 457 found DS-ML weights
     the best between-group estimator.
5. **Scoring: binary-estimated weights applied to continuous values.**
   - Composite of group g: the answer-z of (Z_g w_g).
   - Fused score: w_PRM · Z_PRM + Σ_g w_g · composite_g.
   - Flags: GAP on the answer-z of the fused score.
6. **Secondary scoring (binary).** w_PRM · v_PRM + Σ_g w_g ξ_g + 1e-6 · Z_PRM (the anchor breaks ties), then GAP.
7. **Secondary estimator: HEM.**
   - The exact latent-group EM of the 2016 model (`scripts/experiments/cvf_v2/em.py`, `fit_em(kind='hem')`, with the
     same groups and the PRM as its own group).
   - Linearized: within-group weight log(e₁(1−e₀)/(e₀(1−e₁))) and between-group weight from the transitions, both
     clipped at 0, applied to the continuous values as in step 5.
   - HEM is too slow for every subset. It runs only on the successful subsets and the 300 subsets with the largest
     primary min-margin.

## Comparators per cell

- **PRM alone under the same GAP rule.** Equals STATUS_MAIN "PRM same rule (better scale under GAP)"; checked.
- **The plain mean of the same views:** mean(Z_PRM, Z_S), then GAP.
- **Context only:** CHAIN_N16, TELE_N16, and for the selected subsets a supervised logistic regression on the same
  views (5-fold GroupKFold by source group).

## Search space

All subsets of the 16 channels with |S| ≥ 4: 64,839 subsets × 10 cells. Subsets failing the minimum structure in any
cell are skipped.

## Metric and success

- **Metric.** Official PRMScore under GAP: PRMBench pooled confusion, identical to `prmbench_evaluate`; Socratic
  official pooled F1 with empty steps counted as predicted incorrect. Both on a 0–100 scale.
- **Margin in a cell:** fused − max(PRM alone, mean of S).
- **Success of S:** valid in all 10 cells and margin > 0 in all 10.
- **Speed and exact re-check.** The search uses vectorized answer-z / GAP / top-k. Before the run, its flags must
  equal the evaluator's loop on the PRM view and on 50 random fused scores (PRMScore difference < 1e-9). Every
  subset reported by name is recomputed with the evaluator's own functions.

## Analyses (all pre-specified)

1. **Counts.** Valid subsets, successful subsets (overall and by size), and the distribution of the min-margin.
2. **Channel frequency** among successful subsets against their frequency among valid subsets.
3. **Cross-benchmark.**
   - Select S_PB = argmax over S of the min-margin over the 5 PRMBench cells; report its 5 Socratic margins.
   - Select S_SO the same way on Socratic; report its PRMBench margins.
4. **Leave-one-PRM-out.** For each PRM, select by the min-margin over the other 8 cells and report the two held-out
   cells.
5. **Omri's structural conditions** on the successful subsets versus all valid subsets, label-free, binary, per
   cell:
   - each channel's correlation with the PRM exceeds its mean correlation with the other channels in S;
   - mean pairwise correlation within S.
6. **For the top subsets:** within-answer AUC, HEM, binary scoring, CHAIN_N16 and the supervised LR, as context.

## Mechanism predictions, written before the run

Channel families:
- **Uncertainty level:** q15_H1, q15_VE1, chosen_surprisal, logprob_margin, energy_level, true_tail50.
- **Change / dynamics:** energy_innovation, top15_turnover, top50_js, bocpd_p0, realized_drv.
- **Realized token against the distribution:** realized_z.
- **Spectral:** dominant_freq16.
- **Digits:** digit_alternative, digit_spread, digit_alternative_innovation.

Predictions:
- **M1. Diversity.** Successful subsets take channels from different families. At most one of the three digit
  channels is chosen more often than chance, because they are near-copies (stage 0: residual partial correlation
  .997). This follows the 2016 paper's selection rule: "the most accurate classifiers, from different groups".
- **M2. Which channels.** Channels that stayed above chance on Socratic for every PRM in stage 0 are over-represented.
  Channels below chance in some Socratic cell in stage 0 (for example realized_drv, bocpd_p0, q15_H1 for
  Math-Shepherd) are under-represented.
- **M3. The PRM's weight.** In successful subsets, the PRM keeps the largest between-group weight in most cells.
- **M4. Prior on the outcome** (honest; stage 0 showed telemetry far below the PRM on Socratic): a subset succeeding
  in all 10 cells is rare or absent. The hardest cells are Socratic Qwen-72B and Math-Shepherd. If no subset
  succeeds, the count of cells won and the cross-validation results are the result.

## Labels

Labels enter only the PRMScore evaluation of each subset, i.e. the selection itself. The selection is therefore
label-based development on these two benchmarks: in-sample, not a held-out claim.

Evidence of generalization comes only from:
- the cross-benchmark and leave-one-PRM-out analyses;
- later datasets (ProcessBench, Hard2Verify, Qwen3 self-generated answers), which need their own approval.

## Not in scope

GPU jobs; new inference; any change to frozen results; promoting a subset as the method before held-out data.

## Pre-run checks (2026-10-25, before the search)

`stage1.py check` writes `results/prm_view_selection_v1/stage1/PARITY.json`.
- **Groups.** `groups_from_cov` equals `detect_dependent_groups` on 50/50 random subsets in all 10 cells.
- **Fused scores.** Vectorized GAP flags equal the evaluator's loops on 345/345 valid random fused scores, with
  PRMScore difference 0.
- **PRM view.**
  - The PRM's quantized scores tie exactly, so vectorized flags differ from the loop in 6 cells (the 1e-16 effect
    documented in `evaluate_collection.answer_z`).
  - The PRM-alone comparator is therefore always computed through the evaluator's exact loop. It equals STATUS_MAIN
    in 10/10 cells.
  - The PRM enters the fused scores only as one continuous term, so this does not affect them.
- **Timing.** 0.365 s per subset across all 10 cells, with early exit when a cell has K < 3. About 6.6 core-hours in
  total. 5/40 random subsets were valid in all 10 cells.

## Post-run disclosures (2026-10-26)

**Engineering change before the search, no effect on results.** Workers load light per-cell caches
(`stage1.py cache`), because 16 full workers exceeded memory (about 5 GB each). The caches reproduce chunk 0
bit for bit.

The independent verification found the following, and none of them changes the headline (0 of 12,247 valid subsets
meets the goal):
1. **Unregistered constant.** DS-EM clips ψ, η and the prevalence to [1e-6, 1−1e-6].
   - In Socratic cells the PRM's estimate is often at that boundary, so the floor sets its weight.
   - The secondary counts "2 beat the PRM in all 10" and "4 win all 5 Socratic cells" hold at floors 1e-12, 1e-9 and
     1e-6, and break at 1e-4.
2. **Fitting pool.** On PRMBench it excludes the 758 control answers. That is label-informed metadata, as in stage
   0; with all answers in the pool, one tested margin changes sign.
3. **DS-EM details.**
   - It uses soft-majority initialization and a tolerance on the mean log-likelihood. The pre-registration says
     hard majority and the total.
   - The 500-iteration cap is sometimes hit.
4. **An invalidity rule** (all between-group weights clipped) existed, but never fired.

**Exploratory, not pre-registered.** A fork of this session diagnosed Math-Shepherd's Socratic failure as step
position, and probed position-residualized channels (`results/prm_view_selection_v1/position_probe/`). It also
counted the 8 cells without Math-Shepherd: 9 subsets beat both comparators. Any use of these needs its own
amendment before a rerun.
