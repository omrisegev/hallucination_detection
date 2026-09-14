# Probability normalization and mass ablation v1

Status before execution: **PROTOCOL FROZEN / NOT YET SCORED**  
Frozen on: 2026-09-14  
Population: the existing 13,769-answer localization development contract only.

## Question

Does the apparent loss from the strong standalone escort-varentropy views to
multi-view fusion come from:

1. renormalizing the retained top-K probabilities to sum to one;
2. discarding probability mass outside the retained head; or
3. standardizing each feature separately inside every answer before fusion?

This is a diagnostic development experiment. It does not provide untouched
confirmation and it does not authorize selection of a final deployable method.

## Fixed contracts

- Same 13,769 answers, official step spans, source-group folds and labels as
  `renyi_position_temporal_fusion_v1`.
- Same Top10 token mean within every official step; earliest argmax on ties.
- Same frozen ProcessBench mean-entropy quantile-0.3 gate.
- Same PRMScore quantile-0.8 rule. Any other-answer standardizer uses the same
  outer-fold exclusion, with two-fold exclusion for PRMScore calibration.
- No gate optimization and no alpha selection occurs in this experiment.
- All token/step scores are frozen before evaluation metrics are opened.

## Mathematical identity control

For a retained head with raw probabilities `p_i`, `S=sum_i p_i` and
`q_i=p_i/S`, escort weights satisfy

`q_i^alpha / sum_j q_j^alpha = p_i^alpha / sum_j p_j^alpha`.

The two surprisals differ by the constant `log(S)`, which cannot change a
variance. Therefore a proper escort-varentropy computed from the raw retained
head must equal the same escort-varentropy computed from the conditional head
`q`, up to floating-point error. This is an integrity assertion, not a
performance arm. A naive unnormalized weighted second moment is not called
varentropy and is not promoted as a method.

The scored identity controls must also reproduce:

- frozen `k15__raw` with `ve1_q15_raw`;
- frozen `k50__raw` with `ve1_q50_raw`;
- frozen Stage-3b `view__ve0.75` with `ve075_q15_raw`; and
- frozen four-view `equal` with `equal4_answer_z`.

## Scored roster

### Representation arms

- `ve075_q15_raw`: frozen alpha-0.75 escort varentropy on conditional top 15.
- `ve075_q50_raw`: same definition on conditional top 50.
- `ve075_tailbucket15_raw`: alpha-0.75 escort varentropy on the coarse
  probability distribution `[p1,...,p15,1-sum(p1,...,p15)]`.
- `ve1_q15_raw`: frozen top-15 varentropy.
- `ve1_q50_raw`: frozen top-50 varentropy.
- `ve1_tailbucket15_raw`: ordinary varentropy on the same 16-category coarse
  distribution.
- `tail15_raw`: residual mass `1-sum(p1,...,p15)` alone, high-is-risk.

Every VE arm is oriented within the answer by correlation with the frozen
top-15 varentropy anchor, exactly as in the alpha sweep. The tail arm has its
natural high-is-risk orientation.

### Four-view preprocessing arms

The four views are the frozen `H0lim`, `VE0`, `VE0.75`, and `VE1` bank with
the same answer-local orientation as the completed temporal-fusion study.
Weights are fixed and equal; only preprocessing changes.

- `equal4_raw`: no centering or scaling.
- `equal4_center_only`: subtract each view's within-answer mean.
- `equal4_scale_only`: divide each view by its within-answer standard deviation.
- `equal4_answer_z`: current within-answer centering and scaling.
- `equal4_fold_global_z`: standardize using equal-source-group/equal-answer
  moments from other folds in the same benchmark cell.
- `equal5_tail_fold_global_z`: the preceding arm plus `tail15_raw` as a fifth
  equally weighted, externally standardized view.

The expected exact within-answer invariances are registered in advance:
`equal4_raw` and `equal4_center_only` must have identical step ordering;
`equal4_scale_only` and `equal4_answer_z` must have identical step ordering.
Their pooled calibration may differ.

## Evaluation and contrasts

Report ProcessBench all-eight macro F1, raw exact error localization, PRMBench
mean within-answer AUROC, pooled AUROC, and PRMScore. Also report the maximum
identity-control score error and the residual-tail distribution.

Six primary diagnostic contrasts are family-wise corrected with 10,000 whole
source-group bootstrap draws (99.1667% intervals):

1. `ve075_q50_raw - ve075_q15_raw`
2. `ve075_tailbucket15_raw - ve075_q15_raw`
3. `ve1_q50_raw - ve1_q15_raw`
4. `ve1_tailbucket15_raw - ve1_q15_raw`
5. `equal4_fold_global_z - equal4_answer_z`
6. `equal5_tail_fold_global_z - equal4_fold_global_z`

All other comparisons are descriptive. A representation or preprocessing arm
may be carried to the next experiment only if it improves one primary axis by
at least 0.5 ProcessBench percentage points or 0.002 PRMB within AUROC while
remaining noninferior on the other axis within those same margins. This is a
development promotion rule, not a generalization claim.

## Required artifacts

- resumable feature/score checkpoint and source/code/protocol manifest;
- unit and real-data smoke reviews;
- score-freeze/identity review before evaluation;
- `SCORES.npz`, `METRICS.json`, `CONTRASTS.json`, `COMPARISON.csv`;
- a concise `REPORT.md` and an entry in
  `docs/experiments/RENYI_FUSION_FOLLOWUP_LOG.md`.

