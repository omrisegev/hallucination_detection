# Step387: aligned predictors before fusion

Frozen before new real-data scoring. User authorized testing predictors first,
and fusion only after deciding whether they provide useful error evidence.

## Question
Does an alternative estimate of the same five current telemetry features
improve the successful signed-residual correction? Prediction MSE, residual
agreement, and error localization are separate outcomes. No U-PCR fitting.

## Fixed scope
- All 13,769 answers, 145,597 steps, 6,968,779 tokens; frozen source groups,
  folds, labels, innovation5 features, signs and whole-answer normalization.
- Original innovation5 step scores remain untouched. For each predictor:
  average the five signed standardized residuals at each token, Top10 within
  each step, then base + .25 * std(base) * standardized auxiliary across steps.
  Same frozen tail15 percentile >= .33 gate. No gate, feature, TopK or dose search.
- Predictor target: the same standardized five-vector z_t. Prediction is made
  before ingesting z_t. Context crosses steps; preprocessing remains OFFLINE.
- `ridge`: saved 16-lag + mask + relative-position predictor, 5 outer models;
  source-excluded training, fixed regularization1. Reproduce saved scores.
- `mean16`: mean of available preceding16 vectors, zero without history.
- `bocpd`: five separate exact, untruncated Gaussian segment-mean filters,
  fixed observation variance1, prior mean0/variance1, hazard1/32. Predictive
  mean averages reset and continuation hypotheses BEFORE the current datum.
  Each feature has its own run-length posterior; no claim of independent
  errors between the five telemetry features. This is a declared adaptation
  of reset-before-observation BOCPD, not an error-probability detector.
- `noreset`: same conjugate Gaussian model with hazard0 (prefix sum/(t+1));
  isolates change-point adaptation from shrinkage/background estimation.
- `zero`: zero standardized prediction; shared-current-observation control.
  It is not a history-free end-to-end method: innovation5 already has history.
- Methods with fixed recurrences use current answer only after shared offline
  preprocessing. Ridge borrows other answers. No correctness labels in scoring.
- No TCN/FM/DiFlo retraining in this bounded stage; partial old neural scores
  cannot establish quality. Their queue and STOP_AFTER_JOB remain untouched.

## Evaluation and decision
Primary comparisons: BOCPD minus Ridge/base/noreset; mean16 minus Ridge/base.
Two endpoints PB macro F1 and PRMB within-answer AUC: 10 contrasts, 10,000
paired source-group bootstrap draws, Bonferroni99.5% intervals. All other
contrasts descriptive secondary95%. No equivalence claim from a null interval.
PRMScore uses the frozen .8 quantile: label-free recurrent scores from other
folds; historical Ridge retains its original nested pair-excluded thresholds.
Keep original4, innovation5, same-gate singles, entropy15 and RBM12 references.

Report per-feature prediction MSE (all tokens and t>=16), prediction and
residual correlations, and correlations after removing the shared current
scalar observation linearly within each answer. Those latter diagnostics are
postprocessing, not predictors or proof of conditional error independence.
Report gain/loss of correct peaks relative to Ridge/base and their locations;
an oracle union of existing peaks is not a fusion performance bound.
MSE improvement or different residuals alone cannot promote a detector. A
useful standalone alternative or complementary localization evidence motivates
a later explicitly registered fusion experiment; this stage never fits one.
All task results remain development evidence; the .25 signed Ridge anchor and
innovation5 bank were already selected using this population's labels.

## Validation and runtime
Exact short-sequence partition enumeration for BOCPD, noreset closed form,
prefix-only invariance conditional on fixed normalization, short/constant
traces, scalar readout reconstruction and reference score replay. Independent
PB and pairwise within-AUC audit for every scored method. Resume by answer,
strict metadata allowlist, manifests/hashes, explicit failures, no fallback.
Feasibility checks may measure throughput but never rank quality. Full-data
scoring is capped at two CPU hours; stop with partial status if exceeded,
without interpreting partial quality. No data or historical artifact mutation.
