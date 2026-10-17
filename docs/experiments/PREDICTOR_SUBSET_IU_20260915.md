# Step389 — all 3/4/5-predictor subsets before selecting a fusion

User authorization: compare the five available predictors and choose the best
development configurations. Frozen before scoring any new fusion.

Predictors, in column order: ridge, tcn, bocpd, noreset, mean16. Reuse the
innovation5 seed0 source-excluded models from Steps379/388. No new training.
Enumerate all10 triples,5 quadruples and the quintuple. For each use native
IU-PCR and an equal-weight control,32 configurations total.

Each token contributes one signed residual per predictor:
r[t,j] = mean_k(z[t,k] - predicted_j[t,k]), over the same5 feature targets.
Fit fusion separately inside each answer, without correctness labels.
Center and standardize each residual column over the answer (ddof0).
Both IU and equal score those same standardized residuals. No new feature signs.
IU uses exactly IU_FIT_DEFAULTS: L2,2 PCs, g2 projection1, scale_ratio.25,
300-point g2 grid, no exclusion/difficulty gate/mean fallback. A vectorized grid
calculation must agree with canonical upcr_fit_covariance on independent
numerical tests. Orient the fused signal toward the equal residual signal by
their answer-local covariance; record flips and signed/large coefficients.

Fuse tokens first, Top10 of that single curve within each step, then the
unchanged signed .25 correction scaled to the base step-score standard
deviation. Add innovation5 base ONCE. Same tail15 percentile>=.33 gate.
No gate/gamma/Top10-order/normalization sweep. This study learns one vector of
predictor weights per answer; it does not learn token-varying routing.
Constant columns or nonfinite fits are explicit failures, never silently
replaced by equal. A constant final auxiliary produces zero correction and
is recorded. Near-collinear inputs retain the canonical spectral rule.

Full13,769 answers/145,597 steps/6,968,779 tokens required for quality.
Five outer exclusions and ten pair exclusions for PRMB calibration:
both learned predictors exclude the held and calibration-source folds.
Answer-local predictors/fusion use only the respective answer, and may be
reused across exclusions. Whole-answer normalization remains offline.
Nested fits are reused; recompute required predictions, not model training.
Replay all five singleton corrections against their frozen scores and replay
Ridge/TCN nested corrections against the saved exclusion-specific scores.

References: all16 Step388 rows, including individual features, entropy15,
RBM12, original4, innovation5, the predictors and history controls.
PB and within are co-primary; PRMScore is a reported tie-breaker.
Display every configuration, per-cell PB, size leaders and Pareto frontier.
Report separate PB and within leaders, using the other endpoint, then
PRMScore, then fewer predictors, then lexical order to resolve exact ties.
Do not pretend a tradeoff produces one universally best method.

16 primary IU-minus-matched-equal pairs x2 endpoints;10,000 source-group
paired bootstrap draws; Bonferroni CI=1-.05/32=99.84375%.
All32 candidates versus ridge,tcn__real,bocpd have exploratory95% intervals;
these do not confirm an adaptively selected winner. Selecting a bank here
uses development labels. The shared population and seed0 are not independent
confirmation. A winner needs a frozen study on unused questions.

Save input/source/model hashes, all per-answer weights and fit diagnostics,
coverage, costs and exact scores. Independent PB and pairwise within checks
must cover all48 methods. Preserve previous results and keep flows paused.
