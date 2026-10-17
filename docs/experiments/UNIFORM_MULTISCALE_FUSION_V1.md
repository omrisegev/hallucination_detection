# Uniform multiscale Renyi/varentropy fusion v1

Status before execution: **PROTOCOL FROZEN / NOT YET SCORED**  
Frozen on: 2026-09-14  
Population: the existing 13,769-answer localization development contract only.

## Question

Can one benchmark-independent representation and weighting rule resolve the
q15/q50 and raw-scale tradeoffs found in Experiment 1?

This is an adaptive development experiment. Experiment 1 aggregate outcomes
were already known when this protocol was written. It may select a candidate
for later new-model confirmation, but it is not confirmation itself.

## Uniformity contract

- No feature, support width, preprocessing operator, weight vector, epsilon or
  selection rule may depend on benchmark identity, dataset, model or test cell.
- One model is fitted across both benchmark panels. Each held source-group fold
  receives the same global model; there are no PB and PRMB heads.
- Outer scoring excludes the held fold globally. PRMScore calibration scores
  use a second exclusion so neither the evaluated fold nor the calibration
  answer's fold enters its scale or supervised weight fit.
- ProcessBench and PRMBench receive equal training mass. Cells are equal inside
  a panel, source groups are equal inside a cell, and answers are equal inside a
  group. Known step classes are then globally balanced.
- Same frozen mean-entropy q=.3 ProcessBench gate, PRMScore q=.8 threshold,
  official steps and source-group folds as Experiment 1.
- Each view is reduced to a step by its own Top10 token mean. Fusion follows the
  readout, so learned weights operate on one fixed step-feature matrix and do
  not change which tokens define another view.

## Fixed feature banks

The four views at each support are `H0lim`, `VE0`, `VE0.75`, and `VE1`.
`q15` and `q50` mean the retained head is normalized on exactly 15 or 50 saved
probabilities. Every column is oriented within the answer by its Pearson sign
against the frozen q15 `VE1` anchor. This is label-free and matches the
completed q15 fusion orientation.

- `q15`: four q15 views.
- `q50`: the same four definitions on q50.
- `ms8`: q15 and q50 concatenated; it is the only multiscale bank.

No coarse tail bucket is reopened.

## Fixed weighting arms

For every bank `B`:

1. `B_raw_equal`: equal weights in natural units.
2. `B_scale_equal`: divide columns by one global other-fold scale vector,
   without subtracting a mean, then use equal weights.
3. `B_scale_natural`: after the same scaling, use weights proportional to the
   fitted scales. This must reconstruct the raw-equal ordering and is an
   algebraic integrity control, not a new candidate.
4. `B_scale_simplex`: fit one non-negative weight vector with sum one by
   class-balanced step logistic loss across both benchmark panels. After the
   fit, weights below epsilon `0.02` are set identically to zero and the
   surviving weights are renormalized. The same vector scores every held cell.
5. `B_scale_simplex_centered`: subtract each answer's per-column step mean
   immediately before applying the preceding frozen simplex weights. It must
   preserve within-answer ordering relative to the full simplex arm and tests
   the lost answer-level channel.

The supervised fit uses SLSQP from the equal simplex point, one start, bounds
`[0,1]`, exact sum-one constraint, an unconstrained intercept, L2 coefficient
`1e-3`, maximum 300 iterations and analytic gradients. A non-converged or
non-finite fit is a declared failure; it cannot fall back to equal weights.

Global scales use feature moments with the same panel/cell/group/answer
hierarchy as the supervised loss. Means are recorded but are not subtracted in
the full arms.

The frozen Experiment-1 token-fusion `equal4_raw` is included only as a
continuity reference. It is excluded from candidate selection because its
fusion-before-Top10 readout differs from this experiment's uniform readout.

## Evaluation and selection

Report ProcessBench all-eight macro F1 and raw exact localization; PRMBench
mean within-answer AUROC; mean of held-fold pooled AUROCs; descriptive
concatenated OOF pooled AUROC; and PRMScore.

Primary contrasts use 10,000 paired whole-source-group bootstrap draws and a
family-wise 99.375% interval:

1. `q50_raw_equal - q15_raw_equal`
2. `ms8_raw_equal - q15_raw_equal`
3. `q50_scale_equal - q15_scale_equal`
4. `ms8_scale_equal - q15_scale_equal`
5. `q15_scale_simplex - q15_scale_equal`
6. `q50_scale_simplex - q50_scale_equal`
7. `ms8_scale_simplex - ms8_scale_equal`
8. `ms8_scale_simplex - ms8_scale_simplex_centered`

The benchmark-uniform development candidate is chosen only among the nine
non-centered `{q15,q50,ms8} x {raw_equal,scale_equal,scale_simplex}` arms.
Let `PB*` and `PRM*` be the best values in that roster. For arm `m`,

`regret(m) = max((PB* - PB(m)) / .005, (PRM* - PRM(m)) / .002)`.

Choose the smallest regret; exact ties prefer fewer views, then equal over
simplex over raw, then lexical method name. This balances the two previously
registered noninferiority margins and cannot select one arm per benchmark.

## Required boundaries and artifacts

- unit tests and a real-data smoke before a full run;
- resumable extraction, model and score checkpoints;
- explicit record that supervised labels are used only for other-fold fits;
- hash-frozen OOF scores before aggregate metrics are computed;
- exclusion, reconstruction, orientation and centered-ordering reviews;
- `METRICS.json`, `CONTRASTS.json`, `COMPARISON.csv`, `WEIGHTS.json`,
  `SELECTION.json`, `ERROR_ANALYSIS.json`, `REPORT.md`, and an append-only
  update to `RENYI_FUSION_FOLLOWUP_LOG.md`.

