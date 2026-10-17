# Selected q15 finalist replay v1

Status before execution: **PROTOCOL FROZEN / NOT YET REPLAYED**  
Frozen on: 2026-09-14  
Population: the frozen 13,769-answer localization development contract.

## Question

What is the result of the single deployable specification selected after
Experiments 1 and 1B, and how much does each accepted decision change the
original static and position-temporal implementations?

This is a deterministic development replay, not a new selection experiment and
not untouched confirmation. No feature, support, weight, gate, benchmark or
threshold is tuned in this replay.

## Frozen finalist

1. Use exactly the q15 views `H0lim`, `VE0`, `VE0.75`, and `VE1`.
2. Preserve each view's frozen label-free orientation.
3. Reduce each view separately to an official step by its Top10 token mean.
4. Average the four resulting step scores in their natural units.
5. Do not subtract the answer mean and do not divide by an answer or global
   feature scale.
6. Use no supervised simplex and no benchmark-specific weights.
7. Use the same mean-entropy q=.3 ProcessBench no-error gate, earliest-argmax
   tie rule, and nested PRMScore q=.8 calibration contract as the source runs.

The four per-view step arrays are replayed from the independently reviewed
Rényi position run. Their mean must be bitwise identical to the already frozen
`q15_raw_equal` score from Experiment 1B.

## Frozen comparisons

The finalist is compared with three already-frozen starting points:

- `original_static_fusion_before_top10`: the initial q15 raw equal fusion at
  token level followed by Top10.
- `original_position_equal_z`: the original position experiment's equal fusion,
  which first centers and scales every view inside the answer and then applies
  Top10 to the fused token stream.
- `original_position_local_shrink`: the original answer-local Shrinkage-IU plus
  16-bin position covariance prior, also fitted on the answer-standardized bank
  and read out after token-level fusion.

Primary ProcessBench and PRMB-within contrasts use 10,000 paired whole-source-
group bootstrap draws and a family-wise 98.333% interval over these three fixed
comparisons. PRMScore uses each source run's already exclusion-safe fold
thresholds. The replay reports absolute metrics and deltas; it cannot change the
selected method.

## Required review

- exact 145,597-step roster;
- finite arrays with identical shape;
- finalist reconstruction bitwise equal to frozen `q15_raw_equal`;
- score archive frozen and hashed before aggregate evaluation;
- exact reuse of the registered calibration thresholds;
- machine-readable metrics, contrasts, comparison table and PASS/FAIL review.

