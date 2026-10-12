# Localization short cycle 2 — fixed-group Joint diagnostic

Date: 2026-09-07. Status: registered before fitting the new arms.

## Question

Did answer-only Joint underperform in short cycle 1 because its four temporal
blocks could not support stable data-derived feature clustering, or because the
Joint factor/map itself is not useful on one answer's windows?

## Frozen input and evaluation

- Reuse exactly the 30 answer IDs frozen in short cycle 1.
- Reuse their already-computed 30 window features. Do not reopen labels during
  fitting.
- Keep width 32, stride 32, per-answer active/constant filtering,
  per-answer standardization and orientation, and the fixed official-step
  span-maximum readout.
- Keep the short-cycle-1 internal-group Joint, IU and equal scores as references.
- Freeze all new scores before loading the PRMBench step labels.
- Report available and strict coverage separately. Resample whole answers for
  paired diagnostic intervals; do not treat steps as independent bootstrap
  units.

## New label-free arms

The four groups are fixed by feature provenance and never fitted:

1. entropy-trace features: `epr` through `cusum_shift_idx`;
2. sampled-token/spilled-energy features;
3. energy-series features;
4. next-token-distribution features.

Inactive features are removed per answer. An arm fails closed if fewer than
four groups survive or any surviving group has fewer than three features.

- `joint_fixed4_modelinv_lam0`: fit the same Joint factor model and lambda-zero
  model-inverse map as short cycle 1, using the fixed four groups.
- `fixed4_continuous_lsml`: use continuous L-SML with the same fixed groups.
  This diagnostic separates the fixed partition from the Joint factor fit.

No lambda sweep, graph regularizer, learned trajectory reducer, threshold,
pooled answer fit, or cross-answer calibration is admitted.

## Interpretation gate

- A coverage gain means the data-derived clustering was a stability bottleneck.
- A score gain over the original Joint on their strict common answers means the
  fixed partition also repairs some ranking loss.
- Joint is not revived as the leading localizer unless fixed-group Joint has
  near-complete strict coverage and matches or exceeds IU on their strict
  pairwise comparison.
- If fixed-group Joint remains below IU, prioritize IU/equal for feature fusion
  and study the trajectory axis separately. Do not start a broad Joint or
  lambda sweep from an ambiguous 30-answer result.

