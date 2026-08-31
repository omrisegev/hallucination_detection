# Reasoning Localization 0.3662 — H2/H3 Role-Separated Reranking V1

Status: `COMPLETE`; isolated development-only experiment on the already opened
eight-Qwen ProcessBench population; no promotion or fresh-confirmation claim.

## Question

Do the individually positive Phase-2C edits combine constructively when the
frozen H0 detector retains exclusive authority over clean/error decisions, and
does a donor-only reliability weight improve on equal H2/C8 reranking?

## Frozen arms

- `H0_FAMILY6_TOP10`: exact five-family/top-ten Phase-2C parent, including its
  grouped cross-fitted detector and threshold.
- `H2_CLEAN_C7`: H0 localizer minus the complete sampled-token-energy family,
  minus `energy_series` inside partition energy, plus frozen C7 inside
  `entropy_dynamics`.
- `H3_EQUAL`: equal step-rank fusion of H2 and frozen C8.
- `H3_RELIABILITY`: H2/C8 rank fusion with
  `alpha_C8 = R_C8 / (R_H2 + R_C8)`. Reliability is median rank stability
  under twelve circular within-step moving-block perturbations, fit in five
  label-free folds without an alpha grid.

All candidates rerank only traces that H0 already declares erroneous. H0's
abstentions are copied exactly, so clean abstention must be identical by
construction. Scores and reliability weights froze before labels opened.

## Results

H0 scores macro F1 `0.354261`, exact error `0.267576`, and clean abstention
`0.570739`.

H2 reaches F1 `0.364090`, delta `+0.009829` with four-contrast Bonferroni
simultaneous interval `[-0.000714,+0.020697]`. Exact error improves by
`+0.011340 [+0.001065,+0.021914]`; clean abstention is unchanged.

H3 equal is raw best: F1 `0.366653`, delta `+0.012392
[+0.001769,+0.022807]`, 6/8 cell wins and worst-cell delta `-0.002764`.
Exact error improves by `+0.012941 [+0.003093,+0.022888]`; within-one improves
in all eight cells; clean abstention is exactly unchanged.

H3 reliability reaches F1 `0.364369`, delta `+0.010108
[-0.000542,+0.021037]`. Against H3 equal it changes F1 by `-0.002285
[-0.007659,+0.003087]`. The learned weights are nearly equal
(`alpha_C8` range `0.4813–0.5000`, median `0.4968`), so the reliability proxy
does not add evidence beyond the simpler 50/50 rule.

## Decision boundary

H3 equal has a directional simultaneous interval above zero, but its lower
bound `+0.001769` does not exceed the existing `+0.003` practical-benefit
boundary. Moreover, the four changes were selected after their individual
results were opened. The result is therefore `DEVELOPMENT / NO_PROMOTION` and
requires fresh questions under a frozen H3-equal contract.

The raw `0.366653` is numerically close to and slightly above the historical
`~0.3662` anchor, but the populations, split and detector contracts differ. It
must not be presented as a paired or direct improvement over the historical
Stage-4 score.

Source report SHA-256:
`5e8abfcdf31e1e85ee7409f0a1e1a8800ed5e9025ef31ee3a6a5b7b5ba70ee85`.

