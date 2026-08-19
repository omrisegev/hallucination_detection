# Cross-dataset hallucination manifold diagnostic v1

**Date frozen:** 2026-08-19

**Status:** retrospective supervised diagnostic on existing local artifacts
**Execution boundary:** CPU-only; no Drive mutation, download, cluster job, or new inference

## Question

Do incorrect/hallucinated generations occupy a feature-space geometry that is
shared across datasets and models, rather than a geometry reconstructed anew in
each evaluation cell?

This experiment uses correctness labels on donor cells to define the target.
It is therefore not a label-free detector and cannot rescue the identifiability
claim of DUFS-LIU. It asks the narrower empirical question that must come first:
does a transferable hallucination geometry appear to exist at all in the
current gray-box feature matrix?

## Population and feature contract

- Source: `results/dependency_fusion_raw/cells.npz`.
- Roster: the registered 24 in-scope cells from `scripts/inscope_cells.py`.
- Target: `1 - correctness_label` (one means hallucination/incorrectness).
- Features: the 16 `fixed_stable_v1` confidence-oriented features present in
  every cell. No missing-feature imputation and no post-result feature choice.
- Each stored cell was standardized using its own unlabeled feature marginal.
  The test is consequently transductive with respect to target-cell centering
  and scaling, but test correctness labels never affect a fitted model.

The common roster is frozen as the sorted intersection across all 24 cells and
must equal:

`cusum_max`, `cusum_max_energy`, `cusum_max_spilled`, `epr`, `epr_energy`,
`epr_spilled`, `logprob_margin`, `mean_logprob_entropy`,
`mean_top1_logprob`, `min_energy`, `renyi_entropy_2`, `sw_var_peak`,
`sw_var_peak_energy`, `sw_var_peak_spilled`, `topk_tail_mass`, `varentropy`.

## Splits

### Primary: leave-one-dataset-family-out

Eight folds hold out every cell containing one of `triviaqa`, `hotpotqa`,
`sciq`, `nq_open`, `squad_v2`, `truthfulqa`, `gsm8k`, or `math500`.
This prevents another model on the same benchmark from teaching the held-out
model its target geometry.

### Secondary: leave-one-cell-out

Twenty-four folds hold out one dataset/model cell. This matches the project's
historical LOCO/LOVO diagnostics but is weaker when another cell uses the same
dataset family.

No row-random split is a reported result.

## Frozen models

1. `epr_risk`: negative confidence-oriented EPR; no labels.
2. `mean_confidence_risk`: negative mean of the 16 oriented features; no labels.
3. `iu_pcr_risk`: negative frozen per-cell IU-PCR score; label-free but
   transductively fitted in the target cell, included only as context.
4. `shared_direction`: average of unit donor-cell vectors
   `mean(error) - mean(correct)`; a supervised shared linear direction.
5. `balanced_logistic`: one shared L2 logistic head. Explicit sample weights
   give every donor cell × class block equal total weight. No target-cell
   intercept or calibration is fitted.
6. `ppca_manifold_k4`: one four-dimensional probabilistic-PCA tube per class.
   Class means and within-class covariances are averaged equally over donor
   cells, the omitted eigenspectrum supplies isotropic residual variance, and
   covariance receives fixed 0.10 identity shrinkage. The score is the error
   log density minus the correct log density.
7. `knn_manifold_k5`: for every donor cell and class, retain at most 64
   deterministic support points. For each held-out row, compute the mean
   five-neighbour distance to each donor cell's correct and error supports;
   average the donor log-distance ratios with equal donor-cell weight.

All settings are fixed here. There is no target-fold hyperparameter search.

## Geometry fingerprints

Two cell-level labelled fingerprints are computed independently of the
predictive models:

1. Mean fingerprint: normalized `mean(error) - mean(correct)`.
2. Shape fingerprint: normalized upper triangle of
   `cov(error) - cov(correct)`.

In each primary fold, donor fingerprints are averaged and compared by cosine
to held-out fingerprints. The primary statistic first averages cells within a
held dataset family and then averages the eight families.

The null independently swaps the meanings of correct/error for each entire
cell, which multiplies both fingerprints by `+1` or `-1` while preserving their
magnitude and internal geometry. Ten thousand fixed-seed sign-flip draws give
a one-sided randomization p-value and a null 95th percentile.

## Metrics and decision language

- Compute AUROC and error-class AUPRC separately in every held-out cell.
- Report equal-cell and equal-dataset-family macros.
- Bootstrap dataset families, never rows, for 95% intervals.
- Never concatenate predictions from different fitted folds before computing a
  ranking metric.

Interpretation is frozen as:

- **Typical nonlinear manifold supported:** a manifold model has primary
  equal-family AUROC at least 0.60, its family-bootstrap lower bound exceeds
  0.50, and its paired advantage over balanced logistic is at least +0.5
  percentage points with a lower bound above zero.
- **Shared direction, not a distinct nonlinear manifold:** the mean fingerprint
  transfers (`p <= 0.05`), balanced logistic exceeds 0.60 with lower bound above
  0.50, but neither manifold model clears the nonlinear advantage condition.
- **Shape regularity only:** the covariance fingerprint transfers but predictive
  manifold models do not clear 0.60. This is descriptive geometry, not a useful
  hallucination manifold.
- **No transferable geometry:** neither fingerprint transfers and no supervised
  model clears the predictive condition.

Any positive conclusion is retrospective development evidence. Promotion
requires an untouched dataset family or model family whose feature extraction,
labels, and inclusion are fixed before fitting.

## Required outputs

- `RUN_DEFINITION.json` with source and bundle hashes;
- `PER_CELL_METRICS.csv` for both split schemes;
- `SUMMARY.csv`, `FEATURE_EFFECTS.csv`, and `GEOMETRY.json`;
- a concise `REPORT.md` and diagnostic figures;
- deterministic rerun and focused test pass.
