# Dependency-aware rho cycle v5 — conclusion and next hypothesis

## Decision

**Stop this candidate family.** None of the three development-promoted
covariance-weighted pair solvers passed sealed synthetic validation. Under the
preregistered stop rule, the 24-cell real correctness labels were not replayed.

This rejects the implemented claim, not the premise that pair-equation errors
are dependent. The dependence is directly measurable. What failed was using
the covariance of raw pair products as the weighting covariance for pair values
that have already passed through a nonlinear rank-two-plus-sparse estimator.

## What the experiment established

On the four sparse primary worlds:

| method | full-rho error reduction | held-out AUROC delta | decision |
|---|---:|---:|:---:|
| robust diagonal MAD | -0.35%, CI [-2.51, +1.48] | -0.000 pp | fail |
| diagonal variance | -2.47%, CI [-6.22, +0.68] | +0.000 pp | fail |
| Ledoit-Wolf full GLS | -6.69%, CI [-18.60, +2.63] | +0.000 pp | fail |

Negative error reduction means worse than OLS SU-PCR. The AUROC changes are
effectively zero because two-component PCR filters almost every change made by
the alternative reliability solves.

The label-free failure diagnosis is more informative:

- Ledoit-Wolf GLS improves the centered shape of `rho` by **3.12%**, 95% CI
  **[+0.94, +5.43]**.
- It worsens error in the two PCR-retained reliability coordinates by
  **31.49%**, CI **[-71.63, -1.42]**.
- It worsens the corresponding two-component PCR weight error by **16.85%**,
  with a wide CI **[-40.84, +0.97]**.

Thus the covariance correction is not merely too weak. It improves the part of
the reliability vector that PCR mostly discards while damaging the common/
leading component that controls the deployed score.

## The missed earlier conclusion

This failure reconnects directly to `results/upcr_study/01_g2_criterion`:

- `g2` changes mean AUROC by as much as 9.11 points across its path;
- the existing projection criterion leaves 1.21 mean points against the
  label-peeking oracle `q`;
- its deployed post-exclusion estimate was pinned to the grid ceiling in 24/25
  historical cells;
- widening or replacing only the variance range did not improve performance.

The pair system identifies the *relative shape* of `rho`; its common shift is
the unresolved `g2` direction. V5 improved relative shape but kept the same
Euclidean `g2` criterion. The result therefore exposes the old identifiability
bottleneck rather than solving it. Repeating more covariance estimators while
leaving this step unchanged is unlikely to help.

## Why the GMM argument did not transfer automatically

Inverse moment-covariance weighting is efficient when the moment equations are
correctly specified and the weighting matrix describes the error of the moments
actually supplied to the solve. Here the supplied vector is not the empirical
off-diagonal covariance. It is the output of sparse support detection and
rank-two completion. That transformation introduces bias and changes the error
covariance. The synthetic bias/variance decomposition confirms the tradeoff:
full GLS sometimes reduces variance, but sparse cleaning and non-Gaussian worlds
introduce enough bias to lose overall.

This is consistent with:

- Hansen's generalized method of moments:
  https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/
- Ledoit and Wolf's need to regularize high-dimensional covariance inversions:
  https://doi.org/10.1016/S0047-259X(03)00096-4
- Tenzer et al.'s rank-two-plus-sparse U-PCR construction:
  https://proceedings.mlr.press/v151/tenzer22a.html

## Recommended next cycle (do not run from these results alone)

The next defensible hypothesis is narrower:

> Estimate uncertainty *after* sparse low-rank recovery, and use it only to
> identify/stabilize the leading PCR coordinates and `g2`, not to optimize the
> discarded full reliability vector.

A clean experiment would compare:

1. current Euclidean `g2` projection;
2. row-bootstrap covariance of the recovered rank-two pair vector, followed by
   a generalized projection criterion in the two-dimensional PCR head;
3. cross-fitted `g2` chosen solely by agreement of head scores across row
   halves, with no correctness labels;
4. an OLS-shape/bootstrapped-head hybrid that preserves the successful OLS
   solution unless head uncertainty clears a fixed threshold.

The key falsification tests are:

- Does bootstrap covariance of the *recovered* pair vector predict its actual
  across-repetition covariance better than raw pair-product covariance?
- Does a candidate reduce planted head-coordinate error, not merely full-rho
  error?
- Does it avoid ceiling collapse and remain stable under sparse support changes?
- Does synthetic head-error improvement translate to held-out AUROC before any
  real replay?

If these fail, dependency-aware reliability estimation should be retired for
the current feature family. At that point the evidence says the remaining gains
must come from better information-bearing features or a different identifiable
unsupervised objective, not another solver for the same pair equations.

Evidence files:

- `results/dependency_aware_rho_v5/REPORT.md`
- `results/dependency_aware_rho_v5/summary.json`
- `results/dependency_aware_rho_v5/FAILURE_DIAGNOSTIC.md`
- `results/dependency_aware_rho_v5/failure_diagnostic_bias_variance.csv`

