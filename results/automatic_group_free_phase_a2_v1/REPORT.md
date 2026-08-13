# Automatic group-free IU — Phase A2 multi-environment JBD

- Version: `automatic-group-free-iu-a2-v1-2026-08-13`
- New correctness labels accessed: **no** (the frozen mixed-v2 input contract
  inherits earlier label-informed transforms and signs)
- Evaluation: **nested 5-fold environment covariance reconstruction**, 23 environments
- Primary missing-aware atomic roster: **30 features**; completion
  is refit within training folds and evaluation uses only genuinely observed pairs
- Missing-aware JBD / block-capacity-matched PCA / enclosing full-block PCA
  environment-macro MSE: **0.028700 / 0.032864 / 0.040727**
- Missing-aware JBD minus block-capacity-matched PCA delta, environment-grouped
  95% CI: **-0.00416414 [-0.0121637, 0.000837728]** (gate fails)
- Missing-aware final block sizes: **[3, 19, 1, 1, 1, 1, 1, 1, 1, 1]**
- Missing-aware outer-fold block sizes: **[[16, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [18, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [15, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 1, 19, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]**
- Missing-aware LOEO minimum mechanism-rank ratio: **0.618182**
- Complete-core diagnostic roster: **17 features**
- Complete-core diagnostic JBD configuration: ridge **0.1**, coupling quantile **0.95**
- Complete-core diagnostic recovered block sizes: **[6, 1, 1, 1, 1, 3, 1, 1, 1, 1]**
- Nested MSE — JBD / block-capacity-matched PCA / enclosing full-block PCA / diagonal PCA / AJD / RJD / factorial-JBD / pooled mean: **0.030779 / 0.032399 / 0.045670 / 0.043133 / 0.034320 / 0.038696 / 0.036938 / 0.045670**
- Paired JBD-block-capacity-matched-PCA MSE delta, environment-grouped 95% CI: **-0.0016193 [-0.00536851, 0.00180053]**
- Paired JBD-enclosing-full-block-PCA MSE delta, environment-grouped 95% CI: **-0.0148902 [-0.0210343, -0.0102296]**
- Paired JBD-diagonal-PCA MSE delta (unmatched diagnostic): **-0.0123536 [-0.0189281, -0.00779031]**
- Paired JBD-AJD MSE delta, environment-grouped 95% CI: **-0.00354083 [-0.00758515, -9.15554e-05]**
- Leave-one-environment mechanism overlap: min **0.766826**, median **0.900207**
- Environment-shuffle JBD minus block-capacity-matched PCA delta: **0.000143033**
- Feature permutation / repeatability max error: **3.61e-16 / 0**
- Simulator JBD / block-capacity-matched PCA MSE: **0.0667505 / 0.073461**

## Decision

**CLOSE_MISSING_AWARE_JBD_AS_TARGET_BASIS**. The missing-aware 30-atom result is the primary scope; the
17-atom universally complete run is a diagnostic. A2 passes only if nested
reconstruction beats a pooled-PCA basis with the same recovered block sizes,
mechanism count, and ridge. The full-block PCA row is an enclosing-space
diagnostic, not the capacity-matched gate. The primary matched comparison is
promising but uncertain and is not robustly positive across environments;
the covariance-mechanism span must be stable and nontrivially block identified,
the advantage must collapse under the registered PSD null, the known-block
simulator must be recovered, and exact invariance gates must pass. Here the
matched reconstruction interval crosses zero and the LOEO mechanism-rank ratio
is 0.618, below the frozen 0.70 threshold; both independently close the route.
This phase identifies covariance mechanisms only; it does not claim that any
block is hallucination-related.

The exact/near-duplicate detector-score, affine score reconstruction, and
zero-evidence IU-PCR fallback gates were not run because A2 failed before a
detector, orientation, or trust rule existed. They remain mandatory if this
representation is ever reopened for promotion; their absence cannot be used
as evidence for a positive result.

A3 is closed regardless of this result because A1 failed its own duplicate
robustness premise. The next independent route is A4: use the exact 3,400-item
cross-model ProcessBench surface to ask whether a recovered mechanism tracks a
response-invariant target component rather than scorer-specific nuisance.
