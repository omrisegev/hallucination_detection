# Automatic group-free IU — Phase A1 factorial measurement model

- Version: `automatic-group-free-iu-a1-v1-2026-08-13`
- New correctness labels accessed: **no** (the frozen mixed-v2 input contract
  inherits earlier label-informed transforms and signs)
- Structural train / untouched structural audit environments: **16 / 7**
- Feature roster: **30**, with NaN-preserving incomplete coverage
- Primary basis: **hybrid soft factorial**, selected only by training-cell LOEO reconstruction
- Frozen configuration: `{"alpha": 0.25, "basis_kind": "hybrid", "interaction": true, "random_seed": 0, "rank": 6, "ridge": 0.1}`
- Audit equal-environment RMSE — hybrid / pooled PCA / hard factorial / pooled mean: **0.178911 / 0.186290 / 0.284529 / 0.213495**
- Paired MSE delta vs pooled PCA, grouped 95% CI: **-0.00269494 [-0.00584499, 0.000282023]**
- Paired MSE delta vs median random partition, grouped 95% CI: **-0.122288 [-0.13763, -0.109478]**
- Random-partition fifth-percentile MSE: **0.145219**
- Leave-one-training-environment projector overlap: min **0.942768**, median **0.992953**
- Exact duplicate mass error: **0**
- Near-duplicate combined/original mass ratio at rho=0.999: **3.008822**
- Feature-order permutation / repeatability max error: **4.83e-15 / 0**
- Simulator candidate / pooled MSE: **0.0118385 / 0.0224721**

## Decision

**CLOSE_AS_DETECTOR_BASIS**. The route passes only if the predeclared hybrid representation
beats pooled PCA and cardinality-matched random partitions under equal-environment
weighting on the hash-held-out
environments, remains stable under environment deletion, conserves exact-
duplicate mass, controls a near duplicate, and beats the pooled simulator
baseline. No detector AUROC, correctness target, Family-NRM direction, or
supervised atomic direction participated in selection or in this decision.

Regardless of this A1 decision, Phase A2 proceeds on the raw atomic residual
covariances. A1 may be used inside A3 only when the result above is PASS.
