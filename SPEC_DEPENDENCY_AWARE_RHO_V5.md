# Dependency-aware U-PCR reliability cycle v5

## Question left open by v4

The fixed-stable solver study rejected the full inverse-covariance SDSF tail,
but retained SU-PCR's two-component final solver.  The remaining dependency
question is therefore upstream: can the reliability vector `rho` be estimated
more accurately while leaving the successful PCR truncation unchanged?

U-PCR recovers `rho` from the overdetermined pair equations

`L_ij = rho_i + rho_j - g2`.

The current implementation applies ordinary least squares to these equations.
That is statistically inefficient when pair moments have unequal variances, and
it ignores their dependence: equations `(i,j)` and `(i,k)` are computed from
the same observations and share feature `i`.  Sparse low-rank decomposition and
this sampling-error problem are different.  The former removes systematic
correlated feature errors; the latter concerns uncertainty in the cleaned pair
moments.

## Hypothesis

A generalized method-of-moments (GMM) solve using a regularized estimate of the
pair-moment covariance will reduce planted `rho` error and improve, or at least
not harm, held-out AUROC relative to ordinary-least-squares SU-PCR.  The final
weight rule remains exactly two-component PCR.

## Candidate registry (fixed before scoring)

All candidates use the same rank-two-plus-sparse decomposition, `g2` grid,
projection criterion, and two-component PCR solver.

1. `ols`: current SU-PCR reliability solve; the control.
2. `diag_var`: inverse empirical variances of sample pair products.
3. `diag_mad`: robust diagonal weights from pair-product MADs.
4. `gaussian_gls`: full pair covariance computed from the observed feature
   covariance by the Gaussian fourth-moment identity.
5. `lw_gls`: full empirical pair-product covariance with Ledoit-Wolf shrinkage.
6. `hybrid_gls`: an equal-trace 50/50 blend of the Gaussian structural target
   and Ledoit-Wolf empirical covariance.

Full covariance candidates receive the same condition-number cap of 100.  No
candidate can receive correctness labels.  The fixed hybrid coefficient is an
ablation, not a per-cell fitted hyperparameter.

Hansen's GMM result motivates inverse moment-covariance weighting.  Ledoit and
Wolf motivate shrinkage before a high-dimensional covariance is inverted.  The
Gaussian target follows Isserlis' identity
`Cov(F_i F_j, F_k F_l) = C_ik C_jl + C_il C_jk`.

## Disjoint synthetic design

The planted joint covariance satisfies the U-PCR additive model and optionally
contains sparse dependent-error edges.  Training features are standardized by
training-only statistics.  Test labels are the sign of a held-out continuous
latent response; labels are accessed only after an unsupervised score is fixed.

Worlds:

- clean Gaussian (specificity/no-harm control),
- sparse Gaussian,
- sparse small-sample,
- sparse elliptical heavy tail,
- sparse anisotropic fourth moments with unchanged covariance,
- sparse training contamination (robustness stress).

Development and sealed validation have disjoint deterministic seed namespaces.
Development promotes the three candidates with the best frozen utility.  Only
their sealed labels are opened.  Synthetic primary outcomes are normalized
`rho` RMSE against known truth and held-out AUROC against `ols`.

## Frozen synthetic advancement gates

A non-control candidate advances only if, over the four sparse primary worlds:

- mean relative reduction in `rho` NRMSE is positive and its paired bootstrap
  95% CI lower bound is non-negative;
- mean AUROC delta is positive and its paired 95% CI lower bound is at least
  -0.1 AUROC points;
- clean-world mean AUROC delta is at least -0.5 points;
- primary-world 5th-percentile AUROC delta is at least -2 points.

The RMSE gate establishes the proposed mechanism.  The slightly non-inferiority
AUROC interval acknowledges that PCR deliberately discards most directions in
which a better full `rho` can differ.

## Frozen real-artifact boundary

Only a synthetic-passing winner may be replayed on
`results/dependency_fusion_raw/cells.npz`.  It is frozen before that replay.
The replay is retrospective and can reject the method; success would still
require a genuinely new dataset/model family.  The existing contribution gate
is retained: at least +1 mean AUROC point over SU-PCR, cell-bootstrap lower
bound above zero, QA and math no worse than -0.5 points, and positive
family-macro gain.

## Stop rules

- A failed sealed candidate is not repaired using its validation results.
- A failed real replay is not tuned on the 24 correctness-label vectors.
- This cycle does not reopen full-inverse SDSF, clustered U-PCR, l0-CCA feature
  selection, or direct channel CCA; those mechanisms already have negative
  evidence in the repository.

Executable specification: `scripts/dependency_aware_rho_autoresearch.py`.

## Executed outcome

The full run used 8 development and 16 disjoint sealed repetitions per world.
No candidate passed sealed validation; the decision was
`STOP_SYNTHETIC_HYPOTHESIS_REJECTED`, and the real replay was not run. The
post-hoc, label-free mechanism diagnosis found that Ledoit-Wolf GLS improved
centered rho shape but significantly worsened the two PCR-retained coordinates.
See `DEPENDENCY_AWARE_RHO_V5_CONCLUSION.md` and
`results/dependency_aware_rho_v5/FAILURE_DIAGNOSTIC.md`.
