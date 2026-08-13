# Coupled-Moment Latent-Factor Fusion: methods

## Claim boundary

This is an exploratory adaptation, not a method published in Ibrahim et al.
(2025), and not a theorem-backed extension of continuous U-PCR. The tensor
identifiability results in that survey concern categorical Dawid--Skene
confusion matrices. Our continuous feature model has weaker identifiability.

## Input

Every cell uses the frozen confidence-oriented mixed-v2 contract. The union has
30 features, but a cell keeps only the 19--30 features actually available. No
missing feature is imputed. Let X be samples by features.

## Model and fitting

Covariance selects its leading 6-dimensional subspace Q. We
then fit a symmetric CP model using only third central moments whose three
original feature indexes are different:

    E[X_i X_j X_k] approximately equals
        sum_l kappa_l b_i,l b_j,l b_k,l, for i < j < k.

Repeated-index entries are excluded because feature-specific marginal skew can
create false shared components. The projected tensor is used only to initialize
the masked optimization.

For total rank r, nuisance k=r-1. The target candidate is the component whose
loading has the largest absolute cosine with IU-PCR's label-free rho estimate;
its sign is oriented toward rho. Other components are reconstructed and removed
from X, after which ordinary full-pool two-component IU-PCR is fitted again.
DUFS-LIU is optionally fitted on the same cleaned X with its frozen settings.

## Label-free rank choice and fallbacks

Ranks 1--5 are compared on four deterministic half splits. K=10 repeated
generations stay in the same half. Rank uses held-out third-moment reconstruction
and the smallest-rank one-standard-error rule. A rank above one must also pass:

- all-distinct K3 split stability >= 0.75;
- target-loading stability >= 0.75;
- target-alignment margin >= 0.05;
- within-5%-of-best frequency >= 0.70;
- full-fit seed agreement >= 0.80;
- full-fit target alignment >= 0.20;
- loading Gram condition number <= 100;
- convergence of every split fit and the selected full-data fit.

Failure returns the exact IU-PCR/DUFS-LIU input, not a tuned alternative.

## Comparators and controls

The same mixed-v2 rows are scored by deployed-style U-PCR, IU-PCR, the local
SU-PCR reproduction, SDSF, and frozen DUFS-LIU. A PCA nuisance-deflation arm
tests whether third moments matter beyond covariance. A feature-wise permutation
arm preserves each feature marginal but destroys covariance and higher-order
cross-feature dependence. It is a broad dependency-destruction control, not a
pure third-moment ablation.

Labels are structurally absent from fitting. Scores and diagnostics are hashed
before labels are loaded. Because mixed-v2 and these 24 cells were used during
development, results are retrospective evidence only.
