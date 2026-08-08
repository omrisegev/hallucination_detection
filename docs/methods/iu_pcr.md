# IU-PCR: the common full-pool anchor

## Terms used below

An **expert** is one continuous hallucination feature. (C) is the covariance
between experts, (ho_i=\operatorname{Cov}(f_i,Y)) is the unknown covariance
with correctness, and (U) contains principal components of (C). “Full pool”
means that no feature is removed after (ho) is estimated.

## Paper basis

IU-PCR is the uncorrelated-error version of the U-PCR framework in Tenzer et
al., [Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html),
AISTATS 2022. It builds on Dror et al.,
[Unsupervised Ensemble Regression](https://arxiv.org/abs/1703.02965), 2017.

The letter `I` is retained from the paper's method name. The mathematical
assumption is pairwise uncorrelated errors; it does not require full
probabilistic independence.

## Mathematics

The model, additive covariance system, and estimate (hat\rho) are the same as
in [Deployed U-PCR](deployed_upcr.md):

\[
C_{ij}=\rho_i+\rho_j-g^2,\qquad i\ne j. \tag{1}
\]

The benchmark fixes the final PCR space to the leading two eigenvectors
(U\in\mathbb R^{m\times2}) and computes

\[
w_0=U(U^\top C U)^{-1}U^\top\hat\rho. \tag{2}
\]

This exact vector is the (lambda=0) anchor for every Laplacian-IU method.

## Difference from deployed U-PCR

The benchmark's IU-PCR is deliberately simple and identical across all graph
methods:

- all stable features remain in the solve;
- exactly two principal components are used;
- there is no weak-feature exclusion;
- there is no difficulty stop or simple-average fallback;
- the same L2 additive solve and `scale_ratio=0.25` are used.

Therefore a difference between a graph method and IU-PCR can be attributed to
the graph penalty, not to a different feature subset or PCR dimension. The
implementation requires every graph arm at (lambda=0) to reproduce the
stored IU-PCR score bit for bit.

## Assumptions and failure modes

IU-PCR needs the additive off-diagonal covariance model to be approximately
correct and its two-dimensional leading subspace to contain the useful response
direction. It fails when correlated errors are dense or conditional, when the
leading subspace is a nuisance manifold, or when two components are not an
adequate representation of (hat\rho).

## Frozen settings and possible future hyperparameters

| parameter | benchmark value | possible future study |
|---|---:|---|
| number of PCs | 2 | 1, 2, or a label-free spectral rule |
| additive loss | L2 | L1 robust loss |
| `scale_ratio` | 0.25 | label-free scale diagnostics |
| exclusion | off | full-pool anchor must keep it off |

The current 24-cell run does not tune these choices.

## Required checks

- exact equality between IU-PCR and every LIU arm at (lambda=0);
- condition number of (U^\top C U);
- additive-model projection residual and covariance eigengap;
- score variance and orientation failures;
- AUROC and AUPRC by cell, domain, and dataset family;
- paired changes against deployed U-PCR, including the lower tail.

## Computational cost

The cost is the same order as U-PCR: (O(nm^2+m^3+Gm^2)) time and
(O(nm+m^2)) memory. With at most 30 features, it is far cheaper than training
DUFS or SpecRaGE and building their sample graphs.
