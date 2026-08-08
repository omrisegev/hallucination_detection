# Deployed U-PCR

## Terms used below

There are (m) features and (n) samples. (f_i(x_j)) is the value of feature
(i) on sample (j), and (Y) is the unknown correctness target. (C) is the
observable (m\times m) feature covariance matrix. A principal component is an
eigenvector of (C). The response covariance
(ho_i=\operatorname{Cov}(f_i,Y)) is useful for fusion but cannot be measured
without labels.

## Paper basis

The original U-PCR method is from Dror, Nadler, Bilal, and Kluger,
[Unsupervised Ensemble Regression](https://arxiv.org/abs/1703.02965), 2017.
The broader framework and its uncorrelated- and sparse-error versions are given
by Tenzer et al.,
[Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html),
AISTATS 2022.

The repository file named `Tenzer2022_Crowdsourcing_Regression_Spectral.pdf`
contains the 2017 paper, not the 2022 AISTATS paper. Citations must use the links
above rather than infer the source from that filename.

## Mathematical model

After centering, each expert prediction is modeled as

\[
f_i(x)=g(x)+h_i(x),
\]

where (g(x)=\mathbb E[Y\mid x]) is the best regression function and (h_i)
is feature (i)'s deviation from it. The central assumption is pairwise
uncorrelated deviations:

\[
\mathbb E[h_i h_j]=0 \quad\text{for }i\ne j.
\]

This is weaker than full statistical independence. Under the model, the
off-diagonal covariance entries have an additive form. With the paper's scalar
signal term written as (g^2),

\[
C_{ij}=\rho_i+\rho_j-g^2,\qquad i\ne j. \tag{1}
\]

For a candidate (q) for (g^2), Equation (1) is an overdetermined linear
system in (ho). The implementation solves

\[
\hat\rho(q)=\arg\min_r\sum_{i<j}
\left(C_{ij}+q-r_i-r_j\right)^2. \tag{2}
\]

The candidate whose (hat\rho(q)) is closest to the leading eigenspace of
(C) is selected. If (U_k) contains the selected leading eigenvectors, the
PCR weight is

\[
\hat w=U_k(U_k^\top C U_k)^{-1}U_k^\top\hat\rho. \tag{3}
\]

The final sample score is (s=\hat w^\top F). No correctness label enters
Equations (1)--(3).

## Exact project realization

The benchmark calls `spectral_utils.upcr.upcr_fit` on the confidence-oriented,
z-scored `fixed_stable_v1` matrix. The deployed policy is:

- squared loss in Equation (2);
- `scale_ratio=0.25` for the (g^2) search interval;
- one component for the projection criterion used to choose (g^2);
- automatic one- or two-component final PCR weights;
- remove features with small estimated (ho_i);
- recompute the estimate after exclusion;
- use a simple average when too few features survive;
- do not stop on the paper's difficulty gate.

The fixed feature direction, feature quarantine, scale choice, exclusion
thresholds, recomputation, and fallback are project decisions. They are not a
single estimator copied verbatim from either source paper.

## Assumptions and likely failure modes

1. The features are continuous measurements of the same target.
2. Their directions are already aligned. U-PCR cannot identify the global sign
   from unlabeled covariance alone.
3. Pairwise error covariance is small enough for Equation (1) to be useful.
4. There are at least three informative features and enough samples for a
   stable covariance estimate.
5. The response-scale choice is adequate after z-scoring.

It can fail when many features share errors, when a nuisance factor creates the
leading covariance directions, when feature reliability changes by sample, or
when exclusion removes complementary weak features.

## Hyperparameters in the benchmark

| parameter | value | meaning |
|---|---:|---|
| `scale_ratio` | 0.25 | upper response-variance scale relative to mean feature variance |
| loss | L2 | squared residual in the additive covariance system |
| `lambda2_threshold` | 0.1 | use two final PCs when the second eigenvalue exceeds 10% of `trace(C)` |
| `min_frac` | 0.05 | first weak-feature exclusion threshold |
| `exclude_frac` | 3.0 | keep a feature only if its estimate is at least one third of the maximum |
| `g2_projection_k` | 1 | number of PCs used to select (g^2) |

These are frozen. The 24-cell report does not optimize them.

## Checks required before making a performance claim

- report the number and identity of kept features in every cell;
- report simple-average fallbacks and any abstention;
- report the projection residual, (g^2) boundary hits, and covariance spectrum;
- check weight and score stability under sample bootstrap;
- compare the same rows and feature pool with full-pool IU-PCR;
- report AUROC, AUPRC, per-cell changes, worst cases, and uncertainty intervals;
- never orient or flip a score using the evaluation labels.

## Computational cost

Forming (C) costs (O(nm^2)) time and (O(m^2)) memory after the
(O(nm)) input matrix. The additive system has (m(m-1)/2) equations. With
the analytic shift used in `upcr_fit`, it is solved once; eigendecomposition is
(O(m^3)), and a grid of (G) response-scale candidates adds about
(O(Gm^2)). Here (m\le 30), so covariance construction usually dominates.
