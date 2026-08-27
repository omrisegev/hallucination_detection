# SU-aware pooled graph-roughness sidecar v1

**Status:** retrospective, isolated sidecar; this does not replace any frozen
IU-PCR, DUFS-LIU, Family-NRM, or pooled-roughness result.

## Question

Can the sparse-error decomposition used by SU-PCR improve the successful
family-contribution pooled graph-roughness mechanism when SU is used as a
dependency cleaner rather than transplanted wholesale as the reliability
estimator?

## Population and split

Use the same eligible original mixed-v2 cells and dataset-family grouping as
the Family-NRM and pooled-roughness development analyses.  Evaluation is
nested leave-dataset-family-out.  Every graph, covariance decomposition,
family contribution transform, and pooled direction is label-free.  Labels
are opened only for inner hyperparameter selection and outer evaluation.

## Upstream factorial

For every cell let `C` be the observed feature covariance and let `S` be the
sparse component returned by the existing SU-PCR reproduction.  Define:

1. `observed`: `C_alpha = C`;
2. `all_sparse`: `C_alpha = PSD(C - alpha S)`;
3. `cross_sparse`: retain the estimated within-family dependence and remove
   only entries of `S` that connect different provenance families;
4. `shared_cross`: replace the cell-specific sparse matrix by the
   equal-dataset-family mean cross-family sparse matrix, aligned by feature
   name, then use `PSD(C - alpha S_shared)`.

The cleaning grid is `alpha in {0.25, 0.5, 1.0}`.  Each covariance is crossed
with `rho in {IU, SU}`.  PCR always uses the leading two eigenvectors of the
covariance supplied to the solve.  Thus the two observed arms reproduce IU-PCR
and SU-PCR respectively; no higher-rank SU interpretation is introduced.

The prespecified primary adaptation is `IU rho + cross_sparse`.  The SU-rho,
all-sparse, and shared-cross arms are mechanism controls.

## Family residuals and graph operators

For each upstream weight vector, group score contributions by the frozen six
provenance families, standardize them, and residualize each family against its
own upstream score.  Build duplicate-safe family-residual graphs for:

- union kNN with `k in {5, 7, 15}`;
- adaptive kNN with mean `k=7`.

For cell `e`, save only the label-free moments

`A_e = R_e' L_e R_e / n_e` and `c_e = R_e' L_e b_e / n_e`,

after the existing trace normalization.  The pooled direction is

`d = -lambda (I + lambda A_bar)^(-1) c_bar`.

Calibration uses `lambda in {0.1, 0.3, 1, 3, 10}` and correction-SD trust in
`{0.25/G, 0.5/G, 1/G}`.

## Pooling mechanisms

`equal_group_mean` averages cells inside each dataset family, then dataset
families equally.  `group_geomedian` computes a geometric-median influence
weight over the joint standardized `(A_g, c_g)` operator vector and applies
those scalar weights to the raw group moments.  Because this remains a convex
combination of PSD roughness matrices, it cannot create an indefinite pooled
quadratic solely through robustification.

The robust control is applied to the observed IU arm and to the primary
cross-family-cleaned IU arm.  It tests outlying-environment suppression; it is
not claimed to reproduce the elementwise sparse theorem of SU-PCR.

## Nested comparisons

For every arm and outer family, the inner folds select the full
`(alpha, graph, lambda, trust)` tuple.  Report three outer-fold quantities:

1. selected upstream score minus canonical IU-PCR;
2. graph score minus the same selected upstream score;
3. graph score minus canonical IU-PCR.

Also select the no-graph alpha independently inside the same inner folds.  A
clean-space claim requires the no-graph control to improve.  A graph-mechanism
claim requires a positive graph increment beyond the matched selected upstream
score.  Hyperparameters may not be chosen from the outer fold.

## Interpretation gates

- The current `IU rho + observed C + equal-group pooled graph` arm must first
  reproduce the ongoing development result to numerical tolerance; otherwise
  stop and diagnose protocol mismatch.
- SU contributes through dependency cleaning only if the primary
  `IU + cross_sparse` arm improves over the reproduced observed-IU graph arm,
  not merely over its own weaker upstream score.
- SU-rho and all-sparse arms are controls, not candidates to rescue post hoc.
- Family-bootstrap intervals are descriptive with only eight development
  families.  ProcessBench/SemGrad are retrospective transfer diagnostics.
  PRMBench/HLE or a fresh family is required for confirmation after a method is
  frozen.
