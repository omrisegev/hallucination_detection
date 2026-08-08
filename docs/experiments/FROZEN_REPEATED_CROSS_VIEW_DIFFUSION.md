# Frozen repeated cross-view diffusion experiment

**Date:** 2026-08-07
**Status:** frozen before real correctness labels are opened
**Working name:** RCV-AD-IU-PCR, Repeated Cross-View Alternating-Diffusion IU-PCR

## Question

The family-relevance diagnostic found that the useful family changes across
IU-PCR score regimes, but within-family agreement and a semantic family graph
did not identify the useful family. This experiment removes family reliability
anchors completely and asks a different question:

> Is there a sample manifold that is reproducible across complementary,
> partly independent feature subsets, and does smoothing IU-PCR on that common
> manifold improve its correctness ranking?

The method does not choose a reliable family and does not use a pseudo-label.
It keeps only sample relations that survive a two-view diffusion operation.

## Alternating-diffusion graph

For one partition of the feature indices into disjoint nonempty sets `A` and
`B`, build self-tuning k-nearest-neighbour affinities `W_A` and `W_B`. Let

\[
P_A=D_A^{-1}W_A,\qquad P_B=D_B^{-1}W_B.
\]

The two-view operator is

\[
S_{A,B}=\frac12(P_AP_B+P_BP_A).
\]

It is symmetrized, its diagonal is removed, and each row is reduced to its
largest `k` positive entries before final symmetric union. For `T` frozen
partitions, the consensus graph is the top-k symmetric reduction of

\[
\bar S_T=\frac1T\sum_{t=1}^{T}S_{A_t,B_t}.
\]

The graph is inserted at the same point as DUFS-LIU. With ordinary IU-PCR
score `s=F^T w` and normalized graph Laplacian `L_T`, the final two-dimensional
solve adds

\[
\lambda s^T L_Ts/n
\]

to the projected IU-PCR system. `lambda=0` must return ordinary IU-PCR exactly.

## Partition schemas

Every schema is label-free and feature-order invariant.

1. **Atomic random:** every feature is its own block. Balanced random block
   assignments are a permissive baseline and can leak near-duplicate features
   across the two views.
2. **Dependency blocked:** rank each feature over samples, compute absolute
   Spearman correlations, and use complete-linkage clustering at distance
   `0.15`. Therefore every pair in a dependency block has absolute Spearman
   correlation at least approximately `0.85`. A complete block must stay on
   one side of every partition. This is the registered primary schema.
3. **Family blocked:** the six frozen provenance families are indivisible
   blocks. This is a stronger independence control, not the primary method.

Only partitions in which both sides contain at least 30% of the features are
allowed. Complementary assignments are treated as the same partition because
the operator is symmetric in `A,B`. The primary and atomic schemas use the 16
most balanced deterministic assignments. Some feature pools have fewer than
16 feasible family-blocked assignments; all feasible assignments are then
used and the exact count is reported.

## Frozen parameters

- feature contract: `fixed_stable_v1`;
- partition count: `T=16`;
- convergence prefixes: `T={4,8,16}` for the dependency-blocked schema;
- dependency complete-linkage distance: `0.15`;
- minimum feature fraction per view: `0.30`;
- graph neighbours: primary `k=7`, sensitivity `k={5,11}` for the primary
  schema;
- Laplacian path: `lambda={0,0.03,0.1,0.3,1,3}`;
- primary: dependency-blocked, `T=16`, `k=7`, `lambda=0.1`.

The primary lambda is inherited from the frozen DUFS-LIU baseline. It is not
selected from the current real labels.

## Controls

- deployed U-PCR and ordinary two-component IU-PCR;
- frozen DUFS-LIU at `lambda=0.1`;
- frozen full-feature uniform-graph LIU at `lambda=0.1`;
- atomic-random alternating diffusion;
- family-blocked alternating diffusion;
- dependency consensus with `T=4` and `T=8`;
- dependency consensus using direct arithmetic view averaging instead of
  alternating products;
- a node-permuted version of the primary consensus graph;
- dependency primary with `k=5` and `k=11`;
- the complete registered lambda path for all three partition schemas.

## Label-free diagnostics

For each cell and schema, save:

- exact blocks and exact feature membership of every partition;
- number of unique feasible partitions and side-size ranges;
- consensus graph connectivity and degree diagnostics;
- median centered-kernel alignment between a partition graph and the final
  consensus graph;
- median edge Jaccard between a partition graph and the consensus graph;
- median Spearman agreement between the partition-specific LIU score and the
  consensus LIU score at `lambda=0.1`;
- mean absolute rank change from IU-PCR;
- score agreement between `T=4`, `T=8`, and `T=16`;
- projected roughness, condition number, and weight cosine.

Consistency is a mechanism diagnostic, not evidence of correctness. Stable
nuisance is an explicit failure interpretation.

## Frozen fit and evaluation seam

The fit program receives a physically stripped bundle containing only feature
matrices, feature names, and fixed orientations. It saves every score,
partition, graph diagnostic, input hash, source hash, and reference-score hash.
The report must verify all hashes and create the score-freeze manifest before
it opens `__labels` from the original bundle.

The 24 cells are retrospective development evidence. No method may be selected
from their AUROC and then described as pre-registered.

## Continuation gates

The repeated cross-view direction continues only if the registered primary:

1. improves IU-PCR by at least `+0.20pp` cell-macro;
2. has an equal-dataset-family bootstrap lower bound above zero;
3. improves at least 14 of 24 cells;
4. has no cell worse than `-2pp` against IU-PCR;
5. beats atomic-random splitting;
6. beats family-blocked splitting;
7. beats direct arithmetic view averaging;
8. beats the node-permuted graph;
9. beats frozen DUFS-LIU;
10. has median partition-to-consensus graph CKA at least `0.50`;
11. has median `T=8` versus `T=16` output-score Spearman at least `0.95`.

Failure of gates 5--8 means the alternating-diffusion or dependency-blocking
mechanism was not demonstrated, even if regularization happens to tie IU-PCR.
Failure of gates 10--11 means the repeated construction did not converge.

## Interpretation boundaries

- If atomic random beats dependency blocking, duplicate leakage is a likely
  explanation and the random result is not promoted.
- If dependency blocking works but family blocking fails, the common manifold
  requires mixed provenance in each view.
- If all schemas are stable but do not improve AUROC, the experiment has found
  reproducible geometry rather than correctness-relevant geometry.
- If family blocking alone works, it remains a control result until reproduced
  on new cells; the current family partition was already used in earlier work.
- Alternating diffusion can suppress a signal that exists in only one view.
  A negative result therefore rejects the common-manifold route, not the
  earlier observation of conditional family specialization.
