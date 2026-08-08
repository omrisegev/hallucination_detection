# Frozen graph-coupled family relevance diagnostic

**Date:** 2026-08-07
**Status:** frozen before the new real-data score fit and evaluation
**Working name:** GCFR-U-PCR, Graph-Coupled Family Relevance U-PCR

## Question

The atomic-operator audit showed that stable static geometry does not identify
correctness-relevant features. This experiment tests a different assumption:

> A feature family can be reliable on some samples and noisy on others. Local
> agreement inside a known family may estimate that relevance, and a small
> family graph may stabilize the estimate.

This is not the previous manual-view experiment. That experiment built one
sample graph per family and inferred family reliability from agreement between
views. Here, the family relation is a prior on a sample-local relevance gate
applied directly to fixed IU-PCR feature weights.

## Model

Let `F` contain oriented features, `w` be the ordinary two-component IU-PCR
weight vector, and `g(j)` be the registered family of feature `j`. For sample
`i`, the model uses a family gate `pi[i,g]`:

\[
\tilde s_i =
\frac{\sum_j w_j F_{ji}\pi_{i,g(j)}}
     {\sum_j |w_j|\pi_{i,g(j)}}\sum_j|w_j|.
\]

The final path is

\[
s_i(\alpha)=(1-\alpha)s_i^{IU}+\alpha\tilde s_i.
\]

Raw family evidence is high when oriented feature ranks inside that family
agree on the sample. Singleton families have no direct agreement evidence. A
registered graph connects:

- entropy level -- entropy dynamics -- structural;
- sampled-token energy -- partition energy -- top-k distribution.

For every sample, graph smoothing solves

\[
\pi_i = \arg\min_\pi
\|M^{1/2}(\pi-b_i)\|_2^2
+\beta\pi^T L_F\pi
+0.1\|\pi-\mathbf 1\|_2^2.
\]

`M` marks families with at least two observed features. The result is clipped
for safety and normalized to mean one, so total score scale cannot create a
ranking gain.

## Experiment series

### A. Synthetic mechanism and failure world

Twenty fixed seeds generate two regimes. Related families 0/1 are relevant in
one regime and related families 2/3 in the other.

1. **Independent-noise world:** inactive family members contain independent
   noise. Within-family agreement should identify relevance.
2. **Correlated-nuisance world:** inactive members share the same nuisance.
   They look consistent but do not measure the target. This is the explicit
   falsification world.

The fixed grid is `beta={0,0.3,1,3}` and `alpha={0.25,0.5,1}`. The real-data
primary is selected only from the independent-noise synthetic world by maximum
wins, then mean AUROC change, then worst change. The selected path is
`beta=3, alpha=1`: +0.773pp mean, 20/20 wins, worst +0.556pp. The same path
loses 9.272pp in the correlated-nuisance world. This failure is part of the
claim, not a result to hide.

### B. Frozen real-data score fit

Use the same 24 retrospective cells and `fixed_stable_v1` feature contract as
the frozen view-fusion and atomic audits. The fit program receives a physically
stripped bundle containing only features, names, and fixed orientations. It
saves and hashes all scores, gates, contexts, and diagnostics before evaluation.

Real-data paths:

- correct family graph for every `beta,alpha` pair;
- graph-permuted prior control;
- global gate control;
- sample-permuted local gate control;
- ordinary IU-PCR;
- deployed U-PCR and frozen DUFS-LIU (`lambda=0.1`) references from the earlier
  hash-verified benchmark.

The primary path is fixed at `beta=3, alpha=1` by the synthetic selection.

### C. Conditional-specialization diagnosis

Before labels are opened, save three sample contexts and their quartile bins:

1. trace-length rank;
2. disagreement between family rank centers;
3. ordinary IU-PCR score rank.

After freezing, measure the AUROC of every fixed family expert inside every
valid stratum. A descriptive context oracle chooses the best family separately
in each stratum. Its headroom over one fixed best family is compared with 500
within-cell context permutations. This oracle is an evaluation diagnostic, not
a runnable detector.

The three context tests use Holm correction. A context supports conditional
specialization only if equal-family headroom is at least +0.5pp and adjusted
one-sided permutation `p<=0.05`.

## Continuation gates

All real gates must pass before building a learned mixture of U-PCR experts:

1. the primary GCFR path improves IU-PCR on average;
2. its equal-family bootstrap lower bound is above zero;
3. it improves at least 14 of 24 cells;
4. its worst loss is no worse than -2pp;
5. it beats the `beta=0` no-graph family gate;
6. it beats the permuted-family graph;
7. it beats the global gate;
8. it beats the sample-permuted local gate;
9. it beats frozen DUFS-LIU;
10. at least one preregistered context supports conditional family
    specialization after Holm correction.

Failure of gates 5--8 means the intended family/local mechanism was not shown,
even if the primary happens to tie IU-PCR. Failure of gate 10 means there is no
evidence that these contexts organize family relevance.

## Interpretation boundary

The key assumption is that an irrelevant family becomes internally
inconsistent. The correlated-nuisance synthetic world proves that the method
cannot distinguish a coherent wrong family from a coherent useful family.
Therefore a real gain would support the assumption only for this feature
bundle. A real loss closes within-family consistency as the missing target; it
does not justify tuning `beta`, `alpha`, family edges, or context bins after
reading AUROC.

The 24 cells are reused development evidence, not external confirmation.
