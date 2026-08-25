# B3 family-dependency structure audit

Date: 2026-08-25
Scope: frozen 24-cell, 48,607-row target-free inventory
Status: descriptive structural audit; not a model-selection result

## Question

Before adding another router or residual correction, determine whether the six
provenance families behave like statistical independence blocks, whether their
dependence is approximately low-rank, what remains after explicit controls, and
whether B3 itself removes that dependence.

## Protocol and firewall

`scripts/diagnose_family_dependency_structure_v1.py fit` loaded only the 24
target-free bundles and the frozen five-seed B3 artifacts. It serialized and
hashed every row-aligned state and every label-free table. The separate
`evaluate` phase verified the script, registry, tables, states, bundle hashes,
and ordered row IDs before dynamically importing the label-sidecar module.
Labels are used only for a retrospective within-class dependence diagnostic.

The audit uses equal-cell and equal-dataset-family summaries. Group-aware
five-fold Ridge fits keep sibling generations in the same fold. Pairwise
partial dependence is computed correctly by excluding both tested families,
fitting both held residuals from all remaining families plus cubic log length,
and correlating only the OOF residuals.

## Main results

All values below are equal-dataset-family means unless noted otherwise.

| Diagnostic | Raw family means | B3 family contributions | Interpretation |
|---|---:|---:|---|
| Mean absolute family correlation | 0.5831 | 0.5925 | Strong cross-family dependence |
| After cubic log-length removal | 0.3678 | 0.3763 | Length explains a material part, not all |
| Within-class mean residual correlation | 0.5463 | 0.5563 | Most dependence is not class-mean separation |
| Within-class, class-specific length residual | 0.3589 | 0.3677 | Dependence remains after both controls |
| Top component variance fraction | 0.6587 | 0.6665 | Strong shared axis |
| Top two variance fraction | 0.8359 | 0.8424 | Descriptively low-rank before controls |
| Rank-2 off-diagonal mean residual | 0.0499 | 0.0483 | Rank 2 reconstructs the unadjusted family matrix well |
| Same-family raw-to-B3 Spearman | — | 0.9992 | B3 mostly preserves family ordering |

At the atomic-feature level, within-family dependence is only moderately larger
than between-family dependence: absolute Pearson 0.3817 versus 0.3378 and
absolute Spearman 0.4095 versus 0.3636. The provenance partition is therefore
not an independence partition.

## Duplicate quotient changes the low-rank conclusion

`entropy_level` and `topk_distribution` are almost the same statistical axis:
their raw family correlation is about 0.990 and remains about 0.985 after the
length control. This is consistent with their extraction from overlapping
token-entropy/log-probability information.

After merging those axes and removing length:

| Diagnostic | Raw quotient | B3 quotient |
|---|---:|---:|
| Top component variance fraction | 0.4803 | 0.4889 |
| Top two variance fraction | 0.6879 | 0.6962 |
| Rank-2 off-diagonal residual | 0.1019 | 0.0973 |
| Fraction of residual edges above 0.10 | 0.4967 | 0.5033 |

Thus a pure rank-1/rank-2 model is a useful backbone but is not a complete model
of the controlled dependence. The initially impressive rank-2 fit was partly
driven by length and a near-duplicate cross-family axis.

## Correct pair-excluded partial dependence

Across all family pairs, cross-fitted pair-excluded mean absolute partial
correlation is 0.1854 for raw family means and 0.1873 for B3 contributions.
After excluding the entropy-level/top-k duplicate pair, it falls to 0.1263 and
0.1282; medians are 0.0877 and 0.0896.

The residual is therefore modest overall but not zero. Several edges recur with
stable signs, notably entropy-dynamics/partition-energy,
entropy-dynamics/sampled-token-energy, sampled-token/top-k, and
sampled-token/partition-energy. These are candidates for a sparse cross-block
term. At the descriptive threshold `|partial correlation| > 0.10`, the
leave-one-dataset-family-out raw support contains six or seven edges; pairwise
support Jaccard is 0.964 on average and no lower than 0.857. The B3-contribution
support is numerically identical in size and stability. This passes a first
stability screen, but no support may be frozen until it also wins a
density-matched, length-stratified permutation test; `0.10` is a descriptive
threshold, not a selected hyperparameter.

## Stability

Among the 21 complete six-family cells, after length adjustment:

- mean absolute cosine of the leading raw-family axis: 0.9339;
- mean rank-2 projector similarity: 0.8596;
- mean family-edge-pattern correlation: 0.8439;
- mean edge-pattern correlation after averaging within dataset family: 0.9172.

B3 values are nearly identical (0.9351, 0.8705, 0.8453, and 0.9212). This is
evidence for stable dependence geometry, but not evidence that the geometry is
the hallucination target.

## Independent audit and corrected errors

An independent subagent verified the freeze hashes and independently reproduced
the conditional summaries. It also caught three invalid interpretations in an
earlier draft:

1. conditioning family contributions on their deterministic B3 sum creates a
   collider and is not “dependence left after B3”;
2. correlating two one-vs-rest residuals when each regression contains the other
   tested family is not a partial correlation;
3. class-mean removal followed by one global length slope is not joint
   conditioning when length effects differ by class.

The final script removes the first two diagnostics, uses pair-excluded OOF
residuals, and fits the length basis separately inside each class for the
label-sidecar audit.

## Model implications, without selecting a model yet

The audit rules out three simplistic stories:

- the current provenance families are independent blocks;
- B3 already disentangles their dependence;
- all between-family dependence is captured by one or two clean factors.

The structural model worth testing next is hierarchical:

1. a shared low-rank factor as a common backbone;
2. dense/shrunk covariance inside learned or quotient-corrected blocks;
3. a sparse set of cross-block residual edges;
4. only if a nonlinear permutation audit passes, sparse nonlinear pair terms.

This structure must not be used for hard factor deletion or direct inverse-
covariance weighting: both have already harmed performance in prior experiments.
The common dependence component is nuisance unless its contrast between the two
latent states is identified. Any B3 extension must therefore parameterize a
state-dependent dependence contrast and nest frozen B3 exactly at zero contrast.

## Remaining gates before implementation choice

- held-group rank-1/rank-2 cross-block likelihood versus block-diagonal and
  rank-matched shuffled controls;
- stable sparse residual support under leave-one-cell and
  leave-one-dataset-family-out;
- multivariate block dependence (cross-validated RV/CCA), because family means
  can hide cancellation;
- length-stratified group permutation for nonlinear residual dependence;
- equal-n subsampling because cell sizes range widely;
- explicit comparison of covariance sparsity versus precision sparsity.

Until those gates run, `block-IU`, `block-SU`, and a sparse nonlinear
cross-family layer remain hypotheses rather than selected implementations.
