# Hierarchical and active spectral correction v2

## Research question

The first semi-supervised spectral experiment established that a six-direction
head fitted independently inside each cell is too variable at 5--80 trusted
labels.  It also established that a two-direction head is safer and that U-PCR
pseudo-labels reinforce, rather than repair, the teacher's errors.

This experiment asks two separate questions:

1. Can a correction learned from trusted labels in *other* dataset families
   transfer into a held-out target family?
2. Given the same number of target labels, does label-free Fisher/D-optimal
   acquisition estimate a two-score correction more efficiently than uniform
   acquisition?

The experiment does not test another pseudo-label method.  That branch is
closed unless an independent teacher becomes available.

## Frozen hypotheses

For every target cell, construct a within-cell standardized representation
whose first coordinate is its unlabeled U-PCR score.  The remaining coordinates
are the 16 stable confidence-oriented features shared by all 24 cells after
their linear U-PCR component is removed.  A hierarchical logistic head has one
shared coefficient vector and a nuisance intercept for every donor cell.

The primary candidate is a two-score target head:

- score 1: target-cell U-PCR;
- score 2: the correction learned from same-domain donor cells after excluding
  **every cell in the target dataset family**.

The target head is centred on the transferred donor solution.  Its trusted
labels are selected without reading their labels by greedily maximizing the
log-determinant of the prior-weighted logistic Fisher information matrix.

Primary hypothesis:

> At 20 target labels, `hybrid_domain_active` improves held-out real-cell AUROC
> over U-PCR, local two-direction fitting, and the identical transferred head
> with uniform target acquisition.

Mechanism hypotheses:

- pooling should help in a synthetic world with a shared correction;
- it must not be promoted from a world whose correction changes by family;
- active acquisition should help only when the transferred correction leaves a
  low-dimensional uncertainty that informative target examples can resolve.

## Leakage and transfer boundary

- Real input: `results/dependency_fusion_raw/cells.npz`, reconstructed under
  `confidence-orientation-v1`.
- The common feature vocabulary is the unlabeled intersection of stable feature
  names over all cells.  Labels cannot affect the vocabulary.
- Each cell is split 60/40 with the exact v1 split namespace so v1 local arms
  can be reproduced and paired.
- Standardization, U-PCR, residualization, covariance directions, donor fits,
  and acquisition use training features only.
- A target's held-out labels are used only once, for final AUROC.
- Donor samples come only from donor training partitions.  The entire target
  family is absent from the donor fit (`LOFO`, not merely LOCO).
- Domain and family metadata are fixed before labels are read.
- Donor acquisition is uniform and label-blind: 20 labels per donor cell.
- Uniform and active target acquisition are nested, label-blind policies.  The
  v1 controlled-stratified arm is retained only as an optimistic upper bound;
  it is not a fair active-learning comparator because it uses labels to force
  both classes.

## Frozen methods

1. `upcr`: stable confidence-oriented U-PCR incumbent.
2. `local_controlled2`: v1 two-direction head with controlled stratification;
   reproduction/optimistic control.
3. `local_uniform2`: identical local head with nested uniform acquisition.
4. `local_active2`: identical local head with nested Fisher/D-optimal
   acquisition.
5. `pooled_domain_lofo`: shared donor head from the same broad domain after
   excluding the target family; no target labels.
6. `pooled_all_lofo`: shared donor head from both domains after excluding the
   target family; tests whether broad pooling washes out domain structure.
7. `hybrid_domain_uniform`: transferred two-score head updated with uniformly
   acquired target labels.
8. `hybrid_domain_active`: the same transferred head updated with D-optimal
   target labels.

The U-PCR/local prior strength remains 10 from v1.  The shared donor head uses
prior strength 20 because it estimates 17 shared coefficients; no strength is
tuned on correctness outcomes.  Nuisance donor intercepts have fixed L2
strength 1.  Label budgets are `0, 5, 10, 20, 40, 80`.  Confirmatory counts are
20 split repetitions per real cell and 20 repetitions for each synthetic
meta-world.

## Synthetic meta-worlds

Each world has 12 cells in three four-cell families and uses a disjoint seed
namespace `hierarchical-active-spectral-v2-2026-08-06`.

- `upcr_sufficient`: independent errors; donor correction should be unnecessary.
- `shared_correction`: a correlated weak-feature block is shared across
  families; LOFO pooling should learn a transferable down-weighting.
- `family_shift`: the six informative coordinates change by family while the
  other coordinates are noise; this is the explicit negative-transfer
  falsifier for a shared correction.

### Preflight amendment before the confirmatory run

The ineligible quick run exposed that the first `family_shift` generator used
three six-feature blocks on only 16 columns with modular wraparound.  The blocks
therefore overlapped and did not instantiate the registered non-transfer
mechanism.  Before any confirmatory output was generated, the synthetic matrix
was changed to 18 features with disjoint family blocks `[0:6]`, `[6:12]`, and
`[12:18]`.  A second preflight showed that merely moving a correlated weak
block was still transferable: any small positive donor coefficient repaired
the deliberately catastrophic U-PCR score.  The final negative control instead
makes one disjoint block informative per family and the other two blocks pure
noise.  All informative coordinates keep the same confidence orientation; only
which coordinates carry information changes.  No real-data method, split,
seed, hyperparameter, or gate changed.

## Frozen real-data gates

At 20 target labels, `hybrid_domain_active` must satisfy all of:

1. mean paired cell delta versus `upcr` >= +1.00 percentage point;
2. cell-bootstrap 95% lower bound versus `upcr` > 0;
3. mean delta versus `local_uniform2` >= 0;
4. mean delta versus `hybrid_domain_uniform` >= 0;
5. QA and math deltas versus U-PCR each >= -0.50 points;
6. no more than two cells lose at least 5 points versus U-PCR.

The two mechanisms are also reported separately even if the combined candidate
fails:

- transfer: `hybrid_domain_uniform - local_uniform2`;
- acquisition: `hybrid_domain_active - hybrid_domain_uniform` and
  `local_active2 - local_uniform2`.

## Frozen synthetic gates

At 20 target labels:

- `hybrid_domain_active - upcr` in `shared_correction` >= +1.00 point;
- its 95% lower bound in `shared_correction` > 0;
- `hybrid_domain_active - upcr` in `upcr_sufficient` >= -0.50 points;
- `pooled_domain_lofo - upcr` in `family_shift` >= -1.00 point;
- active minus uniform acquisition is non-negative in `shared_correction`.

These gates establish that the implementation can exploit shared structure and
recognize a transfer boundary.  They cannot rescue a failed real-data claim.

## Interpretation rules

- If pooling helps synthetic shared structure but not real cells, conclude that
  the current correction is not stable across real dataset families.
- If active beats uniform locally but the hybrid does not, acquisition works
  but donor transfer is the bottleneck.
- If pooled-only works and adding target labels loses, the update prior or
  target sample size is the bottleneck.
- If `pooled_all_lofo` beats domain-only, the QA/math split was not the relevant
  hierarchy; report it as an ablation, not a tuned replacement.
- A method that wins only with controlled stratification has not demonstrated a
  deployable acquisition strategy.
- Passing the retrospective real replay permits a prospective family replay;
  it is not final evidence of generalization.
