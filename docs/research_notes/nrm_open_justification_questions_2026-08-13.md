# NRM-CS-IU: open justification questions

**Date:** 2026-08-13
**Status:** unresolved research debt; this note does not modify the frozen v1
algorithm or its confirmation protocol.

## 1. Literature basis for the neutral eigenvalue rule

NRM-CS-IU currently selects the residual-covariance eigenvector whose
eigenvalue is closest to one.  The internal motivation is that every family
residual is standardized: unit covariance is the independent-residual null,
large eigenvalues indicate amplified shared dependence, and near-zero modes
indicate deterministic redundancy.

This is presently a structural heuristic, not a cited identification theorem.
Before presenting it as a principled target/nuisance separator, search the
professional literature for a defensible source or closest analogue, including:

- standardized correlation-matrix nulls and noise eigenmodes;
- spiked-covariance and factor-model separation of common factors from
  idiosyncratic variation;
- residual PCA, whitening, and neutral/noise-subspace selection;
- random-matrix results distinguishing unit/noise bulk modes from spikes;
- ensemble-regression or crowdsourcing results that connect such a mode to a
  shared latent regression target.

The provenance of the rule must be reported honestly: state whether it is a
direct consequence of a cited model, an adaptation of a known diagnostic, or
our own heuristic.  A citation to a unit-noise bulk alone would justify the
spectral geometry, not automatically identify hallucination signal.

## 2. Manual-family assumption

The six current families are assigned by the primitive trajectory from which a
feature was engineered, using the frozen `FEATURE_TO_VIEW` registry:

- entropy level;
- entropy-trajectory dynamics;
- sampled-token/spilled-energy trajectory;
- full-vocabulary log-partition trajectory;
- top-k distribution summaries;
- structural trace length.

This assignment is deterministic, auditable, and independent of correctness
labels.  It establishes feature provenance.  It does **not** establish that
features inside a family share an error factor, that different families are
the correct exchangeability blocks, or that this partition follows from the
original U-PCR assumption that every feature regresses the same scalar target.

Using the partition therefore adds a structural prior and breaks permutation
invariance over input features.  The method must not describe family membership
as something known statistically merely because it is known operationally from
the feature computation graph.

## 3. Required controls before a general method claim

Evaluate the frozen family-based NRM against at least:

1. an atomic, group-free contribution-residual version with one coordinate per
   named feature;
2. label-free learned groups, trained only on source telemetry and transferred
   unchanged;
3. random or cardinality-matched partitions;
4. deterministic refinements and coarsenings of the provenance partition;
5. feature-order permutation and family-label permutation invariance checks.

If the effect survives these controls, the family partition is probably not
load-bearing.  If it survives only for the hand registry, the honest claim is a
provenance-informed fusion method, not a generic consequence of U-PCR.  If an
atomic formulation matches or improves it, prefer the atomic formulation and
remove manual families from the final algorithm.
