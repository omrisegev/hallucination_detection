# Specification: Cardinality-Balanced Contribution-Subspace IU v1

**Date:** 2026-08-12

**Status:** final non-supervised candidate frozen after a post-transfer mechanism
revision; requires a new, pristine external confirmation.

## 1. Decision and disclosure

The current candidate is **Cardinality-Balanced Contribution-Subspace IU
(CB-CS-IU)**.  It promotes the family-cardinality control from the earlier
leverage-balanced audit into a first-class fusion rule.

This choice was made after correctness-labelled reports from both the original
24-cell bundle and ProcessBench had been inspected.  In particular,
cardinality balancing transferred more strongly than L1-leverage balancing on
ProcessBench.  Therefore neither dataset is a prospective confirmation of the
selection between those variants.  They are retrospective selection evidence,
and the exact method below must not be changed before the next external test.

## 2. Target/nuisance interpretation

The supervised HARP-inspired proof-of-concept established that useful target
correction exists in the low-dimensional space of IU-PCR provenance-family
contributions.  It did not make a supervised classifier deployable.

CB-CS-IU makes a narrower, identifiable nuisance claim:

- the ordinary standardized IU score is retained as the target-consensus axis;
- family residuals orthogonal to that score contain degrees of freedom that can
  change ranking without discarding the shared IU signal;
- unequal numbers of engineered measurements per provenance family are a
  design-induced nuisance, not evidence that one source is more truthful;
- the correction removes that multiplicity bias only in the IU-orthogonal
  residual subspace.

This does not claim to identify the full latent truth/false manifold.  It
protects the shared IU target axis while removing one nuisance whose origin is
known independently of correctness labels.  That identifiability is stronger
than treating an empirically smooth or stable sample direction as nuisance.

## 3. Frozen algorithm

Fit ordinary IU-PCR on the unchanged feature-by-sample matrix `F`, obtaining
weights `w`.  Group feature contributions using the frozen provenance
registry:

```text
h_g(x) = sum_{i in g} w_i F_i(x)
b(x)   = sum_g h_g(x).
```

On the same unlabeled fit batch, standardize `b` and every `h_g`.  Regress each
standardized contribution on standardized `b`, then center and standardize its
residual.  Stack the residuals as `R`.  Every residual coordinate has zero
linear covariance with `b` on the fit batch.

For each present family, let

```text
m_g = number of feature coordinates assigned to family g
d_g = mean_h(log m_h) - log m_g.
```

Let `G` be the number of present families and

```text
q = R d / std(R d)
s_CB = b + (1/G) q.
```

The correction is fixed to one equal family share of the standardized IU
score.  There is no strength path, validation-label selection, dataset router,
or target-specific parameter.

If all present families have equal cardinality, or `std(Rd)` is numerically
zero, return standardized IU-PCR exactly.  The affine transform is mapped back
to the original coordinates and returned as

```text
s_CB(x) = w_CB^T F(x) + c.
```

## 4. Deployment contract

CB-CS-IU uses only:

- the existing one-pass feature matrix;
- ordinary IU-PCR weights;
- the existing feature-provenance registry;
- unlabeled moments of the current contribution matrix.

It adds no model inference, feature extraction, prompts, generations, labels,
hidden-state access, attention access, gradients, or model parameters.  The
public entry point is `spectral_utils.cardinality_balanced_iu_fit`.

## 5. Why cardinality, not the original leverage rule

Both rules use the same IU-orthogonal residual geometry and the same fixed
`1/G` trust scale.  They differ only in the observable used to orient the
family direction:

- LB-CS-IU balances realized `sum(abs(w_i))` per family;
- CB-CS-IU balances the fixed number of measurements per family.

On the original 23 eligible cells, both were positive and leverage had a small
point advantage whose paired family interval crossed zero.  On the frozen
ProcessBench transfer, cardinality beat leverage in all six confirmation cells
and by a positive equal-subset interval.  Cardinality also has the cleaner
causal nuisance interpretation: feature-family multiplicity is introduced by
the measurement design itself.

This is a mechanism revision, not a post-hoc claim that ProcessBench confirmed
CB-CS-IU.  The selection penalty is handled by requiring a new benchmark.

## 6. Required pristine confirmation

Before opening labels for a new intrinsic-detection dataset or genuinely new
model family:

1. record hashes of this specification, the implementation, feature registry,
   and evaluation script;
2. freeze eligible cells, exclusion rules, target orientation, ordinary IU and
   DUFS-LIU incumbents, and grouped resampling unit;
3. fit and hash all scores without reading labels;
4. require positive grouped paired AUROC interval versus ordinary IU-PCR;
5. report paired comparisons with DUFS-LIU and LB-CS-IU, but do not select a
   further variant from the confirmation labels;
6. require no cell below -1.0 percentage point and exact identity in degenerate
   equal-cardinality cases;
7. verify score reconstruction, correction scale `1/G`, and IU/correction
   covariance to floating-point tolerance.

Until such a run passes, CB-CS-IU is the best current non-supervised candidate,
not a prospectively confirmed final method.
