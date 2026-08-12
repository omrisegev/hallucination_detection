# Specification: Leverage-Balanced Contribution-Subspace IU v1

**Date:** 2026-08-12

**Status:** retrospectively discovered, formula frozen for a mechanism audit;
not eligible as a prospective confirmation on the 24-cell development bundle.

## 1. Question

Can IU-PCR be improved without labels, new features, additional inference, or
white-box access by correcting provenance-family leverage inside its own fusion
score?

This is the first label-free candidate derived from the supervised
contribution-subspace teacher in
`SPEC_HARP_CONTRIBUTION_SUBSPACE_IU_V1.md`.  It is intentionally not another
sample graph, feature selector, hard filter, or nuisance-deflation operator.

## 2. Discovery disclosure

The candidate was found after inspecting correctness labels on the existing
23 eligible development cells.  Exploratory comparisons included uniform,
family-cardinality, masked-predictability, baseline-loading, signed-correlation,
and IU-weight-leverage directions over a small correction-strength path.

Consequently:

- the current 24-cell run is a retrospective mechanism audit;
- its uncertainty describes sampling/cell variation, not prospective discovery
  uncertainty;
- no result on these cells can confirm the method;
- the exact formula below must be frozen before any new external-family test.

## 3. Frozen algorithm

Fit ordinary IU-PCR on the unchanged mixed-v2 feature matrix `F`, obtaining
weights `w`.  Group per-feature score contributions by the frozen provenance
registry:

```text
h_g(x) = sum_{i in g} w_i F_i(x),
b(x)   = sum_g h_g(x).
```

Standardize `b` and each `h_g` on the same unlabeled batch.  Regress every
standardized contribution on standardized `b` and retain its residual.  Stack
the residuals as `R`.  On the fit batch, each column of `R` has mean zero,
variance one, and zero linear covariance with `b`.

Define observable family leverage

```text
ell_g = sum_{i in g} |w_i|.
```

The frozen leverage-balancing direction is the centered log ratio

```text
d_g = mean_h(log(max(ell_h, eps))) - log(max(ell_g, eps)).
```

Thus a family whose total IU leverage exceeds the geometric family mean is
downweighted, while an underrepresented family is upweighted.  Centering and
the log ratio make the direction invariant to a common rescaling of `w`.

Let

```text
q = R d / std(R d).
```

For `G` present provenance families, the final score is

```text
s_LB = b + (1/G) q.
```

The `1/G` trust scale is fixed from the observable family count.  It limits the
new unit-variance correction to one equal family share of the standardized IU
baseline.  There is no strength path, validation-label selection, or dataset
identity input in the frozen method.

Degenerate identity rule: if all family leverages are equal or `std(Rd)` is
numerically zero, return `b` exactly.

Because the transform is affine, the final score is mapped back exactly to the
original feature coordinates:

```text
s_LB(x) = w_LB^T F(x) + c.
```

The implementation returns `w_LB` and `c` and verifies their score
reconstruction to floating-point tolerance.  The candidate is therefore an
addition to the IU-PCR weight calculation, not an extra detector stacked after
fusion.

## 4. Claim boundary

The algorithm is label-free at fit and deployment.  It uses only:

- the existing one-pass feature matrix;
- ordinary IU-PCR weights;
- the frozen feature-provenance registry;
- unlabeled sample moments of the contribution matrix.

It does not use correctness labels, hallucination types, prompts, generations,
hidden states, logits beyond the existing feature contract, model parameters,
attention, or gradients.  The provenance registry describes measurement
origin rather than hallucination behavior.

Because the formula was discovered on labelled development cells, the present
evidence supports only “label-free execution of a retrospectively selected
rule.”  A new dataset family is required for a confirmed label-free claim.

## 5. Retrospective audit protocol

### 5.1 Fit phase

The fit command must never load any `__labels` key.  For each of the registered
24 cells it writes and hashes:

- ordinary IU score;
- primary leverage-balanced score;
- uniform residual-direction control;
- family-cardinality balancing control;
- reversed-leverage falsifier;
- deterministic within-cell family-permutation controls;
- family leverage, correction coefficients, reconstruction, orthogonality,
  and scale diagnostics.

`spilled_triviaqa_llama8b` is fitted but excluded from aggregate evaluation by
the pre-existing `n_positive < 20` rule, which is applied only in the report
phase.

### 5.2 Report phase

Verify every score hash before opening labels.  Report:

- cell-macro and equal-dataset-family AUROC;
- 20,000-draw equal-family bootstrap intervals;
- wins/losses/ties and worst-cell delta;
- paired difference from each mechanism control;
- score/reconstruction/orthogonality invariants;
- recovery relative to the previously frozen supervised teacher result.

All eight dataset families receive equal weight in the family summary.  The
single-cell families remain a limitation and are not pseudoreplicates.

The already-frozen full-pool mixed-v2 DUFS-LIU scores from
`results/hard_filter_dufs_liu_24cell/` are included as a secondary incumbent
comparison.  This comparison was added after the exploratory primary result
was visible and is therefore descriptive, not an additional preregistered
gate.  Its score hashes and its matched IU score must be verified before use.

## 6. Retrospective continuation gates

These gates determine whether the candidate is worth external confirmation,
not whether it is already confirmed.

1. equal-family AUROC delta over IU is positive with bootstrap lower bound
   above zero;
2. at least 16 wins among the 23 eligible cells;
3. worst-cell delta is no lower than -1.0 percentage point;
4. the primary recovers at least 30% of the supervised teacher's frozen
   equal-family gain (`+0.721pp`);
5. primary mean delta exceeds the uniform and cardinality-only controls;
6. reversed leverage is worse than primary;
7. primary exceeds the mean of permuted-family controls;
8. correction standard deviation equals `1/G`, contribution reconstruction is
   within floating-point tolerance, and baseline/correction covariance is zero
   within `1e-10` in every cell.

Failure of gates 5--7 means the leverage mechanism is unsupported even if the
primary AUROC is positive.

## 7. Required confirmation

If the retrospective gates pass, freeze source hashes and run the identical
formula on a new intrinsic-detection dataset family or a newly generated model
family whose labels were not used in discovery.  Required confirmation gates:

- no formula, scale, family registry, or exclusion change;
- paired AUROC improvement over ordinary IU-PCR with interval lower bound above
  zero at the independent-family level;
- no cell below -1.0pp;
- no degradation of the existing score when the correction degenerates;
- leverage direction remains superior to uniform/cardinality controls.

Only that result can promote the method as the requested non-supervised
algorithmic addition to IU-PCR.
