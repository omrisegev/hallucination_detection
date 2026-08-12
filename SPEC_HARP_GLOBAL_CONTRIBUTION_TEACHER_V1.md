# Specification: HARP-inspired global contribution teacher v1

**Date:** 2026-08-12

**Status:** supervised proof-of-feasibility; retrospective research instrument,
never a deployable unsupervised method.

## Question

The within-cell supervised contribution head already showed that a useful
target correction exists when labels from the same cell are available.  The
new question is harder:

> Is there one correctness-aligned direction in provenance-family contribution
> space that transfers across held-out dataset families and to independent
> benchmark examples?

A positive result proves that the contribution space contains a generalizable
target manifold.  It does not prove that this direction can be identified
without labels.

## Representation

For every cell, independently and without using its labels:

1. fit ordinary mixed-v2 IU-PCR;
2. decompose its score into exact provenance-family contributions;
3. standardize each family contribution and the IU score on that cell;
4. residualize each family contribution against standardized IU;
5. align residual columns to the frozen six-family `VIEW_ORDER`, using zero for
   a family absent from the concrete feature contract.

Let `b_c` be the standardized IU score and `R_c` the aligned residual matrix.
The global supervised teacher is

```text
s_c = b_c + R_c delta.
```

The coefficient of ordinary IU is fixed to one.  `delta=0` is exact IU ranking.
The operation is affine in the existing feature matrix and remains inside the
fusion calculation.

## Supervised fit

Fit one six-dimensional `delta` with class-balanced logistic loss and fixed
ridge strength `0.3`, carried unchanged from the earlier anchored-head PoC.
Every source cell has equal total weight; within each cell, correct and
incorrect examples each receive half of that cell's weight.  There is no
intercept and no strength path.

The target is answer correctness.  ProcessBench correctness means no annotated
reasoning error (`label == -1`); SemGrad correctness is `bem_correct`.

## Evaluations

1. **Original-family LOFO:** train on all original eligible cells outside one
   dataset family, evaluate every cell in the held-out family, and repeat over
   all eight families.
2. **Frozen source-23 transfer:** fit once on all 23 eligible original cells,
   then apply the same `delta` without target labels to:
   - eight Qwen3 ProcessBench cells;
   - four Llama-3.1-8B ProcessBench scorer-family cells;
   - SemGrad SciQ and TruthfulQA independent answer-level examples.

Report cell-macro and equal-group AUROC deltas versus ordinary IU, wins/losses,
worst cell, and 20,000-draw group bootstrap intervals.  ProcessBench model
replicates are grouped by subset.  SemGrad's two datasets remain two groups.

The already-frozen CB-CS-IU score is included only to localize the gap between
supervised target identification and the current label-free proxy.

## Interpretation boundary

- LOFO evidence is retrospective because the original cells have been used in
  extensive project development.
- External labels already existed in the repository, but they do not enter the
  source-23 teacher fit.
- A transfer gain supports a reusable target direction, not a final algorithm.
- Failure of CB where the teacher succeeds means the family contribution space
  is adequate but cardinality is not an adequate nuisance/target identifier.
- The final requested method must replace the supervised `delta` with a
  label-free, self-supervised, or otherwise non-supervised structural rule.
