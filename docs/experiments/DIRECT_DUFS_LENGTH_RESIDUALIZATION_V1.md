# Direct DUFS train-fitted length residualization v1

## Question

After removing the explicit length coordinate, can a residualizer learned only
from training cells remove the *indirect* length geometry while preserving a
hallucination-related geometry?

This is a retrospective mechanism diagnostic. It does not tune a promoted
method and it does not pool target metrics across tasks.

## Frozen validation lanes

- **Global:** 21 cells with an explicit `trace_length` coordinate. Each cell is
  held out in turn; the residualizer is fit on the other 20 cells.
- **ProcessBench:** the four Qwen3-4B development cells fit one residualizer;
  the four Qwen3-8B cells are model-held validation cells.
- **RAGTruth:** the development split fits the residualizer and the
  `original30_full` test graph is the validation graph.

Labels are never used by the residualizer, DUFS, graph construction, or score
fitting.

## Residualizer

1. Delete every explicit length feature.
2. Within each training cell, transform length to `log1p(length)`, robustly
   center it by its median, and robustly scale it by `1.4826 * MAD` (fall back
   to standard deviation when necessary), then clip it to `[-5, 5]`.
3. Form a centered cubic basis: `z`, `z^2`, `z^3`.
4. For each feature name, pool only training cells containing that feature and
   fit a ridge regression of the within-cell-centered feature on that basis
   (`alpha=1e-3`; no target labels).
5. Apply the frozen coefficients to the held-out cell's correspondingly
   normalized length basis. Standardize the residual coordinates without
   labels, refit DUFS with the frozen seeds/epochs, and construct the usual
   union-kNN graph with `k=7`.

Using relative, within-cell length is intentional: feature matrices are already
standardized per cell and absolute token scales differ across models and tasks.
The residualizer therefore tests whether a shared within-cell length mechanism
transfers, not whether one global token-count threshold transfers.

## Conditions

1. `original`: frozen graph with explicit length.
2. `drop_length_refit_gates`: explicit length deleted and DUFS refit.
3. `train_residualized_refit_gates`: explicit length deleted, remaining
   features residualized by the training-only model, and DUFS refit.

## Measurements and decision

For each validation graph, measure target and held-out length smoothness as the
symmetric-normalized-Laplacian energy reduction relative to 200 row
permutations. Also report median absolute feature/length Spearman correlation
and IU-PCR / DUFS-LIU AUROC.

- `RESIDUALIZATION_REVEALS_TARGET_SPECIFIC_GEOMETRY`: target smoothness exceeds
  length smoothness in at least half of the validation cells in every lane.
- `RESIDUALIZATION_REMOVES_LENGTH_BUT_NOT_TARGET_SPECIFIC`: length smoothness is
  materially reduced, but the target does not dominate in every lane.
- `TRAIN_FITTED_RESIDUALIZATION_DOES_NOT_REMOVE_LENGTH_GEOMETRY`: median length
  smoothness falls by less than 20% relative to the no-length condition in at
  least one lane.

RAGTruth contains one test graph, so its result is a split-level diagnostic,
not an estimate of across-cell prevalence.
