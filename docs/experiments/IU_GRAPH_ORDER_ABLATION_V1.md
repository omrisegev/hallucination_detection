# IU graph-order ablation v1

## Question

Does graph regularization help because it changes the feature matrix before
IU-PCR re-estimates the common direction, or because it supplies a constrained
correction in the IU-orthogonal family-residual space?

The current reconstruction roster's `DEEM-B3` is a continuous, graph-free
energy model.  It is not the residual/Laplacian mechanism tested here.  The
new correction arm is therefore named `residual_ridge_correction`, never
`DEEM-B3`.  The signed, already reconstructed `DEEM-B3` score is included only
as a context reference after the new scores freeze.

## Frozen common input

All arms receive the same canonical mixed-v2 confidence matrix.  No feature is
reoriented.  The answer graph is built without targets, with the existing
duplicate-safe symmetric union self-tuning kNN constructor (`k=7`) and the
symmetric normalized Laplacian.

Two graph coordinate systems are kept separate:

1. the canonical standardized IU family-contribution residual matrix `R`,
   used for the matched comparison;
2. the full prepared feature matrix `X`, used only as a raw-geometry
   sensitivity.

## Arms

The unchanged anchors are canonical IU-PCR and equal-family mean.

For every frozen `lambda` in `{0.03, 0.1, 0.3, 1, 3, 10}`:

1. `feature_smooth_residual_graph`: compute
   `Z=(I+lambda L_R)^-1 X`, population-standardize each nonconstant column,
   then refit canonical IU-PCR;
2. `feature_smooth_raw_graph`: the same operation with a graph built directly
   on `X`;
3. `score_smooth_residual_graph`: smooth the already fitted IU score on `L_R`;
4. `residual_ridge_correction`: solve exactly

   `delta=-lambda (I+lambda R^T L_R R/n)^-1 R^T L_R b/n`

   and score `b+R delta`, with no post-hoc correction rescaling.

`lambda=1` is the preregistered unit-strength display point.  `lambda=.03` is
shown because it matches the historical graph-roughness study.  Neither is
selected by labels; the complete response curve is the scientific object.

## Evaluation and claim boundary

Build A and Build B independently consume the two byte-verified prepared
snapshots.  Their score arrays must be byte-identical before labels are read.
Evaluation then joins the already signed frozen-24 prediction snapshot by exact
row ID and uses 20,000 shared source-group bootstrap draws per cell.  Report
AUROC and AUPRC per cell and equal-cell macro, plus paired deltas versus IU-PCR,
equal-family mean, and the signed current `DEEM-B3` context score.

The 24 cells were used retrospectively to develop mixed-v2 and several
mechanisms.  This experiment is D0 mechanism evidence, not independent
validation.  A best lambda observed after label opening is an oracle
sensitivity and cannot be promoted.
