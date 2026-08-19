# Layer-organic NRM methods

Status: **PRELIMINARY / VALIDATION BLOCKED**. This is a retrospective
structural addendum to the frozen white-box v2 benchmark; it does not replace
the registered v2 primary or the earlier four-depth-band NRM addendum.

## Organic feature contract

The primary matrix is derived only from the frozen, label-free
`lens_grid_all` bundles. Each of the 32 residual transformer layers is one
group. Its three internal features are the oriented token means of:

1. residual logit-lens entropy;
2. residual target-token NLL (`-lens_logp_tgt`);
3. residual top-1 surprisal (`-lens_logp_top1`).

Thus the primary matrix has 96 atomic features and 32 groups. Every atomic
column is standardized by the same canonical white-box standardizer. The sole
orientation anchor remains final-layer residual target-token NLL.

KL-to-final is not included in the primary triad because it couples every
layer to the final layer and is therefore not strictly local. A separate
sensitivity includes it as a fourth within-layer feature. The mechanically
zero final-layer KL column is absent, giving 127 features.

## Fusion and NRM

IU-PCR is fitted over the atomic matrix. Its weighted atomic contributions are
summed within each layer, producing 32 layer contributions that reconstruct
the IU score to floating-point precision. NRM then uses the already frozen
rule:

- residualize standardized layer contributions against standardized IU;
- average source-cell residual covariance with equal cell weight;
- select the covariance eigenvector whose eigenvalue is closest to one;
- orient it toward the equal-layer risk direction;
- add the projected correction at trust `1/32`.

No fitting API accepts outcomes. Score and diagnostic bundles are hashed in
`SCORE_FREEZE_MANIFEST.json` before the evaluator calls the raw-label loader.

## Transfer cohorts

Exact layer identity is used only for protocol-eligible 32-layer cells.
Qwen3-8B (36 layers) and the two 40-layer Mistral cells are excluded rather
than silently interpolated.

- `all_32layer` (10 cells): LODO, LOMO, and LOCO source definitions.
- `same_model_llama_six` (6 cells): exact Llama-3.1-8B fixed; each target is
  calibrated from the other five datasets.
- `same_dataset_gsm8k_32layer` (5 cells): GSM8K fixed; each target is
  calibrated from the other four 32-layer models.

The last two are the clean structural controls. The roster is still not a
fully crossed model-by-dataset design.

## Evaluation

The evaluator defines `y_hallucination = 1 - correctness`. Primary metric is
candidate AUROC; secondary metric is AUPRC with prevalence retained per cell.
Uncertainty uses 2,000 deterministic paired bootstrap draws (root seed
`20260812`) resampling whole problem groups within each cell. Identical draws
are reused across methods. Macro values weight cells equally. W/T/L uses
`±0.001` AUROC tie tolerance; Wilcoxon tests are Holm-adjusted and treated as
low-power supporting evidence.
