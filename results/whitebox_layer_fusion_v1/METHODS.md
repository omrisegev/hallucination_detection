# White-box Layer-Fusion Benchmark — Methods

## Status and scope

This is an offline, cross-dataset benchmark for `meta-llama/Llama-3.1-8B-Instruct` using six pre-existing `layer-lens-v1` captures. The result is **PRELIMINARY / VALIDATION BLOCKED** until the corrected live Gate B check passes for all six cells and the two-cell architecture-fidelity pilot passes. No cross-model claim is made.

## Data contract

The source roster is GSM8K T=1.0, TriviaQA T=1.0, SciQ T=1.0, TruthfulQA T=0.5, SQuADv2 T=0.5, and NQ-Open T=0.5. Raw caches and sidecars were copied read-only from `gdrive:hallucination_detection/` and frozen with remote path, size, modification time, remote hash where available, and local SHA-256.

Every sidecar key `i:j` is joined to `raw[i]["candidates"][j]`. The audit checks candidate count, labels, generated-token length, tensor shapes, finite values, model metadata, layer count, and projection metadata. Four GSM8K rows whose sidecar token axis was truncated at 1,024 tokens are explicitly excluded, leaving 30,166 evaluable candidates from 30,170 source candidates. Problem IDs, not candidate rows, are the resampling unit.

## Leakage boundary

Preparation produces label-free `LayerCell` and `FeatureMatrix` artifacts. Fitting receives feature matrices, feature names, a label-free risk anchor, and fixed feature groups, but no labels. Score bundles and their hashes are frozen before evaluation reopens raw labels and defines `y_hallucination = 1 - label`. The fit diagnostics record `labels_seen_during_fit=false`, and evaluation rechecks source, prepared-feature, code, and score hashes.

## Feature contracts

The registered headline contract, `resid-core-32`, creates one expert per layer by equally averaging standardized token means for residual-stream entropy, target-token NLL, top-1 surprisal, and KL-to-final. Final-layer residual target-token NLL is the only global orientation anchor. Mechanically degenerate columns, including exact final-layer KL, are dropped.

Fixed layer subsets are all 32 layers; spaced eight `[0,4,9,13,18,22,27,31]`; and late eight `[24,25,26,27,28,29,30,31]`. The secondary `lens-96` contract uses four metrics at three module positions over the spaced-eight layers. Token count is excluded from fusion inputs and appears as a confound baseline; a fixed sensitivity residualizes every feature against `log1p(token_count)` without labels. Representation-geometry performance is omitted because the original capture implementation and projection semantics have not yet been validated.

## Methods

Controls are final-layer NLL, token length, equal mean, and anchor-oriented PC1. Core methods are deployed U-PCR, IU-PCR, and DUFS-LIU-PCR, using the repository's frozen deployed settings. DUFS uses seeds 11/23/37, 80 epochs, k=7, and lambda=0.1; lambda=0 is asserted to reproduce IU-PCR exactly.

Dependency controls are SU-PCR, canonical continuous L-SML, and cross-band clustered U-PCR with fixed bands `[0–7]`, `[8–15]`, `[16–23]`, `[24–31]`. Matched hierarchical variants fuse inside the same four bands and then across the four virtual experts. Flat and module-by-metric hierarchical variants are also evaluated on `lens-96`. Balanced logistic regression and best-single-layer curves are label-using diagnostic ceilings only and are excluded from headline selection.

## Evaluation

Candidate-level AUROC is primary; AUPRC is secondary and prevalence is reported. The headline is the equal-cell macro across the six datasets, never a pooled-candidate metric. Confidence intervals and method differences use 2,000 deterministic paired bootstrap draws with base seed `20260812`, resampling problem groups within each cell and reusing identical draws across methods. Comparisons also report per-cell deltas, wins/ties/losses with a ±0.001 tie tolerance, worst-cell loss, and Holm-adjusted Wilcoxon tests as low-power supporting evidence.

The two primary registered contrasts are DUFS-LIU-PCR all32 minus final-layer NLL and DUFS-LIU-PCR all32 minus IU-PCR all32. A robust-improvement claim requires both macro-AUROC intervals to exclude zero on the positive side and completion of both validation gates.

## Reproduction

Run the phases from the worktree root with the project's scientific Python environment:

```bash
python scripts/whitebox_layer_fusion_experiment.py prepare
python scripts/whitebox_layer_fusion_experiment.py fit
python scripts/whitebox_layer_fusion_experiment.py evaluate
python scripts/whitebox_layer_fusion_experiment.py report
```

The exact run definition, source freeze, prepared-feature freeze, score freeze, bootstrap-draw manifest, evaluation tables, diagnostics, report manifest, and self-contained HTML report are stored beside this file.
