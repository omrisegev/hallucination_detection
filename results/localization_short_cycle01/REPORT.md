# Localization short cycle 1 — answer-only Joint pilot

Date: 2026-09-07. Status: completed and reviewed. This is a bounded
development pilot, not a publication result and not a replacement for the
24-cell benchmark.

## Question

Can the Joint model-inverse with lambda zero learn a useful localizer from one
long answer's own windows, without borrowing groups, scales, signs, weights,
labels or thresholds from other answers?

## Frozen protocol

- Population: existing PRMBench Qwen3-8B telemetry cell.
- Selection: 30 distinct answer IDs, deterministic SHA-256 ordering among
  answers with 1,024–2,048 tokens. Selection used only row IDs and lengths.
- Representation: original 30 window feature definitions, width 32 and
  non-overlapping fitting windows. Constant/unavailable columns were removed
  per answer and recorded. Token scores were mapped back to official steps;
  the fixed step readout was span maximum.
- Fitting: per-answer standardization, orientation, temporal-block stability
  grouping and weights. Main arm was Joint model-inverse lambda zero. IU and
  equal weights were fitted on exactly the same windows as references.
- No pooled fit, cross-answer parameter, answer label or correctness threshold
  entered the score phase. Scores were frozen before labels were opened.

## Results

| Method | Available answers | Strictly converged answers | Steps in available scores | Pooled step AUROC |
|---|---:|---:|---:|---:|
| Joint model-inverse λ=0 | 26 | 24 | 1,043 | 0.66999 |
| IU | 30 | 30 | 1,239 | 0.70070 |
| Equal average | 30 | 30 | 1,239 | 0.69648 |

On the common 24 strictly converged answers, pooled step AUROC was Joint
`0.69148`, IU `0.72072`, and equal `0.70134`. A descriptive grouped bootstrap
over those 24 answers gave:

- Joint − IU: `−0.02875`, 95% CI `[−0.04837, −0.00768]`.
- Joint − equal: `−0.01016`, 95% CI `[−0.04856, +0.03005]`.

The IU comparison is unfavorable to Joint in this pilot. The equal comparison
is inconclusive. The interval is a pilot diagnostic interval, not a claim about
the full PRMBench population.

## What failed and why it matters

Joint produced no admissible temporal-block partition in 4/30 answers. Two
additional fits produced finite scores but failed the multi-start convergence
requirement and are reported only descriptively. IU and equal succeeded on all
30. The Joint fit therefore has 24/30 strict coverage; no failure was replaced
with a fallback or an all-correct prediction.

The one-answer adaptation uses four contiguous temporal blocks as resampling
units. With roughly 32–60 fitting windows and 29 active features, this is a
small and high-dimensional grouping problem. The pilot shows that the current
Joint grouping contract is fragile in this setting. This is a mechanism and
coverage finding, not evidence that all possible answer-only Joint designs are
impossible.

## Code and provenance review

The score phase passed syntax/import checks. Every row score array was finite;
the 30 selected IDs were unique; all selected lengths were 1,035–1,945 tokens;
the score freeze records `labels_accessed=false`, the cell hash, fitting-module
hash and runner hash. The evaluation records that selection was frozen before
labels and stores the label-file hash. A missing Joint fit remains explicitly
missing in the evaluation table. Pooled and strict-common metrics are separate.

Review findings that remain limitations:

1. The result is pooled over steps, so within-answer dependence is handled only
   in the grouped bootstrap; it is not an independent-step confidence interval.
2. Four temporal blocks are an engineering adaptation, not a validated answer
   grouping contract. Their adequacy needs a separate study if Joint is revived.
3. The fixed span-max readout tests the feature-fusion premise only. It does not
   test a learned chronological trajectory reducer.
4. The 30-answer hash-selected pilot is retrospective and small. It does not
   establish full-cell performance or a publication-ready comparison.

## Decision for the next short cycle

Do not launch a lambda sweep, the two-axis grid, or the 24-cell transfer yet.
The cheapest next question is whether the answer-only Joint grouping is the
problem: compare the same 30 answers with a predeclared non-clustering Joint
reference (fixed four feature groups or the model-inverse without learned
groups), keeping IU/equal and the same score/readout. If that cannot be made
stable, prioritize the simpler IU/equal localizer and study trajectory mapping
separately.

Artifacts: `CONFIG.json`, `SCORES_FROZEN.json`, `EVALUATION.json`, one
`row_*.npz` per selected answer, and the source implementation in
`spectral_utils/short_cycle_localization.py` plus
`scripts/run_short_cycle01.py`.
