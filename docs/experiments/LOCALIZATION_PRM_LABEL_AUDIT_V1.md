# PRMBench label correction and frozen-score bridge — Step 313 amendment

The planned frozen-score forensics stopped on its direct raw-label assertion.
The Qwen3-8B PRMBench cache retains one-based `error_steps`; Claude v2's
`scripts/joint_lsml_optimization_v2/run_v2.py` writes `flags[step]` instead of
`flags[step-1]`. Recent Codex evaluations inherited that label NPZ. Earlier
reviews reconstructed scores and checked labels against this NPZ; that did
not independently verify the raw annotation contract. The discovery is a
real evaluation defect, not evidence of a predictor failure.

Official source checked September 7, 2026:
[PRMBench task evaluator](https://github.com/ssmisya/PRMBench/blob/main/mr_eval/tasks/prmtest_classified/task.py).
Its evaluator subtracts one before indexing the step list. The local port in
`spectral_utils/prmbench.py` already has this conversion. The cluster cache
writer explicitly preserves the original one-based annotations.

## Required repair

1. Preserve every old release, source, prediction, metric and failed audit.
   Record the failure and mark affected PRMB numbers as superseded. Do not
   edit the readonly Claude worktree or silently repair frozen sources.
2. Build new labels for all 6,969 cached PRMB answers. Match exact row IDs,
   step counts and classifications. Out-of-range annotations remain inert,
   following the official evaluator; empty error lists stay all-correct.
   Reconstruct the old buggy rule too, requiring exact old-label replay.
3. Issue `localization-cached-v3-prm-onebased-20260907`, retaining v2 source
   groups/folds, all PB labels and all telemetry. Include exact hashes and
   a reusable tested conversion API. Do not change graph permutation seeds.
4. Replay ALL 98 current110 and 25 original58 answer-only candidates with
   corrected PRMB targets, preserving every other per-row field exactly.
   Recompute pooled/within-answer PRMB metrics, while requiring both native
   and fixed-IU PB endpoints to remain exact. Compare old/new labels on the
   SAME scores; any change is an evaluation correction, not an algorithm gain.
5. Recompute the registered current Step312/Step310 and old58 comparisons
   under v3 labels and v2 source groups. Intervals stay exploratory, unadjusted,
   with 1,000 source-group draws. Keep all arms/registered comparisons rather
   than selecting an attractive subset after seeing the correction.
6. Review all raw labels against the existing official-metric port using
   all-valid model outputs (its wrong-step list provides the complement).
   Independently recompute pooled/within AUROC with pair comparisons and PB
   subset counts; check every unchanged prediction/score/group and a subset
   of explicit bootstrap resamples against sklearn AUC.

No new inference or fitting is needed for these answer-only score bridges.
Claude's multi-answer fitted/label-selected pipelines require corrected-label
and corrected-fold reruns: rescaling an old report cannot repair selection.
Other historical answer-only score sets need their own bridge before reuse.
The original text/peak forensics remains unfinished and must resume against
the new contract. The full fusion-centered research objective stays open.
