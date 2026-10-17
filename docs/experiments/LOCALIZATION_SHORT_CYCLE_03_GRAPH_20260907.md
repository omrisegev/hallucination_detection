# Localization short cycle 3 — answer-only Joint-LIU graph test

Date: 2026-09-07. Status: registered before fitting the graph arms.

## User question

Does the graph-regularized Joint variant that looked strongest in Claude's
PRMBench development experiment work better when every fit is learned from one
long answer's windows?

## Exact candidate and controls

Reuse the 30-answer, width-32, non-overlapping-window contract and frozen
feature matrices from short cycle 1. For every answer with an available
internal Joint partition:

- replay `joint_modelinv_lam0` exactly as the lambda-zero reference;
- fit `internal_joint_liu010_answer_only`, the Hook-3a model-inverse with
  lambda 0.1, adapted-DUFS soft feature gates, and a self-tuning k=7 graph over
  that answer's fitting windows;
- fit `permctl_graph_internal_joint_liu010_answer_only` with the same gates,
  factor model, lambda, graph spectrum and edge-weight multiset, but a fixed
  node relabeling.

The graph candidate and control use only the current answer's unlabeled window
matrix. They do not introduce fixed/external feature groups. Answers for which
short cycle 1 had no admissible internal partition remain unavailable; there is
no provenance fallback.

Keep IU and equal from short cycle 1 as references. Keep the same per-answer
standardization/orientation and span-maximum official-step readout. Freeze new
scores before opening labels. Do not test other lambdas, graphs, window widths,
trajectory reducers, thresholds or pooled fits.

## Required checks

1. The recomputed lambda-zero window score must reproduce the frozen
   short-cycle-1 score before either graph score is admitted.
2. Report graph-versus-lambda-zero weight cosine and score Spearman, adapted-
   DUFS gate stability, graph connectivity, strict fit coverage and failures.
3. Evaluate paired deltas on strict common answers and bootstrap whole answers.
4. Attribute an improvement to graph structure only if the meaningful graph
   beats lambda zero and the node-permuted graph does not reproduce that gain.

## Interpretation

This is an adaptive retrospective diagnostic on the same 30 answers, not fresh
confirmation. A positive result can justify a fresh-subset graph confirmation.
If the meaningful graph does not beat both lambda zero and its permutation
control, stop the answer-only Joint graph line and continue with IU/equal on
the feature axis plus a separate trajectory-axis short cycle.

