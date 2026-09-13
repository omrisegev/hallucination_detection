# Temporal fusion development: integration handoff

Snapshot: 2026-09-14. Integration base:
`codex/conditional-iu-followups-v1`.

This branch already contains the following development lineage. The three
earlier branches are ancestors, not independent changes to merge again.

| Branch | Recorded commit | Scope |
|---|---|---|
| codex/rbm-hierarchical-time-v1 | 2ec11dd0cd84012033b636f6134141677dc72a2c | Frozen RBM12 token logits, local/shared/hierarchical time weights |
| codex/rbm-two-axis-fusion-v1 | 6249db384dc453cea54838079aa6a42dbeb05be6 | Initial within-step two-axis implementation; smoke only, later position amendment |
| codex/answer-position-fusion-v1 | 48138140814072900c63a2da6a6c4dcf48369bb7 | Whole-answer position; Gaussian factor and H1 RBM loading-map rank1/2; regional IU |
| codex/conditional-iu-followups-v1 | dc12221d2 | AIRCC scientific/package snapshot for position prior, graph-local and coefficient GraphTV |

The metadata commit f3fa3cc34 was previously pushed. This handoff and the fetched
result reports are the subsequent update. See the branch HEAD for the new
integration commit; no scientific training/scoring source changed in this update.

## Completed results available in Git

`results/aircc_results_20260913/` holds byte-exact compact original reports,
manifests, review records, per-cell tables, contrasts, fit health and interpretation:
- `answer_position/`: job255722 COMPLETE_REVIEWED.
- `conditional_position/`: job255752 COMPLETE_REVIEWED.
- `graph_local/`: job255753 COMPLETE_REVIEWED.

The fetched files passed remote SHA256 comparison; all PB macros were locally
recomputed from their eight cells. Large raw caches, SQLite checkpoints and NPZ
scores are not in Git. These results are full development evidence, not untouched
confirmation. See FETCH_REVIEW.json for the actual verification scope.

## GraphTV is still running

At 2026-09-14 00:01 Israel time, job255754 was RUNNING, SCORING_TOP10:
4,525 / 13,769 answers checkpointed. Full final results are NOT available yet.
Its source and protocol are already included in this branch.

Remaining stages:
1. Complete all answers' outer and PRMB inner-calibration predictions, including
   real-graph and permuted-graph coefficient optimization.
2. Check coverage, fit health and fold exclusions.
3. Compute held-fold PRMScore q0.8 thresholds and all benchmark metrics.
4. Run the registered 10,000-draw source-group bootstrap and endpoint audit.
5. Emit COMPLETE_REVIEWED, fetch and interpret final reports.

There is no missing implementation or pending approval blocking this job.
An answer checkpoint includes its required outer/inner scores together.
Do not estimate the remaining runtime by multiplying only a single last-answer
duration: answer lengths and PRMB calibration work vary.

AIRCC result root:
`/shared/cycle2_tau_averbuch_prj/omrisegev1/experiments/conditional_iu_20260913_dc12221d2/code/results/conditional_iu_fusion_v1/graph_tv/`.

## Starting a later combined experiment

From the second computer's main repository:
```powershell
git fetch origin
git worktree add -b codex/combined-fusion-v1 ..\combined-fusion-v1 origin/codex/conditional-iu-followups-v1
```

This is a suggested future branch name; the combined experiment has NOT been
created or launched here. Start from this descendant to retain the full lineage,
then integrate only the separately selected branches. Record their exact commits
and inspect conflicts in scorer, normalization, orientation, readout and grouping.
Do not blindly merge all historical worktrees or resolve research records with
ours/theirs; preserve both lines of history.

Portability/data instructions:
`docs/research_notes/CONDITIONAL_IU_SECOND_MACHINE.md`.
Before a run, validate all frozen data using
`scripts/check_conditional_iu_inputs.py --source-root <main-data-repository>`.
The new worktree reads those inputs; Git does not supply the large caches.

Keep v3 labels, v2 source-group folds, original entropy q0.3 gate, token-to-step
mapping, Top10 and earliest argmax, and PRMScore nested q0.8 calibration matched.
Keep answer-local and other-answer unlabeled access explicit. Give a combined
recipe a new manifest/output directory; do not modify the current AIRCC experiment.

The main repository's separate token-local optimization branch and Claude's
parallel branches are not silently merged or committed by this handoff.
Conditional-context RBM and Temporal Convolution remain separate future work.
