# A one-step label error changes our PRMBench comparison

Step313. Corrected labels; exactly the same fusion scores. Review PASS.

6,035/6,969 cached answers have changed target arrays; 15,147/94,203 step labels change. The old rule omitted valid final-step errors in some answers; 227 previously all-correct arrays now contain an error. All eight PB label files are unchanged.

The raw annotations count steps from 1. Claude v2 wrote flags[step] instead of flags[step-1]. Recent Codex runs inherited that NPZ. The official evaluator and our existing port subtract one. Earlier review checked the derived NPZ, which missed this contract error.

Official source: [PRMBench task evaluator](https://github.com/ssmisya/PRMBench/blob/main/mr_eval/tasks/prmtest_classified/task.py).

## Current110 — same 24 PRMB and 86 PB answers

| Method | Previous PRMB (superseded) | Corrected PRMB | Corrected within-answer AUC | PB F1 unchanged |
|---|---:|---:|---:|---:|
| dual__equal | 0.62649 | 0.68298 | 0.78407 | 25.55% |
| dual__iu | 0.63797 | 0.68131 | 0.76881 | 30.16% |
| dual__cond100 | 0.63296 | 0.65381 | 0.75465 | 25.47% |
| dual__cond100_graph010 | 0.63847 | 0.65545 | 0.75348 | 30.22% |
| dual__equal_graph_perm | 0.62170 | 0.69226 | 0.78360 | 31.32% |
| ar1__iu | 0.61334 | 0.66220 | 0.71800 | 23.50% |
| ar1__graph010 | 0.62355 | 0.64714 | 0.67710 | 18.01% |
| last__equal_graph_perm | 0.62473 | 0.69733 | 0.77897 | 25.45% |

## What changes in the research conclusion

- Joint graph100 is no longer nearly tied with IU on PRMB: 0.65545 versus 0.68131. Their paired difference interval still includes zero; no proven two-task winner.
- The original permuted-graph equal-weight control reaches 0.69226 / 31.32%, above IU on both point estimates. This is a control, not evidence that meaningful graph structure or learned Joint weights caused a gain. Do not hide it or promote a label-selected winner.
- The old claim that all21 augmented recipes trail both anchors on both points is withdrawn: 20/21 trail IU and 9/21 trail graph100. All21 still have lower PB than both anchors. Last+equal+permuted graph has PRMB0.69733 but PB25.45%.
- AR+IU PRMB regression is no longer resolved by its exploratory interval. The AR+Joint-graph PB regression remains because PB is unchanged. Fit-validity and group-count observations remain valid.
- PRMB pooled AUROC remains a project endpoint, not the official PRMB category score. All123 method/cohort bundles and 207 registered comparisons are retained. Two cohorts stay separate; this is adaptive development, not untouched confirmation.

## Next action

Use RELEASE_V3.json for every new label read, while retaining v2 source groups/folds and the original graph seed namespace. Resume the interrupted raw-text/peak audit on this contract, then choose a bounded fusion/readout experiment. Other score sets need bridges before comparison. Claude multi-answer fitting/selection needs new labels AND new source-group folds; this bridge cannot repair already trained or selected pipelines. Full comparator coverage, named auxiliary methods, untouched confirmation and historical24 transfer remain open.

## Review

Five contract tests. Direct raw-pickle and official-port checks on all6969 rows, eight PB file identities, 168 unchanged row-field bundles, 11594 score/prediction replays, 123 independent metric bundles and 207 paired point/scope checks. Five explicit1000-draw bootstrap checks. Same-session review with shared metadata reader/official port; no external reviewer.

The original forensics run deliberately stopped on the mismatch. Its six geometry tests and 110-row raw extraction completed; its full alignment/peak AUDIT did not. No complete alignment verdict is claimed.
