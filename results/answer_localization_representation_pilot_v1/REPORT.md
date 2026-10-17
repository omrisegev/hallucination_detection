# Answer-localization representation pilot v1

2026-09-07. Retrospective development stress test; no winner promoted.

The short-window representation improves fit coverage, but the fixed mixture/first-crossing readout fails on ProcessBench. A graph benefit is not established. The next experiment should isolate the chronological readout while retaining these frozen feature scores and the Joint representation hypotheses.

## Frozen scope

58 answers: 12 PRMBench, 46 ProcessBench across four Qwen3-8B subsets. Selection used source-group hashes in three length bins, without labels. This deliberately stresses short and long traces; its aggregate is not a representative full-benchmark estimate. Every selected answer has a distinct registered source group within its cell. All data were already exposed in Claude v2. Qwen3-4B is registered for later replication.

Release: `localization-cached-v1-20260907`. Protocol: `docs/experiments/ANSWER_LOCALIZATION_REPRESENTATION_PILOT_V1.md`. Four representations separate historical borrowed signs, within-answer signs, the original 30 measurements, a 27-coordinate primitive-moment bank, and widths 32/8. The answer-local sign convention retains a declared negative-entropy anchor; it is not anchor-free. Equal, canonical IU and Joint lambda-zero are included, with meaningful/permuted graphs on the local-sign lanes.

## Main findings

- IU/equal coverage increases from 38/58 at width 32 to 58/58 with moments at width 8. Joint coverage is 32/58 in the legacy recipe and 43/58 in moments-8. The new moment bank at width 32 has only 27/58 valid Joint fits; a new feature bank alone did not fix every partition.
- Moments-8 selects K=4 in 14 answers and K=3 in 29. The global-30 local-sign bank also selects K=4 and K=6 in some answers. This refutes a universal 'always three groups' reading of the earlier two-fold diagnosis, but is not an accuracy result.
- On the seven common valid PRMB answers for moments-8 Joint/IU: meaningful graph AUROC 0.66255 versus IU 0.64964, difference +0.01291, CI [-0.03016,+0.04864]. Graph versus lambda zero and permutation is unresolved. These very small common subsets cannot establish a method advantage.
- The BIC-selected mixture followed by the first threshold-crossing step gives PB macro-F1 zero for all 18 fusion arms; the entropy control gives 0.08333. This is a failure of these full pilot pipelines. It is not evidence that all underlying risk rankings are useless.
- For moments-8 IU, 26/46 PB predictions are step 0, 19 are no-error and one is step 1; none hits the true first error. In a post-hoc error-only diagnostic, argmax risk hits 7/25 erroneous answers. That is not a replacement result or a validated alternative readout; it shows why the choice of locator must be isolated.

## Descriptive table: differing availability is explicit

Do not rank this entire table as a common-population leaderboard. PRMB AUROC uses the available valid answers listed; the paired table below uses shared IDs. PB uses all 46 selected answers and counts unavailable predictions as misses in either class.

| Arm | Valid fits / 58 | PRMB answers | PRMB AUROC | PB macro-F1 |
|---|---:|---:|---:|---:|
| `legacy_fixed32__equal` | 38 | 8 | 0.64855 | 0.00000 |
| `legacy_fixed32__iu` | 38 | 8 | 0.61997 | 0.00000 |
| `legacy_fixed32__joint_lambda0` | 32 | 6 | 0.48493 | 0.00000 |
| `global30_local32__equal` | 38 | 8 | 0.63742 | 0.00000 |
| `global30_local32__iu` | 38 | 8 | 0.62263 | 0.00000 |
| `global30_local32__joint_lambda0` | 32 | 6 | 0.48493 | 0.00000 |
| `global30_local32__joint_graph010` | 32 | 6 | 0.49058 | 0.00000 |
| `global30_local32__joint_graph_permuted` | 32 | 6 | 0.47834 | 0.00000 |
| `moments27_local32__equal` | 38 | 8 | 0.61864 | 0.00000 |
| `moments27_local32__iu` | 38 | 8 | 0.61698 | 0.00000 |
| `moments27_local32__joint_lambda0` | 27 | 4 | 0.74320 | 0.00000 |
| `moments27_local32__joint_graph010` | 27 | 4 | 0.73639 | 0.00000 |
| `moments27_local32__joint_graph_permuted` | 27 | 4 | 0.74660 | 0.00000 |
| `moments27_local8__equal` | 58 | 12 | 0.61717 | 0.00000 |
| `moments27_local8__iu` | 58 | 12 | 0.62261 | 0.00000 |
| `moments27_local8__joint_lambda0` | 43 | 7 | 0.66171 | 0.00000 |
| `moments27_local8__joint_graph010` | 43 | 7 | 0.66255 | 0.00000 |
| `moments27_local8__joint_graph_permuted` | 43 | 7 | 0.66115 | 0.00000 |
| `entropy_mean_w8` | 58 | 12 | 0.62587 | 0.08333 |

## Paired PRMB contrasts

Intervals are exploratory, unadjusted source-group bootstraps, 1,000 draws. Common-cohort counts are small. The PB intervals [0,0] for two failed zero-hit pipelines are degenerate resampling results, not proof of population equivalence.

| Left minus right | Common PRMB answers | Observed AUROC difference | 95% interval |
|---|---:|---:|---:|
| `moments27_local8__iu_minus_legacy_fixed32__iu` | 8 | -0.01396 | [-0.08985, 0.1444] |
| `global30_local32__iu_minus_legacy_fixed32__iu` | 8 | 0.00266 | [-0.01824, 0.04546] |
| `moments27_local32__iu_minus_global30_local32__iu` | 8 | -0.00565 | [-0.07302, 0.0701] |
| `moments27_local8__iu_minus_moments27_local32__iu` | 8 | -0.01097 | [-0.06823, 0.08542] |
| `moments27_local8__iu_minus_moments27_local8__equal` | 12 | 0.00543 | [-0.02491, 0.04289] |
| `moments27_local8__iu_minus_entropy_mean_w8` | 12 | -0.00326 | [-0.0524, 0.08318] |
| `global30_local32__joint_graph010_minus_global30_local32__joint_lambda0` | 6 | 0.00565 | [-0.00941, 0.03299] |
| `global30_local32__joint_graph010_minus_global30_local32__joint_graph_permuted` | 6 | 0.01224 | [0.0, 0.04032] |
| `global30_local32__joint_graph010_minus_global30_local32__iu` | 6 | -0.01412 | [-0.06178, 0.02119] |
| `moments27_local32__joint_graph010_minus_moments27_local32__joint_lambda0` | 4 | -0.00680 | [-0.05, 0.01172] |
| `moments27_local32__joint_graph010_minus_moments27_local32__joint_graph_permuted` | 4 | -0.01020 | [-0.05, 0.0] |
| `moments27_local32__joint_graph010_minus_moments27_local32__iu` | 4 | -0.04082 | [-0.125, 0.0] |
| `moments27_local8__joint_graph010_minus_moments27_local8__joint_lambda0` | 7 | 0.00084 | [-0.02305, 0.01701] |
| `moments27_local8__joint_graph010_minus_moments27_local8__joint_graph_permuted` | 7 | 0.00140 | [-0.02974, 0.01998] |
| `moments27_local8__joint_graph010_minus_moments27_local8__iu` | 7 | 0.01291 | [-0.03016, 0.04864] |

## Review and historical continuity

Five scientific-contract tests passed before freezing the run. Independent projection and span replay passed for 583 maps, with maximum projection error 0. Both benchmark endpoints reproduce for all 19 arms. Frozen code and all 116 score/metadata hashes match. Direct source-label rejoins by ID match all 58 evaluated answers, and the 29 raw column names/order match the upstream schema. `REVIEW.json` records these checks; the audit does not prove that mixture states are correctness states.

Scoring finished in about 206 seconds using three CPU workers, with per-answer checkpoints. Labels were decoded only in the later evaluation phase. The original pilots, Claude's worktree and prior frozen benchmark releases were preserved.

The historical 30-long-answer pilot's IU 0.70070 and common-24 graph 0.69188 are on different IDs from this stress test; the new cohort does not retroactively invalidate them or provide a direct delta. The historical recipe is recomputed on the new selected IDs as `legacy_fixed32`. Claude's pooled/calibrated PB numbers use another no-error protocol and belong in a separate lane. The 24-cell final-answer benchmark is a different target and remains a later frozen-candidate transfer experiment.

## Next bounded work

Freeze a chronological readout comparison over the saved window scores: retain this failed first-crossing baseline, include a simple peak locator and a fixed no-error control, then compare HMM/BOCPD and an explicit IMM/switching-filter adaptation. Keep feature-fusion weights fixed so the contribution is measurable. Do not label a Gaussian HMM as an IMM reproduction. Preserve KalmanNet, LOCA, Diverging Flows and graph token/window sampling in their separate registered follow-ups. Joint feature/hyperparameter work remains active, with no higher-K forcing or label-guided parameter selection.

A publication winner still requires full matched benchmark coverage, a declared selection procedure and genuinely untouched confirmation on both tasks. This pilot has not achieved that objective.
