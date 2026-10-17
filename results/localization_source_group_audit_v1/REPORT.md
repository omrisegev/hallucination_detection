# A necessary benchmark correction: keep source questions together

The exposure audit found that different versions of the same question were assigned to different evaluation groups. We created a corrected release and folds, and recomputed uncertainty for the existing answer-only pilot. No fusion winner is established.

## What was wrong

PRMB source_idx identifies a perturbation record. Names such as confidence_prm_train_p1_7 and circular_prm_train_p1_7 can describe versions of the same source question. The previous grouping kept these names separate. ProcessBench also has different answer IDs with identical problem text.
These are groups of source questions used to evaluate experiments. Joint feature groups are a separate part of the fusion method; their fitted values are unchanged by this repair.

| Local cache | Answer rows | Old groups | Corrected source-question groups | Groups crossing old outer folds |
| --- | --- | --- | --- | --- |
| PRMB / Qwen3-8B | 6969 | 6211 | 707 | 707 |
| PB / each of Qwen3-4B and 8B | 3400 | 3400 | 2842 | 363 |

## Evidence from the actual saved data

PRMB contains 758 source-seed IDs. Exact question matches join some seeds into 707 components. There are 835 distinct question hashes occurring under multiple old groups.
Direct text evidence: 812 identical PRMB question hashes cross old outer folds. This observation does not depend on interpreting the source-seed suffix.
PB has 430 repeated-question groups covering 988 answers. 363 of those groups cross old folds. No identical PB question crosses its four subsets in these caches.
There are 66 exact whitespace-normalized question hashes shared by PB and PRMB. The corrected global folds keep them aligned across tasks.
The metadata reader skips NumPy payloads instead of constructing the large telemetry arrays. It reads IDs and question text from the nine local pickle files. Correctness fields exist in those containers but are not used to construct groups. All 3,400 PB question texts match exactly between the 4b and 8b caches.

Official schema: https://github.com/ssmisya/PRMBench#-data-format-for-prmbench . Counts are from the frozen local cache.

## What this means for the experiments

Claude v2 fitted and selected methods across multiple answers using the old folds. Every corrected PRMB component appears in more than one old outer fold. Those results do not establish performance on unseen source questions. The amount and direction of any score bias have not been measured.
Changing group names in an existing prediction file cannot repair those trained fits. We prepared corrected folds; the relevant multi-answer fits, configuration selections and predictions still need to be recomputed.
Our recent window experiments fit each answer independently. Their frozen scores and predictions remain valid for those answers. Their source-group uncertainty and disjointness claims require correction. Twelve PRMB pilot answers form 11 corrected groups, and two of these groups overlap earlier short-cycle cohorts. The 46 PB pilot answers remain 46 distinct groups.
The existing cache was already evaluated by v2. A new sample from it is development replication. Truly untouched publication confirmation still needs a separate exposure audit and data source.

## All 25 point-metric bundles are unchanged

The score bridge reuses the same 58 answers, all scores, targets, validity flags and predictions. Only the resampling group IDs change. The matched table still gives no clear winner: Joint -> IU has promising points over original IU, but context equal has higher headline points.

| Method | PRMB pooled AUC | Within-answer AUC | PB macro F1 |
| --- | --- | --- | --- |
| Original IU | 0.62261 | 0.67567 | 17.71% |
| Original Joint -> IU | 0.63663 | 0.68798 | 27.22% |
| Context equal fusion | 0.66087 | 0.66182 | 27.67% |

## Uncertainty recomputed under corrected source groups

All 32 bridge comparisons are complete: the 30 original registered pairs and the two previously added context-equal comparisons. The selected intervals below still include zero. These are retrospective, unadjusted 95% percentile intervals from 1,000 source-group draws.
The existing bootstrap draws PRMB before PB from one random generator. Changing the PRMB group count changes subsequent PB draws. Small PB interval changes here reflect that Monte Carlo effect; the 46 PB pilot groups themselves remain distinct.

| Comparison | Endpoint | Old interval | Corrected interval |
| --- | --- | --- | --- |
| single__joint0 minus moment__iu | PRMB AUC | [-0.0077, +0.0270] | [-0.0126, +0.0273] |
| single__joint0 minus moment__iu | PB percentage points | [-9.3750, +20.5011] | [-9.2116, +20.7192] |
| single__joint0 minus context__equal | PRMB AUC | [-0.0874, +0.0344] | [-0.0739, +0.0345] |
| single__joint0 minus context__equal | PB percentage points | [-16.8262, +11.7157] | [-18.7392, +10.7649] |
| dual__graph010 minus dual__joint0 | PRMB AUC | [-0.0071, +0.0183] | [-0.0082, +0.0179] |
| dual__graph010 minus dual__joint0 | PB percentage points | [-23.5655, +3.1486] | [-24.0292, +2.6510] |

## What has been delivered

A new immutable release, localization-cached-v2-sourcegroups-20260907, keeps the original rows and raw telemetry hashes and stores both corrected and legacy group IDs.
New five-outer / five-inner assignments preserve source-question isolation across all benchmark/model cells. They have been checked, but multi-answer methods have not yet been refitted on them.
The exposure inventory excludes 94 corrected components represented in the documented earlier Codex short cycles and 58-answer pilot. This is an exclusion list for further development, not a list of all exposure in the project.
No previous release, frozen score file, or Claude worktree was modified. The fixed-recipe replication prototype passed two tests, including replays of all 19 retained arms in three prior routing cases, but no new replication cohort or experiment has been launched.

## Independent review

A separate reviewer rebuilds source components with a sparse graph algorithm, checks every corrected release row and verifies fold isolation. It verifies the original telemetry and label file hashes, all nine metadata-source pickle hashes, and the unchanged 25 metric bundles.
Three corrected 1,000-draw comparisons are independently reproduced by explicitly resampling rows. All four endpoint intervals and defined-draw counts match. Two metadata-decoder tests compare the extracted strings with standard pickle across protocols 4 and 5, shared references and large binary frames.

| Check | Count |
| --- | --- |
| metadata_identity_rows | 10369 |
| source_pickle_hashes | 9 |
| release_rows_checked | 13769 |
| fold_isolation_checks | 30 |
| score_target_decision_row_replays | 58 |
| unchanged_independent_metric_bundles | 25 |

## Next action

Use the corrected identity map to freeze a small source-disjoint development replication. Retain IU, both equal-fusion banks, Joint with explicit IU fallback, and graph lambda-zero/permuted controls. The grouping correction must be applied at data loading before reusing any multi-answer fitting code.
Keep the planned rerun of relevant Claude contenders on corrected folds in the benchmark queue. Do not present the old grouped-CV numbers as source-question-disjoint evidence.
Claude's latest report also proposes minimum feature-group size two. That is a separate Joint identifiability question to audit before changing the existing minimum-three recipe. Joint features/graphs, IU improvements, temporal/geometry/sampling support, the wider comparator panel and the historical 24-cell transfer remain active research work.

- [Frozen repair protocol](../../docs/experiments/LOCALIZATION_SOURCE_GROUP_REPAIR_V2.md)
- [Corrected release](RELEASE_V2.json)
- [Corrected folds, not yet fitted](FOLDS_V2.json)
- [Audit evidence](AUDIT.json)
- [32 interval bridges](CONTRASTS_V2.json)
- [Independent review](REVIEW.json)
- [Previous fusion experiment](../fusion_explicit_fallback_pilot_v1/REPORT.html)
