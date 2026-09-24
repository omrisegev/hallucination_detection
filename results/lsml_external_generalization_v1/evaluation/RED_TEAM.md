# Independent red-team review: external L-SML comparison

Three independent agents reviewed raw per-response artifacts and code without reading the main metrics, contrasts or reports. A recomputed all metrics; B audited all records, hashes and masks; C recomputed observed scores, shuffled labels and checked estimator algebra. The root reconciled their evidence only after all predictions were sealed.

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| All 21 primary numbers are correct on the complete registered population. | confirmed | A and C exactly match all metrics (maximum difference 0); A confusion counts match all21 rows. B checks6190/6190 records,53970 steps,5399900 tokens. Pinned official evaluators agree for all21 rows. `RECONCILIATION.json`, `OFFICIAL_METRIC_REPLAY.json`. |
| Frozen L-SML improves both matched controls on Socratic, including backbone transfer. | confirmed | A independently reproduces both positive deltas per backbone; B verifies identical populations; C gains exceed both descriptive shuffle ranges and verifies weights/threshold algebra. Root corrected intervals are positive for allfour contrasts; observed-disjoint sensitivity agrees. `CONTRASTS.json`, `DISJOINT_CONTRASTS.json`, `independent_null/LABEL_NULL.json`. |
| Frozen L-SML beats the learned-partition equal control on Hard2Verify. | confirmed | Independently reproduced +3.912pp; corrected interval[+0.837,+7.058]pp. B200/200; C observed gain exceeds global and within-answer null ranges. |
| Frozen L-SML is conclusively better than ordinary equal on every external cell. | weakened | Hard2 gain+2.787pp has corrected interval[-0.222,+5.839]pp and lies inside the global-shuffle contrast range. Do not turn the point estimate into an across-benchmark universal claim. |
| Answer-local L-SML improves over its matched ordinary equal control on Socratic. | refuted | A/C reproduce -1.285pp Qwen3 and-1.087pp QwQ; both corrected intervals are negative. B confirms full/native masks and zero fallback decision differences for these ordinary-equal comparisons. |
| One feature perturbation proves the mechanism across the full external population. | weakened | C only perturbed36/6190 answers: FEASIBILITY. Entropy perturbation changes frozen scores36/36, native local scores34/34 and leaves CT7 unchanged, but this is bounded implementation sensitivity, not full-population quality ablation. |
| The disjoint panel proves no underlying problem contamination. | weakened | B exact-match/source-component audit covers all inputs and known development hashes, but metadata lacks a complete PRMB original-question crosswalk; no semantic/paraphrase/pretraining exclusion. |

All main comparisons use100000 paired source-question bootstrap draws, seed20260924, Bonferroni over18 primary contrasts. Label permutations (1000 global;200 within-answer) are diagnostic only; nonzero chance differences arise from differing prediction prevalence. The two Socratic backbones are the same question population, not independent datasets.

Independent evidence: `independent_external/INDEPENDENT_EXTERNAL.json`, `independent_coverage/AUDIT.json`, `independent_coverage/RED_TEAM_B.md`, `independent_null/RED_TEAM_C.md`, `independent_null/ACTUAL_MATH.json`.

Post-evaluation diagnostic checked independently by C: Hard2 frozen-L-SML observed per-answer decision-budget ceiling52.67257%, global-prevalence ceiling65.05190%, and41/42 entirely-correct answers flagged. This bounds the observed decision budgets only; it does not bound all thresholds or L-SML implementations. Socratic backbone disagreement1751/26055 and mean within-answer Spearman .9003189451 over2991 answers also replay exactly (`independent_null/DECISION_BUDGET_CHECK.json`).

Full internal comparison is complete. Reproducing published critic/PRM inference remains unfinished in the broader plan; literature numbers do not constitute reproduced or paired baselines.
