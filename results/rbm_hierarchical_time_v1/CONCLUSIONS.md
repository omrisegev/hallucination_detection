
## Codex hierarchical time fusion v1 ? 2026-09-13 ? COMPLETE REVIEWED

Full 13,769-answer run completed in 1,792.85 seconds (about 30 minutes, excluding implementation and preflight). All scoring and nested calibration covered the registered population. Independent PB, within-answer AUC and official PRMScore reconstruction PASS; 13 frozen reference rows reproduced; four additional DUFS/shared-variance reference rows re-evaluated against matching contract hashes. No numerical failures or nonconverged selected fits; the one-step answer uses the declared rules. 55 unique training exclusion sets and 41,645 answer/exclusion records were checked. Current-answer RBM12 coefficients never changed.

| Frozen RBM12 readout / time weights | PB all8 % | PRMB within AUC | PRMScore |
|---|---:|---:|---:|
| Original Top10 | 36.271 | 0.74520 | 0.62222 |
| All-token mean | 29.429 | 0.66050 | 0.56348 |
| Best contiguous 10 | 31.653 | 0.70476 | 0.60425 |
| Local time | 28.142 | 0.64238 | 0.57764 |
| Shared time | 29.728 | 0.66991 | 0.57134 |
| Hierarchical time | 29.385 | 0.66454 | 0.57272 |
| Shuffled hierarchical time | 29.190 | 0.65985 | 0.56285 |
| Supervised positive time weights | 30.242 | 0.68101 | 0.59483 |

Primary, 10,000 paired source-group draws, 97.5% CI: shared-local PB +1.586 pp [0.044, 3.115], within-AUC +0.02753 [0.02290, 0.03204], but PRMScore -0.00631 [-0.01013, -0.00251]. Hierarchical-shared PB -0.343 pp [-1.260, 0.557], within-AUC -0.00537 [-0.00740, -0.00330]; PRMScore +0.00138 has an interval including zero. Sharing data helps relative to local covariance fitting on two endpoints, not across every metric. Local adaptation supplies no overall gain.

The hierarchy loses to original Top10 in every PB cell: it gains 281 exact successes and loses 608 (269 early, 339 late; zero gate or computation losses). Top10-to-all-mean alone loses 6.842 PB points before fitting any time weights. This isolates a substantial loss from replacing score-adaptive selection with averaging. Supervised fixed positive time weights do not recover it; this diagnostic is not a supervised ceiling for other architectures.

Average shared mass over successive step quarters: 12.6%, 21.5%, 29.7%, 36.2% (uniform: 25% each). Hierarchy: 12.7%, 21.9%, 29.4%, 36.1%. The median answer-level L1 change from shared weights is 0.258, despite similar aggregate profiles. Shuffling yields nearly uniform quarters. Descriptive original-order versus shuffled hierarchy within-AUC +0.00470 [0.00238, 0.00700], PB +0.195 pp [-0.972, 1.348]: evidence for modest within-answer ordering information, not a PB gain.

Boundary diagnostics are not promoted: Top10 without the first token gives PB 36.360% (difference not clear), within-AUC 0.74387 (slightly worse), PRMScore 0.62747 (better). Removing the first token from hierarchy gives 29.546% PB; it does not close the gap. Regions do not create observations: 22,333 of 145,597 steps have fewer than 16 tokens; 91 have one token. Median step length is 31 tokens; median answer has eight steps and alpha=0.304.

Decision: retain original RBM12 Logit + Top10 as this temporal experiment's reference. Do not adopt the tested positive fixed-position averaging hierarchy. More data stabilizes temporal weights but cannot by itself recover information discarded by the readout. A next family should preserve score-adaptive high-risk token selection while isolating added temporal/context information. No tensor, conditional RBM, convolution or new feature experiment was started; discussion is required before the next family. Findings remain development evidence, not untouched confirmation. Historical low-correlation RBM6 and Varentropy50 remain explicit competing references, not replaced by this run.

Artifacts: results/rbm_hierarchical_time_v1/{REPORT.md,COMPARISON.csv,PER_CELL.csv,METRICS.json,CONTRASTS.json,CONTROL_CONTRASTS.json,PRMSCORE_CONTRASTS.json,WEIGHTS.csv,WEIGHT_SUMMARY.json,ANSWER_DIAGNOSTICS.csv,RESULT_REVIEW.json,POSTFIT_REVIEW.json}. Protocol: docs/experiments/RBM_HIERARCHICAL_TIME_V1.md. Preflight archives preserve the JSON-reader/audit repairs; PROFILE_REUSE_REVIEW.json verifies the unchanged extraction and inputs. No original source, Claude run, or historical result was modified. No HTML was produced.
