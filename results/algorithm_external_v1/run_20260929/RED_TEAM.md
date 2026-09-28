# Red team: algorithm_external_v1 run_20260929 (sealed 9beae6efe, evaluated 775ec15a7; protocol b3ef5c2e1 + A1)

- **A** recomputation from the sealed predictions and the labels with fresh code (all 15 levels and all claimed contrasts to 4
  decimals; seals verified against the pre-evaluation commit; 20,000-draw bootstrap with OVERLAP groups and with exact-text groups).
- **B** coverage: 69 arm-cell metrics, 40 category-cells per contrast, length and position panels, leave-one-category-out, equal
  predicted-correct share (threshold-transfer check), official vs within-AUC agreement, development-disjoint panel, refit vs frozen,
  stopping-rule decisions.
- **C** nulls: step-index baseline, same-length whole-answer label swap and within-answer shuffle (300 permutations), linear
  position-trend removal, before/after-error pair split, mechanism of the frozen bank11 L-SML.

Scripts in the session scratchpad (`rt_ext_A`, `rt_ext_B`, `rt_ext_C`).

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| 1. The frozen candidate (DS filter + plain average) on the 13+digit bank beats the earlier incumbent ct7 on both Socratic cells (+0.0446 / +0.0424 PRMScore, Bonferroni); also on 20+d and 51+d; not significant on Hard2Verify | **numbers confirmed; reading weakened: mostly position and partly threshold transfer** | **A**: exact; **B**: broad across categories (positive in 17-18/20, leave-one-out +0.040 to +0.049) but 15-25% of the gap comes from ct7's larger predicted-correct share (equal share: +0.038 / +0.032; Hard2Verify +0.0198 becomes -0.0079). **C**: within-AUC delta +0.0368 / +0.0270 vs same-length swap mean +0.0340 / +0.0204 - content share 8% / 24% (swap p 0.11 / 0.007); after removing each answer's linear position trend -0.002 / +0.009. The gain is a shift away from ct7's early-step preference. The two Socratic cells are the same answers and labels scored with two models (one dataset seen twice) |
| 2. The grouped estimate weighting helps on the 32+digit bank (+0.0323 / +0.0299 PRMScore on Socratic, Bonferroni; within-AUC +0.0425 / +0.0403 / +0.0329 incl. Hard2Verify), as in development | **confirmed; about half content** | **A/B**: Bonferroni above 0 on both Socratic cells (2/3, meets the frozen rule); positive in 18/20 categories, none significantly negative. **C**: content +0.021 to +0.023 above the swap mean (50-56%), +0.014 / +0.015 after trend removal; on Hard2Verify content 76% and positive on both pair types - the cleanest content gain found. It repairs the weakest bank: 32+d grouped is still below 13+d plain and below frozen bank11 L-SML |
| 3. The candidate ties the frozen bank11 L-SML on Socratic and loses on Hard2Verify (-0.0394); the larger banks are below it everywhere | **raw numbers confirmed; the tie is cancellation, and on content the candidate leads** | **B**: the Socratic tie cancels significant category losses (regather_redundancy about -0.06) and gains; the Hard2Verify loss is significant only at unadjusted 95% (Bonferroni [-0.084, +0.001]; removing 13 of 79 groups flips it). **C**: frozen L-SML's advantage is a late-step tilt (it silences energy_innovation, top15_turnover, top50_js, the channels that give the average its early-step preference): L-SML - bank11 average +0.0324 / +0.0361, about 45% positional by the swap null and all of it by trend removal; net of position the 13+d candidate leads L-SML by about +0.014 / +0.008 (p 0.0033) |
| 4. Step position | **new, decisive** | **C**: the step index alone ("later = riskier") has within-answer AUC 0.7307 on Socratic and 0.8649 on Hard2Verify, above every arm (best 0.7011 / 0.6836 / 0.6343). 918 of 2,987 Socratic answers are a clean prefix followed by an all-error suffix. The protocol did not include a step-index baseline row |
| 5. Bank order 16 > 23 > 54 > 35 | **Socratic only** | **B**: all 12 adjacent-pair intervals on Socratic in the claimed direction; Hard2Verify order differs and nothing there is significant |
| 6. Refit on the target is not better than the frozen fit | **confirmed with one exception** | **B**: refit worse in 12/24 (official) and 17/24 (within-AUC); the only threshold-robust exception is R_B23_GRP > F_B23_GRP on within-AUC (+0.019 / +0.005) |
| 7. Stopping rule | **never switched on** | **B**: S1 0.182-0.278 > tau* 0.1734 in 12/12 bank-cells; F_SW identical to F_BASE |
| Metric agreement | | **B**: F_B54_BASE - ct7 is Bonferroni-positive on PRMScore but negative on within-AUC (-0.011 / -0.008); official and within-AUC disagree in sign on 11/28 Hard2Verify and 4-5/28 Socratic contrasts |

## What the author's reading missed
- A step-position baseline row and a position null were not in the external protocol; on these benchmarks position alone beats every
  method, and most of the candidate's advantage over ct7 is position.
- Part of the official-metric gap to ct7 is threshold transfer (different predicted-correct shares), not ranking.
- "Both Socratic cells" are one dataset scored twice; "a tie with L-SML" hides category-level cancellation.
