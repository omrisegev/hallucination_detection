# Fixed fusion recipes on additional source questions

2026-09-07. Development replication; independent review PASS.


## What we learned

The earlier Joint0 -> IU advantage did not recur on this additional cohort. Dual-bank IU has encouraging point estimates, but it does not establish a consistent advantage on both benchmarks or over its matched equal-fusion control. Joint with graph lambda 0.1 also improves over its zero-graph version at the point-estimate level. Its PB interval against the permuted graph excludes zero, but the zero-graph and IU comparisons do not establish a two-task advantage. No winner is promoted.


## Our method is still fusion

We develop IU-PCR and Joint L-SML on the N windows x P features matrix of one answer. Representation, grouping, graph penalties, sampling and temporal models serve this fusion. Any addition must beat the same fusion without it and be compared with simple aggregation using the same addition. This stage changes the development cohort, not the 19 fixed recipes.


## Exactly what was fixed

Both banks have nine primitive telemetry streams and 27 window features. Moment uses mean, SD and slope; context uses level and EMA8/EMA32. Full fitting windows have eight tokens; a final end-anchored full window is scored when needed. Groups use K={3,4,6,8}, minimum size three and the existing stability/validity checks. The native inverse has condition target 1000; graph lambda is 0.1 with zero and node-permutation controls. Normalization, signs, gates, groups and fusion weights are fitted within each answer, with the declared fixed negative-entropy anchor. The peak and no-error rule are unchanged. This is offline processing of a full answer, not a claim of causal online detection.


## Read the method names

Routing is based on fit validity, never correctness labels, peak scores or the no-error decision. A selected readout failure stays a failure. Dual IU is a fixed control defined before scoring; its bank rule still requires Joint eligibility calculations and must not be described as the cost of plain IU.

| Name | Meaning |
|---|---|
| moment / context | The feature bank; these are alternative inputs to the same fusion family. |
| equal / iu | Equal-weight fusion or IU-PCR on that bank. |
| joint0 / graph010 / graph_perm | Joint L-SML with lambda 0 / graph lambda 0.1 / permuted graph lambda 0.1. |
| single | Use moment Joint when valid; otherwise moment IU. |
| dual Joint variants | Use moment Joint, then context Joint if needed, then moment IU. |
| dual__iu / dual__equal | Use context only when moment Joint is invalid and context Joint is valid; otherwise moment. Apply IU or equal fusion on the selected bank. |
| fixed-IU gate | Diagnostic: retain each arm's location peak, but use the original moment-IU binary error/no-error decision. |


## 110 different source-question groups

24 PRMB answers and 86 ProcessBench answers were selected without labels or fit outcomes, with at most eight groups per length bin and no filling of shortages. All 110 corrected groups are distinct and excluded from the documented 94-component Codex pilot inventory. This inventory is not a complete project exposure history. Claude already evaluated the whole cache, so this is development replication, not untouched publication confirmation. The sample stresses length ranges; it is not a prevalence-representative random sample. The corrected release is localization-cached-v2-sourcegroups-20260907; the v1 scoring namespace is retained to preserve the fixed graph-permutation seeds.

| Cell | Tokens in answer | Eligible groups at bin entry | Selected | Quota shortfall |
|---|---|---|---|---|
| prmbench_qwen3_8b | [64, 255] | 462 | 8 | 0 |
| prmbench_qwen3_8b | [256, 1023] | 501 | 8 | 0 |
| prmbench_qwen3_8b | [1024, 2048] | 29 | 8 | 0 |
| pb_gsm8k_q8 | [64, 255] | 174 | 8 | 0 |
| pb_gsm8k_q8 | [256, 1023] | 200 | 8 | 0 |
| pb_gsm8k_q8 | [1024, 2048] | 0 | 0 | 8 |
| pb_math_q8 | [64, 255] | 184 | 8 | 0 |
| pb_math_q8 | [256, 1023] | 693 | 8 | 0 |
| pb_math_q8 | [1024, 2048] | 76 | 8 | 0 |
| pb_olympiadbench_q8 | [64, 255] | 6 | 6 | 2 |
| pb_olympiadbench_q8 | [256, 1023] | 466 | 8 | 0 |
| pb_olympiadbench_q8 | [1024, 2048] | 162 | 8 | 0 |
| pb_omnimath_q8 | [64, 255] | 35 | 8 | 0 |
| pb_omnimath_q8 | [256, 1023] | 721 | 8 | 0 |
| pb_omnimath_q8 | [1024, 2048] | 198 | 8 | 0 |


## Matched results: this 110-answer cohort

Higher is better. PRMB pooled AUC ranks steps across answers; within-answer AUC measures local ranking only on answers with both labels. Full-coverage arms have 16 such PRMB answers. PB is the macro of four subset-level harmonic means of clean-answer accuracy and exact first-error accuracy. All 86 PB answers remain in the denominator, with invalid decisions counted as failures. Pure Joint PRMB rows use different valid populations; use the common-ID contrasts below to compare them. All valid fits in this run also have valid native and fixed-IU decisions.

| Arm | Fit coverage | Valid PRMB | PRMB AUC | Within-answer AUC | PB native F1 | PB fixed-IU gate (diagnostic) |
|---|---|---|---|---|---|---|
| moment__equal | 110/110 | 24/24 | 0.59530 | 0.67342 | 20.70% | 26.38% |
| moment__iu | 110/110 | 24/24 | 0.60140 | 0.67110 | 26.38% | 26.38% |
| moment__joint0 | 78/110 | 17/24 | 0.58642 | 0.59999 | 18.24% | 21.89% |
| moment__graph010 | 78/110 | 17/24 | 0.59145 | 0.62194 | 21.45% | 22.07% |
| moment__graph_perm | 78/110 | 17/24 | 0.58910 | 0.57738 | 17.10% | 20.75% |
| context__equal | 110/110 | 24/24 | 0.66971 | 0.66964 | 13.51% | 26.89% |
| context__iu | 110/110 | 24/24 | 0.67378 | 0.66898 | 18.39% | 29.25% |
| context__joint0 | 102/110 | 21/24 | 0.68397 | 0.69365 | 15.68% | 24.39% |
| context__graph010 | 102/110 | 21/24 | 0.68086 | 0.69540 | 17.88% | 24.88% |
| context__graph_perm | 102/110 | 21/24 | 0.68592 | 0.69365 | 18.40% | 24.88% |
| entropy_parent | 110/110 | 24/24 | 0.60646 | 0.62044 | 14.88% | 20.70% |
| single__joint0 | 110/110 | 24/24 | 0.59733 | 0.62674 | 21.45% | 26.60% |
| single__graph010 | 110/110 | 24/24 | 0.60090 | 0.64458 | 24.13% | 25.69% |
| single__graph_perm | 110/110 | 24/24 | 0.59792 | 0.60837 | 19.42% | 24.57% |
| dual__joint0 | 110/110 | 24/24 | 0.62884 | 0.63034 | 25.33% | 29.25% |
| dual__graph010 | 110/110 | 24/24 | 0.63350 | 0.64817 | 29.94% | 29.53% |
| dual__graph_perm | 110/110 | 24/24 | 0.63015 | 0.61197 | 25.73% | 28.41% |
| dual__equal | 110/110 | 24/24 | 0.62649 | 0.67359 | 25.55% | 29.35% |
| dual__iu | 110/110 | 24/24 | 0.63797 | 0.67470 | 30.16% | 30.37% |


## History stays visible: two separate cohorts

The earlier cohort has 58 answers: 12 PRMB (11 corrected source groups) and 46 PB. The new cohort has 110 answers: 24 PRMB and 86 PB. These columns are historical context, not a paired cross-cohort comparison. All 19 old metric bundles were checked against the corrected Step 305 bridge. The higher or lower score of a recipe on different questions is not an algorithmic improvement. Expanded-K arms remain in the previous report and were not included in this fixed legacy-K replication.

| Arm | Old valid PRMB | Old PRMB AUC | Old PB F1 | New valid PRMB | New PRMB AUC | New PB F1 |
|---|---|---|---|---|---|---|
| moment__equal | 12 | 0.61717 | 17.76% | 24 | 0.59530 | 20.70% |
| moment__iu | 12 | 0.62261 | 17.71% | 24 | 0.60140 | 26.38% |
| moment__joint0 | 7 | 0.66171 | 12.50% | 17 | 0.58642 | 18.24% |
| moment__graph010 | 7 | 0.66255 | 8.33% | 17 | 0.59145 | 21.45% |
| moment__graph_perm | 7 | 0.66115 | 4.17% | 17 | 0.58910 | 17.10% |
| context__equal | 12 | 0.66087 | 27.67% | 24 | 0.66971 | 13.51% |
| context__iu | 12 | 0.65652 | 16.97% | 24 | 0.67378 | 18.39% |
| context__joint0 | 9 | 0.63826 | 27.43% | 21 | 0.68397 | 15.68% |
| context__graph010 | 9 | 0.64297 | 20.19% | 21 | 0.68086 | 17.88% |
| context__graph_perm | 9 | 0.64101 | 19.10% | 21 | 0.68592 | 18.40% |
| entropy_parent | 12 | 0.62587 | 19.85% | 24 | 0.60646 | 14.88% |
| single__joint0 | 12 | 0.63663 | 27.22% | 24 | 0.59733 | 21.45% |
| single__graph010 | 12 | 0.63478 | 14.94% | 24 | 0.60090 | 24.13% |
| single__graph_perm | 12 | 0.63707 | 15.82% | 24 | 0.59792 | 19.42% |
| dual__joint0 | 12 | 0.62859 | 27.22% | 24 | 0.62884 | 25.33% |
| dual__graph010 | 12 | 0.63217 | 14.94% | 24 | 0.63350 | 29.94% |
| dual__graph_perm | 12 | 0.63163 | 15.82% | 24 | 0.63015 | 25.73% |
| dual__equal | 12 | 0.62435 | 17.76% | 24 | 0.62649 | 25.55% |
| dual__iu | 12 | 0.62587 | 17.71% | 24 | 0.63797 | 30.16% |


## 38 registered paired comparisons

Intervals use 1,000 source-group bootstrap draws, stratified by cell. These are exploratory, unadjusted 95% intervals across many comparisons. Undefined draws remain excluded and counted; most full-coverage PB comparisons have 999 defined draws, because one draw loses a required class in a subset. All four intervals, including within-answer AUC and the fixed-IU diagnostic, are stored in CONTRASTS.json. The highlighted graph-permutation result is one development signal; it is not confirmation of graph superiority over Joint0 or IU.

| Left minus right | Common valid PRMB | Delta AUC | AUC CI | Delta PB (pp) | PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| dual__joint0 minus single__joint0 | 24 | +0.03151 | [+0.00577, +0.06422] | +3.87833 | [-1.50357, +10.30349] | 999 |
| dual__joint0 minus moment__iu | 24 | +0.02744 | [-0.02202, +0.06368] | -1.04934 | [-11.35113, +9.95808] | 999 |
| dual__joint0 minus dual__equal | 24 | +0.00235 | [-0.04248, +0.03215] | -0.22036 | [-9.49442, +9.68112] | 999 |
| dual__joint0 minus dual__iu | 24 | -0.00913 | [-0.05057, +0.01758] | -4.82807 | [-13.73805, +3.28370] | 999 |
| single__joint0 minus moment__iu | 24 | -0.00407 | [-0.05195, +0.02417] | -4.92767 | [-14.69726, +3.77607] | 999 |
| single__joint0 minus moment__equal | 24 | +0.00203 | [-0.05070, +0.03451] | +0.75009 | [-8.63748, +10.67593] | 999 |
| dual__graph010 minus single__graph010 | 24 | +0.03259 | [+0.00845, +0.06504] | +5.80776 | [-0.54683, +13.43836] | 999 |
| dual__graph010 minus moment__iu | 24 | +0.03210 | [-0.00918, +0.06687] | +3.55603 | [-6.51980, +14.54138] | 999 |
| dual__graph010 minus dual__equal | 24 | +0.00701 | [-0.02841, +0.03326] | +4.38501 | [-5.49963, +15.26508] | 999 |
| dual__graph010 minus dual__iu | 24 | -0.00448 | [-0.03844, +0.01923] | -0.22271 | [-8.45421, +8.59431] | 999 |
| single__graph010 minus moment__iu | 24 | -0.00050 | [-0.04122, +0.02587] | -2.25173 | [-10.99806, +6.18099] | 999 |
| single__graph010 minus moment__equal | 24 | +0.00561 | [-0.03709, +0.03693] | +3.42602 | [-6.26229, +14.35183] | 999 |
| dual__graph_perm minus single__graph_perm | 24 | +0.03223 | [+0.00644, +0.06395] | +6.30335 | [-0.32207, +14.04168] | 999 |
| dual__graph_perm minus moment__iu | 24 | +0.02875 | [-0.01859, +0.06303] | -0.65350 | [-10.48426, +10.29083] | 999 |
| dual__graph_perm minus dual__equal | 24 | +0.00366 | [-0.03919, +0.03320] | +0.17548 | [-9.09722, +10.30421] | 999 |
| dual__graph_perm minus dual__iu | 24 | -0.00782 | [-0.04875, +0.01749] | -4.43223 | [-12.30531, +3.81804] | 999 |
| single__graph_perm minus moment__iu | 24 | -0.00348 | [-0.04913, +0.02199] | -6.95685 | [-15.42929, +1.55580] | 999 |
| single__graph_perm minus moment__equal | 24 | +0.00262 | [-0.04751, +0.03417] | -1.27909 | [-9.53740, +9.07641] | 999 |
| single__graph010 minus single__joint0 | 24 | +0.00357 | [-0.00855, +0.02105] | +2.67594 | [-4.08570, +9.86138] | 999 |
| single__graph010 minus single__graph_perm | 24 | +0.00298 | [-0.00861, +0.01786] | +4.70511 | [+0.00000, +11.33866] | 999 |
| dual__graph010 minus dual__joint0 | 24 | +0.00466 | [-0.00648, +0.02071] | +4.60537 | [-1.77390, +12.72926] | 999 |
| dual__graph010 minus dual__graph_perm | 24 | +0.00335 | [-0.00772, +0.01687] | +4.20952 | [+0.17476, +9.57830] | 999 |
| dual__equal minus moment__equal | 24 | +0.03119 | [+0.00577, +0.05730] | +4.84878 | [+0.01478, +12.14778] | 999 |
| dual__iu minus moment__iu | 24 | +0.03657 | [+0.00764, +0.06898] | +3.77873 | [-2.64473, +10.61211] | 999 |
| dual__iu minus dual__equal | 24 | +0.01148 | [-0.00161, +0.02540] | +4.60771 | [-1.69733, +11.64046] | 999 |
| dual__joint0 minus context__joint0 | 21 | -0.02918 | [-0.07220, +0.00375] | +9.65195 | [-2.38223, +21.70616] | 1000 |
| dual__graph010 minus context__graph010 | 21 | -0.02236 | [-0.05886, +0.00926] | +12.05924 | [+0.49275, +24.65622] | 1000 |
| single__joint0 minus moment__joint0 | 17 | +0.00000 | [+0.00000, +0.00000] | +3.21520 | [+0.06250, +10.80566] | 1000 |
| single__graph010 minus moment__graph010 | 17 | +0.00000 | [+0.00000, +0.00000] | +2.68045 | [+0.00000, +9.60718] | 1000 |
| single__graph_perm minus moment__graph_perm | 17 | +0.00000 | [+0.00000, +0.00000] | +2.32849 | [+0.00000, +9.16372] | 1000 |
| single__joint0 minus context__equal | 24 | -0.07238 | [-0.12710, -0.03348] | +7.94049 | [-7.43219, +19.34359] | 999 |
| dual__joint0 minus context__equal | 24 | -0.04087 | [-0.08364, -0.00782] | +11.81882 | [-1.86943, +22.58905] | 999 |
| context__equal minus moment__equal | 24 | +0.07441 | [+0.03070, +0.10952] | -7.19041 | [-18.47550, +8.76611] | 999 |
| context__iu minus moment__iu | 24 | +0.07238 | [+0.03010, +0.11287] | -7.98864 | [-20.33947, +6.05640] | 999 |
| context__joint0 minus moment__joint0 | 15 | +0.05288 | [+0.00734, +0.10915] | -2.55843 | [-14.23542, +12.16925] | 1000 |
| context__graph010 minus moment__graph010 | 15 | +0.04080 | [-0.00150, +0.09014] | -3.57103 | [-16.43685, +11.70612] | 1000 |
| context__joint0 minus context__equal | 21 | -0.01261 | [-0.02420, -0.00289] | +2.16687 | [-5.99555, +7.67603] | 1000 |
| context__graph010 minus context__equal | 21 | -0.01572 | [-0.02839, -0.00334] | +4.36495 | [-5.16845, +11.65056] | 1000 |


## ProcessBench: clean answers and exact error locations

The new PB cohort has 33 clean and 53 erroneous answers. Moment IU gets 15 clean and 11 exact errors right; dual IU gets 18 and 12, while dual Joint graph gets 17 and 13. Raw peaks hit 17, 20 and 18 of the 53 erroneous answers respectively. Dual IU has the same GSM score and higher subset F1 in the other three subsets than moment IU, but Omni-Math exact-error hits decrease from six to five. Its higher clean-answer accuracy offsets this in the harmonic score. Do not call this uniform improvement of error localization. Every arm and subset, including those not expanded here, is in EVALUATION.json.

| Arm | PB subset | Clean hits | Exact-error hits | Subset F1 | Valid decisions |
|---|---|---|---|---|---|
| moment__iu | pb_gsm8k_q8 | 5/9 | 1/7 | 22.73% | 16 |
| moment__iu | pb_math_q8 | 3/7 | 2/17 | 18.46% | 24 |
| moment__iu | pb_olympiadbench_q8 | 2/7 | 2/15 | 18.18% | 22 |
| moment__iu | pb_omnimath_q8 | 5/10 | 6/14 | 46.15% | 24 |
| single__joint0 | pb_gsm8k_q8 | 6/9 | 1/7 | 23.53% | 16 |
| single__joint0 | pb_math_q8 | 3/7 | 2/17 | 18.46% | 24 |
| single__joint0 | pb_olympiadbench_q8 | 2/7 | 2/15 | 18.18% | 22 |
| single__joint0 | pb_omnimath_q8 | 2/10 | 5/14 | 25.64% | 24 |
| dual__joint0 | pb_gsm8k_q8 | 6/9 | 1/7 | 23.53% | 16 |
| dual__joint0 | pb_math_q8 | 3/7 | 3/17 | 25.00% | 24 |
| dual__joint0 | pb_olympiadbench_q8 | 2/7 | 3/15 | 23.53% | 22 |
| dual__joint0 | pb_omnimath_q8 | 3/10 | 4/14 | 29.27% | 24 |
| dual__graph010 | pb_gsm8k_q8 | 6/9 | 1/7 | 23.53% | 16 |
| dual__graph010 | pb_math_q8 | 4/7 | 3/17 | 26.97% | 24 |
| dual__graph010 | pb_olympiadbench_q8 | 2/7 | 4/15 | 27.59% | 22 |
| dual__graph010 | pb_omnimath_q8 | 5/10 | 5/14 | 41.67% | 24 |
| dual__graph_perm | pb_gsm8k_q8 | 6/9 | 1/7 | 23.53% | 16 |
| dual__graph_perm | pb_math_q8 | 3/7 | 2/17 | 18.46% | 24 |
| dual__graph_perm | pb_olympiadbench_q8 | 2/7 | 4/15 | 27.59% | 22 |
| dual__graph_perm | pb_omnimath_q8 | 4/10 | 4/14 | 33.33% | 24 |
| dual__equal | pb_gsm8k_q8 | 4/9 | 1/7 | 21.62% | 16 |
| dual__equal | pb_math_q8 | 3/7 | 3/17 | 25.00% | 24 |
| dual__equal | pb_olympiadbench_q8 | 2/7 | 1/15 | 10.81% | 22 |
| dual__equal | pb_omnimath_q8 | 6/10 | 5/14 | 44.78% | 24 |
| dual__iu | pb_gsm8k_q8 | 5/9 | 1/7 | 22.73% | 16 |
| dual__iu | pb_math_q8 | 3/7 | 3/17 | 25.00% | 24 |
| dual__iu | pb_olympiadbench_q8 | 2/7 | 3/15 | 23.53% | 22 |
| dual__iu | pb_omnimath_q8 | 8/10 | 5/14 | 49.38% | 24 |
| context__equal | pb_gsm8k_q8 | 6/9 | 0/7 | 0.00% | 16 |
| context__equal | pb_math_q8 | 4/7 | 2/17 | 19.51% | 24 |
| context__equal | pb_olympiadbench_q8 | 2/7 | 1/15 | 10.81% | 22 |
| context__equal | pb_omnimath_q8 | 7/10 | 2/14 | 23.73% | 24 |


## Coverage and routing are part of the result

Moment Joint is valid for 78/110 answers; context Joint for 102/110. Context rescues 29 moment failures but loses five moment-valid fits. The dual route therefore uses 78 moment Joint, 29 context Joint and three IU fallbacks. Single uses 78 Joint and 32 IU. Moment has 31 inadmissible partitions and one unconverged/blocked multistart fit; context has seven inadmissible partitions and one such fit. These failures are retained. Context Joint has higher pooled AUC than context equal on their unmatched full rows, but on the same 21 valid PRMB answers it is worse: 0.68397 versus 0.69658. Their paired AUC interval is [-0.02420, -0.00289], also exploratory.

| Cell | Answers | Moment Joint valid | Context Joint valid | Dual: moment Joint | Dual: context Joint | Dual: IU fallback |
|---|---|---|---|---|---|---|
| prmbench_qwen3_8b | 24 | 17 | 21 | 17 | 6 | 1 |
| pb_gsm8k_q8 | 16 | 14 | 14 | 14 | 2 | 0 |
| pb_math_q8 | 24 | 16 | 23 | 16 | 7 | 1 |
| pb_olympiadbench_q8 | 22 | 14 | 21 | 14 | 8 | 0 |
| pb_omnimath_q8 | 24 | 17 | 23 | 17 | 6 | 1 |


## Review and compute

Independent reconstruction passed for selection and group exclusions, all 110 raw-input/span and label-ID joins, 220 feature banks and normalizations, 980 weight projections, 1,970 step maps, 1,090 GMM decisions, 880 exact fallback inheritances, 19 complete metric/history bundles and all 38 paired point bundles. Five explicit 1,000-draw bootstrap reconstructions match all four intervals and defined counts. Five representative Joint refits reproduce 15 native inverse heads; the original Joint optimizer and graph-builder kernels were reused. This is not an independent implementation of those two algorithms. Four scientific tests passed before cohort freeze (two scorer/replay and two cohort-selection tests). The report is structurally checked; it has not been visually inspected in a browser.

| Measurement | Value |
|---|---|
| Complete 19-arm scoring, three CPU workers | 213.06 s wall time |
| All 38 paired contrasts | 18.46 s |
| Independent review | 40.52 s |
| Largest independently reconstructed feature difference | 1.4210854715202004e-14 |
| Largest reconstructed weight-projection difference | 1.1102230246251565e-15 |
| Largest refitted inverse-weight difference | 5.473399511402022e-14 |


## What this stage supports next

Keep moment IU, dual IU, dual Joint0 and dual Joint graph as research anchors, with both always-on equal banks and the matched routed equal control. Do not tune these recipes on this cohort. Next audit Claude's proposed minimum FEATURE-group size two: distinguish identifiability of each latent loading from identifiability of the covariance and fusion weights, then check solver behavior before any new measured arm. This concerns our Joint fusion core. Benchmark SOURCE-question groups are a separate correction: Claude's multi-answer fits and selections still require corrected-fold reruns. Further feature/graph development and supporting temporal/geometry/sampling ideas remain open. A complete comparator registry, untouched confirmation and frozen-method transfer to the historical 24 cells are still pending.


## Evidence and code

- [Frozen cohort, protocol/source hashes and comparison roster](MANIFEST.json)
- [Predictions frozen before label evaluation](SCORES_FROZEN.json)
- [All target joins, scores and endpoint bundles](EVALUATION.json)
- [All 38 comparisons and four intervals](CONTRASTS.json)
- [Independent numerical review](REVIEW.json)
- [Previous source-group correction and score bridge](../localization_source_group_audit_v1/REPORT.html)
- [Historical 58-answer fallback experiment](../fusion_explicit_fallback_pilot_v1/REPORT.html)
- [Frozen protocol](../../docs/experiments/FUSION_SOURCE_DISJOINT_REPLICATION_V1.md)
- [Fusion scorer and explicit routing](../../spectral_utils/fusion_replication.py)
- [Independent reviewer](../../scripts/review_fusion_replication_v1.py)
- [Undergraduate guide to our fusion architecture](../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html)
