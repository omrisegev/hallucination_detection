# Checked pair groups: localization results

2026-09-07. Development experiment. Independent review PASS.


## Decision: retain the existing incumbents

The pair extension increases valid Joint fits, but this fixed localization recipe is not an improvement. The dual-bank graph route falls from PRMB AUC 0.63350 / PB F1 29.94% to 0.58974 / 19.29% on the same answers. Its exploratory paired intervals are below zero on both primary endpoints. Single-bank fallback is closer to its old result, but adds no consistent gain. Do not promote the new pair-routing policy. Keep the mathematical repair and the negative result as part of developing our fusion, not as evidence that the entire Joint/graph family is finished.


## Our fusion method is still the center

The architecture remains one answer -> N windows x P features -> IU-PCR / Joint L-SML -> fused trajectory -> official step and no-error decisions. This experiment changes the feature-group minimum and its safe covariance/Jacobian treatment. It does not replace fusion with an auxiliary detector. More admissible partitions and more valid fits are useful engineering properties; they are not a correctness objective.


## What we can show our advisors

We have adapted our existing fusion architecture to observations inside one answer, tested feature banks and graph penalties with matched controls, and repaired two mathematical edge cases in admitting feature pairs. The quality test also identifies a weakness in using fit eligibility to select the feature bank. These are concrete steps in developing IU-PCR and Joint L-SML. They do not yet establish a consistently better localizer. IMM, LOCA, Diverging Flows, KalmanNet and token sampling remain possible supporting components. Each needs the same fusion with and without it, plus simple aggregation with the component, to show what fusion contributes.


## Frozen contract and what was compared

All 110 previously evaluated development answers were retained: 24 PRMB, 16 GSM8K, 24 MATH, 22 OlympiadBench and 24 Omni-MATH. These are one-pass teacher-forced gray-box traces of fixed official answers. Fitting stays inside each answer, with the declared negative-entropy anchor. Width eight, 27 moment/context coordinates, four chronological blocks, K={3,4,6,8}, held admissibility 0.95, five starts, 5000 sweeps, inverse condition 1000 and all seeds are unchanged. Graph lambda is 0 or 0.1, with the same node-permutation control. Predictions were frozen before this evaluator read labels; existing data exposure is still disclosed. No untouched confirmation is claimed.


## Read the new method names

The 19 original arms are copied exactly as anchors. Fourteen new arms bring the total to 33. pair_moment and pair_context are pure checked-pair Joint on the respective bank. joint0 means no graph penalty; graph010 means lambda 0.1; graph_perm uses the permuted graph. pair_single uses moment Joint then moment IU on invalid fits. pair_dual tries context Joint between them. pair_dual__iu/equal use the same new bank route, but apply the unchanged IU/equal map on that bank. A no-error decision or failed readout never triggers another route. The fixed-IU gate remains diagnostic.


## Matched headline rows

Higher is better. Every row here covers all 110 answers. PRMB pooled AUC ranks steps across answers; within-answer AUC uses the 16 answers with both step labels. PB is the mean of four subset-level harmonic means of clean and exact-first-error accuracy. Invalid decisions remain failures on all 86 PB answers.

| Arm | Fit coverage | Valid PRMB | Pooled PRMB AUC | Within-answer AUC | PB native F1 | PB fixed-IU gate: diagnostic |
|---|---|---|---|---|---|---|
| moment__iu | 110/110 | 24/24 | 0.60140 | 0.67110 | 26.38% | 26.38% |
| dual__iu | 110/110 | 24/24 | 0.63797 | 0.67470 | 30.16% | 30.37% |
| dual__graph010 | 110/110 | 24/24 | 0.63350 | 0.64817 | 29.94% | 29.53% |
| single__joint0 | 110/110 | 24/24 | 0.59733 | 0.62674 | 21.45% | 26.60% |
| pair_single__joint0 | 110/110 | 24/24 | 0.59742 | 0.62906 | 19.18% | 23.93% |
| pair_dual__joint0 | 110/110 | 24/24 | 0.58689 | 0.60149 | 18.90% | 23.93% |
| pair_dual__graph010 | 110/110 | 24/24 | 0.58974 | 0.61238 | 19.29% | 25.26% |
| pair_dual__iu | 110/110 | 24/24 | 0.59222 | 0.64353 | 26.10% | 26.38% |
| pair_dual__equal | 110/110 | 24/24 | 0.58522 | 0.64584 | 20.43% | 26.38% |
| context__equal | 110/110 | 24/24 | 0.66971 | 0.66964 | 13.51% | 26.89% |


## All 33 arms, including pure failures

Pure moment Joint coverage rises 78 -> 106; context 102 -> 108. Pure PRMB scores use their own valid answers, so their full-row AUCs are not automatically matched comparisons. Native and fixed-IU readouts are separate endpoints. All valid fits in this run have valid native decisions. The two proposed routes retain all answers.

| Arm | Fit coverage | Valid PRMB | Pooled PRMB AUC | Within-answer AUC | PB native F1 | PB fixed-IU gate: diagnostic |
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
| pair_moment__joint0 | 106/110 | 23/24 | 0.59759 | 0.62590 | 18.90% | 23.66% |
| pair_moment__graph010 | 106/110 | 23/24 | 0.60058 | 0.63752 | 19.29% | 24.98% |
| pair_moment__graph_perm | 106/110 | 23/24 | 0.60166 | 0.62112 | 18.21% | 24.98% |
| pair_context__joint0 | 108/110 | 23/24 | 0.64353 | 0.62774 | 19.60% | 28.23% |
| pair_context__graph010 | 108/110 | 23/24 | 0.64015 | 0.63191 | 21.48% | 29.25% |
| pair_context__graph_perm | 108/110 | 23/24 | 0.64442 | 0.62774 | 22.73% | 29.25% |
| pair_single__joint0 | 110/110 | 24/24 | 0.59742 | 0.62906 | 19.18% | 23.93% |
| pair_single__graph010 | 110/110 | 24/24 | 0.59991 | 0.63995 | 19.49% | 25.26% |
| pair_single__graph_perm | 110/110 | 24/24 | 0.60118 | 0.62458 | 18.49% | 25.26% |
| pair_dual__joint0 | 110/110 | 24/24 | 0.58689 | 0.60149 | 18.90% | 23.93% |
| pair_dual__graph010 | 110/110 | 24/24 | 0.58974 | 0.61238 | 19.29% | 25.26% |
| pair_dual__graph_perm | 110/110 | 24/24 | 0.59091 | 0.59700 | 18.21% | 25.26% |
| pair_dual__equal | 110/110 | 24/24 | 0.58522 | 0.64584 | 20.43% | 26.38% |
| pair_dual__iu | 110/110 | 24/24 | 0.59222 | 0.64353 | 26.10% | 26.38% |


## Paired evidence, with the same IDs

All 63 comparisons were frozen before new quality evaluation. Intervals use 1000 source-group bootstrap draws, stratified by cell, and are exploratory and unadjusted across this roster. The new dual graph versus its old version has AUC delta -0.04376, interval [-0.08151,-0.01527], and PB delta -10.65 points, interval [-21.04,-1.51]. Within-answer and fixed-IU diagnostic intervals for that comparison include zero. The diagnostics do not replace the primary native endpoints.

| New minus reference | Common valid PRMB | Delta AUC | 95% AUC CI | Delta PB (pp) | 95% PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| pair_dual__graph010 minus dual__graph010 | 24 | -0.04376 | [-0.08151, -0.01527] | -10.65080 | [-21.04139, -1.50963] | 999 |
| pair_dual__joint0 minus dual__joint0 | 24 | -0.04195 | [-0.07310, -0.01399] | -6.42908 | [-15.59651, -0.41732] | 999 |
| pair_single__joint0 minus single__joint0 | 24 | +0.00009 | [-0.02640, +0.03465] | -2.27433 | [-8.51688, +0.19646] | 999 |
| pair_dual__iu minus dual__iu | 24 | -0.04575 | [-0.08810, -0.01265] | -4.05514 | [-11.36233, +2.18737] | 999 |
| pair_moment__joint0 minus moment__joint0 | 16 | -0.00196 | [-0.03428, +0.02564] | +0.66445 | [-3.84921, +6.40867] | 999 |
| pair_context__joint0 minus context__joint0 | 20 | +0.00109 | [-0.00405, +0.00857] | +3.92085 | [+0.00000, +11.50134] | 999 |
| pair_dual__graph010 minus pair_dual__joint0 | 24 | +0.00285 | [-0.00538, +0.01315] | +0.38364 | [-5.36725, +6.44427] | 999 |
| pair_dual__graph010 minus pair_dual__graph_perm | 24 | -0.00118 | [-0.00887, +0.00916] | +1.07187 | [+0.00000, +5.91791] | 999 |
| pair_dual__graph010 minus pair_dual__iu | 24 | -0.00249 | [-0.03804, +0.02517] | -6.81836 | [-18.00090, +2.62403] | 999 |


## Why coverage alone gave the wrong expectation

The dual route changes banks on 31 answers: 28 move from context to moment, and three move from moment to context. Three further answers move from moment IU fallback to moment Joint, with the bank unchanged. The IU routing control isolates bank choice: its IU maps never change, yet its PRMB/PB falls from 0.63797/30.16% to 0.59222/26.10%. That is evidence that fit validity is insufficient as a rule for choosing the more useful representation. It does not prove that all of the Joint regression is caused by routing, because its fitted weights can also change.

| Old dual route | New dual route | Answers |
|---|---|---|
| moment_joint | moment_joint | 75 |
| moment_joint | context_joint | 3 |
| context_joint | moment_joint | 28 |
| moment_iu | moment_joint | 3 |
| context_joint | context_joint | 1 |


## Do not misread the pure context comparison

Context Joint0's unmatched full AUC falls from 0.68397 on 21 answers to 0.64353 on 23. On the 20 common valid answers, however, its paired AUC change is +0.00109, and its mean within-answer change is exactly zero. Its PB full-population change is +3.92 points, with interval [0.00,+11.50], not a clear two-task win. Moment Joint0 has common-ID AUC delta -0.00196 on 16 answers and PB delta +0.66 points, both intervals including zero. The paired tables distinguish representation/weight changes from who could be scored.


## ProcessBench: what changed in actual decisions

The cohort contains 33 clean and 53 erroneous answers. The old dual graph gets 17 clean and 13 exact errors right; the new dual graph gets 10 and 11. Its raw peak hits 14 erroneous answers, versus 18 before. Both the no-error decision and localization need work. Subset-balanced F1 and total successes are different summaries; both are visible.

| Arm | Clean hits | Exact-error hits | Raw peak hits | GSM F1 | MATH F1 | Olympiad F1 | Omni F1 |
|---|---|---|---|---|---|---|---|
| moment__iu | 15/33 | 11/53 | 17/53 | 22.73% | 18.46% | 18.18% | 46.15% |
| dual__iu | 18/33 | 12/53 | 20/53 | 22.73% | 25.00% | 23.53% | 49.38% |
| dual__graph010 | 17/33 | 13/53 | 18/53 | 23.53% | 26.97% | 27.59% | 41.67% |
| single__joint0 | 13/33 | 10/53 | 15/53 | 23.53% | 18.46% | 18.18% | 25.64% |
| pair_single__joint0 | 10/33 | 9/53 | 14/53 | 22.73% | 16.67% | 13.79% | 23.53% |
| pair_dual__joint0 | 9/33 | 9/53 | 13/53 | 21.62% | 16.67% | 13.79% | 23.53% |
| pair_dual__graph010 | 10/33 | 11/53 | 14/53 | 22.73% | 25.00% | 13.79% | 15.62% |
| pair_dual__iu | 14/33 | 11/53 | 16/53 | 21.62% | 18.46% | 18.18% | 46.15% |
| pair_dual__equal | 11/33 | 9/53 | 15/53 | 21.62% | 16.67% | 10.81% | 32.61% |
| context__equal | 19/33 | 5/53 | 15/53 | 0.00% | 19.51% | 10.81% | 23.73% |


## Historical context remains visible

All 19 prior 110-answer endpoint bundles replay exactly. The older 58-answer cohort is shown separately below. Changes across these two question cohorts are not algorithmic gains. New pair arms have no older-58 result in this experiment. Claude's multi-answer scores remain a separate fitting protocol requiring corrected-fold refits; this answer-only comparison does not repair those scores.

| Frozen anchor | Old valid PRMB | Old-58 AUC | Old-58 PB F1 | Current valid PRMB | Current-110 AUC | Current-110 PB F1 |
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


## All 63 registered comparisons

The JSON includes all four intervals for each comparison, including within-answer AUC and the fixed-IU gate diagnostic. Undefined bootstrap draws are counted, not silently treated as zero.

| New minus reference | Common valid PRMB | Delta AUC | 95% AUC CI | Delta PB (pp) | 95% PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| pair_moment__joint0 minus moment__joint0 | 16 | -0.00196 | [-0.03428, +0.02564] | +0.66445 | [-3.84921, +6.40867] | 999 |
| pair_moment__joint0 minus moment__iu | 23 | -0.00402 | [-0.04736, +0.02734] | -7.47842 | [-17.85053, +1.35699] | 999 |
| pair_moment__joint0 minus moment__equal | 23 | +0.00338 | [-0.04368, +0.03909] | -1.80066 | [-12.79787, +8.16789] | 999 |
| pair_moment__graph010 minus moment__graph010 | 16 | -0.00269 | [-0.03712, +0.03056] | -2.16259 | [-9.13507, +4.76282] | 999 |
| pair_moment__graph010 minus moment__iu | 23 | -0.00103 | [-0.04013, +0.02909] | -7.09477 | [-18.49667, +2.94516] | 999 |
| pair_moment__graph010 minus moment__equal | 23 | +0.00637 | [-0.03655, +0.04027] | -1.41702 | [-13.21963, +10.73339] | 999 |
| pair_moment__graph_perm minus moment__graph_perm | 16 | -0.00244 | [-0.02855, +0.01987] | +1.11870 | [-6.12883, +7.57129] | 999 |
| pair_moment__graph_perm minus moment__iu | 23 | +0.00005 | [-0.03984, +0.02984] | -8.16664 | [-19.73732, +1.51029] | 999 |
| pair_moment__graph_perm minus moment__equal | 23 | +0.00745 | [-0.03718, +0.04112] | -2.48888 | [-14.24644, +9.42137] | 999 |
| pair_moment__graph010 minus pair_moment__joint0 | 23 | +0.00299 | [-0.00575, +0.01295] | +0.38364 | [-5.75735, +6.86186] | 999 |
| pair_moment__graph010 minus pair_moment__graph_perm | 23 | -0.00108 | [-0.00965, +0.00949] | +1.07187 | [+0.00000, +6.48234] | 999 |
| pair_context__joint0 minus context__joint0 | 20 | +0.00109 | [-0.00405, +0.00857] | +3.92085 | [+0.00000, +11.50134] | 999 |
| pair_context__joint0 minus context__iu | 23 | -0.01847 | [-0.03576, -0.00591] | +1.20820 | [-5.09299, +8.25770] | 999 |
| pair_context__joint0 minus context__equal | 23 | -0.01370 | [-0.02576, -0.00272] | +6.08772 | [-1.59241, +13.21572] | 999 |
| pair_context__graph010 minus context__graph010 | 20 | +0.00340 | [-0.00121, +0.01081] | +3.60198 | [+0.00000, +11.19608] | 999 |
| pair_context__graph010 minus context__iu | 23 | -0.02184 | [-0.03666, -0.00800] | +3.08741 | [-3.10016, +9.34348] | 999 |
| pair_context__graph010 minus context__equal | 23 | -0.01708 | [-0.02802, -0.00330] | +7.96693 | [-1.51529, +16.18701] | 999 |
| pair_context__graph_perm minus context__graph_perm | 20 | +0.00095 | [-0.00362, +0.00733] | +4.32787 | [+0.00000, +12.50469] | 999 |
| pair_context__graph_perm minus context__iu | 23 | -0.01758 | [-0.02912, -0.00731] | +4.33656 | [-0.64583, +11.03942] | 999 |
| pair_context__graph_perm minus context__equal | 23 | -0.01281 | [-0.02294, -0.00169] | +9.21608 | [+0.77231, +17.06531] | 999 |
| pair_context__graph010 minus pair_context__joint0 | 23 | -0.00338 | [-0.01035, +0.00488] | +1.87921 | [-3.57852, +7.48223] | 999 |
| pair_context__graph010 minus pair_context__graph_perm | 23 | -0.00427 | [-0.01166, +0.00344] | -1.24915 | [-5.98213, +0.00000] | 999 |
| pair_single__joint0 minus single__joint0 | 24 | +0.00009 | [-0.02640, +0.03465] | -2.27433 | [-8.51688, +0.19646] | 999 |
| pair_single__joint0 minus moment__iu | 24 | -0.00398 | [-0.04319, +0.02638] | -7.20201 | [-17.24355, +0.67999] | 999 |
| pair_single__joint0 minus context__equal | 24 | -0.07229 | [-0.12328, -0.03123] | +5.66616 | [-10.27333, +16.42526] | 999 |
| pair_single__graph010 minus single__graph010 | 24 | -0.00099 | [-0.02533, +0.03021] | -4.64251 | [-13.63284, +2.42316] | 999 |
| pair_single__graph010 minus moment__iu | 24 | -0.00149 | [-0.03761, +0.02663] | -6.89424 | [-17.91333, +2.58558] | 999 |
| pair_single__graph010 minus context__equal | 24 | -0.06980 | [-0.11730, -0.03046] | +5.97392 | [-9.87246, +17.29430] | 999 |
| pair_single__graph_perm minus single__graph_perm | 24 | +0.00325 | [-0.02032, +0.03384] | -0.93338 | [-10.37378, +5.60683] | 999 |
| pair_single__graph_perm minus moment__iu | 24 | -0.00023 | [-0.03800, +0.02816] | -7.89023 | [-19.08951, +1.32560] | 999 |
| pair_single__graph_perm minus context__equal | 24 | -0.06854 | [-0.11510, -0.02912] | +4.97793 | [-11.48385, +15.78613] | 999 |
| pair_single__graph010 minus pair_single__joint0 | 24 | +0.00249 | [-0.00607, +0.01340] | +0.30777 | [-5.46614, +6.26004] | 999 |
| pair_single__graph010 minus pair_single__graph_perm | 24 | -0.00127 | [-0.00948, +0.00975] | +0.99599 | [+0.00000, +5.77189] | 999 |
| pair_dual__joint0 minus dual__joint0 | 24 | -0.04195 | [-0.07310, -0.01399] | -6.42908 | [-15.59651, -0.41732] | 999 |
| pair_dual__joint0 minus moment__iu | 24 | -0.01451 | [-0.06642, +0.02021] | -7.47842 | [-18.12317, +0.09744] | 999 |
| pair_dual__joint0 minus context__equal | 24 | -0.08282 | [-0.13382, -0.04557] | +5.38975 | [-10.27333, +15.61965] | 999 |
| pair_dual__graph010 minus dual__graph010 | 24 | -0.04376 | [-0.08151, -0.01527] | -10.65080 | [-21.04139, -1.50963] | 999 |
| pair_dual__graph010 minus moment__iu | 24 | -0.01166 | [-0.06002, +0.02173] | -7.09477 | [-18.23342, +2.42249] | 999 |
| pair_dual__graph010 minus context__equal | 24 | -0.07997 | [-0.12449, -0.04560] | +5.77339 | [-10.12223, +16.82094] | 999 |
| pair_dual__graph_perm minus dual__graph_perm | 24 | -0.03924 | [-0.06988, -0.01363] | -7.51315 | [-19.28879, +1.27624] | 999 |
| pair_dual__graph_perm minus moment__iu | 24 | -0.01049 | [-0.06125, +0.02174] | -8.16664 | [-19.92369, +1.03496] | 999 |
| pair_dual__graph_perm minus context__equal | 24 | -0.07880 | [-0.12544, -0.04443] | +4.70152 | [-11.57184, +15.12821] | 999 |
| pair_dual__graph010 minus pair_dual__joint0 | 24 | +0.00285 | [-0.00538, +0.01315] | +0.38364 | [-5.36725, +6.44427] | 999 |
| pair_dual__graph010 minus pair_dual__graph_perm | 24 | -0.00118 | [-0.00887, +0.00916] | +1.07187 | [+0.00000, +5.91791] | 999 |
| pair_dual__joint0 minus pair_single__joint0 | 24 | -0.01053 | [-0.04579, +0.00000] | -0.27641 | [-3.12500, +0.00000] | 999 |
| pair_dual__joint0 minus pair_dual__equal | 24 | +0.00167 | [-0.03975, +0.03615] | -1.52425 | [-10.95628, +7.14594] | 999 |
| pair_dual__joint0 minus pair_dual__iu | 24 | -0.00533 | [-0.04540, +0.02451] | -7.20201 | [-17.60396, +0.83055] | 999 |
| pair_dual__graph010 minus pair_single__graph010 | 24 | -0.01017 | [-0.04525, +0.00000] | -0.20053 | [-2.17076, +0.00000] | 999 |
| pair_dual__graph010 minus pair_dual__equal | 24 | +0.00452 | [-0.03366, +0.03646] | -1.14060 | [-11.28854, +10.09365] | 999 |
| pair_dual__graph010 minus pair_dual__iu | 24 | -0.00249 | [-0.03804, +0.02517] | -6.81836 | [-18.00090, +2.62403] | 999 |
| pair_dual__graph_perm minus pair_single__graph_perm | 24 | -0.01026 | [-0.04492, +0.00000] | -0.27641 | [-3.12500, +0.00000] | 999 |
| pair_dual__graph_perm minus pair_dual__equal | 24 | +0.00570 | [-0.03319, +0.03702] | -2.21247 | [-13.01357, +8.26129] | 999 |
| pair_dual__graph_perm minus pair_dual__iu | 24 | -0.00131 | [-0.03879, +0.02564] | -7.89023 | [-19.19737, +1.56394] | 999 |
| pair_dual__equal minus dual__equal | 24 | -0.04127 | [-0.08503, -0.01204] | -5.12519 | [-13.06125, -0.23866] | 999 |
| pair_dual__equal minus moment__equal | 24 | -0.01008 | [-0.04553, +0.00000] | -0.27641 | [-2.82068, +0.00000] | 999 |
| pair_dual__equal minus context__equal | 24 | -0.08449 | [-0.11445, -0.05298] | +6.91399 | [-8.78233, +17.78918] | 999 |
| pair_dual__iu minus dual__iu | 24 | -0.04575 | [-0.08810, -0.01265] | -4.05514 | [-11.36233, +2.18737] | 999 |
| pair_dual__iu minus moment__iu | 24 | -0.00918 | [-0.04259, +0.00000] | -0.27641 | [-2.97703, +0.00000] | 999 |
| pair_dual__iu minus context__iu | 24 | -0.08156 | [-0.12088, -0.04568] | +7.71223 | [-6.44456, +19.10671] | 999 |
| pair_dual__iu minus pair_dual__equal | 24 | +0.00701 | [-0.01025, +0.02172] | +5.67776 | [-0.22184, +13.55358] | 999 |
| pair_context__joint0 minus pair_moment__joint0 | 22 | +0.07525 | [+0.03866, +0.11886] | +0.69797 | [-9.33358, +13.83344] | 1000 |
| pair_context__graph010 minus pair_moment__graph010 | 22 | +0.06742 | [+0.03244, +0.10963] | +2.19354 | [-10.56603, +16.45519] | 1000 |
| pair_context__graph_perm minus pair_moment__graph_perm | 22 | +0.07217 | [+0.03285, +0.11780] | +4.51456 | [-7.58079, +19.68777] | 1000 |


## Independent review and runtime

Review passes for 110 direct label/source-group joins, 4710 exact parent arrays, 2090 parent metadata records, 220 audit groupings, 219 fitted covariance/factor replays, 67 independent pair-product Jacobians, 642 graph/native inverse projections, step maps and GMM decisions, 880 route inheritances, all 33 metric bundles and all 63 paired point bundles. Six explicit 1000-draw bootstraps match all four intervals and defined counts. Same-answer gates replay in 176 bank fits; representative newly computed gate recipes are refitted. The Joint optimizer, DUFS and graph-builder kernels were reused; the Laplacian, inverse projection, decisions and endpoints were reconstructed independently. The two pre-freeze tests supplement the eleven preceding pair/audit tests.

| Measurement | Value |
|---|---|
| Scoring wall time, three workers | 157.49 s |
| All 63 contrasts | 31.31 s |
| Independent review | 36.50 s |
| Largest reconstructed risk difference | 5.595524044110789e-14 |
| Browser visual inspection | Not run; structural and link checks only |


## Next short question: the native inverse

Do not replace the incumbent routes with the pair-eligibility route. The next bounded experiment should keep the original Joint fits, feature groups and bank routing fixed and test stronger native inverse conditioning. A post-evaluation unlabeled diagnostic shows that 104/106 valid pair-moment maps and 97/108 pair-context maps sit at the condition-1000 cap; the median old/new condition on their common fits is also 1000. This motivates a conditioning test, not a claim that conditioning caused the observed errors. Start with the lambda-zero native map to isolate that change, retaining the existing graph and IU/equal anchors; add graph-dose interaction only if justified. Full comparator coverage, corrected-fold multi-answer replay, supporting fusion ideas, untouched confirmation and historical24 transfer remain open.


## Evidence and code

- [Frozen protocol/source registry](MANIFEST.json)
- [All labels, scores and metrics](EVALUATION.json)
- [All paired intervals](CONTRASTS.json)
- [Independent review](REVIEW.json)
- [Post-evaluation routing/conditioning diagnostics](DIAGNOSTICS.json)
- [Observed two-test PASS record](TESTS.txt)
- [Pair covariance/Jacobian explanation](../joint_pair_identifiability_audit_v1/REPORT.html)
- [Prior 110-answer results](../fusion_replication_v1/REPORT.html)
- [Checked pair-fusion scoring and routing](../../spectral_utils/fusion_pair_quality.py)
- [Frozen quality protocol](../../docs/experiments/JOINT_PAIR_LOCALIZATION_QUALITY_V1.md)
- [Visual guide to our fusion method](../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html)
