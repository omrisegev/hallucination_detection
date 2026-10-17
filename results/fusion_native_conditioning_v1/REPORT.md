# Original Joint inverse conditioning

2026-09-07. Development evidence. Independent review PASS.


## Decision: keep developing Joint, retain the references

Stronger native conditioning modestly improves Joint over its original condition-1000 map. Dual Joint at condition 30 gives PRMB AUC 0.63639 and PB F1 28.43%, versus original Joint0 0.62884 and 25.33%. However, both paired improvement intervals include zero, and dual IU remains higher at 0.63797 and 30.16%. This is a useful direction for the same fusion method, not a demonstrated two-task winner or an optimal setting.


## What changed inside our fusion

The original Joint model supplies a covariance C and global loading v. Its native weights solve (C + alpha I) w = v. Diagonal ridge alpha reduces the influence of directions with very small variance. A smaller allowed condition number generally requires more ridge. We tested 1000, 300, 100 and 30, with 1000 the unchanged reference. This parameter is different from graph lambda; all new heads here have lambda zero.


## A controlled extension of the original method

All 110 answers are the same previously evaluated development examples: 24 PRMB, and 16 GSM8K, 24 MATH, 22 OlympiadBench and 24 Omni-MATH. These are fixed official answers scored with one teacher-forced gray-box model pass. Normalization, groups, Joint weights and gates are fitted inside each answer. The declared negative-entropy anchor remains. The original fit did not persist C and v, so we reproduced each valid fit on its original matrix and partition, requiring the condition-1000 weights, scores and decisions to replay before new scoring. We now save C, v and group loadings for reuse. All 180 valid fits passed; the 40 invalid bank fits remain invalid. No new pair admission or bank switching took place.


## Full-coverage comparisons on the same answers

All rows in this table cover the entire cohort. Dual means the ORIGINAL route: moment Joint if valid, otherwise context Joint if valid, otherwise moment IU. The matched dual IU and equal controls use the same bank route. Pure moment/context failures appear in the complete table below. The within-answer endpoint uses 16 mixed-label PRMB answers here; pooled AUC includes cross-answer comparisons.

| Arm | Fit coverage | Valid PRMB | Pooled AUC | Within-answer AUC | PB native F1 | PB fixed-IU: diagnostic |
|---|---|---|---|---|---|---|
| moment__iu | 110/110 | 24/24 | 0.60140 | 0.67110 | 26.38% | 26.38% |
| dual__equal | 110/110 | 24/24 | 0.62649 | 0.67359 | 25.55% | 29.35% |
| dual__iu | 110/110 | 24/24 | 0.63797 | 0.67470 | 30.16% | 30.37% |
| dual__joint0 | 110/110 | 24/24 | 0.62884 | 0.63034 | 25.33% | 29.25% |
| dual__cond300 | 110/110 | 24/24 | 0.63079 | 0.62728 | 25.47% | 27.61% |
| dual__cond100 | 110/110 | 24/24 | 0.63296 | 0.64149 | 25.47% | 27.61% |
| dual__cond30 | 110/110 | 24/24 | 0.63639 | 0.67117 | 28.43% | 28.73% |
| dual__graph010 | 110/110 | 24/24 | 0.63350 | 0.64817 | 29.94% | 29.53% |
| single__joint0 | 110/110 | 24/24 | 0.59733 | 0.62674 | 21.45% | 26.60% |
| single__cond30 | 110/110 | 24/24 | 0.60154 | 0.66757 | 23.87% | 25.69% |
| context__equal | 110/110 | 24/24 | 0.66971 | 0.66964 | 13.51% | 26.89% |


## The measured dose response

The strongest of the three new tested settings has the highest dual Joint point estimates. It is not a proven optimum. Plots use narrow vertical ranges to show small changes, and equally spaced tested settings rather than a numeric horizontal scale. These are point estimates; paired uncertainty is shown next.


## Paired evidence limits the conclusion

Dual condition30 minus original Joint0 is +0.00755 AUC, interval [-0.00062,+0.01777], and +3.10 PB points, interval [-2.58,+9.74]. Both include zero. Against dual IU and matched equal, both primary intervals also include zero. These are 1000-draw source-group intervals, exploratory and unadjusted across 69 registered comparisons. A positive within-answer interval against the permuted-graph control is a secondary finding, not a two-task win against IU or the real graph.

| New minus reference | Common PRMB | Delta AUC | 95% AUC CI | Delta PB (pp) | 95% PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| dual__cond30 minus dual__joint0 | 24 | +0.00755 | [-0.00062, +0.01777] | +3.09959 | [-2.58407, +9.73883] | 999 |
| dual__cond30 minus dual__iu | 24 | -0.00158 | [-0.03819, +0.02318] | -1.72848 | [-10.05628, +5.91008] | 999 |
| dual__cond30 minus dual__equal | 24 | +0.00990 | [-0.02932, +0.03747] | +2.87923 | [-6.23458, +12.50686] | 999 |
| dual__cond30 minus dual__graph010 | 24 | +0.00289 | [-0.01282, +0.01529] | -1.50577 | [-6.76644, +2.28625] | 999 |
| dual__cond30 minus dual__graph_perm | 24 | +0.00624 | [-0.00114, +0.01631] | +2.70375 | [-1.80766, +8.15279] | 999 |
| single__cond30 minus single__joint0 | 24 | +0.00420 | [-0.00232, +0.01346] | +2.41327 | [-3.88536, +9.64636] | 999 |
| single__cond30 minus moment__iu | 24 | +0.00014 | [-0.04341, +0.02621] | -2.51440 | [-11.94564, +6.17542] | 999 |
| context__cond30 minus context__equal | 21 | -0.00859 | [-0.01748, -0.00100] | +2.45927 | [-5.40883, +8.11039] | 1000 |
| moment__cond30 minus moment__joint0 | 17 | +0.00739 | [-0.00315, +0.02313] | +2.26982 | [-2.00893, +8.08644] | 1000 |


## Keep the pure-bank coverage caveat

Context condition30 has full-row PRMB AUC 0.68799 on 21 valid answers, versus context equal 0.66971 on all 24. That ordering is misleading as a method comparison: on the SAME 21 answers, context equal scores 0.69658. The paired Joint-minus-equal difference is -0.00859, interval [-0.01748,-0.00100]. Stronger conditioning has not removed this weakness. Pure moment retains 78/110 fits and context 102/110. The original single route stays 78 Joint/32 IU, and dual stays 78 moment Joint/29 context Joint/3 IU.


## ProcessBench: gating and location still differ

Dual condition30 gets 16 clean answers and 12 exact first errors right, versus original Joint0 14 and 11. The total raw-peak hit count stays 17/53; identical totals do not mean identical answer-level hits. With the fixed-original-IU gate, PB falls from 29.25% to 28.73%. Within-answer PRMB AUC rises 0.63034 -> 0.67117, but its paired improvement interval includes zero. The native PB increase alone does not demonstrate better peak localization.

| Arm | Clean hits | Exact-error hits | Raw peak hits | GSM F1 | MATH F1 | Olympiad F1 | Omni F1 |
|---|---|---|---|---|---|---|---|
| moment__iu | 15/33 | 11/53 | 17/53 | 22.73% | 18.46% | 18.18% | 46.15% |
| dual__iu | 18/33 | 12/53 | 20/53 | 22.73% | 25.00% | 23.53% | 49.38% |
| single__joint0 | 13/33 | 10/53 | 15/53 | 23.53% | 18.46% | 18.18% | 25.64% |
| single__cond30 | 14/33 | 11/53 | 15/53 | 23.53% | 18.46% | 18.18% | 35.29% |
| dual__joint0 | 14/33 | 11/53 | 17/53 | 23.53% | 25.00% | 23.53% | 29.27% |
| dual__cond30 | 16/33 | 12/53 | 17/53 | 23.53% | 25.00% | 23.53% | 41.67% |
| dual__graph010 | 17/33 | 13/53 | 18/53 | 23.53% | 26.97% | 27.59% | 41.67% |


## All 45 arms

All 33 preceding arms replay exactly, including the negative checked-pair controls. Twelve new heads change only native conditioning. Invalid fits and decisions remain visible. Pure-bank AUCs use their own valid IDs; use paired common-ID contrasts to compare them.

| Arm | Fit coverage | Valid PRMB | Pooled AUC | Within-answer AUC | PB native F1 | PB fixed-IU: diagnostic |
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
| moment__cond30 | 78/110 | 17/24 | 0.59381 | 0.65025 | 20.51% | 22.07% |
| moment__cond100 | 78/110 | 17/24 | 0.59060 | 0.61371 | 18.53% | 20.75% |
| moment__cond300 | 78/110 | 17/24 | 0.59038 | 0.59623 | 18.53% | 20.75% |
| context__cond30 | 102/110 | 21/24 | 0.68799 | 0.70502 | 15.97% | 24.39% |
| context__cond100 | 102/110 | 21/24 | 0.68519 | 0.70502 | 15.97% | 24.39% |
| context__cond300 | 102/110 | 21/24 | 0.68445 | 0.70327 | 15.97% | 24.39% |
| single__cond30 | 110/110 | 24/24 | 0.60154 | 0.66757 | 23.87% | 25.69% |
| single__cond100 | 110/110 | 24/24 | 0.59901 | 0.63789 | 21.17% | 24.57% |
| single__cond300 | 110/110 | 24/24 | 0.59819 | 0.62368 | 21.17% | 24.57% |
| dual__cond30 | 110/110 | 24/24 | 0.63639 | 0.67117 | 28.43% | 28.73% |
| dual__cond100 | 110/110 | 24/24 | 0.63296 | 0.64149 | 25.47% | 27.61% |
| dual__cond300 | 110/110 | 24/24 | 0.63079 | 0.62728 | 25.47% | 27.61% |


## Historical context, with separate populations

The older 58-answer results are retained below. Changing the evaluated cohort is not an algorithmic gain. The three new conditioning doses have no old-58 result in this experiment. Claude's multi-answer results use another fitting protocol and still require corrected-fold refits. This answer-only experiment does not repair those historical fits.

| Anchor | Old valid PRMB | Old-58 AUC | Old-58 PB | Current valid PRMB | Current-110 AUC | Current-110 PB |
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


## All 69 registered paired comparisons

Every comparison is retained. CONTRASTS.json additionally includes within-answer and fixed-IU-gate intervals. Undefined bootstrap draws are counted instead of being imputed as zero.

| New minus reference | Common PRMB | Delta AUC | 95% AUC CI | Delta PB (pp) | 95% PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| moment__cond30 minus moment__joint0 | 17 | +0.00739 | [-0.00315, +0.02313] | +2.26982 | [-2.00893, +8.08644] | 1000 |
| moment__cond30 minus moment__iu | 17 | +0.00846 | [-0.04164, +0.04165] | -5.87304 | [-16.78617, +2.28658] | 1000 |
| moment__cond30 minus moment__equal | 17 | +0.01981 | [-0.03991, +0.05952] | -0.19529 | [-10.20240, +8.08826] | 1000 |
| moment__cond30 minus moment__graph010 | 17 | +0.00236 | [-0.02061, +0.02480] | -0.94086 | [-8.20806, +5.16043] | 1000 |
| moment__cond30 minus moment__graph_perm | 17 | +0.00471 | [-0.00670, +0.02205] | +3.41229 | [+0.00000, +9.05993] | 1000 |
| moment__cond100 minus moment__joint0 | 17 | +0.00418 | [-0.00366, +0.01378] | +0.29225 | [-3.44828, +5.55840] | 1000 |
| moment__cond100 minus moment__iu | 17 | +0.00525 | [-0.04871, +0.03806] | -7.85062 | [-17.85046, +0.00503] | 1000 |
| moment__cond100 minus moment__equal | 17 | +0.01660 | [-0.04860, +0.05646] | -2.17286 | [-11.77797, +5.62965] | 1000 |
| moment__cond100 minus moment__graph010 | 17 | -0.00086 | [-0.02331, +0.01797] | -2.91843 | [-9.38098, +3.87618] | 1000 |
| moment__cond100 minus moment__graph_perm | 17 | +0.00150 | [-0.01022, +0.01361] | +1.43472 | [+0.00000, +6.41129] | 1000 |
| moment__cond300 minus moment__joint0 | 17 | +0.00396 | [-0.00192, +0.01087] | +0.29225 | [-3.44828, +5.55840] | 1000 |
| moment__cond300 minus moment__iu | 17 | +0.00503 | [-0.05043, +0.03724] | -7.85062 | [-17.85046, +0.00503] | 1000 |
| moment__cond300 minus moment__equal | 17 | +0.01638 | [-0.04795, +0.05659] | -2.17286 | [-11.77797, +5.62965] | 1000 |
| moment__cond300 minus moment__graph010 | 17 | -0.00107 | [-0.02329, +0.01492] | -2.91843 | [-9.38098, +3.87618] | 1000 |
| moment__cond300 minus moment__graph_perm | 17 | +0.00129 | [-0.01162, +0.01235] | +1.43472 | [+0.00000, +6.41129] | 1000 |
| context__cond30 minus context__joint0 | 21 | +0.00402 | [-0.00440, +0.01193] | +0.29240 | [+0.00000, +2.43968] | 1000 |
| context__cond30 minus context__iu | 21 | -0.01419 | [-0.02839, -0.00277] | -2.42025 | [-10.09634, +3.32704] | 1000 |
| context__cond30 minus context__equal | 21 | -0.00859 | [-0.01748, -0.00100] | +2.45927 | [-5.40883, +8.11039] | 1000 |
| context__cond30 minus context__graph010 | 21 | +0.00713 | [-0.00056, +0.01449] | -1.90568 | [-6.49205, +1.78311] | 1000 |
| context__cond30 minus context__graph_perm | 21 | +0.00207 | [-0.00516, +0.00834] | -2.42894 | [-6.98208, +0.00000] | 1000 |
| context__cond100 minus context__joint0 | 21 | +0.00122 | [-0.00684, +0.00790] | +0.29240 | [+0.00000, +2.43968] | 1000 |
| context__cond100 minus context__iu | 21 | -0.01700 | [-0.03208, -0.00563] | -2.42025 | [-10.09634, +3.32704] | 1000 |
| context__cond100 minus context__equal | 21 | -0.01139 | [-0.02118, -0.00298] | +2.45927 | [-5.40883, +8.11039] | 1000 |
| context__cond100 minus context__graph010 | 21 | +0.00433 | [-0.00341, +0.01104] | -1.90568 | [-6.49205, +1.78311] | 1000 |
| context__cond100 minus context__graph_perm | 21 | -0.00073 | [-0.00888, +0.00579] | -2.42894 | [-6.98208, +0.00000] | 1000 |
| context__cond300 minus context__joint0 | 21 | +0.00049 | [-0.00421, +0.00412] | +0.29240 | [+0.00000, +2.43968] | 1000 |
| context__cond300 minus context__iu | 21 | -0.01773 | [-0.03392, -0.00652] | -2.42025 | [-10.09634, +3.32704] | 1000 |
| context__cond300 minus context__equal | 21 | -0.01212 | [-0.02176, -0.00418] | +2.45927 | [-5.40883, +8.11039] | 1000 |
| context__cond300 minus context__graph010 | 21 | +0.00359 | [-0.00489, +0.01128] | -1.90568 | [-6.49205, +1.78311] | 1000 |
| context__cond300 minus context__graph_perm | 21 | -0.00146 | [-0.01040, +0.00563] | -2.42894 | [-6.98208, +0.00000] | 1000 |
| single__cond30 minus single__joint0 | 24 | +0.00420 | [-0.00232, +0.01346] | +2.41327 | [-3.88536, +9.64636] | 999 |
| single__cond30 minus moment__iu | 24 | +0.00014 | [-0.04341, +0.02621] | -2.51440 | [-11.94564, +6.17542] | 999 |
| single__cond30 minus moment__equal | 24 | +0.00624 | [-0.03955, +0.03747] | +3.16336 | [-5.31811, +13.05650] | 999 |
| single__cond30 minus single__graph010 | 24 | +0.00063 | [-0.01694, +0.01624] | -0.26266 | [-6.40194, +5.50178] | 999 |
| single__cond30 minus single__graph_perm | 24 | +0.00362 | [-0.00364, +0.01317] | +4.44245 | [+0.00000, +10.61869] | 999 |
| single__cond100 minus single__joint0 | 24 | +0.00167 | [-0.00260, +0.00774] | -0.28726 | [-5.13105, +5.89584] | 999 |
| single__cond100 minus moment__iu | 24 | -0.00240 | [-0.04763, +0.02436] | -5.21493 | [-12.50401, +2.63244] | 999 |
| single__cond100 minus moment__equal | 24 | +0.00371 | [-0.04594, +0.03615] | +0.46283 | [-6.79882, +9.82609] | 999 |
| single__cond100 minus single__graph010 | 24 | -0.00190 | [-0.01900, +0.01279] | -2.96320 | [-9.51574, +4.34628] | 999 |
| single__cond100 minus single__graph_perm | 24 | +0.00108 | [-0.00567, +0.00828] | +1.74192 | [+0.00000, +6.94541] | 999 |
| single__cond300 minus single__joint0 | 24 | +0.00086 | [-0.00199, +0.00485] | -0.28726 | [-5.13105, +5.89584] | 999 |
| single__cond300 minus moment__iu | 24 | -0.00321 | [-0.04871, +0.02423] | -5.21493 | [-12.50401, +2.63244] | 999 |
| single__cond300 minus moment__equal | 24 | +0.00289 | [-0.04723, +0.03540] | +0.46283 | [-6.79882, +9.82609] | 999 |
| single__cond300 minus single__graph010 | 24 | -0.00271 | [-0.01934, +0.01047] | -2.96320 | [-9.51574, +4.34628] | 999 |
| single__cond300 minus single__graph_perm | 24 | +0.00027 | [-0.00642, +0.00678] | +1.74192 | [+0.00000, +6.94541] | 999 |
| dual__cond30 minus dual__joint0 | 24 | +0.00755 | [-0.00062, +0.01777] | +3.09959 | [-2.58407, +9.73883] | 999 |
| dual__cond30 minus dual__iu | 24 | -0.00158 | [-0.03819, +0.02318] | -1.72848 | [-10.05628, +5.91008] | 999 |
| dual__cond30 minus dual__equal | 24 | +0.00990 | [-0.02932, +0.03747] | +2.87923 | [-6.23458, +12.50686] | 999 |
| dual__cond30 minus dual__graph010 | 24 | +0.00289 | [-0.01282, +0.01529] | -1.50577 | [-6.76644, +2.28625] | 999 |
| dual__cond30 minus dual__graph_perm | 24 | +0.00624 | [-0.00114, +0.01631] | +2.70375 | [-1.80766, +8.15279] | 999 |
| dual__cond100 minus dual__joint0 | 24 | +0.00411 | [-0.00186, +0.01148] | +0.13922 | [-4.27240, +5.84191] | 999 |
| dual__cond100 minus dual__iu | 24 | -0.00502 | [-0.04406, +0.02016] | -4.68885 | [-12.24909, +3.15296] | 999 |
| dual__cond100 minus dual__equal | 24 | +0.00646 | [-0.03385, +0.03523] | -0.08114 | [-9.06330, +9.71970] | 999 |
| dual__cond100 minus dual__graph010 | 24 | -0.00054 | [-0.01519, +0.01118] | -4.46614 | [-10.73148, +0.91559] | 999 |
| dual__cond100 minus dual__graph_perm | 24 | +0.00280 | [-0.00354, +0.01004] | -0.25662 | [-3.75341, +3.17473] | 999 |
| dual__cond300 minus dual__joint0 | 24 | +0.00194 | [-0.00183, +0.00647] | +0.13922 | [-4.27240, +5.84191] | 999 |
| dual__cond300 minus dual__iu | 24 | -0.00719 | [-0.04614, +0.01823] | -4.68885 | [-12.24909, +3.15296] | 999 |
| dual__cond300 minus dual__equal | 24 | +0.00429 | [-0.03658, +0.03341] | -0.08114 | [-9.06330, +9.71970] | 999 |
| dual__cond300 minus dual__graph010 | 24 | -0.00271 | [-0.01770, +0.00822] | -4.46614 | [-10.73148, +0.91559] | 999 |
| dual__cond300 minus dual__graph_perm | 24 | +0.00063 | [-0.00698, +0.00738] | -0.25662 | [-3.75341, +3.17473] | 999 |
| dual__cond30 minus context__equal | 24 | -0.03332 | [-0.07390, -0.00002] | +14.91842 | [+1.65137, +25.02507] | 999 |
| context__cond30 minus moment__cond30 | 15 | +0.04725 | [-0.00158, +0.10506] | -4.53585 | [-16.76064, +10.58982] | 1000 |
| dual__cond30 minus single__cond30 | 24 | +0.03486 | [+0.00854, +0.06690] | +4.56465 | [-1.33454, +11.35626] | 999 |
| dual__cond100 minus context__equal | 24 | -0.03675 | [-0.07852, -0.00328] | +11.95804 | [-0.19028, +21.81441] | 999 |
| context__cond100 minus moment__cond100 | 15 | +0.04643 | [-0.00283, +0.10899] | -2.55828 | [-14.55335, +11.55014] | 1000 |
| dual__cond100 minus single__cond100 | 24 | +0.03395 | [+0.00767, +0.06616] | +4.30481 | [-2.08459, +10.66847] | 999 |
| dual__cond300 minus context__equal | 24 | -0.03892 | [-0.08173, -0.00531] | +11.95804 | [-0.19028, +21.81441] | 999 |
| context__cond300 minus moment__cond300 | 15 | +0.04863 | [+0.00115, +0.10843] | -2.55828 | [-14.55335, +11.55014] | 1000 |
| dual__cond300 minus single__cond300 | 24 | +0.03259 | [+0.00641, +0.06548] | +4.30481 | [-2.08459, +10.66847] | 999 |


## Code and findings review

Three pre-freeze tests pass: analytical ridge behavior, original score/route replay, and preservation of failed new heads/readouts. Independent review passes for 110 label/group joins, 8411 exact parent arrays, 3630 parent metadata records, 180 covariance constructions, 720 inverse/step/GMM reconstructions, 660 fixed-route inheritances, 45 metric bundles and 69 paired point bundles. Ten representative original fits replay. Six explicit 1000-draw bootstraps match all four intervals and defined counts. The optimizer and sklearn GMM kernels are reused. The initial representative refit differed at about 1e-11 when using an independently ordered normalization; exact source-recipe replay now preserves the reduction order. Independent input/algebra checks remain, and no frozen scientific source, score or tolerance was changed.

| Measurement | Value |
|---|---|
| Scoring, three CPU workers | 63.11 s |
| 69 contrasts | 33.44 s |
| Final review | 32.12 s |
| Maximum reconstructed risk difference | 5.595524044110789e-14 |
| Browser visual inspection | Not run; structure and local links checked |


## Next bounded question: does graph structure still help?

The conditioning effect is large enough to change Joint scores and some decisions, but does not establish superiority. Keep all three frozen caps, the original fits and routes, and test their interaction with the existing graph lambda 0.1 and its permutation control. Reuse the saved original covariances and same-answer DUFS gates. This separates graph structure from generic inverse regularization without widening K or searching new graph doses. Do not choose a best cap from these labels or promote a two-task winner. The broader IU/Joint program, supporting ideas, corrected-fold refits, full comparator coverage, untouched confirmation and historical24 transfer remain open.


## Evidence and code

- [Frozen methods and comparisons](MANIFEST.json)
- [All metrics and decisions](EVALUATION.json)
- [All paired intervals](CONTRASTS.json)
- [Independent review](REVIEW.json)
- [Captured test output](TESTS.txt)
- [Test command and source hashes](TEST_EXECUTION.json)
- [Previous checked-pair result](../fusion_pair_quality_v1/REPORT.html)
- [Original 110-answer fusion references](../fusion_replication_v1/REPORT.html)
- [Conditioning and fixed routing code](../../spectral_utils/fusion_native_conditioning.py)
- [Frozen protocol](../../docs/experiments/FUSION_NATIVE_CONDITIONING_V1.md)
- [Visual guide to our fusion](../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html)
