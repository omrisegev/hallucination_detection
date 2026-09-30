# Graph structure after native conditioning

2026-09-07. Development evidence. Independent review PASS.


## Decision: retain Joint with graphs, without declaring a winner

Dual Joint with graph0.1 and condition100 gives PRMB AUC 0.63847 and PB F1 30.22%. Dual IU gives 0.63797 and 30.16%. This crosses IU at the point-estimate level, but the differences are only +0.00050 AUC and +0.06 PB points. Their paired intervals are [-0.03409,+0.02323] and [-7.75,+8.18] points. There is no demonstrated superiority or optimal cap. The result keeps our Joint/graph direction viable while showing that more work is needed for a consistent two-task advantage.


## The same fusion, with two distinct controls

Native Joint solves a regularized inverse using its fitted covariance and global loading. Here graph lambda remains 0.1; condition caps remain 30/100/300. We hold original C/v/u, feature groups, bank routes and normalization fixed. A node permutation tests whether graph alignment matters. The additional graph-smoothed equal control substitutes identity covariance and uniform loading under the SAME graph mechanism, testing what Joint contributes beyond simple aggregation with a graph.


## Fixed benchmark and fitting scope

All 110 previously evaluated development answers remain: 24 PRMB and 86 PB over GSM8K, MATH, OlympiadBench and Omni-MATH. Qwen3-8b scores fixed official answers through one teacher-forced gray-box model pass. The width-eight banks, same-answer fitting and declared negative-entropy anchor remain. All 45 prior arms replay exactly; 24 native graph heads and eight simple graph controls bring the total to 77. No Joint fit is rerun. Original real/permuted graph scores at condition1000 replay before new scoring. Original DUFS gates replay in 182 bank records; the same recipe is computed in 38 remaining banks for simple controls. New predictions were frozen before this evaluator read labels; this is not untouched confirmation.


## Interaction matrix: dual Joint

Each entry is PRMB AUC / PB native F1. All cells cover the same 110 answers with the original dual route. The graph improves PB points most at caps 100 and 300, but point ordering is not a significance test. All three caps were retained; no best cap is selected from these labels.

| Condition cap | No graph | Our graph, lambda0.1 | Permuted graph, lambda0.1 |
|---|---|---|---|
| 1000 | 0.62884 / 25.33% | 0.63350 / 29.94% | 0.63015 / 25.73% |
| 300 | 0.63079 / 25.47% | 0.63684 / 29.94% | 0.63241 / 24.71% |
| 100 | 0.63296 / 25.47% | 0.63847 / 30.22% | 0.63504 / 25.47% |
| 30 | 0.63639 / 28.43% | 0.63838 / 29.45% | 0.63712 / 28.43% |


## Matched full-coverage references

The new graph-smoothed equal control also stays in the comparison. Its permuted version has higher PB than native Joint at condition100, but lower PRMB AUC. Different endpoint leaders do not establish a single consistently better method. The fixed-IU gate is diagnostic only.

| Arm | Fit coverage | Valid PRMB | Pooled AUC | Within-answer AUC | PB native F1 | PB fixed-IU: diagnostic |
|---|---|---|---|---|---|---|
| moment__iu | 110/110 | 24/24 | 0.60140 | 0.67110 | 26.38% | 26.38% |
| dual__iu | 110/110 | 24/24 | 0.63797 | 0.67470 | 30.16% | 30.37% |
| dual__equal | 110/110 | 24/24 | 0.62649 | 0.67359 | 25.55% | 29.35% |
| dual__graph010 | 110/110 | 24/24 | 0.63350 | 0.64817 | 29.94% | 29.53% |
| dual__cond30_graph010 | 110/110 | 24/24 | 0.63838 | 0.68588 | 29.45% | 29.53% |
| dual__cond100_graph010 | 110/110 | 24/24 | 0.63847 | 0.66244 | 30.22% | 29.53% |
| dual__cond300_graph010 | 110/110 | 24/24 | 0.63684 | 0.64682 | 29.94% | 29.53% |
| dual__equal_graph010 | 110/110 | 24/24 | 0.62993 | 0.66107 | 26.62% | 29.25% |
| dual__equal_graph_perm | 110/110 | 24/24 | 0.62170 | 0.63973 | 31.32% | 28.23% |
| context__equal | 110/110 | 24/24 | 0.66971 | 0.66964 | 13.51% | 26.89% |


## Paired evidence for graph structure and fusion

At condition100 the aligned graph adds +4.75 PB points over both no graph and permutation, with interval [0.00,+10.37]. At condition300 it beats permutation by +5.22 points, interval [+0.61,+11.28], while the PRMB interval still includes zero. Comparisons against IU and the matched equal-graph control do not establish a two-task advantage. These are unadjusted exploratory 1000-draw source-group intervals across 101 registered comparisons; the isolated positive PB interval is not a confirmatory claim.

| New minus reference | Common PRMB | Delta AUC | 95% AUC CI | Delta PB (pp) | 95% PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| dual__cond30_graph010 minus dual__cond30 | 24 | +0.00199 | [-0.00370, +0.01054] | +1.01420 | [+0.00000, +4.24634] | 999 |
| dual__cond30_graph010 minus dual__cond30_graph_perm | 24 | +0.00127 | [-0.00344, +0.00779] | +1.01420 | [+0.00000, +4.24634] | 999 |
| dual__cond30_graph010 minus dual__iu | 24 | +0.00041 | [-0.03153, +0.02303] | -0.71428 | [-8.68449, +7.29237] | 999 |
| dual__cond30_graph010 minus dual__equal_graph010 | 24 | +0.00845 | [-0.02479, +0.03499] | +2.82337 | [-6.78875, +12.82991] | 999 |
| dual__cond100_graph010 minus dual__cond100 | 24 | +0.00552 | [-0.00100, +0.01464] | +4.75193 | [+0.00000, +10.36643] | 999 |
| dual__cond100_graph010 minus dual__cond100_graph_perm | 24 | +0.00344 | [-0.00229, +0.01150] | +4.75193 | [+0.00000, +10.36643] | 999 |
| dual__cond100_graph010 minus dual__iu | 24 | +0.00050 | [-0.03409, +0.02323] | +0.06308 | [-7.74974, +8.17915] | 999 |
| dual__cond100_graph010 minus dual__equal_graph010 | 24 | +0.00854 | [-0.02693, +0.03489] | +3.60073 | [-5.81974, +13.66341] | 999 |
| dual__cond300_graph010 minus dual__cond300 | 24 | +0.00606 | [-0.00207, +0.01799] | +4.46614 | [-0.91559, +10.73148] | 999 |
| dual__cond300_graph010 minus dual__cond300_graph_perm | 24 | +0.00443 | [-0.00310, +0.01553] | +5.22372 | [+0.61348, +11.27753] | 999 |
| dual__cond300_graph010 minus dual__iu | 24 | -0.00113 | [-0.03491, +0.02170] | -0.22271 | [-8.45421, +8.59431] | 999 |
| dual__cond300_graph010 minus dual__equal_graph010 | 24 | +0.00692 | [-0.02755, +0.03386] | +3.31494 | [-6.42129, +13.79765] | 999 |


## What the simple graph control tells us

Dual equal-graph gives PRMB 0.62993 / PB 26.62%, versus plain dual equal 0.62649 / 25.55%. Both improvement intervals include zero. Its permuted control gives 0.62170 / 31.32%, so the real graph is not uniformly better for every fusion core. Joint at condition100 exceeds real equal-graph in both points, but both paired intervals include zero. We have not established that the learned Joint structure adds a reliable advantage under this graph. The simple control is exactly equal fusion at lambda zero; its graph condition is at most 3.18289 here, below every cap, so it is identical across the tested caps.

| New minus reference | Common PRMB | Delta AUC | 95% AUC CI | Delta PB (pp) | 95% PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| dual__equal_graph010 minus dual__equal | 24 | +0.00344 | [-0.00643, +0.01346] | +1.07006 | [-4.14456, +6.45505] | 999 |
| dual__equal_graph010 minus dual__equal_graph_perm | 24 | +0.00823 | [-0.00969, +0.02559] | -4.69957 | [-14.86994, +2.50321] | 999 |


## Preserve the pure-bank comparison rule

Pure moment/native graph retains 78/110 valid fits and context 102/110; composites and simple controls cover all answers. Context condition100 graph AUC is 0.68452 on 21 valid PRMB answers. Context equal on the same 21 scores 0.69658, giving delta -0.01206, interval [-0.02432,-0.00124]. Its full-row comparison against equal on all 24 would conceal this weakness. Use shared valid IDs, while PB keeps invalid decisions as failures on the full population.


## ProcessBench: inspect the actual decisions

Dual IU gets 18 clean and 12 exact errors right; Joint condition100 with the real graph gets 17 and 13. Both therefore get 30 answers correct in total, while subset-balanced F1 differs slightly. A tiny macro lead is not a uniform improvement. Raw peaks hit 20 errors for IU and 18 for Joint; fixed-IU PB is 30.37% versus 29.53%. Within-answer PRMB is also higher for IU (0.67470 versus 0.66244).

| Arm | Clean hits | Exact-error hits | Raw peak hits | GSM F1 | MATH F1 | Olympiad F1 | Omni F1 |
|---|---|---|---|---|---|---|---|
| dual__iu | 18/33 | 12/53 | 20/53 | 22.73% | 25.00% | 23.53% | 49.38% |
| dual__graph010 | 17/33 | 13/53 | 18/53 | 23.53% | 26.97% | 27.59% | 41.67% |
| dual__cond30_graph010 | 16/33 | 13/53 | 18/53 | 23.53% | 25.00% | 27.59% | 41.67% |
| dual__cond100_graph010 | 17/33 | 13/53 | 18/53 | 23.53% | 25.00% | 27.59% | 44.78% |
| dual__cond100_graph_perm | 16/33 | 10/53 | 16/53 | 23.53% | 18.46% | 23.53% | 36.36% |
| dual__equal_graph010 | 17/33 | 10/53 | 18/53 | 22.73% | 25.00% | 18.18% | 40.58% |
| dual__equal_graph_perm | 19/33 | 11/53 | 18/53 | 40.00% | 25.00% | 18.18% | 42.11% |


## A more useful next clue: shared misses

This is a POST-EVALUATION diagnostic that uses error labels, not a new scoring method. Of 53 erroneous PB answers, both IU and Joint raw peaks hit 16; only IU hits four; only Joint hits two; neither hits 31. The Joint hit sets are identical across condition1000/300/100/30. Even a perfect chooser restricted to these existing peak locations could hit only 22/53. This is NOT a ceiling for combining full trajectories, reranking steps or adding features: those operations can produce other locations. It shows why simply switching between these heads is unlikely to solve the main localization problem.


## All 77 arms and their failures

All 45 preceding arms are exact historical anchors. Twenty-four conditioned native graph heads and eight simple graph controls are new. No-error and readout failures cannot change the original route. Equal-graph routes follow the original bank choice, using moment equal-graph where the native route falls back to IU.

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
| moment__cond30_graph010 | 78/110 | 17/24 | 0.59510 | 0.66835 | 20.51% | 22.07% |
| moment__cond30_graph_perm | 78/110 | 17/24 | 0.59488 | 0.64968 | 20.51% | 22.07% |
| moment__cond100_graph010 | 78/110 | 17/24 | 0.59510 | 0.63950 | 21.79% | 22.07% |
| moment__cond100_graph_perm | 78/110 | 17/24 | 0.59317 | 0.62140 | 18.53% | 20.75% |
| moment__cond300_graph010 | 78/110 | 17/24 | 0.59488 | 0.62027 | 21.45% | 22.07% |
| moment__cond300_graph_perm | 78/110 | 17/24 | 0.59081 | 0.58166 | 17.10% | 20.75% |
| context__cond30_graph010 | 102/110 | 21/24 | 0.68646 | 0.71157 | 18.40% | 24.88% |
| context__cond30_graph_perm | 102/110 | 21/24 | 0.68762 | 0.70502 | 17.92% | 24.39% |
| context__cond100_graph010 | 102/110 | 21/24 | 0.68452 | 0.70502 | 18.40% | 24.88% |
| context__cond100_graph_perm | 102/110 | 21/24 | 0.68470 | 0.70327 | 17.92% | 24.39% |
| context__cond300_graph010 | 102/110 | 21/24 | 0.68183 | 0.70502 | 18.40% | 24.88% |
| context__cond300_graph_perm | 102/110 | 21/24 | 0.68409 | 0.70327 | 17.92% | 24.39% |
| single__cond30_graph010 | 110/110 | 24/24 | 0.60262 | 0.68228 | 23.87% | 25.69% |
| single__cond30_graph_perm | 110/110 | 24/24 | 0.60298 | 0.66712 | 23.87% | 25.69% |
| single__cond100_graph010 | 110/110 | 24/24 | 0.60325 | 0.65884 | 25.39% | 25.69% |
| single__cond100_graph_perm | 110/110 | 24/24 | 0.60063 | 0.64414 | 21.17% | 24.57% |
| single__cond300_graph010 | 110/110 | 24/24 | 0.60298 | 0.64322 | 24.13% | 25.69% |
| single__cond300_graph_perm | 110/110 | 24/24 | 0.59937 | 0.61185 | 19.42% | 24.57% |
| dual__cond30_graph010 | 110/110 | 24/24 | 0.63838 | 0.68588 | 29.45% | 29.53% |
| dual__cond30_graph_perm | 110/110 | 24/24 | 0.63712 | 0.67071 | 28.43% | 28.73% |
| dual__cond100_graph010 | 110/110 | 24/24 | 0.63847 | 0.66244 | 30.22% | 29.53% |
| dual__cond100_graph_perm | 110/110 | 24/24 | 0.63504 | 0.64774 | 25.47% | 27.61% |
| dual__cond300_graph010 | 110/110 | 24/24 | 0.63684 | 0.64682 | 29.94% | 29.53% |
| dual__cond300_graph_perm | 110/110 | 24/24 | 0.63241 | 0.61544 | 24.71% | 27.61% |
| moment__equal_graph010 | 110/110 | 24/24 | 0.59647 | 0.66013 | 21.53% | 25.26% |
| moment__equal_graph_perm | 110/110 | 24/24 | 0.59367 | 0.63956 | 29.09% | 26.89% |
| context__equal_graph010 | 110/110 | 24/24 | 0.66618 | 0.67040 | 17.32% | 28.23% |
| context__equal_graph_perm | 110/110 | 24/24 | 0.66582 | 0.66573 | 13.69% | 26.89% |
| single__equal_graph010 | 110/110 | 24/24 | 0.59647 | 0.66013 | 21.53% | 25.26% |
| single__equal_graph_perm | 110/110 | 24/24 | 0.59367 | 0.63956 | 29.09% | 26.89% |
| dual__equal_graph010 | 110/110 | 24/24 | 0.62993 | 0.66107 | 26.62% | 29.25% |
| dual__equal_graph_perm | 110/110 | 24/24 | 0.62170 | 0.63973 | 31.32% | 28.23% |


## History remains visible under its original contract

Older-58 and current-110 scores use different question cohorts and are shown separately. Cross-cohort changes are not algorithmic gains. The new graph-condition arms have no old-58 result here. Claude's multi-answer results still require corrected-fold refits; this answer-only experiment does not repair them.

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


## All 101 registered comparisons

All four interval families, including within-answer AUC and fixed-IU PB, are in CONTRASTS.json. Undefined bootstrap draws are counted, not set to zero.

| New minus reference | Common PRMB | Delta AUC | 95% AUC CI | Delta PB (pp) | 95% PB CI (pp) | Defined PB draws |
|---|---|---|---|---|---|---|
| moment__cond30_graph010 minus moment__cond30 | 17 | +0.00129 | [-0.00761, +0.01649] | +0.00000 | [+0.00000, +0.00000] | 1000 |
| moment__cond30_graph010 minus moment__cond30_graph_perm | 17 | +0.00021 | [-0.00753, +0.01128] | +0.00000 | [+0.00000, +0.00000] | 1000 |
| moment__cond30_graph010 minus moment__graph010 | 17 | +0.00364 | [-0.01117, +0.02172] | -0.94086 | [-8.20806, +5.16043] | 1000 |
| moment__cond30_graph010 minus moment__iu | 17 | +0.00975 | [-0.03414, +0.03765] | -5.87304 | [-16.78617, +2.28658] | 1000 |
| moment__cond30_graph010 minus moment__equal | 17 | +0.02110 | [-0.03224, +0.05739] | -0.19529 | [-10.20240, +8.08826] | 1000 |
| moment__cond30_graph010 minus moment__equal_graph010 | 17 | +0.02131 | [-0.02861, +0.05465] | -1.02570 | [-13.24220, +7.39940] | 1000 |
| moment__cond30_graph_perm minus moment__equal_graph_perm | 17 | +0.02795 | [-0.03185, +0.07335] | -8.58283 | [-21.84541, +2.53798] | 1000 |
| moment__cond100_graph010 minus moment__cond100 | 17 | +0.00450 | [-0.00572, +0.01924] | +3.25936 | [+0.00000, +8.09026] | 1000 |
| moment__cond100_graph010 minus moment__cond100_graph_perm | 17 | +0.00193 | [-0.00669, +0.01321] | +3.25936 | [+0.00000, +8.09026] | 1000 |
| moment__cond100_graph010 minus moment__graph010 | 17 | +0.00364 | [-0.00985, +0.01684] | +0.34093 | [-5.26316, +5.23130] | 1000 |
| moment__cond100_graph010 minus moment__iu | 17 | +0.00975 | [-0.03848, +0.03853] | -4.59126 | [-14.67221, +3.33797] | 1000 |
| moment__cond100_graph010 minus moment__equal | 17 | +0.02110 | [-0.03519, +0.05721] | +1.08650 | [-9.26723, +10.26446] | 1000 |
| moment__cond100_graph010 minus moment__equal_graph010 | 17 | +0.02131 | [-0.03339, +0.05591] | +0.25609 | [-10.80677, +8.36182] | 1000 |
| moment__cond100_graph_perm minus moment__equal_graph_perm | 17 | +0.02624 | [-0.03835, +0.07192] | -10.56041 | [-23.80505, +0.75234] | 1000 |
| moment__cond300_graph010 minus moment__cond300 | 17 | +0.00450 | [-0.00715, +0.02314] | +2.91843 | [-3.87618, +9.38098] | 1000 |
| moment__cond300_graph010 minus moment__cond300_graph_perm | 17 | +0.00407 | [-0.00660, +0.02177] | +4.35315 | [+0.00000, +10.45351] | 1000 |
| moment__cond300_graph010 minus moment__graph010 | 17 | +0.00343 | [-0.00319, +0.00984] | +0.00000 | [+0.00000, +0.00000] | 1000 |
| moment__cond300_graph010 minus moment__iu | 17 | +0.00953 | [-0.03707, +0.03863] | -4.93218 | [-15.68127, +3.61178] | 1000 |
| moment__cond300_graph010 minus moment__equal | 17 | +0.02088 | [-0.03362, +0.05804] | +0.74557 | [-10.71604, +10.95921] | 1000 |
| moment__cond300_graph010 minus moment__equal_graph010 | 17 | +0.02110 | [-0.02907, +0.05512] | -0.08484 | [-12.03718, +8.46369] | 1000 |
| moment__cond300_graph_perm minus moment__equal_graph_perm | 17 | +0.02388 | [-0.04288, +0.06953] | -11.99513 | [-24.95093, -0.21639] | 1000 |
| moment__equal_graph010 minus moment__equal | 24 | +0.00118 | [-0.01117, +0.01325] | +0.83041 | [-2.80888, +6.45418] | 999 |
| moment__equal_graph010 minus moment__equal_graph_perm | 24 | +0.00280 | [-0.01685, +0.02160] | -7.55714 | [-17.18457, +1.07948] | 999 |
| context__cond30_graph010 minus context__cond30 | 21 | -0.00152 | [-0.00514, +0.00258] | +2.42894 | [+0.00000, +6.98208] | 1000 |
| context__cond30_graph010 minus context__cond30_graph_perm | 21 | -0.00116 | [-0.00425, +0.00186] | +0.48450 | [+0.00000, +2.99242] | 1000 |
| context__cond30_graph010 minus context__graph010 | 21 | +0.00560 | [-0.00188, +0.01415] | +0.52326 | [+0.00000, +3.30032] | 1000 |
| context__cond30_graph010 minus context__iu | 21 | -0.01572 | [-0.02904, -0.00407] | +0.00869 | [-7.59828, +5.30385] | 1000 |
| context__cond30_graph010 minus context__equal | 21 | -0.01011 | [-0.01907, -0.00143] | +4.88821 | [-4.21007, +11.76249] | 1000 |
| context__cond30_graph010 minus context__equal_graph010 | 21 | -0.00999 | [-0.01527, -0.00414] | +1.08292 | [-8.96586, +9.19141] | 1000 |
| context__cond30_graph_perm minus context__equal_graph_perm | 21 | -0.00408 | [-0.01393, +0.00598] | +4.22180 | [-5.33717, +10.96606] | 1000 |
| context__cond100_graph010 minus context__cond100 | 21 | -0.00067 | [-0.00671, +0.00567] | +2.42894 | [+0.00000, +6.98208] | 1000 |
| context__cond100_graph010 minus context__cond100_graph_perm | 21 | -0.00018 | [-0.00550, +0.00589] | +0.48450 | [+0.00000, +2.99242] | 1000 |
| context__cond100_graph010 minus context__graph010 | 21 | +0.00365 | [-0.00072, +0.00832] | +0.52326 | [+0.00000, +3.30032] | 1000 |
| context__cond100_graph010 minus context__iu | 21 | -0.01767 | [-0.03380, -0.00536] | +0.00869 | [-7.59828, +5.30385] | 1000 |
| context__cond100_graph010 minus context__equal | 21 | -0.01206 | [-0.02432, -0.00124] | +4.88821 | [-4.21007, +11.76249] | 1000 |
| context__cond100_graph010 minus context__equal_graph010 | 21 | -0.01194 | [-0.02030, -0.00615] | +1.08292 | [-8.96586, +9.19141] | 1000 |
| context__cond100_graph_perm minus context__equal_graph_perm | 21 | -0.00701 | [-0.01796, +0.00300] | +4.22180 | [-5.33717, +10.96606] | 1000 |
| context__cond300_graph010 minus context__cond300 | 21 | -0.00262 | [-0.01045, +0.00600] | +2.42894 | [+0.00000, +6.98208] | 1000 |
| context__cond300_graph010 minus context__cond300_graph_perm | 21 | -0.00225 | [-0.00877, +0.00402] | +0.48450 | [+0.00000, +2.99242] | 1000 |
| context__cond300_graph010 minus context__graph010 | 21 | +0.00097 | [-0.00112, +0.00359] | +0.52326 | [+0.00000, +3.30032] | 1000 |
| context__cond300_graph010 minus context__iu | 21 | -0.02035 | [-0.03714, -0.00709] | +0.00869 | [-7.59828, +5.30385] | 1000 |
| context__cond300_graph010 minus context__equal | 21 | -0.01474 | [-0.02679, -0.00257] | +4.88821 | [-4.21007, +11.76249] | 1000 |
| context__cond300_graph010 minus context__equal_graph010 | 21 | -0.01462 | [-0.02481, -0.00680] | +1.08292 | [-8.96586, +9.19141] | 1000 |
| context__cond300_graph_perm minus context__equal_graph_perm | 21 | -0.00761 | [-0.01847, +0.00395] | +4.22180 | [-5.33717, +10.96606] | 1000 |
| context__equal_graph010 minus context__equal | 24 | -0.00353 | [-0.01222, +0.00421] | +3.80529 | [-0.66994, +9.46681] | 999 |
| context__equal_graph010 minus context__equal_graph_perm | 24 | +0.00036 | [-0.00989, +0.00899] | +3.62337 | [-3.13179, +9.40164] | 999 |
| single__cond30_graph010 minus single__cond30 | 24 | +0.00108 | [-0.00515, +0.01076] | +0.00000 | [+0.00000, +0.00000] | 999 |
| single__cond30_graph010 minus single__cond30_graph_perm | 24 | -0.00036 | [-0.00549, +0.00722] | +0.00000 | [+0.00000, +0.00000] | 999 |
| single__cond30_graph010 minus single__graph010 | 24 | +0.00172 | [-0.00957, +0.01470] | -0.26266 | [-6.40194, +5.50178] | 999 |
| single__cond30_graph010 minus moment__iu | 24 | +0.00122 | [-0.03717, +0.02525] | -2.51440 | [-11.94564, +6.17542] | 999 |
| single__cond30_graph010 minus moment__equal | 24 | +0.00732 | [-0.03537, +0.03624] | +3.16336 | [-5.31811, +13.05650] | 999 |
| single__cond30_graph010 minus single__equal_graph010 | 24 | +0.00615 | [-0.03476, +0.03400] | +2.33295 | [-6.88000, +11.67225] | 999 |
| single__cond30_graph_perm minus single__equal_graph_perm | 24 | +0.00931 | [-0.04011, +0.04879] | -5.22419 | [-18.16562, +7.56362] | 999 |
| single__cond100_graph010 minus single__cond100 | 24 | +0.00425 | [-0.00336, +0.01532] | +4.22183 | [+0.00000, +10.06936] | 999 |
| single__cond100_graph010 minus single__cond100_graph_perm | 24 | +0.00262 | [-0.00476, +0.01174] | +4.22183 | [+0.00000, +10.06936] | 999 |
| single__cond100_graph010 minus single__graph010 | 24 | +0.00235 | [-0.00657, +0.01180] | +1.25863 | [-1.88266, +6.25610] | 999 |
| single__cond100_graph010 minus moment__iu | 24 | +0.00185 | [-0.03689, +0.02666] | -0.99310 | [-9.16630, +6.65161] | 999 |
| single__cond100_graph010 minus moment__equal | 24 | +0.00796 | [-0.03520, +0.03590] | +4.68466 | [-4.21758, +14.32015] | 999 |
| single__cond100_graph010 minus single__equal_graph010 | 24 | +0.00678 | [-0.03620, +0.03544] | +3.85425 | [-5.34373, +13.08030] | 999 |
| single__cond100_graph_perm minus single__equal_graph_perm | 24 | +0.00696 | [-0.04550, +0.04678] | -7.92472 | [-19.42765, +5.11455] | 999 |
| single__cond300_graph010 minus single__cond300 | 24 | +0.00479 | [-0.00473, +0.01836] | +2.96320 | [-4.34628, +9.51574] | 999 |
| single__cond300_graph010 minus single__cond300_graph_perm | 24 | +0.00362 | [-0.00569, +0.01705] | +4.70511 | [+0.00000, +11.33866] | 999 |
| single__cond300_graph010 minus single__graph010 | 24 | +0.00208 | [-0.00276, +0.00740] | +0.00000 | [+0.00000, +0.00000] | 999 |
| single__cond300_graph010 minus moment__iu | 24 | +0.00158 | [-0.03752, +0.02689] | -2.25173 | [-10.99806, +6.18099] | 999 |
| single__cond300_graph010 minus moment__equal | 24 | +0.00769 | [-0.03363, +0.03750] | +3.42602 | [-6.26229, +14.35183] | 999 |
| single__cond300_graph010 minus single__equal_graph010 | 24 | +0.00651 | [-0.03599, +0.03627] | +2.59561 | [-6.92359, +12.25523] | 999 |
| single__cond300_graph_perm minus single__equal_graph_perm | 24 | +0.00570 | [-0.04747, +0.04438] | -9.66664 | [-22.86322, +3.65482] | 999 |
| single__equal_graph010 minus moment__equal | 24 | +0.00118 | [-0.01117, +0.01325] | +0.83041 | [-2.80888, +6.45418] | 999 |
| single__equal_graph010 minus single__equal_graph_perm | 24 | +0.00280 | [-0.01685, +0.02160] | -7.55714 | [-17.18457, +1.07948] | 999 |
| dual__cond30_graph010 minus dual__cond30 | 24 | +0.00199 | [-0.00370, +0.01054] | +1.01420 | [+0.00000, +4.24634] | 999 |
| dual__cond30_graph010 minus dual__cond30_graph_perm | 24 | +0.00127 | [-0.00344, +0.00779] | +1.01420 | [+0.00000, +4.24634] | 999 |
| dual__cond30_graph010 minus dual__graph010 | 24 | +0.00488 | [-0.00522, +0.01520] | -0.49157 | [-4.76543, +2.74169] | 999 |
| dual__cond30_graph010 minus dual__iu | 24 | +0.00041 | [-0.03153, +0.02303] | -0.71428 | [-8.68449, +7.29237] | 999 |
| dual__cond30_graph010 minus dual__equal | 24 | +0.01189 | [-0.02206, +0.03789] | +3.89343 | [-5.25882, +13.78907] | 999 |
| dual__cond30_graph010 minus dual__equal_graph010 | 24 | +0.00845 | [-0.02479, +0.03499] | +2.82337 | [-6.78875, +12.82991] | 999 |
| dual__cond30_graph_perm minus dual__equal_graph_perm | 24 | +0.01542 | [-0.02659, +0.04891] | -2.89040 | [-16.91512, +9.66626] | 999 |
| dual__cond100_graph010 minus dual__cond100 | 24 | +0.00552 | [-0.00100, +0.01464] | +4.75193 | [+0.00000, +10.36643] | 999 |
| dual__cond100_graph010 minus dual__cond100_graph_perm | 24 | +0.00344 | [-0.00229, +0.01150] | +4.75193 | [+0.00000, +10.36643] | 999 |
| dual__cond100_graph010 minus dual__graph010 | 24 | +0.00497 | [-0.00347, +0.01277] | +0.28579 | [-3.10865, +3.30929] | 999 |
| dual__cond100_graph010 minus dual__iu | 24 | +0.00050 | [-0.03409, +0.02323] | +0.06308 | [-7.74974, +8.17915] | 999 |
| dual__cond100_graph010 minus dual__equal | 24 | +0.01198 | [-0.02418, +0.03775] | +4.67080 | [-4.83505, +15.01502] | 999 |
| dual__cond100_graph010 minus dual__equal_graph010 | 24 | +0.00854 | [-0.02693, +0.03489] | +3.60073 | [-5.81974, +13.66341] | 999 |
| dual__cond100_graph_perm minus dual__equal_graph_perm | 24 | +0.01334 | [-0.03000, +0.04722] | -5.85077 | [-18.75402, +7.10591] | 999 |
| dual__cond300_graph010 minus dual__cond300 | 24 | +0.00606 | [-0.00207, +0.01799] | +4.46614 | [-0.91559, +10.73148] | 999 |
| dual__cond300_graph010 minus dual__cond300_graph_perm | 24 | +0.00443 | [-0.00310, +0.01553] | +5.22372 | [+0.61348, +11.27753] | 999 |
| dual__cond300_graph010 minus dual__graph010 | 24 | +0.00335 | [-0.00149, +0.00796] | +0.00000 | [+0.00000, +0.00000] | 999 |
| dual__cond300_graph010 minus dual__iu | 24 | -0.00113 | [-0.03491, +0.02170] | -0.22271 | [-8.45421, +8.59431] | 999 |
| dual__cond300_graph010 minus dual__equal | 24 | +0.01035 | [-0.02401, +0.03664] | +4.38501 | [-5.49963, +15.26508] | 999 |
| dual__cond300_graph010 minus dual__equal_graph010 | 24 | +0.00692 | [-0.02755, +0.03386] | +3.31494 | [-6.42129, +13.79765] | 999 |
| dual__cond300_graph_perm minus dual__equal_graph_perm | 24 | +0.01071 | [-0.03376, +0.04433] | -6.60835 | [-19.86251, +6.15563] | 999 |
| dual__equal_graph010 minus dual__equal | 24 | +0.00344 | [-0.00643, +0.01346] | +1.07006 | [-4.14456, +6.45505] | 999 |
| dual__equal_graph010 minus dual__equal_graph_perm | 24 | +0.00823 | [-0.00969, +0.02559] | -4.69957 | [-14.86994, +2.50321] | 999 |
| dual__cond30_graph010 minus context__equal | 24 | -0.03133 | [-0.07072, +0.00087] | +15.93262 | [+1.90336, +26.25424] | 999 |
| dual__cond30_graph010 minus single__cond30_graph010 | 24 | +0.03576 | [+0.01019, +0.06839] | +5.57885 | [-0.96333, +12.72144] | 999 |
| context__cond30_graph010 minus moment__cond30_graph010 | 15 | +0.04849 | [-0.00000, +0.10562] | -2.10691 | [-15.10199, +13.70225] | 1000 |
| dual__cond100_graph010 minus context__equal | 24 | -0.03124 | [-0.07156, +0.00095] | +16.70998 | [+2.89075, +27.07010] | 999 |
| dual__cond100_graph010 minus single__cond100_graph010 | 24 | +0.03522 | [+0.00952, +0.06739] | +4.83491 | [-1.44062, +11.84681] | 999 |
| context__cond100_graph010 minus moment__cond100_graph010 | 15 | +0.04190 | [-0.00382, +0.10021] | -3.38870 | [-16.13262, +11.77640] | 1000 |
| dual__cond300_graph010 minus context__equal | 24 | -0.03287 | [-0.07278, -0.00117] | +16.42419 | [+2.71315, +27.01553] | 999 |
| dual__cond300_graph010 minus single__cond300_graph010 | 24 | +0.03386 | [+0.00902, +0.06570] | +5.80776 | [-0.54683, +13.43836] | 999 |
| context__cond300_graph010 minus moment__cond300_graph010 | 15 | +0.03805 | [-0.00575, +0.09342] | -3.04777 | [-16.29693, +12.81648] | 1000 |


## Code and findings review

Three pre-freeze tests pass. Independent review verifies 110 label/group joins, 11351 exact parent arrays, 4950 metadata records, 182 gate replays, 220 reconstructed source graphs and zero-graph equal replays, 440 Laplacians and equal-control cap-invariance checks, 1880 inverse/step/GMM reconstructions, 1760 route inheritances, all 77 metric bundles and 101 paired point bundles. Seven explicit 1000-draw bootstraps match all four intervals and defined counts. New gate recipes receive ten representative refits. Graph-builder, DUFS and GMM kernels are reused; Laplacians, trace matching, inverses and endpoints are reconstructed independently. No Joint refits. The later overlap diagnostic is separately rechecked from Boolean success sets by this renderer; it is not represented as a preregistered comparison.

| Measurement | Value |
|---|---|
| Scoring, three CPU workers | 51.35 s |
| 101 contrasts | 50.53 s |
| Independent review | 68.70 s |
| Maximum reconstructed risk difference | 5.0487392044828994e-14 |
| Browser visual inspection | Not run; HTML structure and local links checked |


## Next bounded direction: complementary information for fusion

Keep IU and the original/conditioned Joint graph recipes as frozen references. Do not widen the same dose grid just to chase a tiny lead on these 110 answers. The next short stage should audit the old AR/Kalman innovation code, then test one same-answer prediction-residual view inside the existing feature matrix, with unchanged-core and matched equal controls. Old final-answer scalar innovations were highly correlated with entropy; the new view must demonstrate additional information rather than rename that old signal. This supports the requested KalmanNet/flow track without claiming that a simple predictor implements either named method. It also does not rule out improved trajectory readout. Broader supporting tracks, corrected-fold multi-answer refits, full comparators, untouched confirmation and historical24 transfer remain open.


## Evidence and code

- [Frozen experiment](MANIFEST.json)
- [All metrics and decisions](EVALUATION.json)
- [All paired intervals](CONTRASTS.json)
- [Independent review](REVIEW.json)
- [Post-evaluation overlap counts](ERROR_OVERLAP.json)
- [Separate overlap validation and scope](DIAGNOSTICS.json)
- [Captured tests](TESTS.txt)
- [Preceding native-conditioning results](../fusion_native_conditioning_v1/REPORT.html)
- [Original fusion references](../fusion_replication_v1/REPORT.html)
- [Graph, simple-control and routing code](../../spectral_utils/fusion_graph_conditioning.py)
- [Frozen protocol](../../docs/experiments/FUSION_GRAPH_CONDITIONING_V1.md)
- [Visual guide to our fusion](../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html)
