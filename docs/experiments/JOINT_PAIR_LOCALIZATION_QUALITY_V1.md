# Checked pair-group Joint: matched localization quality

2026-09-07. Development experiment, not untouched confirmation.

## Frozen question

Does the greater fitting coverage from Step 307 improve localization when
we keep our Joint fusion, its native inverse and graph controls? Test the
checked pair representation, not naive pair admission. Use
`joint_pair_jacobian.fit_joint_pairs_checked`, including the zero-product
nuisance correction. Preserve all prior frozen code and scores.

## Fixed data and methods

Reuse all 110 answers and corrected source groups of fusion_replication_v1:
24 PRMB and 86 PB, Qwen3-8b, one teacher-forced gray-box model pass on each
fixed official answer. All are development data already evaluated in the
project. There is no new source-disjoint confirmation claim. Refit from
each answer only; no external fitted signs, groups, gates or thresholds.
The declared negative-entropy anchor remains part of the engineering recipe.

Exactly 33 reported arms: all 19 frozen replication anchors, plus 14 new
arms. New pure moment/context Joint each uses lambda 0, graph lambda .1,
and permuted-graph lambda .1. New single/dual fallbacks use those three
Joint variants, and new dual-routed IU/equal controls use the same bank
eligibility. Keep original moment IU as the last fallback. Routes depend
on fit validity only; no-error decisions and readout failures do not switch
the selected method. Both pure failures and composite behavior are reported.

Reuse width eight, the 27-coordinate moment/context banks, K={3,4,6,8},
four chronological blocks, held admissibility .95, five optimizer starts,
5000-sweep cap, fixed seeds and native inverse condition 1000. Lower the
group minimum to two and use the audited pair covariance/native-map and
Jacobian guards. No wider K or condition-number dose. Reuse same-answer
parent gates where present; otherwise compute the same DUFS adaptation
(seeds 0/1/2, 120 epochs). Graph k=7 and the old scoring identity namespace
remain fixed. Parent features and scores must replay, and new fits must
match the unlabeled audit's covariance/validity before quality evaluation.

Score every window and official step with the unchanged mapping, native
GMM binary gate and peak location. The fixed-original-IU gate is diagnostic
only. Invalid decisions count as failures on PB; paired PRMB comparisons
use common valid IDs. Report full coverage and within-answer AUC beside
pooled PRMB AUC and PB macro harmonic F1. No threshold tuning from labels.

## Comparisons and execution

The runner freezes 63 distinct comparisons before predictions or new
quality evaluation: each pure pair arm against its minimum-three parent,
same-bank IU/equal; graph-zero/permutation contrasts in both banks and
both fallback policies; fallback arms against their old policy, moment IU
and context equal; dual-versus-single and matched dual IU/equal; routed
IU/equal versus historical and always-bank controls; paired context-versus-
moment heads. Report all 33 methods and all 63 contrasts, with the prior
110-answer endpoints replayed exactly and the older 58-answer results as
separate historical context. Exploratory unadjusted 1,000-draw source-group
bootstrap, seed 2026090706, four existing endpoints and undefined counts.
Do not declare a winner from one favorable comparison among this roster.

Test unchanged-partition score replay, genuine pair-fit audit agreement,
infeasible-pair routing and readout-failure preservation before freezing.
Use at most three CPU workers, a 1200-second scoring cap per invocation,
checkpoint completed answers and finish in-flight fits at the cap. Freeze
all predictions before reading the benchmark labels in the evaluator.
Independent review must verify parent replay, direct labels, raw feature
identity, pair fits/Jacobian, native inverse/graph heads, GMMs, routing,
all endpoints/point contrasts and representative explicit bootstraps.
Produce HTML and Markdown results, with failures and limitations. Greater
fit coverage alone is not the success criterion. Keep the full fusion
mandate, corrected-fold multi-answer replay, wider supporting tracks,
untouched confirmation and historical24 transfer open.
