# Original Joint inverse conditioning: fixed fits and routes

2026-09-07. Bounded development experiment. Fusion remains the contribution.

## Decision question

Does stronger conditioning of the native Joint inverse improve localization
without changing the original feature banks, feature groups or bank routing?
The preceding pair extension increased coverage but its new routing degraded
quality. Condition diagnostics motivate this separate test; they do not prove
that numerical conditioning caused errors. No graph-dose interaction yet.

## Frozen contract

Retain all 110 answers and corrected source groups from fusion_replication_v1:
24 PRMB and 86 PB, five cells, Qwen3-8b. These cached official-answer traces
use one teacher-forced model pass and were already evaluated. All fitting
quantities belong to the same answer; the declared negative-entropy anchor
and fixed engineering constants remain. This is development, not confirmation.

Copy all 33 Step-308 arms exactly, including the 19 Step-306 anchors and
the 14 negative pair-extension controls. Add 12 arms: original moment,
context, single fallback and dual fallback, each with target native inverse
condition 30, 100 or 300. Condition 1000 is their existing Joint0 reference.
All new heads have graph lambda zero. The graph 0.1 and permutation anchors
remain in the tables and paired comparisons. A condition target is not a
graph lambda. Smaller targets usually imply more diagonal ridge, not more
data or a different grouping model. No label-guided per-answer dose selection.

Use original minimum-three feature groups and original fit validity. The
old run did not persist covariance/loadings. Reproduce each valid original
fit on the saved same-answer matrix and partition with the same five starts,
5000-sweep cap and seed 2026090601. Require original fit guards, off-diagonal
misfit, condition-1000 weights/window scores/step scores and native decisions
to replay before admitting new heads. Save the reproduced C, v and u. Do
not use the new pair covariance as a substitute. Invalid original fits stay
invalid even when another rule might admit them.

Keep both original routes exactly: single uses moment Joint then moment IU;
dual tries context Joint before IU when moment Joint is invalid. No route
change based on the new score, a no-error decision or a failed readout.
Normalization, width-eight windows, fit rows, feature signs and official
step mapping stay fixed. Score every window and step. Refit the same native
GMM gate to each new fused score; preserve the fixed-original-IU gate as a
separate diagnostic. Failed decisions remain failures in all-population PB.

## Comparison and reporting

Freeze 69 paired comparisons: every new arm versus its same-bank or routed
Joint0, IU, equal, graph0.1 and permuted-graph references (60); each new dual
versus context equal (3); each new context versus corresponding new moment
(3); and each new dual versus corresponding new single (3). Preserve all
45 arms, all contrasts and failures. No best-dose claim from the roster's
highest point alone. Use common-valid PRMB IDs and all-population PB, pooled
and within-answer AUC, native and fixed-IU decisions, subgroup counts,
coverage and runtime. Retain the older 58-answer table separately.

Use the existing exploratory unadjusted 1000-draw source-group bootstrap,
seed 2026090706; count undefined draws. Require a consistent two-task gain
against matched IU/equal and original Joint/graph references before moving
any candidate toward confirmation. This test does not establish an optimal
condition target or an untouched publication result.

Test known-spectrum ridge behavior and exact parent replay on original
moment/context/IU route cases, plus preservation of failed new readouts and
heads, before freeze. Maximum three CPU workers and 600 seconds per scoring
invocation, checkpointing completed answers and finishing in-flight fits.
Freeze scores before the new evaluator opens labels. Independently review
normalization, original-fit replay, covariance construction, inverse solves,
native gates, step maps, fixed routes, all metrics and point contrasts, plus
representative explicit bootstrap replays. Document reuse of optimizer and
GMM kernels rather than claiming a fully independent implementation.

Produce reviewed HTML and Markdown results. Keep Joint/graph and IU work,
supporting IMM/LOCA/flows/KalmanNet/sampling ideas, corrected-fold multi-answer
refits, full comparator coverage, untouched confirmation and historical24
transfer in scope. No standalone replacement detector is introduced.
