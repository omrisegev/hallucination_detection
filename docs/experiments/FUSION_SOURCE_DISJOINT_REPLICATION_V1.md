# Fixed fusion recipes on additional source questions

Date: 2026-09-07. Development replication, not untouched confirmation.

## Frozen question

Does the Joint -> IU fallback advantage over original IU recur on additional
source-question groups, and how does it compare with both equal-fusion banks
and the unchanged graph controls? Keep feature fusion central. No feature,
grouping, lambda, sampling, readout or threshold tuning in this stage.

## Population and deterministic selection

Use `localization-cached-v2-sourcegroups-20260907` from
`results/localization_source_group_audit_v1/RELEASE_V2.json`. Exclude the
94 canonical components listed in that audit: documented earlier Codex short
cycles and the 58-answer pilot. The whole v2 cache was already evaluated by
Claude; these exclusions do not establish complete project non-exposure.

Cells, in this order: PRMB Qwen3-8b, PB GSM8K/Math/OlympiadBench/Omni-Math
Qwen3-8b. Use the previous length bins [64,255], [256,1023], [1024,2048].
Select at most eight unique corrected source groups per bin, with no filling
from other bins. Maintain a global selected-group set across cells and bins.
Sort candidates by SHA256(selection_namespace / cell / group_id), breaking
ties by SHA256(row_id); selection_namespace is release_id / pilot_id.
Choose the first available row per group. No labels or fit outcomes enter
selection. Persist all shortages; GSM8K has no remaining eligible long-bin
groups and OlympiadBench has fewer than eight short-bin groups.

This is a length-stratified stress-cohort replication. Changing sample size
and group identity is explicit. Compare methods within the new cohort and
show the old cohort in a separate panel; do not call a cross-cohort score
difference an algorithmic gain. The same release, targets, scoring rules and
metric functions are used. No candidate wins merely by having a larger score
than it had on the earlier sample.

## Methods and scoring

Exactly 19 arms: five original-moment-bank cores and five context-bank cores
(equal, IU, Joint lambda 0, graph lambda .1, permuted graph lambda .1), entropy,
six Joint/IU fallbacks (single/dual times three Joint variants), and two
dual-routed equal/IU controls. Expanded-K variants are historical references
from the context pilot and are not part of this fixed legacy-K replication.

Reuse the nine primitive streams, 27 moment or context coordinates, width
eight, original orientation/normalization, K={3,4,6,8}, minimum group size
three, Joint fit validity guards, target inverse condition 1000, and native
GMM + peak decision. Model fitting stays inside each answer. Keep the declared
negative-entropy anchor and label-free hyperparameters. Graph permutation
identity uses `localization-cached-v1-20260907 / cell / row_id / moments27_local8`;
the new grouping release must not change the old scoring namespace.

Single: moment Joint when valid, otherwise moment IU. Dual: moment Joint,
then context Joint, then moment IU. A selected no-error decision or failed
readout does not trigger further routing. Invalid fits/decisions stay visible.
The shared original-IU gate remains a diagnostic only. If that reference
gate is unavailable on a new answer, the fixed-IU diagnostic is unavailable
for that answer, while native decisions can still be evaluated. Never filter
the cohort based on IU or Joint fit success.

Both caches contain teacher-forced, one-pass gray-box telemetry of supplied
official answers. No new generation or inference is required. The new scorer
must replay the 19 retained methods in three existing routing cases before
the cohort is frozen. A degenerate-answer test must preserve failures.

## Evaluation and decision

Read labels only after predictions are frozen. Report the existing pooled
PRMB AUC and mean within-answer AUC, PB macro harmonic F1, per-cell clean/error
successes, fit/decision coverage, routing counts, length support and runtime.
PB includes all selected answers; invalid decisions count as failures.
PRMB paired comparisons use common valid IDs and disclose their counts.

The 38 paired contrasts are frozen in the runner: the preceding 32 fallback
comparisons (now including both context-equal checks), context-vs-moment
equal/IU/Joint0/graph, and context Joint0/graph versus context equal. Use
1,000 source-group bootstrap draws, seed 2026090706, retaining undefined draws
and the fixed-IU-gate diagnostic. Intervals are unadjusted and exploratory.
Do not tune on this cohort after reading results. No winner is guaranteed.

Review group exclusion/selection without labels, raw window extraction,
feature/normalization/weight projection, Joint validity and grouping guards,
native GMM and step mapping, exact fallback inheritance, direct label joins,
all endpoints and representative independent explicit bootstrap comparisons.
The earlier capsule audits support the fusion kernels; disclose reused
kernels and distinguish score reconstruction from a full independent refit.

At most three CPU workers and 1,200 seconds of scoring per invocation, with
one answer task submitted per worker at a time. Checkpoint completed answers;
on reaching the cap, finish currently running tasks and pause remaining
submissions. Do not kill a valid in-flight fit on an observation timeout.
Unexpected exceptions stop the stage for investigation, not silent imputation.
Complete numerical review and HTML/Markdown reporting at this stage boundary.
Corrected-fold multi-answer replay, further Joint/IU development, supporting
methods, the full comparator registry, untouched confirmation and 24-cell
transfer remain open.
