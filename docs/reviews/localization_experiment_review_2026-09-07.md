# Localization experiment review and continuation decision

Status checked: 2026-09-07 00:35 Israel time.

**Subsequent scope correction, September 7:** The recommendation below to
close Joint/graph development is superseded by Omri's explicit research
mandate: `../experiments/LOCALIZATION_RESEARCH_MANDATE_20260907.md`.
The results apply to the tested roster/width/lambda, not to every alternative
Joint representation. An independent follow-up audit also checks borrowed
feature signs, actual import lineage and source-group uncertainty. Preserve
this original review as the historical decision, with these qualifications.

## Decision

The answer-only direction remains promising, but the current evidence supports
IU/equal rather than Joint. The exact graph suggestion has now been tested. It
does not improve answer-only Joint, and a node-permuted graph performs slightly
better. Stop Joint graph/lambda development for this representation.

The next short cycle should use a fresh deterministic set of long answers,
confirm answer-only IU versus equal, and only then compare a small frozen set of
trajectory readouts. Do not force K above 3, start a broad target-condition
sweep, or transfer a selected localizer to the historical 24 cells yet.

## Codex short cycles

| Cycle | Question | Strict coverage | Main result | Conclusion |
|---|---|---:|---|---|
| 1 | Can one answer's windows fit Joint, IU and equal? | Joint 24/30; IU/equal 30/30 | common-24 AUROC: IU 0.72072, equal 0.70134, Joint 0.69148 | Answer-only fitting is viable; current Joint is fragile and below IU. |
| 2 | Is learned feature clustering the Joint bottleneck? | fixed-group Joint 27/30; fixed-group continuous 30/30 | fixed Joint 0.65721 available; continuous 0.68916; IU 0.70070 | Fixed groups improve coverage, not ranking. Joint factor/model-inverse remains the larger problem. |
| 3 | Does the requested Joint-LIU graph at lambda 0.1 help? | graph/lambda-zero 24/30 strict | graph 0.69188, lambda zero 0.69148, permuted graph 0.69578, IU 0.72072 | No attributable graph gain. IU remains higher. |

Cycle 2 was a useful diagnostic proposed by Codex, but it was not the graph
test Omri requested. That scope mistake is recorded explicitly in its report.
Cycle 3 is the direct correction and implements the exact meaningful-graph,
lambda-zero and node-permutation comparison.

The cycle-3 paired intervals are:

- meaningful graph minus lambda zero: +0.00040, 95% interval
  [-0.00566, +0.00841];
- meaningful graph minus permuted graph: -0.00390
  [-0.00959, +0.00237];
- meaningful graph minus IU: -0.02884
  [-0.04851, -0.00560].

The graph maps are close to lambda zero (median score Spearman 0.9969), and the
meaningful and permuted graph scores are also almost identical (median 0.9964).
This explains why changing the graph does not create a stable localization
gain.

## Claude optimization-v2 review

Claude's completed scientific outputs tell the same mechanism story on the
pooled/folded PRMBench protocol:

- tuned Joint/L-SML reaches 0.6724 versus tuned IU 0.6665 on PRMBench;
- the ungated model-inverse lambda-zero reference is higher at 0.6734;
- the node-permuted graph is also higher than the meaningful graph at 0.6732;
- `internal_joint_liu010` is 0.6724 and lambda 0.5 falls to 0.6683;
- on ProcessBench, tuned Joint is 0.3437 versus IU 0.3493, and the internal
  Joint/continuous successors are catastrophic;
- Module B's selected learned trajectory composition harms the frozen span-max
  reference. The fixed max/mean blend is positive development evidence, but it
  is not a clean answer-only label-free winner.

The apparently successful Claude graph row therefore does not show a graph
mechanism. The later contrast output already confirms:

- permuted graph minus tuned IU: +0.0067 [0.0036, 0.0097];
- lambda zero minus tuned IU: +0.0069 [0.0037, 0.0100];
- meaningful lambda-0.1 graph minus lambda zero: -0.0010
  [-0.0015, -0.0005].

The gain comes from the repaired model-inverse head relative to the old
hierarchical head in Claude's pooled setting. It does not come from LIU graph
regularization.

## What Claude is currently running

The active process is
`scripts/joint_lsml_optimization_v2/report_v2.py --stage contrasts`, PID
145884. It started at 22:50 on September 6 and is still consuming CPU. This is
post-evaluation reporting: six PRMBench and three ProcessBench paired
bootstraps. It is not a target-condition optimization experiment.

At the status check, four of the six PRMBench contrasts were logged and
`evaluation/report_contrasts.json` had not yet been written. The process is
alive and its error log is empty. It should be allowed to finish. Claude's
branch is `claude/joint-lsml-optimization-v2`, 17 commits ahead of its remote;
the report script and result directory are still untracked because Claude
planned to render, document, commit and push after the contrasts finish.

Claude also completed a two-fold label-free K diagnostic. K=4/5/6/7 were all
inadmissible because at least one group had fewer than three features; K=8 is
structurally impossible with 23 features under the same minimum group size.
Forcing K above 3 would therefore turn the current result into blocked fits. It
would not repair the method. Claude has not started the requested
target-condition scan.

## Code and evidence audit

For Codex cycle 3:

- lambda zero reproduces cycle 1 exactly, maximum absolute error 0.0;
- all protocol, source and implementation hashes match their score freeze;
- all stored arrays are finite and aligned to official steps;
- missing partitions and unconverged fits remain explicit;
- independent `sklearn` AUROC recomputation matches the report.

The main evidence limitation is adaptivity. Cycles 2 and 3 were designed after
cycle-1 results on the same 30 answers were known. They are mechanism
diagnostics, not fresh confirmation. One graph permutation per answer is also
insufficient to estimate the full null distribution, although more
permutations cannot rescue the observed near-zero graph gain over lambda zero.

Cycle 2 has a minor documentation gap: its code applied the existing
three-unit small-m guard to one answer whose spilled-energy group lost one
inactive feature. The behavior is hash-bound and does not affect the Joint
graph conclusion.

## Continuation instructions

1. Let Claude's current contrast process finish. Verify that
   `report_contrasts.json`, the final rendered report, HISTORY/PROGRESS update,
   commit and push all appear. Do not start another heavy process in his
   worktree before that closure.
2. Do not force K > 3. If grouping is revisited later, the estimator must first
   support groups smaller than three or use a different identification model.
   Fixed groups already showed that coverage alone does not recover accuracy.
3. Do not test a larger graph lambda. Both Claude's pooled sequence and the
   answer-only graph control reject that direction.
4. Freeze a fresh second cohort of 30 long PRMBench answers, excluding the
   current IDs. Re-run only answer-only IU and equal with the existing width-32
   span-max contract. This is the cheapest confirmation of the 0.70 result.
5. If IU confirms, freeze IU as the feature-axis fuser and run one small
   trajectory/readout comparison on another untouched cohort: span max,
   span mean, and the predeclared 0.5 max/mean blend transferred from Claude.
   Report step AUROC, first-error localization, tolerance-one hit and clean
   abstention. No label-selected alpha.
6. Keep target-condition optimization as a separate pooled-Joint research
   branch. It is relevant to Claude's model-inverse head, but it should use
   nested training selection and untouched confirmation data. It is secondary
   to the current answer-only localization goal.
7. Keep DUFS token/window selection as a distinct backlog experiment. The
   failed graph regularizer neither validates nor invalidates point selection.
