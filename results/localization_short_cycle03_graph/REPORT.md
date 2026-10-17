# Localization short cycle 3 — requested answer-only Joint-LIU graph test

Date: 2026-09-07. Status: completed and reviewed.

## Question

Does Claude's strongest graph-regularized Joint candidate,
`internal_joint_liu010`, improve localization when Joint is fitted from the
windows of one answer only?

## Frozen comparison

The score phase reused the exact 30 answers, width-32 window matrices,
per-answer preprocessing, internal partitions and span-maximum step readout
from short cycle 1. On every answer with an available internal partition it
computed:

- the lambda-zero model-inverse reference, reproduced from scratch;
- the requested lambda-0.1 Joint-LIU map with adapted-DUFS feature gates and a
  self-tuning k=7 graph over that answer's fitting windows;
- the same lambda, gates and graph weights after a fixed node relabeling.

No fixed/external groups, fallback, pooled fit or additional lambda entered
this test. Scores were frozen before the evaluator loaded step labels.

## Result

All three Joint maps are available on the same 26/30 answers and strictly
converged on the same 24/30. On the 24 strict answers (977 official steps):

| Method | Step AUROC |
|---|---:|
| IU | **0.72072** |
| Node-permuted graph Joint, lambda 0.1 | 0.69578 |
| Meaningful graph Joint, lambda 0.1 | 0.69188 |
| Joint model-inverse, lambda 0 | 0.69148 |

Paired grouped-bootstrap diagnostics, resampling whole answers:

| Contrast | Delta | 95% interval |
|---|---:|---:|
| Meaningful graph − lambda 0 | +0.00040 | [-0.00566, +0.00841] |
| Permuted graph − lambda 0 | +0.00430 | [-0.00201, +0.01221] |
| Meaningful graph − permuted graph | -0.00390 | [-0.00959, +0.00237] |
| Meaningful graph − IU | -0.02884 | [-0.04851, -0.00560] |

The requested graph does not improve the answer-only Joint method in a
meaningful or attributable way. Its observed gain over lambda zero is 0.04
AUROC points, while the graph with destroyed node identity gains more. The
meaningful graph remains below IU, with the paired interval excluding zero.

## Mechanism and code review

- The lambda-zero window score reproduced the short-cycle-1 artifact exactly
  on all 26 available answers (maximum absolute error 0.0).
- Protocol, runner, fitting module, adapted-DUFS and source-freeze hashes match
  their frozen values.
- All saved arrays are finite and aligned with the official steps. Missing
  partitions remain explicit; there is no fallback.
- An independent `sklearn` calculation reproduced AUROC 0.691881 for the real
  graph, 0.695781 for the permuted graph, 0.691484 for lambda zero and 0.720719
  for IU.
- All 26 graphs were connected. Adapted-DUFS seed variability was bounded
  (maximum mean per-feature seed standard deviation 0.0634).
- The graph maps are close to lambda zero: median score Spearman is 0.9969 and
  median standardized-weight cosine is 0.9866. Real and permuted graph scores
  are also nearly identical (median Spearman 0.9964).

The result uses one fixed graph permutation per answer. More permutations
would estimate control variability more precisely, but are not needed for the
decision: the meaningful graph has essentially zero gain over lambda zero and
is already below IU. This is also an adaptive retrospective pilot on the same
30 answers, so none of its intervals are fresh-confirmation claims.

## Decision

Stop the answer-only Joint graph line for this representation. A larger lambda
is not supported: Claude's pooled experiment already shows degradation from
lambda 0 to 0.1 to 0.5, and the answer-only lambda-0.1 graph is inert here.

Use IU as the current answer-only feature-fusion reference, retain equal as a
close simplicity control, and make the next bounded experiment a trajectory-
axis comparison on a fresh, deterministically frozen set of long answers.
Keep the DUFS token/window-selection proposal in the backlog as a separate
sampling method; it is not validated by this regularization result.

