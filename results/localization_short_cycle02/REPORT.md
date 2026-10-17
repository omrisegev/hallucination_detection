# Localization short cycle 2 — fixed-group diagnostic

Date: 2026-09-07. Status: completed and reviewed.

## Scope correction

This experiment answers the assistant-proposed question of whether data-derived
feature clustering caused the answer-only Joint failure. It does **not** test
Omri's requested graph-regularized `internal_joint_liu010` variant. That graph
test is a separate next short cycle with a meaningful-graph and permuted-graph
comparison.

## Frozen design

The experiment reused exactly the 30 answers, width-32 non-overlapping windows,
per-answer preprocessing and span-maximum step readout from short cycle 1. Four
label-free groups were fixed by feature source: entropy trace, sampled-token
spilled energy, energy series, and next-token distribution. New scores were
frozen before the evaluator opened the step labels.

The two new arms were fixed-group Joint with the lambda-zero model-inverse map,
and continuous L-SML with the same four groups. Existing internal-group Joint,
IU and equal scores were carried unchanged as references.

## Results

| Method | Available | Strictly converged | Pooled available step AUROC |
|---|---:|---:|---:|
| Internal-group Joint lambda 0 | 26/30 | 24/30 | 0.66999 |
| Fixed-group Joint lambda 0 | 30/30 | 27/30 | 0.65721 |
| Fixed-group continuous L-SML | 30/30 | 30/30 | 0.68916 |
| IU | 30/30 | 30/30 | 0.70070 |
| Equal | 30/30 | 30/30 | 0.69648 |

Strict pairwise comparisons resample whole answers:

| Contrast | Answers | Left AUROC | Right AUROC | Delta | Grouped 95% interval |
|---|---:|---:|---:|---:|---:|
| Fixed-group Joint − IU | 27 | 0.62577 | 0.67159 | -0.04582 | [-0.07203, -0.01768] |
| Fixed-group Joint − internal Joint | 21 | 0.63597 | 0.65295 | -0.01699 | [-0.03806, +0.01162] |
| Fixed-group continuous − fixed-group Joint | 27 | 0.65654 | 0.62577 | +0.03078 | [+0.00236, +0.05546] |
| Fixed-group continuous − IU | 30 | 0.68916 | 0.70070 | -0.01154 | [-0.02919, +0.00225] |
| IU − equal | 30 | 0.70070 | 0.69648 | +0.00422 | [-0.02093, +0.03029] |

## Interpretation

Fixed groups remove the partition-availability failure: strict Joint coverage
rises from 24 to 27 answers and a finite score exists for all 30. They do not
repair ranking. Fixed-group Joint is worse than IU and is not better than the
original internal-group Joint on their common strict answers.

Continuous L-SML is better than the Joint model-inverse under exactly the same
fixed partition. This points to the current Joint factor/model-inverse stage,
rather than clustering alone, as a performance bottleneck. IU remains the
descriptive leader, while IU and equal are not separated by this small pilot.

## Code and finding review

- The protocol, fitting module, runner and short-cycle-1 source hashes match the
  values frozen before evaluation.
- All 30 score files exist; every stored array is finite and aligned to the
  official step count.
- An independent `sklearn` AUROC calculation reproduced every reported
  pairwise value.
- Missing and unconverged fits remain explicit; no fallback score is inserted.
- Fixed-group Joint converged strictly on 27/30 answers. Continuous L-SML
  produced an admissible score on 30/30.
- Score stability under the two quarter-omission refits was high for Joint
  (minimum Spearman 0.9535) but weaker for continuous L-SML (minimum 0.7708).
  Several raw-coordinate weight maps had negative cosine despite similar score
  rankings, showing that the highly collinear feature basis does not identify a
  stable coefficient interpretation.

One documentation gap was found. The code used the repository's existing
`small_m_guard=True` for continuous L-SML. It had no numerical effect on 29
answers; on one answer an inactive spilled feature left a three-feature group,
so that within-group stage used equal standardized weights. The registered
protocol did not spell out this guard. The exact behavior is hash-bound in the
frozen code, but the continuous-L-SML result should retain this caveat.

This is an adaptive retrospective diagnostic: its design was chosen after
short-cycle-1 results on the same 30 answers were known. It cannot serve as
fresh confirmation or a publication claim.

## Decision

Do not continue a fixed-group Joint sweep. The experiment improves our
diagnosis but does not answer the requested graph hypothesis. Test exactly
`internal_joint_liu010` on the answer-only window matrix, with lambda zero and
a node-permuted graph as controls. Unless the meaningful graph beats both,
prioritize IU/equal feature fusion and move the short-cycle program to the
trajectory axis on a fresh answer subset.

