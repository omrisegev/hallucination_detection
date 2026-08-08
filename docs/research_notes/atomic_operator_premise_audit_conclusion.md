# Atomic operator premise audit: conclusion and next junction

**Date:** 2026-08-07
**Protocol:** `atomic-operator-premise-v2-2026-08-07`
**Run fingerprint:** `db50c5ff15349c29f7c0570aa717d2a1047466b02a5beefefc1cb66f96b18566`
**Decision:** Stop AOG-IU-PCR for the registered proxy. Do not run Phase 1.

## Question

The previous frozen benchmark showed that sample-specific SpecRaGE weights and
stable micro-view clusters did not improve IU-PCR. Atomic views were safer, and
there was a small signal in global weighting. This suggested a simpler
DUFS-inspired idea: apply one global continuous gate directly to the atomic
feature Laplacians inside the IU-PCR subspace.

Before building that learner, Phase 0 asked a smaller question:

> Can a fixed score computed without correctness labels identify which atomic
> feature Laplacian will improve IU-PCR?

The score combined cross-fitted smoothness agreement, bootstrap operator
stability, rank-change stability, and bounded actuation. It was computed and
hashed for all 24 cells before correctness labels were opened.

## Main result

The answer is no for this proxy.

| quantity | result |
|---|---:|
| median within-cell Spearman(proxy, atomic usefulness) | -0.312 |
| equal-family mean association | -0.032 |
| family-bootstrap interval | [-0.319, +0.249] |
| feature-identity permutation p-value | 0.690 |
| exact eight-family sign-flip p-value | 0.582 |
| families with positive top-minus-bottom contrast | 3 of 8 |
| top-proxy atom, equal-family change vs IU-PCR | -1.178 pp |
| top-proxy atom, cell-macro change vs IU-PCR | -0.838 pp |
| top-proxy wins/losses | 7 / 17 |
| worst top-proxy loss | -3.658 pp |
| label-only oracle atomic headroom | +0.483 pp equal-family; +0.447 pp cell-macro |

Only 3 of the 15 preregistered continuation gates passed. The full report is
in `results/atomic_operator_premise_audit_v2/REPORT.md`.

The oracle uses the same labels for choosing and evaluating the best atom. It
is therefore an optimistic headroom diagnostic, not an achievable or
generalizing detector.

## What problem was exposed

This is a **label-free target-identifiability problem**, not an optimization or
numerical problem.

The static feature covariance can tell us that a feature graph is reproducible,
agrees with an IU-PCR pseudo-score, and changes the fitted ranking. It cannot
tell us whether that geometry represents correctness or a nuisance such as
answer length, generic difficulty, confidence scale, or model-specific trace
shape. Removing the candidate feature and its near-correlated clones does not
create an independent semantic target; the pseudo-target still comes from the
same feature system.

This explains how a stable score became harmful. The proxy rewarded operators
that produced a large and reproducible change. In these data, large actuation
was usually negatively related to usefulness. `full_actuation` had a median
within-cell association of -0.441, and rank-change reproducibility had -0.377.
The proxy converged accurately, but it converged to the wrong ordering.

## Why parameter tuning cannot rescue this result

The registered grid recomputed the complete proxy for all nine combinations
of graph neighbourhood size and Laplacian strength:

- `k` in 7, 15, 30;
- `lambda` in 0.3, 1, 3.

Every combination had a negative median association and a negative
top-minus-bottom contrast. The least negative point, `k=30, lambda=0.3`, still
had median Spearman -0.108. Smaller `lambda` reduced the damage but did not
reverse the feature ordering. Larger `lambda` amplified the same error.

The proxy ranking was already nearly converged after four subsamples: its
median rank agreement with the 40-subsample result was 0.990. More bootstrap
samples therefore improve numerical precision, not target information. The
0.90, 0.95, and 0.99 duplicate-threshold diagnostics also gave no positive
evidence, and they change only the alignment component.

Parameters still exist, but they have limited roles:

| parameter | what it changes | conclusion from this audit |
|---|---|---|
| graph `k` | local versus broad feature neighbourhoods | no tested value identified useful atoms |
| `lambda` | strength of the Laplacian correction | lower values reduce harm; they do not fix selection |
| duplicate threshold | which near-clones are removed from cross-fitting | no positive sensitivity result |
| bootstrap count/fraction/cap | precision of stability estimates | already converged; not the bottleneck |
| proxy component weights | definition of the target used by a future learner | must not be tuned post hoc on these labels |

A new experiment may change these values only after it introduces a new source
of target information and freezes a new hypothesis. Searching this data for a
better proxy mixture would overfit a failed objective.

## Connection to the earlier experiments

1. **DUFS-LIU:** stable graph regularization tied IU-PCR. It did not prove that
   a Laplacian is target-relevant.
2. **CA-SpecRaGE:** sample-specific weights did not beat global or permuted
   controls. Local reliability was not identified.
3. **Fusion-aware micro-views:** the clusters were reproducible and harmful.
   Stable grouping was not enough.
4. **Atomic Phase 0:** the simpler global premise also failed. Stability,
   agreement, and actuation still did not identify correctness-relevant
   operators.

Together, these results close the current DUFS/graph-gating line. They do not
show that every graph regularizer is impossible. They show that the existing
static, label-free geometry is missing the information needed to choose one.

## Next scientific junction

Keep confidence-oriented U-PCR/IU-PCR as the incumbent. Keep DUFS-LIU, uniform
atomic fusion, and the atomic operator machinery as controls and diagnostics.
Do not build the planned AOG gate learner.

The next premise must add an **independent interventional self-supervised
signal**. Candidate sources include repeated generations, benign prompt or
decoding perturbations, evidence-conditioned answers, or semantic
answer-consistency views. These interventions may reveal which signals remain
reliable when the generated answer changes. They add information that cannot
be recovered from the covariance of the same static feature matrix.

The main failure condition is systematic hallucination: a model can be
consistently wrong under every perturbation. Therefore consistency cannot be
assumed to equal correctness. It must first pass a frozen premise audit with
leave-one-family-out transfer, absolute improvement gates, and the same tail
safety checks used here.

The mathematical backbone remains Tenzer et al.'s continuous U-PCR/IU-PCR.
DUFS remains useful only as a differentiable optimization mechanism after a
valid self-supervised target exists. SpecRaGE and DEEM do not solve the missing
target by themselves: they can model dependence, but unlabeled dependence is
not the same as hallucination correctness.

The next work should therefore be a short literature-and-data design phase for
an interventional target, followed by another premise test. It should not be a
new fusion learner or a larger hyperparameter search.

## Audit status

The label-free fit used a physically stripped bundle and completed all 24
cells. Source files, input files, dependency versions, and every score and
diagnostic artifact were verified before labels were read. An independent
pre-run review approved the v2 protocol. A separate post-run review reproduced
the headline numbers and approved the negative conclusion. No result is an
external confirmation because all 24 cells are retrospective development
data.
