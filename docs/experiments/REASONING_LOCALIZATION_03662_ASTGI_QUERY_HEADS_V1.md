# Reasoning Localization 0.3662 — ASTGI-Inspired Query Heads v1

Status: **Q1 completed with no promotion; Q2--Q4 closed as NOT_RUN_BY_GATE (Steps 345/346)**

Freeze note: the later-rung design below remains historical.  Q1 did not pass
its premise gate, so Q2--Q4 and both transfer tasks were never evaluated.

## Claim boundary

This is an ASTGI-inspired repository adaptation, not a paper-exact
reproduction and not evidence that a graph improves localization. The proposal
supplied to the project attributes a strong mean-pooling ablation to ASTGI, but
an exact paper citation and paper-faithful component map have not yet been
registered. That statement is motivation only and cannot support a gate or a
claim in the report.

The branch is motivated primarily by repository evidence:

- the matched historical 2x2 audit attributes the favorable end-to-end point
  movement mainly to the H0 detector, not to an isolated H3 localizer;
- H2 is the raw-best current compact localizer in the matched historical
  replay but remains unconfirmed; and
- H3/C8 has a supported, task-specific PRMBench ranking advantage while its
  ProcessBench onset contribution is unresolved.

The objective is therefore one shared compact observation representation with
three explicitly separate roles, never one scalar averaged across tasks:

```text
compact token/family observations
  -> frozen response-detection head D0
  -> onset query head O        -> ProcessBench first error
  -> state-error query head S  -> PRMBench every-step ranking
```

## Frozen role parents

- `D0 = P2C_F6_TOP10_REFERENCE`: current H0 response detector and abstention
  decision. Candidate heads must copy it exactly.
- `O0 = P3A_H2_EQUAL_OUTER_REFERENCE`: H2 onset parent: entropy level;
  entropy dynamics plus C7; partition energy without `energy_series`; top-k
  distribution; no sampled-token energy.
- `S0 = P2F_H3_EQUAL_C8_RERANK_PRM`: H3 state-ranking parent on PRMBench, with
  C8 retained as an external state expert.

The related `P2D_H3_EQUAL_C8_RERANK` ProcessBench record remains a diagnostic
comparator, not the onset parent. Current ProcessBench and PRMBench labels are
already development-open; no result on them can be fresh confirmation.

## Shared observation object

Each real, unpadded observation is

```text
v[i,t,f] = (source group i, token/fixed-bin time t,
            frozen family/channel f, value x, step-boundary relation b, mask)
```

Only the H2 compact family roster plus the separately named C8 state stream is
eligible. All 29 views, structural trace-length replication, and new DSP
channels are forbidden. Right padding may exist only inside batches and is
excluded by an explicit mask from every fit, attention denominator,
neighborhood, loss, and score.

Step-boundary relation may encode observed token offset from the current step
start/end for completed-trace localization. A future early arm may use only
current-step/prefix-relative coordinates; total-answer fractions, future step
boundaries, and unseen suffix statistics are forbidden.

## Ordered Q ladder

### Q1 — point query pooling, no graph

`P3T_Q1_POINT_QUERY` applies one frozen query-conditioned pooling rule directly
to observations. It introduces no neighborhood and no message passing.

- Onset query: emphasizes new positive innovation, local jump, C7
  burst/rebound, boundary crossing, and newly emerging family disagreement.
- State query: emphasizes persistent residual/innovation, sustained high risk,
  persistent family disagreement, and the frozen C8 state contribution.

The generic form is

```text
a[q,i] = masked_softmax(g(relative_time, boundary_relation,
                          channel_embedding, observation, query_type))
```

Before execution the registry must freeze the exact analytic function or the
task-blind fitting objective, dimensions, initialization, regularization,
orientation, seeds, and tie breaks. ProcessBench and PRMBench labels may not
fit or choose `g`, a query embedding, a feature, or a reducer. The detector,
outer family fusion, and task-specific parent reducers remain unchanged.

Mandatory controls: exact O0/S0/D0 aliases; mean pooling; the same point
representation with query identity permuted; feature permutation; and removal
of step-boundary relation. Q1 must beat or establish preregistered
noninferiority to its own task parent; task metrics are never averaged.

### Q2 — donor-learned causal coordinates

`P3T_Q2_LEARNED_COORD` may open only after Q1 passes its registered premise.
It learns compact family-by-time coordinates on donor/calibration responses
using exactly one frozen task-blind objective chosen before task labels:
masked feature reconstruction, next-observation prediction, or cross-family
prediction. The executable registry may instantiate only one.

Reconstruction or prediction quality is a fit diagnostic, never evidence of
better localization. Coordinates, standardization, dimension, orientation,
and strength are frozen before evaluation. Held responses are projection-only
and all scorer copies of a question share one fold. The zero-coordinate
strength setting must alias Q1 exactly.

### Q3 — adaptive causal neighborhood

`P3T_Q3_CAUSAL_NEIGHBOR` may open only after Q2. Neighbors obey
`t[j] <= t[i]`; the single K value and distance are selected by a frozen
label-free stability rule or nested donor split, never task metrics. Q3 uses
relation-aware weighted aggregation but no iterative propagation.

Mandatory controls are temporal chain only, time-only nearest neighbors,
cardinality-matched random neighbors, time permutation, feature permutation,
Q2 query-only, mean pooling, and exact zero-strength Q2 alias. Improvement
over uniform chance is insufficient. Q3 must beat Q2 and every topology/null
control relevant to its claim. This is a new premise and does not reopen the
failed STEP-CUT conductance graph.

### Q4 — one propagation layer

`P3T_Q4_ONE_LAYER` may open only if Q3 establishes neighborhood value. It adds
one residual, relation-aware message-passing layer followed by layer
normalization. No depth sweep is permitted. Zero propagation strength must
alias Q3. Oversmoothing, random-neighbor, feature-permutation, and
time-permutation controls are mandatory.

Only a Q-ladder survivor may become an input parent for
`P3T_T3_TWO_AXIS_LOWRANK`. Hierarchical family fusion and query aggregation
remain separate factors: the first Q experiment uses the frozen H2/H3 parents,
not a jointly tuned hierarchical expert. A later contrast may change their
order one factor at a time.

## Fit, leakage, and evaluation contract

- All standardization, embeddings, self-supervised losses, coordinate maps,
  query parameters, K, neighborhood strength, orientation, and stopping use
  donor/calibration rows only.
- Held responses are projection-only. Whole source questions are bootstrap and
  fold units; scorer copies never split across folds.
- ProcessBench and PRMBench labels are evaluation-only. No task label selects
  a loss, head, hyperparameter, direction, or checkpoint.
- Q1--Q4 use a compact frozen roster and one registered change per rung. There
  is no ASTGI-by-DSP-by-hierarchical factorial search.
- Missing, degenerate, single-class, or failed-control states are explicit;
  they are never zeros.

ProcessBench reports onset F1, exact, within-one, clean abstention, W/T/L,
worst cell, and paired whole-question intervals versus O0. PRMBench separately
reports AUROC/AUPRC, error-family and source-stratum behavior, worst family,
and grouped intervals versus S0. D0 abstention mismatch must be zero.

A dual-head architecture may advance only when the onset head is
noninferior-or-better than O0 under a preregistered margin, the state head is
noninferior-or-better than S0 under its separate margin, and D0 is unchanged.
It may be classified as a ProcessBench or PRMBench specialist when only one
head improves. There is no combined score and no averaging away a task loss.

## Phase 5 boundary

Only the onset head can transfer automatically to early detection, and only
after a separate prefix-safe freeze. Its observations, coordinates,
neighbors, pooling, normalization, and detector must be suffix invariant. C8
state ranking and the PRMBench state head do not transfer automatically.

## Deferred and forbidden in this branch

- end-to-end supervised ASTGI on ProcessBench or PRMBench labels;
- all 29 token views;
- more than one propagation layer or a depth sweep;
- K, embedding-size, filter, checkpoint, or head selection by task metrics;
- joint ASTGI × DSP × hierarchical-U-PCR search;
- bidirectional/future edges in the causal or early lanes;
- interpreting self-supervised reconstruction improvement as localization
  evidence.
