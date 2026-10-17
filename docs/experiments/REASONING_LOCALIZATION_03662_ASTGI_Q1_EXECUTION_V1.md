# Reasoning Localization 0.3662 — ASTGI-Q1 point-query execution v1

Status: **completed ProcessBench development run; inconclusive macro-F1, exact-error degradation, no promotion**

Freeze note: Q1 macro-F1 delta is
`-0.009507 [-0.019618,+0.000516]`; its separately preserved exact-error delta
is `-0.010644 [-0.020123,-0.001210]`.  Q2 was not opened.

This document freezes the first executable rung of the ASTGI-inspired query
ladder. It is a repository adaptation, not a paper-exact reproduction. The
run changes one operation only: point-query pooling over the already frozen H2
family observations. The detector, ProcessBench population, fit preparation,
top-ten step reducer, threshold protocol, and bootstrap estimator remain
unchanged.

## Question and parent

Does a fixed onset-oriented query over compact H2 family observations improve
first-error localization over the exact H2 equal-family/top-ten parent, or is
any movement explained by query permutation or a boundary-position prior?

Parent: `P3A_H2_EQUAL_OUTER_REFERENCE` (the exact frozen H2 local score). The
H0 response detector and its clean/error abstention decision remain the sole
authority. Q1 only reranks H0 non-abstentions.

## Exact query function

For each real token, the frozen H2 family-risk vector is

```text
z_t = [entropy_level, entropy_dynamics_plus_C7,
       partition_energy_without_energy_series, topk_distribution].
```

The vector is the existing donor-fit standardized H2 risk construction; no
labels or ProcessBench outcomes enter it. The onset query has the fixed,
untuned family prior

```text
q_onset = [0.20, 0.40, 0.20, 0.20], temperature = 1.0,
boundary coefficient gamma = 0.05.
```

For token `t`, `a_t = softmax(z_t / temperature + q_onset)` and the query score
is `r_t = a_t dot z_t + gamma * (1 - u_t)`, where `u_t` is the normalized
position inside its observed step (`u_t=0` at the first token and `u_t=1` at
the last). The same frozen top-ten mean reducer maps token scores to step
scores. This is an analytic point-pooling rule; there is no learned query
embedding, grid, checkpoint selection, or label-based fitting.

The exact controls are score-frozen in the same pass:

1. `MEAN_ALIAS`: equal family mean; must alias the H2 parent at <=1e-12.
2. `QUERY_PERMUTED`: reverse `q_onset`; tests whether the named family roles
   matter rather than a generic softmax reshaping.
3. `NO_BOUNDARY`: the same query with `gamma=0`; tests whether a position prior
   explains a movement.

Controls are diagnostics, not extra tunable candidates. A positive movement
   over uniform chance is not sufficient for a query-mechanism claim.

## Contract and population

- Eight current-common Qwen ProcessBench cells, the same rows, folds, spans,
  source groups and five-fold threshold protocol as the H2 parent.
- Score freeze precedes label import. Held rows are projection-only; scorer
  copies of a source question stay in one fold.
- H0 clean/error decisions are copied exactly; abstention mismatches must be
  zero for every arm.
- Primary inference is 20,000 paired whole-question grouped bootstrap draws,
  with Bonferroni simultaneous intervals across the three macro-F1 contrasts:
  query-minus-parent, query-minus-permuted, and query-minus-no-boundary.
- Practical bounds are +0.003 benefit and -0.005 harm. A CI crossing zero is
  `PROMISING_UNCONFIRMED` when the point estimate is positive, otherwise
  `INCONCLUSIVE`; it is not generic rejection.
- Required secondary outputs: exact error, within-one, clean abstention,
  overall decision, per-cell W/T/L, worst cell, prediction flips and
  step-length strata.

## Promotion and transfer gates

Q1 cannot open Q2 unless it beats its exact parent and both controls under the
registered premise gate, with no exact/clean/worst-cell material regression.
Current labels are development-only, so even a passing development premise
requires fresh-question confirmation. PRMBench transfer is not opened by this
ProcessBench-only run. Any later state-query transfer must be frozen separately
against the H3/C8 PRMBench parent; no cross-task aggregate is allowed.

The Q1 artifact must expose the formula, parent alias, control aliases,
causal/completed-trace boundary, source hashes and plot-data hash in the live
report.
