# Reasoning Localization 0.3662 — Phase-3 context-conditioned DUFS family expert v1

Status: **complete on the opened eight-Qwen ProcessBench development panel;
context premise not supported**

## Question

The completed P3E attribution ladder found a positive but unconfirmed point
estimate when ordinary IU replaced equal compression only inside
`entropy_dynamics`.  This study asks two narrower questions:

1. Does DUFS-LIU improve the dynamics expert when DUFS sees only dynamics?
2. Can the other compact H2 views improve the dynamics expert indirectly by
   defining its donor-sample graph, without entering its output weights?

The second arm is a **context-conditioned family expert**, not a pure
within-family intervention.  Its final score uses only dynamics coordinates,
but its Laplacian is learned from all compact H2 member views.

## Frozen roster and factorization

The member matrix is the exact P3D/P3E compact H2 roster: one entropy-level
view, fourteen entropy-dynamics views including frozen C7, three partition
views excluding `energy_series`, and six top-k views.  Sampled-token energy and
structural views remain absent.

For donor-standardized all-view matrix `F_all` and dynamics submatrix `F_dyn`:

```
F_all -> parameter-free DUFS gates -> donor kNN graph W_all
F_dyn + W_all -> lambda-regularized two-component IU -> dynamics score
dynamics score + unchanged equal level/partition/top-k -> equal outer mean
-> frozen top-ten step reducer -> frozen H0 detector/reranker
```

All fitting uses five grouped donor folds.  Held responses are projection-only.
ProcessBench and PRMBench labels do not select gates, graphs, signs, lambda,
features, or variants.

## Frozen variants

1. `P3F0_DYNAMICS_IU_PARENT`: exact P3E1 ordinary-IU dynamics parent.
2. `P3F1_DYNAMICS_LOCAL_DUFS_LIU`: DUFS gates and graph use only `F_dyn`;
   LIU also solves only on `F_dyn`.
3. `P3F2_DYNAMICS_CONTEXT_DUFS_LIU`: DUFS gates and graph use `F_all`, while
   LIU weights and the emitted family score use only `F_dyn`.
4. `P3F3_DYNAMICS_CONTEXT_PERM_CONTROL`: same as F2, except every non-dynamics
   family is circularly shifted inside each donor response by a deterministic
   non-zero row-specific offset before DUFS.  The control preserves response
   membership, marginal sequences, and cross-channel structure among the
   outside families while destroying their token alignment with dynamics.

No arm changes the entropy-level, partition, or top-k family score, the equal
outer fusion, H0 clean/error decisions, or the top-ten reducer.

## DUFS/LIU contract

- DUFS: parameter-free Eq.-7 adaptation, seeds `11,23,37`, 80 epochs, soft
  RMS-normalized gates, no thresholding.
- Graph: symmetric self-tuning `k=7` nearest-neighbour graph on donor tokens.
- LIU: fixed `lambda=0.1`, the exact ordinary-IU two-component subspace and
  configuration used by P3E.
- Orientation: donor-only correlation with the equal dynamics anchor.
- Exact invariant: `lambda=0` must reproduce F0 for local, context, and
  permuted-context graphs to numerical tolerance `1e-12` before labels open.
- Every scorer copy of a source question stays in one fold.  Token rows are
  structured observations, not bootstrap units.

## Contrasts, inference, and gates

The four primary macro-F1 contrasts are:

1. F1 minus F0: value of family-local DUFS geometry.
2. F2 minus F0: net value of context-conditioned DUFS.
3. F2 minus F1: incremental value of outside-family context.
4. F2 minus F3: aligned context versus the deterministic context control.

They share 20,000 paired whole-question bootstrap draws and one Bonferroni
simultaneous family.  Report exact-error, within-one, clean abstention, W/T/L,
worst cell, graph diagnostics, gate stability, parent aliases, and prediction
flips.  A positive point with an interval crossing zero is
`PROMISING_UNCONFIRMED`, not rejection.

Formal promotion requires F2 minus F0 point delta at least `+0.003`, its
simultaneous lower bound above `+0.003`, exact-error delta at least `-0.010`,
worst-cell delta at least `-0.020`, zero H0 abstention mismatches, all aliases
passing, and finite donor fits.  A claim that outside-family context helps also
requires F2 to beat both F1 and F3 with simultaneous lower bounds above zero.
The opened development population cannot provide fresh confirmation.

## Conditional next step

A single top-k family-local DUFS control may open only after this dynamics
ladder is evaluated and only if F1 or F2 has a positive point estimate versus
F0 without supported material harm or a hard validity failure.  STG/SU,
DUFS pruning, B3, and L-SML are not part of this experiment.

## Completed result

Every score tree was frozen before labels, every lambda-zero and P3E parent
alias has maximum error zero, and every H0 abstention alias is exact.

- F0 dynamics IU parent: `0.366876`.
- F1 family-local DUFS-LIU: `0.367044`, delta `+0.000168` with simultaneous
  CI `[-0.001466,+0.001818]`.
- F2 all-H2 context DUFS-LIU: `0.366870`, delta `-0.000006` versus F0 with CI
  `[-0.001577,+0.001586]`.
- F2 minus F1 is `-0.000174 [-0.000726,0.000000]`.
- F2 minus the aligned-context permutation control is
  `+0.000020 [-0.000986,+0.001021]`.

The local DUFS arm is descriptively positive but effectively tied to its IU
parent.  The all-H2 context does not improve the family expert and does not
beat the required controls.  The context mechanism is therefore not
supported; this is not supported material harm.  The preregistered loose
eligibility rule opens only the single top-k family-local DUFS control.
