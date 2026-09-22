# Reasoning Localization 0.3662 — Phase-3 top-k local DUFS control v1

Status: **complete on the opened eight-Qwen ProcessBench development panel;
no promotion**

## Question

P3E3 found a small positive point estimate for ordinary IU inside the six-view
top-k family.  P3F then found that dynamics-local DUFS was harmless but nearly
identical to its IU parent, while all-H2 contextual geometry was unsupported.
The frozen P3F gate permits one secondary control: does family-local DUFS add
anything to the top-k IU expert?

## Frozen variants

1. `P3K0_TOPK_IU_PARENT`: exact P3E3 top-k ordinary-IU parent.
2. `P3K1_TOPK_LOCAL_DUFS_LIU`: parameter-free DUFS gates and `k=7` graph on
   the six donor-standardized top-k views, followed by dynamics-independent
   two-component LIU at `lambda=0.1` on the same six views.

All other H2 family scores use equal compression.  The equal outer mean, H0
clean/error decisions, and top-ten reducer are unchanged.  Five grouped donor
folds, seeds `11,23,37`, 80 DUFS epochs, donor-only orientation, held-response
projection, and 20,000 paired whole-question bootstrap draws are frozen.

`lambda=0` must reproduce K0 with maximum error at most `1e-12`; K0 must alias
P3E3 to the same tolerance.  No task labels select gates, weights, signs,
lambda, or hyperparameters.

## Evaluation and gate

The single primary contrast is K1 minus K0 in ProcessBench macro F1.  Report
exact error, within-one, clean abstention, W/T/L, worst cell, prediction flips,
gate diagnostics, and aliases.  Promotion requires point delta at least
`+0.003`, CI lower bound above `+0.003`, exact-error delta at least `-0.010`,
worst cell at least `-0.020`, and all validity aliases.  A positive estimate
whose interval crosses zero is promising unconfirmed, not rejected.

This control does not reopen contextual DUFS, partition DUFS, pruning, STG,
B3, L-SML, or PRMBench transfer.

## Completed result

K0 aliases P3E3 exactly and every lambda-zero alias has maximum error zero.
K0 scores `0.365603`; K1 scores `0.365538`.  The paired macro-F1 delta is
`-0.000065 [-0.001657,+0.001623]`, with cell W/T/L `3/1/4` and worst-cell
delta `-0.003210`.  H0 abstention mismatches are zero.

The interval is compatible with a small benefit or harm and the point is
negligibly negative.  The result is `INCONCLUSIVE`, not supported harm, but it
does not support DUFS as an improvement over ordinary IU inside top-k.  No
additional DUFS family or contextual branch opens from this control.
