# Family-residual graph LIU v1 — development report

## Result

The nested leave-dataset-family-out procedure changed AUROC versus ordinary
IU-PCR by **-0.005pp** (equal-family bootstrap
95% CI **[-0.016, +0.005]pp**), with
3/8 positive held-out families
and worst-family change -0.035pp.

The fixed label-free default changed AUROC by
+0.006pp.  The historical DUFS-LIU arm changed
it by +0.069pp, while frozen Family-NRM's
reference change was +0.277pp.  The nested
procedure recovered -1.7% of the NRM
point gain; `D_0.5` was -0.143pp and `D_0.3` was
-0.088pp.

Secondary AUPRC changed by
+0.004pp under the same held-family
choices (95% family bootstrap
[-0.001,
+0.008]pp).

## Mechanism attribution

Across held-out families, the selected arm minus the matched hybrid-graph U2
arm was -0.006pp.  Its advantage over
the same readout on the ordinary DUFS graph was
-0.024pp.  Direct score diffusion is
evaluated only after selection in the fixed-control phase and cannot affect
this hyperparameter search.

## Promotion gates

- `ci_lower_gt_zero`: FAIL
- `point_at_least_0p10pp`: FAIL
- `six_of_eight_positive`: FAIL
- `worst_at_least_minus_0p50pp`: PASS
- `six_nonzero_folds`: PASS
- `d30_lower_nonnegative`: FAIL

Overall promotion: **FAIL**.

## Frozen development finalist

`u2__e0100__b0000__k15__l0003`

This is a retrospective development estimate: all fits and scores were frozen
without labels, but labels selected the final configuration across the original
eight dataset families.  Existing external datasets can test frozen transfer,
not prospective confirmation.
