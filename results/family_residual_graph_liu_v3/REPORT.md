# Family-residual graph LIU v3 — post-audit development report

## Result

The nested leave-dataset-family-out procedure changed AUROC versus ordinary
IU-PCR by **+0.018pp** (equal-family bootstrap
95% CI **[-0.041, +0.080]pp**), with
4/8 positive held-out families
and worst-family change -0.131pp.

This primary is the strict self-safe **union-kNN bug repair**. The separately
reported adaptive-only sensitivity changed AUROC by
+0.015pp (CI
[-0.005,
+0.050]pp); allowing union and adaptive
to compete changed it by +0.018pp (CI
[-0.042,
+0.081]pp). These are retrospective
topology-rescue sensitivities, not new confirmation.

The fixed label-free default changed AUROC by
+0.008pp. The corrected union-kNN
DUFS-coordinate arm changed
it by +0.068pp, while frozen Family-NRM's
reference change was +0.277pp.  The nested
procedure recovered 6.6% of the NRM
point gain; `D_0.5` was -0.120pp and `D_0.3` was
-0.065pp.

Secondary AUPRC changed by
+0.013pp under the same held-family
choices (95% family bootstrap
[-0.017,
+0.041]pp).

## Mechanism attribution

Across held-out families, the selected arm minus the matched hybrid-graph U2
arm was +0.018pp.  Its advantage over
the same readout on the ordinary DUFS graph was
-0.014pp.  Direct score diffusion is
evaluated only after selection in the fixed-control phase and cannot affect
this hyperparameter search.

The selected graph met the registered connectivity mechanism criterion in
22/
24 cells. Connectivity did not
filter utility HPO; it is a separate connectivity mechanism gate:
**PASS**.
This connectivity gate is necessary but not sufficient for mechanism
promotion; the post-selection matched and permutation controls are still
required.

## Promotion gates

- `ci_lower_gt_zero`: FAIL
- `point_at_least_0p10pp`: FAIL
- `six_of_eight_positive`: FAIL
- `worst_at_least_minus_0p50pp`: PASS
- `six_nonzero_folds`: PASS
- `d30_lower_nonnegative`: FAIL

Utility promotion: **FAIL**.

## Frozen development finalist

`u2__gunion__e0100__b0000__k07__l0003`

This is a retrospective development estimate: all fits and scores were frozen
without labels, but labels selected the final configuration across the original
eight dataset families.  Existing external datasets can test frozen transfer,
not prospective confirmation.
