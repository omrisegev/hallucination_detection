# Reasoning Localization 0.3662 — Phase-3 dynamics STG-SU v1

Status: **completed development diagnostic; support mechanism not established and no promotion (Steps 344/346)**

Freeze note: the prospective execution contract below is retained for audit.
The completed result is STG-SU minus dynamics-IU
`-0.000114 [-0.002964,+0.002697]`; the learned-support controls do not establish
the proposed mechanism, and no PRMBench transfer opened.

## Question and boundary

Ordinary IU inside `entropy_dynamics` is the strongest compact family expert
found by P3E.  DUFS geometry did not improve it.  This study asks whether a
different dependence model—SU-PCR with a fold-stable stochastic-gate sparse
error support—can improve that same family without changing any other system
component.

This is an adaptation of the corrected final-answer STG-SU mechanism to one
token-local family.  It is not evidence that the historical detector already
transfers to localization.

## Frozen variants

1. `P3S0_DYNAMICS_IU_PARENT`: exact P3E1 dynamics-IU parent.
2. `P3S1_DYNAMICS_CANONICAL_SU`: canonical fixed-threshold SU-PCR on the same
   fourteen dynamics views.
3. `P3S2_DYNAMICS_STG_SU`: replace only canonical SU's support extractor with
   donor-nested STG consensus; then refit the unchanged SU-PCR predictor.
4. `P3S3_DYNAMICS_STG_PERMUTED_SUPPORT`: deterministically permute the learned
   STG support across dynamics feature labels before the same SU refit.
5. `P3S4_DYNAMICS_RANDOM_SUPPORT_CONTROL`: arithmetic mean of twenty
   cardinality-matched random-support SU refits.  Individual random arms are
   controls and cannot enter the leaderboard.

Every arm emits one dynamics family score. Entropy level, partition and top-k
remain equal-compressed; outer family fusion remains equal; H0 clean/error
decisions and the top-ten reducer are unchanged.

## Fit contract

- Five outer source-question folds; held responses are projection-only.
- Inside each outer donor set, source questions are deterministically assigned
  to five nested STG folds. Tokens remain structured observations and inherit
  their response fold.
- STG uses seeds `11,23,37`, 120 epochs, sigma `0.5`, learning rate `0.05`,
  and the frozen penalty roster `{0.1,1,3,4,5}`.
- A pair must have mean gate probability at least `0.75` and recur in at least
  four of five nested folds (`minimum_fold_fraction=0.8`).
- Penalty selection is label-free: minimum held-covariance error among
  supports satisfying the SU sparse-support theorem. No support cap is applied.
- Canonical and fixed-support refits use the exact repository `SU_CONFIG`.
- Orientation is donor-only against the equal dynamics anchor.
- Twenty random supports use seeds `2026083111` through `2026083130`; the
  feature-permutation seed is `2026083102`.
- All score trees and diagnostics are hashed before ProcessBench labels open.

## Primary contrasts and inference

The five primary macro-F1 contrasts are:

1. S1 minus S0: canonical SU versus the strongest IU parent.
2. S2 minus S0: net STG-SU value.
3. S2 minus S1: whether STG repairs canonical support extraction.
4. S2 minus S3: learned support versus feature-label permutation.
5. S2 minus S4: learned support versus cardinality-matched random support.

Use 20,000 paired whole-question bootstrap draws with Bonferroni-simultaneous
intervals across all five. Report exact error, within-one, clean abstention,
W/T/L, worst cell, prediction flips, selected pair counts, chosen penalties,
outer-fold support Jaccard, theorem validity, and convergence.

Formal promotion requires S2 to beat S0 with point delta at least `+0.003`
and simultaneous lower bound above `+0.003`, exact-error delta at least
`-0.010`, worst cell at least `-0.020`, zero H0 abstention mismatches, and all
finite/convergence/theorem checks.  A claim that learned support matters also
requires S2 to beat S3 and S4 with simultaneous lower bounds above zero.

A CI crossing zero means promising-unconfirmed or inconclusive, never generic
rejection. Supported material harm, invalid support, leakage, alias failure,
or nonconvergence is a hard stop. The opened population cannot provide fresh
confirmation.

## Conditional next steps

No top-k STG, STG prune/refit, outer STG, or PRMBench transfer opens unless S2
passes its exact-parent and support-mechanism gates. B3/L-SML and tensor/query
remain distinct later mechanisms and are not crossed with this branch.
