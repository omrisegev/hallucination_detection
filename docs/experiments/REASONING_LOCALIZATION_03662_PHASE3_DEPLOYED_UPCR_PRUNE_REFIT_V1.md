# Reasoning Localization 0.3662 — Phase 3 deployed U-PCR and prune/refit amendment v1

Status: **complete on the opened eight-Qwen ProcessBench development panel;
no promotion**

Frozen execution registry:
`results/reasoning_localization_03662_v1/phase_3/deployed_upcr_prune_refit/P3D_COMPACT_VIEW_EXECUTION_REGISTRY.json`.
The registry, all 24 member names, five grouped folds, 20 random-mask seeds,
six macro-F1 contrasts and practical bounds were frozen before labels were
imported.

## Why this amendment exists

The completed P3B/P3C experiments used ordinary two-component IU-PCR.  They
did not test the repository's deployed U-PCR policy.  These estimators share
the same additive off-diagonal covariance model, but they are not aliases.

Deployed U-PCR performs two estimation passes:

1. fit the additive covariance model on the full fit-side view pool and
   estimate each view's response covariance `rho_hat_i`;
2. remove weak views using the frozen project thresholds, then recompute the
   covariance estimate and PCR weights on the survivors.

The removal criterion is **not** ordinary observed correlation with a task
label or with the eventual ProcessBench score.  It is the method's label-free
estimate `rho_hat_i = Cov(f_i, Y)` under the U-PCR model.  The exact deployed
policy uses `scale_ratio=0.25`, L2, automatic one/two-component final PCR,
`min_frac=0.05`, `exclude_frac=3.0`, recomputation after exclusion, and a
simple-average fallback when fewer than five experts remain.  See
`docs/methods/deployed_upcr.md` and `spectral_utils/upcr.py`.

This correction matters because the historical 0.3662 family6 head used
ordinary cross-family IU-PCR, while the new P3B used full-pool ordinary IU-PCR
over the four H2 family scores.  Neither run tests deployed U-PCR's
weak-expert exclusion and refit.

## Eligibility boundary

Exact deployed U-PCR is not informative on the four outer H2 family scores:
the frozen deployed policy has only four inputs and therefore necessarily
enters its `<5` simple-average fallback.  Such a run is an expected alias of
the equal-family parent, not evidence for or against pruning.  It must be
reported as `NOT_APPLICABLE_BY_DIMENSION`, not ranked as a new fusion method.

The first meaningful test therefore uses the compact H2 **member-view** pool:

- singleton entropy level;
- the frozen entropy-dynamics members plus C7;
- partition-energy members except `energy_series`;
- the six top-k distribution members;
- no sampled-token-energy or structural members, matching H2's roster.

The executable registry must enumerate the exact member names and count before
scores are opened.  No member may be added because it improves audit F1.

## P3D compact matched ladder

The bounded ladder is:

1. `P3D0_H2_VIEW_FULLPOOL_IU`: ordinary two-component IU-PCR on the exact
   compact member-view matrix, with exclusion disabled.  This is the matched
   full-pool spectral control, not a claim that it aliases equal-family H2.
2. `P3D1_H2_VIEW_DEPLOYED_UPCR`: exact deployed U-PCR on the same matrix,
   including the full-pool `rho_hat`, frozen weak-view mask, survivor refit,
   automatic one/two-PC rule, and documented simple-average fallback.
3. `P3D2_H2_VIEW_MASK_EQUAL_CONTROL`: use P3D1's frozen survivor mask but
   equal-weight the retained standardized views.  This separates selection
   from U-PCR reweighting.
4. `P3D3_H2_VIEW_RANDOM_MASK_CONTROL`: cardinality-matched random masks, with
   seeds and the aggregation rule frozen before labels.  This asks whether
   the estimated-`rho` mask does more than merely reduce dimension.

All candidates retain the H0 clean/error decision and top-ten reducer.  They
may rerank only H0 non-abstentions.  The primary method contrast is P3D1 minus
P3D0; P3D2 and P3D3 are mechanism controls.  Equal-family H2 and the strongest
atomic parent remain system comparators but are not substituted for the exact
same-matrix P3D0 parent.

## Fit, stability, and leakage contract

- Fit-side standardization, `rho_hat`, keep masks, component count, sign, and
  refit use calibration/donor rows only.
- All scorer copies of one source question remain in one fit/evaluation fold.
- ProcessBench and PRMBench labels cannot select a threshold, survivor count,
  view, sign, component count, or fallback.
- The no-exclusion setting must reproduce P3D0 to numerical tolerance.
- Report the full and survivor `rho_hat`, mask, number kept, fallback status,
  component count, projection residual, `g2` boundary status, weights, and
  score variance for every cell.
- Repeat the mask on five grouped donor folds.  Report per-view selection
  frequency and Jaccard stability.  Instability is evidence against the
  pruning mechanism even when the point F1 is positive.
- Bootstrap and uncertainty units remain whole source questions.  Extra token
  observations do not create independent samples.

## Applying the prune/refit idea to other fusion variants

Prune/refit is registered as a **method-native wrapper**, not a generic
post-hoc filter and not an automatic combinatorial expansion.

- ordinary IU-PCR may use the matched full-pool additive-model `rho_hat`, then
  refit the same fixed two-component IU rule on the frozen survivors;
- SU-PCR may use only its sparse-error-corrected reliability estimate after
  its support/identifiability premise passes;
- STG-SU may use only fold-stable stochastic-gate support and must retain a
  cardinality-matched random-support control;
- DUFS-LIU may prune only from donor-fold-stable DUFS gates before the LIU
  refit; audit/test F1 cannot choose the gate threshold;
- L-SML/B3 is eligible only if a method-native label-free reliability and an
  exact no-pruning parent alias are specified in advance;
- tensor/query variants may prune coordinates only from donor-only loading or
  stability criteria and cannot borrow U-PCR `rho_hat` without a separately
  justified model contract.

For every eligible method, test at most one frozen full-pool/prune-refit pair.
Required controls are the exact full-pool parent, no-pruning alias,
cardinality-matched random mask, and mask-only equal-weight control when
defined.  A method that has not survived its own unpruned gate does not earn a
pruned rerun merely because pruning exists.

## Evaluation and promotion

Use the current common ProcessBench rows, folds, spans, H0 detector, top-ten
reducer, and grouped bootstrap.  Report macro F1, exact error, within-one,
clean abstention, W/T/L, worst cell, prediction flips, keep-set stability, and
paired multiplicity-valid intervals.  A positive point estimate whose
interval crosses zero is `PROMISING_UNCONFIRMED`, not rejected.

Promotion requires P3D1 to beat both the exact same-matrix P3D0 parent and the
strongest compact system parent, while preserving the registered exact-error,
clean-abstention, and worst-cell bounds.  Only a frozen ProcessBench survivor
transfers to PRMBench; the two task estimands remain separate.  Phase 5 may use
only prefix-safe member views and donor statistics.

## Completed result

The ladder used five-fold grouped cross-fitting: fit-side imputation,
standardization, orientation, `rho_hat`, exclusion, component choice and refit
used only the four donor folds, while every held response was projection-only.
The no-pruning alias and frozen H2 source alias both had maximum absolute error
zero.  H0 abstention mismatches were zero for every arm and random-mask seed.

ProcessBench macro F1 was:

| arm | F1 |
|---|---:|
| equal-family H2 system parent | 0.364090 |
| P3D0 member-view full-pool IU | 0.354240 |
| P3D1 deployed U-PCR | 0.356740 |
| P3D2 rho-mask equal control | 0.353551 |
| P3D3 mean of 20 random-mask controls | 0.354007 |

P3D1 minus its exact same-matrix P3D0 parent is `+0.002499`, with the
six-contrast simultaneous interval `[-0.008683,+0.013781]` and cell W/T/L
`5/1/2`.  This is `PROMISING_UNCONFIRMED`, not rejection.  P3D1 minus H2 is
`-0.007350 [-0.017885,+0.003263]`, W/T/L `1/2/5`; therefore it does not pass
the system-parent gate.  Its increment over the equal survivor-mask control is
`+0.003189 [-0.006005,+0.012580]`, while the rho-mask equal control minus the
random-mask reference is `-0.000457 [-0.009545,+0.008547]`.  Neither mechanism
contrast supports a positive claim.

The masks themselves are reproducible: the minimum, over cells, of the
within-cell mean pairwise fold Jaccard is `0.9571`; 12--14 of 24 views survive,
no fold invokes the simple-mean fallback, and every fold selects two final
components.  Entropy level, C7 onset, six top-k views and five entropy-dynamics
views are retained in every cell/fold; several other spectral views are
consistently removed.  Stable exclusion is therefore established as a
descriptive mechanism property, but stability did not translate into a
supported localization gain.

Verdict: `PROMISING_UNCONFIRMED_VS_FULLPOOL__NO_PROMOTION__MECHANISM_NOT_SUPPORTED`.
No PRMBench transfer is opened by this result.  Other method-native
prune/refit wrappers remain survivor-gated; this run does not reopen methods
whose unpruned parent failed.
