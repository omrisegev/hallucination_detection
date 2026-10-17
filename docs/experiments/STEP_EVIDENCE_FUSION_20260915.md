# Step392: complementary step evidence, before evaluation

## Decision and history

User authorized trying fusion after the common-miss audit. One fixed bank,
not a subset or window-length sweep. The hypothesis is that context residuals,
sustained uncertainty and end-of-step uncertainty provide complementary evidence
about the same step. This is a development experiment; last4 was motivated by
label-selected failures in Claude's audit. No claim of untouched confirmation.

History checked: RBM hierarchical-time v1 already compared Top10, all-token mean
and contiguous10 (PB36.271,29.429,31.653 respectively); Module B in Joint-LSML v2
already fused ten order-statistic views. Neither establishes novelty of temporal
summaries. The present distinction is combining heterogeneous step evidence as
a bounded correction to innovation5, retaining its original Top10 once.
Step389 already fused five *predictor residual trajectories* before Top10.
Here fitting rows are the current answer's steps, and columns are three different
step summaries. Few steps are a declared estimator limitation.

## Fixed algorithm

Base b_s: mean of separate Top10 over the five natural-unit innovation5 streams.
Three auxiliary columns for each step s:
1. context: Top10 of the signed TCN residual, equivalently its already saved
   standardized step correction. Reuse all15 source-excluded TCN score sets.
2. end: mean of the last min(4,length) tokens, then mean across five raw streams.
3. sustained: maximum mean of a contiguous min(10,length)-token window, separately
   in each raw stream, then mean over streams. Peak windows may differ by stream.

Standardize each column over steps of the current answer (population SD).
Single column, equal mean, or canonical IU two-PC weighted sum -> standardize
result over steps -> final b + .25*sd(b)*z(auxiliary). Base appears ONCE.
This preserves answer base mean and fixed correction amplitude, not total output
variance. High auxiliary values retain the established high-error convention.
No labels set signs; IU global sign aligns its score with equal standardized
evidence. No per-answer choice based on labels.

IU uses fast_iu, verified against upcr_fit_covariance with IU_FIT_DEFAULTS:
L2 additive fit,300-point g2 grid,scale_ratio=.25,two PCs, no exclusion/difficulty
gate. This is the maintained regression IU, not legacy nadler_fuse or the
label-based orientation helper. Correlations/Spearman>=.75 are diagnosed, not
used to choose another bank, consistent with the existing correlated-view IU
experiments. The three-pair additive identity is exactly identified; its fit
residual cannot validate its assumptions.
If fewer than3 steps, any constant view, or second covariance eigenvalue<=1e-10,
declare IU unavailable and use equal as an EXPLICIT fallback. Report each reason,
weights, native coverage and full-population policy results. Other numerical
failures abort rather than silently falling back.

## Roster and controls

Three singleton corrections: context (must replay TCN), end, sustained.
Equal and IU over all three. Equal context+end and context+sustained for ablation.
Equal and IU with the two shape summaries extracted after a shared permutation
of tokens within each step; TCN correction untouched. Same deterministic
UID-derived seed; all five raw streams share a permutation. This controls the
use of within-step order without disrupting cross-stream alignment.
No new first-error decoder, alpha, gate, feature bank or model training.

All16 Step388 references, plus Step389 IU PB leader and equal within leader.
Same13769 answers/145597 steps, source groups and label release. PB fixed frozen
tail15 percentile>=.33 (transductive). PRMScore q=.8 calibrated separately for
each held fold using scores from pair-excluded TCN models; do not calibrate from
ordinary OOF outputs of models exposed to the evaluation fold.
Answer-local normalization/fusion with externally fitted source-excluded TCN
and transductive PB gate: whole system is offline, not answer-only.

## Evaluation frozen before opening new results

Five primary pairs: IU-equal, IU-TCN, equal-TCN, IU-IU-shuffled,
equal-equal-shuffled. Two endpoints PB and within;10000 paired source-group
bootstrap draws,99.5% intervals (10 comparisons). Other comparisons descriptive.
PRMScore and cell results retained. No selected winner promotion in this cycle.
Report gains/losses vs TCN and innovation5 and recovery among original885/707,
separately before/after gate. Original miss cohorts remain fixed diagnostic
cohorts, not training sets or replacement quality populations.
No quality conclusion from smoke data. Full-population scoring and independent
metric replay required. Budget: one hour CPU; resumable per-answer extraction,
saved outer/nested scores. Stop at reviewed report; no automatic follow-up sweep.

## Acceptance checks

Independent scalar extraction (including short steps), full baseline and context
replay, canonical weights on deterministic audit rows, identity at zero
amplitude, explicit constants/short-answer behavior, reproducible permutation,
full IDs/spans/coverage and source hashes. Fitting APIs receive numeric evidence
only; correctness labels are loaded only in evaluation. Recompute all PB and
within metrics independently; preserve original inputs and Claude outputs.
