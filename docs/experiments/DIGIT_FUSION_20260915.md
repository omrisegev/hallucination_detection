# Step393: independently replay digit disagreement and test useful fusion

User authorized a bounded full-data experiment inspired by Claude's digit view.
Data are already exposed development data. Seven candidate views were screened
using labelled error cohorts by Claude; fixed gamma does not remove that exposure.

## Audit before scoring

Independently verify ASCII digit IDs from cached Qwen3-4B/8B tokenizer vocabulary,
raw provided-token/top1 alignment, spans and source hashes. Reconstruct binary
d_t from raw gen_token_ids and top50 IDs: both tokens are single digits and differ.
Replay every saved Claude auxiliary and gamma .25 /1 score independently.
No new model forward pass. Teacher-forced provided answers, not sampled answers.
The old participation ratio3.55 concerns all seven added views, not digit alone;
compute bank5 versus bank5+digit explicitly. Zero-variance digit answers must
not be interpreted as independent experts.

## Fixed roster, no label-driven resweep

1. Direct digit auxiliary (diagnostic).
2. innovation5 + .25*sd(base)*z(Top10 digit), Claude primary replay.
3. Same gamma1, prior secondary replay, not a new gamma search.
4. Same .25 correction using provided-digit presence (opportunity/length control).
5. Same .25 correction using per-step disagreement count / provided-digit count,
   zero when no digit exists (rate control; different readout, explicitly).
6. Same .25 correction after permuting disagreements among provided-digit positions
   within each answer, preserving total disagreements and digit positions;
   fixed UID-derived seed, no labels (density/location null).
7. TCN corrected score + the .25 digit correction on the innovation5 scale.
8. Base + .25*sd(base)*z(z(TCN correction)+z(digit auxiliary)).
   This amplitude-matched version distinguishes direction from simply adding
   twice as much correction. Two-view equal is allowed; do not call it IU.
9-12. Existing five raw token streams versus the same five plus digit, each with
   answer-token-standardized equal and canonical IU2PC. Fit covariance on all
   answer tokens; take Top10 separately per standardized stream, then weighted
   sum (same order in all four arms). Return final step score to the base mean
   and SD by a positive affine map. This is direct bank fusion, not a .25 residual
   correction; equal/IU and bank5/bank6 pairs isolate those changes.
   Constant columns are zero and excluded from the covariance solve. For <3
   active views or rank<2, explicitly record equal fallback. Negative native IU
   coefficients are retained; global sign aligns with equal high-risk evidence.

No operator extension, gate change, group/feature selection, learned gamma or
new training in this cycle. Check those later only after these results.

## Population, access and inference

All13769 answers/145597 steps, unchanged source groups and labels; fixed
tail15 gate q=.33. Fitting/normalization functions accept no labels. Base/digit
and bank fusion fit within answer. TCN uses existing source-excluded models:
reuse five outer and ten pair-exclusion score sets to calibrate each held PRMB
fold's q=.8 without models trained on that fold. For answer-only arms, ordinary
other-fold unlabeled quantiles suffice; still use the same calibration roster.
The complete system remains offline and PB-gate-transductive.

Six primary comparisons x PB/within,10000 paired source-group draws, CI99.5833%:
digit025-base; digit025-presence; digit025-permuted; matchedTCNdigit-TCN;
bank6IU-bank6equal; bank6IU-bank5IU. Other comparisons including sum versus
matched amplitude, bank6equal-bank5equal and rate are exploratory95%.
Retain Claude's original97.5% contrast as historical replay, not extra confirmation.
Full same-gate references: Step388 sixteen methods and Step389 two leading
tradeoffs. No winner-selected deployment; show Pareto/endpoint tradeoffs.

## Required diagnostics / acceptance

Per-cell metrics, coverage, source-group paired intervals, exact PB gains/losses
vs base and TCN, recovery among fixed885 raw misses/707 open misses. Constant
digit fraction, disagreement count/opportunities, bank correlations and
participation ratio5 vs6 (and conditional on nonconstant digit), IU coefficient
and fallback diagnostics. Density and rate controls address exposure, not a
proof that every length effect is removed. Sparse binary maxima have ties:
do not reuse the43% token-maximum threshold as an acceptance rule.
Independent scalar stream/readout replay; constant/short-span tests; canonical
IU comparison; independent full PB/within reconstruction. Preserve Claude source.
No small-cohort quality inference. One hour execution budget, save extracted
features and scored arrays before evaluation; stop after the reviewed stage.
