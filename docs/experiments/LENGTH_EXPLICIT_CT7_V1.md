# Explicit step length for CT7 v1 (Step 420 [Claude]) - PRE-REGISTRATION

Written and committed before any score of the new arms is computed. Decisions by Omri, 2026-09-17:
neutralizer = length-calibrated Top10; length view = log n; BOCPD may be recomputed at token level.

## Question

How much of the frozen CT7 result is evidence and how much is the hidden step-length prior carried by
the Top10 step readout? (CT7 peaks on the longest step in 43% of ProcessBench error answers; the first
error is the longest step in 29.7%.)

## Fixed design

Population, gate, folds, metrics, bootstrap: identical to Step 413-418 (13,769 answers; frozen
non-digit tail15 gate; PB eight-cell macro-F1; PRMB mean within-answer AUROC; 10,000-draw paired
source-group bootstrap, 95% intervals). No fitted parameter in any arm; equal weights.

Token streams, unchanged from CT7:
* H0lim, ve0, ve0.75, ve1, H0lim prefix innovation: token columns of `digitfree_broad50.token_bank`,
  re-extracted from the raw pickles; the innovation stream's first token is invalid as in `step_bank`.
* BOCPD residual: from `temporal_context_data_v1` token features (innovation5 bank, whole-answer
  mean/scale), `bocpd_mean` hazard 1/32, token residual = mean over the five features of (z - prediction).
* Chosen-token view: unchanged CT7 view (pooled z-test, step 0 neutralized; already length-free).

Exactness gates before any new arm is scored (the run stops if either fails):
1. The re-extracted Top10 of the five bank streams equals Codex's frozen extraction (float32 tolerance 1e-4).
2. The answer-standardized Top10 of the recomputed BOCPD token residual equals the CT7 BOCPD view (1e-8),
   and the step scores base + .25 * std(base) * astd(Top10) equal the historical BOCPD scores (1e-10).
3. CT7 rebuilt from these re-extracted Top10 readouts equals the frozen CT7 development scores (1e-6).

Length-calibrated readout (`spectral_utils/length_calibrated_readout.py`): for a step with m valid tokens,
z = (Top10 mean - mean of Top10 over contiguous m-token windows of the same answer) / max(window sd,
.5 * token sd / sqrt(min(10, m))); at most 64 evenly spaced windows; single-step answers give 0.
Then answer-standardized like every view. No step-0 rule is added to these six streams (as in CT7).

Length view: log of the step's token count, answer-standardized. Orientation fixed a priori by a
constant per-token hazard model (longer step, higher chance of containing the error), not by labels.

## Arms

| id | views | role |
|---|---|---|
| CT7 | frozen candidate | reference, length hidden |
| CT7-replay | CT7 rebuilt from the re-extracted Top10 | exactness gate |
| LX7 | six length-calibrated streams + chosen-token view | evidence only |
| LX8 | LX7 + log-length view | evidence + declared length prior |
| LEN | log-length view alone | prior only |
| CT7+LEN | CT7 + log-length view | double-counting reference |

## Contrasts (all reported, both endpoints)

LX7 - CT7 (what the hidden length contributed, with sign reversed), LX8 - LX7 (what the declared
length adds), LX8 - CT7 (does making it explicit lose anything), CT7+LEN - CT7.

## Diagnostics

Within-answer correlation of each fused score with log step length; share of ProcessBench error
answers whose peak is the longest step (truth 29.7%); effective conditionally independent views
(PRMB labels for measurement only) for LX7 and LX8; mean view value by step index (descriptive).

## Decision language fixed in advance

* If LX7 is below CT7 on an endpoint with an interval excluding zero, that part of CT7 was length prior.
* If LX8 recovers CT7 within intervals, the length prior can be declared explicitly without loss.
* No arm is promoted from this run; CT7 stays frozen. Any adopted arm becomes a new frozen candidate
  with its own confirmation requirement.

## Amendment 1 (2026-09-17, before any new arm was scored)

Gate 3 as written (CT7 rebuilt from the re-extracted readouts within 1e-6 of the frozen scores) FAILED
at 1.151e-06. Diagnosis before any change: the frozen bank extraction is stored as float32, while the
re-extraction computes in float64. Casting the re-extracted Top10 to float32 reproduces the stored
extraction EXACTLY (max absolute difference 0.0 over all steps and all five streams), and the candidate
rebuilt from it matches the frozen development scores exactly (0.0). The residual was therefore storage
precision, not a difference in the readout.

Gates 1 and 3 are amended to exact equality after casting the re-extraction to float32, which is
stricter than the original tolerances; the float64 differences (1.90e-06 for the readouts, 1.151e-06 for
the rebuilt candidate) are reported as diagnostics. The new arms keep full float64 precision. Gate 2 was
unchanged and passed. No arm had been scored when this amendment was written.

## Amendment 2 (2026-09-17, before any new arm was scored)

Gate 3 as amended (exact equality after the float32 cast) FAILED at 9.99e-16. Diagnosis before any
change: the rebuild used the RECOMPUTED BOCPD view, which differs from the frozen view within the
gate-2 tolerance (1e-8); divided by seven views that residual reaches machine precision in the mean.
Requiring exactly zero there conflated the readout claim with the BOCPD recomputation.

Gate 3 now rebuilds the candidate with the float32-cast bank readouts and the FROZEN BOCPD view, and
requires exact equality (the readout claim). The rebuild with the recomputed BOCPD view is reported as
a diagnostic and must stay below 1e-9, one order tighter than gate 2. Still no arm had been scored.
