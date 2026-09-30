# Full gate-versus-location diagnostic

Decision question: how much of the full ProcessBench gap can be recovered
by changing only the no-error decision while preserving the answer-only
fusion scores and selected steps? This takes priority over new fusion,
feature, graph or sampling sweeps. Existing full runs continue unchanged.

Use all 13,769 cached model-answer records, including all 6,800 ProcessBench
rows. Same v3 labels, canonical source groups and five fixed outer folds.
This is retrospective development analysis, not untouched confirmation.
No new fusion fit, feature extraction, graph fit or model inference.

## 1. Cross the existing location and gate outputs

Current IU and corrected historical IU supply a 2-by-2 table: current/current,
current/historical, historical/current and historical/historical. Preserve
each location arm's saved peak. The gate donor contributes only its saved
error/no-error decision and decision validity. Combined validity requires
both a valid location score and a valid donor decision. All failures remain
in the denominator. Historical PB peaks come from its top10 readout, not
the PRMB spanmax ranking curve.

Report correct-peak overlap, all cells and Q4/Q8/all macros. The first direct
full-data check, before this protocol, gave Q8 20.3828/31.6751/21.5468/34.2940%.
Thus this block is explicitly post-hoc. Report the two intervention orders,
their interaction and the symmetric average allocation of the gap. This is
an operational decomposition of two frozen pipelines, not a causal claim
that GMM alone caused that fraction. Historical gate access includes other
training answers, a different detector representation, and label calibration.

## 2. Change only the gate on our saved fusion curves

Four pre-existing full-data cores: routed IU, routed equal, Joint graph with
condition100, and the static mean of IU plus that graph curve. These retain
the learned-fusion versus equal control and both fusion axes. No new weights,
route, group, feature, sign or peak is selected. Three gates per core:

- Original native GMM/BIC decision, copied exactly.
- Hybrid calibrated raw-peak gate. Detector is the maximum saved fused
  window risk. For each held-out outer source fold, choose one threshold from
  the 99 outer-training quantiles (0.01 through0.99), maximizing the eight-cell
  PB harmonic macro. First/lower threshold wins ties. No held-out source
  group supplies calibration labels. Existing answer-only scores do not need
  inner model refits: they have no fitted cross-answer dependency. This is
  label-calibrated decision-making, explicitly not the primary unsupervised
  method. It tests whether a threshold on our own score is sufficient.
- Fixed normal-reference control. Standardize the saved window risk using
  its original nonoverlapping fitting windows. Let S be its maximum and N
  the number of fitting windows. Call error when `N*normal_survival(S)<=0.05`.
  The level0.05 is fixed before inspecting this control's outcomes. The
  normal tail is an engineering reference, not a guaranteed p-value under
  dependent, nonnormal, self-standardized traces. A zero/invalid scale is an
  invalid decision, retained as a failure. No level search or promotion.

These controls change the gate only, but the calibrated and fixed controls
use different gate statistics. Their difference is not pure calibration
attribution. A failure of one maximum-based statistic does not rule out an
informative gate based on other available telemetry. The historical decision
swap distinguishes access to its whole gate from calibrating our own score.

## Report and checks

Preserve original PRMB score curves and within-answer results. A gate-only
change cannot improve those rankings; do not imply a new two-task winner.
Report PB per-cell clean/error accuracy, Q4/Q8/all harmonic macros, coverage,
gate error/no-error discrimination and the unchanged peak accuracy.

Use1,000 paired canonical-source draws, seed2026090707, shared across scorers
and methods. CIs condition on the fitted calibration thresholds and saved
predictions; they do not redo threshold selection and are exploratory.
Compare each new gate with its native core; compare learned fusion with equal
and the graph/mean cores with IU under the same new gate. Keep the historical
IU and CONT context and the two swap interventions visible.

Validate the four original corners, source-fold separation, fixed peaks,
independent direct PB counts, threshold-grid maxima and tied-score threshold
behavior. Review the code and findings once, focusing on scientific validity.
Keep hashes and detailed provenance in machine-readable files, not the main
scientific narrative. Based on this full-data result, design at most one
next label-free gate serving the existing fusion, with matched controls.
