# Review request: label-free error-count family A–D

The user wants Claude to examine the completed experiment. This document is a
handoff; it does not claim Claude has reviewed it or authorize new experiments.

## Read first

1. README.md and FINDINGS_HE.md; report/index_github.html.
2. PROTOCOL.json and chronological AMENDMENTS.json (especially the numerical
   bridge repair and the quota-mechanism diagnostic's information timeline).
3. review/FINAL_REVIEW.md and its independent arithmetic receipts.
4. evaluation/METRICS.csv, CONTRASTS.csv, SUCCESS.json, CHECKPOINT_STATUS.csv,
   and FOLD_METRICS.csv. Inspect per-type MS valid F1 and PS/DR/MS Sensitivity.
5. diagnostics/QUOTA_SCIENCE.json, EMISSION_CALIBRATION.json,
   COMPONENT_RESULTS.json and COUNT_ORACLES.json.

## Questions to examine

- Do fitting interfaces avoid target labels, types, PRM scores and holdout
  statistics? Are groups, weights, survivors, normalizations, thresholds and
  triple selection donor-only with source-disjoint FOLDS_V2 identities?
- Does the compensated raw/answer-z bridge preserve masks, affine coefficients
  and canonical group scores at the unchanged 1e-9 tolerance? Distinguish the
  first numerical failure from later genuine infeasible MoM emissions.
- Is IU-PCR canonical, fitted once per final bank/fold, and ordering shared
  across all quantity arms? Are ranking metrics strictly within answers?
- Are the positive-orientation, conditional voter independence and conditional
  step-independence assumptions justified or merely assumed? B selects one
  stable triple; stability and low correlation do not establish independence.
- Does D separate prior refitting (rho=0) from the structural-zero component,
  and report boundary/convergence/identification checks without silent repairs?
- Are posterior/count-PMF mean and lower median comparisons otherwise identical?
- Are noncontrols, synthetic controls and all true-zero cohorts distinguished,
  failure counts missing rather than zero, and partial coverage explicitly shown?
- Do all seven candidates remain in the corrected declaration family, with
  shared 20,000 source-group draws and the one-percentage-point recall allowance?
- Does the quota diagnostic support sensitivity without claiming MoM is
  mathematically the vote fraction or choosing q=10% because it matches truth?
- Is the conclusion balanced: no protected improvement, but LOCAL can improve
  over old MoM; R2 retains similar PRMScore and higher recall despite worse MAE;
  stable B improves count association without calibrated allocation; D improves
  clean abstention while missing more erroneous answers?
- Are oracle component gains kept nondeployable and nonadditive, and next-step
  continuous/dependence modelling clearly a hypothesis rather than a proven fix?

## Code and limits

Main code: spectral_utils/error_count.py, error_count_inputs.py,
error_count_evaluation.py; scripts/run_label_free_error_count_v1.py and
evaluate_label_free_error_count_v1.py. The separate final renderer is
scripts/report_label_free_error_count_v1.py. Synthetic tests are under tests/.
Diagnostic and independent audit scripts are archived beside their results.

Read-only static and aggregate arithmetic review can be done from Git. Full
array-level reconstruction needs the catalogued original numerical bundle and
the inherited advisor inputs. State what was actually reproduced; do not claim
full reproduction when arrays are unavailable. Do not refit, repair failed
arms, select thresholds, or overwrite any frozen artifact as part of this review.

Return accept/partial-accept/push-back findings with file/line evidence, severity,
and a distinction between arithmetic bugs, assumption failures and claim limits.
