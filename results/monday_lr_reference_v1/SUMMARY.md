# Supervised logistic-regression reference per nested bank (2026-10-07)

Supervised reference, not a method. It uses PRMBench labels.

## Result
- **PRMBench:** LR beats the best of our 4 label-free methods on every bank.
  - The within-answer AUC gap is 0.026–0.068, and every paired source-group 95% interval is above 0.
  - The gap grows with bank size: 0.026–0.029 on 11–16 channels, 0.059–0.068 on 40–56 channels.
- **ProcessBench:** the same LR, trained on PRMBench and applied unchanged, is within −0.013 to +0.030 of MEAN_MOM_ga.
  - 8 of 9 intervals include 0. N43 does not.
  - The best ProcessBench value is still plain equal weights on N16 (0.414). LR on N16 scores 0.402.
- **Direction:** step scores agree with MEAN_MOM, with within-answer correlation 0.54–0.85. Channel weights do not agree: plain cosine is −0.35 to 0.45.
  - LR gives negative weights to 5 of 11 (N11) up to 27 of 56 (N56) channels, using signed contrasts between correlated channels.
  - Measured in the channel covariance, the cosine equals the step-score correlation.

| bank | LR AUC | our best (method) | gap [95% CI] | LR−plain avg | LR−oriented | PB LR | PB ours mean | corr(LR, MEAN_MOM) | weight cosine | sign agreement vs oriented |
|---|---|---|---|---|---|---|---|---|---|---|
| N11 | .7853 | .7576 MEAN_MOM_ga | +.028 [.024,.032] | +.036 | +.030 | .334 | n/a | .851 | .448 | .545 |
| N13 | .8142 | .7850 TCI_ONLY_ga | +.029 [.025,.033] | +.039 | +.039 | .362 | .358 | .776 | .333 | .615 |
| N22 | .8198 | .7736 MEAN_MOM_ga | +.046 [.042,.051] | +.052 | +.051 | .358 | .373 | .664 | .244 | .636 |
| N40 | .8265 | .7633 HEM_MOM_cc | +.063 [.058,.068] | +.081 | +.054 | .376 | .346 | .535 | .021 | .525 |
| N53 | .8304 | .7621 MEAN_MOM_cc | +.068 [.063,.073] | +.078 | +.062 | .366 | .360 | .557 | −.114 | .509 |
| N14 | .7981 | .7698 MEAN_MOM_cc | +.028 [.025,.032] | +.034 | +.025 | .380 | .377 | .844 | −.087 | .643 |
| N16 | .8224 | .7961 HEM_MOM_cc | +.026 [.023,.030] | +.036 | +.032 | .402 | .403 | .787 | −.184 | .688 |
| N25 | .8264 | .7783 TCI_ONLY_ga | +.048 [.043,.052] | +.046 | +.044 | .395 | .392 | .666 | −.082 | .600 |
| N43 | .8336 | .7744 HEM_MOM_cc | +.059 [.054,.064] | +.076 | +.050 | .412 | .384 | .567 | −.350 | .512 |
| N56 | .8355 | .7749 MEAN_MOM_cc | +.061 [.055,.066] | +.074 | +.060 | .401 | .381 | .629 | −.299 | .482 |

- **"Our best"** is picked after seeing the scores, so each gap is a lower bound.
- **N11 on ProcessBench:** one cell has only 2 groups, so our methods are undefined there.

## Validation
- N13 and N16 LR out-of-fold AUC (0.814239, 0.822366) match the earlier estimator_input_v1 reference exactly.
- The label-free per-answer AUCs match v1/v2 PER_ANSWER.npz exactly.
- The DS survivors match v1 FIT_DIAGNOSTICS.
- A fold-matched ProcessBench transfer differs from the all-fold model by at most 0.0035.
- Every fit converged. Fold weight cosine is ≥ 0.994.

## Deviations from the earlier LR spec
1. LR is trained on PRMBench only.
2. Implied MEAN_MOM weights use minimum-norm least squares, because near-duplicate channels in N53/N56 make plain OLS unstable. R² is 0.993–0.998.
3. Two measures were added: score correlation and cosine in the channel covariance.
4. "Family" in the L3 figures is the nesting block, not a meaning-based family.

Files: PROTOCOL.json (written before the run), run.py, LR_SCORES.csv, LR_COEFFICIENTS.csv, COEF_COMPARISON.csv, GAPS.csv, PARITY.json, figures/L1–L3.
