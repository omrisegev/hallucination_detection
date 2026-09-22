# Step388: complete the aligned TCN predictor, seed0, before fusion

User approved completing TCN on all answers with the successful signed
correction and the existing predictor references. Frozen before new fitting.

- innovation5 only, seed0 only. Same TelemetryTCN architecture, Gaussian
  telemetry-prediction loss, AdamW3e-4/weight_decay1e-3, batch256, <=50000
  updates, fixed validation every500, six stale validations stop. Reuse the
  existing fold0 checkpoint and scores after checking source/data/split hashes.
- Complete four remaining outer fits and ten pair-excluded calibration fits.
  Up to3 independent CPU jobs, each one torch/BLAS thread. No flow queue restart,
  FM/DiFlo modifications, bank search, dose search or predictor fusion.
- Same five standardized targets, 16 previous tokens + mask + relative position.
  Whole-answer means/signs/scales remain offline. TCN predicts BEFORE current
  observation; context crosses steps. No correctness labels in fitting/scoring.
- Registered outcome: original innovation5 step score plus .25 signed residual
  correction. Average residuals across features, Top10 within step, center and
  standardize auxiliary across steps, multiply by original step-score SD.
  Frozen tail15 percentile >=.33 gate. All13769 answers/145597 steps/6968779
  tokens; source folds unchanged. PRMScore .8 thresholds use pair-excluded fits.
- Real, history-slot shuffled, zero-history versions from the same fitted
  model. These are inference interventions, not separately refitted nulls.
  The innovation target already contains history; zero is not an end-to-end
  history-free method. The existing scorer also saves other diagnostic readouts;
  they are not candidate searches in this study and do not select a winner.
- Primary6 pairs: real TCN minus Ridge, BOCPD, noreset, innovation5,
  shuffled TCN, zero-history TCN. PB/within are12 primary endpoints,
  10000 paired source-group bootstrap draws, Bonferroni CI1-.05/12.
  Report primary advantages/tradeoffs; null intervals are not equivalence.
  Keep all13 Step387 references with exact scores and thresholds.
- Report all-token and after16 MSE by feature, real/shuffled/zero sensitivity,
  residual/prediction correlations with Ridge and shared-current partial
  correlations, PB gain/loss and early/middle/late strata. Diagnostics do not
  establish conditional independence or a label-free routing rule.
- Independent scalar signed-readout reconstruction and full PB/pairwise AUC;
  group/train/validation/held separation, complete nested coverage, frozen
  fold0 replay, code and input hashes, finite predictions and explicit failures.
- Conclusion concerns seed0 only, on development data previously used for bank
  and signed-dose selection. A positive result requires later seed robustness
  and untouched confirmation. A single-seed loss does not close TCN.
- Complete/report this bounded predictor study before fitting U-PCR or other
  fusion. Better prediction loss or differing residuals alone is insufficient.
