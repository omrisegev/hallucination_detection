# Same-answer prediction view: feasibility audit v1

Date: 2026-09-07. Step 311. Scope: source-history and label-free feasibility.
This stage does not fit or evaluate new localization heads. IU-PCR and Joint
L-SML remain the method. A prediction residual can only add measurements to
their existing matrix; it is not a replacement detector.

## Decision question

Can a small predictor fitted strictly inside one answer produce usable extra
window features, and how similar are these to the existing measurements?
Feasibility and decorrelation cannot establish correctness information. A
noise feature can be different and still harm localization.

## Historical audit, before implementation

The old scalar AR/Kalman final-answer pilot is not the whole history:

| Existing path | Actual role and fitting scope | Evidence boundary |
|---|---|---|
| `spectral_utils/temporal_models.py`: `ar_innovation_scores`, `kalman_innovation_scores` | Whole-trace scalar AR(2) in-sample error or ordinary constant-velocity Kalman innovations | Step 151: one 200-answer GSM8K cell; AR MSE .717, KF NIS .703, DeepConf .735, L-SML5 .754. Not a localization comparison and not learned KalmanNet |
| `spectral_utils/local_online_comprehensive.py`: `causal_operator_matrices`, `fit_trajectory_head_prepared` | Positive deviation from prior slow EMA; matrices from multiple calibration answers stacked to fit IU | Existing report says event coordinates were inconsistent. Selected local family6 used level, not innovation; its transfer .3662 versus entropy .3614 is a finalist result, not the innovation arm's score |
| `spectral_utils/unified_causal_iu.py`: `_OneChannelDSP.update` | Signed `innovation64 = value - previous_ewma[64]` among causal coordinates | Causal coordinate construction is not proof of answer-only normalization, fusion fitting or parameter selection |
| `spectral_utils/token_temporal_innovation_b3.py`: `fit_innovation_map`, `fit_token_b3_ladder` | Lag/time predictors and optional cross-stream support fit on donor questions; residuals added to continuous B3 | Implemented protocol and tests located; a filename search of local results/cache found no matching named Phase-2 freeze/evaluation. This is not proof it never ran elsewhere |
| `spectral_utils/ciw_cross_scale_localization.py` | Token coordinates predicted from whole-answer coordinate means and CIW answer score; donor-row cross-fitting and IU input blend | Existing 13-cell primary: PB .308301 / PRMB .582489; previous adapter .309136 / .581138. No two-task improvement. Different fitting and response-head contract from current 110 answers |

These observations correct an incomplete history search. They do not reopen
an old method under a new name. The bounded new difference is a self-predictor
whose coefficients use ONLY preceding pairs from the SAME answer, producing
extra columns for the current width-eight fusion representation. This is an
AR(1) adaptation, not KalmanNet, a flow, or a causal-discovery graph.

## Frozen mechanics

- Use the exact 110 Step-306 input shards and corrected source-group identities;
  no new cohort selection. All are already exposed development answers.
- Nine existing primitive streams, unchanged moment27/context27 banks and
  width-eight grid/fit indices. Append nine mean absolute residual columns.
  Keep the original 27 columns byte-for-byte. No new model inference.
- At token t use only pairs k=1,...,t-1. Fit a univariate intercept/slope
  from prefix pair means/covariance; clip the OLS slope to [-1,1]. Shrink the
  fitted prediction toward x[t-1] with eta=(t-1)/(t-1+16). Use Welford
  updates after predicting the current token. No external fitted quantities.
- Two controls: last observation and EMA32 initialized at the first token,
  both predicted before incorporating the target. No parameter search.
- First token has no prediction and is excluded. First window has seven
  residual observations; other width-eight windows have eight. Original
  feature support stays eight. The end-anchored overlapping window is scored
  but is not an extra independent fitting row.
- Diagnostic prediction MSE uses tokens t>=17 (zero-based), after at least
  16 past pairs. Every stream/answer remains in the diagnostic; no label or
  best-performing stream selection. Zero denominator gives undefined ratio.
- For all three residual controls and both banks, report Spearman correlation
  with entropy level and maximum absolute correlation with each original
  column. Report active-column counts and centered matrix rank, which can
  saturate at N-1. These are redundancy diagnostics, not error information.
- No correctness labels, previous EVALUATION files or error-hit sets opened
  by the runner. Hash source closure, raw shards, original score/metadata
  snapshots, protocol and test source before running. Do not mutate originals.
- CPU execution cap 180 seconds; atomic per-answer checkpoints permit resume.
  Independent batch predictor/EMA/window/rank/correlation review, meaningful
  synthetic tests, source hashes and HTML link checks are required.

## Gate for the next quality experiment

Do not promote a method from this audit. If mechanics and support pass, the
next bounded quality test must include unchanged IU/Joint/graph anchors,
the SAME cores with the extra columns, and equal aggregation with the same
columns. Keep a last-observation residual control to test whether fitting
the predictor adds anything. Keep graph-zero/permutation controls when
testing graph Joint. Explicitly register augmented-Joint failures and fallback
before labels; bank eligibility is not a validated quality selector.

Use both PRMBench and ProcessBench metrics, common-coverage comparisons,
within-answer ranking, original-IU-gate diagnostic and source-group paired
uncertainty. Freeze the roster before quality evaluation. The predictor is
causal but full-answer normalization/fusion/step decisions remain offline.
Predictability, feature diversity and a healthy Joint fit do not establish a
localization win. Broader named supporting methods and confirmation stay open.
