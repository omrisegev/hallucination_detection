# Cumulative vote fusion v2 — frozen development protocol

Authorized experiment, 2026-09-22. Branch `codex/cumulative-vote-fusion-v2`, base
`c3b018bd1`. No acceptance threshold or automatic promotion. This is development
on an already inspected population, not confirmation on untouched data.

## Population and provenance

All 6,800 ProcessBench answers (4 subsets, Qwen3-4B/8B; 4,442 erroneous) and,
separately, 6,969 PRMBench/Qwen3-8B answers. Labels v3, source-group folds v2.
Models fit separately per task/scorer, with PB subsets pooled within scorer.
The same source group cannot cross outer or inner folds, including different
scorers and variants. Main training includes clean answers. PB error-only
training is a separately named Top5 control. Single-step PB answers contribute
no training thresholds but remain in evaluation with prediction 0.

Input files, sizes and SHA256 hashes are recorded in INPUT_FREEZE.json. Code
hashes and actual source snapshots are recorded before fits in RUN_FREEZE.json.
Paths are configuration, not hardcoded Python paths or cross-worktree imports.
Existing aligned token matrices are reused as data, with all token/step counts,
feature names and bounds checked. Historical CT7 and token anchors must pass.
Historical baselines missing 21 answers keep explicit coverage and IDs; no silent
imputation makes them full-population comparisons.

## Profiles and models

11 risk-oriented, digit-free channels. Token profiles use within-answer robust
median/IQR normalization (std fallback), exactly the reviewed feature-bank
readout family. Ordered readouts: Top5, Top10, max, mean, negative LoG sigma2.5
then Top5, absolute centered CUSUM then Top5, onset80. Onset uses
min+0.8(max-min), keeps the prefix through first crossing, excludes the suffix;
constant profiles cross at step0. There is no numerical tie perturbation.

Main roster fixes Top5 for every channel. Separate label-selected roster chooses
each channel's readout on outer-training answers only: PB equal-subset SLA on
erroneous training answers, PRMB within-answer AUROC on eligible training
answers, excluding onset80. Inner-fold fits reselect using inner training only.
Exact ties follow the above order.

PB binary: argmax profile, votes 2I(s_j<=n)-1. Soft: standardize each channel's
finite step profile within answer, softmax temperature1, cumulative votes 2F-1.
Both train over every internal threshold n=0..S-2. Equal subset mass, equal
trainable-answer mass within subset, equally divided over that answer's rows.
PRMB binary: strictly above within-answer median is positive; ties negative.
PRMB continuous: within-answer standardized scores. Each answer has equal mass.
There is no assumption that half the steps are erroneous.

Seven spectral arms: binary equal/spectral/binary-LSML; soft
equal/spectral/continuous-LSML; binary continuous-LSML bridge. Simple spectral
uses weighted, training-standardized columns in both encodings. Binary LSML
uses raw votes and preserves within-group sign, with zero mapped to +1.
Continuous models save and apply training mean/std. Spectral factors retain
negative weights. Group search uses weighted covariance, the canonical
Jaffe score matrix/residual, K=2..8 and deterministic spectral clustering.
Small groups and residuals are saved; no theoretical reliability claim is made.

PB evaluates virtual all-negative and all-positive endpoints as well as every
internal threshold. Orientation is fit-only, requiring the positive endpoint
to be higher. PAVA, nonnegative differences and normalization produce a readout
mass (not a calibrated posterior). Mode with earliest tie is primary, median
secondary. Invalid endpoint range or readout is a recorded native failure;
equal of the same encoding supplies the full-population table with a flag.
PRMB uses direct scores, without cumulative encoding/PAVA/mode.

## EM

Both DS and hierarchical EM run regardless of spectral results, on binary votes
only. Hierarchical groups are discovered on the corresponding training matrix
and then held fixed. The model is Y -> alpha_g -> votes_j, with exact
marginalization over group states, matching the latent structure in
https://proceedings.mlr.press/v51/jaffe16.pdf . DS is the independent-observation
special case. Labels never enter EM.

Weighted likelihood; 5 fixed starts: spectral score responsibilities, equal
consensus responsibilities, 3 seeded logit perturbations (sd0.25) of spectral
initial parameters. Best training likelihood chooses the start. Max1000 EM
updates; stop after3 relative improvements below1e-8; probability clip1e-6.
Repeated binary observation patterns are combined with summed weights, an exact
sufficient compression of the same likelihood. Save all likelihood trajectories,
convergence, start selection, fitted probabilities and boundary/small-group flags.
Initialization seed20260922. Reliability parameters remain model estimates.

## Evaluation and uncertainty

PB gate-free SLA primary: equal macro of8 cells, with each cell and scorer
reported; tolerance1, early/late, distance and median diagnostic. Shared-gate
F1 uses immutable CT7 decisions on all answers. Each cell's clean accuracy and
gated exact-error accuracy enter a harmonic mean, followed by macro across cells.
Native historical gates are reported separately. No gate or threshold is fitted.

Mind-the-Gap replay uses the existing local adapter's adjusted EMA span5 and
worst negative flux within official spans. Final scores are reconstructed from
the original raw top-k logprobs in float64, with every source ID/span verified. The prior
evidence-drop artifact used unadjusted EMA: retained separately, not substituted
for this adapter. Its float32 evidence gives identical peak indices to the final
float64 replay (maximum step-score difference 6.72e-8). Paper figures are context
only: Qwen3 and teacher forcing are shared, but prompt/revision/population details
and the underspecified token-to-step rule prevent a reproduction claim.

PRMB primary: mean within-answer AUROC, explicit mixed-label eligibility (6,030).
Step AUROC/AP are calculated within outer folds and then averaged. PRMScore uses
the existing official port (valid=1), with its disclosed redundancy/circular
validity adaptation and exclusion of synthetic-correct controls from pooled F1.
Fixed threshold: training-risk quantile0.8. Tuned: 50 quantiles from0.5 to0.99,
chosen by pooled inner-OOF PRMScore. Each inner fit calibrates thresholds on its
own training predictions and applies them to inner holdout; selected quantile
is recalibrated on outer-training predictions. This avoids pooling raw scores
from models with different scales. Same grid for every method. Frozen historical
score functions have no new fusion fit, but use identical fold calibration and
threshold budget. Prior development access is disclosed. Supervised PRM separate.

Controls: each single channel, longest token span, fixed early/middle/late
relative-position profiles. Token-order control independently permutes each
channel inside each answer at one fixed seed, rebuilds Top5 profiles and refits
all models. This is a mechanism probe, not a null-distribution estimate.

10,000 common source-group bootstrap draws, preserving scorer linkage, support
primary-endpoint intervals and paired contrasts. Report exact contrast count,
Holm-adjusted p-values and Monte Carlo resolution; do not promote a winner.
Secondary fold-ranking diagnostics are descriptive, not an OOF pooled AUC.
Planned contrasts: softness for equal/spectral/continuous; bridge binary-LSML vs
continuous; learned vs equal; selected vs Top5; DS/HEM vs spectral and each other;
new methods vs CT7/MindGap; shuffle and error-only controls vs main counterpart.

## Execution and artifacts

`python scripts/experiments/run_cumulative_vote_v2.py --config
configs/cumulative_vote_fusion_v2.json --stage all`

Stages prepare, spectral, em, inner, report can be resumed independently.
Each job writes predictions, fold model pickle, transparent JSON parameters and
training/testing source groups. Final tables include coverage/fallbacks/runtime,
all OOF step scores, answer locators/gates, CSV/JSON, and Hebrew HTML. Previous
results are read-only. Llama historical replay belongs only in an appendix.
