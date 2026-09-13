# Whole-answer position fusion: Gaussian factor, RBM and IU-PCR

User-authorized amendment, 2026-09-13. Branch codex/answer-position-fusion-v1,
base 6249db384. The previous within-step experiment ended at smoke27 PASS;
its source and artifacts stay unchanged. No full result from it is claimed.

## Question

Does feature reliability vary with position in the COMPLETE answer? Sixteen
equal token-interval regions span the full cached answer, without resetting at
reasoning-step boundaries. Steps remain only the official readout/label units.
The cache is teacher-forced scoring of provided reasoning answers; this is
offline gray-box localization, not a newly generated answer or causal detector.
Answer length is known at scoring time. Relative position is not token dynamics.

Keep the exact answer-standardized RBM12 feature bank: entropy15, varentropy15,
moment3_15, selected_surprisal, selected_squared, selected_cubed, moment4_15,
selected_power4, moment5_15, selected_power5, moment6_15, selected_power6.
No raw source or frozen original RBM model is changed. All token observations
participate in fitting: source groups have equal weight, answers within a group
equal weight, and tokens within an answer equal weight. Region overlap assigns
fractional token observations at boundaries, not independent copies.

## Algorithms (names are not interchangeable)

Gaussian factor: x|j = mu_j + w_j*h + epsilon, h~N(0,1), diagonal D shared
over regions. Estimate mu from training answers. Fit centered regional covariance
by Gaussian likelihood, W=UV', rank1 or rank2. Fixed-weight baseline uses pooled
mean and covariance. A mean-only control keeps pooled coefficients but subtracts
training regional means: it separates position centering from weight changes.
Score with the exact Gaussian posterior-mean coefficients and fitted intercept.
The previous zero-regional-mean restriction has been removed explicitly; this
is not a silent replay of that earlier implementation.

Gaussian/Bernoulli RBM: the project's existing exact H1 energy,
E_j(x,h)=||x-a||^2/2-h*(b+x.w_j). Shared visible bias a and hidden bias b;
identity conditional visible variance. W=UV' with rank1/rank2; stationary W has
identical rows. Log Z_j (minus the Gaussian constant) is
softplus(b+a.w_j+||w_j||^2/2). Optimize weighted exact likelihood over ALL training
tokens; no CD, labels, pseudo-labels, hidden-class sampling or extra hidden unit.
Output the oriented posterior LOGIT, not the sigmoid. This is a position-aware
extension of the original RBM; it does not consume neighboring token context.

Factor and RBM: two deterministic starts, L-BFGS-B maxiter1000, ftol1e-10,
gtol1e-6, float64. Fixed ridge .5e-4 mean(W^2); select by pure NLL; retain both
starts and convergence information. Factor noise floor remains
max(1e-8,1e-3*mean covariance diagonal). Finite iteration-capped results are scored
and flagged, not described as convergence. Varentropy15 loading is the fixed
risk-sign anchor; it is an assumption, not unsupervised identification of truth.

IU-PCR: call the existing upcr_fit_covariance with IU_FIT_DEFAULTS (two PCR
components, no exclusion/fallback). Use genuine CENTERED regional covariance,
training mean/scale, and transform coefficients/intercept back to the same x
coordinates. Regional raw moments are shrunk toward pooled moments with
alpha=G/(G+16), where G is the training source-group count. This is a fixed
engineering shrinkage rule, not an estimated effective independent token count.
Orient by training covariance with varentropy15. Fixed and mean-only controls
isolate pooling, centering and regional coefficient changes. The regional IU
extension is not a rank1/rank2 parameterization or a new name for an eigenvector.

These objectives model agreements/density, not semantic correctness. More
answers reduce sampling variability; they cannot repair false independence or
identifiability assumptions. Cross-family comparisons also change the latent
family and preprocessing. Within-family controls carry the mechanism claims.

## Roster and controls

- Factor: stationary, stationary coefficients with position mean, rank1, rank2,
  rank2 with shuffled position assignments.
- IU-PCR: stationary, stationary coefficients with position mean, regional,
  regional with shuffled position assignments.
- Exact H1 RBM: stationary, rank1, rank2, rank2 with shuffled position assignments.
- Original answer-local RBM12 Logit+Top10, equal features, position-only early
  and late controls; thirteen frozen reference rows from the prior logit run.

Shuffle the entire answer's POSITION ASSIGNMENTS with an answer-ID seed, in
training and scoring. Never move feature rows between steps, never move labels,
and never shuffle each feature independently. Thus stationary models and
token identities stay invariant, while feature-position alignment is removed.
This tests broad relative-position information, not short-range causal order.

## Contract and decisions

Same 13,769-answer development population; v3 annotations, v2 canonical source
groups/folds. Other-answer UNLABELLED training separately within each cell.
No held outer-fold group enters fitting. PRMScore q=.8 uses inner-fold scores
from models excluding both the outer test fold and the inner calibration fold.
Same external mean-entropy q=.3 gate for ProcessBench; Top10 token mean in each
official step, earliest argmax tie, no first_near_max. Coverage/failure is visible.
The inherited answer-local normalization uses the observed test answer only;
other test answers do not alter any fitted model or calibration threshold.

Five primary contrasts: factor rank1 minus mean-only control, factor rank2 minus
rank1; RBM rank1 minus stationary, RBM rank2 minus rank1; regional IU minus its
mean-only control. Ten thousand source-group paired bootstrap draws; 99% CIs
for these five contrasts (Bonferroni family .05), 95% for descriptive controls.
Keep PB all8/Q4/Q8/per-cell and exact/early/late/clean/masked-peak diagnostics;
PRMB within-answer, pooled and fold-mean AUC, PRMScore, denominator/coverage.
Intervals condition on saved fits/thresholds and do not include prior selection.
No arbitrary performance gate. No winner or confirmation claim from a smoke.

## Implementation and execution

New module spectral_utils/answer_position_fusion.py and dedicated runner
scripts/run_answer_position_fusion.py. Exact source/cache hashes, group lists,
feature order, weights, biases, fit health and failure reasons saved. Checkpoints
per model arm permit interrupted fits to resume without refitting completed arms.
One process, BLAS1; minimum 4GiB free RAM before cache loading; eight-hour cap.
Raw caches are read one cell at a time and are never copied. RBM fitting keeps
weighted token blocks in RAM; smoke forecasts cost, not benchmark efficacy.

Tests: whole-answer clock and no step reset, segmentation-independent token
scores/statistics, fractional short-answer overlap, canonical IU replay, exact
historical RBM objective/gradient reduction, independent two-Gaussian density,
finite differences at both ranks, missing/constant data, unchanged original
Top10, explicit source-fold exclusion and label mutation firewall, independent
metrics and nested calibration review. Smoke27 then full evaluation, returning
results in chat. No HTML, no Conditional-context RBM/TCN/CD or new feature bank.

Sources of the ideas, not claims of paper-exact reproduction:
- Project moment_rbm_fusion.rbm_objective and rbm_literature_completion.ExactRBM.
- Hinton's Gaussian-visible RBM guidance:
  https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
- Factor Analysis model: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
- Project canonical upcr.upcr_fit_covariance and laplacian_upcr.IU_FIT_DEFAULTS.
