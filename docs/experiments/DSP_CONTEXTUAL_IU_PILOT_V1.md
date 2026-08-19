# DSP-Contextual IU Router pilot v1

**Frozen date:** 2026-08-19  
**Status:** retrospective, CPU-only, existing-cache premise test

## Question and claim boundary

This program tests whether already registered causal DSP states organize the
local reliability of the six provenance-balanced feature families for
ProcessBench first-error Localization and causal Early Detection.  Existing
feature orientations remain fixed.  DSP states are neighbourhood context, not
verified nuisance variables and not correctness pseudo-labels.

The prior +2.833 AUROC-point family-specialization number was a label oracle
effect-modification diagnostic.  It is not a routed score and is not evidence
that a runnable router works.  This protocol requires a label-blind runnable
score and evaluates it only after its reference moments, contexts, neighbours,
and scores are frozen.

All available ProcessBench populations have been opened historically.  Any
result is retrospective premise evidence, not external confirmation.  No new
inference, GPU/cluster work, download, or Google Drive mutation is authorized.

## Frozen inputs and splits

Use the deterministic source-question split and repeated-scorer grouping from
`LOCAL_ONLINE_COMPREHENSIVE_V1.md`:

- calibration: 40%, used for all label-blind references and detector thresholds;
- development: Qwen3-4B GSM8K and MATH;
- architecture: Qwen3-4B OlympiadBench and OmniMath;
- audit: Qwen3-8B and Llama-3.1-8B, all four families.

Scorer copies of one source question are repeated measurements.  Resampling is
by source identity and carries copies together.  Dataset families have equal
macro weight.  Fit bundles are physically stripped of correctness, first-error
labels, final-answer correctness, and step spans.

Localization fuses `family6__level` (six coordinates).  Early fuses
`family6__fast_slow` (twelve coordinates) at absolute budgets
16, 32, 64, 128, 256, and 512.  The primary Early endpoint is the equal-family
mean of AUROC at 64 and 128 among traces longer than the budget.

## Frozen context and router

Core context contains calibration-ECDF IU score, `log1p` current position or
budget, and MAD of the six family contributions.  DSP context appends the six
family-resolved values of `innovation`, `shortlong`, `positive_mean`,
`persistence`, and `recovery`.  Every coordinate is robustly scaled on
calibration; the core block and each six-wide DSP block have equal Euclidean
distance leverage.

Localization keeps 32 equal-position landmarks per calibration question.
Early keeps one prefix landmark per question and budget.  A query takes the
nearest landmark from every source question, then the nearest
`min(Q,max(n_min+8,ceil(sqrt(Q))))` distinct questions, where
`n_min=max(32,4m)`.  The eight-question headroom is necessary because a
non-uniform Gaussian kernel on exactly `n_min` neighbours has effective size
strictly below `n_min`.  Gaussian bandwidth is the last included question.
Effective sample size is computed over questions.

The local covariance is centered with kernel weights and shrunk to the global
question-balanced covariance with `alpha=n_eff/(n_eff+4m)`.  If
`n_eff < max(32,4m)`, the top-two covariance is ill-conditioned, or IU fitting
is invalid, scoring returns the exact global IU weights.  Valid local weights
are sign-aligned with global IU and normalized to its L1 leverage.

The registered controls are global IU, IU-rank-by-position bins, core-context
kNN, DSP-context kNN, ordinary IU after direct DSP feature augmentation,
permuted context, random neighbours, and global weights.

## S0 synthetic gate

Twenty fixed seeds cover an informative switching regime, context-independent
null, coherent nuisance, observational equivalence, abrupt onset, variable
length, and repeated-token worlds.  Continue only when the informative world
wins at least 18/20 seeds with mean gain at least +0.005, the null does not
actuate, coherent-nuisance mean/worst loss are no worse than -0.005/-0.020,
token duplication is invariant, and suffix/chunk/fallback mechanics pass.

Observational equivalence is a claim-boundary check: identical observations
must yield identical scores; the method is not required or permitted to infer
which latent bit is correctness.

## S1 routing-premise gate

The label-blind family routing signal is normalized absolute IU weight mass.
After score freeze, family utility is:

- Localization: true-first-error step-top-five mean minus the strongest other
  step for each family, on erroneous traces;
- Early: class-balanced signed calibration-ECDF family-expert rank.

The primary statistic is within-question Spearman across the six families.
Continue only if its grouped 95% interval is above zero, at least three of four
dataset families are positive, DSP exceeds core context in point estimate, and
the observed statistic exceeds the 95th percentile of the fixed context
permutation distribution.  A positive estimate with an interval crossing zero
may use the frozen Qwen architecture cells once as a power extension; no method
choice may change.

## S2 end-to-end gate

Localization reports ProcessBench F1, exact, tolerance-one, clean abstention,
detector AUROC/AUPRC, and signed onset delay.  Early reports AUROC/AUPRC at all
budgets and ever-warning behaviour at calibration 5% and 10% false-warning
targets.

Each task continues independently only with at least +0.005 primary improvement
over matched global IU, grouped lower confidence bound above zero, worst-family
loss no worse than -0.010 Local or -0.015 Early, and a point-estimate win over
bins, direct DSP augmentation, and core-context routing.

## S3 manifold diagnostic gate

S3 opens only after S1 and S2 pass, or after positive S2 point estimates with a
predeclared neighbourhood-instability failure.  LPCA dimension is the smallest
explaining 90% variance, clipped to 2--8, with `d-1`/`d+1` sensitivity.
LTSREx is diagnostic only: it audits which DSP functions parameterize the
sampled manifold and whether budget or missingness dominates.  LEGO is allowed
only if LPCA median bootstrap principal angle exceeds 15 degrees and LEGO
reduces it by at least 25%.  Connection-Laplacian smoothing is excluded because
it is transductive and can blur first-error onset.

No DSP coordinate may be called nuisance without an external nuisance-validating
intervention.  In its absence the method is `tangent-conditioned IU`, not
`Nuisance-Tangent IU-PCR`.  Added complexity requires at least +0.0025 over
DSP-kNN with positive paired interval and no onset delay.

## S4 audit and decision

Audit the frozen surviving task paths on Qwen3-8B and Llama-3.1-8B.  A fresh
confirmation request requires a positive interval against the strongest Tier-A
direct competitor and wins in at least three of four families.  The only final
statuses are:

- `STOP_NO_ROUTING_SIGNAL`
- `STOP_ROUTER_NO_GAIN`
- `DIAGNOSTIC_ONLY`
- `RESEARCH_CANDIDATE`
- `REQUEST_FRESH_CONFIRMATION`

Every stage writes question/cell tables, grouped intervals, diagnostics, score
hashes, and an explicit gate decision.  Final outputs are `REPORT.md`,
`REPORT.html`, `DECISION.json`, `AUDIT.json`, and `RUN_MANIFEST.json` under
`results/dsp_contextual_iu_pilot_v1/`.
