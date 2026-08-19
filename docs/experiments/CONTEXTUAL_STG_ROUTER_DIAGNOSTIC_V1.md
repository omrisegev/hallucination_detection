# Contextual-STG Router Diagnostic v1

## Status and scientific claim boundary

This is a retrospective, supervised mechanism diagnostic on already-opened
Qwen3-4B ProcessBench calibration/development partitions.  It is not a
label-free detector, external confirmation, or a promotion experiment.  It
asks one question only: do the currently measured core/DSP contexts contain
enough information for a supervised conditional gate to improve over a global
combination of the same family features?

The explanatory feature directions remain frozen in risk orientation.  The
conditional model may only vary non-negative family leverage.  Context has no
direct path to its prediction.  Therefore the experiment concerns conditional
reliability, not sign/orientation.

No inference, GPU, cluster, download, Drive access, architecture/audit label,
or fresh ProcessBench cell is allowed.

## Frozen cells and partitions

- Model: Qwen3-4B.
- Families: GSM8K and MATH, reported separately and as an equal-family macro.
- Fit: the existing deterministic `calibration` partition from
  `local-online-v1`.
- Evaluation: its existing `development` partition.
- `architecture` and `audit` remain unopened.
- All resampling and weighting is at source-question level.

## Inputs

All views are derived causally from `family6`, whose six risk-oriented families
and calibration scaling are frozen by the existing comprehensive Local/Online
pipeline.

### Localization

One sample is one valid reasoning step.  Its six explanatory features are the
top-five mean within the step for each `family6__level` curve.  The binary
training target is one only for the annotated first-error step.  Clean steps
and non-onset steps are zero.

### Early detection

One sample is one source question at budget 64 or 128, when the trace is longer
than that budget.  The twelve explanatory features are the endpoint values of
`family6__fast_slow`.  The target is answer-wrong.

### Contexts

`core` contains the calibration ECDF rank of a label-blind IU score, log token
position/budget, and MAD of the six label-blind family contributions.

`dsp` appends endpoint/step-top-five summaries for `innovation`, `shortlong`,
`positive_mean`, `persistence`, and `recovery` for all six families (30
coordinates).  Context construction never reads a target.

## Models and controls

- `global_lr`: balanced logistic regression on explanatory features only.
- `context_only_lr`: balanced logistic regression on context only.
- `augmented_lr`: balanced logistic regression on explanatory plus context.
- `cstg_core`: oriented c-STG using core context.
- `cstg_dsp`: oriented c-STG using core plus DSP context.
- `cstg_dsp_permuted`: same model after independent deterministic row
  permutation of context in fit and evaluation partitions.

The c-STG Gaussian relaxation is
`gate=clip(mu(context)+Normal(0,0.5), 0, 1)` during fitting and
`clip(mu(context),0,1)` at evaluation.  The hypernetwork has one 16-unit ReLU
layer.  The prediction head is linear with non-negative normalized feature
weights.  Fixed optimization: Adam, learning rate 0.005, weight decay 1e-4,
600 maximum epochs, 200 minimum epochs, patience 100, sparsity coefficient
0.01.  Scores are averaged over seeds 11, 23, and 47.  No development result
may select a hyperparameter.

Logistic controls use class-balanced fitting.  c-STG additionally gives every
source question equal total loss mass before class balancing.

## Metrics

Localization converts step logits into a per-question maximum detector and
argmax locator.  The detection threshold is selected on calibration only.
Primary metric is ProcessBench F1; exact error localization, clean abstention,
within-one, and detector AUROC are secondary.

Early detection primary is equal-family mean AUROC over budgets 64 and 128;
AUPRC and per-budget/family results are secondary.  Ranking metrics are
computed independently per family/budget and then averaged; out-of-fold scores
from different fits are never concatenated into one AUROC.

Paired 95% intervals use 2,000 deterministic source-question bootstrap draws.

## Decision

`CONTEXT_HAS_ROUTING_SIGNAL` requires, for at least one task:

1. `cstg_dsp - global_lr >= 0.005` on the task primary;
2. paired interval lower bound above zero;
3. no family loss beyond 0.010 Localization or 0.015 Early;
4. `cstg_dsp` exceeds `augmented_lr` and `cstg_core` at the point estimate;
5. `cstg_dsp_permuted` does not reproduce the gain.

Otherwise the decision is `STOP_CONTEXT_NOT_SUFFICIENT`.  A positive result
only opens a separately registered DiSC/intervention/LTSREx audit.  It does not
authorize LTSREx, LEGO, or a final label-free router by itself.
