# Early/online hallucination detection from existing caches — v1

Status: frozen CPU-only retrospective screen, 2026-08-16. This protocol does
not authorize new inference, GPU jobs, raw-data mutation, or an exact
competitor reproduction.

## Question

Can the existing token-telemetry caches support a useful answer to the common
problem behind the recent online-detection papers: predict final-answer error
before generation ends? A second, method-specific question is whether the
causal prefix score has already converged to the score that the same frozen
model assigns to the completed trace.

## Units and labels

- Unit of evaluation: one generated trace. Traces sharing a question remain in
  the same split and bootstrap group.
- Target: final-answer error (`1 - correctness`).
- Absolute monitoring budgets: 16, 32, 64, 128, 256, and 512 generated tokens.
- The primary fixed-budget risk set contains only traces whose final length is
  strictly greater than the budget. Finished traces are not silently mixed
  into an "early" cohort.
- Cells with one target class, fewer than 30 usable traces, or fewer than 10
  traces of each class are descriptive/blocked, not headline evidence.

## Frozen scorers

The calibration-question half is used to fit the label-free feature transform,
IU-PCR weights, and score orientation once. No model component is refit at a
later budget. Labels enter only threshold/declaration calibration.

1. `iu28_no_length` (primary): the 28 token-resolved streams in
   `token_feature_views.py`; final trace length is absent.
2. `iu29_elapsed_length` (task adapter): the same streams plus the length of
   the currently observed prefix. This is not the literal historical
   `trace_length` feature until the trace ends, and must be named an adapter.
3. Entropy controls: prefix mean entropy and prefix maximum entropy.
4. `deepconf_entropy_w{32,64}`: the existing entropy-based lowest-group-
   confidence proxy. It is not called code-exact DeepConf because raw-logit
   equivalence has not been established.

For IU-PCR, a prefix is rebuilt from `row[:budget]`; a full-trace feature matrix
is never constructed and sliced. The decision score is the maximum token risk
over the causally rebuilt prefix. The frozen mixed-v2 transform and two-
component, no-exclusion IU-PCR settings match the fixed application pipeline.
All-NaN telemetry families are dropped by the fit and recorded in diagnostics.

## Split and fit

- Deterministic grouped split by stable question/group identifier: 50%
  calibration, 50% evaluation.
- The unlabeled fit samples the same number of token rows from every calibration
  trace, preventing long answers from dominating the covariance fit.
- The evaluation half is never used to fit transforms, IU weights, score
  orientation, decision thresholds, tolerances, or stopping thresholds.
- Results are reported per cell. A future multi-cell macro must weight cells
  equally and use grouped question-level intervals.

## Convergence endpoints

For every method, budget, and at-risk evaluation cohort, report:

- AUROC and AUPRC for final error;
- Spearman and Kendall correlation with the completed-trace score;
- mean absolute score error normalized by the calibration final-score standard
  deviation;
- agreement with the frozen final-score decision threshold;
- flip rate relative to the preceding available budget;
- above-chance AUROC recovery,
  `(AUROC_b - 0.5) / (AUROC_final - 0.5)`, when defined.

Per trace, retain the complete score trajectory, the oracle last decision flip
(diagnostic only), the first budget inside the frozen final-score tolerance,
and the number of threshold crossings.

## Early declaration policy

The calibration half chooses two score thresholds: a high threshold declares
"hallucination", a low threshold declares "not hallucination", and the middle
continues/abstains. A declaration requires two consecutive monitored budgets on
the same side. Threshold pairs are searched only on calibration trajectories.
The primary constraint is the question-level probability of **ever making a
wrong declaration over the complete monitoring horizon**, not a separate
fixed-time false-positive rate. Among calibration-feasible pairs, maximize
coverage and then prefer earlier declarations. Default calibration tolerance is
10%; held-out performance is reported without repair.

Report held-out coverage, ever-wrong rate, false alarm (hallucination declared
on a correct answer), false clearance (non-hallucination declared on a wrong
answer), decision budget, and potential remaining tokens. "Potential" is not
realized token saving: no forced-closure branch was generated.

## Competitor claim map in this screen

- DeepConf: prefix confidence and offline ranking/filtering proxy where K>1
  exists; no native single-trace stopping claim is attributed to it.
- REFRAIN: text-replay trigger only where complete generated text is present;
  association and potential suffix savings only, never realized accuracy.
- Streaming Hallucination Detection: same prefix-to-final-error target; hidden-
  state and proprietary/asset-gated ceilings remain unavailable.
- LEASH: entropy/logit-margin sequential controls where their telemetry exists.
- Online Auditing of Information Flow: supplies the error-delay formulation;
  its supervised sequential rule is a separate access tier.
- uPRM: only on step-labelled ProcessBench and not pooled with answer-level
  early detection.
- HALT/noncausal methods: ceilings only.

## Gate for exact paper-condition reproduction

New inference is justified only if the existing-data screen shows a grouped
paired improvement over the strongest same-access baseline whose interval
excludes zero, the same direction across more than one model/dataset family,
useful held-out ever-wrong operating points, and robustness to absolute budgets,
length, and truncation. A one-cell positive is a pilot, not confirmation.
