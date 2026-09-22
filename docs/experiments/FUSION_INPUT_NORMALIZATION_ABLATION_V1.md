# Fusion input-normalization ablation v1

Status before execution: **PROTOCOL FROZEN / NOT YET SCORED**

## Question

Do the frozen four-view Renyi fusion algorithms benefit from receiving the
oriented features in their natural units, rather than the current independent
within-answer z-score?

## Fixed components

- Population, official steps, source-group folds and labels are inherited from
  `renyi_position_temporal_fusion_v1`.
- Feature bank is fixed to q15 `H0lim`, `VE0`, `VE0.75`, `VE1`; no H1/Hinf or
  q50 feature is added in this ablation.
- Orientation, Top10 token-to-step readout and earliest-argmax prediction are
  unchanged.
- ProcessBench uses the development-frozen tail15 Top10 gate with one uniform
  q=.33. PRMBench has no answer gate and retains the nested q=.8 calibration
  contract.
- No labels enter a fusion fit. No normalization or solver is selected on an
  individual benchmark or cell.

## Input representations

For the already oriented natural-unit matrix `X`, with its within-answer
column mean `mu` and standard deviation `sigma`, compare:

1. `answer_z`: `(X-mu)/sigma`, the current implementation.
2. `scale_only`: `X/sigma`, which retains each answer's absolute feature level.
3. `raw`: `X`, the literal natural-unit input.

The literal `scale_only` and `raw` arms intentionally test bypassing centering.
IU-PCR documents z-scored input as its expected contract, so these arms are
empirical deviations and their covariance/condition diagnostics must be
reported; they are not silently re-centered inside the answer-local solver.

## Solvers

- equal token fusion;
- answer-local IU-PCR;
- other-answer stationary IU-PCR;
- other-answer position-varying IU-PCR;
- answer-local shrinkage IU with a pooled prior;
- answer-local shrinkage IU with the real position prior;
- the matched position scale-only control.

All other-answer priors are rebuilt in the representation being evaluated,
with the original outer and two-fold source-group exclusions.

Joint L-SML is explicitly excluded from this run. The four-view bank cannot
satisfy its structural requirement of at least three groups with at least
three varying members per group, and the full multistart fit is deferred unless
this cheaper ablation finds a useful non-z input.

## Endpoints and decision

Report PB raw exact, PB all-eight under the frozen q=.33 gate, PRMB within and
fold AUROC, pooled OOF AUROC, and PRMScore. Every raw/scale-only solver is
compared with its own answer-z version, and the best arm is also compared with
the current development-frozen q15 per-view-Top10 locator.

A non-z representation is carried into Experiment 3 only if it improves PB by
at least 0.2 percentage points or PRMB within by at least .002 without losing
more than the same margin on the other primary endpoint. This is a development
screen, not external confirmation.
