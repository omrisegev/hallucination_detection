# Fair-protocol re-scoring of the N16 L-SML / IU-PCR step scores (2026-10-16)

Nothing was refitted. The scores come from `../SCORES_*.npz`, which were fitted label-free. Labels are used only for scoring.
Protocol definitions are reused from `results/agent_step_telemetry_v1_bias_audit_2026-10-16/`:
- same-format test (`struct`, verbatim);
- length matching (0.8 ≤ len ratio ≤ 1.25);
- residualization: label-free per-subset OLS on log step length (+ position for `lenpos`);
- first error vs the clean (+1) steps before it;
- injected twins (a group with exactly one safe and one injected run).

Length is the baseline's raw step token count. Bootstrap: 1000 draws, seed 0, over pairs/trajectories/runs.
Replication check: the baseline rows reproduce the audit, e.g. TPRM ToolSandbox same-format entropy 0.752, APB first-error vs prior entropy 0.659, len 0.698 (length-residualized 0.555), AFT injected entropy at m 0.820.

Run:
```
python -I fair.py <scratchpad/out> .. <inputs_v2_20261013> <apb ref dir> <apb judge dir> .
```
Output: `FAIR_METRICS.json`.

## Choices
- **TPRM.**
  - bfcl, tooltalk and ToolSandbox: same-format pairs only, which here are all pairs except 17 in ToolSandbox.
  - GTA: all 118 pairs, as a format probe; only 4 are same-format.
  - Residualization coefficients are fitted over all actions of the subset.
- **APB.**
  - Units are the 468 error trajectories with at least one +1 step before the first -1.
  - "pairwise" is the mean over trajectories of P(first error > each prior clean step).
  - "pooled AUROC" is the audit's `first_err_pooled`: first-error steps vs every +1 step that precedes any error, across all trajectories.
  - Localization is the argmax over {prior +1 steps, first error}, with ties counted at their expected value. Random is 1/(prefix length).
  - Position wins 1.000 by construction (the first error is always last in the prefix). Raw results here are therefore position-confounded, so `lenpos` is the honest column.
  - Judge: the Qwen3-8B judge's -1/0/+1 label is used as the score (-1 → 1, 0 → .5, +1 → 0). Its official `first_neg1_idx_acc` 0.407 also credits "no error" trajectories, so it is a different metric and not comparable.
- **AFT (a), diagnosed.**
  - Each diagnosed run is scored on the mean over steps up to and including its mistake step.
  - The audit defines no rule for safe runs, so the coordinator's fallback was used: each safe run is truncated to the step count of a random same-domain diagnosed run (seed 20261016). Safe runs shorter than the cut are kept whole (agentic 40/405, math 48/396).
  - Localization over the online prefix is the same as "argmax is the last step"; the full-run argmax (the audit's definition) is also given.
  - Only cellz scores are used, because ansz normalizes with future steps. The cellz fit itself is transductive over all unlabeled steps.
- **AFT (b), injected.** Twins are compared at the mistake step only. A length-residualized version (per-domain OLS) was added as an extra control.

## Results
See `FAIR_METRICS.json`; the summary is in the hand-off to the coordinator.

## Main-session check (2026-10-16): AFT diagnosed localization is mostly a position prior
Constant-index baselines (step chosen with labels, so an oracle upper bound for a fixed position):
- agentic diagnosed (n=322): always picking step index 3 hits 0.342 > IU-PCR 0.233. IU-PCR argmax lands on index 1/2/3 in 240/322 runs.
- math diagnosed (n=64): always picking step index 1 hits 0.313; IU-PCR picks index 1 in 38/64 runs and hits 0.359.
So the diagnosed-localization "signal" is largely an early-step position effect, not error-specific.
