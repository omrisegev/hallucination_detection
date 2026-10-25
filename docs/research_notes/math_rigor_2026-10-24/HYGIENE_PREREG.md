# Pre-registration: statistical hygiene for CHAIN_N16 (W7, W11, W12), 2026-10-25

ID `prm-boost-hygiene-v1`. JSON: `results/prm_boost_math_rigor_v1/HYGIENE_PREREG.json`. Code: `stage7_hygiene.py`.
**This file, the JSON and the code are committed before the first run.** The method is unchanged: CHAIN_N16, fitted
without labels. Labels only score.

## Why

The math review raised three hygiene points.
- **W11.** The bootstrap CIs keep the fitted scores fixed, so they ignore the variance of the fit itself.
- **W7.** The telemetry partition is chosen by a heuristic and nobody has checked whether it is stable.
- **W12.** The fit is transductive: it is fitted on the same answers it scores.

Parity was checked before writing this pre-registration. CHAIN refitted with the frozen fit/apply code on all rows
matches the exported CHAIN to 2.1e-7 (PRMBench Qwen-7B). One fit takes 0.8 s.

## Procedures

**W11, refit bootstrap (E1).**
- **Draws.** 200 draws of the source groups with replacement, seed 20261026. External cells resample answers.
- **Refit.** Each draw refits the whole of CHAIN on the resampled rows: telemetry partition, group weights, cross
  weights and sign.
- **Apply.** The frozen refit is applied to every answer. Per-answer within-AUC and GAP counts are weighted by the
  draw's multiplicities.
- **Comparison.** On the same draws, the fixed-score bootstrap uses the exported CHAIN.
- **Endpoint.** Δ = CHAIN − PRM in within-AUC, with a 95% percentile CI.
- **Reported without a criterion.** Δ in GAP PRMScore, where the PRM side is GAP on the PRM score; and the CI-width
  ratio, refit over fixed.

**W7, partition stability (E2).**
- **ARI.** For each refit, the ARI between its telemetry partition and the full-data partition, over the 16 channels.
- **K.** The modal K and its share of the draws.

**W12, split-half inductive fit (E3).**
- **Split.** Stage 4's length-stratified 50/50 split of the source groups, seeds 20261025 + 0..9.
- **Inductive arm.** Fit CHAIN on half A without labels, freeze it, and apply it to half B.
- **Comparison.** On B, compare it with the exported transductive CHAIN, fitted on everything, and with the PRM.
- **Metrics on B.** Within-AUC and GAP PRMScore.

## Pass rules (PRMBench, 5 PRMs)

- **E1.** The refit CI of the within-AUC Δ excludes 0 with the same sign as the fixed-score CI, in ≥ 4/5 PRMs. A
  fixed-score CI that already includes 0 counts as met when the refit CI also includes 0.
- **E2.** Median ARI ≥ .80 and modal-K share ≥ .80, in ≥ 4/5 PRMs.
- **E3.** Over the 10 seeds, in ≥ 4/5 PRMs, all three hold:
  - the median of (inductive − transductive) within-AUC on B is ≥ −.005;
  - the median of (inductive − transductive) GAP PRMScore on B is ≥ −0.5;
  - the median of (inductive − PRM) has the same sign as the median of (transductive − PRM).
- **Verdict.** Pass if E1, E2 and E3 all hold; partial if one or two hold; fail if none holds.

## Reported without a criterion
- Socratic and Hard2Verify. Hard2Verify is evaluated on the 177-answer common set, and its fits use the whole cell.
- Per-PRM K distribution, and the telemetry channels that most often change group.

A partial or a fail is recorded with `/negative-result`. Nothing is tuned after the results.
