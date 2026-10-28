# Between-group estimators (HEM, spectral start, plain SML) and the next-step PRM view in CHAIN (CPU, 2026-10-28)

**Pre-registration:** `docs/experiments/PRM_VIEW_SELECTION_ESTIMATORS_NEXTSTEP_PREREG_2026-10-28.md` (240bc0ea).
Development data; labels only score.

**Cells:** PRMBench, Socratic and Hard2Verify × Qwen-7B, Qwen-72B, PURE and RLHFlow (12 cells).

**Parity before reading:**
- Part A: DS-EM reference against `stage1/SEARCH.npz`, `stage2/SEARCH.npz` and `h2v_holdout/`: 40/40, max diff 0.0.
- Part B: recomputed CHAIN_N16 against STATUS_MAIN: 12/12, max diff 1.4e-14.

**Code and outputs:** `estimators.py`, `RESULTS.json`, `RESULTS_A.csv`, `RESULTS_B.csv`, `PARITY.json`, `run.log`.
The run took about 8 minutes on one process.

## Part A: changing only the between-group estimator

There are 48 comparisons against the DS-EM reference: {SUB5, N16} × {+PRM, alone} × 12 cells. A gain or a loss means
the paired 95% interval excludes 0.

| Arm | Gains | Losses | Ties |
|---|---|---|---|
| DS-EM, spectral start | 0 | 3 | 45 |
| HEM, spectral start (primary) | 8 | 16 | 24 |
| HEM, majority start | 8 | 16 | 24 |
| plain SML between groups (no EM) | 9 | 18 | 21 |

**Pre-registered verdicts.**
- **The spectral start changes nothing.**
  - HEM with the spectral start equals HEM with the majority start in 48/48 comparisons. `fit_em` keeps the
    best-likelihood start among 5, and the starts converge to the same solution.
  - DS-EM with the spectral start equals the reference in 45/48.
- **HEM is not preferred over DS-EM.** The rule asked for at least 4 gains and no losses; it has 8 gains and
  16 losses.
  - HEM loses on PRMBench: N16 −0.55 to −2.36, in 7/8.
  - It gains on Socratic RLHFlow (+3.7 to +5.0) and Hard2Verify RLHFlow alone (+4.5).
  - It loses heavily where it matters most, Socratic strong PRMs with SUB5 + PRM: Qwen-72B −4.8, PURE −5.6.
- **Plain SML between groups is worse on balance:** 9 gains, 18 losses. Its gains are concentrated in N16 + PRM,
  and its largest is Hard2Verify RLHFlow (+15.2), where the PRM is near useless.

## Part B: the next-step PRM reward as its own CHAIN group

| Cell | PRM | CHAIN | CHAIN + next | Δ vs CHAIN [95% CI] | next weight share |
|---|---|---|---|---|---|
| PRMBench Qwen-7B | 68.01 | 70.40 | 70.95 | +0.55 [+0.29, +0.82] | .12 |
| PRMBench Qwen-72B | 69.49 | 71.09 | 72.06 | +0.97 [+0.67, +1.27] | .09 |
| PRMBench PURE | 66.24 | 69.51 | 70.14 | +0.63 [+0.37, +0.90] | .12 |
| PRMBench RLHFlow | 59.39 | 66.25 | 64.64 | −1.62 [−2.05, −1.22] | .22 |
| Socratic Qwen-7B | 70.10 | 66.89 | 67.63 | +0.74 [+0.41, +1.06] | .07 |
| Socratic Qwen-72B | 71.72 | 57.43 | 57.68 | +0.24 [+0.08, +0.41] | .02 |
| Socratic PURE | 70.59 | 64.99 | 66.28 | +1.29 [+0.92, +1.65] | .11 |
| Socratic RLHFlow | 50.38 | 57.06 | 48.88 | −8.17 [−9.19, −7.21] | .13 |
| Hard2Verify Qwen-7B | 45.82 | 44.27 | 45.49 | +1.22 [−0.97, +3.72] | .02 |
| Hard2Verify Qwen-72B | 35.52 | 43.37 | 43.37 | 0.00 | .00 |
| Hard2Verify PURE | 50.76 | 39.96 | 39.96 | −0.01 [−1.15, +1.06] | .01 |
| Hard2Verify RLHFlow | 5.11 | 33.06 | 25.04 | −8.02 [−12.51, −3.37] | .16 |

**Pre-registered verdict: not met.**
- The rule asked for at least 4 gains and no losses. The result is 6 gains, 3 losses and 3 ties.
- **The gains:** every non-RLHFlow cell on PRMBench and Socratic.
- **The losses:** every RLHFlow cell. There the PRM is weak, so its next-step reading is weak too. It also correlates
  more with the PRM itself (r .42–.72) than in the other cells (.05–.49).
- **Socratic strong PRMs:** the view narrows the gap to the PRM by +0.24 to +1.29, but does not close it. CHAIN +
  next is still −2.5, −14.1 and −4.3 below the PRM.

**Reading (post hoc, not pre-registered).** The next-step view is a small, consistent addition when the PRM is good.
With a weak PRM it adds that PRM's error a second time.

## Limits

- All three benchmarks were inspected before, and SUB5 was selected on PRMBench and Socratic labels.
- Every estimator here assumes conditional independence between groups given y, which stage 0 and Step 481 showed is
  violated. This run confirms that the estimator choice does not remove the strong-PRM Socratic gap.
