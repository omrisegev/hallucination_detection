# PRM-anchored view selection, stage 0 (descriptive, CPU, 2026-10-25)

Pre-registration: `docs/experiments/PRM_VIEW_SELECTION_STAGE0_PREREG_2026-10-25.md`, committed before the run
(104a3bfc), with amendments 1 and 2. Code: `stage0.py`; helpers `spectral_utils/binary_view_estimation.py` (tests
5/5). Runtime about 90 s per variant.
- **Views.** 17: the PRM plus N16 backbone telemetry, answer-z, oriented by the PRM.
- **Cells.** 2 benchmarks × 5 PRMs.
- **Pool.** PRMBench: 83,370 steps per PRM. Socratic: 26,048.
- **Parity.** The PRM view's GAP flags reproduce the frozen STATUS_MAIN "PRM same rule" PRMScore in 10/10 cells.
- **Independent check.** An agent with its own code reproduced the checked values (signs, ψ/η/balanced accuracy,
  PRMScore, within-AUC, two-factor u/λ, misfit blocks) exactly.

## Predictions

| | Prediction | Verdict |
|---|---|---|
| P1 | On PRMBench, label-free balanced-accuracy estimates rank views like the truth (Spearman ≥ .7 in ≥ 4/5 PRMs) | **Holds.** DS ρ = .91, .96, .75, .86, .61. **But on Socratic it fails:** ρ = .87, .76, .72, **−.79, −.52** |
| P2 | Telemetry accuracy drops more from PRMBench to Socratic than the PRM's | **Holds, 5/5** (table below). The I-ratio clause is void (amendment 2) |
| P3 | Rank-one misfit sits in telemetry–telemetry pairs, not PRM–telemetry | **Holds on PRMBench, 5/5. Reverses on Socratic** for 72B, PURE, RLHFlow and Math-Shepherd (PRM–telemetry misfit .38–.88 vs .18–.44) |
| P4 | Nuisance factor carries no label signal given the error factor (\|partial r\| < .05) | **Mixed.** 5/10 cells; fails for the weak PRMs and for PURE on Socratic |

P2, within-answer AUC of the continuous view. "Telemetry" is the median over the 16 channels.

| PRM | PRM: PRMBench → Socratic | Telemetry median: PRMBench → Socratic |
|---|---|---|
| Qwen-7B | .801 → .824 | .724 → .673 |
| Qwen-72B | .795 → .814 | .703 → .637 |
| PURE | .763 → .796 | .694 → .628 |
| RLHFlow | .662 → .549 | .678 → .537 |
| Math-Shepherd | .684 → .718 | .631 → .498 |

On Socratic, 1–6 channels per PRM are below chance in their label-free orientation.

## What this says

1. **The PRM gets better on Socratic; the telemetry gets worse,** in 4/5 PRMs (RLHFlow: both get worse). This is
   consistent with the PRMBench telemetry signal being partly benchmark-specific, but does not prove the
   perturbation-artifact hypothesis.
2. **Label-free CI estimators over-trust telemetry.**
   - Estimated minus true balanced accuracy averages +.04 to +.07 for telemetry on PRMBench (up to +.15 on
     Socratic), against −.09 to +.06 for the PRM.
   - Correlated telemetry channels inflate the rank-one fit.
3. **The failure mechanism on Socratic.** The latent class locks onto the telemetry consensus and the PRM becomes the
   outlier (PRM–telemetry misfit > telemetry–telemetry). For RLHFlow and Math-Shepherd, the ranking of views by the
   label-free estimates is anti-correlated with the truth.
4. **Descriptive fused scores** (`FUSED_DESCRIPTIVE.csv`; not candidates):
   - binary DS / spectral fusion of the 17 GAP votes is above the PRM on PRMBench (+0.6 to +5.6), but below CHAIN;
   - on Socratic it is below the PRM in 5/5;
   - the anchored two-factor error score is the only one above the PRM on Socratic, and only for Qwen-7B
     (70.87 vs 70.10).
5. **Implication for stage 1.** A label-free criterion fitted on the target benchmark cannot by itself be trusted to
   keep harmful channels out: on Socratic it ranks them wrongly. Selection must be anchored more strongly on the PRM,
   e.g. accept a view only if it agrees with the PRM beyond the telemetry consensus, and it must be tested on a
   benchmark not used for design.

## Files

- `BINARY_VIEWS.csv`: per view. Label-free estimates, then `VAL_*` columns.
- `CELL_SUMMARY.csv`, `FUSED_DESCRIPTIVE.csv`, `RUN_META.json`, `run*.log`.
- `sensitivity_orient-az_pool-keep/` and `sensitivity_orient-az_pool-all/`: amendment 2.
