# Pre-registration: risk-controlled clean-answer threshold, calibrated track (2026-10-25)

ID `prm-boost-calibrated-clean-v1`. Design: `CALIBRATED_CLEAN_THRESHOLD_DESIGN.md`. Machine-readable:
`results/prm_boost_math_rigor_v1/CALIBRATED_CLEAN_PREREG.json`. Code: `stage4_calibrated_clean.py` and
`spectral_utils/risk_control.py` (12 unit tests pass, including a coverage check: violation rate ≤ δ over 400
synthetic calibration draws).

**This file, the JSON and the code are committed before the first run.**

**Access level: calibrated.** Labels are used on the calibration split to choose τ. The results are reported apart
from the label-free method.

## Rule
- **Score.** S = Σ_steps (1 − reward), the PRM's expected error count per answer. Called clean iff S < τ.
- **Choosing τ.** Fixed-sequence Learn-then-Test with Hoeffding–Bentkus p-values (`spectral_utils/risk_control.py`).
  - Grid: 200 quantiles of S on the calibration answers, tested from the smallest τ up.
  - Risk: the false-clean rate, i.e. the share of erroneous calibration answers with S < τ.
  - Settings: α = 0.05 (primary) and 0.02, δ = 0.10.
  - If no τ is certified, nothing is called clean, and the rule reduces to GAP.
- **Flags.** Answers called clean get no flags. Every other answer gets GAP on CHAIN_N16, unchanged.

## Data and splits
- **PRMBench, all 6,969 answers including controls.**
  - Calibration/test: 50/50 by source group, stratified by mean answer length (adjacent pairs of groups).
  - 10 seeds, from 20261025 upward.
- **Hard2Verify, 177-answer common set.** Transfer test with the seed-0 PRMBench τ. There is no guarantee there,
  because exchangeability does not hold.
- **5 PRMs.**

## Endpoints, PRMBench test split
1. **Guarantee.** Realized test false-clean rate ≤ α in ≥ 9 of 10 seeds.
2. **Usefulness.** Pooled step macro-F1 ×100 on all test answers (incl. controls), two-step minus GAP.
   Seed 0, paired bootstrap over test source groups, 2,000 draws.
3. **Official-pool non-inferiority.** PRMScore-formula difference on the non-control test answers, seed 0,
   ≥ −0.5.

Also reported: clean-recall, mean deltas over the 10 seeds, the number of seeds with no τ certified, and the
Hard2Verify transfer (realized false-clean rate, and the official delta with a CI over answers).

## Verdict (α = 0.05)
- **Pass:** endpoint 1 holds for ≥ 4/5 PRMs, endpoint 2's CI is above 0 for ≥ 3/5, and endpoint 3 holds for
  ≥ 4/5.
- **Partial:** endpoint 1 holds for ≥ 4/5, plus one of endpoints 2 and 3.
- **Fail:** anything else.

A fail or a partial is recorded with `/negative-result`. Nothing is tuned after the results.

## Expectations, written down in advance
- From the design's break-even: calling clean pays only if the share of clean answers recalled is about ≥ 0.3 at
  α = 0.05.
- The label-free version already showed that the native threshold calls 23–39% of erroneous answers clean.
- A certified τ may therefore recall few clean answers, which gives a valid but useless rule. That outcome is
  acceptable and will be reported as is.
