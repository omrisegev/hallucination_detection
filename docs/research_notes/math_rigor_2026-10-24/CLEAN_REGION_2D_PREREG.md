# Pre-registration: 2-D clean region in (S, number of steps), calibrated track (2026-10-25)

ID `prm-boost-clean-region-2d-v1`. JSON: `results/prm_boost_math_rigor_v1/CLEAN_REGION_2D_PREREG.json`. Code:
`stage6_clean_region_2d.py`. **This file, the JSON and the code are committed before the first run.**

## Why

The 1-D threshold on S = Σ(1 − r) kept its guarantee, but its verdict was partial (Step 471). It recalled few
clean answers outside the two Qwen PRMs, and it did not transfer to Hard2Verify. The reason is that S grows with
answer length. This test checks whether letting the boundary depend on length fixes that, before trying a learned
region (CoCoNet).

## Variants (the access level is calibrated; labels are used on the calibration split only)

- **Primary, `mondrian_len3`.**
  - Strata: terciles of answer length (number of steps) on the calibration answers.
  - One τ per tercile, each certified by the Step 471 procedure on that tercile's erroneous answers: Learn-then-Test
    with Hoeffding–Bentkus p-values, α = 0.05, δ = 0.10.
  - Rule: an answer is called clean iff S < τ of its tercile.
  - This is a piecewise-constant 2-D boundary. The false-clean rate in each stratum is controlled, so the mixture
    is controlled too.
- **Secondary, `mean_p`.** One τ on S / n, the mean error probability per step.

**Inherited unchanged from `CALIBRATED_CLEAN_PREREG.md`:**
- the PRMBench 50/50 source-group splits with 10 seeds;
- the flags (no flags for answers called clean, GAP on CHAIN_N16 otherwise);
- endpoints 1–3, their thresholds, and the verdict rule.

## Reported without a criterion
- Hard2Verify transfer with the seed-0 rule. The strata cut points come from PRMBench.
- Comparison with the 1-D rule of Step 471.

A fail or a partial is recorded with `/negative-result`. Nothing is tuned after the results.
