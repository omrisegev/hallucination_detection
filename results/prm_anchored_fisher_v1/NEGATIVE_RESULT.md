# Negative result: PRM-anchored Fisher fusion (P5) is not identified on the existing views

**Hypothesis (one sentence):** weighting CHAIN_N16's cross-stage views by anchored Fisher weights, without labels,
beats CHAIN_N16 and does not fall below a strong PRM on Socratic.

**Date / step:** 2026-10-28, Step 483. The pre-registration was committed before the run: `7203cb67`,
`docs/experiments/PRM_ANCHORED_FISHER_PREREG_2026-10-28.md`.

**Population:** PRMBench (6,969 answers), Socratic-PRMBench (2,995) and Hard2Verify (177 answers scored) × 5 PRMs.
Fitting is per cell and transductive, like CHAIN.

## Arms compared
| Arm | What it is | Result (12 line cells) |
|---|---|---|
| reference | CHAIN_N16 (shipped) | — |
| simple control | the PRM alone (same rule) | — |
| candidate | ANCH_AUTO (rank route, else cumulant; fallback to the PRM view) | identified 3/12; 4 gains, 7 losses against CHAIN |
| variants | ANCH_RANK, ANCH_CUM, ANCH_CUMW, ANCH_CCA, ANCH_AUTO_G (gate), ANCH_AUTO_FBC (fallback to CHAIN) | the same identification failure |
| label-using | ORACLE_FISHER (LDA, in-sample) | +0.7 to +1.1 over CHAIN on PRMBench; below the PRM on 2/3 strong-PRM Socratic cells under GAP |

**Win/loss against CHAIN_N16:** 4 gains, 7 losses, 1 tie. The 4 gains are fallbacks to the PRM view, not
identified fusions.
- Effect on PRMBench, where it falls back in 4/4 line cells: −4.0 to −6.9 PRMScore, every CI below 0.
- Against the PRM, on the 3 strong-PRM Socratic cells: −0.6 [−1.06, −0.16], −3.6 [−4.01, −3.05] and
  −0.6 [−0.98, −0.23].

## Three plausible reasons it failed
1. **The error factor is too small against the nuisance in the telemetry block.** The rank route returns a flat
   profile. Step 469: 82–100% of the off-diagonal covariance is within-class structure.
2. **The third-order cumulants do not match the model.** The cumulant route puts u₀ too low, and the implied
   telemetry loadings exceed the view variance. Step 469: the signs fit in only 21/44 views.
3. **The anchor's noise is not independent of the telemetry's.** Within-class |corr| is .13–.35, because both come
   from the same backbone. That violates P5's assumption directly.

## Verdict
- [ ] closes the DIRECTION
- [x] closes this IMPLEMENTATION: anchored identification on CHAIN_N16's own-backbone telemetry views

**What stays open:** the same algorithm with views whose noise is not shared with the PRM. For example, the
question-blind PRM difference and leave-one-step-out influence. The module is reusable for that as is.
