# Pre-registration: AUTO vs CHAIN with GAP, and CHAIN under forced K (W7 part 2, W8), 2026-10-25

ID `prm-boost-auto-forcedk-v1`. JSON: `results/prm_boost_math_rigor_v1/AUTO_FORCEDK_PREREG.json`. Code:
`stage8_auto_forced_k.py`. **This file, the JSON and the code are committed before the first run.** All fitting is
label-free. Labels only score.

## Why
- **W7 part 2.** CHAIN puts the PRM in a cross-stage view of its own (assumption A7). AUTO runs the same algorithm on
  the PRM and the telemetry together and absorbs the PRM into a telemetry group in 15/20 bank-sweep cells. The only
  existing AUTO numbers use Q80 and an older bank code path. The question: does CHAIN's design cost anything against
  AUTO under GAP?
- **W8.** P3 predicts that the PRM's cross-stage share is about 1/(K+1), so it is set by K and not by quality. The
  review asked how sensitive the share and the PRMScore are to K.

## Arms
- **CHAIN_N16.** Exactly as exported. Parity is asserted against the exported scores before anything else.
- **AUTO_N16.** `lsml_continuous_guarded` on [z(PRM), N16 channels] with free grouping, oriented by the PRM. This is
  `prm_boosting_bank_sweep_v1`'s definition.
- **CHAIN_N16 with forced K.** The telemetry partition uses `K_range=[K]` for K = 2..8. Everything else is unchanged.
- **Readout.** Every arm uses GAP flags. Metrics: within-AUC, and GAP PRMScore (official pool).
- **PRM share.** Two measures:
  - the cross-stage share |w_PRM| / Σ|w| (CHAIN only);
  - the least-squares share of z(PRM) in the fused score (`prm_weight_share`, every arm).

## Decision rules (PRMBench, 5 PRMs)

**A, AUTO vs CHAIN.** A paired bootstrap of AUTO − CHAIN, 2000 draws, using the collection's source-group weights.
- **AUTO preferred:** the GAP-PRMScore CI is above 0 in ≥ 3/5 PRMs.
- **CHAIN preferred:** the within-AUC CI is below 0 in ≥ 3/5 PRMs, and in no PRM is AUTO significantly better on both
  metrics.
- **Otherwise:** inconclusive.

**B1, the P3 prediction.** The Spearman correlation between K and the cross-stage PRM share is negative in ≥ 4/5
PRMs.

**B2, K robustness.** The automatic-K CHAIN's GAP PRMScore is at least the best forced-K PRMScore − 0.5, in ≥ 4/5
PRMs.

## Reported without a criterion
- Socratic and Hard2Verify, with the same rules applied.
- Share × (K+1) for each K.
- Whether AUTO leaves the PRM alone.
- The PRMScore range over K.

Nothing is tuned after the results. If CHAIN is not preferred, or B2 fails, the outcome is recorded with
`/negative-result`.
