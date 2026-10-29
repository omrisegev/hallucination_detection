# ds_gate_e1b_fuse: corrected-E1 gate on FUSE global-threshold votes (E1b)

**Result: FAILED all pass criteria in every arm (stopped as frozen, no tuning).** The absolute votes fix the
grouping problem only on SUB5; with 17 or 16 voters the S-hat search drives most thresholds to the 0.98 grid edge
(votes on ~2% of steps) and the vote covariance then splits into K = 2 groups, which the protocol counts as invalid.

Protocol: PROTOCOL.json (written before results); self-review and unit checks: REVIEW.md, test_shat.py (PASS).
Code: votes_e1b.py (entry point `fuse_votes(bench, prm, arm)`, cached votes in cache/ for E2), run_e1b.py.
Outputs: E1B_RESULTS_<arm>.csv, E1B_FITS_<arm>.json (chosen quantiles, vote rates, groups), E1B_VERDICT_<arm>.json,
run_auto.log, run_sa.log.

## Validity

| arm | valid cells | invalid (K = 2) | missing raw |
|---|---|---|---|
| AUTO (PRM tuned + N16 backbone) | 1 / 16 (CRV-GSM8K Qwen-7B) | 11 | 4 (PRMBench: PRM-backbone raw only on AIRCC) |
| AUTO_NATIVE (PRM native flag) | 2 / 16 | 10 | 4 |
| TELE16 standalone | 6 / 16 (all CRV) | 10 (all PRMBench, all ProcessBench) | 0 |
| SUB5 standalone | 16 / 16 | 0 | 0 |

S-hat falls sharply under the search (e.g. ProcessBench Qwen-7B AUTO 48.7 -> 0.12; SUB5 GAP-vote S-hat 0.01-0.78
vs FUSE-vote S-hat 0.000-0.009), but the minimizer favours extreme, sparse votes: in AUTO 9-12 of 17 voters end at
the 0.98 quantile.

## SUB5 (all 16 cells valid), mean over the 4 PRMs

| benchmark | est. share with error | true share | ECE | clean closed @rho | erroneous kept @rho | within-AUC q | within-AUC fused (ref) | GAP | gated @rho | gated cost | gated 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PRMBench (PRMScore) | 0.847 | 0.866 | 0.165 | 0.108 | 0.786 | 0.606 | 0.763 | 66.33 | 64.10 | 64.23 | 65.05 |
| ProcessBench (F1) | 0.743 | 0.653 | 0.376 | 0.293 | 0.712 | 0.583 | 0.715 | 0.0 | 24.14 | 22.53 | 22.26 |
| CRV-Arith (F1) | 0.872 | 0.083 | 0.789 | 0.272 | 0.888 | 0.631 | 0.519 | 45.86 | 48.51 | 47.99 | 46.99 |
| CRV-GSM8K (F1) | 0.790 | 0.210 | 0.618 | 0.280 | 0.816 | 0.530 | 0.563 | 47.63 | 50.61 | 50.36 | 49.88 |

- Prevalence: closer than E1 on PRMBench (0.80-0.90 vs 0.866) and ProcessBench (0.66-0.82 vs 0.653; PURE 0.657),
  still far off on CRV (0.73-0.94 vs 0.08 / 0.21). D1 fails (needs ProcessBench and both CRV within 0.05 for 3/4 PRMs).
- Gate: on PRMBench it closes 16-28% of erroneous answers and PRMScore drops 1-4 points vs GAP. On ProcessBench it
  closes 29% of clean answers on average but also 29% of erroneous ones (D4 needs >= 30% clean closed with >= 85%
  erroneous kept: only RLHFlow 0.41 / 0.86 passes, 1 of 4).
- Gated F1 beats plain GAP on CRV (+2 to +5) and ProcessBench (GAP always flags, so 0 there; gated 14-35), as any
  gate that closes some answers would; the step posterior q ranks steps worse than the arm's fused L-SML score on
  PRMBench and ProcessBench (within-AUC 0.61 vs 0.76, 0.58 vs 0.72).

## AUTO / TELE16 valid cells (few; details in the CSVs)

AUTO CRV-GSM8K Qwen-7B: est. 0.66 vs true 0.21, ECE 0.48, F1 GAP 49.99 -> gated 55.8. AUTO_NATIVE CRV-Arith Qwen-7B:
est. 0.77 vs 0.083. TELE16 CRV cells: est. 0.70-0.89 vs 0.08 / 0.21. Prevalence remains badly overestimated.
