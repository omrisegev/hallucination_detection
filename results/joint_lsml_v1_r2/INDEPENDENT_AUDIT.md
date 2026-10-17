# Independent audit — Joint L-SML v1 R2

Verdict: **PASS for bounded label-free structural claims; no blocker.**

The independent auditor reconstructed the result without using benchmark
outcomes:

- All 18 Task-1 donor-score rankings were reconstructed from the sanitized NPZ
  inputs and frozen C-v2 ledger. All 18 pass the 0.99 gate; the range is
  0.9933622788–0.9999503673 and maximum ledger error is 3.33e-16.
- All 16 fitted-lane covariances were reconstructed with maximum absolute error
  2.26e-13. All 96 saved weight-map Spearman values match within 2.22e-16.
- Joint model matrices, objectives and misfits match within 2e-15; hard-L-SML
  misfits match within 4.44e-16.
- The 16 fitted and two blocked lanes reproduce exactly. Every fitted lane has
  lower off-diagonal misfit, with absolute improvement 0.02649–0.08338. All five
  starts converge in every fitted lane, objective traces are monotone,
  multistart checks pass, and the profiled Jacobian has full rank.
- Raw orientation reproduces with no sign mismatch and maximum loading error
  1.28e-14. The global roster is exactly 23/28: three weak streams, two
  sign-unstable streams, and five degree-rejected streams, with exact fallback
  signs and removal reasons.
- All registered source/input hashes, result artifact hashes and payload hashes
  pass. The Agent-A payloads satisfy its exact schemas. Sanitized NPZ members are
  only `raw`, `row_ids`, and `token_offsets`; no outcome or fused score array was
  found.

Claim limitations:

- The 16/16 misfit comparison is conditional on partition admissibility. Both
  blocked lanes are `v2_active28` lanes.
- The minimum four-map donor-score agreement is only 0.708240; structural fit
  does not establish localization efficacy.
- Diagonal residual clipping occurs in 14/16 fitted lanes, at no more than two
  coordinates per lane and maximum clipped mass 0.05452.
- R2 is a post-failure engineering continuation. The frozen protocol records
  that R1 failed closed before result materialization; R2 changed only blocked
  lane handling, not thresholds or estimator logic.
