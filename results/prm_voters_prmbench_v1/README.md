# PRM voters on PRMBench (2026-10-18): no-go

**Voters:**
- 5 PRMs: Qwen2.5-Math-PRM-7B and -72B, PURE, RLHFlow, MathShepherd.
- 2 uPRM judges: Qwen3-8B and Qwen2.5-14B-Instruct.
- The Qwen2.5-72B-Instruct critic.

**Fusion:** each voter is a label-free view. Fusion uses continuous L-SML, and per-answer error counts use MoM.

**Runs:**
- Script: `scripts/run_prm_voters_prmbench.py`, at this branch's commit.
- Inputs: AIRCC jobs 287340-287361 for the judges and the critic, and `prm_boost_gen_v1` for the PRMs.
- Analysis: CPU job 287942. Run script `run_voters.sh`, log `prm_voters_analysis_287942.out`, numbers `*/RESULTS.json`.

**Go/no-go rule, fixed before the run:** the fusion must beat the best single PRM, and must also beat PRM boosting. It does neither, so the direction closes.

## Results (official PRMScore, 6,211 noncontrols)

| run | rows | L-SML (Q80) | best single (Q80) | L-SML minus best [95% CI] | MoM count MAE |
|---|---|---|---|---|---|
| 5 PRMs only | 6,969 | 65.80 | PRM-72B 68.68 | -2.88 [-3.28, -2.48] | 1.39 (PRM-72B 1.32) |
| 5 PRMs, MOM_COUNT rule | 6,969 | 67.13 | PRM-72B 67.41 | -0.28 [-0.54, -0.01] | - |
| all 8 voters | 4,648 (see below) | 63.49 | PRM-72B 66.41 | -2.92 [-3.46, -2.40] | MoM fit failed |

For comparison, PRM-72B boosted with its own backbone telemetry reaches 69.84 (`results/prm_boosting_generalization_v1`), and the clean N16 recipe reaches 68.81 (`results/prm_boosting_controls_v1`).

**Counts:** the true mean error count is 2.14 per answer. MoM estimates 1.42, so it underestimates, as it did on ProcessBench. The critic's cumulative flags overestimate (4.26).

## Data issue found and fixed

The step-voters driver keeps a judge's per-step view only if its candidate scores reproduce S(T+1) within a relative tolerance of 1e-4. bf16 scores miss this on 6,593 of 6,969 rows (Qwen3-8B) and 6,509 of 6,969 (Qwen2.5-14B). The pilots also missed it, on 33/35 and 35/35 rows. That contradicts the pilot README, which says there were no recovery errors.

The candidate scores are saved, so the analysis now recovers the per-step view from them and keeps rows within 0.02 nats (`--recovery-tol`, unit test `tests/test_prm_voters_recovery.py`). On Qwen2.5-14B the median gap is 0.004, but the 90th percentile is 8 nats, so 2,238 (14B) and 399 (8B) rows stay unusable. That is why the all-voters run covers only 4,648 rows.

The PRM-only verdict does not depend on this, because it joins all 6,969 rows.
