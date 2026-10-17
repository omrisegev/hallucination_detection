# PRM boosting generalization v1: results (2026-10-14)

This is a frozen-recipe transfer to 4 more PRMs on PRMBench (6,969 answers). Nothing was re-selected or tuned:
`S3 = ½·az(PRM risk) + ½·mean az(energy_level, hist_entropy_cusum_abs_series, realized_drv)`.
The 3 channels come from each PRM's own backbone, in the same forward as its reward. Code: Codex branch
`codex/prm-boosting-generalization-v1` (fba8e8d9). AIRCC jobs: 283804 (T1), 2838xx (pilots / full runs / banks),
283943, 283948 and 283953 (evaluation). Raw telemetry and banks: AIRCC `results/prm_boost_gen_v1/`.

## Gates
- **T1 (Qwen PRM-7B regression).** Bit-identical to v1: every gap is 0.0 (`t1_qwen_prm/manifest.json`).
- **G1.** Every PRM reproduces its leaderboard Overall under its own native rule:
  72B 68.18 (68.2), PURE 65.31 (65.3), RLHFlow 54.43 (54.4), MathShepherd 47.02 (47.0).
- **G2.** Median token surprisal is below 1 nat for every model (72B .62, PURE .09, RLHFlow .20, MathShepherd .57).
  RLHFlow logged 211 step/text mismatches. All 211 decode exactly once `clean_up_tokenization_spaces` is turned off
  (`G2_DECODE_DIAGNOSTIC_rlhflow_prm.json`), so they are decode artifacts and the alignment is fine. Amendment A1
  (`evaluate_amended.py`) records this.
- **G3.** The Qwen3 bank rebuild reproduces the frozen bank exactly (max abs error 0.0).
- **Amendment A2 (MathShepherd).** The N16 fit fails with a constant column. The digit channels hard-code the
  Qwen/Llama-3 digit token IDs 15-24, and those IDs are not digits in MathShepherd's Mistral/Llama-2 vocabulary
  (`digit_active` .9998). The N16_own and EQ2_own arms are withheld. S3 contains no digit channels.

## Main table
Official PRMScore and within-AUC. "native" is each PRM's own rule; every other arm uses ANSWER_Z_Q80.

| PRM | PRM native | PRM Q80 | N16 own | ½PRM+½N16 | **S3 (frozen)** | S3 − native [95% CI] |
|---|---|---|---|---|---|---|
| Qwen2.5-Math-PRM-7B (selection model) | 65.46 / .801 | 65.82 | 67.20 / .813 | 68.53 / .840 | **69.38 / .851** | +3.93 [3.24, 4.60] |
| Qwen2.5-Math-PRM-72B | 68.18 / .795 | 66.24 | 67.50 / .822 | 69.39 / .849 | **69.39 / .854** | +1.21 [0.62, 1.76] |
| PURE-PRM-7B | 65.31 / .763 | 63.27 | 65.94 / .784 | 67.06 / .816 | **66.77 / .817** | +1.46 [0.85, 2.13] |
| RLHFlow Llama3.1-8B PRM | 54.43 / .662 | 56.46 | **63.67 / .755** | 61.80 / .749 | 61.00 / .738 | +6.57 [5.95, 7.21] |
| MathShepherd-Mistral-7B | 47.02 / .684 | 57.71 | (A2) | (A2) | **63.15 / .770** | +16.13 [14.72, 17.48] |

The paired source-group bootstrap (707 groups, 2,000 draws) also gives S3 − PRM Q80:
- 72B: +3.15
- PURE: +3.50
- RLHFlow: +4.54
- MathShepherd: +5.44

Every interval excludes 0. The CSVs in `eval/` and `eval_amended/` give per-classification PRMScore,
plus the S3_q3 and S6 arms.

## Reading
- **The 3-channel recipe transfers.** Frozen as selected on PRM-7B, S3 beats every PRM's own native PRMScore,
  and its within-AUC gain is +.054 to +.086.
- **The gain grows as the PRM gets weaker.** It is about +1.2 to +1.5 for the two strong PRMs and +6.6 / +16.1 for
  the two weak ones.
- **On RLHFlow, our label-free N16 beats S3.** N16 alone, built from RLHFlow's backbone, scores 63.67, ahead of
  S3's 61.00 (S3 − N16: −2.67 [−3.25, −2.13]). Giving the PRM half the weight helps strong PRMs and hurts when
  the PRM is weak.
- **Caveats.**
  - The channels were label-selected on Qwen-PRM-7B.
  - Each PRM is label-trained, so the fused score inherits that supervision. Only the fusion step is label-free.
  - PRMBench is previously inspected development data, not an untouched test.
  - Skywork (n+1 rewards) and ReasonEval (no LM head) were not run.
