# PRM boosting CPU controls (2026-10-18)

Omri approved the CPU controls (axis C of `docs/post_advisor_2026-10-17/05_PRM_BOOSTING_EVIDENCE_PLAN_HE.md`).
AIRCC CPU job 287777, script `lsml_controls.py`, run from `eval_mirror_prm_gen_v1`.
Inputs, orientation, the ANSWER_Z_Q80 flag rule and the paired source-group bootstrap are the same as in
`../prm_boosting_generalization_v1/run_v1/eval_lsml4/lsml_four.py`. Parity: PRM_native and LSML_CONT
replay those results exactly for all 5 PRMs. All fusion is label-free; labels are used only for scoring.
This is development data on PRMBench, and nothing here is external evidence.

## PRMScore (official, 6,211 noncontrols)

| arm | Qwen-7B | Qwen-72B | PURE | RLHFlow | MathShepherd |
|---|---|---|---|---|---|
| PRM native | 65.49 | 68.18 | 65.31 | 54.43 | 47.02 |
| LSML_CONT (3 label-chosen channels) | 69.48 | 69.84 | 66.69 | 63.53 | 63.68 |
| **HALF_N16** (½ PRM + ½ mean of N16; no channel choice) | 68.12 | 68.81 | 66.49 | 60.46 | 62.52 |
| **LSML_N16_PRMG** (L-SML, PRM its own group + N16) | 67.80 | 68.83 | 66.88 | 64.34 | 60.58 |
| LSML_N16_AUTO (automatic groups) | 67.49 | 67.83 | 66.67 | 63.95 | 61.70 |
| LSML_RESID (channels minus position and log length) | 68.72 | 67.90 | 65.65 | 63.36 | 63.37 |
| PERM_ANS (channels moved to another answer, same step count) | 53.45 | 53.42 | 52.26 | 52.69 | 51.50 |
| NATIVECOUNT (fused order, PRM-native number of flags) | 67.11 | 69.49 | 66.06 | 59.28 | 47.00 |

Ablations (`DROP_*`, `ONLY_*`) and per-category scores are in `CONTROLS_RESULTS_*.csv`. Every contrast
with its 95% CI, against PRM_native and against LSML_CONT, is in `CONTROLS_BOOTSTRAP_*.csv`.

## What it shows

1. **The boosting does not depend on the label-chosen channels.**
   - HALF_N16 beats the native PRM on both metrics in all 5 PRMs, and every CI excludes 0:
     - PRMScore: +2.63, +0.63, +1.17, +6.03, +15.51
     - within-AUC: +0.036 to +0.080
   - LSML_N16_PRMG also beats native in all 5: +2.31, +0.65, +1.57, +9.91, +13.57.
   - The label-chosen 3-channel recipe is higher by about 0.2–3 points; on RLHFlow LSML_N16_PRMG is higher than it (+0.81).
2. **Step alignment carries the information.** Moving the channels to another answer with the same number
   of steps keeps their position profile but destroys the gain. The fused score then falls far below
   the PRM alone (within-AUC about 0.53–0.60).
3. **Most of the gain is not position or step length.**
   - After residualizing the channels on position and log step length, within-AUC still beats native in all 5.
   - PRMScore beats native in 3 of 5. The CI includes 0 for Qwen-72B and for PURE.
4. **realized_drv carries most of the weight.**
   - Dropping it costs 2.0–4.6 PRMScore.
   - hist_entropy_cusum_abs_series contributes little: dropping it costs 0 to 1.4.
5. **The PRM's native flag count does not help.** Using it with the fused order is below Q80 in all 5,
   strongly so for the weak PRMs. That idea for the per-answer count is negative.

Caveats:
- N16 was itself chosen as the highlight bank after seeing Qwen3-8B results (not PRM results).
- HALF_N16 is a fixed ½ weight. The L-SML version is LSML_N16_PRMG.
