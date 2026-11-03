# CAT34 vs Z17 on Socratic-PRMBench, 4 PRMs (2026-11-03)

The same experiment as `../README.md`, run on the external Socratic-PRMBench set (2,995 answers, 26,052 non-empty
steps): local CPU, about 16 s per PRM. Script: `auto_raw_plus_z_socratic.py`; tables: `make_tables.py` ->
`TABLES.md`. Socratic is external data, but it was already inspected post hoc in earlier work
(`../../sml_weight_estimator_v1/`), so this is not an untouched test.

**Bottom line.** The pre-declared rule is NOT met on Socratic.
- CAT34 beats Z17 on 2 of 4 PRMs: Qwen-7B +1.10, PURE +0.86 PRMScore.
- It loses on the other 2: Qwen-72B -1.47, RLHFlow -0.49. All four CIs exclude 0.
- Within-AUC rises on 3 PRMs (+0.007 to +0.032) and falls on Qwen-72B (-0.017).
- Both fused arms stay below the PRM alone on Qwen-7B and Qwen-72B.

## Same method as PRMBench

- **Inputs:** each PRM's own backbone telemetry on Socratic. The files are bank_v2 `FEATURES.npz` + `REWARDS.npz`
  (AIRCC jobs such as 288317; local copy `cache/prm_boost_external_v1/full_<prm>_socratic/bank_v2/`).
  The same 17 channels: PRM + N16.
- **Z17:** the stored Socratic AUTO_N16 code path (`../../sml_weight_estimator_v1/socratic_sign/prm_placement.py`,
  mode `signed`, i.e. the shipped SML). It uses `lsml_continuous_guarded`, which drops constant views before
  the fit. No view was constant here, so it is exactly `lsml_continuous`.
- **CAT34:** Z17 plus the same 17 channels after a global rank -> normal transform. The transform pools all
  non-empty Socratic steps of that PRM. Digit channels are ranked over their active steps and set to 0
  elsewhere; the raw PRM view is g(logit(1-r)). The same orientation anchor, z(PRM). Nothing is tuned.
- **Metric:** the official Socratic PRMScore, (F1_correct + F1_error)/2 over included steps (shown x100), as in
  `../../prm_boost_ablation5_v1/eval_socratic_ablation5.py`.
  - Decision rule: Q80 on the answer-z score over non-empty steps; empty steps are predicted incorrect.
  - Within-AUC is computed on included non-empty steps.
  - The bootstrap reweights answers (multinomial, 1,000 draws). There is no source-group bootstrap: Socratic
    has no source groups in these files.
- **References:** PRM_native (the PRM's own threshold), PRMPROB_Q80 (z(1-r) under Q80, i.e. PRM_Q80_PROB), and
  POS_ONLY_Q80 (relative step position under Q80).

## Parity: PASS

Z17 equals the stored `AUTO_N16_signed_Q80` row of `RESULTS_socratic_placement.csv` exactly on all 4 PRMs
(PRMScore and within-AUC). The PRM, PRMPROB and POS_ONLY rows also equal
`prm_boost_ablation5_v1/socratic/RESULTS_socratic.csv`.

| PRM | Z17 (stored AUTO_N16) PRMScore | within-AUC |
|---|---|---|
| Qwen-7B | 64.28 | 0.7220 |
| Qwen-72B | 59.57 | 0.6612 |
| PURE | 63.39 | 0.6600 |
| RLHFlow | 57.56 | 0.5953 |

## Results

| PRM | Z17 | CAT34 | CAT34 - Z17 (95% CI) | within-AUC Z17 -> CAT34 | PRM_native | PRM_Q80_PROB | position only (PRMScore / within-AUC) |
|---|---|---|---|---|---|---|---|
| Qwen-7B | 64.28 | 65.37 | +1.10 [+0.82, +1.37] | 0.722 -> 0.754 | 68.09 | 67.69 | 58.01 / 0.731 |
| Qwen-72B | 59.57 | 58.10 | -1.47 [-1.73, -1.18] | 0.661 -> 0.644 | 71.96 | 67.35 | 58.01 / 0.731 |
| PURE | 63.39 | 64.26 | +0.86 [+0.61, +1.12] | 0.660 -> 0.680 | 63.71 | 63.49 | 58.01 / 0.731 |
| RLHFlow | 57.56 | 57.07 | -0.49 [-0.82, -0.17] | 0.595 -> 0.602 | 48.12 | 49.29 | 58.01 / 0.731 |

**Position.** Relative step position alone has within-AUC 0.731. That beats both Z17 and CAT34 on Qwen-72B,
PURE and RLHFlow, and is about equal to them on Qwen-7B. Under Q80 its PRMScore is low (58.0).

**Answer-level AUC is not reported.** Only 6 of the 2,995 Socratic answers have no error step, so an AUC that
separates answers with and without an error rests on 6 negatives and is meaningless. The column is in
`RESULTS.csv` only for completeness.

## Diagnostics

- **|corr(raw_i, z_i)|:** 0.63-0.98, the same on clean and error steps (Table 4). This is the same pattern as
  on PRMBench.
- **Grouping:** raw_i lands in the same CAT34 group as z_i for 17, 17, 16 and 17 of 17 channels. The only split
  is `chosen_surprisal` on PURE. The PRM pair (z and raw) is a group of its own in all 4 PRMs.
- **Leading eigenvalue share:** 0.36-0.39 for Z17, 0.34-0.36 for CAT34.

## Reading

On Socratic the answer-level signal that the raw copy adds back does not help consistently. On PRMBench the
same addition helped 3 of 4 PRMs. On Socratic nearly every answer contains an error, so between-answer level
carries little information about which answers are wrong.

The two PRMs where CAT34 loses are also the ones where Z17 itself is weakest relative to position.
