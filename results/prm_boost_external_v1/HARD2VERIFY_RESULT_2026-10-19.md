# Hard2Verify result (2026-10-19): overflow rows removed

Omri's decision (2026-10-19): drop the answers that exceed a PRM's 4,096-token context for now.
The bank builder already skips rows whose telemetry failed, so no code changed. CPU job 288968 ran
`ext_down_hard2verify.sh` (bank_v2 + `prm_boost_score_hard2verify.py`, label-free). The evaluation
ran locally with `eval_external.py` (gold labels stay local).

Two views:
- `eval_hard2verify/per_prm/`: each PRM on the answers it can read (Qwen-7B 181, Qwen-72B 181,
  PURE 182, MathShepherd 177, RLHFlow 200).
- `eval_hard2verify/common_177/`: all PRMs on the 177 answers every PRM can read
  (`common_rows.py`). This is the comparable view; the numbers barely move.

Official metric is Balanced F1. Common 177 answers, metric / within-AUC:

| PRM | PRM native | PRM Q80 | HALF_N16 (Q80) | LSML_N16_PRMG | TELE_N16 (no PRM) |
|---|---|---|---|---|---|
| Qwen-7B | 0.163 / 0.638 | 0.394 / 0.638 | **0.401** / 0.610 | 0.381 / 0.588 | 0.392 / 0.561 |
| Qwen-72B | 0.311 / 0.513 | 0.367 / 0.513 | **0.383** / 0.531 | 0.380 / 0.565 | 0.367 / 0.562 |
| PURE | **0.526** / 0.763 | 0.442 / 0.763 | 0.427 / 0.715 | 0.336 / 0.481 | 0.336 / 0.480 |
| RLHFlow | 0.059 / 0.185 | 0.138 / 0.185 | 0.171 / 0.245 | 0.336 / 0.482 | **0.348** / 0.500 |
| MathShepherd | 0.079 / 0.525 | 0.340 / 0.525 | 0.367 / 0.536 | 0.381 / 0.535 | 0.383 / 0.535 |

HALF_N16 minus PRM Q80 (same decision rule), paired bootstrap over answers, 95% CI:
Qwen-7B +0.007 [-0.023, +0.035]; Qwen-72B +0.017 [-0.001, +0.034]; PURE -0.016 [-0.047, +0.014];
RLHFlow +0.033 [+0.005, +0.061]; MathShepherd +0.026 [-0.000, +0.054]. Within-AUC: worse for
Qwen-7B (-0.027) and PURE (-0.048), better for RLHFlow (+0.060), within noise for the others.

Caveats:
- Parity: Qwen-7B PRM on these rows gives Q80 0.394 / within 0.638, matching the earlier
  external PRM run on all 200 rows (0.391 / 0.634).
- Step position alone gives within-AUC 0.866 on Hard2Verify, higher than every method here, so
  within-AUC on this benchmark is dominated by where the errors sit, not by local evidence.
- RLHFlow's risk falls along the answer (mean per-answer correlation with step index -0.86), so
  its within-AUC is 0.185, well below chance. The run status was COMPLETE; the 3 step-text
  mismatches flagged in the timing run were not re-checked.
- PRM native Balanced F1 is very low for several PRMs because the 0.5 threshold marks almost
  every step correct on this benchmark.
- 200 (or 177) answers only; intervals are wide.

Large files left on AIRCC: `full_<prm>_hard2verify/bank_v2/{FEATURES,REWARDS}.npz` and
`scores_hard2verify/SCORES_<prm>.npz` under `$SHARED/results/prm_boost_external_v1/`.
