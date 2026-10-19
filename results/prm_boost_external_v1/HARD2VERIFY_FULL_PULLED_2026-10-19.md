# Hard2Verify full telemetry runs (AIRCC 288430-288439), pulled 2026-10-19

Two jobs per PRM (link 1 and its resume link 2), finished 2026-10-18 23:53 to 2026-10-19 00:05.
Telemetry collection finished for all 5 PRMs on all 200 Hard2Verify rows.

| PRM | status | rows over context (no telemetry) |
|---|---|---|
| rlhflow_prm | COMPLETE | 0 |
| qwen_prm (7B) | COMPLETE_WITH_FAILURES | 19 |
| qwen_prm_72b | COMPLETE_WITH_FAILURES | 19 |
| pure_prm | COMPLETE_WITH_FAILURES | 18 |
| math_shepherd | COMPLETE_WITH_FAILURES | 23 |

Slurm shows FAILED for the 8 non-RLHFlow jobs because the driver exits non-zero and prints
"fusion blocked" when any row overflows the 4,096-token context. This is the expected
behaviour seen in the timing runs, not a crash.

Not done yet: the downstream step `ext_down_hard2verify.sh` (bank_v2 build + scoring with
`prm_boost_score_hard2verify.py`) has not run. There is no `bank_v2/` and no
`scores_hard2verify/` on AIRCC, so there are no Hard2Verify boosting numbers yet. A decision is
needed on how to treat the overflow rows before scoring.

Pulled here: each run's `manifest.json`, the slurm logs, the downstream script and scorer.
Left on AIRCC (not in git): `external_telemetry.pkl` (120-167 MB each) and `run_identity.json`
(1.1 MB each) under
`/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_external_v1/full_<prm>_hard2verify/`.
