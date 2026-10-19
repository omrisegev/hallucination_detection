# Agent step telemetry full run (AIRCC 284668-284670, code 813d11b), pulled 2026-10-19

Qwen3-8B teacher-forced telemetry, completed 2026-10-14 21:38-21:45. All three alignment gates
report PASS (via the reviewed Qwen3-8B bf16 prefix exception, commit 813d11b).

| dataset | job | records |
|---|---|---|
| apb | 284668 | 991 |
| tprm | 284669 | 891 |
| aft | 284670 | 2,277 |

Pulled here: ALIGNMENT_GATE / TIMING / TOKENIZATION for the full runs, the smoke runs
(813d11b and e61d7fd) without their per-record files, the compute estimate, and the full-run
slurm logs. No analysis of these records has been run yet.

Left on AIRCC (not in git, about 4.1 GB): the per-answer `records/*.record.json` under
`/shared/cycle2_tau_averbuch_prj/omrisegev1/results/agent_step_telemetry_v1/{apb,tprm,aft}_qwen3_8b_813d11b_full/records/`.
