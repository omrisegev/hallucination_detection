# External PRM boosting: environment parity and timing (2026-10-18)

The external test of PRM boosting runs the same frozen recipe on Socratic-PRMBench and Hard2Verify. The recipe is the PRM step reward plus its own backbone telemetry, fused label-free. Plan: `docs/post_advisor_2026-10-17/05_PRM_BOOSTING_EVIDENCE_PLAN_HE.md`.

- **Driver:** `cluster/run_prmbench_prm_telemetry.py --answers` reads the label-free `answers.json` used by the Qwen3 external telemetry, so the uids and step lists are the same. It is submitted with `cluster/submit_prm_telemetry_offline.sbatch`, which installs from the offline wheelhouse.
- **Code on AIRCC:** `$SHARED/code_integration_7534b8bf4`.
- **Outputs:** `$SHARED/results/prm_boost_external_v1/`.
- **Not done yet:** no labels were read and nothing was scored. The full run waits for Omri's budget decision.

## Environment parity (T1, job 287910)

qwen_prm on the first 40 PRMBench rows was compared with the 2026-10-13 telemetry: **PASS**, max abs gaps {'rewards': 0.0, 'token_entropies': 0.0, 'token_spilled_energies': 0.0, 'token_logsumexp': 0.0}.

## Timing (12 examples per pair, jobs 287911-287920)

| adapter | benchmark | rows | elapsed s | s/row | full-run GPU-h | max input tokens | context overflow | step-text mismatch | status |
|---|---|---|---|---|---|---|---|---|---|
| math_shepherd | hard2verify | 12 | 11.6 | 0.97 | 0.05 | 4642 | 2 | 0 | COMPLETE_WITH_FAILURES |
| math_shepherd | socratic | 12 | 11.0 | 0.92 | 0.76 | 1512 | 0 | 0 | COMPLETE |
| pure_prm | hard2verify | 12 | 10.6 | 0.88 | 0.05 | 4411 | 2 | 0 | COMPLETE_WITH_FAILURES |
| pure_prm | socratic | 12 | 10.3 | 0.86 | 0.71 | 1342 | 0 | 0 | COMPLETE |
| qwen_prm_72b | hard2verify | 12 | 87.5 | 7.29 | 0.40 | 4443 | 2 | 0 | COMPLETE_WITH_FAILURES |
| qwen_prm_72b | socratic | 12 | 70.2 | 5.85 | 4.87 | 1365 | 0 | 0 | COMPLETE |
| qwen_prm | hard2verify | 12 | 6.9 | 0.58 | 0.03 | 4441 | 2 | 0 | COMPLETE_WITH_FAILURES |
| qwen_prm | socratic | 12 | 7.4 | 0.62 | 0.51 | 1372 | 0 | 0 | COMPLETE |
| rlhflow_prm | hard2verify | 12 | 7.4 | 0.61 | 0.03 | 4462 | 0 | 3 | COMPLETE |
| rlhflow_prm | socratic | 12 | 9.9 | 0.82 | 0.69 | 1363 | 0 | 0 | COMPLETE |

The full run needs about **8.1 GPU-h** of compute, plus about 1.5 minutes of setup per job. Socratic has 2,995 answers and Hard2Verify has 200.

## Issues found

- **Hard2Verify, Qwen2.5-Math family** (qwen_prm, qwen_prm_72b, pure_prm, math_shepherd): 2 of 12 answers exceed the 4,096-token context and are recorded as counted failures.
  - Proposed handling: no truncation. Compare all PRMs on the answers every model can read, and report coverage.
- **rlhflow_prm on Hard2Verify:** 3 step-text mismatches. This must be checked before the full run.

## Full run: Socratic only (Omri, 2026-10-18)

Omri approved running Socratic first. Hard2Verify runs only if the Socratic result is good.

Jobs are linear 2-link chains, all with the frozen recipe and the same code dir:

| adapter | jobs | wall per link |
|---|---|---|
| qwen_prm | 288069 → 288070 | 3 h |
| pure_prm | 288071 → 288072 | 3 h |
| rlhflow_prm | 288073 → 288074 | 3 h |
| math_shepherd | 288075 → 288076 | 3 h |
| qwen_prm_72b | 288077 → 288078 | 8 h |

Outputs go to `$SHARED/results/prm_boost_external_v1/full_<adapter>_socratic/`.
