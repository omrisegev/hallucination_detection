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

## Downstream pipeline, frozen before any full external result

1. `build_bank_external.py` (AIRCC CPU, cwd `code_thinkfix_bank_553de79`): computes the same per-step 56-channel bank as on PRMBench, using the frozen `rebuild_source_bank_v4.one`. It applies mask1, drops empty steps, writes the external_banks_v4 layout, and saves rewards and native flags to `REWARDS.npz`. Labels are not read.
2. `score_external.py` (AIRCC CPU, cwd `eval_mirror_prm_gen_v1`): computes the label-free arms on each PRM's own unlabeled cell. The arms are PRM, HALF_N16, LSML_N16_PRMG and LSML_CONT, with the same code path as `results/prm_boosting_controls_v1/lsml_controls.py`. Output: `SCORES.npz`.
3. `eval_external.py` (local; the gold labels never leave this machine): reports the official benchmark metric and within-AUC under PRM native and ANSWER_Z_Q80, with a paired bootstrap over answers.

Smoke check: steps 1-3 ran end to end on the 12-row timing cells (jobs 288087 and a scoring smoke). Those numbers are a pipeline check only, not a result.

## Socratic result (2026-10-18): the boosting does not transfer consistently

CPU job 288317 built the banks and scored the cells without labels. The local evaluation (`eval_external.py`) wrote `eval_socratic/RESULTS_socratic.csv` and `eval_socratic/BOOTSTRAP_socratic.csv`, with a paired bootstrap over 2,995 answers.

Parity check: PRM-7B native gives 0.6809 PRMScore and 0.8236 within-AUC, the same as the earlier external PRM run (0.680 / 0.824).

| PRM | PRM native | PRM Q80 | HALF_N16 (Q80) | LSML_N16_PRMG | LSML_CONT (3 ch.) |
|---|---|---|---|---|---|
| Qwen-7B | **0.681** | 0.674 | 0.674 | 0.651 | 0.647 |
| Qwen-72B | **0.720** | 0.669 | 0.673 | 0.587 | 0.575 |
| PURE | 0.637 | 0.641 | **0.676** | 0.644 | 0.624 |
| RLHFlow | 0.481 | 0.491 | 0.517 | **0.576** | 0.534 |
| MathShepherd | **0.644** | 0.595 | 0.621 | 0.500 | 0.509 |

The table shows official PRMScore. Within-AUC for HALF_N16 minus PRM:

| PRM | Δ within-AUC | 95% CI |
|---|---|---|
| MathShepherd | +0.017 | [+0.012, +0.023] |
| PURE | +0.017 | [+0.011, +0.022] |
| Qwen-7B | −0.011 | [−0.015, −0.007] |
| Qwen-72B | −0.005 | [−0.010, −0.000] |
| RLHFlow | +0.022 | [+0.017, +0.028] |

Reading:
- **L-SML fusions break on the strong PRMs.** Within-AUC drops by 0.07–0.27. L-SML gives the telemetry much more weight than the PRM: the fused score's correlation with the PRM is only 0.06–0.53. RLHFlow is the exception and gains +9.4 PRMScore.
- **The clean ½·PRM + ½·N16 recipe beats the PRM under the same Q80 rule in all 5 models** (+0.0 to +3.5). Against each PRM's native threshold it wins only for PURE (+3.9) and RLHFlow (+3.6). It loses for Qwen-7B (−0.7), Qwen-72B (−4.7) and MathShepherd (−2.3).
- **The pre-registered success rule fails on Socratic.** That rule required a positive gain with a CI above 0. Hard2Verify is therefore not run, per Omri's condition.
