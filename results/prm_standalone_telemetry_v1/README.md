# prm_standalone_telemetry_v1: telemetry from each PRM's base model, standalone (PRMBench)

Plan, models, gates and the step-by-step handoff for the AIRCC machine:
[docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md](../../docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md).

**Status (2026-10-22, AIRCC):** all five standalone runs COMPLETE (6,969/6,969 rows each, 0 failed rows, 0 context
overflow); downstream P1 parity PASS; local backbone parity PASS for all five PRMs. Results below.

| path | what |
|---|---|
| `aircc/common.sh` | paths and the per-PRM model / prompt-style / wall table |
| `aircc/prefetch.sh` | login-node download of the standalone checkpoints |
| `aircc/submit_gpu.sh` | `timing` or `full` telemetry jobs, one per PRM (`cluster/submit_prm_telemetry_offline.sbatch`) |
| `aircc/downstream.sbatch` | CPU: P1 bank parity, then bank + ablation5 export per PRM (tag `<prm>__standalone`) |
| `aircc/upload_drive.sh` | rclone to `gdrive:hallucination_detection/cluster_results/prm_standalone_telemetry_v1/` |
| `standalone_table.py` | local: one PRMBench table per PRM, parity with `prm_boost_collection_v1/tables/STATUS_MAIN.csv` |

Driver change (`cluster/run_prmbench_prm_telemetry.py`):
- `--mode instruct` accepts any `--model`, with `--prompt-style qwen_cot_chat | chat | plain`.
- An empty step that maps to no token is written as an empty span (`fill_empty_spans`).

Tests: `tests/test_standalone_prompt.py` (5 pass).

Local CPU smoke (tiny-random Qwen2 / Llama-3.1 / Mistral models, 3 fixture rows):
- all three styles COMPLETE with the existing output schema;
- the `qwen_cot_chat` output is byte-identical to the driver before this change;
- `standalone_table.py --smoke-no-parity` runs on synthetic scores for all five PRMs.

## Runs (AIRCC, 2026-10-22)

Frozen code: `code_standalone_v1` (SYNC_COMMIT 41b63ac). Manifests, run identities and job logs: `aircc_runs/`.

| PRM | prompt style | timing job (12 rows) | timing s | full job | full rows | failed | context overflow | max input tokens | full s |
|---|---|---|---|---|---|---|---|---|---|
| qwen_prm_7b | qwen_cot_chat | 290704 | 8.7 | 290711 | 6969 | 0 | 0 | 3555 | 388.9 |
| qwen_prm_72b | qwen_cot_chat | 290716 | 77.5 | 290722 | 6969 | 0 | 0 | 3555 | 849.9 |
| pure_prm | qwen_cot_chat | 290705 | 7.4 | 290712 | 6969 | 0 | 0 | 3555 | 357.2 |
| rlhflow_prm | chat | 290706 | 7.8 | 290713 | 6969 | 0 | 0 | 3251 | 337.0 |
| math_shepherd | plain | 290707 | 7.7 | 290714 | 6969 | 0 | 0 | 3682 | 402.6 |

All timing and full jobs ended COMPLETED 0:0. No row exceeded 4,096 input tokens (the Qwen-Math context), and no
step-text mismatch was recorded. `SA_WALL` for the 7-8B runs was raised from 2 h to 4 h before submission (the
unbatched timing rate was close to 2 h); the measured full runs took 6-14 minutes.

Downstream (CPU): job 290729 FAILED at P1 with `mask failures` (3 step spans already empty in the source, e.g.
`step 1 span (9, 9) becomes empty after dropping token 0`; `aircc_runs/logs/sa_downstream_290729_FAILURE.json`).
Cause: `downstream.sbatch` pointed `--mask-code` at `code_thinkfix_rescore_v1`, while the frozen backbone banks
(job 283836) were built with `code_prm_boost_v1/mask_fixed`, which keeps an already-empty `(k,k)` span empty.
After the fix, job 290730 COMPLETED: P1 rebuild of the pure_prm backbone bank equals the frozen bank on every key
(`values` max diff 0.0; `aircc_runs/PARITY_P1_bank_pure_prm.json`), five standalone banks + ablation5 exports,
`SA_DOWNSTREAM_DONE`. Hashes: `aircc_runs/MD5_abl5.txt` (the local copies of the five ABL5_SCORES files match).

Cross-check: the Qwen-Math-7B-Instruct standalone run equals the 2026-10-12 run (job 282803) bit for bit on the first
40 rows (token ids, entropies, spilled energies, logsumexp, top-50 ids/logprobs, step spans: max abs diff 0), and
its manifest reproduces that run's surprisal median/mean and entropy median to every printed digit.

Local tables (`standalone_table.py`): backbone parity with `prm_boost_collection_v1/tables/STATUS_MAIN.csv` PASS on
6 backbone rows for each of the five PRMs. Outputs: `tables/STANDALONE_MAIN.csv`, `STANDALONE_CONTRASTS.csv`,
`STANDALONE_TABLES.md` (all rules, all arms).

## Results (PRMBench, 6,969 answers; development data)

Within-AUC and official PRMScore under the GAP rule (other rules in `tables/`).

| PRM | PRM alone (GAP) | backbone mean | backbone L-SML | standalone mean | standalone L-SML | PRM + backbone L-SML |
|---|---|---|---|---|---|---|
| qwen_prm_7b | .801 / 68.01 | .805 / 68.60 | .818 / 69.26 | .782 / 67.14 | .793 / 68.26 | .838 / 70.40 |
| qwen_prm_72b | .795 / 69.49 | .805 / 68.11 | .823 / 69.68 | .780 / 67.39 | .781 / 67.80 | .843 / 71.09 |
| pure_prm | .763 / 66.24 | .785 / 67.98 | .792 / 68.71 | .769 / 67.06 | .770 / 67.63 | .816 / 69.51 |
| rlhflow_prm | .662 / 59.39 | .756 / 65.57 | .767 / 66.17 | .783 / 67.45 | .788 / 68.20 | .775 / 66.25 |
| math_shepherd | .684 / 60.21 | .740 / 64.22 | .748 / 64.97 | .754 / 66.13 | .760 / 66.94 | .778 / 66.39 |

PRM native threshold (Q80 column of the native arm): qwen_prm_7b 65.49, qwen_prm_72b 68.18, pure_prm 65.31,
rlhflow_prm 54.43, math_shepherd 47.02.

Standalone minus backbone, paired source-group bootstrap (707 groups, 2000 draws), GAP rule, 95% CI:

| PRM | L-SML: PRMScore | L-SML: within-AUC | plain mean: PRMScore | plain mean: within-AUC |
|---|---|---|---|---|
| qwen_prm_7b | -0.99 [-1.41, -0.58] | -0.024 [-0.028, -0.021] | -1.46 [-1.88, -1.06] | -0.023 [-0.026, -0.020] |
| qwen_prm_72b | -1.88 [-2.37, -1.42] | -0.042 [-0.046, -0.038] | -0.72 [-1.16, -0.29] | -0.025 [-0.029, -0.022] |
| pure_prm | -1.08 [-1.39, -0.75] | -0.021 [-0.024, -0.019] | -0.93 [-1.26, -0.59] | -0.016 [-0.019, -0.014] |
| rlhflow_prm | +2.03 [+1.61, +2.45] | +0.021 [+0.017, +0.025] | +1.89 [+1.46, +2.30] | +0.027 [+0.023, +0.030] |
| math_shepherd | +1.97 [+1.51, +2.42] | +0.012 [+0.008, +0.016] | +1.91 [+1.51, +2.32] | +0.014 [+0.010, +0.017] |

Reading: for the three Qwen-based PRMs the standalone base model's telemetry is weaker than the PRM backbone's
telemetry; for RLHFlow (Llama-3.1-8B-Instruct) and Math-Shepherd (Mistral-7B-v0.1) it is stronger. Standalone
L-SML vs the PRM alone (GAP PRMScore): +0.26 [-0.36, +0.85] (qwen_prm_7b), -1.69 [-2.36, -1.04] (qwen_prm_72b),
+1.40 [+0.78, +1.99] (pure_prm), +8.81 [+8.10, +9.55] (rlhflow_prm), +6.73 [+6.00, +7.53] (math_shepherd).
These are development results on PRMBench; they do not establish a new method.

Drive: `gdrive:hallucination_detection/cluster_results/prm_standalone_telemetry_v1/` (abl5/, full_<prm>/ manifests
without pkls, bank_<prm>/, jobs/): 51 objects, 510.669 MiB (abl5 25 files 209.655 MiB, `rclone check` 0 differences;
full_<prm> 2 files 8.809 MiB each; bank_<prm> 2 files ~51.3-51.4 MiB each; jobs 6 files 32 KiB). Raw telemetry pkls stay on AIRCC under
`results/prm_standalone_telemetry_v1/full_<prm>/`.
