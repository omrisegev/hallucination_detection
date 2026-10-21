# prm_standalone_telemetry_v1: telemetry from each PRM's base model, standalone (PRMBench)

Plan, models, gates and the step-by-step handoff for the AIRCC machine:
[docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md](../../docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md).

**Status (2026-10-22, Mac):** code and tests only. No GPU job has run.

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
