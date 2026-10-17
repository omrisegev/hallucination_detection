# raw_channel_readout_fusion_v1 — pilot feasibility outputs only

Protocol and full-run instructions: `docs/experiments/RAW_CHANNEL_READOUT_FUSION_V1.md`.

| directory | cells | answers (erroneous) | purpose |
|---|---|---|---|
| `pilot_llama31_8b_tau1` | Llama-3.1-8B pilot gsm8k + math | 60 (30) | machinery check, τ = 1 |
| `pilot_llama31_8b_tau0.25`, `_tau0.5`, `_tau2` | same | 60 (30) | τ sensitivity (descriptive) |
| `pilot_qwen3_4b_gsm8k_tau1` | Qwen3-4B pilot gsm8k | 30 (30) | second scorer, τ→0 identity 1.00 in all folds |

Each directory holds `REPORT.md` (readout grid, binary-vs-soft table, fits), `REPORT.json`, and
`PREDICTIONS.csv` (per-answer out-of-fold predictions for every rule).

These are 30-answer pilot caches. Per CLAUDE.md (2026-09-07) they check implementation and
runtime only; they do not rank readouts, compare binary with soft, or promote anything. The
full cells (3,400 answers per scorer) have not been run here because the telemetry is not in git
and exceeds the Drive connector's 10 MB limit.
