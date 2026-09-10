# Direct probability fusion: data audit

Required vocabulary ranks: **K=15**.
The existing `token_entropies` use top-15 probabilities, renormalized before Shannon entropy.
The token top-10 readout is a separate aggregation over token positions inside a step.

## Readiness

- Localization: 9/9 artifacts ready.
- Historical complete-answer benchmark: 24/24 cells ready.
- This audit uses cached gray-box data only. It does not use or alter white-box captures.

## Historical cells

| cell | rows | min K | token range | status |
|---|---:|---:|---:|---|
| `epr_triviaqa_mistral24b` | 1000 | 50 | 2–64 | READY |
| `losnet_hotpotqa_mistral7b` | 500 | 1000 | 33–512 | READY |
| `sciq_llama8b` | 1000 | 50 | 2–196 | READY |
| `se_nq_open_llama8b` | 10000 | 50 | 2–128 | READY |
| `se_squad_v2_llama8b` | 10000 | 50 | 2–198 | READY |
| `seiclr_triviaqa_opt30b` | 5000 | 50 | 3–64 | READY |
| `semenergy_triviaqa_qwen3_8b` | 5000 | 50 | 6–64 | READY |
| `spilled_triviaqa_llama8b` | 500 | 50 | 2–256 | READY |
| `truthfulqa_llama8b` | 8170 | 50 | 2–128 | READY |
| `ars_gsm8k_r1distill8b` | 500 | 50 | 189–1406 | READY |
| `internalstates_gsm8k_qwen25_7b` | 500 | 50 | 120–2048 | READY |
| `lapeigvals_gsm8k_llama3b` | 1319 | 50 | 79–888 | READY |
| `lapeigvals_gsm8k_llama8b` | 500 | 50 | 81–2048 | READY |
| `lapeigvals_gsm8k_mistral24b` | 1319 | 50 | 83–760 | READY |
| `lapeigvals_gsm8k_nemo` | 1319 | 50 | 83–1915 | READY |
| `lapeigvals_gsm8k_phi35` | 1319 | 50 | 110–2048 | READY |
| `noise_gsm8k_mistral7b` | 1319 | 50 | 91–2048 | READY |
| `noise_gsm8k_phi3mini` | 1319 | 50 | 85–1552 | READY |
| `math500_dsmath7b` | 300 | 50 | 30–2048 | READY |
| `math500_qwenmath7b` | 300 | 50 | 27–2048 | READY |
| `math500_r1distill8b` | 300 | 50 | 249–2048 | READY |
| `math500_r1distill8b_mn4096` | 300 | 50 | 203–4096 | READY |
| `trace_gsm8k_llama8b_k10` | 5000 | 50 | 89–2048 | READY |
| `trace_math500_qwenmath15b_k10` | 3000 | 50 | 82–2048 | READY |
