# Selected-token and tail preflight

- Localization artifacts ready: 9/9.
- Historical cells ready: 24/24.
- Audited tokens: 6,968,779 localization and 8,949,580 historical.
- PB/PRMB tokens are teacher-forced scored answer tokens; historical tokens are generated/sampled outputs.
- A selected token outside saved Top-50 is valid because its probability is stored separately.

| artifact/cell | tokens | selected in Top-1 | Top-15 | Top-50 | max ATP error | status |
|---|---:|---:|---:|---:|---:|---|
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_gsm8k.pkl` | 114502 | 83.5% | 99.0% | 99.5% | 0.00e+00 | READY |
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_math.pkl` | 523899 | 84.2% | 99.1% | 99.7% | 0.00e+00 | READY |
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_olympiadbench.pkl` | 781284 | 84.6% | 99.3% | 99.8% | 0.00e+00 | READY |
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_omnimath.pkl` | 773607 | 82.0% | 99.0% | 99.7% | 0.00e+00 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl` | 114502 | 83.9% | 99.0% | 99.5% | 0.00e+00 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_math.pkl` | 523899 | 84.2% | 99.1% | 99.7% | 0.00e+00 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_olympiadbench.pkl` | 781284 | 84.5% | 99.2% | 99.8% | 0.00e+00 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_omnimath.pkl` | 773607 | 82.0% | 99.0% | 99.7% | 0.00e+00 | READY |
| `dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl` | 2582195 | 74.9% | 95.9% | 98.0% | 0.00e+00 | READY |
| `epr_triviaqa_mistral24b` | 18768 | 82.4% | 99.2% | 100.0% | 0.00e+00 | READY |
| `losnet_hotpotqa_mistral7b` | 111538 | 100.0% | 100.0% | 100.0% | 0.00e+00 | READY |
| `sciq_llama8b` | 14503 | 81.0% | 99.4% | 100.0% | 0.00e+00 | READY |
| `se_nq_open_llama8b` | 291965 | 90.8% | 99.9% | 100.0% | 0.00e+00 | READY |
| `se_squad_v2_llama8b` | 87710 | 94.3% | 100.0% | 100.0% | 0.00e+00 | READY |
| `seiclr_triviaqa_opt30b` | 319305 | 89.0% | 99.5% | 100.0% | 0.00e+00 | READY |
| `semenergy_triviaqa_qwen3_8b` | 72078 | 97.6% | 100.0% | 100.0% | 0.00e+00 | READY |
| `spilled_triviaqa_llama8b` | 8904 | 71.5% | 96.9% | 100.0% | 0.00e+00 | READY |
| `truthfulqa_llama8b` | 514520 | 88.5% | 100.0% | 100.0% | 0.00e+00 | READY |
| `ars_gsm8k_r1distill8b` | 224633 | 100.0% | 100.0% | 100.0% | 0.00e+00 | READY |
| `internalstates_gsm8k_qwen25_7b` | 173459 | 93.4% | 100.0% | 100.0% | 0.00e+00 | READY |
| `lapeigvals_gsm8k_llama3b` | 309209 | 87.4% | 99.7% | 100.0% | 0.00e+00 | READY |
| `lapeigvals_gsm8k_llama8b` | 132190 | 83.4% | 99.1% | 100.0% | 0.00e+00 | READY |
| `lapeigvals_gsm8k_mistral24b` | 337782 | 95.6% | 99.9% | 100.0% | 0.00e+00 | READY |
| `lapeigvals_gsm8k_nemo` | 357886 | 93.1% | 99.9% | 100.0% | 0.00e+00 | READY |
| `lapeigvals_gsm8k_phi35` | 408196 | 93.2% | 99.9% | 100.0% | 0.00e+00 | READY |
| `noise_gsm8k_mistral7b` | 442970 | 88.9% | 99.9% | 100.0% | 0.00e+00 | READY |
| `noise_gsm8k_phi3mini` | 352311 | 90.2% | 99.9% | 100.0% | 0.00e+00 | READY |
| `math500_dsmath7b` | 159307 | 61.5% | 92.5% | 99.5% | 0.00e+00 | READY |
| `math500_qwenmath7b` | 472931 | 19.1% | 45.4% | 94.2% | 0.00e+00 | READY |
| `math500_r1distill8b` | 469303 | 63.3% | 93.8% | 99.9% | 0.00e+00 | READY |
| `math500_r1distill8b_mn4096` | 813758 | 55.5% | 90.4% | 99.9% | 0.00e+00 | READY |
| `trace_gsm8k_llama8b_k10` | 1330251 | 84.0% | 99.3% | 100.0% | 0.00e+00 | READY |
| `trace_math500_qwenmath15b_k10` | 1526103 | 80.8% | 98.4% | 100.0% | 0.00e+00 | READY |
