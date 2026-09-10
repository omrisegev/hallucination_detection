# Selected-token and tail preflight

- Localization artifacts ready: 9/9.
- Historical cells ready: 24/24.
- Audited tokens: 6,968,779 localization and 8,949,580 historical.
- PB/PRMB tokens are teacher-forced scored answer tokens; historical tokens are generated/sampled outputs.
- A selected token outside saved Top-50 is valid because its probability is stored separately.

| artifact/cell | tokens | selected Top-15 | tail >1e-6 | tail >1e-3 | tail >1e-2 | min within-row tail SD | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_gsm8k.pkl` | 114502 | 99.0% | 46.4% | 5.7% | 1.1% | 9.11e-06 | READY |
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_math.pkl` | 523899 | 99.1% | 43.1% | 8.1% | 2.8% | 8.55e-06 | READY |
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_olympiadbench.pkl` | 781284 | 99.3% | 43.9% | 9.2% | 3.2% | 1.21e-04 | READY |
| `dataset_cache/repgrid/pb_qwen3_4b/processbench_omnimath.pkl` | 773607 | 99.0% | 50.5% | 12.9% | 5.0% | 6.26e-05 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl` | 114502 | 99.0% | 52.7% | 7.5% | 1.7% | 1.04e-05 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_math.pkl` | 523899 | 99.1% | 46.3% | 9.7% | 3.7% | 4.66e-05 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_olympiadbench.pkl` | 781284 | 99.2% | 46.5% | 10.7% | 4.1% | 1.44e-04 | READY |
| `dataset_cache/repgrid/pb_qwen3_8b/processbench_omnimath.pkl` | 773607 | 99.0% | 53.4% | 14.6% | 6.2% | 1.15e-04 | READY |
| `dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl` | 2582195 | 95.9% | 62.4% | 19.0% | 7.9% | 4.25e-05 | READY |
| `epr_triviaqa_mistral24b` | 18768 | 99.2% | 93.8% | 37.1% | 13.6% | 7.20e-05 | READY |
| `losnet_hotpotqa_mistral7b` | 111538 | 100.0% | 56.3% | 13.7% | 4.7% | 6.50e-04 | READY |
| `sciq_llama8b` | 14503 | 99.4% | 78.6% | 26.1% | 11.0% | 7.73e-07 | READY |
| `se_nq_open_llama8b` | 291965 | 99.9% | 24.3% | 4.1% | 0.9% | 0.00e+00 | READY |
| `se_squad_v2_llama8b` | 87710 | 100.0% | 11.0% | 0.6% | 0.0% | 0.00e+00 | READY |
| `seiclr_triviaqa_opt30b` | 319305 | 99.5% | 47.2% | 15.1% | 8.4% | 3.86e-06 | READY |
| `semenergy_triviaqa_qwen3_8b` | 72078 | 100.0% | 5.9% | 1.6% | 0.5% | 0.00e+00 | READY |
| `spilled_triviaqa_llama8b` | 8904 | 96.9% | 89.3% | 51.0% | 34.2% | 1.84e-06 | READY |
| `truthfulqa_llama8b` | 514520 | 100.0% | 26.5% | 3.7% | 0.7% | 0.00e+00 | READY |
| `ars_gsm8k_r1distill8b` | 224633 | 100.0% | 69.3% | 6.1% | 0.7% | 2.58e-04 | READY |
| `internalstates_gsm8k_qwen25_7b` | 173459 | 100.0% | 22.3% | 2.5% | 0.6% | 9.13e-07 | READY |
| `lapeigvals_gsm8k_llama3b` | 309209 | 99.7% | 76.0% | 13.8% | 4.9% | 7.03e-05 | READY |
| `lapeigvals_gsm8k_llama8b` | 132190 | 99.1% | 71.0% | 19.8% | 10.1% | 1.40e-04 | READY |
| `lapeigvals_gsm8k_mistral24b` | 337782 | 99.9% | 48.2% | 5.2% | 2.2% | 5.15e-06 | READY |
| `lapeigvals_gsm8k_nemo` | 357886 | 99.9% | 42.3% | 4.0% | 0.9% | 2.11e-05 | READY |
| `lapeigvals_gsm8k_phi35` | 408196 | 99.9% | 29.2% | 3.4% | 1.0% | 6.46e-06 | READY |
| `noise_gsm8k_mistral7b` | 442970 | 99.9% | 64.7% | 13.3% | 3.3% | 2.61e-04 | READY |
| `noise_gsm8k_phi3mini` | 352311 | 99.9% | 69.5% | 12.2% | 2.6% | 2.73e-04 | READY |
| `math500_dsmath7b` | 159307 | 92.5% | 99.8% | 76.8% | 54.7% | 7.80e-03 | READY |
| `math500_qwenmath7b` | 472931 | 45.4% | 100.0% | 97.2% | 89.6% | 2.54e-03 | READY |
| `math500_r1distill8b` | 469303 | 93.8% | 100.0% | 80.1% | 55.3% | 7.20e-03 | READY |
| `math500_r1distill8b_mn4096` | 813758 | 90.4% | 100.0% | 87.5% | 69.0% | 4.50e-03 | READY |
| `trace_gsm8k_llama8b_k10` | 1330251 | 99.3% | 70.5% | 18.4% | 8.8% | 1.07e-04 | READY |
| `trace_math500_qwenmath15b_k10` | 1526103 | 98.4% | 90.8% | 39.2% | 19.7% | 3.90e-05 | READY |
