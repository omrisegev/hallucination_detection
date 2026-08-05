# Dependency-aware fusion experiment — results

Configuration hash: `568dc60530928f54`. Loaded cells: **24**. 
Failed arm/seed records: **113**.

Labels were used only to compute the metrics below, after every method score was frozen.

## Arm summary

| arm | n | macro AUROC | QA | math | median fit s | failures |
|---|---:|---:|---:|---:|---:|---:|
| `keep.iu_pcr` | 24 | 0.7742 | 0.7585 | 0.7836 | 0.003 | 0 |
| `deployed.upcr_signrho` | 24 | 0.7741 | 0.7585 | 0.7834 | 0.003 | 0 |
| `keep.su_pcr_reproduction` | 24 | 0.7736 | 0.7584 | 0.7826 | 0.003 | 0 |
| `deployed.dufs_lsml` | 24 | 0.7687 | 0.7520 | 0.7787 | 0.267 | 0 |
| `full.su_pcr_reproduction` | 24 | 0.7668 | 0.7347 | 0.7861 | 0.003 | 0 |
| `full.deem_deep_hard_ensemble` | 24 | 0.7544 | 0.7296 | 0.7692 | 45.737 | 0 |
| `full.iu_pcr` | 24 | 0.7542 | 0.7580 | 0.7519 | 0.003 | 0 |
| `full.deem_deep_hard_seed0` | 24 | 0.7517 | 0.7326 | 0.7631 | 45.737 | 0 |
| `keep.iu_ridge` | 24 | 0.7512 | 0.7452 | 0.7549 | 0.003 | 0 |
| `full.deem_irbm_hard_ensemble` | 24 | 0.7464 | 0.7403 | 0.7501 | 35.346 | 0 |
| `full.deem_irbm_hard_seed0` | 24 | 0.7464 | 0.7403 | 0.7500 | 35.346 | 0 |
| `keep.sdsf` | 24 | 0.7353 | 0.7273 | 0.7401 | 0.003 | 0 |
| `full.iu_ridge` | 24 | 0.7318 | 0.7316 | 0.7319 | 0.003 | 0 |
| `full.sdsf` | 24 | 0.7104 | 0.6939 | 0.7202 | 0.003 | 0 |

## Paired contrasts

Positive means the candidate is better. Deltas are AUROC percentage points.

| contrast | reference → candidate | mean [95% CI] | QA/math | dataset macro [CI] | W/L | p | Holm |
|---|---|---:|---:|---:|---:|---:|---:|
| `H1_sparse_reliability` | `full.iu_pcr` → `full.su_pcr_reproduction` | +1.26 [-1.78, +6.33] | -2.33/+3.42 | -0.40 [-4.38, +3.52] | 14/8 | 0.7332 | 1 |
| `H2_dependency_weights` | `full.su_pcr_reproduction` → `full.sdsf` | -5.65 [-10.19, -2.73] | -4.08/-6.59 | -3.77 [-6.93, -1.22] | 2/22 | 5.96e-07 | 1.788e-06 |
| `A1_ridge_without_sparse` | `full.iu_pcr` → `full.iu_ridge` | -2.24 [-3.41, -1.20] | -2.64/-2.00 | -1.33 [-2.76, -0.16] | 3/21 | 6.39e-05 | nan |
| `A3_deem_depth_hard` | `full.deem_irbm_hard_ensemble` → `full.deem_deep_hard_ensemble` | +0.80 [-1.34, +4.44] | -1.06/+1.91 | +0.11 [-1.70, +2.67] | 8/16 | 0.03665 | nan |
| `P1_sdsf_vs_deployed` | `deployed.upcr_signrho` → `full.sdsf` | -6.37 [-11.03, -3.15] | -6.46/-6.32 | -5.41 [-9.53, -1.66] | 2/22 | 5.96e-07 | nan |
| `P3_sdsf_vs_dufs` | `deployed.dufs_lsml` → `full.sdsf` | -5.83 [-10.69, -2.58] | -5.81/-5.84 | -5.06 [-9.42, -1.28] | 3/21 | 2.98e-06 | nan |
| `K1_keep_sparse_reliability` | `keep.iu_pcr` → `keep.su_pcr_reproduction` | -0.07 [-0.20, +0.01] | -0.01/-0.10 | -0.02 [-0.11, +0.04] | 8/10 | 0.446 | nan |
| `K2_keep_dependency_weights` | `keep.su_pcr_reproduction` → `keep.sdsf` | -3.82 [-5.53, -2.41] | -3.11/-4.25 | -2.60 [-4.05, -1.26] | 0/24 | 1.192e-07 | nan |
| `A4_factorial_interaction` | `(iu_ridge-iu_pcr)` → `(sdsf-su_pcr)` | -3.41 [-8.07, -0.79] | -1.44/-4.58 | -2.44 [-5.39, -0.48] | 4/20 | 0.0002052 | nan |

## DEEM seed stability

AUROC standard deviation and range are computed across the registered seeds per cell.

| arena.arm | complete cells | mean cell std | max cell range |
|---|---:|---:|---:|
| `full.deem_deep_hard` | 24 | 1.32pp | 38.43pp |
| `full.deem_irbm_hard` | 24 | 0.02pp | 0.34pp |

## Sparse decomposition diagnostics

| arena | cells | converged | theorem support | median sparse fraction | median residual |
|---|---:|---:|---:|---:|---:|
| `full` | 24 | 24/24 | 21/24 | 0.0035 | 0.1531 |
| `keep` | 24 | 24/24 | 23/24 | 0.0000 | 0.1196 |

## Registered interpretation gate

SDSF advances as a contribution only if `H2_dependency_weights` has mean gain at least +1.0pp, a positive cell-bootstrap lower bound, Holm-adjusted p<0.05, QA and math deltas each at least -0.5pp, a positive equal-dataset macro, and at least 90% primary-arena decomposition convergence. `H1` alone is evidence for the published sparse-error correction, not for our new weighting method.

## Failures

- `epr_triviaqa_mistral24b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `epr_triviaqa_mistral24b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `epr_triviaqa_mistral24b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `epr_triviaqa_mistral24b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `losnet_hotpotqa_mistral7b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `losnet_hotpotqa_mistral7b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `losnet_hotpotqa_mistral7b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `losnet_hotpotqa_mistral7b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `sciq_llama8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `sciq_llama8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `sciq_llama8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `sciq_llama8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `sciq_llama8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `se_nq_open_llama8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `se_nq_open_llama8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `se_nq_open_llama8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `se_nq_open_llama8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `se_nq_open_llama8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `se_squad_v2_llama8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `se_squad_v2_llama8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `se_squad_v2_llama8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `se_squad_v2_llama8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `se_squad_v2_llama8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `seiclr_triviaqa_opt30b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `seiclr_triviaqa_opt30b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `seiclr_triviaqa_opt30b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `seiclr_triviaqa_opt30b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `semenergy_triviaqa_qwen3_8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `semenergy_triviaqa_qwen3_8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `semenergy_triviaqa_qwen3_8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `semenergy_triviaqa_qwen3_8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `semenergy_triviaqa_qwen3_8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `spilled_triviaqa_llama8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `spilled_triviaqa_llama8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `spilled_triviaqa_llama8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `spilled_triviaqa_llama8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `spilled_triviaqa_llama8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `truthfulqa_llama8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `truthfulqa_llama8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `truthfulqa_llama8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `truthfulqa_llama8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `truthfulqa_llama8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `ars_gsm8k_r1distill8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `ars_gsm8k_r1distill8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `ars_gsm8k_r1distill8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `ars_gsm8k_r1distill8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `ars_gsm8k_r1distill8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `internalstates_gsm8k_qwen25_7b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `internalstates_gsm8k_qwen25_7b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `internalstates_gsm8k_qwen25_7b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `internalstates_gsm8k_qwen25_7b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `internalstates_gsm8k_qwen25_7b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama3b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama3b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama3b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama3b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama3b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_llama8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_mistral24b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_mistral24b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_mistral24b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_mistral24b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_mistral24b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_nemo/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_nemo/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_nemo/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_nemo/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_nemo/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_phi35/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_phi35/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_phi35/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_phi35/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `lapeigvals_gsm8k_phi35/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_mistral7b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_mistral7b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_mistral7b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_mistral7b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_mistral7b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_phi3mini/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_phi3mini/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_phi3mini/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_phi3mini/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `noise_gsm8k_phi3mini/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `math500_dsmath7b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `math500_dsmath7b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `math500_dsmath7b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `math500_qwenmath7b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `math500_qwenmath7b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `math500_qwenmath7b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b_mn4096/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b_mn4096/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b_mn4096/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b_mn4096/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `math500_r1distill8b_mn4096/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `trace_gsm8k_llama8b_k10/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `trace_gsm8k_llama8b_k10/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `trace_gsm8k_llama8b_k10/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `trace_gsm8k_llama8b_k10/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `trace_gsm8k_llama8b_k10/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
- `trace_math500_qwenmath15b_k10/full/deem_deep_soft` seed=0: ValueError: method returned a non-finite or constant score
- `trace_math500_qwenmath15b_k10/full/deem_deep_soft` seed=1: ValueError: method returned a non-finite or constant score
- `trace_math500_qwenmath15b_k10/full/deem_deep_soft` seed=2: ValueError: method returned a non-finite or constant score
- `trace_math500_qwenmath15b_k10/full/deem_deep_soft` seed=3: ValueError: method returned a non-finite or constant score
- `trace_math500_qwenmath15b_k10/full/deem_deep_soft` seed=4: ValueError: method returned a non-finite or constant score
