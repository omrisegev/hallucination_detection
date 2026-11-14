# Test 2 tables (exact first-error accuracy on erroneous answers, %, 95% CI)

## ProcessBench (2,221 erroneous answers, macro over 4 subsets)

| cell (base model) | TELE16 + GAP | TELE16 argmax | PRM native | position + GAP | always step 0 | TELE16+GAP minus PRM |
|---|---|---|---|---|---|---|
| qwen_prm_7b (Qwen2.5-Math-7B-Instruct) | 38.47 [36.09, 40.76] | 39.48 | 69.54 | 8.40 | 13.50 | -31.07 [-33.78, -28.28] |
| qwen_prm_72b (Qwen2.5-Math-72B-Instruct) | 38.10 [35.62, 40.45] | 39.11 | 73.91 | 8.40 | 13.50 | -35.81 [-38.60, -33.13] |
| pure_prm (Qwen2.5-Math-7B) | 31.29 [29.19, 33.49] | 35.84 | 47.97 | 8.40 | 13.50 | -16.68 [-19.71, -13.34] |
| rlhflow_prm (Llama-3.1-8B-Instruct) | 34.29 [32.10, 36.67] | 37.06 | 28.52 | 8.40 | 13.50 | +5.77 [+2.90, +8.57] |

LLM critics, ProcessBench paper Tables 5 and 7, "error" accuracy, macro of 4 subsets (checked against
papers/extracted/processbench-...md): Qwen2.5-Math-7B-Instruct 11.2; Qwen2.5-Math-72B-Instruct 31.1;
Llama-3.1-8B-Instruct 37.7 (its correct-answer accuracy is about 6%); Qwen2.5-72B-Instruct 46.1; GPT-4o-0806 53.9.
The critic can answer -1 ("no error"); our rule cannot, because the error is known. The comparison favours us.

## PRMBench (6,035 erroneous answers)

| cell (base model) | TELE16 + GAP | TELE16 argmax | PRM native | position + GAP | always step 0 | TELE16+GAP minus PRM |
|---|---|---|---|---|---|---|
| qwen_prm_7b | 46.83 [45.53, 48.04] | 42.87 | 45.75 | 9.31 | 2.63 | +1.08 [-0.53, +2.72] |
| qwen_prm_72b | 45.55 [44.31, 46.76] | 43.76 | 50.56 | 9.31 | 2.63 | -5.00 [-6.58, -3.45] |
| pure_prm | 46.83 [45.60, 48.00] | 45.34 | 41.31 | 9.31 | 2.63 | +5.52 [+3.89, +7.16] |
| rlhflow_prm | 49.53 [48.24, 50.77] | 45.48 | 21.18 | 9.31 | 2.63 | +28.35 [+26.78, +29.86] |

Per category, qwen_prm_7b cell (TELE16 + GAP / PRM): circular 56.5/42.5, confidence 49.2/67.4,
counterfactual 43.4/57.9, deception 25.9/50.9, domain inconsistency 56.4/48.7, missing condition 33.3/29.9,
redundancy 52.5/16.8, step contradiction 57.0/52.0. Other cells: TEST2_PRMBENCH.csv.
