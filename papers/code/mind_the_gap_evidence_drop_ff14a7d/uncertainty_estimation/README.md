# Uncertainty Estimation for LLM Reasoning

Uncertainty estimation methods for selective prediction in LLM mathematical reasoning.
Tested on Qwen3-8B with MATH and GSM8K datasets.

## Overview

This repository provides two uncertainty estimation methods:

### LN_E (Length-Normalized Entropy)
A baseline method that computes risk as the negative mean of token log-probabilities.
- Simple and efficient
- Lower scores indicate higher uncertainty
- Direct arithmetic mean of token logprobs

### LN_E_drop (Running Mean + EMA + Drop Detection)
An enhanced method that captures quality drops during generation:
1. Computes running mean of token confidences over time
2. Applies EMA smoothing to reduce noise
3. Detects significant quality drops (tokens that lower overall confidence)
4. Averages the K worst drops as the final risk score

## Repository Structure

```
uncertainty_estimation/
├── utils/
│   ├── metrics.py          # Core risk calculation functions
│   └── vllm_utils.py       # vLLM setup and distributed inference
├── math/
│   ├── ln_e.py             # LN_E evaluation for MATH dataset
│   ├── ln_e_drop.py        # LN_E_drop evaluation for MATH dataset
│   └── eval.py             # Selective prediction evaluation for MATH
├── gsm8k/
│   ├── ln_e.py             # LN_E evaluation for GSM8K dataset
│   ├── ln_e_drop.py        # LN_E_drop evaluation for GSM8K dataset
│   └── eval.py             # Selective prediction evaluation for GSM8K
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### MATH Dataset

**LN_E (Baseline):**
```bash
python math/ln_e.py \
    --model /path/to/Qwen3-8B \
    --dataset_path /path/to/math_eval.jsonl \
    --output_dir ./results/math \
    --tensor_parallel_size 1
```

**LN_E_drop (Enhanced):**
```bash
python math/ln_e_drop.py \
    --model /path/to/Qwen3-8B \
    --dataset_path /path/to/math_eval.jsonl \
    --output_dir ./results/math \
    --ema_span 5.0 \
    --drop_k 5 \
    --tensor_parallel_size 1
```

**Evaluate with threshold:**
```bash
python math/eval.py \
    --tau 0.5 \
    --result_file ./results/math/eval_LN_E_drop_final_Qwen3-8B.json
```

### GSM8K Dataset

**LN_E (Baseline):**
```bash
python gsm8k/ln_e.py \
    --model /path/to/Qwen3-8B \
    --dataset_path /path/to/gsm8k_eval.jsonl \
    --output_dir ./results/gsm8k \
    --tensor_parallel_size 1
```

**LN_E_drop (Enhanced):**
```bash
python gsm8k/ln_e_drop.py \
    --model /path/to/Qwen3-8B \
    --dataset_path /path/to/gsm8k_eval.jsonl \
    --output_dir ./results/gsm8k \
    --ema_span 5.0 \
    --drop_k 5 \
    --tensor_parallel_size 1
```

**Evaluate with threshold:**
```bash
python gsm8k/eval.py \
    --tau 0.5 \
    --result_file ./results/gsm8k/eval_LN_E_drop_final_Qwen3-8B.json \
    --dataset_name GSM8K
```

## Parameters

### Inference Parameters
- `--model`: Path to the language model
- `--dataset_path`: Path to dataset (.jsonl format)
- `--output_dir`: Directory for output files
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism
- `--gpu_memory_utilization`: GPU memory fraction (default: 0.6-0.65)

### LN_E_drop Specific Parameters
- `--ema_span`: EMA smoothing window size (default: 5.0)
- `--drop_k`: Number of worst drops to average (default: 5)
- `--logprobs`: Number of top logprobs to retrieve (default: 20)

### Evaluation Parameters
- `--tau`: Risk threshold for selective prediction (calibrated on validation set)
- `--result_file`: Path to evaluation results JSON

## Output Files

Each method generates three files:
- `{prefix}_LN_E_raw_{model}.json`: Raw inference results with logprob curves
- `{prefix}_LN_E_final_{model}.json`: Final results with risk scores
- `{prefix}_LN_E_final_{model}_D0_scores.json`: Risk scores for incorrect predictions (for calibration)

Prefix is determined by dataset filename:
- `cal_*` for calibration datasets
- `eval_*` for evaluation datasets

## Key Metrics (from eval.py)

- **Coverage**: Percentage of samples accepted (not rejected)
- **Selective Accuracy**: Accuracy on accepted samples
- **Conditional Risk**: Error rate on accepted samples (should be < alpha)
- **Type I Error (FNR)**: Percentage of errors that were accepted
- **Type II Error (FPR)**: Percentage of correct answers that were rejected

## Citation

If you use this code, please cite:
```

```

## License

MIT License
