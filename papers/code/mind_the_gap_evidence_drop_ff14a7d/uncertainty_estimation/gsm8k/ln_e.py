#!/usr/bin/env python3
"""
LN_E Evaluation for GSM8K Dataset.

Baseline method using Length-Normalized Entropy (arithmetic mean of logprobs).
"""

import json
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import get_logprob_curve, calculate_risk_baseline_mean
from utils.vllm_utils import add_vllm_tp_args, llm_common_kwargs, setup_vllm_distributed_and_tp

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass


def format_gsm8k_prompt(question):
    """Format GSM8K dataset prompt with few-shot example."""
    few_shot = (
        "Question: There are 5 birds on a tree. 2 fly away. How many are left?\n"
        "Answer: There were originally 5 birds. 2 flew away. So 5 - 2 = 3.\n"
        "#### 3\n\n"
    )
    return f"{few_shot}Question: {question}\nAnswer:"


def extract_gsm8k_answer(text):
    """Extract answer after #### marker."""
    if text is None:
        return None
    if "####" not in text:
        return None
    return text.split("####")[-1].strip()


def is_gsm8k_correct(pred_str, gt_str):
    """Check if two GSM8K answers are numerically equivalent."""
    if pred_str is None or gt_str is None:
        return False

    def clean_num(s):
        s = str(s).strip().replace(',', '')
        try:
            return float(s)
        except:
            return None

    pred_val = clean_num(pred_str)
    gt_val = clean_num(gt_str)

    if pred_val is None or gt_val is None:
        return False

    return abs(pred_val - gt_val) < 1e-6


def truncate_response(text, logprobs_seq, tokenizer):
    """Truncate text and logprobs at #### answer."""
    if text is None:
        return text, logprobs_seq, None

    pred_ans = extract_gsm8k_answer(text)
    if pred_ans is None:
        return text, logprobs_seq, None

    idx = text.rfind("####")
    if idx == -1:
        return text, logprobs_seq, pred_ans

    truncated_text = text[:idx]
    tokens = tokenizer.encode(truncated_text, add_special_tokens=False)
    valid_len = min(len(tokens), len(logprobs_seq))
    return truncated_text, logprobs_seq[:valid_len], pred_ans


def run_inference(args):
    """Run inference to collect logprob curves."""
    print(f"\n[Mode: INFERENCE] Loading Model: {args.model}")

    tp = setup_vllm_distributed_and_tp(args)
    llm = LLM(**llm_common_kwargs(args, tp))
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=1024,
        logprobs=20,
        stop_token_ids=[tokenizer.eos_token_id],
        stop=["Question:", "Answer:", "\n\n\n"]
    )

    print(f"Loading Dataset: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at {args.dataset_path}")
        exit(1)

    data_list = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            data_list = json.loads(content)
        else:
            for line in content.split('\n'):
                if line.strip():
                    data_list.append(json.loads(line))

    prompts = [format_gsm8k_prompt(item['question']) for item in data_list]
    print(f"Running inference on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    raw_data = []
    print("Extracting LOGPROB curves...")
    for i, output in enumerate(tqdm(outputs)):
        item = data_list[i]
        full_text = output.outputs[0].text
        full_logprobs = output.outputs[0].logprobs

        trunc_text, trunc_logprobs, pred_ans = truncate_response(full_text, full_logprobs, tokenizer)
        if pred_ans is None:
            pred_ans = "N/A"

        gt_full = item.get('answer', '')
        gt_ans = extract_gsm8k_answer(gt_full)
        if gt_ans is None:
            gt_ans = "N/A"

        is_correct = is_gsm8k_correct(pred_ans, gt_ans)
        prob_curve = get_logprob_curve(trunc_logprobs)

        raw_data.append({
            "question": item['question'],
            "pred_answer": pred_ans,
            "gt_answer": gt_ans,
            "is_correct": int(is_correct),
            "raw_curve": prob_curve,
            "generated_text": trunc_text,
            "metric_type": "LN_E"
        })

    print(f"Saving RAW data to: {args.raw_data_path}")
    with open(args.raw_data_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)

    return raw_data


def run_analysis(raw_data, args):
    """Analyze raw data to compute risk scores."""
    print(f"\n[Mode: ANALYSIS] Calculating Risk using [LN_E]...")

    final_results = []
    incorrect_risks = []

    for item in tqdm(raw_data):
        raw_curve = item['raw_curve']
        risk_score = calculate_risk_baseline_mean(raw_curve)

        result_item = item.copy()
        result_item['risk_score'] = risk_score
        if 'raw_curve' in result_item:
            del result_item['raw_curve']

        final_results.append(result_item)
        if not result_item['is_correct']:
            incorrect_risks.append(risk_score)

    total = len(final_results)
    correct = sum(1 for r in final_results if r['is_correct'])
    acc = correct / total if total > 0 else 0

    print("=" * 40)
    print(f"GSM8K LN_E Stats:")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {acc*100:.2f}% ({correct}/{total})")
    print(f"Wrong Samples (D_0): {len(incorrect_risks)}")
    print("=" * 40)

    print(f"Saving FINAL results to: {args.save_path}")
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    d0_path = args.save_path.replace(".json", "_D0_scores.json")
    print(f"Saving D_0 Scores to: {d0_path}")
    with open(d0_path, 'w') as f:
        json.dump(incorrect_risks, f)


if __name__ == "__main__":
    parser = ArgumentParser(description="LN_E Evaluation for GSM8K Dataset")
    parser.add_argument('--model', type=str, required=True, help="Path to the model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to GSM8K dataset (.jsonl)")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    parser.add_argument("--use_stored_data", action="store_true", help="Reuse previously stored raw curves")
    parser.add_argument("--ema_span", type=float, default=5.0, help="Ignored (for compatibility)")
    parser.add_argument("--drop_k", type=int, default=5, help="Ignored (for compatibility)")
    add_vllm_tp_args(parser, default_gpu_memory_utilization=0.6)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_name = args.model.rstrip('/').split('/')[-1]
    base_name = os.path.basename(args.dataset_path)

    if "calibration" in base_name:
        prefix = "cal"
    elif "evaluation" in base_name:
        prefix = "eval"
    else:
        prefix = "gsm8k"

    args.raw_data_path = os.path.join(args.output_dir, f"{prefix}_LN_E_raw_{model_name}.json")
    args.save_path = os.path.join(args.output_dir, f"{prefix}_LN_E_final_{model_name}.json")

    raw_data = []
    if args.use_stored_data and os.path.exists(args.raw_data_path):
        print(f"Loading stored raw data from: {args.raw_data_path}")
        with open(args.raw_data_path, 'r') as f:
            raw_data = json.load(f)
    else:
        raw_data = run_inference(args)

    run_analysis(raw_data, args)
