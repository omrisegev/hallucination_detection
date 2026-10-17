#!/usr/bin/env python3
"""
Result evaluation script for MATH dataset.

Computes selective prediction metrics given a threshold tau.
"""

import json
import os
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict


def eval_accuracy(results):
    """Compute accuracy: (correct_count, total_count, accuracy)"""
    if not results:
        return 0, 0, 0.0
    preds = [1 if res["is_correct"] else 0 for res in results]
    return sum(preds), len(preds), sum(preds) / len(preds)


def bootstrap_confidence_interval(data, num_bootstrap_samples=10000, confidence_level=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    data = np.array(data).astype(float)
    if len(data) == 0:
        return 0.0, 0.0

    indices = np.random.randint(0, len(data), size=(num_bootstrap_samples, len(data)))
    samples = data[indices]
    bootstrap_means = np.mean(samples, axis=1)

    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    print(f"    -> 95% Bootstrap CI: ({ci_lower*100:.2f}%, {ci_upper*100:.2f}%)")
    return ci_lower, ci_upper


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate selective prediction results for MATH")
    parser.add_argument("--tau", type=float, required=True, help="Risk threshold value (from calibration)")
    parser.add_argument("--result_file", type=str, required=True, help="Path to evaluation results (.json)")
    args = parser.parse_args()

    file_path = args.result_file
    print(f"Loading results from: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    with open(file_path, 'r') as f:
        data = json.load(f)

    tau = args.tau
    print(f"Using Risk Threshold (tau): {tau:.6f}")
    print("(Policy: Accept if Score <= tau, Reject if Score > tau)")

    uncertain_results = []
    certain_results = []
    level_stats = defaultdict(lambda: {"total": 0, "accepted": 0, "correct_accepted": 0})

    for d in data:
        score = d["risk_score"]
        level = d.get("level", "Unknown")
        if level is None:
            level = "Unknown"

        level_stats[level]["total"] += 1

        if score <= tau:
            certain_results.append(d)
            level_stats[level]["accepted"] += 1
            if d["is_correct"]:
                level_stats[level]["correct_accepted"] += 1
        else:
            uncertain_results.append(d)

    print('\n' + '=' * 50)
    print(' OVERALL PERFORMANCE (No Rejection)')
    print('=' * 50)
    correct_num, total_num, total_acc = eval_accuracy(data)
    wrong_num = total_num - correct_num

    print(f"Total Samples:        {total_num}")
    print(f"Overall Accuracy:     {total_acc*100:.2f}% ({correct_num}/{total_num})")

    print('\n' + '=' * 50)
    print(f' ACCEPTED GROUP (Risk <= {tau:.6f})')
    print('=' * 50)
    certain_correct, certain_total, certain_acc = eval_accuracy(certain_results)
    coverage = certain_total / total_num if total_num > 0 else 0

    print(f"Accepted Samples:     {certain_total}")
    print(f"Coverage:             {coverage*100:.2f}% (Samples Retained)")
    print(f"Selective Accuracy:   {certain_acc*100:.2f}% ({certain_correct}/{certain_total})")

    if certain_results:
        certain_preds = [1 if res["is_correct"] else 0 for res in certain_results]
        bootstrap_confidence_interval(certain_preds)

    print('\n' + '=' * 50)
    print(f' REJECTED GROUP (Risk > {tau:.6f})')
    print('=' * 50)
    uncertain_correct, uncertain_total, uncertain_acc = eval_accuracy(uncertain_results)

    print(f"Rejected Samples:     {uncertain_total}")
    print(f"Rejection Rate:       {(uncertain_total/total_num)*100:.2f}%")
    print(f"Accuracy in Rejected: {uncertain_acc*100:.2f}% (Should be low)")

    print('\n' + '=' * 50)
    print(' KEY RISK METRICS (For Paper)')
    print('=' * 50)

    conditional_risk = 1.0 - certain_acc
    print(f"Conditional Risk:     {conditional_risk*100:.2f}% (Expected < Alpha)")

    accepted_errors = certain_total - certain_correct
    type1_error = accepted_errors / wrong_num if wrong_num > 0 else 0
    print(f"Type I Error (FNR):   {type1_error*100:.2f}% (Errors Accepted / Total Errors)")
    print(f"Error Recall:         {(1.0 - type1_error)*100:.2f}% (Target: High)")

    rejected_corrects = uncertain_correct
    type2_error = rejected_corrects / correct_num if correct_num > 0 else 0
    print(f"Type II Error (FPR):  {type2_error*100:.2f}% (Corrects Rejected / Total Corrects)")

    print('\n' + '=' * 50)
    print(' BREAKDOWN BY DIFFICULTY LEVEL (MATH)')
    print('=' * 50)
    print(f"{'Level':<10} | {'Total':<8} | {'Coverage':<10} | {'Sel. Acc':<10}")
    print("-" * 46)

    sorted_levels = sorted(level_stats.keys(), key=lambda x: str(x) if x != "Unknown" else "z")

    for lvl in sorted_levels:
        stats = level_stats[lvl]
        tot = stats['total']
        if tot == 0:
            continue

        acc_n = stats['accepted']
        corr_acc_n = stats['correct_accepted']

        cov = acc_n / tot
        sel_acc = corr_acc_n / acc_n if acc_n > 0 else 0.0

        print(f"{str(lvl):<10} | {tot:<8} | {cov*100:6.2f}%    | {sel_acc*100:6.2f}%")

    print('\n' + '-' * 50)
    if total_acc > 0:
        improvement = certain_acc - total_acc
        print(f"Summary: By rejecting {(uncertain_total/total_num)*100:.1f}% of samples, accuracy improved by {improvement*100:.2f}%.")
    print('-' * 50)
