#!/usr/bin/env python3
"""Two-stage ProcessBench evaluation: detect an error, then localize it.

All detector and locator scores are label-free.  Within each repeated split,
calibration labels choose only the sequence-level operating threshold.  The
evaluation half remains untouched.  Shannon receives the identical treatment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

SUPPORT = Path(__file__).resolve().parent
LOC = SUPPORT / "localization"
sys.path[:0] = [str(SUPPORT), str(LOC)]
from optimize_localization import load_rows, score_candidates
from evidence_drop import EVIDENCE_FNS, evidence_drop_risk
from localization_metrics import NO_ERROR, processbench_f1, sla


def _argmax_locator(by_row):
    output = []
    for values in by_row:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        output.append(int(np.nanargmax(values)) if finite.any() else NO_ERROR)
    return np.asarray(output, dtype=int)


def _zscore(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-12)


def best_threshold(risk, locator, labels, indices):
    """Exact calibration-set F1 optimum over every possible flagged prefix."""
    risk = np.asarray(risk, float)[indices]
    locator = np.asarray(locator, int)[indices]
    labels = np.asarray(labels, int)[indices]
    order = np.argsort(-risk, kind="mergesort")
    r = risk[order]
    y = labels[order]
    p = locator[order]
    n_err = max(int(np.sum(y != NO_ERROR)), 1)
    n_clean = max(int(np.sum(y == NO_ERROR)), 1)
    hit = ((y != NO_ERROR) & (p == y)).astype(int)
    clean_flag = (y == NO_ERROR).astype(int)
    cum_hit = np.concatenate([[0], np.cumsum(hit)])
    cum_clean_flag = np.concatenate([[0], np.cumsum(clean_flag)])
    acc_err = cum_hit / n_err
    acc_clean = (n_clean - cum_clean_flag) / n_clean
    denom = acc_err + acc_clean
    f1 = np.divide(2 * acc_err * acc_clean, denom,
                   out=np.zeros_like(denom, dtype=float), where=denom > 0)
    k = int(np.flatnonzero(f1 == np.max(f1))[0])
    if k == 0:
        tau = np.inf
    elif k == len(r):
        tau = -np.inf
    else:
        tau = float((r[k - 1] + r[k]) / 2.0)
    return tau, float(f1[k])


def evaluate_two_stage(risk, locator, labels, n_splits=100, seed=0):
    labels = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    output = {key: [] for key in (
        "f1", "acc_erroneous", "acc_correct", "sla", "sla_tol1", "tau", "cal_f1"
    )}
    for _ in range(n_splits):
        perm = rng.permutation(len(labels))
        cal, ev = perm[:len(labels) // 2], perm[len(labels) // 2:]
        tau, cal_f1 = best_threshold(risk, locator, labels, cal)
        pred = np.where(np.asarray(risk)[ev] > tau, np.asarray(locator)[ev], NO_ERROR)
        scored = processbench_f1(pred, labels[ev])
        output["f1"].append(scored["f1"])
        output["acc_erroneous"].append(scored["acc_erroneous"])
        output["acc_correct"].append(scored["acc_correct"])
        output["sla"].append(sla(pred, labels[ev], 0))
        output["sla_tol1"].append(sla(pred, labels[ev], 1))
        output["tau"].append(tau)
        output["cal_f1"].append(cal_f1)
    result = {"n": len(labels), "n_splits": n_splits}
    for key, values in output.items():
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        result[key] = float(finite.mean()) if len(finite) else float("nan")
        result[key + "_sd"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
    return result


def score_two_stage(rows):
    step_scores, fit_diag = score_candidates(rows)
    labels_deferred = True

    detectors = {
        "shannon_sequence": np.asarray([
            evidence_drop_risk(EVIDENCE_FNS["shannon"](row, 20), M=5, ema_span=5)
            for row in rows
        ]),
        "pos_core_sequence": np.asarray([np.nanmax(v) for v in step_scores["pos_core_max"]]),
        "pos_full_sequence": np.asarray([np.nanmax(v) for v in step_scores["pos_full_max"]]),
        "pos_mode_sequence": np.asarray([np.nanmax(v) for v in step_scores["pos_core_mode_max"]]),
    }
    detectors["hybrid_core75_shannon25"] = (
        0.75 * _zscore(detectors["pos_core_sequence"])
        + 0.25 * _zscore(detectors["shannon_sequence"])
    )
    detectors["hybrid_core50_shannon50"] = (
        0.50 * _zscore(detectors["pos_core_sequence"])
        + 0.50 * _zscore(detectors["shannon_sequence"])
    )

    locators = {
        name: _argmax_locator(step_scores[source]) for name, source in {
            "shannon_locator": "shannon_drop",
            "core_level_locator": "pos_core_max",
            "core_rise_locator": "pos_core_rise",
            "onset_locator": "pos_onset_max",
            "mixed_locator": "pos_mixed_max",
            "mode_locator": "pos_core_mode_max",
            "mode_blend_locator": "pos_core_mode_blend25",
        }.items()
    }
    return detectors, locators, fit_diag, labels_deferred


def run(path, out_dir):
    rows = load_rows(path)
    detectors, locators, fit_diag, frozen = score_two_stage(rows)

    # Open labels only after the score-generating stage has returned.
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    records = []
    for detector_name, risk in detectors.items():
        for locator_name, locator in locators.items():
            records.append({
                "detector": detector_name, "locator": locator_name,
                "method": detector_name + "__" + locator_name,
                **evaluate_two_stage(risk, locator, labels),
            })
    records.sort(key=lambda row: row["f1"], reverse=True)
    subset = os.path.basename(path).removeprefix("processbench_").removesuffix(".pkl")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{subset}__two_stage.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(records[0]))
        writer.writeheader(); writer.writerows(records)
    with open(os.path.join(out_dir, f"{subset}__two_stage_diag.json"), "w") as f:
        json.dump({"scores_frozen_before_labels": frozen, "fit": fit_diag},
                  f, indent=2, default=str)
    print("\n", subset)
    for row in records[:12]:
        print(f"{row['method']:64s} F1={100*row['f1']:6.2f} "
              f"err={100*row['acc_erroneous']:6.2f} clean={100*row['acc_correct']:6.2f} "
              f"SLA1={100*row['sla_tol1']:6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subsets", default=None)
    args = parser.parse_args()
    wanted = set(args.subsets.split(",")) if args.subsets else None
    for name in sorted(os.listdir(args.data_dir)):
        if not (name.startswith("processbench_") and name.endswith(".pkl")):
            continue
        subset = name.removeprefix("processbench_").removesuffix(".pkl")
        if wanted is not None and subset not in wanted:
            continue
        run(os.path.join(args.data_dir, name), args.out_dir)


if __name__ == "__main__":
    main()
