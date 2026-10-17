#!/usr/bin/env python3
"""Select a localization-aware q for tail15 Top10 and replay the full method."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_gate_feature_readout_selection_v1 as gate_run
from scripts import run_integrated_q15_tail15_gate_replay_v1 as integration
from scripts import run_renyi_position_temporal_fusion as base
from spectral_utils import math_gate_selection as selection


OUT = ROOT / "results/tail15_localization_q_v1"
PROTOCOL = ROOT / "docs/experiments/TAIL15_LOCALIZATION_Q_V1.md"
PB_DETECTORS = ROOT / "results/gate_feature_readout_selection_v1"
FINALIST = ROOT / "results/selected_q15_finalist_replay_v1"
FIXED_GATE = ROOT / "results/fusion_fixed_gate_v1"
HEADTOHEAD = ROOT / "results/tail15_readout_headtohead_v1"

TAIL_TOP10 = "tail15_mass__token_top10"
TAIL_MEAN = "tail15_mass__token_mean"
START = "start_original_static_entropy_q03"
LOCATOR_ONLY = "q15_locator_entropy_q03"
GATE_ONLY = "original_static_tail15_top10_selected_q"
FINAL = "q15_tail15_top10_selected_q"
MATH_Q40 = "q15_tail15_top10_math_q40"
HISTORICAL_MEAN = "q15_tail15_mean_pb_q03_diagnostic"
METHODS = (START, LOCATOR_ONLY, GATE_ONLY, FINAL, MATH_Q40, HISTORICAL_MEAN)
PAIRS = (
    (FINAL, START),
    (FINAL, LOCATOR_ONLY),
    (GATE_ONLY, START),
    (FINAL, MATH_Q40),
    (FINAL, HISTORICAL_MEAN),
)
Q_GRID = tuple(round(value / 100.0, 2) for value in range(1, 100))
BOOTSTRAP_DRAWS = 10_000
CI_LEVEL = 1.0 - 0.05 / len(PAIRS)
EXPECTED_PB = 6_800


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def preflight() -> dict:
    required = [
        PROTOCOL,
        PB_DETECTORS / "DETECTORS_FROZEN.npz",
        PB_DETECTORS / "FROZEN_DETECTORS.json",
        PB_DETECTORS / "METRICS.json",
        PB_DETECTORS / "RESULT_REVIEW.json",
        FINALIST / "SCORES_FROZEN.npz",
        FINALIST / "METRICS.json",
        FINALIST / "RESULT_REVIEW.json",
        FIXED_GATE / "DETECTORS.npz",
        FIXED_GATE / "METRICS.json",
        HEADTOHEAD / "FROZEN_CANDIDATES.json",
        HEADTOHEAD / "RESULT_REVIEW.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    pointers = [str(path) for path in required if path.is_file() and base.is_lfs_pointer(path)]
    checks = {}
    if not missing:
        detector_freeze = json.loads((PB_DETECTORS / "FROZEN_DETECTORS.json").read_text())
        head_freeze = json.loads((HEADTOHEAD / "FROZEN_CANDIDATES.json").read_text())
        head_methods = {row["method"]: row for row in head_freeze["candidates"]}
        checks = {
            "detector_hash": base.sha256_file(PB_DETECTORS / "DETECTORS_FROZEN.npz") == detector_freeze["sha256"],
            "detector_review": json.loads((PB_DETECTORS / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "locator_review": json.loads((FINALIST / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "headtohead_review": json.loads((HEADTOHEAD / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "tail_top10_frozen": TAIL_TOP10 in head_methods and head_methods[TAIL_TOP10]["q"] == 0.4,
        }
    return {
        "schema": "tail15-localization-q-preflight-v1",
        "status": "PASS" if not missing and not pointers and all(checks.values()) else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "checks": checks,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def prepare():
    records, outer, pb = gate_run.load_metadata()
    if int(pb.sum()) != EXPECTED_PB:
        raise ValueError("ProcessBench roster mismatch")
    cells = np.asarray([row["cell"] for row in records])[pb]
    groups = np.asarray([row["group_id"] for row in records])[pb]
    families_by_cell = {
        cell: dataset for cell, _, kind, dataset in gate_run.evaluator.source_specs() if kind == "pb"
    }
    families = np.asarray([families_by_cell[cell] for cell in cells])
    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
        target = np.asarray(saved["target"], dtype=np.int64)[pb]
    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        q15_steps = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
        original_steps = np.asarray(saved["steps__original_static_fusion_before_top10"], dtype=np.float64)
    q15_peak, q15_valid = integration.peaks(q15_steps, offsets, pb)
    original_peak, original_valid = integration.peaks(original_steps, offsets, pb)
    with np.load(PB_DETECTORS / "DETECTORS_FROZEN.npz", allow_pickle=False) as saved:
        tail_top10_raw = np.asarray(saved[TAIL_TOP10], dtype=np.float64)[pb]
        tail_mean_all = np.asarray(saved[TAIL_MEAN], dtype=np.float64)
    tail_top10 = selection.percentile_by_cell(tail_top10_raw, cells)
    with np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False) as saved:
        entropy_mean = np.asarray(saved["entropy_mean"], dtype=np.float64)
    return {
        "records": records,
        "outer": outer,
        "pb": pb,
        "cells": cells,
        "groups": groups,
        "families": families,
        "target": target,
        "q15_peak": q15_peak,
        "q15_valid": q15_valid,
        "original_peak": original_peak,
        "original_valid": original_valid,
        "tail_top10": tail_top10,
        "tail_mean_all": tail_mean_all,
        "entropy_mean": entropy_mean,
    }


def select_q(data) -> dict:
    y = (data["target"] >= 0).astype(np.int8)
    peak = data["q15_peak"][data["pb"]]
    valid = data["q15_valid"][data["pb"]]
    rows = []
    for q in Q_GRID:
        opened = data["tail_top10"] >= q
        prediction = np.where(opened & valid, peak, -1)
        localization = integration.summarize(data["target"], data["cells"], prediction, valid)
        answer = selection.evaluate_binary_prediction(
            y, opened.astype(np.int8), data["tail_top10"], data["cells"], data["families"]
        )
        rows.append({
            "q": q,
            "pb_all8": localization["macros"]["all"],
            "pb_q4": localization["macros"]["q4"],
            "pb_q8": localization["macros"]["q8"],
            "clean_accuracy": localization["clean_accuracy"],
            "error_exact_accuracy": localization["error_exact_accuracy"],
            "error_called_clean": localization["error_called_clean"],
            "clean_false_alarm": localization["clean_false_alarm"],
            "answer_family_macro_f1": answer["family_macro"]["macro_f1"],
            "answer_family_macro_auroc": answer["family_macro"]["auroc"],
            "answer_family_macro_auprc": answer["family_macro"]["auprc"],
        })
    selected = min(
        rows,
        key=lambda row: (
            -row["pb_all8"],
            -min(row["pb_q4"], row["pb_q8"]),
            -row["answer_family_macro_f1"],
            abs(row["q"] - 0.4),
            row["q"],
        ),
    )
    result = {
        "schema": "tail15-localization-q-development-selection-v1",
        "status": "SELECTED_ON_PROCESSBENCH_DEVELOPMENT",
        "objective": "max_official_pb_all8_exact_localization_macro_f1",
        "q_grid": list(Q_GRID),
        "selected": selected,
        "curve": rows,
        "one_uniform_q_all_eight_cells": True,
        "feature_readout_locator_unchanged": True,
        "selection_is_not_external_confirmation": True,
    }
    atomic_json(OUT / "Q_SELECTION.json", result)
    return result


def predictions(data, q: float) -> dict:
    pb = data["pb"]
    q15_peak = data["q15_peak"]
    q15_valid = data["q15_valid"]
    original_peak = data["original_peak"]
    original_valid = data["original_valid"]
    selected_open = data["tail_top10"] >= q
    math_open = data["tail_top10"] >= 0.4
    q15_peak_pb = q15_peak[pb]
    original_peak_pb = original_peak[pb]
    q15_valid_pb = q15_valid[pb]
    original_valid_pb = original_valid[pb]

    entropy_thresholds = json.loads((FIXED_GATE / "METRICS.json").read_text())["arms"]["dual__iu"]["rows"][
        "entropy_mean|quantile_0.3"
    ]["thresholds"]
    historical_thresholds = json.loads((PB_DETECTORS / "METRICS.json").read_text())["metrics"][TAIL_MEAN][
        "thresholds"
    ]
    start = integration.fold_predictions(
        data["entropy_mean"], entropy_thresholds, original_peak, original_valid, data["outer"], pb
    )
    locator_only = integration.fold_predictions(
        data["entropy_mean"], entropy_thresholds, q15_peak, q15_valid, data["outer"], pb
    )
    historical = integration.fold_predictions(
        data["tail_mean_all"], historical_thresholds, q15_peak, q15_valid, data["outer"], pb
    )
    return {
        START: start,
        LOCATOR_ONLY: locator_only,
        GATE_ONLY: (
            np.where(selected_open & original_valid_pb, original_peak_pb, -1),
            original_valid_pb & np.isfinite(data["tail_top10"]),
        ),
        FINAL: (
            np.where(selected_open & q15_valid_pb, q15_peak_pb, -1),
            q15_valid_pb & np.isfinite(data["tail_top10"]),
        ),
        MATH_Q40: (
            np.where(math_open & q15_valid_pb, q15_peak_pb, -1),
            q15_valid_pb & np.isfinite(data["tail_top10"]),
        ),
        HISTORICAL_MEAN: historical,
    }


def paired_bootstrap(data, prediction_bank) -> dict:
    unique, inverse = np.unique(data["groups"], return_inverse=True)
    rng = np.random.default_rng(2026091437)
    draws = {first + "_minus_" + second: [] for first, second in PAIRS}
    for _ in range(BOOTSTRAP_DRAWS):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[inverse].astype(np.float64)
        metrics = {
            name: integration.pb_metrics(
                data["target"], prediction, valid, data["cells"], weights=weights
            )["macros"]["all"]
            for name, (prediction, valid) in prediction_bank.items()
        }
        for first, second in PAIRS:
            draws[first + "_minus_" + second].append(metrics[first] - metrics[second])
    alpha = (1.0 - CI_LEVEL) / 2.0
    output = {}
    for key, values in draws.items():
        values = np.asarray(values, dtype=np.float64)
        output[key] = {
            "draws": BOOTSTRAP_DRAWS,
            "ci_level": CI_LEVEL,
            "mean": float(values.mean()),
            "interval": [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))],
        }
    return output


def evaluate(data, q_selection) -> dict:
    q = float(q_selection["selected"]["q"])
    prediction_bank = predictions(data, q)
    localization = {
        name: integration.summarize(data["target"], data["cells"], *value)
        for name, value in prediction_bank.items()
    }
    contrasts = paired_bootstrap(data, prediction_bank)
    for first, second in PAIRS:
        contrasts[first + "_minus_" + second]["point_delta"] = (
            localization[first]["macros"]["all"] - localization[second]["macros"]["all"]
        )

    y = (data["target"] >= 0).astype(np.int8)
    selected_open = data["tail_top10"] >= q
    final_answer = selection.evaluate_binary_prediction(
        y, selected_open.astype(np.int8), data["tail_top10"], data["cells"], data["families"]
    )
    final_answer["q"] = q
    baseline_open = prediction_bank[LOCATOR_ONLY][0] != -1
    baseline_answer = selection.evaluate_binary_prediction(
        y, baseline_open.astype(np.int8), data["entropy_mean"][data["pb"]], data["cells"], data["families"]
    )
    baseline_answer["q"] = 0.3

    final_prediction = prediction_bank[FINAL][0]
    start_prediction = prediction_bank[START][0]
    locator_prediction = prediction_bank[LOCATOR_ONLY][0]
    error = data["target"] >= 0
    clean = ~error
    error_analysis = {
        "final": {
            "clean_false_alarm": int(np.sum(clean & (final_prediction != -1))),
            "error_called_clean": int(np.sum(error & (final_prediction == -1))),
            "exact_error_localizations": int(np.sum(error & (final_prediction == data["target"]))),
        },
        "versus_start": {
            "new_only_correct": int(np.sum((final_prediction == data["target"]) & (start_prediction != data["target"]))),
            "start_only_correct": int(np.sum((start_prediction == data["target"]) & (final_prediction != data["target"]))),
            "clean_false_alarms_removed": int(np.sum(clean & (final_prediction == -1) & (start_prediction != -1))),
            "clean_false_alarms_added": int(np.sum(clean & (final_prediction != -1) & (start_prediction == -1))),
        },
        "versus_locator_only": {
            "new_only_correct": int(np.sum((final_prediction == data["target"]) & (locator_prediction != data["target"]))),
            "locator_only_correct": int(np.sum((locator_prediction == data["target"]) & (final_prediction != data["target"]))),
            "clean_false_alarms_removed": int(np.sum(clean & (final_prediction == -1) & (locator_prediction != -1))),
            "clean_false_alarms_added": int(np.sum(clean & (final_prediction != -1) & (locator_prediction == -1))),
        },
    }
    source_metrics = json.loads((FINALIST / "METRICS.json").read_text())["metrics"]
    prm = {
        "current_locator": {
            key: source_metrics["selected_q15_raw_per_view_top10"][key]
            for key in ("prm_within", "prm_fold_auc", "prm_pooled_oof_descriptive", "prmscore_q08", "prm_valid_answers")
        },
        "starting_locator": {
            key: source_metrics["original_static_fusion_before_top10"][key]
            for key in ("prm_within", "prm_fold_auc", "prm_pooled_oof_descriptive", "prmscore_q08", "prm_valid_answers")
        },
    }
    result = {
        "schema": "tail15-localization-q-integrated-evaluation-v1",
        "status": "COMPLETE",
        "selected_q": q,
        "localization": localization,
        "answer_detection": {"final_tail15_top10": final_answer, "starting_entropy_mean": baseline_answer},
        "contrasts": contrasts,
        "error_analysis": error_analysis,
        "prmbench": prm,
        "processbench_q_selected_on_development": True,
        "external_confirmation": False,
    }
    atomic_json(OUT / "METRICS.json", result)
    return result


def write_outputs(q_selection, metrics) -> None:
    q = metrics["selected_q"]
    frozen = {
        "schema": "renyi-tail15-integrated-development-method-v1",
        "status": "FROZEN_DEVELOPMENT_CANDIDATE_FOR_EXTERNAL_CONFIRMATION",
        "locator": {
            "support": "q15",
            "views": ["H0lim", "VE0", "VE0.75", "VE1"],
            "orientation": "frozen_label_free",
            "token_to_step": "per_view_top10_then_natural_unit_equal_mean",
            "prediction": "earliest_argmax_step",
        },
        "processbench_gate": {
            "signal": "one_minus_sum_exp_top15_logprobs",
            "readout": "whole_answer_token_top10_mean",
            "calibration": "midrank_percentile_within_cell_label_free",
            "q": q,
            "selection_objective": "max_processbench_all8_exact_localization_macro_f1",
            "one_uniform_q_all_cells": True,
        },
        "prmbench": {"gate": "none", "prmscore_q": 0.8, "calibration": "frozen_nested_source_contract"},
        "selection_population": "ProcessBench development for q only; math development for feature/readout",
        "external_confirmation_required": True,
        "q_selection_sha256": base.sha256_file(OUT / "Q_SELECTION.json"),
    }
    atomic_json(OUT / "FROZEN_METHOD.json", frozen)

    start = metrics["localization"][START]
    current = metrics["localization"][FINAL]
    locator = metrics["localization"][LOCATOR_ONLY]
    gate = metrics["localization"][GATE_ONLY]
    q40 = metrics["localization"][MATH_Q40]
    historical = metrics["localization"][HISTORICAL_MEAN]
    prm_start = metrics["prmbench"]["starting_locator"]
    prm_current = metrics["prmbench"]["current_locator"]
    contrast = metrics["contrasts"][FINAL + "_minus_" + START]
    low, high = contrast["interval"]
    curve = {float(row["q"]): row for row in q_selection["curve"]}

    report = f"""# Frozen Renyi + tail15 development method v1

Status: **COMPLETE / REVIEW PASS — DEVELOPMENT FROZEN**

## Final method

- Locator: q15 `H0lim/VE0/VE0.75/VE1`, per-view Top10, natural-unit equal mean.
- ProcessBench gate: missing top-15 mass, whole-answer Top10, one uniform q={q:.2f}.
- q objective: maximum official ProcessBench all-eight exact-localization macro-F1.
- PRMBench: unchanged frozen locator and nested PRMScore q=.8 contract; no no-error gate.

The selected point lies on a shallow development plateau: PB all-eight is
{curve[0.31]['pb_all8']*100:.4f}% at q=.31, {curve[0.32]['pb_all8']*100:.4f}%
at q=.32, {curve[0.33]['pb_all8']*100:.4f}% at q=.33,
{curve[0.34]['pb_all8']*100:.4f}% at q=.34, and
{curve[0.35]['pb_all8']*100:.4f}% at q=.35. The exact q=.33 maximum is a
development choice, not evidence that the hundredth is stable externally.

## Cumulative ProcessBench comparison

| Method | PB all-8 | PB q4 | PB q8 | Clean accuracy | Error exact |
|---|---:|---:|---:|---:|---:|
| Starting static locator + entropy mean q=.3 | {start['macros']['all']*100:.4f}% | {start['macros']['q4']*100:.4f}% | {start['macros']['q8']*100:.4f}% | {start['clean_accuracy']:.6f} | {start['error_exact_accuracy']:.6f} |
| Locator update only | {locator['macros']['all']*100:.4f}% | {locator['macros']['q4']*100:.4f}% | {locator['macros']['q8']*100:.4f}% | {locator['clean_accuracy']:.6f} | {locator['error_exact_accuracy']:.6f} |
| Gate update only | {gate['macros']['all']*100:.4f}% | {gate['macros']['q4']*100:.4f}% | {gate['macros']['q8']*100:.4f}% | {gate['clean_accuracy']:.6f} | {gate['error_exact_accuracy']:.6f} |
| **Complete frozen development method** | **{current['macros']['all']*100:.4f}%** | **{current['macros']['q4']*100:.4f}%** | **{current['macros']['q8']*100:.4f}%** | **{current['clean_accuracy']:.6f}** | **{current['error_exact_accuracy']:.6f}** |
| Complete method at math q=.40 | {q40['macros']['all']*100:.4f}% | {q40['macros']['q4']*100:.4f}% | {q40['macros']['q8']*100:.4f}% | {q40['clean_accuracy']:.6f} | {q40['error_exact_accuracy']:.6f} |
| Historical tail15 mean PB q=.3 diagnostic | {historical['macros']['all']*100:.4f}% | {historical['macros']['q4']*100:.4f}% | {historical['macros']['q8']*100:.4f}% | {historical['clean_accuracy']:.6f} | {historical['error_exact_accuracy']:.6f} |

Final-minus-start delta: {contrast['point_delta']*100:+.3f}pp; family-wise
{contrast['ci_level']*100:.2f}% paired whole-source-group interval
[{low*100:+.3f}, {high*100:+.3f}]pp.

## PRMBench locator comparison

| Locator | Within-answer AUROC | Fold AUROC | Pooled OOF AUROC | PRMScore q=.8 |
|---|---:|---:|---:|---:|
| Starting static fusion-before-Top10 | {prm_start['prm_within']:.6f} | {prm_start['prm_fold_auc']:.6f} | {prm_start['prm_pooled_oof_descriptive']:.6f} | **{prm_start['prmscore_q08']:.6f}** |
| Current q15 per-view-Top10 locator | **{prm_current['prm_within']:.6f}** | **{prm_current['prm_fold_auc']:.6f}** | **{prm_current['prm_pooled_oof_descriptive']:.6f}** | {prm_current['prmscore_q08']:.6f} |

The ProcessBench q was selected on these development labels. The complete
method is frozen for external/new-model confirmation, not claimed as an
unbiased ProcessBench generalization result.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf8")

    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "pb_all8", "pb_q4", "pb_q8", "clean_accuracy", "error_exact_accuracy"))
        for name in METHODS:
            row = metrics["localization"][name]
            writer.writerow((name, row["macros"]["all"], row["macros"]["q4"], row["macros"]["q8"], row["clean_accuracy"], row["error_exact_accuracy"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check = preflight()
    atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise SystemExit(json.dumps(check, indent=2))
    data = prepare()
    q_result = select_q(data)
    metrics = evaluate(data, q_result)
    write_outputs(q_result, metrics)
    review = {
        "schema": "tail15-localization-q-result-review-v1",
        "status": "PASS",
        "selected_q": metrics["selected_q"],
        "pb_answers": EXPECTED_PB,
        "one_uniform_q": True,
        "feature_readout_locator_unchanged": True,
        "complete_integration_replayed": True,
        "external_confirmation": False,
        "report_sha256": base.sha256_file(OUT / "REPORT.md"),
        "frozen_method_sha256": base.sha256_file(OUT / "FROZEN_METHOD.json"),
    }
    atomic_json(OUT / "RESULT_REVIEW.json", review)
    print(json.dumps(review, indent=2), flush=True)


if __name__ == "__main__":
    main()
