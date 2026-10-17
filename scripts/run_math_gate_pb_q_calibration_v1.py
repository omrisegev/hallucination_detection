#!/usr/bin/env python3
"""PB-development q calibration for the math-selected answer-gate representation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_gate_feature_readout_selection_v1 as gate_run
from scripts import run_integrated_q15_tail15_gate_replay_v1 as old_integration
from scripts import run_renyi_position_temporal_fusion as base
from spectral_utils import math_gate_selection as selection


SOURCE = ROOT / "results/math_gate_processbench_transfer_v1"
MATH = ROOT / "results/math_gate_development_v1"
OUT = ROOT / "results/math_gate_pb_q_calibration_v1"
METHODS = (
    "original_static_entropy",
    "q15_finalist_entropy",
    "integrated_q15_tail15",
    "math_frozen_q45",
    "math_selected_pb_calibrated_q",
)
PAIRS = (
    ("math_selected_pb_calibrated_q", "integrated_q15_tail15"),
    ("math_selected_pb_calibrated_q", "q15_finalist_entropy"),
    ("math_selected_pb_calibrated_q", "original_static_entropy"),
    ("math_selected_pb_calibrated_q", "math_frozen_q45"),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        SOURCE / "RESULT_REVIEW.json",
        SOURCE / "METRICS.json",
        SOURCE / "PREDICTIONS.npz",
        SOURCE / "DETECTOR_FREEZE.json",
        MATH / "FROZEN_GATE.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    review = None if missing else json.loads(required[0].read_text(encoding="utf8"))["status"]
    preflight = {
        "schema": "math-gate-pb-q-calibration-preflight-v1",
        "status": "PASS" if not missing and review == "PASS" else "BLOCKED",
        "missing": missing,
        "source_review": review,
        "representation_and_fusion_frozen": True,
        "only_q_may_change": True,
    }
    base.atomic_json(OUT / "PREFLIGHT.json", preflight)
    if preflight["status"] != "PASS":
        raise SystemExit(2)

    gate_run.configure()
    records, _, pb = gate_run.load_metadata()
    pb_indices = np.flatnonzero(pb)
    cells = np.asarray([records[index]["cell"] for index in pb_indices])
    groups = np.asarray([records[index]["group_id"] for index in pb_indices])
    family_by_cell = {
        cell: dataset
        for cell, _, kind, dataset in gate_run.evaluator.source_specs()
        if kind == "pb"
    }
    families = np.asarray([family_by_cell[cell] for cell in cells])
    with np.load(SOURCE / "PREDICTIONS.npz", allow_pickle=False) as saved:
        gate_score = np.asarray(saved["gate_score"], dtype=np.float64)
        target = np.asarray(saved["target"], dtype=np.int64)
        peak = np.asarray(saved["q15_peak"], dtype=np.int64)
        prior_predictions = {
            "original_static_entropy": np.asarray(saved["original_static_entropy"], dtype=np.int64),
            "q15_finalist_entropy": np.asarray(saved["q15_finalist_entropy"], dtype=np.int64),
            "integrated_q15_tail15": np.asarray(saved["integrated_q15_tail15"], dtype=np.int64),
            "math_frozen_q45": np.asarray(saved["math_frozen_gate"], dtype=np.int64),
        }
    y_error = (target >= 0).astype(np.int8)
    q_selection = selection.select_q(y_error, gate_score, cells, families)
    selected_q = float(q_selection["selected_q"])
    calibrated = np.where(gate_score >= selected_q, peak, -1)
    predictions = {**prior_predictions, "math_selected_pb_calibrated_q": calibrated}
    valid = np.ones(len(target), dtype=bool)
    localization = {
        name: old_integration.summarize(target, cells, prediction, valid)
        for name, prediction in predictions.items()
    }
    localization_curve = []
    for q in selection.Q_GRID:
        prediction = np.where(gate_score >= q, peak, -1)
        row = old_integration.summarize(target, cells, prediction, valid)
        localization_curve.append({
            "q": q,
            "pb_all8": row["macros"]["all"],
            "pb_q4": row["macros"]["q4"],
            "pb_q8": row["macros"]["q8"],
            "clean_accuracy": row["clean_accuracy"],
            "error_exact_accuracy": row["error_exact_accuracy"],
        })
    paired = {name: (prediction, valid) for name, prediction in predictions.items()}
    contrasts = old_integration.paired_bootstrap(target, cells, groups, paired, PAIRS)
    for first, second in PAIRS:
        key = first + "_minus_" + second
        contrasts[key]["point_delta"] = (
            localization[first]["macros"]["all"] - localization[second]["macros"]["all"]
        )

    frozen = json.loads((MATH / "FROZEN_GATE.json").read_text(encoding="utf8"))
    candidate = {
        "schema": "math-selected-pb-q-calibrated-gate-v1",
        "status": "DEVELOPMENT_CANDIDATE_FROZEN_FOR_EXTERNAL_TRANSFER",
        "arm": frozen["arm"],
        "features": frozen["features"],
        "percentile_calibration": frozen["percentile_calibration"],
        "q": selected_q,
        "q_selection_objective": "ProcessBench family-first answer-level clean/error macro-F1",
        "feature_and_fusion_selection_population": "15 historical math/reasoning cells",
        "q_calibration_population": "8 ProcessBench development cells",
        "one_common_q_all_processbench_cells": True,
        "not_external_confirmation": True,
        "math_frozen_gate_sha256": base.sha256_file(MATH / "FROZEN_GATE.json"),
        "pb_detector_freeze_sha256": base.sha256_file(SOURCE / "DETECTOR_FREEZE.json"),
    }
    base.atomic_json(OUT / "FROZEN_CANDIDATE.json", candidate)
    error = target >= 0
    clean = ~error
    frozen_open = prior_predictions["math_frozen_q45"] != -1
    calibrated_open = calibrated != -1
    metrics = {
        "schema": "math-gate-pb-q-calibration-metrics-v1",
        "status": "COMPLETE",
        "selected_q": selected_q,
        "answer_detection_q_selection": q_selection,
        "localization": localization,
        "localization_curve_diagnostic_not_selection_objective": localization_curve,
        "contrasts": contrasts,
        "q45_to_selected_q": {
            "additional_answers_opened": int(np.sum(calibrated_open & ~frozen_open)),
            "answers_newly_closed": int(np.sum(~calibrated_open & frozen_open)),
            "additional_clean_false_alarms": int(np.sum(clean & calibrated_open & ~frozen_open)),
            "erroneous_answers_reopened": int(np.sum(error & calibrated_open & ~frozen_open)),
            "erroneous_exact_localizations_recovered": int(np.sum(error & (calibrated == target) & (prior_predictions["math_frozen_q45"] != target))),
            "clean_decisions_lost": int(np.sum(clean & (prior_predictions["math_frozen_q45"] == target) & (calibrated != target))),
        },
        "representation_or_fusion_reselected_on_processbench": False,
        "q_selected_on_processbench": True,
        "development_not_external_confirmation": True,
    }
    base.atomic_json(OUT / "METRICS.json", metrics)
    base.atomic_json(OUT / "RESULT_REVIEW.json", {
        "schema": "math-gate-pb-q-calibration-review-v1",
        "status": "PASS",
        "selected_q": selected_q,
        "selection_objective": "answer-level macro-F1 only",
        "uniform_q": True,
        "features_unchanged": candidate["features"] == frozen["features"],
        "fusion_unchanged": candidate["arm"] == frozen["arm"],
        "external_confirmation": False,
    })
    print("selected_q", selected_q)
    print("method,pb_all8,pb_q4,pb_q8")
    for name in METHODS:
        row = localization[name]["macros"]
        print(f"{name},{row['all']},{row['q4']},{row['q8']}")


if __name__ == "__main__":
    main()
