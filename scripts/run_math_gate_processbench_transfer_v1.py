#!/usr/bin/env python3
"""Transfer the math-frozen answer gate to the frozen ProcessBench locator."""
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
from spectral_utils import gate_feature_readout as feature_model
from spectral_utils import math_gate_selection as selection


OUT = ROOT / "results/math_gate_processbench_transfer_v1"
MATH = ROOT / "results/math_gate_development_v1"
FINALIST = ROOT / "results/selected_q15_finalist_replay_v1"
FIXED_GATE = ROOT / "results/fusion_fixed_gate_v1"
OLD_GATE = ROOT / "results/gate_feature_readout_selection_v1"
EXPECTED_PB = 6_800
EXPECTED_STEPS = 145_597
METHODS = (
    "original_static_entropy",
    "q15_finalist_entropy",
    "integrated_q15_tail15",
    "math_frozen_gate",
)
PAIRS = (
    ("math_frozen_gate", "integrated_q15_tail15"),
    ("math_frozen_gate", "q15_finalist_entropy"),
    ("math_frozen_gate", "original_static_entropy"),
)


def verify_math_freeze() -> dict:
    record = json.loads((MATH / "FREEZE_RECORD.json").read_text(encoding="utf8"))
    frozen_path = Path(record["frozen_gate"])
    if record["status"] != "PASS" or base.sha256_file(frozen_path) != record["frozen_gate_sha256"]:
        raise ValueError("math gate freeze record mismatch")
    frozen = json.loads(frozen_path.read_text(encoding="utf8"))
    if frozen["status"] != "FROZEN_BEFORE_PROCESSBENCH" or frozen["processbench_labels_seen"]:
        raise ValueError("math gate was not frozen before ProcessBench")
    return frozen


def preflight() -> dict:
    gate_run.configure()
    frozen = verify_math_freeze()
    paths = [
        MATH / "FREEZE_RECORD.json",
        MATH / "FROZEN_GATE.json",
        FINALIST / "SCORES_FROZEN.npz",
        FINALIST / "RESULT_REVIEW.json",
        FIXED_GATE / "DETECTORS.npz",
        FIXED_GATE / "METRICS.json",
        OLD_GATE / "DETECTORS_FROZEN.npz",
        OLD_GATE / "METRICS.json",
    ]
    paths.extend(path for _, path, kind, _ in gate_run.evaluator.source_specs() if kind == "pb")
    missing = [str(path) for path in paths if not Path(path).is_file()]
    pointers = [str(path) for path in paths if Path(path).is_file() and base.is_lfs_pointer(Path(path))]
    selected = list(frozen["features"])
    invalid = [name for name in selected if name not in feature_model.TEMPORAL_METHODS]
    return {
        "schema": "math-gate-processbench-transfer-preflight-v1",
        "status": "PASS" if not missing and not pointers and not invalid else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "invalid_selected_features": invalid,
        "math_frozen_arm": frozen["arm"],
        "math_frozen_features": selected,
        "math_frozen_q": frozen["q"],
        "python": sys.executable,
    }


def extract_detectors(records, pb, frozen) -> dict:
    pb_indices = np.flatnonzero(pb)
    local_index = {int(global_index): position for position, global_index in enumerate(pb_indices)}
    selected = list(frozen["features"])
    matrix = np.full((len(pb_indices), len(selected)), np.nan, dtype=np.float64)
    cells = np.asarray([records[index]["cell"] for index in pb_indices])
    uids = np.asarray([records[index]["uid"] for index in pb_indices])
    source_hashes = {}
    for cell, path, kind, dataset in gate_run.evaluator.source_specs():
        if kind != "pb":
            continue
        path = Path(path)
        print("[hash]", cell, flush=True)
        source_hashes[str(path.resolve())] = base.sha256_file(path)
        indices = [int(index) for index in pb_indices if records[index]["cell"] == cell]
        rows = gate_run.evaluator.old._source_row_map(
            gate_run.evaluator.old.load_pickle(path), kind=kind, dataset=dataset
        )
        for completed, index in enumerate(indices, start=1):
            record = records[index]
            row = rows[record["row_id"]]
            logprobs = np.asarray(gate_run.evaluator.old._topk_payload(row)["logprobs"], dtype=np.float64)
            entropy = np.asarray(row["token_entropies"], dtype=np.float64)
            if logprobs.shape != (len(entropy), 50):
                raise ValueError("PB telemetry alignment mismatch: " + record["uid"])
            detectors = feature_model.answer_temporal_detectors(logprobs, entropy)
            matrix[local_index[index]] = [detectors[name] for name in selected]
            if completed % 250 == 0 or completed == len(indices):
                print(f"[extract] {cell} {completed}/{len(indices)}", flush=True)
        del rows
    if not np.isfinite(matrix).all() or len(set(uids.astype(str))) != EXPECTED_PB:
        raise ValueError("PB detector archive is incomplete")
    archive = OUT / "DETECTORS_FROZEN.npz"
    np.savez_compressed(
        archive,
        X_raw=matrix,
        names=np.asarray(selected),
        pb_index=pb_indices,
        cell=cells,
        uid=uids,
    )
    record = {
        "schema": "math-gate-processbench-detector-freeze-v1",
        "status": "DETECTORS_FROZEN_BEFORE_TARGET_EVALUATION",
        "archive": str(archive.resolve()),
        "archive_sha256": base.sha256_file(archive),
        "math_gate_sha256": base.sha256_file(MATH / "FROZEN_GATE.json"),
        "features": selected,
        "answers": len(pb_indices),
        "source_hashes": source_hashes,
        "labels_loaded_during_extraction": False,
    }
    base.atomic_json(OUT / "DETECTOR_FREEZE.json", record)
    return record


def load_or_extract(records, pb, frozen) -> dict:
    record_path = OUT / "DETECTOR_FREEZE.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf8"))
        if (
            record.get("math_gate_sha256") == base.sha256_file(MATH / "FROZEN_GATE.json")
            and Path(record["archive"]).is_file()
            and base.sha256_file(Path(record["archive"])) == record["archive_sha256"]
            and record.get("features") == list(frozen["features"])
        ):
            print("[reuse] frozen PB detector archive", flush=True)
            return record
        raise ValueError("existing PB detector freeze does not match math gate")
    return extract_detectors(records, pb, frozen)


def evaluate(records, pb, outer, frozen, detector_freeze) -> dict:
    if base.sha256_file(Path(detector_freeze["archive"])) != detector_freeze["archive_sha256"]:
        raise ValueError("PB detector archive changed after freeze")
    if base.sha256_file(MATH / "FROZEN_GATE.json") != detector_freeze["math_gate_sha256"]:
        raise ValueError("math gate changed after PB detector extraction")
    with np.load(detector_freeze["archive"], allow_pickle=False) as saved:
        matrix = np.asarray(saved["X_raw"], dtype=np.float64)
        names = saved["names"].astype(str).tolist()
        cells = saved["cell"].astype(str)
        pb_indices = np.asarray(saved["pb_index"], dtype=np.int64)
    if not np.array_equal(pb_indices, np.flatnonzero(pb)):
        raise ValueError("PB detector index mismatch")

    # This is the first point at which the ProcessBench target is opened.
    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        target_all = np.asarray(saved["target"], dtype=np.int64)
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    if int(offsets[-1]) != EXPECTED_STEPS:
        raise ValueError("ProcessBench step roster mismatch")
    target = target_all[pb]
    families_by_cell = {
        cell: dataset
        for cell, _, kind, dataset in gate_run.evaluator.source_specs()
        if kind == "pb"
    }
    families = np.asarray([families_by_cell[cell] for cell in cells])
    gate_score = selection.apply_frozen_fusion(matrix, names, cells, frozen)
    gate_open = gate_score >= float(frozen["q"])

    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        q15_steps = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
        original_steps = np.asarray(saved["steps__original_static_fusion_before_top10"], dtype=np.float64)
    q15_peak, q15_valid = old_integration.peaks(q15_steps, offsets, pb)
    original_peak, original_valid = old_integration.peaks(original_steps, offsets, pb)
    q15_peak_pb = q15_peak[pb]
    q15_valid_pb = q15_valid[pb]
    new_prediction = np.where(gate_open & q15_valid_pb, q15_peak_pb, -1)
    new_valid = q15_valid_pb & np.isfinite(gate_score)

    with np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False) as saved:
        entropy = np.asarray(saved["entropy_mean"], dtype=np.float64)
    with np.load(OLD_GATE / "DETECTORS_FROZEN.npz", allow_pickle=False) as saved:
        tail15 = np.asarray(saved["tail15_mass__token_mean"], dtype=np.float64)
    entropy_thresholds = json.loads((FIXED_GATE / "METRICS.json").read_text(encoding="utf8"))[
        "arms"
    ]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"]
    tail_thresholds = json.loads((OLD_GATE / "METRICS.json").read_text(encoding="utf8"))[
        "metrics"
    ]["tail15_mass__token_mean"]["thresholds"]
    predictions = {
        "original_static_entropy": old_integration.fold_predictions(
            entropy, entropy_thresholds, original_peak, original_valid, outer, pb
        ),
        "q15_finalist_entropy": old_integration.fold_predictions(
            entropy, entropy_thresholds, q15_peak, q15_valid, outer, pb
        ),
        "integrated_q15_tail15": old_integration.fold_predictions(
            tail15, tail_thresholds, q15_peak, q15_valid, outer, pb
        ),
        "math_frozen_gate": (new_prediction, new_valid),
    }
    groups = np.asarray([records[index]["group_id"] for index in pb_indices])
    metrics = {
        name: old_integration.summarize(target, cells, *predictions[name])
        for name in METHODS
    }
    continuous_gate_scores = {
        "original_static_entropy": entropy[pb],
        "q15_finalist_entropy": entropy[pb],
        "integrated_q15_tail15": tail15[pb],
        "math_frozen_gate": gate_score,
    }
    binary = {
        name: selection.evaluate_binary_prediction(
            (target >= 0).astype(np.int8),
            (predictions[name][0] != -1).astype(np.int8),
            continuous_gate_scores[name],
            cells,
            families,
        )
        for name in METHODS
    }
    binary["math_frozen_gate"]["q"] = float(frozen["q"])
    contrasts = old_integration.paired_bootstrap(target, cells, groups, predictions, PAIRS)
    for first, second in PAIRS:
        key = first + "_minus_" + second
        contrasts[key]["point_delta"] = (
            metrics[first]["macros"]["all"] - metrics[second]["macros"]["all"]
        )
    error = target >= 0
    clean = ~error
    error_analysis = {
        "answers": int(len(target)),
        "clean": int(clean.sum()),
        "error": int(error.sum()),
        "gate_open": int(gate_open.sum()),
        "gate_closed": int((~gate_open).sum()),
        "locator_exact_before_gate": int(np.sum(error & (q15_peak_pb == target))),
        "locator_exact_suppressed_by_gate": int(np.sum(error & (q15_peak_pb == target) & ~gate_open)),
        "clean_false_alarms_after_gate": int(np.sum(clean & gate_open)),
        "clean_peaks_suppressed": int(np.sum(clean & ~gate_open)),
        "comparators": {},
    }
    for comparator in ("integrated_q15_tail15", "q15_finalist_entropy"):
        baseline_prediction, _ = predictions[comparator]
        baseline_open = baseline_prediction != -1
        error_analysis["comparators"][comparator] = {
            "prediction_disagreements": int(np.sum(new_prediction != baseline_prediction)),
            "new_only_correct": int(np.sum((new_prediction == target) & (baseline_prediction != target))),
            "baseline_only_correct": int(np.sum((baseline_prediction == target) & (new_prediction != target))),
            "clean_false_alarms_removed": int(np.sum(clean & ~gate_open & baseline_open)),
            "clean_false_alarms_added": int(np.sum(clean & gate_open & ~baseline_open)),
            "errors_newly_suppressed": int(np.sum(error & ~gate_open & baseline_open)),
            "errors_newly_opened": int(np.sum(error & gate_open & ~baseline_open)),
            "answer_gate_new_only_correct": int(np.sum((gate_open == error) & (baseline_open != error))),
            "answer_gate_baseline_only_correct": int(np.sum((baseline_open == error) & (gate_open != error))),
        }
    result = {
        "schema": "math-gate-processbench-transfer-metrics-v1",
        "status": "COMPLETE",
        "math_frozen_gate": {
            "arm": frozen["arm"],
            "features": frozen["features"],
            "q": frozen["q"],
        },
        "localization": metrics,
        "answer_detection": binary,
        "contrasts": contrasts,
        "error_analysis": error_analysis,
        "selection_or_recalibration_on_processbench": False,
        "development_transfer_not_external_confirmation": True,
    }
    base.atomic_json(OUT / "METRICS.json", result)
    np.savez_compressed(
        OUT / "PREDICTIONS.npz",
        gate_score=gate_score,
        gate_open=gate_open,
        target=target,
        q15_peak=q15_peak_pb,
        **{name: predictions[name][0] for name in METHODS},
    )
    review = {
        "schema": "math-gate-processbench-transfer-review-v1",
        "status": "PASS",
        "pb_answers": len(target),
        "finite_gate_scores": bool(np.isfinite(gate_score).all()),
        "detectors_frozen_before_target": True,
        "math_gate_hash_unchanged": True,
        "processbench_selection": False,
        "common_method_and_q_all_cells": True,
    }
    base.atomic_json(OUT / "RESULT_REVIEW.json", review)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check = preflight()
    base.atomic_json(OUT / "PREFLIGHT.json", check)
    print(json.dumps(check, indent=2), flush=True)
    if check["status"] != "PASS":
        raise SystemExit(2)
    frozen = verify_math_freeze()
    records, outer, pb = gate_run.load_metadata()
    detector_freeze = load_or_extract(records, pb, frozen)
    result = evaluate(records, pb, outer, frozen, detector_freeze)
    print("method,pb_all8,pb_q4,pb_q8,clean_accuracy,error_exact_accuracy")
    for name in METHODS:
        row = result["localization"][name]
        print(
            f"{name},{row['macros']['all']},{row['macros']['q4']},{row['macros']['q8']},"
            f"{row['clean_accuracy']},{row['error_exact_accuracy']}"
        )
    print(
        "answer_gate_family_macro_f1",
        result["answer_detection"]["math_frozen_gate"]["family_macro"]["macro_f1"],
    )


if __name__ == "__main__":
    main()
