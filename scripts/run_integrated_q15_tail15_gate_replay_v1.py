"""Replay cumulative q15 locator and selected tail15 gate decisions."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_direct_probability_temporal as evaluator
from scripts import run_renyi_position_temporal_fusion as base
from spectral_utils.historical_fusion_evaluation import pb_metrics


OUT = ROOT / "results/integrated_q15_tail15_gate_replay_v1"
PROTOCOL = ROOT / "docs/experiments/INTEGRATED_Q15_TAIL15_GATE_REPLAY_V1.md"
FINALIST = ROOT / "results/selected_q15_finalist_replay_v1"
GATE_SELECTION = ROOT / "results/gate_feature_readout_selection_v1"
FIXED_GATE = ROOT / "results/fusion_fixed_gate_v1"
UNIFORM = ROOT / "results/uniform_multiscale_fusion_v1"
SELECTED_GATE = "tail15_mass__token_mean"
METHODS = (
    "original_static_entropy",
    "q15_finalist_entropy",
    "original_static_tail15",
    "integrated_q15_tail15",
)
PAIRS = (
    ("integrated_q15_tail15", "q15_finalist_entropy"),
    ("integrated_q15_tail15", "original_static_entropy"),
    ("original_static_tail15", "original_static_entropy"),
    ("integrated_q15_tail15", "original_static_tail15"),
)
BOOTSTRAP_DRAWS = 10_000
CI_LEVEL = 1.0 - 0.05 / len(PAIRS)
EXPECTED_ANSWERS = 13_769
EXPECTED_PB = 6_800
EXPECTED_STEPS = 145_597


def gated_prediction(detector, threshold, peak, valid):
    detector = np.asarray(detector, dtype=np.float64)
    peak = np.asarray(peak, dtype=np.int64)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(detector)
    prediction = np.where(usable & (detector >= float(threshold)), peak, -1)
    return prediction, usable


def required_inputs():
    return [
        PROTOCOL,
        Path(__file__),
        ROOT / "scripts/test_integrated_q15_tail15_gate_replay.py",
        FINALIST / "SCORES_FROZEN.npz",
        FINALIST / "METRICS.json",
        FINALIST / "RESULT_REVIEW.json",
        GATE_SELECTION / "DETECTORS_FROZEN.npz",
        GATE_SELECTION / "METRICS.json",
        GATE_SELECTION / "SELECTION.json",
        GATE_SELECTION / "RESULT_REVIEW.json",
        FIXED_GATE / "DETECTORS.npz",
        FIXED_GATE / "METRICS.json",
        UNIFORM / "MANIFEST.json",
    ]


def configure_and_load():
    uniform = json.loads((UNIFORM / "MANIFEST.json").read_text(encoding="utf8"))
    base.configure_sources(Path(uniform["source_root"]), Path(uniform["contract_root"]))
    records = json.loads(
        (evaluator.old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8")
    )["records"]
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        joined = {name: np.asarray(saved[name]) for name in saved.files}
    if len(records) != EXPECTED_ANSWERS or int(joined["offsets"][-1]) != EXPECTED_STEPS:
        raise ValueError("integration benchmark roster mismatch")
    pb = np.asarray([row["cell"].startswith("pb_") for row in records])
    if int(pb.sum()) != EXPECTED_PB:
        raise ValueError("integration PB roster mismatch")
    folds = json.loads(evaluator.old.FOLDS.read_text(encoding="utf8"))["outer"]
    outer = np.asarray([int(folds[row["group_id"]]) for row in records], dtype=np.int64)
    return records, joined, pb, outer


def preflight():
    missing = [str(path) for path in required_inputs() if not path.is_file()]
    pointers = [str(path) for path in required_inputs() if path.is_file() and base.is_lfs_pointer(path)]
    selection = None
    reviews = {}
    if not missing:
        selection = json.loads((GATE_SELECTION / "SELECTION.json").read_text(encoding="utf8"))["selected"]
        reviews = {
            "finalist": json.loads((FINALIST / "RESULT_REVIEW.json").read_text(encoding="utf8"))["status"],
            "gate_selection": json.loads((GATE_SELECTION / "RESULT_REVIEW.json").read_text(encoding="utf8"))["status"],
        }
    return {
        "schema": "integrated-q15-tail15-preflight-v1",
        "status": "PASS" if not missing and not pointers and selection == SELECTED_GATE and set(reviews.values()) == {"PASS"} else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "selected_gate": selection,
        "source_reviews": reviews,
        "python": sys.executable,
    }


def peaks(score, offsets, pb):
    peak = np.full(len(pb), -1, dtype=np.int64)
    valid = np.zeros(len(pb), dtype=bool)
    for index in np.flatnonzero(pb):
        value = score[offsets[index]:offsets[index + 1]]
        if len(value) and np.isfinite(value).all():
            peak[index] = int(np.argmax(value))
            valid[index] = True
    return peak, valid


def fold_predictions(detector, thresholds, peak, valid, outer, pb):
    prediction = np.full(int(pb.sum()), -1, dtype=np.int64)
    usable = np.zeros(int(pb.sum()), dtype=bool)
    fold_pb = outer[pb]
    detector_pb = detector[pb]
    peak_pb = peak[pb]
    valid_pb = valid[pb]
    for fold in sorted(set(fold_pb)):
        mask = fold_pb == fold
        pred, ok = gated_prediction(detector_pb[mask], thresholds[str(int(fold))], peak_pb[mask], valid_pb[mask])
        prediction[mask] = pred
        usable[mask] = ok
    return prediction, usable


def summarize(target, cells, prediction, valid):
    result = pb_metrics(target, prediction, valid, cells)
    erroneous = target >= 0
    clean = target < 0
    return {
        "macros": result["macros"],
        "cells": result["cells"],
        "clean_accuracy": float(np.mean(prediction[clean] == -1)),
        "error_exact_accuracy": float(np.mean(prediction[erroneous] == target[erroneous])),
        "error_called_clean": int(np.sum(erroneous & (prediction == -1))),
        "clean_false_alarm": int(np.sum(clean & (prediction != -1))),
        "valid_answers": int(valid.sum()),
    }


def paired_bootstrap(target, cells, groups, predictions, pairs):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(2026091417)
    draws = {first + "_minus_" + second: [] for first, second in pairs}
    for _ in range(BOOTSTRAP_DRAWS):
        count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weight = count[inverse].astype(np.float64)
        values = {}
        for name in {item for pair in pairs for item in pair}:
            prediction, valid = predictions[name]
            values[name] = pb_metrics(target, prediction, valid, cells, weights=weight)["macros"]["all"]
        for first, second in pairs:
            draws[first + "_minus_" + second].append(values[first] - values[second])
    alpha = (1.0 - CI_LEVEL) / 2.0
    output = {}
    for key, values in draws.items():
        values = np.asarray(values, dtype=np.float64)
        output[key] = {
            "draws": int(len(values)),
            "ci_level": CI_LEVEL,
            "mean": float(values.mean()),
            "interval": [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))],
        }
    return output


def main():
    check = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    base.atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise FileNotFoundError(base.dumps(check))
    records, joined, pb, outer = configure_and_load()

    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        q15 = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
        original = np.asarray(saved["steps__original_static_fusion_before_top10"], dtype=np.float64)
    with np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False) as saved:
        entropy = np.asarray(saved["entropy_mean"], dtype=np.float64)
    with np.load(GATE_SELECTION / "DETECTORS_FROZEN.npz", allow_pickle=False) as saved:
        tail15 = np.asarray(saved[SELECTED_GATE], dtype=np.float64)
    entropy_thresholds = json.loads((FIXED_GATE / "METRICS.json").read_text(encoding="utf8"))[
        "arms"
    ]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"]
    tail_thresholds = json.loads((GATE_SELECTION / "METRICS.json").read_text(encoding="utf8"))[
        "metrics"
    ][SELECTED_GATE]["thresholds"]

    q15_peak, q15_valid = peaks(q15, joined["offsets"], pb)
    original_peak, original_valid = peaks(original, joined["offsets"], pb)
    predictions = {
        "original_static_entropy": fold_predictions(entropy, entropy_thresholds, original_peak, original_valid, outer, pb),
        "q15_finalist_entropy": fold_predictions(entropy, entropy_thresholds, q15_peak, q15_valid, outer, pb),
        "original_static_tail15": fold_predictions(tail15, tail_thresholds, original_peak, original_valid, outer, pb),
        "integrated_q15_tail15": fold_predictions(tail15, tail_thresholds, q15_peak, q15_valid, outer, pb),
    }
    cells = np.asarray([row["cell"] for row in records])[pb]
    groups = np.asarray([row["group_id"] for row in records])[pb]
    target = joined["target"][pb]
    metrics = {name: summarize(target, cells, *predictions[name]) for name in METHODS}

    old_metrics = json.loads((FINALIST / "METRICS.json").read_text(encoding="utf8"))["metrics"]
    np.testing.assert_allclose(
        metrics["q15_finalist_entropy"]["macros"]["all"],
        old_metrics["selected_q15_raw_per_view_top10"]["pb_all8"], atol=0, rtol=0,
    )
    np.testing.assert_allclose(
        metrics["original_static_entropy"]["macros"]["all"],
        old_metrics["original_static_fusion_before_top10"]["pb_all8"], atol=0, rtol=0,
    )
    selected_metric = json.loads((GATE_SELECTION / "METRICS.json").read_text(encoding="utf8"))[
        "metrics"
    ][SELECTED_GATE]["macros"]["all"]
    np.testing.assert_allclose(metrics["integrated_q15_tail15"]["macros"]["all"], selected_metric, atol=0, rtol=0)

    contrasts = paired_bootstrap(target, cells, groups, predictions, PAIRS)
    for first, second in PAIRS:
        contrasts[first + "_minus_" + second]["point_delta"] = (
            metrics[first]["macros"]["all"] - metrics[second]["macros"]["all"]
        )
    incremental = contrasts["integrated_q15_tail15_minus_q15_finalist_entropy"]["point_delta"]
    cumulative = contrasts["integrated_q15_tail15_minus_original_static_entropy"]["point_delta"]
    decision = {
        "schema": "integrated-q15-tail15-decision-v1",
        "rule": "positive incremental and cumulative PB point deltas",
        "incremental_delta": incremental,
        "cumulative_delta": cumulative,
        "point_gate_pass": bool(incremental > 0 and cumulative > 0),
        "recommendation": "retain_tail15_mean_as_next_gate_candidate" if incremental > 0 and cumulative > 0 else "retain_entropy_gate",
        "development_integration_not_confirmation": True,
    }
    base.atomic_json(OUT / "METRICS.json", {"schema": "integrated-q15-tail15-metrics-v1", "metrics": metrics})
    base.atomic_json(OUT / "CONTRASTS.json", contrasts)
    base.atomic_json(OUT / "DECISION.json", decision)
    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ("method", "pb_all8", "pb_q4", "pb_q8", "clean_accuracy", "error_exact_accuracy", "error_called_clean", "clean_false_alarm")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in METHODS:
            value = metrics[name]
            writer.writerow({
                "method": name,
                "pb_all8": value["macros"]["all"],
                "pb_q4": value["macros"]["q4"],
                "pb_q8": value["macros"]["q8"],
                **{key: value[key] for key in fields[4:]},
            })
    manifest = {
        "schema": "integrated-q15-tail15-replay-v1",
        "selected_gate": SELECTED_GATE,
        "methods": list(METHODS),
        "pairs": [list(pair) for pair in PAIRS],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "familywise_ci": CI_LEVEL},
        "hashes": {str(path): base.sha256_file(path) for path in required_inputs()},
        "same_development_benchmark_integration": True,
    }
    base.atomic_json(OUT / "MANIFEST.json", manifest)
    base.atomic_json(OUT / "RESULT_REVIEW.json", {
        "schema": "integrated-q15-tail15-result-review-v1",
        "status": "PASS",
        "answers": len(records),
        "pb_answers": int(pb.sum()),
        "steps": int(joined["offsets"][-1]),
        "original_static_exact_replay": True,
        "q15_finalist_exact_replay": True,
        "selected_gate_exact_replay": True,
        "no_reselection": True,
        "development_integration_not_confirmation": True,
    })
    base.atomic_json(OUT / "RUN_STATE.json", {"status": "COMPLETE_REVIEWED", "pb_answers": int(pb.sum())})
    print("method,pb_all8,pb_q4,pb_q8,clean_accuracy,error_exact_accuracy")
    for name in METHODS:
        value = metrics[name]
        print(",".join(map(str, (
            name, value["macros"]["all"], value["macros"]["q4"], value["macros"]["q8"],
            value["clean_accuracy"], value["error_exact_accuracy"],
        ))))
    print("decision", base.dumps(decision))


if __name__ == "__main__":
    main()
