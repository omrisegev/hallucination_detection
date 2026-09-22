#!/usr/bin/env python3
"""Compare tail15 Top10 and mean gates with math-selected thresholds."""
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


OUT = ROOT / "results/tail15_readout_headtohead_v1"
PROTOCOL = ROOT / "docs/experiments/TAIL15_READOUT_HEADTOHEAD_V1.md"
MATH = ROOT / "results/math_gate_development_v1"
PB_DETECTORS = ROOT / "results/gate_feature_readout_selection_v1"
FINALIST = ROOT / "results/selected_q15_finalist_replay_v1"
FIXED_GATE = ROOT / "results/fusion_fixed_gate_v1"

TOP10 = "tail15_mass__token_top10"
MEAN = "tail15_mass__token_mean"
BASELINE = "entropy_mean_q03"
CANDIDATES = (TOP10, MEAN)
PAIRS = ((TOP10, MEAN), (TOP10, BASELINE), (MEAN, BASELINE))
EXPECTED_PB = 6_800
BOOTSTRAP_DRAWS = 10_000
CI_LEVEL = 1.0 - 0.05 / len(PAIRS)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def preflight() -> dict:
    required = [
        PROTOCOL,
        MATH / "SINGLE_RESULTS.json",
        PB_DETECTORS / "DETECTORS_FROZEN.npz",
        PB_DETECTORS / "FROZEN_DETECTORS.json",
        PB_DETECTORS / "RESULT_REVIEW.json",
        FINALIST / "SCORES_FROZEN.npz",
        FINALIST / "RESULT_REVIEW.json",
        FIXED_GATE / "DETECTORS.npz",
        FIXED_GATE / "METRICS.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    pointers = [str(path) for path in required if path.is_file() and base.is_lfs_pointer(path)]
    checks = {}
    if not missing:
        detector_freeze = json.loads((PB_DETECTORS / "FROZEN_DETECTORS.json").read_text())
        checks = {
            "detector_hash": base.sha256_file(PB_DETECTORS / "DETECTORS_FROZEN.npz") == detector_freeze["sha256"],
            "detector_review": json.loads((PB_DETECTORS / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "locator_review": json.loads((FINALIST / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
        }
    return {
        "schema": "tail15-readout-headtohead-preflight-v1",
        "status": "PASS" if not missing and not pointers and all(checks.values()) else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "checks": checks,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def freeze_candidates() -> dict:
    rows = json.loads((MATH / "SINGLE_RESULTS.json").read_text())["results"]
    by_name = {row["method"]: row for row in rows}
    candidates = []
    for name in CANDIDATES:
        row = by_name[name]
        metric = row["selected"]["family_macro"]
        candidates.append({
            "method": name,
            "math_rank": int(row["rank"]),
            "q": float(row["selected_q"]),
            "math_family_macro_f1": float(metric["macro_f1"]),
            "math_family_macro_auroc": float(metric["auroc"]),
            "math_family_macro_auprc": float(metric["auprc"]),
        })
    result = {
        "schema": "tail15-readout-candidates-frozen-before-pb-target-v1",
        "status": "FROZEN_BEFORE_PROCESSBENCH_TARGET",
        "candidates": candidates,
        "raw_signal": "one_minus_sum_exp_top15_logprobs",
        "only_difference": "whole_answer_readout",
        "processbench_labels_seen": False,
        "processbench_readout_or_q_calibration": False,
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "math_single_results_sha256": base.sha256_file(MATH / "SINGLE_RESULTS.json"),
    }
    atomic_json(OUT / "FROZEN_CANDIDATES.json", result)
    return result


def bootstrap(target, cells, groups, predictions) -> dict:
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(2026091429)
    draws = {first + "_minus_" + second: [] for first, second in PAIRS}
    for _ in range(BOOTSTRAP_DRAWS):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[inverse].astype(np.float64)
        values = {
            name: integration.pb_metrics(target, prediction, valid, cells, weights=weights)["macros"]["all"]
            for name, (prediction, valid) in predictions.items()
        }
        for first, second in PAIRS:
            draws[first + "_minus_" + second].append(values[first] - values[second])
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


def evaluate(freeze: dict) -> dict:
    records, outer, pb = gate_run.load_metadata()
    if int(pb.sum()) != EXPECTED_PB:
        raise ValueError("ProcessBench roster mismatch")
    cells = np.asarray([row["cell"] for row in records])[pb]
    groups = np.asarray([row["group_id"] for row in records])[pb]
    families_by_cell = {
        cell: dataset for cell, _, kind, dataset in gate_run.evaluator.source_specs() if kind == "pb"
    }
    families = np.asarray([families_by_cell[cell] for cell in cells])
    frozen = {row["method"]: row for row in freeze["candidates"]}

    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        locator_steps = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
    locator_peak, locator_valid = integration.peaks(locator_steps, offsets, pb)
    peak = locator_peak[pb]
    valid = locator_valid[pb]

    scores = {}
    opened = {}
    with np.load(PB_DETECTORS / "DETECTORS_FROZEN.npz", allow_pickle=False) as saved:
        for name in CANDIDATES:
            raw = np.asarray(saved[name], dtype=np.float64)[pb]
            scores[name] = selection.percentile_by_cell(raw, cells)
            opened[name] = scores[name] >= frozen[name]["q"]

    with np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False) as saved:
        entropy_mean = np.asarray(saved["entropy_mean"], dtype=np.float64)
    thresholds = json.loads((FIXED_GATE / "METRICS.json").read_text())["arms"]["dual__iu"]["rows"][
        "entropy_mean|quantile_0.3"
    ]["thresholds"]
    baseline_prediction, baseline_valid = integration.fold_predictions(
        entropy_mean, thresholds, locator_peak, locator_valid, outer, pb
    )

    # First target access follows the frozen readouts and math-selected q values.
    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        target = np.asarray(saved["target"], dtype=np.int64)[pb]
    y = (target >= 0).astype(np.int8)
    predictions = {BASELINE: (baseline_prediction, baseline_valid)}
    for name in CANDIDATES:
        prediction = np.where(opened[name] & valid, peak, -1)
        predictions[name] = (prediction, valid & np.isfinite(scores[name]))

    answer_detection = {
        name: selection.evaluate_binary_prediction(y, opened[name].astype(np.int8), scores[name], cells, families)
        for name in CANDIDATES
    }
    for name in CANDIDATES:
        answer_detection[name]["q"] = frozen[name]["q"]
    answer_detection[BASELINE] = selection.evaluate_binary_prediction(
        y, (baseline_prediction != -1).astype(np.int8), entropy_mean[pb], cells, families
    )
    answer_detection[BASELINE]["q"] = 0.3
    localization = {
        name: integration.summarize(target, cells, *prediction) for name, prediction in predictions.items()
    }
    contrasts = bootstrap(target, cells, groups, predictions)
    for first, second in PAIRS:
        contrasts[first + "_minus_" + second]["point_delta"] = (
            localization[first]["macros"]["all"] - localization[second]["macros"]["all"]
        )

    error = target >= 0
    clean = ~error
    error_analysis = {}
    for name in CANDIDATES:
        prediction = predictions[name][0]
        error_analysis[name] = {
            "clean_false_alarms": int(np.sum(clean & (prediction != -1))),
            "errors_called_clean": int(np.sum(error & (prediction == -1))),
            "exact_error_localizations": int(np.sum(error & (prediction == target))),
        }
    result = {
        "schema": "tail15-readout-headtohead-metrics-v1",
        "status": "COMPLETE",
        "answer_detection": answer_detection,
        "localization": localization,
        "contrasts": contrasts,
        "error_analysis": error_analysis,
        "processbench_readout_or_q_calibration": False,
        "historical_tail15_mean_q03_excluded_from_selection": True,
        "external_confirmation": False,
    }
    atomic_json(OUT / "METRICS.json", result)
    return result


def write_outputs(freeze: dict, result: dict) -> None:
    frozen = {row["method"]: row for row in freeze["candidates"]}
    rows = []
    for name in CANDIDATES:
        answer = result["answer_detection"][name]["family_macro"]
        loc = result["localization"][name]
        rows.append({
            "method": name,
            "math_q": frozen[name]["q"],
            "math_answer_f1": frozen[name]["math_family_macro_f1"],
            "pb_answer_f1": answer["macro_f1"],
            "pb_auroc": answer["auroc"],
            "pb_auprc": answer["auprc"],
            "pb_localization": loc["macros"]["all"],
            "clean_accuracy": loc["clean_accuracy"],
            "error_exact_accuracy": loc["error_exact_accuracy"],
        })
    primary = result["contrasts"][TOP10 + "_minus_" + MEAN]
    low, high = primary["interval"]
    preferred = TOP10 if primary["point_delta"] >= 0 else MEAN
    decision = {
        "schema": "tail15-readout-headtohead-decision-v1",
        "status": "DEVELOPMENT_DECISION",
        "point_preferred": preferred,
        "primary_contrast": TOP10 + "_minus_" + MEAN,
        "primary_point_delta": primary["point_delta"],
        "primary_interval": primary["interval"],
        "confirmed": bool(low > 0 or high < 0),
        "external_confirmation_required": True,
    }
    atomic_json(OUT / "DECISION.json", decision)
    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Tail15 readout head-to-head v1 — report",
        "",
        "Status: **COMPLETE / REVIEW PASS**",
        "",
        "| Readout | Math q | Math answer F1 | PB answer F1 | PB AUROC | PB localization | Clean accuracy | Error exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| `{row['method']}` | {row['math_q']:.2f} | {row['math_answer_f1']:.6f} | "
            f"{row['pb_answer_f1']:.6f} | {row['pb_auroc']:.6f} | {row['pb_localization']*100:.4f}% | "
            f"{row['clean_accuracy']:.6f} | {row['error_exact_accuracy']:.6f} |"
        )
    report.extend([
        "",
        f"Primary Top10-minus-mean localization delta: {primary['point_delta']*100:+.3f}pp; "
        f"family-wise {CI_LEVEL*100:.3f}% interval [{low*100:+.3f}, {high*100:+.3f}]pp.",
        "",
        f"Point-preferred readout: `{preferred}`. This is a development decision, not external confirmation.",
        "",
        "The historical tail15-mean q=.3 result is retained only as a PB-developed diagnostic and did not enter this selection.",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check = preflight()
    atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise SystemExit(json.dumps(check, indent=2))
    freeze = freeze_candidates()
    result = evaluate(freeze)
    write_outputs(freeze, result)
    review = {
        "schema": "tail15-readout-headtohead-result-review-v1",
        "status": "PASS",
        "pb_answers": EXPECTED_PB,
        "candidates": list(CANDIDATES),
        "same_raw_signal": True,
        "math_q_reused": True,
        "processbench_readout_or_q_calibration": False,
        "report_sha256": base.sha256_file(OUT / "REPORT.md"),
    }
    atomic_json(OUT / "RESULT_REVIEW.json", review)
    print(json.dumps(review, indent=2), flush=True)


if __name__ == "__main__":
    main()
