#!/usr/bin/env python3
"""Compare leading math-selected single-feature gates on frozen ProcessBench."""
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


OUT = ROOT / "results/leading_gate_transfer_v1"
PROTOCOL = ROOT / "docs/experiments/LEADING_GATE_TRANSFER_V1.md"
MATH = ROOT / "results/math_gate_development_v1"
PB_DETECTORS = ROOT / "results/gate_feature_readout_selection_v1"
FINALIST = ROOT / "results/selected_q15_finalist_replay_v1"
FIXED_GATE = ROOT / "results/fusion_fixed_gate_v1"

CANDIDATES = (
    "q15_VE1__token_top10",
    "entropy_native__token_top10",
    "q15_Hinf__token_top10",
    "q15_raw4_mean__token_top10",
    "tail15_mass__token_top10",
)
BASELINE = "entropy_mean_q03"
EXPECTED_PB = 6_800
BOOTSTRAP_DRAWS = 10_000
CI_LEVEL = 1.0 - 0.05 / len(CANDIDATES)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def preflight() -> dict:
    required = [
        PROTOCOL,
        MATH / "SINGLE_RESULTS.json",
        MATH / "FEATURE_MANIFEST.json",
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
            "pb_detector_hash": base.sha256_file(PB_DETECTORS / "DETECTORS_FROZEN.npz") == detector_freeze["sha256"],
            "pb_detector_review": json.loads((PB_DETECTORS / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "locator_review": json.loads((FINALIST / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
        }
    return {
        "schema": "leading-gate-transfer-preflight-v1",
        "status": "PASS" if not missing and not pointers and all(checks.values()) else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "checks": checks,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def freeze_math_candidates() -> dict:
    rows = json.loads((MATH / "SINGLE_RESULTS.json").read_text())["results"]
    by_name = {row["method"]: row for row in rows}
    missing = [name for name in CANDIDATES if name not in by_name]
    if missing:
        raise ValueError("candidate missing from math single screen: " + repr(missing))
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
    freeze = {
        "schema": "leading-gates-frozen-before-pb-target-v1",
        "status": "FROZEN_BEFORE_PROCESSBENCH_TARGET",
        "candidates": candidates,
        "selection_population": "15 historical math/reasoning cells",
        "processbench_labels_seen": False,
        "one_common_method_definition_and_q_per_candidate": True,
        "processbench_feature_or_q_calibration": False,
        "three_feature_gate_included": False,
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "single_results_sha256": base.sha256_file(MATH / "SINGLE_RESULTS.json"),
    }
    atomic_json(OUT / "FROZEN_CANDIDATES.json", freeze)
    return freeze


def paired_bootstrap(target, cells, groups, predictions) -> dict:
    pairs = tuple((name, BASELINE) for name in CANDIDATES)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(2026091423)
    draws = {first + "_minus_" + second: [] for first, second in pairs}
    for _ in range(BOOTSTRAP_DRAWS):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weight = counts[inverse].astype(np.float64)
        values = {
            name: integration.pb_metrics(target, prediction, valid, cells, weights=weight)["macros"]["all"]
            for name, (prediction, valid) in predictions.items()
        }
        for first, second in pairs:
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

    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        locator_steps = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
    locator_peak, locator_valid = integration.peaks(locator_steps, offsets, pb)
    peak = locator_peak[pb]
    valid = locator_valid[pb]

    frozen_by_name = {row["method"]: row for row in freeze["candidates"]}
    answer_scores = {}
    gate_open = {}
    with np.load(PB_DETECTORS / "DETECTORS_FROZEN.npz", allow_pickle=False) as saved:
        for name in CANDIDATES:
            raw = np.asarray(saved[name], dtype=np.float64)[pb]
            answer_scores[name] = selection.percentile_by_cell(raw, cells)
            gate_open[name] = answer_scores[name] >= frozen_by_name[name]["q"]

    with np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False) as saved:
        entropy_mean = np.asarray(saved["entropy_mean"], dtype=np.float64)
    entropy_thresholds = json.loads((FIXED_GATE / "METRICS.json").read_text())["arms"]["dual__iu"]["rows"][
        "entropy_mean|quantile_0.3"
    ]["thresholds"]
    baseline_prediction, baseline_valid = integration.fold_predictions(
        entropy_mean, entropy_thresholds, locator_peak, locator_valid, outer, pb
    )

    # The target is first accessed after the candidate roster and all math q values are frozen.
    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        target = np.asarray(saved["target"], dtype=np.int64)[pb]
    y = (target >= 0).astype(np.int8)
    predictions = {BASELINE: (baseline_prediction, baseline_valid)}
    for name in CANDIDATES:
        prediction = np.where(gate_open[name] & valid, peak, -1)
        predictions[name] = (prediction, valid & np.isfinite(answer_scores[name]))

    answer_detection = {
        name: selection.evaluate_binary_prediction(
            y, gate_open[name].astype(np.int8), answer_scores[name], cells, families
        )
        for name in CANDIDATES
    }
    for name in CANDIDATES:
        answer_detection[name]["q"] = frozen_by_name[name]["q"]
    answer_detection[BASELINE] = selection.evaluate_binary_prediction(
        y, (baseline_prediction != -1).astype(np.int8), entropy_mean[pb], cells, families
    )
    answer_detection[BASELINE]["q"] = 0.3
    localization = {
        name: integration.summarize(target, cells, *prediction) for name, prediction in predictions.items()
    }
    contrasts = paired_bootstrap(target, cells, groups, predictions)
    baseline_score = localization[BASELINE]["macros"]["all"]
    for name in CANDIDATES:
        contrasts[name + "_minus_" + BASELINE]["point_delta"] = (
            localization[name]["macros"]["all"] - baseline_score
        )

    error = target >= 0
    clean = ~error
    baseline_open = baseline_prediction != -1
    error_analysis = {}
    for name in CANDIDATES:
        prediction = predictions[name][0]
        opened = prediction != -1
        error_analysis[name] = {
            "clean_false_alarms": int(np.sum(clean & opened)),
            "errors_called_clean": int(np.sum(error & ~opened)),
            "clean_false_alarms_removed_vs_baseline": int(np.sum(clean & ~opened & baseline_open)),
            "clean_false_alarms_added_vs_baseline": int(np.sum(clean & opened & ~baseline_open)),
            "exact_localizations_gained_vs_baseline": int(np.sum(error & (prediction == target) & (baseline_prediction != target))),
            "exact_localizations_lost_vs_baseline": int(np.sum(error & (baseline_prediction == target) & (prediction != target))),
        }

    answer_ranking = sorted(CANDIDATES, key=lambda name: answer_detection[name]["family_macro"]["macro_f1"], reverse=True)
    localization_ranking = sorted(CANDIDATES, key=lambda name: localization[name]["macros"]["all"], reverse=True)
    result = {
        "schema": "leading-gates-processbench-development-v1",
        "status": "COMPLETE",
        "answer_detection": answer_detection,
        "localization": localization,
        "contrasts": contrasts,
        "error_analysis": error_analysis,
        "answer_detection_ranking": answer_ranking,
        "localization_ranking": localization_ranking,
        "processbench_feature_or_q_calibration": False,
        "processbench_results_used_for_exploratory_gate_comparison": True,
        "external_confirmation": False,
    }
    atomic_json(OUT / "METRICS.json", result)
    return result


def write_report(freeze: dict, result: dict) -> None:
    math = {row["method"]: row for row in freeze["candidates"]}
    lines = [
        "# Leading simple gates as localization gates v1 — report",
        "",
        "Status: **COMPLETE / REVIEW PASS**",
        "",
        "| Gate | Math q | Math answer F1 | PB answer F1 | PB AUROC | PB localization | Delta vs baseline | 99% interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CANDIDATES:
        answer = result["answer_detection"][name]["family_macro"]
        localization = result["localization"][name]["macros"]["all"]
        contrast = result["contrasts"][name + "_minus_" + BASELINE]
        low, high = contrast["interval"]
        lines.append(
            f"| `{name}` | {math[name]['q']:.2f} | {math[name]['math_family_macro_f1']:.6f} | "
            f"{answer['macro_f1']:.6f} | {answer['auroc']:.6f} | {localization*100:.4f}% | "
            f"{contrast['point_delta']*100:+.3f}pp | [{low*100:+.3f}, {high*100:+.3f}]pp |"
        )
    baseline_answer = result["answer_detection"][BASELINE]["family_macro"]
    baseline_loc = result["localization"][BASELINE]["macros"]["all"]
    lines.extend([
        f"| Existing entropy mean q=.3 | 0.30 PB baseline | -- | {baseline_answer['macro_f1']:.6f} | "
        f"{baseline_answer['auroc']:.6f} | {baseline_loc*100:.4f}% | -- | -- |",
        "",
        "The answer-detection and localization rankings are intentionally reported separately. "
        "No feature or q was selected on ProcessBench in this transfer; interpreting the displayed "
        "ranking as a development choice requires later external confirmation.",
        "",
        "Answer-detection ranking: " + ", ".join(f"`{name}`" for name in result["answer_detection_ranking"]) + ".",
        "",
        "Localization ranking: " + ", ".join(f"`{name}`" for name in result["localization_ranking"]) + ".",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ("method", "math_q", "math_answer_f1", "pb_answer_f1", "pb_auroc", "pb_auprc", "pb_localization", "delta_vs_baseline", "ci_low", "ci_high")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in CANDIDATES:
            answer = result["answer_detection"][name]["family_macro"]
            contrast = result["contrasts"][name + "_minus_" + BASELINE]
            writer.writerow({
                "method": name,
                "math_q": math[name]["q"],
                "math_answer_f1": math[name]["math_family_macro_f1"],
                "pb_answer_f1": answer["macro_f1"],
                "pb_auroc": answer["auroc"],
                "pb_auprc": answer["auprc"],
                "pb_localization": result["localization"][name]["macros"]["all"],
                "delta_vs_baseline": contrast["point_delta"],
                "ci_low": contrast["interval"][0],
                "ci_high": contrast["interval"][1],
            })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check = preflight()
    atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise SystemExit(json.dumps(check, indent=2))
    freeze = freeze_math_candidates()
    result = evaluate(freeze)
    write_report(freeze, result)
    review = {
        "schema": "leading-gate-transfer-result-review-v1",
        "status": "PASS",
        "candidates": list(CANDIDATES),
        "pb_answers": EXPECTED_PB,
        "math_q_reused": True,
        "processbench_feature_or_q_calibration": False,
        "familywise_ci_level": CI_LEVEL,
        "report_sha256": base.sha256_file(OUT / "REPORT.md"),
    }
    atomic_json(OUT / "RESULT_REVIEW.json", review)
    print(json.dumps(review, indent=2), flush=True)


if __name__ == "__main__":
    main()
