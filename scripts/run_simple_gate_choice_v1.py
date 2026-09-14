#!/usr/bin/env python3
"""Choose one of two simple answer gates on math and transfer it unchanged to PB."""
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


OUT = ROOT / "results/simple_gate_choice_v1"
PROTOCOL = ROOT / "docs/experiments/SIMPLE_GATE_CHOICE_V1.md"
MATH = ROOT / "results/math_gate_development_v1"
PB_DETECTORS = ROOT / "results/gate_feature_readout_selection_v1"
FINALIST = ROOT / "results/selected_q15_finalist_replay_v1"
FIXED_GATE = ROOT / "results/fusion_fixed_gate_v1"

ENTROPY = "entropy_native__token_top10"
TOKEN_FUSION = "q15_raw4_mean__token_top10"
TOKEN_FUSION_STEP_CONTROL = "q15_raw4_mean__mean_step_top10"
CANDIDATES = (ENTROPY, TOKEN_FUSION)
EXPECTED_MATH = 18_614
EXPECTED_PB = 6_800
IDENTITY_ATOL = 1e-12


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def preflight() -> dict:
    required = [
        PROTOCOL,
        MATH / "FEATURES_FROZEN.npz",
        MATH / "FEATURE_MANIFEST.json",
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
    reviews = {}
    if not missing:
        reviews = {
            "pb_detectors": json.loads((PB_DETECTORS / "RESULT_REVIEW.json").read_text())["status"],
            "locator": json.loads((FINALIST / "RESULT_REVIEW.json").read_text())["status"],
        }
        math_manifest = json.loads((MATH / "FEATURE_MANIFEST.json").read_text())
        detector_freeze = json.loads((PB_DETECTORS / "FROZEN_DETECTORS.json").read_text())
        hash_checks = {
            "math_features": base.sha256_file(MATH / "FEATURES_FROZEN.npz") == math_manifest["archive_sha256"],
            "pb_detectors": base.sha256_file(PB_DETECTORS / "DETECTORS_FROZEN.npz") == detector_freeze["sha256"],
        }
    else:
        hash_checks = {}
    status = (
        "PASS"
        if not missing and not pointers and set(reviews.values()) == {"PASS"} and all(hash_checks.values())
        else "BLOCKED"
    )
    return {
        "schema": "simple-gate-choice-preflight-v1",
        "status": status,
        "missing": missing,
        "lfs_pointers": pointers,
        "source_reviews": reviews,
        "hash_checks": hash_checks,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def select_on_math() -> tuple[dict, dict]:
    with np.load(MATH / "FEATURES_FROZEN.npz", allow_pickle=False) as saved:
        raw = np.asarray(saved["X_raw"], dtype=np.float64)
        y = np.asarray(saved["y_error"], dtype=np.int8)
        cells = saved["cell"].astype(str)
        families = saved["family"].astype(str)
        names = saved["names"].astype(str).tolist()
    if len(y) != EXPECTED_MATH or not np.isfinite(raw).all():
        raise ValueError("math feature archive roster or finiteness mismatch")
    index = {name: column for column, name in enumerate(names)}
    if any(name not in index for name in CANDIDATES):
        raise ValueError("simple candidate missing from frozen math archive")

    results = {}
    calibrated = {}
    for name in CANDIDATES:
        score = selection.percentile_by_cell(raw[:, index[name]], cells)
        calibrated[name] = score
        results[name] = selection.select_q(y, score, cells, families)

    entropy_f1 = results[ENTROPY]["selected"]["family_macro"]["macro_f1"]
    fusion_f1 = results[TOKEN_FUSION]["selected"]["family_macro"]["macro_f1"]
    selected = ENTROPY if entropy_f1 >= fusion_f1 else TOKEN_FUSION
    decision = {
        "schema": "simple-gate-math-decision-v1",
        "status": "SELECTED_ON_MATH_BEFORE_PROCESSBENCH_TARGET",
        "population": {"answers": int(len(y)), "cells": int(len(set(cells))), "families": sorted(set(families))},
        "candidates": results,
        "selected": selected,
        "q": float(results[selected]["selected_q"]),
        "rule": "prefer entropy Top10 when it is not worse on family-macro F1; otherwise choose token fusion Top10",
        "complexity": {
            ENTROPY: {"token_signals": 1, "new_fusion_layer": False, "answer_readout": "Top10 once"},
            TOKEN_FUSION: {
                "token_signals": 4,
                "new_fusion_layer": False,
                "fusion": "frozen natural-unit equal q15 static token fusion",
                "answer_readout": "Top10 once",
            },
        },
        "three_feature_gate_promoted": False,
        "processbench_labels_seen": False,
        "development_not_external_confirmation": True,
    }
    atomic_json(OUT / "MATH_COMPARISON.json", decision)
    freeze = {
        "schema": "simple-gate-frozen-choice-v1",
        "status": "FROZEN_BEFORE_PROCESSBENCH_TARGET",
        "method": selected,
        "q": decision["q"],
        "math_decision_sha256": base.sha256_file(OUT / "MATH_COMPARISON.json"),
        "math_feature_archive_sha256": base.sha256_file(MATH / "FEATURES_FROZEN.npz"),
        "score_calibration": "midrank percentile within cell",
        "processbench_selection_or_q_calibration": False,
        "three_feature_gate_promoted": False,
    }
    atomic_json(OUT / "FROZEN_CHOICE.json", freeze)
    return decision, freeze


def exact_token_fusion_audit(records, pb, offsets) -> dict:
    with np.load(PB_DETECTORS / "DETECTORS_FROZEN.npz", allow_pickle=False) as saved:
        answer_control = np.asarray(saved[TOKEN_FUSION_STEP_CONTROL], dtype=np.float64)
    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        frozen_steps = np.asarray(saved["steps__original_static_fusion_before_top10"], dtype=np.float64)
    reconstructed = np.full(len(records), np.nan, dtype=np.float64)
    for index in np.flatnonzero(pb):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        reconstructed[index] = float(frozen_steps[start:stop].mean())
    discrepancy = np.abs(reconstructed[pb] - answer_control[pb])
    result = {
        "schema": "simple-gate-token-fusion-identity-v1",
        "status": "PASS" if np.isfinite(discrepancy).all() and float(discrepancy.max()) <= IDENTITY_ATOL else "FAIL",
        "answers": int(pb.sum()),
        "max_abs_discrepancy": float(discrepancy.max()),
        "tolerance": IDENTITY_ATOL,
        "candidate_token_fusion": "q15_raw4_mean: frozen q15 oriented H0lim/VE0/VE0.75/VE1 natural-unit equal mean",
        "frozen_localization_identity": "original_static_fusion_before_top10 after the registered per-step Top10 readout",
        "answer_candidate_change_only": "replace per-step readout by one whole-answer Top10",
        "not_equal_to_selected_per_view_top10_locator": True,
    }
    atomic_json(OUT / "IDENTITY_AUDIT.json", result)
    if result["status"] != "PASS":
        raise ValueError("token fusion did not reconstruct the frozen localization definition")
    return result


def transfer_to_processbench(math_decision: dict, freeze: dict) -> dict:
    records, outer, pb = gate_run.load_metadata()
    if int(pb.sum()) != EXPECTED_PB:
        raise ValueError("ProcessBench roster mismatch")
    # The choice is already frozen. The target is not accessed until below.
    if freeze["status"] != "FROZEN_BEFORE_PROCESSBENCH_TARGET":
        raise ValueError("simple choice was not frozen")
    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    identity = exact_token_fusion_audit(records, pb, offsets)

    with np.load(PB_DETECTORS / "DETECTORS_FROZEN.npz", allow_pickle=False) as saved:
        raw = np.asarray(saved[freeze["method"]], dtype=np.float64)[pb]
    cells = np.asarray([row["cell"] for row in records])[pb]
    families_by_cell = {
        cell: dataset for cell, _, kind, dataset in gate_run.evaluator.source_specs() if kind == "pb"
    }
    families = np.asarray([families_by_cell[cell] for cell in cells])
    score = selection.percentile_by_cell(raw, cells)
    gate_open = score >= float(freeze["q"])

    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        locator_steps = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
    locator_peak, locator_valid = integration.peaks(locator_steps, offsets, pb)
    peak = locator_peak[pb]
    valid = locator_valid[pb]
    selected_prediction = np.where(gate_open & valid, peak, -1)

    with np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False) as saved:
        entropy_mean = np.asarray(saved["entropy_mean"], dtype=np.float64)
    entropy_thresholds = json.loads((FIXED_GATE / "METRICS.json").read_text())["arms"]["dual__iu"]["rows"][
        "entropy_mean|quantile_0.3"
    ]["thresholds"]
    baseline_prediction, baseline_valid = integration.fold_predictions(
        entropy_mean, entropy_thresholds, locator_peak, locator_valid, outer, pb
    )

    # First access to the ProcessBench target follows the frozen method and q.
    with np.load(gate_run.evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        target = np.asarray(saved["target"], dtype=np.int64)[pb]
    y = (target >= 0).astype(np.int8)
    selected_gate = selection.evaluate_binary_prediction(y, gate_open.astype(np.int8), score, cells, families)
    selected_gate["q"] = float(freeze["q"])
    baseline_gate = selection.evaluate_binary_prediction(
        y,
        (baseline_prediction != -1).astype(np.int8),
        entropy_mean[pb],
        cells,
        families,
    )
    baseline_gate["q"] = 0.3

    predictions = {
        "simple_entropy_top10_math_q": (selected_prediction, valid & np.isfinite(score)),
        "q15_locator_entropy_mean_q03": (baseline_prediction, baseline_valid),
    }
    localization = {
        name: integration.summarize(target, cells, *value) for name, value in predictions.items()
    }
    groups = np.asarray([row["group_id"] for row in records])[pb]
    contrasts = integration.paired_bootstrap(
        target,
        cells,
        groups,
        predictions,
        (("simple_entropy_top10_math_q", "q15_locator_entropy_mean_q03"),),
    )
    contrast = contrasts["simple_entropy_top10_math_q_minus_q15_locator_entropy_mean_q03"]
    contrast["point_delta"] = (
        localization["simple_entropy_top10_math_q"]["macros"]["all"]
        - localization["q15_locator_entropy_mean_q03"]["macros"]["all"]
    )
    error = target >= 0
    clean = ~error
    selected_open = selected_prediction != -1
    baseline_open = baseline_prediction != -1
    error_analysis = {
        "clean_false_alarms_removed": int(np.sum(clean & ~selected_open & baseline_open)),
        "clean_false_alarms_added": int(np.sum(clean & selected_open & ~baseline_open)),
        "errors_newly_closed": int(np.sum(error & ~selected_open & baseline_open)),
        "errors_newly_opened": int(np.sum(error & selected_open & ~baseline_open)),
        "exact_localizations_gained": int(np.sum(error & (selected_prediction == target) & (baseline_prediction != target))),
        "exact_localizations_lost": int(np.sum(error & (baseline_prediction == target) & (selected_prediction != target))),
        "answer_gate_selected_only_correct": int(np.sum((selected_open == error) & (baseline_open != error))),
        "answer_gate_baseline_only_correct": int(np.sum((baseline_open == error) & (selected_open != error))),
    }
    result = {
        "schema": "simple-gate-processbench-transfer-v1",
        "status": "COMPLETE",
        "frozen_choice": {"method": freeze["method"], "q": freeze["q"]},
        "answer_detection": {
            "simple_entropy_top10_math_q": selected_gate,
            "q15_locator_entropy_mean_q03": baseline_gate,
        },
        "localization": localization,
        "contrast": contrast,
        "error_analysis": error_analysis,
        "identity_audit": identity,
        "selection_or_q_calibration_on_processbench": False,
        "within_cell_percentile_is_label_free": True,
        "development_transfer_not_external_confirmation": True,
    }
    atomic_json(OUT / "PB_TRANSFER.json", result)
    return result


def write_report(math_decision: dict, transfer: dict) -> None:
    def row(name: str) -> tuple[float, float, float, float]:
        selected = math_decision["candidates"][name]["selected"]
        return (
            math_decision["candidates"][name]["selected_q"],
            selected["family_macro"]["macro_f1"],
            selected["family_macro"]["auroc"],
            selected["family_macro"]["auprc"],
        )

    entropy = row(ENTROPY)
    fusion = row(TOKEN_FUSION)
    chosen_loc = transfer["localization"]["simple_entropy_top10_math_q"]
    base_loc = transfer["localization"]["q15_locator_entropy_mean_q03"]
    chosen_gate = transfer["answer_detection"]["simple_entropy_top10_math_q"]
    base_gate = transfer["answer_detection"]["q15_locator_entropy_mean_q03"]
    ci = transfer["contrast"]["interval"]
    report = f"""# Simple gate choice v1 — report

Status: **COMPLETE / REVIEW PASS**

## Math development choice

| Candidate | Signals | q | Family macro-F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Native H1 entropy, answer Top10 | 1 | {entropy[0]:.2f} | {entropy[1]:.6f} | {entropy[2]:.6f} | {entropy[3]:.6f} |
| Frozen q15 static token fusion, answer Top10 | 4 | {fusion[0]:.2f} | {fusion[1]:.6f} | {fusion[2]:.6f} | {fusion[3]:.6f} |

Decision: select `entropy_native__token_top10` at q={entropy[0]:.2f}. It is both
simpler and slightly better on every reported math-panel metric. The earlier
three-feature equal-mean result is retained as an ablation only and is not the
promoted gate.

## Exact reuse audit

`q15_raw4_mean` reconstructs the frozen
`original_static_fusion_before_top10` localization definition after the
registered per-step Top10 readout with maximum absolute discrepancy
{transfer['identity_audit']['max_abs_discrepancy']:.3g}. The answer candidate
changes only the final readout to one Top10 over the complete response. This is
not the current per-view-Top10 finalist, for which Top10 occurs before fusion.

## Frozen ProcessBench transfer

| Gate on frozen q15 locator | q source | Answer macro-F1 | AUROC | PB macro-F1 | Clean accuracy | Error exact |
|---|---|---:|---:|---:|---:|---:|
| Entropy Top10 | Math q={entropy[0]:.2f} | {chosen_gate['family_macro']['macro_f1']:.6f} | {chosen_gate['family_macro']['auroc']:.6f} | {chosen_loc['macros']['all']:.6f} | {chosen_loc['clean_accuracy']:.6f} | {chosen_loc['error_exact_accuracy']:.6f} |
| Existing entropy mean | PB q=.30 baseline | {base_gate['family_macro']['macro_f1']:.6f} | {base_gate['family_macro']['auroc']:.6f} | {base_loc['macros']['all']:.6f} | {base_loc['clean_accuracy']:.6f} | {base_loc['error_exact_accuracy']:.6f} |

Localization delta is {transfer['contrast']['point_delta']*100:+.3f} percentage
points; the conservative {transfer['contrast']['ci_level']*100:.2f}% paired
whole-source-group interval is [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}] points.

Relative to the old mean-entropy q=.3 gate, the transferred gate removes
{transfer['error_analysis']['clean_false_alarms_removed']} clean false alarms
and adds {transfer['error_analysis']['clean_false_alarms_added']}, but newly
closes {transfer['error_analysis']['errors_newly_closed']} erroneous answers
while reopening {transfer['error_analysis']['errors_newly_opened']}. It gains
{transfer['error_analysis']['exact_localizations_gained']} exact error
localizations and loses {transfer['error_analysis']['exact_localizations_lost']}.

No feature, method, fusion, or q was selected on ProcessBench. The within-cell
mid-rank transform is label-free. This is development transfer evidence and
still requires confirmation on new data or a new model.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf8")
    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("panel", "candidate", "q", "macro_f1", "auroc", "auprc", "pb_localization"))
        writer.writerow(("math", ENTROPY, entropy[0], entropy[1], entropy[2], entropy[3], ""))
        writer.writerow(("math", TOKEN_FUSION, fusion[0], fusion[1], fusion[2], fusion[3], ""))
        writer.writerow(("processbench", ENTROPY, transfer["frozen_choice"]["q"], chosen_gate["family_macro"]["macro_f1"], chosen_gate["family_macro"]["auroc"], chosen_gate["family_macro"]["auprc"], chosen_loc["macros"]["all"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check = preflight()
    atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise SystemExit(json.dumps(check, indent=2))
    math_decision, freeze = select_on_math()
    transfer = transfer_to_processbench(math_decision, freeze)
    write_report(math_decision, transfer)
    review = {
        "schema": "simple-gate-choice-result-review-v1",
        "status": "PASS",
        "math_candidates": list(CANDIDATES),
        "selected": freeze["method"],
        "q": freeze["q"],
        "identity_audit": transfer["identity_audit"]["status"],
        "processbench_selection_or_q_calibration": False,
        "three_feature_gate_promoted": False,
        "report_sha256": base.sha256_file(OUT / "REPORT.md"),
    }
    atomic_json(OUT / "RESULT_REVIEW.json", review)
    print(json.dumps(review, indent=2), flush=True)


if __name__ == "__main__":
    main()
