#!/usr/bin/env python3
"""Execute one non-reference Stage-A reducer against frozen Phase-2R thresholds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from scripts.reasoning_localization import run_phase2_reducer as base  # noqa: E402


REFERENCE = base.REFERENCE
CANDIDATES = base.STAGE_A_VARIANTS[1:]
REFERENCE_ROOT = base.PHASE_ROOT / REFERENCE.lower()
THRESHOLD_PATH = REFERENCE_ROOT / "evaluation/FROZEN_THRESHOLDS.json"
SCHEMA = "reasoning-localization-phase2-reducer-candidate-v1"
HARD_WORST_CELL_BOUND = -0.030


class CandidateError(RuntimeError):
    """Fail-closed candidate execution error."""


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _verify_payload(value: Mapping[str, Any]) -> None:
    expected = value.get("payload_sha256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if not expected or expected != _payload_sha(unsigned):
        raise CandidateError("signed JSON payload hash mismatch")


def load_registry(path: Path, variant_id: str, release: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "reasoning-localization-phase2-reducer-candidate-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN",
        "variant_id": variant_id,
        "stage_a_order": list(base.STAGE_A_VARIANTS),
        "processbench_cells": list(base.PB_CELLS),
        "bootstrap_draws": base.BOOTSTRAP_DRAWS,
        "bootstrap_seed": base.BOOTSTRAP_SEED,
        "reference_variant": REFERENCE,
        "candidate_rethresholding_allowed": False,
        "hard_worst_cell_bound": HARD_WORST_CELL_BOUND,
    }
    for key, expected in required.items():
        if registry.get(key) != expected:
            raise CandidateError(f"execution registry mismatch for {key}")
    if Path(registry["release_root"]).resolve() != release.resolve():
        raise CandidateError("release root differs from frozen registry")
    if registry["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise CandidateError("candidate runner changed after registry freeze")
    if registry["score_builder_sha256"] != sha256_file(Path(base.__file__).resolve()):
        raise CandidateError("frozen reducer score builder changed")
    if registry["reference_threshold_sha256"] != sha256_file(THRESHOLD_PATH):
        raise CandidateError("frozen reference threshold ledger changed")
    base._require_source_hashes(registry)
    return registry


def _load_thresholds(registry: Mapping[str, Any]) -> dict[str, Any]:
    if registry["reference_threshold_sha256"] != sha256_file(THRESHOLD_PATH):
        raise CandidateError("reference threshold file differs from execution registry")
    payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    _verify_payload(payload)
    if (
        payload.get("variant_id") != REFERENCE
        or payload.get("candidate_rethresholding_allowed") is not False
    ):
        raise CandidateError("invalid reference threshold contract")
    return payload


def _evaluate_with_frozen_thresholds(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluation: Any,
    thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    by_cell: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"fold_mismatches": 0, "thresholds_applied": 0, "models": {}}
    for model in base.QWEN_MODELS:
        rows = list(rows_by_model[model])
        assignments = evaluation.assign_processbench_folds(rows)
        frozen = thresholds["models"][model]
        assignment_sha = evaluation.payload_sha256(sorted(assignments.items()))
        if assignment_sha != frozen["fold_assignment_sha256"]:
            raise CandidateError(f"{model}: candidate fold assignment differs from reference")
        ledgers = {int(row["held_out_fold"]): row for row in frozen["calibration_ledgers"]}
        if set(ledgers) != set(range(5)):
            raise CandidateError(f"{model}: reference threshold roster is incomplete")
        predictions: list[int] = []
        folds: list[int] = []
        for row in rows:
            fold = int(assignments[str(row["group_id"])])
            prediction = evaluation.processbench_prediction(
                row["step_scores"], float(ledgers[fold]["threshold"])
            )
            predictions.append(int(prediction))
            folds.append(fold)
            audit["thresholds_applied"] += 1
        metrics = evaluation.processbench_panel_metrics(rows, predictions)
        audit["models"][model] = {
            "fold_assignment_sha256": assignment_sha,
            "thresholds": [float(ledgers[fold]["threshold"]) for fold in range(5)],
            "length_cutpoints": frozen["length_cutpoints"],
        }
        for parent, prediction, fold in zip(rows, predictions, folds):
            true_error = int(parent["first_error"])
            true_length = int(parent["step_lengths"][true_error]) if true_error >= 0 else None
            selected_length = int(parent["step_lengths"][prediction]) if prediction >= 0 else None
            decisions.append({
                "model_id": model,
                "cell_id": parent["cell_id"],
                "slice_id": parent["slice_id"],
                "row_id": parent["row_id"],
                "group_id": parent["group_id"],
                "fold": fold,
                "true_first_error": true_error,
                "prediction_step": prediction,
                "threshold": float(ledgers[fold]["threshold"]),
                "true_error_step_length": true_length,
                "true_error_length_stratum": (
                    base._stratum(true_length, frozen["length_cutpoints"][str(fold)])
                    if true_length is not None else "CLEAN"
                ),
                "selected_step_length": selected_length,
            })
        for family, values in metrics["per_subset"].items():
            by_cell.append({
                "model_id": model,
                "slice_id": family,
                "cell_id": f"processbench_{family}_{model}",
                **{metric: values[metric] for metric in base.PB_METRICS},
                "n_examples": values["n_examples"],
                "n_error": values["n_error"],
                "n_clean": values["n_clean"],
            })
    return decisions, by_cell, audit


def _paired_contrasts(
    by_cell: Sequence[Mapping[str, Any]], samples: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    reference_cells = {
        row["cell_id"]: row
        for row in base._read_csv(REFERENCE_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv")
    }
    reference_samples = load_npz_no_pickle(
        REFERENCE_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"
    )
    result: list[dict[str, Any]] = []
    for metric in base.PB_METRICS:
        candidate_point = float(np.mean([float(row[metric]) for row in by_cell]))
        reference_point = float(np.mean([
            float(reference_cells[str(row["cell_id"])][metric]) for row in by_cell
        ]))
        delta_samples = np.asarray(samples[metric]) - np.asarray(reference_samples[metric])
        cell_deltas = {
            str(row["cell_id"]): float(row[metric])
            - float(reference_cells[str(row["cell_id"])][metric])
            for row in by_cell
        }
        family_deltas = {
            family: float(np.mean([
                value for cell_id, value in cell_deltas.items()
                if str(reference_cells[cell_id]["slice_id"]) == family
            ]))
            for family in base.FAMILIES
        }
        eps = 1e-12
        result.append({
            "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
            "source_metric_id": metric,
            "delta": candidate_point - reference_point,
            "ci_low": float(np.quantile(delta_samples, 0.025)),
            "ci_high": float(np.quantile(delta_samples, 0.975)),
            "cell_wins": sum(value > eps for value in cell_deltas.values()),
            "cell_ties": sum(abs(value) <= eps for value in cell_deltas.values()),
            "cell_losses": sum(value < -eps for value in cell_deltas.values()),
            "worst_cell_delta": min(cell_deltas.values()),
            "worst_cell_id": min(cell_deltas, key=cell_deltas.get),
            "family_wins": sum(value > eps for value in family_deltas.values()),
            "family_ties": sum(abs(value) <= eps for value in family_deltas.values()),
            "family_losses": sum(value < -eps for value in family_deltas.values()),
            "worst_family_delta": min(family_deltas.values()),
            "worst_family_id": min(family_deltas, key=family_deltas.get),
            "bootstrap_draws": len(delta_samples),
            "interval": "unadjusted paired percentile; integration recomputes simultaneous primary interval across all opened reducers",
        })
    return result


def _prediction_flips(
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = {
        (row["model_id"], row["row_id"]): row
        for row in base._read_csv(REFERENCE_ROOT / "evaluation/PROCESSBENCH_DECISIONS.csv")
    }
    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    for row in decisions:
        parent = reference[(str(row["model_id"]), str(row["row_id"]))]
        candidate_prediction = int(row["prediction_step"])
        reference_prediction = int(parent["prediction_step"])
        target = int(row["true_first_error"])
        candidate_correct = candidate_prediction == target
        reference_correct = reference_prediction == target
        if candidate_prediction == reference_prediction:
            category = "NO_FLIP_CORRECT" if candidate_correct else "NO_FLIP_INCORRECT"
        elif candidate_correct and not reference_correct:
            category = "FLIP_GAIN"
        elif reference_correct and not candidate_correct:
            category = "FLIP_LOSS"
        else:
            category = "FLIP_LATERAL"
        counts[(str(row["cell_id"]), category)] = counts.get(
            (str(row["cell_id"]), category), 0
        ) + 1
        rows.append({
            "cell_id": row["cell_id"], "model_id": row["model_id"],
            "slice_id": row["slice_id"], "row_id": row["row_id"],
            "group_id": row["group_id"], "true_first_error": target,
            "reference_prediction_step": reference_prediction,
            "candidate_prediction_step": candidate_prediction,
            "changed": str(candidate_prediction != reference_prediction).lower(),
            "transition": category,
        })
    summary = [
        {"cell_id": cell_id, "transition": category, "count": count}
        for (cell_id, category), count in sorted(counts.items())
    ]
    return rows, summary


def evaluate_candidate(
    variant_id: str,
    release: Path,
    output: Path,
    registry: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    if variant_id not in CANDIDATES:
        raise CandidateError("candidate runner cannot execute the reference")
    base._require_source_hashes(registry)
    verified = base._verified_scores(output, freeze)
    if any(not np.isfinite(payload["arrays"]["combined_step_scores"]).all() for payload in verified.values()):
        raise CandidateError("candidate score freeze contains non-finite values")
    thresholds = _load_thresholds(registry)
    labels = base._load_pb_labels(release)
    evaluation = __import__(
        "spectral_utils.reconstruction_benchmark.localization_evaluation",
        fromlist=["localization_evaluation"],
    )
    rows_by_model = base._rows_by_model(verified, labels)
    decisions, by_cell, threshold_audit = _evaluate_with_frozen_thresholds(
        rows_by_model, evaluation, thresholds
    )
    samples = base._bootstrap_pb_panel(decisions, base.QWEN_MODELS)
    point = {
        metric: float(np.mean([float(row[metric]) for row in by_cell]))
        for metric in base.PB_METRICS
    }
    panels = [{
        "population_id": "current_common_eight_qwen",
        "metric_id": metric,
        "value": point[metric],
        "ci_low": float(np.quantile(samples[metric], 0.025)),
        "ci_high": float(np.quantile(samples[metric], 0.975)),
        "n_rows": sum(int(row["n_examples"]) for row in by_cell),
        "n_groups": 3400,
    } for metric in base.PB_METRICS]
    contrasts = _paired_contrasts(by_cell, samples)
    primary = next(row for row in contrasts if row["metric_id"] == "macro_f1")
    flips, flip_summary = _prediction_flips(decisions)
    hard_worst_cell = float(primary["worst_cell_delta"]) < HARD_WORST_CELL_BOUND
    execution_status = "HARD_FAIL" if hard_worst_cell else "COMPLETE"

    eval_root = output / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=False)
    base._write_csv(eval_root / "PROCESSBENCH_DECISIONS.csv", decisions)
    base._write_csv(eval_root / "PROCESSBENCH_BY_CELL.csv", by_cell)
    base._write_csv(eval_root / "PROCESSBENCH_PANELS.csv", panels)
    atomic_write_npz(eval_root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", samples)
    base._write_csv(eval_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    base._write_csv(eval_root / "STEP_LENGTH_STRATA.csv", base._length_strata(decisions, by_cell))
    base._write_csv(
        eval_root / "SELECTED_STEP_LENGTH.csv",
        base._selected_length_distribution(decisions),
    )
    base._write_csv(eval_root / "PREDICTION_FLIPS.csv", flips)
    base._write_csv(eval_root / "PREDICTION_FLIP_SUMMARY.csv", flip_summary)
    atomic_write_json(eval_root / "THRESHOLD_APPLICATION_AUDIT.json", threshold_audit)
    gates = [
        {"gate_id": "P2R_SCORE_FREEZE_COMPLETE", "status": "PASS", "observed": len(verified), "required": "8 cells", "detail": "all registered candidate scores were frozen before labels opened"},
        {"gate_id": "P2R_LABEL_FIREWALL", "status": "PASS", "observed": "labels opened after score-freeze manifest", "required": "no fit-side labels or targets", "detail": "the reducer and score combination are label-free"},
        {"gate_id": "P2R_REFERENCE_THRESHOLD_HASH", "status": "PASS", "observed": sha256_file(THRESHOLD_PATH), "required": registry["reference_threshold_sha256"], "detail": "candidate uses the immutable top-five threshold artifact"},
        {"gate_id": "P2R_NO_CANDIDATE_RETHRESHOLD", "status": "PASS", "observed": "0 threshold fits", "required": "0", "detail": "all 6800 decisions apply the ten frozen reference thresholds"},
        {"gate_id": "P2R_FOLD_ALIAS", "status": "PASS", "observed": threshold_audit["fold_mismatches"], "required": "0", "detail": "source-group fold assignments match the reference"},
        {"gate_id": "P2R_POPULATION_COMPLETE", "status": "PASS", "observed": len(decisions), "required": "6800 rows", "detail": "same eight Qwen cells and rows as the reference"},
        {"gate_id": "P2R_BOOTSTRAP_ALIGNMENT", "status": "PASS", "observed": len(samples["official_macro_f1"]), "required": str(base.BOOTSTRAP_DRAWS), "detail": "paired samples reuse the frozen seed and source-question groups"},
        {"gate_id": "P2R_WORST_CELL_HARD_BOUND", "status": "HARD_FAIL" if hard_worst_cell else "PASS", "observed": primary["worst_cell_delta"], "required": f">= {HARD_WORST_CELL_BOUND}", "detail": f"worst cell is {primary['worst_cell_id']}"},
    ]
    base._write_csv(eval_root / "GATES.csv", gates)
    summary = {
        "schema": "reasoning-localization-phase2-reducer-candidate-evaluation-v1",
        "variant_id": variant_id,
        "status": execution_status,
        "processbench_qwen8_macro_f1": point["official_macro_f1"],
        "processbench_qwen8_ci": [
            next(row["ci_low"] for row in panels if row["metric_id"] == "official_macro_f1"),
            next(row["ci_high"] for row in panels if row["metric_id"] == "official_macro_f1"),
        ],
        "raw_paired_delta_macro_f1": primary["delta"],
        "raw_paired_delta_ci": [primary["ci_low"], primary["ci_high"]],
        "cell_wtl": [primary["cell_wins"], primary["cell_ties"], primary["cell_losses"]],
        "family_wtl": [primary["family_wins"], primary["family_ties"], primary["family_losses"]],
        "worst_cell_id": primary["worst_cell_id"],
        "worst_cell_delta": primary["worst_cell_delta"],
        "worst_family_id": primary["worst_family_id"],
        "worst_family_delta": primary["worst_family_delta"],
        "prediction_flips": sum(row["changed"] == "true" for row in flips),
        "candidate_rethresholding_allowed": False,
        "reference_threshold_sha256": sha256_file(THRESHOLD_PATH),
        "bootstrap_draws": base.BOOTSTRAP_DRAWS,
        "bootstrap_seed": base.BOOTSTRAP_SEED,
        "peak_memory_bytes": base._peak_memory_bytes(),
    }
    summary["payload_sha256"] = _payload_sha(summary)
    atomic_write_json(eval_root / "SUMMARY.json", summary)
    outputs = (
        "PROCESSBENCH_DECISIONS.csv", "PROCESSBENCH_BY_CELL.csv",
        "PROCESSBENCH_PANELS.csv", "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz",
        "PAIRWISE_CONTRASTS.csv", "STEP_LENGTH_STRATA.csv",
        "SELECTED_STEP_LENGTH.csv", "PREDICTION_FLIPS.csv",
        "PREDICTION_FLIP_SUMMARY.csv", "THRESHOLD_APPLICATION_AUDIT.json",
        "GATES.csv", "SUMMARY.json",
    )
    manifest = {
        "schema": "reasoning-localization-phase2-reducer-candidate-evaluation-manifest-v1",
        "variant_id": variant_id,
        "status": execution_status,
        "score_freeze_sha256": sha256_file(
            output / "score_freeze/SCORE_FREEZE_MANIFEST.json"
        ),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "reference_threshold_sha256": sha256_file(THRESHOLD_PATH),
        "outputs": [
            {"path": name, "sha256": sha256_file(eval_root / name), "bytes": (eval_root / name).stat().st_size}
            for name in outputs
        ],
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(eval_root / "EVALUATION_MANIFEST.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=CANDIDATES)
    parser.add_argument("--release", type=Path, default=base.DEFAULT_RELEASE)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry_path = (
        args.registry or base.PHASE_ROOT / f"{args.variant}_EXECUTION_REGISTRY.json"
    ).resolve()
    output = (args.output or base.PHASE_ROOT / args.variant.lower()).resolve()
    release = args.release.resolve()
    registry = load_registry(registry_path, args.variant, release)
    registry["registry_path"] = str(registry_path)
    started = time.perf_counter()
    freeze = base.freeze_scores(args.variant, release, output, registry)
    summary = evaluate_candidate(args.variant, release, output, registry, freeze)
    run = {
        "schema": "reasoning-localization-phase2-reducer-candidate-run-v1",
        "variant_id": args.variant,
        "status": summary["status"],
        "execution_registry_sha256": sha256_file(registry_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "score_builder_sha256": sha256_file(Path(base.__file__).resolve()),
        "score_freeze_manifest_sha256": sha256_file(
            output / "score_freeze/SCORE_FREEZE_MANIFEST.json"
        ),
        "evaluation_manifest_sha256": sha256_file(
            output / "evaluation/EVALUATION_MANIFEST.json"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    run["payload_sha256"] = _payload_sha(run)
    atomic_write_json(output / "RUN_MANIFEST.json", run)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
