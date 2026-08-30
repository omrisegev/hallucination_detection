#!/usr/bin/env python3
"""Run Phase-0 state S2A: replace only the historical answer detector.

The parent is the completed P0-S1 family6-level / step-max bridge.  This
runner first reconstructs that parent from the exact historical sources, then
keeps its rows, roles, local representation, fitted local head, step locator,
and threshold objective fixed.  The sole scientific change is the answer-risk
head: RegisteredGlobal mixed-v2 ordinary IU becomes the preregistered
calibration-only mixed-v2 DUFS-LIU detector (k=7, lambda=0.1, seeds
11/23/37, 80 epochs).  P0-S2B's purely local detector is intentionally not
opened by this runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.reasoning_localization import run_phase0_reducer_bridge as s1  # noqa: E402
from spectral_utils.online_localization_fusion import (  # noqa: E402
    GLOBAL_DUFS_EPOCHS,
    GLOBAL_DUFS_SEEDS,
    GLOBAL_K,
    GLOBAL_LAMBDA,
    fit_frozen_global_gl_liu,
)


STATE_ID = "P0_S2A_MODERN_DETECTOR_BRIDGE"
VARIANT_ID = "P0_S2A_FAMILY6_STEP_MAX_DUFS_DETECTOR"
PARENT_VARIANT_ID = "P0_S1_FAMILY6_STEP_MAX"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082901
NUMERIC_TOLERANCE = 1e-12
METRIC_FIELDS = s1.METRIC_FIELDS
DEFAULT_OUTPUT = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "p0_s2a_detector_bridge"
)
DEFAULT_REGISTRY = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "P0_S2A_EXECUTION_REGISTRY.json"
)


class BridgeError(RuntimeError):
    """Raised when the frozen S2A contract or S1 reconstruction fails."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise BridgeError(f"missing {label}: {path}")
    observed = s1.sha256_file(path)
    if observed != expected:
        raise BridgeError(
            f"{label} SHA mismatch: {path}\nexpected={expected}\nobserved={observed}"
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("state_id") != STATE_ID:
        raise BridgeError("execution registry state_id mismatch")
    if registry.get("status") != "FROZEN_BEFORE_RUN":
        raise BridgeError("execution registry must be FROZEN_BEFORE_RUN")
    require_hash(Path(__file__).resolve(), registry["runner_sha256"], "frozen runner")
    if registry.get("bootstrap_draws") != BOOTSTRAP_DRAWS:
        raise BridgeError("bootstrap draw count differs from the frozen S2A contract")
    if registry.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise BridgeError("bootstrap seed differs from the frozen S2A contract")
    if registry.get("single_changed_factor") != {
        "factor": "answer_detector",
        "from": "RegisteredGlobal mixed-v2 ordinary IU",
        "to": "calibration-only answer_dufs_liu_mixed",
    }:
        raise BridgeError("S2A must change exactly the registered detector factor")
    expected = {
        "feature_contract": "mixed-v2",
        "fit_population": "historical calibration rows only",
        "include_elapsed_length": False,
        "k": GLOBAL_K,
        "lambda": GLOBAL_LAMBDA,
        "dufs_seeds": list(GLOBAL_DUFS_SEEDS),
        "dufs_epochs": GLOBAL_DUFS_EPOCHS,
        "labels_seen_during_fit": False,
    }
    if registry.get("modern_detector") != expected:
        raise BridgeError("modern detector parameters differ from the frozen S2A contract")
    return registry


def preflight_sources(registry: Mapping[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    s1_spec = registry["s1_execution_registry"]
    s1_path = resolve_path(s1_spec["path"])
    require_hash(s1_path, s1_spec["sha256"], "S1 execution registry")
    s1_registry = s1.load_registry(s1_path)
    inherited = s1.preflight_sources(s1_registry)
    sources.extend(inherited["sources"])
    sources.append({
        "role": "s1_execution_registry", "path": s1_spec["path"],
        "sha256": s1_spec["sha256"], "bytes": s1_path.stat().st_size,
    })
    for role in ("s1_artifacts", "code_dependencies"):
        for spec in registry[role]:
            path = resolve_path(spec["path"])
            require_hash(path, spec["sha256"], f"{role} {spec['path']}")
            sources.append({
                "role": role[:-1], "path": spec["path"],
                "sha256": spec["sha256"], "bytes": path.stat().st_size,
            })
    return {
        "sources": sources,
        "sources_sha256": s1.canonical_sha256(sources),
        "inherited_s1_sources_sha256": inherited["sources_sha256"],
    }


def artifact_by_role(registry: Mapping[str, Any], role: str) -> Path:
    matches = [row for row in registry["s1_artifacts"] if row["role"] == role]
    if len(matches) != 1:
        raise BridgeError(f"expected one registered S1 artifact for role={role}")
    return resolve_path(matches[0]["path"])


def compare_parent_cell(
    model: str,
    family: str,
    reconstructed_records: Sequence[Mapping[str, Any]],
    reconstructed_metric: Mapping[str, Any],
    frozen_records: Sequence[Mapping[str, Any]],
    frozen_metric: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        row["unit"]: row for row in frozen_records
        if row["model"] == model and row["family"] == family
    }
    observed = {str(row["unit"]): row for row in reconstructed_records}
    if set(expected) != set(observed):
        raise BridgeError(f"S1 unit mismatch for {model}/{family}")
    locator_mismatches = prediction_mismatches = target_mismatches = 0
    max_score_delta = 0.0
    for unit in sorted(expected):
        left, right = observed[unit], expected[unit]
        locator_mismatches += int(int(left["locator"]) != int(right["locator"]))
        prediction_mismatches += int(int(left["prediction"]) != int(right["prediction"]))
        target_mismatches += int(int(left["target"]) != int(right["target"]))
        max_score_delta = max(max_score_delta, abs(float(left["score"]) - float(right["score"])))
    metric_delta = max(
        abs(float(reconstructed_metric[field]) - float(frozen_metric[field]))
        for field in (*METRIC_FIELDS, "threshold", "calibration_f1")
    )
    if (
        locator_mismatches or prediction_mismatches or target_mismatches
        or max_score_delta > NUMERIC_TOLERANCE
        or metric_delta > NUMERIC_TOLERANCE
    ):
        raise BridgeError(
            f"S1 reconstruction hard failure for {model}/{family}: "
            f"locator={locator_mismatches}, prediction={prediction_mismatches}, "
            f"target={target_mismatches}, score_delta={max_score_delta}, "
            f"metric_delta={metric_delta}"
        )
    return {
        "model": model, "family": family, "n_audit": len(observed),
        "s1_locator_mismatches": locator_mismatches,
        "s1_prediction_mismatches": prediction_mismatches,
        "s1_target_mismatches": target_mismatches,
        "s1_max_abs_score_delta": max_score_delta,
        "s1_metric_max_abs_delta": metric_delta,
        "s1_reconstruction_exact": True,
    }


def run_cell(
    model: str,
    family: str,
    raw_spec: Mapping[str, Any],
    checkpoint_spec: Mapping[str, Any],
    frozen_records: Sequence[Mapping[str, Any]],
    frozen_metric: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any],
]:
    parent_records, parent_metrics, s0_audit, _, _, population_cell = s1.run_cell(
        model, family, raw_spec, checkpoint_spec
    )
    if len(parent_metrics) != 1:
        raise BridgeError(f"unexpected reconstructed S1 metric count for {model}/{family}")
    parent_metric = parent_metrics[0]
    s1_audit = compare_parent_cell(
        model, family, parent_records, parent_metric,
        frozen_records, frozen_metric,
    )

    rows = s1.load_rows(s1.resolve_registered_path(raw_spec))
    for row in rows:
        row["_stage"] = s1._stage_partition(family, row["_unit"])
    calibration = [row for row in rows if row["_stage"] == "calibration"]
    audit = [row for row in rows if row["_stage"] == "audit"]
    if not calibration or not audit:
        raise BridgeError(f"empty calibration/audit role for {model}/{family}")

    # Rebuild the unchanged family6-level calibration locator.  Audit locators
    # are taken from the checksum-verified S1 parent records.
    references = s1.fit_references(calibration)
    prepared_cal = [s1.prepare_trace(row, references) for row in calibration]
    local_head = s1.fit_trajectory_head_prepared(
        prepared_cal, name="finalist_local", representation="family6",
        operators=("level",),
    )
    cal_curves = [
        local_head.curve_from_level(item.representations["family6"])
        for item in prepared_cal
    ]
    cal_locator = np.asarray([
        s1._peak_locator(curve, row) for curve, row in zip(cal_curves, calibration)
    ])

    detector = fit_frozen_global_gl_liu(
        calibration,
        include_elapsed_length=False,
        dufs_epochs=GLOBAL_DUFS_EPOCHS,
    )
    if detector.diagnostics.get("labels_seen_during_fit") is not False:
        raise BridgeError("modern detector did not preserve the label-free fit contract")
    if detector.diagnostics.get("head") != "answer_dufs_liu_mixed":
        raise BridgeError("unexpected modern detector head")
    cal_score = np.asarray([detector.risk(row, None) for row in calibration])
    audit_score = np.asarray([detector.risk(row, None) for row in audit])
    cal_target = np.asarray([int(row["label"]) for row in calibration])
    audit_target = np.asarray([int(row["label"]) for row in audit])

    parent_lookup = {str(row["unit"]): dict(row) for row in parent_records}
    audit_units = [str(row["_unit"]) for row in audit]
    if set(parent_lookup) != set(audit_units):
        raise BridgeError(f"S1 audit order/coverage mismatch for {model}/{family}")
    ordered_parent = [parent_lookup[unit] for unit in audit_units]
    audit_locator = np.asarray([int(row["locator"]) for row in ordered_parent])
    threshold, calibration_f1 = s1._best_threshold(
        cal_score, cal_locator, cal_target
    )
    prediction = np.where(audit_score > threshold, audit_locator, -1)
    result = s1._processbench(prediction, audit_target)

    candidate_records: list[dict[str, Any]] = []
    normalized_parent: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    for row, target, score, locator, predicted, parent in zip(
        audit, audit_target, audit_score, audit_locator, prediction, ordered_parent
    ):
        candidate = dict(parent)
        candidate.update({
            "candidate": VARIANT_ID,
            "score": float(score),
            "locator": int(locator),
            "prediction": int(predicted),
            "detector": "answer_dufs_liu_mixed_calibration_only",
            "reducer": "step_max_token_argmax",
        })
        candidate_records.append(candidate)
        parent_row = dict(parent)
        parent_row["candidate"] = PARENT_VARIANT_ID
        normalized_parent.append(parent_row)
        flips.append({
            "model": model, "family": family, "unit": row["_unit"],
            "target": int(target), "s1_locator": int(parent["locator"]),
            "s2a_locator": int(locator),
            "s1_prediction": int(parent["prediction"]),
            "s2a_prediction": int(predicted),
            "flip_kind": s1.flip_kind(
                int(target), int(parent["prediction"]), int(predicted)
            ),
        })
    metric = {
        "candidate": VARIANT_ID, "model": model, "family": family,
        "task": "local", "primary": result["f1"], **result,
        "threshold": float(threshold), "calibration_f1": float(calibration_f1),
        "n": len(audit), "access_tier": "A",
        "detector": "answer_dufs_liu_mixed_calibration_only",
        "reducer": "step_max_token_argmax",
    }
    detector_audit = {
        "model": model, "family": family,
        "head": detector.diagnostics["head"],
        "n_fit_traces": detector.diagnostics["n_fit_traces"],
        "n_features": detector.diagnostics["n_features"],
        "include_elapsed_length": detector.diagnostics["include_elapsed_length"],
        "dufs_epochs": detector.diagnostics["dufs_epochs"],
        "dufs_seeds": detector.diagnostics["dufs_seeds"],
        "k": detector.diagnostics["k"],
        "lambda": detector.diagnostics["lambda"],
        "labels_seen_during_fit": detector.diagnostics["labels_seen_during_fit"],
        "feature_names": detector.diagnostics["feature_names"],
        "dufs_effective_feature_count": detector.diagnostics["dufs_effective_feature_count"],
    }
    reconstruction = {**s1_audit, "s0_reconstruction_exact_inside_s1": s0_audit["s0_reconstruction_exact"]}
    return (
        candidate_records, [metric], normalized_parent, flips,
        reconstruction, detector_audit, population_cell,
    )


def aggregate_metrics(cell_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = s1.aggregate_metrics(cell_metrics)
    for row in rows:
        row["variant_id"] = VARIANT_ID
    return rows


def paired_contrasts(
    candidate_records: Sequence[Mapping[str, Any]],
    parent_records: Sequence[Mapping[str, Any]],
    candidate_cells: Sequence[Mapping[str, Any]],
    parent_cell_metrics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Reuse the already-tested paired bootstrap implementation under temporary
    # method aliases, then restore the S2A lineage labels.
    candidate_alias = [
        {**row, "candidate": s1.VARIANT_ID} for row in candidate_records
    ]
    parent_alias = [
        {**row, "candidate": s1.PARENT_VARIANT_ID} for row in parent_records
    ]
    contrasts, family_rows = s1.paired_contrasts(
        candidate_alias, parent_alias, candidate_cells, parent_cell_metrics
    )
    for row in contrasts:
        row["left_variant_id"] = VARIANT_ID
        row["right_variant_id"] = PARENT_VARIANT_ID
    return contrasts, family_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = args.registry.resolve()
    registry = load_registry(registry_path)
    source_manifest = preflight_sources(registry)
    if args.preflight_only:
        print(json.dumps({
            "state_id": STATE_ID, "status": "PREFLIGHT_PASS", **source_manifest,
        }, indent=2, sort_keys=True))
        return 0

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise BridgeError(f"output directory must be new or empty: {output}")

    s1_registry = s1.load_registry(resolve_path(registry["s1_execution_registry"]["path"]))
    raw_specs, checkpoint_specs = s1.load_cell_specs(s1_registry)
    frozen_records = read_csv(artifact_by_role(registry, "per_question"))
    frozen_cell_metrics = read_csv(artifact_by_role(registry, "cell_metrics"))
    frozen_metric_lookup = {
        (row["model"], row["family"]): row for row in frozen_cell_metrics
    }
    frozen_population = json.loads(
        artifact_by_role(registry, "population").read_text(encoding="utf-8")
    )

    candidate_records: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    detector_audits: list[dict[str, Any]] = []
    population_cells: list[dict[str, Any]] = []
    for model in s1.MODELS:
        for family in s1.FAMILIES:
            print(f"P0-S2A {model}/{family}: S1 reconstruction and detector bridge", flush=True)
            cell = run_cell(
                model, family,
                raw_specs[(model, family)], checkpoint_specs[(model, family)],
                frozen_records, frozen_metric_lookup[(model, family)],
            )
            records, metrics, parents, cell_flips, reconstruction, detector_audit, population_cell = cell
            candidate_records.extend(records)
            candidate_metrics.extend(metrics)
            parent_records.extend(parents)
            flips.extend(cell_flips)
            reconstruction_rows.append(reconstruction)
            detector_audits.append(detector_audit)
            population_cells.append(population_cell)

    population = dict(frozen_population)
    population["state_id"] = STATE_ID
    population["cells"] = population_cells
    expected = registry["expected_population"]
    if any(population[key] != expected[key] for key in (
        "n_cells", "n_scorer_rows", "n_source_question_groups",
        "source_question_group_sha256",
    )):
        raise BridgeError(f"S2A population differs from frozen S1: {population}")
    if len(candidate_records) != population["n_scorer_rows"]:
        raise BridgeError("S2A scorer-row count differs from the frozen population")

    aggregate = aggregate_metrics(candidate_metrics)
    contrasts, family_deltas = paired_contrasts(
        candidate_records, parent_records, candidate_metrics, frozen_metric_lookup
    )
    flip_counts = {
        kind: sum(row["flip_kind"] == kind for row in flips)
        for kind in sorted({row["flip_kind"] for row in flips})
    }
    source_manifest_after = preflight_sources(registry)
    if source_manifest_after != source_manifest:
        raise BridgeError("registered source changed during the read-only S2A run")

    output.mkdir(parents=True, exist_ok=True)
    s1.write_csv(output / "P0_S2A_LOCAL_PER_QUESTION.csv", candidate_records)
    s1.write_csv(output / "P0_S2A_LOCAL_CELL_METRICS.csv", candidate_metrics)
    s1.write_csv(output / "P0_S2A_LOCAL_AGGREGATE.csv", aggregate)
    s1.write_csv(output / "P0_S2A_CONTRASTS.csv", contrasts)
    s1.write_csv(output / "P0_S2A_FAMILY_DELTAS.csv", family_deltas)
    s1.write_csv(output / "P0_S2A_RECONSTRUCTION_AUDIT.csv", reconstruction_rows)
    s1.write_csv(output / "P0_S2A_DETECTOR_AUDIT.csv", detector_audits)
    s1.write_csv(output / "P0_S2A_PREDICTION_FLIPS.csv", flips)
    s1.write_csv(
        output / "P0_S2A_PREDICTION_FLIP_SUMMARY.csv",
        [{"flip_kind": kind, "count": count} for kind, count in sorted(flip_counts.items())],
    )
    s1.write_json(output / "P0_S2A_POPULATION.json", population)
    gates = [
        {"gate_id": "P0_S2A_S1_RECONSTRUCTION_EXACT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2A_ONE_FACTOR_DETECTOR_ONLY", "observed": "answer_dufs_liu_mixed_calibration_only", "threshold": "answer_dufs_liu_mixed_calibration_only", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2A_POPULATION_HASH_UNCHANGED", "observed": population["source_question_group_sha256"], "threshold": expected["source_question_group_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2A_LABEL_FREE_DETECTOR_FIT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2A_BOOTSTRAP_DRAWS", "observed": BOOTSTRAP_DRAWS, "threshold": BOOTSTRAP_DRAWS, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2A_SOURCE_HASHES_STABLE", "observed": source_manifest_after["sources_sha256"], "threshold": source_manifest["sources_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
    ]
    s1.write_csv(output / "P0_S2A_GATES.csv", gates)
    verification = {
        "schema": "reasoning_localization_p0_s2a_verification_v1",
        "state_id": STATE_ID, "status": "COMPLETE",
        "variant_id": VARIANT_ID, "parent_variant_id": PARENT_VARIANT_ID,
        "single_changed_factor": registry["single_changed_factor"],
        "s1_reconstruction_exact": all(row["s1_reconstruction_exact"] for row in reconstruction_rows),
        "aggregate": aggregate, "contrasts": contrasts,
        "prediction_flip_counts": flip_counts,
        "detector_audits": detector_audits,
        "population_sha256": population["source_question_group_sha256"],
        "source_manifest": source_manifest,
        "new_model_inference": False, "gpu_hours": 0,
        "label_free_method_fit": True, "source_mutation": False,
        "s2b_opened": False,
    }
    s1.write_json(output / "P0_S2A_VERIFICATION.json", verification)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    run_manifest = {
        "schema": "reasoning_localization_p0_s2a_run_manifest_v1",
        "state_id": STATE_ID, "status": "COMPLETE",
        "source_commit": s1.git_head(),
        "runner_sha256": s1.sha256_file(Path(__file__).resolve()),
        "execution_registry_sha256": s1.sha256_file(registry_path),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "new_model_inference": False, "gpu_hours": 0,
        "label_free_method_fit": True, "source_mutation": False,
        "outputs": [
            {"file": path.name, "sha256": s1.sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    s1.write_json(output / "RUN_MANIFEST.json", run_manifest)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BridgeError, s1.BridgeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
