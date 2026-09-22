#!/usr/bin/env python3
"""Run Phase-0 state S3A: replace only family6 risk with raw entropy.

The parent is P0-S2B.  Historical rows and roles, the purely-local detector
construction, step-max reduction, threshold objective, access tier, and paired
bootstrap remain fixed.  Both detector and locator consume the same curve in
both states; the sole scientific change is family6-level fitted risk -> raw
token entropy.
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

from scripts.reasoning_localization import run_phase0_detector_bridge as s2a  # noqa: E402
from scripts.reasoning_localization import run_phase0_pure_local_detector_bridge as s2b  # noqa: E402
from scripts.reasoning_localization import run_phase0_reducer_bridge as s1  # noqa: E402


STATE_ID = "P0_S3A_RAW_ENTROPY_REPRESENTATION_BRIDGE"
VARIANT_ID = "P0_S3A_RAW_ENTROPY_STEP_MAX_LOCAL_DETECTOR"
PARENT_VARIANT_ID = "P0_S2B_FAMILY6_STEP_MAX_LOCAL_DETECTOR"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082901
NUMERIC_TOLERANCE = 1e-12
METRIC_FIELDS = s1.METRIC_FIELDS
DEFAULT_OUTPUT = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "p0_s3a_raw_entropy_representation_bridge"
)
DEFAULT_REGISTRY = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "P0_S3A_EXECUTION_REGISTRY.json"
)


class BridgeError(RuntimeError):
    """Raised when the frozen S3A contract or S2B reconstruction fails."""


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
        raise BridgeError("bootstrap draw count differs from frozen S3A")
    if registry.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise BridgeError("bootstrap seed differs from frozen S3A")
    if registry.get("single_changed_factor") != {
        "factor": "local_risk_representation",
        "from": "fitted family6 level risk curve",
        "to": "raw token entropy",
    }:
        raise BridgeError("S3A must change exactly the registered representation")
    if registry.get("raw_entropy_contract") != {
        "curve": "token_entropies",
        "orientation": "higher_is_riskier",
        "answer_detector": "max_t token_entropy[t]",
        "step_locator": "token argmax mapped to known step span",
        "labels_seen_during_representation_construction": False,
    }:
        raise BridgeError("raw-entropy contract differs from frozen S3A")
    return registry


def preflight_sources(registry: Mapping[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    spec = registry["s2b_execution_registry"]
    path = resolve_path(spec["path"])
    require_hash(path, spec["sha256"], "S2B execution registry")
    parent_registry = s2b.load_registry(path)
    inherited = s2b.preflight_sources(parent_registry)
    sources.extend(inherited["sources"])
    sources.append({
        "role": "s2b_execution_registry", "path": spec["path"],
        "sha256": spec["sha256"], "bytes": path.stat().st_size,
    })
    for role in ("s2b_artifacts", "code_dependencies"):
        for item in registry[role]:
            item_path = resolve_path(item["path"])
            require_hash(item_path, item["sha256"], f"{role} {item['path']}")
            sources.append({
                "role": role[:-1], "path": item["path"],
                "sha256": item["sha256"], "bytes": item_path.stat().st_size,
            })
    return {
        "sources": sources,
        "sources_sha256": s1.canonical_sha256(sources),
        "inherited_s2b_sources_sha256": inherited["sources_sha256"],
    }


def artifact_by_role(registry: Mapping[str, Any], role: str) -> Path:
    matches = [row for row in registry["s2b_artifacts"] if row["role"] == role]
    if len(matches) != 1:
        raise BridgeError(f"expected one registered S2B artifact for role={role}")
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
        raise BridgeError(f"S2B unit mismatch for {model}/{family}")
    locator_mismatches = prediction_mismatches = target_mismatches = 0
    max_score_delta = 0.0
    for unit in sorted(expected):
        left, right = observed[unit], expected[unit]
        locator_mismatches += int(int(left["locator"]) != int(right["locator"]))
        prediction_mismatches += int(int(left["prediction"]) != int(right["prediction"]))
        target_mismatches += int(int(left["target"]) != int(right["target"]))
        max_score_delta = max(
            max_score_delta, abs(float(left["score"]) - float(right["score"]))
        )
    metric_delta = max(
        abs(float(reconstructed_metric[field]) - float(frozen_metric[field]))
        for field in (*METRIC_FIELDS, "threshold", "calibration_f1")
    )
    if (
        locator_mismatches or prediction_mismatches or target_mismatches
        or max_score_delta > NUMERIC_TOLERANCE or metric_delta > NUMERIC_TOLERANCE
    ):
        raise BridgeError(
            f"S2B reconstruction hard failure for {model}/{family}: "
            f"locator={locator_mismatches}, prediction={prediction_mismatches}, "
            f"target={target_mismatches}, score_delta={max_score_delta}, "
            f"metric_delta={metric_delta}"
        )
    return {
        "model": model, "family": family, "n_audit": len(observed),
        "s2b_locator_mismatches": locator_mismatches,
        "s2b_prediction_mismatches": prediction_mismatches,
        "s2b_target_mismatches": target_mismatches,
        "s2b_max_abs_score_delta": max_score_delta,
        "s2b_metric_max_abs_delta": metric_delta,
        "s2b_reconstruction_exact": True,
    }


def run_cell(
    model: str,
    family: str,
    raw_spec: Mapping[str, Any],
    checkpoint_spec: Mapping[str, Any],
    frozen_s1_records: Sequence[Mapping[str, Any]],
    frozen_s1_metric: Mapping[str, Any],
    frozen_s2a_records: Sequence[Mapping[str, Any]],
    frozen_s2a_metric: Mapping[str, Any],
    frozen_s2b_records: Sequence[Mapping[str, Any]],
    frozen_s2b_metric: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], dict[str, Any], dict[str, Any],
]:
    parent = s2b.run_cell(
        model, family, raw_spec, checkpoint_spec,
        frozen_s1_records, frozen_s1_metric,
        frozen_s2a_records, frozen_s2a_metric,
    )
    parent_records, parent_metrics, _, _, s2b_audit, _, population_cell = parent
    if len(parent_metrics) != 1:
        raise BridgeError(f"unexpected S2B metric count for {model}/{family}")
    reconstruction = compare_parent_cell(
        model, family, parent_records, parent_metrics[0],
        frozen_s2b_records, frozen_s2b_metric,
    )
    reconstruction["s2a_reconstruction_exact_inside_s2b"] = s2b_audit["s2a_reconstruction_exact"]

    rows = s1.load_rows(s1.resolve_registered_path(raw_spec))
    for row in rows:
        row["_stage"] = s1._stage_partition(family, row["_unit"])
    calibration = [row for row in rows if row["_stage"] == "calibration"]
    audit = [row for row in rows if row["_stage"] == "audit"]
    cal_curves = [np.asarray(row["token_entropies"], dtype=float) for row in calibration]
    audit_curves = [np.asarray(row["token_entropies"], dtype=float) for row in audit]
    if any(curve.ndim != 1 or not len(curve) or not np.isfinite(curve).all() for curve in (*cal_curves, *audit_curves)):
        raise BridgeError(f"invalid raw entropy curve for {model}/{family}")
    cal_score = np.asarray([float(np.max(curve)) for curve in cal_curves])
    audit_score = np.asarray([float(np.max(curve)) for curve in audit_curves])
    cal_locator = np.asarray([
        s1._peak_locator(curve, row) for curve, row in zip(cal_curves, calibration)
    ])
    audit_locator = np.asarray([
        s1._peak_locator(curve, row) for curve, row in zip(audit_curves, audit)
    ])
    cal_target = np.asarray([int(row["label"]) for row in calibration])
    audit_target = np.asarray([int(row["label"]) for row in audit])
    threshold, calibration_f1 = s1._best_threshold(cal_score, cal_locator, cal_target)
    prediction = np.where(audit_score > threshold, audit_locator, -1)
    result = s1._processbench(prediction, audit_target)

    parent_lookup = {str(row["unit"]): dict(row) for row in parent_records}
    audit_units = [str(row["_unit"]) for row in audit]
    if set(parent_lookup) != set(audit_units):
        raise BridgeError(f"S2B audit order/coverage mismatch for {model}/{family}")
    ordered_parent = [parent_lookup[unit] for unit in audit_units]
    candidate_records: list[dict[str, Any]] = []
    normalized_parent: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    for row, target, score, locator, predicted, parent_row in zip(
        audit, audit_target, audit_score, audit_locator, prediction, ordered_parent
    ):
        candidate = dict(parent_row)
        candidate.update({
            "candidate": VARIANT_ID, "score": float(score),
            "locator": int(locator), "prediction": int(predicted),
            "detector": "raw_entropy_max", "representation": "raw_entropy",
            "reducer": "step_max_token_argmax",
        })
        candidate_records.append(candidate)
        normalized = dict(parent_row)
        normalized["candidate"] = PARENT_VARIANT_ID
        normalized_parent.append(normalized)
        flips.append({
            "model": model, "family": family, "unit": row["_unit"],
            "target": int(target),
            "s2b_locator": int(parent_row["locator"]), "s3a_locator": int(locator),
            "s2b_prediction": int(parent_row["prediction"]), "s3a_prediction": int(predicted),
            "locator_changed": int(parent_row["locator"]) != int(locator),
            "flip_kind": s1.flip_kind(
                int(target), int(parent_row["prediction"]), int(predicted)
            ),
        })
    metric = {
        "candidate": VARIANT_ID, "model": model, "family": family,
        "task": "local", "primary": result["f1"], **result,
        "threshold": float(threshold), "calibration_f1": float(calibration_f1),
        "n": len(audit), "access_tier": "A", "detector": "raw_entropy_max",
        "representation": "raw_entropy", "reducer": "step_max_token_argmax",
    }
    representation_audit = {
        "model": model, "family": family,
        "representation": "raw_entropy", "curve_field": "token_entropies",
        "n_calibration": len(calibration), "n_audit": len(audit),
        "labels_seen_during_representation_construction": False,
        "all_curves_finite": True,
        "locator_changes_vs_s2b": sum(row["locator_changed"] for row in flips),
    }
    return (
        candidate_records, [metric], normalized_parent, flips,
        reconstruction, representation_audit, population_cell,
    )


def paired_contrasts(
    candidate_records: Sequence[Mapping[str, Any]],
    parent_records: Sequence[Mapping[str, Any]],
    candidate_cells: Sequence[Mapping[str, Any]],
    parent_cell_metrics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_alias = [{**row, "candidate": s1.VARIANT_ID} for row in candidate_records]
    parent_alias = [{**row, "candidate": s1.PARENT_VARIANT_ID} for row in parent_records]
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

    s2b_registry = s2b.load_registry(resolve_path(registry["s2b_execution_registry"]["path"]))
    s2a_registry = s2a.load_registry(resolve_path(s2b_registry["s2a_execution_registry"]["path"]))
    s1_registry = s1.load_registry(resolve_path(s2a_registry["s1_execution_registry"]["path"]))
    raw_specs, checkpoint_specs = s1.load_cell_specs(s1_registry)
    frozen_s1_records = read_csv(s2a.artifact_by_role(s2a_registry, "per_question"))
    frozen_s1_metrics = read_csv(s2a.artifact_by_role(s2a_registry, "cell_metrics"))
    frozen_s1_lookup = {(r["model"], r["family"]): r for r in frozen_s1_metrics}
    frozen_s2a_records = read_csv(s2b.artifact_by_role(s2b_registry, "per_question"))
    frozen_s2a_metrics = read_csv(s2b.artifact_by_role(s2b_registry, "cell_metrics"))
    frozen_s2a_lookup = {(r["model"], r["family"]): r for r in frozen_s2a_metrics}
    frozen_s2b_records = read_csv(artifact_by_role(registry, "per_question"))
    frozen_s2b_metrics = read_csv(artifact_by_role(registry, "cell_metrics"))
    frozen_s2b_lookup = {(r["model"], r["family"]): r for r in frozen_s2b_metrics}
    frozen_population = json.loads(artifact_by_role(registry, "population").read_text(encoding="utf-8"))

    candidate_records: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    representation_audits: list[dict[str, Any]] = []
    population_cells: list[dict[str, Any]] = []
    for model in s1.MODELS:
        for family in s1.FAMILIES:
            print(f"P0-S3A {model}/{family}: S2B reconstruction and raw-entropy bridge", flush=True)
            cell = run_cell(
                model, family, raw_specs[(model, family)], checkpoint_specs[(model, family)],
                frozen_s1_records, frozen_s1_lookup[(model, family)],
                frozen_s2a_records, frozen_s2a_lookup[(model, family)],
                frozen_s2b_records, frozen_s2b_lookup[(model, family)],
            )
            records, metrics, parents, cell_flips, reconstruction, rep_audit, pop_cell = cell
            candidate_records.extend(records); candidate_metrics.extend(metrics)
            parent_records.extend(parents); flips.extend(cell_flips)
            reconstruction_rows.append(reconstruction); representation_audits.append(rep_audit)
            population_cells.append(pop_cell)

    population = dict(frozen_population)
    population["state_id"] = STATE_ID
    population["cells"] = population_cells
    expected = registry["expected_population"]
    if any(population[key] != expected[key] for key in (
        "n_cells", "n_scorer_rows", "n_source_question_groups",
        "source_question_group_sha256",
    )):
        raise BridgeError(f"S3A population differs from S2B: {population}")
    if len(candidate_records) != population["n_scorer_rows"]:
        raise BridgeError("S3A scorer-row count differs from frozen population")

    aggregate = s1.aggregate_metrics(candidate_metrics)
    for row in aggregate:
        row["variant_id"] = VARIANT_ID
    contrasts, family_deltas = paired_contrasts(
        candidate_records, parent_records, candidate_metrics, frozen_s2b_lookup
    )
    flip_counts = {
        kind: sum(row["flip_kind"] == kind for row in flips)
        for kind in sorted({row["flip_kind"] for row in flips})
    }
    source_manifest_after = preflight_sources(registry)
    if source_manifest_after != source_manifest:
        raise BridgeError("registered source changed during read-only S3A run")

    output.mkdir(parents=True, exist_ok=True)
    s1.write_csv(output / "P0_S3A_LOCAL_PER_QUESTION.csv", candidate_records)
    s1.write_csv(output / "P0_S3A_LOCAL_CELL_METRICS.csv", candidate_metrics)
    s1.write_csv(output / "P0_S3A_LOCAL_AGGREGATE.csv", aggregate)
    s1.write_csv(output / "P0_S3A_CONTRASTS.csv", contrasts)
    s1.write_csv(output / "P0_S3A_FAMILY_DELTAS.csv", family_deltas)
    s1.write_csv(output / "P0_S3A_RECONSTRUCTION_AUDIT.csv", reconstruction_rows)
    s1.write_csv(output / "P0_S3A_REPRESENTATION_AUDIT.csv", representation_audits)
    s1.write_csv(output / "P0_S3A_PREDICTION_FLIPS.csv", flips)
    s1.write_csv(
        output / "P0_S3A_PREDICTION_FLIP_SUMMARY.csv",
        [{"flip_kind": kind, "count": count} for kind, count in sorted(flip_counts.items())],
    )
    s1.write_json(output / "P0_S3A_POPULATION.json", population)
    gates = [
        {"gate_id": "P0_S3A_S2B_RECONSTRUCTION_EXACT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S3A_ONE_FACTOR_REPRESENTATION_ONLY", "observed": "raw_token_entropy", "threshold": "raw_token_entropy", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S3A_RAW_CURVES_FINITE", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S3A_POPULATION_HASH_UNCHANGED", "observed": population["source_question_group_sha256"], "threshold": expected["source_question_group_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S3A_LABEL_FREE_REPRESENTATION", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S3A_BOOTSTRAP_DRAWS", "observed": BOOTSTRAP_DRAWS, "threshold": BOOTSTRAP_DRAWS, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S3A_SOURCE_HASHES_STABLE", "observed": source_manifest_after["sources_sha256"], "threshold": source_manifest["sources_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
    ]
    s1.write_csv(output / "P0_S3A_GATES.csv", gates)
    verification = {
        "schema": "reasoning_localization_p0_s3a_verification_v1",
        "state_id": STATE_ID, "status": "COMPLETE",
        "variant_id": VARIANT_ID, "parent_variant_id": PARENT_VARIANT_ID,
        "single_changed_factor": registry["single_changed_factor"],
        "s2b_reconstruction_exact": all(row["s2b_reconstruction_exact"] for row in reconstruction_rows),
        "aggregate": aggregate, "contrasts": contrasts,
        "prediction_flip_counts": flip_counts,
        "locator_change_count": sum(row["locator_changed"] for row in flips),
        "representation_audits": representation_audits,
        "population_sha256": population["source_question_group_sha256"],
        "source_manifest": source_manifest,
        "new_model_inference": False, "gpu_hours": 0,
        "label_free_method_fit": True, "source_mutation": False,
        "iu29_bridge_opened": False,
    }
    s1.write_json(output / "P0_S3A_VERIFICATION.json", verification)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    run_manifest = {
        "schema": "reasoning_localization_p0_s3a_run_manifest_v1",
        "state_id": STATE_ID, "status": "COMPLETE", "source_commit": s1.git_head(),
        "runner_sha256": s1.sha256_file(Path(__file__).resolve()),
        "execution_registry_sha256": s1.sha256_file(registry_path),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "new_model_inference": False,
        "gpu_hours": 0, "label_free_method_fit": True, "source_mutation": False,
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
    except (BridgeError, s2b.BridgeError, s2a.BridgeError, s1.BridgeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
