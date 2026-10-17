#!/usr/bin/env python3
"""Run the bounded Phase-0 reducer-by-detector interaction control.

The completed P0-S2A state uses the calibration-only DUFS-LIU answer
detector with a step-max local reducer.  This runner reconstructs that state
exactly, then changes only the reducer back to the historical step-top-five
mean.  Together with S0, S1, and S2A this supplies the fourth cell of the
registered 2x2 audit:

    RegisteredGlobal/top5   RegisteredGlobal/max
    DUFS-LIU/top5           DUFS-LIU/max

The run reports adjacent and anchor contrasts separately and computes a
paired difference-in-differences interaction.  It is retrospective audit
evidence only.  The original purely-local P0-S2B state remains unopened.
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
from scripts.reasoning_localization import run_phase0_reducer_bridge as s1  # noqa: E402
from spectral_utils.online_localization_fusion import (  # noqa: E402
    GLOBAL_DUFS_EPOCHS,
    fit_frozen_global_gl_liu,
)


STATE_ID = "P0_S2I_REDUCER_DETECTOR_INTERACTION_CONTROL"
VARIANT_ID = "P0_S2I_FAMILY6_TOP5_DUFS_DETECTOR"
PARENT_VARIANT_ID = s2a.VARIANT_ID
ANCHOR_VARIANT_ID = s1.PARENT_VARIANT_ID
REGISTERED_MAX_VARIANT_ID = s1.VARIANT_ID
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082901
NUMERIC_TOLERANCE = 1e-12
METRIC_FIELDS = s1.METRIC_FIELDS
DEFAULT_OUTPUT = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "p0_s2i_interaction_control"
)
DEFAULT_REGISTRY = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "P0_S2I_EXECUTION_REGISTRY.json"
)


class InteractionError(RuntimeError):
    """Raised when the frozen interaction-control contract is violated."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise InteractionError(f"missing {label}: {path}")
    observed = s1.sha256_file(path)
    if observed != expected:
        raise InteractionError(
            f"{label} SHA mismatch: {path}\nexpected={expected}\nobserved={observed}"
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact_path(registry: Mapping[str, Any], role: str) -> Path:
    matches = [row for row in registry["source_artifacts"] if row["role"] == role]
    if len(matches) != 1:
        raise InteractionError(f"expected exactly one source artifact for role={role}")
    return resolve_path(matches[0]["path"])


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("state_id") != STATE_ID:
        raise InteractionError("execution registry state_id mismatch")
    if registry.get("status") != "FROZEN_BEFORE_RUN":
        raise InteractionError("execution registry must be FROZEN_BEFORE_RUN")
    require_hash(Path(__file__).resolve(), registry["runner_sha256"], "frozen runner")
    if registry.get("single_changed_factor") != {
        "factor": "step_reducer",
        "from": "step_max_token_argmax",
        "to": "step_top5mean",
    }:
        raise InteractionError("interaction control must change only the reducer")
    if registry.get("bootstrap_draws") != BOOTSTRAP_DRAWS:
        raise InteractionError("bootstrap draw count differs from the frozen contract")
    if registry.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise InteractionError("bootstrap seed differs from the frozen contract")
    return registry


def preflight_sources(registry: Mapping[str, Any]) -> dict[str, Any]:
    s2a_spec = registry["s2a_execution_registry"]
    s2a_registry_path = resolve_path(s2a_spec["path"])
    require_hash(s2a_registry_path, s2a_spec["sha256"], "S2A execution registry")
    s2a_registry = s2a.load_registry(s2a_registry_path)
    inherited = s2a.preflight_sources(s2a_registry)
    sources = list(inherited["sources"])
    sources.append({
        "role": "s2a_execution_registry",
        "path": s2a_spec["path"],
        "sha256": s2a_spec["sha256"],
        "bytes": s2a_registry_path.stat().st_size,
    })
    for spec in registry["source_artifacts"]:
        path = resolve_path(spec["path"])
        require_hash(path, spec["sha256"], f"source artifact {spec['role']}")
        sources.append({
            "role": spec["role"], "path": spec["path"],
            "sha256": spec["sha256"], "bytes": path.stat().st_size,
        })
    return {
        "sources": sources,
        "sources_sha256": s1.canonical_sha256(sources),
        "inherited_s2a_sources_sha256": inherited["sources_sha256"],
    }


def normalize_records(
    rows: Sequence[Mapping[str, Any]], variant_id: str
) -> list[dict[str, Any]]:
    return [{**dict(row), "candidate": variant_id} for row in rows]


def select_historical_finalist(
    rows: Sequence[Mapping[str, Any]], label: str
) -> list[dict[str, Any]]:
    selected = [
        dict(row) for row in rows
        if row.get("candidate") == "finalist_global_detector_local_locator"
    ]
    if not selected:
        raise InteractionError(f"historical finalist selector is empty for {label}")
    return selected


def compare_s2a_cell(
    model: str,
    family: str,
    records: Sequence[Mapping[str, Any]],
    metric: Mapping[str, Any],
    frozen_records: Sequence[Mapping[str, Any]],
    frozen_metric: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        str(row["unit"]): row for row in frozen_records
        if row["model"] == model and row["family"] == family
    }
    observed = {str(row["unit"]): row for row in records}
    if set(expected) != set(observed):
        raise InteractionError(f"S2A unit mismatch for {model}/{family}")
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
        abs(float(metric[field]) - float(frozen_metric[field]))
        for field in (*METRIC_FIELDS, "threshold", "calibration_f1")
    )
    if (
        locator_mismatches or prediction_mismatches or target_mismatches
        or max_score_delta > NUMERIC_TOLERANCE
        or metric_delta > NUMERIC_TOLERANCE
    ):
        raise InteractionError(
            f"S2A reconstruction hard failure for {model}/{family}: "
            f"locator={locator_mismatches}, prediction={prediction_mismatches}, "
            f"target={target_mismatches}, score_delta={max_score_delta}, "
            f"metric_delta={metric_delta}"
        )
    return {
        "model": model, "family": family, "n_audit": len(observed),
        "s2a_locator_mismatches": locator_mismatches,
        "s2a_prediction_mismatches": prediction_mismatches,
        "s2a_target_mismatches": target_mismatches,
        "s2a_max_abs_score_delta": max_score_delta,
        "s2a_metric_max_abs_delta": metric_delta,
        "s2a_reconstruction_exact": True,
    }


def run_cell(
    model: str,
    family: str,
    raw_spec: Mapping[str, Any],
    frozen_records: Sequence[Mapping[str, Any]],
    frozen_metric: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]],
    dict[str, Any], list[dict[str, Any]], dict[str, Any],
]:
    rows = s1.load_rows(s1.resolve_registered_path(raw_spec))
    for row in rows:
        row["_stage"] = s1._stage_partition(family, row["_unit"])
    calibration = [row for row in rows if row["_stage"] == "calibration"]
    audit = [row for row in rows if row["_stage"] == "audit"]
    if not calibration or not audit:
        raise InteractionError(f"empty calibration/audit role for {model}/{family}")

    references = s1.fit_references(calibration)
    prepared_cal = [s1.prepare_trace(row, references) for row in calibration]
    prepared_audit = [s1.prepare_trace(row, references) for row in audit]
    local_head = s1.fit_trajectory_head_prepared(
        prepared_cal, name="finalist_local", representation="family6",
        operators=("level",),
    )
    cal_curves = [
        local_head.curve_from_level(item.representations["family6"])
        for item in prepared_cal
    ]
    audit_curves = [
        local_head.curve_from_level(item.representations["family6"])
        for item in prepared_audit
    ]
    detector = fit_frozen_global_gl_liu(
        calibration, include_elapsed_length=False,
        dufs_epochs=GLOBAL_DUFS_EPOCHS,
    )
    if detector.diagnostics.get("labels_seen_during_fit") is not False:
        raise InteractionError("DUFS detector did not preserve label-free fitting")
    if detector.diagnostics.get("head") != "answer_dufs_liu_mixed":
        raise InteractionError("unexpected DUFS detector head")

    cal_scores = np.asarray([detector.risk(row, None) for row in calibration])
    audit_scores = np.asarray([detector.risk(row, None) for row in audit])
    cal_targets = np.asarray([int(row["label"]) for row in calibration])
    audit_targets = np.asarray([int(row["label"]) for row in audit])

    # Hard gate: reconstruct the full S2A max/DUFS cell before opening top5.
    cal_max = np.asarray([
        s1._peak_locator(curve, row) for curve, row in zip(cal_curves, calibration)
    ])
    audit_max = np.asarray([
        s1._peak_locator(curve, row) for curve, row in zip(audit_curves, audit)
    ])
    max_threshold, max_calibration_f1 = s1._best_threshold(
        cal_scores, cal_max, cal_targets
    )
    max_predictions = np.where(audit_scores > max_threshold, audit_max, -1)
    max_result = s1._processbench(max_predictions, audit_targets)
    parent_records = []
    for row, score, locator, prediction, target in zip(
        audit, audit_scores, audit_max, max_predictions, audit_targets
    ):
        parent_records.append({
            "model": model, "family": family, "unit": row["_unit"],
            "candidate": PARENT_VARIANT_ID, "target": int(target),
            "score": float(score), "locator": int(locator),
            "prediction": int(prediction),
        })
    parent_metric = {
        "candidate": PARENT_VARIANT_ID, "model": model, "family": family,
        "task": "local", "primary": max_result["f1"], **max_result,
        "threshold": float(max_threshold),
        "calibration_f1": float(max_calibration_f1),
    }
    reconstruction = compare_s2a_cell(
        model, family, parent_records, parent_metric,
        frozen_records, frozen_metric,
    )

    # Sole opened factor: max becomes the historical within-step top-five mean.
    cal_top5 = np.asarray([
        s1._step_top5_locator(curve, row)
        for curve, row in zip(cal_curves, calibration)
    ])
    audit_top5 = np.asarray([
        s1._step_top5_locator(curve, row)
        for curve, row in zip(audit_curves, audit)
    ])
    threshold, calibration_f1 = s1._best_threshold(
        cal_scores, cal_top5, cal_targets
    )
    prediction = np.where(audit_scores > threshold, audit_top5, -1)
    result = s1._processbench(prediction, audit_targets)

    frozen_lookup = {
        str(row["unit"]): dict(row) for row in frozen_records
        if row["model"] == model and row["family"] == family
    }
    candidate_records: list[dict[str, Any]] = []
    normalized_parent: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    for row, target, score, locator, predicted in zip(
        audit, audit_targets, audit_scores, audit_top5, prediction
    ):
        unit = str(row["_unit"])
        parent = frozen_lookup[unit]
        candidate = dict(parent)
        candidate.update({
            "candidate": VARIANT_ID, "target": int(target),
            "score": float(score), "locator": int(locator),
            "prediction": int(predicted), "reducer": "step_top5mean",
        })
        candidate_records.append(candidate)
        normalized_parent.append({**parent, "candidate": PARENT_VARIANT_ID})
        flips.append({
            "model": model, "family": family, "unit": unit,
            "target": int(target),
            "s2a_locator": int(parent["locator"]),
            "s2i_locator": int(locator),
            "s2a_prediction": int(parent["prediction"]),
            "s2i_prediction": int(predicted),
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
        "reducer": "step_top5mean",
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
    }
    return (
        candidate_records, metric, normalized_parent, reconstruction,
        flips, detector_audit,
    )


def cell_metric_lookup(
    rows: Sequence[Mapping[str, Any]], variant_id: str | None = None
) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for row in rows:
        item = dict(row)
        if variant_id is not None:
            item["candidate"] = variant_id
        output[(str(item["model"]), str(item["family"]))] = item
    return output


def factorial_effects(
    records_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    cells_by_method: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    methods = (
        ANCHOR_VARIANT_ID, REGISTERED_MAX_VARIANT_ID,
        PARENT_VARIANT_ID, VARIANT_ID,
    )
    effects = {
        "P0_S1_VS_S0_POOLING_REGISTEREDGLOBAL": {
            REGISTERED_MAX_VARIANT_ID: 1.0, ANCHOR_VARIANT_ID: -1.0,
        },
        "P0_S2A_VS_S1_DETECTOR_STEP_MAX": {
            PARENT_VARIANT_ID: 1.0, REGISTERED_MAX_VARIANT_ID: -1.0,
        },
        "P0_S2I_VS_S2A_ADJACENT_POOLING": {
            VARIANT_ID: 1.0, PARENT_VARIANT_ID: -1.0,
        },
        "P0_S2I_VS_S0_DETECTOR_TOP5": {
            VARIANT_ID: 1.0, ANCHOR_VARIANT_ID: -1.0,
        },
        "P0_S2A_VS_S0_CUMULATIVE_ANCHOR": {
            PARENT_VARIANT_ID: 1.0, ANCHOR_VARIANT_ID: -1.0,
        },
        "P0_REDUCER_X_DETECTOR_INTERACTION": {
            PARENT_VARIANT_ID: 1.0,
            REGISTERED_MAX_VARIANT_ID: -1.0,
            VARIANT_ID: -1.0,
            ANCHOR_VARIANT_ID: 1.0,
        },
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    family_draws: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    family_points: dict[str, dict[str, dict[str, float]]] = {}
    cell_effects: dict[str, dict[str, list[float]]] = {
        effect: {field: [] for field in METRIC_FIELDS} for effect in effects
    }
    for family in s1.FAMILIES:
        models = sorted({
            str(row["model"]) for method in methods
            for row in records_by_method[method] if row["family"] == family
        })
        units = sorted(set.intersection(*[
            {
                str(row["unit"]) for row in records_by_method[method]
                if row["family"] == family and row["model"] == model
            }
            for method in methods for model in models
        ]))
        indexes = rng.integers(0, len(units), size=(BOOTSTRAP_DRAWS, len(units)))
        method_draws: dict[str, dict[str, np.ndarray]] = {}
        method_points: dict[str, dict[str, float]] = {}
        for method in methods:
            by_field = {field: [] for field in METRIC_FIELDS}
            for model in models:
                lookup = {
                    str(row["unit"]): row for row in records_by_method[method]
                    if row["family"] == family and row["model"] == model
                }
                if set(lookup) != set(units):
                    raise InteractionError(
                        f"factorial coverage mismatch for {method}/{model}/{family}"
                    )
                prediction = np.asarray([
                    int(lookup[unit]["prediction"]) for unit in units
                ])
                target = np.asarray([int(lookup[unit]["target"]) for unit in units])
                drawn = s1.draw_metrics(prediction, target, indexes)
                for field in METRIC_FIELDS:
                    by_field[field].append(drawn[field])
            method_draws[method] = {
                field: np.nanmean(np.vstack(by_field[field]), axis=0)
                for field in METRIC_FIELDS
            }
            method_points[method] = {
                field: float(np.mean([
                    float(cells_by_method[method][(model, family)][field])
                    for model in models
                ]))
                for field in METRIC_FIELDS
            }
        family_draws[family] = {}
        family_points[family] = {}
        for effect_id, coefficients in effects.items():
            family_draws[family][effect_id] = {}
            family_points[family][effect_id] = {}
            for field in METRIC_FIELDS:
                family_draws[family][effect_id][field] = sum(
                    coefficient * method_draws[method][field]
                    for method, coefficient in coefficients.items()
                )
                family_points[family][effect_id][field] = float(sum(
                    coefficient * method_points[method][field]
                    for method, coefficient in coefficients.items()
                ))
                for model in models:
                    cell_effects[effect_id][field].append(float(sum(
                        coefficient * float(
                            cells_by_method[method][(model, family)][field]
                        )
                        for method, coefficient in coefficients.items()
                    )))

    output: list[dict[str, Any]] = []
    for effect_id in effects:
        for field in METRIC_FIELDS:
            draws = np.mean(np.vstack([
                family_draws[family][effect_id][field]
                for family in s1.FAMILIES
            ]), axis=0)
            low, high = np.quantile(draws, (0.025, 0.975))
            points = [
                family_points[family][effect_id][field] for family in s1.FAMILIES
            ]
            output.append({
                "effect_id": effect_id,
                "metric_id": "macro_f1" if field == "f1" else field,
                "delta": float(np.mean(points)),
                "ci_low": float(low), "ci_high": float(high),
                "wins": int(sum(value > 0 for value in points)),
                "ties": int(sum(value == 0 for value in points)),
                "losses": int(sum(value < 0 for value in points)),
                "worst_unit_delta": float(min(cell_effects[effect_id][field])),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "n_groups": 635,
                "status": "COMPLETE", "evidence_status": "RETROSPECTIVE",
            })
    return output


def compare_frozen_edge(
    effects: Sequence[Mapping[str, Any]],
    effect_id: str,
    frozen_path: Path,
) -> float:
    observed = {
        row["metric_id"]: row for row in effects if row["effect_id"] == effect_id
    }
    frozen = {row["metric_id"]: row for row in read_csv(frozen_path)}
    if set(observed) != set(frozen):
        raise InteractionError(f"frozen edge metric roster mismatch for {effect_id}")
    fields = ("delta", "ci_low", "ci_high", "worst_unit_delta")
    maximum = max(
        abs(float(observed[metric][field]) - float(frozen[metric][field]))
        for metric in observed for field in fields
    )
    if maximum > NUMERIC_TOLERANCE:
        raise InteractionError(
            f"factorial bootstrap failed to reconstruct {effect_id}: {maximum}"
        )
    return maximum


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
        raise InteractionError(f"output directory must be new or empty: {output}")

    s2a_registry = s2a.load_registry(
        resolve_path(registry["s2a_execution_registry"]["path"])
    )
    s1_registry = s1.load_registry(
        resolve_path(s2a_registry["s1_execution_registry"]["path"])
    )
    raw_specs, _ = s1.load_cell_specs(s1_registry)
    frozen_s2a_records = read_csv(artifact_path(registry, "s2a_per_question"))
    frozen_s2a_cells = read_csv(artifact_path(registry, "s2a_cell_metrics"))
    frozen_s2a_cell_lookup = cell_metric_lookup(
        frozen_s2a_cells, PARENT_VARIANT_ID
    )
    frozen_population = json.loads(
        artifact_path(registry, "s2a_population").read_text(encoding="utf-8")
    )

    candidate_records: list[dict[str, Any]] = []
    candidate_cells: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    detector_audits: list[dict[str, Any]] = []
    for model in s1.MODELS:
        for family in s1.FAMILIES:
            print(
                f"P0-S2I {model}/{family}: S2A reconstruction and top5 control",
                flush=True,
            )
            records, metric, parents, reconstruction, flips, detector_audit = run_cell(
                model, family, raw_specs[(model, family)],
                frozen_s2a_records, frozen_s2a_cell_lookup[(model, family)],
            )
            candidate_records.extend(records)
            candidate_cells.append(metric)
            parent_records.extend(parents)
            reconstruction_rows.append(reconstruction)
            flip_rows.extend(flips)
            detector_audits.append(detector_audit)

    expected = registry["expected_population"]
    if any(frozen_population[key] != expected[key] for key in (
        "n_cells", "n_scorer_rows", "n_source_question_groups",
        "source_question_group_sha256",
    )):
        raise InteractionError("S2I population differs from the frozen S2A population")
    if len(candidate_records) != expected["n_scorer_rows"]:
        raise InteractionError("S2I scorer-row count differs from the frozen population")

    s0_records = normalize_records(
        select_historical_finalist(
            read_csv(artifact_path(registry, "s0_per_question")),
            "S0 per-question records",
        ),
        ANCHOR_VARIANT_ID,
    )
    s1_records = normalize_records(
        read_csv(artifact_path(registry, "s1_per_question")),
        REGISTERED_MAX_VARIANT_ID,
    )
    s2a_records = normalize_records(frozen_s2a_records, PARENT_VARIANT_ID)
    s0_cells = cell_metric_lookup(
        select_historical_finalist(
            read_csv(artifact_path(registry, "s0_cell_metrics")),
            "S0 cell metrics",
        ),
        ANCHOR_VARIANT_ID,
    )
    s1_cells = cell_metric_lookup(
        read_csv(artifact_path(registry, "s1_cell_metrics")),
        REGISTERED_MAX_VARIANT_ID,
    )
    candidate_cell_lookup = cell_metric_lookup(candidate_cells, VARIANT_ID)
    effects = factorial_effects(
        {
            ANCHOR_VARIANT_ID: s0_records,
            REGISTERED_MAX_VARIANT_ID: s1_records,
            PARENT_VARIANT_ID: s2a_records,
            VARIANT_ID: candidate_records,
        },
        {
            ANCHOR_VARIANT_ID: s0_cells,
            REGISTERED_MAX_VARIANT_ID: s1_cells,
            PARENT_VARIANT_ID: frozen_s2a_cell_lookup,
            VARIANT_ID: candidate_cell_lookup,
        },
    )
    s1_edge_delta = compare_frozen_edge(
        effects, "P0_S1_VS_S0_POOLING_REGISTEREDGLOBAL",
        artifact_path(registry, "s1_contrasts"),
    )
    s2a_edge_delta = compare_frozen_edge(
        effects, "P0_S2A_VS_S1_DETECTOR_STEP_MAX",
        artifact_path(registry, "s2a_contrasts"),
    )
    aggregate = s2a.aggregate_metrics(candidate_cells)
    for row in aggregate:
        row["variant_id"] = VARIANT_ID
    flip_counts = {
        kind: sum(row["flip_kind"] == kind for row in flip_rows)
        for kind in sorted({row["flip_kind"] for row in flip_rows})
    }
    source_manifest_after = preflight_sources(registry)
    if source_manifest_after != source_manifest:
        raise InteractionError("registered source changed during the read-only run")

    output.mkdir(parents=True, exist_ok=True)
    s1.write_csv(output / "P0_S2I_LOCAL_PER_QUESTION.csv", candidate_records)
    s1.write_csv(output / "P0_S2I_LOCAL_CELL_METRICS.csv", candidate_cells)
    s1.write_csv(output / "P0_S2I_LOCAL_AGGREGATE.csv", aggregate)
    s1.write_csv(output / "P0_S2I_FACTORIAL_EFFECTS.csv", effects)
    s1.write_csv(output / "P0_S2I_RECONSTRUCTION_AUDIT.csv", reconstruction_rows)
    s1.write_csv(output / "P0_S2I_DETECTOR_AUDIT.csv", detector_audits)
    s1.write_csv(output / "P0_S2I_PREDICTION_FLIPS.csv", flip_rows)
    s1.write_csv(
        output / "P0_S2I_PREDICTION_FLIP_SUMMARY.csv",
        [{"flip_kind": kind, "count": count} for kind, count in sorted(flip_counts.items())],
    )
    population = dict(frozen_population)
    population["state_id"] = STATE_ID
    s1.write_json(output / "P0_S2I_POPULATION.json", population)
    gates = [
        {"gate_id": "P0_S2I_S2A_RECONSTRUCTION_EXACT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2I_ONE_FACTOR_REDUCER_ONLY", "observed": "step_top5mean", "threshold": "step_top5mean", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2I_POPULATION_HASH_UNCHANGED", "observed": population["source_question_group_sha256"], "threshold": expected["source_question_group_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2I_LABEL_FREE_DETECTOR_FIT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2I_EXISTING_EDGES_RECONSTRUCTED", "observed": str(max(s1_edge_delta, s2a_edge_delta)), "threshold": str(NUMERIC_TOLERANCE), "direction": "le", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S2I_SOURCE_HASHES_STABLE", "observed": source_manifest_after["sources_sha256"], "threshold": source_manifest["sources_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
    ]
    s1.write_csv(output / "P0_S2I_GATES.csv", gates)
    verification = {
        "schema": "reasoning_localization_p0_s2i_verification_v1",
        "state_id": STATE_ID, "status": "COMPLETE",
        "variant_id": VARIANT_ID, "parent_variant_id": PARENT_VARIANT_ID,
        "anchor_variant_id": ANCHOR_VARIANT_ID,
        "single_changed_factor": registry["single_changed_factor"],
        "s2a_reconstruction_exact": all(
            row["s2a_reconstruction_exact"] for row in reconstruction_rows
        ),
        "existing_edge_max_abs_delta": max(s1_edge_delta, s2a_edge_delta),
        "aggregate": aggregate, "factorial_effects": effects,
        "prediction_flip_counts": flip_counts,
        "detector_audits": detector_audits,
        "population_sha256": population["source_question_group_sha256"],
        "source_manifest": source_manifest,
        "new_model_inference": False, "gpu_hours": 0,
        "label_free_method_fit": True, "source_mutation": False,
        "s2b_opened": False,
    }
    s1.write_json(output / "P0_S2I_VERIFICATION.json", verification)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    run_manifest = {
        "schema": "reasoning_localization_p0_s2i_run_manifest_v1",
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
    except (InteractionError, s1.BridgeError, s2a.BridgeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
