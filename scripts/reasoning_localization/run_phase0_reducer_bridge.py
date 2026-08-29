#!/usr/bin/env python3
"""Run Phase-0 state S1: change only top-five step pooling to step max.

The runner refits the frozen historical family6/local and RegisteredGlobal
heads from the exact Stage-4 calibration rows.  Before evaluating the new
reducer it must reconstruct every S0 audit locator and prediction.  The only
scientific change is then `_step_top5_locator` -> `_peak_locator`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_global_local_online_architecture_v2 import (  # noqa: E402
    _best_threshold,
    _peak_locator,
    _processbench,
    fit_registered_global,
    load_rows,
)
from scripts.run_local_online_comprehensive_stage1 import (  # noqa: E402
    _stage_partition,
    _step_top5_locator,
)
from scripts.run_local_online_comprehensive_stage4 import (  # noqa: E402
    FAMILIES,
    FINALIST,
    MODELS,
    PROTOCOL,
    PROTOCOL_SHA256,
    _metadata,
)
from spectral_utils.local_online_comprehensive import (  # noqa: E402
    fit_references,
    fit_trajectory_head_prepared,
    prepare_trace,
)


STATE_ID = "P0_S1_REDUCER_BRIDGE"
VARIANT_ID = "P0_S1_FAMILY6_STEP_MAX"
PARENT_VARIANT_ID = "R2_HISTORICAL_FAMILY6_BRIDGE"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082901
NUMERIC_TOLERANCE = 1e-12
METRIC_FIELDS = ("f1", "exact_error", "clean_abstention", "within_one")
DEFAULT_OUTPUT = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "p0_s1_reducer_bridge"
)
DEFAULT_REGISTRY = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "P0_S1_EXECUTION_REGISTRY.json"
)


class BridgeError(RuntimeError):
    """Raised when the frozen S1 contract or S0 reconstruction fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise BridgeError(f"refusing to write empty S1 table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise BridgeError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise BridgeError(
            f"{label} SHA mismatch: {path}\nexpected={expected}\nobserved={observed}"
        )


def resolve_registered_path(spec: Mapping[str, Any]) -> Path:
    path = Path(spec["path"])
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("state_id") != STATE_ID:
        raise BridgeError("execution registry state_id mismatch")
    if registry.get("status") != "FROZEN_BEFORE_RUN":
        raise BridgeError("execution registry must be FROZEN_BEFORE_RUN")
    require_hash(Path(__file__).resolve(), registry["runner_sha256"], "frozen runner")
    require_hash(PROTOCOL, registry["protocol_sha256"], "frozen protocol")
    if registry["protocol_sha256"] != PROTOCOL_SHA256:
        raise BridgeError("registry protocol hash differs from historical code gate")
    if registry.get("bootstrap_draws") != BOOTSTRAP_DRAWS:
        raise BridgeError("bootstrap draw count differs from the frozen S1 contract")
    if registry.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise BridgeError("bootstrap seed differs from the frozen S1 contract")
    factor = registry.get("single_changed_factor", {})
    if factor != {
        "factor": "step_reducer",
        "from": "step_top5mean",
        "to": "step_max_token_argmax",
    }:
        raise BridgeError("S1 must change exactly the registered reducer factor")
    return registry


def preflight_sources(registry: Mapping[str, Any]) -> dict[str, Any]:
    expected_cells = {(model, family) for model in MODELS for family in FAMILIES}
    sources: list[dict[str, Any]] = []
    for role in ("raw_sources", "checkpoints"):
        specs = {(row["model"], row["family"]): row for row in registry[role]}
        if set(specs) != expected_cells:
            raise BridgeError(f"{role} roster differs from the exact eight cells")
        for model, family in sorted(expected_cells):
            spec = specs[(model, family)]
            path = resolve_registered_path(spec)
            require_hash(path, spec["sha256"], f"{role} {model}/{family}")
            sources.append({
                "role": role[:-1], "model": model, "family": family,
                "path": spec["path"], "sha256": spec["sha256"],
                "bytes": path.stat().st_size,
            })
    for role in ("s0_artifacts", "code_dependencies"):
        for spec in registry[role]:
            path = resolve_registered_path(spec)
            require_hash(path, spec["sha256"], f"{role} {spec['path']}")
            sources.append({
                "role": role[:-1], "path": spec["path"],
                "sha256": spec["sha256"], "bytes": path.stat().st_size,
            })
    return {
        "sources": sources,
        "sources_sha256": canonical_sha256(sources),
    }


def max_numeric_delta(left: Any, right: Any, label: str = "root") -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise BridgeError(f"diagnostic structure mismatch at {label}")
        return max(
            (max_numeric_delta(left[key], right[key], f"{label}.{key}") for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise BridgeError(f"diagnostic length mismatch at {label}")
        return max(
            (max_numeric_delta(a, b, f"{label}[{index}]") for index, (a, b) in enumerate(zip(left, right))),
            default=0.0,
        )
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        if bool(left) != bool(right):
            raise BridgeError(f"diagnostic boolean mismatch at {label}")
        return 0.0
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        a, b = float(left), float(right)
        if np.isnan(a) and np.isnan(b):
            return 0.0
        if not np.isfinite(a) or not np.isfinite(b):
            if a != b:
                raise BridgeError(f"diagnostic nonfinite mismatch at {label}")
            return 0.0
        return abs(a - b)
    if left != right:
        raise BridgeError(f"diagnostic value mismatch at {label}: {left!r} != {right!r}")
    return 0.0


def load_cell_specs(registry: Mapping[str, Any]) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    raw = {(row["model"], row["family"]): row for row in registry["raw_sources"]}
    checkpoints = {(row["model"], row["family"]): row for row in registry["checkpoints"]}
    return raw, checkpoints


def flip_kind(target: int, parent: int, candidate: int) -> str:
    if parent == candidate:
        return "NO_FLIP"
    if target == -1:
        if parent == -1 and candidate != -1:
            return "CLEAN_TO_FALSE_POSITIVE"
        if parent != -1 and candidate == -1:
            return "FALSE_POSITIVE_TO_CLEAN"
        return "CLEAN_WRONG_STEP_CHANGED"
    if parent != target and candidate == target:
        return "ERROR_TO_EXACT"
    if parent == target and candidate != target:
        return "EXACT_TO_NONEXACT"
    if parent == -1 and candidate != -1:
        return "ERROR_ABSTENTION_TO_LOCALIZED"
    if parent != -1 and candidate == -1:
        return "ERROR_LOCALIZED_TO_ABSTENTION"
    return "ERROR_NONEXACT_CHANGED"


def run_cell(
    model: str,
    family: str,
    raw_spec: Mapping[str, Any],
    checkpoint_spec: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows = load_rows(resolve_registered_path(raw_spec))
    for row in rows:
        row["_stage"] = _stage_partition(family, row["_unit"])
    calibration = [row for row in rows if row["_stage"] == "calibration"]
    audit = [row for row in rows if row["_stage"] == "audit"]
    if not calibration or not audit:
        raise BridgeError(f"empty calibration/audit role for {model}/{family}")

    with resolve_registered_path(checkpoint_spec).open("rb") as handle:
        checkpoint = pickle.load(handle)
    parent_records = [
        dict(row) for row in checkpoint["local_records"]
        if row["candidate"] == FINALIST
    ]
    parent_metrics = [
        row for row in checkpoint["local_metrics"]
        if row["candidate"] == FINALIST
    ]
    if len(parent_metrics) != 1 or len(parent_records) != len(audit):
        raise BridgeError(f"unexpected S0 checkpoint shape for {model}/{family}")

    references = fit_references(calibration)
    prepared_cal = [prepare_trace(row, references) for row in calibration]
    prepared_audit = [prepare_trace(row, references) for row in audit]
    local_head = fit_trajectory_head_prepared(
        prepared_cal, name="finalist_local", representation="family6",
        operators=("level",),
    )
    global_model = fit_registered_global(calibration)
    cal_curves = [
        local_head.curve_from_level(item.representations["family6"])
        for item in prepared_cal
    ]
    audit_curves = [
        local_head.curve_from_level(item.representations["family6"])
        for item in prepared_audit
    ]
    cal_scores = np.asarray([global_model.score(row, None) for row in calibration])
    audit_scores = np.asarray([global_model.score(row, None) for row in audit])
    cal_targets = np.asarray([int(row["label"]) for row in calibration])
    audit_targets = np.asarray([int(row["label"]) for row in audit])

    # Hard reconstruction gate: the refit must reproduce S0 before S1 opens.
    cal_top5 = np.asarray([
        _step_top5_locator(curve, row)
        for curve, row in zip(cal_curves, calibration)
    ])
    audit_top5 = np.asarray([
        _step_top5_locator(curve, row)
        for curve, row in zip(audit_curves, audit)
    ])
    top5_threshold, top5_calibration_f1 = _best_threshold(
        cal_scores, cal_top5, cal_targets
    )
    top5_predictions = np.where(audit_scores > top5_threshold, audit_top5, -1)
    top5_result = _processbench(top5_predictions, audit_targets)
    parent_lookup = {row["unit"]: row for row in parent_records}
    audit_units = [row["_unit"] for row in audit]
    if set(parent_lookup) != set(audit_units):
        raise BridgeError(f"S0 unit mismatch for {model}/{family}")
    expected_parent = [parent_lookup[unit] for unit in audit_units]
    expected_scores = np.asarray([float(row["score"]) for row in expected_parent])
    expected_locators = np.asarray([int(row["locator"]) for row in expected_parent])
    expected_predictions = np.asarray([int(row["prediction"]) for row in expected_parent])
    expected_targets = np.asarray([int(row["target"]) for row in expected_parent])
    max_score_delta = float(np.max(np.abs(audit_scores - expected_scores)))
    locator_mismatches = int(np.sum(audit_top5 != expected_locators))
    prediction_mismatches = int(np.sum(top5_predictions != expected_predictions))
    target_mismatches = int(np.sum(audit_targets != expected_targets))
    expected_metric = parent_metrics[0]
    metric_max_delta = max(
        abs(float(top5_result[field]) - float(expected_metric[field]))
        for field in METRIC_FIELDS
    )
    threshold_delta = abs(float(top5_threshold) - float(expected_metric["threshold"]))
    diagnostics_delta = max(
        max_numeric_delta(references.as_dict(), checkpoint["diagnostics"]["references"], "references"),
        max_numeric_delta(local_head.diagnostics, checkpoint["diagnostics"]["local"], "local"),
        max_numeric_delta(global_model.diagnostics, checkpoint["diagnostics"]["global"], "global"),
    )
    if (
        locator_mismatches or prediction_mismatches or target_mismatches
        or max_score_delta > NUMERIC_TOLERANCE
        or metric_max_delta > NUMERIC_TOLERANCE
        or threshold_delta > NUMERIC_TOLERANCE
        or diagnostics_delta > NUMERIC_TOLERANCE
    ):
        raise BridgeError(
            f"S0 reconstruction hard failure for {model}/{family}: "
            f"locator={locator_mismatches}, prediction={prediction_mismatches}, "
            f"target={target_mismatches}, score_delta={max_score_delta}, "
            f"metric_delta={metric_max_delta}, threshold_delta={threshold_delta}, "
            f"diagnostics_delta={diagnostics_delta}"
        )

    # The single opened factor: top-five pooling becomes token argmax -> step.
    cal_max = np.asarray([
        _peak_locator(curve, row) for curve, row in zip(cal_curves, calibration)
    ])
    audit_max = np.asarray([
        _peak_locator(curve, row) for curve, row in zip(audit_curves, audit)
    ])
    threshold, calibration_f1 = _best_threshold(cal_scores, cal_max, cal_targets)
    predictions = np.where(audit_scores > threshold, audit_max, -1)
    result = _processbench(predictions, audit_targets)
    cuts = tuple(np.quantile(
        [len(row["token_entropies"]) for row in calibration], (1 / 3, 2 / 3)
    ))
    candidate_records: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    normalized_parent: list[dict[str, Any]] = []
    for row, target, score, locator, prediction, parent in zip(
        audit, audit_targets, audit_scores, audit_max, predictions, expected_parent
    ):
        candidate_records.append({
            "candidate": VARIANT_ID, "model": model, "family": family,
            "unit": row["_unit"], "task": "local", "target": int(target),
            "score": float(score), "locator": int(locator),
            "prediction": int(prediction), "access_tier": "A",
            "reducer": "step_max_token_argmax", **_metadata(row, cuts),
        })
        parent_row = dict(parent)
        parent_row["candidate"] = PARENT_VARIANT_ID
        normalized_parent.append(parent_row)
        flips.append({
            "model": model, "family": family, "unit": row["_unit"],
            "target": int(target), "s0_locator": int(parent["locator"]),
            "s1_locator": int(locator), "s0_prediction": int(parent["prediction"]),
            "s1_prediction": int(prediction),
            "flip_kind": flip_kind(int(target), int(parent["prediction"]), int(prediction)),
        })
    metric = {
        "candidate": VARIANT_ID, "model": model, "family": family,
        "task": "local", "primary": result["f1"], **result,
        "threshold": float(threshold), "calibration_f1": float(calibration_f1),
        "n": len(audit), "access_tier": "A", "reducer": "step_max_token_argmax",
    }
    reconstruction = {
        "model": model, "family": family, "n_audit": len(audit),
        "s0_locator_mismatches": locator_mismatches,
        "s0_prediction_mismatches": prediction_mismatches,
        "s0_target_mismatches": target_mismatches,
        "s0_max_abs_score_delta": max_score_delta,
        "s0_metric_max_abs_delta": metric_max_delta,
        "s0_threshold_abs_delta": threshold_delta,
        "s0_diagnostics_max_abs_delta": diagnostics_delta,
        "s0_reconstruction_exact": True,
        "s0_threshold": float(top5_threshold),
        "s0_calibration_f1": float(top5_calibration_f1),
    }
    population_cell = {
        "model": model, "family": family, "n_scorer_rows": len(audit),
        "raw_source_path": raw_spec["path"], "raw_source_sha256": raw_spec["sha256"],
        "checkpoint_path": checkpoint_spec["path"],
        "checkpoint_sha256": checkpoint_spec["sha256"],
        "unit_sha256": hashlib.sha256("\n".join(sorted(audit_units)).encode("utf-8")).hexdigest(),
    }
    return candidate_records, [metric], reconstruction, flips, normalized_parent, population_cell


def aggregate_metrics(cell_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for field in METRIC_FIELDS:
        output.append({
            "variant_id": VARIANT_ID,
            "metric_id": "macro_f1" if field == "f1" else field,
            "value": float(np.mean([float(row[field]) for row in cell_metrics])),
            "status": "COMPLETE",
            "evidence_status": "RETROSPECTIVE",
        })
    return output


def draw_metrics(prediction: np.ndarray, target: np.ndarray, indexes: np.ndarray) -> dict[str, np.ndarray]:
    pred = prediction[indexes]
    truth = target[indexes]
    error = truth != -1
    clean = ~error
    n_error = error.sum(axis=1)
    n_clean = clean.sum(axis=1)
    exact = np.divide(
        ((pred == truth) & error).sum(axis=1), n_error,
        out=np.full(len(indexes), np.nan), where=n_error > 0,
    )
    abstain = np.divide(
        ((pred == -1) & clean).sum(axis=1), n_clean,
        out=np.full(len(indexes), np.nan), where=n_clean > 0,
    )
    f1 = np.divide(
        2.0 * exact * abstain, exact + abstain,
        out=np.zeros(len(indexes), dtype=float),
        where=np.isfinite(exact) & np.isfinite(abstain) & ((exact + abstain) > 0),
    )
    within_one = np.divide(
        ((pred != -1) & error & (np.abs(pred - truth) <= 1)).sum(axis=1), n_error,
        out=np.full(len(indexes), np.nan), where=n_error > 0,
    )
    return {
        "f1": f1, "exact_error": exact,
        "clean_abstention": abstain, "within_one": within_one,
    }


def paired_contrasts(
    candidate_records: Sequence[Mapping[str, Any]],
    parent_records: Sequence[Mapping[str, Any]],
    candidate_cells: Sequence[Mapping[str, Any]],
    parent_cell_metrics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records = list(candidate_records) + list(parent_records)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    family_draws: dict[str, dict[str, np.ndarray]] = {}
    family_points: dict[str, dict[str, float]] = {}
    cell_deltas: dict[str, list[float]] = {field: [] for field in METRIC_FIELDS}
    family_rows: list[dict[str, Any]] = []
    candidate_cell_lookup = {
        (row["model"], row["family"]): row for row in candidate_cells
    }
    for family in FAMILIES:
        family_records = [row for row in all_records if row["family"] == family]
        models = sorted({row["model"] for row in family_records})
        units = sorted(set.intersection(*[
            {row["unit"] for row in family_records if row["candidate"] == method and row["model"] == model}
            for method in (VARIANT_ID, PARENT_VARIANT_ID) for model in models
        ]))
        indexes = rng.integers(0, len(units), size=(BOOTSTRAP_DRAWS, len(units)))
        method_draws: dict[str, dict[str, list[np.ndarray]]] = {
            method: {field: [] for field in METRIC_FIELDS}
            for method in (VARIANT_ID, PARENT_VARIANT_ID)
        }
        for method in (VARIANT_ID, PARENT_VARIANT_ID):
            for model in models:
                lookup = {
                    row["unit"]: row for row in family_records
                    if row["candidate"] == method and row["model"] == model
                }
                if set(lookup) != set(units):
                    raise BridgeError(f"paired coverage mismatch for {method}/{model}/{family}")
                predictions = np.asarray([int(lookup[unit]["prediction"]) for unit in units])
                targets = np.asarray([int(lookup[unit]["target"]) for unit in units])
                values = draw_metrics(predictions, targets, indexes)
                for field in METRIC_FIELDS:
                    method_draws[method][field].append(values[field])
        family_draws[family] = {}
        family_points[family] = {}
        for field in METRIC_FIELDS:
            candidate_draw = np.nanmean(np.vstack(method_draws[VARIANT_ID][field]), axis=0)
            parent_draw = np.nanmean(np.vstack(method_draws[PARENT_VARIANT_ID][field]), axis=0)
            family_draws[family][field] = candidate_draw - parent_draw
            point_deltas = []
            for model in models:
                candidate_cell = candidate_cell_lookup[(model, family)]
                parent_cell = parent_cell_metrics[(model, family)]
                delta = float(candidate_cell[field]) - float(parent_cell[field])
                point_deltas.append(delta)
                cell_deltas[field].append(delta)
            family_points[family][field] = float(np.mean(point_deltas))
            family_rows.append({
                "family": family,
                "metric_id": "macro_f1" if field == "f1" else field,
                "delta": family_points[family][field],
                "n_groups": len(units),
            })

    contrasts = []
    for field in METRIC_FIELDS:
        draws = np.mean(np.vstack([
            family_draws[family][field] for family in FAMILIES
        ]), axis=0)
        low, high = np.quantile(draws, (0.025, 0.975))
        points = [family_points[family][field] for family in FAMILIES]
        contrast = {
            "left_variant_id": VARIANT_ID,
            "right_variant_id": PARENT_VARIANT_ID,
            "metric_id": "macro_f1" if field == "f1" else field,
            "delta": float(np.mean(points)),
            "ci_low": float(low),
            "ci_high": float(high),
            "wins": int(sum(value > 0 for value in points)),
            "ties": int(sum(value == 0 for value in points)),
            "losses": int(sum(value < 0 for value in points)),
            "worst_unit_delta": float(min(cell_deltas[field])),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_groups": 635,
            "status": "COMPLETE",
            "evidence_status": "RETROSPECTIVE",
        }
        contrasts.append(contrast)
    return contrasts, family_rows


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


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
    raw_specs, checkpoint_specs = load_cell_specs(registry)
    candidate_records: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    population_cells: list[dict[str, Any]] = []
    parent_cell_metrics: dict[tuple[str, str], Mapping[str, Any]] = {}
    for model in MODELS:
        for family in FAMILIES:
            print(f"P0-S1 {model}/{family}: refit and S0 reconstruction", flush=True)
            cell_records, cell_metrics, reconstruction, flips, cell_parent, population_cell = run_cell(
                model, family, raw_specs[(model, family)], checkpoint_specs[(model, family)]
            )
            with resolve_registered_path(checkpoint_specs[(model, family)]).open("rb") as handle:
                checkpoint = pickle.load(handle)
            parent_metric = next(
                row for row in checkpoint["local_metrics"] if row["candidate"] == FINALIST
            )
            parent_cell_metrics[(model, family)] = parent_metric
            candidate_records.extend(cell_records)
            candidate_metrics.extend(cell_metrics)
            parent_records.extend(cell_parent)
            reconstruction_rows.append(reconstruction)
            flip_rows.extend(flips)
            population_cells.append(population_cell)

    family_units: dict[str, tuple[str, ...]] = {}
    for cell in population_cells:
        units = tuple(sorted(
            row["unit"] for row in candidate_records
            if row["model"] == cell["model"] and row["family"] == cell["family"]
        ))
        if cell["family"] in family_units and family_units[cell["family"]] != units:
            raise BridgeError(f"source-question identity mismatch across models for {cell['family']}")
        family_units.setdefault(cell["family"], units)
    group_keys = [
        f"{family}|{unit}" for family in FAMILIES for unit in family_units[family]
    ]
    population = {
        "schema": "reasoning_localization_p0_population_v1",
        "population_id": "historical_stage4_eight_cell_audit",
        "models": list(MODELS), "families": list(FAMILIES),
        "cells": population_cells, "n_cells": len(population_cells),
        "n_scorer_rows": len(candidate_records),
        "n_source_question_groups": len(group_keys),
        "source_question_group_sha256": hashlib.sha256("\n".join(group_keys).encode("utf-8")).hexdigest(),
        "calibration_role": "historical deterministic 40 percent per family",
        "evaluation_role": "historical deterministic 20 percent audit per family",
        "scorer_copies_grouped": True,
    }
    expected = registry["expected_population"]
    if any(population[key] != expected[key] for key in (
        "n_cells", "n_scorer_rows", "n_source_question_groups",
        "source_question_group_sha256",
    )):
        raise BridgeError(f"S1 population differs from frozen S0: {population}")

    aggregate = aggregate_metrics(candidate_metrics)
    contrasts, family_deltas = paired_contrasts(
        candidate_records, parent_records, candidate_metrics, parent_cell_metrics
    )
    flip_counts = {
        kind: sum(row["flip_kind"] == kind for row in flip_rows)
        for kind in sorted({row["flip_kind"] for row in flip_rows})
    }
    source_manifest_after = preflight_sources(registry)
    if source_manifest_after != source_manifest:
        raise BridgeError("registered source changed during the read-only S1 run")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "P0_S1_LOCAL_PER_QUESTION.csv", candidate_records)
    write_csv(output / "P0_S1_LOCAL_CELL_METRICS.csv", candidate_metrics)
    write_csv(output / "P0_S1_LOCAL_AGGREGATE.csv", aggregate)
    write_csv(output / "P0_S1_CONTRASTS.csv", contrasts)
    write_csv(output / "P0_S1_FAMILY_DELTAS.csv", family_deltas)
    write_csv(output / "P0_S1_RECONSTRUCTION_AUDIT.csv", reconstruction_rows)
    write_csv(output / "P0_S1_PREDICTION_FLIPS.csv", flip_rows)
    write_csv(
        output / "P0_S1_PREDICTION_FLIP_SUMMARY.csv",
        [
            {"flip_kind": kind, "count": count}
            for kind, count in sorted(flip_counts.items())
        ],
    )
    write_json(output / "P0_S1_POPULATION.json", population)
    gates = [
        {"gate_id": "P0_S1_S0_RECONSTRUCTION_EXACT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S1_ONE_FACTOR_REDUCER_ONLY", "observed": "step_max_token_argmax", "threshold": "step_max_token_argmax", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S1_POPULATION_HASH_UNCHANGED", "observed": population["source_question_group_sha256"], "threshold": expected["source_question_group_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S1_PAIRED_GROUP_COVERAGE", "observed": population["n_source_question_groups"], "threshold": 635, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S1_BOOTSTRAP_DRAWS", "observed": BOOTSTRAP_DRAWS, "threshold": BOOTSTRAP_DRAWS, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S1_SOURCE_HASHES_STABLE", "observed": source_manifest_after["sources_sha256"], "threshold": source_manifest["sources_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
    ]
    write_csv(output / "P0_S1_GATES.csv", gates)
    verification = {
        "schema": "reasoning_localization_p0_s1_verification_v1",
        "state_id": STATE_ID, "status": "COMPLETE",
        "variant_id": VARIANT_ID, "parent_variant_id": PARENT_VARIANT_ID,
        "single_changed_factor": registry["single_changed_factor"],
        "s0_reconstruction_exact": all(row["s0_reconstruction_exact"] for row in reconstruction_rows),
        "aggregate": aggregate, "contrasts": contrasts,
        "prediction_flip_counts": flip_counts,
        "population_sha256": population["source_question_group_sha256"],
        "source_manifest": source_manifest,
        "new_inference": False, "gpu_hours": 0, "source_mutation": False,
    }
    write_json(output / "P0_S1_VERIFICATION.json", verification)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    run_manifest = {
        "schema": "reasoning_localization_p0_s1_run_manifest_v1",
        "state_id": STATE_ID, "status": "COMPLETE",
        "source_commit": git_head(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "execution_registry_sha256": sha256_file(registry_path),
        "protocol_sha256": PROTOCOL_SHA256,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "new_inference": False, "gpu_hours": 0, "source_mutation": False,
        "outputs": [
            {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    write_json(output / "RUN_MANIFEST.json", run_manifest)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
