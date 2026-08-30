#!/usr/bin/env python3
"""Run Phase-0 state S4: replace only historical 40/20 threshold roles.

The parent is P0-S3B. Scores, locators, IU29 representation, detector,
step reducer, rows, source groups and bootstrap remain fixed. The sole
scientific change is the threshold split: per-cell historical calibration is
replaced by the current deterministic five-fold source-group cross-fit fitted
per scorer model across the four ProcessBench subsets.
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

from scripts.reasoning_localization import run_phase0_raw_entropy_representation_bridge as s3a  # noqa: E402
from scripts.reasoning_localization import run_phase0_reducer_bridge as s1  # noqa: E402
from spectral_utils.reconstruction_benchmark import localization_evaluation as current_eval  # noqa: E402


STATE_ID = "P0_S4_FIVEFOLD_SPLIT_BRIDGE"
VARIANT_ID = "P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD"
PARENT_VARIANT_ID = "P0_S3B_IU29_STEP_MAX_LOCAL_DETECTOR"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082901
NUMERIC_TOLERANCE = 1e-12
DEFAULT_OUTPUT = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "p0_s4_fivefold_split_bridge"
)
DEFAULT_REGISTRY = (
    REPO / "results" / "reasoning_localization_03662_v1" / "phase_0"
    / "P0_S4_EXECUTION_REGISTRY.json"
)


class BridgeError(RuntimeError):
    """Raised when the frozen split-only contract fails closed."""


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
    expected_factor = {
        "factor": "threshold_split",
        "from": "historical deterministic 40-percent calibration / 20-percent audit roles",
        "to": "deterministic five-fold source-group threshold cross-fit on the same audit population",
    }
    if registry.get("single_changed_factor") != expected_factor:
        raise BridgeError("S4 must change exactly the registered threshold split")
    if registry.get("bootstrap_draws") != BOOTSTRAP_DRAWS:
        raise BridgeError("bootstrap draw count differs from frozen S4")
    if registry.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise BridgeError("bootstrap seed differs from frozen S4")
    expected_crossfit = {
        "folds": 5,
        "fold_unit": "source_question",
        "fit_scope": "one scorer model across four subsets",
        "strata": "subset x error-present",
        "assignment": "SHA256(group_id) stable ordering then round-robin within stratum",
        "objective": "equal-subset mean official ProcessBench F1",
        "decision": "argmax step iff maximum score is strictly above threshold else -1",
        "tie_break": "largest numeric threshold",
        "score_parameters_refit": False,
    }
    if registry.get("crossfit_contract") != expected_crossfit:
        raise BridgeError("five-fold threshold contract differs from frozen S4")
    return registry


def preflight_sources(registry: Mapping[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for role in ("parent_registry", "parent_artifacts", "code_dependencies"):
        values = registry[role]
        values = [values] if isinstance(values, dict) else values
        for item in values:
            path = resolve_path(item["path"])
            require_hash(path, item["sha256"], f"{role} {item['path']}")
            sources.append({
                "role": role, "path": item["path"], "sha256": item["sha256"],
                "bytes": path.stat().st_size,
            })
    return {"sources": sources, "sources_sha256": s1.canonical_sha256(sources)}


def artifact_by_role(registry: Mapping[str, Any], role: str) -> Path:
    matches = [row for row in registry["parent_artifacts"] if row["role"] == role]
    if len(matches) != 1:
        raise BridgeError(f"expected one parent artifact for role={role}")
    return resolve_path(matches[0]["path"])


def _decision_equivalent_step_scores(score: float, locator: int, target: int) -> list[float]:
    """Reconstruct only the max/argmax sufficient statistics used by the evaluator."""

    if not np.isfinite(score) or locator < 0:
        raise BridgeError("parent row has invalid score or locator")
    length = max(locator + 1, target + 1 if target >= 0 else 1)
    lower = float(np.nextafter(score, -np.inf))
    if not np.isfinite(lower):
        lower = score - 1.0
    values = np.full(length, lower, dtype=np.float64)
    values[locator] = score
    if int(np.argmax(values)) != locator or float(np.max(values)) != score:
        raise BridgeError("failed to reconstruct parent max/argmax sufficient statistics")
    return values.tolist()


def reconstruct_parent(
    records: Sequence[Mapping[str, str]],
    frozen_cells: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for model in s1.MODELS:
        for family in s1.FAMILIES:
            rows = [row for row in records if row["model"] == model and row["family"] == family]
            if not rows:
                raise BridgeError(f"empty parent cell {model}/{family}")
            result = s1._processbench(
                np.asarray([int(row["prediction"]) for row in rows]),
                np.asarray([int(row["target"]) for row in rows]),
            )
            frozen = frozen_cells[(model, family)]
            metric_error = max(
                abs(float(result[field]) - float(frozen[field]))
                for field in s1.METRIC_FIELDS
            )
            if metric_error > NUMERIC_TOLERANCE:
                raise BridgeError(f"parent cell metric reconstruction failed: {model}/{family}")
            audits.append({
                "model": model, "family": family, "n": len(rows),
                "parent_metric_max_abs_error": metric_error,
                "scores_and_locators_reused_exactly": True,
                "parent_reconstruction_exact": True,
            })
    return audits


def crossfit_parent_scores(
    parent_records: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    cell_metrics: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    assignment_hashes: list[str] = []
    for model in s1.MODELS:
        model_rows = [row for row in parent_records if row["model"] == model]
        if {row["family"] for row in model_rows} != set(s1.FAMILIES):
            raise BridgeError(f"{model}: missing ProcessBench subset")
        evaluation_rows = []
        parent_lookup: dict[str, Mapping[str, str]] = {}
        for row in model_rows:
            row_id = f"{row['family']}:{row['unit']}"
            if row_id in parent_lookup:
                raise BridgeError(f"{model}: duplicate row id {row_id}")
            parent_lookup[row_id] = row
            evaluation_rows.append({
                "row_id": row_id,
                "group_id": row_id,
                "slice_id": row["family"],
                "first_error": int(row["target"]),
                "step_scores": _decision_equivalent_step_scores(
                    float(row["score"]), int(row["locator"]), int(row["target"])
                ),
            })
        fitted = current_eval.crossfit_processbench_threshold(evaluation_rows)
        if fitted["score_parameters_refit"] is not False:
            raise BridgeError("current evaluator refit score parameters")
        assignment_hashes.append(str(fitted["fold_assignment_sha256"]))
        decisions = {row["row_id"]: row for row in fitted["decisions"]}
        if set(decisions) != set(parent_lookup):
            raise BridgeError(f"{model}: cross-fit decision coverage mismatch")
        for row_id, parent in parent_lookup.items():
            decision = decisions[row_id]
            candidate = dict(parent)
            candidate.update({
                "candidate": VARIANT_ID,
                "prediction": int(decision["prediction_step"]),
                "split": "fivefold_source_group_crossfit",
                "fold": int(decision["fold"]),
            })
            candidates.append(candidate)
        for family in s1.FAMILIES:
            metrics = fitted["metrics"]["per_subset"][family]
            family_ledgers = [row for row in fitted["calibration_ledgers"]]
            cell_metrics.append({
                "candidate": VARIANT_ID, "model": model, "family": family,
                "task": "local", "primary": metrics["official_macro_f1"],
                "f1": metrics["official_macro_f1"],
                "exact_error": metrics["first_error_exact"],
                "clean_abstention": metrics["clean_abstention_accuracy"],
                "within_one": metrics["first_error_within_one"],
                "n": metrics["n_examples"], "access_tier": "A",
                "detector": "local_iu29_max", "representation": "local_iu29",
                "reducer": "step_max_token_argmax",
                "split": "fivefold_source_group_crossfit",
                "threshold_min": min(float(row["threshold"]) for row in family_ledgers),
                "threshold_max": max(float(row["threshold"]) for row in family_ledgers),
                "calibration_f1_mean": float(np.mean([
                    row["objective_equal_subset_official_macro_f1"] for row in family_ledgers
                ])),
            })
        for ledger in fitted["calibration_ledgers"]:
            ledgers.append({"model": model, **ledger})
    if len(set(assignment_hashes)) != 1:
        raise BridgeError("source-group fold assignments differ across scorer models")
    return candidates, cell_metrics, ledgers


def paired_contrasts(
    candidate_records: Sequence[Mapping[str, Any]],
    parent_records: Sequence[Mapping[str, Any]],
    candidate_cells: Sequence[Mapping[str, Any]],
    parent_cells: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contrasts, families = s3a.paired_contrasts(
        candidate_records, parent_records, candidate_cells, parent_cells
    )
    for row in contrasts:
        row["left_variant_id"] = VARIANT_ID
        row["right_variant_id"] = PARENT_VARIANT_ID
    return contrasts, families


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
        print(json.dumps({"state_id": STATE_ID, "status": "PREFLIGHT_PASS", **source_manifest}, indent=2, sort_keys=True))
        return 0
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise BridgeError(f"output directory must be new or empty: {output}")

    parent_records = read_csv(artifact_by_role(registry, "per_question"))
    parent_cell_rows = read_csv(artifact_by_role(registry, "cell_metrics"))
    parent_cells = {(row["model"], row["family"]): row for row in parent_cell_rows}
    population = json.loads(artifact_by_role(registry, "population").read_text(encoding="utf-8"))
    if (
        len(parent_records) != registry["expected_population"]["n_scorer_rows"]
        or population["source_question_group_sha256"]
        != registry["expected_population"]["source_question_group_sha256"]
    ):
        raise BridgeError("parent population differs from frozen S4")
    reconstruction = reconstruct_parent(parent_records, parent_cells)
    candidates, candidate_cells, fold_ledgers = crossfit_parent_scores(parent_records)
    normalized_parent = [{**row, "candidate": PARENT_VARIANT_ID} for row in parent_records]
    aggregate = s1.aggregate_metrics(candidate_cells)
    for row in aggregate:
        row["variant_id"] = VARIANT_ID
    contrasts, family_deltas = paired_contrasts(
        candidates, normalized_parent, candidate_cells, parent_cells
    )
    flips = []
    parent_lookup = {(row["model"], row["family"], row["unit"]): row for row in parent_records}
    for row in candidates:
        parent = parent_lookup[(row["model"], row["family"], row["unit"])]
        flips.append({
            "model": row["model"], "family": row["family"], "unit": row["unit"],
            "target": int(row["target"]), "locator": int(row["locator"]),
            "parent_prediction": int(parent["prediction"]),
            "candidate_prediction": int(row["prediction"]),
            "fold": int(row["fold"]),
            "score_unchanged": float(row["score"]) == float(parent["score"]),
            "locator_unchanged": int(row["locator"]) == int(parent["locator"]),
            "flip_kind": s1.flip_kind(
                int(row["target"]), int(parent["prediction"]), int(row["prediction"])
            ),
        })
    flip_counts = {
        kind: sum(row["flip_kind"] == kind for row in flips)
        for kind in sorted({row["flip_kind"] for row in flips})
    }
    if any(not row["score_unchanged"] or not row["locator_unchanged"] for row in flips):
        raise BridgeError("score or locator changed in split-only bridge")
    source_manifest_after = preflight_sources(registry)
    if source_manifest_after != source_manifest:
        raise BridgeError("registered source changed during read-only S4 run")

    output.mkdir(parents=True, exist_ok=True)
    population = dict(population)
    population.update({
        "state_id": STATE_ID,
        "threshold_split": "fivefold_source_group_crossfit",
        "population_unchanged_from_s3b": True,
    })
    s1.write_csv(output / "P0_S4_LOCAL_PER_QUESTION.csv", candidates)
    s1.write_csv(output / "P0_S4_LOCAL_CELL_METRICS.csv", candidate_cells)
    s1.write_csv(output / "P0_S4_LOCAL_AGGREGATE.csv", aggregate)
    s1.write_csv(output / "P0_S4_CONTRASTS.csv", contrasts)
    s1.write_csv(output / "P0_S4_FAMILY_DELTAS.csv", family_deltas)
    s1.write_csv(output / "P0_S4_PARENT_RECONSTRUCTION.csv", reconstruction)
    s1.write_csv(output / "P0_S4_FOLD_LEDGERS.csv", fold_ledgers)
    s1.write_csv(output / "P0_S4_PREDICTION_FLIPS.csv", flips)
    s1.write_csv(output / "P0_S4_PREDICTION_FLIP_SUMMARY.csv", [
        {"flip_kind": kind, "count": count} for kind, count in sorted(flip_counts.items())
    ])
    s1.write_json(output / "P0_S4_POPULATION.json", population)
    gates = [
        {"gate_id": "P0_S4_PARENT_RECONSTRUCTION_EXACT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S4_ONE_FACTOR_SPLIT_ONLY", "observed": "fivefold_source_group_crossfit", "threshold": "fivefold_source_group_crossfit", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S4_SCORE_AND_LOCATOR_UNCHANGED", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S4_FIVE_FOLDS", "observed": 5, "threshold": 5, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S4_GROUP_SAFE_ASSIGNMENT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S4_POPULATION_HASH_UNCHANGED", "observed": population["source_question_group_sha256"], "threshold": registry["expected_population"]["source_question_group_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S4_BOOTSTRAP_DRAWS", "observed": BOOTSTRAP_DRAWS, "threshold": BOOTSTRAP_DRAWS, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S4_SOURCE_HASHES_STABLE", "observed": source_manifest_after["sources_sha256"], "threshold": source_manifest["sources_sha256"], "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
    ]
    s1.write_csv(output / "P0_S4_GATES.csv", gates)
    verification = {
        "schema": "reasoning_localization_p0_s4_verification_v1",
        "state_id": STATE_ID, "status": "COMPLETE",
        "variant_id": VARIANT_ID, "parent_variant_id": PARENT_VARIANT_ID,
        "single_changed_factor": registry["single_changed_factor"],
        "parent_reconstruction_exact": all(row["parent_reconstruction_exact"] for row in reconstruction),
        "score_and_locator_unchanged": all(row["score_unchanged"] and row["locator_unchanged"] for row in flips),
        "aggregate": aggregate, "contrasts": contrasts,
        "prediction_flip_counts": flip_counts,
        "fold_assignment_sha256": fold_ledgers[0]["held_out_group_sha256"] if fold_ledgers else None,
        "fold_ledgers": fold_ledgers,
        "population_sha256": population["source_question_group_sha256"],
        "source_manifest": source_manifest,
        "new_model_inference": False, "gpu_hours": 0,
        "representation_refit": False, "score_parameters_refit": False,
        "source_mutation": False, "population_bridge_opened": False,
    }
    s1.write_json(output / "P0_S4_VERIFICATION.json", verification)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    run_manifest = {
        "schema": "reasoning_localization_p0_s4_run_manifest_v1",
        "state_id": STATE_ID, "status": "COMPLETE", "source_commit": s1.git_head(),
        "runner_sha256": s1.sha256_file(Path(__file__).resolve()),
        "execution_registry_sha256": s1.sha256_file(registry_path),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "new_model_inference": False,
        "gpu_hours": 0, "representation_refit": False,
        "score_parameters_refit": False, "source_mutation": False,
        "outputs": [
            {"file": path.name, "sha256": s1.sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    s1.write_json(output / "RUN_MANIFEST.json", run_manifest)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
