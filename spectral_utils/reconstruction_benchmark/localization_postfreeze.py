"""Complete post-score-freeze ProcessBench and PRMBench evaluation pipeline.

The public entry points in this module never accept caller-supplied metrics or
decisions.  They rederive every atomic row from one independently frozen build,
after fully revalidating the localization and upstream external A/B trees.  The
A/B verifier repeats that derivation for both builds and compares the resulting
bytes, so two identical fabricated tables cannot pass.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import csv
import errno
from io import StringIO
import json
import os
import pickle
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .external_final_answer import (
    apply_external_id_contract,
    identity_key_id,
    keyed_opaque_external_id,
    load_external_registry,
    load_identity_key,
    load_raw_feature_cell,
    resolve_sources,
    verify_sources,
)
from .io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_tree_manifest,
    deterministic_npz_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .localization_ab import (
    DEFAULT_EXTERNAL_REGISTRY,
    DEFAULT_LOCALIZATION_REGISTRY,
    DEFAULT_POPULATION_REGISTRY,
    DEFAULT_SOURCE_ROOT,
    REPO_ROOT,
    assert_localization_ab_certificate,
)
from .localization_contract import (
    load_localization_registry,
    payload_sha256,
    primary_system_roster,
)
from .localization_evaluation import (
    DEFAULT_BOOTSTRAP_DRAWS,
    LOCALIZATION_DECISION_FIELDS,
    METRIC_FIELDS,
    PRMBENCH_BOOTSTRAP_SEED,
    PRMBENCH_ERROR_FAMILIES,
    PROCESSBENCH_BOOTSTRAP_SEED,
    PROCESSBENCH_SUBSETS,
    UNDEFINED_SINGLE_CLASS,
    assign_processbench_folds,
    crossfit_processbench_threshold,
    evaluator_contract,
    grouped_bootstrap_metric_map,
    prmbench_step_metrics,
    processbench_panel_metrics,
)
from .localization_fit import load_localization_score_bundle
from .localization_postfreeze_amendment import (
    DEFAULT_LOCALIZATION_POSTFREEZE_AMENDMENT,
    EXPECTED_SCORE_VERIFIER_GIT_HEAD,
    apply_localization_postfreeze_amendment,
    load_localization_postfreeze_amendment,
    validate_observed_prmbench_oob_audit,
)
from .methods import PRIMARY_METHOD_IDS


EVALUATION_MANIFEST_SCHEMA_VERSION = "reconstruction-localization-evaluation-manifest-v3"
EVALUATION_AB_SCHEMA_VERSION = "reconstruction-localization-evaluation-ab-v3"
PB_METRICS = (
    "official_macro_f1", "first_error_exact", "first_error_within_one",
    "clean_abstention_accuracy", "overall_decision_accuracy",
)
PRM_METRICS = ("auroc", "auprc", "mean_risk", "risk_q90", "coverage")
EXPECTED_ARTIFACTS = (
    "bootstrap_ledger.json", "calibration_ledgers.json", "contrasts_long.csv",
    "coverage_long.csv", "localization_decisions.csv", "metrics_long.csv",
    "prmbench_steps.npz", "source_provenance.json",
)
CONTRAST_FIELDS = (
    "task_id", "dataset_id", "population_id", "cell_id", "slice_id", "model_id",
    "metric_id", "candidate_system_id", "reference_system_id", "delta", "ci_low",
    "ci_high", "n_valid", "status", "comparison_group_id", "bootstrap_unit",
    "bootstrap_draws", "cohort_id", "candidate_run_hash", "reference_run_hash",
    "candidate_access_level", "candidate_fidelity", "reference_access_level",
    "reference_fidelity",
)
COVERAGE_FIELDS = (
    "task_id", "dataset_id", "population_id", "cell_id", "slice_id", "model_id",
    "system_id", "n_expected", "n_scored", "n_excluded", "n_failed", "n_fallback",
    "status", "access_level", "fidelity", "comparison_group_id", "cohort_id",
    "run_hash",
)
EVALUATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/localization.json",
    "configs/reconstruction_benchmark_v1/localization_postfreeze_amendment_v1.json",
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "spectral_utils/reconstruction_benchmark/localization_ab.py",
    "spectral_utils/reconstruction_benchmark/localization_contract.py",
    "spectral_utils/reconstruction_benchmark/localization_evaluation.py",
    "spectral_utils/reconstruction_benchmark/localization_postfreeze.py",
    "spectral_utils/reconstruction_benchmark/localization_postfreeze_amendment.py",
    "scripts/reconstruction_benchmark/evaluate_localization.py",
    "scripts/reconstruction_benchmark/verify_localization_evaluation_ab.py",
)


@dataclass(frozen=True)
class SystemMeta:
    system_id: str
    access_level: str
    fidelity: str
    comparison_group_id: str
    role: str


@dataclass(frozen=True)
class PBCell:
    cell_id: str
    population_id: str
    model_id: str
    slice_id: str
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    first_error: np.ndarray
    segment_offsets: np.ndarray
    core_system_ids: tuple[str, ...]
    core_scores: np.ndarray
    comparator_predictions: Mapping[str, tuple[int | None, ...]]
    comparator_coverage: Mapping[str, np.ndarray]
    run_hashes: Mapping[str, str]
    source_records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PRMPanel:
    cell_id: str
    population_id: str
    model_id: str
    response_row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    error_families: tuple[str, ...]
    step_offsets: np.ndarray
    step_labels: np.ndarray
    system_ids: tuple[str, ...]
    system_scores: np.ndarray
    run_hashes: Mapping[str, str]
    source_records: tuple[Mapping[str, Any], ...]
    postfreeze_amendment_audit: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DerivedLocalizationEvaluation:
    files: Mapping[str, bytes]
    manifest_core: Mapping[str, Any]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _hashed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    if payload.pop("payload_sha256", None) != payload_sha256(payload):
        raise RuntimeError(f"payload hash failed: {path}")
    return value


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    expected = set(fields)
    values = []
    for row in rows:
        if set(row) != expected:
            raise RuntimeError(f"localization tidy schema drifted: {sorted(set(row) ^ expected)}")
        values.append(dict(row))
    values.sort(key=lambda row: tuple(str(row[field]) for field in fields))
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _cohort_id(row_ids: Sequence[str]) -> str:
    return "cohortv1_" + payload_sha256(list(map(str, row_ids)))


def _run_hash(values: Sequence[str]) -> str:
    return "runv1_" + payload_sha256(list(map(str, values)))


def _repo_state(repo: Path) -> dict[str, Any]:
    """Bind repository identity without opening any label-bearing source file."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    value = {
        "git_head": head,
        "git_clean": not bool(status.strip()),
        "git_status_sha256": sha256_bytes(status.encode("utf-8")),
    }
    value["snapshot_sha256"] = payload_sha256(value)
    return value


def _score_verifier_repo_snapshot(repo: Path, *, required_git_head: str) -> dict[str, Any]:
    state = _repo_state(repo)
    if state["git_clean"] is not True:
        raise RuntimeError("score verifier repository must be clean")
    if state["git_head"] != required_git_head:
        raise RuntimeError("score verifier repository is not at the amended frozen HEAD")
    value = {
        "repo_role": "score_ab_verifier",
        "required_git_head": required_git_head,
        **{key: state[key] for key in ("git_head", "git_clean", "git_status_sha256")},
    }
    value["snapshot_sha256"] = payload_sha256(value)
    return value


def _source_snapshot(repo: Path) -> dict[str, Any]:
    state = _repo_state(repo)
    files = [
        {"path": relative, "sha256": sha256_file(repo / relative)}
        for relative in EVALUATION_SOURCE_FILES
    ]
    value = {
        "repo_role": "postfreeze_evaluator",
        **{key: state[key] for key in ("git_head", "git_clean", "git_status_sha256")},
        "files": files,
    }
    value["snapshot_sha256"] = payload_sha256(value)
    return value


def _core_system_meta(dataset_id: str) -> dict[str, SystemMeta]:
    rows = primary_system_roster(PRIMARY_METHOD_IDS)
    output = {}
    for row in rows:
        role = str(row["role"])
        fidelity = (
            "frozen_response_plus_token_midrank_geomean"
            if role == "primary_localization_adapter"
            else "registered_label_free_adapter_null"
        )
        output[row["system_id"]] = SystemMeta(
            system_id=row["system_id"],
            access_level="saved_output_probability_telemetry_one_pass",
            fidelity=fidelity,
            comparison_group_id=f"{dataset_id}_core_common_access",
            role=role,
        )
    return output


def _comparator_meta(config: Mapping[str, Any], dataset_id: str) -> dict[str, SystemMeta]:
    output = {}
    for row in config["comparators"]:
        if row["dataset_id"] != dataset_id:
            continue
        system_id = str(row["system_id"])
        output[system_id] = SystemMeta(
            system_id=system_id,
            access_level=str(row["access_level"]),
            fidelity=str(row["fidelity"]),
            comparison_group_id=f"{dataset_id}_context_{system_id}",
            role="published_or_trained_context",
        )
    return output


def _load_build_scores(
    *, release_root: Path, release_id: str, build_id: str,
    certificate: Mapping[str, Any],
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, np.ndarray], str]], dict[tuple[str, str], tuple[dict[str, Any], dict[str, np.ndarray], str]], dict[str, Any]]:
    root = release_root / release_id / f"build_{build_id}/localization"
    freeze_path = root / "fit/SCORE_FREEZE_MANIFEST.json"
    freeze = _hashed_json(freeze_path)
    build_binding = certificate.get("builds", {}).get(build_id, {})
    if (
        freeze.get("schema_version") != "reconstruction-localization-score-freeze-v1"
        or freeze.get("release_id") != release_id
        or freeze.get("build_id") != build_id
        or freeze.get("scientific_full") is not True
        or freeze.get("target_data_opened") is not False
        or freeze.get("response_scores_refit") is not False
        or sha256_file(freeze_path) != build_binding.get("score_freeze_sha256")
        or canonical_tree_manifest(root / "fit")["tree_sha256"]
        != build_binding.get("fit_tree_sha256")
        or canonical_tree_manifest(root / "comparator_projections")["tree_sha256"]
        != build_binding.get("projection_tree_sha256")
    ):
        raise RuntimeError("post-freeze score tree no longer matches localization A/B certificate")
    core = {}
    seen = set()
    for summary in freeze["records"]:
        cell_id = str(summary["cell_id"])
        if cell_id in seen:
            raise RuntimeError("duplicate localization score summary during evaluation")
        seen.add(cell_id)
        record_path = (root / "fit" / str(summary["record_path"])).resolve()
        try:
            record_path.relative_to((root / "fit").resolve())
        except ValueError as exc:
            raise RuntimeError("post-freeze score record escaped the fit tree") from exc
        if sha256_file(record_path) != summary.get("record_file_sha256"):
            raise RuntimeError(f"{cell_id}: post-freeze record file hash drifted")
        record, arrays = load_localization_score_bundle(record_path)
        expected = {
            "cell_id": record["cell_id"],
            "population_id": record["population_id"],
            "dataset_id": record["dataset_id"],
            "model_id": record["model_id"],
            "slice_id": record["slice_id"],
            "record_sha256": record["record_sha256"],
            "score_sha256": record["score_sha256"],
            "n_rows": record["n_rows"],
            "n_segments": record["n_segments"],
            "n_systems": record["n_systems"],
        }
        if any(summary.get(field) != value for field, value in expected.items()):
            raise RuntimeError(f"{cell_id}: post-freeze record/summary binding drifted")
        if (
            tuple(record.get("system_ids", ()))
            != tuple(row["system_id"] for row in primary_system_roster(PRIMARY_METHOD_IDS))
            or tuple(record.get("method_ids", ()))
            != tuple(row["method_id"] for row in primary_system_roster(PRIMARY_METHOD_IDS))
            or tuple(record.get("adapter_ids", ()))
            != tuple(row["adapter_id"] for row in primary_system_roster(PRIMARY_METHOD_IDS))
            or int(record.get("n_systems", -1)) != 27
        ):
            raise RuntimeError(f"{cell_id}: post-freeze system roster is not canonical")
        score_file = record_path.parent / str(record["score_path"])
        core[cell_id] = (record, arrays, sha256_file(score_file))
    projection_root = root / "comparator_projections"
    projection_manifest = _hashed_json(projection_root / "MANIFEST.json")
    projections = {}
    for record in projection_manifest["records"]:
        key = (str(record["system_id"]), str(record["cell_id"]))
        if key in projections:
            raise RuntimeError("duplicate comparator projection during evaluation")
        path = (projection_root / str(record["artifact_path"])).resolve()
        try:
            path.relative_to(projection_root.resolve())
        except ValueError as exc:
            raise RuntimeError("post-freeze comparator projection escaped its tree") from exc
        if sha256_file(path) != record["artifact_sha256"]:
            raise RuntimeError(f"comparator projection changed before evaluation: {key}")
        projections[key] = (dict(record), load_npz_no_pickle(path), sha256_file(path))
    if len(core) != 13 or len(projections) != 37:
        raise RuntimeError("post-freeze score/projection roster is not exact 13 cells/37 projections")
    return core, projections, freeze


def _ordered_source_rows(
    *, registry: Any, spec: Any, source_root: Path, identity_key: bytes
) -> tuple[Any, list[Mapping[str, Any]], Any, tuple[Mapping[str, Any], ...]]:
    sources = resolve_sources(registry, spec, repo=source_root)
    verified = tuple(verify_sources(sources, include_labels=True))
    raw = load_raw_feature_cell(spec, sources)
    cache = _load_pickle(sources.feature_files[0].path)
    if not isinstance(cache, Mapping):
        raise RuntimeError(f"{spec.cell_id}: post-freeze source is not a mapping")
    key_name = "id" if spec.dataset_id == "processbench" else "idx"
    by_id = {str(row.get(key_name, "")): row for row in cache.values()}
    if len(by_id) != len(cache) or "" in by_id:
        raise RuntimeError(f"{spec.cell_id}: post-freeze raw IDs are empty/duplicated")
    if not all(raw_id and raw_id in by_id for raw_id in raw.row_ids):
        raise RuntimeError(f"{spec.cell_id}: post-freeze source roster does not join")
    identity = apply_external_id_contract(
        registry, spec, raw.row_ids, raw.group_ids, identity_key=identity_key
    )
    order = sorted(range(len(identity.row_ids)), key=lambda index: identity.row_ids[index])
    rows = [by_id[raw.row_ids[index]] for index in order]
    ordered_identity = type(identity)(
        row_ids=tuple(identity.row_ids[index] for index in order),
        group_ids=tuple(identity.group_ids[index] for index in order),
        contract_binding=identity.contract_binding,
        row_namespace_sha256=identity.row_namespace_sha256,
        group_namespace_sha256=identity.group_namespace_sha256,
    )
    return raw, rows, ordered_identity, verified


def _processbench_group_id(
    *, identity_key: bytes, slice_id: str, raw_id: str
) -> str:
    return keyed_opaque_external_id(
        identity_key=identity_key,
        kind="group",
        namespace={
            "contract_version": "reconstruction-localization-pb-group-v1",
            "dataset_id": "processbench",
            "slice_id": str(slice_id),
            "scope": "shared_across_all_scorer_models",
        },
        raw=raw_id,
    )


def _load_processbench_cells(
    *,
    config: Mapping[str, Any],
    registry: Any,
    source_root: Path,
    identity_key: bytes,
    core: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray], str]],
    projections: Mapping[tuple[str, str], tuple[dict[str, Any], dict[str, np.ndarray], str]],
) -> dict[str, PBCell]:
    comparator_ids = tuple(_comparator_meta(config, "processbench"))
    cells = {}
    label_roster_by_subset: dict[str, tuple[tuple[str, int], ...]] = {}
    for cell_id in config["processbench"]["source_cells"]:
        spec = registry.by_cell[cell_id]
        record, arrays, core_hash = core[cell_id]
        _raw, rows, identity, verified = _ordered_source_rows(
            registry=registry, spec=spec, source_root=source_root, identity_key=identity_key
        )
        row_ids = tuple(map(str, arrays["row_ids"].tolist()))
        if row_ids != identity.row_ids:
            raise RuntimeError(f"{cell_id}: post-freeze ProcessBench row join failed")
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        core_scores = np.asarray(arrays["system_scores"], dtype=np.float64)
        core_system_ids = tuple(map(str, arrays["system_ids"].tolist()))
        labels = []
        group_ids = []
        raw_label_roster = []
        for index, row in enumerate(rows):
            label = int(row.get("label"))
            steps = row.get("steps")
            spans = row.get("step_token_spans")
            n_segments = int(offsets[index + 1] - offsets[index])
            if (
                label < -1
                or not isinstance(steps, Sequence)
                or isinstance(steps, (str, bytes))
                or not isinstance(spans, Sequence)
                or isinstance(spans, (str, bytes))
                or len(steps) != n_segments
                or len(spans) != n_segments
                or (label != -1 and label >= n_segments)
            ):
                raise RuntimeError(f"{cell_id}: invalid post-freeze first-error target")
            raw_id = str(row["id"])
            labels.append(label)
            group_ids.append(_processbench_group_id(
                identity_key=identity_key, slice_id=spec.slice_id, raw_id=raw_id
            ))
            raw_label_roster.append((raw_id, label))
        expected_balance = config["processbench"]["expected_first_error_balance_by_subset"][spec.slice_id]
        n_error = sum(value != -1 for value in labels)
        n_clean = len(labels) - n_error
        if n_error != int(expected_balance["error"]) or n_clean != int(expected_balance["clean"]):
            raise RuntimeError(f"{cell_id}: ProcessBench first-error class balance drifted")
        roster = tuple(sorted(raw_label_roster))
        previous = label_roster_by_subset.setdefault(str(spec.slice_id), roster)
        if previous != roster:
            raise RuntimeError(f"{cell_id}: ProcessBench labels differ across scorer models")

        comparator_predictions = {}
        comparator_coverage = {}
        run_hashes = {
            system_id: _run_hash([core_hash, system_id])
            for system_id in core_system_ids
        }
        for system_id in comparator_ids:
            projection_record, projection_arrays, projection_hash = projections[(system_id, cell_id)]
            projection_rows = tuple(map(str, projection_arrays["row_ids"].tolist()))
            if projection_rows != row_ids:
                raise RuntimeError(f"{cell_id}/{system_id}: comparator row cohort differs")
            coverage = np.asarray(projection_arrays["coverage"], dtype=np.int8)
            native = np.asarray(projection_arrays["native_prediction"], dtype=np.int64)
            decisions: list[int | None] = []
            for index, (available, prediction) in enumerate(zip(coverage, native)):
                if not int(available):
                    decisions.append(None)
                    continue
                value = int(prediction)
                n_segments = int(offsets[index + 1] - offsets[index])
                if value < -1 or value >= n_segments:
                    raise RuntimeError(
                        f"{cell_id}/{system_id}: native comparator prediction is out of range"
                    )
                decisions.append(value)
            comparator_predictions[system_id] = tuple(decisions)
            comparator_coverage[system_id] = coverage
            run_hashes[system_id] = _run_hash([projection_hash, system_id])
            if projection_record["dataset_id"] != "processbench":
                raise RuntimeError("ProcessBench comparator projection metadata drifted")
        cells[cell_id] = PBCell(
            cell_id=cell_id,
            population_id=str(config["processbench"]["population_id_by_model"][spec.model_id]),
            model_id=str(spec.model_id),
            slice_id=str(spec.slice_id),
            row_ids=row_ids,
            group_ids=tuple(group_ids),
            first_error=np.asarray(labels, dtype=np.int64),
            segment_offsets=offsets,
            core_system_ids=core_system_ids,
            core_scores=core_scores,
            comparator_predictions=comparator_predictions,
            comparator_coverage=comparator_coverage,
            run_hashes=run_hashes,
            source_records=verified,
        )
    if len(cells) != 12:
        raise RuntimeError("ProcessBench evaluation did not load the exact 12 cells")
    return cells


def _partition_prmbench_error_steps(
    error_steps: Any, *, n_steps: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Apply the official one-based membership semantics without repair."""

    if type(n_steps) is not int or n_steps < 1:
        raise RuntimeError("PRMBench realized step count is malformed")
    if (
        not isinstance(error_steps, Sequence)
        or isinstance(error_steps, (str, bytes))
        or any(type(value) is not int for value in error_steps)
    ):
        raise RuntimeError("PRMBench error_steps must contain exact one-based integers")
    values = tuple(error_steps)
    if len(values) != len(set(values)):
        raise RuntimeError("PRMBench error_steps contain duplicate annotations")
    if any(value < 1 for value in values):
        raise RuntimeError("PRMBench error_steps contain zero/negative annotations")
    return (
        tuple(value for value in values if value <= n_steps),
        tuple(value for value in values if value > n_steps),
    )


def _load_prmbench_panel(
    *,
    config: Mapping[str, Any],
    amendment: Mapping[str, Any],
    registry: Any,
    source_root: Path,
    identity_key: bytes,
    core: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray], str]],
    projections: Mapping[tuple[str, str], tuple[dict[str, Any], dict[str, np.ndarray], str]],
) -> PRMPanel:
    cell_id = str(config["prmbench"]["source_cell"])
    spec = registry.by_cell[cell_id]
    _record, arrays, core_hash = core[cell_id]
    _raw, rows, identity, verified = _ordered_source_rows(
        registry=registry, spec=spec, source_root=source_root, identity_key=identity_key
    )
    full_row_ids = tuple(map(str, arrays["row_ids"].tolist()))
    if full_row_ids != identity.row_ids:
        raise RuntimeError("PRMBench post-freeze row join failed")
    full_offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
    full_core_scores = np.asarray(arrays["system_scores"], dtype=np.float64)
    core_system_ids = tuple(map(str, arrays["system_ids"].tolist()))
    selected = [index for index, row in enumerate(rows) if str(row.get("classification")) != "correct"]
    response_row_ids = tuple(full_row_ids[index] for index in selected)
    group_ids = tuple(identity.group_ids[index] for index in selected)
    families = tuple(str(rows[index]["classification"]) for index in selected)
    if len(response_row_ids) != int(config["prmbench"]["expected_error_responses"]):
        raise RuntimeError("PRMBench error-response roster count drifted")
    if len(set(group_ids)) != len(group_ids):
        raise RuntimeError("PRMBench source_idx groups are not one-per-error-response")
    if set(families) != set(PRMBENCH_ERROR_FAMILIES):
        raise RuntimeError("PRMBench does not contain the exact nine error families")

    score_parts = []
    step_labels = []
    step_offsets = [0]
    oob_records: list[dict[str, Any]] = []
    all_annotations: list[int] = []
    zero_count = 0
    negative_count = 0
    duplicate_annotation_rows = 0
    for index in selected:
        lo, hi = map(int, full_offsets[index:index + 2])
        row = rows[index]
        n_steps = hi - lo
        steps = row.get("steps")
        spans = row.get("step_token_spans")
        error_steps = row.get("error_steps")
        if (
            not isinstance(steps, Sequence)
            or isinstance(steps, (str, bytes))
            or not isinstance(spans, Sequence)
            or isinstance(spans, (str, bytes))
            or len(steps) != n_steps
            or len(spans) != n_steps
            or not isinstance(error_steps, Sequence)
            or isinstance(error_steps, (str, bytes))
        ):
            raise RuntimeError("PRMBench step target/span roster is malformed")
        if isinstance(error_steps, Sequence) and not isinstance(error_steps, (str, bytes)):
            exact_values = list(error_steps)
            if all(type(value) is int for value in exact_values):
                all_annotations.extend(exact_values)
                zero_count += sum(value == 0 for value in exact_values)
                negative_count += sum(value < 0 for value in exact_values)
                duplicate_annotation_rows += int(len(exact_values) != len(set(exact_values)))
        valid_errors, invalid_errors = _partition_prmbench_error_steps(
            error_steps, n_steps=n_steps,
        )
        errors = set(valid_errors)
        if invalid_errors:
            oob_records.append({
                "idx": str(row.get("idx")),
                "family": str(row.get("classification")),
                "n_steps": n_steps,
                "error_steps": list(error_steps),
                "invalid": list(invalid_errors),
            })
        step_labels.extend(int(step_index + 1 in errors) for step_index in range(n_steps))
        score_parts.append(full_core_scores[:, lo:hi])
        step_offsets.append(step_offsets[-1] + n_steps)
    amendment_audit = validate_observed_prmbench_oob_audit(
        oob_records,
        amendment,
        all_annotation_count=len(all_annotations),
        minimum_annotation=min(all_annotations) if all_annotations else 0,
        zero_count=zero_count,
        negative_count=negative_count,
        duplicate_annotation_rows=duplicate_annotation_rows,
    )
    matrix_parts = [np.concatenate(score_parts, axis=1)]
    system_ids = list(core_system_ids)
    run_hashes = {
        system_id: _run_hash([core_hash, system_id])
        for system_id in core_system_ids
    }

    comparator_meta = _comparator_meta(config, "prmbench")
    for system_id in comparator_meta:
        projection_record, projection_arrays, projection_hash = projections[(system_id, cell_id)]
        projection_rows = tuple(map(str, projection_arrays["row_ids"].tolist()))
        projection_offsets = np.asarray(projection_arrays["score_offsets"], dtype=np.int64)
        projection_scores = np.asarray(projection_arrays["score"], dtype=np.float64)
        if projection_rows != full_row_ids:
            raise RuntimeError("PRMBench comparator row cohort differs")
        parts = []
        for index in selected:
            lo, hi = map(int, projection_offsets[index:index + 2])
            expected = int(full_offsets[index + 1] - full_offsets[index])
            if hi - lo != expected:
                raise RuntimeError("PRMBench comparator step count differs from telemetry")
            parts.append(projection_scores[lo:hi])
        matrix_parts.append(np.concatenate(parts)[None, :])
        system_ids.append(system_id)
        run_hashes[system_id] = _run_hash([projection_hash, system_id])
        if projection_record["dataset_id"] != "prmbench":
            raise RuntimeError("PRMBench comparator projection metadata drifted")
    system_scores = np.vstack(matrix_parts)
    labels_array = np.asarray(step_labels, dtype=np.int8)
    offsets_array = np.asarray(step_offsets, dtype=np.int64)
    if (
        system_scores.shape != (28, int(config["prmbench"]["expected_steps"]))
        or len(labels_array) != int(config["prmbench"]["expected_steps"])
        or int(labels_array.sum()) != int(config["prmbench"]["expected_positive_steps"])
    ):
        raise RuntimeError("PRMBench exact response/step/system roster drifted")
    expected_by_family = config["prmbench"]["expected_by_family"]
    for family in PRMBENCH_ERROR_FAMILIES:
        response_indices = [i for i, value in enumerate(families) if value == family]
        step_indices = np.concatenate([
            np.arange(offsets_array[index], offsets_array[index + 1], dtype=np.int64)
            for index in response_indices
        ])
        expected = expected_by_family[family]
        observed = {
            "responses": len(response_indices),
            "steps": len(step_indices),
            "positive_steps": int(labels_array[step_indices].sum()),
        }
        if observed != {key: int(value) for key, value in expected.items()}:
            raise RuntimeError(f"PRMBench registered family counts drifted: {family}")
    if prmbench_step_metrics(
        labels_array[
            np.concatenate([
                np.arange(offsets_array[index], offsets_array[index + 1], dtype=np.int64)
                for index, family in enumerate(families) if family == "multi_solutions"
            ])
        ],
        system_scores[0, np.concatenate([
            np.arange(offsets_array[index], offsets_array[index + 1], dtype=np.int64)
            for index, family in enumerate(families) if family == "multi_solutions"
        ])],
    )["status"] != UNDEFINED_SINGLE_CLASS:
        raise RuntimeError("PRMBench multi_solutions is no longer single-class")
    return PRMPanel(
        cell_id=cell_id,
        population_id=str(config["prmbench"]["population_id"]),
        model_id=str(config["prmbench"]["model_id"]),
        response_row_ids=response_row_ids,
        group_ids=group_ids,
        error_families=families,
        step_offsets=offsets_array,
        step_labels=labels_array,
        system_ids=tuple(system_ids),
        system_scores=system_scores,
        run_hashes=run_hashes,
        source_records=verified,
        postfreeze_amendment_audit=amendment_audit,
    )


def _require_bootstrap_execution(
    result: Mapping[str, Any], *, draws: int, unit: str,
) -> None:
    """Reject declared draws that were not actually sampled and evaluated."""

    if (
        int(result.get("draws", -1)) != int(draws)
        or int(result.get("draws_executed", -1)) != int(draws)
        or result.get("bootstrap_unit") != unit
        or result.get("paired_payload") is not True
        or not isinstance(result.get("draw_stream_sha256"), str)
        or len(str(result.get("draw_stream_sha256"))) != 64
        or not isinstance(result.get("sample_stream_sha256"), str)
        or len(str(result.get("sample_stream_sha256"))) != 64
    ):
        raise RuntimeError("bootstrap execution did not perform the exact registered draws")
    samples = result.get("samples")
    if not isinstance(samples, Mapping) or set(samples) != set(result.get("statistics", {})):
        raise RuntimeError("paired bootstrap samples are missing or have a different roster")
    if any(len(values) != int(draws) for values in samples.values()):
        raise RuntimeError("bootstrap sample vector length differs from executed draws")
    for metric_id, summary in result["statistics"].items():
        values = np.asarray([
            np.nan if value is None else float(value) for value in samples[metric_id]
        ], dtype=np.float64)
        n_valid = int(np.isfinite(values).sum())
        if int(summary.get("n_valid", -1)) != n_valid:
            raise RuntimeError("bootstrap interval does not bind its actual valid draws")


def _bootstrap_ledger_row(
    *, execution_id: str, dataset_id: str, model_id: str, slice_id: str,
    system_id: str, result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_id": execution_id,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "slice_id": slice_id,
        "system_id": system_id,
        "draws": int(result["draws"]),
        "draws_executed": int(result["draws_executed"]),
        "seed": int(result["seed"]),
        "bootstrap_unit": str(result["bootstrap_unit"]),
        "n_groups": int(result["n_groups"]),
        "n_groups_by_stratum": dict(result["n_groups_by_stratum"]),
        "draw_stream_sha256": str(result["draw_stream_sha256"]),
        "sample_stream_sha256": str(result["sample_stream_sha256"]),
        "metric_n_valid": {
            metric_id: int(summary["n_valid"])
            for metric_id, summary in sorted(result["statistics"].items())
        },
    }


def _float_or_blank(value: Any) -> float | str:
    if value is None:
        return ""
    number = float(value)
    return number if np.isfinite(number) else ""


def _pb_reference_system(system_id: str) -> str:
    if system_id.endswith("__response_only_null_v1"):
        return "iu_pcr__response_only_null_v1"
    return "iu_pcr__loc_geomean_v1"


def _paired_contrast(
    *, candidate: Mapping[str, Any], reference: Mapping[str, Any],
    candidate_samples: Sequence[float | None],
    reference_samples: Sequence[float | None], draws: int,
) -> tuple[float | str, float | str, float | str, int, str]:
    if len(candidate_samples) != draws or len(reference_samples) != draws:
        raise RuntimeError("paired contrast did not receive the exact draw count")
    left = np.asarray([
        np.nan if value is None else float(value) for value in candidate_samples
    ], dtype=np.float64)
    right = np.asarray([
        np.nan if value is None else float(value) for value in reference_samples
    ], dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    deltas = left[valid] - right[valid]
    point = (
        "" if candidate["value"] == "" or reference["value"] == ""
        else float(candidate["value"]) - float(reference["value"])
    )
    if not len(deltas):
        return point, "", "", 0, UNDEFINED_SINGLE_CLASS
    low, high = np.percentile(deltas, [2.5, 97.5])
    return point, float(low), float(high), int(len(deltas)), "OK"


def _shared_group_draw_counts(
    *, group_ids: Sequence[str], strata: Sequence[str], draws: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one exact grouped draw stream for every system in a slice."""

    if len(group_ids) != len(strata) or not group_ids:
        raise ValueError("shared bootstrap group/stratum rows must align")
    group_strata: dict[str, str] = {}
    for group_id, stratum in zip(map(str, group_ids), map(str, strata)):
        previous = group_strata.setdefault(group_id, stratum)
        if previous != stratum:
            raise RuntimeError("one bootstrap source group crosses registered strata")
    roster = tuple(sorted(group_strata))
    group_position = {group_id: index for index, group_id in enumerate(roster)}
    row_group_index = np.asarray(
        [group_position[str(group_id)] for group_id in group_ids], dtype=np.int32
    )
    by_stratum: dict[str, list[int]] = {}
    for group_id in roster:
        by_stratum.setdefault(group_strata[group_id], []).append(group_position[group_id])
    counts = np.zeros((draws, len(roster)), dtype=np.uint16)
    rng = np.random.default_rng(int(seed))
    draw_hasher = __import__("hashlib").sha256()
    draws_executed = 0
    for draw_index in range(draws):
        for stratum in sorted(by_stratum):
            source = np.asarray(by_stratum[stratum], dtype=np.int32)
            picks = rng.integers(0, len(source), size=len(source))
            draw_hasher.update(stratum.encode("utf-8") + b"\0")
            draw_hasher.update(np.asarray(picks, dtype="<i8").tobytes(order="C"))
            selected = source[picks]
            values = np.bincount(selected, minlength=len(roster))
            if int(values.max(initial=0)) > np.iinfo(np.uint16).max:
                raise RuntimeError("bootstrap group multiplicity exceeds uint16")
            counts[draw_index] += values.astype(np.uint16)
        draws_executed = draw_index + 1
    metadata = {
        "draws": draws,
        "draws_executed": draws_executed,
        "seed": int(seed),
        "n_groups": len(roster),
        "n_groups_by_stratum": {
            key: len(value) for key, value in sorted(by_stratum.items())
        },
        "draw_stream_sha256": draw_hasher.hexdigest(),
    }
    return counts, row_group_index, metadata


def _sample_stream_sha256(samples: Mapping[str, Sequence[float]]) -> str:
    hasher = __import__("hashlib").sha256()
    for metric_id in sorted(samples):
        hasher.update(metric_id.encode("utf-8") + b"\0")
        values = np.asarray(samples[metric_id], dtype="<f8")
        values = np.where(np.isnan(values), np.float64(np.nan), values).astype("<f8")
        hasher.update(values.tobytes(order="C"))
    return hasher.hexdigest()


def _intervals_from_samples(
    *, point: Mapping[str, float | None], samples: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    output = {}
    for metric_id, values_raw in samples.items():
        values = np.asarray(values_raw, dtype=np.float64)
        valid = values[np.isfinite(values)]
        low, high = (
            np.percentile(valid, [2.5, 97.5])
            if len(valid) else (float("nan"), float("nan"))
        )
        value = point[metric_id]
        output[metric_id] = {
            "point": None if value is None else float(value),
            "ci_low": None if not len(valid) else float(low),
            "ci_high": None if not len(valid) else float(high),
            "n_valid": int(len(valid)),
            "status": "OK" if len(valid) else UNDEFINED_SINGLE_CLASS,
        }
    return output


def _weighted_prm_samples(
    *, labels: np.ndarray, scores: np.ndarray, row_group_index: np.ndarray,
    draw_counts: np.ndarray, chunk_size: int = 32,
) -> dict[str, np.ndarray]:
    """Exact source-group bootstrap metrics without resampling/sorting each draw."""

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(row_group_index, dtype=np.int32)
    if y.shape != s.shape or groups.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("weighted PRMBench bootstrap arrays must be finite and aligned")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    sorted_groups = groups[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_scores) != 0)].astype(np.int64)
    tie_scores = sorted_scores[starts]
    n_draws = len(draw_counts)
    samples = {
        metric_id: np.full(n_draws, np.nan, dtype=np.float64)
        for metric_id in PRM_METRICS
    }
    samples["coverage"].fill(1.0)
    y_pos = sorted_y.astype(np.uint16)
    y_neg = (1 - sorted_y).astype(np.uint16)
    for lo in range(0, n_draws, int(chunk_size)):
        hi = min(n_draws, lo + int(chunk_size))
        step_weight = draw_counts[lo:hi, sorted_groups]
        positive = np.add.reduceat(
            step_weight * y_pos[None, :], starts, axis=1, dtype=np.int32
        )
        negative = np.add.reduceat(
            step_weight * y_neg[None, :], starts, axis=1, dtype=np.int32
        )
        total_positive = positive.sum(axis=1, dtype=np.int64)
        total_negative = negative.sum(axis=1, dtype=np.int64)
        total = total_positive + total_negative
        cumulative_negative = np.cumsum(negative, axis=1, dtype=np.int64)
        numerator = np.sum(
            positive * (cumulative_negative - negative + 0.5 * negative), axis=1
        )
        denominator = total_positive * total_negative
        valid_auc = denominator > 0
        auc = np.full(hi - lo, np.nan, dtype=np.float64)
        auc[valid_auc] = numerator[valid_auc] / denominator[valid_auc]
        samples["auroc"][lo:hi] = auc

        positive_desc = positive[:, ::-1]
        negative_desc = negative[:, ::-1]
        cumulative_positive = np.cumsum(positive_desc, axis=1, dtype=np.int64)
        cumulative_seen = cumulative_positive + np.cumsum(
            negative_desc, axis=1, dtype=np.int64
        )
        precision = np.divide(
            cumulative_positive, cumulative_seen,
            out=np.zeros_like(cumulative_positive, dtype=np.float64),
            where=cumulative_seen > 0,
        )
        ap_numerator = np.sum(positive_desc * precision, axis=1)
        ap = np.full(hi - lo, np.nan, dtype=np.float64)
        valid_ap = total_positive > 0
        ap[valid_ap] = ap_numerator[valid_ap] / total_positive[valid_ap]
        samples["auprc"][lo:hi] = ap

        block_weight = positive + negative
        weighted_sum = block_weight @ tie_scores
        samples["mean_risk"][lo:hi] = weighted_sum / total
        cumulative = np.cumsum(block_weight, axis=1, dtype=np.int64)
        rank = 0.90 * (total - 1)
        lower_rank = np.floor(rank).astype(np.int64)
        upper_rank = np.ceil(rank).astype(np.int64)
        lower_index = np.argmax(cumulative > lower_rank[:, None], axis=1)
        upper_index = np.argmax(cumulative > upper_rank[:, None], axis=1)
        fraction = rank - lower_rank
        samples["risk_q90"][lo:hi] = (
            tie_scores[lower_index] * (1.0 - fraction)
            + tie_scores[upper_index] * fraction
        )
    return samples


def _shared_prm_bootstrap_results(
    *, labels: np.ndarray, score_matrix: np.ndarray, group_ids: Sequence[str],
    strata: Sequence[str], system_ids: Sequence[str], draws: int, seed: int,
) -> dict[str, dict[str, Any]]:
    counts, group_index, metadata = _shared_group_draw_counts(
        group_ids=group_ids, strata=strata, draws=draws, seed=seed
    )
    output = {}
    for system_index, system_id in enumerate(system_ids):
        point_full = prmbench_step_metrics(labels, score_matrix[system_index])
        point = {metric_id: point_full[metric_id] for metric_id in PRM_METRICS}
        samples = _weighted_prm_samples(
            labels=labels, scores=score_matrix[system_index],
            row_group_index=group_index, draw_counts=counts,
        )
        result = {
            **metadata,
            "alpha": 0.05,
            "bootstrap_unit": "source_idx",
            "paired_payload": True,
            "statistics": _intervals_from_samples(point=point, samples=samples),
            "samples": samples,
            "sample_stream_sha256": _sample_stream_sha256(samples),
        }
        _require_bootstrap_execution(result, draws=draws, unit="source_idx")
        output[str(system_id)] = result
    return output


def _processbench_rows_for_system(
    cells: Sequence[PBCell], *, system_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        system_index = (
            cell.core_system_ids.index(system_id)
            if system_id in cell.core_system_ids else None
        )
        for row_index, row_id in enumerate(cell.row_ids):
            row = {
                "row_id": row_id,
                "group_id": cell.group_ids[row_index],
                "slice_id": cell.slice_id,
                "first_error": int(cell.first_error[row_index]),
                "bootstrap_stratum": (
                    f"{cell.slice_id}::"
                    f"{'error' if int(cell.first_error[row_index]) != -1 else 'clean'}"
                ),
            }
            if system_index is not None:
                lo, hi = map(int, cell.segment_offsets[row_index:row_index + 2])
                row["step_scores"] = cell.core_scores[system_index, lo:hi].tolist()
            else:
                row["prediction_step"] = cell.comparator_predictions[system_id][row_index]
            rows.append(row)
    return rows


def _pb_bootstrap_statistic(
    sample: list[Mapping[str, Any]],
) -> Mapping[str, float | None]:
    panel = processbench_panel_metrics(
        sample, [row["prediction_step"] for row in sample]
    )
    output: dict[str, float | None] = {}
    for subset in PROCESSBENCH_SUBSETS:
        for metric_id in PB_METRICS:
            output[f"{subset}::{metric_id}"] = panel["per_subset"][subset][metric_id]
    for metric_id in PB_METRICS:
        output[f"all_four_subsets::{metric_id}"] = panel["aggregate"][metric_id]
    return output


def _shared_pb_bootstrap_results(
    *, rows: Sequence[Mapping[str, Any]],
    predictions_by_system: Mapping[str, Sequence[int | None]],
    draws: int, seed: int,
) -> dict[str, dict[str, Any]]:
    """Exact paired ProcessBench bootstrap using one shared group-count matrix."""

    rows = list(rows)
    group_ids = [str(row["group_id"]) for row in rows]
    strata = [str(row["bootstrap_stratum"]) for row in rows]
    counts, group_index, metadata = _shared_group_draw_counts(
        group_ids=group_ids, strata=strata, draws=draws, seed=seed
    )
    n_groups = int(metadata["n_groups"])

    def weighted(values: np.ndarray) -> np.ndarray:
        grouped = np.bincount(
            group_index, weights=np.asarray(values, dtype=np.float64), minlength=n_groups
        )
        return np.asarray(counts @ grouped, dtype=np.float64)

    labels = np.asarray([int(row["first_error"]) for row in rows], dtype=np.int64)
    slices = np.asarray([str(row["slice_id"]) for row in rows], dtype="<U32")
    denominators = {}
    for subset in PROCESSBENCH_SUBSETS:
        subset_mask = slices == subset
        error_mask = subset_mask & (labels != -1)
        clean_mask = subset_mask & (labels == -1)
        denominators[subset] = {
            "error": weighted(error_mask),
            "clean": weighted(clean_mask),
            "all": weighted(subset_mask),
        }
        if (
            np.any(denominators[subset]["error"] <= 0)
            or np.any(denominators[subset]["clean"] <= 0)
        ):
            raise RuntimeError("stratified ProcessBench bootstrap lost class support")

    output = {}
    for system_id, predictions_raw in predictions_by_system.items():
        if len(predictions_raw) != len(rows):
            raise RuntimeError("ProcessBench shared prediction vector is misaligned")
        parsed = np.asarray([value is not None for value in predictions_raw], dtype=bool)
        predictions = np.asarray([
            -2 if value is None else int(value) for value in predictions_raw
        ], dtype=np.int64)
        samples: dict[str, np.ndarray] = {}
        per_subset_point = processbench_panel_metrics(rows, predictions_raw)["per_subset"]
        for subset in PROCESSBENCH_SUBSETS:
            subset_mask = slices == subset
            error_mask = subset_mask & (labels != -1)
            clean_mask = subset_mask & (labels == -1)
            exact = weighted(error_mask & parsed & (predictions == labels))
            exact /= denominators[subset]["error"]
            abstention = weighted(clean_mask & parsed & (predictions == -1))
            abstention /= denominators[subset]["clean"]
            denominator = exact + abstention
            official = np.divide(
                2.0 * exact * abstention, denominator,
                out=np.zeros_like(denominator), where=denominator > 0,
            )
            within_one = weighted(
                error_mask & parsed & (predictions != -1)
                & (np.abs(predictions - labels) <= 1)
            ) / denominators[subset]["error"]
            overall = weighted(subset_mask & parsed & (predictions == labels))
            overall /= denominators[subset]["all"]
            samples[f"{subset}::official_macro_f1"] = official
            samples[f"{subset}::first_error_exact"] = exact
            samples[f"{subset}::first_error_within_one"] = within_one
            samples[f"{subset}::clean_abstention_accuracy"] = abstention
            samples[f"{subset}::overall_decision_accuracy"] = overall
        for metric_id in PB_METRICS:
            samples[f"all_four_subsets::{metric_id}"] = np.mean(
                np.vstack([
                    samples[f"{subset}::{metric_id}"] for subset in PROCESSBENCH_SUBSETS
                ]), axis=0,
            )
        point = {
            f"{subset}::{metric_id}": per_subset_point[subset][metric_id]
            for subset in PROCESSBENCH_SUBSETS for metric_id in PB_METRICS
        }
        panel_point = processbench_panel_metrics(rows, predictions_raw)["aggregate"]
        point.update({
            f"all_four_subsets::{metric_id}": panel_point[metric_id]
            for metric_id in PB_METRICS
        })
        result = {
            **metadata,
            "alpha": 0.05,
            "bootstrap_unit": "source_question",
            "paired_payload": True,
            "statistics": _intervals_from_samples(point=point, samples=samples),
            "samples": samples,
            "sample_stream_sha256": _sample_stream_sha256(samples),
        }
        _require_bootstrap_execution(result, draws=draws, unit="source_question")
        output[str(system_id)] = result
    return output


def _evaluate_processbench(
    *, config: Mapping[str, Any], cells_by_id: Mapping[str, PBCell], draws: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
]:
    core_meta = _core_system_meta("processbench")
    comparator_meta = _comparator_meta(config, "processbench")
    systems = {**core_meta, **comparator_meta}
    if len(core_meta) != 27 or len(comparator_meta) != 3 or len(systems) != 30:
        raise RuntimeError("ProcessBench system roster must be exact 27 core + 3 context")

    decisions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    metric_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    samples_index: dict[tuple[str, str, str, str], Sequence[float | None]] = {}
    fold_hashes: set[str] = set()

    for model_id in map(str, config["processbench"]["models"]):
        cells = sorted(
            (cell for cell in cells_by_id.values() if cell.model_id == model_id),
            key=lambda cell: PROCESSBENCH_SUBSETS.index(cell.slice_id),
        )
        if tuple(cell.slice_id for cell in cells) != PROCESSBENCH_SUBSETS:
            raise RuntimeError(f"{model_id}: ProcessBench four-subset roster is incomplete")
        cell_by_slice = {cell.slice_id: cell for cell in cells}
        population_ids = {cell.population_id for cell in cells}
        if len(population_ids) != 1:
            raise RuntimeError(f"{model_id}: ProcessBench population binding differs by subset")
        population_id = next(iter(population_ids))
        system_rows_by_id: dict[str, list[dict[str, Any]]] = {}
        predictions_by_id: dict[str, list[int | None]] = {}

        for system_id, meta in systems.items():
            system_rows = _processbench_rows_for_system(cells, system_id=system_id)
            if system_id in core_meta:
                fitted = crossfit_processbench_threshold(system_rows)
                predictions = fitted["predictions"]
                fold_hashes.add(str(fitted["fold_assignment_sha256"]))
                for ledger in fitted["calibration_ledgers"]:
                    value = {
                        **ledger,
                        "dataset_id": "processbench",
                        "model_id": model_id,
                        "system_id": system_id,
                        "population_id": population_id,
                        "score_freeze_precedes_threshold_fit": True,
                        "score_parameters_refit": False,
                    }
                    value["evaluation_ledger_sha256"] = payload_sha256(value)
                    calibration.append(value)
                folds = {
                    row["row_id"]: int(row["fold"])
                    for row in fitted["decisions"]
                }
            else:
                predictions = [row["prediction_step"] for row in system_rows]
                assignments = assign_processbench_folds(system_rows)
                folds = {
                    row["row_id"]: assignments[str(row["group_id"])]
                    for row in system_rows
                }
            for row, prediction in zip(system_rows, predictions):
                row["prediction_step"] = prediction
            system_rows_by_id[system_id] = system_rows
            predictions_by_id[system_id] = list(predictions)

            for cell in cells:
                indices = [
                    index for index, row in enumerate(system_rows)
                    if row["slice_id"] == cell.slice_id
                ]
                cohort = _cohort_id(cell.row_ids)
                for index in indices:
                    row = system_rows[index]
                    prediction = predictions[index]
                    decisions.append({
                        "task_id": "first_error_localization",
                        "dataset_id": "processbench",
                        "population_id": population_id,
                        "cell_id": cell.cell_id,
                        "slice_id": cell.slice_id,
                        "model_id": model_id,
                        "system_id": system_id,
                        "row_id": row["row_id"],
                        "cohort_id": cohort,
                        "group_id": row["group_id"],
                        "fold": folds[row["row_id"]],
                        "prediction_step": "" if prediction is None else int(prediction),
                        "true_first_error": int(row["first_error"]),
                        "status": "OK" if prediction is not None else "UNSCORABLE_COMPARATOR_ROW",
                        "access_level": meta.access_level,
                        "fidelity": meta.fidelity,
                        "comparison_group_id": meta.comparison_group_id,
                        "run_hash": cell.run_hashes[system_id],
                    })
                n_scored = len(indices)
                if system_id in comparator_meta:
                    n_scored = int(cell.comparator_coverage[system_id].sum())
                coverage.append({
                    "task_id": "first_error_localization",
                    "dataset_id": "processbench",
                    "population_id": population_id,
                    "cell_id": cell.cell_id,
                    "slice_id": cell.slice_id,
                    "model_id": model_id,
                    "system_id": system_id,
                    "n_expected": len(indices),
                    "n_scored": n_scored,
                    "n_excluded": 0,
                    "n_failed": len(indices) - n_scored,
                    "n_fallback": 0,
                    "status": "OK" if n_scored == len(indices) else "PARTIAL_COVERAGE",
                    "access_level": meta.access_level,
                    "fidelity": meta.fidelity,
                    "comparison_group_id": meta.comparison_group_id,
                    "cohort_id": cohort,
                    "run_hash": cell.run_hashes[system_id],
                })

        base_rows = system_rows_by_id[next(iter(systems))]
        shared_results = _shared_pb_bootstrap_results(
            rows=base_rows, predictions_by_system=predictions_by_id,
            draws=draws, seed=PROCESSBENCH_BOOTSTRAP_SEED,
        )
        if len({row["draw_stream_sha256"] for row in shared_results.values()}) != 1:
            raise RuntimeError("ProcessBench paired systems used different bootstrap draws")
        for system_id, meta in systems.items():
            system_rows = system_rows_by_id[system_id]
            predictions = predictions_by_id[system_id]
            result = shared_results[system_id]
            executions.append(_bootstrap_ledger_row(
                execution_id=f"processbench::{model_id}::{system_id}",
                dataset_id="processbench", model_id=model_id,
                slice_id="all_four_subsets", system_id=system_id, result=result,
            ))
            panel_point = processbench_panel_metrics(system_rows, predictions)
            for slice_id in (*PROCESSBENCH_SUBSETS, "all_four_subsets"):
                if slice_id == "all_four_subsets":
                    point = panel_point["aggregate"]
                    cell_id = f"processbench_panel_{model_id}"
                    scope_rows = system_rows
                    scope_cells = cells
                else:
                    point = panel_point["per_subset"][slice_id]
                    cell_id = cell_by_slice[slice_id].cell_id
                    scope_rows = [row for row in system_rows if row["slice_id"] == slice_id]
                    scope_cells = [cell_by_slice[slice_id]]
                cohort = _cohort_id([row["row_id"] for row in scope_rows])
                run_hash = _run_hash([
                    system_id, *[cell.run_hashes[system_id] for cell in scope_cells]
                ])
                n_positive = sum(int(row["first_error"] != -1) for row in scope_rows)
                n_negative = len(scope_rows) - n_positive
                for metric_id in PB_METRICS:
                    summary = result["statistics"][f"{slice_id}::{metric_id}"]
                    row = {
                        "task_id": "first_error_localization",
                        "dataset_id": "processbench",
                        "population_id": population_id,
                        "cell_id": cell_id,
                        "slice_id": slice_id,
                        "model_id": model_id,
                        "system_id": system_id,
                        "metric_id": metric_id,
                        "value": _float_or_blank(point[metric_id]),
                        "ci_low": _float_or_blank(summary["ci_low"]),
                        "ci_high": _float_or_blank(summary["ci_high"]),
                        "n_examples": len(scope_rows),
                        "n_positive": n_positive,
                        "n_negative": n_negative,
                        "status": str(summary["status"]),
                        "access_level": meta.access_level,
                        "fidelity": meta.fidelity,
                        "comparison_group_id": meta.comparison_group_id,
                        "bootstrap_unit": "source_question",
                        "bootstrap_draws": draws,
                        "cohort_id": cohort,
                        "run_hash": run_hash,
                    }
                    metrics.append(row)
                    metric_index[(model_id, slice_id, system_id, metric_id)] = row
                    samples_index[(model_id, slice_id, system_id, metric_id)] = (
                        result["samples"][f"{slice_id}::{metric_id}"]
                    )

        # Aggregate coverage is derived from the exact four cell rows.
        for system_id, meta in systems.items():
            pieces = [
                row for row in coverage
                if row["dataset_id"] == "processbench"
                and row["model_id"] == model_id
                and row["system_id"] == system_id
                and row["slice_id"] in PROCESSBENCH_SUBSETS
            ]
            coverage.append({
                "task_id": "first_error_localization",
                "dataset_id": "processbench",
                "population_id": population_id,
                "cell_id": f"processbench_panel_{model_id}",
                "slice_id": "all_four_subsets",
                "model_id": model_id,
                "system_id": system_id,
                "n_expected": sum(int(row["n_expected"]) for row in pieces),
                "n_scored": sum(int(row["n_scored"]) for row in pieces),
                "n_excluded": 0,
                "n_failed": sum(int(row["n_failed"]) for row in pieces),
                "n_fallback": 0,
                "status": "OK" if all(row["status"] == "OK" for row in pieces)
                else "PARTIAL_COVERAGE",
                "access_level": meta.access_level,
                "fidelity": meta.fidelity,
                "comparison_group_id": meta.comparison_group_id,
                "cohort_id": _cohort_id([
                    row_id for cell in cells for row_id in cell.row_ids
                ]),
                "run_hash": _run_hash([
                    system_id, *[cell.run_hashes[system_id] for cell in cells]
                ]),
            })

        for candidate_id in core_meta:
            reference_id = _pb_reference_system(candidate_id)
            for slice_id in (*PROCESSBENCH_SUBSETS, "all_four_subsets"):
                for metric_id in PB_METRICS:
                    candidate = metric_index[(model_id, slice_id, candidate_id, metric_id)]
                    reference = metric_index[(model_id, slice_id, reference_id, metric_id)]
                    delta, low, high, n_valid, status = _paired_contrast(
                        candidate=candidate, reference=reference,
                        candidate_samples=samples_index[(model_id, slice_id, candidate_id, metric_id)],
                        reference_samples=samples_index[(model_id, slice_id, reference_id, metric_id)],
                        draws=draws,
                    )
                    contrasts.append({
                        "task_id": candidate["task_id"],
                        "dataset_id": candidate["dataset_id"],
                        "population_id": candidate["population_id"],
                        "cell_id": candidate["cell_id"],
                        "slice_id": slice_id,
                        "model_id": model_id,
                        "metric_id": metric_id,
                        "candidate_system_id": candidate_id,
                        "reference_system_id": reference_id,
                        "delta": delta,
                        "ci_low": low,
                        "ci_high": high,
                        "n_valid": n_valid,
                        "status": status,
                        "comparison_group_id": core_meta[candidate_id].comparison_group_id,
                        "bootstrap_unit": "source_question",
                        "bootstrap_draws": draws,
                        "cohort_id": candidate["cohort_id"],
                        "candidate_run_hash": candidate["run_hash"],
                        "reference_run_hash": reference["run_hash"],
                        "candidate_access_level": candidate["access_level"],
                        "candidate_fidelity": candidate["fidelity"],
                        "reference_access_level": reference["access_level"],
                        "reference_fidelity": reference["fidelity"],
                    })
        for key in [key for key in samples_index if key[0] == model_id]:
            del samples_index[key]
        for key in [key for key in metric_index if key[0] == model_id]:
            del metric_index[key]

    if len(fold_hashes) != 1:
        raise RuntimeError("ProcessBench five-fold source assignments differ across systems/models")
    return decisions, metrics, contrasts, coverage, calibration, executions


def _prm_reference_system(system_id: str) -> str:
    return _pb_reference_system(system_id)


def _evaluate_prmbench(
    *, config: Mapping[str, Any], panel: PRMPanel, draws: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], bytes,
]:
    core_meta = _core_system_meta("prmbench")
    comparator_meta = _comparator_meta(config, "prmbench")
    systems = {**core_meta, **comparator_meta}
    if (
        len(core_meta) != 27 or len(comparator_meta) != 1 or len(systems) != 28
        or tuple(systems) != panel.system_ids
    ):
        raise RuntimeError("PRMBench system roster must be exact 27 core + 1 context")

    response_index_by_step = np.repeat(
        np.arange(len(panel.response_row_ids), dtype=np.int64),
        np.diff(panel.step_offsets),
    )
    if len(response_index_by_step) != len(panel.step_labels):
        raise RuntimeError("PRMBench step-to-response mapping is malformed")
    step_group_ids = np.asarray(
        [panel.group_ids[index] for index in response_index_by_step], dtype="<U80"
    )
    step_families = np.asarray(
        [panel.error_families[index] for index in response_index_by_step], dtype="<U40"
    )
    slices = (str(config["prmbench"]["overall_slice"]), *PRMBENCH_ERROR_FAMILIES)
    slice_indices: dict[str, np.ndarray] = {
        slices[0]: np.arange(len(panel.step_labels), dtype=np.int64),
        **{
            family: np.flatnonzero(step_families == family)
            for family in PRMBENCH_ERROR_FAMILIES
        },
    }
    response_indices_by_slice = {
        slice_id: np.unique(response_index_by_step[indices])
        for slice_id, indices in slice_indices.items()
    }

    metric_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []

    # Slice-major evaluation executes one grouped draw stream, then applies its
    # integer group weights to all 28 systems. Samples are discarded after the
    # paired contrasts for that slice, bounding memory independently of ten slices.
    for slice_number, slice_id in enumerate(slices):
        indices = slice_indices[slice_id]
        response_indices = response_indices_by_slice[slice_id]
        results = _shared_prm_bootstrap_results(
            labels=panel.step_labels[indices],
            score_matrix=panel.system_scores[:, indices],
            group_ids=[str(step_group_ids[index]) for index in indices],
            strata=[str(step_families[index]) for index in indices],
            system_ids=panel.system_ids,
            draws=draws,
            seed=PRMBENCH_BOOTSTRAP_SEED + slice_number,
        )
        if len({row["draw_stream_sha256"] for row in results.values()}) != 1:
            raise RuntimeError("PRMBench paired systems used different bootstrap draws")
        response_row_ids = [panel.response_row_ids[index] for index in response_indices]
        cohort = _cohort_id(response_row_ids)
        local_metrics: dict[tuple[str, str], dict[str, Any]] = {}
        for system_index, system_id in enumerate(panel.system_ids):
            meta = systems[system_id]
            result = results[system_id]
            executions.append(_bootstrap_ledger_row(
                execution_id=f"prmbench::{slice_id}::{system_id}",
                dataset_id="prmbench", model_id=panel.model_id,
                slice_id=slice_id, system_id=system_id, result=result,
            ))
            point = prmbench_step_metrics(
                panel.step_labels[indices], panel.system_scores[system_index, indices]
            )
            run_hash = _run_hash([
                panel.run_hashes[system_id], system_id, slice_id,
                payload_sha256(response_row_ids),
            ])
            for metric_id in PRM_METRICS:
                summary = result["statistics"][metric_id]
                status = point["status"] if metric_id in ("auroc", "auprc") else "OK"
                row = {
                    "task_id": "step_error_localization",
                    "dataset_id": "prmbench",
                    "population_id": panel.population_id,
                    "cell_id": panel.cell_id,
                    "slice_id": slice_id,
                    "model_id": panel.model_id,
                    "system_id": system_id,
                    "metric_id": metric_id,
                    "value": _float_or_blank(point[metric_id]),
                    "ci_low": _float_or_blank(summary["ci_low"]),
                    "ci_high": _float_or_blank(summary["ci_high"]),
                    "n_examples": int(point["n_examples"]),
                    "n_positive": int(point["n_positive"]),
                    "n_negative": int(point["n_negative"]),
                    "status": status,
                    "access_level": meta.access_level,
                    "fidelity": meta.fidelity,
                    "comparison_group_id": meta.comparison_group_id,
                    "bootstrap_unit": "source_idx",
                    "bootstrap_draws": draws,
                    "cohort_id": cohort,
                    "run_hash": run_hash,
                }
                metric_rows.append(row)
                local_metrics[(system_id, metric_id)] = row
            n_scored = int(np.isfinite(panel.system_scores[system_index, indices]).sum())
            coverage_rows.append({
                "task_id": "step_error_localization",
                "dataset_id": "prmbench",
                "population_id": panel.population_id,
                "cell_id": panel.cell_id,
                "slice_id": slice_id,
                "model_id": panel.model_id,
                "system_id": system_id,
                "n_expected": len(indices),
                "n_scored": n_scored,
                "n_excluded": 0,
                "n_failed": len(indices) - n_scored,
                "n_fallback": 0,
                "status": "OK" if n_scored == len(indices) else "PARTIAL_COVERAGE",
                "access_level": meta.access_level,
                "fidelity": meta.fidelity,
                "comparison_group_id": meta.comparison_group_id,
                "cohort_id": cohort,
                "run_hash": run_hash,
            })

        for candidate_id in core_meta:
            reference_id = _prm_reference_system(candidate_id)
            for metric_id in PRM_METRICS:
                candidate = local_metrics[(candidate_id, metric_id)]
                reference = local_metrics[(reference_id, metric_id)]
                delta, low, high, n_valid, status = _paired_contrast(
                    candidate=candidate, reference=reference,
                    candidate_samples=results[candidate_id]["samples"][metric_id],
                    reference_samples=results[reference_id]["samples"][metric_id],
                    draws=draws,
                )
                contrast_rows.append({
                    "task_id": candidate["task_id"],
                    "dataset_id": candidate["dataset_id"],
                    "population_id": candidate["population_id"],
                    "cell_id": candidate["cell_id"],
                    "slice_id": slice_id,
                    "model_id": panel.model_id,
                    "metric_id": metric_id,
                    "candidate_system_id": candidate_id,
                    "reference_system_id": reference_id,
                    "delta": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "n_valid": n_valid,
                    "status": status,
                    "comparison_group_id": core_meta[candidate_id].comparison_group_id,
                    "bootstrap_unit": "source_idx",
                    "bootstrap_draws": draws,
                    "cohort_id": candidate["cohort_id"],
                    "candidate_run_hash": candidate["run_hash"],
                    "reference_run_hash": reference["run_hash"],
                    "candidate_access_level": candidate["access_level"],
                    "candidate_fidelity": candidate["fidelity"],
                    "reference_access_level": reference["access_level"],
                    "reference_fidelity": reference["fidelity"],
                })

    prm_npz = deterministic_npz_bytes({
        "response_row_ids": np.asarray(panel.response_row_ids, dtype="<U80"),
        "group_ids": np.asarray(panel.group_ids, dtype="<U80"),
        "error_families": np.asarray(panel.error_families, dtype="<U40"),
        "step_offsets": panel.step_offsets.astype("<i8", copy=False),
        "step_labels": panel.step_labels.astype("<i1", copy=False),
        "system_ids": np.asarray(panel.system_ids, dtype="<U100"),
        "system_scores": panel.system_scores.astype("<f8", copy=False),
    })
    return metric_rows, contrast_rows, coverage_rows, executions, prm_npz


def _validate_evaluation_tables(
    *, config: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]], calibration: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]], draws: int,
) -> dict[str, Any]:
    """Prove exact scientific completeness from atomic rows, never metadata alone."""

    core_ids = tuple(row["system_id"] for row in primary_system_roster(PRIMARY_METHOD_IDS))
    pb_context = tuple(_comparator_meta(config, "processbench"))
    prm_context = tuple(_comparator_meta(config, "prmbench"))
    pb_systems = (*core_ids, *pb_context)
    prm_systems = (*core_ids, *prm_context)
    pb_cells = tuple(map(str, config["processbench"]["source_cells"]))
    cell_parts = {
        cell_id: (
            str(cell_id).removeprefix("processbench_").rsplit("_", 2)[0],
            next(
                model for model in map(str, config["processbench"]["models"])
                if cell_id.endswith("_" + model)
            ),
        )
        for cell_id in pb_cells
    }
    expected_rows = {
        subset: int(value)
        for subset, value in config["processbench"]["expected_rows_by_subset"].items()
    }
    pb_scopes = {
        (model_id, subset, cell_id)
        for cell_id, (subset, model_id) in cell_parts.items()
    } | {
        (model_id, "all_four_subsets", f"processbench_panel_{model_id}")
        for model_id in map(str, config["processbench"]["models"])
    }
    prm_slices = (
        str(config["prmbench"]["overall_slice"]), *PRMBENCH_ERROR_FAMILIES
    )

    decision_counts: dict[tuple[str, str], int] = {}
    group_folds: dict[str, int] = {}
    decision_keys = set()
    for row in decisions:
        if set(row) != set(LOCALIZATION_DECISION_FIELDS):
            raise RuntimeError("localization decision schema is not exact")
        if row["dataset_id"] != "processbench" or row["system_id"] not in pb_systems:
            raise RuntimeError("localization decision roster includes a non-ProcessBench row")
        if row["cell_id"] not in cell_parts:
            raise RuntimeError("localization decision references an unknown ProcessBench cell")
        if isinstance(row["prediction_step"], (bool, np.bool_)):
            raise RuntimeError("localization integer decision was coerced to boolean")
        if row["prediction_step"] != "" and int(row["prediction_step"]) != row["prediction_step"]:
            raise RuntimeError("localization decision is not integer typed")
        key = (str(row["cell_id"]), str(row["system_id"]), str(row["row_id"]))
        if key in decision_keys:
            raise RuntimeError("duplicate atomic localization decision")
        decision_keys.add(key)
        pair = (str(row["cell_id"]), str(row["system_id"]))
        decision_counts[pair] = decision_counts.get(pair, 0) + 1
        group_id = str(row["group_id"])
        fold = int(row["fold"])
        previous = group_folds.setdefault(group_id, fold)
        if previous != fold or fold not in range(5):
            raise RuntimeError("ProcessBench source fold differs across scorer/system rows")
    expected_decision_pairs = {(cell, system) for cell in pb_cells for system in pb_systems}
    if set(decision_counts) != expected_decision_pairs:
        raise RuntimeError("ProcessBench decision table is not exact 12 cells x 30 systems")
    for (cell_id, _system_id), count in decision_counts.items():
        subset = cell_parts[cell_id][0]
        if count != expected_rows[subset]:
            raise RuntimeError("ProcessBench atomic decision count drifted")

    metric_keys = set()
    for row in metrics:
        if set(row) != set(METRIC_FIELDS):
            raise RuntimeError("localization metric schema is not exact")
        if int(row["bootstrap_draws"]) != draws:
            raise RuntimeError("metric row declares a bootstrap count that was not registered")
        key = (
            str(row["dataset_id"]), str(row["model_id"]), str(row["slice_id"]),
            str(row["cell_id"]), str(row["system_id"]), str(row["metric_id"]),
        )
        if key in metric_keys:
            raise RuntimeError("duplicate localization metric row")
        metric_keys.add(key)
    expected_metric_keys = {
        ("processbench", model, slice_id, cell_id, system, metric)
        for model, slice_id, cell_id in pb_scopes
        for system in pb_systems for metric in PB_METRICS
    } | {
        ("prmbench", str(config["prmbench"]["model_id"]), slice_id,
         str(config["prmbench"]["source_cell"]), system, metric)
        for slice_id in prm_slices for system in prm_systems for metric in PRM_METRICS
    }
    if metric_keys != expected_metric_keys:
        raise RuntimeError("localization metric table is not the exact PB/PRM Cartesian roster")
    for row in metrics:
        if (
            row["dataset_id"] == "prmbench"
            and row["slice_id"] == "multi_solutions"
            and row["metric_id"] in ("auroc", "auprc")
        ):
            if row["status"] != UNDEFINED_SINGLE_CLASS or row["value"] != "":
                raise RuntimeError("PRMBench multi_solutions undefined metric was hidden")

    coverage_keys = set()
    for row in coverage:
        if set(row) != set(COVERAGE_FIELDS):
            raise RuntimeError("localization coverage schema is not exact")
        key = (
            str(row["dataset_id"]), str(row["model_id"]), str(row["slice_id"]),
            str(row["cell_id"]), str(row["system_id"]),
        )
        if key in coverage_keys:
            raise RuntimeError("duplicate localization coverage row")
        coverage_keys.add(key)
        if int(row["n_scored"]) + int(row["n_failed"]) != int(row["n_expected"]):
            raise RuntimeError("localization coverage arithmetic is not exact")
    expected_coverage_keys = {
        key[:-1] for key in expected_metric_keys
    }
    if coverage_keys != expected_coverage_keys:
        raise RuntimeError("localization coverage table is not the exact PB/PRM roster")

    contrast_keys = set()
    for row in contrasts:
        if set(row) != set(CONTRAST_FIELDS) or int(row["bootstrap_draws"]) != draws:
            raise RuntimeError("localization contrast schema/draw count is invalid")
        key = (
            str(row["dataset_id"]), str(row["model_id"]), str(row["slice_id"]),
            str(row["cell_id"]), str(row["candidate_system_id"]), str(row["metric_id"]),
        )
        if key in contrast_keys:
            raise RuntimeError("duplicate localization paired contrast")
        contrast_keys.add(key)
        if row["candidate_system_id"] not in core_ids:
            raise RuntimeError("context comparator entered a core paired contrast")
    expected_contrast_keys = {
        ("processbench", model, slice_id, cell_id, system, metric)
        for model, slice_id, cell_id in pb_scopes
        for system in core_ids for metric in PB_METRICS
    } | {
        ("prmbench", str(config["prmbench"]["model_id"]), slice_id,
         str(config["prmbench"]["source_cell"]), system, metric)
        for slice_id in prm_slices for system in core_ids for metric in PRM_METRICS
    }
    if contrast_keys != expected_contrast_keys:
        raise RuntimeError("localization contrasts omit a PB cell or PRMBench family")

    calibration_keys = set()
    for row in calibration:
        key = (str(row["model_id"]), str(row["system_id"]), int(row["held_out_fold"]))
        if key in calibration_keys or row["system_id"] not in core_ids:
            raise RuntimeError("ProcessBench threshold calibration ledger is duplicated/invalid")
        if (
            int(row["held_out_fold"]) not in range(5)
            or row.get("score_freeze_precedes_threshold_fit") is not True
            or row.get("score_parameters_refit") is not False
        ):
            raise RuntimeError("ProcessBench threshold was not strict post-freeze five-fold fit")
        calibration_keys.add(key)
    expected_calibration = {
        (model, system, fold)
        for model in map(str, config["processbench"]["models"])
        for system in core_ids for fold in range(5)
    }
    if calibration_keys != expected_calibration:
        raise RuntimeError("ProcessBench threshold ledger is not exact 3 x 27 x 5")

    execution_keys = set()
    stream_by_scope: dict[tuple[str, str, str], str] = {}
    for row in executions:
        if (
            int(row.get("draws", -1)) != draws
            or int(row.get("draws_executed", -1)) != draws
        ):
            raise RuntimeError("bootstrap ledger metadata does not prove actual draws")
        key = str(row["execution_id"])
        if key in execution_keys:
            raise RuntimeError("duplicate localization bootstrap execution")
        execution_keys.add(key)
        scope = (str(row["dataset_id"]), str(row["model_id"]), str(row["slice_id"]))
        previous = stream_by_scope.setdefault(scope, str(row["draw_stream_sha256"]))
        if previous != row["draw_stream_sha256"]:
            raise RuntimeError("paired localization systems did not share bootstrap draws")
        expected_unit = "source_question" if row["dataset_id"] == "processbench" else "source_idx"
        if row["bootstrap_unit"] != expected_unit:
            raise RuntimeError("localization bootstrap used the wrong grouping unit")
        expected_metric_roster = (
            {f"{slice_id}::{metric}" for slice_id in (*PROCESSBENCH_SUBSETS, "all_four_subsets")
             for metric in PB_METRICS}
            if row["dataset_id"] == "processbench" else set(PRM_METRICS)
        )
        if set(row.get("metric_n_valid", {})) != expected_metric_roster:
            raise RuntimeError("localization bootstrap statistic roster is incomplete")
    expected_executions = {
        f"processbench::{model}::{system}"
        for model in map(str, config["processbench"]["models"])
        for system in pb_systems
    } | {
        f"prmbench::{slice_id}::{system}"
        for slice_id in prm_slices for system in prm_systems
    }
    if execution_keys != expected_executions:
        raise RuntimeError("localization bootstrap execution roster is incomplete")

    summary = {
        "n_processbench_cells": len(pb_cells),
        "n_processbench_core_systems": len(core_ids),
        "n_processbench_context_systems": len(pb_context),
        "n_prmbench_families": len(PRMBENCH_ERROR_FAMILIES),
        "n_prmbench_core_systems": len(core_ids),
        "n_prmbench_context_systems": len(prm_context),
        "n_decisions": len(decisions),
        "n_metrics": len(metrics),
        "n_contrasts": len(contrasts),
        "n_coverage": len(coverage),
        "n_calibration_ledgers": len(calibration),
        "n_bootstrap_executions": len(executions),
        "bootstrap_draws_required_and_executed": draws,
        "processbench_metric_roster": list(PB_METRICS),
        "prmbench_metric_roster": list(PRM_METRICS),
        "prmbench_slice_roster": list(prm_slices),
    }
    summary["completeness_sha256"] = payload_sha256(summary)
    return summary


def _source_provenance_bytes(
    *, cells: Mapping[str, PBCell], panel: PRMPanel,
    localization_registry_path: Path, external_registry_path: Path,
    population_registry_path: Path,
    amendment: Mapping[str, Any],
) -> bytes:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in [
        *(record for cell in cells.values() for record in cell.source_records),
        *panel.source_records,
    ]:
        row = dict(item)
        key = (str(row.get("path")), str(row.get("sha256")), str(row.get("role")))
        previous = records.setdefault(key, row)
        if previous != row:
            raise RuntimeError("post-freeze source provenance differs for one artifact")
    value = {
        "schema_version": "reconstruction-localization-label-source-provenance-v2",
        "stage": "post_score_ab_freeze_only",
        "source_root_contract": "registered relative paths beneath verified source overlay",
        "localization_registry_sha256": sha256_file(localization_registry_path),
        "external_registry_sha256": sha256_file(external_registry_path),
        "population_registry_sha256": sha256_file(population_registry_path),
        "postfreeze_amendment": {
            "schema_version": amendment["schema_version"],
            "amendment_id": amendment["amendment_id"],
            "file_sha256": amendment["file_sha256"],
            "payload_sha256": amendment["payload_sha256"],
            "score_verifier_required_git_head": amendment["score_verifier_repo"][
                "required_git_head"
            ],
            "semantics": amendment["semantics"],
            "observed_oob_audit": panel.postfreeze_amendment_audit,
            "disclosure": amendment["disclosure"],
        },
        "records": [records[key] for key in sorted(records)],
    }
    value["payload_sha256"] = payload_sha256(value)
    return _json_bytes(value)


def derive_localization_evaluation(
    *, release_id: str, build_id: str, release_root: str | Path,
    identity_key_path: str | Path | None = None,
    localization_ab_certificate_path: str | Path | None = None,
    localization_registry_path: str | Path = DEFAULT_LOCALIZATION_REGISTRY,
    external_registry_path: str | Path = DEFAULT_EXTERNAL_REGISTRY,
    population_registry_path: str | Path = DEFAULT_POPULATION_REGISTRY,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    score_verifier_repo: str | Path,
    evaluation_repo: str | Path = REPO_ROOT,
    localization_postfreeze_amendment_path: str | Path = (
        DEFAULT_LOCALIZATION_POSTFREEZE_AMENDMENT
    ),
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> DerivedLocalizationEvaluation:
    """Rederive every post-label output from one exact frozen score build."""

    if build_id not in {"A", "B"}:
        raise ValueError("localization evaluation build must be A or B")
    if int(bootstrap_draws) != bootstrap_draws or int(bootstrap_draws) < 1:
        raise ValueError("localization evaluation bootstrap draws must be positive integer")
    draws = int(bootstrap_draws)
    score_verifier_repo_path = Path(score_verifier_repo).resolve()
    evaluation_repo_path = Path(evaluation_repo).resolve()
    release_root_path = Path(release_root).resolve()
    source_root_path = Path(source_root).resolve()
    localization_registry_file = Path(localization_registry_path).resolve()
    external_registry_file = Path(external_registry_path).resolve()
    population_registry_file = Path(population_registry_path).resolve()
    amendment_path = Path(localization_postfreeze_amendment_path).resolve()
    certificate_path = (
        Path(localization_ab_certificate_path).resolve()
        if localization_ab_certificate_path is not None
        else release_root_path / release_id / "localization/AB_VERIFICATION.json"
    )

    # Repository state is target-free.  Require the dedicated verifier checkout
    # to be exactly the score-frozen commit before revalidating the certificate.
    score_verifier_snapshot = _score_verifier_repo_snapshot(
        score_verifier_repo_path,
        required_git_head=EXPECTED_SCORE_VERIFIER_GIT_HEAD,
    )
    score_registry_files = {
        "localization": score_verifier_repo_path
        / "configs/reconstruction_benchmark_v1/localization.json",
        "external": score_verifier_repo_path
        / "configs/reconstruction_benchmark_v1/external_final_answer.json",
        "populations": score_verifier_repo_path
        / "configs/reconstruction_benchmark_v1/populations.json",
    }
    evaluation_registry_files = {
        "localization": localization_registry_file,
        "external": external_registry_file,
        "populations": population_registry_file,
    }
    if any(
        sha256_file(score_registry_files[key])
        != sha256_file(evaluation_registry_files[key])
        for key in score_registry_files
    ):
        raise RuntimeError("score-verifier and evaluation registry bytes differ")
    score_source_root = score_verifier_repo_path / (
        "results/reconstruction_benchmark_v1/source_overlays/external_final_answer_v1"
    )

    # This full recomputation is intentionally the last gate before any
    # label-bearing amendment or raw target container is opened.
    certificate = assert_localization_ab_certificate(
        certificate_path,
        release_id=release_id,
        release_root=release_root_path,
        localization_registry_path=score_registry_files["localization"],
        external_registry_path=score_registry_files["external"],
        population_registry_path=score_registry_files["populations"],
        source_root=score_source_root,
        repo=score_verifier_repo_path,
    )
    amendment = load_localization_postfreeze_amendment(
        amendment_path,
        release_id=release_id,
        localization_registry_path=localization_registry_file,
        score_ab_certificate_path=certificate_path,
        score_ab_certificate=certificate,
        source_root=source_root_path,
    )
    if amendment["score_verifier_repo"]["required_git_head"] != (
        score_verifier_snapshot["git_head"]
    ):
        raise RuntimeError("score verifier snapshot differs from post-freeze amendment")
    external_release_id = str(certificate["external_release_id"])
    key_path = (
        Path(identity_key_path).resolve()
        if identity_key_path is not None
        else release_root_path.parent / "private_control" / external_release_id
        / "external_final_answer/external-id-v2.key"
    )
    identity_key = load_identity_key(key_path)
    identity_binding = certificate.get("identity_contract", {})
    if identity_key_id(identity_key) != identity_binding.get("key_id"):
        raise RuntimeError("post-freeze identity key differs from signed localization inputs")

    score_bound_config = load_localization_registry(localization_registry_file)
    config = apply_localization_postfreeze_amendment(score_bound_config, amendment)
    registry = load_external_registry(
        repo=evaluation_repo_path,
        registry_path=external_registry_file,
        population_registry_path=population_registry_file,
    )
    core, projections, freeze = _load_build_scores(
        release_root=release_root_path, release_id=release_id, build_id=build_id,
        certificate=certificate,
    )
    expected_cells = {
        *map(str, config["processbench"]["source_cells"]),
        str(config["prmbench"]["source_cell"]),
    }
    if set(core) != expected_cells:
        raise RuntimeError("evaluation score build is not the exact 13-cell roster")
    cells = _load_processbench_cells(
        config=config, registry=registry, source_root=source_root_path,
        identity_key=identity_key, core=core, projections=projections,
    )
    panel = _load_prmbench_panel(
        config=config, amendment=amendment, registry=registry, source_root=source_root_path,
        identity_key=identity_key, core=core, projections=projections,
    )
    pb_decisions, pb_metrics, pb_contrasts, pb_coverage, calibration, pb_executions = (
        _evaluate_processbench(config=config, cells_by_id=cells, draws=draws)
    )
    prm_metrics, prm_contrasts, prm_coverage, prm_executions, prm_npz = (
        _evaluate_prmbench(config=config, panel=panel, draws=draws)
    )
    metrics = [*pb_metrics, *prm_metrics]
    contrasts = [*pb_contrasts, *prm_contrasts]
    coverage = [*pb_coverage, *prm_coverage]
    executions = [*pb_executions, *prm_executions]
    completeness = _validate_evaluation_tables(
        config=config, decisions=pb_decisions, metrics=metrics,
        contrasts=contrasts, coverage=coverage, calibration=calibration,
        executions=executions, draws=draws,
    )

    calibration_value = {
        "schema_version": "reconstruction-localization-pb-calibration-ledgers-v1",
        "score_freeze_precedes_threshold_fit": True,
        "folds": 5,
        "fit_scope": "one_scorer_model_across_four_subsets",
        "ledgers": sorted(
            calibration,
            key=lambda row: (row["model_id"], row["system_id"], row["held_out_fold"]),
        ),
    }
    calibration_value["payload_sha256"] = payload_sha256(calibration_value)
    bootstrap_value = {
        "schema_version": "reconstruction-localization-bootstrap-ledger-v2",
        "draws_required": draws,
        "draws_executed_per_execution": draws,
        "paired": True,
        "processbench_unit": "source_question",
        "prmbench_unit": "source_idx",
        "executions": sorted(executions, key=lambda row: row["execution_id"]),
    }
    bootstrap_value["payload_sha256"] = payload_sha256(bootstrap_value)
    files = {
        "bootstrap_ledger.json": _json_bytes(bootstrap_value),
        "calibration_ledgers.json": _json_bytes(calibration_value),
        "contrasts_long.csv": _csv_bytes(contrasts, CONTRAST_FIELDS),
        "coverage_long.csv": _csv_bytes(coverage, COVERAGE_FIELDS),
        "localization_decisions.csv": _csv_bytes(
            pb_decisions, LOCALIZATION_DECISION_FIELDS
        ),
        "metrics_long.csv": _csv_bytes(metrics, METRIC_FIELDS),
        "prmbench_steps.npz": prm_npz,
        "source_provenance.json": _source_provenance_bytes(
            cells=cells, panel=panel,
            localization_registry_path=localization_registry_file,
            external_registry_path=external_registry_file,
            population_registry_path=population_registry_file,
            amendment=amendment,
        ),
    }
    if tuple(sorted(files)) != tuple(sorted(EXPECTED_ARTIFACTS)):
        raise AssertionError("localization evaluation artifact roster drifted")
    artifact_rows = [
        {"path": path, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        for path, payload in sorted(files.items())
    ]
    manifest_core = {
        "score_ab_certificate_sha256": certificate["certificate_sha256"],
        "score_ab_certificate_file_sha256": sha256_file(certificate_path),
        "score_freeze_payload_sha256": freeze["payload_sha256"],
        "postfreeze_amendment": {
            "schema_version": amendment["schema_version"],
            "amendment_id": amendment["amendment_id"],
            "file_sha256": amendment["file_sha256"],
            "payload_sha256": amendment["payload_sha256"],
            "semantics": amendment["semantics"],
            "observed_oob_audit": panel.postfreeze_amendment_audit,
            "effective_prmbench_counts": amendment["effective_prmbench_counts"],
        },
        "identity_key_id": identity_key_id(identity_key),
        "target_data_opened_only_after_score_ab_pass": True,
        "response_scores_refit": False,
        "historical_075_025_blend_used": False,
        "evaluator_contract": evaluator_contract(),
        "bootstrap_draws": draws,
        "completeness": completeness,
        "artifacts": artifact_rows,
        "artifacts_sha256": payload_sha256(artifact_rows),
        "score_verifier_repo_snapshot": score_verifier_snapshot,
        "evaluation_source_snapshot": _source_snapshot(evaluation_repo_path),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    return DerivedLocalizationEvaluation(files=files, manifest_core=manifest_core)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing a raced target."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        operation = getattr(libc, "renamex_np", None)
        if operation is None:
            raise RuntimeError(
                "atomic no-replace localization publication is unavailable"
            )
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise RuntimeError(
                "atomic no-replace localization publication is unavailable"
            )
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(-100, source_bytes, -100, target_bytes, 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError(
            f"atomic no-replace localization publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            f"localization evaluation output already exists: {target}"
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(target))


class _AtomicEvaluationStage:
    """Build an unpublished sibling tree and expose it by one rename."""

    def __init__(self, final_path: Path) -> None:
        requested = Path(os.path.abspath(os.fspath(final_path)))
        try:
            parent = requested.parent.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError("localization evaluation parent is absent") from error
        if not parent.is_dir():
            raise RuntimeError("localization evaluation parent must be a directory")
        self.final_path = parent / requested.name
        if os.path.lexists(self.final_path):
            raise FileExistsError(
                f"localization evaluation output already exists: {self.final_path}"
            )
        self.path = Path(tempfile.mkdtemp(
            prefix=f".{self.final_path.name}.staging-", dir=parent,
        ))
        self.committed = False

    def commit(self) -> None:
        if self.committed:
            raise RuntimeError("localization evaluation stage was already committed")
        if os.path.lexists(self.final_path):
            raise FileExistsError(
                f"localization evaluation output already exists: {self.final_path}"
            )
        _fsync_directory(self.path)
        _rename_directory_noreplace(self.path, self.final_path)
        _fsync_directory(self.final_path.parent)
        self.committed = True

    def cleanup(self) -> None:
        if not self.committed and self.path.exists():
            shutil.rmtree(self.path)


def _read_regular_file_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileExistsError("localization evaluation certificate is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_immutable_certificate(path: Path, payload: bytes) -> None:
    """Atomically publish a no-clobber certificate; allow identical reruns."""

    requested = Path(os.path.abspath(os.fspath(path)))
    try:
        parent = requested.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError("localization evaluation certificate parent is absent") from error
    if not parent.is_dir():
        raise RuntimeError("localization evaluation certificate parent must be a directory")
    target = parent / requested.name
    try:
        existing = _read_regular_file_nofollow(target)
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise FileExistsError("localization evaluation certificate target is unsafe") from error
    if existing is not None:
        if existing != payload:
            raise FileExistsError("localization evaluation certificate already differs")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            try:
                raced = _read_regular_file_nofollow(target)
            except OSError as error:
                raise FileExistsError(
                    "localization evaluation certificate was claimed unsafely"
                ) from error
            if raced != payload:
                raise FileExistsError("localization evaluation certificate already differs")
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_localization_evaluation_build(
    *, release_id: str, build_id: str, release_root: str | Path,
    output_root: str | Path | None = None, scientific_full: bool = True,
    **derive_kwargs: Any,
) -> dict[str, Any]:
    draws = int(derive_kwargs.get("bootstrap_draws", DEFAULT_BOOTSTRAP_DRAWS))
    if scientific_full and draws != DEFAULT_BOOTSTRAP_DRAWS:
        raise RuntimeError("scientific localization evaluation requires exactly 20,000 draws")
    evaluation_repo_path = Path(
        derive_kwargs.get("evaluation_repo", REPO_ROOT)
    ).resolve()
    if scientific_full and _repo_state(evaluation_repo_path)["git_clean"] is not True:
        raise RuntimeError("scientific localization evaluation requires a clean worktree")
    derived = derive_localization_evaluation(
        release_id=release_id, build_id=build_id, release_root=release_root,
        **derive_kwargs,
    )
    requested_root = (
        Path(output_root) if output_root is not None
        else Path(release_root).resolve() / release_id / f"build_{build_id}"
        / "localization/evaluation"
    )
    manifest = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "build_id": build_id,
        "status": "PASS",
        "scientific_full": bool(scientific_full),
        **derived.manifest_core,
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    stage = _AtomicEvaluationStage(requested_root)
    try:
        for path, payload in derived.files.items():
            atomic_write_bytes(stage.path / path, payload)
        atomic_write_json(stage.path / "MANIFEST.json", manifest)
        stage.commit()
    finally:
        stage.cleanup()
    return manifest


def _validate_evaluation_build_against_derivation(
    *, root: Path, release_id: str, build_id: str,
    derived: DerivedLocalizationEvaluation,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("evaluation build root must be a real directory")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("evaluation build contains a symlink")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    expected_files = {*EXPECTED_ARTIFACTS, "MANIFEST.json"}
    if actual_files != expected_files:
        raise RuntimeError("evaluation build contains missing or unregistered artifacts")
    manifest = _hashed_json(root / "MANIFEST.json")
    expected_manifest_fields = {
        "schema_version", "release_id", "build_id", "status", "scientific_full",
        *derived.manifest_core.keys(), "payload_sha256",
    }
    if (
        set(manifest) != expected_manifest_fields
        or
        manifest.get("schema_version") != EVALUATION_MANIFEST_SCHEMA_VERSION
        or manifest.get("release_id") != release_id
        or manifest.get("build_id") != build_id
        or manifest.get("status") != "PASS"
        or manifest.get("scientific_full") is not True
        or int(manifest.get("bootstrap_draws", -1)) != DEFAULT_BOOTSTRAP_DRAWS
    ):
        raise RuntimeError("evaluation build is not a complete scientific build")
    for field, expected in derived.manifest_core.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"evaluation manifest differs from rederivation: {field}")
    for path, payload in derived.files.items():
        actual = (root / path).read_bytes()
        if actual != payload:
            raise RuntimeError(f"evaluation artifact differs from rederivation: {path}")
    tree = canonical_tree_manifest(root)
    return {
        "manifest": manifest,
        "manifest_file_sha256": sha256_file(root / "MANIFEST.json"),
        "tree_sha256": tree["tree_sha256"],
        "artifact_sha256": {
            path: sha256_file(root / path) for path in EXPECTED_ARTIFACTS
        },
    }


def verify_localization_evaluation_ab(
    *, release_id: str, release_root: str | Path,
    identity_key_path: str | Path | None = None,
    output_path: str | Path | None = None,
    localization_ab_certificate_path: str | Path | None = None,
    localization_registry_path: str | Path = DEFAULT_LOCALIZATION_REGISTRY,
    external_registry_path: str | Path = DEFAULT_EXTERNAL_REGISTRY,
    population_registry_path: str | Path = DEFAULT_POPULATION_REGISTRY,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    score_verifier_repo: str | Path,
    evaluation_repo: str | Path = REPO_ROOT,
    localization_postfreeze_amendment_path: str | Path = (
        DEFAULT_LOCALIZATION_POSTFREEZE_AMENDMENT
    ),
) -> dict[str, Any]:
    """Rederive both builds, prove completeness, then require byte identity."""

    release_root_path = Path(release_root).resolve()
    if _repo_state(Path(evaluation_repo).resolve())["git_clean"] is not True:
        raise RuntimeError("scientific localization evaluation verification requires clean git")
    derived = {}
    validated = {}
    for build_id in ("A", "B"):
        derived[build_id] = derive_localization_evaluation(
            release_id=release_id, build_id=build_id,
            release_root=release_root_path, identity_key_path=identity_key_path,
            localization_ab_certificate_path=localization_ab_certificate_path,
            localization_registry_path=localization_registry_path,
            external_registry_path=external_registry_path,
            population_registry_path=population_registry_path,
            source_root=source_root,
            score_verifier_repo=score_verifier_repo,
            evaluation_repo=evaluation_repo,
            localization_postfreeze_amendment_path=(
                localization_postfreeze_amendment_path
            ),
            bootstrap_draws=DEFAULT_BOOTSTRAP_DRAWS,
        )
        root = release_root_path / release_id / f"build_{build_id}"
        root = root / "localization/evaluation"
        validated[build_id] = _validate_evaluation_build_against_derivation(
            root=root, release_id=release_id, build_id=build_id,
            derived=derived[build_id],
        )
    if derived["A"].files != derived["B"].files:
        raise RuntimeError("independent localization evaluation derivations differ")
    if derived["A"].manifest_core != derived["B"].manifest_core:
        raise RuntimeError("independent localization evaluation manifests differ")
    if validated["A"]["artifact_sha256"] != validated["B"]["artifact_sha256"]:
        raise RuntimeError("localization evaluation A/B artifacts are not byte-identical")
    completeness = derived["A"].manifest_core["completeness"]
    if completeness != derived["B"].manifest_core["completeness"]:
        raise RuntimeError("localization evaluation A/B completeness differs")
    certificate = {
        "schema_version": EVALUATION_AB_SCHEMA_VERSION,
        "release_id": release_id,
        "status": "PASS",
        "scientific_full": True,
        "bootstrap_draws_executed_per_execution": DEFAULT_BOOTSTRAP_DRAWS,
        "score_ab_certificate_sha256": derived["A"].manifest_core[
            "score_ab_certificate_sha256"
        ],
        "score_ab_certificate_file_sha256": derived["A"].manifest_core[
            "score_ab_certificate_file_sha256"
        ],
        "postfreeze_amendment": derived["A"].manifest_core[
            "postfreeze_amendment"
        ],
        "score_verifier_repo_snapshot": derived["A"].manifest_core[
            "score_verifier_repo_snapshot"
        ],
        "evaluation_source_snapshot": derived["A"].manifest_core[
            "evaluation_source_snapshot"
        ],
        "completeness": completeness,
        "artifact_sha256": validated["A"]["artifact_sha256"],
        "builds": {
            build_id: {
                "manifest_file_sha256": validated[build_id]["manifest_file_sha256"],
                "tree_sha256": validated[build_id]["tree_sha256"],
            }
            for build_id in ("A", "B")
        },
    }
    certificate["certificate_sha256"] = payload_sha256(certificate)
    target = (
        Path(output_path) if output_path is not None
        else release_root_path / release_id / "localization/EVALUATION_AB_VERIFICATION.json"
    )
    _write_immutable_certificate(target, _json_bytes(certificate))
    return certificate


__all__ = [
    "CONTRAST_FIELDS", "COVERAGE_FIELDS", "DerivedLocalizationEvaluation",
    "EVALUATION_AB_SCHEMA_VERSION", "EVALUATION_MANIFEST_SCHEMA_VERSION",
    "EXPECTED_ARTIFACTS", "derive_localization_evaluation",
    "verify_localization_evaluation_ab", "write_localization_evaluation_build",
]
