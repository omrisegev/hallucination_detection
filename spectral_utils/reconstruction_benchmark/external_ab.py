"""Integrity and independent A/B gates for external final-answer evaluation.

This module contains no label loader.  It verifies the complete target-free
input and score trees and issues a content-addressed A/B certificate.  The
evaluation process may open labels only after re-validating that certificate
against the still-current build trees and reconstruction contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .external_final_answer import (
    CANONICAL_FEATURE_NAMES,
    PREPARED_SCHEMA_VERSION,
    SCORE_FREEZE_SCHEMA_VERSION,
    ExternalRegistry,
    load_prepared_external_cell,
)
from .io import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_tree_manifest,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .methods import PRIMARY_METHOD_IDS, PRIMARY_METHOD_SPECS
from ..dufs_liu_feature_contract import CONTRACT_VERSION


AB_CERTIFICATE_SCHEMA_VERSION = "reconstruction-external-ab-certificate-v1"
INPUT_MANIFEST_SCHEMA_VERSION = "reconstruction-external-target-free-build-v1"
SUCCESS = {"OK", "OK_FALLBACK"}

FEATURE_CONFIG_PATH = "configs/reconstruction_benchmark_v1/feature_contract.json"
METHOD_CONFIG_PATH = "configs/reconstruction_benchmark_v1/methods.json"
TRANSFORM_SOURCE_PATH = "spectral_utils/dufs_liu_feature_contract.py"
ORIENTATION_SOURCE_PATH = "spectral_utils/feature_contract.py"
FEATURE_ROSTER_SOURCE_PATH = "spectral_utils/specrage_views.py"

REQUIRED_PREPARATION_SOURCES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    FEATURE_CONFIG_PATH,
    "configs/reconstruction_benchmark_v1/populations.json",
    "scripts/reconstruction_benchmark/prepare_external_final_answer.py",
    TRANSFORM_SOURCE_PATH,
    ORIENTATION_SOURCE_PATH,
    "spectral_utils/feature_utils.py",
    "spectral_utils/repgrid_scoring.py",
    FEATURE_ROSTER_SOURCE_PATH,
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/io.py",
)
AB_VERIFICATION_SOURCES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    FEATURE_CONFIG_PATH,
    METHOD_CONFIG_PATH,
    "configs/reconstruction_benchmark_v1/populations.json",
    "scripts/reconstruction_benchmark/verify_external_final_answer_ab.py",
    "spectral_utils/reconstruction_benchmark/external_ab.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
)


def _payload_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _verify_payload(value: Mapping[str, Any], *, field: str, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop(field, None)
    if recorded != _payload_hash(payload):
        raise RuntimeError(f"{name} {field} failed")


def _safe_child(root: Path, relative: str, *, description: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{description} escapes its root: {relative!r}") from error
    return candidate


def current_feature_contract_bindings(repo: str | Path) -> dict[str, Any]:
    """Validate and return the current transform/orientation/roster bindings."""

    root = Path(repo).resolve()
    config_path = root / FEATURE_CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "reconstruction-feature-contract-v1":
        raise RuntimeError("unexpected reconstruction feature-contract schema")
    if config.get("contract_id") != CONTRACT_VERSION:
        raise RuntimeError("feature-contract config does not name the executable mixed-v2 contract")
    if config.get("preprocessing_count") != 1 or config.get("nominal_feature_count") != len(CANONICAL_FEATURE_NAMES):
        raise RuntimeError("feature-contract preprocessing count or nominal roster size drifted")
    expected_sources = {
        "transform_source": TRANSFORM_SOURCE_PATH,
        "orientation_source": ORIENTATION_SOURCE_PATH,
    }
    for key, relative in expected_sources.items():
        if config.get(key) != relative:
            raise RuntimeError(f"feature-contract {key} changed: {config.get(key)!r}")
        observed = sha256_file(root / relative)
        if config.get(f"{key}_sha256") != observed:
            raise RuntimeError(f"feature-contract {key} hash is stale")
    declared_roster_source = str(config.get("roster_source", ""))
    if not declared_roster_source:
        raise RuntimeError("feature-contract config lacks its declared roster source")
    roster_path = _safe_child(root, declared_roster_source, description="declared contract roster")
    if config.get("roster_source_sha256") != sha256_file(roster_path):
        raise RuntimeError("feature-contract declared roster-source hash is stale")
    return {
        "feature_contract_id": CONTRACT_VERSION,
        "feature_contract_config_sha256": sha256_file(config_path),
        "transform_source_sha256": sha256_file(root / TRANSFORM_SOURCE_PATH),
        "orientation_source_sha256": sha256_file(root / ORIENTATION_SOURCE_PATH),
        "feature_roster_source_sha256": sha256_file(root / FEATURE_ROSTER_SOURCE_PATH),
        "declared_roster_source": declared_roster_source,
        "declared_roster_source_sha256": sha256_file(roster_path),
        "nominal_feature_roster_sha256": _payload_hash(list(CANONICAL_FEATURE_NAMES)),
        "nominal_feature_count": len(CANONICAL_FEATURE_NAMES),
    }


def verify_current_source_snapshot(
    snapshot: Mapping[str, Any],
    *,
    repo: str | Path,
    required_paths: Sequence[str] = (),
    name: str,
) -> None:
    """Reject a self-consistent snapshot whose source files are now stale."""

    payload = dict(snapshot)
    recorded = payload.pop("snapshot_sha256", None)
    if recorded != _payload_hash(payload):
        raise RuntimeError(f"{name} snapshot payload hash failed")
    files = snapshot.get("files", ())
    paths = [str(item.get("path", "")) for item in files]
    if any(not value for value in paths) or len(set(paths)) != len(paths):
        raise RuntimeError(f"{name} snapshot has empty or duplicate paths")
    missing = sorted(set(map(str, required_paths)) - set(paths))
    if missing:
        raise RuntimeError(f"{name} snapshot omits required sources: {missing}")
    root = Path(repo).resolve()
    for item in files:
        path = _safe_child(root, str(item["path"]), description=f"{name} source")
        if not path.is_file() or sha256_file(path) != str(item.get("sha256")):
            raise RuntimeError(f"{name} source changed or is missing: {item['path']}")


def _verify_target_free_source_records(record: Mapping[str, Any], *, source_root: Path) -> None:
    rows = record.get("source_files", ())
    if not rows:
        raise RuntimeError(f"{record.get('cell_id')}: eligible input lacks source provenance")
    seen: set[str] = set()
    for item in rows:
        relative = str(item.get("path", ""))
        if not relative or relative in seen:
            raise RuntimeError(f"{record.get('cell_id')}: invalid source-file roster")
        seen.add(relative)
        if item.get("role") == "label":
            raise RuntimeError(f"{record.get('cell_id')}: label file crossed the fitting boundary")
        path = _safe_child(source_root, relative, description="external telemetry source")
        if not path.is_file():
            raise RuntimeError(f"{record.get('cell_id')}: source disappeared: {relative}")
        if int(item.get("size_bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"{record.get('cell_id')}: source size changed: {relative}")
        if str(item.get("sha256")) != sha256_file(path):
            raise RuntimeError(f"{record.get('cell_id')}: source hash changed: {relative}")


def validate_scientific_input_manifest(
    path: str | Path,
    *,
    registry: ExternalRegistry,
    repo: str | Path,
    input_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a complete, current, target-free preparation tree."""

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_payload(manifest, field="payload_sha256", name="external input manifest")
    if manifest.get("schema_version") != INPUT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unexpected external input-manifest schema")
    if manifest.get("scientific_full_build") is not True:
        raise RuntimeError("partial external input build cannot be published")
    if manifest.get("applicability_complete") is not True or manifest.get("complete_eligible_roster") is not True:
        raise RuntimeError("external applicability/eligible roster is incomplete")
    if manifest.get("external_registry_sha256") != registry.sha256:
        raise RuntimeError("external input manifest binds another registry")
    if manifest.get("population_registry_sha256") != registry.population_registry_sha256:
        raise RuntimeError("external input manifest binds another population registry")
    if manifest.get("feature_contract_id") != CONTRACT_VERSION:
        raise RuntimeError("external input manifest binds another feature contract")
    if manifest.get("mixed_v2_applied_exactly_once") is not True:
        raise RuntimeError("external input manifest does not attest one mixed-v2 pass")
    if manifest.get("labels_opened") is not False or manifest.get("historical_scores_opened") is not False:
        raise RuntimeError("external input manifest does not prove target isolation")
    snapshot = manifest.get("preparation_source_snapshot", {})
    if manifest.get("preparation_source_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise RuntimeError("external input manifest preparation-snapshot binding failed")
    verify_current_source_snapshot(
        snapshot,
        repo=repo,
        required_paths=REQUIRED_PREPARATION_SOURCES,
        name="external preparation",
    )
    bindings = current_feature_contract_bindings(repo)

    rows = manifest.get("cells", ())
    ids = [str(item.get("cell_id", "")) for item in rows]
    expected_ids = [item.cell_id for item in registry.cells]
    if ids != expected_ids:
        raise RuntimeError("external input manifest is not the exact ordered registered-cell roster")
    if int(manifest.get("n_registered_cells", -1)) != len(expected_ids):
        raise RuntimeError("external input registered-cell count drifted")
    expected_runnable = sum(item.fit_policy == "run_if_compatible" for item in registry.cells)
    if int(manifest.get("n_runnable_cells", -1)) != expected_runnable:
        raise RuntimeError("external input runnable-cell count drifted")
    source_root = Path(str(manifest.get("source_root", ""))).resolve()
    if not source_root.is_dir():
        raise RuntimeError("external input source root is unavailable")
    artifact_root = Path(input_root) if input_root is not None else manifest_path.parent
    eligible: list[str] = []
    for row, spec in zip(rows, registry.cells):
        if row.get("population_id") != spec.population_id or int(row.get("expected_rows", -1)) != spec.expected_rows:
            raise RuntimeError(f"{spec.cell_id}: input population/count binding changed")
        status = str(row.get("status", ""))
        if spec.fit_policy == "forbidden":
            if status != spec.configured_status or row.get("prepared") is not False:
                raise RuntimeError(f"{spec.cell_id}: terminal applicability status changed")
            continue
        if status not in {"ELIGIBLE", "INCOMPATIBLE_FEATURE_CONTRACT"}:
            raise RuntimeError(f"{spec.cell_id}: unresolved scientific applicability status {status!r}")
        if status != "ELIGIBLE":
            if row.get("prepared") is not False or not row.get("reason"):
                raise RuntimeError(f"{spec.cell_id}: incompatible status lacks explicit evidence")
            continue
        eligible.append(spec.cell_id)
        exact = {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "cell_id": spec.cell_id,
            "population_id": spec.population_id,
            "dataset_id": spec.dataset_id,
            "model_id": spec.model_id,
            "slice_id": spec.slice_id,
            "domain": spec.domain,
            "comparison_group_id": spec.comparison_group_id,
            "panel_role": spec.panel_role,
            "adapter_id": spec.adapter_id,
            "n_rows": spec.expected_rows,
            "feature_contract_id": CONTRACT_VERSION,
            "preprocessing_steps": [CONTRACT_VERSION],
            "mixed_v2_applied_count": 1,
            "labels_opened": False,
            "historical_scores_opened": False,
        }
        for key, value in exact.items():
            if row.get(key) != value:
                raise RuntimeError(f"{spec.cell_id}: prepared input field {key} drifted")
        names = tuple(map(str, row.get("feature_names", ())))
        canonical_subset = tuple(name for name in CANONICAL_FEATURE_NAMES if name in set(names))
        if names != canonical_subset or len(names) < 3 or int(row.get("n_features", -1)) != len(names):
            raise RuntimeError(f"{spec.cell_id}: invalid present-feature roster")
        absent = [name for name in CANONICAL_FEATURE_NAMES if name not in set(names)]
        if row.get("absent_feature_names") != absent:
            raise RuntimeError(f"{spec.cell_id}: absent-feature roster drifted")
        if row.get("present_feature_roster_sha256") != _payload_hash(list(names)):
            raise RuntimeError(f"{spec.cell_id}: present-feature roster hash failed")
        if row.get("nominal_feature_roster_sha256") != bindings["nominal_feature_roster_sha256"]:
            raise RuntimeError(f"{spec.cell_id}: nominal feature roster hash failed")
        if int(row.get("nominal_feature_count", -1)) != len(CANONICAL_FEATURE_NAMES):
            raise RuntimeError(f"{spec.cell_id}: nominal feature count failed")
        _verify_target_free_source_records(row, source_root=source_root)
        artifact_relative = str(row.get("artifact_path", ""))
        if not artifact_relative:
            raise RuntimeError(f"{spec.cell_id}: eligible input lacks an artifact")
        load_prepared_external_cell(
            artifact_path=_safe_child(artifact_root, artifact_relative, description="prepared artifact"),
            record=row,
        )
    if int(manifest.get("n_prepared_cells", -1)) != len(eligible):
        raise RuntimeError("external input prepared-cell count drifted")
    manifest["_validated_feature_contract_bindings"] = bindings
    manifest["_eligible_cell_ids"] = eligible
    return manifest


def _load_current_method_contract(repo: Path) -> tuple[str, dict[str, Mapping[str, Any]]]:
    path = repo / METHOD_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("methods", ())
    ids = tuple(str(item.get("method_id")) for item in rows)
    if ids != PRIMARY_METHOD_IDS or int(raw.get("primary_roster_size", -1)) != len(PRIMARY_METHOD_IDS):
        raise RuntimeError("current method registry is not the executable 13-method roster")
    by_id = {str(item["method_id"]): item for item in rows}
    for method_id in PRIMARY_METHOD_IDS:
        spec = PRIMARY_METHOD_SPECS[method_id]
        if by_id[method_id].get("method_version_id") != spec.method_version_id:
            raise RuntimeError(f"{method_id}: current method version disagreement")
        if by_id[method_id].get("config_sha256") != spec.config_sha256:
            raise RuntimeError(f"{method_id}: current method config disagreement")
    return sha256_file(path), by_id


def validate_scientific_score_freeze(
    path: str | Path,
    *,
    registry: ExternalRegistry,
    repo: str | Path,
    input_root: str | Path,
    fit_root: str | Path | None = None,
    input_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact publication score roster and every frozen artifact."""

    freeze_path = Path(path)
    fit = Path(fit_root) if fit_root is not None else freeze_path.parent
    root = Path(repo).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _verify_payload(freeze, field="payload_sha256", name="external score freeze")
    if freeze.get("schema_version") != SCORE_FREEZE_SCHEMA_VERSION:
        raise RuntimeError("unexpected external score-freeze schema")
    if freeze.get("scientific_full") is not True:
        raise RuntimeError("debug or partial score freeze cannot open publication labels")
    if freeze.get("all_expected_scores_present") is not True:
        raise RuntimeError("external score freeze is incomplete")
    isolation = {
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "historical_scores_opened": False,
        "donors_used": False,
        "family_nrm_pgrd_regime": "A_within_cell_fully_unsupervised",
    }
    for key, value in isolation.items():
        if freeze.get(key) != value:
            raise RuntimeError(f"external score freeze violates {key}")
    if freeze.get("external_registry_sha256") != registry.sha256:
        raise RuntimeError("external score freeze binds another registry")
    if freeze.get("population_registry_sha256") != registry.population_registry_sha256:
        raise RuntimeError("external score freeze binds another population registry")
    if tuple(freeze.get("method_ids", ())) != PRIMARY_METHOD_IDS:
        raise RuntimeError("external score freeze is not the exact primary 13-method roster")
    current_method_sha, method_rows = _load_current_method_contract(root)
    if freeze.get("method_registry_sha256") != current_method_sha:
        raise RuntimeError("external score freeze method registry is stale")
    bindings = current_feature_contract_bindings(root)
    if freeze.get("feature_contract_bindings") != bindings:
        raise RuntimeError("external score freeze feature-contract bindings are stale")
    manifest_path = Path(input_root) / "MANIFEST.json"
    manifest = dict(input_manifest) if input_manifest is not None else validate_scientific_input_manifest(
        manifest_path, registry=registry, repo=root, input_root=input_root,
    )
    if freeze.get("input_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("external score freeze/input manifest file binding failed")
    if freeze.get("input_manifest_payload_sha256") != manifest.get("payload_sha256"):
        raise RuntimeError("external score freeze/input manifest payload binding failed")
    if freeze.get("preparation_source_snapshot_sha256") != manifest.get("preparation_source_snapshot_sha256"):
        raise RuntimeError("external score freeze/preparation snapshot binding failed")
    prefit_path = fit / "FIT_SOURCE_SNAPSHOT.json"
    prefit = json.loads(prefit_path.read_text(encoding="utf-8"))
    _verify_payload(prefit, field="payload_sha256", name="external prefit manifest")
    if freeze.get("prefit_snapshot_sha256") != sha256_file(prefit_path):
        raise RuntimeError("external score freeze/prefit file binding failed")
    if prefit.get("scientific_full") is not True:
        raise RuntimeError("external prefit snapshot is not scientific/full")
    expected_prefit = {
        "release_id": freeze.get("release_id"),
        "build_id": freeze.get("build_id"),
        "input_manifest_sha256": freeze.get("input_manifest_sha256"),
        "input_manifest_payload_sha256": freeze.get("input_manifest_payload_sha256"),
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "preparation_source_snapshot_sha256": manifest.get("preparation_source_snapshot_sha256"),
        "method_ids": list(PRIMARY_METHOD_IDS),
        "cell_ids": list(manifest.get("_eligible_cell_ids", ())),
    }
    for key, value in expected_prefit.items():
        if prefit.get(key) != value:
            raise RuntimeError(f"external prefit {key} binding failed")
    if prefit.get("feature_contract_bindings") != bindings:
        raise RuntimeError("external prefit feature-contract binding failed")
    snapshot = prefit.get("source_snapshot", {})
    if prefit.get("source_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise RuntimeError("external prefit/source snapshot binding failed")
    verify_current_source_snapshot(snapshot, repo=root, name="external fitting")
    if freeze.get("source_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise RuntimeError("external score freeze/fitting snapshot binding failed")

    eligible = tuple(manifest.get("_eligible_cell_ids", ()))
    if tuple(freeze.get("cell_ids", ())) != eligible:
        raise RuntimeError("external score freeze is not the exact eligible-cell roster")
    expected_statuses = [
        {"cell_id": item["cell_id"], "status": item["status"]}
        for item in manifest["cells"]
    ]
    if freeze.get("applicability_statuses") != expected_statuses:
        raise RuntimeError("external score freeze applicability roster differs from preparation")
    expected_n = len(eligible) * len(PRIMARY_METHOD_IDS)
    if int(freeze.get("expected_records", -1)) != expected_n or int(freeze.get("n_records", -1)) != expected_n:
        raise RuntimeError("external score freeze record count failed")
    records = freeze.get("records", ())
    pairs = [(str(item.get("cell_id")), str(item.get("method_id"))) for item in records]
    expected_pairs = [(cell_id, method_id) for cell_id in eligible for method_id in PRIMARY_METHOD_IDS]
    if pairs != expected_pairs:
        raise RuntimeError("external score freeze record order/roster is not the exact Cartesian product")
    prepared_by_cell = {item["cell_id"]: item for item in manifest["cells"] if item.get("status") == "ELIGIBLE"}
    for record in records:
        cell_id, method_id = str(record["cell_id"]), str(record["method_id"])
        method = method_rows[method_id]
        exact = {
            "population_id": registry.by_cell[cell_id].population_id,
            "method_version_id": method["method_version_id"],
            "config_sha256": method["config_sha256"],
            "status": record.get("status"),
            "prepared_matrix_sha256": prepared_by_cell[cell_id]["prepared_matrix_sha256"],
        }
        if record.get("status") not in SUCCESS:
            raise RuntimeError(f"{cell_id}/{method_id}: frozen fit was not successful")
        for key, value in exact.items():
            if record.get(key) != value:
                raise RuntimeError(f"{cell_id}/{method_id}: frozen {key} drifted")
        for key in ("record_path", "score_path", "artifact_index_path"):
            if not record.get(key):
                raise RuntimeError(f"{cell_id}/{method_id}: missing {key}")
        record_path = _safe_child(fit, str(record["record_path"]), description="method record")
        score_path = _safe_child(fit, str(record["score_path"]), description="method score")
        index_path = _safe_child(fit, str(record["artifact_index_path"]), description="artifact index")
        if sha256_file(record_path) != record.get("record_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: RECORD hash failed")
        if sha256_file(score_path) != record.get("score_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: score hash failed")
        if sha256_file(index_path) != record.get("artifact_index_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: artifact-index hash failed")
        if record.get("artifacts_path") is None:
            if record.get("artifacts_sha256") is not None:
                raise RuntimeError(f"{cell_id}/{method_id}: inconsistent optional artifact binding")
        else:
            artifacts_path = _safe_child(fit, str(record["artifacts_path"]), description="method artifacts")
            if sha256_file(artifacts_path) != record.get("artifacts_sha256"):
                raise RuntimeError(f"{cell_id}/{method_id}: artifact hash failed")
        disk_record = json.loads(record_path.read_text(encoding="utf-8"))
        for key in (
            "method_id", "method_version_id", "config_sha256", "status", "population_id",
            "cell_id", "prepared_matrix_sha256", "score_sha256", "artifacts_sha256",
            "artifact_index_sha256",
        ):
            if disk_record.get(key) != record.get(key):
                raise RuntimeError(f"{cell_id}/{method_id}: freeze/RECORD {key} mismatch")
        bundle = load_npz_no_pickle(score_path)
        if set(bundle) != {"row_ids", "score"}:
            raise RuntimeError(f"{cell_id}/{method_id}: unexpected score members")
        prepared_path = _safe_child(Path(input_root), str(prepared_by_cell[cell_id]["artifact_path"]), description="prepared input")
        prepared = load_npz_no_pickle(prepared_path)
        if tuple(map(str, bundle["row_ids"].tolist())) != tuple(map(str, prepared["row_ids"].tolist())):
            raise RuntimeError(f"{cell_id}/{method_id}: score/prepared row roster mismatch")
    freeze["_validated_input_manifest"] = manifest
    return freeze


def _tree_binding(path: Path) -> dict[str, Any]:
    value = canonical_tree_manifest(path)
    return {"root": str(path.resolve()), **value}


def _verification_source_snapshot(repo: str | Path) -> dict[str, Any]:
    root = Path(repo).resolve()
    value = {
        "files": [
            {"path": relative, "sha256": sha256_file(root / relative)}
            for relative in AB_VERIFICATION_SOURCES
        ]
    }
    value["snapshot_sha256"] = _payload_hash(value)
    return value


def _public_manifest_view(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if not key.startswith("_") and key not in {"build_id", "payload_sha256"}
    }


def verify_external_ab(
    *,
    release_id: str,
    release_root: str | Path,
    registry: ExternalRegistry,
    repo: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify both independent builds and issue an exact content certificate."""

    release = Path(release_root) / release_id
    audits: dict[str, dict[str, Any]] = {}
    for build_id in ("A", "B"):
        root = release / f"build_{build_id}" / "external_final_answer"
        inputs, fit = root / "inputs", root / "fit"
        manifest_path = inputs / "MANIFEST.json"
        manifest = validate_scientific_input_manifest(
            manifest_path, registry=registry, repo=repo, input_root=inputs,
        )
        freeze_path = fit / "SCORE_FREEZE_MANIFEST.json"
        freeze = validate_scientific_score_freeze(
            freeze_path,
            registry=registry,
            repo=repo,
            input_root=inputs,
            fit_root=fit,
            input_manifest=manifest,
        )
        if manifest.get("release_id") != release_id or freeze.get("release_id") != release_id:
            raise RuntimeError(f"build {build_id}: release ID binding failed")
        if manifest.get("build_id") != build_id or freeze.get("build_id") != build_id:
            raise RuntimeError(f"build {build_id}: build ID binding failed")
        audits[build_id] = {
            "manifest": manifest,
            "freeze": freeze,
            "input_manifest_sha256": sha256_file(manifest_path),
            "input_manifest_payload_sha256": manifest["payload_sha256"],
            "score_freeze_sha256": sha256_file(freeze_path),
            "score_freeze_payload_sha256": freeze["payload_sha256"],
            "input_tree": _tree_binding(inputs),
            "fit_tree": _tree_binding(fit),
        }
    left, right = audits["A"], audits["B"]
    if _public_manifest_view(left["manifest"]) != _public_manifest_view(right["manifest"]):
        raise RuntimeError("A/B input manifests differ beyond build identity")
    left_freeze, right_freeze = left["freeze"], right["freeze"]
    if left_freeze["cell_ids"] != right_freeze["cell_ids"] or left_freeze["method_ids"] != right_freeze["method_ids"]:
        raise RuntimeError("A/B fitted rosters differ")
    comparisons: list[dict[str, Any]] = []
    for left_record, right_record in zip(left_freeze["records"], right_freeze["records"]):
        identity = (left_record["cell_id"], left_record["method_id"])
        if identity != (right_record["cell_id"], right_record["method_id"]):
            raise RuntimeError("A/B score record order differs")
        exact_fields = (
            "method_version_id", "config_sha256", "status", "prepared_matrix_sha256",
            "score_sha256", "record_sha256", "artifacts_sha256", "artifact_index_sha256",
        )
        unequal = [key for key in exact_fields if left_record.get(key) != right_record.get(key)]
        if unequal:
            raise RuntimeError(f"A/B outputs differ for {identity}: {unequal}")
        comparisons.append({
            "cell_id": identity[0],
            "method_id": identity[1],
            **{key: left_record.get(key) for key in exact_fields},
        })
    input_left = {item["cell_id"]: item for item in left["manifest"]["cells"] if item.get("status") == "ELIGIBLE"}
    input_right = {item["cell_id"]: item for item in right["manifest"]["cells"] if item.get("status") == "ELIGIBLE"}
    for cell_id in left_freeze["cell_ids"]:
        for key in (
            "artifact_sha256", "prepared_matrix_sha256", "row_signature_sha256",
            "present_feature_roster_sha256", "nominal_feature_roster_sha256", "source_files",
            "source_inventory_bindings", "transform_details",
        ):
            if input_left[cell_id].get(key) != input_right[cell_id].get(key):
                raise RuntimeError(f"A/B prepared inputs differ for {cell_id}: {key}")

    bindings = current_feature_contract_bindings(repo)
    certificate = {
        "schema_version": AB_CERTIFICATE_SCHEMA_VERSION,
        "release_id": release_id,
        "status": "PASS",
        "scientific_full": True,
        "attestation_algorithm": "sha256-canonical-json-v1",
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "method_registry_sha256": sha256_file(Path(repo) / METHOD_CONFIG_PATH),
        "feature_contract_bindings": bindings,
        "verification_source_snapshot": _verification_source_snapshot(repo),
        "method_ids": list(PRIMARY_METHOD_IDS),
        "cell_ids": list(left_freeze["cell_ids"]),
        "n_method_comparisons": len(comparisons),
        "comparison_records": comparisons,
        "comparison_records_sha256": _payload_hash(comparisons),
        "builds": {
            build_id: {
                key: value
                for key, value in audits[build_id].items()
                if key not in {"manifest", "freeze"}
            }
            for build_id in ("A", "B")
        },
    }
    certificate["certificate_sha256"] = _payload_hash(certificate)
    target = Path(output_path) if output_path is not None else release / "external_final_answer" / "AB_VERIFICATION.json"
    atomic_write_json(target, certificate)
    return certificate


def assert_external_ab_certificate(
    path: str | Path,
    *,
    release_id: str,
    release_root: str | Path,
    selected_build: str,
    registry: ExternalRegistry,
    repo: str | Path,
) -> dict[str, Any]:
    """Revalidate a PASS certificate and both unchanged trees before labels."""

    if selected_build not in {"A", "B"}:
        raise RuntimeError("selected build must be A or B")
    certificate = json.loads(Path(path).read_text(encoding="utf-8"))
    _verify_payload(certificate, field="certificate_sha256", name="external A/B certificate")
    if certificate.get("schema_version") != AB_CERTIFICATE_SCHEMA_VERSION or certificate.get("status") != "PASS":
        raise RuntimeError("a passing external A/B certificate is required before labels")
    if certificate.get("scientific_full") is not True or certificate.get("release_id") != release_id:
        raise RuntimeError("external A/B certificate is partial or for another release")
    if certificate.get("external_registry_sha256") != registry.sha256:
        raise RuntimeError("external A/B certificate registry is stale")
    if certificate.get("population_registry_sha256") != registry.population_registry_sha256:
        raise RuntimeError("external A/B certificate population registry is stale")
    if certificate.get("feature_contract_bindings") != current_feature_contract_bindings(repo):
        raise RuntimeError("external A/B certificate feature contract is stale")
    verification_snapshot = certificate.get("verification_source_snapshot", {})
    verify_current_source_snapshot(
        verification_snapshot,
        repo=repo,
        required_paths=AB_VERIFICATION_SOURCES,
        name="external A/B verification",
    )
    method_sha, _ = _load_current_method_contract(Path(repo).resolve())
    if certificate.get("method_registry_sha256") != method_sha:
        raise RuntimeError("external A/B certificate method registry is stale")
    if tuple(certificate.get("method_ids", ())) != PRIMARY_METHOD_IDS:
        raise RuntimeError("external A/B certificate method roster is not the exact 13")
    cell_ids = tuple(map(str, certificate.get("cell_ids", ())))
    if not cell_ids or len(set(cell_ids)) != len(cell_ids):
        raise RuntimeError("external A/B certificate has an invalid eligible-cell roster")
    if int(certificate.get("n_method_comparisons", -1)) != len(cell_ids) * len(PRIMARY_METHOD_IDS):
        raise RuntimeError("external A/B certificate comparison count is not the exact Cartesian roster")
    comparison_digest = str(certificate.get("comparison_records_sha256", ""))
    comparison_records = certificate.get("comparison_records", ())
    if (
        not isinstance(comparison_records, list)
        or len(comparison_records) != len(cell_ids) * len(PRIMARY_METHOD_IDS)
        or comparison_digest != _payload_hash(comparison_records)
    ):
        raise RuntimeError("external A/B certificate lacks its comparison-record digest")
    release = Path(release_root) / release_id
    for build_id in ("A", "B"):
        root = release / f"build_{build_id}" / "external_final_answer"
        expected = certificate.get("builds", {}).get(build_id, {})
        input_manifest = root / "inputs" / "MANIFEST.json"
        score_freeze = root / "fit" / "SCORE_FREEZE_MANIFEST.json"
        if expected.get("input_manifest_sha256") != sha256_file(input_manifest):
            raise RuntimeError(f"build {build_id}: input manifest changed after A/B certification")
        if expected.get("score_freeze_sha256") != sha256_file(score_freeze):
            raise RuntimeError(f"build {build_id}: score freeze changed after A/B certification")
        current_input_tree = canonical_tree_manifest(root / "inputs")
        current_fit_tree = canonical_tree_manifest(root / "fit")
        if expected.get("input_tree", {}).get("tree_sha256") != current_input_tree["tree_sha256"]:
            raise RuntimeError(f"build {build_id}: input tree changed after A/B certification")
        if expected.get("fit_tree", {}).get("tree_sha256") != current_fit_tree["tree_sha256"]:
            raise RuntimeError(f"build {build_id}: fit tree changed after A/B certification")
    return certificate


__all__ = [
    "AB_CERTIFICATE_SCHEMA_VERSION",
    "AB_VERIFICATION_SOURCES",
    "FEATURE_CONFIG_PATH",
    "FEATURE_ROSTER_SOURCE_PATH",
    "ORIENTATION_SOURCE_PATH",
    "REQUIRED_PREPARATION_SOURCES",
    "TRANSFORM_SOURCE_PATH",
    "assert_external_ab_certificate",
    "current_feature_contract_bindings",
    "validate_scientific_input_manifest",
    "validate_scientific_score_freeze",
    "verify_current_source_snapshot",
    "verify_external_ab",
]
