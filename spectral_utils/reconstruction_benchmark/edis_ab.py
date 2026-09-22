"""Controller-only exact A/B verification for the EDIS reconstruction lane.

No labels are loaded here.  The verifier checks the two independent prepared
trees, all 12x13 score records, the private source/group commitments, and the
current target-free registry before issuing a public certificate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .edis_fit import (
    PREFIT_SCHEMA,
    SCORE_FREEZE_SCHEMA,
    load_fit_registry,
    load_prepared_cell,
)
from .fit_firewall import validate_fit_audit_policy
from .edis_preparation import (
    PREPARATION_SOURCE_PATHS,
    PRIVATE_PROVENANCE_SCHEMA,
    assert_expected_preparation_status_roster,
    load_preparation_registry,
    load_preparation_status,
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


CERTIFICATE_SCHEMA = "reconstruction-edis-ab-certificate-v1"
EVALUATION_CERTIFICATE_SCHEMA = "reconstruction-edis-evaluation-ab-certificate-v1"
EVALUATION_SCHEMA = "reconstruction-edis-evaluation-v1"
FIT_SOURCE_PATHS = (
    "configs/reconstruction_benchmark_v1/edis_target_free.json",
    "configs/reconstruction_benchmark_v1/fit_safe_feature_contract.json",
    "configs/reconstruction_benchmark_v1/fit_safe_feature_roster.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "scripts/reconstruction_benchmark/run_edis_methods.py",
    "scripts/reconstruction_benchmark/edis_fit_worker.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/edis_ab.py",
    "spectral_utils/reconstruction_benchmark/edis_fit.py",
    "spectral_utils/reconstruction_benchmark/edis_identity.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
    "spectral_utils/reconstruction_benchmark/serialization.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/selectors/a2_groupfs.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/specrage_laplacian.py",
    "spectral_utils/fusion_aware_views.py",
    "spectral_utils/residual_graph_deem.py",
    "spectral_utils/contribution_subspace.py",
    "spectral_utils/graph_topology.py",
)
_SUCCESS = {"OK", "OK_FALLBACK"}


def _payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _verify_payload(value: Mapping[str, Any], *, field: str, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop(field, None)
    if recorded != _payload_sha256(payload):
        raise RuntimeError(f"{name} {field} failed")


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"artifact escapes root: {relative!r}") from error
    return path


def canonical_evaluation_table_sha256(
    *,
    table: str,
    rows: list[Mapping[str, Any]],
    release_id: str,
    build_id: str,
) -> str:
    """Hash a tidy EDIS table after removing only its A/B run identifier."""

    from spectral_utils.reconstruction_reporting.schemas import table_sha256

    expected_run = f"{release_id}::edis::{build_id}::postfreeze"
    canonical_run = f"{release_id}::edis::<BUILD>::postfreeze"
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("run_id") != expected_run:
            raise RuntimeError(
                f"EDIS {table} row {index} has an unexpected build run identifier"
            )
        normalized.append({**dict(row), "run_id": canonical_run})
    if not normalized:
        raise RuntimeError(f"EDIS {table} table is unexpectedly empty")
    return table_sha256(table, normalized)


def verify_current_source_snapshot(
    snapshot: Mapping[str, Any],
    *,
    repo: Path,
    name: str,
    require_clean: bool,
    expected_paths: tuple[str, ...],
) -> None:
    payload = dict(snapshot)
    recorded = payload.pop("snapshot_sha256", None)
    if recorded != _payload_sha256(payload):
        raise RuntimeError(f"{name} snapshot hash failed")
    if require_clean and snapshot.get("git_clean") is not True:
        raise RuntimeError(f"{name} snapshot was not created from a clean worktree")
    rows = snapshot.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{name} snapshot has no source files")
    if [str(row.get("path", "")) for row in rows] != list(expected_paths):
        raise RuntimeError(f"{name} snapshot source roster drifted")
    seen: set[str] = set()
    root = repo.resolve()
    for row in rows:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise RuntimeError(f"{name} snapshot has empty/duplicate source paths")
        seen.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"{name} source escapes repository: {relative}") from error
        if sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"{name} source changed or is missing: {relative}")


def load_private_provenance(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _verify_payload(raw, field="payload_sha256", name="EDIS private preparation provenance")
    if raw.get("schema_version") != PRIVATE_PROVENANCE_SCHEMA:
        raise RuntimeError("unexpected EDIS private preparation schema")
    if raw.get("labels_opened") is not False or raw.get("historical_scores_opened") is not False:
        raise RuntimeError("EDIS private preparation did not attest target isolation")
    scientific_full = raw.get("scientific_full_build") is True
    descriptive_partial = raw.get("partial_descriptive_build") is True
    status_only = raw.get("status_only_build") is True
    if (
        sum((scientific_full, descriptive_partial, status_only)) != 1
        or raw.get("headline_eligible") is not False
    ):
        raise RuntimeError("EDIS private preparation full/partial status drifted")
    rows = raw.get("cells")
    if not isinstance(rows, list) or len(rows) != 12:
        raise RuntimeError("EDIS private preparation roster is incomplete")
    ids = [str(item.get("cell_id", "")) for item in rows]
    if any(not value for value in ids) or len(set(ids)) != 12:
        raise RuntimeError("EDIS private preparation roster is invalid")
    for row in rows:
        if row.get("status") not in {
            "READY",
            "BLOCKED_TRACE_BELOW_FROZEN_MIN",
            "BLOCKED_MALFORMED_TELEMETRY",
            "BLOCKED_PARTIAL_FEATURE_AVAILABILITY",
            "BLOCKED_ASSET",
        }:
            raise RuntimeError(f"{row.get('cell_id')}: unknown private cell status")
        for key in ("source", "source_manifest"):
            item = row.get(key, {})
            if not item.get("path") or not item.get("sha256") or int(item.get("size_bytes", -1)) <= 0:
                raise RuntimeError(f"{row.get('cell_id')}: private source binding is incomplete")
        if row.get("status") != "BLOCKED_ASSET":
            if len(str(row.get("group_membership_commitment_sha256", ""))) != 64:
                raise RuntimeError(f"{row.get('cell_id')}: group commitment is malformed")
            if len(str(row.get("question_roster_commitment_sha256", ""))) != 64:
                raise RuntimeError(f"{row.get('cell_id')}: question-roster commitment is malformed")
    ready_count = sum(row.get("status") == "READY" for row in rows)
    if (
        (scientific_full and ready_count != 12)
        or (descriptive_partial and not 0 <= ready_count < 12)
        or raw.get("fit_registry_available") is not (not status_only)
        or (not status_only and raw.get("status_roster_contract_match") is not True)
        or raw.get("aggregate_metrics_allowed") is not scientific_full
        or len(str(raw.get("preparation_status_commitment_sha256", ""))) != 64
    ):
        raise RuntimeError("EDIS private cell-status accounting failed")
    private_binding = raw.get("private_identity_contract")
    private_commitment = raw.get("private_identity_contract_commitment_sha256")
    if (
        not isinstance(private_binding, Mapping)
        or private_commitment != _payload_sha256(private_binding)
        or private_binding.get("group_namespace_scope")
        != "dataset_question_content_postfreeze_only"
    ):
        raise RuntimeError("EDIS private identity-contract commitment failed")
    return raw


def validate_score_freeze(
    *,
    fit_root: str | Path,
    input_root: str | Path,
    expected_build: str,
    repo: str | Path,
    private_audit_policy_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = Path(fit_root)
    inputs = Path(input_root)
    registry_path = inputs / "FIT_REGISTRY.json"
    registry = load_fit_registry(registry_path)
    freeze_path = fit / "SCORE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _verify_payload(freeze, field="payload_sha256", name="EDIS score freeze")
    if freeze.get("schema_version") != SCORE_FREEZE_SCHEMA:
        raise RuntimeError("unexpected EDIS score-freeze schema")
    if freeze.get("build_id") != expected_build or registry.get("build_id") != expected_build:
        raise RuntimeError("EDIS build binding failed")
    if freeze.get("release_id") != registry.get("release_id"):
        raise RuntimeError("EDIS release binding failed")
    expected_full = registry["scientific_full_build"] is True
    expected_partial = registry["partial_descriptive_build"] is True
    preparation_status_path = inputs.parent / "PREPARATION_STATUS.json"
    preparation_status = load_preparation_status(preparation_status_path)
    ready_status_ids = [
        row["cell_id"]
        for row in preparation_status["cells"]
        if row["status"] == "READY"
    ]
    if (
        preparation_status["status_commitment_sha256"]
        != registry["preparation_status_commitment_sha256"]
        or preparation_status.get("release_id") != freeze.get("release_id")
        or preparation_status.get("build_id") != expected_build
        or ready_status_ids != [row["cell_id"] for row in registry["cells"]]
        or preparation_status["scientific_full_build"] is not expected_full
        or preparation_status["partial_descriptive_build"] is not expected_partial
    ):
        raise RuntimeError("EDIS preparation status is not bound to the score freeze")
    attestations = {
        "scientific_full": expected_full,
        "descriptive_partial": expected_partial,
        "headline_eligible": False,
        "aggregate_metrics_allowed": registry["aggregate_metrics_allowed"],
        "preparation_status_commitment_sha256": registry[
            "preparation_status_commitment_sha256"
        ],
        "all_expected_scores_present": True,
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "raw_sources_opened_by_fit": False,
        "group_structure_opened_by_fit": False,
        "historical_scores_opened": False,
        "donors_used": False,
        "score_semantics": "higher_is_incorrect",
        "fit_isolation_tier": "trusted_first_party_python_audit_hook_v1",
        "firewall_violations": [],
    }
    for key, expected in attestations.items():
        if freeze.get(key) != expected:
            raise RuntimeError(f"EDIS score-freeze {key} attestation failed")
    prefit_path = fit / "FIT_SOURCE_SNAPSHOT.json"
    if sha256_file(prefit_path) != freeze.get("prefit_sha256"):
        raise RuntimeError("EDIS score freeze prefit file binding failed")
    prefit = json.loads(prefit_path.read_text(encoding="utf-8"))
    _verify_payload(prefit, field="payload_sha256", name="EDIS prefit snapshot")
    if (
        prefit.get("schema_version") != PREFIT_SCHEMA
        or prefit.get("release_id") != freeze.get("release_id")
        or prefit.get("build_id") != expected_build
        or prefit.get("scientific_full") is not expected_full
        or prefit.get("descriptive_partial") is not expected_partial
        or prefit.get("headline_eligible") is not False
        or prefit.get("aggregate_metrics_allowed")
        != registry["aggregate_metrics_allowed"]
        or prefit.get("preparation_status_commitment_sha256")
        != registry["preparation_status_commitment_sha256"]
        or prefit.get("fit_registry_sha256") != freeze.get("fit_registry_sha256")
        or prefit.get("fit_registry_payload_sha256")
        != freeze.get("fit_registry_payload_sha256")
        or prefit.get("identity_contract") != registry.get("identity_contract")
        or prefit.get("method_ids") != list(PRIMARY_METHOD_IDS)
        or prefit.get("cell_ids") != [row["cell_id"] for row in registry["cells"]]
    ):
        raise RuntimeError("EDIS prefit snapshot binding drifted")
    verify_current_source_snapshot(
        prefit.get("fit_source_snapshot", {}),
        repo=Path(repo),
        name="EDIS fit source",
        require_clean=True,
        expected_paths=FIT_SOURCE_PATHS,
    )
    private_policy_path = Path(private_audit_policy_path)
    if sha256_file(private_policy_path) != freeze.get(
        "private_audit_policy_file_sha256"
    ):
        raise RuntimeError("EDIS private audit-policy file binding failed")
    policy = validate_fit_audit_policy(
        json.loads(private_policy_path.read_text(encoding="utf-8"))
    )
    if policy.get("policy_sha256") != freeze.get("audit_policy_sha256"):
        raise RuntimeError("EDIS audit-policy payload binding failed")
    expected_probe_ids = [row["probe_id"] for row in policy["forbidden_probes"]]
    if [row["probe_id"] for row in freeze.get("denial_probes", ())] != expected_probe_ids:
        raise RuntimeError("EDIS denial probes differ from the frozen audit policy")
    capsule = fit.parent / "fit_capsule"
    if canonical_tree_manifest(capsule)["tree_sha256"] != freeze.get(
        "capsule_tree_sha256"
    ):
        raise RuntimeError("EDIS fit capsule changed after fitting")
    if freeze.get("fit_registry_sha256") != sha256_file(registry_path):
        raise RuntimeError("EDIS score freeze does not bind the fit-registry file")
    if freeze.get("fit_registry_payload_sha256") != registry.get("payload_sha256"):
        raise RuntimeError("EDIS score freeze does not bind the fit-registry payload")
    if freeze.get("identity_contract") != registry.get("identity_contract"):
        raise RuntimeError("EDIS score freeze identity binding drifted")
    denial_probes = freeze.get("denial_probes")
    if (
        not isinstance(denial_probes, list)
        or len(denial_probes) < 6
        or any(
            set(item) != {"probe_id", "read_denied"}
            or item.get("read_denied") is not True
            for item in denial_probes
        )
    ):
        raise RuntimeError("EDIS score freeze lacks complete firewall denial probes")
    worker_path = fit / "WORKER_RESULT_MANIFEST.json"
    if sha256_file(worker_path) != freeze.get("worker_result_sha256"):
        raise RuntimeError("EDIS score freeze worker-result file binding failed")
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    _verify_payload(worker, field="payload_sha256", name="EDIS worker result")
    if worker.get("payload_sha256") != freeze.get("worker_result_payload_sha256"):
        raise RuntimeError("EDIS score freeze worker-result payload binding failed")
    if (
        worker.get("audit_policy_sha256") != freeze.get("audit_policy_sha256")
        or worker.get("denial_probes") != denial_probes
        or worker.get("firewall_violations") != []
        or worker.get("all_candidate_scores_present") is not True
        or worker.get("scientific_full_build") is not expected_full
        or worker.get("partial_descriptive_build") is not expected_partial
        or worker.get("headline_eligible") is not False
        or worker.get("aggregate_metrics_allowed")
        != registry["aggregate_metrics_allowed"]
        or worker.get("preparation_status_commitment_sha256")
        != registry["preparation_status_commitment_sha256"]
    ):
        raise RuntimeError("EDIS score freeze worker isolation attestation drifted")
    if tuple(freeze.get("method_ids", ())) != PRIMARY_METHOD_IDS:
        raise RuntimeError("EDIS score freeze is not the canonical 13-method roster")
    cell_ids = tuple(str(item["cell_id"]) for item in registry["cells"])
    if tuple(freeze.get("cell_ids", ())) != cell_ids:
        raise RuntimeError("EDIS score-freeze cell roster drifted")
    expected_n = len(cell_ids) * len(PRIMARY_METHOD_IDS)
    records = freeze.get("records", ())
    expected_pairs = [(cell, method) for cell in cell_ids for method in PRIMARY_METHOD_IDS]
    observed_pairs = [(str(row.get("cell_id")), str(row.get("method_id"))) for row in records]
    if (
        int(freeze.get("n_records", -1)) != expected_n
        or int(freeze.get("expected_records", -1)) != expected_n
        or observed_pairs != expected_pairs
    ):
        raise RuntimeError("EDIS score-freeze Cartesian roster is incomplete or reordered")
    by_cell = {str(item["cell_id"]): item for item in registry["cells"]}
    method_specs = PRIMARY_METHOD_SPECS
    for row in records:
        cell_id, method_id = str(row["cell_id"]), str(row["method_id"])
        if row.get("status") not in _SUCCESS:
            raise RuntimeError(f"{cell_id}/{method_id}: scientific freeze contains a failed method")
        spec = method_specs[method_id]
        if row.get("method_version_id") != spec.method_version_id or row.get("config_sha256") != spec.config_sha256:
            raise RuntimeError(f"{cell_id}/{method_id}: method contract drifted")
        if row.get("prepared_matrix_sha256") != by_cell[cell_id]["prepared_matrix_sha256"]:
            raise RuntimeError(f"{cell_id}/{method_id}: prepared matrix binding drifted")
        if row.get("row_roster_sha256") != by_cell[cell_id]["row_roster_sha256"]:
            raise RuntimeError(f"{cell_id}/{method_id}: row roster binding drifted")
        record_path = _safe_child(fit, str(row["record_path"]))
        score_path = _safe_child(fit, str(row["score_path"]))
        index_path = _safe_child(fit, str(row["artifact_index_path"]))
        if sha256_file(record_path) != row.get("record_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: method record hash failed")
        if sha256_file(score_path) != row.get("score_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: score hash failed")
        if sha256_file(index_path) != row.get("artifact_index_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: artifact-index hash failed")
        if row.get("artifacts_path") is None:
            if row.get("artifacts_sha256") is not None:
                raise RuntimeError(f"{cell_id}/{method_id}: optional artifact binding is inconsistent")
        else:
            artifact_path = _safe_child(fit, str(row["artifacts_path"]))
            if sha256_file(artifact_path) != row.get("artifacts_sha256"):
                raise RuntimeError(f"{cell_id}/{method_id}: method artifact hash failed")
        score = load_npz_no_pickle(score_path)
        if set(score) != {"row_ids", "score"}:
            raise RuntimeError(f"{cell_id}/{method_id}: fit-visible score contains unexpected members")
        prepared = load_prepared_cell(
            artifact_path=inputs / by_cell[cell_id]["artifact_path"],
            record=by_cell[cell_id],
            identity_binding=registry["identity_contract"],
        )
        score_rows = tuple(map(str, score["row_ids"].tolist()))
        values = np.asarray(score["score"], dtype=float)
        if score_rows != prepared.row_ids or values.shape != (len(prepared.row_ids),) or not np.isfinite(values).all():
            raise RuntimeError(f"{cell_id}/{method_id}: score/prepared cohort mismatch")
    return registry, freeze


def _public_registry_view(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: child
        for key, child in value.items()
        if key not in {"build_id", "payload_sha256"}
    }


def _private_view(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lane_id": value.get("lane_id"),
        "preparation_registry_sha256": value.get("preparation_registry_sha256"),
        "identity_contract": value.get("identity_contract"),
        "private_identity_contract": value.get("private_identity_contract"),
        "private_identity_contract_commitment_sha256": value.get(
            "private_identity_contract_commitment_sha256"
        ),
        "preparation_source_snapshot": value.get("preparation_source_snapshot"),
        "scientific_full_build": value.get("scientific_full_build"),
        "partial_descriptive_build": value.get("partial_descriptive_build"),
        "status_only_build": value.get("status_only_build"),
        "status_roster_contract_match": value.get("status_roster_contract_match"),
        "headline_eligible": value.get("headline_eligible"),
        "aggregate_metrics_allowed": value.get("aggregate_metrics_allowed"),
        "preparation_status_commitment_sha256": value.get(
            "preparation_status_commitment_sha256"
        ),
        "trace_status_contract_id": value.get("trace_status_contract_id"),
        "cells": value.get("cells"),
    }


def verify_ab(
    *,
    release_id: str,
    release_root: str | Path,
    private_control_root: str | Path,
    preparation_registry_path: str | Path,
    repo: str | Path,
) -> Mapping[str, Any]:
    release = Path(release_root) / release_id
    expected_registry_sha = sha256_file(preparation_registry_path)
    preparation_registry = load_preparation_registry(preparation_registry_path)
    audits: dict[str, dict[str, Any]] = {}
    for build in ("A", "B"):
        lane = release / f"build_{build}" / "edis"
        inputs, fit = lane / "inputs", lane / "fit"
        private_path = Path(private_control_root) / release_id / "edis" / f"build_{build}" / "PREPARATION_PROVENANCE.json"
        private_policy_path = private_path.parent / "FIT_AUDIT_POLICY.json"
        registry, freeze = validate_score_freeze(
            fit_root=fit,
            input_root=inputs,
            expected_build=build,
            repo=repo,
            private_audit_policy_path=private_policy_path,
        )
        private = load_private_provenance(private_path)
        preparation_status_path = lane / "PREPARATION_STATUS.json"
        preparation_status = load_preparation_status(preparation_status_path)
        assert_expected_preparation_status_roster(
            registry=preparation_registry, status=preparation_status
        )
        verify_current_source_snapshot(
            private.get("preparation_source_snapshot", {}),
            repo=Path(repo),
            name=f"EDIS preparation build {build}",
            require_clean=False,
            expected_paths=PREPARATION_SOURCE_PATHS,
        )
        if registry.get("release_id") != release_id or freeze.get("release_id") != release_id or private.get("release_id") != release_id:
            raise RuntimeError(f"build {build}: EDIS release binding failed")
        if registry.get("preparation_registry_sha256") != expected_registry_sha or private.get("preparation_registry_sha256") != expected_registry_sha:
            raise RuntimeError(f"build {build}: current EDIS registry binding failed")
        if private.get("identity_contract") != registry.get("identity_contract"):
            raise RuntimeError(f"build {build}: public/private identity binding differs")
        if private.get("private_identity_contract_commitment_sha256") != registry.get(
            "private_identity_contract_commitment_sha256"
        ):
            raise RuntimeError(
                f"build {build}: private identity-contract commitment differs"
            )
        if (
            preparation_status["status_commitment_sha256"]
            != registry["preparation_status_commitment_sha256"]
            or private.get("preparation_status_commitment_sha256")
            != registry["preparation_status_commitment_sha256"]
        ):
            raise RuntimeError(
                f"build {build}: preparation-status commitment differs"
            )
        audits[build] = {
            "registry": registry,
            "freeze": freeze,
            "private": private,
            "fit_registry_sha256": sha256_file(inputs / "FIT_REGISTRY.json"),
            "fit_registry_payload_sha256": registry["payload_sha256"],
            "score_freeze_sha256": sha256_file(fit / "SCORE_FREEZE_MANIFEST.json"),
            "score_freeze_payload_sha256": freeze["payload_sha256"],
            "private_provenance_sha256": sha256_file(private_path),
            "private_audit_policy_sha256": sha256_file(private_policy_path),
            "preparation_status": preparation_status,
            "preparation_status_file_sha256": sha256_file(preparation_status_path),
            "input_tree": canonical_tree_manifest(inputs),
            "fit_tree": canonical_tree_manifest(fit),
        }
    left, right = audits["A"], audits["B"]
    if _public_registry_view(left["registry"]) != _public_registry_view(right["registry"]):
        raise RuntimeError("EDIS A/B fit-safe registries differ beyond build identity")
    if _private_view(left["private"]) != _private_view(right["private"]):
        raise RuntimeError("EDIS A/B source or controller-only group commitments differ")
    left_status = {
        key: value
        for key, value in left["preparation_status"].items()
        if key not in {"build_id", "payload_sha256"}
    }
    right_status = {
        key: value
        for key, value in right["preparation_status"].items()
        if key not in {"build_id", "payload_sha256"}
    }
    if left_status != right_status:
        raise RuntimeError("EDIS A/B preparation cell-status rosters differ")
    comparisons: list[dict[str, Any]] = []
    for left_row, right_row in zip(left["freeze"]["records"], right["freeze"]["records"]):
        identity = (left_row["cell_id"], left_row["method_id"])
        if identity != (right_row["cell_id"], right_row["method_id"]):
            raise RuntimeError("EDIS A/B method record order differs")
        fields = (
            "method_version_id", "config_sha256", "status",
            "prepared_matrix_sha256", "row_roster_sha256", "score_sha256",
            "record_sha256", "artifacts_sha256", "artifact_index_sha256",
        )
        unequal = [field for field in fields if left_row.get(field) != right_row.get(field)]
        if unequal:
            raise RuntimeError(f"EDIS A/B outputs differ for {identity}: {unequal}")
        comparisons.append({
            "cell_id": identity[0],
            "method_id": identity[1],
            **{field: left_row.get(field) for field in fields},
        })
    certificate = {
        "schema_version": CERTIFICATE_SCHEMA,
        "release_id": release_id,
        "lane_id": left["registry"]["lane_id"],
        "status": "PASS",
        "scientific_full": left["registry"]["scientific_full_build"],
        "descriptive_partial": left["registry"]["partial_descriptive_build"],
        "headline_eligible": False,
        "aggregate_metrics_allowed": left["registry"]["aggregate_metrics_allowed"],
        "certificate_scope": (
            "FULL_READY_CELL_ROSTER"
            if left["registry"]["scientific_full_build"]
            else "DESCRIPTIVE_PARTIAL_READY_CELLS_ONLY"
        ),
        "identity_contract": left["registry"]["identity_contract"],
        "preparation_registry_sha256": expected_registry_sha,
        "preparation_status_commitment_sha256": left["registry"][
            "preparation_status_commitment_sha256"
        ],
        "method_ids": list(PRIMARY_METHOD_IDS),
        "cell_ids": list(left["freeze"]["cell_ids"]),
        "registered_cell_statuses": left["preparation_status"]["cells"],
        "registered_cell_count": left["preparation_status"]["registered_cell_count"],
        "ready_cell_count": left["preparation_status"]["ready_cell_count"],
        "blocked_cell_count": left["preparation_status"]["blocked_cell_count"],
        "n_method_comparisons": len(comparisons),
        "comparison_records": comparisons,
        "comparison_records_sha256": _payload_sha256(comparisons),
        "private_group_commitments_verified_equal": True,
        "private_identifiers_or_paths_serialized": False,
        "builds": {
            build: {
                key: value
                for key, value in audits[build].items()
                if key not in {"registry", "freeze", "private", "preparation_status"}
            }
            for build in ("A", "B")
        },
    }
    certificate["certificate_sha256"] = _payload_sha256(certificate)
    target = release / "edis" / "AB_VERIFICATION.json"
    atomic_write_json(target, certificate)
    return certificate


def assert_ab_certificate(
    *,
    path: str | Path,
    release_id: str,
    release_root: str | Path,
    selected_build: str,
    preparation_registry_path: str | Path,
    private_control_root: str | Path,
    repo: str | Path,
    allow_descriptive_partial: bool = False,
) -> dict[str, Any]:
    if selected_build not in {"A", "B"}:
        raise ValueError("selected_build must be A or B")
    certificate = json.loads(Path(path).read_text(encoding="utf-8"))
    _verify_payload(certificate, field="certificate_sha256", name="EDIS A/B certificate")
    if certificate.get("schema_version") != CERTIFICATE_SCHEMA or certificate.get("status") != "PASS":
        raise RuntimeError("a passing EDIS A/B certificate is required")
    scientific_full = certificate.get("scientific_full") is True
    descriptive_partial = certificate.get("descriptive_partial") is True
    if (
        certificate.get("release_id") != release_id
        or scientific_full == descriptive_partial
        or certificate.get("headline_eligible") is not False
        or certificate.get("aggregate_metrics_allowed") is not scientific_full
        or certificate.get("certificate_scope")
        != (
            "FULL_READY_CELL_ROSTER"
            if scientific_full
            else "DESCRIPTIVE_PARTIAL_READY_CELLS_ONLY"
        )
    ):
        raise RuntimeError("EDIS A/B certificate full/partial contract failed")
    if descriptive_partial and not allow_descriptive_partial:
        raise RuntimeError(
            "the current EDIS post-freeze evaluator is full-roster only; a "
            "descriptive partial certificate must not open blocked-cell labels"
        )
    if certificate.get("preparation_registry_sha256") != sha256_file(preparation_registry_path):
        raise RuntimeError("EDIS A/B certificate target-free registry is stale")
    preparation_registry = load_preparation_registry(preparation_registry_path)
    assert_expected_preparation_status_roster(
        registry=preparation_registry,
        status={"cells": certificate.get("registered_cell_statuses")},
    )
    cell_ids = tuple(map(str, certificate.get("cell_ids", ())))
    status_rows = certificate.get("registered_cell_statuses")
    if not isinstance(status_rows, list) or len(status_rows) != 12:
        raise RuntimeError("EDIS A/B certificate does not expose all 12 cell statuses")
    ready_ids = tuple(
        str(row.get("cell_id", ""))
        for row in status_rows
        if row.get("status") == "READY"
    )
    blocked = len(status_rows) - len(ready_ids)
    if (
        ready_ids != cell_ids
        or int(certificate.get("registered_cell_count", -1)) != 12
        or int(certificate.get("ready_cell_count", -1)) != len(ready_ids)
        or int(certificate.get("blocked_cell_count", -1)) != blocked
        or (scientific_full and blocked != 0)
        or (descriptive_partial and not 0 < len(ready_ids) < 12)
    ):
        raise RuntimeError("EDIS A/B certificate cell-status accounting failed")
    if tuple(certificate.get("method_ids", ())) != PRIMARY_METHOD_IDS:
        raise RuntimeError("EDIS A/B certificate method roster drifted")
    if int(certificate.get("n_method_comparisons", -1)) != len(cell_ids) * len(PRIMARY_METHOD_IDS):
        raise RuntimeError("EDIS A/B certificate comparison count failed")
    if certificate.get("comparison_records_sha256") != _payload_sha256(certificate.get("comparison_records", ())):
        raise RuntimeError("EDIS A/B certificate comparison commitment failed")
    release = Path(release_root) / release_id
    for build in ("A", "B"):
        lane = release / f"build_{build}" / "edis"
        expected = certificate["builds"][build]
        private_path = Path(private_control_root) / release_id / "edis" / f"build_{build}" / "PREPARATION_PROVENANCE.json"
        private_policy_path = private_path.parent / "FIT_AUDIT_POLICY.json"
        if expected["fit_registry_sha256"] != sha256_file(lane / "inputs" / "FIT_REGISTRY.json"):
            raise RuntimeError(f"build {build}: EDIS fit registry changed after certification")
        if expected["score_freeze_sha256"] != sha256_file(lane / "fit" / "SCORE_FREEZE_MANIFEST.json"):
            raise RuntimeError(f"build {build}: EDIS score freeze changed after certification")
        if expected["input_tree"]["tree_sha256"] != canonical_tree_manifest(lane / "inputs")["tree_sha256"]:
            raise RuntimeError(f"build {build}: EDIS input tree changed after certification")
        if expected["fit_tree"]["tree_sha256"] != canonical_tree_manifest(lane / "fit")["tree_sha256"]:
            raise RuntimeError(f"build {build}: EDIS fit tree changed after certification")
        if expected["private_provenance_sha256"] != sha256_file(private_path):
            raise RuntimeError(f"build {build}: EDIS private provenance changed after certification")
        if expected["private_audit_policy_sha256"] != sha256_file(private_policy_path):
            raise RuntimeError(f"build {build}: EDIS private audit policy changed after certification")
        preparation_status_path = lane / "PREPARATION_STATUS.json"
        if expected["preparation_status_file_sha256"] != sha256_file(
            preparation_status_path
        ):
            raise RuntimeError(
                f"build {build}: EDIS preparation status changed after certification"
            )
        preparation_status = load_preparation_status(preparation_status_path)
        if preparation_status["status_commitment_sha256"] != certificate.get(
            "preparation_status_commitment_sha256"
        ):
            raise RuntimeError(
                f"build {build}: EDIS preparation status commitment drifted"
            )
        validate_score_freeze(
            fit_root=lane / "fit",
            input_root=lane / "inputs",
            expected_build=build,
            repo=repo,
            private_audit_policy_path=private_policy_path,
        )
        private = load_private_provenance(private_path)
        verify_current_source_snapshot(
            private.get("preparation_source_snapshot", {}),
            repo=Path(repo),
            name=f"EDIS preparation build {build}",
            require_clean=False,
            expected_paths=PREPARATION_SOURCE_PATHS,
        )
    return certificate


def _verify_evaluation_build(
    *,
    release: Path,
    release_id: str,
    build_id: str,
    score_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every persisted post-freeze table and label-artifact binding."""

    from spectral_utils.reconstruction_reporting.io import read_parquet, read_tidy_csv
    from spectral_utils.reconstruction_reporting.schemas import table_sha256

    output = release / f"build_{build_id}" / "edis" / "evaluation"
    manifest_path = output / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_payload(
        manifest, field="payload_sha256", name=f"EDIS evaluation build {build_id}"
    )
    descriptive_partial = score_certificate.get("descriptive_partial") is True
    required = {
        "schema_version": EVALUATION_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "lane_id": "edis_aime_reconstruction_v1",
        "ab_certificate_sha256": score_certificate["certificate_sha256"],
        "labels_opened_only_after_score_freeze_and_ab_pass": True,
        "historical_scores_copied": False,
        "bootstrap_draws": 20_000,
        "bootstrap_unit": "source_question",
        "positive_class": "incorrect",
        "metrics": ["auroc", "auprc"],
        "preparation_status_commitment_sha256": score_certificate[
            "preparation_status_commitment_sha256"
        ],
        "scientific_full": not descriptive_partial,
        "descriptive_partial": descriptive_partial,
        "partial_evaluation_mode": descriptive_partial,
        "registered_cell_count": 12,
        "ready_cell_count": len(score_certificate["cell_ids"]),
        "blocked_cell_count": 12 - len(score_certificate["cell_ids"]),
        "registered_cell_statuses": score_certificate["registered_cell_statuses"],
        "raw_row_labels_opened_cell_ids": score_certificate["cell_ids"],
        "blocked_raw_row_label_artifacts_opened": False,
        "postfreeze_target_summary_registry_loaded": True,
        "aggregation": (
            "per_temperature_only; dataset_and_task_aggregates_forbidden_in_partial_release"
            if descriptive_partial
            else (
                "per_temperature; equal_temperature_per_dataset; "
                "equal_dataset_after_equal_temperature"
            )
        ),
        "dataset_task_aggregates_emitted": not descriptive_partial,
        "track": "multi_sample_inference",
        "access_contract_id": "gray_box_multi_pass",
        "headline_eligible": False,
        "publication_eligible": False,
        "evidence_status": (
            "DESCRIPTIVE_PARTIAL_GATE_FAILED"
            if descriptive_partial else "DESCRIPTIVE_GATE_FAILED"
        ),
        "one_pass_leaderboard_combination_forbidden": True,
        "evaluation_ab_certificate_required_for_release": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"EDIS evaluation build {build_id} {key} contract drifted"
            )
    if manifest.get("score_freeze_sha256") != score_certificate["builds"][build_id][
        "score_freeze_sha256"
    ]:
        raise RuntimeError(
            f"EDIS evaluation build {build_id} is bound to another score freeze"
        )
    artifacts = manifest.get("artifacts")
    canonical_hashes = manifest.get("canonical_table_sha256")
    table_names = ("predictions", "metrics", "contrasts", "coverage")
    if set(artifacts or {}) != set(table_names) or set(canonical_hashes or {}) != set(
        table_names
    ):
        raise RuntimeError(f"EDIS evaluation build {build_id} table roster drifted")
    table_audits: dict[str, Any] = {}
    table_rows: dict[str, list[dict[str, Any]]] = {}
    for table in table_names:
        records = artifacts[table]
        if set(records or {}) != {"csv", "parquet"}:
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{table} artifact roster drifted"
            )
        csv_record, parquet_record = records["csv"], records["parquet"]
        csv_path = _safe_child(output, str(csv_record.get("path", "")))
        parquet_path = _safe_child(output, str(parquet_record.get("path", "")))
        if sha256_file(csv_path) != csv_record.get("file_sha256"):
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{table} CSV hash failed"
            )
        if sha256_file(parquet_path) != parquet_record.get("file_sha256"):
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{table} Parquet hash failed"
            )
        csv_rows = read_tidy_csv(csv_path, table)
        parquet_rows = read_parquet(parquet_path, table)
        csv_logical = table_sha256(table, csv_rows)
        parquet_logical = table_sha256(table, parquet_rows)
        if (
            csv_logical != csv_record.get("logical_sha256")
            or parquet_logical != parquet_record.get("logical_sha256")
            or csv_logical != parquet_logical
            or len(csv_rows) != int(csv_record.get("row_count", -1))
            or len(parquet_rows) != int(parquet_record.get("row_count", -1))
        ):
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{table} logical artifact binding failed"
            )
        canonical = canonical_evaluation_table_sha256(
            table=table,
            rows=csv_rows,
            release_id=release_id,
            build_id=build_id,
        )
        if canonical != canonical_hashes[table]:
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{table} canonical hash failed"
            )
        table_audits[table] = {
            "row_count": len(csv_rows),
            "canonical_table_sha256": canonical,
            "csv_file_sha256": csv_record["file_sha256"],
            "parquet_file_sha256": parquet_record["file_sha256"],
        }
        table_rows[table] = csv_rows
    _validate_evaluation_table_rosters(
        table_rows=table_rows,
        score_certificate=score_certificate,
        descriptive_partial=descriptive_partial,
    )
    label_rows = manifest.get("label_artifacts")
    if not isinstance(label_rows, list) or len(label_rows) != len(
        score_certificate["cell_ids"]
    ):
        raise RuntimeError(f"EDIS evaluation build {build_id} label roster drifted")
    expected_cells = list(score_certificate["cell_ids"])
    if [str(row.get("cell_id", "")) for row in label_rows] != expected_cells:
        raise RuntimeError(
            f"EDIS evaluation build {build_id} label cell order drifted"
        )
    label_audits: list[dict[str, str]] = []
    label_bindings: dict[str, dict[str, tuple[str, int]]] = {}
    expected_rows_by_cell = {
        str(row["cell_id"]): int(row["expected_rows"])
        for row in score_certificate["registered_cell_statuses"]
    }
    for row in label_rows:
        path = _safe_child(output, str(row.get("artifact_path", "")))
        if sha256_file(path) != row.get("artifact_sha256"):
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{row.get('cell_id')} label hash failed"
            )
        arrays = load_npz_no_pickle(path)
        if set(arrays) != {"row_ids", "group_ids", "incorrect"}:
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{row.get('cell_id')} label members drifted"
            )
        row_ids = tuple(map(str, arrays["row_ids"].tolist()))
        group_ids = tuple(map(str, arrays["group_ids"].tolist()))
        labels = np.asarray(arrays["incorrect"], dtype=np.int8)
        if (
            not row_ids
            or len(row_ids) != expected_rows_by_cell[str(row["cell_id"])]
            or len(set(row_ids)) != len(row_ids)
            or row_ids != tuple(sorted(row_ids))
            or len(group_ids) != len(row_ids)
            or labels.shape != (len(row_ids),)
            or not set(labels.tolist()).issubset({0, 1})
        ):
            raise RuntimeError(
                f"EDIS evaluation build {build_id}/{row.get('cell_id')} label cohort is invalid"
            )
        label_audits.append(
            {
                "cell_id": str(row["cell_id"]),
                "artifact_sha256": str(row["artifact_sha256"]),
            }
        )
        label_bindings[str(row["cell_id"])] = {
            row_id: (group_id, int(label))
            for row_id, group_id, label in zip(row_ids, group_ids, labels.tolist())
        }
    _validate_prediction_label_bindings(
        predictions=table_rows["predictions"],
        label_bindings=label_bindings,
        ready_cell_ids=list(map(str, score_certificate["cell_ids"])),
    )
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "table_audits": table_audits,
        "label_audits": label_audits,
        "output_tree": canonical_tree_manifest(output),
    }


def _validate_evaluation_table_rosters(
    *,
    table_rows: Mapping[str, list[Mapping[str, Any]]],
    score_certificate: Mapping[str, Any],
    descriptive_partial: bool,
) -> None:
    """Reject A/B-identical but incomplete EDIS reporting tables."""

    registered = [
        str(row["cell_id"])
        for row in score_certificate["registered_cell_statuses"]
    ]
    status_by_cell = {
        str(row["cell_id"]): row
        for row in score_certificate["registered_cell_statuses"]
    }
    if any(int(status_by_cell[cell].get("expected_rows", -1)) <= 0 for cell in registered):
        raise RuntimeError("EDIS certificate cell status lacks expected_rows")
    ready = list(map(str, score_certificate["cell_ids"]))
    blocked = [cell for cell in registered if cell not in set(ready)]
    methods = list(PRIMARY_METHOD_IDS)
    expected_coverage = {(cell, method) for cell in registered for method in methods}
    coverage_rows = table_rows["coverage"]
    coverage_pairs = [
        (str(row["cell_id"]), str(row["method_id"]))
        for row in coverage_rows
    ]
    if len(coverage_pairs) != len(expected_coverage) or set(coverage_pairs) != expected_coverage:
        raise RuntimeError("EDIS evaluation coverage is not the unique complete 12x13 roster")
    coverage_by_pair = {
        (str(row["cell_id"]), str(row["method_id"])): row
        for row in coverage_rows
    }
    for cell in blocked:
        for method in methods:
            row = coverage_by_pair[(cell, method)]
            expected_n = int(status_by_cell[cell]["expected_rows"])
            if (
                row["status"] != "INPUT_INVALID"
                or int(row["expected_n"]) != expected_n
                or int(row["eligible_n"]) != 0
                or int(row["scored_n"]) != 0
                or int(row["failed_n"]) != expected_n
                or float(row["coverage_fraction"]) != 0.0
            ):
                raise RuntimeError("EDIS blocked-cell coverage contract drifted")
    for cell in ready:
        expected_n = int(status_by_cell[cell]["expected_rows"])
        for method in methods:
            row = coverage_by_pair[(cell, method)]
            if (
                int(row["expected_n"]) != expected_n
                or int(row["eligible_n"]) != expected_n
                or int(row["scored_n"]) != expected_n
                or int(row["excluded_n"]) != 0
                or int(row["failed_n"]) != 0
                or float(row["coverage_fraction"]) != 1.0
            ):
                raise RuntimeError("EDIS ready-cell coverage differs from expected_rows")

    expected_ready_pairs = {(cell, method) for cell in ready for method in methods}
    prediction_pairs = [
        (str(row["cell_id"]), str(row["method_id"]))
        for row in table_rows["predictions"]
    ]
    if set(prediction_pairs) != expected_ready_pairs:
        raise RuntimeError("EDIS predictions do not cover every ready cell/method")
    prediction_counts: dict[tuple[str, str], int] = {}
    for pair in prediction_pairs:
        prediction_counts[pair] = prediction_counts.get(pair, 0) + 1
    if any(
        prediction_counts[pair] != int(coverage_by_pair[pair]["scored_n"])
        for pair in expected_ready_pairs
    ):
        raise RuntimeError("EDIS prediction counts differ from ready-cell coverage")
    for pair in expected_ready_pairs:
        row_ids = [
            str(row.get("row_id", ""))
            for row in table_rows["predictions"]
            if (str(row["cell_id"]), str(row["method_id"])) == pair
        ]
        if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
            raise RuntimeError("EDIS prediction row IDs are empty or duplicated")

    if descriptive_partial:
        expected_metrics = {
            (cell, method, metric)
            for cell in ready for method in methods for metric in ("auroc", "auprc")
        }
        observed_metrics = [
            (str(row["cell_id"]), str(row["method_id"]), str(row["metric_id"]))
            for row in table_rows["metrics"]
        ]
        if (
            len(observed_metrics) != len(expected_metrics)
            or set(observed_metrics) != expected_metrics
            or any(row["aggregation_level"] != "cell" for row in table_rows["metrics"])
        ):
            raise RuntimeError("EDIS partial metrics Cartesian roster is incomplete")
        candidate_methods = [method for method in methods if method != "iu_pcr"]
        expected_contrasts = {
            (cell, method, metric)
            for cell in ready
            for method in candidate_methods
            for metric in ("auroc", "auprc")
        }
        observed_contrasts = [
            (str(row["cell_id"]), str(row["method_id"]), str(row["metric_id"]))
            for row in table_rows["contrasts"]
        ]
        if (
            len(observed_contrasts) != len(expected_contrasts)
            or set(observed_contrasts) != expected_contrasts
            or any(row["aggregation_level"] != "cell" for row in table_rows["contrasts"])
        ):
            raise RuntimeError("EDIS partial contrasts Cartesian roster is incomplete")


def _validate_prediction_label_bindings(
    *,
    predictions: list[Mapping[str, Any]],
    label_bindings: Mapping[str, Mapping[str, tuple[str, int]]],
    ready_cell_ids: list[str],
) -> None:
    """Bind predictions to label rows independent of persisted sort order."""

    for cell_id in ready_cell_ids:
        expected = label_bindings.get(cell_id)
        if not expected:
            raise RuntimeError(f"{cell_id}: EDIS label binding is absent")
        expected_rows = set(expected)
        for method_id in PRIMARY_METHOD_IDS:
            rows = [
                row for row in predictions
                if str(row["cell_id"]) == cell_id
                and str(row["method_id"]) == method_id
            ]
            observed_ids = [str(row["row_id"]) for row in rows]
            if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_rows:
                raise RuntimeError(
                    f"{cell_id}/{method_id}: prediction/label row roster differs"
                )
            for row in rows:
                row_id = str(row["row_id"])
                expected_group, expected_label = expected[row_id]
                if (
                    str(row["group_id"]) != expected_group
                    or int(row["label"]) != expected_label
                ):
                    raise RuntimeError(
                        f"{cell_id}/{method_id}: prediction group/label binding differs"
                    )


def verify_evaluation_ab(
    *,
    release_id: str,
    release_root: str | Path,
    private_control_root: str | Path,
    preparation_registry_path: str | Path,
    repo: str | Path,
    score_certificate_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Certify A/B identity of all post-label EDIS metrics and report tables."""

    release = Path(release_root) / release_id
    score_path = (
        Path(score_certificate_path)
        if score_certificate_path is not None
        else release / "edis" / "AB_VERIFICATION.json"
    )
    score_certificate = assert_ab_certificate(
        path=score_path,
        release_id=release_id,
        release_root=release_root,
        selected_build="A",
        preparation_registry_path=preparation_registry_path,
        private_control_root=private_control_root,
        repo=repo,
        allow_descriptive_partial=True,
    )
    audits = {
        build: _verify_evaluation_build(
            release=release,
            release_id=release_id,
            build_id=build,
            score_certificate=score_certificate,
        )
        for build in ("A", "B")
    }
    left, right = audits["A"], audits["B"]
    if left["table_audits"] != right["table_audits"]:
        differing = [
            table
            for table in ("predictions", "metrics", "contrasts", "coverage")
            if left["table_audits"][table]["canonical_table_sha256"]
            != right["table_audits"][table]["canonical_table_sha256"]
        ]
        if differing:
            raise RuntimeError(f"EDIS A/B post-freeze tables differ: {differing}")
        # File hashes are expected to differ because the persisted run_id is
        # build-specific; only row counts and canonical hashes are comparable.
        for table in ("predictions", "metrics", "contrasts", "coverage"):
            for field in ("row_count", "canonical_table_sha256"):
                if left["table_audits"][table][field] != right["table_audits"][table][field]:
                    raise RuntimeError(
                        f"EDIS A/B post-freeze {table} {field} differs"
                    )
    if left["label_audits"] != right["label_audits"]:
        raise RuntimeError("EDIS A/B post-freeze label/group artifacts differ")
    comparison_fields = (
        "lane_id",
        "preparation_registry_sha256",
        "postfreeze_registry_sha256",
        "identity_contract",
        "bootstrap_draws",
        "bootstrap_unit",
        "positive_class",
        "metrics",
        "aggregation",
        "track",
        "access_contract_id",
        "headline_eligible",
        "publication_eligible",
        "scientific_full",
        "descriptive_partial",
        "partial_evaluation_mode",
        "registered_cell_count",
        "ready_cell_count",
        "blocked_cell_count",
        "registered_cell_statuses",
        "raw_row_labels_opened_cell_ids",
        "blocked_raw_row_label_artifacts_opened",
        "postfreeze_target_summary_registry_loaded",
        "dataset_task_aggregates_emitted",
        "preparation_status_commitment_sha256",
        "evidence_status",
        "gate_audit",
        "one_pass_leaderboard_combination_forbidden",
    )
    for field in comparison_fields:
        if left["manifest"].get(field) != right["manifest"].get(field):
            raise RuntimeError(f"EDIS A/B evaluation manifest {field} differs")
    canonical_tables = {
        table: left["table_audits"][table]["canonical_table_sha256"]
        for table in ("predictions", "metrics", "contrasts", "coverage")
    }
    certificate = {
        "schema_version": EVALUATION_CERTIFICATE_SCHEMA,
        "release_id": release_id,
        "lane_id": left["manifest"]["lane_id"],
        "status": "PASS",
        "score_ab_certificate_sha256": score_certificate["certificate_sha256"],
        "canonical_table_sha256": canonical_tables,
        "label_artifacts": left["label_audits"],
        "bootstrap_draws": 20_000,
        "bootstrap_unit": "source_question",
        "gate_audit": left["manifest"]["gate_audit"],
        "headline_eligible": False,
        "publication_eligible": False,
        "scientific_full": score_certificate["scientific_full"],
        "descriptive_partial": score_certificate["descriptive_partial"],
        "partial_evaluation_mode": score_certificate["descriptive_partial"],
        "evidence_status": left["manifest"]["evidence_status"],
        "builds": {
            build: {
                "manifest_sha256": audits[build]["manifest_sha256"],
                "manifest_payload_sha256": audits[build]["manifest_payload_sha256"],
                "output_tree": audits[build]["output_tree"],
            }
            for build in ("A", "B")
        },
    }
    certificate["certificate_sha256"] = _payload_sha256(certificate)
    target = release / "edis" / "EVALUATION_AB_VERIFICATION.json"
    atomic_write_json(target, certificate)
    return certificate


__all__ = [
    "CERTIFICATE_SCHEMA",
    "EVALUATION_CERTIFICATE_SCHEMA",
    "FIT_SOURCE_PATHS",
    "assert_ab_certificate",
    "canonical_evaluation_table_sha256",
    "load_private_provenance",
    "validate_score_freeze",
    "verify_current_source_snapshot",
    "verify_ab",
    "verify_evaluation_ab",
]
