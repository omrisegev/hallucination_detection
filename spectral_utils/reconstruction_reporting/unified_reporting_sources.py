"""Authenticate reviewed, content-addressed sources for unified reporting.

Authentication stops at producer-certified post-label evaluation artifacts.  It
does not invoke producer rederivation code and it never walks a release tree or
opens a file that is absent from the reviewed source lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

from .unified_reporting_schemas import (
    UnifiedReportingError,
    canonical_sha256,
)


SOURCE_LOCK_SCHEMA = "reconstruction-unified-reporting-source-lock-v1"
CONTRACT_SCHEMA = "reconstruction-unified-reporting-contract-v1"


@dataclass(frozen=True)
class AuthenticatedSource:
    source_id: str
    source_release_id: str
    source_binding_id: str
    certified: bool
    source_status: str
    source_root_id: str | None
    logical_binding_sha256: str
    lock_record: Mapping[str, Any]
    certificate: Mapping[str, Any] | None
    manifest: Mapping[str, Any] | None
    files: Mapping[str, bytes]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise UnifiedReportingError(f"{where} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise UnifiedReportingError(f"{where} must be a lowercase SHA-256") from exc
    if value != value.lower():
        raise UnifiedReportingError(f"{where} must be lowercase")
    return value


def _safe_relative(value: Any, *, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise UnifiedReportingError(f"{where} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnifiedReportingError(f"{where} is unsafe: {value!r}")
    return path


def read_locked_file(root: str | Path, relative: str, expected_sha256: str) -> bytes:
    """Read one locked regular file without following any path-component symlink."""

    expected_sha256 = _validate_sha(expected_sha256, where="expected file hash")
    parts = _safe_relative(relative, where="locked file path").parts
    root_path = Path(root).resolve(strict=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(root_path, directory_flags)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise UnifiedReportingError(f"locked source is not a regular file: {relative}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise UnifiedReportingError(f"locked source changed while being read: {relative}")
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise UnifiedReportingError(
                f"locked source hash mismatch for {relative}: expected={expected_sha256}, observed={observed}"
            )
        return b"".join(chunks)
    except OSError as exc:
        raise UnifiedReportingError(f"cannot safely open locked source {relative}: {exc}") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _json(payload: bytes, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnifiedReportingError(f"{where} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise UnifiedReportingError(f"{where} must be a JSON object")
    return value


def _schema(value: Mapping[str, Any]) -> Any:
    return value.get("schema_version", value.get("schema"))


def _verify_json_binding(value: Mapping[str, Any], binding: Mapping[str, Any], *, where: str) -> None:
    if _schema(value) != binding.get("schema_version"):
        raise UnifiedReportingError(f"{where} schema drift")
    field = binding.get("self_hash_field")
    expected = _validate_sha(binding.get("self_hash"), where=f"{where} locked self hash")
    if not isinstance(field, str) or value.get(field) != expected:
        raise UnifiedReportingError(f"{where} self-hash field drift")
    body = dict(value)
    observed = body.pop(field, None)
    if observed != canonical_sha256(body):
        raise UnifiedReportingError(f"{where} canonical self hash is invalid")
    if "status_field" in binding:
        if value.get(binding["status_field"]) != binding.get("status_value"):
            raise UnifiedReportingError(f"{where} is not in the reviewed passing state")


def _artifact_hash_from_list(manifest: Mapping[str, Any], basename: str) -> str | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for item in artifacts:
        if isinstance(item, Mapping) and item.get("path") == basename:
            return item.get("sha256")
    return None


def _verify_chain(source_id: str, record: Mapping[str, Any], certificate: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> None:
    files = record.get("files")
    if not isinstance(files, Mapping):
        raise UnifiedReportingError(f"{source_id} files lock is missing")
    expected = {name: item["file_sha256"] for name, item in files.items()}
    if source_id == "frozen24":
        by_name = {
            item.get("path"): item.get("file_sha256")
            for item in certificate.get("artifacts", [])
            if isinstance(item, Mapping)
        }
        for name, filename in (
            ("metrics", "metrics_long.csv"), ("contrasts", "contrasts_long.csv"),
            ("coverage", "coverage_long.csv"),
        ):
            if by_name.get(filename) != expected[name]:
                raise UnifiedReportingError(f"frozen24 certificate does not bind {filename}")
        if certificate.get("scientific_publication_eligible") is not True:
            raise UnifiedReportingError("frozen24 bridge is not publication eligible")
        return
    if manifest is None:
        raise UnifiedReportingError(f"{source_id} requires a Build A manifest")
    manifest_sha = record["manifest"]["file_sha256"]
    if source_id == "external_v3":
        if certificate.get("builds", {}).get("A", {}).get("evaluation_manifest_file_sha256") != manifest_sha:
            raise UnifiedReportingError("external certificate does not bind Build A manifest")
        identity = certificate.get("byte_identity", {})
        if identity.get("metrics_long.csv") != expected["metrics"] or identity.get("contrasts_long.csv") != expected["contrasts"]:
            raise UnifiedReportingError("external certificate reporting table drift")
        if manifest.get("metrics_sha256") != expected["metrics"] or manifest.get("contrasts_sha256") != expected["contrasts"]:
            raise UnifiedReportingError("external manifest reporting table drift")
        if not isinstance(manifest.get("population_checks"), list):
            raise UnifiedReportingError("external manifest population_checks are missing")
        return
    if source_id == "edis_v2":
        build = certificate.get("builds", {}).get("A", {})
        if build.get("manifest_sha256") != manifest_sha:
            raise UnifiedReportingError("EDIS certificate does not bind Build A manifest")
        tree = {
            item.get("path"): item.get("sha256")
            for item in build.get("output_tree", {}).get("files", [])
            if isinstance(item, Mapping)
        }
        paths = {"metrics": "metrics_long.csv", "contrasts": "contrasts_long.csv", "coverage": "coverage_long.csv"}
        for name, path in paths.items():
            artifact = manifest.get("artifacts", {}).get(name, {}).get("csv", {})
            if tree.get(path) != expected[name] or artifact.get("file_sha256") != expected[name]:
                raise UnifiedReportingError(f"EDIS chain does not bind {path}")
        return
    if source_id == "localization_v1":
        if certificate.get("builds", {}).get("A", {}).get("manifest_file_sha256") != manifest_sha:
            raise UnifiedReportingError("localization certificate does not bind Build A manifest")
        paths = {
            "metrics": "metrics_long.csv", "contrasts": "contrasts_long.csv",
            "coverage": "coverage_long.csv", "localization_decisions": "localization_decisions.csv",
        }
        for name, path in paths.items():
            if _artifact_hash_from_list(manifest, path) != expected[name]:
                raise UnifiedReportingError(f"localization manifest does not bind {path}")
        return
    if source_id == "prefix_v1":
        paths = {"metrics": "METRICS.json", "contrasts": "CONTRASTS.json"}
        for name, path in paths.items():
            if certificate.get("artifacts", {}).get(path) != expected[name]:
                raise UnifiedReportingError(f"prefix certificate does not bind {path}")
            if manifest.get("artifacts", {}).get(path) != expected[name]:
                raise UnifiedReportingError(f"prefix manifest does not bind {path}")
        if certificate.get("cross_task_macro_allowed") is not False or certificate.get("cross_budget_macro_allowed") is not False:
            raise UnifiedReportingError("prefix macro boundary drift")
        if certificate.get("stopping_claim_allowed") is not False:
            raise UnifiedReportingError("prefix stopping boundary drift")
        return
    if source_id == "leash_stopping":
        required_files = {
            "aggregate_metrics": "aggregate_metrics.csv",
            "bootstrap_intervals": "bootstrap_intervals.csv",
            "cell_metrics": "cell_metrics.csv",
            "contrasts": "contrasts.csv",
            "coverage": "coverage.csv",
            "frontier": "frontier.csv",
        }
        if set(expected) != set(required_files):
            raise UnifiedReportingError(
                "LEASH source lock must contain only certified aggregate reporting tables"
            )
        if (
            certificate.get("schema_version")
            != "reconstruction-leash-stopping-evaluation-ab-v1"
            or manifest.get("schema_version")
            != "reconstruction-leash-stopping-evaluation-v1"
        ):
            raise UnifiedReportingError("LEASH certificate/manifest schema drift")
        if manifest.get("lane_id") != "leash_actual_stopping_v1":
            raise UnifiedReportingError("LEASH manifest lane drift")
        if certificate.get("lane_id") != manifest.get("lane_id"):
            raise UnifiedReportingError("LEASH certificate/manifest lane mismatch")
        tree_hashes = certificate.get("evaluation_tree_sha256", {})
        if (
            not isinstance(tree_hashes, Mapping)
            or tree_hashes.get("A") != tree_hashes.get("B")
            or tree_hashes.get("A") != certificate.get("rederived_evaluation_tree_sha256")
            or certificate.get("status") != "PASS"
            or certificate.get("byte_identical") is not True
            or certificate.get("transitive_rederivation") is not True
            or certificate.get("grouped_bootstrap_rederived") is not True
            or certificate.get("private_outcomes_reparsed") is not True
            or certificate.get("searchable_output_contract_verified") is not True
        ):
            raise UnifiedReportingError("LEASH evaluation certificate is not a passing rederivation")
        if (
            certificate.get("paper_exact_claim") is not False
            or certificate.get("conceptual_objective_reproduced_as_equation") is not False
            or certificate.get("matched_accuracy_claim") is not False
            or manifest.get("paper_exact_claim") is not False
            or manifest.get("conceptual_objective_reproduced_as_equation") is not False
            or manifest.get("matched_accuracy_claim") is not False
            or manifest.get("cross_task_or_access_macro") is not False
            or manifest.get("proxy_stopping") is not False
            or manifest.get("policy_execution_evaluated") is not True
            or manifest.get("all_policy_stops_have_realized_closure") is not True
            or manifest.get("fidelity") != "paper-specified-partial"
        ):
            raise UnifiedReportingError("LEASH scientific claim boundary drift")
        if manifest.get("bootstrap") != {
            "draws": 2000,
            "paired_across_arms_and_model_copies": True,
            "seed": 2026082406,
            "stratification": "within dataset",
            "unit": "source question",
        }:
            raise UnifiedReportingError("LEASH grouped-bootstrap manifest drift")
        for field in (
            "registry_sha256", "preparation_ab_certificate_sha256",
            "fit_ab_certificate_sha256",
        ):
            if certificate.get(field) != manifest.get(field):
                raise UnifiedReportingError(f"LEASH certificate/manifest {field} mismatch")
        manifest_tables = manifest.get("tables")
        row_counts = certificate.get("rows_by_table")
        if not isinstance(manifest_tables, Mapping) or not isinstance(row_counts, Mapping):
            raise UnifiedReportingError("LEASH table contracts are missing")
        expected_roster = {
            "aggregate_metrics", "bootstrap_intervals", "cell_metrics", "contrasts",
            "coverage", "frontier", "per_question",
        }
        if set(manifest_tables) != expected_roster or set(row_counts) != expected_roster:
            raise UnifiedReportingError("LEASH certified table roster drift")
        for role, filename in required_files.items():
            table = manifest_tables.get(role)
            if not isinstance(table, Mapping):
                raise UnifiedReportingError(f"LEASH manifest omits {role}")
            if (
                table.get("files", {}).get(filename) != expected[role]
                or table.get("row_count") != row_counts.get(role)
            ):
                raise UnifiedReportingError(f"LEASH chain does not bind {filename}")
        return
    if source_id == "rag_evidence":
        required_files = {
            "metrics": "metrics.csv",
            "contrasts": "contrasts.csv",
            "panel_status": "panel_status.csv",
        }
        if set(expected) != set(required_files):
            raise UnifiedReportingError(
                "RAG source lock must contain only aggregate/status reporting tables"
            )
        if (
            certificate.get("schema_version")
            != "reconstruction-rag-evidence-evaluation-ab-v1"
            or manifest.get("schema_version")
            != "reconstruction-rag-evidence-evaluation-v1"
        ):
            raise UnifiedReportingError("RAG certificate/manifest schema drift")
        if (
            certificate.get("status") != "PASS"
            or certificate.get("scientific_full_required") is not True
            or certificate.get("transitive_source_rederivation") is not True
            or certificate.get("independent_postfreeze_reevaluation") is not True
            or certificate.get("cross_panel_macro_computed") is not False
            or certificate.get("refchecker_settings_pooled") is not False
        ):
            raise UnifiedReportingError("RAG evaluation certificate is not a passing rederivation")
        comparisons = certificate.get("comparisons")
        comparison_files = (
            "metrics.csv", "predictions.csv", "contrasts.csv", "panel_status.csv",
        )
        expected_comparisons = {
            *comparison_files,
            "source_snapshot_identity", "bootstrap_identity", "panel_status_identity",
            "independent_panel_status_matches_A",
            "independent_panel_status_matches_B",
            *{
                f"independent_{name}_matches_{build_id}"
                for name in comparison_files
                for build_id in ("A", "B")
            },
        }
        if (
            not isinstance(comparisons, Mapping)
            or set(comparisons) != expected_comparisons
            or not all(value is True for value in comparisons.values())
        ):
            raise UnifiedReportingError("RAG evaluation A/B comparison gate failed")
        build_a = certificate.get("builds", {}).get("A", {})
        if (
            build_a.get("evaluation_manifest_sha256") != manifest_sha
            or build_a.get("evaluation_manifest_payload_sha256")
            != record["manifest"].get("self_hash")
        ):
            raise UnifiedReportingError("RAG certificate does not bind Build A manifest")
        if (
            manifest.get("build_id") != "A"
            or manifest.get("lane_id") != "rag_evidence_benchmark_v1"
            or certificate.get("score_ab_sha256")
            != manifest.get("score_ab_certificate_sha256")
            or any(
                certificate.get(field) != manifest.get(field)
                for field in (
                    "score_sha256", "private_label_sha256",
                    "source_binding_sha256", "evaluation_repo_snapshot",
                    "score_verifier_repo_snapshot", "isolated_score_authentication",
                    "shared_repository_contract",
                )
            )
        ):
            raise UnifiedReportingError("RAG certificate/manifest provenance mismatch")
        for field in (
            "score_ab_sha256", "score_sha256", "private_label_sha256",
            "source_binding_sha256",
        ):
            _validate_sha(certificate.get(field), where=f"RAG certificate {field}")
        if (
            manifest.get("scientific_full") is not True
            or manifest.get("cross_panel_macro_computed") is not False
            or manifest.get("refchecker_settings_pooled") is not False
            or manifest.get("historical_scores_copied") is not False
        ):
            raise UnifiedReportingError("RAG manifest scientific boundary drift")
        bootstrap = manifest.get("bootstrap", {})
        if bootstrap != {
            "draws_requested": 20000,
            "group": "panel-registered source group",
            "paired_contrasts": True,
            "seed": 2026082407,
        }:
            raise UnifiedReportingError("RAG grouped-bootstrap contract drift")
        reporting = certificate.get("reporting_files")
        descriptors = manifest.get("files")
        if not isinstance(reporting, Mapping) or set(reporting) != {
            "metrics.csv", "predictions.csv", "contrasts.csv", "panel_status.csv",
        }:
            raise UnifiedReportingError("RAG certificate reporting-file roster drift")
        if not isinstance(descriptors, list) or len(descriptors) != 4:
            raise UnifiedReportingError("RAG manifest reporting-file roster drift")
        by_path = {
            item.get("path"): item
            for item in descriptors
            if isinstance(item, Mapping)
        }
        if set(by_path) != set(reporting):
            raise UnifiedReportingError("RAG manifest reporting-file descriptors drift")
        for path, digest in reporting.items():
            if by_path[path].get("sha256") != digest:
                raise UnifiedReportingError(f"RAG manifest/certificate mismatch for {path}")
        for role, filename in required_files.items():
            if reporting.get(filename) != expected[role]:
                raise UnifiedReportingError(f"RAG chain does not bind {filename}")
        panel_status = manifest.get("panel_status")
        expected_panels = (
            "ragtruth_evidence_contrast_answer",
            "ragtruth_evidence_contrast_sentence",
            "ragtruth_evidence_contrast_token",
            "gasp_protocol_sentence",
            "lettucedetect_example",
            "refchecker_threeway",
            "refchecker_binary_claim",
        )
        if (
            not isinstance(panel_status, list)
            or tuple(row.get("panel_id") for row in panel_status if isinstance(row, Mapping))
            != expected_panels
            or any(
                not isinstance(row, Mapping)
                or row.get("status") != "PASS"
                or row.get("cross_panel_macro_contribution") != "FORBIDDEN"
                for row in panel_status
            )
        ):
            raise UnifiedReportingError("RAG manifest panel-status boundary drift")
        return
    if source_id in {"winner_frozen24", "winner_external_v3"}:
        build = certificate.get("builds", {}).get("A", {})
        if build.get("manifest_file_sha256") != manifest_sha:
            raise UnifiedReportingError("winner certificate does not bind Build A manifest")
        paths = {
            "winner_reference_sets": "winner_reference_sets.csv",
            "winner_reference_contrasts": "winner_reference_contrasts.csv",
        }
        identity = certificate.get("byte_identity", {})
        for name, path in paths.items():
            manifest_hash = manifest.get("files", {}).get(path, {}).get("sha256")
            if identity.get(path) != expected[name] or manifest_hash != expected[name]:
                raise UnifiedReportingError(f"winner chain does not bind {path}")
        inference = manifest.get("inference_contract", {})
        if (
            inference.get("equivalence_claim") is not False
            or inference.get("simultaneous_or_familywise_coverage") is not False
            or inference.get("winner_selection_adjusted") is not False
            or inference.get("multiplicity_adjustment") != "NONE"
        ):
            raise UnifiedReportingError("winner-reference inference boundary drift")
        return
    raise UnifiedReportingError(f"no certified-source chain validator for {source_id}")


def parse_contract_bytes(payload: bytes, *, where: str = "unified reporting contract") -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnifiedReportingError(f"cannot parse {where}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != CONTRACT_SCHEMA:
        raise UnifiedReportingError("unified reporting contract schema drift")
    lanes = value.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        raise UnifiedReportingError("unified reporting contract lanes are missing")
    allowed_adapters = {
        "frozen24_v1", "external_v3", "edis_v2", "localization_v1",
        "prefix_v1", "winner_reference_v1", "leash_v1", "rag_evidence_v1",
        "not_certified",
    }
    required = {
        "adapter", "lane_id", "task_id", "default_prediction_unit",
        "default_estimand_id", "report_partition",
    }
    for source_id, lane in lanes.items():
        if not isinstance(source_id, str) or not source_id or not isinstance(lane, Mapping):
            raise UnifiedReportingError("unified reporting lane records are invalid")
        missing = required - set(lane)
        if missing or any(not isinstance(lane[field], str) or not lane[field] for field in required):
            raise UnifiedReportingError(f"contract lane {source_id} field drift: missing={sorted(missing)}")
        if lane["adapter"] not in allowed_adapters:
            raise UnifiedReportingError(f"contract lane {source_id} has an unsupported adapter")
        if lane["report_partition"] not in {"primary", "context"}:
            raise UnifiedReportingError(f"contract lane {source_id} has an invalid partition")
        if lane["adapter"] == "leash_v1":
            if (
                source_id != "leash_stopping"
                or lane["lane_id"] != "leash_actual_stopping"
                or lane["task_id"] != "adaptive_stopping"
                or lane["default_prediction_unit"] != "source_question"
                or lane["default_estimand_id"] != "accuracy_vs_realized_compute"
                or lane["report_partition"] != "context"
                or lane.get("allowed_arms") != ["cot", "leash", "nocot"]
                or lane.get("reference_arm") != "cot"
                or lane.get("bootstrap_draws") != 2000
            ):
                raise UnifiedReportingError("LEASH unified-reporting contract drift")
        if lane["adapter"] == "rag_evidence_v1":
            expected_panels = [
                "ragtruth_evidence_contrast_answer",
                "ragtruth_evidence_contrast_sentence",
                "ragtruth_evidence_contrast_token",
                "gasp_protocol_sentence",
                "lettucedetect_example",
                "refchecker_threeway",
                "refchecker_binary_claim",
            ]
            if (
                source_id != "rag_evidence"
                or lane["lane_id"] != "rag_evidence"
                or lane["task_id"] != "rag_evidence_evaluation"
                or lane["default_prediction_unit"] != "panel_registered_unit"
                or lane["default_estimand_id"] != "panel_registered_rag_evidence"
                or lane["report_partition"] != "context"
                or lane.get("panel_ids") != expected_panels
                or lane.get("refchecker_subgroups")
                != ["accurate_context", "noisy_context", "zero_context"]
                or lane.get("bootstrap_draws") != 20000
            ):
                raise UnifiedReportingError("RAG unified-reporting contract drift")
        if lane["adapter"] == "winner_reference_v1":
            base = lane.get("base_source")
            if not isinstance(base, str) or base not in lanes or base == source_id:
                raise UnifiedReportingError(f"winner lane {source_id} has an invalid base_source")
        elif "base_source" in lane:
            raise UnifiedReportingError(f"non-winner lane {source_id} declares base_source")
    if not isinstance(value.get("access_partitions", {}), Mapping):
        raise UnifiedReportingError("contract access_partitions must be an object")
    if not isinstance(value.get("claim_boundaries", {}), Mapping):
        raise UnifiedReportingError("contract claim_boundaries must be an object")
    return value


def load_contract(path: str | Path) -> dict[str, Any]:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise UnifiedReportingError(f"cannot load unified reporting contract: {exc}") from exc
    return parse_contract_bytes(payload)


def parse_source_lock_bytes(payload: bytes, *, where: str = "unified reporting source lock") -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnifiedReportingError(f"cannot parse {where}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SOURCE_LOCK_SCHEMA:
        raise UnifiedReportingError("unified reporting source-lock schema drift")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise UnifiedReportingError("unified reporting source lock has no sources")
    ids = [item.get("source_id") for item in sources if isinstance(item, Mapping)]
    if len(ids) != len(sources) or len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
        raise UnifiedReportingError("source lock source_id values are invalid or duplicated")
    for record in sources:
        source_id = record["source_id"]
        if type(record.get("certified")) is not bool:
            raise UnifiedReportingError(f"source {source_id} certified flag must be boolean")
        if not isinstance(record.get("source_release_id"), str) or not record["source_release_id"]:
            raise UnifiedReportingError(f"source {source_id} source_release_id is invalid")
        if record["certified"] is False:
            forbidden = {"certificate", "manifest", "files", "source_root_id"} & set(record)
            if (
                record.get("status") != "NOT_CERTIFIED"
                or record["source_release_id"] != "NOT_CERTIFIED"
                or forbidden
            ):
                raise UnifiedReportingError(
                    f"uncertified source {source_id} is not a source-closed placeholder"
                )
            continue
        root_id = record.get("source_root_id")
        if not isinstance(root_id, str) or not root_id:
            raise UnifiedReportingError(f"certified source {source_id} has no root alias")
        bindings: list[tuple[str, Mapping[str, Any]]] = []
        certificate = record.get("certificate")
        if not isinstance(certificate, Mapping):
            raise UnifiedReportingError(f"certified source {source_id} has no certificate lock")
        bindings.append(("certificate", certificate))
        manifest = record.get("manifest")
        if manifest is not None:
            if not isinstance(manifest, Mapping):
                raise UnifiedReportingError(f"source {source_id} manifest lock is invalid")
            bindings.append(("manifest", manifest))
        files = record.get("files")
        if not isinstance(files, Mapping) or not files:
            raise UnifiedReportingError(f"certified source {source_id} has no artifact locks")
        for name, binding in files.items():
            if not isinstance(name, str) or not name or not isinstance(binding, Mapping):
                raise UnifiedReportingError(f"source {source_id} artifact lock is invalid")
            if binding.get("format") not in {"csv", "json"}:
                raise UnifiedReportingError(f"source {source_id} artifact format is invalid")
            bindings.append((f"files.{name}", binding))
        observed_paths: set[str] = set()
        for name, binding in bindings:
            path_value = binding.get("path")
            _safe_relative(path_value, where=f"{source_id}.{name}.path")
            _validate_sha(binding.get("file_sha256"), where=f"{source_id}.{name}.file_sha256")
            if path_value in observed_paths:
                raise UnifiedReportingError(f"source {source_id} reuses a locked path")
            observed_paths.add(path_value)
        for name, binding in (("certificate", certificate), ("manifest", manifest)):
            if binding is None:
                continue
            if not isinstance(binding.get("schema_version"), str) or not binding["schema_version"]:
                raise UnifiedReportingError(f"source {source_id} {name} schema lock is invalid")
            if not isinstance(binding.get("self_hash_field"), str) or not binding["self_hash_field"]:
                raise UnifiedReportingError(f"source {source_id} {name} self-hash field is invalid")
            _validate_sha(binding.get("self_hash"), where=f"{source_id}.{name}.self_hash")
    return value


def load_source_lock(path: str | Path) -> dict[str, Any]:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise UnifiedReportingError(f"cannot load unified reporting source lock: {exc}") from exc
    return parse_source_lock_bytes(payload)


def validate_contract_source_lock(
    contract: Mapping[str, Any], source_lock: Mapping[str, Any]
) -> None:
    lanes = contract.get("lanes", {})
    records = {record["source_id"]: record for record in source_lock.get("sources", [])}
    if set(lanes) != set(records):
        raise UnifiedReportingError(
            "contract/source-lock source roster mismatch: "
            f"contract_only={sorted(set(lanes) - set(records))}, "
            f"lock_only={sorted(set(records) - set(lanes))}"
        )
    for source_id, lane in lanes.items():
        certified = records[source_id].get("certified") is True
        if certified and lane["adapter"] == "not_certified":
            raise UnifiedReportingError(f"source certification/adapter mismatch for {source_id}")
        if not certified and lane["adapter"] != "not_certified":
            future_placeholders = {
                "leash_stopping": "leash_v1",
                "rag_evidence": "rag_evidence_v1",
            }
            if future_placeholders.get(source_id) != lane["adapter"]:
                raise UnifiedReportingError(
                    f"source certification/adapter mismatch for {source_id}"
                )
        if lane["adapter"] == "winner_reference_v1":
            base = lane["base_source"]
            if records[base].get("certified") is not True:
                raise UnifiedReportingError(f"winner source {source_id} has an uncertified base")


def authenticate_sources(
    source_lock: Mapping[str, Any],
    *,
    source_roots: Mapping[str, str | Path],
) -> list[AuthenticatedSource]:
    output: list[AuthenticatedSource] = []
    for record in source_lock["sources"]:
        source_id = str(record["source_id"])
        certified = record.get("certified") is True
        logical_binding_sha256 = canonical_sha256(record)
        source_binding_id = f"sourcev1_{logical_binding_sha256[:24]}"
        if not certified:
            if record.get("status") != "NOT_CERTIFIED" or record.get("source_release_id") != "NOT_CERTIFIED":
                raise UnifiedReportingError(f"uncertified source {source_id} is not an explicit placeholder")
            output.append(
                AuthenticatedSource(
                    source_id=source_id, source_release_id="NOT_CERTIFIED",
                    source_binding_id=source_binding_id, certified=False,
                    source_status="NOT_CERTIFIED", source_root_id=None,
                    logical_binding_sha256=logical_binding_sha256,
                    lock_record=record, certificate=None, manifest=None, files={},
                )
            )
            continue
        root_id = record.get("source_root_id")
        if not isinstance(root_id, str) or root_id not in source_roots:
            raise UnifiedReportingError(f"missing runtime source root for {source_id}: {root_id!r}")
        root = source_roots[root_id]
        certificate_binding = record.get("certificate")
        if not isinstance(certificate_binding, Mapping):
            raise UnifiedReportingError(f"{source_id} certificate binding is missing")
        certificate_payload = read_locked_file(
            root, certificate_binding["path"], certificate_binding["file_sha256"]
        )
        certificate = _json(certificate_payload, where=f"{source_id} certificate")
        _verify_json_binding(certificate, certificate_binding, where=f"{source_id} certificate")
        source_release_id = str(record["source_release_id"])
        if "release_id" in certificate and certificate.get("release_id") != source_release_id:
            raise UnifiedReportingError(f"{source_id} certificate release_id drift")
        manifest = None
        manifest_binding = record.get("manifest")
        if manifest_binding is not None:
            if not isinstance(manifest_binding, Mapping):
                raise UnifiedReportingError(f"{source_id} manifest binding is invalid")
            manifest_payload = read_locked_file(
                root, manifest_binding["path"], manifest_binding["file_sha256"]
            )
            manifest = _json(manifest_payload, where=f"{source_id} manifest")
            _verify_json_binding(manifest, manifest_binding, where=f"{source_id} manifest")
            if "release_id" in manifest and manifest.get("release_id") != source_release_id:
                raise UnifiedReportingError(f"{source_id} manifest release_id drift")
        _verify_chain(source_id, record, certificate, manifest)
        files: dict[str, bytes] = {}
        for name, binding in sorted(record.get("files", {}).items()):
            files[name] = read_locked_file(root, binding["path"], binding["file_sha256"])
        output.append(
            AuthenticatedSource(
                source_id=source_id,
                source_release_id=source_release_id,
                source_binding_id=source_binding_id, certified=True,
                source_status="CERTIFIED", source_root_id=root_id,
                logical_binding_sha256=logical_binding_sha256,
                lock_record=record, certificate=certificate, manifest=manifest,
                files=files,
            )
        )
    return output


def source_lock_sha256(source_lock: Mapping[str, Any]) -> str:
    return canonical_sha256(source_lock)


__all__ = [
    "AuthenticatedSource", "CONTRACT_SCHEMA", "SOURCE_LOCK_SCHEMA",
    "authenticate_sources", "load_contract", "load_source_lock",
    "parse_contract_bytes", "parse_source_lock_bytes", "read_locked_file",
    "source_lock_sha256", "validate_contract_source_lock",
]
