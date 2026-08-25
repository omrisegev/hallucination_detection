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
        "prefix_v1", "winner_reference_v1", "not_certified",
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
        if (lane["adapter"] == "not_certified") is certified:
            raise UnifiedReportingError(f"source certification/adapter mismatch for {source_id}")
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
