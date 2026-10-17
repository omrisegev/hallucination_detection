"""Fail-closed contracts for the reconstruction RAG-evidence benchmark lane.

The lane deliberately contains several incomparable panels.  A panel is the
smallest unit that shares a dataset, prediction unit, access regime and
estimand.  This module makes that separation executable instead of relying on
report prose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import errno
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import pickle
import secrets
import stat
import sys
from typing import Any

import numpy as np

from .io import canonical_json_bytes, sha256_bytes, sha256_file


REGISTRY_SCHEMA = "reconstruction-rag-evidence-registry-v1"
FIT_INPUT_SCHEMA = "reconstruction-rag-evidence-fit-input-v1"
PRIVATE_LABEL_SCHEMA = "reconstruction-rag-evidence-private-labels-v1"
PREPARATION_SCHEMA = "reconstruction-rag-evidence-preparation-v1"
PREPARATION_AB_SCHEMA = "reconstruction-rag-evidence-preparation-ab-v1"
SCORE_SCHEMA = "reconstruction-rag-evidence-scores-v1"
SCORE_FREEZE_SCHEMA = "reconstruction-rag-evidence-score-freeze-v1"
SCORE_AB_SCHEMA = "reconstruction-rag-evidence-score-ab-v1"
EVALUATION_SCHEMA = "reconstruction-rag-evidence-evaluation-v1"
EVALUATION_AB_SCHEMA = "reconstruction-rag-evidence-evaluation-ab-v1"

FIT_INPUT_FILENAME = "FIT_INPUT.pkl"
PRIVATE_LABEL_FILENAME = "PRIVATE_LABELS.pkl"
PREPARATION_MANIFEST_FILENAME = "PREPARATION_MANIFEST.json"
SCORES_FILENAME = "SCORES.npz"
SCORE_MANIFEST_FILENAME = "SCORE_FREEZE.json"
EVALUATION_MANIFEST_FILENAME = "EVALUATION_MANIFEST.json"

PANEL_IDS = (
    "ragtruth_evidence_contrast_answer",
    "ragtruth_evidence_contrast_sentence",
    "ragtruth_evidence_contrast_token",
    "gasp_protocol_sentence",
    "lettucedetect_example",
    "refchecker_threeway",
    "refchecker_binary_claim",
)
REFCHECKER_SETTINGS = (
    "accurate_context",
    "noisy_context",
    "zero_context",
)
SOURCE_ASSET_IDS = (
    "ragtruth_official_response", "ragtruth_dev_cache", "ragtruth_dev_manifest",
    "ragtruth_test_cache", "ragtruth_test_manifest", "gasp_cache", "gasp_manifest",
    "lettuce_cache", "lettuce_manifest", "refchecker_cache",
    "refchecker_nli_predictions", "refchecker_manifest", "tokenizer_config",
    "tokenizer_model_config", "tokenizer_json", "tokenizer_merges", "tokenizer_vocab",
)
EXPECTED_SOURCE_ASSET_ROSTER_SHA256 = (
    "2000737296b1e1c28d05ceabe0465ef12fbd409c3d7e05e3c7d6cac36034cdea"
)


class RagEvidenceContractError(RuntimeError):
    """A RAG evidence artifact violated the frozen registry or stage contract."""


def payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def add_payload_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output["payload_sha256"] = payload_sha256(output)
    return output


def verify_payload(value: Mapping[str, Any], *, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != payload_sha256(payload):
        raise RagEvidenceContractError(f"{name} payload SHA-256 failed")


def add_pickle_payload_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind an array-bearing mapping without expanding large arrays to JSON."""

    output = dict(value)
    output["payload_sha256"] = sha256_bytes(pickle.dumps(output, protocol=5))
    return output


def verify_pickle_payload(value: Mapping[str, Any], *, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != sha256_bytes(pickle.dumps(payload, protocol=5)):
        raise RagEvidenceContractError(f"{name} pickle-payload SHA-256 failed")


def _require_exact_keys(value: Mapping[str, Any], expected: Sequence[str], *, name: str) -> None:
    if set(value) != set(expected):
        raise RagEvidenceContractError(
            f"{name} keys drifted: expected={sorted(expected)}, observed={sorted(value)}"
        )


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        value = json.loads(
            read_bound_file_bytes(
                registry_path, name="RAG evidence registry"
            ).decode("utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RagEvidenceContractError(f"invalid RAG evidence registry: {registry_path}") from error
    if value.get("schema_version") != REGISTRY_SCHEMA:
        raise RagEvidenceContractError("unexpected RAG evidence registry schema")
    panels = value.get("panels")
    if not isinstance(panels, list) or tuple(row.get("panel_id") for row in panels) != PANEL_IDS:
        raise RagEvidenceContractError("RAG evidence panel roster/order drifted")
    identities = [(row.get("dataset"), row.get("unit"), row.get("access"), row.get("estimand")) for row in panels]
    if len(set(identities)) != len(identities):
        raise RagEvidenceContractError("RAG evidence panels do not have unique access/estimand identities")
    registered_methods = value.get("method_contract")
    used_methods = {
        str(method)
        for panel in panels
        for method in panel.get("methods", ())
    }
    if not isinstance(registered_methods, Mapping) or set(registered_methods) != used_methods:
        raise RagEvidenceContractError("RAG evidence method contract/usage roster drifted")
    if value.get("evaluation", {}).get("cross_panel_macro") != "FORBIDDEN":
        raise RagEvidenceContractError("RAG evidence cross-panel macro is not fail-closed")
    if value.get("evaluation", {}).get("refchecker_setting_pooling") != "FORBIDDEN":
        raise RagEvidenceContractError("RefChecker setting pooling is not fail-closed")
    bootstrap = value.get("evaluation", {}).get("bootstrap", {})
    if int(bootstrap.get("draws", -1)) != 20_000 or bootstrap.get("resampling") != "complete source groups with replacement":
        raise RagEvidenceContractError("RAG evidence grouped-bootstrap contract drifted")
    ref_panels = [row for row in panels if row["panel_id"].startswith("refchecker_")]
    if any(tuple(row.get("required_subgroups", ())) != REFCHECKER_SETTINGS for row in ref_panels):
        raise RagEvidenceContractError("RefChecker setting roster/order drifted")
    forbidden = set(value.get("fit_visibility", {}).get("forbidden_fields", ()))
    required_forbidden = {
        "label", "labels", "gold", "human_label", "label_unsupported",
        "response_label", "span_labels", "source_id", "example_id",
        "bootstrap_group", "private_labels", "target",
    }
    if not required_forbidden.issubset(forbidden):
        raise RagEvidenceContractError("RAG fit forbidden-field roster is incomplete")
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise RagEvidenceContractError("RAG evidence source roster is absent")
    if tuple(sources) != SOURCE_ASSET_IDS:
        raise RagEvidenceContractError("RAG evidence source-asset roster/order drifted")
    for asset_id, item in sources.items():
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size_bytes"}:
            raise RagEvidenceContractError(f"malformed RAG source asset: {asset_id}")
        if Path(str(item["path"])).is_absolute() or ".." in Path(str(item["path"])).parts:
            raise RagEvidenceContractError(f"unsafe RAG source path: {asset_id}")
        digest = str(item["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RagEvidenceContractError(f"malformed RAG source hash: {asset_id}")
        if int(item["size_bytes"]) <= 0:
            raise RagEvidenceContractError(f"malformed RAG source size: {asset_id}")
    return value


def bind_source_assets(source_root: str | Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every asset through held no-follow fds and return its binding."""

    with BoundRagSourceAssets(source_root, registry) as held:
        return dict(held.binding)


def validate_source_binding(
    binding: Mapping[str, Any], *, source_root: str | Path, registry: Mapping[str, Any]
) -> None:
    observed = bind_source_assets(source_root, registry)
    if dict(binding) != observed:
        raise RagEvidenceContractError("RAG transitive source binding changed")


def _forbidden_key_match(key: str, forbidden: set[str]) -> bool:
    normalized = key.strip().lower()
    if normalized in forbidden:
        return True
    return normalized.startswith("gold_") or normalized.endswith("_label") or normalized.endswith("_labels")


def validate_fit_sanitization(value: Any, *, forbidden_fields: Sequence[str], path: str = "fit_input") -> None:
    """Recursively reject target/group fields before a fit worker can open data."""

    forbidden = {str(item).strip().lower() for item in forbidden_fields}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _forbidden_key_match(str(key), forbidden):
                raise RagEvidenceContractError(f"forbidden fit-visible field at {path}.{key}")
            validate_fit_sanitization(item, forbidden_fields=forbidden, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_fit_sanitization(item, forbidden_fields=forbidden, path=f"{path}[{index}]")
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise RagEvidenceContractError(f"object array forbidden at {path}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RagEvidenceContractError(f"non-finite fit-visible array at {path}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RagEvidenceContractError(f"non-finite fit-visible scalar at {path}")


def validate_fit_input(value: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        value,
        (
            "schema_version", "lane_id", "contract_version", "panels",
            "rosters", "source_asset_roster_sha256", "historical_scores_opened",
            "targets_opened_by_fit", "payload_sha256",
        ),
        name="RAG fit input",
    )
    verify_pickle_payload(value, name="RAG fit input")
    if value.get("schema_version") != FIT_INPUT_SCHEMA or value.get("lane_id") != registry["lane_id"]:
        raise RagEvidenceContractError("RAG fit input registry/schema binding failed")
    if value.get("historical_scores_opened") is not False or value.get("targets_opened_by_fit") is not False:
        raise RagEvidenceContractError("RAG fit input claims target or historical-score access")
    panels = value.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != {"ragtruth", "gasp", "lettuce", "refchecker"}:
        raise RagEvidenceContractError("RAG fit input panel bundle drifted")
    validate_fit_sanitization(
        value, forbidden_fields=registry["fit_visibility"]["forbidden_fields"]
    )
    return dict(value)


def validate_private_labels(value: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        value,
        (
            "schema_version", "lane_id", "ragtruth", "gasp", "lettuce",
            "refchecker", "rosters", "private_target_audit",
            "source_asset_roster_sha256", "payload_sha256",
        ),
        name="RAG private labels",
    )
    verify_pickle_payload(value, name="RAG private labels")
    if value.get("schema_version") != PRIVATE_LABEL_SCHEMA or value.get("lane_id") != registry["lane_id"]:
        raise RagEvidenceContractError("RAG private-label registry/schema binding failed")
    return dict(value)


def pickle_bytes(value: Any) -> bytes:
    """Deterministic for canonical ordered builtins/NumPy arrays in this lane."""

    return pickle.dumps(value, protocol=5)


def load_pickle(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    name: str = "RAG pickle",
) -> Any:
    payload = read_bound_file_bytes(
        path, expected_sha256=expected_sha256, name=name
    )
    return pickle.load(BytesIO(payload))


def load_fit_input(
    path: str | Path,
    registry: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    payload = read_bound_file_bytes(
        path, expected_sha256=expected_sha256, name="RAG fit input"
    )
    return load_fit_input_bytes(payload, registry)


def load_fit_input_bytes(
    payload: bytes, registry: Mapping[str, Any]
) -> dict[str, Any]:
    value = pickle.load(BytesIO(payload))
    return _validate_loaded_fit_input(value, registry)


def load_fit_input_handle(
    handle: Any, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Parse a fit input from an already-held descriptor stream."""

    return _validate_loaded_fit_input(pickle.load(handle), registry)


def _validate_loaded_fit_input(
    value: Any, registry: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RagEvidenceContractError("RAG fit input is not a mapping")
    return validate_fit_input(value, registry)


def load_private_labels(
    path: str | Path,
    registry: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    payload = read_bound_file_bytes(
        path, expected_sha256=expected_sha256, name="RAG private labels"
    )
    return load_private_labels_bytes(payload, registry)


def load_private_labels_bytes(
    payload: bytes, registry: Mapping[str, Any]
) -> dict[str, Any]:
    value = pickle.load(BytesIO(payload))
    if not isinstance(value, Mapping):
        raise RagEvidenceContractError("RAG private labels are not a mapping")
    return validate_private_labels(value, registry)


def _child_name(name: str) -> bytes:
    if (
        not name
        or "\x00" in name
        or name in {".", ".."}
        or Path(name).name != name
    ):
        raise ValueError(f"unsafe RAG parent-relative child name: {name!r}")
    return os.fsencode(name)


def validate_artifact_identifier(value: str, *, name: str = "artifact identifier") -> str:
    """Accept one conservative path component and reject traversal/ambiguity."""

    text = str(value)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if (
        not text
        or len(text) > 128
        or text.startswith(".")
        or text.endswith(".")
        or any(character not in allowed for character in text)
        or Path(text).name != text
    ):
        raise RagEvidenceContractError(f"unsafe {name}: {text!r}")
    return text


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise RagEvidenceContractError("RAG publication parent is not a directory")
    return int(value.st_dev), int(value.st_ino)


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise RuntimeError("secure RAG parent-dirfd publication is unavailable")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_chain(path: Path, *, create: bool) -> int:
    """Open every absolute-path component with no symlink following."""

    absolute = path.absolute()
    parts = absolute.parts
    if not absolute.is_absolute() or not parts or parts[0] != absolute.anchor:
        raise RagEvidenceContractError(f"RAG directory path is not absolute: {path}")
    if any(part in {"", ".", ".."} or "\x00" in part for part in parts[1:]):
        raise RagEvidenceContractError(f"unsafe RAG directory path: {path}")
    flags = _directory_open_flags()
    descriptor = os.open(absolute.anchor, flags)
    try:
        _directory_identity(os.fstat(descriptor))
        for part in parts[1:]:
            _child_name(part)
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            _directory_identity(os.fstat(child))
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_bound_parent(
    parent: Path, *, create: bool = True
) -> tuple[int, tuple[int, int]]:
    descriptor = _open_directory_chain(parent, create=create)
    return descriptor, _directory_identity(os.fstat(descriptor))


def _assert_parent_binding(
    parent: Path, descriptor: int, expected: tuple[int, int]
) -> None:
    if _directory_identity(os.fstat(descriptor)) != expected:
        raise RagEvidenceContractError("RAG held publication parent inode changed")
    try:
        observed_descriptor = _open_directory_chain(parent, create=False)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RagEvidenceContractError("RAG publication parent path was replaced") from error
    try:
        observed = _directory_identity(os.fstat(observed_descriptor))
        if observed != expected:
            raise RagEvidenceContractError(
                "RAG publication parent path inode was replaced"
            )
    finally:
        os.close(observed_descriptor)


class BoundRagFile:
    """A verified regular file held by fd from hash through parse and rebind."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
        name: str = "RAG file",
    ) -> None:
        self.path = Path(path).absolute()
        self.name = name
        _child_name(self.path.name)
        self.parent_descriptor, self.parent_identity = _open_bound_parent(
            self.path.parent, create=False
        )
        self.descriptor = -1
        self.identity: tuple[int, int] | None = None
        self._initial_signature: tuple[int, int, int, int, int] | None = None
        self.sha256 = ""
        self.size_bytes = -1
        try:
            self.descriptor = os.open(
                self.path.name,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self.parent_descriptor,
            )
            value = os.fstat(self.descriptor)
            if not stat.S_ISREG(value.st_mode):
                raise RagEvidenceContractError(f"{name} is not a regular file")
            self.identity = (int(value.st_dev), int(value.st_ino))
            self._initial_signature = self._signature(value)
            self.size_bytes = int(value.st_size)
            self.sha256 = self._hash_held_bytes()
            if self._signature(os.fstat(self.descriptor)) != self._initial_signature:
                raise RagEvidenceContractError(f"{name} changed during initial hash")
            if expected_size is not None and self.size_bytes != int(expected_size):
                raise RagEvidenceContractError(
                    f"{name} size differs: {self.size_bytes} != {expected_size}"
                )
            if expected_sha256 is not None and self.sha256 != expected_sha256:
                raise RagEvidenceContractError(
                    f"{name} SHA-256 differs: {self.sha256} != {expected_sha256}"
                )
            self._assert_rebound()
        except Exception:
            self.close(verify=False)
            raise

    @staticmethod
    def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    def _hash_held_bytes(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(self.descriptor, 1024 * 1024, offset)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        return digest.hexdigest()

    def _assert_rebound(self) -> None:
        if self.descriptor < 0 or self.identity is None:
            raise RuntimeError(f"{self.name} binding is closed")
        _assert_entry_identity(
            self.parent_descriptor,
            self.path.name,
            self.identity,
            require_directory=False,
        )
        _assert_parent_binding(
            self.path.parent,
            self.parent_descriptor,
            self.parent_identity,
        )

    def open(self):
        """Return a binary stream for the held inode, never for its pathname."""

        if self.descriptor < 0:
            raise RuntimeError(f"{self.name} binding is closed")
        duplicate = os.dup(self.descriptor)
        os.lseek(duplicate, 0, os.SEEK_SET)
        return os.fdopen(duplicate, "rb")

    def read_bytes(self) -> bytes:
        if self.descriptor < 0:
            raise RuntimeError(f"{self.name} binding is closed")
        blocks: list[bytes] = []
        offset = 0
        while True:
            block = os.pread(self.descriptor, 1024 * 1024, offset)
            if not block:
                break
            blocks.append(block)
            offset += len(block)
        payload = b"".join(blocks)
        if sha256_bytes(payload) != self.sha256:
            raise RagEvidenceContractError(f"{self.name} changed during read")
        return payload

    def verify_stable(self) -> None:
        if self.descriptor < 0 or self._initial_signature is None:
            raise RuntimeError(f"{self.name} binding is closed")
        if self._signature(os.fstat(self.descriptor)) != self._initial_signature:
            raise RagEvidenceContractError(f"{self.name} held inode changed")
        if self._hash_held_bytes() != self.sha256:
            raise RagEvidenceContractError(f"{self.name} bytes changed after parse")
        self._assert_rebound()

    def close(self, *, verify: bool = True) -> None:
        if self.parent_descriptor < 0:
            return
        failure: BaseException | None = None
        if verify and self.descriptor >= 0:
            try:
                self.verify_stable()
            except BaseException as error:
                failure = error
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        os.close(self.parent_descriptor)
        self.parent_descriptor = -1
        if failure is not None:
            raise failure

    def __enter__(self) -> "BoundRagFile":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(verify=True)


class BoundRagSourceAssets:
    """Hold the complete registered source roster through adapter parsing."""

    def __init__(self, source_root: str | Path, registry: Mapping[str, Any]) -> None:
        self.root = Path(source_root).absolute()
        self.files: dict[str, BoundRagFile] = {}
        assets: list[dict[str, Any]] = []
        try:
            for asset_id, item in registry["sources"].items():
                relative = Path(str(item["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RagEvidenceContractError(
                        f"RAG source escapes source root: {asset_id}"
                    )
                path = self.root / relative
                held = BoundRagFile(
                    path,
                    expected_sha256=str(item["sha256"]),
                    expected_size=int(item["size_bytes"]),
                    name=f"RAG source asset {asset_id}",
                )
                self.files[str(asset_id)] = held
                assets.append({
                    "asset_id": str(asset_id),
                    "path": str(item["path"]),
                    "size_bytes": held.size_bytes,
                    "sha256": held.sha256,
                })
            binding = {
                "source_root": str(self.root),
                "assets": assets,
                "asset_roster_sha256": payload_sha256(assets),
            }
            if (
                binding["asset_roster_sha256"]
                != EXPECTED_SOURCE_ASSET_ROSTER_SHA256
            ):
                raise RagEvidenceContractError(
                    "RAG evidence source roster commitment drifted"
                )
            binding["binding_sha256"] = payload_sha256(binding)
            self.binding = binding
        except Exception:
            self.close(verify=False)
            raise

    def __getitem__(self, asset_id: str) -> BoundRagFile:
        return self.files[asset_id]

    def verify_stable(self) -> None:
        for held in self.files.values():
            held.verify_stable()

    def close(self, *, verify: bool = True) -> None:
        failure: BaseException | None = None
        if verify:
            try:
                self.verify_stable()
            except BaseException as error:
                failure = error
        for held in reversed(tuple(self.files.values())):
            try:
                held.close(verify=False)
            except BaseException as error:
                if failure is None:
                    failure = error
        self.files.clear()
        if failure is not None:
            raise failure

    def __enter__(self) -> "BoundRagSourceAssets":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(verify=True)


def read_bound_file_bytes(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    name: str = "RAG file",
) -> bytes:
    """Hash, read, and end-rebind exactly one held regular-file inode."""

    with BoundRagFile(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        name=name,
    ) as held:
        return held.read_bytes()


class BoundRagTree:
    """Hold one directory root and reject aliases, links, and tree mutation.

    The canonical root inode remains open for the complete verification
    operation.  Every component below it is traversed with ``openat`` and
    ``O_NOFOLLOW``; only directories and regular files are accepted.  The
    initial relative-path/inode roster is rederived before the binding is
    released, so replacing the canonical root or any descendant cannot turn a
    successful verification into authorization for another tree.
    """

    def __init__(self, path: str | Path, *, name: str = "RAG artifact tree") -> None:
        self.path = Path(path).absolute()
        self.name = name
        _child_name(self.path.name)
        self.parent_descriptor, self.parent_identity = _open_bound_parent(
            self.path.parent, create=False
        )
        self.descriptor = -1
        self.identity: tuple[int, int] | None = None
        self.files: dict[str, tuple[int, int, int]] = {}
        self.directories: dict[str, tuple[int, int]] = {}
        try:
            self.identity = _entry_identity(
                self.parent_descriptor, self.path.name, require_directory=True
            )
            self.descriptor = os.open(
                self.path.name,
                _directory_open_flags(),
                dir_fd=self.parent_descriptor,
            )
            if _directory_identity(os.fstat(self.descriptor)) != self.identity:
                raise RagEvidenceContractError(
                    f"{name} changed while its root was opened"
                )
            self.files, self.directories = self._scan_tree()
            self._assert_rebound()
        except Exception:
            self.close(verify=False)
            raise

    @staticmethod
    def _open_child_directory(parent_descriptor: int, child: str) -> int:
        _child_name(child)
        descriptor = os.open(
            child, _directory_open_flags(), dir_fd=parent_descriptor
        )
        _directory_identity(os.fstat(descriptor))
        return descriptor

    def _scan_directory(
        self,
        descriptor: int,
        prefix: str,
        files: dict[str, tuple[int, int, int]],
        directories: dict[str, tuple[int, int]],
    ) -> None:
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise RagEvidenceContractError(
                f"cannot enumerate held {self.name}"
            ) from error
        for child in names:
            _child_name(child)
            relative = f"{prefix}/{child}" if prefix else child
            value = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
            identity = (int(value.st_dev), int(value.st_ino))
            if stat.S_ISLNK(value.st_mode):
                raise RagEvidenceContractError(
                    f"symlink forbidden in {self.name}: {relative}"
                )
            if stat.S_ISDIR(value.st_mode):
                child_descriptor = self._open_child_directory(descriptor, child)
                try:
                    if _directory_identity(os.fstat(child_descriptor)) != identity:
                        raise RagEvidenceContractError(
                            f"directory changed while scanning {self.name}: {relative}"
                        )
                    directories[relative] = identity
                    self._scan_directory(
                        child_descriptor, relative, files, directories
                    )
                    if _directory_identity(os.fstat(child_descriptor)) != identity:
                        raise RagEvidenceContractError(
                            f"directory changed after scanning {self.name}: {relative}"
                        )
                    _assert_entry_identity(
                        descriptor, child, identity, require_directory=True
                    )
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(value.st_mode):
                file_descriptor = os.open(
                    child,
                    os.O_RDONLY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(file_descriptor)
                    opened_identity = (int(opened.st_dev), int(opened.st_ino))
                    if not stat.S_ISREG(opened.st_mode) or opened_identity != identity:
                        raise RagEvidenceContractError(
                            f"file changed while scanning {self.name}: {relative}"
                        )
                    files[relative] = (
                        opened_identity[0], opened_identity[1], int(opened.st_size)
                    )
                    _assert_entry_identity(
                        descriptor, child, identity, require_directory=False
                    )
                finally:
                    os.close(file_descriptor)
            else:
                raise RagEvidenceContractError(
                    f"special file forbidden in {self.name}: {relative}"
                )

    def _scan_tree(
        self,
    ) -> tuple[dict[str, tuple[int, int, int]], dict[str, tuple[int, int]]]:
        if self.descriptor < 0:
            raise RuntimeError(f"{self.name} binding is closed")
        files: dict[str, tuple[int, int, int]] = {}
        directories: dict[str, tuple[int, int]] = {}
        self._scan_directory(self.descriptor, "", files, directories)
        return files, directories

    def _assert_rebound(self) -> None:
        if self.descriptor < 0 or self.identity is None:
            raise RuntimeError(f"{self.name} binding is closed")
        if _directory_identity(os.fstat(self.descriptor)) != self.identity:
            raise RagEvidenceContractError(f"held {self.name} root inode changed")
        _assert_entry_identity(
            self.parent_descriptor,
            self.path.name,
            self.identity,
            require_directory=True,
        )
        _assert_parent_binding(
            self.path.parent, self.parent_descriptor, self.parent_identity
        )

    def verify_stable(self) -> None:
        self._assert_rebound()
        files, directories = self._scan_tree()
        if files != self.files or directories != self.directories:
            raise RagEvidenceContractError(f"{self.name} inode roster changed")
        self._assert_rebound()

    @property
    def regular_file_identities(self) -> set[tuple[int, int]]:
        return {(row[0], row[1]) for row in self.files.values()}

    def close(self, *, verify: bool = True) -> None:
        if self.parent_descriptor < 0:
            return
        failure: BaseException | None = None
        if verify and self.descriptor >= 0:
            try:
                self.verify_stable()
            except BaseException as error:
                failure = error
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        os.close(self.parent_descriptor)
        self.parent_descriptor = -1
        if failure is not None:
            raise failure

    def __enter__(self) -> "BoundRagTree":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(verify=True)


def assert_physical_tree_independence(
    left: BoundRagTree,
    right: BoundRagTree,
    *,
    name: str = "RAG A/B artifact trees",
) -> None:
    """Require genuinely distinct roots and no cross-build hard-linked files."""

    if left.identity is None or right.identity is None:
        raise RuntimeError("cannot compare closed RAG tree bindings")
    if left.identity == right.identity:
        raise RagEvidenceContractError(f"{name} share one directory inode")
    shared = left.regular_file_identities & right.regular_file_identities
    if shared:
        raise RagEvidenceContractError(
            f"{name} share {len(shared)} regular-file inode(s)"
        )


def _entry_identity(
    descriptor: int, name: str, *, require_directory: bool | None = None
) -> tuple[int, int]:
    value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    if require_directory is True and not stat.S_ISDIR(value.st_mode):
        raise RagEvidenceContractError(f"RAG publication entry is not a directory: {name}")
    if require_directory is False and not stat.S_ISREG(value.st_mode):
        raise RagEvidenceContractError(f"RAG publication entry is not a regular file: {name}")
    return int(value.st_dev), int(value.st_ino)


def _assert_entry_identity(
    descriptor: int, name: str, expected: tuple[int, int],
    *, require_directory: bool | None = None,
) -> None:
    try:
        observed = _entry_identity(
            descriptor, name, require_directory=require_directory
        )
    except FileNotFoundError as error:
        raise RagEvidenceContractError(f"RAG publication entry disappeared: {name}") from error
    if observed != expected:
        raise RagEvidenceContractError(f"RAG publication entry inode was replaced: {name}")


def _rename_directory_noreplace_at(
    parent_descriptor: int, source_name: str, target_name: str,
    *, require_directory: bool | None = True,
) -> tuple[int, int] | None:
    source = _child_name(source_name)
    target = _child_name(target_name)
    source_identity = _entry_identity(
        parent_descriptor,
        source_name,
        require_directory=require_directory,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        if operation is None:
            raise RuntimeError("atomic no-replace RAG publication is unavailable")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            parent_descriptor, source, parent_descriptor, target, 0x00000004
        )
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise RuntimeError("atomic no-replace RAG publication is unavailable")
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(
            parent_descriptor, source, parent_descriptor, target, 1
        )
    else:
        raise RuntimeError(f"atomic no-replace RAG publication is unsupported on {sys.platform}")
    if result != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"RAG output already exists: {target_name}")
        raise OSError(number, os.strerror(number), target_name)
    try:
        target_identity = _entry_identity(
            parent_descriptor,
            target_name,
            require_directory=None,
        )
    except OSError:
        # The no-replace syscall already succeeded.  Never raise before the
        # caller records that the canonical final name is now the active name;
        # a missing/substituted final is handled by caller quarantine.
        return None
    # Return what the syscall actually moved.  Callers compare this to their
    # held object fd; coordinated source-name substitution is never inferred
    # away from a pre-syscall stat.
    return target_identity


def _quarantine_entry_at(
    parent_descriptor: int,
    source_name: str,
    *,
    label: str,
    require_directory: bool | None,
) -> tuple[str, tuple[int, int]]:
    """Move the entry currently at ``source_name`` aside; never delete it."""

    safe_label = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in label
    )[:80] or "entry"
    for _ in range(128):
        quarantine_name = (
            f".{safe_label}.rag-evidence-quarantine-{secrets.token_hex(12)}"
        )
        try:
            identity = _rename_directory_noreplace_at(
                parent_descriptor,
                source_name,
                quarantine_name,
                require_directory=require_directory,
            )
        except FileExistsError:
            continue
        if identity is None:
            raise RagEvidenceContractError(
                "RAG quarantine target disappeared after no-replace rename"
            )
        os.fsync(parent_descriptor)
        return quarantine_name, identity
    raise FileExistsError("cannot allocate a unique RAG evidence quarantine")


class AtomicRagDirectory:
    """Publish through a held, inode-bound parent dirfd with no replacement.

    Sensitive controller outputs must use :meth:`write_bytes`/`write_json`,
    which never re-resolve ``path``.  The pathname view exists for external
    tools and is guarded at their boundaries; those tools require an exclusive
    writer for the release parent because portable macOS has no directory-fd
    pathname suitable for subprocesses.
    """

    def __init__(self, final_path: str | Path) -> None:
        self.final_path = Path(final_path).absolute()
        self._final_name = self.final_path.name
        _child_name(self._final_name)
        self._parent_descriptor, self._parent_identity = _open_bound_parent(
            self.final_path.parent, create=True
        )
        self._closed = False
        self._stage_descriptor = -1
        self._stage_name = ""
        self._stage_identity: tuple[int, int] | None = None
        self._active_name: str | None = None
        self.quarantine_name: str | None = None
        try:
            try:
                os.stat(
                    self._final_name,
                    dir_fd=self._parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(f"RAG output already exists: {self.final_path}")
            for _ in range(128):
                candidate = (
                    f".{self._final_name}.rag-staging-{secrets.token_hex(12)}"
                )
                try:
                    os.mkdir(candidate, 0o700, dir_fd=self._parent_descriptor)
                    self._stage_name = candidate
                    self._active_name = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise FileExistsError("cannot allocate a unique RAG staging directory")
            self._stage_descriptor = os.open(
                self._stage_name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._parent_descriptor,
            )
            self._stage_identity = _directory_identity(
                os.fstat(self._stage_descriptor)
            )
            _assert_entry_identity(
                self._parent_descriptor, self._stage_name,
                self._stage_identity, require_directory=True,
            )
            _assert_parent_binding(
                self.final_path.parent,
                self._parent_descriptor,
                self._parent_identity,
            )
            os.fsync(self._parent_descriptor)
        except Exception:
            if self._stage_descriptor >= 0:
                os.close(self._stage_descriptor)
            if self._active_name:
                try:
                    self.quarantine_name, _ = _quarantine_entry_at(
                        self._parent_descriptor,
                        self._active_name,
                        label=f"{self._final_name}-constructor-failure",
                        require_directory=None,
                    )
                except (FileNotFoundError, RagEvidenceContractError, OSError):
                    pass
            os.close(self._parent_descriptor)
            self._closed = True
            raise
        self.path = self.final_path.parent / self._stage_name
        self.committed = False

    def assert_path_binding(self) -> None:
        """Fail if the external-tool pathname no longer names the held stage."""

        if self._closed or self.committed:
            raise RuntimeError("RAG stage pathname is not writable")
        _assert_parent_binding(
            self.final_path.parent,
            self._parent_descriptor,
            self._parent_identity,
        )
        _assert_entry_identity(
            self._parent_descriptor,
            self._stage_name,
            self._stage_identity,
            require_directory=True,
        )
        observed = _directory_identity(os.stat(self.path, follow_symlinks=False))
        if observed != self._stage_identity:
            raise RagEvidenceContractError("RAG staging pathname inode was replaced")
        if _directory_identity(os.fstat(self._stage_descriptor)) != self._stage_identity:
            raise RagEvidenceContractError("RAG held staging directory inode changed")

    def _open_relative_parent(
        self, relative_path: str | Path
    ) -> tuple[int, str]:
        if self._closed or self.committed:
            raise RuntimeError("RAG stage is not writable")
        value = Path(relative_path)
        if value.is_absolute() or not value.parts:
            raise ValueError(f"unsafe RAG stage-relative path: {relative_path!r}")
        parts = value.parts
        if any(part in {"", ".", ".."} or "\x00" in part for part in parts):
            raise ValueError(f"unsafe RAG stage-relative path: {relative_path!r}")
        for part in parts:
            _child_name(part)
        if _directory_identity(os.fstat(self._stage_descriptor)) != self._stage_identity:
            raise RagEvidenceContractError("RAG held staging directory inode changed")
        descriptor = os.dup(self._stage_descriptor)
        try:
            for part in parts[:-1]:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
                child = os.open(
                    part,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                _directory_identity(os.fstat(child))
                os.close(descriptor)
                descriptor = child
            return descriptor, parts[-1]
        except Exception:
            os.close(descriptor)
            raise

    def write_bytes(self, relative_path: str | Path, payload: bytes) -> str:
        """Write an unpublished stage file without re-resolving the parent path."""

        parent_descriptor, name = self._open_relative_parent(relative_path)
        descriptor = -1
        identity: tuple[int, int] | None = None
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise RagEvidenceContractError("RAG stage output is not a regular file")
            identity = (int(opened.st_dev), int(opened.st_ino))
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("short write while creating RAG stage file")
                written += count
            os.fsync(descriptor)
            _assert_entry_identity(
                parent_descriptor, name, identity, require_directory=False
            )
            os.fsync(parent_descriptor)
            return sha256_bytes(payload)
        except Exception:
            if identity is not None:
                try:
                    _quarantine_entry_at(
                        parent_descriptor,
                        name,
                        label=f"stage-write-{name}",
                        require_directory=None,
                    )
                except (FileNotFoundError, RagEvidenceContractError, OSError):
                    pass
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_descriptor)

    def write_json(self, relative_path: str | Path, value: Any) -> str:
        return self.write_bytes(
            relative_path, canonical_json_bytes(value) + b"\n"
        )

    def commit(self) -> None:
        if self._closed:
            raise RuntimeError("RAG artifact stage is closed")
        if self.committed:
            raise RuntimeError("RAG artifact stage was already committed")
        _assert_parent_binding(
            self.final_path.parent,
            self._parent_descriptor,
            self._parent_identity,
        )
        _assert_entry_identity(
            self._parent_descriptor,
            self._stage_name,
            self._stage_identity,
            require_directory=True,
        )
        if _directory_identity(os.fstat(self._stage_descriptor)) != self._stage_identity:
            raise RagEvidenceContractError("RAG held staging directory inode changed")
        os.fsync(self._stage_descriptor)
        published_identity = _rename_directory_noreplace_at(
            self._parent_descriptor, self._stage_name, self._final_name
        )
        self._active_name = self._final_name
        try:
            if published_identity != self._stage_identity:
                raise RagEvidenceContractError(
                    "RAG publish moved a substituted staging inode"
                )
            _assert_entry_identity(
                self._parent_descriptor,
                self._final_name,
                self._stage_identity,
                require_directory=True,
            )
            _assert_parent_binding(
                self.final_path.parent,
                self._parent_descriptor,
                self._parent_identity,
            )
            os.fsync(self._parent_descriptor)
        except Exception:
            try:
                self.quarantine_name, _ = _quarantine_entry_at(
                    self._parent_descriptor,
                    self._final_name,
                    label=f"{self._final_name}-failed-publish",
                    require_directory=None,
                )
            finally:
                self._active_name = None
            raise
        self.committed = True

    def rollback(self) -> None:
        """Move a just-published tree back to staging after a paired-tree race."""

        if not self.committed:
            return
        if self._closed:
            raise RuntimeError("RAG artifact stage is closed")
        self.quarantine_name, _ = _quarantine_entry_at(
            self._parent_descriptor,
            self._final_name,
            label=f"{self._final_name}-paired-rollback",
            require_directory=None,
        )
        self.committed = False
        self._active_name = None

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            if not self.committed and self._active_name:
                try:
                    self.quarantine_name, _ = _quarantine_entry_at(
                        self._parent_descriptor,
                        self._active_name,
                        label=f"{self._final_name}-cleanup",
                        require_directory=None,
                    )
                except FileNotFoundError:
                    pass
                finally:
                    self._active_name = None
        finally:
            if self._stage_descriptor >= 0:
                os.close(self._stage_descriptor)
                self._stage_descriptor = -1
            os.close(self._parent_descriptor)
            self._closed = True

    def __enter__(self) -> "AtomicRagDirectory":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()


def write_json_noreplace(path: str | Path, value: Any) -> str:
    """Publish a certificate through an inode-bound no-replace rename.

    Failed temporaries and any substituted object actually moved to the final
    name are quarantined as evidence.  Cleanup never unlinks a mutable name.
    """

    target = Path(path).absolute()
    target_name = target.name
    _child_name(target_name)
    payload = canonical_json_bytes(value) + b"\n"
    parent_descriptor, parent_identity = _open_bound_parent(
        target.parent, create=True
    )
    descriptor = -1
    temporary_name = ""
    temporary_identity: tuple[int, int] | None = None
    active_name: str | None = None
    published = False
    try:
        for _ in range(128):
            candidate = f".{target_name}.{secrets.token_hex(12)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                temporary_name = candidate
                active_name = candidate
                temporary_stat = os.fstat(descriptor)
                if not stat.S_ISREG(temporary_stat.st_mode):
                    raise RagEvidenceContractError(
                        "RAG certificate temporary is not regular"
                    )
                temporary_identity = (
                    int(temporary_stat.st_dev), int(temporary_stat.st_ino)
                )
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("cannot allocate a unique RAG certificate temporary")
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write while creating RAG certificate")
            written += count
        os.fsync(descriptor)
        temporary_stat = os.fstat(descriptor)
        if (
            int(temporary_stat.st_dev), int(temporary_stat.st_ino)
        ) != temporary_identity:
            raise RagEvidenceContractError("RAG held certificate inode changed")
        _assert_parent_binding(target.parent, parent_descriptor, parent_identity)
        _assert_entry_identity(
            parent_descriptor,
            temporary_name,
            temporary_identity,
            require_directory=False,
        )
        try:
            published_identity = _rename_directory_noreplace_at(
                parent_descriptor,
                temporary_name,
                target_name,
                require_directory=False,
            )
            active_name = target_name
        except FileExistsError as error:
            raise FileExistsError(f"RAG certificate already exists: {target}") from error
        try:
            if published_identity != temporary_identity:
                raise RagEvidenceContractError(
                    "RAG certificate publish moved a substituted temporary inode"
                )
            _assert_entry_identity(
                parent_descriptor,
                target_name,
                temporary_identity,
                require_directory=False,
            )
            _assert_parent_binding(target.parent, parent_descriptor, parent_identity)
            os.fsync(parent_descriptor)
        except Exception:
            try:
                _quarantine_entry_at(
                    parent_descriptor,
                    target_name,
                    label=f"{target_name}-failed-publish",
                    require_directory=None,
                )
            finally:
                active_name = None
            raise
        published = True
        active_name = None
        return sha256_bytes(payload)
    finally:
        if not published and active_name:
            try:
                _quarantine_entry_at(
                    parent_descriptor,
                    active_name,
                    label=f"{target_name}-temporary",
                    require_directory=None,
                )
            except (FileNotFoundError, RagEvidenceContractError, OSError):
                pass
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


__all__ = [
    "AtomicRagDirectory", "BoundRagFile", "BoundRagSourceAssets",
    "BoundRagTree", "EVALUATION_AB_SCHEMA", "EVALUATION_MANIFEST_FILENAME",
    "EVALUATION_SCHEMA", "FIT_INPUT_FILENAME", "FIT_INPUT_SCHEMA", "PANEL_IDS",
    "PREPARATION_AB_SCHEMA", "PREPARATION_MANIFEST_FILENAME", "PREPARATION_SCHEMA",
    "PRIVATE_LABEL_FILENAME", "PRIVATE_LABEL_SCHEMA", "REFCHECKER_SETTINGS",
    "SOURCE_ASSET_IDS", "EXPECTED_SOURCE_ASSET_ROSTER_SHA256",
    "RagEvidenceContractError", "SCORE_AB_SCHEMA", "SCORE_FREEZE_SCHEMA",
    "SCORE_MANIFEST_FILENAME", "SCORE_SCHEMA", "SCORES_FILENAME", "add_payload_sha256",
    "add_pickle_payload_sha256",
    "assert_physical_tree_independence", "bind_source_assets", "load_fit_input",
    "load_fit_input_bytes", "load_fit_input_handle", "load_pickle", "load_private_labels",
    "load_private_labels_bytes", "load_registry",
    "payload_sha256", "pickle_bytes", "validate_fit_input", "validate_fit_sanitization",
    "validate_artifact_identifier", "validate_private_labels",
    "read_bound_file_bytes", "validate_source_binding", "verify_payload",
    "write_json_noreplace",
]
