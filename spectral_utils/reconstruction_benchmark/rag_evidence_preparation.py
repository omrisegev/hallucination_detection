"""Prepare two independently reconstructable, target-isolated RAG builds."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import stat
import tempfile
from typing import Any

import numpy as np

from spectral_utils.fixed_application_pipelines import condition_trace_row
from spectral_utils.ragtruth_evidence_contrast import (
    adapt_cache,
    load_cache_handle as load_ragtruth_cache_handle,
    load_official_responses_handle,
)

from .io import canonical_json_bytes, sha256_bytes, sha256_file
from .rag_evidence_contract import (
    AtomicRagDirectory,
    BoundRagSourceAssets,
    FIT_INPUT_FILENAME,
    FIT_INPUT_SCHEMA,
    PREPARATION_MANIFEST_FILENAME,
    PREPARATION_SCHEMA,
    PRIVATE_LABEL_FILENAME,
    PRIVATE_LABEL_SCHEMA,
    RagEvidenceContractError,
    _assert_entry_identity,
    _assert_parent_binding,
    _directory_identity,
    _directory_open_flags,
    _entry_identity,
    _open_bound_parent,
    _quarantine_entry_at,
    add_payload_sha256,
    add_pickle_payload_sha256,
    load_registry,
    payload_sha256,
    pickle_bytes,
    read_bound_file_bytes,
    validate_artifact_identifier,
    validate_fit_input,
    validate_private_labels,
    verify_payload,
)


PREPARATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/rag_evidence.json",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/fixed_application_pipelines.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/ragtruth_evidence_contrast.py",
    "spectral_utils/repeated_measurement_reliability.py",
    "spectral_utils/token_feature_views.py",
    "spectral_utils/upcr.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_contract.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_preparation.py",
)
PAIR_TRANSACTION_FILENAME = "PAIR_TRANSACTION.json"
PAIR_TRANSACTION_SCHEMA = "reconstruction-rag-evidence-pair-transaction-v1"


def rag_evidence_pair_transaction(
    *, reconstruction: Mapping[str, Any], release_id: str, build_id: str
) -> dict[str, Any]:
    return add_payload_sha256({
        "schema_version": PAIR_TRANSACTION_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "lane_id": reconstruction["registry"]["lane_id"],
        "fit_input_sha256": reconstruction["fit_input_sha256"],
        "private_label_sha256": reconstruction["private_label_sha256"],
        "source_binding_sha256": reconstruction["source_binding"]["binding_sha256"],
    })


def _preparation_manifest(
    *,
    reconstruction: Mapping[str, Any],
    private_build: Path,
    release_id: str,
    build_id: str,
    scientific_full: bool,
    pair_transaction: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the one canonical manifest accepted by creation and recovery."""

    marker_bytes = canonical_json_bytes(pair_transaction) + b"\n"
    return add_payload_sha256({
        "schema_version": PREPARATION_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "lane_id": reconstruction["registry"]["lane_id"],
        "scientific_full": bool(scientific_full),
        "pair_transaction_id": pair_transaction["payload_sha256"],
        "pair_transaction": {
            "path": PAIR_TRANSACTION_FILENAME,
            "private_path": str(
                (private_build / PAIR_TRANSACTION_FILENAME).absolute()
            ),
            "sha256": sha256_bytes(marker_bytes),
            "size_bytes": len(marker_bytes),
        },
        "fit_input": {
            "path": f"inputs/{FIT_INPUT_FILENAME}",
            "sha256": reconstruction["fit_input_sha256"],
            "size_bytes": len(reconstruction["fit_input_bytes"]),
            "target_fields_present": False,
        },
        "private_labels": {
            "path": str((private_build / PRIVATE_LABEL_FILENAME).absolute()),
            "sha256": reconstruction["private_label_sha256"],
            "size_bytes": len(reconstruction["private_label_bytes"]),
        },
        "source_binding": reconstruction["source_binding"],
        "source_binding_sha256": reconstruction["source_binding"]["binding_sha256"],
        "source_snapshot": reconstruction["source_snapshot"],
        "rosters": reconstruction["rosters"],
        "labels_exposed_to_fit": False,
        "historical_scores_opened": False,
    })


def _recovery_file_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise RagEvidenceContractError("RAG recovery entry is not a regular file")
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _read_recovery_descriptor(descriptor: int) -> bytes:
    blocks: list[bytes] = []
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            break
        blocks.append(block)
        offset += len(block)
    return b"".join(blocks)


class _RecoveryFileBinding:
    """One recovery file held from first exact-byte validation to pair decision."""

    def __init__(
        self, parent_descriptor: int, name: str, *, expected_bytes: bytes
    ) -> None:
        self.parent_descriptor = parent_descriptor
        self.name = name
        self.expected_bytes = expected_bytes
        self.descriptor = -1
        self.identity: tuple[int, int] | None = None
        self.signature: tuple[int, int, int, int, int] | None = None
        try:
            self.descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
            self.signature = _recovery_file_signature(os.fstat(self.descriptor))
            self.identity = self.signature[:2]
            payload = _read_recovery_descriptor(self.descriptor)
            if payload != expected_bytes:
                raise RagEvidenceContractError(
                    f"RAG recovery file differs from retry: {name}"
                )
            if _recovery_file_signature(os.fstat(self.descriptor)) != self.signature:
                raise RagEvidenceContractError(
                    f"RAG recovery file changed during validation: {name}"
                )
            _assert_entry_identity(
                parent_descriptor, name, self.identity, require_directory=False
            )
        except Exception:
            self.close()
            raise

    def verify_stable(self) -> None:
        if self.descriptor < 0 or self.identity is None or self.signature is None:
            raise RuntimeError("RAG recovery file binding is closed")
        if _recovery_file_signature(os.fstat(self.descriptor)) != self.signature:
            raise RagEvidenceContractError(
                f"RAG recovery file inode/content metadata changed: {self.name}"
            )
        if _read_recovery_descriptor(self.descriptor) != self.expected_bytes:
            raise RagEvidenceContractError(
                f"RAG recovery file content changed: {self.name}"
            )
        if _recovery_file_signature(os.fstat(self.descriptor)) != self.signature:
            raise RagEvidenceContractError(
                f"RAG recovery file changed during revalidation: {self.name}"
            )
        _assert_entry_identity(
            self.parent_descriptor,
            self.name,
            self.identity,
            require_directory=False,
        )

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class _RecoveryDirectoryBinding:
    """Held nested recovery directory and its canonical parent-relative name."""

    def __init__(self, parent_descriptor: int, name: str) -> None:
        self.parent_descriptor = parent_descriptor
        self.name = name
        self.descriptor = os.open(
            name, _directory_open_flags(), dir_fd=parent_descriptor
        )
        try:
            self.identity = _directory_identity(os.fstat(self.descriptor))
            _assert_entry_identity(
                parent_descriptor, name, self.identity, require_directory=True
            )
        except Exception:
            self.close()
            raise

    def verify_stable(self) -> None:
        if self.descriptor < 0:
            raise RuntimeError("RAG recovery directory binding is closed")
        if _directory_identity(os.fstat(self.descriptor)) != self.identity:
            raise RagEvidenceContractError(
                f"RAG held recovery directory inode changed: {self.name}"
            )
        _assert_entry_identity(
            self.parent_descriptor,
            self.name,
            self.identity,
            require_directory=True,
        )

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class _RecoveryArtifactBinding:
    """Held parent/object descriptors for one member of a preparation pair."""

    def __init__(self, path: Path) -> None:
        self.path = path.absolute()
        self.parent_descriptor, self.parent_identity = _open_bound_parent(
            self.path.parent, create=True
        )
        self.identity: tuple[int, int] | None = None
        self.child_descriptor = -1
        self.file_bindings: list[_RecoveryFileBinding] = []
        self.directory_bindings: list[_RecoveryDirectoryBinding] = []
        self.rosters: list[tuple[int, frozenset[str], str]] = []
        try:
            try:
                self.identity = _entry_identity(
                    self.parent_descriptor,
                    self.path.name,
                    require_directory=True,
                )
            except FileNotFoundError:
                return
            self.child_descriptor = os.open(
                self.path.name,
                _directory_open_flags(),
                dir_fd=self.parent_descriptor,
            )
            if _directory_identity(os.fstat(self.child_descriptor)) != self.identity:
                raise RagEvidenceContractError(
                    "RAG recovery artifact inode changed while opening"
                )
        except Exception:
            self.close()
            raise

    def bind_roster(
        self, descriptor: int, expected: set[str], *, name: str
    ) -> None:
        frozen = frozenset(expected)
        if frozenset(os.listdir(descriptor)) != frozen:
            raise RagEvidenceContractError(f"RAG {name} file roster drifted")
        self.rosters.append((descriptor, frozen, name))

    def bind_directory(
        self, parent_descriptor: int, name: str
    ) -> _RecoveryDirectoryBinding:
        held = _RecoveryDirectoryBinding(parent_descriptor, name)
        self.directory_bindings.append(held)
        return held

    def bind_file(
        self, parent_descriptor: int, name: str, *, expected_bytes: bytes
    ) -> bytes:
        held = _RecoveryFileBinding(
            parent_descriptor, name, expected_bytes=expected_bytes
        )
        self.file_bindings.append(held)
        return expected_bytes

    def verify_descendants(self) -> None:
        if self.identity is None or self.child_descriptor < 0:
            raise RagEvidenceContractError("RAG recovery artifact is absent")
        if _directory_identity(os.fstat(self.child_descriptor)) != self.identity:
            raise RagEvidenceContractError("RAG held recovery artifact inode changed")
        for held in self.directory_bindings:
            held.verify_stable()
        for descriptor, expected, name in self.rosters:
            if frozenset(os.listdir(descriptor)) != expected:
                raise RagEvidenceContractError(f"RAG {name} file roster drifted")
        for held in self.file_bindings:
            held.verify_stable()
        for descriptor, expected, name in self.rosters:
            if frozenset(os.listdir(descriptor)) != expected:
                raise RagEvidenceContractError(f"RAG {name} file roster drifted")
        for held in self.directory_bindings:
            held.verify_stable()

    def close(self) -> None:
        for held in reversed(self.file_bindings):
            held.close()
        self.file_bindings.clear()
        for held in reversed(self.directory_bindings):
            held.close()
        self.directory_bindings.clear()
        self.rosters.clear()
        if self.child_descriptor >= 0:
            os.close(self.child_descriptor)
            self.child_descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1


def _validate_recovery_artifact(
    binding: _RecoveryArtifactBinding,
    *,
    kind: str,
    expected_marker_bytes: bytes,
    expected_manifest_bytes: bytes,
    reconstruction: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate one artifact only through its already-held directory fd."""

    if binding.identity is None or binding.child_descriptor < 0:
        raise RagEvidenceContractError(f"RAG {kind} recovery artifact is absent")
    if _directory_identity(os.fstat(binding.child_descriptor)) != binding.identity:
        raise RagEvidenceContractError("RAG held recovery artifact inode changed")
    try:
        binding.bind_file(
            binding.child_descriptor,
            PAIR_TRANSACTION_FILENAME,
            expected_bytes=expected_marker_bytes,
        )
    except RagEvidenceContractError as error:
        raise RagEvidenceContractError(
            "RAG orphan pair marker differs from retry"
        ) from error
    recovered_manifest: dict[str, Any] | None = None
    if kind == "private":
        binding.bind_roster(
            binding.child_descriptor,
            {PRIVATE_LABEL_FILENAME, PAIR_TRANSACTION_FILENAME},
            name="private orphan",
        )
        try:
            binding.bind_file(
                binding.child_descriptor,
                PRIVATE_LABEL_FILENAME,
                expected_bytes=reconstruction["private_label_bytes"],
            )
        except RagEvidenceContractError as error:
            raise RagEvidenceContractError(
                "RAG private orphan payload differs from retry"
            ) from error
    elif kind == "public":
        binding.bind_roster(
            binding.child_descriptor,
            {"inputs", PREPARATION_MANIFEST_FILENAME, PAIR_TRANSACTION_FILENAME},
            name="public orphan",
        )
        inputs = binding.bind_directory(binding.child_descriptor, "inputs")
        binding.bind_roster(
            inputs.descriptor, {FIT_INPUT_FILENAME}, name="public orphan input"
        )
        try:
            binding.bind_file(
                inputs.descriptor,
                FIT_INPUT_FILENAME,
                expected_bytes=reconstruction["fit_input_bytes"],
            )
        except RagEvidenceContractError as error:
            raise RagEvidenceContractError(
                "RAG public orphan fit input differs from retry"
            ) from error
        try:
            manifest_bytes = binding.bind_file(
                binding.child_descriptor,
                PREPARATION_MANIFEST_FILENAME,
                expected_bytes=expected_manifest_bytes,
            )
        except RagEvidenceContractError as error:
            raise RagEvidenceContractError(
                "RAG recovered preparation manifest is not the exact "
                "registered-source reconstruction"
            ) from error
        recovered_manifest = json.loads(manifest_bytes)
        verify_payload(
            recovered_manifest, name="RAG recovered preparation manifest"
        )
    else:
        raise ValueError(kind)
    if _directory_identity(os.fstat(binding.child_descriptor)) != binding.identity:
        raise RagEvidenceContractError("RAG held recovery artifact inode changed")
    binding.verify_descendants()
    return recovered_manifest


def _observed_recovery_identity(
    binding: _RecoveryArtifactBinding,
) -> tuple[int, int] | None:
    try:
        return _entry_identity(
            binding.parent_descriptor,
            binding.path.name,
            require_directory=None,
        )
    except FileNotFoundError:
        return None


def _quarantine_recovery_canonical_entries(
    bindings: tuple[_RecoveryArtifactBinding, _RecoveryArtifactBinding],
) -> None:
    """Preserve and clear every current canonical pair entry after drift."""

    for _ in range(128):
        found = False
        for binding in bindings:
            if _observed_recovery_identity(binding) is None:
                continue
            found = True
            _quarantine_entry_at(
                binding.parent_descriptor,
                binding.path.name,
                label=f"{binding.path.name}-recovery-binding-drift",
                require_directory=None,
            )
        if not found:
            for binding in bindings:
                _assert_parent_binding(
                    binding.path.parent,
                    binding.parent_descriptor,
                    binding.parent_identity,
                )
                if _observed_recovery_identity(binding) is not None:
                    break
            else:
                return
    raise RagEvidenceContractError(
        "RAG canonical recovery entries could not be stably quarantined"
    )


def _reassert_recovery_pair(
    bindings: tuple[_RecoveryArtifactBinding, _RecoveryArtifactBinding],
    expected: tuple[tuple[int, int] | None, tuple[int, int] | None],
    *,
    context: str,
) -> None:
    """Bind both canonical names together immediately before recovery success."""

    try:
        for binding, expected_identity in zip(bindings, expected, strict=True):
            _assert_parent_binding(
                binding.path.parent,
                binding.parent_descriptor,
                binding.parent_identity,
            )
            if expected_identity is not None:
                if (
                    _directory_identity(os.fstat(binding.child_descriptor))
                    != expected_identity
                ):
                    raise RagEvidenceContractError(
                        "RAG held recovery artifact inode changed"
                    )
                binding.verify_descendants()
    except Exception as error:
        _quarantine_recovery_canonical_entries(bindings)
        if any(_observed_recovery_identity(item) is not None for item in bindings):
            raise RagEvidenceContractError(
                "RAG recovery canonical entries remained after descendant-drift "
                "quarantine"
            ) from error
        raise RagEvidenceContractError(
            f"RAG preparation pair descendant drifted during {context}"
        ) from error
    observed = tuple(_observed_recovery_identity(item) for item in bindings)
    if observed == expected:
        return
    _quarantine_recovery_canonical_entries(bindings)
    if any(_observed_recovery_identity(item) is not None for item in bindings):
        raise RagEvidenceContractError(
            "RAG recovery canonical entries remained after quarantine"
        )
    raise RagEvidenceContractError(
        f"RAG preparation pair binding drifted during {context}"
    )


def _recover_incomplete_pair(
    *, release_build: Path, private_build: Path,
    marker: Mapping[str, Any], reconstruction: Mapping[str, Any],
    release_id: str, build_id: str, scientific_full: bool,
) -> dict[str, Any] | None:
    expected_manifest = _preparation_manifest(
        reconstruction=reconstruction,
        private_build=private_build,
        release_id=release_id,
        build_id=build_id,
        scientific_full=scientific_full,
        pair_transaction=marker,
    )
    expected_manifest_bytes = canonical_json_bytes(expected_manifest) + b"\n"
    marker_bytes = canonical_json_bytes(marker) + b"\n"
    public_binding = _RecoveryArtifactBinding(release_build)
    try:
        private_binding = _RecoveryArtifactBinding(private_build)
    except Exception:
        public_binding.close()
        raise
    bindings = (public_binding, private_binding)
    try:
        expected = (public_binding.identity, private_binding.identity)
        public_exists = public_binding.identity is not None
        private_exists = private_binding.identity is not None
        if public_exists and private_exists:
            manifest = _validate_recovery_artifact(
                public_binding,
                kind="public",
                expected_marker_bytes=marker_bytes,
                expected_manifest_bytes=expected_manifest_bytes,
                reconstruction=reconstruction,
            )
            _validate_recovery_artifact(
                private_binding,
                kind="private",
                expected_marker_bytes=marker_bytes,
                expected_manifest_bytes=expected_manifest_bytes,
                reconstruction=reconstruction,
            )
            _reassert_recovery_pair(
                bindings, expected, context="complete-pair validation"
            )
            if manifest is None:
                raise RagEvidenceContractError(
                    "RAG complete pair has no public manifest"
                )
            return manifest
        if not public_exists and not private_exists:
            _reassert_recovery_pair(
                bindings, expected, context="empty-pair validation"
            )
            return None
        orphan = public_binding if public_exists else private_binding
        kind = "public" if public_exists else "private"
        _validate_recovery_artifact(
            orphan,
            kind=kind,
            expected_marker_bytes=marker_bytes,
            expected_manifest_bytes=expected_manifest_bytes,
            reconstruction=reconstruction,
        )
        _reassert_recovery_pair(
            bindings, expected, context=f"{kind}-orphan validation"
        )
        _, moved_identity = _quarantine_entry_at(
            orphan.parent_descriptor,
            orphan.path.name,
            label=f"{orphan.path.name}-recovered-{kind}-orphan",
            require_directory=None,
        )
        if moved_identity != orphan.identity:
            _quarantine_recovery_canonical_entries(bindings)
            raise RagEvidenceContractError(
                "RAG orphan source was substituted during recovery quarantine"
            )
        _reassert_recovery_pair(
            bindings, (None, None), context=f"{kind}-orphan mutation"
        )
        return None
    finally:
        private_binding.close()
        public_binding.close()


def _load_pickle_bytes(payload: bytes) -> Any:
    """Parse only bytes already authenticated from a held source descriptor."""

    return pickle.loads(payload)


def _condition_from_row(row: Mapping[str, Any], *, include_exact_jsd: bool = False) -> dict[str, Any]:
    output = {
        "token_entropies": np.asarray(row["token_entropies"], dtype=np.float64),
        "token_spilled_energies": np.asarray(row["token_spilled_energies"], dtype=np.float64),
        "token_logsumexp": np.asarray(row["token_logsumexp"], dtype=np.float64),
        "top_k_logprobs": {
            "ids": np.asarray(row["top_k_logprobs"]["ids"], dtype=np.int64),
            "logprobs": np.asarray(row["top_k_logprobs"]["logprobs"], dtype=np.float64),
        },
    }
    if include_exact_jsd and row.get("token_jsd_vs_full") is not None:
        output["token_jsd_vs_full"] = np.asarray(row["token_jsd_vs_full"], dtype=np.float64)
    return output


def _condition_from_trace(trace: Any) -> dict[str, Any]:
    raw = condition_trace_row(trace)
    return {
        "token_entropies": np.asarray(raw["token_entropies"], dtype=np.float64),
        "token_spilled_energies": np.asarray(raw["token_spilled_energies"], dtype=np.float64),
        "token_logsumexp": np.asarray(raw["token_logsumexp"], dtype=np.float64),
        "top_k_logprobs": {
            "ids": np.asarray(raw["top_k_logprobs"]["ids"], dtype=np.int64),
            "logprobs": np.asarray(raw["top_k_logprobs"]["logprobs"], dtype=np.float64),
        },
    }


def _token_targets(response: Any, official: Mapping[str, Any]) -> np.ndarray:
    labels = list(official.get("labels") or [])
    output = np.zeros(len(response.token_offsets), dtype=np.uint8)
    for index, (start, end) in enumerate(response.token_offsets):
        output[index] = int(any(
            int(end) > int(span["start"]) and int(start) < int(span["end"])
            for span in labels
        ))
    return output


def _prepare_ragtruth(
    *, sources: BoundRagSourceAssets, registry: Mapping[str, Any], tokenizer: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    official_asset = sources["ragtruth_official_response"]
    official = load_official_responses_handle(BytesIO(official_asset.read_bytes()))
    fit_splits: dict[str, list[dict[str, Any]]] = {}
    private_splits: dict[str, list[dict[str, Any]]] = {}
    roster: dict[str, Any] = {}
    for split, asset_id in (("dev", "ragtruth_dev_cache"), ("test", "ragtruth_test_cache")):
        cache_asset = sources[asset_id]
        manifest_asset = sources[f"ragtruth_{split}_manifest"]
        raw_cache = load_ragtruth_cache_handle(BytesIO(cache_asset.read_bytes()))
        sidecar_manifest = json.loads(manifest_asset.read_bytes().decode("utf-8"))
        cache_path = Path(registry["sources"][asset_id]["path"])
        official_path = Path(
            registry["sources"]["ragtruth_official_response"]["path"]
        )
        dataset, labels, diagnostics = adapt_cache(
            cache_path,
            official_path,
            tokenizer,
            raw_cache=raw_cache,
            official_responses=official,
            sidecar_manifest=sidecar_manifest,
            cache_sha256=cache_asset.sha256,
            sidecar_manifest_sha256=manifest_asset.sha256,
        )
        fit_rows: list[dict[str, Any]] = []
        private_rows: list[dict[str, Any]] = []
        for index, response in enumerate(dataset.responses):
            unit_id = f"rt_{split}_{index:06d}"
            sentence_windows = []
            sentence_targets = []
            for sentence in response.sentences:
                sentence_id = f"{unit_id}_s{sentence.index:04d}"
                sentence_windows.append({
                    "unit_id": sentence_id,
                    "start": int(sentence.token_start),
                    "end": int(sentence.token_end),
                })
                sentence_targets.append({
                    "unit_id": sentence_id,
                    "label": int(labels.sentence[f"{response.response_id}::sent_{sentence.index}"].hallucinated),
                })
            fit_rows.append({
                "unit_id": unit_id,
                "task_type": response.task_type,
                "conditions": {
                    name: _condition_from_trace(trace)
                    for name, trace in sorted(response.conditions.items())
                },
                "sentence_windows": sentence_windows,
            })
            private_rows.append({
                "unit_id": unit_id,
                "source_id": response.source_id,
                "task_type": response.task_type,
                "response_label": int(labels.response[response.response_id].hallucinated),
                "sentence_labels": sentence_targets,
                "token_labels": _token_targets(response, official[response.response_id]),
            })
        expected = registry["expected_rosters"][f"ragtruth_{split}"]
        observed = {
            "conditions": int(diagnostics["n_conditions"]),
            "responses": len(fit_rows),
            "sources": len({row["source_id"] for row in private_rows}),
        }
        if observed != expected:
            raise RagEvidenceContractError(
                f"RAGTruth {split} roster drifted: expected={expected}, observed={observed}"
            )
        fit_splits[split] = fit_rows
        private_splits[split] = private_rows
        roster[split] = {
            **observed,
            "sentences": sum(len(row["sentence_windows"]) for row in fit_rows),
            "tokens": sum(len(row["token_labels"]) for row in private_rows),
        }
    return {"splits": fit_splits}, {"splits": private_splits}, roster


def _token_windows_for_gasp(row: Mapping[str, Any], tokenizer: Any) -> list[tuple[int, int]]:
    encoded = tokenizer(
        str(row["response"]), add_special_tokens=False, return_offsets_mapping=True
    )
    observed = np.asarray(encoded["input_ids"], dtype=np.int64)
    expected = np.asarray(row["gen_token_ids"], dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise RagEvidenceContractError(
            f"GASP response {row['response_id']} does not reproduce stored tokens"
        )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    output: list[tuple[int, int]] = []
    for char_start, char_end in row["sentence_spans"]:
        indexes = [
            index for index, (start, end) in enumerate(offsets)
            if end > int(char_start) and start < int(char_end)
        ]
        if indexes:
            output.append((min(indexes), max(indexes) + 1))
        else:
            output.append((-1, -1))
    return output


def _prepare_gasp(
    *, sources: BoundRagSourceAssets, registry: Mapping[str, Any], tokenizer: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = load_ragtruth_cache_handle(BytesIO(sources["gasp_cache"].read_bytes()))
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in raw.values():
        grouped[str(row["response_id"])][str(row["condition"])] = row
    fit_rows, private_rows = [], []
    for response_index, response_id in enumerate(sorted(grouped, key=int)):
        conditions = grouped[response_id]
        if "full" not in conditions or "noctx" not in conditions:
            raise RagEvidenceContractError(f"GASP response lacks full/noctx: {response_id}")
        full = conditions["full"]
        windows = _token_windows_for_gasp(full, tokenizer)
        response_unit_id = f"gasp_r{response_index:04d}"
        fit_windows, private_sentences = [], []
        for sentence_index, ((char_start, char_end), (start, end)) in enumerate(
            zip(full["sentence_spans"], windows, strict=True)
        ):
            if start < 0:
                continue
            unit_id = f"{response_unit_id}_s{sentence_index:04d}"
            fit_windows.append({"unit_id": unit_id, "start": start, "end": end})
            hallucinated = int(any(
                max(int(char_start), int(item["start"]))
                < min(int(char_end), int(item["end"]))
                for item in full.get("span_labels", [])
            ))
            private_sentences.append({
                "unit_id": unit_id,
                "source_id": str(full["source_id"]),
                "task_type": str(full["task_type"]),
                "label": hallucinated,
            })
        fit_rows.append({
            "response_unit_id": response_unit_id,
            "task_type": str(full["task_type"]),
            "conditions": {
                name: _condition_from_row(row, include_exact_jsd=True)
                for name, row in sorted(conditions.items())
            },
            "sentence_windows": fit_windows,
        })
        private_rows.extend(private_sentences)
    expected = registry["expected_rosters"]["gasp"]
    observed = {
        "conditions": len(raw),
        "responses": len(fit_rows),
        "sources": len({row["source_id"] for row in private_rows}),
    }
    if observed != expected:
        raise RagEvidenceContractError(f"GASP roster drifted: expected={expected}, observed={observed}")
    roster = {**observed, "sentences": len(private_rows)}
    return {"rows": fit_rows}, {"sentences": private_rows}, roster


def _prepare_lettuce(
    *, sources: BoundRagSourceAssets, registry: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = _load_pickle_bytes(sources["lettuce_cache"].read_bytes())
    if not isinstance(raw, Mapping):
        raise RagEvidenceContractError("Lettuce cache root is not a mapping")
    ordered = sorted(raw.values(), key=lambda row: int(row["response_id"]))
    fit_rows, private_rows = [], []
    for index, row in enumerate(ordered):
        unit_id = f"lettuce_{index:06d}"
        fit_rows.append({
            "unit_id": unit_id,
            "task_type": str(row["task_type"]),
            "binary_prediction": int(bool(row["pred_hallucinated"])),
            "maximum_token_probability": float(max(row.get("token_probs") or [0.0])),
            "truncated": int(bool(row["truncated"])),
        })
        private_rows.append({
            "unit_id": unit_id,
            "source_id": str(row["source_id"]),
            "task_type": str(row["task_type"]),
            "label": int(bool(row["gold_hallucinated"])),
        })
    expected = registry["expected_rosters"]["lettuce"]
    target_audit = {
        "examples": len(fit_rows),
        "gold_positive": sum(row["label"] for row in private_rows),
        "truncated": sum(row["truncated"] for row in fit_rows),
    }
    if target_audit != expected:
        raise RagEvidenceContractError(
            f"Lettuce roster drifted: expected={expected}, observed={target_audit}"
        )
    # The fit-visible roster is deliberately target-free.  Gold prevalence is
    # retained only in the private bundle and post-freeze evaluator.
    public_roster = {
        "examples": len(fit_rows),
        "truncated": sum(row["truncated"] for row in fit_rows),
    }
    return (
        {"rows": fit_rows},
        {"rows": private_rows, "target_audit": target_audit},
        public_roster,
    )


def _prepare_refchecker(
    *, sources: BoundRagSourceAssets, registry: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = _load_pickle_bytes(sources["refchecker_cache"].read_bytes())
    predictions = json.loads(
        sources["refchecker_nli_predictions"].read_bytes().decode("utf-8")
    )
    by_key = {str(row["claim_key"]): row for row in predictions}
    if len(by_key) != len(predictions):
        raise RagEvidenceContractError("duplicate RefChecker NLI prediction key")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in raw.values():
        claim_key = "|".join((
            str(row["setting"]), str(row["generator"]), str(row["example_id"]),
            str(int(row["claim_index"])),
        ))
        grouped[claim_key][str(row["condition"])] = row
    fit_rows, private_rows = [], []
    for index, claim_key in enumerate(sorted(grouped)):
        conditions = grouped[claim_key]
        if set(conditions) != {"full", "noctx"}:
            raise RagEvidenceContractError(f"RefChecker claim pair is incomplete: {claim_key}")
        full = conditions["full"]
        prediction = by_key.get(claim_key)
        if prediction is None:
            raise RagEvidenceContractError(f"RefChecker NLI prediction is absent: {claim_key}")
        if str(prediction["human_label"]) != str(full["human_label"]):
            raise RagEvidenceContractError(f"RefChecker NLI/gold join failed: {claim_key}")
        unit_id = f"ref_{index:06d}"
        fit_rows.append({
            "unit_id": unit_id,
            "setting": str(full["setting"]),
            "generator": str(full["generator"]),
            "conditions": {
                name: _condition_from_row(row)
                for name, row in sorted(conditions.items())
            },
            "nli_prediction": str(prediction["predicted_label"]),
        })
        private_rows.append({
            "unit_id": unit_id,
            "example_id": str(full["example_id"]),
            "setting": str(full["setting"]),
            "human_label": str(full["human_label"]),
            "label_unsupported": int(full["label_unsupported"]),
        })
    expected = registry["expected_rosters"]["refchecker"]
    setting_counts = Counter(row["setting"] for row in private_rows)
    observed = {
        "conditions": len(raw),
        "claims": len(fit_rows),
        "settings": {name: setting_counts[name] for name in expected["settings"]},
    }
    if observed != expected:
        raise RagEvidenceContractError(
            f"RefChecker roster drifted: expected={expected}, observed={observed}"
        )
    if set(by_key) != set(grouped):
        raise RagEvidenceContractError("RefChecker NLI prediction roster differs from telemetry")
    return {"rows": fit_rows}, {"rows": private_rows}, observed


def reconstruct_rag_evidence_preparation(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    include_payloads: bool = True,
) -> dict[str, Any]:
    """Rederive the complete sanitized/private split from hash-pinned raw assets."""

    repo_path = Path(repo).resolve(strict=True)
    registry_file = Path(registry_path).absolute()
    registry = load_registry(registry_file)
    root = Path(source_root).absolute()

    from transformers import AutoTokenizer

    with BoundRagSourceAssets(root, registry) as sources, tempfile.TemporaryDirectory(
        prefix="rag-bound-tokenizer-"
    ) as tokenizer_directory:
        tokenizer_root = Path(tokenizer_directory)
        for asset_id in (
            "tokenizer_config",
            "tokenizer_model_config",
            "tokenizer_json",
            "tokenizer_merges",
            "tokenizer_vocab",
        ):
            filename = Path(registry["sources"][asset_id]["path"]).name
            (tokenizer_root / filename).write_bytes(sources[asset_id].read_bytes())
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_root), local_files_only=True
        )
        rag_fit, rag_private, rag_roster = _prepare_ragtruth(
            sources=sources, registry=registry, tokenizer=tokenizer
        )
        gasp_fit, gasp_private, gasp_roster = _prepare_gasp(
            sources=sources, registry=registry, tokenizer=tokenizer
        )
        lettuce_fit, lettuce_private, lettuce_roster = _prepare_lettuce(
            sources=sources, registry=registry
        )
        ref_fit, ref_private, ref_roster = _prepare_refchecker(
            sources=sources, registry=registry
        )
        sources.verify_stable()
        source_binding = dict(sources.binding)
    rosters = {
        "ragtruth": rag_roster,
        "gasp": gasp_roster,
        "lettuce": lettuce_roster,
        "refchecker": ref_roster,
    }
    fit_input = add_pickle_payload_sha256({
        "schema_version": FIT_INPUT_SCHEMA,
        "lane_id": registry["lane_id"],
        "contract_version": registry["method_contract"]["fixed_rag_iu_pcr"]["feature_contract"],
        "panels": {
            "ragtruth": rag_fit,
            "gasp": gasp_fit,
            "lettuce": lettuce_fit,
            "refchecker": ref_fit,
        },
        "rosters": rosters,
        "source_asset_roster_sha256": source_binding["asset_roster_sha256"],
        "historical_scores_opened": False,
        "targets_opened_by_fit": False,
    })
    private_labels = add_pickle_payload_sha256({
        "schema_version": PRIVATE_LABEL_SCHEMA,
        "lane_id": registry["lane_id"],
        "ragtruth": rag_private,
        "gasp": gasp_private,
        "lettuce": lettuce_private,
        "refchecker": ref_private,
        "rosters": rosters,
        "private_target_audit": {
            "lettuce": lettuce_private["target_audit"],
        },
        "source_asset_roster_sha256": source_binding["asset_roster_sha256"],
    })
    validate_fit_input(fit_input, registry)
    validate_private_labels(private_labels, registry)
    fit_bytes = pickle_bytes(fit_input)
    fit_sha256 = hashlib.sha256(fit_bytes).hexdigest()
    fit_size = len(fit_bytes)
    retained_fit_bytes = fit_bytes if include_payloads else None
    if not include_payloads:
        del fit_bytes
    private_bytes = pickle_bytes(private_labels)
    private_sha256 = hashlib.sha256(private_bytes).hexdigest()
    private_size = len(private_bytes)
    retained_private_bytes = private_bytes if include_payloads else None
    if not include_payloads:
        del private_bytes
    registry_relative = (
        registry_file.relative_to(repo_path).as_posix()
        if registry_file.is_relative_to(repo_path)
        else str(registry_file)
    )
    source_snapshot = {
        "registry": {
            "path": registry_relative,
            "sha256": sha256_bytes(
                read_bound_file_bytes(registry_file, name="RAG registry snapshot")
            ),
        },
        "files": [
            {
                "path": relative,
                "sha256": sha256_bytes(
                    read_bound_file_bytes(
                        repo_path / relative,
                        name=f"RAG preparation source {relative}",
                    )
                ),
            }
            for relative in PREPARATION_SOURCE_FILES
        ],
    }
    source_snapshot["snapshot_sha256"] = payload_sha256(source_snapshot)
    output = {
        "registry": registry,
        "fit_input_sha256": fit_sha256,
        "fit_input_size_bytes": fit_size,
        "private_label_sha256": private_sha256,
        "private_label_size_bytes": private_size,
        "source_binding": source_binding,
        "source_snapshot": source_snapshot,
        "rosters": rosters,
    }
    if include_payloads:
        output.update({
            "fit_input": fit_input,
            "private_labels": private_labels,
            "fit_input_bytes": retained_fit_bytes,
            "private_label_bytes": retained_private_bytes,
        })
    return output


def prepare_rag_evidence_build(
    *,
    repo: str | Path,
    registry_path: str | Path,
    source_root: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    build_id: str,
    scientific_full: bool,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    if build_id not in {"A", "B"}:
        raise RagEvidenceContractError("RAG preparation build must be A or B")
    reconstruction = reconstruct_rag_evidence_preparation(
        repo=repo, registry_path=registry_path, source_root=source_root
    )
    release_build = Path(release_root) / release_id / "rag_evidence" / build_id
    private_build = Path(private_root) / release_id / "rag_evidence" / build_id
    pair_transaction = rag_evidence_pair_transaction(
        reconstruction=reconstruction,
        release_id=release_id,
        build_id=build_id,
    )
    existing_manifest = _recover_incomplete_pair(
        release_build=release_build,
        private_build=private_build,
        marker=pair_transaction,
        reconstruction=reconstruction,
        release_id=release_id,
        build_id=build_id,
        scientific_full=bool(scientific_full),
    )
    if existing_manifest is not None:
        return existing_manifest
    public_stage = AtomicRagDirectory(release_build)
    private_stage = AtomicRagDirectory(private_build)
    try:
        fit_sha = public_stage.write_bytes(
            Path("inputs") / FIT_INPUT_FILENAME,
            reconstruction["fit_input_bytes"],
        )
        private_sha = private_stage.write_bytes(
            PRIVATE_LABEL_FILENAME,
            reconstruction["private_label_bytes"],
        )
        pair_marker_sha = public_stage.write_json(
            PAIR_TRANSACTION_FILENAME, pair_transaction
        )
        private_pair_marker_sha = private_stage.write_json(
            PAIR_TRANSACTION_FILENAME, pair_transaction
        )
        expected_marker_sha = sha256_bytes(
            canonical_json_bytes(pair_transaction) + b"\n"
        )
        if (
            pair_marker_sha != expected_marker_sha
            or private_pair_marker_sha != expected_marker_sha
            or fit_sha != reconstruction["fit_input_sha256"]
            or private_sha != reconstruction["private_label_sha256"]
        ):
            raise RagEvidenceContractError(
                "RAG prepared payload digest diverged from reconstruction"
            )
        manifest = _preparation_manifest(
            reconstruction=reconstruction,
            private_build=private_build,
            release_id=release_id,
            build_id=build_id,
            scientific_full=bool(scientific_full),
            pair_transaction=pair_transaction,
        )
        public_stage.write_json(PREPARATION_MANIFEST_FILENAME, manifest)
        private_stage.commit()
        try:
            public_stage.commit()
        except Exception:
            private_stage.rollback()
            raise
        return manifest
    finally:
        public_stage.cleanup()
        private_stage.cleanup()


__all__ = [
    "PAIR_TRANSACTION_FILENAME", "PREPARATION_SOURCE_FILES",
    "prepare_rag_evidence_build", "rag_evidence_pair_transaction",
    "reconstruct_rag_evidence_preparation",
]
