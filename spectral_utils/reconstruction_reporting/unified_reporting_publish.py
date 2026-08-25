"""No-clobber publication and A/B verification for unified reporting v1."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

from .unified_reporting_bridge import BRIDGE_SCHEMA, advisor_inputs, build_unified_rows
from .unified_reporting_schemas import (
    SCHEMA_VERSION,
    TABLE_SCHEMAS,
    UnifiedReportingError,
    assert_csv_parquet_parity,
    canonical_json_bytes,
    canonical_sha256,
    csv_bytes,
    parquet_bytes,
    read_csv_bytes,
    schema_bundle,
)
from .unified_reporting_sources import (
    AuthenticatedSource,
    authenticate_sources,
    load_contract,
    load_source_lock,
    parse_contract_bytes,
    parse_source_lock_bytes,
    validate_contract_source_lock,
)


MANIFEST_SCHEMA = "reconstruction-unified-reporting-manifest-v1"
COMPLETION_SCHEMA = "reconstruction-unified-reporting-completion-v1"
AB_CERTIFICATE_SCHEMA = "reconstruction-unified-reporting-ab-verification-v1"

ROOT_FILES = frozenset(
    {
        "REPORT_CONTRACT.json",
        "SOURCE_LOCK.json",
        "TABLE_SCHEMAS.json",
        "ADVISOR_INPUTS.json",
        "BRIDGE_MANIFEST.json",
        "RELEASE_COMPLETE.json",
    }
)
TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class VerifiedUnifiedRelease:
    root: Path
    root_device: int
    root_inode: int
    manifest: Mapping[str, Any]
    completion: Mapping[str, Any]
    manifest_file_sha256: str
    completion_file_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _token(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        raise UnifiedReportingError(f"{where} is not a safe stable identifier")
    return value


def _with_hash(value: Mapping[str, Any], *, field: str = "payload_sha256") -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise UnifiedReportingError(f"self-hash field already exists: {field}")
    result[field] = canonical_sha256(result)
    return result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _safe_parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise UnifiedReportingError(f"unsafe release-relative path: {relative!r}")
    return path.parts


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise UnifiedReportingError("descriptor-safe directory traversal is unsupported")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_read_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise UnifiedReportingError("descriptor-safe regular-file reads are unsupported")
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


def _open_inventory_file(directory_fd: int, name: str) -> int:
    """Open an inventory file, tolerating APFS rejection of nonblocking large-file opens."""

    try:
        return os.open(name, _file_read_flags(), dir_fd=directory_fd)
    except PermissionError as exc:
        if sys.platform != "darwin" or exc.errno != errno.EPERM:
            raise
        return os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd,
        )


def _exclusive_write_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise UnifiedReportingError("descriptor-safe regular-file creation is unsupported")
    return os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _stable_stat(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode), int(info.st_nlink), int(info.st_size),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def _rename_stable_stat(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """State that must survive rename; POSIX permits rename to advance ctime."""

    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode), int(info.st_nlink),
        int(info.st_size), int(info.st_mtime_ns),
    )


def _open_directory_path_nofollow(path: str | Path, *, where: str) -> tuple[Path, int, os.stat_result]:
    """Open every component without following symlinks and return the held directory."""

    requested = Path(os.path.abspath(os.fspath(path)))
    parts = requested.parts
    if not parts or not requested.is_absolute():
        raise UnifiedReportingError(f"{where} must resolve to an absolute directory path")
    descriptor: int | None = None
    try:
        descriptor = os.open(requested.anchor, _directory_flags())
        for component in parts[1:]:
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise UnifiedReportingError(f"{where} is not a real directory")
        return requested, descriptor, info
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise UnifiedReportingError(f"cannot safely open {where} {requested}: {exc}") from exc


def _safe_name(name: str, *, where: str) -> str:
    if "/" in name or name in {"", ".", ".."}:
        raise UnifiedReportingError(f"unsafe {where}: {name!r}")
    return name


def _open_parent_for_new_path(path: str | Path, *, where: str) -> tuple[Path, Path, str, int, os.stat_result]:
    requested = Path(os.path.abspath(os.fspath(path)))
    name = _safe_name(requested.name, where=f"{where} name")
    parent, descriptor, info = _open_directory_path_nofollow(
        requested.parent, where=f"{where} parent",
    )
    return requested, parent, name, descriptor, info


def _assert_directory_path_identity(path: Path, expected: os.stat_result, *, where: str) -> None:
    _, descriptor, observed = _open_directory_path_nofollow(path, where=where)
    try:
        if _identity(observed) != _identity(expected):
            raise UnifiedReportingError(f"{where} changed identity")
    finally:
        os.close(descriptor)


def _mkdir_owned_fd(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    name = _safe_name(name, where="output directory name")
    descriptor: int | None = None
    try:
        os.mkdir(name, mode=0o755, dir_fd=parent_fd)
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise UnifiedReportingError(f"created output path is not a directory: {name}")
        os.fsync(parent_fd)
        return descriptor, info
    except FileExistsError as exc:
        raise UnifiedReportingError(f"release directory already exists (no clobber): {name}") from exc
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise UnifiedReportingError(f"cannot create release directory {name}: {exc}") from exc


def _write_owned_fd(directory_fd: int, name: str, payload: bytes) -> os.stat_result:
    name = _safe_name(name, where="output filename")
    file_flags = _exclusive_write_flags()
    descriptor: int | None = None
    try:
        descriptor = os.open(name, file_flags, 0o644, dir_fd=directory_fd)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise UnifiedReportingError(f"short write for {name}")
            view = view[written:]
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise UnifiedReportingError(f"created output path is not a regular file: {name}")
        os.close(descriptor)
        descriptor = None
        os.fsync(directory_fd)
        return info
    except FileExistsError as exc:
        raise UnifiedReportingError(f"release file already exists (no clobber): {name}") from exc
    except OSError as exc:
        raise UnifiedReportingError(f"cannot write release file {name}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _create_staging_directory(parent_fd: int, final_name: str) -> tuple[str, int, os.stat_result]:
    for _ in range(64):
        stage_name = f".{final_name}.staging-{secrets.token_hex(12)}"
        try:
            descriptor, info = _mkdir_owned_fd(parent_fd, stage_name)
            return stage_name, descriptor, info
        except UnifiedReportingError as exc:
            if "already exists" not in str(exc):
                raise
    raise UnifiedReportingError("cannot allocate a unique same-parent staging directory")


def _rename_noreplace(parent_fd: int, source_name: str, target_name: str) -> None:
    """Atomically rename within one directory and fail if the target exists."""

    source = os.fsencode(_safe_name(source_name, where="staging entry"))
    target = os.fsencode(_safe_name(target_name, where="publication entry"))
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        function = libc.renameatx_np
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, source, parent_fd, target, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, source, parent_fd, target, 0x00000001)  # RENAME_NOREPLACE
    else:
        raise UnifiedReportingError("atomic no-replace publication is unsupported on this platform")
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise UnifiedReportingError(f"publication target already exists (no clobber): {target_name}")
        raise UnifiedReportingError(
            f"atomic no-replace publication failed for {target_name}: {os.strerror(observed_errno)}"
        )


def _entry_stat(parent_fd: int, name: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise UnifiedReportingError(f"cannot inspect directory entry {name}: {exc}") from exc


def _quarantine_canonical_entry(parent_fd: int, name: str) -> list[str]:
    """Make a failed publication name absent without deleting its current entry.

    The entry at ``name`` may no longer be the inode that this process
    published.  Consequently rollback must not unlink by pathname.  Each
    observed entry is moved, type-agnostically and with kernel no-replace
    semantics, to an evidence-preserving quarantine name through the already
    held parent descriptor.  The loop also closes the deterministic recreate
    window between rename and the absence check.
    """

    name = _safe_name(name, where="failed publication entry")
    quarantined: list[str] = []
    for _ in range(64):
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.fsync(parent_fd)
            return quarantined
        except OSError as exc:
            raise UnifiedReportingError(
                f"cannot inspect failed publication entry {name}: {exc}"
            ) from exc
        quarantine_name = f".{name}.invalid-{secrets.token_hex(12)}"
        try:
            _rename_noreplace(parent_fd, name, quarantine_name)
        except UnifiedReportingError:
            # A concurrent disappearance is already the required safe state.
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                os.fsync(parent_fd)
                return quarantined
            raise
        quarantined.append(quarantine_name)
        os.fsync(parent_fd)
    raise UnifiedReportingError(
        f"failed publication entry was recreated too many times: {name}"
    )


def _assert_entry_identity(
    parent_fd: int, name: str, expected: os.stat_result, *, directory: bool, where: str,
) -> None:
    observed = _entry_stat(parent_fd, name)
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_kind(observed.st_mode) or _identity(observed) != _identity(expected):
        raise UnifiedReportingError(f"{where} changed identity")


def _cleanup_tree_fd(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=directory_fd)
            try:
                _cleanup_tree_fd(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _cleanup_staging_directory(
    parent_fd: int, stage_name: str, stage_fd: int, stage_info: os.stat_result,
) -> None:
    try:
        _assert_entry_identity(
            parent_fd, stage_name, stage_info, directory=True, where="staging directory",
        )
    except UnifiedReportingError:
        return
    try:
        _cleanup_tree_fd(stage_fd)
        os.fsync(stage_fd)
        os.rmdir(stage_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise UnifiedReportingError(f"cannot clean owned staging directory: {exc}") from exc


@dataclass
class _CanonicalPublicationState:
    attempted: bool = False
    exposed: bool = False


def _publish_staging_directory(
    *, parent_path: Path, parent_fd: int, parent_info: os.stat_result,
    stage_name: str, stage_fd: int, stage_info: os.stat_result, final_name: str,
    publication_state: _CanonicalPublicationState,
) -> Path:
    os.fsync(stage_fd)
    os.fsync(parent_fd)
    _assert_directory_path_identity(parent_path, parent_info, where="publication parent")
    _assert_entry_identity(
        parent_fd, stage_name, stage_info, directory=True, where="staging directory",
    )
    publication_state.attempted = True
    _rename_noreplace(parent_fd, stage_name, final_name)
    # This assignment must be the first operation after the successful kernel
    # rename.  Every later failure is rollback-eligible even if the canonical
    # name is subsequently replaced by an unrelated inode or file type.
    publication_state.exposed = True
    os.fsync(parent_fd)
    _assert_entry_identity(
        parent_fd, final_name, stage_info, directory=True, where="published release",
    )
    _assert_directory_path_identity(parent_path, parent_info, where="publication parent")
    return parent_path / final_name


def _fd_is_within(directory_fd: int, forbidden: set[tuple[int, int]]) -> bool:
    current = os.dup(directory_fd)
    try:
        while True:
            here = _identity(os.fstat(current))
            if here in forbidden:
                return True
            parent = os.open("..", _directory_flags(), dir_fd=current)
            parent_identity = _identity(os.fstat(parent))
            if parent_identity == here:
                os.close(parent)
                return False
            os.close(current)
            current = parent
    finally:
        os.close(current)


def _write_external_no_clobber(
    path: str | Path, payload: bytes, *, forbidden_ancestors: set[tuple[int, int]] | None = None,
    precommit: Callable[[], None] | None = None,
    postcommit: Callable[[], None] | None = None,
) -> Path:
    requested, parent_path, name, parent_fd, parent_info = _open_parent_for_new_path(
        path, where="certificate output",
    )
    if forbidden_ancestors and _fd_is_within(parent_fd, forbidden_ancestors):
        os.close(parent_fd)
        raise UnifiedReportingError("certificate destination is inside an authenticated build tree")
    temp_name = f".{name}.staging-{secrets.token_hex(12)}"
    descriptor: int | None = None
    temp_info: os.stat_result | None = None
    rename_attempted = False
    publication_returned = False
    succeeded = False
    try:
        descriptor = os.open(
            temp_name,
            _exclusive_write_flags(),
            0o444,
            dir_fd=parent_fd,
        )
        temp_info = os.fstat(descriptor)
        if not stat.S_ISREG(temp_info.st_mode):
            raise UnifiedReportingError("certificate staging path is not a regular file")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise UnifiedReportingError("short write for verification certificate")
            view = view[written:]
        os.fsync(descriptor)
        temp_info = os.fstat(descriptor)
        os.fsync(parent_fd)
        _assert_directory_path_identity(parent_path, parent_info, where="certificate parent")
        _assert_entry_identity(
            parent_fd, temp_name, temp_info, directory=False, where="certificate staging file",
        )
        if forbidden_ancestors and _fd_is_within(parent_fd, forbidden_ancestors):
            raise UnifiedReportingError("certificate destination moved inside an authenticated build tree")
        _assert_directory_path_identity(parent_path, parent_info, where="certificate parent")
        _assert_entry_identity(
            parent_fd, temp_name, temp_info, directory=False, where="certificate staging file",
        )
        if _stable_stat(os.fstat(descriptor)) != _stable_stat(temp_info):
            raise UnifiedReportingError("certificate staging file changed before publication")
        # The held build inventories are rescanned immediately on both sides
        # of the single namespace operation that exposes the certificate.
        if precommit is not None:
            precommit()
        rename_attempted = True
        _rename_noreplace(parent_fd, temp_name, name)
        publication_returned = True
        if postcommit is not None:
            postcommit()
        os.fsync(parent_fd)
        _assert_entry_identity(
            parent_fd, name, temp_info, directory=False, where="published certificate",
        )
        published_info = os.fstat(descriptor)
        if _rename_stable_stat(published_info) != _rename_stable_stat(temp_info):
            raise UnifiedReportingError("published certificate changed identity or contents")
        if _read_held_regular_fd(
            descriptor, _stable_stat(published_info), "published verification certificate",
        ) != payload:
            raise UnifiedReportingError("published certificate bytes changed")
        if forbidden_ancestors and _fd_is_within(parent_fd, forbidden_ancestors):
            raise UnifiedReportingError("published certificate parent moved inside an authenticated build tree")
        _assert_directory_path_identity(parent_path, parent_info, where="certificate parent")
        # Recheck both the authenticated inputs and the canonical certificate
        # binding at the last possible point before reporting success.  The
        # immediate post-rename scan above closes the commit window; this scan
        # also closes mutations attempted during the certificate self-checks.
        if postcommit is not None:
            postcommit()
        _assert_entry_identity(
            parent_fd, name, temp_info, directory=False,
            where="published certificate",
        )
        if _stable_stat(os.fstat(descriptor)) != _stable_stat(published_info):
            raise UnifiedReportingError("published certificate changed before success")
        succeeded = True
        return requested
    except OSError as exc:
        raise UnifiedReportingError(f"cannot publish verification certificate: {exc}") from exc
    finally:
        cleanup_error: Exception | None = None
        if not succeeded and temp_info is not None:
            try:
                if descriptor is None or _identity(os.fstat(descriptor)) != _identity(temp_info):
                    raise UnifiedReportingError("certificate staging descriptor changed identity")
                # If the rename returned, the canonical name binds the held
                # inode, or an attempted rename changed either the staging
                # binding or held-inode state, the canonical name may contain
                # our certificate or unrelated replacement data.  Link count
                # alone cannot resolve this ambiguity: an attacker may unlink
                # the genuine committed inode first.
                temp_present = False
                try:
                    observed_temp = os.stat(
                        temp_name, dir_fd=parent_fd, follow_symlinks=False,
                    )
                    temp_present = _identity(observed_temp) == _identity(temp_info)
                except FileNotFoundError:
                    temp_present = False
                try:
                    observed_canonical = os.stat(
                        name, dir_fd=parent_fd, follow_symlinks=False,
                    )
                    canonical_is_held_certificate = (
                        _identity(observed_canonical) == _identity(temp_info)
                    )
                except FileNotFoundError:
                    canonical_is_held_certificate = False
                held_certificate_changed = (
                    _stable_stat(os.fstat(descriptor)) != _stable_stat(temp_info)
                )
                publication_may_have_happened = (
                    publication_returned
                    or canonical_is_held_certificate
                    or (
                        rename_attempted
                        and (not temp_present or held_certificate_changed)
                    )
                )
                if publication_may_have_happened:
                    _quarantine_canonical_entry(parent_fd, name)
                    try:
                        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise UnifiedReportingError(
                            "failed certificate publication left a canonical target"
                        )
                elif temp_present:
                    _assert_entry_identity(
                        parent_fd, temp_name, temp_info, directory=False,
                        where="certificate staging file",
                    )
                    os.unlink(temp_name, dir_fd=parent_fd)
                    if os.fstat(descriptor).st_nlink != 0:
                        raise UnifiedReportingError(
                            "certificate staging unlink did not remove the held inode"
                        )
                    os.fsync(parent_fd)
            except (OSError, UnifiedReportingError) as exc:
                cleanup_error = exc
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        if cleanup_error is not None:
            raise UnifiedReportingError(
                f"certificate publication rollback failed: {cleanup_error}"
            ) from cleanup_error


def _read_held_regular_fd(
    descriptor: int, expected: tuple[int, ...], relative: str,
) -> bytes:
    """Read one inventory-bound release inode through its already-held descriptor."""

    if not hasattr(os, "pread"):
        raise UnifiedReportingError("descriptor-stable release reads are unsupported")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stable_stat(before) != expected
        ):
            raise UnifiedReportingError(f"held release file changed before read: {relative}")
        chunks: list[bytes] = []
        offset = 0
        while True:
            block = os.pread(descriptor, 1 << 20, offset)
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
        if after.st_nlink != 1 or _stable_stat(after) != expected:
            raise UnifiedReportingError(f"held release file changed while being read: {relative}")
        return b"".join(chunks)
    except OSError as exc:
        raise UnifiedReportingError(f"cannot safely read held release file {relative}: {exc}") from exc


def _assert_held_file_states(
    file_fds: Mapping[str, int], inventory: Mapping[str, tuple[int, ...]],
) -> None:
    for relative, descriptor in file_fds.items():
        observed = os.fstat(descriptor)
        if observed.st_nlink != 1 or _stable_stat(observed) != inventory[relative]:
            raise UnifiedReportingError(
                f"held release file inode changed during verification: {relative}"
            )


def _parse_hashed_json(
    payload: bytes, *, where: str, schema: str, hash_field: str = "payload_sha256"
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnifiedReportingError(f"{where} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise UnifiedReportingError(f"{where} schema drift")
    body = dict(value)
    observed = body.pop(hash_field, None)
    if observed != canonical_sha256(body):
        raise UnifiedReportingError(f"{where} self hash mismatch")
    return value


def _source_relation_checks(
    source_lock: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    bindings = {row["source_id"]: row for row in tables["source_bindings"]}
    records = {record["source_id"]: record for record in source_lock["sources"]}
    if set(bindings) != set(records):
        raise UnifiedReportingError("source_bindings does not cover the exact source lock")
    uncertified_binding_ids: set[str] = set()
    for source_id, record in records.items():
        row = bindings[source_id]
        logical_hash = canonical_sha256(record)
        expected_id = f"sourcev1_{logical_hash[:24]}"
        certified = record.get("certified") is True
        if (
            row["source_binding_id"] != expected_id
            or row["logical_binding_sha256"] != logical_hash
            or row["certified"] is not certified
            or row["source_release_id"] != record["source_release_id"]
            or row["source_status"] != ("CERTIFIED" if certified else "NOT_CERTIFIED")
        ):
            raise UnifiedReportingError(f"source binding drift for {source_id}")
        if not certified:
            uncertified_binding_ids.add(expected_id)
    artifacts = tables["source_artifacts"]
    expected_artifacts: set[tuple[str, str, str, str]] = set()
    for source_id, record in records.items():
        if record.get("certified") is not True:
            continue
        binding_id = bindings[source_id]["source_binding_id"]
        expected_artifacts.add(
            (binding_id, "certificate", "certificate", record["certificate"]["file_sha256"])
        )
        if "manifest" in record:
            expected_artifacts.add(
                (binding_id, "manifest", "manifest", record["manifest"]["file_sha256"])
            )
        expected_artifacts.update(
            (binding_id, "evaluation_artifact", name, binding["file_sha256"])
            for name, binding in record.get("files", {}).items()
        )
    observed_artifacts = {
        (row["source_binding_id"], row["artifact_role"], row["logical_name"], row["file_sha256"])
        for row in artifacts
        if row["authenticated"] is True
    }
    if observed_artifacts != expected_artifacts or len(observed_artifacts) != len(artifacts):
        raise UnifiedReportingError("source_artifacts does not match the reviewed file lock")
    placeholder_counts = {binding_id: 0 for binding_id in uncertified_binding_ids}
    for row in tables["status"]:
        if row["source_binding_id"] in uncertified_binding_ids:
            if not (
                row["status_scope"] == "lane"
                and row["status_class"] == "NOT_CERTIFIED"
                and row["source_status"] == "NOT_CERTIFIED"
                and row["rankable"] is False
            ):
                raise UnifiedReportingError("uncertified lane emitted a non-placeholder status")
            placeholder_counts[row["source_binding_id"]] += 1
    if any(count != 1 for count in placeholder_counts.values()):
        raise UnifiedReportingError("each uncertified lane must emit exactly one status placeholder")
    for table, rows in tables.items():
        if table in {"source_bindings", "status"}:
            continue
        if any(row["source_binding_id"] in uncertified_binding_ids for row in rows):
            raise UnifiedReportingError(f"uncertified lane emitted rows in {table}")


def _release_content(
    *, release_id: str, contract: Mapping[str, Any], source_lock: Mapping[str, Any],
    sources: Sequence[AuthenticatedSource], files: Mapping[str, Mapping[str, Any]],
    table_manifest: Mapping[str, Mapping[str, Any]], advisor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "bridge_schema": BRIDGE_SCHEMA,
        "table_schema": SCHEMA_VERSION,
        "release_id": release_id,
        "contract_sha256": canonical_sha256(contract),
        "source_lock_sha256": canonical_sha256(source_lock),
        "source_binding_ids": sorted(source.source_binding_id for source in sources),
        "source_statuses": {source.source_id: source.source_status for source in sources},
        "runtime_source_root_ids": sorted(
            {source.source_root_id for source in sources if source.source_root_id is not None}
        ),
        "files": dict(sorted(files.items())),
        "tables": dict(table_manifest),
        "advisor_payload_sha256": advisor["payload_sha256"],
        "certified_post_label_artifacts_only": True,
        "raw_or_private_label_sources_opened": False,
        "absolute_runtime_paths_recorded": False,
        "cross_task_aggregation_computed": False,
        "cross_prediction_unit_aggregation_computed": False,
        "cross_access_level_aggregation_computed": False,
        "winner_reference_equivalence_claim": False,
    }


def build_unified_release(
    *,
    release_id: str,
    build_id: str,
    output_root: str | Path,
    contract_path: str | Path,
    source_lock_path: str | Path,
    source_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Authenticate, normalize, and publish one immutable release directory.

    The complete tree is written in a same-parent hidden staging directory and
    becomes visible only through an atomic no-replace rename.
    """

    release_id = _token(release_id, where="release_id")
    build_id = _token(build_id, where="build_id")
    contract = load_contract(contract_path)
    source_lock = load_source_lock(source_lock_path)
    validate_contract_source_lock(contract, source_lock)
    sources = authenticate_sources(source_lock, source_roots=source_roots)
    tables = build_unified_rows(release_id=release_id, contract=contract, sources=sources)
    _source_relation_checks(source_lock, tables)
    advisor = advisor_inputs(release_id=release_id, contract=contract, tables=tables)

    contract_payload = canonical_json_bytes(contract) + b"\n"
    source_lock_payload = canonical_json_bytes(source_lock) + b"\n"
    schema_document = _with_hash(schema_bundle())
    schema_payload = _json_bytes(schema_document)
    advisor_payload = _json_bytes(advisor)

    fixed_payloads = {
        "REPORT_CONTRACT.json": contract_payload,
        "SOURCE_LOCK.json": source_lock_payload,
        "TABLE_SCHEMAS.json": schema_payload,
        "ADVISOR_INPUTS.json": advisor_payload,
    }
    release_payloads = dict(fixed_payloads)
    requested, parent_path, final_name, parent_fd, parent_info = _open_parent_for_new_path(
        output_root, where="output root",
    )
    try:
        os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        os.close(parent_fd)
        raise UnifiedReportingError(f"cannot inspect output root target: {exc}") from exc
    else:
        os.close(parent_fd)
        raise UnifiedReportingError(f"output root already exists (no clobber): {requested}")

    stage_name: str | None = None
    stage_fd: int | None = None
    stage_info: os.stat_result | None = None
    table_fd: int | None = None
    stage_file_fds: dict[str, int] = {}
    stage_inventory: Mapping[str, tuple[int, ...]] | None = None
    publication_state = _CanonicalPublicationState()
    published = False
    files: dict[str, dict[str, Any]] = {}
    table_manifest: dict[str, dict[str, Any]] = {}
    try:
        stage_name, stage_fd, stage_info = _create_staging_directory(parent_fd, final_name)
        table_fd, _ = _mkdir_owned_fd(stage_fd, "tables")
        for name, payload in fixed_payloads.items():
            _write_owned_fd(stage_fd, name, payload)
            files[name] = {"file_sha256": _sha256(payload), "bytes": len(payload)}

        for table in sorted(TABLE_SCHEMAS):
            csv_payload, normalized = csv_bytes(table, tables[table])
            parquet_payload, _ = parquet_bytes(table, normalized)
            logical_sha = assert_csv_parquet_parity(table, csv_payload, parquet_payload)
            csv_name = f"{table}.csv"
            parquet_name = f"{table}.parquet"
            _write_owned_fd(table_fd, csv_name, csv_payload)
            _write_owned_fd(table_fd, parquet_name, parquet_payload)
            csv_path, parquet_path = f"tables/{csv_name}", f"tables/{parquet_name}"
            release_payloads[csv_path] = csv_payload
            release_payloads[parquet_path] = parquet_payload
            files[csv_path] = {"file_sha256": _sha256(csv_payload), "bytes": len(csv_payload)}
            files[parquet_path] = {"file_sha256": _sha256(parquet_payload), "bytes": len(parquet_payload)}
            table_manifest[table] = {
                "row_count": len(normalized),
                "logical_sha256": logical_sha,
                "csv_path": csv_path,
                "csv_file_sha256": files[csv_path]["file_sha256"],
                "parquet_path": parquet_path,
                "parquet_file_sha256": files[parquet_path]["file_sha256"],
            }

        content = _release_content(
            release_id=release_id, contract=contract, source_lock=source_lock,
            sources=sources, files=files, table_manifest=table_manifest,
            advisor=advisor,
        )
        release_content_sha = canonical_sha256(content)
        manifest = _with_hash({
            "schema_version": MANIFEST_SCHEMA,
            "status": "READY",
            "release_id": release_id,
            "build_id": build_id,
            "release_content_sha256": release_content_sha,
            "content": content,
            "completion_marker": "RELEASE_COMPLETE.json",
        })
        manifest_payload = _json_bytes(manifest)
        _write_owned_fd(stage_fd, "BRIDGE_MANIFEST.json", manifest_payload)
        release_payloads["BRIDGE_MANIFEST.json"] = manifest_payload
        completion = _with_hash(
            {
                "schema_version": COMPLETION_SCHEMA,
                "status": "COMPLETE",
                "release_id": release_id,
                "build_id": build_id,
                "manifest_file_sha256": _sha256(manifest_payload),
                "manifest_payload_sha256": manifest["payload_sha256"],
                "release_content_sha256": release_content_sha,
            },
            field="completion_sha256",
        )
        # The completion marker is written inside the unpublished stage, then
        # the entire complete tree is committed in one namespace operation.
        completion_payload = _json_bytes(completion)
        _write_owned_fd(stage_fd, "RELEASE_COMPLETE.json", completion_payload)
        release_payloads["RELEASE_COMPLETE.json"] = completion_payload
        os.fsync(table_fd)
        os.fsync(stage_fd)
        stage_inventory = _inventory_state(stage_fd, table_fd)
        stage_file_fds, stage_inventory = _open_inventory_files(
            stage_fd, table_fd, stage_inventory,
        )
        if set(release_payloads) != set(stage_file_fds):
            raise UnifiedReportingError("generated payload roster differs from the staging inventory")
        for relative, payload in release_payloads.items():
            observed = _read_held_regular_fd(
                stage_file_fds[relative], stage_inventory[relative], relative,
            )
            if observed != payload:
                raise UnifiedReportingError(
                    f"staging payload differs from generated bytes: {relative}"
                )
        if _inventory_state(stage_fd, table_fd) != stage_inventory:
            raise UnifiedReportingError("staging inventory changed before atomic publication")
        root = _publish_staging_directory(
            parent_path=parent_path, parent_fd=parent_fd, parent_info=parent_info,
            stage_name=stage_name, stage_fd=stage_fd, stage_info=stage_info,
            final_name=final_name, publication_state=publication_state,
        )
        _assert_held_file_states(stage_file_fds, stage_inventory)
        published_inventory = _inventory_state(stage_fd, table_fd)
        if any(
            published_inventory[name] != stage_inventory[name]
            for name in stage_inventory
            if name != "@root"
        ):
            raise UnifiedReportingError("published release differs from the authenticated staging tree")
        _assert_entry_identity(
            parent_fd, final_name, stage_info, directory=True,
            where="published release before success",
        )
        _assert_directory_path_identity(
            parent_path, parent_info, where="publication parent before success",
        )
        _assert_held_file_states(stage_file_fds, stage_inventory)
        if _inventory_state(stage_fd, table_fd) != published_inventory:
            raise UnifiedReportingError("published release changed before success")
        _assert_directory_path_identity(
            parent_path, parent_info, where="publication parent at success boundary",
        )
        _assert_entry_identity(
            parent_fd, final_name, stage_info, directory=True,
            where="published release at success boundary",
        )
        published = True
    finally:
        rollback_error: Exception | None = None
        if not published:
            try:
                publication_may_have_happened = publication_state.exposed
                stage_entry_matches = False
                if stage_name is not None and stage_fd is not None and stage_info is not None:
                    try:
                        observed_stage = os.stat(
                            stage_name, dir_fd=parent_fd, follow_symlinks=False,
                        )
                        stage_entry_matches = (
                            stat.S_ISDIR(observed_stage.st_mode)
                            and _identity(observed_stage) == _identity(stage_info)
                        )
                    except FileNotFoundError:
                        stage_entry_matches = False

                    try:
                        observed_canonical = os.stat(
                            final_name, dir_fd=parent_fd, follow_symlinks=False,
                        )
                        canonical_present = True
                        canonical_is_held_stage = (
                            _identity(observed_canonical) == _identity(stage_info)
                        )
                    except FileNotFoundError:
                        canonical_present = False
                        canonical_is_held_stage = False

                    held_stage_info = os.fstat(stage_fd)
                    held_stage_changed = (
                        stage_inventory is not None
                        and _stable_stat(held_stage_info) != stage_inventory["@root"]
                    )
                    # A missing/replaced staging name after a reported rename
                    # failure is commit-uncertain.  The held inode may be the
                    # canonical release, may have been displaced as evidence,
                    # or the canonical name may already hold a replacement.
                    # None of those states permits ordinary staging cleanup.
                    publication_may_have_happened = (
                        publication_may_have_happened
                        or canonical_is_held_stage
                        or (publication_state.attempted and held_stage_changed)
                        or (
                            not stage_entry_matches
                            and (held_stage_info.st_nlink > 0 or canonical_present)
                        )
                    )

                if publication_may_have_happened:
                    _quarantine_canonical_entry(parent_fd, final_name)
                    try:
                        os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise UnifiedReportingError(
                            "failed release publication left a canonical target"
                        )
                elif (
                    stage_entry_matches
                    and stage_name is not None and stage_fd is not None and stage_info is not None
                ):
                    _cleanup_staging_directory(parent_fd, stage_name, stage_fd, stage_info)
            except Exception as exc:
                rollback_error = exc
        for descriptor in stage_file_fds.values():
            os.close(descriptor)
        if table_fd is not None:
            os.close(table_fd)
        if stage_fd is not None:
            os.close(stage_fd)
        os.close(parent_fd)
        if rollback_error is not None:
            raise UnifiedReportingError(
                f"release publication rollback failed: {rollback_error}"
            ) from rollback_error
    return {
        "output_root": str(root),
        "release_id": release_id,
        "build_id": build_id,
        "release_content_sha256": release_content_sha,
        "table_rows": {table: value["row_count"] for table, value in table_manifest.items()},
        "status": "COMPLETE",
    }


def _inventory_state(root_fd: int, tables_fd: int) -> dict[str, tuple[int, ...]]:
    expected_root = set(ROOT_FILES) | {"tables"}
    expected_tables = {
        f"{table}.{suffix}" for table in TABLE_SCHEMAS for suffix in ("csv", "parquet")
    }
    try:
        observed_root = set(os.listdir(root_fd))
        observed_tables = set(os.listdir(tables_fd))
    except OSError as exc:
        raise UnifiedReportingError(f"cannot scan held release inventory: {exc}") from exc
    if observed_root != expected_root:
        raise UnifiedReportingError(
            f"release root inventory drift: expected={sorted(expected_root)}, observed={sorted(observed_root)}"
        )
    if observed_tables != expected_tables:
        raise UnifiedReportingError(
            f"release table inventory drift: expected={sorted(expected_tables)}, observed={sorted(observed_tables)}"
        )
    state: dict[str, tuple[int, ...]] = {
        "@root": _stable_stat(os.fstat(root_fd)),
        "@tables": _stable_stat(os.fstat(tables_fd)),
    }
    try:
        table_entry = os.stat("tables", dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(table_entry.st_mode) or _identity(table_entry) != _identity(os.fstat(tables_fd)):
            raise UnifiedReportingError("release tables entry is not the held directory")
        state["tables"] = _stable_stat(table_entry)
        for name in ROOT_FILES:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise UnifiedReportingError(f"release root entry is not a regular file: {name}")
            if info.st_nlink != 1:
                raise UnifiedReportingError(
                    f"release file must have exactly one hard link: {name}"
                )
            state[name] = _stable_stat(info)
        for name in expected_tables:
            info = os.stat(name, dir_fd=tables_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise UnifiedReportingError(f"release table entry is not a regular file: {name}")
            if info.st_nlink != 1:
                raise UnifiedReportingError(
                    f"release file must have exactly one hard link: tables/{name}"
                )
            state[f"tables/{name}"] = _stable_stat(info)
    except OSError as exc:
        raise UnifiedReportingError(f"cannot inspect held release inventory: {exc}") from exc
    return state


@dataclass
class _HeldVerifiedRelease:
    verified: VerifiedUnifiedRelease
    root_fd: int
    tables_fd: int
    file_fds: Mapping[str, int]
    root_info: os.stat_result
    tables_info: os.stat_result
    initial_inventory: Mapping[str, tuple[int, ...]]

    def close(self) -> None:
        for descriptor in self.file_fds.values():
            os.close(descriptor)
        os.close(self.tables_fd)
        os.close(self.root_fd)


def _assert_held_release_unchanged(held: _HeldVerifiedRelease) -> None:
    _assert_held_file_states(held.file_fds, held.initial_inventory)
    if _inventory_state(held.root_fd, held.tables_fd) != held.initial_inventory:
        raise UnifiedReportingError("held release inventory or inode state changed during verification")
    _assert_entry_identity(
        held.root_fd, "tables", held.tables_info, directory=True, where="release tables directory",
    )
    _assert_directory_path_identity(
        held.verified.root, held.root_info, where="unified release root",
    )


def _open_inventory_files(
    root_fd: int, tables_fd: int, inventory: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, int], dict[str, tuple[int, ...]]]:
    """Open every expected release file once and bind it to the initial inventory."""

    relative_paths = set(ROOT_FILES) | {
        f"tables/{table}.{suffix}" for table in TABLE_SCHEMAS for suffix in ("csv", "parquet")
    }
    opened: dict[str, int] = {}
    bound_inventory = dict(inventory)
    descriptor: int | None = None
    try:
        for relative in sorted(relative_paths):
            parts = _safe_parts(relative)
            directory_fd = tables_fd if parts[0] == "tables" else root_fd
            name = parts[-1]
            descriptor = _open_inventory_file(directory_fd, name)
            info = os.fstat(descriptor)
            observed = _stable_stat(info)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                # On APFS, the first read-only open can materialize access
                # metadata and advance ctime.  Bind that descriptor-observed
                # ctime only after every identity/content field still matches
                # the initial path inventory; subsequent checks use the full
                # rebound state, including ctime.
                or observed[:-1] != inventory[relative][:-1]
            ):
                raise UnifiedReportingError(
                    f"release file does not match the initial inventory: {relative}"
                )
            bound_inventory[relative] = observed
            opened[relative] = descriptor
            descriptor = None
        return opened, bound_inventory
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        for descriptor in opened.values():
            os.close(descriptor)
        raise UnifiedReportingError(f"cannot bind release files to held descriptors: {exc}") from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        for descriptor in opened.values():
            os.close(descriptor)
        raise


def _open_held_release(
    root: str | Path,
) -> tuple[
    Path, int, int, Mapping[str, int], os.stat_result, os.stat_result,
    Mapping[str, tuple[int, ...]],
]:
    release_root, root_fd, root_info = _open_directory_path_nofollow(
        root, where="unified release root",
    )
    tables_fd: int | None = None
    file_fds: dict[str, int] = {}
    try:
        tables_fd = os.open("tables", _directory_flags(), dir_fd=root_fd)
        tables_info = os.fstat(tables_fd)
        inventory = _inventory_state(root_fd, tables_fd)
        file_fds, inventory = _open_inventory_files(root_fd, tables_fd, inventory)
        if _inventory_state(root_fd, tables_fd) != inventory:
            raise UnifiedReportingError("release inventory changed while binding file descriptors")
        return release_root, root_fd, tables_fd, file_fds, root_info, tables_info, inventory
    except OSError as exc:
        for descriptor in file_fds.values():
            os.close(descriptor)
        if tables_fd is not None:
            os.close(tables_fd)
        os.close(root_fd)
        raise UnifiedReportingError(f"cannot safely open release tables directory: {exc}") from exc
    except Exception:
        for descriptor in file_fds.values():
            os.close(descriptor)
        if tables_fd is not None:
            os.close(tables_fd)
        os.close(root_fd)
        raise


def _verify_completed_release_held(
    root: str | Path, *, trusted_contract: Mapping[str, Any] | None = None,
    trusted_source_lock: Mapping[str, Any] | None = None,
    source_roots: Mapping[str, str | Path] | None = None,
) -> _HeldVerifiedRelease:
    trusted = trusted_contract is not None or trusted_source_lock is not None or source_roots is not None
    if trusted and (
        trusted_contract is None or trusted_source_lock is None or source_roots is None
    ):
        raise UnifiedReportingError("trusted verification requires contract, source lock, and source roots")
    (
        release_root, root_fd, tables_fd, file_fds, root_info, tables_info, inventory,
    ) = _open_held_release(root)
    try:
        def read_release_file(relative: str) -> bytes:
            return _read_held_regular_fd(file_fds[relative], inventory[relative], relative)

        completion_payload = read_release_file("RELEASE_COMPLETE.json")
        completion = _parse_hashed_json(
            completion_payload, where="release completion", schema=COMPLETION_SCHEMA,
            hash_field="completion_sha256",
        )
        if completion.get("status") != "COMPLETE":
            raise UnifiedReportingError("unified release is not complete")
        manifest_payload = read_release_file("BRIDGE_MANIFEST.json")
        if _sha256(manifest_payload) != completion.get("manifest_file_sha256"):
            raise UnifiedReportingError("completion does not bind the bridge manifest bytes")
        manifest = _parse_hashed_json(
            manifest_payload, where="bridge manifest", schema=MANIFEST_SCHEMA
        )
        if (
            manifest.get("status") != "READY"
            or manifest.get("payload_sha256") != completion.get("manifest_payload_sha256")
            or manifest.get("release_content_sha256") != completion.get("release_content_sha256")
            or manifest.get("release_id") != completion.get("release_id")
            or manifest.get("build_id") != completion.get("build_id")
            or manifest.get("completion_marker") != "RELEASE_COMPLETE.json"
        ):
            raise UnifiedReportingError("completion/manifest binding mismatch")
        release_id = _token(manifest.get("release_id"), where="embedded release_id")
        _token(manifest.get("build_id"), where="embedded build_id")
        content = manifest.get("content")
        if not isinstance(content, Mapping) or canonical_sha256(content) != manifest["release_content_sha256"]:
            raise UnifiedReportingError("release content hash mismatch")
        required_flags = {
            "certified_post_label_artifacts_only": True,
            "raw_or_private_label_sources_opened": False,
            "absolute_runtime_paths_recorded": False,
            "cross_task_aggregation_computed": False,
            "cross_prediction_unit_aggregation_computed": False,
            "cross_access_level_aggregation_computed": False,
            "winner_reference_equivalence_claim": False,
        }
        if any(content.get(name) is not value for name, value in required_flags.items()):
            raise UnifiedReportingError("release scientific safety flags drift")

        files = content.get("files")
        table_manifest = content.get("tables")
        if not isinstance(files, Mapping) or not isinstance(table_manifest, Mapping):
            raise UnifiedReportingError("release file/table manifest is missing")
        if set(table_manifest) != set(TABLE_SCHEMAS):
            raise UnifiedReportingError("release table manifest does not match schemas")
        expected_manifest_files = (ROOT_FILES - {"BRIDGE_MANIFEST.json", "RELEASE_COMPLETE.json"}) | {
            f"tables/{table}.{suffix}" for table in TABLE_SCHEMAS for suffix in ("csv", "parquet")
        }
        if set(files) != expected_manifest_files:
            raise UnifiedReportingError("release content file inventory drift")

        fixed: dict[str, bytes] = {}
        for relative in sorted(ROOT_FILES - {"BRIDGE_MANIFEST.json", "RELEASE_COMPLETE.json"}):
            record = files.get(relative)
            if not isinstance(record, Mapping):
                raise UnifiedReportingError(f"release file record missing: {relative}")
            payload = read_release_file(relative)
            if _sha256(payload) != record.get("file_sha256") or len(payload) != record.get("bytes"):
                raise UnifiedReportingError(f"release file binding mismatch: {relative}")
            fixed[relative] = payload
        contract = parse_contract_bytes(
            fixed["REPORT_CONTRACT.json"], where="embedded unified reporting contract",
        )
        source_lock = parse_source_lock_bytes(
            fixed["SOURCE_LOCK.json"], where="embedded unified reporting source lock",
        )
        if canonical_sha256(contract) != content.get("contract_sha256"):
            raise UnifiedReportingError("embedded contract hash mismatch")
        if canonical_sha256(source_lock) != content.get("source_lock_sha256"):
            raise UnifiedReportingError("embedded source-lock hash mismatch")
        validate_contract_source_lock(contract, source_lock)
        schema_document = _parse_hashed_json(
            fixed["TABLE_SCHEMAS.json"], where="table schema bundle", schema=SCHEMA_VERSION
        )
        if {key: value for key, value in schema_document.items() if key != "payload_sha256"} != schema_bundle():
            raise UnifiedReportingError("table schema bundle differs from the verifier")
        advisor = _parse_hashed_json(
            fixed["ADVISOR_INPUTS.json"], where="advisor inputs",
            schema="reconstruction-unified-advisor-inputs-v1",
        )
        if advisor["payload_sha256"] != content.get("advisor_payload_sha256"):
            raise UnifiedReportingError("advisor inputs are not bound by the manifest")

        expected_tables: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
        expected_advisor: Mapping[str, Any] | None = None
        authenticated: Sequence[AuthenticatedSource] | None = None
        expected_files: dict[str, dict[str, Any]] = {}
        expected_table_manifest: dict[str, dict[str, Any]] = {}
        if trusted:
            assert trusted_contract is not None and trusted_source_lock is not None and source_roots is not None
            validate_contract_source_lock(trusted_contract, trusted_source_lock)
            trusted_contract_payload = canonical_json_bytes(trusted_contract) + b"\n"
            trusted_lock_payload = canonical_json_bytes(trusted_source_lock) + b"\n"
            if fixed["REPORT_CONTRACT.json"] != trusted_contract_payload:
                raise UnifiedReportingError("embedded contract differs from the trusted verifier contract")
            if fixed["SOURCE_LOCK.json"] != trusted_lock_payload:
                raise UnifiedReportingError("embedded source lock differs from the trusted verifier lock")
            authenticated = authenticate_sources(trusted_source_lock, source_roots=source_roots)
            expected_tables = build_unified_rows(
                release_id=release_id, contract=trusted_contract, sources=authenticated,
            )
            _source_relation_checks(trusted_source_lock, expected_tables)
            expected_advisor = advisor_inputs(
                release_id=release_id, contract=trusted_contract, tables=expected_tables,
            )
            expected_fixed = {
                "REPORT_CONTRACT.json": trusted_contract_payload,
                "SOURCE_LOCK.json": trusted_lock_payload,
                "TABLE_SCHEMAS.json": _json_bytes(_with_hash(schema_bundle())),
                "ADVISOR_INPUTS.json": _json_bytes(expected_advisor),
            }
            for name, payload in expected_fixed.items():
                if fixed[name] != payload:
                    raise UnifiedReportingError(f"{name} differs from independent source rederivation")
                expected_files[name] = {"file_sha256": _sha256(payload), "bytes": len(payload)}

        loaded_tables: dict[str, list[dict[str, Any]]] = {}
        all_binding_ids: set[str] | None = None
        uncertified_ids: set[str] = set()
        for table in sorted(TABLE_SCHEMAS):
            record = table_manifest[table]
            if not isinstance(record, Mapping):
                raise UnifiedReportingError(f"table manifest record is invalid for {table}")
            csv_path = f"tables/{table}.csv"
            parquet_path = f"tables/{table}.parquet"
            if record.get("csv_path") != csv_path or record.get("parquet_path") != parquet_path:
                raise UnifiedReportingError(f"table path drift for {table}")
            csv_record = files.get(csv_path)
            parquet_record = files.get(parquet_path)
            if not isinstance(csv_record, Mapping) or not isinstance(parquet_record, Mapping):
                raise UnifiedReportingError(f"table file record is invalid for {table}")
            csv_payload = read_release_file(csv_path)
            parquet_payload = read_release_file(parquet_path)
            if (
                _sha256(csv_payload) != record.get("csv_file_sha256")
                or _sha256(parquet_payload) != record.get("parquet_file_sha256")
                or csv_record.get("file_sha256") != record.get("csv_file_sha256")
                or parquet_record.get("file_sha256") != record.get("parquet_file_sha256")
                or len(csv_payload) != csv_record.get("bytes")
                or len(parquet_payload) != parquet_record.get("bytes")
            ):
                raise UnifiedReportingError(f"table file binding drift for {table}")
            logical_sha = assert_csv_parquet_parity(table, csv_payload, parquet_payload)
            if logical_sha != record.get("logical_sha256"):
                raise UnifiedReportingError(f"table logical hash mismatch for {table}")
            rows = read_csv_bytes(table, csv_payload)
            if len(rows) != record.get("row_count"):
                raise UnifiedReportingError(f"table row count mismatch for {table}")
            if expected_tables is not None:
                expected_csv, normalized = csv_bytes(table, expected_tables[table])
                expected_parquet, _ = parquet_bytes(table, normalized)
                expected_logical = assert_csv_parquet_parity(table, expected_csv, expected_parquet)
                if csv_payload != expected_csv or parquet_payload != expected_parquet:
                    raise UnifiedReportingError(
                        f"{table} bytes differ from independent authenticated-source rederivation"
                    )
                expected_files[csv_path] = {
                    "file_sha256": _sha256(expected_csv), "bytes": len(expected_csv),
                }
                expected_files[parquet_path] = {
                    "file_sha256": _sha256(expected_parquet), "bytes": len(expected_parquet),
                }
                expected_table_manifest[table] = {
                    "row_count": len(normalized), "logical_sha256": expected_logical,
                    "csv_path": csv_path, "csv_file_sha256": _sha256(expected_csv),
                    "parquet_path": parquet_path,
                    "parquet_file_sha256": _sha256(expected_parquet),
                }
            loaded_tables[table] = rows
            if table == "source_bindings":
                all_binding_ids = {row["source_binding_id"] for row in rows}
                uncertified_ids = {
                    row["source_binding_id"] for row in rows if row["certified"] is False
                }
        if all_binding_ids is None:
            raise UnifiedReportingError("source binding table was not read")
        for table, rows in loaded_tables.items():
            if table == "source_bindings":
                continue
            for row in rows:
                binding_id = row.get("source_binding_id")
                if binding_id not in all_binding_ids:
                    raise UnifiedReportingError(f"{table} references an unknown source binding")
                if table != "status" and binding_id in uncertified_ids:
                    raise UnifiedReportingError(f"uncertified source emitted {table} rows")
        _source_relation_checks(source_lock, loaded_tables)

        if trusted:
            assert authenticated is not None and expected_advisor is not None
            expected_content = _release_content(
                release_id=release_id, contract=trusted_contract,
                source_lock=trusted_source_lock, sources=authenticated,
                files=expected_files, table_manifest=expected_table_manifest,
                advisor=expected_advisor,
            )
            if content != expected_content:
                raise UnifiedReportingError("bridge manifest content differs from independent rederivation")
            expected_release_sha = canonical_sha256(expected_content)
            expected_manifest = _with_hash({
                "schema_version": MANIFEST_SCHEMA, "status": "READY",
                "release_id": release_id, "build_id": manifest["build_id"],
                "release_content_sha256": expected_release_sha,
                "content": expected_content, "completion_marker": "RELEASE_COMPLETE.json",
            })
            if manifest != expected_manifest or manifest_payload != _json_bytes(expected_manifest):
                raise UnifiedReportingError("bridge manifest is not the exact independently rederived manifest")
            expected_completion = _with_hash({
                "schema_version": COMPLETION_SCHEMA, "status": "COMPLETE",
                "release_id": release_id, "build_id": manifest["build_id"],
                "manifest_file_sha256": _sha256(_json_bytes(expected_manifest)),
                "manifest_payload_sha256": expected_manifest["payload_sha256"],
                "release_content_sha256": expected_release_sha,
            }, field="completion_sha256")
            if completion != expected_completion or completion_payload != _json_bytes(expected_completion):
                raise UnifiedReportingError("completion marker is not the exact independently rederived marker")

        verified = VerifiedUnifiedRelease(
            root=release_root, root_device=int(root_info.st_dev), root_inode=int(root_info.st_ino),
            manifest=manifest, completion=completion,
            manifest_file_sha256=_sha256(manifest_payload),
            completion_file_sha256=_sha256(completion_payload),
        )
        held = _HeldVerifiedRelease(
            verified=verified, root_fd=root_fd, tables_fd=tables_fd, file_fds=file_fds,
            root_info=root_info, tables_info=tables_info, initial_inventory=inventory,
        )
        _assert_held_release_unchanged(held)
        return held
    except Exception:
        for descriptor in file_fds.values():
            os.close(descriptor)
        os.close(tables_fd)
        os.close(root_fd)
        raise


def verify_completed_release(root: str | Path) -> VerifiedUnifiedRelease:
    held = _verify_completed_release_held(root)
    try:
        return held.verified
    finally:
        held.close()


def verify_unified_release_ab(
    *, build_a: str | Path, build_b: str | Path, certificate_path: str | Path,
    contract_path: str | Path, source_lock_path: str | Path,
    source_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    trusted_contract = load_contract(contract_path)
    trusted_source_lock = load_source_lock(source_lock_path)
    validate_contract_source_lock(trusted_contract, trusted_source_lock)
    first: _HeldVerifiedRelease | None = None
    second: _HeldVerifiedRelease | None = None
    try:
        first = _verify_completed_release_held(
            build_a, trusted_contract=trusted_contract,
            trusted_source_lock=trusted_source_lock, source_roots=source_roots,
        )
        second = _verify_completed_release_held(
            build_b, trusted_contract=trusted_contract,
            trusted_source_lock=trusted_source_lock, source_roots=source_roots,
        )
        if (first.verified.root_device, first.verified.root_inode) == (
            second.verified.root_device, second.verified.root_inode
        ):
            raise UnifiedReportingError("A/B builds must be distinct directory inodes")
        first_file_inodes = {
            _identity(os.fstat(descriptor)) for descriptor in first.file_fds.values()
        }
        second_file_inodes = {
            _identity(os.fstat(descriptor)) for descriptor in second.file_fds.values()
        }
        if first_file_inodes & second_file_inodes:
            raise UnifiedReportingError("A/B builds share release file inodes")
        if first.verified.manifest["build_id"] == second.verified.manifest["build_id"]:
            raise UnifiedReportingError("A/B builds must have distinct build_id values")
        if first.verified.manifest["release_id"] != second.verified.manifest["release_id"]:
            raise UnifiedReportingError("A/B release_id mismatch")
        if first.verified.manifest["content"] != second.verified.manifest["content"]:
            raise UnifiedReportingError("A/B full unified reporting content mismatch")
        if (
            first.verified.manifest["release_content_sha256"]
            != second.verified.manifest["release_content_sha256"]
        ):
            raise UnifiedReportingError("A/B unified reporting content hash mismatch")
        certificate = _with_hash(
            {
                "schema_version": AB_CERTIFICATE_SCHEMA,
                "status": "PASS",
                "release_id": first.verified.manifest["release_id"],
                "release_content_sha256": first.verified.manifest["release_content_sha256"],
                "contract_sha256": canonical_sha256(trusted_contract),
                "source_lock_sha256": canonical_sha256(trusted_source_lock),
                "source_binding_ids": list(first.verified.manifest["content"]["source_binding_ids"]),
                "builds": {
                    "A": {
                        "build_id": first.verified.manifest["build_id"],
                        "manifest_file_sha256": first.verified.manifest_file_sha256,
                        "manifest_payload_sha256": first.verified.manifest["payload_sha256"],
                        "completion_file_sha256": first.verified.completion_file_sha256,
                        "completion_payload_sha256": first.verified.completion["completion_sha256"],
                    },
                    "B": {
                        "build_id": second.verified.manifest["build_id"],
                        "manifest_file_sha256": second.verified.manifest_file_sha256,
                        "manifest_payload_sha256": second.verified.manifest["payload_sha256"],
                        "completion_file_sha256": second.verified.completion_file_sha256,
                        "completion_payload_sha256": second.verified.completion["completion_sha256"],
                    },
                },
                "byte_identical_content_files": True,
                "full_manifest_rederived": True,
                "csv_parquet_logical_parity": True,
                "source_bindings_authenticated": True,
                "independent_source_rederivation": True,
                "absolute_runtime_paths_recorded": False,
            },
            field="certificate_sha256",
        )

        def precommit() -> None:
            assert first is not None and second is not None
            _assert_held_release_unchanged(first)
            _assert_held_release_unchanged(second)

        def postcommit() -> None:
            assert first is not None and second is not None
            _assert_held_release_unchanged(first)
            _assert_held_release_unchanged(second)

        _write_external_no_clobber(
            certificate_path, _json_bytes(certificate),
            forbidden_ancestors={
                (first.verified.root_device, first.verified.root_inode),
                (second.verified.root_device, second.verified.root_inode),
            },
            precommit=precommit,
            postcommit=postcommit,
        )
        return certificate
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()


__all__ = [
    "AB_CERTIFICATE_SCHEMA", "COMPLETION_SCHEMA", "MANIFEST_SCHEMA",
    "VerifiedUnifiedRelease", "build_unified_release",
    "verify_completed_release", "verify_unified_release_ab",
]
