"""Fail-closed contracts for the actual-execution LEASH stopping lane."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import ctypes
import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
from typing import Any

from .io import (
    canonical_json_bytes,
    sha256_bytes,
)


REGISTRY_SCHEMA = "reconstruction-leash-stopping-registry-v1"
PREPARATION_SCHEMA = "reconstruction-leash-stopping-preparation-v1"
PRIVATE_OUTCOME_SCHEMA = "reconstruction-leash-stopping-private-outcomes-v1"
PREPARATION_AB_SCHEMA = "reconstruction-leash-stopping-preparation-ab-v1"
FIT_SCHEMA = "reconstruction-leash-stopping-fit-v1"
FIT_AB_SCHEMA = "reconstruction-leash-stopping-fit-ab-v1"
EVALUATION_SCHEMA = "reconstruction-leash-stopping-evaluation-v1"
EVALUATION_AB_SCHEMA = "reconstruction-leash-stopping-evaluation-ab-v1"

ARMS = ("cot", "leash", "nocot")
DATASETS = ("aqua", "gsm8k")
READY_STATUS = "READY"
BLOCKED_STATUS = "PROTOCOL_GATE_FAILED"
FIDELITY = "paper-specified-partial"
FIT_ALLOWED_FIELDS = frozenset(
    {
        "row_id", "group_id", "cell_id", "population_id", "dataset_revision",
        "dataset", "question_id", "model", "model_revision", "arm", "method_id",
        "trace_key", "source_artifact_sha256", "n_reasoning_tokens",
        "n_closure_tokens", "n_total_tokens", "wall_s", "stopped_early",
        "closure_generated", "stop_reason", "setting_label", "fidelity",
        "actual_policy_execution_observed",
        "policy_replay_verified", "policy_replay_fired", "policy_replay_stop_index",
        "closure_evidence_verified",
    }
)
FIT_FORBIDDEN_FIELDS = frozenset(
    {
        "answer", "answer_text", "gold", "gold_answer", "label", "labels", "correct",
        "stored_correct", "pred_answer", "prediction", "stored_prediction", "parse_status",
        "stored_parse_status", "parser_failure", "problem", "prompt", "response", "target",
    }
)
SEARCHABLE_TABLES = (
    "coverage", "per_question", "cell_metrics", "contrasts", "frontier",
    "aggregate_metrics", "bootstrap_intervals",
)

SOURCE_GUARD_CODE_PATHS = {
    "worker": "scripts/reconstruction_benchmark/leash_source_guard_worker.py",
    "contract": "spectral_utils/reconstruction_benchmark/leash_contract.py",
    "preparation": "spectral_utils/reconstruction_benchmark/leash_preparation.py",
    "canonical_io": "spectral_utils/reconstruction_benchmark/io.py",
    "stopping_adapter": "spectral_utils/fair_comparisons/stopping.py",
    "comparison_registry": "spectral_utils/fair_comparisons/registry.py",
    "policy_callback": "spectral_utils/paper_exact/leash.py",
    "source_manifest": "spectral_utils/paper_exact/manifest.py",
    "source_evaluator": "spectral_utils/paper_exact/evaluator.py",
}


class LeashContractError(RuntimeError):
    """A LEASH source or released artifact violated the frozen contract."""


def payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def source_guard_closure_sha256(registry: Mapping[str, Any]) -> str:
    """Digest the exact ordered source-guard code closure declared by registry."""

    guard_files = registry.get("source_contract", {}).get("source_guard_code_files", {})
    records = [
        {
            "asset_id": name,
            "path": SOURCE_GUARD_CODE_PATHS[name],
            "sha256": guard_files.get(name, {}).get("sha256"),
        }
        for name in sorted(SOURCE_GUARD_CODE_PATHS)
    ]
    return payload_sha256(records)


def add_payload_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output["payload_sha256"] = payload_sha256(output)
    return output


def verify_payload(value: Mapping[str, Any], *, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != payload_sha256(payload):
        raise LeashContractError(f"{name} payload SHA-256 failed")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    return load_bound_json(path, name=name)


def load_jsonl(path: str | Path, *, name: str) -> list[dict[str, Any]]:
    return load_bound_jsonl(path, name=name)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_relative(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LeashContractError(f"{name} path is empty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise LeashContractError(f"{name} path is unsafe: {value!r}")
    return value


def resolve_source_path(source_root: str | Path, relative: str, *, name: str) -> Path:
    relative = _safe_relative(relative, name=name)
    root = Path(source_root).resolve(strict=True)
    unresolved = root
    for component in PurePosixPath(relative).parts:
        unresolved = unresolved / component
        if unresolved.is_symlink():
            raise LeashContractError(f"{name} source path contains a symlink")
    candidate = unresolved.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise LeashContractError(f"{name} escapes source root") from error
    return candidate


def assert_no_symlinks(root: str | Path, *, name: str) -> None:
    base = Path(root)
    if base.is_symlink():
        raise LeashContractError(f"{name} root is a symlink")
    if any(path.is_symlink() for path in base.rglob("*")):
        raise LeashContractError(f"{name} contains a symlink")


def load_registry(path: str | Path) -> dict[str, Any]:
    value = load_bound_json(path, name="LEASH registry")
    if value.get("schema_version") != REGISTRY_SCHEMA:
        raise LeashContractError("unexpected LEASH registry schema")
    if value.get("fidelity") != FIDELITY:
        raise LeashContractError("LEASH fidelity must remain paper-specified-partial")
    boundary = value.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise LeashContractError("LEASH claim boundary is missing")
    if boundary.get("conceptual_objective_status") != "CONCEPTUAL_ONLY_NOT_REPRODUCED_EQUATION":
        raise LeashContractError("conceptual objective cannot be an equation reproduction")
    if boundary.get("paper_exact_status") != "FORBIDDEN":
        raise LeashContractError("paper-exact status must be forbidden")
    if boundary.get("matched_accuracy_status") != "FORBIDDEN":
        raise LeashContractError("matched-accuracy status must be forbidden")
    forbidden_claims = " ".join(str(item).lower() for item in boundary.get("forbidden", ()))
    for phrase in ("paper-exact", "reproduced equation", "matched-accuracy", "mistral"):
        if phrase not in forbidden_claims:
            raise LeashContractError(f"LEASH forbidden claim roster lacks {phrase!r}")

    policy = value.get("policy_contract")
    if not isinstance(policy, Mapping) or tuple(policy.get("arms", ())) != ARMS:
        raise LeashContractError("LEASH arm roster/order drifted")
    if policy.get("setting_label") != "central" or policy.get("proxy_stopping") is not False:
        raise LeashContractError("LEASH actual-execution policy contract drifted")
    if policy.get("forced_closure_definition") != "stopped_early and closure_generated":
        raise LeashContractError("LEASH forced-closure definition drifted")
    if (
        policy.get("acquisition_repo_commit") != "4b6b81015971fc332db603468ff69c2925cc3084"
        or policy.get("acquisition_repo_dirty") is not False
    ):
        raise LeashContractError("LEASH clean acquisition commit binding drifted")
    published = policy.get("published_constants")
    if published != {"k": 8, "L": 5, "eps_H": 0.005, "delta_M": 0.05, "m": 64, "M": 320}:
        raise LeashContractError("published LEASH constant roster drifted")
    declared = policy.get("declared_not_paper_specified")
    if declared != {"B": 30.0, "tau_p": 0.95, "w": 16, "gamma": 0.1}:
        raise LeashContractError("declared LEASH constant roster drifted")

    visibility = value.get("fit_visibility", {})
    if set(visibility.get("allowed_fields", ())) != FIT_ALLOWED_FIELDS:
        raise LeashContractError("LEASH fit-visible field roster drifted")
    if set(visibility.get("forbidden_field_names", ())) != FIT_FORBIDDEN_FIELDS:
        raise LeashContractError("LEASH fit-forbidden field roster drifted")

    source = value.get("source_contract", {})
    implementation_files = source.get("implementation_files")
    if not isinstance(implementation_files, Mapping) or set(implementation_files) != {
        "acquisition_runner", "policy_callback", "acquisition_evaluator", "frozen_stopping_loader"
    }:
        raise LeashContractError("LEASH implementation source roster drifted")
    for name, item in implementation_files.items():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or not _is_sha256(item.get("sha256"))
        ):
            raise LeashContractError(f"LEASH implementation binding is malformed: {name}")
        _safe_relative(item.get("path"), name=f"implementation::{name}")
    guard_files = source.get("source_guard_code_files")
    if not isinstance(guard_files, Mapping) or set(guard_files) != set(
        SOURCE_GUARD_CODE_PATHS
    ):
        raise LeashContractError("LEASH source-guard code closure roster drifted")
    for name, expected_path in SOURCE_GUARD_CODE_PATHS.items():
        item = guard_files[name]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or item.get("path") != expected_path
            or not _is_sha256(item.get("sha256"))
        ):
            raise LeashContractError(
                f"LEASH source-guard code binding is malformed: {name}"
            )
        _safe_relative(item["path"], name=f"source_guard::{name}")
    ready = source.get("ready_runs")
    blocked = source.get("blocked_runs")
    if not isinstance(ready, list) or len(ready) != 6:
        raise LeashContractError("LEASH registry must contain six ready cells")
    if not isinstance(blocked, list) or len(blocked) != 2:
        raise LeashContractError("LEASH registry must contain two blocked cells")
    run_ids: set[str] = set()
    ready_pairs: set[tuple[str, str]] = set()
    for item in ready:
        required = {
            "run_id", "path", "dataset", "model", "expected_questions", "expected_traces",
            "expected_leash_policy_stops",
            "file_count", "bytes_total", "tree_sha256", "manifest_sha256", "index_sha256",
            "status_sha256", "summary_sha256",
        }
        if not isinstance(item, Mapping) or set(item) != required:
            raise LeashContractError("ready LEASH source entry keys drifted")
        run_id = str(item["run_id"])
        if run_id in run_ids:
            raise LeashContractError(f"duplicate LEASH run ID {run_id}")
        run_ids.add(run_id)
        _safe_relative(item["path"], name=run_id)
        if item["dataset"] not in DATASETS:
            raise LeashContractError(f"unsupported LEASH dataset {item['dataset']!r}")
        ready_pairs.add((str(item["dataset"]), str(item["model"])))
        if int(item["expected_traces"]) != int(item["expected_questions"]) * len(ARMS):
            raise LeashContractError(f"{run_id} trace count is not questions x arms")
        if not 0 < int(item["expected_leash_policy_stops"]) <= int(item["expected_questions"]):
            raise LeashContractError(f"{run_id} registered LEASH stop count is invalid")
        if int(item["file_count"]) <= 0 or int(item["bytes_total"]) <= 0:
            raise LeashContractError(f"{run_id} source size is invalid")
        for field in ("tree_sha256", "manifest_sha256", "index_sha256", "status_sha256", "summary_sha256"):
            if not _is_sha256(item[field]):
                raise LeashContractError(f"{run_id} has invalid {field}")
    if len(ready_pairs) != 6 or {dataset for dataset, _ in ready_pairs} != set(DATASETS):
        raise LeashContractError("ready LEASH cell matrix is incomplete")

    for item in blocked:
        required = {
            "run_id", "path", "dataset", "model", "expected_traces", "expected_failed",
            "coverage_status", "failure_signature", "files",
        }
        if not isinstance(item, Mapping) or set(item) != required:
            raise LeashContractError("blocked LEASH source entry keys drifted")
        run_id = str(item["run_id"])
        if run_id in run_ids:
            raise LeashContractError(f"duplicate LEASH run ID {run_id}")
        run_ids.add(run_id)
        _safe_relative(item["path"], name=run_id)
        if item["coverage_status"] != BLOCKED_STATUS:
            raise LeashContractError("blocked Mistral status must be PROTOCOL_GATE_FAILED")
        if int(item["expected_failed"]) != int(item["expected_traces"]):
            raise LeashContractError("blocked Mistral cell must bind all expected failures")
        files = item["files"]
        if not isinstance(files, Mapping) or set(files) != {
            "RUN_MANIFEST.json", "GATE_S2-leash-full.json", "STATUS.json"
        } or any(not _is_sha256(digest) for digest in files.values()):
            raise LeashContractError("blocked Mistral file roster/hash drifted")

    population = value.get("population", {})
    if int(population.get("ready_cells", -1)) != 6 or int(population.get("blocked_cells", -1)) != 2:
        raise LeashContractError("LEASH population cell counts drifted")
    if int(population.get("expected_ready_traces", -1)) != sum(
        int(item["expected_traces"]) for item in ready
    ):
        raise LeashContractError("LEASH population trace count drifted")
    if set(population.get("expected_models", ())) != {str(item["model"]) for item in ready}:
        raise LeashContractError("LEASH registered ready-model roster drifted")
    expected_questions = population.get("expected_ready_questions_by_dataset")
    if not isinstance(expected_questions, Mapping) or set(expected_questions) != set(DATASETS):
        raise LeashContractError("LEASH dataset question-count roster drifted")
    for dataset in DATASETS:
        observed = {
            int(item["expected_questions"]) for item in ready if item["dataset"] == dataset
        }
        if observed != {int(expected_questions[dataset])}:
            raise LeashContractError(f"LEASH {dataset} question count drifted across models")
    if {str(item["dataset"]) for item in blocked} != set(DATASETS):
        raise LeashContractError("blocked Mistral dataset roster drifted")

    evaluation = value.get("evaluation", {})
    bootstrap = evaluation.get("bootstrap", {})
    if (
        int(bootstrap.get("draws", 0)) <= 0
        or bootstrap.get("unit") != "source question"
        or bootstrap.get("stratification") != "within dataset"
        or bootstrap.get("paired_across_arms_and_model_copies") is not True
        or evaluation.get("cross_task_or_access_macro") != "FORBIDDEN"
    ):
        raise LeashContractError("LEASH grouped-bootstrap contract drifted")
    output = value.get("output_contract", {})
    if tuple(output.get("searchable_tables", ())) != SEARCHABLE_TABLES:
        raise LeashContractError("LEASH searchable-table roster drifted")
    if output.get("atomic_no_clobber") is not True:
        raise LeashContractError("LEASH outputs must be atomic and no-clobber")
    return value


def validate_fit_row(row: Mapping[str, Any]) -> None:
    if set(row) != FIT_ALLOWED_FIELDS:
        raise LeashContractError(
            f"fit row field drift: missing={sorted(FIT_ALLOWED_FIELDS - set(row))}, "
            f"extra={sorted(set(row) - FIT_ALLOWED_FIELDS)}"
        )
    lowered = {str(key).lower() for key in row}
    leaked = lowered.intersection(FIT_FORBIDDEN_FIELDS)
    if leaked:
        raise LeashContractError(f"fit row leaks outcome fields: {sorted(leaked)}")
    if row.get("fidelity") != FIDELITY or row.get("setting_label") != "central":
        raise LeashContractError("fit row fidelity/setting drifted")
    if row.get("arm") not in ARMS or row.get("method_id") != f"{row.get('arm')}|central":
        raise LeashContractError("fit row arm/method mismatch")
    is_leash = row.get("arm") == "leash"
    if row.get("actual_policy_execution_observed") is not is_leash:
        raise LeashContractError("fit row policy-execution evidence does not match its arm")
    if row.get("policy_replay_verified") is not is_leash:
        raise LeashContractError("fit row policy replay status does not match its arm")
    if row.get("closure_evidence_verified") is not True:
        raise LeashContractError("fit row lacks token-level closure evidence")
    if is_leash:
        if row.get("policy_replay_fired") is not bool(row.get("stopped_early")):
            raise LeashContractError("fit row replay firing disagrees with stopped_early")
        expected_index = int(row["n_reasoning_tokens"]) if row["stopped_early"] else None
        if row.get("policy_replay_stop_index") != expected_index:
            raise LeashContractError("fit row replay stop index drifted")
    elif row.get("policy_replay_fired") is not None or row.get("policy_replay_stop_index") is not None:
        raise LeashContractError("control arm cannot carry LEASH replay output")
    for field in ("n_reasoning_tokens", "n_closure_tokens", "n_total_tokens"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LeashContractError(f"fit row has invalid {field}")
    if row["n_total_tokens"] != row["n_reasoning_tokens"] + row["n_closure_tokens"]:
        raise LeashContractError("fit row total tokens do not equal reasoning plus closure")
    if not isinstance(row.get("stopped_early"), bool) or not isinstance(row.get("closure_generated"), bool):
        raise LeashContractError("fit row stopping flags are not boolean")


def assert_no_forbidden_keys(value: Any, *, path: str = "$") -> None:
    """Recursively reject target/text fields from every fit-visible structure."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if name in FIT_FORBIDDEN_FIELDS:
                raise LeashContractError(f"fit-visible outcome leak at {path}.{key}")
            assert_no_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_no_forbidden_keys(child, path=f"{path}[{index}]")


def _rename_entry_noreplace_at(
    source_parent_fd: int,
    source_name: str,
    target_parent_fd: int,
    target_name: str,
) -> None:
    """Atomically rename held-dirfd entries without replacement."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes, target_bytes = os.fsencode(source_name), os.fsencode(target_name)
    if sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        if operation is None:
            raise RuntimeError("atomic no-replace LEASH publication is unavailable")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_parent_fd, source_bytes, target_parent_fd, target_bytes, 0x00000004
        )
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise RuntimeError("atomic no-replace LEASH publication is unavailable")
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(source_parent_fd, source_bytes, target_parent_fd, target_bytes, 1)
    else:
        raise RuntimeError(f"atomic no-replace LEASH publication is unsupported on {sys.platform}")
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"LEASH output entry already exists: {target_name}")
    raise OSError(number, os.strerror(number), target_name)


def _rename_directory_noreplace_at(
    parent_fd: int, source_name: str, target_name: str
) -> None:
    """Compatibility wrapper for a no-replace rename within one held parent."""

    _rename_entry_noreplace_at(parent_fd, source_name, parent_fd, target_name)


def _lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute normalized path without following any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_read_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _safe_stage_leaf(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
        or "\\" in value
    ):
        raise LeashContractError(f"unsafe LEASH stage filename: {value!r}")
    return value


def validate_safe_component(value: str, *, name: str = "LEASH path component") -> str:
    """Return one strict ASCII path component or fail before path composition."""

    if (
        not isinstance(value, str)
        or len(value) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
        or value in {".", ".."}
        or Path(value).is_absolute()
        or "/" in value
        or "\\" in value
    ):
        raise LeashContractError(f"unsafe {name}: {value!r}")
    return value


def _read_all(descriptor: int, *, max_bytes: int | None = None) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        total += len(block)
        if max_bytes is not None and total > max_bytes:
            raise LeashContractError("LEASH bound file exceeds registered byte size")
        chunks.append(block)


def _component_is_symlink(parent_fd: int, component: str) -> bool:
    try:
        return stat.S_ISLNK(
            os.stat(component, dir_fd=parent_fd, follow_symlinks=False).st_mode
        )
    except FileNotFoundError:
        return False


def _open_directory_nofollow(
    path: str | Path, *, create: bool, name: str
) -> tuple[Path, int]:
    """Traverse a directory with held dirfds, rejecting symlinks at every level."""

    candidate = _lexical_absolute_path(path)
    flags = _directory_open_flags()
    descriptor = os.open(candidate.anchor, flags)
    walked = Path(candidate.anchor)
    try:
        for component in candidate.parts[1:]:
            walked /= component
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise LeashContractError(f"{name} directory is missing: {walked}")
                try:
                    os.mkdir(component, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator won.  The no-follow open below decides
                    # whether it created a genuine directory or a forbidden link.
                    pass
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as error:
                    if _component_is_symlink(descriptor, component):
                        raise LeashContractError(
                            f"{name} path contains a symlink component: {walked}"
                        ) from error
                    raise LeashContractError(
                        f"cannot open {name} directory component: {walked}"
                    ) from error
            except OSError as error:
                if _component_is_symlink(descriptor, component):
                    raise LeashContractError(
                        f"{name} path contains a symlink component: {walked}"
                    ) from error
                raise LeashContractError(
                    f"cannot open {name} directory component: {walked}"
                ) from error
            os.close(descriptor)
            descriptor = child
        return candidate, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _verify_directory_binding(
    path: Path, descriptor: int, identity: tuple[int, int], *, name: str
) -> None:
    """Prove the lexical path still names the held no-follow directory."""

    try:
        _, probe = _open_directory_nofollow(path, create=False, name=name)
    except LeashContractError as error:
        raise LeashContractError(f"{name} path binding changed") from error
    try:
        if _identity(os.fstat(probe)) != identity or _identity(os.fstat(descriptor)) != identity:
            raise LeashContractError(f"{name} path binding changed")
    finally:
        os.close(probe)


def _read_regular_entry_at(
    parent_fd: int, leaf: str, *, name: str, expected_bytes: int | None = None
) -> tuple[bytes, tuple[int, int]]:
    """Read one named regular file once, with name/fd identity checks around I/O."""

    observed = _entry_stat(parent_fd, leaf)
    if observed is None or not stat.S_ISREG(observed.st_mode):
        raise LeashContractError(f"{name} is missing or is not a regular file")
    expected_identity = _identity(observed)
    try:
        descriptor = os.open(leaf, _file_read_open_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise LeashContractError(f"cannot open {name} without following links") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != expected_identity:
            raise LeashContractError(f"{name} name-to-fd identity binding changed")
        _require_entry_identity(parent_fd, leaf, expected_identity, label=name)
        payload = _read_all(descriptor, max_bytes=expected_bytes)
        if _identity(os.fstat(descriptor)) != expected_identity:
            raise LeashContractError(f"{name} descriptor identity changed during read")
        _require_entry_identity(parent_fd, leaf, expected_identity, label=name)
        # A malicious lstat hook can exchange the name after producing the
        # expected stat result.  Re-open the canonical name and bind that actual
        # object back to the descriptor identity before accepting the bytes.
        probe = os.open(leaf, _file_read_open_flags(), dir_fd=parent_fd)
        try:
            probe_info = os.fstat(probe)
            if not stat.S_ISREG(probe_info.st_mode) or _identity(probe_info) != expected_identity:
                raise LeashContractError(f"{name} final name-to-fd identity binding changed")
        finally:
            os.close(probe)
        return payload, expected_identity
    finally:
        os.close(descriptor)


def read_bound_bytes(
    path: str | Path,
    *,
    name: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    """Read a canonical file through held no-follow parent and object descriptors."""

    target = _lexical_absolute_path(path)
    if not target.name:
        raise LeashContractError(f"{name} path lacks a final component")
    parent, parent_fd = _open_directory_nofollow(
        target.parent, create=False, name=f"{name} parent"
    )
    parent_identity = _identity(os.fstat(parent_fd))
    try:
        _verify_directory_binding(
            parent, parent_fd, parent_identity, name=f"{name} parent"
        )
        payload, file_identity = _read_regular_entry_at(
            parent_fd, target.name, name=name, expected_bytes=expected_bytes
        )
        if expected_bytes is not None and len(payload) != int(expected_bytes):
            raise LeashContractError(f"{name} byte-size binding failed")
        if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
            raise LeashContractError(f"{name} SHA-256 binding failed")
        _verify_directory_binding(
            parent, parent_fd, parent_identity, name=f"{name} parent"
        )
        # The parent-path verification above is deliberately followed by one
        # final actual open of the leaf; neither a deceptive stat nor a parent
        # exchange can make bytes from a different inode pass.
        final_probe = os.open(target.name, _file_read_open_flags(), dir_fd=parent_fd)
        try:
            final_info = os.fstat(final_probe)
            if (
                not stat.S_ISREG(final_info.st_mode)
                or _identity(final_info) != file_identity
            ):
                raise LeashContractError(f"{name} final name-to-fd binding changed")
        finally:
            os.close(final_probe)
        return payload
    finally:
        os.close(parent_fd)


def read_authenticated_source_guard_code(
    code_root: str | Path, registry: Mapping[str, Any]
) -> dict[str, bytes]:
    """Read every pinned guard module once and return path-keyed exact bytes."""

    output: dict[str, bytes] = {}
    specs = registry["source_contract"]["source_guard_code_files"]
    for asset_id in sorted(SOURCE_GUARD_CODE_PATHS):
        spec = specs[asset_id]
        relative = SOURCE_GUARD_CODE_PATHS[asset_id]
        candidate = resolve_source_path(
            code_root, relative, name=f"LEASH source guard code::{asset_id}"
        )
        payload = read_bound_bytes(
            candidate,
            name=f"LEASH source guard code {asset_id}",
            expected_sha256=str(spec["sha256"]),
        )
        if relative in output:
            raise LeashContractError("duplicate LEASH source-guard capsule path")
        output[relative] = payload
    return output


def parse_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LeashContractError(f"invalid {name}") from error
    if not isinstance(value, dict):
        raise LeashContractError(f"{name} must be a JSON object")
    return value


def parse_jsonl_bytes(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise LeashContractError(f"invalid UTF-8 in {name}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise LeashContractError(f"{name} contains blank line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise LeashContractError(f"invalid {name} line {line_number}") from error
        if not isinstance(row, dict):
            raise LeashContractError(f"{name} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise LeashContractError(f"{name} is empty")
    return rows


def load_bound_json(path: str | Path, *, name: str) -> dict[str, Any]:
    return parse_json_bytes(read_bound_bytes(path, name=name), name=name)


def load_bound_jsonl(path: str | Path, *, name: str) -> list[dict[str, Any]]:
    return parse_jsonl_bytes(read_bound_bytes(path, name=name), name=name)


def bound_json_sha256(
    path: str | Path, expected: Mapping[str, Any], *, name: str
) -> str:
    """Hash and parse one bound JSON file from the exact same captured bytes."""

    payload = read_bound_bytes(path, name=name)
    if parse_json_bytes(payload, name=name) != dict(expected):
        raise LeashContractError(f"{name} content binding drifted")
    return sha256_bytes(payload)


def _bound_tree_files(
    directory_fd: int,
    *,
    prefix: PurePosixPath,
    name: str,
    physical: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    initial_names = sorted(os.listdir(directory_fd))
    files: list[dict[str, Any]] = []
    for leaf in initial_names:
        if leaf in {".", ".."} or "/" in leaf or "\\" in leaf:
            raise LeashContractError(f"{name} contains an unsafe entry name")
        observed = _entry_stat(directory_fd, leaf)
        if observed is None:
            raise LeashContractError(f"{name} changed during tree snapshot")
        relative = prefix / leaf
        if stat.S_ISLNK(observed.st_mode):
            raise LeashContractError(f"{name} contains a symlink")
        if stat.S_ISDIR(observed.st_mode):
            child = os.open(leaf, _directory_open_flags(), dir_fd=directory_fd)
            try:
                if _identity(os.fstat(child)) != _identity(observed):
                    raise LeashContractError(f"{name} directory identity changed")
                if physical is not None:
                    opened = os.fstat(child)
                    physical.append(
                        {
                            "path": relative.as_posix(),
                            "kind": "directory",
                            "dev": int(opened.st_dev),
                            "ino": int(opened.st_ino),
                            "nlink": int(opened.st_nlink),
                        }
                    )
                files.extend(
                    _bound_tree_files(
                        child, prefix=relative, name=name, physical=physical
                    )
                )
                _require_entry_identity(
                    directory_fd, leaf, _identity(observed), label=f"{name} directory"
                )
                probe = os.open(leaf, _directory_open_flags(), dir_fd=directory_fd)
                try:
                    if _identity(os.fstat(probe)) != _identity(observed):
                        raise LeashContractError(f"{name} final directory binding changed")
                finally:
                    os.close(probe)
            finally:
                os.close(child)
        elif stat.S_ISREG(observed.st_mode):
            payload, file_identity = _read_regular_entry_at(
                directory_fd, leaf, name=f"{name}/{relative.as_posix()}"
            )
            rebound = _entry_stat(directory_fd, leaf)
            if rebound is None or _identity(rebound) != file_identity:
                raise LeashContractError(f"{name} file identity changed")
            if physical is not None:
                physical.append(
                    {
                        "path": relative.as_posix(),
                        "kind": "file",
                        "dev": int(rebound.st_dev),
                        "ino": int(rebound.st_ino),
                        "nlink": int(rebound.st_nlink),
                    }
                )
            files.append(
                {
                    "path": relative.as_posix(),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )
        else:
            raise LeashContractError(f"{name} contains a non-file, non-directory entry")
    if sorted(os.listdir(directory_fd)) != initial_names:
        raise LeashContractError(f"{name} entry roster changed during tree snapshot")
    return files


def bound_tree_manifest(root: str | Path, *, name: str) -> dict[str, Any]:
    """Hash a tree through held no-follow dirfds and same-fd file bytes."""

    lexical, root_fd = _open_directory_nofollow(root, create=False, name=name)
    root_identity = _identity(os.fstat(root_fd))
    try:
        files = _bound_tree_files(root_fd, prefix=PurePosixPath(), name=name)
        _verify_directory_binding(lexical, root_fd, root_identity, name=name)
    finally:
        os.close(root_fd)
    payload: dict[str, Any] = {
        "schema_version": "canonical-tree-manifest-v1",
        "files": files,
    }
    payload["tree_sha256"] = sha256_bytes(canonical_json_bytes(files))
    return payload


def bound_tree_physical_snapshot(root: str | Path, *, name: str) -> dict[str, Any]:
    """Capture exact bytes plus inode/link evidence through held no-follow fds."""

    lexical, root_fd = _open_directory_nofollow(root, create=False, name=name)
    root_info = os.fstat(root_fd)
    root_identity = _identity(root_info)
    physical = [
        {
            "path": ".",
            "kind": "directory",
            "dev": int(root_info.st_dev),
            "ino": int(root_info.st_ino),
            "nlink": int(root_info.st_nlink),
        }
    ]
    try:
        files = _bound_tree_files(
            root_fd, prefix=PurePosixPath(), name=name, physical=physical
        )
        _verify_directory_binding(lexical, root_fd, root_identity, name=name)
        if _identity(os.fstat(root_fd)) != root_identity:
            raise LeashContractError(f"{name} root identity changed")
    finally:
        os.close(root_fd)
    tree: dict[str, Any] = {
        "schema_version": "canonical-tree-manifest-v1",
        "files": files,
    }
    tree["tree_sha256"] = sha256_bytes(canonical_json_bytes(files))
    return {
        "tree": tree,
        "root_identity": [root_identity[0], root_identity[1]],
        "entries": sorted(physical, key=lambda item: (item["path"], item["kind"])),
    }


def require_physically_disjoint_trees(
    roots: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    """Reject A/B aliasing, hardlinks, and drift across a complete gate read."""

    if len(roots) < 2:
        raise LeashContractError("physical-disjointness gate requires at least two trees")
    snapshots = {
        label: bound_tree_physical_snapshot(path, name=label)
        for label, path in roots.items()
    }
    owner_by_identity: dict[tuple[int, int], tuple[str, str]] = {}
    for label, snapshot in snapshots.items():
        local_identities: set[tuple[int, int]] = set()
        for entry in snapshot["entries"]:
            identity = (int(entry["dev"]), int(entry["ino"]))
            if identity in local_identities:
                raise LeashContractError(
                    f"{label} contains physically aliased entries"
                )
            local_identities.add(identity)
            if entry["kind"] == "file" and int(entry["nlink"]) != 1:
                raise LeashContractError(
                    f"{label} contains a hardlinked regular file: {entry['path']}"
                )
            previous = owner_by_identity.get(identity)
            if previous is not None:
                raise LeashContractError(
                    f"LEASH A/B trees are not physically disjoint: "
                    f"{previous[0]}/{previous[1]} aliases {label}/{entry['path']}"
                )
            owner_by_identity[identity] = (label, str(entry["path"]))

    # Re-read both bytes and inode/link evidence after the cross-tree comparison.
    # This closes a coordinated alias/substitution performed while its peer was
    # being inspected.
    for label, path in roots.items():
        rebound = bound_tree_physical_snapshot(path, name=f"{label} final binding")
        if rebound != snapshots[label]:
            raise LeashContractError(f"{label} changed during physical A/B verification")
    return {label: snapshot["tree"] for label, snapshot in snapshots.items()}


def _entry_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _require_entry_identity(
    parent_fd: int, name: str, identity: tuple[int, int], *, label: str
) -> None:
    observed = _entry_stat(parent_fd, name)
    if observed is None or _identity(observed) != identity:
        raise LeashContractError(f"{label} name-to-fd identity binding changed")


def _unique_entry_name(parent_fd: int, *, prefix: str, directory: bool) -> tuple[str, int | None]:
    for _ in range(128):
        candidate = f"{prefix}{secrets.token_hex(16)}"
        try:
            if directory:
                os.mkdir(candidate, mode=0o700, dir_fd=parent_fd)
                return candidate, None
            flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            return candidate, os.open(candidate, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique LEASH staging entry")


def _entry_names_by_identity(
    parent_fd: int, identity: tuple[int, int]
) -> list[str]:
    matches: list[str] = []
    for entry in sorted(os.listdir(parent_fd)):
        observed = _entry_stat(parent_fd, entry)
        if observed is not None and _identity(observed) == identity:
            matches.append(entry)
    return matches


def _quarantine_entry_by_identity(
    parent_fd: int,
    *,
    identity: tuple[int, int],
    preferred_name: str,
    quarantine_prefix: str,
    name: str,
) -> str:
    """Atomically move the matching inode to a random recovery name.

    A candidate is validated only *after* the atomic rename.  If a concurrent
    substitution won the race, that entry is restored rather than deleted and
    the search continues for the genuine inode.
    """

    for _ in range(16):
        candidates = [preferred_name]
        candidates.extend(
            entry
            for entry in _entry_names_by_identity(parent_fd, identity)
            if entry != preferred_name
        )
        for candidate in candidates:
            observed = _entry_stat(parent_fd, candidate)
            if observed is None or _identity(observed) != identity:
                continue
            quarantine = f"{quarantine_prefix}{secrets.token_hex(16)}"
            try:
                _rename_directory_noreplace_at(parent_fd, candidate, quarantine)
            except FileNotFoundError:
                continue
            quarantined = _entry_stat(parent_fd, quarantine)
            if quarantined is not None and _identity(quarantined) == identity:
                os.fsync(parent_fd)
                return quarantine
            try:
                _rename_directory_noreplace_at(parent_fd, quarantine, candidate)
            except (FileNotFoundError, FileExistsError):
                pass
        if not _entry_names_by_identity(parent_fd, identity):
            raise LeashContractError(f"{name} lost its recoverable directory entry")
    raise LeashContractError(f"{name} could not be identity-quarantined")


def _clear_directory_fd(directory_fd: int) -> None:
    """Delete staging-owned contents through the held directory object."""

    for entry in os.listdir(directory_fd):
        observed = _entry_stat(directory_fd, entry)
        if observed is None:
            continue
        if stat.S_ISDIR(observed.st_mode):
            child_fd = os.open(entry, _directory_open_flags(), dir_fd=directory_fd)
            try:
                if _identity(os.fstat(child_fd)) != _identity(observed):
                    raise LeashContractError("LEASH staging child identity changed")
                _clear_directory_fd(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(entry, dir_fd=directory_fd)
        else:
            # The held staging directory owns its current leaf entries.  This
            # never dereferences the mutable parent-level staging name.
            os.unlink(entry, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _remove_empty_directory_by_identity(
    parent_fd: int,
    directory_fd: int,
    *,
    identity: tuple[int, int],
    preferred_name: str,
    name: str,
) -> None:
    """Quarantine the genuine empty directory, then remove only that object."""

    if os.listdir(directory_fd):
        raise LeashContractError(f"{name} is not empty after fd-bound cleanup")
    _remove_known_empty_directory_by_identity(
        parent_fd,
        identity=identity,
        preferred_name=preferred_name,
        name=name,
    )


def _remove_known_empty_directory_by_identity(
    parent_fd: int,
    *,
    identity: tuple[int, int],
    preferred_name: str,
    name: str,
) -> None:
    """Quarantine a just-created empty directory without deleting by mutable name.

    POSIX has no rmdir-by-inode primitive.  Deleting the returned quarantine
    pathname would reintroduce a name-substitution race in which an unrelated
    empty directory can be exchanged immediately before ``rmdir``.  Recovery
    entries therefore remain for explicit forensic garbage collection.
    """

    quarantine = _quarantine_entry_by_identity(
        parent_fd,
        identity=identity,
        preferred_name=preferred_name,
        quarantine_prefix=f".{preferred_name}.leash-empty-quarantine-",
        name=name,
    )
    os.fsync(parent_fd)


class AtomicLeashDirectory:
    """Build and publish through one held, no-follow parent directory fd."""

    def __init__(self, final_path: str | Path) -> None:
        self.final_path = _lexical_absolute_path(final_path)
        if not self.final_path.name:
            raise LeashContractError("LEASH output path lacks a final component")
        parent, self._parent_fd = _open_directory_nofollow(
            self.final_path.parent, create=True, name="LEASH output parent"
        )
        self._parent_identity = _identity(os.fstat(self._parent_fd))
        self._closed = False
        self.committed = False
        self._entry_name: str | None = None
        self._stage_fd: int | None = None
        self._stage_identity: tuple[int, int] | None = None
        try:
            _verify_directory_binding(
                parent,
                self._parent_fd,
                self._parent_identity,
                name="LEASH output parent",
            )
            if _entry_stat(self._parent_fd, self.final_path.name) is not None:
                raise FileExistsError(
                    f"LEASH output directory already exists: {self.final_path}"
                )
            self._stage_name, _ = _unique_entry_name(
                self._parent_fd,
                prefix=f".{self.final_path.name}.leash-staging-",
                directory=True,
            )
            self._entry_name = self._stage_name
            stage_info = _entry_stat(self._parent_fd, self._stage_name)
            if stage_info is None or not stat.S_ISDIR(stage_info.st_mode):
                raise LeashContractError("LEASH staging directory creation failed")
            self._stage_identity = _identity(stage_info)
            try:
                self._stage_fd = os.open(
                    self._stage_name, _directory_open_flags(), dir_fd=self._parent_fd
                )
            except BaseException:
                _remove_known_empty_directory_by_identity(
                    self._parent_fd,
                    identity=self._stage_identity,
                    preferred_name=self._stage_name,
                    name="LEASH just-created staging directory",
                )
                self._entry_name = None
                raise
            bound_name = _entry_stat(self._parent_fd, self._stage_name)
            if (
                _identity(os.fstat(self._stage_fd)) != self._stage_identity
                or bound_name is None
                or _identity(bound_name) != self._stage_identity
            ):
                os.close(self._stage_fd)
                self._stage_fd = None
                _remove_known_empty_directory_by_identity(
                    self._parent_fd,
                    identity=self._stage_identity,
                    preferred_name=self._stage_name,
                    name="LEASH just-created staging directory",
                )
                self._entry_name = None
                raise LeashContractError("LEASH staging directory identity changed")
            os.fsync(self._parent_fd)
        except BaseException:
            try:
                if self._stage_fd is not None and self._stage_identity is not None:
                    recovery = _quarantine_entry_by_identity(
                        self._parent_fd,
                        identity=self._stage_identity,
                        preferred_name=self._stage_name,
                        quarantine_prefix=f".{self.final_path.name}.leash-recovery-",
                        name="LEASH staging directory after initialization failure",
                    )
                    self._entry_name = recovery
            finally:
                if self._stage_fd is not None:
                    os.close(self._stage_fd)
                os.close(self._parent_fd)
                self._closed = True
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("LEASH stage is closed")

    def _require_stage_identity(self) -> None:
        if (
            self._stage_fd is None
            or self._stage_identity is None
            or _identity(os.fstat(self._stage_fd)) != self._stage_identity
        ):
            raise LeashContractError("LEASH staging directory identity changed")
        info = _entry_stat(self._parent_fd, self._stage_name)
        if (
            info is None
            or not stat.S_ISDIR(info.st_mode)
            or _identity(info) != self._stage_identity
        ):
            raise LeashContractError("LEASH staging directory identity changed")

    def write_bytes(self, filename: str, payload: bytes) -> str:
        """Create one flat staging file through the held stage directory fd."""

        self._require_open()
        if self.committed:
            raise RuntimeError("cannot write a committed LEASH stage")
        self._require_stage_identity()
        if self._stage_fd is None:  # pragma: no cover - guarded above
            raise RuntimeError("LEASH stage descriptor is unavailable")
        leaf = _safe_stage_leaf(filename)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(leaf, flags, 0o600, dir_fd=self._stage_fd)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(self._stage_fd)
        return sha256_bytes(payload)

    def write_json(self, filename: str, value: Any) -> str:
        return self.write_bytes(filename, canonical_json_bytes(value) + b"\n")

    def read_bytes(self, filename: str) -> bytes:
        """Read one flat staging file through the held stage directory fd."""

        self._require_open()
        self._require_stage_identity()
        if self._stage_fd is None:  # pragma: no cover - guarded above
            raise RuntimeError("LEASH stage descriptor is unavailable")
        leaf = _safe_stage_leaf(filename)
        observed = _entry_stat(self._stage_fd, leaf)
        if observed is None or not stat.S_ISREG(observed.st_mode):
            raise LeashContractError(f"LEASH stage file is missing or non-regular: {leaf}")
        descriptor = os.open(leaf, _file_read_open_flags(), dir_fd=self._stage_fd)
        try:
            if _identity(os.fstat(descriptor)) != _identity(observed):
                raise LeashContractError(f"LEASH stage file identity changed: {leaf}")
            return _read_all(descriptor)
        finally:
            os.close(descriptor)

    def load_json(self, filename: str, *, name: str) -> dict[str, Any]:
        try:
            value = json.loads(self.read_bytes(filename).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LeashContractError(f"invalid {name} in LEASH stage") from error
        if not isinstance(value, dict):
            raise LeashContractError(f"{name} in LEASH stage must be a JSON object")
        return value

    def tree_manifest(self) -> dict[str, Any]:
        """Hash the flat staged tree entirely through the held directory fd."""

        self._require_open()
        self._require_stage_identity()
        if self._stage_fd is None:  # pragma: no cover - guarded above
            raise RuntimeError("LEASH stage descriptor is unavailable")
        files: list[dict[str, Any]] = []
        for leaf in sorted(os.listdir(self._stage_fd)):
            _safe_stage_leaf(leaf)
            payload = self.read_bytes(leaf)
            files.append(
                {"path": leaf, "bytes": len(payload), "sha256": sha256_bytes(payload)}
            )
        manifest: dict[str, Any] = {
            "schema_version": "canonical-tree-manifest-v1",
            "files": files,
        }
        manifest["tree_sha256"] = sha256_bytes(canonical_json_bytes(files))
        return manifest

    def commit(self) -> None:
        self._require_open()
        if self.committed:
            raise RuntimeError("LEASH stage was already committed")
        self._require_stage_identity()
        _verify_directory_binding(
            self.final_path.parent,
            self._parent_fd,
            self._parent_identity,
            name="LEASH output parent",
        )
        _rename_directory_noreplace_at(
            self._parent_fd, self._stage_name, self.final_path.name
        )
        self._entry_name = self.final_path.name
        published_fd: int | None = None
        try:
            published = _entry_stat(self._parent_fd, self.final_path.name)
            if published is None or _identity(published) != self._stage_identity:
                raise LeashContractError("LEASH published directory identity changed")
            published_fd = os.open(
                self.final_path.name, _directory_open_flags(), dir_fd=self._parent_fd
            )
            if _identity(os.fstat(published_fd)) != self._stage_identity:
                raise LeashContractError("LEASH published directory fd identity changed")
            os.fsync(published_fd)
            os.fsync(self._parent_fd)
            _verify_directory_binding(
                self.final_path.parent,
                self._parent_fd,
                self._parent_identity,
                name="LEASH output parent",
            )
            _require_entry_identity(
                self._parent_fd,
                self.final_path.name,
                self._stage_identity,
                label="LEASH published directory",
            )
            # Re-open after the final stat-based assertion.  This catches an
            # exchange performed inside a deceptive stat hook before it returns.
            final_probe = os.open(
                self.final_path.name, _directory_open_flags(), dir_fd=self._parent_fd
            )
            try:
                if _identity(os.fstat(final_probe)) != self._stage_identity:
                    raise LeashContractError(
                        "LEASH published directory final binding changed"
                    )
                os.fsync(final_probe)
            finally:
                os.close(final_probe)
            _verify_directory_binding(
                self.final_path.parent,
                self._parent_fd,
                self._parent_identity,
                name="LEASH output parent",
            )
            last_probe = os.open(
                self.final_path.name, _directory_open_flags(), dir_fd=self._parent_fd
            )
            try:
                if _identity(os.fstat(last_probe)) != self._stage_identity:
                    raise LeashContractError(
                        "LEASH published directory last binding changed"
                    )
            finally:
                os.close(last_probe)
        except BaseException as error:
            recovery_notes: list[str] = []
            actual = _entry_stat(self._parent_fd, self.final_path.name)
            if actual is not None and _identity(actual) != self._stage_identity:
                try:
                    foreign = _quarantine_entry_by_identity(
                        self._parent_fd,
                        identity=_identity(actual),
                        preferred_name=self.final_path.name,
                        quarantine_prefix=f".{self.final_path.name}.leash-foreign-recovery-",
                        name="substituted LEASH published directory",
                    )
                    recovery_notes.append(f"foreign publication preserved as {foreign!r}")
                except BaseException as quarantine_error:
                    recovery_notes.append(
                        f"foreign publication quarantine failed: {quarantine_error}"
                    )
            try:
                genuine = _quarantine_entry_by_identity(
                    self._parent_fd,
                    identity=self._stage_identity,
                    preferred_name=self.final_path.name,
                    quarantine_prefix=f".{self.final_path.name}.leash-recovery-",
                    name="genuine LEASH published directory",
                )
                self._entry_name = genuine
                recovery_notes.append(f"genuine publication preserved as {genuine!r}")
            except BaseException as quarantine_error:
                recovery_notes.append(
                    f"genuine publication quarantine failed: {quarantine_error}"
                )
                # Cleanup must never clear through a name whose identity was not
                # recovered after a failed publication.
                self.committed = True
            if hasattr(error, "add_note"):
                for note in recovery_notes:
                    error.add_note(note)
            raise
        finally:
            if published_fd is not None:
                os.close(published_fd)
        self.committed = True

    def rollback(self) -> None:
        self._require_open()
        if self.committed:
            published = _entry_stat(self._parent_fd, self.final_path.name)
            if published is None or _identity(published) != self._stage_identity:
                raise LeashContractError("LEASH published directory identity changed")
            _rename_directory_noreplace_at(
                self._parent_fd, self.final_path.name, self._stage_name
            )
            self._entry_name = self._stage_name
            self.committed = False

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            if not self.committed and self._entry_name == self._stage_name:
                if (
                    self._stage_fd is None
                    or self._stage_identity is None
                    or _identity(os.fstat(self._stage_fd)) != self._stage_identity
                ):
                    raise LeashContractError("LEASH staging directory identity changed")
                recovery = _quarantine_entry_by_identity(
                    self._parent_fd,
                    identity=self._stage_identity,
                    preferred_name=self._stage_name,
                    quarantine_prefix=f".{self.final_path.name}.leash-recovery-",
                    name="LEASH staging directory cleanup",
                )
                self._entry_name = recovery
        finally:
            if self._stage_fd is not None:
                os.close(self._stage_fd)
            os.close(self._parent_fd)
            self._closed = True


def leash_tree_write_bytes(
    stage: AtomicLeashDirectory, filename: str, payload: bytes
) -> str:
    """Write a tree leaf through the held stage directory object."""

    leaf = _safe_stage_leaf(filename)
    return stage.write_bytes(leaf, payload)


def leash_tree_write_json(
    stage: AtomicLeashDirectory, filename: str, value: Any
) -> str:
    leaf = _safe_stage_leaf(filename)
    return stage.write_json(leaf, value)


def leash_tree_manifest(stage: AtomicLeashDirectory) -> dict[str, Any]:
    return stage.tree_manifest()


def leash_tree_load_json(
    stage: AtomicLeashDirectory, filename: str, *, name: str
) -> dict[str, Any]:
    leaf = _safe_stage_leaf(filename)
    return stage.load_json(leaf, name=name)


def write_json_noreplace(path: str | Path, value: Any) -> str:
    target = _lexical_absolute_path(path)
    if not target.name:
        raise LeashContractError("LEASH certificate path lacks a final component")
    parent, parent_fd = _open_directory_nofollow(
        target.parent, create=True, name="LEASH certificate parent"
    )
    parent_identity = _identity(os.fstat(parent_fd))
    if _entry_stat(parent_fd, target.name) is not None:
        os.close(parent_fd)
        raise FileExistsError(f"LEASH certificate already exists: {target}")
    payload = canonical_json_bytes(value) + b"\n"
    try:
        temporary_name, descriptor_or_none = _unique_entry_name(
            parent_fd, prefix=f".{target.name}.leash-tmp-", directory=False
        )
    except BaseException:
        os.close(parent_fd)
        raise
    if descriptor_or_none is None:  # pragma: no cover - impossible by construction
        os.close(parent_fd)
        raise RuntimeError("LEASH certificate staging descriptor is missing")
    descriptor = descriptor_or_none
    temporary_identity = _identity(os.fstat(descriptor))
    published_fd: int | None = None
    try:
        _require_entry_identity(
            parent_fd,
            temporary_name,
            temporary_identity,
            label="LEASH certificate staging file",
        )
        _verify_directory_binding(
            parent, parent_fd, parent_identity, name="LEASH certificate parent"
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        _verify_directory_binding(
            parent, parent_fd, parent_identity, name="LEASH certificate parent"
        )
        _require_entry_identity(
            parent_fd,
            temporary_name,
            temporary_identity,
            label="LEASH certificate staging file",
        )
        try:
            _rename_entry_noreplace_at(
                parent_fd, temporary_name, parent_fd, target.name
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"LEASH certificate already exists: {target}"
            ) from error
        os.fsync(parent_fd)
        _verify_directory_binding(
            parent, parent_fd, parent_identity, name="LEASH certificate parent"
        )
        published = _entry_stat(parent_fd, target.name)
        if published is None:
            raise LeashContractError("LEASH published certificate disappeared")
        if _identity(published) != temporary_identity:
            foreign_identity = _identity(published)
            foreign_quarantine = _quarantine_entry_by_identity(
                parent_fd,
                identity=foreign_identity,
                preferred_name=target.name,
                quarantine_prefix=f".{target.name}.leash-foreign-recovery-",
                name="substituted LEASH certificate publication",
            )
            error = LeashContractError("LEASH published certificate identity changed")
            if hasattr(error, "add_note"):
                error.add_note(
                    f"substituted certificate preserved as {foreign_quarantine!r}"
                )
            raise error
        published_fd = os.open(target.name, _file_read_open_flags(), dir_fd=parent_fd)
        published_info = os.fstat(published_fd)
        if (
            not stat.S_ISREG(published_info.st_mode)
            or _identity(published_info) != temporary_identity
        ):
            raise LeashContractError("LEASH published certificate fd identity changed")
        os.lseek(published_fd, 0, os.SEEK_SET)
        if _read_all(published_fd) != payload:
            raise LeashContractError("LEASH published certificate bytes changed")
        _require_entry_identity(
            parent_fd,
            target.name,
            temporary_identity,
            label="LEASH published certificate",
        )
        # Re-open after the last stat-based assertion so an exchange performed
        # inside that assertion cannot be accepted.
        final_probe = os.open(target.name, _file_read_open_flags(), dir_fd=parent_fd)
        try:
            final_info = os.fstat(final_probe)
            if (
                not stat.S_ISREG(final_info.st_mode)
                or _identity(final_info) != temporary_identity
            ):
                raise LeashContractError("LEASH published certificate final binding changed")
        finally:
            os.close(final_probe)
        _verify_directory_binding(
            parent, parent_fd, parent_identity, name="LEASH certificate parent"
        )
        last_probe = os.open(target.name, _file_read_open_flags(), dir_fd=parent_fd)
        try:
            if _identity(os.fstat(last_probe)) != temporary_identity:
                raise LeashContractError("LEASH published certificate last binding changed")
        finally:
            os.close(last_probe)
        return sha256_bytes(payload)
    except BaseException as error:
        try:
            actual = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            actual = None
        if actual is not None and _identity(actual) != temporary_identity:
            try:
                foreign_quarantine = _quarantine_entry_by_identity(
                    parent_fd,
                    identity=_identity(actual),
                    preferred_name=target.name,
                    quarantine_prefix=f".{target.name}.leash-foreign-recovery-",
                    name="substituted LEASH certificate publication",
                )
                if hasattr(error, "add_note"):
                    error.add_note(
                        f"substituted certificate preserved as {foreign_quarantine!r}"
                    )
            except BaseException as quarantine_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        f"substituted certificate quarantine failed: {quarantine_error}"
                    )
        quarantine = _quarantine_entry_by_identity(
            parent_fd,
            identity=temporary_identity,
            preferred_name=temporary_name,
            quarantine_prefix=f".{target.name}.leash-recovery-",
            name="LEASH certificate staging file",
        )
        if hasattr(error, "add_note"):
            error.add_note(
                f"genuine certificate staging file preserved as recoverable entry {quarantine!r}"
            )
        raise
    finally:
        if published_fd is not None:
            os.close(published_fd)
        os.close(descriptor)
        os.close(parent_fd)


def verify_ready_tree(source_root: str | Path, spec: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = resolve_source_path(source_root, str(spec["path"]), name=str(spec["run_id"]))
    if not path.is_dir():
        raise LeashContractError(f"ready LEASH source is not a directory: {path}")
    assert_no_symlinks(path, name=f"ready LEASH source {spec['run_id']}")
    tree = bound_tree_manifest(path, name=f"ready LEASH source {spec['run_id']}")
    observed_bytes = sum(int(item["bytes"]) for item in tree["files"])
    if (
        len(tree["files"]) != int(spec["file_count"])
        or observed_bytes != int(spec["bytes_total"])
        or tree["tree_sha256"] != spec["tree_sha256"]
    ):
        raise LeashContractError(f"source tree binding failed for {spec['run_id']}")
    control = {
        "RUN_MANIFEST.json": spec["manifest_sha256"],
        "INDEX.jsonl": spec["index_sha256"],
        "STATUS.json": spec["status_sha256"],
        "SUMMARY.json": spec["summary_sha256"],
    }
    by_path = {item["path"]: item["sha256"] for item in tree["files"]}
    if any(by_path.get(name) != digest for name, digest in control.items()):
        raise LeashContractError(f"source control-file binding failed for {spec['run_id']}")
    return path, tree


def verify_blocked_run(source_root: str | Path, spec: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = resolve_source_path(source_root, str(spec["path"]), name=str(spec["run_id"]))
    if not path.is_dir():
        raise LeashContractError(f"blocked LEASH source is not a directory: {path}")
    assert_no_symlinks(path, name=f"blocked LEASH source {spec['run_id']}")
    bound_payloads: dict[str, bytes] = {}
    for name, digest in spec["files"].items():
        file_path = path / name
        bound_payloads[name] = read_bound_bytes(
            file_path,
            name=f"blocked source {spec['run_id']}/{name}",
            expected_sha256=digest,
        )
    manifest = parse_json_bytes(
        bound_payloads["RUN_MANIFEST.json"], name="blocked RUN_MANIFEST"
    )
    status = parse_json_bytes(bound_payloads["STATUS.json"], name="blocked STATUS")
    gate = parse_json_bytes(
        bound_payloads["GATE_S2-leash-full.json"], name="blocked acquisition gate"
    )
    if (
        manifest.get("run_id") != spec["run_id"]
        or manifest.get("model_id") != spec["model"]
        or manifest.get("fidelity") != FIDELITY
        or int(manifest.get("expected_traces", -1)) != int(spec["expected_traces"])
    ):
        raise LeashContractError(f"blocked manifest semantics drifted for {spec['run_id']}")
    if (
        status.get("complete") is not True
        or int(status.get("n_expected", -1)) != int(spec["expected_traces"])
        or int(status.get("n_failed", -1)) != int(spec["expected_failed"])
        or int(status.get("n_finished", -1)) != 0
        or int(status.get("n_shards", -1)) != 0
    ):
        raise LeashContractError(f"blocked status semantics drifted for {spec['run_id']}")
    failures = status.get("failures")
    if not isinstance(failures, list) or not failures or any(
        spec["failure_signature"] not in str(item.get("reason")) for item in failures
    ):
        raise LeashContractError(f"blocked failure signature drifted for {spec['run_id']}")
    if gate.get("passed") is not True:
        raise LeashContractError(
            f"{spec['run_id']} acquisition-manifest gate unexpectedly did not pass"
        )
    return path, {
        "run_id": spec["run_id"],
        "dataset": spec["dataset"],
        "model": spec["model"],
        "coverage_status": BLOCKED_STATUS,
        "reason": "all expected traces failed because the base tokenizer had no chat template",
        "n_expected": int(spec["expected_traces"]),
        "n_finished": 0,
        "n_failed": int(spec["expected_failed"]),
        "acquisition_manifest_gate_passed": True,
        "usable_for_evaluation": False,
        "fidelity": FIDELITY,
    }


def manifest_files(root: str | Path) -> dict[str, str]:
    tree = bound_tree_manifest(root, name="LEASH manifest tree")
    return {item["path"]: item["sha256"] for item in tree["files"]}


__all__ = [
    "ARMS", "AtomicLeashDirectory", "BLOCKED_STATUS", "DATASETS", "EVALUATION_AB_SCHEMA",
    "EVALUATION_SCHEMA", "FIDELITY", "FIT_AB_SCHEMA", "FIT_ALLOWED_FIELDS", "FIT_FORBIDDEN_FIELDS",
    "FIT_SCHEMA", "LeashContractError", "PREPARATION_AB_SCHEMA", "PREPARATION_SCHEMA",
    "PRIVATE_OUTCOME_SCHEMA", "READY_STATUS", "REGISTRY_SCHEMA", "SEARCHABLE_TABLES",
    "SOURCE_GUARD_CODE_PATHS",
    "add_payload_sha256", "assert_no_forbidden_keys", "bound_json_sha256",
    "bound_tree_manifest", "bound_tree_physical_snapshot",
    "canonical_jsonl_bytes", "load_bound_json", "load_bound_jsonl", "load_json",
    "assert_no_symlinks",
    "leash_tree_load_json", "leash_tree_manifest", "leash_tree_write_bytes",
    "leash_tree_write_json", "load_jsonl", "load_registry", "manifest_files",
    "parse_json_bytes", "parse_jsonl_bytes", "payload_sha256", "read_bound_bytes",
    "read_authenticated_source_guard_code", "require_physically_disjoint_trees",
    "resolve_source_path", "source_guard_closure_sha256", "validate_safe_component",
    "validate_fit_row", "verify_blocked_run", "verify_payload", "verify_ready_tree",
    "write_json_noreplace",
]
