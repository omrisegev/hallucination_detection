"""Post-freeze all-pairs and point-winner contrast artifacts.

This module is deliberately downstream of the scientific evaluators.  It
never fits a method, changes a score, or writes into a scientific release.  It
accepts only already-certified evaluation payloads and derives paired
contrasts from the exact shared grouped-bootstrap draw contract.

The phrase "winner-reference set" has a narrow meaning here: a method is in
the set when its *direct paired* 95% bootstrap interval versus the observed
point winner contains numerical zero.  This is not an equivalence test, does
not adjust for selecting the winner on the same data, and has no simultaneous
or family-wise coverage guarantee.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import csv
import errno
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import external_evaluation as _external
from .io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "reconstruction-winner-reference-contrasts-v1"
DRAW_AUDIT_SCHEMA_VERSION = "reconstruction-winner-reference-draw-audit-v1"
AB_VERIFICATION_SCHEMA_VERSION = "reconstruction-winner-reference-ab-verification-v1"

CANONICAL_DRAWS = 20_000
METRIC_DIRECTIONS = {
    "auroc": 1,
    "auprc": 1,
    "aurc_x1000": -1,
}
NUMERICAL_ZERO_ATOL = {
    "auroc": 1e-14,
    "auprc": 1e-14,
    "aurc_x1000": 2e-10,
}
SUCCESS_STATUSES = frozenset({"OK", "OK_FALLBACK"})

WINNER_CONTRAST_CODE_PATHS = (
    "spectral_utils/reconstruction_benchmark/winner_contrasts.py",
    "scripts/reconstruction_benchmark/build_winner_contrasts.py",
    "scripts/reconstruction_benchmark/verify_winner_contrasts_ab.py",
    "scripts/reconstruction_benchmark/test_winner_contrasts.py",
)
_WINNER_CONTRAST_REPO = Path(__file__).resolve().parents[2]

ALL_PAIRS_FIELDS = (
    "comparison_group_id", "source_comparison_group_id", "lane_id",
    "population_id", "scope_type", "scope_value", "record_level",
    "cell_id", "dataset_id", "model_id", "slice_id", "cell_ids_json",
    "n_cells", "aggregation", "metric_id", "higher_is_better",
    "method_a_id", "method_b_id", "method_a_value", "method_b_value",
    "delta_a_minus_b", "delta_ci_low", "delta_ci_high",
    "oriented_advantage_a_over_b", "oriented_ci_low", "oriented_ci_high",
    "probability_oriented_advantage_le_zero", "bootstrap_unit",
    "bootstrap_draws", "bootstrap_valid_draws", "relation",
    "multiplicity_adjustment", "status",
)

WINNER_CONTRAST_FIELDS = (
    "comparison_group_id", "source_comparison_group_id", "lane_id",
    "population_id", "scope_type", "scope_value", "record_level",
    "cell_id", "dataset_id", "model_id", "slice_id", "cell_ids_json",
    "n_cells", "aggregation", "metric_id", "higher_is_better",
    "point_winner_method_ids_json", "winner_reference_method_id",
    "candidate_method_id", "winner_value", "candidate_value",
    "delta_candidate_minus_winner", "delta_ci_low", "delta_ci_high",
    "oriented_advantage_candidate_over_winner", "oriented_ci_low",
    "oriented_ci_high", "probability_oriented_advantage_le_zero",
    "bootstrap_unit", "bootstrap_draws", "bootstrap_valid_draws",
    "membership_status", "in_winner_reference_set",
    "multiplicity_adjustment", "status",
)

WINNER_SET_FIELDS = (
    "comparison_group_id", "source_comparison_group_id", "lane_id",
    "population_id", "scope_type", "scope_value", "record_level",
    "cell_id", "dataset_id", "model_id", "slice_id", "cell_ids_json",
    "n_cells", "aggregation", "metric_id", "higher_is_better",
    "point_winner_method_ids_json", "winner_reference_method_id",
    "method_id", "method_value", "membership_status",
    "in_winner_reference_set", "interpretation",
)


class WinnerContrastError(RuntimeError):
    """Raised when a certified source or derived artifact fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WinnerContrastError(message)


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def _sorted_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted((str(value) for value in values), key=_utf8_key))


def _json_text(value: object) -> str:
    return canonical_json_bytes(value).decode("ascii")


def _validate_hashed_payload(value: Mapping[str, Any], *, name: str,
                             field: str = "payload_sha256") -> None:
    payload = dict(value)
    recorded = payload.pop(field, None)
    _require(isinstance(recorded, str), f"{name}: missing {field}")
    _require(
        recorded == sha256_bytes(canonical_json_bytes(payload)),
        f"{name}: {field} mismatch",
    )


def _require_plain_file(path: Path, *, name: str) -> None:
    _require(path.exists(), f"{name}: missing file {path}")
    _require(path.is_file() and not path.is_symlink(), f"{name}: not a plain file {path}")


def _git_stdout(repo_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(repo_root), *arguments),
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise WinnerContrastError(
            "cannot capture the winner-contrast git repository snapshot"
        ) from error
    return completed.stdout


def _validate_repo_snapshot(
    snapshot: Mapping[str, Any], *, name: str,
) -> None:
    expected_fields = {
        "git_head", "git_clean", "git_status_sha256", "code_files",
        "snapshot_sha256",
    }
    _require(set(snapshot) == expected_fields,
             f"{name}: repository snapshot field roster drifted")
    head = snapshot.get("git_head")
    _require(
        isinstance(head, str) and len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head),
        f"{name}: invalid git HEAD",
    )
    _require(snapshot.get("git_clean") is True,
             f"{name}: repository snapshot is not clean")
    status_hash = snapshot.get("git_status_sha256")
    _require(
        status_hash == sha256_bytes(b""),
        f"{name}: clean git status digest mismatch",
    )
    code_files = snapshot.get("code_files")
    _require(
        isinstance(code_files, Mapping)
        and set(code_files) == set(WINNER_CONTRAST_CODE_PATHS),
        f"{name}: exact winner-contrast code roster drifted",
    )
    for relative_path in WINNER_CONTRAST_CODE_PATHS:
        binding = code_files[relative_path]
        _require(
            isinstance(binding, Mapping)
            and set(binding) == {"sha256", "bytes"}
            and isinstance(binding.get("sha256"), str)
            and len(str(binding["sha256"])) == 64
            and all(
                character in "0123456789abcdef"
                for character in str(binding["sha256"])
            )
            and isinstance(binding.get("bytes"), int)
            and int(binding["bytes"]) >= 0,
            f"{name}: invalid code binding for {relative_path}",
        )
    payload = dict(snapshot)
    recorded = payload.pop("snapshot_sha256")
    _require(
        recorded == sha256_bytes(canonical_json_bytes(payload)),
        f"{name}: repository snapshot payload hash mismatch",
    )


def _capture_repo_snapshot(
    repo_root: str | Path,
    *,
    code_paths: Sequence[str] = WINNER_CONTRAST_CODE_PATHS,
) -> dict[str, Any]:
    """Capture a clean committed checkout and the exact producer code roster."""

    root = Path(repo_root).resolve()
    _require(tuple(code_paths) == WINNER_CONTRAST_CODE_PATHS,
             "winner-contrast code snapshot must use the exact frozen roster")
    top_level = Path(
        os.fsdecode(_git_stdout(root, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    _require(top_level == root,
             "winner-contrast repository root differs from git top level")
    head = os.fsdecode(
        _git_stdout(root, "rev-parse", "--verify", "HEAD")
    ).strip()
    status = _git_stdout(
        root, "status", "--porcelain=v1", "--untracked-files=normal",
    )
    _require(status == b"",
             "winner-contrast publication/verification requires a clean git checkout")
    tracked = {
        line for line in os.fsdecode(
            _git_stdout(root, "ls-files", "--", *WINNER_CONTRAST_CODE_PATHS)
        ).splitlines() if line
    }
    _require(
        tracked == set(WINNER_CONTRAST_CODE_PATHS),
        "winner-contrast code snapshot contains an untracked or missing source file",
    )
    code_files: dict[str, dict[str, Any]] = {}
    for relative_path in WINNER_CONTRAST_CODE_PATHS:
        path = root / relative_path
        _require_plain_file(path, name=f"winner-contrast code file {relative_path}")
        payload = path.read_bytes()
        code_files[relative_path] = {
            "sha256": sha256_bytes(payload), "bytes": len(payload),
        }
    snapshot: dict[str, Any] = {
        "git_head": head,
        "git_clean": True,
        "git_status_sha256": sha256_bytes(status),
        "code_files": code_files,
    }
    snapshot["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
    _validate_repo_snapshot(snapshot, name="captured winner-contrast repo snapshot")
    return snapshot


def _current_repo_snapshot() -> dict[str, Any]:
    return _capture_repo_snapshot(_WINNER_CONTRAST_REPO)


def _require_repo_snapshot_match(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, context: str,
) -> None:
    _validate_repo_snapshot(expected, name=f"{context} expected snapshot")
    _validate_repo_snapshot(observed, name=f"{context} observed snapshot")
    _require(expected.get("git_head") == observed.get("git_head"),
             f"{context}: git HEAD mismatch")
    _require(
        expected.get("git_status_sha256") == observed.get("git_status_sha256"),
        f"{context}: git clean-status digest mismatch",
    )
    _require(expected.get("code_files") == observed.get("code_files"),
             f"{context}: winner-contrast code roster/hash mismatch")
    _require(expected == observed,
             f"{context}: repository snapshot mismatch")


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing a raced target."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        operation = getattr(libc, "renamex_np", None)
        _require(operation is not None,
                 "atomic no-replace directory publication is unavailable")
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        _require(operation is not None,
                 "atomic no-replace directory publication is unavailable")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(-100, source_bytes, -100, target_bytes, 1)  # RENAME_NOREPLACE
    else:
        raise WinnerContrastError(
            f"atomic no-replace directory publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"winner-contrast output already exists: {target}")
    raise OSError(error_number, os.strerror(error_number), os.fspath(target))


def _atomic_write_bytes_noclobber(target: Path, payload: bytes) -> None:
    """Stage bytes, then atomically hard-link them into a new final path."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.stage.", dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"winner-contrast A/B certificate already exists: {target}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    _require_plain_file(path, name=name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser error is incidental
        raise WinnerContrastError(f"{name}: invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"{name}: JSON root is not an object")
    return value


def _read_csv(path: Path, *, name: str) -> list[dict[str, str]]:
    _require_plain_file(path, name=name)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"{name}: missing CSV header")
        return [dict(row) for row in reader]


def _parse_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", ""}:
        return False
    raise WinnerContrastError(f"{name}: invalid boolean {value!r}")


def _float_equal(left: float, right: float, *, atol: float = 2e-12) -> bool:
    return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol))


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return sha256_bytes(array.tobytes(order="C"))


def _quantiles(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    _require(len(finite) > 0, "paired contrast has no valid bootstrap draws")
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


@dataclass(frozen=True)
class ScopeMetric:
    """One method-complete comparison scope and one metric."""

    lane_id: str
    population_id: str
    scope_type: str
    scope_value: str
    record_level: str
    cell_id: str
    dataset_id: str
    model_id: str
    slice_id: str
    cell_ids: tuple[str, ...]
    aggregation: str
    source_comparison_group_id: str
    bootstrap_unit: str
    bootstrap_seed: int | None
    linked_resampling: bool
    stratified_by_label: bool
    bootstrap_draws_requested: int
    points: Mapping[str, float]
    draws: Mapping[str, np.ndarray]
    method_statuses: Mapping[str, str]


@dataclass(frozen=True)
class EvaluatedScopeMetric(ScopeMetric):
    metric: str

    @property
    def metric_id(self) -> str:
        return self.metric


@dataclass(frozen=True)
class LoadedSource:
    source_type: str
    lane_id: str
    method_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    scopes: tuple[EvaluatedScopeMetric, ...]
    source_binding: Mapping[str, Any]


def _comparison_group_id(scope: EvaluatedScopeMetric) -> str:
    payload = {
        "lane_id": scope.lane_id,
        "population_id": scope.population_id,
        "scope_type": scope.scope_type,
        "scope_value": scope.scope_value,
        "record_level": scope.record_level,
        "cell_ids": list(scope.cell_ids),
        "metric_id": scope.metric_id,
        "source_comparison_group_id": scope.source_comparison_group_id,
    }
    return "winner_reference::" + sha256_bytes(canonical_json_bytes(payload))[:24]


def _base_row(scope: EvaluatedScopeMetric) -> dict[str, Any]:
    return {
        "comparison_group_id": _comparison_group_id(scope),
        "source_comparison_group_id": scope.source_comparison_group_id,
        "lane_id": scope.lane_id,
        "population_id": scope.population_id,
        "scope_type": scope.scope_type,
        "scope_value": scope.scope_value,
        "record_level": scope.record_level,
        "cell_id": scope.cell_id,
        "dataset_id": scope.dataset_id,
        "model_id": scope.model_id,
        "slice_id": scope.slice_id,
        "cell_ids_json": _json_text(list(scope.cell_ids)),
        "n_cells": len(scope.cell_ids),
        "aggregation": scope.aggregation,
        "metric_id": scope.metric_id,
        "higher_is_better": METRIC_DIRECTIONS[scope.metric_id] == 1,
    }


def _external_cell_draws(
    *, labels: np.ndarray, scores_by_method: Mapping[str, np.ndarray],
    group_ids: Sequence[str], draws: int, seed: int,
    stratify_by_label: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Exact draw-producing counterpart of ``grouped_paired_bootstrap``.

    The implementation intentionally uses the frozen evaluator's private
    metric primitives and preserves its RNG nesting.  The source evaluator is
    not modified; the derived manifest binds its file hash.
    """

    y, groups, methods, scores = _external._validate_scores(  # type: ignore[attr-defined]
        labels=labels, scores_by_method=scores_by_method,
        group_ids=group_ids, reference_method=methods_reference(scores_by_method),
    )
    roster, members = _external._group_members(  # type: ignore[attr-defined]
        groups, canonical_order=False,
    )
    group_labels = (
        _external._pure_group_labels(y, roster, members)  # type: ignore[attr-defined]
        if stratify_by_label else None
    )
    row_group_positions = _external._row_group_positions(  # type: ignore[attr-defined]
        n_rows=len(y), roster=roster, members=members,
    )
    group_positive = np.bincount(
        row_group_positions, weights=y, minlength=len(roster),
    ).astype(np.int64, copy=False)
    group_rows = np.bincount(
        row_group_positions, minlength=len(roster),
    ).astype(np.int64, copy=False)
    group_negative = group_rows - group_positive
    plans = {
        method_id: _external._metric_plan(  # type: ignore[attr-defined]
            labels=y, score=scores[method_id],
            row_group_positions=row_group_positions,
        )
        for method_id in methods
    }
    rng = np.random.default_rng(int(seed))
    chunks = {
        method_id: {metric: [] for metric in _external.METRIC_IDS}
        for method_id in methods
    }
    generated = 0
    valid_total = 0
    while generated < int(draws):
        batch = _external._bootstrap_batch_size(  # type: ignore[attr-defined]
            n_rows=len(y), n_groups=len(roster),
            remaining=int(draws) - generated,
        )
        counts = np.empty((batch, len(roster)), dtype=np.int64)
        for draw_index in range(batch):
            counts[draw_index] = _external._sample_group_counts(  # type: ignore[attr-defined]
                roster=roster, rng=rng, group_labels=group_labels,
            )
        generated += batch
        valid = ((counts @ group_positive > 0) & (counts @ group_negative > 0))
        if not np.any(valid):
            continue
        valid_counts = counts[valid]
        valid_total += int(np.sum(valid))
        for method_id in methods:
            observed = _external._weighted_binary_metric_batch(  # type: ignore[attr-defined]
                group_counts=valid_counts, plan=plans[method_id],
            )
            for metric in _external.METRIC_IDS:
                chunks[method_id][metric].append(observed[metric])
    _require(valid_total > 0, "external cell bootstrap has no valid draws")
    values = {
        method_id: {
            metric: np.concatenate(chunks[method_id][metric])
            for metric in _external.METRIC_IDS
        }
        for method_id in methods
    }
    points_by_method = {
        method_id: _external.binary_metric_values(y, scores[method_id])
        for method_id in methods
    }
    return points_by_method, values, {
        "valid_draws": valid_total,
        "n_rows": len(y),
        "n_groups": len(roster),
    }


def methods_reference(scores_by_method: Mapping[str, object]) -> str:
    methods = _sorted_strings(scores_by_method)
    _require(bool(methods), "score mapping is empty")
    return "iu_pcr" if "iu_pcr" in methods else methods[0]


def _external_population_draws(
    *, cells: Mapping[str, Mapping[str, object]],
    link_keys: Mapping[str, str] | None, draws: int, seed: int,
    weighting: str, stratify_by_label: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Exact draw-producing counterpart of the frozen population evaluator."""

    _require(bool(cells), "population bootstrap requires cells")
    cell_ids = _sorted_strings(cells)
    if weighting == "single_cell":
        _require(len(cell_ids) == 1, "single_cell weighting requires one cell")
    else:
        _require(weighting == "equal_cell", "unsupported population weighting")
    if link_keys is None:
        effective_links = {
            cell_id: f"__independent__:{cell_id}" for cell_id in cell_ids
        }
    else:
        _require(set(link_keys) == set(cell_ids), "population link-key roster mismatch")
        effective_links = {cell_id: str(link_keys[cell_id]) for cell_id in cell_ids}

    state: dict[str, dict[str, Any]] = {}
    method_ids: tuple[str, ...] | None = None
    for cell_id in cell_ids:
        cell = cells[cell_id]
        score_mapping = cell.get("scores_by_method")
        _require(isinstance(score_mapping, Mapping), f"{cell_id}: score mapping missing")
        y, groups, methods, scores = _external._validate_scores(  # type: ignore[attr-defined]
            labels=cell["labels"], scores_by_method=score_mapping,
            group_ids=cell["group_ids"],
            reference_method=methods_reference(score_mapping),
        )
        if method_ids is None:
            method_ids = methods
        _require(methods == method_ids, f"{cell_id}: method roster drift")
        roster, members = _external._group_members(  # type: ignore[attr-defined]
            groups, canonical_order=True,
        )
        group_labels = (
            _external._pure_group_labels(y, roster, members)  # type: ignore[attr-defined]
            if stratify_by_label else None
        )
        positions = _external._row_group_positions(  # type: ignore[attr-defined]
            n_rows=len(y), roster=roster, members=members,
        )
        group_positive = np.bincount(
            positions, weights=y, minlength=len(roster),
        ).astype(np.int64, copy=False)
        group_rows = np.bincount(
            positions, minlength=len(roster),
        ).astype(np.int64, copy=False)
        state[cell_id] = {
            "labels": y, "scores": scores, "roster": roster,
            "members": members, "group_labels": group_labels,
            "group_positive": group_positive,
            "group_negative": group_rows - group_positive,
            "plans": {
                method_id: _external._metric_plan(  # type: ignore[attr-defined]
                    labels=y, score=scores[method_id],
                    row_group_positions=positions,
                )
                for method_id in methods
            },
        }
    _require(method_ids is not None, "population method roster missing")

    block_cells: dict[str, list[str]] = {}
    for cell_id in cell_ids:
        block_cells.setdefault(effective_links[cell_id], []).append(cell_id)
    block_state: dict[str, dict[str, Any]] = {}
    link_audits: list[dict[str, Any]] = []
    for link_key in _sorted_strings(block_cells):
        linked = _sorted_strings(block_cells[link_key])
        first = linked[0]
        roster = state[first]["roster"]
        members = state[first]["members"]
        counts = {group: int(len(members[group])) for group in roster}
        for cell_id in linked[1:]:
            _require(state[cell_id]["roster"] == roster,
                     f"{link_key}: linked group roster mismatch")
            other = state[cell_id]["members"]
            _require(
                {group: int(len(other[group])) for group in roster} == counts,
                f"{link_key}: linked member-count mismatch",
            )
            if stratify_by_label:
                _require(state[cell_id]["group_labels"] == state[first]["group_labels"],
                         f"{link_key}: linked group-label mismatch")
        block_state[link_key] = {
            "roster": roster,
            "group_labels": state[first]["group_labels"],
        }
        link_audits.append({
            "link_key": link_key,
            "cell_ids": list(linked),
            "linked": len(linked) > 1,
            "n_groups": len(roster),
        })

    cell_points = {
        cell_id: {
            method_id: _external.binary_metric_values(
                state[cell_id]["labels"], state[cell_id]["scores"][method_id]
            )
            for method_id in method_ids
        }
        for cell_id in cell_ids
    }
    points = {
        method_id: {
            metric: float(np.mean([
                cell_points[cell_id][method_id][metric] for cell_id in cell_ids
            ]))
            for metric in _external.METRIC_IDS
        }
        for method_id in method_ids
    }

    chunks = {
        method_id: {metric: [] for metric in _external.METRIC_IDS}
        for method_id in method_ids
    }
    rng = np.random.default_rng(int(seed))
    generated = 0
    valid_total = 0
    block_keys = _sorted_strings(block_state)
    max_rows = max(len(state[cell_id]["labels"]) for cell_id in cell_ids)
    total_groups = sum(len(block_state[key]["roster"]) for key in block_keys)
    while generated < int(draws):
        batch = _external._bootstrap_batch_size(  # type: ignore[attr-defined]
            n_rows=max_rows, n_groups=total_groups,
            remaining=int(draws) - generated,
        )
        counts_by_block = {
            key: np.empty((batch, len(block_state[key]["roster"])), dtype=np.int64)
            for key in block_keys
        }
        for draw_index in range(batch):
            for key in block_keys:
                counts_by_block[key][draw_index] = _external._sample_group_counts(  # type: ignore[attr-defined]
                    roster=block_state[key]["roster"], rng=rng,
                    group_labels=block_state[key]["group_labels"],
                )
        generated += batch
        valid = np.ones(batch, dtype=bool)
        for cell_id in cell_ids:
            counts = counts_by_block[effective_links[cell_id]]
            valid &= (
                (counts @ state[cell_id]["group_positive"] > 0)
                & (counts @ state[cell_id]["group_negative"] > 0)
            )
        if not np.any(valid):
            continue
        valid_total += int(np.sum(valid))
        per_method = {
            method_id: {metric: [] for metric in _external.METRIC_IDS}
            for method_id in method_ids
        }
        for cell_id in cell_ids:
            valid_counts = counts_by_block[effective_links[cell_id]][valid]
            for method_id in method_ids:
                observed = _external._weighted_binary_metric_batch(  # type: ignore[attr-defined]
                    group_counts=valid_counts,
                    plan=state[cell_id]["plans"][method_id],
                )
                for metric in _external.METRIC_IDS:
                    per_method[method_id][metric].append(observed[metric])
        for method_id in method_ids:
            for metric in _external.METRIC_IDS:
                chunks[method_id][metric].append(np.mean(
                    np.stack(per_method[method_id][metric], axis=0), axis=0,
                ))
    _require(valid_total > 0, "external population bootstrap has no valid draws")
    values = {
        method_id: {
            metric: np.concatenate(chunks[method_id][metric])
            for metric in _external.METRIC_IDS
        }
        for method_id in method_ids
    }
    return points, values, {
        "valid_draws": valid_total,
        "cell_ids": list(cell_ids),
        "link_blocks": link_audits,
        "linked_resampling": any(row["linked"] for row in link_audits),
    }


def _bound_file(root: Path, relative: object, *, name: str) -> Path:
    text = str(relative)
    _require(bool(text), f"{name}: empty relative path")
    candidate = Path(text)
    _require(not candidate.is_absolute(), f"{name}: absolute path is forbidden")
    resolved_root = root.resolve()
    cursor = root
    for part in candidate.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{name}: symlink path component is forbidden")
    resolved = (root / candidate).resolve()
    _require(resolved == resolved_root or resolved_root in resolved.parents,
             f"{name}: path escapes certified root")
    _require_plain_file(resolved, name=name)
    return resolved


def _verify_bound_hash(path: Path, expected: object, *, name: str) -> None:
    _require(isinstance(expected, str) and len(expected) == 64,
             f"{name}: missing SHA-256 binding")
    _require(sha256_file(path) == expected, f"{name}: file SHA-256 mismatch")


def _metric_atol(metric: str) -> float:
    return 4e-10 if metric == "aurc_x1000" else 4e-13


def _check_number(observed: object, expected: float, *, metric: str,
                  name: str) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise WinnerContrastError(f"{name}: nonnumeric source value") from exc
    _require(_float_equal(value, expected, atol=_metric_atol(metric)),
             f"{name}: recomputation mismatch ({value!r} != {expected!r})")


def load_frozen24_source(evaluation_dir: str | Path) -> LoadedSource:
    """Load and independently validate the frozen 24-cell shared draws."""

    requested_root = Path(os.path.abspath(os.fspath(evaluation_dir)))
    _require(requested_root.is_dir() and not requested_root.is_symlink(),
             "frozen24 evaluation directory is missing or unsafe")
    root = requested_root.resolve()
    manifest_path = root / "EVALUATION_MANIFEST.json"
    manifest = _load_json(manifest_path, name="frozen24 evaluation manifest")
    _validate_hashed_payload(manifest, name="frozen24 evaluation manifest")
    _require(manifest.get("schema_version") == "reconstruction-evaluation-manifest-v1",
             "frozen24 evaluation manifest schema drifted")
    _require(manifest.get("status") == "OK" and manifest.get("headline_status") == "OK",
             "frozen24 source is not a certified OK evaluation")
    _require(int(manifest.get("bootstrap_draws", -1)) == CANONICAL_DRAWS,
             "frozen24 source does not use 20,000 bootstrap draws")
    _require(int(manifest.get("canonical_bootstrap_draws", -1)) == CANONICAL_DRAWS,
             "frozen24 canonical draw contract drifted")

    provenance = manifest.get("input_provenance")
    _require(isinstance(provenance, Mapping),
             "frozen24 evaluation input provenance is missing")
    release_root = root.parent
    repository_root = Path(__file__).resolve().parents[2]
    score_certificate_path = release_root / "SCORE_AB_VERIFICATION.json"
    score_certificate = _load_json(
        score_certificate_path, name="frozen24 score A/B certificate",
    )
    _validate_hashed_payload(
        score_certificate, name="frozen24 score A/B certificate",
    )
    _require(
        score_certificate.get("schema_version")
        == "reconstruction-score-ab-verification-v1"
        and score_certificate.get("pass") is True
        and int(score_certificate.get("n_cells", -1)) == 24
        and int(score_certificate.get("n_methods", -1)) == 13
        and int(score_certificate.get("n_pairs", -1)) == 312,
        "frozen24 score A/B certificate contract drifted",
    )
    _require(
        provenance.get("score_ab_verification_sha256")
        == sha256_file(score_certificate_path),
        "frozen24 evaluation lost its score A/B certificate binding",
    )
    frozen_source_paths = {
        "evaluation_module_sha256": (
            repository_root / "spectral_utils/reconstruction_benchmark/evaluation.py"
        ),
        "cell_registry_sha256": (
            repository_root / "configs/reconstruction_benchmark_v1/frozen24_cells.json"
        ),
        "method_registry_sha256": (
            repository_root / "configs/reconstruction_benchmark_v1/methods.json"
        ),
        "group_manifest_sha256": (
            release_root / "group_sidecars/GROUP_SIDECARS.json"
        ),
        "label_bundle_sha256": (
            repository_root / "results/dependency_fusion_raw/cells.npz"
        ),
    }
    for field, path in frozen_source_paths.items():
        _require_plain_file(path, name=f"frozen24 {field} source")
        _require(provenance.get(field) == sha256_file(path),
                 f"frozen24 current-source binding failed: {field}")
    evaluator_cli_path = (
        repository_root / "scripts/reconstruction_benchmark/evaluate_24cell_release.py"
    )
    _require_plain_file(evaluator_cli_path, name="frozen24 evaluator CLI")
    _require(manifest.get("evaluator_cli_sha256") == sha256_file(evaluator_cli_path),
             "frozen24 evaluator CLI binding failed")
    for build_id in ("A", "B"):
        build_root = release_root / f"build_{build_id}"
        freeze_path = build_root / "fit/SCORE_FREEZE_MANIFEST.json"
        input_path = build_root / "inputs/MANIFEST.json"
        _require_plain_file(freeze_path, name=f"frozen24 freeze {build_id}")
        _require_plain_file(input_path, name=f"frozen24 input manifest {build_id}")
        _require(
            score_certificate.get(f"freeze_{build_id}_sha256")
            == provenance.get(f"freeze_{build_id}_sha256")
            == sha256_file(freeze_path),
            f"frozen24 freeze certificate binding failed: {build_id}",
        )
        _require(
            score_certificate.get(f"input_manifest_{build_id}_sha256")
            == provenance.get(f"input_manifest_{build_id}_sha256")
            == sha256_file(input_path),
            f"frozen24 input certificate binding failed: {build_id}",
        )

    evaluation_path = _bound_file(
        root, manifest.get("evaluation_path"), name="frozen24 evaluation payload",
    )
    bootstrap_path = _bound_file(
        root, manifest.get("bootstrap_path"), name="frozen24 bootstrap archive",
    )
    prediction_path = _bound_file(
        root, manifest.get("prediction_snapshot_path"),
        name="frozen24 prediction snapshot",
    )
    _verify_bound_hash(evaluation_path, manifest.get("evaluation_sha256"),
                       name="frozen24 evaluation payload")
    _verify_bound_hash(bootstrap_path, manifest.get("bootstrap_sha256"),
                       name="frozen24 bootstrap archive")
    _verify_bound_hash(prediction_path, manifest.get("prediction_snapshot_sha256"),
                       name="frozen24 prediction snapshot")

    evaluation = _load_json(evaluation_path, name="frozen24 evaluation payload")
    _validate_hashed_payload(evaluation, name="frozen24 evaluation payload")
    _require(evaluation.get("schema_version") == "reconstruction-24cell-evaluation-v1",
             "frozen24 evaluation schema drifted")
    _require(evaluation.get("status") == "OK" and evaluation.get("headline_status") == "OK",
             "frozen24 evaluation is not headline-certified")
    _require(evaluation.get("positive_class") == "incorrect",
             "frozen24 positive-class contract drifted")
    _require(evaluation.get("score_semantics") == "higher_is_incorrect",
             "frozen24 score semantics drifted")
    method_ids = tuple(map(str, evaluation.get("method_ids", ())))
    _require(len(method_ids) == 13 and len(set(method_ids)) == 13,
             "frozen24 exact-13 method roster failed")
    _require(evaluation.get("reference_method_id") == "iu_pcr",
             "frozen24 reference method drifted")
    _require(int(evaluation.get("n_cells", -1)) == 24,
             "frozen24 cell count drifted")
    _require(tuple(map(str, score_certificate.get("method_ids", ()))) == method_ids,
             "frozen24 score-certificate method roster drifted")
    cell_draw_manifests = evaluation.get("bootstrap", {}).get("cell_draw_manifests")
    _require(isinstance(cell_draw_manifests, list),
             "frozen24 cell draw manifest roster is missing")
    _require(
        tuple(map(str, score_certificate.get("cell_ids", ())))
        == tuple(str(row.get("cell_id", "")) for row in cell_draw_manifests),
        "frozen24 score-certificate cell roster drifted",
    )

    arrays = load_npz_no_pickle(bootstrap_path)
    cell_rows = evaluation.get("cell_metrics")
    aggregate_rows = evaluation.get("aggregate_metrics")
    contrast_rows = evaluation.get("paired_contrasts_vs_iu_pcr")
    _require(isinstance(cell_rows, list) and isinstance(aggregate_rows, list)
             and isinstance(contrast_rows, list),
             "frozen24 evaluation row tables are missing")
    expected_array_keys = {
        f"cell__{row['cell_id']}__{row['method_id']}__{row['metric']}"
        for row in cell_rows
    }
    _require(set(arrays) == expected_array_keys and len(arrays) == 24 * 13 * 2,
             "frozen24 bootstrap archive member roster drifted")
    for key, value in arrays.items():
        _require(value.dtype == np.dtype("float64") and value.shape == (CANONICAL_DRAWS,),
                 f"frozen24 bootstrap member shape/dtype drifted: {key}")

    metadata_by_cell: dict[str, dict[str, Any]] = {}
    for row in cell_rows:
        cell_id = str(row["cell_id"])
        metadata = {
            "dataset_id": str(row.get("dataset_id", "")),
            "model_id": str(row.get("model_id", "")),
            "domain": str(row.get("domain", "")),
        }
        previous = metadata_by_cell.setdefault(cell_id, metadata)
        _require(previous == metadata, f"frozen24 cell metadata drifted: {cell_id}")

    aggregate_index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in aggregate_rows:
        key = (str(row["scope_type"]), str(row["scope_value"]),
               str(row["metric"]), str(row["method_id"]))
        _require(key not in aggregate_index, f"duplicate frozen24 aggregate row: {key}")
        aggregate_index[key] = row
        grouped.setdefault(key[:3], []).append(row)
    _require(len(aggregate_index) == 42 * 13 * 2,
             "frozen24 aggregate scope roster drifted")

    contrast_index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in contrast_rows:
        key = (str(row["scope_type"]), str(row["scope_value"]),
               str(row["metric"]), str(row["candidate_method_id"]))
        _require(key not in contrast_index, f"duplicate frozen24 IU contrast: {key}")
        contrast_index[key] = row
    _require(len(contrast_index) == 42 * 12 * 2,
             "frozen24 IU-only contrast roster drifted")

    scopes: list[EvaluatedScopeMetric] = []
    for (scope_type, scope_value, metric), rows in sorted(
        grouped.items(), key=lambda item: tuple(_utf8_key(x) for x in item[0]),
    ):
        _require(metric in {"auroc", "auprc"}, f"unknown frozen24 metric: {metric}")
        by_method = {str(row["method_id"]): row for row in rows}
        _require(set(by_method) == set(method_ids) and len(rows) == len(method_ids),
                 f"frozen24 scope is not method-complete: {(scope_type, scope_value, metric)}")
        first = rows[0]
        cell_ids = tuple(map(str, first.get("cell_ids", ())))
        _require(bool(cell_ids) and len(cell_ids) == int(first.get("n_cells", -1)),
                 "frozen24 aggregate cell roster is invalid")
        for row in rows:
            _require(tuple(map(str, row.get("cell_ids", ()))) == cell_ids,
                     "frozen24 aggregate cell roster differs by method")
            _require(int(row.get("bootstrap_draws_requested", -1)) == CANONICAL_DRAWS,
                     "frozen24 aggregate draw count drifted")
            _require(str(row.get("status")) in SUCCESS_STATUSES,
                     "frozen24 aggregate includes a failed method")

        points: dict[str, float] = {}
        scope_draws: dict[str, np.ndarray] = {}
        statuses: dict[str, str] = {}
        for method_id in method_ids:
            row = by_method[method_id]
            matrix = np.vstack([
                arrays[f"cell__{cell_id}__{method_id}__{metric}"]
                for cell_id in cell_ids
            ])
            valid = np.all(np.isfinite(matrix), axis=0)
            values = np.full(CANONICAL_DRAWS, np.nan, dtype=np.float64)
            values[valid] = np.mean(matrix[:, valid], axis=0)
            point = float(row["estimate"])
            points[method_id] = point
            scope_draws[method_id] = values
            statuses[method_id] = str(row["status"])
            finite = values[np.isfinite(values)]
            low, high = _quantiles(finite)
            _require(len(finite) == int(row["bootstrap_draws_valid"]),
                     "frozen24 aggregate valid-draw count mismatch")
            _check_number(row["ci_lower"], low, metric=metric,
                          name="frozen24 aggregate lower CI")
            _check_number(row["ci_upper"], high, metric=metric,
                          name="frozen24 aggregate upper CI")
            _check_number(row["bootstrap_median"], float(np.quantile(finite, 0.5)),
                          metric=metric, name="frozen24 aggregate median")

        reference = "iu_pcr"
        for candidate in method_ids:
            if candidate == reference:
                continue
            source = contrast_index[(scope_type, scope_value, metric, candidate)]
            candidate_matrix = np.vstack([
                arrays[f"cell__{cell_id}__{candidate}__{metric}"] for cell_id in cell_ids
            ])
            reference_matrix = np.vstack([
                arrays[f"cell__{cell_id}__{reference}__{metric}"] for cell_id in cell_ids
            ])
            valid = np.all(np.isfinite(candidate_matrix) & np.isfinite(reference_matrix), axis=0)
            delta_draws = np.mean(
                candidate_matrix[:, valid] - reference_matrix[:, valid], axis=0,
            )
            low, high = _quantiles(delta_draws)
            _require(len(delta_draws) == int(source["bootstrap_draws_valid"]),
                     "frozen24 IU contrast valid-draw mismatch")
            _check_number(source["delta"], points[candidate] - points[reference],
                          metric=metric, name="frozen24 IU point contrast")
            _check_number(source["ci_lower"], low, metric=metric,
                          name="frozen24 IU lower contrast CI")
            _check_number(source["ci_upper"], high, metric=metric,
                          name="frozen24 IU upper contrast CI")
            _check_number(
                source["bootstrap_probability_delta_positive"],
                float(np.mean(delta_draws > 0.0)), metric=metric,
                name="frozen24 IU contrast probability",
            )

        if scope_type == "cell":
            meta = metadata_by_cell[scope_value]
            record_level = "cell"
            cell_id = scope_value
            dataset_id = meta["dataset_id"]
            model_id = meta["model_id"]
        else:
            record_level = "aggregate"
            cell_id = "__aggregate__"
            dataset_id = "__multiple__"
            model_id = "__multiple__"
        scopes.append(EvaluatedScopeMetric(
            lane_id="frozen24_response", population_id=str(evaluation["population_id"]),
            scope_type=scope_type, scope_value=scope_value,
            record_level=record_level, cell_id=cell_id,
            dataset_id=dataset_id, model_id=model_id, slice_id=scope_value,
            cell_ids=cell_ids, aggregation=str(first["aggregation"]),
            source_comparison_group_id=(
                f"frozen24_response::{scope_type}::{scope_value}::{metric}"
            ),
            bootstrap_unit="verified_source_group_within_cell",
            bootstrap_seed=None, linked_resampling=False,
            stratified_by_label=False, bootstrap_draws_requested=CANONICAL_DRAWS,
            points=points, draws=scope_draws, method_statuses=statuses,
            metric=metric,
        ))

    return LoadedSource(
        source_type="frozen24_shared_draw_archive",
        lane_id="frozen24_response",
        method_ids=method_ids,
        metric_ids=("auroc", "auprc"),
        scopes=tuple(scopes),
        source_binding={
            "population_id": evaluation["population_id"],
            "evaluation_manifest_sha256": sha256_file(manifest_path),
            "evaluation_manifest_payload_sha256": manifest["payload_sha256"],
            "evaluation_payload_sha256": manifest["evaluation_sha256"],
            "bootstrap_archive_sha256": manifest["bootstrap_sha256"],
            "prediction_snapshot_sha256": manifest["prediction_snapshot_sha256"],
            "score_ab_certificate_sha256": sha256_file(score_certificate_path),
            "score_ab_certificate_payload_sha256": score_certificate["payload_sha256"],
            "evaluation_module_sha256": provenance["evaluation_module_sha256"],
            "evaluator_cli_sha256": manifest["evaluator_cli_sha256"],
            "bootstrap_draws": CANONICAL_DRAWS,
        },
    )


def _npz_text(array: np.ndarray, *, name: str) -> str:
    values = np.asarray(array)
    _require(values.shape == (1,) and values.dtype.kind in {"U", "S"},
             f"{name}: expected one text scalar")
    return str(values[0])


def _external_scope_rows(
    metric_rows: Sequence[Mapping[str, str]],
    contrast_rows: Sequence[Mapping[str, str]],
) -> tuple[
    dict[tuple[str, str, str, str], Mapping[str, str]],
    dict[tuple[str, str, str, str], Mapping[str, str]],
]:
    metrics: dict[tuple[str, str, str, str], Mapping[str, str]] = {}
    contrasts: dict[tuple[str, str, str, str], Mapping[str, str]] = {}
    for row in metric_rows:
        level = str(row["record_level"])
        scope_id = str(row["cell_id"] if level == "cell" else row["population_id"])
        key = (level, scope_id, str(row["metric_id"]), str(row["method_id"]))
        _require(key not in metrics, f"duplicate external metric row: {key}")
        metrics[key] = row
    for row in contrast_rows:
        level = str(row["record_level"])
        scope_id = str(row["cell_id"] if level == "cell" else row["population_id"])
        key = (level, scope_id, str(row["metric_id"]), str(row["method_id"]))
        _require(key not in contrasts, f"duplicate external contrast row: {key}")
        contrasts[key] = row
    return metrics, contrasts


def _validate_external_recomputation(
    *, level: str, scope_id: str, method_ids: Sequence[str],
    points: Mapping[str, Mapping[str, float]],
    draws: Mapping[str, Mapping[str, np.ndarray]],
    metric_index: Mapping[tuple[str, str, str, str], Mapping[str, str]],
    contrast_index: Mapping[tuple[str, str, str, str], Mapping[str, str]],
) -> None:
    for method_id in method_ids:
        for metric in METRIC_DIRECTIONS:
            key = (level, scope_id, metric, method_id)
            _require(key in metric_index, f"missing external metric row: {key}")
            row = metric_index[key]
            values = np.asarray(draws[method_id][metric], dtype=float)
            _require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
                     f"external recomputed draws invalid: {key}")
            low, high = _quantiles(values)
            _check_number(row["value"], points[method_id][metric], metric=metric,
                          name=f"external point metric {key}")
            _check_number(row["ci_low"], low, metric=metric,
                          name=f"external lower metric CI {key}")
            _check_number(row["ci_high"], high, metric=metric,
                          name=f"external upper metric CI {key}")
            _require(int(row["bootstrap_draws"]) == CANONICAL_DRAWS,
                     f"external requested draw count drifted: {key}")
            _require(int(row["bootstrap_valid_draws"]) == len(values),
                     f"external valid draw count drifted: {key}")
            _require(str(row["status"]) in SUCCESS_STATUSES,
                     f"external source method failed: {key}")
            if method_id == "iu_pcr":
                continue
            contrast_key = (level, scope_id, metric, method_id)
            _require(contrast_key in contrast_index,
                     f"missing external IU contrast: {contrast_key}")
            contrast = contrast_index[contrast_key]
            _require(contrast["reference_method_id"] == "iu_pcr",
                     f"external reference drifted: {contrast_key}")
            delta_draws = values - np.asarray(draws["iu_pcr"][metric], dtype=float)
            delta_low, delta_high = _quantiles(delta_draws)
            _check_number(
                contrast["delta"],
                points[method_id][metric] - points["iu_pcr"][metric],
                metric=metric, name=f"external IU point contrast {contrast_key}",
            )
            _check_number(contrast["ci_low"], delta_low, metric=metric,
                          name=f"external IU lower contrast CI {contrast_key}")
            _check_number(contrast["ci_high"], delta_high, metric=metric,
                          name=f"external IU upper contrast CI {contrast_key}")
            probability = _external._probability_delta_le_zero(  # type: ignore[attr-defined]
                delta_draws, metric=metric,
            )
            _check_number(
                contrast["probability_delta_le_zero"], probability, metric=metric,
                name=f"external IU contrast probability {contrast_key}",
            )
            _require(int(contrast["bootstrap_valid_draws"]) == len(delta_draws),
                     f"external IU contrast valid draw count drifted: {contrast_key}")


def load_external_source(
    evaluation_dir: str | Path,
    evaluation_ab_certificate: str | Path,
) -> LoadedSource:
    """Recompute exact shared draws from A/B-certified scores and labels."""

    requested_root = Path(os.path.abspath(os.fspath(evaluation_dir)))
    _require(requested_root.is_dir() and not requested_root.is_symlink(),
             "external evaluation directory is missing or unsafe")
    root = requested_root.resolve()
    manifest_path = root / "MANIFEST.json"
    manifest = _load_json(manifest_path, name="external evaluation manifest")
    _validate_hashed_payload(manifest, name="external evaluation manifest")
    _require(manifest.get("schema_version") == "reconstruction-external-evaluation-v2",
             "external evaluation schema drifted")
    _require(manifest.get("scientific_full") is True
             and manifest.get("ab_verification_status") == "PASS",
             "external evaluation is not a scientific-full A/B PASS")
    _require(manifest.get("score_semantics") == "higher_is_incorrect"
             and manifest.get("positive_class") == "incorrect",
             "external evaluation label/score semantics drifted")
    _require(int(manifest.get("bootstrap_draws", -1)) == CANONICAL_DRAWS,
             "external evaluation does not use 20,000 draws")
    build_id = str(manifest.get("build_id", ""))
    _require(build_id in {"A", "B"}, "external build ID is not A or B")

    certificate_path = Path(evaluation_ab_certificate).resolve()
    certificate = _load_json(certificate_path, name="external evaluation A/B certificate")
    _validate_hashed_payload(
        certificate, name="external evaluation A/B certificate",
        field="certificate_sha256",
    )
    _require(
        certificate.get("schema_version")
        == "reconstruction-external-evaluation-ab-certificate-v1"
        and certificate.get("status") == "PASS"
        and certificate.get("scientific_full") is True,
        "external evaluation A/B certificate is not a scientific-full PASS",
    )
    _require(certificate.get("release_id") == manifest.get("release_id"),
             "external release binding differs from A/B certificate")
    _require(certificate.get("external_registry_sha256")
             == manifest.get("external_registry_sha256"),
             "external registry binding differs from A/B certificate")
    certified_builds = certificate.get("builds")
    _require(isinstance(certified_builds, Mapping) and build_id in certified_builds,
             "external A/B certificate lacks selected build")
    build_binding = certified_builds[build_id]
    _require(isinstance(build_binding, Mapping), "external build binding is invalid")
    _require(build_binding.get("evaluation_manifest_file_sha256")
             == sha256_file(manifest_path),
             "external evaluation manifest is not the A/B-certified file")
    _require(build_binding.get("evaluation_manifest_payload_sha256")
             == manifest.get("payload_sha256"),
             "external evaluation manifest payload is not A/B-certified")

    method_ids = tuple(map(str, certificate.get("method_ids", ())))
    metric_ids = tuple(map(str, certificate.get("metric_ids", ())))
    _require(len(method_ids) == 13 and len(set(method_ids)) == 13
             and "iu_pcr" in method_ids,
             "external A/B certificate exact-13 roster failed")
    _require(metric_ids == ("auroc", "auprc", "aurc_x1000"),
             "external metric roster drifted")
    _require(int(certificate.get("bootstrap_draws", -1)) == CANONICAL_DRAWS,
             "external A/B certificate draw count drifted")

    source_snapshot = manifest.get("evaluation_source_snapshot")
    _require(isinstance(source_snapshot, Mapping),
             "external evaluation source snapshot is missing")
    source_files = source_snapshot.get("files")
    _require(isinstance(source_files, list),
             "external evaluation source file roster is missing")
    source_hash_by_path = {
        str(row["path"]): str(row["sha256"])
        for row in source_files if isinstance(row, Mapping)
    }
    evaluator_relative = (
        "spectral_utils/reconstruction_benchmark/external_evaluation.py"
    )
    registry_relative = (
        "configs/reconstruction_benchmark_v1/external_final_answer.json"
    )
    evaluator_hash = sha256_file(Path(_external.__file__))
    _require(source_hash_by_path.get(evaluator_relative) == evaluator_hash,
             "current external evaluator differs from frozen source snapshot")
    repo_root = Path(_external.__file__).resolve().parents[2]
    registry_path = repo_root / registry_relative
    _require_plain_file(registry_path, name="frozen external registry")
    _require(source_hash_by_path.get(registry_relative) == sha256_file(registry_path)
             == manifest.get("external_registry_sha256"),
             "current external registry differs from frozen source snapshot")
    registry_payload = _load_json(registry_path, name="frozen external registry")
    aggregate_rules = registry_payload.get("population_aggregates")
    _require(isinstance(aggregate_rules, Mapping),
             "frozen external registry aggregate rules are missing")

    metrics_path = _bound_file(root, manifest.get("metrics_path"),
                               name="external metrics table")
    contrasts_path = _bound_file(root, manifest.get("contrasts_path"),
                                 name="external contrasts table")
    _verify_bound_hash(metrics_path, manifest.get("metrics_sha256"),
                       name="external metrics table")
    _verify_bound_hash(contrasts_path, manifest.get("contrasts_sha256"),
                       name="external contrasts table")
    byte_identity = certificate.get("byte_identity")
    _require(isinstance(byte_identity, Mapping)
             and byte_identity.get("metrics_long.csv") == manifest.get("metrics_sha256")
             and byte_identity.get("contrasts_long.csv") == manifest.get("contrasts_sha256"),
             "external evaluation tables are not the A/B byte-identical tables")
    metric_rows = _read_csv(metrics_path, name="external metrics table")
    contrast_rows = _read_csv(contrasts_path, name="external contrasts table")
    _require(len(metric_rows) == int(manifest.get("n_metric_rows", -1))
             == int(certificate.get("n_metric_rows", -2)),
             "external metric row count binding failed")
    _require(len(contrast_rows) == int(manifest.get("n_contrast_rows", -1))
             == int(certificate.get("n_contrast_rows", -2)),
             "external contrast row count binding failed")
    metric_index, contrast_index = _external_scope_rows(metric_rows, contrast_rows)

    fit_root = root.parent / "fit"
    _require(fit_root.is_dir() and not fit_root.is_symlink(),
             "external sibling fit directory is missing or unsafe")
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = _load_json(freeze_path, name="external score-freeze manifest")
    _validate_hashed_payload(freeze, name="external score-freeze manifest")
    _require(freeze.get("schema_version") == "reconstruction-external-score-freeze-v2"
             and freeze.get("scientific_full") is True
             and freeze.get("all_expected_scores_present") is True,
             "external score freeze is not complete scientific-full")
    _require(freeze.get("build_id") == build_id
             and freeze.get("release_id") == manifest.get("release_id"),
             "external score-freeze release/build binding drifted")
    _require(tuple(map(str, freeze.get("method_ids", ()))) == method_ids,
             "external score-freeze method roster drifted")
    _require(manifest.get("score_freeze_sha256") == sha256_file(freeze_path)
             == build_binding.get("score_freeze_sha256"),
             "external score-freeze file hash binding failed")
    _require(manifest.get("score_freeze_payload_sha256") == freeze.get("payload_sha256")
             == build_binding.get("score_freeze_payload_sha256"),
             "external score-freeze payload binding failed")

    records = freeze.get("records")
    _require(isinstance(records, list), "external score records are missing")
    record_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    score_roster: list[dict[str, str]] = []
    for record in records:
        _require(isinstance(record, Mapping), "external score-freeze record is invalid")
        key = (str(record.get("cell_id", "")), str(record.get("method_id", "")))
        _require(key not in record_index, f"duplicate external score record: {key}")
        _require(str(record.get("status")) in SUCCESS_STATUSES,
                 f"external score record failed: {key}")
        record_path = _bound_file(fit_root, record.get("record_path"),
                                  name=f"external score record {key}")
        score_path = _bound_file(fit_root, record.get("score_path"),
                                 name=f"external score artifact {key}")
        _verify_bound_hash(record_path, record.get("record_sha256"),
                           name=f"external score record {key}")
        _verify_bound_hash(score_path, record.get("score_sha256"),
                           name=f"external score artifact {key}")
        detail = _load_json(record_path, name=f"external score record {key}")
        _require(detail.get("cell_id") == key[0] and detail.get("method_id") == key[1]
                 and detail.get("score_sha256") == record.get("score_sha256")
                 and detail.get("status") == record.get("status"),
                 f"external score record content binding failed: {key}")
        record_index[key] = record
        score_roster.append({
            "cell_id": key[0], "method_id": key[1],
            "score_sha256": str(record["score_sha256"]),
            "status": str(record["status"]),
        })

    label_records = manifest.get("label_records")
    _require(isinstance(label_records, list), "external label records are missing")
    certified_label_hashes = byte_identity.get("label_artifacts")
    _require(isinstance(certified_label_hashes, Mapping),
             "external A/B certificate lacks label byte identities")
    labels_by_cell: dict[str, dict[str, Any]] = {}
    label_roster: list[dict[str, str]] = []
    for record in label_records:
        _require(isinstance(record, Mapping), "external label record is invalid")
        cell_id = str(record.get("cell_id", ""))
        _require(cell_id and cell_id not in labels_by_cell,
                 f"duplicate external label record: {cell_id}")
        relative = str(record.get("artifact_path", ""))
        label_path = _bound_file(root, relative, name=f"external labels {cell_id}")
        _verify_bound_hash(label_path, record.get("artifact_sha256"),
                           name=f"external labels {cell_id}")
        _require(certified_label_hashes.get(relative) == record.get("artifact_sha256"),
                 f"external label artifact is not A/B byte-identical: {cell_id}")
        bundle = load_npz_no_pickle(label_path)
        _require(set(bundle) == {
            "row_ids", "group_ids", "incorrect", "id_contract_version",
            "id_contract_sha256", "identity_key_id", "row_namespace_sha256",
            "group_namespace_sha256",
        }, f"external label member roster drifted: {cell_id}")
        row_ids = np.asarray(bundle["row_ids"])
        group_ids = np.asarray(bundle["group_ids"])
        incorrect = np.asarray(bundle["incorrect"])
        _require(row_ids.ndim == group_ids.ndim == incorrect.ndim == 1
                 and len(row_ids) == len(group_ids) == len(incorrect)
                 == int(record.get("n_rows", -1)),
                 f"external label array shape drifted: {cell_id}")
        _require(incorrect.dtype == np.dtype("int8")
                 and set(map(int, np.unique(incorrect))) <= {0, 1}
                 and len(np.unique(incorrect)) == 2,
                 f"external labels are not two-class int8: {cell_id}")
        _require(_npz_text(bundle["row_namespace_sha256"], name=cell_id)
                 == record.get("row_namespace_sha256"),
                 f"external label row namespace drifted: {cell_id}")
        labels_by_cell[cell_id] = {
            "row_ids": tuple(map(str, row_ids)),
            "group_ids": tuple(map(str, group_ids)),
            "incorrect": incorrect.astype(np.int8, copy=False),
            "artifact_sha256": str(record["artifact_sha256"]),
        }
        label_roster.append({
            "cell_id": cell_id,
            "artifact_sha256": str(record["artifact_sha256"]),
        })

    certified_cells = tuple(map(str, certificate.get("cell_ids", ())))
    _require(set(labels_by_cell) == set(certified_cells),
             "external label cell roster differs from A/B certificate")
    _require(set(record_index) == {
        (cell_id, method_id) for cell_id in certified_cells for method_id in method_ids
    }, "external score record roster is not exact cell-by-method")

    evaluation_cells: dict[str, dict[str, Any]] = {}
    for cell_id in certified_cells:
        label = labels_by_cell[cell_id]
        scores_by_method: dict[str, np.ndarray] = {}
        statuses: dict[str, str] = {}
        for method_id in method_ids:
            record = record_index[(cell_id, method_id)]
            bundle = load_npz_no_pickle(_bound_file(
                fit_root, record["score_path"],
                name=f"external score artifact {(cell_id, method_id)}",
            ))
            _require(set(bundle) == {
                "row_ids", "score", "id_contract_version", "id_contract_sha256",
                "identity_key_id", "row_namespace_sha256", "row_roster_sha256",
            }, f"external score member roster drifted: {(cell_id, method_id)}")
            _require(tuple(map(str, bundle["row_ids"])) == label["row_ids"],
                     f"external signed score/label row join failed: {(cell_id, method_id)}")
            score = np.asarray(bundle["score"], dtype=float)
            _require(score.shape == label["incorrect"].shape and np.isfinite(score).all(),
                     f"external score vector is invalid: {(cell_id, method_id)}")
            scores_by_method[method_id] = score
            statuses[method_id] = str(record["status"])
        evaluation_cells[cell_id] = {
            "labels": label["incorrect"], "group_ids": label["group_ids"],
            "scores_by_method": scores_by_method, "method_statuses": statuses,
        }

    population_checks = certificate.get("population_checks")
    _require(population_checks == manifest.get("population_checks")
             and isinstance(population_checks, list),
             "external population checks differ from A/B certificate")
    check_by_population = {
        str(check["population_id"]): check for check in population_checks
    }
    scopes: list[EvaluatedScopeMetric] = []
    registry_hash = str(manifest["external_registry_sha256"])
    for cell_id in certified_cells:
        sample_rows = [
            metric_index[("cell", cell_id, metric_ids[0], method_id)]
            for method_id in method_ids
        ]
        first = sample_rows[0]
        _require(all(row["population_id"] == first["population_id"] for row in sample_rows),
                 f"external cell population metadata drifted: {cell_id}")
        population_id = str(first["population_id"])
        stratified = _parse_bool(first.get("stratified_by_label", False),
                                 name=f"external cell stratification {cell_id}")
        aggregate_rule = aggregate_rules.get(population_id)
        _require(isinstance(aggregate_rule, Mapping),
                 f"frozen aggregate rule missing: {population_id}")
        _require(stratified == (
            aggregate_rule.get("bootstrap") == "source_group_stratified_by_label"
        ), f"external cell stratification differs from frozen registry: {cell_id}")
        seed_payload = f"{registry_hash}:{cell_id}:grouped-paired-bootstrap-v1".encode("utf-8")
        seed = int(aggregate_rule.get(
            "seed", int(sha256_bytes(seed_payload)[:8], 16),
        ))
        points, draw_map, draw_meta = _external_cell_draws(
            labels=evaluation_cells[cell_id]["labels"],
            scores_by_method=evaluation_cells[cell_id]["scores_by_method"],
            group_ids=evaluation_cells[cell_id]["group_ids"],
            draws=CANONICAL_DRAWS, seed=seed,
            stratify_by_label=stratified,
        )
        _validate_external_recomputation(
            level="cell", scope_id=cell_id, method_ids=method_ids,
            points=points, draws=draw_map, metric_index=metric_index,
            contrast_index=contrast_index,
        )
        for metric in metric_ids:
            rows = [metric_index[("cell", cell_id, metric, method_id)]
                    for method_id in method_ids]
            group_ids = {row["comparison_group_id"] for row in rows}
            _require(len(group_ids) == 1, f"external cell comparison group drifted: {cell_id}")
            scopes.append(EvaluatedScopeMetric(
                lane_id="external_final_answer", population_id=population_id,
                scope_type="cell", scope_value=cell_id, record_level="cell",
                cell_id=cell_id, dataset_id=str(first["dataset_id"]),
                model_id=str(first["model_id"]), slice_id=str(first["slice_id"]),
                cell_ids=(cell_id,), aggregation="single_cell",
                source_comparison_group_id=next(iter(group_ids)),
                bootstrap_unit=str(first["bootstrap_unit"]), bootstrap_seed=seed,
                linked_resampling=False, stratified_by_label=stratified,
                bootstrap_draws_requested=CANONICAL_DRAWS,
                points={method_id: points[method_id][metric] for method_id in method_ids},
                draws={method_id: draw_map[method_id][metric] for method_id in method_ids},
                method_statuses=evaluation_cells[cell_id]["method_statuses"],
                metric=metric,
            ))
        _require(int(draw_meta["valid_draws"]) > 0,
                 f"external cell has no valid draws: {cell_id}")

    aggregated_population_ids: list[str] = []
    for population_id, check in sorted(check_by_population.items(), key=lambda x: _utf8_key(x[0])):
        if check.get("status") != "OK_AGGREGATED":
            continue
        aggregated_population_ids.append(population_id)
        blocks = check.get("link_blocks")
        _require(isinstance(blocks, list) and blocks,
                 f"external population link blocks missing: {population_id}")
        link_keys: dict[str, str] = {}
        cell_ids: list[str] = []
        for block in blocks:
            _require(isinstance(block, Mapping),
                     f"external population link block invalid: {population_id}")
            for cell_id in map(str, block.get("cell_ids", ())):
                _require(cell_id not in link_keys,
                         f"external population cell repeated in link blocks: {cell_id}")
                link_keys[cell_id] = str(block["link_key"])
                cell_ids.append(cell_id)
        canonical_cells = _sorted_strings(cell_ids)
        _require(set(canonical_cells) <= set(evaluation_cells),
                 f"external population uses an uncertified cell: {population_id}")
        first_source = metric_index[("population", population_id, metric_ids[0], method_ids[0])]
        stratified = _parse_bool(first_source.get("stratified_by_label", False),
                                 name=f"external population stratification {population_id}")
        weighting = str(check["weighting"])
        aggregate_rule = aggregate_rules.get(population_id)
        _require(isinstance(aggregate_rule, Mapping)
                 and aggregate_rule.get("enabled") is True
                 and str(aggregate_rule.get("weighting")) == weighting,
                 f"external population rule differs from frozen registry: {population_id}")
        population_seed_payload = (
            f"{registry_hash}:{population_id}:population-grouped-paired-bootstrap-v1"
        ).encode("utf-8")
        registered_seed = int(aggregate_rule.get(
            "seed", int(sha256_bytes(population_seed_payload)[:8], 16),
        ))
        _require(registered_seed == int(check["seed"]),
                 f"external population seed differs from frozen registry: {population_id}")
        points, draw_map, draw_meta = _external_population_draws(
            cells={cell_id: evaluation_cells[cell_id] for cell_id in canonical_cells},
            link_keys=link_keys, draws=CANONICAL_DRAWS,
            seed=int(check["seed"]), weighting=weighting,
            stratify_by_label=stratified,
        )
        _validate_external_recomputation(
            level="population", scope_id=population_id, method_ids=method_ids,
            points=points, draws=draw_map, metric_index=metric_index,
            contrast_index=contrast_index,
        )
        for metric in metric_ids:
            rows = [metric_index[("population", population_id, metric, method_id)]
                    for method_id in method_ids]
            comparison_groups = {row["comparison_group_id"] for row in rows}
            _require(len(comparison_groups) == 1,
                     f"external population comparison group drifted: {population_id}")
            statuses = {method_id: str(
                metric_index[("population", population_id, metric, method_id)]["status"]
            ) for method_id in method_ids}
            scopes.append(EvaluatedScopeMetric(
                lane_id="external_final_answer", population_id=population_id,
                scope_type="population", scope_value=population_id,
                record_level="population", cell_id=str(first_source["cell_id"]),
                dataset_id=str(first_source["dataset_id"]),
                model_id=str(first_source["model_id"]),
                slice_id=str(first_source["slice_id"]), cell_ids=canonical_cells,
                aggregation=weighting,
                source_comparison_group_id=next(iter(comparison_groups)),
                bootstrap_unit=str(first_source["bootstrap_unit"]),
                bootstrap_seed=int(check["seed"]),
                linked_resampling=bool(draw_meta["linked_resampling"]),
                stratified_by_label=stratified,
                bootstrap_draws_requested=CANONICAL_DRAWS,
                points={method_id: points[method_id][metric] for method_id in method_ids},
                draws={method_id: draw_map[method_id][metric] for method_id in method_ids},
                method_statuses=statuses, metric=metric,
            ))

    expected_scope_keys = {
        (str(row["record_level"]),
         str(row["cell_id"] if row["record_level"] == "cell" else row["population_id"]),
         str(row["metric_id"]))
        for row in metric_rows
    }
    observed_scope_keys = {(scope.record_level, scope.scope_value, scope.metric_id)
                           for scope in scopes}
    _require(observed_scope_keys == expected_scope_keys,
             "external derived scope roster differs from certified metrics table")
    _require(len(scopes) == (len(certified_cells) + len(aggregated_population_ids)) * 3,
             "external scope count failed")

    return LoadedSource(
        source_type="external_v3_signed_score_label_recomputation",
        lane_id="external_final_answer", method_ids=method_ids,
        metric_ids=metric_ids, scopes=tuple(scopes),
        source_binding={
            "release_id": manifest["release_id"], "build_id": build_id,
            "evaluation_manifest_sha256": sha256_file(manifest_path),
            "evaluation_manifest_payload_sha256": manifest["payload_sha256"],
            "evaluation_ab_certificate_sha256": certificate["certificate_sha256"],
            "evaluation_ab_certificate_file_sha256": sha256_file(certificate_path),
            "score_freeze_manifest_sha256": sha256_file(freeze_path),
            "score_freeze_manifest_payload_sha256": freeze["payload_sha256"],
            "metrics_sha256": manifest["metrics_sha256"],
            "contrasts_sha256": manifest["contrasts_sha256"],
            "score_artifact_roster_sha256": sha256_bytes(canonical_json_bytes(score_roster)),
            "label_artifact_roster_sha256": sha256_bytes(canonical_json_bytes(label_roster)),
            "external_registry_sha256": registry_hash,
            "external_evaluation_module_sha256": evaluator_hash,
            "bootstrap_draws": CANONICAL_DRAWS,
            "aggregate_scope_policy": {
                "included": aggregated_population_ids,
                "excluded_checks": [
                    {"population_id": str(check["population_id"]),
                     "status": str(check["status"])}
                    for check in population_checks
                    if check.get("status") != "OK_AGGREGATED"
                ],
                "cross_task_macro": "FORBIDDEN_AND_ABSENT",
            },
        },
    )


def _scope_sort_key(scope: EvaluatedScopeMetric) -> tuple[bytes, ...]:
    return tuple(_utf8_key(value) for value in (
        scope.lane_id, scope.population_id, scope.scope_type,
        scope.scope_value, scope.metric_id,
    ))


def _paired_summary(
    *, left: np.ndarray, right: np.ndarray, metric: str,
) -> dict[str, float | int]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    _require(a.shape == b.shape and a.ndim == 1,
             "paired bootstrap draw arrays diverged")
    valid = np.isfinite(a) & np.isfinite(b)
    _require(bool(np.any(valid)), "paired comparison has no valid draws")
    delta = a[valid] - b[valid]
    oriented = METRIC_DIRECTIONS[metric] * delta
    raw_low, raw_high = _quantiles(delta)
    oriented_low, oriented_high = _quantiles(oriented)
    tolerance = NUMERICAL_ZERO_ATOL[metric]
    return {
        "delta_ci_low": raw_low, "delta_ci_high": raw_high,
        "oriented_ci_low": oriented_low, "oriented_ci_high": oriented_high,
        "probability_oriented_advantage_le_zero": float(
            np.mean(oriented <= tolerance)
        ),
        "bootstrap_valid_draws": int(len(delta)),
    }


def derive_winner_contrasts(source: LoadedSource) -> dict[str, Any]:
    """Derive all pairs and direct point-winner reference sets in memory."""

    methods = _sorted_strings(source.method_ids)
    _require(len(methods) >= 2 and len(methods) == len(set(methods)),
             "derived source method roster is invalid")
    _require(set(source.metric_ids) <= set(METRIC_DIRECTIONS),
             "derived source metric roster is invalid")
    scopes = tuple(sorted(source.scopes, key=_scope_sort_key))
    scope_keys = [
        (scope.lane_id, scope.population_id, scope.scope_type,
         scope.scope_value, scope.metric_id)
        for scope in scopes
    ]
    _require(len(scope_keys) == len(set(scope_keys)),
             "derived source contains duplicate scopes")

    pair_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    audit_scopes: list[dict[str, Any]] = []
    for scope in scopes:
        metric = scope.metric_id
        _require(metric in source.metric_ids,
                 f"scope metric is outside source roster: {metric}")
        _require(scope.bootstrap_draws_requested == CANONICAL_DRAWS,
                 "scope does not use the canonical 20,000 requested draws")
        _require(set(scope.points) == set(methods)
                 and set(scope.draws) == set(methods)
                 and set(scope.method_statuses) == set(methods),
                 "scope is not method-complete")
        draw_arrays = {
            method_id: np.asarray(scope.draws[method_id], dtype=float)
            for method_id in methods
        }
        _require(all(values.ndim == 1 for values in draw_arrays.values()),
                 "scope shared draw arrays must be one-dimensional")
        lengths = {len(values) for values in draw_arrays.values()}
        _require(len(lengths) == 1, "scope shared draw arrays have unequal lengths")
        _require(next(iter(lengths)) == CANONICAL_DRAWS,
                 "scope shared draw arrays do not contain exactly 20,000 draws")
        shared_finite_mask = np.isfinite(draw_arrays[methods[0]])
        _require(bool(np.any(shared_finite_mask)),
                 "scope shared draw arrays have no jointly valid draw")
        _require(all(np.array_equal(np.isfinite(draw_arrays[method_id]), shared_finite_mask)
                     for method_id in methods[1:]),
                 "scope methods do not share the same accepted draw indexes")
        _require(all(str(scope.method_statuses[method_id]) in SUCCESS_STATUSES
                     for method_id in methods),
                 "scope contains a failed method")
        oriented_points = {
            method_id: METRIC_DIRECTIONS[metric] * float(scope.points[method_id])
            for method_id in methods
        }
        best = max(oriented_points.values())
        point_winners = tuple(
            method_id for method_id in methods
            if abs(oriented_points[method_id] - best) <= NUMERICAL_ZERO_ATOL[metric]
        )
        _require(bool(point_winners), "scope has no deterministic point winner")
        representative = point_winners[0]
        base = _base_row(scope)

        for left_index, method_a in enumerate(methods):
            for method_b in methods[left_index + 1:]:
                summary = _paired_summary(
                    left=scope.draws[method_a], right=scope.draws[method_b],
                    metric=metric,
                )
                raw_point = float(scope.points[method_a]) - float(scope.points[method_b])
                oriented_point = METRIC_DIRECTIONS[metric] * raw_point
                tolerance = NUMERICAL_ZERO_ATOL[metric]
                if float(summary["oriented_ci_low"]) > tolerance:
                    relation = "A_BETTER"
                elif float(summary["oriented_ci_high"]) < -tolerance:
                    relation = "B_BETTER"
                else:
                    relation = "NOT_SEPARATED_95CI"
                status = (
                    "OK_FALLBACK" if "OK_FALLBACK" in {
                        scope.method_statuses[method_a], scope.method_statuses[method_b]
                    } else "OK"
                )
                pair_rows.append({
                    **base,
                    "method_a_id": method_a, "method_b_id": method_b,
                    "method_a_value": float(scope.points[method_a]),
                    "method_b_value": float(scope.points[method_b]),
                    "delta_a_minus_b": raw_point,
                    "delta_ci_low": summary["delta_ci_low"],
                    "delta_ci_high": summary["delta_ci_high"],
                    "oriented_advantage_a_over_b": oriented_point,
                    "oriented_ci_low": summary["oriented_ci_low"],
                    "oriented_ci_high": summary["oriented_ci_high"],
                    "probability_oriented_advantage_le_zero": summary[
                        "probability_oriented_advantage_le_zero"
                    ],
                    "bootstrap_unit": scope.bootstrap_unit,
                    "bootstrap_draws": scope.bootstrap_draws_requested,
                    "bootstrap_valid_draws": summary["bootstrap_valid_draws"],
                    "relation": relation,
                    "multiplicity_adjustment": "NONE",
                    "status": status,
                })

        point_winners_json = _json_text(list(point_winners))
        for candidate in methods:
            summary = _paired_summary(
                left=scope.draws[candidate], right=scope.draws[representative],
                metric=metric,
            )
            raw_point = float(scope.points[candidate]) - float(scope.points[representative])
            oriented_point = METRIC_DIRECTIONS[metric] * raw_point
            interval_contains_zero = (
                float(summary["oriented_ci_low"]) <= NUMERICAL_ZERO_ATOL[metric]
                and float(summary["oriented_ci_high"]) >= -NUMERICAL_ZERO_ATOL[metric]
            )
            if candidate in point_winners:
                _require(
                    interval_contains_zero,
                    "numerical point-winner tie is separated from the UTF-8 "
                    "representative by its direct paired interval",
                )
                membership = "POINT_WINNER"
                included = True
            elif interval_contains_zero:
                membership = "NOT_SEPARATED_FROM_POINT_WINNER_95CI"
                included = True
            else:
                membership = "SEPARATED_FROM_POINT_WINNER_95CI"
                included = False
            status = (
                "OK_FALLBACK" if "OK_FALLBACK" in {
                    scope.method_statuses[candidate],
                    scope.method_statuses[representative],
                } else "OK"
            )
            common = {
                **base,
                "point_winner_method_ids_json": point_winners_json,
                "winner_reference_method_id": representative,
                "candidate_method_id": candidate,
                "winner_value": float(scope.points[representative]),
                "candidate_value": float(scope.points[candidate]),
                "delta_candidate_minus_winner": raw_point,
                "delta_ci_low": summary["delta_ci_low"],
                "delta_ci_high": summary["delta_ci_high"],
                "oriented_advantage_candidate_over_winner": oriented_point,
                "oriented_ci_low": summary["oriented_ci_low"],
                "oriented_ci_high": summary["oriented_ci_high"],
                "probability_oriented_advantage_le_zero": summary[
                    "probability_oriented_advantage_le_zero"
                ],
                "bootstrap_unit": scope.bootstrap_unit,
                "bootstrap_draws": scope.bootstrap_draws_requested,
                "bootstrap_valid_draws": summary["bootstrap_valid_draws"],
                "membership_status": membership,
                "in_winner_reference_set": included,
                "multiplicity_adjustment": "NONE",
                "status": status,
            }
            winner_rows.append(common)
            set_rows.append({
                **base,
                "point_winner_method_ids_json": point_winners_json,
                "winner_reference_method_id": representative,
                "method_id": candidate,
                "method_value": float(scope.points[candidate]),
                "membership_status": membership,
                "in_winner_reference_set": included,
                "interpretation": (
                    "membership is determined by whether the direct paired percentile "
                    "interval versus the observed point winner contains numerical zero; "
                    "not equivalence or simultaneous inference"
                ),
            })

        draw_hashes = {
            method_id: _array_sha256(np.asarray(scope.draws[method_id], dtype=float))
            for method_id in methods
        }
        finite_counts = {
            method_id: int(np.sum(np.isfinite(scope.draws[method_id])))
            for method_id in methods
        }
        audit_scopes.append({
            "comparison_group_id": base["comparison_group_id"],
            "source_comparison_group_id": scope.source_comparison_group_id,
            "population_id": scope.population_id,
            "scope_type": scope.scope_type, "scope_value": scope.scope_value,
            "record_level": scope.record_level, "cell_ids": list(scope.cell_ids),
            "metric_id": metric, "bootstrap_unit": scope.bootstrap_unit,
            "bootstrap_seed": scope.bootstrap_seed,
            "bootstrap_draws_requested": scope.bootstrap_draws_requested,
            "shared_draw_array_length": next(iter(lengths)),
            "shared_finite_draw_mask_sha256": _array_sha256(
                shared_finite_mask.astype(np.uint8)
            ),
            "linked_resampling": scope.linked_resampling,
            "stratified_by_label": scope.stratified_by_label,
            "method_points": {method_id: float(scope.points[method_id])
                              for method_id in methods},
            "method_draw_sha256": draw_hashes,
            "method_finite_draws": finite_counts,
            "joint_draw_roster_sha256": sha256_bytes(canonical_json_bytes([
                {"method_id": method_id, "draw_sha256": draw_hashes[method_id],
                 "finite_draws": finite_counts[method_id]}
                for method_id in methods
            ])),
            "point_winner_method_ids": list(point_winners),
            "winner_reference_method_id": representative,
        })

    audit: dict[str, Any] = {
        "schema_version": DRAW_AUDIT_SCHEMA_VERSION,
        "status": "OK",
        "source_type": source.source_type,
        "lane_id": source.lane_id,
        "method_ids": list(methods), "metric_ids": list(source.metric_ids),
        "bootstrap_draws_requested": CANONICAL_DRAWS,
        "paired_draw_contract": (
            "same accepted grouped-bootstrap draw index for every method within "
            "one explicit scope and metric"
        ),
        "aggregate_draw_contract": (
            "equal-cell source scopes use same-index cell means; no cross-task macro is derived"
        ),
        "n_scopes": len(scopes), "scopes": audit_scopes,
    }
    audit["payload_sha256"] = sha256_bytes(canonical_json_bytes(audit))
    return {
        "all_pairs": pair_rows,
        "winner_contrasts": winner_rows,
        "winner_sets": set_rows,
        "draw_audit": audit,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), extrasaction="raise", lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        _require(set(row) == set(fields),
                 "derived CSV row does not match the frozen field roster")
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue().encode("utf-8")


def _derived_artifact_payloads(
    source: LoadedSource,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    derived = derive_winner_contrasts(source)
    payloads = {
        "all_pairs_contrasts.csv": _csv_bytes(
            derived["all_pairs"], ALL_PAIRS_FIELDS,
        ),
        "winner_reference_contrasts.csv": _csv_bytes(
            derived["winner_contrasts"], WINNER_CONTRAST_FIELDS,
        ),
        "winner_reference_sets.csv": _csv_bytes(
            derived["winner_sets"], WINNER_SET_FIELDS,
        ),
        "DRAW_AUDIT.json": canonical_json_bytes(derived["draw_audit"]) + b"\n",
    }
    return derived, payloads


def _expected_artifact_manifest(
    source: LoadedSource, *, replica_id: str,
    derived: Mapping[str, Any], payloads: Mapping[str, bytes],
    repo_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_repo_snapshot(repo_snapshot, name="winner-contrast manifest snapshot")
    files = {
        name: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
        for name, payload in payloads.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "OK", "replica_id": replica_id,
        "source_type": source.source_type, "lane_id": source.lane_id,
        "source_binding": dict(source.source_binding),
        "method_ids": list(_sorted_strings(source.method_ids)),
        "metric_ids": list(source.metric_ids),
        "bootstrap_draws_requested": CANONICAL_DRAWS,
        "scope_policy": {
            "explicit_source_scopes_only": True,
            "cross_task_macro_derived": False,
            "pairing": "all unordered method pairs within one scope and metric",
            "winner_reference": (
                "UTF-8-first representative of numerical point-estimate ties"
            ),
        },
        "inference_contract": {
            "interval": "paired percentile 95% bootstrap CI",
            "winner_set_membership": (
                "direct paired CI versus observed point winner contains numerical zero"
            ),
            "marginal_ci_overlap_used": False,
            "equivalence_claim": False,
            "winner_selection_adjusted": False,
            "simultaneous_or_familywise_coverage": False,
            "multiplicity_adjustment": "NONE",
        },
        "row_counts": {
            "all_pairs_contrasts.csv": len(derived["all_pairs"]),
            "winner_reference_contrasts.csv": len(derived["winner_contrasts"]),
            "winner_reference_sets.csv": len(derived["winner_sets"]),
        },
        "n_comparison_scopes": len(source.scopes),
        "files": files,
        "code_snapshot": dict(repo_snapshot),
    }
    manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def publish_winner_contrasts(
    source: LoadedSource, *, output_dir: str | Path, replica_id: str,
) -> dict[str, Any]:
    """Publish one immutable, content-bound downstream artifact directory."""

    _require(isinstance(replica_id, str) and bool(replica_id.strip()),
             "replica_id must be a nonempty string")
    target = Path(os.path.abspath(os.fspath(output_dir)))
    _require(not target.exists() and not target.is_symlink(),
             f"winner-contrast output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    repo_snapshot = _current_repo_snapshot()
    derived, payloads = _derived_artifact_payloads(source)
    _require_repo_snapshot_match(
        repo_snapshot, _current_repo_snapshot(),
        context="winner-contrast publication derivation",
    )
    manifest = _expected_artifact_manifest(
        source, replica_id=replica_id, derived=derived, payloads=payloads,
        repo_snapshot=repo_snapshot,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent))
    try:
        for name, payload in payloads.items():
            atomic_write_bytes(staging / name, payload)
        atomic_write_json(staging / "MANIFEST.json", manifest)
        _require_repo_snapshot_match(
            repo_snapshot, _current_repo_snapshot(),
            context="winner-contrast publication commit",
        )
        _rename_directory_noreplace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def verify_winner_contrast_artifact(
    path: str | Path, *, source: LoadedSource,
) -> dict[str, Any]:
    """Rederive and verify every artifact byte from its certified source."""

    requested_root = Path(os.path.abspath(os.fspath(path)))
    _require(requested_root.is_dir() and not requested_root.is_symlink(),
             "winner-contrast artifact directory is missing or unsafe")
    root = requested_root.resolve()
    manifest_path = root / "MANIFEST.json"
    manifest = _load_json(manifest_path, name="winner-contrast manifest")
    _validate_hashed_payload(manifest, name="winner-contrast manifest")
    _require(manifest.get("schema_version") == SCHEMA_VERSION
             and manifest.get("status") == "OK",
             "winner-contrast manifest is not an OK v1 artifact")
    _require(isinstance(manifest.get("replica_id"), str)
             and bool(manifest["replica_id"].strip()),
             "winner-contrast manifest has an invalid replica ID")
    current_snapshot = _current_repo_snapshot()
    manifest_snapshot = manifest.get("code_snapshot")
    _require(isinstance(manifest_snapshot, Mapping),
             "winner-contrast manifest lacks a repository code snapshot")
    _require_repo_snapshot_match(
        manifest_snapshot, current_snapshot,
        context="winner-contrast artifact verification",
    )
    files = manifest.get("files")
    expected_names = {
        "all_pairs_contrasts.csv", "winner_reference_contrasts.csv",
        "winner_reference_sets.csv", "DRAW_AUDIT.json",
    }
    _require(isinstance(files, Mapping) and set(files) == expected_names,
             "winner-contrast file roster drifted")
    observed_names = {
        child.name for child in root.iterdir() if child.is_file()
    }
    _require(observed_names == expected_names | {"MANIFEST.json"}
             and not any(child.is_symlink() for child in root.iterdir()),
             "winner-contrast directory contains missing, extra, or symlink files")
    raw_files: dict[str, bytes] = {}
    for name in sorted(expected_names, key=_utf8_key):
        file_path = root / name
        _require_plain_file(file_path, name=f"winner-contrast file {name}")
        payload = file_path.read_bytes()
        binding = files[name]
        _require(isinstance(binding, Mapping)
                 and binding.get("sha256") == sha256_bytes(payload)
                 and int(binding.get("bytes", -1)) == len(payload),
                 f"winner-contrast file binding failed: {name}")
        raw_files[name] = payload

    expected_derived, expected_payloads = _derived_artifact_payloads(source)
    _require_repo_snapshot_match(
        current_snapshot, _current_repo_snapshot(),
        context="winner-contrast verification derivation",
    )
    for name in sorted(expected_names, key=_utf8_key):
        _require(
            raw_files[name] == expected_payloads[name],
            f"winner-contrast artifact differs from exact source rederivation: {name}",
        )
    expected_manifest = _expected_artifact_manifest(
        source,
        replica_id=str(manifest.get("replica_id", "")),
        derived=expected_derived,
        payloads=expected_payloads,
        repo_snapshot=current_snapshot,
    )
    _require(
        manifest == expected_manifest,
        "winner-contrast manifest differs from exact source/code rederivation",
    )

    table_specs = (
        ("all_pairs_contrasts.csv", ALL_PAIRS_FIELDS, "all_pairs"),
        ("winner_reference_contrasts.csv", WINNER_CONTRAST_FIELDS, "winner"),
        ("winner_reference_sets.csv", WINNER_SET_FIELDS, "sets"),
    )
    tables: dict[str, list[dict[str, str]]] = {}
    for name, fields, key in table_specs:
        with io.StringIO(raw_files[name].decode("utf-8"), newline="") as stream:
            reader = csv.DictReader(stream)
            _require(tuple(reader.fieldnames or ()) == tuple(fields),
                     f"winner-contrast CSV header drifted: {name}")
            rows = [dict(row) for row in reader]
        expected_count = int(manifest["row_counts"][name])
        _require(len(rows) == expected_count,
                 f"winner-contrast CSV row count drifted: {name}")
        tables[key] = rows

    audit = json.loads(raw_files["DRAW_AUDIT.json"])
    _require(isinstance(audit, dict), "winner-contrast draw audit is invalid")
    _validate_hashed_payload(audit, name="winner-contrast draw audit")
    _require(audit.get("schema_version") == DRAW_AUDIT_SCHEMA_VERSION
             and audit.get("status") == "OK",
             "winner-contrast draw audit schema/status drifted")
    methods = _sorted_strings(manifest.get("method_ids", ()))
    _require(tuple(manifest.get("method_ids", ())) == methods,
             "winner-contrast method roster is not canonical")
    n_scopes = int(manifest.get("n_comparison_scopes", -1))
    _require(n_scopes == int(audit.get("n_scopes", -2)),
             "winner-contrast scope count differs from draw audit")
    expected_pairs = len(methods) * (len(methods) - 1) // 2
    pair_groups: dict[str, list[dict[str, str]]] = {}
    winner_groups: dict[str, list[dict[str, str]]] = {}
    set_groups: dict[str, list[dict[str, str]]] = {}
    for row in tables["all_pairs"]:
        pair_groups.setdefault(row["comparison_group_id"], []).append(row)
    for row in tables["winner"]:
        winner_groups.setdefault(row["comparison_group_id"], []).append(row)
    for row in tables["sets"]:
        set_groups.setdefault(row["comparison_group_id"], []).append(row)
    _require(set(pair_groups) == set(winner_groups) == set(set_groups)
             and len(pair_groups) == n_scopes,
             "winner-contrast comparison-group roster drifted")
    expected_pair_roster = {
        (left, right) for index, left in enumerate(methods)
        for right in methods[index + 1:]
    }
    for group_id in sorted(pair_groups, key=_utf8_key):
        pairs = pair_groups[group_id]
        winners = winner_groups[group_id]
        sets = set_groups[group_id]
        _require(len(pairs) == expected_pairs
                 and {(row["method_a_id"], row["method_b_id"]) for row in pairs}
                 == expected_pair_roster,
                 f"all-pairs roster incomplete: {group_id}")
        _require(len(winners) == len(methods)
                 and {row["candidate_method_id"] for row in winners} == set(methods),
                 f"winner-reference contrast roster incomplete: {group_id}")
        _require(len(sets) == len(methods)
                 and {row["method_id"] for row in sets} == set(methods),
                 f"winner-reference set roster incomplete: {group_id}")
        winner_membership = {
            row["candidate_method_id"]: (
                row["membership_status"],
                _parse_bool(row["in_winner_reference_set"], name="winner membership"),
            ) for row in winners
        }
        set_membership = {
            row["method_id"]: (
                row["membership_status"],
                _parse_bool(row["in_winner_reference_set"], name="set membership"),
            ) for row in sets
        }
        _require(winner_membership == set_membership,
                 f"winner-reference tables disagree: {group_id}")
        _require(any(status == "POINT_WINNER" and included
                     for status, included in winner_membership.values()),
                 f"winner-reference set has no point winner: {group_id}")

    return {
        "root": str(root), "manifest": manifest,
        "manifest_file_sha256": sha256_file(manifest_path),
        "raw_files": raw_files,
        "draw_audit": audit,
    }


_AB_NORMALIZED_SOURCE_FIELDS = frozenset({
    "build_id", "evaluation_manifest_sha256",
    "evaluation_manifest_payload_sha256", "score_freeze_manifest_sha256",
    "score_freeze_manifest_payload_sha256",
})


def _normalized_ab_manifest(manifest: Mapping[str, Any]) -> bytes:
    value = json.loads(json.dumps(manifest))
    value["replica_id"] = "<REPLICA>"
    source = value.get("source_binding")
    _require(isinstance(source, dict), "winner-contrast source binding is invalid")
    if value.get("source_type") == "external_v3_signed_score_label_recomputation":
        _require(_AB_NORMALIZED_SOURCE_FIELDS <= set(source),
                 "external winner-contrast manifest lacks A/B-normalized bindings")
        for field in _AB_NORMALIZED_SOURCE_FIELDS:
            source[field] = f"<BUILD_{field.upper()}>"
    value["payload_sha256"] = "<MANIFEST_PAYLOAD_SHA256>"
    return canonical_json_bytes(value)


def verify_winner_contrasts_ab(
    build_a: str | Path, build_b: str | Path, *, source_a: LoadedSource,
    source_b: LoadedSource, output_path: str | Path,
) -> dict[str, Any]:
    """Rederive both builds from certified sources, then issue a PASS cert."""

    left = verify_winner_contrast_artifact(build_a, source=source_a)
    right = verify_winner_contrast_artifact(build_b, source=source_b)
    _require(Path(left["root"]) != Path(right["root"]),
             "A/B verification requires two distinct artifact directories")
    left_manifest = left["manifest"]
    right_manifest = right["manifest"]
    _require(
        (left_manifest["replica_id"], right_manifest["replica_id"]) == ("A", "B"),
        "A/B verification requires replica A followed by replica B",
    )
    if (
        left_manifest.get("source_type")
        == "external_v3_signed_score_label_recomputation"
        or right_manifest.get("source_type")
        == "external_v3_signed_score_label_recomputation"
    ):
        _require(
            left_manifest.get("source_type")
            == right_manifest.get("source_type")
            == "external_v3_signed_score_label_recomputation",
            "external winner-contrast A/B source types diverged",
        )
        _require(
            (
                left_manifest.get("source_binding", {}).get("build_id"),
                right_manifest.get("source_binding", {}).get("build_id"),
            ) == ("A", "B"),
            "external winner-contrast A/B must use source builds A then B",
        )
    for name in (
        "all_pairs_contrasts.csv", "winner_reference_contrasts.csv",
        "winner_reference_sets.csv", "DRAW_AUDIT.json",
    ):
        _require(left["raw_files"][name] == right["raw_files"][name],
                 f"winner-contrast A/B byte identity failed: {name}")
    normalized_left = _normalized_ab_manifest(left_manifest)
    normalized_right = _normalized_ab_manifest(right_manifest)
    _require(normalized_left == normalized_right,
             "winner-contrast A/B manifests differ outside explicit normalization")

    normalized_source_fields = (
        sorted(_AB_NORMALIZED_SOURCE_FIELDS)
        if left_manifest["source_type"]
        == "external_v3_signed_score_label_recomputation"
        else []
    )
    run_repo_snapshot = _current_repo_snapshot()
    _require_repo_snapshot_match(
        left_manifest["code_snapshot"], run_repo_snapshot,
        context="winner-contrast A/B certification",
    )
    certificate: dict[str, Any] = {
        "schema_version": AB_VERIFICATION_SCHEMA_VERSION,
        "status": "PASS", "lane_id": left_manifest["lane_id"],
        "source_type": left_manifest["source_type"],
        "replica_ids": [left_manifest["replica_id"], right_manifest["replica_id"]],
        "normalization_contract": {
            "top_level": ["replica_id", "payload_sha256"],
            "source_binding_build_fields": normalized_source_fields,
            "all_other_manifest_fields": "byte-identical canonical JSON",
            "derived_tables_and_draw_audit": "byte-identical",
        },
        "exact_source_rederivation": {
            "performed": True,
            "build_A_source_binding_sha256": sha256_bytes(
                canonical_json_bytes(source_a.source_binding)
            ),
            "build_B_source_binding_sha256": sha256_bytes(
                canonical_json_bytes(source_b.source_binding)
            ),
        },
        "run_repo_snapshot": run_repo_snapshot,
        "normalized_manifest_sha256": sha256_bytes(normalized_left),
        "byte_identity": {
            name: sha256_bytes(left["raw_files"][name])
            for name in (
                "all_pairs_contrasts.csv", "winner_reference_contrasts.csv",
                "winner_reference_sets.csv", "DRAW_AUDIT.json",
            )
        },
        "builds": {
            "A": {
                "replica_id": left_manifest["replica_id"],
                "manifest_file_sha256": left["manifest_file_sha256"],
                "manifest_payload_sha256": left_manifest["payload_sha256"],
            },
            "B": {
                "replica_id": right_manifest["replica_id"],
                "manifest_file_sha256": right["manifest_file_sha256"],
                "manifest_payload_sha256": right_manifest["payload_sha256"],
            },
        },
    }
    certificate["certificate_sha256"] = sha256_bytes(canonical_json_bytes(certificate))
    target = Path(os.path.abspath(os.fspath(output_path)))
    _atomic_write_bytes_noclobber(
        target, canonical_json_bytes(certificate) + b"\n",
    )
    return certificate


__all__ = [
    "AB_VERIFICATION_SCHEMA_VERSION", "ALL_PAIRS_FIELDS", "CANONICAL_DRAWS",
    "DRAW_AUDIT_SCHEMA_VERSION", "EvaluatedScopeMetric", "LoadedSource",
    "SCHEMA_VERSION", "WINNER_CONTRAST_CODE_PATHS", "WINNER_CONTRAST_FIELDS",
    "WINNER_SET_FIELDS",
    "WinnerContrastError", "derive_winner_contrasts", "load_external_source",
    "load_frozen24_source", "publish_winner_contrasts",
    "verify_winner_contrast_artifact", "verify_winner_contrasts_ab",
]
