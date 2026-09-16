"""Dependence diagnostics for the Fusion Independence Atlas.

This module is deliberately downstream of signal extraction.  Correctness
annotations are allowed here because they define operational errors; none of
the APIs in this file are used to extract a signal or fit fusion weights.

The central contract is :class:`ErrorMatrix`: one matrix has exactly one error
target and one observation resolution.  In particular, predictor residuals,
raw ProcessBench misses, PRMBench pairwise misorderings, gate errors, and final
ProcessBench errors cannot accidentally be pooled into one dependence test.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge


class Resolution(str, Enum):
    TOKEN = "token"
    ANSWER = "answer"
    PAIR = "pair"


class ErrorTarget(str, Enum):
    PREDICTOR_RESIDUAL = "predictor_residual"
    PB_RAW_LOCATOR_MISS = "pb_raw_locator_miss"
    PRMB_PAIRWISE_MISORDER = "prmb_pairwise_misorder"
    GATE_FALSE_OPEN = "gate_false_open"
    GATE_FALSE_CLOSE = "gate_false_close"
    PB_FINAL_ERROR = "pb_final_error"


class DependenceStatus(str, Enum):
    INDEPENDENCE_COMPATIBLE = "INDEPENDENCE_COMPATIBLE"
    DEPENDENT_COMPLEMENTARY = "DEPENDENT_COMPLEMENTARY"
    REDUNDANT = "REDUNDANT"
    UNRESOLVED = "UNRESOLVED"


_TARGET_RESOLUTION = {
    ErrorTarget.PREDICTOR_RESIDUAL: Resolution.TOKEN,
    ErrorTarget.PB_RAW_LOCATOR_MISS: Resolution.ANSWER,
    ErrorTarget.PRMB_PAIRWISE_MISORDER: Resolution.PAIR,
    ErrorTarget.GATE_FALSE_OPEN: Resolution.ANSWER,
    ErrorTarget.GATE_FALSE_CLOSE: Resolution.ANSWER,
    ErrorTarget.PB_FINAL_ERROR: Resolution.ANSWER,
}
_BOUNDED_TARGETS = frozenset(set(ErrorTarget) - {ErrorTarget.PREDICTOR_RESIDUAL})
_BINARY_TARGETS = frozenset(
    {
        ErrorTarget.PB_RAW_LOCATOR_MISS,
        ErrorTarget.GATE_FALSE_OPEN,
        ErrorTarget.GATE_FALSE_CLOSE,
        ErrorTarget.PB_FINAL_ERROR,
    }
)


def _one_dimensional(value: Sequence, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or (length is not None and len(array) != length):
        suffix = "" if length is None else f" of length {length}"
        raise ValueError(f"{name} must be a one-dimensional array{suffix}")
    return array


def _take_indices(selection: Sequence[int] | np.ndarray, length: int) -> np.ndarray:
    """Normalize an integer selection or a true boolean mask to row indices."""
    selection = np.asarray(selection)
    if selection.ndim != 1:
        raise ValueError("row selection must be one-dimensional")
    if np.issubdtype(selection.dtype, np.bool_):
        if len(selection) != length:
            raise ValueError("boolean row mask has the wrong length")
        return np.flatnonzero(selection)
    if not np.issubdtype(selection.dtype, np.integer):
        raise TypeError("row selection must contain booleans or integers")
    indices = selection.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= length):
        raise IndexError("row selection is out of bounds")
    return indices


def _arrays_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    """Compare metadata arrays without applying ``isnan`` to string dtypes."""
    if left is None or right is None:
        return left is None and right is None
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _hash_array(digest: "hashlib._Hash", value: np.ndarray | None) -> None:
    if value is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    digest.update(memoryview(array).cast("B"))


@dataclass(frozen=True)
class ObservationMetadata:
    """Label-independent nuisance metadata aligned with error observations."""

    groups: np.ndarray
    cells: np.ndarray
    tokens: np.ndarray
    steps: np.ndarray
    folds: np.ndarray | None = None
    first_error_position: np.ndarray | None = None
    digit_opportunities: np.ndarray | None = None

    def __post_init__(self) -> None:
        groups = _one_dimensional(self.groups, "groups").astype(str)
        n = len(groups)
        cells = _one_dimensional(self.cells, "cells", n).astype(str)
        tokens = _one_dimensional(self.tokens, "tokens", n).astype(np.float64)
        steps = _one_dimensional(self.steps, "steps", n).astype(np.float64)
        if n == 0 or np.any(groups == "") or np.any(cells == ""):
            raise ValueError("metadata requires nonempty observations, groups, and cells")
        if not np.isfinite(tokens).all() or not np.isfinite(steps).all() or np.any(tokens < 0) or np.any(steps < 0):
            raise ValueError("tokens and steps must be finite and nonnegative")

        folds = None
        if self.folds is not None:
            folds = _one_dimensional(self.folds, "folds", n).astype(np.int64)
            if np.any(folds < 0):
                raise ValueError("fold identifiers must be nonnegative")
            # One vectorized pass replaces one full ``groups == group`` scan
            # per source group.  At Atlas scale the old validation performed
            # ~24B Unicode comparisons every time token metadata was sliced.
            _, group_codes = np.unique(groups, return_inverse=True)
            group_count = int(group_codes.max()) + 1
            minimum = np.full(group_count, np.iinfo(np.int64).max, dtype=np.int64)
            maximum = np.full(group_count, -1, dtype=np.int64)
            np.minimum.at(minimum, group_codes, folds)
            np.maximum.at(maximum, group_codes, folds)
            if np.any(minimum != maximum):
                raise ValueError("a source group cannot occur in multiple folds")

        first = None
        if self.first_error_position is not None:
            first = _one_dimensional(self.first_error_position, "first_error_position", n).astype(np.float64)
            finite = np.isfinite(first)
            if np.any((first[finite] < 0) | (first[finite] > 1)):
                raise ValueError("relative first-error position must lie in [0, 1]")

        digit = None
        if self.digit_opportunities is not None:
            digit = _one_dimensional(self.digit_opportunities, "digit_opportunities", n).astype(np.float64)
            finite = np.isfinite(digit)
            if np.any(digit[finite] < 0):
                raise ValueError("digit opportunities must be nonnegative")

        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "first_error_position", first)
        object.__setattr__(self, "digit_opportunities", digit)

    def __len__(self) -> int:
        return len(self.groups)

    def take(self, indices: Sequence[int] | np.ndarray) -> "ObservationMetadata":
        indices = _take_indices(indices, len(self))
        optional = lambda value: None if value is None else value[indices]
        return ObservationMetadata(
            groups=self.groups[indices],
            cells=self.cells[indices],
            tokens=self.tokens[indices],
            steps=self.steps[indices],
            folds=optional(self.folds),
            first_error_position=optional(self.first_error_position),
            digit_opportunities=optional(self.digit_opportunities),
        )


def _method_matrix(value: Sequence, n: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n:
        raise ValueError(f"{name} must have shape [observations, methods]")
    return array


def _normalise_topk(
    topk_sets: Sequence[Sequence[Iterable[int] | None]] | None,
    n: int,
    m: int,
) -> tuple[tuple[tuple[frozenset[int], ...], ...] | None, np.ndarray | None]:
    if topk_sets is None:
        return None, None
    if len(topk_sets) != m:
        raise ValueError("top-k evidence must be method-major")
    result = []
    valid = np.ones((n, m), dtype=bool)
    for column, method in enumerate(topk_sets):
        if len(method) != n:
            raise ValueError("top-k evidence is not observation-aligned")
        observations = []
        for row, observation in enumerate(method):
            if observation is None:
                valid[row, column] = False
                observations.append(frozenset())
                continue
            selected = frozenset(int(item) for item in observation)
            if not selected:
                valid[row, column] = False
            observations.append(selected)
        result.append(tuple(observations))
    return tuple(result), valid


@dataclass(frozen=True)
class ErrorMatrix:
    """A homogeneous operational-error matrix.

    Values are high-is-error.  All non-predictor targets are bounded in
    ``[0, 1]``.  PRMBench ties are represented by exactly ``0.5``.
    """

    target: ErrorTarget
    resolution: Resolution
    method_names: tuple[str, ...]
    values: np.ndarray
    metadata: ObservationMetadata
    scores: np.ndarray | None = None
    peaks: np.ndarray | None = None
    topk_sets: tuple[tuple[frozenset[int], ...], ...] | None = None
    active: np.ndarray | None = None
    identity: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        target = ErrorTarget(self.target)
        resolution = Resolution(self.resolution)
        if _TARGET_RESOLUTION[target] != resolution:
            raise ValueError(f"{target.value} requires {_TARGET_RESOLUTION[target].value} resolution")
        names = tuple(str(name) for name in self.method_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("method names must be nonempty and unique")
        values = _method_matrix(self.values, len(self.metadata), "errors").astype(np.float64)
        if values.shape[1] != len(names):
            raise ValueError("error columns must align with method names")
        active = (
            np.ones(values.shape, dtype=bool)
            if self.active is None
            else _method_matrix(self.active, len(self.metadata), "active").astype(bool)
        )
        if active.shape != values.shape:
            raise ValueError("active masks must align with errors")
        finite_values = np.isfinite(values)
        if np.any(active & ~finite_values):
            raise ValueError("active error observations must be finite")
        # ``active`` describes availability of the operational error itself.
        # Optional scores, peaks and Top-k sets have their own evidence masks
        # inside ``pair_diagnostics`` and must not silently remove a defined
        # inactive-as-closed/failure outcome from the population denominator.
        values = np.where(finite_values, values, 0.0)
        if target in _BOUNDED_TARGETS and np.any((values[active] < 0) | (values[active] > 1)):
            raise ValueError("discrete operational errors must lie in [0, 1]")
        if target in _BINARY_TARGETS and not np.isin(values[active], (0.0, 1.0)).all():
            raise ValueError(f"{target.value} requires binary errors")
        if target == ErrorTarget.PRMB_PAIRWISE_MISORDER and not np.isin(values[active], (0.0, 0.5, 1.0)).all():
            raise ValueError("PRMB pairwise errors must be 0, 0.5 (tie), or 1")

        scores = None
        if self.scores is not None:
            scores = _method_matrix(self.scores, len(self.metadata), "scores").astype(np.float64)
            if scores.shape != values.shape:
                raise ValueError("scores must align with errors")
        peaks = None
        if self.peaks is not None:
            peaks = _method_matrix(self.peaks, len(self.metadata), "peaks")
            if peaks.shape != values.shape or not np.issubdtype(peaks.dtype, np.integer):
                raise ValueError("peaks must be integer-valued and align with errors")
            peaks = peaks.astype(np.int64)
        topk, _ = _normalise_topk(self.topk_sets, len(self.metadata), len(names))

        digest = hashlib.sha256()
        digest.update(target.value.encode())
        digest.update(resolution.value.encode())
        digest.update("\0".join(names).encode())
        for value in (
            values,
            active,
            self.metadata.groups,
            self.metadata.cells,
            self.metadata.tokens,
            self.metadata.steps,
            self.metadata.folds,
            self.metadata.first_error_position,
            self.metadata.digit_opportunities,
        ):
            _hash_array(digest, value)

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "method_names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "peaks", peaks)
        object.__setattr__(self, "topk_sets", topk)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "identity", digest.hexdigest())

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def active_mask(self) -> np.ndarray:
        return self.active

    def take(self, indices: Sequence[int] | np.ndarray) -> "ErrorMatrix":
        indices = _take_indices(indices, len(self.metadata))
        topk = None
        if self.topk_sets is not None:
            topk = tuple(tuple(method[index] for index in indices) for method in self.topk_sets)
        return ErrorMatrix(
            target=self.target,
            resolution=self.resolution,
            method_names=self.method_names,
            values=self.values[indices],
            metadata=self.metadata.take(indices),
            scores=None if self.scores is None else self.scores[indices],
            peaks=None if self.peaks is None else self.peaks[indices],
            topk_sets=topk,
            active=self.active[indices],
        )

    def with_values(self, values: np.ndarray) -> "ErrorMatrix":
        """Return the same typed matrix with recalibrated error values."""
        return replace(self, values=values)


def combine_error_matrices(*matrices: ErrorMatrix) -> ErrorMatrix:
    """Column-bind homogeneous matrices; mixed targets/resolutions are invalid."""
    if not matrices:
        raise ValueError("at least one error matrix is required")
    first = matrices[0]
    for matrix in matrices[1:]:
        if matrix.target != first.target or matrix.resolution != first.resolution:
            raise ValueError("cannot combine different error targets or resolutions")
        if len(matrix.metadata) != len(first.metadata):
            raise ValueError("error matrices are not observation-aligned")
        for field in ("groups", "cells", "tokens", "steps", "folds", "first_error_position", "digit_opportunities"):
            left, right = getattr(first.metadata, field), getattr(matrix.metadata, field)
            if not _arrays_equal(left, right):
                raise ValueError(f"error matrices disagree on metadata field {field}")
    scores = None
    if all(matrix.scores is not None for matrix in matrices):
        scores = np.column_stack([matrix.scores for matrix in matrices])
    elif any(matrix.scores is not None for matrix in matrices):
        raise ValueError("score evidence must be present for every combined matrix")
    peaks = None
    if all(matrix.peaks is not None for matrix in matrices):
        peaks = np.column_stack([matrix.peaks for matrix in matrices])
    elif any(matrix.peaks is not None for matrix in matrices):
        raise ValueError("peak evidence must be present for every combined matrix")
    topk = None
    if all(matrix.topk_sets is not None for matrix in matrices):
        topk = tuple(item for matrix in matrices for item in matrix.topk_sets)
    elif any(matrix.topk_sets is not None for matrix in matrices):
        raise ValueError("top-k evidence must be present for every combined matrix")
    return ErrorMatrix(
        first.target,
        first.resolution,
        tuple(name for matrix in matrices for name in matrix.method_names),
        np.column_stack([matrix.values for matrix in matrices]),
        first.metadata,
        scores=scores,
        peaks=peaks,
        topk_sets=topk,
        active=np.column_stack([matrix.active for matrix in matrices]),
    )


def predictor_residual_matrix(
    observed: Sequence[float],
    predictions: np.ndarray,
    method_names: Sequence[str],
    metadata: ObservationMetadata,
    *,
    active: np.ndarray | None = None,
) -> ErrorMatrix:
    observed = _one_dimensional(observed, "observed", len(metadata)).astype(np.float64)
    predictions = _method_matrix(predictions, len(metadata), "predictions").astype(np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("observed primitive values must be finite")
    active_values = np.isfinite(predictions)
    if active is not None:
        active_values &= _method_matrix(active, len(metadata), "active").astype(bool)
    safe_predictions = np.where(np.isfinite(predictions), predictions, observed[:, None])
    return ErrorMatrix(
        ErrorTarget.PREDICTOR_RESIDUAL,
        Resolution.TOKEN,
        tuple(method_names),
        observed[:, None] - safe_predictions,
        metadata,
        scores=safe_predictions,
        active=active_values,
    )


def pb_raw_locator_miss_matrix(
    peaks: np.ndarray,
    target_steps: Sequence[int],
    valid: np.ndarray,
    method_names: Sequence[str],
    metadata: ObservationMetadata,
    *,
    scores: np.ndarray | None = None,
    active: np.ndarray | None = None,
    topk_sets: Sequence[Sequence[Iterable[int] | None]] | None = None,
    retain_invalid_as_failures: bool = False,
) -> ErrorMatrix:
    target = _one_dimensional(target_steps, "target_steps", len(metadata)).astype(np.int64)
    peaks = _method_matrix(peaks, len(metadata), "peaks").astype(np.int64)
    valid = _method_matrix(valid, len(metadata), "valid").astype(bool)
    if peaks.shape != valid.shape or peaks.shape[1] != len(method_names):
        raise ValueError("locator arrays do not align")
    pb = np.char.startswith(metadata.cells.astype(str), "pb_")
    keep = pb & (target >= 0)
    if not np.any(keep):
        raise ValueError("raw locator misses require erroneous ProcessBench answers")
    errors = (~valid[keep] | (peaks[keep] != target[keep, None])).astype(np.float64)
    score_values = None if scores is None else _method_matrix(scores, len(metadata), "scores")[keep]
    signal_active = valid & (peaks >= 0)
    if active is not None:
        signal_active &= _method_matrix(active, len(metadata), "active").astype(bool)
    active_values = np.ones_like(signal_active, dtype=bool) if retain_invalid_as_failures else signal_active
    selected_topk = None
    if topk_sets is not None:
        if len(topk_sets) != len(method_names):
            raise ValueError("raw locator top-k evidence is not method-aligned")
        kept = np.flatnonzero(keep)
        selected_topk = tuple(
            tuple(
                frozenset() if retain_invalid_as_failures and method[index] is None else method[index]
                for index in kept
            )
            for method in topk_sets
        )
    return ErrorMatrix(
        ErrorTarget.PB_RAW_LOCATOR_MISS,
        Resolution.ANSWER,
        tuple(method_names),
        errors,
        metadata.take(np.flatnonzero(keep)),
        scores=score_values,
        peaks=peaks[keep],
        topk_sets=selected_topk,
        active=active_values[keep],
    )


def prmb_pairwise_misorder_matrix(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    method_names: Sequence[str],
    metadata: ObservationMetadata,
    *,
    higher_is_positive: bool = True,
    active: np.ndarray | None = None,
) -> ErrorMatrix:
    positive = _method_matrix(positive_scores, len(metadata), "positive_scores").astype(np.float64)
    negative = _method_matrix(negative_scores, len(metadata), "negative_scores").astype(np.float64)
    if positive.shape != negative.shape or positive.shape[1] != len(method_names):
        raise ValueError("PRMB score pairs do not align")
    active_values = np.isfinite(positive) & np.isfinite(negative)
    if active is not None:
        active_values &= _method_matrix(active, len(metadata), "active").astype(bool)
    safe_positive = np.where(np.isfinite(positive), positive, 0.0)
    safe_negative = np.where(np.isfinite(negative), negative, 0.0)
    difference = safe_positive - safe_negative if higher_is_positive else safe_negative - safe_positive
    values = np.where(difference < 0, 1.0, np.where(difference == 0, 0.5, 0.0))
    return ErrorMatrix(
        ErrorTarget.PRMB_PAIRWISE_MISORDER,
        Resolution.PAIR,
        tuple(method_names),
        values,
        metadata,
        scores=difference,
        active=active_values,
    )


def _gate_error_matrix(
    gate_open: np.ndarray,
    relevant: Sequence[bool],
    method_names: Sequence[str],
    metadata: ObservationMetadata,
    target: ErrorTarget,
    scores: np.ndarray | None,
    active: np.ndarray | None,
) -> ErrorMatrix:
    opened = _method_matrix(gate_open, len(metadata), "gate_open").astype(bool)
    if opened.shape[1] != len(method_names):
        raise ValueError("gate decisions do not align with method names")
    relevant = _one_dimensional(relevant, "relevant", len(metadata)).astype(bool)
    if not relevant.any():
        raise ValueError("gate error matrix has no relevant answers")
    errors = opened[relevant] if target == ErrorTarget.GATE_FALSE_OPEN else ~opened[relevant]
    score_values = None if scores is None else _method_matrix(scores, len(metadata), "scores")[relevant]
    active_values = None if active is None else _method_matrix(active, len(metadata), "active")[relevant]
    return ErrorMatrix(
        target,
        Resolution.ANSWER,
        tuple(method_names),
        errors.astype(np.float64),
        metadata.take(np.flatnonzero(relevant)),
        scores=score_values,
        active=active_values,
    )


def gate_false_open_matrix(
    gate_open: np.ndarray,
    clean_mask: Sequence[bool],
    method_names: Sequence[str],
    metadata: ObservationMetadata,
    *,
    scores: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> ErrorMatrix:
    return _gate_error_matrix(gate_open, clean_mask, method_names, metadata, ErrorTarget.GATE_FALSE_OPEN, scores, active)


def gate_false_close_matrix(
    gate_open: np.ndarray,
    erroneous_mask: Sequence[bool],
    method_names: Sequence[str],
    metadata: ObservationMetadata,
    *,
    scores: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> ErrorMatrix:
    return _gate_error_matrix(gate_open, erroneous_mask, method_names, metadata, ErrorTarget.GATE_FALSE_CLOSE, scores, active)


def final_pb_error_matrix(
    predictions: np.ndarray,
    target_steps: Sequence[int],
    valid: np.ndarray,
    method_names: Sequence[str],
    metadata: ObservationMetadata,
    *,
    scores: np.ndarray | None = None,
    active: np.ndarray | None = None,
    retain_invalid_as_failures: bool = False,
) -> ErrorMatrix:
    target = _one_dimensional(target_steps, "target_steps", len(metadata)).astype(np.int64)
    predictions = _method_matrix(predictions, len(metadata), "predictions").astype(np.int64)
    valid = _method_matrix(valid, len(metadata), "valid").astype(bool)
    if predictions.shape != valid.shape or predictions.shape[1] != len(method_names):
        raise ValueError("final PB arrays do not align")
    pb = np.char.startswith(metadata.cells.astype(str), "pb_")
    if not pb.any():
        raise ValueError("final PB errors require ProcessBench answers")
    errors = (~valid[pb] | (predictions[pb] != target[pb, None])).astype(np.float64)
    score_values = None if scores is None else _method_matrix(scores, len(metadata), "scores")[pb]
    signal_active = valid.copy()
    if active is not None:
        signal_active &= _method_matrix(active, len(metadata), "active").astype(bool)
    active_values = np.ones_like(signal_active, dtype=bool) if retain_invalid_as_failures else signal_active
    return ErrorMatrix(
        ErrorTarget.PB_FINAL_ERROR,
        Resolution.ANSWER,
        tuple(method_names),
        errors,
        metadata.take(np.flatnonzero(pb)),
        scores=score_values,
        # ``-1`` is a valid final clean prediction, not an invalid locator peak.
        peaks=None,
        active=active_values[pb],
    )


@dataclass(frozen=True)
class GateErrorMatrices:
    """Full-population gate calibration followed by separate error targets."""

    false_open: ErrorMatrix
    false_close: ErrorMatrix
    gate_open: np.ndarray
    gate_percentiles: np.ndarray
    active: np.ndarray


def rank_fused_gate_decisions(
    detector_scores: np.ndarray,
    detector_names: Sequence[str],
    cells: Sequence[str],
    fusion_sets: Mapping[str, Sequence[str | int]],
    *,
    q: float = 0.33,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-cell midrank, equal-rank fuse, rerank, then threshold.

    Inactive detectors are omitted row-wise from the positive mean.  A fused
    gate is inactive only when none of its registered detectors is active.
    """
    scores = np.asarray(detector_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(detector_names) or not len(scores):
        raise ValueError("detector scores must have shape [answers, detectors]")
    names = tuple(str(name) for name in detector_names)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("detector names must be nonempty and unique")
    cells = _one_dimensional(cells, "cells", len(scores)).astype(str)
    detector_active = np.isfinite(scores)
    if active is not None:
        detector_active &= _method_matrix(active, len(scores), "active").astype(bool)
    safe_scores = np.where(detector_active, scores, 0.0)
    ranked = np.full_like(safe_scores, np.nan)
    for cell in sorted(set(cells)):
        in_cell = cells == cell
        for column in range(scores.shape[1]):
            keep = in_cell & detector_active[:, column]
            if keep.any():
                ranked[keep, column] = (rankdata(safe_scores[keep, column], method="average") - 0.5) / int(keep.sum())

    fusion_names = tuple(str(name) for name in fusion_sets)
    if not fusion_names or len(set(fusion_names)) != len(fusion_names):
        raise ValueError("fusion-set names must be nonempty and unique")
    fused = np.full((len(scores), len(fusion_names)), np.nan, dtype=np.float64)
    fused_active = np.zeros_like(fused, dtype=bool)
    for output, fusion_name in enumerate(fusion_names):
        raw_members = tuple(fusion_sets[fusion_name])
        if not raw_members:
            raise ValueError(f"gate fusion {fusion_name} is empty")
        members = tuple(names.index(member) if isinstance(member, str) else int(member) for member in raw_members)
        if len(set(members)) != len(members) or any(member < 0 or member >= len(names) for member in members):
            raise ValueError(f"gate fusion {fusion_name} has invalid members")
        member_active = detector_active[:, members]
        count = member_active.sum(axis=1)
        fused_active[:, output] = count > 0
        numerator = np.where(member_active, ranked[:, members], 0.0).sum(axis=1)
        fused[fused_active[:, output], output] = numerator[fused_active[:, output]] / count[fused_active[:, output]]

    percentiles = np.full_like(fused, np.nan)
    for cell in sorted(set(cells)):
        in_cell = cells == cell
        for column in range(len(fusion_names)):
            keep = in_cell & fused_active[:, column]
            if keep.any():
                percentiles[keep, column] = (rankdata(fused[keep, column], method="average") - 0.5) / int(keep.sum())
    if not (0 < q < 1):
        raise ValueError("gate threshold must lie in (0, 1)")
    opened = fused_active & (percentiles >= float(q))
    return opened, percentiles, fused_active


def gate_error_matrices_from_scores(
    detector_scores: np.ndarray,
    detector_names: Sequence[str],
    fusion_sets: Mapping[str, Sequence[str | int]],
    erroneous_mask: Sequence[bool],
    metadata: ObservationMetadata,
    *,
    q: float = 0.33,
    detector_active: np.ndarray | None = None,
    processbench_mask: Sequence[bool] | None = None,
    retain_inactive_as_closed: bool = False,
) -> GateErrorMatrices:
    """Calibrate gates on the full roster, then derive FO and FC separately."""
    erroneous = _one_dimensional(erroneous_mask, "erroneous_mask", len(metadata)).astype(bool)
    if processbench_mask is None:
        pb = np.char.startswith(metadata.cells.astype(str), "pb_")
    else:
        pb = _one_dimensional(processbench_mask, "processbench_mask", len(metadata)).astype(bool)
    if np.any(erroneous & ~pb) or not pb.any():
        raise ValueError("erroneous gate labels must be a subset of ProcessBench")
    opened, percentiles, active = rank_fused_gate_decisions(
        detector_scores,
        detector_names,
        metadata.cells,
        fusion_sets,
        q=q,
        active=detector_active,
    )
    method_names = tuple(str(name) for name in fusion_sets)
    false_open = gate_false_open_matrix(
        opened,
        pb & ~erroneous,
        method_names,
        metadata,
        scores=percentiles,
        active=None if retain_inactive_as_closed else active,
    )
    false_close = gate_false_close_matrix(
        opened,
        erroneous,
        method_names,
        metadata,
        scores=percentiles,
        active=None if retain_inactive_as_closed else active,
    )
    return GateErrorMatrices(false_open, false_close, opened, percentiles, active)


@dataclass(frozen=True)
class FullPopulationGateBootstrap:
    point_errors: GateErrorMatrices
    statistics: "BootstrapSamples"


GateBootstrapStatistic = Callable[[GateErrorMatrices], float | Sequence[float] | np.ndarray]


def full_population_gate_bootstrap(
    detector_scores: np.ndarray,
    detector_names: Sequence[str],
    fusion_sets: Mapping[str, Sequence[str | int]],
    erroneous_mask: Sequence[bool],
    metadata: ObservationMetadata,
    statistic: GateBootstrapStatistic,
    *,
    q: float = 0.33,
    detector_active: np.ndarray | None = None,
    processbench_mask: Sequence[bool] | None = None,
    draws: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    retain_inactive_as_closed: bool = False,
) -> FullPopulationGateBootstrap:
    """Bootstrap full answers and recalibrate the complete gate in every draw."""
    if draws <= 0 or not (0 < confidence < 1):
        raise ValueError("bootstrap draws and confidence are invalid")
    scores = _method_matrix(detector_scores, len(metadata), "detector_scores").astype(np.float64)
    erroneous = _one_dimensional(erroneous_mask, "erroneous_mask", len(metadata)).astype(bool)
    active = None if detector_active is None else _method_matrix(detector_active, len(metadata), "detector_active").astype(bool)
    pb = None if processbench_mask is None else _one_dimensional(processbench_mask, "processbench_mask", len(metadata)).astype(bool)
    point_errors = gate_error_matrices_from_scores(
        scores,
        detector_names,
        fusion_sets,
        erroneous,
        metadata,
        q=q,
        detector_active=active,
        processbench_mask=pb,
        retain_inactive_as_closed=retain_inactive_as_closed,
    )
    point = np.atleast_1d(np.asarray(statistic(point_errors), dtype=np.float64))
    if point.ndim != 1:
        raise ValueError("gate bootstrap statistic must return a scalar or vector")
    _, rows = _group_rows(metadata.groups)
    if len(rows) < 2:
        raise ValueError("source-group bootstrap requires at least two groups")
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(draws), len(point)), dtype=np.float64)
    for draw in range(int(draws)):
        indices = _bootstrap_indices(rows, rng)
        errors = gate_error_matrices_from_scores(
            scores[indices],
            detector_names,
            fusion_sets,
            erroneous[indices],
            metadata.take(indices),
            q=q,
            detector_active=None if active is None else active[indices],
            processbench_mask=None if pb is None else pb[indices],
            retain_inactive_as_closed=retain_inactive_as_closed,
        )
        value = np.atleast_1d(np.asarray(statistic(errors), dtype=np.float64))
        if value.shape != point.shape:
            raise ValueError("gate bootstrap statistic changed shape")
        samples[draw] = value
    finite = np.isfinite(samples)
    applicable = np.isfinite(point)
    result = BootstrapSamples(
        point=point,
        samples=samples,
        seed=int(seed),
        confidence=float(confidence),
        requested_draws=int(draws),
        finite_counts=tuple(int(value) for value in finite.sum(axis=0)),
        fully_finite_draws=(
            int(np.all(finite[:, applicable], axis=1).sum()) if applicable.any() else 0
        ),
    )
    return FullPopulationGateBootstrap(point_errors, result)


def deterministic_group_folds(groups: Sequence[str], n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Assign whole source groups to deterministic, approximately balanced folds."""
    groups = _one_dimensional(groups, "groups").astype(str)
    unique, counts = np.unique(groups, return_counts=True)
    if n_splits < 2 or len(unique) < 2:
        raise ValueError("cross-fitting requires at least two source groups and folds")
    actual = min(int(n_splits), len(unique))
    keyed = sorted(
        zip(unique, counts),
        key=lambda item: (
            -int(item[1]),
            hashlib.sha256(f"fusion-atlas-fold:{seed}:{item[0]}".encode()).digest(),
        ),
    )
    loads = np.zeros(actual, dtype=np.int64)
    assignment: dict[str, int] = {}
    for group, count in keyed:
        fold = int(np.argmin(loads))
        assignment[str(group)] = fold
        loads[fold] += int(count)
    return np.array([assignment[str(group)] for group in groups], dtype=np.int64)


@dataclass(frozen=True)
class NuisanceDesign:
    values: np.ndarray
    columns: tuple[str, ...]
    continuous: tuple[int, ...]


def nuisance_design(metadata: ObservationMetadata) -> NuisanceDesign:
    """Build the registered label-independent nuisance design."""
    columns: list[str] = []
    blocks: list[np.ndarray] = []
    continuous: list[int] = []
    categories = sorted(set(metadata.cells))
    # Drop one reference cell; the estimators supply an intercept.
    for cell in categories[1:]:
        columns.append(f"cell={cell}")
        blocks.append((metadata.cells == cell).astype(np.float64))
    for name, values in (
        ("log1p_tokens", np.log1p(metadata.tokens)),
        ("log1p_steps", np.log1p(metadata.steps)),
    ):
        continuous.append(len(columns))
        columns.append(name)
        blocks.append(values)
    if metadata.first_error_position is not None:
        missing = ~np.isfinite(metadata.first_error_position)
        continuous.append(len(columns))
        columns.append("first_error_position")
        blocks.append(np.where(missing, 0.0, metadata.first_error_position))
        if missing.any():
            columns.append("first_error_position_missing")
            blocks.append(missing.astype(np.float64))
    if metadata.digit_opportunities is not None:
        missing = ~np.isfinite(metadata.digit_opportunities)
        continuous.append(len(columns))
        columns.append("log1p_digit_opportunities")
        blocks.append(np.log1p(np.where(missing, 0.0, metadata.digit_opportunities)))
        if missing.any():
            columns.append("digit_opportunities_missing")
            blocks.append(missing.astype(np.float64))
    matrix = np.column_stack(blocks).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("nuisance design is nonfinite")
    return NuisanceDesign(matrix, tuple(columns), tuple(continuous))


@dataclass(frozen=True)
class ConditionedErrors:
    expected: np.ndarray
    residuals: np.ndarray
    folds: np.ndarray
    design_columns: tuple[str, ...]
    model_kinds: tuple[tuple[str, ...], ...]
    train_groups_by_fold: Mapping[int, frozenset[str]]
    source_identity: str
    method_names: tuple[str, ...]
    active: np.ndarray

    def validate(self, matrix: ErrorMatrix) -> None:
        if (
            self.source_identity != matrix.identity
            or self.method_names != matrix.method_names
            or self.expected.shape != matrix.values.shape
            or self.residuals.shape != matrix.values.shape
            or self.active.shape != matrix.values.shape
            or not np.array_equal(self.active, matrix.active)
        ):
            raise ValueError("conditioned errors belong to a different matrix or row order")

    def take(
        self,
        indices: Sequence[int] | np.ndarray,
        source: ErrorMatrix,
    ) -> "ConditionedErrors":
        indices = _take_indices(indices, len(self.folds))
        if source.shape != (len(indices), self.expected.shape[1]):
            raise ValueError("conditioned subset does not align with its source matrix")
        return ConditionedErrors(
            self.expected[indices],
            self.residuals[indices],
            self.folds[indices],
            self.design_columns,
            self.model_kinds,
            self.train_groups_by_fold,
            source.identity,
            source.method_names,
            self.active[indices],
        )


def _fold_scaled_design(design: NuisanceDesign, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_x = design.values[train].copy()
    test_x = design.values[test].copy()
    if design.continuous:
        index = np.asarray(design.continuous, dtype=np.int64)
        mean = train_x[:, index].mean(axis=0)
        scale = train_x[:, index].std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        train_x[:, index] = (train_x[:, index] - mean) / scale
        test_x[:, index] = (test_x[:, index] - mean) / scale
    return train_x, test_x


def _restore_prevalence(probability: np.ndarray, prevalence: float) -> np.ndarray:
    """Undo the artificial 50/50 prior induced by balanced class weights."""
    probability = np.clip(probability, 1e-9, 1 - 1e-9)
    odds = probability / (1.0 - probability)
    odds *= prevalence / (1.0 - prevalence)
    return np.clip(odds / (1.0 + odds), 1e-9, 1 - 1e-9)


def cross_fit_nuisance(
    matrix: ErrorMatrix,
    *,
    n_splits: int = 5,
    seed: int = 0,
    ridge_alpha: float = 1.0,
    logistic_c: float = 1.0,
) -> ConditionedErrors:
    """Fit nuisance expectations only on source groups outside each held fold.

    Exact binary columns use balanced logistic regression.  Continuous
    predictor residuals and PRMB columns containing half-ties use ridge.  The
    only returned predictions are fold-local held predictions; they must not be
    concatenated and treated as one globally calibrated ranking score.
    """
    if ridge_alpha < 0 or logistic_c <= 0:
        raise ValueError("regularization parameters must be positive")
    design = nuisance_design(matrix.metadata)
    folds = (
        deterministic_group_folds(matrix.metadata.groups, n_splits=n_splits, seed=seed)
        if matrix.metadata.folds is None
        else matrix.metadata.folds.copy()
    )
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two populated folds")
    expected = np.full_like(matrix.values, np.nan, dtype=np.float64)
    kinds: list[list[str]] = [[] for _ in matrix.method_names]
    train_groups: dict[int, frozenset[str]] = {}
    for fold in unique_folds:
        held = folds == fold
        train = ~held
        if not held.any() or not train.any():
            raise ValueError("empty train or held partition")
        held_groups = set(matrix.metadata.groups[held])
        fitted_groups = frozenset(str(group) for group in matrix.metadata.groups[train])
        if held_groups & set(fitted_groups):
            raise ValueError("source-group leakage in nuisance cross-fit")
        train_groups[fold] = fitted_groups
        for column in range(matrix.values.shape[1]):
            train_column = train & matrix.active[:, column]
            held_column = held & matrix.active[:, column]
            if not held_column.any():
                kinds[column].append("inactive_fold")
                continue
            if not train_column.any():
                raise ValueError(f"method {matrix.method_names[column]} has no active training observations")
            train_x, held_x = _fold_scaled_design(design, train_column, held_column)
            y = matrix.values[train_column, column]
            binary = np.isin(y, (0.0, 1.0)).all() and matrix.target != ErrorTarget.PREDICTOR_RESIDUAL
            if binary and len(np.unique(y)) == 2:
                estimator = LogisticRegression(
                    C=float(logistic_c),
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=int(seed),
                )
                estimator.fit(train_x, y.astype(np.int8))
                probability = estimator.predict_proba(held_x)[:, 1]
                expected[held_column, column] = _restore_prevalence(probability, float(y.mean()))
                kinds[column].append("balanced_logistic")
            elif binary:
                expected[held_column, column] = float(y[0])
                kinds[column].append("constant_binary")
            else:
                estimator = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
                estimator.fit(train_x, y)
                expected[held_column, column] = estimator.predict(held_x)
                if matrix.target in _BOUNDED_TARGETS:
                    expected[held_column, column] = np.clip(expected[held_column, column], 1e-9, 1 - 1e-9)
                kinds[column].append("ridge")
    if not np.isfinite(expected[matrix.active]).all():
        raise FloatingPointError("nuisance cross-fit produced nonfinite predictions")
    residuals = np.full_like(matrix.values, np.nan, dtype=np.float64)
    residuals[matrix.active] = matrix.values[matrix.active] - expected[matrix.active]
    return ConditionedErrors(
        expected=expected,
        residuals=residuals,
        folds=folds,
        design_columns=design.columns,
        model_kinds=tuple(tuple(item) for item in kinds),
        train_groups_by_fold=train_groups,
        source_identity=matrix.identity,
        method_names=matrix.method_names,
        active=matrix.active.copy(),
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-15 else float("nan")


def _soft_table(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.mean(left * right),
            np.mean(left * (1.0 - right)),
            np.mean((1.0 - left) * right),
            np.mean((1.0 - left) * (1.0 - right)),
        ],
        dtype=np.float64,
    )


def _mutual_information(table: np.ndarray) -> float:
    table = np.maximum(np.asarray(table, dtype=np.float64), 0.0)
    total = float(table.sum())
    if total <= 0:
        return float("nan")
    table /= total
    rows = np.array([table[0] + table[1], table[2] + table[3]])
    columns = np.array([table[0] + table[2], table[1] + table[3]])
    result = 0.0
    for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        if table[index] > 0 and rows[row] > 0 and columns[column] > 0:
            result += float(table[index] * np.log(table[index] / (rows[row] * columns[column])))
    return result


def _conditional_odds_ratio(
    left_expected: np.ndarray,
    right_expected: np.ndarray,
    left_residual: np.ndarray,
    right_residual: np.ndarray,
) -> float:
    """Residual-adjusted soft-table OR diagnostic.

    This is intentionally a descriptive P2 heuristic, not a conditional-logit
    coefficient: nuisance-independent cell mass is adjusted by mean residual
    covariance and a Haldane half-observation correction prevents infinities.
    Bootstrap intervals therefore describe this registered diagnostic only.
    """
    base = _soft_table(left_expected, right_expected)
    delta = float(np.mean(left_residual * right_residual))
    adjusted = base + np.array([delta, -delta, -delta, delta])
    # A half-observation correction makes separation finite while preserving
    # OR=1 for an exactly independent adjusted table.
    counts = np.maximum(adjusted * len(left_expected), 0.0) + 0.5
    return float((counts[0] * counts[3]) / (counts[1] * counts[2]))


def _mean_jaccard(
    left: Sequence[frozenset[int]], right: Sequence[frozenset[int]]
) -> float:
    values = []
    for a, b in zip(left, right):
        union = a | b
        values.append(1.0 if not union else len(a & b) / len(union))
    return float(np.mean(values)) if values else float("nan")


@dataclass(frozen=True)
class PairDiagnostics:
    left: str
    right: str
    target: ErrorTarget
    resolution: Resolution
    n: int
    source_groups: int
    left_successes: float
    left_failures: float
    right_successes: float
    right_failures: float
    conditional_phi: float
    conditional_odds_ratio: float
    joint_failure_ratio: float
    mutual_information: float
    unique_left_wins: float
    unique_right_wins: float
    score_spearman: float
    topk_jaccard: float
    peak_agreement: float
    exact_error_match: bool

    @property
    def conditional_residual_correlation(self) -> float:
        return self.conditional_phi

    def as_dict(self) -> dict:
        result = {
            key: (value.value if isinstance(value, Enum) else value)
            for key, value in self.__dict__.items()
        }
        result["conditional_residual_correlation"] = self.conditional_phi
        return result


def pair_diagnostics(
    matrix: ErrorMatrix,
    left: int | str,
    right: int | str,
    *,
    conditioned: ConditionedErrors | None = None,
) -> PairDiagnostics:
    if conditioned is None:
        conditioned = cross_fit_nuisance(matrix)
    conditioned.validate(matrix)
    index = lambda value: matrix.method_names.index(value) if isinstance(value, str) else int(value)
    a, b = index(left), index(right)
    if a == b or not (0 <= a < matrix.shape[1] and 0 <= b < matrix.shape[1]):
        raise ValueError("pair requires two distinct valid method columns")
    active = matrix.active[:, a] & matrix.active[:, b]
    bounded = matrix.target in _BOUNDED_TARGETS
    if not active.any():
        support_value = 0.0 if bounded else float("nan")
        return PairDiagnostics(
            left=matrix.method_names[a],
            right=matrix.method_names[b],
            target=matrix.target,
            resolution=matrix.resolution,
            n=0,
            source_groups=0,
            left_successes=support_value,
            left_failures=support_value,
            right_successes=support_value,
            right_failures=support_value,
            conditional_phi=float("nan"),
            conditional_odds_ratio=float("nan"),
            joint_failure_ratio=float("nan"),
            mutual_information=float("nan"),
            unique_left_wins=support_value,
            unique_right_wins=support_value,
            score_spearman=float("nan"),
            topk_jaccard=float("nan"),
            peak_agreement=float("nan"),
            exact_error_match=bool(np.array_equal(matrix.active[:, a], matrix.active[:, b])),
        )
    y_a, y_b = matrix.values[active, a], matrix.values[active, b]
    e_a, e_b = conditioned.expected[active, a], conditioned.expected[active, b]
    r_a, r_b = conditioned.residuals[active, a], conditioned.residuals[active, b]
    expected_joint = float(np.mean(e_a * e_b))
    observed_joint = float(np.mean(y_a * y_b))
    score_spearman = float("nan")
    if matrix.scores is not None:
        score_rows = active & np.isfinite(matrix.scores[:, a]) & np.isfinite(matrix.scores[:, b])
        if score_rows.any():
            score_spearman = _correlation(
                rankdata(matrix.scores[score_rows, a]),
                rankdata(matrix.scores[score_rows, b]),
            )
    jaccard = float("nan")
    if matrix.topk_sets is not None:
        rows = np.asarray([
            row for row in np.flatnonzero(active)
            if matrix.topk_sets[a][row] and matrix.topk_sets[b][row]
        ], dtype=np.int64)
        if len(rows):
            jaccard = _mean_jaccard(
                tuple(matrix.topk_sets[a][row] for row in rows),
                tuple(matrix.topk_sets[b][row] for row in rows),
            )
    peak_agreement = float("nan")
    if matrix.peaks is not None:
        valid = active & (matrix.peaks[:, a] >= 0) & (matrix.peaks[:, b] >= 0)
        if valid.any():
            peak_agreement = float(np.mean(matrix.peaks[valid, a] == matrix.peaks[valid, b]))
    return PairDiagnostics(
        left=matrix.method_names[a],
        right=matrix.method_names[b],
        target=matrix.target,
        resolution=matrix.resolution,
        n=int(active.sum()),
        source_groups=len(np.unique(matrix.metadata.groups[active])),
        left_successes=float(np.sum(1.0 - y_a)) if bounded else float("nan"),
        left_failures=float(np.sum(y_a)) if bounded else float("nan"),
        right_successes=float(np.sum(1.0 - y_b)) if bounded else float("nan"),
        right_failures=float(np.sum(y_b)) if bounded else float("nan"),
        conditional_phi=_correlation(r_a, r_b),
        conditional_odds_ratio=(
            _conditional_odds_ratio(e_a, e_b, r_a, r_b)
            if bounded
            else float("nan")
        ),
        joint_failure_ratio=(observed_joint / expected_joint if bounded and expected_joint > 1e-15 else float("nan")),
        mutual_information=(
            _mutual_information(_soft_table(y_a, y_b))
            if bounded
            else float("nan")
        ),
        unique_left_wins=float(np.sum((1.0 - y_a) * y_b)) if bounded else float("nan"),
        unique_right_wins=float(np.sum(y_a * (1.0 - y_b))) if bounded else float("nan"),
        score_spearman=score_spearman,
        topk_jaccard=jaccard,
        peak_agreement=peak_agreement,
        exact_error_match=bool(
            np.array_equal(matrix.active[:, a], matrix.active[:, b])
            and np.array_equal(y_a, y_b)
        ),
    )


def all_pair_diagnostics(
    matrix: ErrorMatrix, *, conditioned: ConditionedErrors | None = None
) -> dict[tuple[str, str], PairDiagnostics]:
    if conditioned is None:
        conditioned = cross_fit_nuisance(matrix)
    return {
        (matrix.method_names[a], matrix.method_names[b]): pair_diagnostics(
            matrix, a, b, conditioned=conditioned
        )
        for a in range(matrix.shape[1])
        for b in range(a + 1, matrix.shape[1])
    }


WithinDrawCallback = Callable[[ErrorMatrix, int], ErrorMatrix | np.ndarray]


def _group_rows(groups: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return lexicographically ordered groups with one linear grouping plan."""
    values = _one_dimensional(groups, "groups").astype(str)
    unique, inverse = np.unique(values, return_inverse=True)
    if not len(unique):
        return unique, ()
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique))
    boundaries = np.r_[0, np.cumsum(counts)]
    return unique, tuple(
        order[int(boundaries[index]):int(boundaries[index + 1])]
        for index in range(len(unique))
    )


def _bootstrap_indices(rows: tuple[np.ndarray, ...], rng: np.random.Generator) -> np.ndarray:
    chosen = rng.integers(0, len(rows), size=len(rows))
    return np.concatenate([rows[index] for index in chosen])


@dataclass(frozen=True)
class BootstrapSamples:
    point: np.ndarray
    samples: np.ndarray
    seed: int
    confidence: float
    requested_draws: int
    finite_counts: tuple[int, ...]
    fully_finite_draws: int


def source_group_bootstrap(
    matrix: ErrorMatrix,
    statistic: Callable[[ErrorMatrix], float | Sequence[float] | np.ndarray],
    *,
    draws: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    within_draw: WithinDrawCallback | None = None,
) -> BootstrapSamples:
    """Evaluate a statistic on common source-group bootstrap draws.

    ``within_draw`` runs after resampling and before the statistic.  Gate
    callers use it to recompute cell midranks, rank fusion, and the 0.33
    decision inside every draw rather than resampling frozen gate decisions.
    """
    if draws <= 0 or not (0 < confidence < 1):
        raise ValueError("bootstrap draws and confidence are invalid")
    point = np.atleast_1d(np.asarray(statistic(matrix), dtype=np.float64))
    if point.ndim != 1:
        raise ValueError("bootstrap statistic must return a scalar or vector")
    _, rows = _group_rows(matrix.metadata.groups)
    if len(rows) < 2:
        raise ValueError("source-group bootstrap requires at least two groups")
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(draws), len(point)), dtype=np.float64)
    for draw in range(int(draws)):
        sample = matrix.take(_bootstrap_indices(rows, rng))
        if within_draw is not None:
            recalibrated = within_draw(sample, draw)
            sample = sample.with_values(recalibrated) if isinstance(recalibrated, np.ndarray) else recalibrated
            if not isinstance(sample, ErrorMatrix):
                raise TypeError("within-draw callback must return ErrorMatrix or an error array")
            if sample.target != matrix.target or sample.resolution != matrix.resolution or sample.method_names != matrix.method_names:
                raise ValueError("within-draw callback changed the error contract")
        value = np.atleast_1d(np.asarray(statistic(sample), dtype=np.float64))
        if value.shape != point.shape:
            raise ValueError("bootstrap statistic changed shape")
        samples[draw] = value
    finite = np.isfinite(samples)
    applicable = np.isfinite(point)
    return BootstrapSamples(
        point,
        samples,
        int(seed),
        float(confidence),
        int(draws),
        tuple(int(value) for value in finite.sum(axis=0)),
        int(np.all(finite[:, applicable], axis=1).sum()) if applicable.any() else 0,
    )


def simultaneous_intervals(
    point: Sequence[float] | np.ndarray,
    samples: np.ndarray,
    *,
    confidence: float = 0.95,
    min_finite_fraction: float = 0.95,
) -> np.ndarray:
    """Bonferroni simultaneous percentile intervals for a statistic vector."""
    point = np.atleast_1d(np.asarray(point, dtype=np.float64))
    samples = np.asarray(samples, dtype=np.float64)
    if (
        samples.ndim != 2
        or samples.shape[1] != len(point)
        or len(samples) == 0
        or not (0 < confidence < 1)
        or not (0 < min_finite_fraction <= 1)
    ):
        raise ValueError("invalid simultaneous-interval inputs")
    result = np.full((len(point), 2), np.nan, dtype=np.float64)
    usable = np.isfinite(point) & np.any(np.isfinite(samples), axis=0)
    family = max(1, int(usable.sum()))
    tail = (1.0 - confidence) / (2.0 * family)
    for column in np.flatnonzero(usable):
        finite = samples[np.isfinite(samples[:, column]), column]
        required = int(math.ceil(min_finite_fraction * len(samples)))
        if len(finite) < required:
            raise ValueError(
                f"bootstrap statistic {column} has only {len(finite)}/{len(samples)} finite draws; "
                f"requires {required}"
            )
        if len(finite):
            result[column] = np.quantile(finite, [tail, 1.0 - tail])
    return result


@dataclass(frozen=True)
class PairInterval:
    phi_low: float
    phi_high: float
    odds_ratio_low: float
    odds_ratio_high: float


@dataclass(frozen=True)
class PairBootstrapResult:
    diagnostics: Mapping[tuple[str, str], PairDiagnostics]
    intervals: Mapping[tuple[str, str], PairInterval]
    statuses: Mapping[tuple[str, str], DependenceStatus]
    draws: int
    seed: int
    finite_counts: tuple[int, ...]
    fully_finite_draws: int
    common_draws: bool = True


def _pair_vector(
    matrix: ErrorMatrix,
    pairs: Sequence[tuple[int, int]],
    conditioned: ConditionedErrors,
) -> np.ndarray:
    diagnostics = [pair_diagnostics(matrix, a, b, conditioned=conditioned) for a, b in pairs]
    phi = [item.conditional_phi for item in diagnostics]
    log_odds = [
        math.log(item.conditional_odds_ratio)
        if np.isfinite(item.conditional_odds_ratio) and item.conditional_odds_ratio > 0
        else float("nan")
        for item in diagnostics
    ]
    return np.asarray(phi + log_odds, dtype=np.float64)


def _pair_bootstrap_from_group_sufficient_statistics(
    matrix: ErrorMatrix,
    pairs: Sequence[tuple[int, int]],
    conditioned: ConditionedErrors,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap phi/log-OR without copying token-scale matrices per draw."""
    groups, inverse = np.unique(matrix.metadata.groups, return_inverse=True)
    if len(groups) < 2:
        raise ValueError("source-group bootstrap requires at least two groups")
    # One common multinomial source-group draw plan for every pair.
    rng = np.random.default_rng(int(seed))
    group_counts = np.empty((int(draws), len(groups)), dtype=np.float32)
    for draw in range(int(draws)):
        sampled = rng.integers(0, len(groups), size=len(groups))
        group_counts[draw] = np.bincount(sampled, minlength=len(groups))
    result = np.full((int(draws), 2 * len(pairs)), np.nan, dtype=np.float64)
    # Process pair batches so candidate-rich matrices do not allocate one
    # enormous draws x pairs x statistic tensor.
    for batch_start in range(0, len(pairs), 64):
        batch = pairs[batch_start:batch_start + 64]
        sufficient = np.zeros((len(groups), len(batch) * 10), dtype=np.float64)
        odds_applicable = np.zeros(len(batch), dtype=bool)
        for local, (left, right) in enumerate(batch):
            active = matrix.active[:, left] & matrix.active[:, right]
            rows = np.flatnonzero(active)
            if not len(rows):
                continue
            group = inverse[rows]
            ra = conditioned.residuals[rows, left]
            rb = conditioned.residuals[rows, right]
            ea = conditioned.expected[rows, left]
            eb = conditioned.expected[rows, right]
            values = (
                np.ones(len(rows)), ra, rb, ra * ra, rb * rb, ra * rb,
                (1.0 - ea) * (1.0 - eb), (1.0 - ea) * eb,
                ea * (1.0 - eb), ea * eb,
            )
            for statistic, value in enumerate(values):
                sufficient[:, local * 10 + statistic] = np.bincount(
                    group, weights=value, minlength=len(groups),
                )
            odds_applicable[local] = np.isfinite(
                pair_diagnostics(matrix, left, right, conditioned=conditioned).conditional_odds_ratio
            )
        totals = (group_counts @ sufficient).reshape(int(draws), len(batch), 10)
        n = totals[:, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_a = totals[:, :, 1] / n
            mean_b = totals[:, :, 2] / n
            covariance = totals[:, :, 5] / n - mean_a * mean_b
            variance_a = totals[:, :, 3] / n - mean_a * mean_a
            variance_b = totals[:, :, 4] / n - mean_b * mean_b
            phi = covariance / np.sqrt(np.maximum(variance_a * variance_b, 0.0))
        phi[(n <= 0) | (variance_a <= 1e-15) | (variance_b <= 1e-15)] = np.nan
        result[:, batch_start:batch_start + len(batch)] = phi
        base = np.divide(
            totals[:, :, 6:10], n[:, :, None],
            out=np.full_like(totals[:, :, 6:10], np.nan), where=n[:, :, None] > 0,
        )
        delta = np.divide(
            totals[:, :, 5], n, out=np.full_like(n, np.nan), where=n > 0,
        )
        adjusted = base + delta[:, :, None] * np.asarray((1.0, -1.0, -1.0, 1.0))
        corrected = np.maximum(adjusted * n[:, :, None], 0.0) + 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            log_or = np.log(
                (corrected[:, :, 0] * corrected[:, :, 3])
                / (corrected[:, :, 1] * corrected[:, :, 2])
            )
        log_or[:, ~odds_applicable] = np.nan
        result[:, len(pairs) + batch_start:len(pairs) + batch_start + len(batch)] = log_or
    return result


def bootstrap_pair_diagnostics(
    matrix: ErrorMatrix,
    *,
    conditioned: ConditionedErrors | None = None,
    draws: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    within_draw: WithinDrawCallback | None = None,
    min_successes: float = 50,
    min_failures: float = 50,
    min_groups: int = 20,
) -> PairBootstrapResult:
    """Bootstrap all pairs with identical source-group draws.

    Nuisance fits are frozen before ordinary diagnostic bootstraps.  If a gate
    callback changes decisions inside a draw, nuisance expectations are
    cross-fitted again on that recalibrated draw.
    """
    if draws <= 0 or not (0 < confidence < 1):
        raise ValueError("bootstrap draws and confidence are invalid")
    conditioned = cross_fit_nuisance(matrix) if conditioned is None else conditioned
    pairs = [(a, b) for a in range(matrix.shape[1]) for b in range(a + 1, matrix.shape[1])]
    if not pairs:
        raise ValueError("pair bootstrap requires at least two methods")
    point_diagnostics = {
        (matrix.method_names[a], matrix.method_names[b]): pair_diagnostics(
            matrix, a, b, conditioned=conditioned
        )
        for a, b in pairs
    }
    point = _pair_vector(matrix, pairs, conditioned)
    if within_draw is None:
        samples = _pair_bootstrap_from_group_sufficient_statistics(
            matrix, pairs, conditioned, draws=int(draws), seed=int(seed),
        )
    else:
        _, rows = _group_rows(matrix.metadata.groups)
        if len(rows) < 2:
            raise ValueError("source-group bootstrap requires at least two groups")
        rng = np.random.default_rng(int(seed))
        samples = np.empty((int(draws), len(point)), dtype=np.float64)
        for draw in range(int(draws)):
            indices = _bootstrap_indices(rows, rng)
            sample = matrix.take(indices)
            recalibrated = within_draw(sample, draw)
            sample = sample.with_values(recalibrated) if isinstance(recalibrated, np.ndarray) else recalibrated
            if not isinstance(sample, ErrorMatrix):
                raise TypeError("within-draw callback must return ErrorMatrix or an error array")
            if sample.target != matrix.target or sample.resolution != matrix.resolution or sample.method_names != matrix.method_names:
                raise ValueError("within-draw callback changed the error contract")
            sample_conditioned = cross_fit_nuisance(sample, seed=seed)
            samples[draw] = _pair_vector(sample, pairs, sample_conditioned)
    intervals_array = simultaneous_intervals(point, samples, confidence=confidence)
    count = len(pairs)
    intervals: dict[tuple[str, str], PairInterval] = {}
    statuses: dict[tuple[str, str], DependenceStatus] = {}
    for position, (a, b) in enumerate(pairs):
        key = (matrix.method_names[a], matrix.method_names[b])
        log_low, log_high = intervals_array[count + position]
        interval = PairInterval(
            phi_low=float(intervals_array[position, 0]),
            phi_high=float(intervals_array[position, 1]),
            odds_ratio_low=float(np.exp(log_low)) if np.isfinite(log_low) else float("nan"),
            odds_ratio_high=float(np.exp(log_high)) if np.isfinite(log_high) else float("nan"),
        )
        intervals[key] = interval
        statuses[key] = classify_pair(
            point_diagnostics[key],
            interval,
            min_successes=min_successes,
            min_failures=min_failures,
            min_groups=min_groups,
        )
    finite = np.isfinite(samples)
    applicable = np.isfinite(point)
    return PairBootstrapResult(
        point_diagnostics,
        intervals,
        statuses,
        int(draws),
        int(seed),
        tuple(int(value) for value in finite.sum(axis=0)),
        int(np.all(finite[:, applicable], axis=1).sum()) if applicable.any() else 0,
    )


def classify_pair(
    diagnostics: PairDiagnostics,
    interval: PairInterval,
    *,
    min_successes: float = 50,
    min_failures: float = 50,
    min_groups: int = 20,
    phi_limit: float = 0.20,
    odds_ratio_bounds: tuple[float, float] = (0.67, 1.50),
    held_fold_unique_successes: float | None = None,
    oof_fusion_improved: bool | None = None,
) -> DependenceStatus:
    if diagnostics.target == ErrorTarget.PREDICTOR_RESIDUAL:
        correlation_finite = np.isfinite([interval.phi_low, interval.phi_high]).all()
        if diagnostics.source_groups < min_groups or not correlation_finite:
            return DependenceStatus.UNRESOLVED
        if interval.phi_low >= -float(phi_limit) and interval.phi_high <= float(phi_limit):
            return DependenceStatus.INDEPENDENCE_COMPATIBLE
        if diagnostics.exact_error_match:
            return DependenceStatus.REDUNDANT
        # Continuous residuals have no principled success/failure count or
        # odds ratio.  Complementarity requires a separate downstream OOF
        # utility result and is therefore not inferred from residual signs.
        return DependenceStatus.UNRESOLVED
    support = (
        diagnostics.source_groups >= min_groups
        and min(diagnostics.left_successes, diagnostics.right_successes) >= min_successes
        and min(diagnostics.left_failures, diagnostics.right_failures) >= min_failures
    )
    limits = np.asarray(
        [interval.phi_low, interval.phi_high, interval.odds_ratio_low, interval.odds_ratio_high],
        dtype=np.float64,
    )
    if not support or not np.isfinite(limits).all():
        return DependenceStatus.UNRESOLVED
    compatible = (
        interval.phi_low >= -float(phi_limit)
        and interval.phi_high <= float(phi_limit)
        and interval.odds_ratio_low >= float(odds_ratio_bounds[0])
        and interval.odds_ratio_high <= float(odds_ratio_bounds[1])
    )
    if compatible:
        return DependenceStatus.INDEPENDENCE_COMPATIBLE
    unique = (
        max(diagnostics.unique_left_wins, diagnostics.unique_right_wins)
        if held_fold_unique_successes is None
        else float(held_fold_unique_successes)
    )
    if oof_fusion_improved is True and unique > 0:
        return DependenceStatus.DEPENDENT_COMPLEMENTARY
    if diagnostics.exact_error_match or (
        diagnostics.unique_left_wins + diagnostics.unique_right_wins < 1.0
    ):
        return DependenceStatus.REDUNDANT
    return DependenceStatus.UNRESOLVED


def _group_point(residuals: np.ndarray) -> tuple[float, float]:
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[1] < 2:
        raise ValueError("group diagnostics require at least two methods")
    if len(residuals) < 2:
        return float("nan"), 0.0
    centered = residuals - residuals.mean(axis=0)
    scale = centered.std(axis=0)
    if np.any(scale <= 1e-12):
        return float("nan"), 0.0
    z = centered / scale
    correlation = (z.T @ z) / len(z)
    off_diagonal = np.abs(correlation[np.triu_indices(len(scale), 1)])
    eigenvalues = np.maximum(np.linalg.eigvalsh((correlation + correlation.T) / 2), 0.0)
    denominator = float(eigenvalues @ eigenvalues)
    effective_rank = float(eigenvalues.sum() ** 2 / denominator) if denominator > 1e-15 else 0.0
    return float(off_diagonal.max()), effective_rank


@dataclass(frozen=True)
class GroupDiagnostics:
    members: tuple[str, ...]
    n: int
    source_groups: int
    max_conditional_residual_correlation: float
    max_correlation_ci: tuple[float, float]
    effective_rank: float
    effective_rank_ci: tuple[float, float]
    required_effective_rank: float
    status: DependenceStatus
    bootstrap_draws: int
    finite_draws: tuple[int, int]


def group_diagnostics(
    matrix: ErrorMatrix,
    members: Sequence[int | str],
    *,
    conditioned: ConditionedErrors | None = None,
    draws: int = 1_000,
    seed: int = 0,
    confidence: float = 0.95,
    min_successes: float = 50,
    min_failures: float = 50,
    min_groups: int = 20,
) -> GroupDiagnostics:
    if draws <= 0 or not (0 < confidence < 1):
        raise ValueError("bootstrap draws and confidence are invalid")
    indices = tuple(
        matrix.method_names.index(member) if isinstance(member, str) else int(member)
        for member in members
    )
    if len(indices) < 2 or len(indices) > 6 or len(set(indices)) != len(indices):
        raise ValueError("fusion groups must contain two to six distinct methods")
    if any(index < 0 or index >= matrix.shape[1] for index in indices):
        raise ValueError("unknown group member")
    conditioned = cross_fit_nuisance(matrix) if conditioned is None else conditioned
    conditioned.validate(matrix)
    active = np.all(matrix.active[:, indices], axis=1)
    residuals = conditioned.residuals[np.ix_(active, indices)]
    point = np.asarray(_group_point(residuals))
    active_groups = matrix.metadata.groups[active]
    _, rows = _group_rows(active_groups)
    if len(rows) < 2:
        intervals = np.full((2, 2), np.nan)
        samples = np.full((int(draws), 2), np.nan)
    else:
        rng = np.random.default_rng(int(seed))
        samples = np.empty((int(draws), 2), dtype=np.float64)
        for draw in range(int(draws)):
            samples[draw] = _group_point(residuals[_bootstrap_indices(rows, rng)])
        intervals = simultaneous_intervals(point, samples, confidence=confidence)
    bounded_support = True
    if matrix.target in _BOUNDED_TARGETS:
        values = matrix.values[np.ix_(active, indices)]
        bounded_support = bool(
            np.all(values.sum(axis=0) >= min_failures)
            and np.all((1.0 - values).sum(axis=0) >= min_successes)
        )
    supported = len(rows) >= min_groups and bounded_support and np.isfinite(intervals).all()
    required_rank = 0.70 * len(indices)
    passed = supported and intervals[0, 1] <= 0.25 and intervals[1, 0] >= required_rank
    names = tuple(matrix.method_names[index] for index in indices)
    return GroupDiagnostics(
        members=names,
        n=int(active.sum()),
        source_groups=len(rows),
        max_conditional_residual_correlation=float(point[0]),
        max_correlation_ci=(float(intervals[0, 0]), float(intervals[0, 1])),
        effective_rank=float(point[1]),
        effective_rank_ci=(float(intervals[1, 0]), float(intervals[1, 1])),
        required_effective_rank=required_rank,
        status=(
            DependenceStatus.INDEPENDENCE_COMPATIBLE
            if passed
            else DependenceStatus.UNRESOLVED
        ),
        bootstrap_draws=int(draws),
        finite_draws=tuple(int(value) for value in np.isfinite(samples).sum(axis=0)),
    )


def build_compatibility_graph(
    method_names: Sequence[str],
    pair_statuses: Mapping[tuple[str, str], DependenceStatus | str],
) -> dict[str, frozenset[str]]:
    """Build an undirected graph containing only independence-compatible edges."""
    names = tuple(str(name) for name in method_names)
    if len(set(names)) != len(names):
        raise ValueError("graph method names must be unique")
    graph: dict[str, set[str]] = {name: set() for name in names}
    known = set(names)
    for (left, right), raw_status in pair_statuses.items():
        if left not in known or right not in known or left == right:
            raise ValueError("pair status names do not match the graph")
        status = DependenceStatus(raw_status)
        if status == DependenceStatus.INDEPENDENCE_COMPATIBLE:
            graph[left].add(right)
            graph[right].add(left)
    return {name: frozenset(sorted(neighbours)) for name, neighbours in graph.items()}


def enumerate_cliques(
    graph: Mapping[str, Iterable[str]],
    *,
    min_size: int = 2,
    max_size: int = 6,
    max_results: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Deterministically enumerate every clique within the registered size cap."""
    if min_size < 1 or max_size < min_size or max_size > 6:
        raise ValueError("clique sizes must satisfy 1 <= min <= max <= 6")
    nodes = tuple(sorted(str(node) for node in graph))
    adjacency = {node: set(str(value) for value in graph[node]) for node in nodes}
    if any(node in adjacency[node] or not adjacency[node].issubset(nodes) for node in nodes):
        raise ValueError("compatibility graph has invalid neighbours")
    for node in nodes:
        if any(node not in adjacency[neighbour] for neighbour in adjacency[node]):
            raise ValueError("compatibility graph must be symmetric")
    output: list[tuple[str, ...]] = []

    def extend(prefix: tuple[str, ...], candidates: tuple[str, ...]) -> None:
        for position, node in enumerate(candidates):
            clique = prefix + (node,)
            if len(clique) >= min_size:
                output.append(clique)
                if max_results is not None and len(output) > max_results:
                    raise RuntimeError("clique result limit exceeded")
            if len(clique) < max_size:
                remaining = tuple(
                    candidate
                    for candidate in candidates[position + 1 :]
                    if candidate in adjacency[node]
                )
                extend(clique, remaining)

    extend((), nodes)
    return tuple(output)


# ---------------------------------------------------------------------------
# Atlas v1 filesystem backends


class AtlasBackendError(RuntimeError):
    """A frozen Atlas artifact is absent, incomplete, or internally inconsistent."""


def _enumerate_cliques_complete(
    graph: Mapping[str, Iterable[str]],
    *,
    min_size: int = 2,
    max_size: int = 6,
    max_results: int = 100_000,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate every requested clique or fail before returning any subset."""
    if int(max_results) <= 0:
        raise ValueError("max_results must be positive")
    try:
        return enumerate_cliques(
            graph, min_size=min_size, max_size=max_size,
            max_results=int(max_results),
        )
    except RuntimeError as error:
        raise AtlasBackendError(
            f"clique set would exceed non-truncating limit {int(max_results)}"
        ) from error


def clique_diagnostics_preflight(
    clique_count: int,
    draws: int,
    group_count: int,
) -> dict[str, int | bool | str]:
    """Describe exact clique-bootstrap resource geometry before it runs.

    The streamed implementation spools one two-statistic sample matrix at a
    time.  It never changes the requested clique set; callers must either run
    every enumerated clique or fail closed during complete enumeration.
    """
    clique_count = int(clique_count)
    draws = int(draws)
    group_count = int(group_count)
    if clique_count < 0 or draws <= 0 or group_count <= 0:
        raise ValueError("invalid clique diagnostic preflight dimensions")
    per_clique = draws * 2 * np.dtype(np.float64).itemsize
    return {
        "schema": "fusion-independence-atlas-v1/clique-preflight-v1",
        "non_truncating": True,
        "cliques": clique_count,
        "draws": draws,
        "groups": group_count,
        "legacy_materialized_sample_bytes": clique_count * per_clique,
        "streamed_peak_sample_bytes": per_clique if clique_count else 0,
        "streamed_spool_bytes": clique_count * per_clique,
        "bootstrap_multiplicity_bytes": (
            draws * group_count * np.dtype(np.int32).itemsize
        ),
    }


def _json_ready(value):
    """Convert NumPy/dataclass values to deterministic strict-JSON values."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: _json_ready(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    os.replace(temporary, path)


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(root: Path, manifest: Mapping, key: str, candidates: Sequence[Path]) -> Path:
    raw = manifest.get(key)
    if raw is not None:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
        raise AtlasBackendError(f"manifest {key} does not exist: {path}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise AtlasBackendError(f"missing required artifact {key}: " + ", ".join(map(str, candidates)))


@dataclass(frozen=True)
class _AtlasInputs:
    extract_root: Path
    predictor_root: Path
    reconciliation_root: Path
    contract_root: Path
    bundle_root: Path
    metadata: tuple[Mapping, ...]
    joined_records: tuple[Mapping, ...]
    joined: Mapping[str, np.ndarray]
    registry: Mapping
    readout_names: tuple[str, ...]
    signal_specs: Mapping[str, Mapping]
    aliases: Mapping[str, str]

    @property
    def answer_count(self) -> int:
        return len(self.metadata)

    @property
    def offsets(self) -> np.ndarray:
        return np.asarray(self.joined["offsets"], dtype=np.int64)


def _load_atlas_inputs(
    extract_root: Path,
    predictor_root: Path,
    reconciliation_root: Path,
    contract_root: Path,
) -> _AtlasInputs:
    extract_root = Path(extract_root)
    predictor_root = Path(predictor_root)
    reconciliation_root = Path(reconciliation_root)
    contract_root = Path(contract_root)
    manifest_path = extract_root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise AtlasBackendError(f"missing extraction manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    answers = int(manifest.get("answers", -1))
    if answers <= 0:
        raise AtlasBackendError("extraction manifest has no positive answer count")
    status_path = extract_root / "STATUS.json"
    if status_path.is_file() and _read_json(status_path).get("status") != "COMPLETE":
        raise AtlasBackendError("atomic extraction is not COMPLETE")
    completion_path = extract_root / "completion.npy"
    if completion_path.is_file():
        completion = np.load(completion_path, mmap_mode="r", allow_pickle=False)
        if completion.shape != (answers,) or not bool(np.asarray(completion).all()):
            raise AtlasBackendError("atomic extraction completion bitmap is incomplete")
    answer_root = extract_root / "answers"
    answer_paths = tuple(answer_root / f"{index:05d}.npz" for index in range(answers))
    missing = [str(path) for path in answer_paths if not path.is_file()]
    if missing:
        raise AtlasBackendError(
            f"atomic extraction is incomplete: {len(missing)}/{answers} answer files missing; first={missing[0]}"
        )

    bundle_root = _resolve_artifact(
        extract_root,
        manifest,
        "bundle_root",
        (extract_root.parent / "bundle/data", extract_root.parent / "bundle"),
    )
    expected_bundle_freeze = manifest.get("bundle_freeze_sha256")
    if expected_bundle_freeze is not None:
        freeze_path = bundle_root / "FREEZE.json"
        if not freeze_path.is_file() or _file_sha256(freeze_path) != str(expected_bundle_freeze):
            raise AtlasBackendError("extraction manifest bundle FREEZE binding has drifted")
    metadata_path = bundle_root / "METADATA.json"
    if not metadata_path.is_file():
        raise AtlasBackendError(f"missing bundle metadata: {metadata_path}")
    metadata = tuple(_read_json(metadata_path))
    if len(metadata) != answers:
        raise AtlasBackendError("bundle metadata and extraction answer count disagree")
    required_metadata = {
        "uid", "cell", "group_id", "offset", "tokens", "step_start", "step_stop", "fold",
    }
    for index, row in enumerate(metadata):
        absent = required_metadata - set(row)
        if absent:
            raise AtlasBackendError(f"bundle metadata row {index} misses {sorted(absent)}")
        if int(row["step_stop"]) <= int(row["step_start"]) or int(row["tokens"]) <= 0:
            raise AtlasBackendError(f"invalid bundle metadata extents at answer {index}")

    joined_json = _resolve_artifact(
        contract_root,
        manifest,
        "joined_json",
        (
            contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.json",
            contract_root / "JOINED.json",
        ),
    )
    joined_npz = _resolve_artifact(
        contract_root,
        manifest,
        "joined_npz",
        (
            contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.npz",
            contract_root / "JOINED.npz",
        ),
    )
    joined_payload = _read_json(joined_json)
    records = tuple(joined_payload.get("records", ()))
    if len(records) != answers:
        raise AtlasBackendError("JOINED records and extraction answer count disagree")
    for index, (meta, record) in enumerate(zip(metadata, records)):
        if str(meta["uid"]) != str(record.get("uid")):
            raise AtlasBackendError(f"JOINED/extraction UID drift at answer {index}")
    with np.load(joined_npz, allow_pickle=False) as saved:
        required = {"offsets", "labels", "target"}
        if not required.issubset(saved.files):
            raise AtlasBackendError(f"JOINED arrays miss {sorted(required - set(saved.files))}")
        joined = {name: np.asarray(saved[name]).copy() for name in saved.files}
    expected_offsets = np.asarray(
        [0] + [int(row["step_stop"]) for row in metadata], dtype=np.int64,
    )
    if joined["offsets"].shape != (answers + 1,) or not np.array_equal(joined["offsets"], expected_offsets):
        raise AtlasBackendError("JOINED step offsets drift from the primitive bundle")
    if joined["target"].shape != (answers,) or joined["labels"].shape != (int(expected_offsets[-1]),):
        raise AtlasBackendError("JOINED annotation arrays have incompatible shapes")

    registry_path = _resolve_artifact(
        extract_root,
        manifest,
        "registry_path",
        (extract_root.parent / "registry/REGISTRY.json",),
    )
    registry = _read_json(registry_path)
    signal_specs = {str(row["name"]): row for row in registry.get("signals", ())}
    aliases = {str(key): str(value) for key, value in registry.get("aliases", {}).items()}
    try:
        from spectral_utils.fusion_signal_registry import READOUT_NAMES
        default_readouts = tuple(READOUT_NAMES)
    except Exception:  # pragma: no cover - only an installation integrity fallback
        default_readouts = ()
    manifest_readouts = tuple(str(value) for value in manifest.get("readout_names", ()))
    readout_names = manifest_readouts or default_readouts
    if not readout_names:
        raise AtlasBackendError("readout names are absent from extraction and registry runtime")

    freeze_path = predictor_root / "FREEZE.json"
    if not freeze_path.is_file():
        raise AtlasBackendError(
            "predictor outputs are incomplete: FREEZE.json is required; singleton OOF or partial fits are not accepted"
        )
    freeze = _read_json(freeze_path)
    if freeze.get("labels_used") is not False or len(freeze.get("targets", ())) != 4:
        raise AtlasBackendError("predictor freeze violates the four primitive label-free target contract")
    predictor_manifest_path = predictor_root / "MANIFEST.json"
    if predictor_manifest_path.is_file():
        predictor_manifest = _read_json(predictor_manifest_path)
        predictor_bundle = Path(predictor_manifest.get("bundle_root", bundle_root))
        if not predictor_bundle.is_absolute():
            predictor_bundle = predictor_root / predictor_bundle
        if predictor_bundle.resolve() != bundle_root.resolve():
            raise AtlasBackendError("predictor and extraction manifests bind different bundles")
        predictor_freeze = predictor_manifest.get("bundle_freeze_sha256")
        if expected_bundle_freeze is not None and str(predictor_freeze) != str(expected_bundle_freeze):
            raise AtlasBackendError("predictor and extraction bundle FREEZE hashes disagree")
    ledger_path = reconciliation_root / "LEDGER.json"
    if not ledger_path.is_file():
        raise AtlasBackendError("historical reconciliation ledger is missing")

    return _AtlasInputs(
        extract_root=extract_root,
        predictor_root=predictor_root,
        reconciliation_root=reconciliation_root,
        contract_root=contract_root,
        bundle_root=bundle_root,
        metadata=metadata,
        joined_records=records,
        joined=joined,
        registry=registry,
        readout_names=tuple(readout_names),
        signal_specs=signal_specs,
        aliases=aliases,
    )


@dataclass(frozen=True)
class _CandidateDefinition:
    name: str
    point: str
    signal: str
    family: str
    access_scope: str
    readout: str | None
    decoder: str | None
    score_key: str
    active_key: str | None
    readout_index: int | None
    decoder_index: int | None
    native: bool = False
    answer_only: bool = False
    gate_token: bool = False
    cost: float = 1.0


_DECODER_COLUMNS = ("argmax", "first_near_max_025", "persistent_q90_3")


def _family_for_signal(inputs: _AtlasInputs, signal: str) -> tuple[str, str]:
    canonical = inputs.aliases.get(signal, signal)
    spec = inputs.signal_specs.get(canonical, inputs.signal_specs.get(signal, {}))
    family = str(spec.get("provenance_family", signal.split(".", 1)[0]))
    access = str(spec.get("access_scope", "answer_only"))
    return family, access


def _candidate_definitions(inputs: _AtlasInputs) -> tuple[_CandidateDefinition, ...]:
    """Discover the packed extraction schema without opening any labels."""
    first = inputs.extract_root / "answers/00000.npz"
    output: list[_CandidateDefinition] = []
    with np.load(first, allow_pickle=False) as saved:
        keys = set(saved.files)
        for key in sorted(keys):
            if not key.startswith("step__") or not key.endswith("__readouts"):
                continue
            signal = key[len("step__") : -len("__readouts")]
            canonical = inputs.aliases.get(signal, signal)
            if canonical != signal and canonical in inputs.signal_specs:
                # Exact aliases remain visible in the registry ledger, but are
                # not expanded into thousands of duplicate experts.
                continue
            spec = inputs.signal_specs.get(canonical, {})
            if spec and spec.get("status") != "ELIGIBLE":
                continue
            family, access = _family_for_signal(inputs, signal)
            active_key = f"step__{signal}__readout_active"
            decision_key = f"decision__{signal}__readouts"
            if active_key not in keys or decision_key not in keys:
                raise AtlasBackendError(f"packed signal {signal} lacks readout masks or decisions")
            shape = saved[key].shape
            if len(shape) != 2 or shape[1] != len(inputs.readout_names):
                raise AtlasBackendError(f"packed readout axis drift for {signal}")
            decisions = saved[decision_key]
            if decisions.shape != (len(inputs.readout_names), len(_DECODER_COLUMNS)):
                raise AtlasBackendError(f"packed decoder axis drift for {signal}")
            for readout_index, readout in enumerate(inputs.readout_names):
                stem = f"{signal}::{readout}"
                # A token candidate denotes the *operation order* "fuse the
                # raw token streams, then apply this readout".  Its singleton
                # downstream error is identical to the corresponding step
                # expert, but multi-member evaluation in the nested backend
                # consumes the raw token store and therefore does not relabel
                # post-readout fusion as token fusion.
                output.append(_CandidateDefinition(
                    name=f"token::{stem}", point="token_pre_readout", signal=signal,
                    family=family, access_scope=access, readout=readout, decoder="argmax",
                    score_key=key, active_key=active_key, readout_index=readout_index,
                    decoder_index=0,
                ))
                output.append(_CandidateDefinition(
                    name=f"step::{stem}", point="step_post_readout", signal=signal,
                    family=family, access_scope=access, readout=readout, decoder="argmax",
                    score_key=key, active_key=active_key, readout_index=readout_index,
                    decoder_index=0,
                ))
                for decoder_index, decoder in enumerate(_DECODER_COLUMNS):
                    output.append(_CandidateDefinition(
                        name=f"decoder::{stem}::{decoder}", point="decoder", signal=signal,
                        family=family, access_scope=access, readout=readout, decoder=decoder,
                        score_key=key, active_key=active_key, readout_index=readout_index,
                        decoder_index=decoder_index,
                    ))
                if readout == "top5" and f"decision__{signal}__step_top5" in keys:
                    output.append(_CandidateDefinition(
                        name=f"decoder::{signal}::step_top5", point="decoder", signal=signal,
                        family=family, access_scope=access, readout="top5", decoder="step_top5",
                        score_key=key, active_key=active_key, readout_index=readout_index,
                        decoder_index=3,
                    ))
        for key in sorted(keys):
            if not key.startswith("step__") or not key.endswith("__native__values"):
                continue
            signal = key[len("step__") : -len("__native__values")]
            spec = inputs.signal_specs.get(inputs.aliases.get(signal, signal), {})
            if spec and spec.get("status") != "ELIGIBLE":
                continue
            family, access = _family_for_signal(inputs, signal)
            active_key = f"step__{signal}__native__active"
            if active_key not in keys:
                raise AtlasBackendError(f"native step signal {signal} lacks an active mask")
            for point in ("step_post_readout", "decoder", "answer_gate"):
                if point == "answer_gate":
                    # Gate candidates use the locked whole-answer bank below,
                    # not arbitrary step readouts.
                    continue
                output.append(_CandidateDefinition(
                    name=f"{point}::{signal}::native", point=point, signal=signal,
                    family=family, access_scope=access, readout="native", decoder="argmax",
                    score_key=key, active_key=active_key, readout_index=None,
                    decoder_index=0, native=True,
                ))

        # Background candidates are scoped to one primitive and one readout.
        # The residual is materialised exactly once as observed minus the
        # registered background; multi-member nested evaluation fuses
        # backgrounds in primitive units before this residual/readout.
        primitive_names = ("H0lim", "VE0", "VE0.75", "VE1")
        background_kinds = (
            "prefix_mean", "mean16", "no_reset", "bocpd_hazard_1_32",
            "source_excluded_ridge", "source_excluded_tcn",
        )
        for primitive in primitive_names:
            for kind in background_kinds:
                registry_name = f"background.{primitive}.{kind}"
                family, access = _family_for_signal(inputs, registry_name)
                for readout_index, readout in enumerate(inputs.readout_names):
                    output.append(_CandidateDefinition(
                        name=f"background::{primitive}::{kind}::{readout}",
                        point="background", signal=f"{primitive}:{kind}", family=family,
                        access_scope=access, readout=readout, decoder="argmax",
                        score_key=f"BACKGROUND:{primitive}:{kind}", active_key=None,
                        readout_index=readout_index, decoder_index=0,
                    ))
        gate_sources = {
            "entropy_native": "q15.H1_native",
            "q15_H0lim": "q15.H0lim",
            "q15_VE0": "q15.VE0",
            "q15_VE0.75": "q15.VE0.75",
            "q15_VE1": "q15.VE1",
            "q15_H1": "q15.H1_native",
            "q15_Hinf": "q15.Hinf",
            "raw_neglogp1": "direct_probability.rank_1_risk",
            "tail15_mass": "direct_probability.residual_tail_mass",
            "tail50_mass": "step395.tail50",
            "q15_raw4_mean": "DERIVED:q15_raw4_mean",
        }
        gate_readouts = (
            "token_mean", "token_top10", "token_q75", "token_q90", "token_q95",
            "rolling8_max", "region_q1_mean", "region_q2_mean", "region_q3_mean",
            "region_q4_mean", "position_slope", "mean_step_top10",
        )
        for gate_signal, source in gate_sources.items():
            source_key = source if source.startswith("DERIVED:") else f"token__{source}__values"
            active_key = None if source.startswith("DERIVED:") else f"token__{source}__active"
            if not source.startswith("DERIVED:") and (source_key not in keys or active_key not in keys):
                raise AtlasBackendError(f"locked gate source is absent from extraction: {gate_signal}/{source}")
            family, access = (
                ("renyi_entropy", "answer_only")
                if gate_signal == "q15_raw4_mean"
                else _family_for_signal(inputs, source)
            )
            for readout in gate_readouts:
                output.append(_CandidateDefinition(
                    name=f"answer_gate::{gate_signal}::{readout}", point="answer_gate",
                    signal=gate_signal, family=family, access_scope=access,
                    readout=readout, decoder=None, score_key=source_key,
                    active_key=active_key, readout_index=None, decoder_index=None,
                    gate_token=True,
                ))
        # The three historical digit answer summaries are separate gate
        # experts.  ``active_key`` names the opportunity *values* here (not
        # their all-active storage mask); _gate_token_values handles this
        # registered special case explicitly.
        if {
            "token__digit.disagreement__values",
            "token__digit.opportunity__values",
        }.issubset(keys):
            for summary in ("digit_count", "digit_rate", "digit_presence"):
                output.append(_CandidateDefinition(
                    name=f"answer_gate::{summary}", point="answer_gate",
                    signal=summary, family="digit", access_scope="answer_only",
                    readout=summary, decoder=None,
                    score_key="token__digit.disagreement__values",
                    active_key="token__digit.opportunity__values",
                    readout_index=None, decoder_index=None, gate_token=True,
                ))
        for key in sorted(keys):
            if not key.startswith("answer__"):
                continue
            signal = key[len("answer__") :]
            family, access = _family_for_signal(inputs, signal)
            output.append(_CandidateDefinition(
                name=f"answer_gate::{signal}", point="answer_gate", signal=signal,
                family=family, access_scope=access, readout=None, decoder=None,
                score_key=key, active_key=None, readout_index=None, decoder_index=None,
                answer_only=True,
            ))
        if "decision__earlier_ve_peak" in keys:
            output.append(_CandidateDefinition(
                name="decoder::earlier_ve_peak", point="decoder", signal="earlier_ve_peak",
                family="fixed_decoder", access_scope="answer_only", readout=None,
                decoder="earlier_ve_peak", score_key="decision__earlier_ve_peak",
                active_key=None, readout_index=None, decoder_index=4, answer_only=False,
            ))
    if not output:
        raise AtlasBackendError("atomic extraction exposes no eligible candidate definitions")
    names = [row.name for row in output]
    if len(names) != len(set(names)):
        raise AtlasBackendError("candidate discovery produced duplicate stable names")
    return tuple(output)


def _risk_pair_auc(step_labels: np.ndarray, scores: np.ndarray, active: np.ndarray) -> float:
    keep = active & np.isin(step_labels, (0, 1))
    labels = step_labels[keep]
    values = scores[keep]
    positive = values[labels == 1]
    negative = values[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    # Frozen JOINED labels use 1 as the positive class for the score AUC.
    # Therefore a larger score on label 1 is a correctly ordered pair.
    difference = positive[:, None] - negative[None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def _canonical_pb_score(
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    cells: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    cells = np.asarray(cells).astype(str)
    keep = np.char.startswith(cells, "pb_") if mask is None else np.asarray(mask, dtype=bool)
    values = []
    for cell in sorted(set(cells[keep])):
        local = keep & (cells == cell)
        clean = local & (target == -1)
        error = local & (target >= 0)
        if not clean.any() or not error.any():
            return float("nan")
        correct = valid & (prediction == target)
        clean_accuracy = float(np.mean(correct[clean]))
        error_accuracy = float(np.mean(correct[error]))
        denominator = clean_accuracy + error_accuracy
        values.append(2.0 * clean_accuracy * error_accuracy / denominator if denominator else 0.0)
    return float(np.mean(values)) if values else float("nan")


def _gate_token_values(answer, definition: _CandidateDefinition) -> tuple[np.ndarray, np.ndarray]:
    if definition.readout in {"digit_count", "digit_rate", "digit_presence"}:
        values = np.asarray(answer["token__digit.disagreement__values"], dtype=np.float64)
        opportunities = np.asarray(
            answer["token__digit.opportunity__values"], dtype=np.float64,
        )
        if values.shape != opportunities.shape:
            raise AtlasBackendError("digit disagreement/opportunity gate streams drift")
        return values, opportunities > 0
    if definition.score_key == "DERIVED:q15_raw4_mean":
        names = ("q15.H0lim", "q15.VE0", "q15.VE0.75", "q15.VE1")
        values = np.column_stack([
            np.asarray(answer[f"token__{name}__values"], dtype=np.float64) for name in names
        ]).mean(axis=1)
        active = np.logical_and.reduce([
            np.asarray(answer[f"token__{name}__active"], dtype=bool) for name in names
        ])
        return values, active
    return (
        np.asarray(answer[definition.score_key], dtype=np.float64),
        np.asarray(answer[definition.active_key], dtype=bool),
    )


def _gate_answer_readout(
    values: np.ndarray,
    active: np.ndarray,
    readout: str,
    spans: np.ndarray | None,
) -> float:
    if readout == "digit_count":
        if values.shape != active.shape or not np.isfinite(values).all():
            raise ValueError("digit count streams are malformed")
        return float(values.sum())
    if readout == "digit_rate":
        if values.shape != active.shape or not np.isfinite(values).all():
            raise ValueError("digit rate streams are malformed")
        opportunities = int(np.sum(active))
        return float(values.sum() / opportunities) if opportunities else 0.0
    if readout == "digit_presence":
        if values.shape != active.shape:
            raise ValueError("digit presence streams are malformed")
        return float(bool(np.any(active)))
    selected = np.asarray(values, dtype=np.float64)[np.asarray(active, dtype=bool)]
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("gate readout has no finite active tokens")
    if readout == "token_mean":
        return float(selected.mean())
    if readout == "token_top10":
        count = min(10, len(selected)); return float(np.partition(selected, len(selected) - count)[-count:].mean())
    if readout in {"token_q75", "token_q90", "token_q95"}:
        return float(np.quantile(selected, int(readout[-2:]) / 100.0))
    if readout == "rolling8_max":
        width = min(8, len(values)); kernel = np.ones(width) / width
        # Rolling windows crossing inactive tokens are excluded.
        numer = np.convolve(np.where(active, values, 0.0), kernel, mode="valid")
        denom = np.convolve(active.astype(float), kernel, mode="valid")
        usable = denom >= 1.0 - 1e-12
        return float(numer[usable].max()) if usable.any() else float(selected.mean())
    if readout.startswith("region_q"):
        quarter = int(readout[len("region_q")]) - 1
        position = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
        region = active & (position >= quarter / 4.0) & (position < (quarter + 1) / 4.0)
        if region.any():
            return float(values[region].mean())
        center = (quarter + 0.5) / 4.0
        live = np.flatnonzero(active)
        return float(values[live[np.argmin(np.abs(position[live] - center))]])
    if readout == "position_slope":
        if len(selected) < 2:
            return 0.0
        position = (np.flatnonzero(active) + 0.5) / len(values)
        return float(np.polyfit(position, selected, 1)[0])
    if readout == "mean_step_top10":
        if spans is None:
            raise AtlasBackendError("mean_step_top10 requires the bundle step spans")
        from spectral_utils.fusion_signal_registry import readout_steps
        score, available = readout_steps(values, spans, "top10", active_mask=active)
        if not available.any():
            raise ValueError("mean_step_top10 has no active step")
        return float(score[available].mean())
    raise ValueError(f"unknown gate readout {readout}")


def _digit025_scores(answer) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the frozen innovation5 + .25 digit step trajectory."""
    from spectral_utils.fusion_signal_registry import READOUT_NAMES
    top10 = tuple(READOUT_NAMES).index("top10")
    signals = (
        "q15.H0lim", "q15.VE0", "q15.VE0.75", "q15.VE1",
        "q15.H0lim.prefix_mean_innovation",
    )
    keys = [f"step__{name}__readouts" for name in signals]
    digit_key = "step__digit.disagreement__readouts"
    if any(key not in answer for key in keys) or digit_key not in answer:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)
    columns = [np.asarray(answer[key], dtype=np.float64)[:, top10] for key in keys]
    if not columns or any(column.shape != columns[0].shape for column in columns):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)
    base = np.mean(columns, axis=0)
    digit = np.asarray(answer[digit_key], dtype=np.float64)[:, top10]
    if not len(base) or digit.shape != base.shape or not np.isfinite(base).all() or not np.isfinite(digit).all():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)
    digit_scale = float(digit.std())
    digit_z = np.zeros_like(digit) if digit_scale <= 1e-12 else (digit - digit.mean()) / digit_scale
    fused = base + 0.25 * float(base.std()) * digit_z
    return fused, np.ones(len(fused), dtype=bool)


def _digit025_peak(answer) -> int:
    """Reconstruct the frozen innovation5 + .25 digit locator label-free."""
    fused, active = _digit025_scores(answer)
    return int(np.argmax(fused)) if active.any() else -1


def _digit025_within_values(inputs: "_AtlasInputs") -> np.ndarray:
    """Return the frozen digit025 per-answer PRMB AUC ledger.

    Keeping this answer-level ledger in ``EVALUATION.npz`` lets the nested
    search compare every locator with exactly the same incumbent and lets a
    gate candidate inherit the frozen locator's within-answer metric without
    reopening packed labels in the weight-fitting path.
    """
    labels = np.asarray(inputs.joined["labels"], dtype=np.int8)
    values = np.full(inputs.answer_count, np.nan, dtype=np.float64)
    for answer_index, row in enumerate(inputs.metadata):
        if not str(row["cell"]).startswith("prmbench_"):
            continue
        start, stop = map(int, inputs.offsets[answer_index:answer_index + 2])
        path = inputs.extract_root / "answers" / f"{answer_index:05d}.npz"
        with np.load(path, allow_pickle=False) as answer:
            score, active = _digit025_scores(answer)
        if score.shape != (stop - start,) or active.shape != score.shape:
            continue
        values[answer_index] = _risk_pair_auc(labels[start:stop], score, active)
    return values


_PRIMITIVE_NAMES = ("H0lim", "VE0", "VE0.75", "VE1")
_BACKGROUND_KINDS = (
    "prefix_mean", "mean16", "no_reset", "bocpd_hazard_1_32",
    "source_excluded_ridge", "source_excluded_tcn",
)


@dataclass(frozen=True)
class _BackgroundArrays:
    observed: np.ndarray
    fixed: np.ndarray
    learned_oof: np.ndarray
    learned_oof_active: np.ndarray
    learned_inner: np.ndarray | None = None
    learned_inner_active: np.ndarray | None = None


def _open_background_arrays(inputs: _AtlasInputs, *, require_inner: bool = False) -> _BackgroundArrays:
    paths = {
        "observed": inputs.bundle_root / "primitive_levels.npy",
        "fixed": inputs.predictor_root / "fixed/backgrounds.npy",
        "learned_oof": inputs.predictor_root / "learned_oof/backgrounds.npy",
        "learned_oof_active": inputs.predictor_root / "learned_oof/active.npy",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AtlasBackendError("frozen background arrays are missing: " + ", ".join(missing))
    observed = np.load(paths["observed"], mmap_mode="r", allow_pickle=False)
    fixed = np.load(paths["fixed"], mmap_mode="r", allow_pickle=False)
    learned_oof = np.load(paths["learned_oof"], mmap_mode="r", allow_pickle=False)
    learned_active = np.load(paths["learned_oof_active"], mmap_mode="r", allow_pickle=False)
    token_count = len(observed)
    if observed.shape != (token_count, 4) or fixed.shape != (token_count, 4, 4):
        raise AtlasBackendError("fixed background axes drift from [tokens,4 predictors,4 primitives]")
    if learned_oof.shape != (token_count, 2, 4) or learned_active.shape not in {
        (token_count,), (token_count, 2),
    }:
        raise AtlasBackendError("learned OOF background axes drift")
    inner = inner_active = None
    inner_path = inputs.predictor_root / "learned_inner/backgrounds.npy"
    inner_active_path = inputs.predictor_root / "learned_inner/active.npy"
    if inner_path.is_file() and inner_active_path.is_file():
        inner = np.load(inner_path, mmap_mode="r", allow_pickle=False)
        inner_active = np.load(inner_active_path, mmap_mode="r", allow_pickle=False)
        if inner.shape != (token_count, 5, 2, 4) or inner_active.shape != (token_count, 5):
            raise AtlasBackendError("learned-inner axes drift from [tokens,5,2,4]")
    elif require_inner:
        raise AtlasBackendError(
            "nested background search requires learned_inner/backgrounds.npy and active.npy"
        )
    return _BackgroundArrays(
        observed=observed, fixed=fixed, learned_oof=learned_oof,
        learned_oof_active=learned_active, learned_inner=inner,
        learned_inner_active=inner_active,
    )


def _background_tokens(
    arrays: _BackgroundArrays,
    token_slice: slice,
    primitive: str,
    kind: str,
    *,
    inner_outer_fold: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed, predictor and active mask in one primitive's units."""
    if primitive not in _PRIMITIVE_NAMES or kind not in _BACKGROUND_KINDS:
        raise AtlasBackendError(f"unknown background coordinate {primitive}/{kind}")
    primitive_index = _PRIMITIVE_NAMES.index(primitive)
    observed = np.asarray(arrays.observed[token_slice, primitive_index], dtype=np.float64)
    if kind in _BACKGROUND_KINDS[:4]:
        predictor_index = _BACKGROUND_KINDS.index(kind)
        prediction = np.asarray(
            arrays.fixed[token_slice, predictor_index, primitive_index], dtype=np.float64,
        )
        active = np.ones(len(observed), dtype=bool)
    else:
        learned_index = _BACKGROUND_KINDS.index(kind) - 4
        if inner_outer_fold is None:
            prediction = np.asarray(
                arrays.learned_oof[token_slice, learned_index, primitive_index], dtype=np.float64,
            )
            learned_active = np.asarray(arrays.learned_oof_active[token_slice], dtype=bool)
            active = learned_active if learned_active.ndim == 1 else learned_active[:, learned_index]
        else:
            if arrays.learned_inner is None or arrays.learned_inner_active is None:
                raise AtlasBackendError("pair-excluded learned backgrounds are unavailable")
            prediction = np.asarray(
                arrays.learned_inner[token_slice, inner_outer_fold, learned_index, primitive_index],
                dtype=np.float64,
            )
            active = np.asarray(
                arrays.learned_inner_active[token_slice, inner_outer_fold], dtype=bool,
            )
    if len(active):
        active = np.asarray(active, dtype=bool).copy()
        active[0] = False
    active &= np.isfinite(observed) & np.isfinite(prediction)
    return observed, prediction, active


def _extract_background_candidate(
    inputs: _AtlasInputs,
    arrays: _BackgroundArrays,
    answer_index: int,
    definition: _CandidateDefinition,
    spans: np.ndarray,
    *,
    inner_outer_fold: int | None = None,
    residual_cache: dict[tuple[str, str, int | None], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float, bool]:
    from spectral_utils.fusion_signal_registry import readout_steps

    _, primitive, kind = definition.score_key.split(":", 2)
    row = inputs.metadata[answer_index]
    token_start = int(row["offset"])
    token_stop = token_start + int(row["tokens"])
    cache_key = (primitive, kind, inner_outer_fold)
    cached = None if residual_cache is None else residual_cache.get(cache_key)
    if cached is None:
        observed, prediction, active = _background_tokens(
            arrays, slice(token_start, token_stop), primitive, kind,
            inner_outer_fold=inner_outer_fold,
        )
        residual = observed - prediction
        if residual_cache is not None:
            residual_cache[cache_key] = (residual, active)
    else:
        residual, active = cached
    score, step_active = readout_steps(
        residual, spans, str(definition.readout), active_mask=active,
    )
    if not step_active.any():
        return score, step_active, -1, 0.0, False
    rows = np.flatnonzero(step_active)
    peak = int(rows[np.argmax(score[rows])])
    return score, step_active, peak, float(score[peak]), True


def _extract_candidate(
    answer,
    definition: _CandidateDefinition,
    spans: np.ndarray | None = None,
    *,
    fixed_peak: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, int, float, bool]:
    """Return step score/mask, peak, answer detector score, detector-active."""
    if definition.answer_only:
        raw = np.asarray(answer[definition.score_key], dtype=np.float64)
        if raw.size != 1 or not np.isfinite(raw).all():
            return None, None, -1, 0.0, False
        # Answer-only gates never choose a step.  Their final-PB utility uses
        # one deterministic, label-free frozen locator when present.
        locator_peak = _digit025_peak(answer) if fixed_peak is None else int(fixed_peak)
        return None, None, locator_peak, float(raw.reshape(-1)[0]), locator_peak >= 0
    if definition.gate_token:
        values, active = _gate_token_values(answer, definition)
        try:
            detector = _gate_answer_readout(values, active, str(definition.readout), spans)
        except ValueError:
            return None, None, -1, 0.0, False
        locator_peak = _digit025_peak(answer) if fixed_peak is None else int(fixed_peak)
        return None, None, locator_peak, detector, locator_peak >= 0
    if definition.decoder == "earlier_ve_peak":
        peak = int(np.asarray(answer[definition.score_key]).reshape(()))
        answer_keys = answer.files if hasattr(answer, "files") else answer.keys()
        step_key = next((key for key in answer_keys if key.startswith("step__") and key.endswith("__readouts")), None)
        if step_key is None:
            raise AtlasBackendError("earlier-VE decoder has no aligned packed step axis")
        steps = int(np.asarray(answer[step_key]).shape[0])
        score = np.zeros(steps, dtype=np.float64)
        active = np.ones(steps, dtype=bool)
        if 0 <= peak < steps:
            score[peak] = 1.0
            return score, active, peak, 1.0, True
        return score, active, -1, 0.0, False
    score = np.asarray(answer[definition.score_key], dtype=np.float64)
    active = np.asarray(answer[definition.active_key], dtype=bool)
    if definition.native:
        score = score.reshape(-1)
        active = active.reshape(-1)
        peak = int(np.flatnonzero(active)[np.argmax(score[active])]) if active.any() else -1
    else:
        score = score[:, int(definition.readout_index)]
        active = active[:, int(definition.readout_index)]
        if definition.decoder == "step_top5":
            peak = int(np.asarray(answer[f"decision__{definition.signal}__step_top5"]).reshape(()))
        else:
            decision_key = f"decision__{definition.signal}__readouts"
            peak = int(np.asarray(answer[decision_key])[int(definition.readout_index), int(definition.decoder_index)])
    if score.ndim != 1 or active.shape != score.shape:
        raise AtlasBackendError(f"candidate {definition.name} has a malformed packed vector")
    active &= np.isfinite(score)
    if not active.any() or peak < 0 or peak >= len(score):
        return score, active, -1, 0.0, False
    detector = float(np.max(score[active]))
    return score, active, peak, detector, True


def _screen_candidates(
    inputs: _AtlasInputs,
    definitions: Sequence[_CandidateDefinition],
    *,
    incumbent_gate: np.ndarray | None = None,
) -> tuple[list[dict], tuple[_CandidateDefinition, ...]]:
    """Utility screen followed by the frozen two-per-family cap.

    Labels are read only here, downstream of extraction.  They rank the
    roster; they are never passed to a fusion-weight fit.
    """
    n = len(definitions)
    fold_count = 5
    pb_hits = np.zeros((n, fold_count), dtype=np.float64)
    pb_count = np.zeros((n, fold_count), dtype=np.int64)
    within_sum = np.zeros((n, fold_count), dtype=np.float64)
    within_count = np.zeros((n, fold_count), dtype=np.int64)
    identities = [hashlib.sha256() for _ in definitions]
    targets = np.asarray(inputs.joined["target"], dtype=np.int64)
    labels = np.asarray(inputs.joined["labels"], dtype=np.int8)
    offsets = inputs.offsets
    answer_metadata = _answer_metadata(inputs)
    if incumbent_gate is None:
        incumbent_gate, incumbent_error = _incumbent_gate(inputs, answer_metadata)
        if incumbent_error is not None:
            raise AtlasBackendError(
                "family-cap screening requires the frozen current gate: " + incumbent_error
            )
    incumbent_gate = np.asarray(incumbent_gate, dtype=bool)
    if incumbent_gate.shape != (inputs.answer_count,):
        raise AtlasBackendError("family-cap incumbent gate is not answer-aligned")
    cell_names = tuple(sorted({
        str(row["cell"]) for row in inputs.metadata if str(row["cell"]).startswith("pb_")
    }))
    cell_index = {name: index for index, name in enumerate(cell_names)}
    clean_hits = np.zeros((n, fold_count, len(cell_names)), dtype=np.int32)
    error_hits = np.zeros_like(clean_hits)
    clean_totals = np.zeros((fold_count, len(cell_names)), dtype=np.int32)
    error_totals = np.zeros_like(clean_totals)
    for answer_index, row in enumerate(inputs.metadata):
        cell = str(row["cell"])
        if cell not in cell_index:
            continue
        fold = int(row["fold"]); column = cell_index[cell]
        if targets[answer_index] == -1:
            clean_totals[fold, column] += 1
        elif targets[answer_index] >= 0:
            error_totals[fold, column] += 1
    by_signature: dict[tuple[str, int | None, int | None, bool], list[int]] = {}
    for index, definition in enumerate(definitions):
        signature = (
            definition.score_key, definition.active_key, definition.readout_index,
            definition.decoder_index, definition.native, definition.answer_only,
            definition.gate_token, definition.readout,
        )
        by_signature.setdefault(signature, []).append(index)
    gate_indexes = np.asarray(
        [index for index, row in enumerate(definitions) if row.point == "answer_gate"], dtype=np.int64,
    )
    gate_scores = np.zeros((inputs.answer_count, len(gate_indexes)), dtype=np.float32)
    gate_active = np.zeros((inputs.answer_count, len(gate_indexes)), dtype=bool)
    gate_peaks = np.full((inputs.answer_count, len(gate_indexes)), -1, dtype=np.int32)
    gate_column = {int(index): column for column, index in enumerate(gate_indexes)}
    spans_path = inputs.bundle_root / "step_spans.npy"
    if not spans_path.is_file():
        raise AtlasBackendError("bundle step_spans.npy is required for the locked gate readout bank")
    global_spans = np.load(spans_path, mmap_mode="r", allow_pickle=False)
    background_arrays = _open_background_arrays(inputs)
    for answer_index in range(inputs.answer_count):
        path = inputs.extract_root / "answers" / f"{answer_index:05d}.npz"
        target = int(targets[answer_index])
        fold = int(inputs.metadata[answer_index]["fold"])
        if fold not in range(fold_count):
            raise AtlasBackendError("Atlas nested roster requires exactly five fold identifiers 0..4")
        step_labels = labels[offsets[answer_index] : offsets[answer_index + 1]]
        row = inputs.metadata[answer_index]
        spans = np.asarray(global_spans[int(row["step_start"]):int(row["step_stop"])], dtype=np.int64)
        spans = spans - int(row.get("offset", spans[0, 0]))
        with np.load(path, allow_pickle=False) as archive:
            # ``NpzFile.__getitem__`` inflates a zip member on every access.
            # The same packed arrays feed thousands of candidate definitions,
            # so materialise each small per-answer member exactly once.
            saved = {name: np.asarray(archive[name]) for name in archive.files}
            fixed_peak = _digit025_peak(saved)
            background_cache: dict[
                tuple[str, str, int | None], tuple[np.ndarray, np.ndarray]
            ] = {}
            auc_cache: dict[tuple, float] = {}
            for signature, indexes in by_signature.items():
                representative = definitions[indexes[0]]
                if representative.point == "background":
                    extracted = _extract_background_candidate(
                        inputs, background_arrays, answer_index, representative, spans,
                        residual_cache=background_cache,
                    )
                else:
                    extracted = _extract_candidate(
                        saved, representative, spans, fixed_peak=fixed_peak,
                    )
                score, active, peak, detector, detector_active = extracted
                auc = float("nan")
                if str(inputs.metadata[answer_index]["cell"]).startswith("prmbench_") and score is not None:
                    auc_key = (
                        representative.score_key, representative.active_key,
                        representative.readout_index, representative.native,
                    )
                    if auc_key not in auc_cache:
                        auc_cache[auc_key] = _risk_pair_auc(step_labels, score, active)
                    auc = auc_cache[auc_key]
                for candidate_index in indexes:
                    identities[candidate_index].update(definitions[candidate_index].point.encode("utf8"))
                    if definitions[candidate_index].point == "token_pre_readout":
                        raw_key = f"token__{definitions[candidate_index].signal}__values"
                        raw_active_key = f"token__{definitions[candidate_index].signal}__active"
                        _hash_array(identities[candidate_index], np.asarray(saved[raw_key]))
                        _hash_array(
                            identities[candidate_index],
                            np.asarray(saved[raw_active_key], dtype=bool),
                        )
                        identities[candidate_index].update(
                            str(definitions[candidate_index].readout).encode("utf8")
                        )
                    if score is None:
                        identities[candidate_index].update(b"<no-step-score>")
                    else:
                        _hash_array(identities[candidate_index], np.asarray(score))
                        _hash_array(identities[candidate_index], np.asarray(active, dtype=bool))
                    _hash_array(identities[candidate_index], np.asarray([peak], dtype=np.int64))
                    _hash_array(
                        identities[candidate_index],
                        np.asarray([detector, float(detector_active)], dtype=np.float64),
                    )
                    if str(inputs.metadata[answer_index]["cell"]).startswith("pb_") and target >= 0:
                        pb_count[candidate_index, fold] += 1
                        pb_hits[candidate_index, fold] += float(detector_active and peak == target)
                    cell = str(inputs.metadata[answer_index]["cell"])
                    if cell in cell_index and definitions[candidate_index].point != "answer_gate":
                        prediction = peak if incumbent_gate[answer_index] else -1
                        decision_valid = bool((not incumbent_gate[answer_index]) or detector_active)
                        correct = bool(decision_valid and prediction == target)
                        if target == -1:
                            clean_hits[candidate_index, fold, cell_index[cell]] += int(correct)
                        elif target >= 0:
                            error_hits[candidate_index, fold, cell_index[cell]] += int(correct)
                    if np.isfinite(auc):
                        within_count[candidate_index, fold] += 1
                        within_sum[candidate_index, fold] += auc
                    if candidate_index in gate_column:
                        column = gate_column[candidate_index]
                        gate_scores[answer_index, column] = detector
                        gate_active[answer_index, column] = detector_active
                        gate_peaks[answer_index, column] = peak
    rows = []
    for index, definition in enumerate(definitions):
        total_pb_count = int(pb_count[index].sum())
        total_within_count = int(within_count[index].sum())
        pb = float(pb_hits[index].sum() / total_pb_count) if total_pb_count else float("nan")
        within = float(within_sum[index].sum() / total_within_count) if total_within_count else float("nan")
        rows.append({
            "name": definition.name,
            "point": definition.point,
            "signal": definition.signal,
            "family": definition.family,
            "access_scope": definition.access_scope,
            "readout": definition.readout,
            "decoder": definition.decoder,
            "raw_pb": pb,
            "prmb_within": within,
            "pb_answers": total_pb_count,
            "prmb_answers": total_within_count,
            "semantic_hash": identities[index].hexdigest(),
            "selected_outer_folds": [],
            "status": "SCREENED",
        })

    # Gate utilities are final PB accuracy on outer-training answers after
    # within-cell midrank and the fixed .33 decision.  This is computed once
    # per outer split; held-fold labels never enter that split's shortlist.
    gate_train_pb = np.full((len(gate_indexes), fold_count), np.nan, dtype=np.float64)
    if len(gate_indexes):
        all_cells = np.asarray([row["cell"] for row in inputs.metadata])
        all_folds = np.asarray([row["fold"] for row in inputs.metadata], dtype=np.int64)
        pb = np.char.startswith(all_cells.astype(str), "pb_")
        for outer in range(fold_count):
            train = all_folds != outer
            fusion_sets = {str(column): (column,) for column in range(len(gate_indexes))}
            opened, _, active = rank_fused_gate_decisions(
                gate_scores[train], tuple(str(column) for column in range(len(gate_indexes))),
                all_cells[train], fusion_sets, q=0.33, active=gate_active[train],
            )
            local_target = targets[train]
            local_pb = pb[train]
            for column in range(len(gate_indexes)):
                prediction = np.where(opened[:, column], gate_peaks[train, column], -1)
                valid = (~opened[:, column]) | (
                    active[:, column] & (gate_peaks[train, column] >= 0)
                )
                if local_pb.any():
                    gate_train_pb[column, outer] = _canonical_pb_score(
                        local_target, prediction, valid, all_cells[train], local_pb,
                    )

    canonical_by_identity: dict[tuple[str, str], int] = {}
    duplicate_of: dict[int, int] = {}
    for index, definition in enumerate(definitions):
        identity = (definition.point, identities[index].hexdigest())
        if identity in canonical_by_identity:
            duplicate_of[index] = canonical_by_identity[identity]
            rows[index]["status"] = "EXACT_DUPLICATE"
            rows[index]["canonical"] = definitions[duplicate_of[index]].name
        else:
            canonical_by_identity[identity] = index

    selected_indexes: set[int] = set()
    for outer in range(fold_count):
        for point_family in sorted({(row.point, row.family) for row in definitions}):
            indexes = [
                index for index, row in enumerate(definitions)
                if (row.point, row.family) == point_family and index not in duplicate_of
            ]
            if not indexes:
                continue
            scored = []
            for index in indexes:
                if definitions[index].point == "answer_gate":
                    pb_value = gate_train_pb[gate_column[index], outer]
                else:
                    training_folds = np.asarray([fold != outer for fold in range(fold_count)])
                    values = []
                    for cell_column in range(len(cell_names)):
                        clean_total = int(clean_totals[training_folds, cell_column].sum())
                        error_total = int(error_totals[training_folds, cell_column].sum())
                        if not clean_total or not error_total:
                            values = []
                            break
                        clean_accuracy = float(
                            clean_hits[index, training_folds, cell_column].sum() / clean_total
                        )
                        error_accuracy = float(
                            error_hits[index, training_folds, cell_column].sum() / error_total
                        )
                        denominator = clean_accuracy + error_accuracy
                        values.append(
                            2.0 * clean_accuracy * error_accuracy / denominator
                            if denominator else 0.0
                        )
                    pb_value = float(np.mean(values)) if values else float("nan")
                within_den = int(within_count[index].sum() - within_count[index, outer])
                within_value = (
                    float((within_sum[index].sum() - within_sum[index, outer]) / within_den)
                    if within_den else float("nan")
                )
                scored.append((index, pb_value, within_value))
            scored.sort(key=lambda item: (
                -(item[1] if np.isfinite(item[1]) else -np.inf),
                -(item[2] if np.isfinite(item[2]) else -np.inf),
                definitions[item[0]].name,
            ))
            leader_index, leader_pb, leader_within = scored[0]
            selected_indexes.add(leader_index)
            rows[leader_index]["selected_outer_folds"].append(outer)
            eligible = []
            for index, pb_value, within_value in scored[1:]:
                pb_ok = not np.isfinite(leader_pb) or (np.isfinite(pb_value) and pb_value >= leader_pb - 0.01)
                within_ok = not np.isfinite(leader_within) or (
                    np.isfinite(within_value) and within_value >= leader_within - 0.005
                )
                if pb_ok and within_ok:
                    structural_distance = sum((
                        definitions[index].signal != definitions[leader_index].signal,
                        definitions[index].readout != definitions[leader_index].readout,
                        definitions[index].decoder != definitions[leader_index].decoder,
                    ))
                    eligible.append((structural_distance, pb_value, within_value, index))
            if eligible:
                diversity_index = max(
                    eligible,
                    key=lambda item: (
                        item[0], item[1] if np.isfinite(item[1]) else -np.inf,
                        item[2] if np.isfinite(item[2]) else -np.inf,
                        tuple(-ord(char) for char in definitions[item[3]].name),
                    ),
                )[3]
                selected_indexes.add(diversity_index)
                rows[diversity_index]["selected_outer_folds"].append(outer)
    selected = tuple(definitions[index] for index in sorted(selected_indexes, key=lambda index: definitions[index].name))
    for index in selected_indexes:
        rows[index]["status"] = "OUTER_TRAINING_FAMILY_REPRESENTATIVE"
    return rows, selected


@dataclass(frozen=True)
class _Consolidated:
    names: tuple[str, ...]
    points: tuple[str, ...]
    families: tuple[str, ...]
    access_scopes: tuple[str, ...]
    costs: np.ndarray
    step_scores: np.ndarray
    step_active: np.ndarray
    peaks: np.ndarray
    answer_scores: np.ndarray
    answer_active: np.ndarray
    definitions: tuple[_CandidateDefinition, ...]
    aliases: Mapping[str, str]


def _array_identity(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _consolidate_selected(
    inputs: _AtlasInputs,
    selected: Sequence[_CandidateDefinition],
    output_root: Path,
) -> _Consolidated:
    offsets = inputs.offsets
    total_steps = int(offsets[-1])
    answers = inputs.answer_count
    count = len(selected)
    step_scores = np.zeros((total_steps, count), dtype=np.float32)
    step_active = np.zeros((total_steps, count), dtype=bool)
    peaks = np.full((answers, count), -1, dtype=np.int32)
    answer_scores = np.zeros((answers, count), dtype=np.float32)
    answer_active = np.zeros((answers, count), dtype=bool)
    background_arrays = _open_background_arrays(inputs)
    token_signals = tuple(sorted({
        definition.signal for definition in selected
        if definition.point == "token_pre_readout"
    }))
    token_column = {name: index for index, name in enumerate(token_signals)}
    token_values = token_active = None
    if token_signals:
        token_count = int(background_arrays.observed.shape[0])
        token_values = np.lib.format.open_memmap(
            Path(output_root) / "TOKEN_VALUES.npy", mode="w+", dtype="float32",
            shape=(token_count, len(token_signals)),
        )
        token_active = np.lib.format.open_memmap(
            Path(output_root) / "TOKEN_ACTIVE.npy", mode="w+", dtype="bool",
            shape=(token_count, len(token_signals)),
        )
        token_values[:] = 0.0
        token_active[:] = False
    by_signature: dict[tuple, list[int]] = {}
    for index, definition in enumerate(selected):
        signature = (
            definition.score_key, definition.active_key, definition.readout_index,
            definition.decoder_index, definition.native, definition.answer_only,
            definition.gate_token, definition.readout,
        )
        by_signature.setdefault(signature, []).append(index)
    spans_path = inputs.bundle_root / "step_spans.npy"
    global_spans = np.load(spans_path, mmap_mode="r", allow_pickle=False)
    for answer_index in range(answers):
        start, stop = map(int, offsets[answer_index : answer_index + 2])
        row = inputs.metadata[answer_index]
        spans = np.asarray(global_spans[int(row["step_start"]):int(row["step_stop"])], dtype=np.int64)
        spans = spans - int(row.get("offset", spans[0, 0]))
        path = inputs.extract_root / "answers" / f"{answer_index:05d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
            fixed_peak = _digit025_peak(saved)
            background_cache: dict[
                tuple[str, str, int | None], tuple[np.ndarray, np.ndarray]
            ] = {}
            if token_values is not None and token_active is not None:
                token_start = int(row["offset"])
                token_stop = token_start + int(row["tokens"])
                for signal, column in token_column.items():
                    values_key = f"token__{signal}__values"
                    active_key = f"token__{signal}__active"
                    if values_key not in saved or active_key not in saved:
                        raise AtlasBackendError(f"selected token signal is absent: {signal}")
                    values = np.asarray(saved[values_key], dtype=np.float32)
                    active = np.asarray(saved[active_key], dtype=bool)
                    if values.shape != (token_stop - token_start,) or active.shape != values.shape:
                        raise AtlasBackendError(f"selected token signal is malformed: {signal}")
                    token_values[token_start:token_stop, column] = np.where(active, values, 0.0)
                    token_active[token_start:token_stop, column] = active
            for indexes in by_signature.values():
                definition = selected[indexes[0]]
                if definition.point == "background":
                    score, active, peak, detector, detector_active = _extract_background_candidate(
                        inputs, background_arrays, answer_index, definition, spans,
                        residual_cache=background_cache,
                    )
                else:
                    score, active, peak, detector, detector_active = _extract_candidate(
                        saved, definition, spans, fixed_peak=fixed_peak,
                    )
                for index in indexes:
                    local = selected[index]
                    peaks[answer_index, index] = peak
                    answer_scores[answer_index, index] = detector
                    answer_active[answer_index, index] = detector_active
                    if score is None:
                        continue
                    if len(score) != stop - start:
                        raise AtlasBackendError(f"step count drift for {local.name} at answer {answer_index}")
                    if local.point == "decoder":
                        # Decision-resolution fusion is a vote over one-hot
                        # step decisions, not flat fusion with step scores.
                        if peak >= 0:
                            step_scores[start + peak, index] = 1.0
                            step_active[start:stop, index] = True
                    else:
                        step_scores[start:stop, index] = np.where(active, score, 0.0)
                        step_active[start:stop, index] = active

    identities: dict[tuple[str, str], str] = {}
    keep: list[int] = []
    aliases: dict[str, str] = {}
    for index, definition in enumerate(selected):
        if definition.point == "token_pre_readout":
            if token_values is None or token_active is None:
                raise AtlasBackendError("token candidate lacks raw token identity store")
            column = token_column[definition.signal]
            identity = _array_identity(
                np.asarray(token_values[:, column]), np.asarray(token_active[:, column]),
                np.frombuffer(str(definition.readout).encode("utf8"), dtype=np.uint8),
            )
        elif definition.point == "answer_gate":
            identity = _array_identity(answer_scores[:, index], answer_active[:, index])
        else:
            identity = _array_identity(step_scores[:, index], step_active[:, index], peaks[:, index])
        key = (definition.point, identity)
        if key in identities:
            aliases[definition.name] = identities[key]
        else:
            identities[key] = definition.name
            keep.append(index)
    if not keep:
        raise AtlasBackendError("exact de-duplication removed every eligible candidate")
    definitions = tuple(selected[index] for index in keep)
    consolidated = _Consolidated(
        names=tuple(row.name for row in definitions),
        points=tuple(row.point for row in definitions),
        families=tuple(row.family for row in definitions),
        access_scopes=tuple(row.access_scope for row in definitions),
        costs=np.asarray([row.cost for row in definitions], dtype=np.float64),
        step_scores=step_scores[:, keep].astype(np.float32, copy=False),
        step_active=step_active[:, keep],
        peaks=peaks[:, keep],
        answer_scores=answer_scores[:, keep].astype(np.float32, copy=False),
        answer_active=answer_active[:, keep],
        definitions=definitions,
        aliases=aliases,
    )
    path = Path(output_root) / "CONSOLIDATED.npz"
    temporary = path.with_suffix(".npz.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            names=np.asarray(consolidated.names),
            points=np.asarray(consolidated.points),
            families=np.asarray(consolidated.families),
            access_scopes=np.asarray(consolidated.access_scopes),
            costs=consolidated.costs.astype(np.float32),
            step_scores=consolidated.step_scores,
            step_active=consolidated.step_active,
            peaks=consolidated.peaks,
            answer_scores=consolidated.answer_scores,
            answer_active=consolidated.answer_active,
            offsets=offsets,
        )
    os.replace(temporary, path)
    if token_values is not None and token_active is not None:
        token_values.flush(); token_active.flush()
        del token_values, token_active
        _write_json(Path(output_root) / "TOKEN_INDEX.json", {
            "schema": "fusion-independence-atlas-v1/token-store-v1",
            "signals": list(token_signals),
            "candidate_map": {
                definition.name: {
                    "signal": definition.signal,
                    "column": token_column[definition.signal],
                    "readout": definition.readout,
                }
                for definition in definitions
                if definition.point == "token_pre_readout"
            },
            "values": "TOKEN_VALUES.npy", "active": "TOKEN_ACTIVE.npy",
        })
    return consolidated


def _load_consolidated(dependence_root: Path) -> _Consolidated:
    dependence_root = Path(dependence_root)
    candidates_payload = _read_json(dependence_root / "CANDIDATES.json")
    rows = {str(row["name"]): row for row in candidates_payload["eligible"]}
    with np.load(dependence_root / "CONSOLIDATED.npz", allow_pickle=False) as saved:
        names = tuple(str(value) for value in saved["names"].tolist())
        points = tuple(str(value) for value in saved["points"].tolist())
        families = tuple(str(value) for value in saved["families"].tolist())
        scopes = tuple(str(value) for value in saved["access_scopes"].tolist())
        definitions = tuple(_CandidateDefinition(
            name=name,
            point=points[index],
            signal=str(rows[name].get("signal", name)),
            family=families[index],
            access_scope=scopes[index],
            readout=rows[name].get("readout"),
            decoder=rows[name].get("decoder"),
            score_key="frozen",
            active_key=None,
            readout_index=None,
            decoder_index=None,
            cost=float(rows[name].get("cost", 1.0)),
        ) for index, name in enumerate(names))
        return _Consolidated(
            names=names,
            points=points,
            families=families,
            access_scopes=scopes,
            costs=np.asarray(saved["costs"], dtype=np.float64),
            step_scores=np.asarray(saved["step_scores"], dtype=np.float64),
            step_active=np.asarray(saved["step_active"], dtype=bool),
            peaks=np.asarray(saved["peaks"], dtype=np.int64),
            answer_scores=np.asarray(saved["answer_scores"], dtype=np.float64),
            answer_active=np.asarray(saved["answer_active"], dtype=bool),
            definitions=definitions,
            aliases=candidates_payload.get("aliases", {}),
        )


def _answer_metadata(inputs: _AtlasInputs) -> ObservationMetadata:
    target = np.asarray(inputs.joined["target"], dtype=np.int64)
    steps = np.asarray([int(row["step_stop"]) - int(row["step_start"]) for row in inputs.metadata])
    first = np.full(inputs.answer_count, np.nan, dtype=np.float64)
    erroneous = target >= 0
    first[erroneous] = target[erroneous] / np.maximum(steps[erroneous] - 1, 1)
    opportunities = np.zeros(inputs.answer_count, dtype=np.float64)
    for index in range(inputs.answer_count):
        path = inputs.extract_root / "answers" / f"{index:05d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            key = "token__digit.opportunity__values"
            if key in saved.files:
                opportunities[index] = float(np.asarray(saved[key], dtype=np.float64).sum())
    return ObservationMetadata(
        groups=np.asarray([row["group_id"] for row in inputs.metadata]),
        cells=np.asarray([row["cell"] for row in inputs.metadata]),
        tokens=np.asarray([row["tokens"] for row in inputs.metadata]),
        steps=steps,
        folds=np.asarray([row["fold"] for row in inputs.metadata]),
        first_error_position=first,
        digit_opportunities=opportunities,
    )


def _incumbent_gate(inputs: _AtlasInputs, metadata: ObservationMetadata) -> tuple[np.ndarray, str | None]:
    """Replay the frozen Step-396 tail15 + digit-rate answer gate."""
    candidates = (
        inputs.extract_root.parent / "baseline_replay/SCORES_FROZEN.npz",
        inputs.contract_root / "results/temporal_research_baseline_v1/SCORES_FROZEN.npz",
    )
    archive = next((path for path in candidates if path.is_file()), None)
    if archive is None:
        return np.zeros(inputs.answer_count, dtype=bool), "frozen tail15 gate_raw archive is unavailable"
    with np.load(archive, allow_pickle=False) as saved:
        if "gate_raw" not in saved.files or saved["gate_raw"].shape != (inputs.answer_count,):
            return np.zeros(inputs.answer_count, dtype=bool), "frozen tail15 gate_raw has incompatible shape"
        tail = np.asarray(saved["gate_raw"], dtype=np.float64)
    count = np.zeros(inputs.answer_count, dtype=np.float64)
    for index in range(inputs.answer_count):
        with np.load(inputs.extract_root / "answers" / f"{index:05d}.npz", allow_pickle=False) as saved:
            key = "token__digit.disagreement__values"
            if key not in saved.files:
                return np.zeros(inputs.answer_count, dtype=bool), "digit disagreement stream is unavailable"
            count[index] = float(np.asarray(saved[key], dtype=np.float64).sum())
    rate = np.divide(
        count, np.maximum(metadata.digit_opportunities, 1.0),
        out=np.zeros_like(count), where=np.isfinite(metadata.digit_opportunities),
    )
    pb = np.char.startswith(metadata.cells.astype(str), "pb_")
    detector_active = np.column_stack((pb & np.isfinite(tail), pb & np.isfinite(rate)))
    opened, _, active = rank_fused_gate_decisions(
        np.column_stack((tail, rate)), ("tail15", "digit_rate"), metadata.cells,
        {"equal_rank_tail15_digit_rate": ("tail15", "digit_rate")},
        q=0.33, active=detector_active,
    )
    gate = opened[:, 0]
    gate[~active[:, 0]] = False
    return gate, None


def _token_metadata(inputs: _AtlasInputs) -> ObservationMetadata:
    tokens = np.asarray([int(row["tokens"]) for row in inputs.metadata], dtype=np.int64)
    groups = np.repeat(np.asarray([row["group_id"] for row in inputs.metadata]), tokens)
    cells = np.repeat(np.asarray([row["cell"] for row in inputs.metadata]), tokens)
    folds = np.repeat(np.asarray([row["fold"] for row in inputs.metadata]), tokens)
    answer_steps = np.asarray([int(row["step_stop"]) - int(row["step_start"]) for row in inputs.metadata])
    return ObservationMetadata(
        groups=groups,
        cells=cells,
        tokens=np.repeat(tokens, tokens),
        steps=np.repeat(answer_steps, tokens),
        folds=folds,
    )


def _iter_predictor_residual_matrices(inputs: _AtlasInputs):
    """Yield one primitive's predictor matrix at a time to bound token-scale RAM."""
    primitive_path = inputs.bundle_root / "primitive_levels.npy"
    fixed_path = inputs.predictor_root / "fixed/backgrounds.npy"
    learned_path = inputs.predictor_root / "learned_oof/backgrounds.npy"
    learned_active_path = inputs.predictor_root / "learned_oof/active.npy"
    for path in (primitive_path, fixed_path, learned_path, learned_active_path):
        if not path.is_file():
            raise AtlasBackendError(f"frozen predictor array is missing: {path}")
    observed = np.load(primitive_path, mmap_mode="r", allow_pickle=False)
    fixed = np.load(fixed_path, mmap_mode="r", allow_pickle=False)
    learned = np.load(learned_path, mmap_mode="r", allow_pickle=False)
    learned_active = np.load(learned_active_path, mmap_mode="r", allow_pickle=False)
    if observed.ndim != 2 or observed.shape[1] != 4:
        raise AtlasBackendError("primitive bundle is not [tokens,4]")
    if fixed.shape != (len(observed), 4, 4) or learned.shape != (len(observed), 2, 4):
        raise AtlasBackendError("predictor arrays do not contain exactly four primitive targets")
    if learned_active.shape not in ((len(observed),), (len(observed), 2)):
        raise AtlasBackendError("learned predictor active mask has an unsupported shape")
    method_names = ("prefix_mean", "mean16", "no_reset", "bocpd", "source_excluded_ridge", "source_excluded_tcn")
    metadata = _token_metadata(inputs)
    first_token = np.zeros(len(observed), dtype=bool)
    offsets = np.r_[0, np.cumsum([int(row["tokens"]) for row in inputs.metadata])]
    first_token[offsets[:-1]] = True
    active = np.ones((len(observed), 6), dtype=bool)
    if learned_active.ndim == 1:
        active[:, 4:] &= np.asarray(learned_active)[:, None]
    else:
        active[:, 4:] &= np.asarray(learned_active)
    active[first_token] = False
    primitive_names = ("H0lim", "VE0", "VE0.75", "VE1")
    for primitive, name in enumerate(primitive_names):
        predictions = np.column_stack((fixed[:, :, primitive], learned[:, :, primitive]))
        matrix = predictor_residual_matrix(
            np.asarray(observed[:, primitive], dtype=np.float64),
            predictions,
            method_names,
            metadata,
            active=active,
        )
        del predictions
        yield name, matrix
        del matrix


def _predictor_residual_matrices(inputs: _AtlasInputs) -> dict[str, ErrorMatrix]:
    """Compatibility wrapper; production orchestration consumes the iterator."""
    return dict(_iter_predictor_residual_matrices(inputs))


def _point_indexes(consolidated: _Consolidated, point: str) -> np.ndarray:
    return np.asarray([index for index, value in enumerate(consolidated.points) if value == point], dtype=np.int64)


def _pb_raw_matrix(
    inputs: _AtlasInputs,
    consolidated: _Consolidated,
    metadata: ObservationMetadata,
    point: str,
) -> ErrorMatrix | None:
    indexes = _point_indexes(consolidated, point)
    if not len(indexes):
        return None
    offsets = inputs.offsets
    topk_sets = []
    for index in indexes:
        method = []
        for answer_index in range(inputs.answer_count):
            start, stop = map(int, offsets[answer_index:answer_index + 2])
            active = consolidated.step_active[start:stop, index]
            rows = np.flatnonzero(active)
            if not len(rows):
                method.append(None)
                continue
            keep = min(10, len(rows))
            local_scores = consolidated.step_scores[start:stop, index][rows]
            chosen = rows[np.argpartition(local_scores, len(rows) - keep)[-keep:]]
            method.append(frozenset(int(value) for value in chosen))
        topk_sets.append(tuple(method))
    return pb_raw_locator_miss_matrix(
        consolidated.peaks[:, indexes],
        inputs.joined["target"],
        consolidated.answer_active[:, indexes],
        tuple(consolidated.names[index] for index in indexes),
        metadata,
        scores=np.where(
            consolidated.answer_active[:, indexes],
            consolidated.answer_scores[:, indexes],
            np.nan,
        ),
        active=consolidated.answer_active[:, indexes],
        topk_sets=tuple(topk_sets),
        retain_invalid_as_failures=True,
    )


def _prmb_pair_matrix(
    inputs: _AtlasInputs,
    consolidated: _Consolidated,
    point: str,
) -> ErrorMatrix | None:
    indexes = _point_indexes(consolidated, point)
    if not len(indexes):
        return None
    offsets = inputs.offsets
    labels = np.asarray(inputs.joined["labels"], dtype=np.int8)
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    groups: list[str] = []
    cells: list[str] = []
    tokens: list[int] = []
    steps: list[int] = []
    folds: list[int] = []
    active_rows: list[np.ndarray] = []
    for answer_index, row in enumerate(inputs.metadata):
        if not str(row["cell"]).startswith("prmbench_"):
            continue
        start, stop = map(int, offsets[answer_index : answer_index + 2])
        local_labels = labels[start:stop]
        positive = np.flatnonzero(local_labels == 1)
        negative = np.flatnonzero(local_labels == 0)
        for pos in positive:
            for neg in negative:
                positive_rows.append(consolidated.step_scores[start + pos, indexes])
                negative_rows.append(consolidated.step_scores[start + neg, indexes])
                active_rows.append(
                    consolidated.step_active[start + pos, indexes]
                    & consolidated.step_active[start + neg, indexes]
                )
                groups.append(str(row["group_id"])); cells.append(str(row["cell"]))
                tokens.append(int(row["tokens"])); steps.append(stop - start); folds.append(int(row["fold"]))
    if not positive_rows:
        return None
    metadata = ObservationMetadata(
        groups=np.asarray(groups), cells=np.asarray(cells), tokens=np.asarray(tokens),
        steps=np.asarray(steps), folds=np.asarray(folds),
    )
    return prmb_pairwise_misorder_matrix(
        np.asarray(positive_rows), np.asarray(negative_rows),
        tuple(consolidated.names[index] for index in indexes), metadata,
        higher_is_positive=True, active=np.asarray(active_rows),
    )


def _analyse_error_matrix(
    key: str,
    matrix: ErrorMatrix,
    *,
    draws: int,
    seed: int,
) -> tuple[list[dict], dict[tuple[str, str], DependenceStatus], str | None]:
    if matrix.shape[1] < 2:
        return [], {}, None
    conditioned = cross_fit_nuisance(matrix, n_splits=5, seed=seed)
    try:
        result = bootstrap_pair_diagnostics(
            matrix, conditioned=conditioned, draws=draws, seed=seed,
            min_successes=50, min_failures=50, min_groups=20,
        )
        bootstrap_error = None
    except (ValueError, FloatingPointError) as error:
        # Insufficient/degenerate support is a resolved UNRESOLVED outcome,
        # not permission to invent a finite interval.
        diagnostics = all_pair_diagnostics(matrix, conditioned=conditioned)
        result = None
        bootstrap_error = f"{type(error).__name__}: {error}"
    rows: list[dict] = []
    statuses: dict[tuple[str, str], DependenceStatus] = {}
    if result is None:
        items = diagnostics.items()
    else:
        items = result.diagnostics.items()
    for pair, diagnostics in sorted(items):
        if result is None:
            interval = None
            status = DependenceStatus.UNRESOLVED
        else:
            interval = result.intervals[pair]
            status = result.statuses[pair]
        statuses[pair] = status
        rows.append({
            "matrix": key,
            **diagnostics.as_dict(),
            "interval": None if interval is None else _json_ready(interval),
            "status": status.value,
            "bootstrap_draws": int(draws),
            "bootstrap_error": bootstrap_error,
        })
    return rows, statuses, bootstrap_error


def _combined_point_graph(
    point: str,
    names: Sequence[str],
    statuses_by_matrix: Mapping[str, Mapping[tuple[str, str], DependenceStatus]],
) -> tuple[dict[str, frozenset[str]], dict[tuple[str, str], DependenceStatus]]:
    combined: dict[tuple[str, str], DependenceStatus] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = (left, right)
            reverse = (right, left)
            observed = []
            for matrix_key, statuses in statuses_by_matrix.items():
                if not matrix_key.startswith(point + ":"):
                    continue
                if key in statuses:
                    observed.append(statuses[key])
                elif reverse in statuses:
                    observed.append(statuses[reverse])
            if observed and all(value == DependenceStatus.INDEPENDENCE_COMPATIBLE for value in observed):
                combined[key] = DependenceStatus.INDEPENDENCE_COMPATIBLE
            elif any(value == DependenceStatus.REDUNDANT for value in observed):
                combined[key] = DependenceStatus.REDUNDANT
            elif any(value == DependenceStatus.DEPENDENT_COMPLEMENTARY for value in observed):
                combined[key] = DependenceStatus.DEPENDENT_COMPLEMENTARY
            else:
                combined[key] = DependenceStatus.UNRESOLVED
    return build_compatibility_graph(names, combined), combined


def _weighted_group_point(residuals: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    residuals = np.asarray(residuals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if residuals.ndim != 2 or len(residuals) != len(weights) or total <= 0:
        return float("nan"), float("nan")
    mean = (residuals * weights[:, None]).sum(axis=0) / total
    centered = residuals - mean
    variance = (centered * centered * weights[:, None]).sum(axis=0) / total
    scale = np.sqrt(np.maximum(variance, 0.0))
    if np.any(scale <= 1e-12):
        return float("nan"), 0.0
    z = centered / scale
    correlation = (z * weights[:, None]).T @ z / total
    off_diagonal = np.abs(correlation[np.triu_indices(residuals.shape[1], 1)])
    eigenvalues = np.maximum(np.linalg.eigvalsh((correlation + correlation.T) / 2.0), 0.0)
    denominator = float(eigenvalues @ eigenvalues)
    effective_rank = float(eigenvalues.sum() ** 2 / denominator) if denominator > 1e-15 else 0.0
    return float(off_diagonal.max()), effective_rank


def _common_group_diagnostics(
    matrix: ErrorMatrix,
    cliques: Sequence[Sequence[str]],
    *,
    conditioned: ConditionedErrors,
    draws: int,
    seed: int,
) -> list[GroupDiagnostics]:
    """Evaluate all cliques with one paired source-group bootstrap plan.

    The simultaneous interval spans both statistics for every clique at this
    insertion point.  This avoids the multiplicity and runtime bug caused by
    starting an unrelated 10k-draw bootstrap for each clique.
    """
    if not cliques:
        return []
    conditioned.validate(matrix)
    unique_groups, row_group = np.unique(
        matrix.metadata.groups, return_inverse=True,
    )
    row_group = np.asarray(row_group, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    multiplicities = np.zeros((int(draws), len(unique_groups)), dtype=np.int32)
    for draw in range(int(draws)):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        multiplicities[draw] = np.bincount(sampled, minlength=len(unique_groups))

    name_to_index = {name: index for index, name in enumerate(matrix.method_names)}
    payload: list[dict[str, object]] = []
    point_values: list[float] = []
    required_finite = int(math.ceil(0.95 * int(draws)))

    # Exact samples are written once and read one clique at a time.  This keeps
    # RAM O(draws + draws*groups), rather than O(draws*number_of_cliques), while
    # preserving the common paired draw plan and the original Bonferroni family.
    with tempfile.TemporaryFile() as spool:
        for members in cliques:
            indexes = np.asarray([name_to_index[name] for name in members], dtype=np.int64)
            active = np.all(matrix.active[:, indexes], axis=1)
            residuals = conditioned.residuals[np.ix_(active, indexes)]
            local_group = row_group[active]
            point = np.asarray(
                _weighted_group_point(residuals, np.ones(len(residuals))),
                dtype=np.float64,
            )
            width = len(indexes)
            sufficient = np.zeros(
                (len(unique_groups), 1 + width + width * width), dtype=np.float64,
            )
            sufficient[:, 0] = np.bincount(local_group, minlength=len(unique_groups))
            cursor = 1
            for column in range(width):
                sufficient[:, cursor + column] = np.bincount(
                    local_group, weights=residuals[:, column], minlength=len(unique_groups),
                )
            cursor += width
            for left in range(width):
                for right in range(width):
                    sufficient[:, cursor + left * width + right] = np.bincount(
                        local_group,
                        weights=residuals[:, left] * residuals[:, right],
                        minlength=len(unique_groups),
                    )
            totals = multiplicities @ sufficient
            n = totals[:, 0]
            sums = totals[:, 1:1 + width]
            cross = totals[:, 1 + width:].reshape(int(draws), width, width)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = sums / n[:, None]
                covariance = cross / n[:, None, None] - mean[:, :, None] * mean[:, None, :]
                scale = np.sqrt(
                    np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0)
                )
                correlation = covariance / (scale[:, :, None] * scale[:, None, :])
            invalid = (n <= 0) | np.any(scale <= 1e-12, axis=1) \
                | ~np.isfinite(correlation).all(axis=(1, 2))
            triangle = np.triu_indices(width, 1)
            maximum = np.max(np.abs(correlation[:, triangle[0], triangle[1]]), axis=1)
            safe_correlation = np.where(np.isfinite(correlation), correlation, 0.0)
            eigenvalues = np.maximum(
                np.linalg.eigvalsh(
                    (safe_correlation + np.swapaxes(safe_correlation, 1, 2)) / 2.0
                ),
                0.0,
            )
            denominator = np.sum(eigenvalues * eigenvalues, axis=1)
            effective = np.divide(
                np.sum(eigenvalues, axis=1) ** 2, denominator,
                out=np.zeros_like(denominator), where=denominator > 1e-15,
            )
            samples = np.column_stack((maximum, effective))
            samples[invalid] = np.nan
            offset = spool.tell()
            spool.write(np.asarray(samples, dtype=np.float64).tobytes(order="C"))
            finite_draws = tuple(int(value) for value in np.isfinite(samples).sum(axis=0))
            values = matrix.values[np.ix_(active, indexes)]
            bounded_support = True
            if matrix.target in _BOUNDED_TARGETS:
                bounded_support = bool(
                    np.all(values.sum(axis=0) >= 50)
                    and np.all((1.0 - values).sum(axis=0) >= 50)
                )
            payload.append({
                "members": tuple(members),
                "width": width,
                "n": int(active.sum()),
                "source_groups": int(np.unique(row_group[active]).size),
                "point": point,
                "finite_draws": finite_draws,
                "bounded_support": bounded_support,
                "offset": offset,
            })
            point_values.extend(point.tolist())

        point_array = np.asarray(point_values, dtype=np.float64)
        usable = np.asarray([
            np.isfinite(point_array[2 * index + statistic])
            and int(row["finite_draws"][statistic]) > 0
            for index, row in enumerate(payload)
            for statistic in range(2)
        ], dtype=bool)
        family = max(1, int(usable.sum()))
        tail = (1.0 - 0.95) / (2.0 * family)
        global_interval_error = any(
            usable[2 * index + statistic]
            and int(row["finite_draws"][statistic]) < required_finite
            for index, row in enumerate(payload)
            for statistic in range(2)
        )

        output = []
        sample_bytes = int(draws) * 2 * np.dtype(np.float64).itemsize
        for index, row in enumerate(payload):
            spool.seek(int(row["offset"]))
            samples = np.frombuffer(spool.read(sample_bytes), dtype=np.float64).reshape(
                int(draws), 2,
            )
            intervals = np.full((2, 2), np.nan, dtype=np.float64)
            if not global_interval_error:
                for statistic in range(2):
                    if not usable[2 * index + statistic]:
                        continue
                    finite = samples[np.isfinite(samples[:, statistic]), statistic]
                    intervals[statistic] = np.quantile(finite, [tail, 1.0 - tail])
            point = np.asarray(row["point"], dtype=np.float64)
            source_groups = int(row["source_groups"])
            supported = (
                source_groups >= 20
                and bool(row["bounded_support"])
                and np.isfinite(intervals).all()
            )
            width = int(row["width"])
            required_rank = 0.70 * width
            passed = (
                supported
                and intervals[0, 1] <= 0.25
                and intervals[1, 0] >= required_rank
            )
            output.append(GroupDiagnostics(
                members=tuple(row["members"]),
                n=int(row["n"]),
                source_groups=source_groups,
                max_conditional_residual_correlation=float(point[0]),
                max_correlation_ci=(float(intervals[0, 0]), float(intervals[0, 1])),
                effective_rank=float(point[1]),
                effective_rank_ci=(float(intervals[1, 0]), float(intervals[1, 1])),
                required_effective_rank=required_rank,
                status=(
                    DependenceStatus.INDEPENDENCE_COMPATIBLE
                    if passed else DependenceStatus.UNRESOLVED
                ),
                bootstrap_draws=int(draws),
                finite_draws=tuple(row["finite_draws"]),
            ))
        return output


def run_atlas_dependence(
    *,
    extract_root,
    predictor_root,
    reconciliation_root,
    contract_root,
    output_root,
    draws: int = 10_000,
    seed: int = 39_615,
):
    """Build the real, typed Atlas error matrices and compatibility graph.

    Extraction and predictor artifacts must already be frozen and complete.
    Correctness annotations are opened here, never by the registry, extractor,
    predictor, or fusion-weight routines.
    """
    if int(draws) <= 0:
        raise ValueError("draws must be positive")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = _load_atlas_inputs(
        Path(extract_root), Path(predictor_root), Path(reconciliation_root), Path(contract_root),
    )
    answer_metadata = _answer_metadata(inputs)
    incumbent_gate_open, incumbent_gate_error = _incumbent_gate(inputs, answer_metadata)
    if incumbent_gate_error is not None:
        raise AtlasBackendError(
            "family-cap screening requires the frozen current gate: " + incumbent_gate_error
        )
    definitions = _candidate_definitions(inputs)
    screening_rows, selected = _screen_candidates(
        inputs, definitions, incumbent_gate=incumbent_gate_open,
    )
    consolidated = _consolidate_selected(inputs, selected, output_root)
    screened = {row["name"]: row for row in screening_rows}
    eligible_rows = []
    for index, definition in enumerate(consolidated.definitions):
        row = dict(screened[definition.name])
        row.update({"cost": float(consolidated.costs[index]), "exact_duplicate": False})
        eligible_rows.append(row)
    pre_cap_aliases = {
        str(row["name"]): str(row["canonical"])
        for row in screening_rows if row.get("status") == "EXACT_DUPLICATE"
    }
    all_aliases = {**pre_cap_aliases, **dict(consolidated.aliases)}
    for alias, canonical in all_aliases.items():
        if alias in screened:
            screened[alias]["status"] = "EXACT_DUPLICATE"
            screened[alias]["canonical"] = canonical
    reconciliation = _read_json(Path(reconciliation_root) / "LEDGER.json")
    inventory_keys = ("historical_unique_score_arrays", "historical_unique_peak_vectors")
    missing_inventory = [key for key in inventory_keys if key not in reconciliation]
    if missing_inventory:
        raise AtlasBackendError(
            "reconciliation ledger lacks frozen inventory counts: " + ", ".join(missing_inventory)
        )
    historical_columns = []
    for row in reconciliation.get("columns", ()):
        historical_columns.append({
            "name": str(row.get("name", "")),
            "archive": str(row.get("archive", "")),
            "status": "REPORT_ONLY",
            "artifact_status": str(row.get("artifact_status", "REPORT_ONLY")),
            "score_sha256": row.get("score_sha256"),
            "peak_sha256": row.get("peak_sha256"),
            "canonical_score": row.get("canonical_score"),
            "exact_score_duplicate": bool(row.get("exact_score_duplicate", False)),
            "peak_verified": bool(row.get("peak_verified", False)),
            "included_historically": bool(row.get("included", False)),
            "control_tagged": bool(row.get("control_tagged", False)),
            "promotion_eligible": False,
            "reason": (
                "historical ledger does not bind a registered insertion point, "
                "resolution, family and reproducible extraction recipe"
            ),
        })
    declared_columns = reconciliation.get("scored_columns")
    if declared_columns is not None and int(declared_columns) != len(historical_columns):
        raise AtlasBackendError(
            "historical reconciliation column count disagrees with scored_columns"
        )
    available_score_columns = int(reconciliation.get("available_score_columns", 0))
    represented_score_columns = sum(row["score_sha256"] is not None for row in historical_columns)
    if historical_columns and represented_score_columns != available_score_columns:
        raise AtlasBackendError(
            "historical reconciliation available-score count disagrees with column identities"
        )
    represented_unique_scores = len({
        row["score_sha256"] for row in historical_columns if row["score_sha256"] is not None
    })
    declared_unique_scores = reconciliation.get("available_unique_score_arrays")
    if (
        declared_unique_scores is not None
        and represented_unique_scores != int(declared_unique_scores)
    ):
        raise AtlasBackendError(
            "historical reconciliation unique-score count disagrees with column identities"
        )
    represented_verified_peaks = len({
        row["peak_sha256"]
        for row in historical_columns
        if row["peak_verified"] and row["peak_sha256"] is not None
    })
    declared_verified_peaks = reconciliation.get("available_verified_peak_vectors")
    if (
        declared_verified_peaks is not None
        and represented_verified_peaks != int(declared_verified_peaks)
    ):
        raise AtlasBackendError(
            "historical reconciliation verified-peak count disagrees with column identities"
        )
    candidate_payload = {
        "schema": "fusion-independence-atlas-v1/candidates-v1",
        "eligible": eligible_rows,
        "aliases": all_aliases,
        "screening": screening_rows,
        "family_cap": 2,
        "family_cap_scope": "independently inside each of five outer-training partitions",
        "held_fold_labels_used_for_its_roster": False,
        "exact_dedup_before_pairwise": True,
        "historical_inventory": {
            "unique_score_arrays": reconciliation["historical_unique_score_arrays"],
            "unique_peak_vectors": reconciliation["historical_unique_peak_vectors"],
            "available_archives": reconciliation.get("available_archives", 0),
            "report_only_archives": reconciliation.get("report_only_archives", 0),
            "represented_columns": len(historical_columns),
            "represented_available_unique_score_arrays": represented_unique_scores,
            "represented_available_verified_peak_vectors": represented_verified_peaks,
            "policy": "Only reconstructible frozen arrays are eligible; missing bulk remains REPORT_ONLY.",
        },
        "historical_report_only": historical_columns,
    }
    _write_json(output_root / "CANDIDATES.json", candidate_payload)

    gate_candidate_indexes = _point_indexes(consolidated, "answer_gate")
    digit025_peaks = (
        consolidated.peaks[:, gate_candidate_indexes[0]]
        if len(gate_candidate_indexes)
        else np.full(inputs.answer_count, -1, dtype=np.int64)
    )
    digit025_within = _digit025_within_values(inputs)
    evaluation_path = output_root / "EVALUATION.npz"
    evaluation_tmp = evaluation_path.with_suffix(".npz.tmp")
    with evaluation_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            target=np.asarray(inputs.joined["target"], dtype=np.int32),
            labels=np.asarray(inputs.joined["labels"], dtype=np.int8),
            offsets=inputs.offsets,
            folds=np.asarray([row["fold"] for row in inputs.metadata], dtype=np.int8),
            cells=answer_metadata.cells,
            groups=answer_metadata.groups,
            tokens=answer_metadata.tokens.astype(np.int32),
            steps=answer_metadata.steps.astype(np.int32),
            incumbent_gate_open=incumbent_gate_open,
            digit025_peaks=digit025_peaks.astype(np.int32),
            digit025_within=digit025_within,
        )
    os.replace(evaluation_tmp, evaluation_path)
    # Token-scale predictor residuals are intentionally not retained here.
    # They are opened and analysed one primitive at a time below; operational
    # answer/pair matrices are small enough to keep as graph references.
    operational_matrices: dict[str, ErrorMatrix] = {}
    for point in ("background", "token_pre_readout", "step_post_readout", "decoder"):
        raw = _pb_raw_matrix(inputs, consolidated, answer_metadata, point)
        pair = _prmb_pair_matrix(inputs, consolidated, point)
        if raw is not None:
            operational_matrices[f"{point}:pb_raw_locator_miss"] = raw
        if pair is not None:
            operational_matrices[f"{point}:prmb_pairwise_misorder"] = pair

    gate_indexes = _point_indexes(consolidated, "answer_gate")
    gate_bootstrap = None
    if len(gate_indexes):
        gate_names = tuple(consolidated.names[index] for index in gate_indexes)
        fusion_sets = {name: (name,) for name in gate_names}
        target = np.asarray(inputs.joined["target"], dtype=np.int64)
        pb = np.char.startswith(answer_metadata.cells.astype(str), "pb_")
        erroneous = pb & (target >= 0)
        gate_errors = gate_error_matrices_from_scores(
            consolidated.answer_scores[:, gate_indexes], gate_names, fusion_sets,
            erroneous, answer_metadata, q=0.33,
            detector_active=consolidated.answer_active[:, gate_indexes], processbench_mask=pb,
            retain_inactive_as_closed=True,
        )
        operational_matrices["answer_gate:gate_false_open"] = gate_errors.false_open
        operational_matrices["answer_gate:gate_false_close"] = gate_errors.false_close
        prediction = np.where(gate_errors.gate_open, consolidated.peaks[:, gate_indexes], -1)
        # An inactive detector is a defined closed gate.  A locator is needed
        # only on rows the gate actually opens.
        final_valid = (~gate_errors.gate_open) | (
            gate_errors.active & (consolidated.peaks[:, gate_indexes] >= 0)
        )
        operational_matrices["answer_gate:pb_final"] = final_pb_error_matrix(
            prediction, target, final_valid, gate_names, answer_metadata,
            scores=gate_errors.gate_percentiles, active=final_valid,
            retain_invalid_as_failures=True,
        )
        try:
            gate_bootstrap_result = full_population_gate_bootstrap(
                consolidated.answer_scores[:, gate_indexes], gate_names, fusion_sets,
                erroneous, answer_metadata,
                lambda errors: np.r_[
                    errors.false_open.values.mean(axis=0), errors.false_close.values.mean(axis=0),
                ],
                q=0.33, detector_active=consolidated.answer_active[:, gate_indexes],
                processbench_mask=pb, draws=draws, seed=seed,
                retain_inactive_as_closed=True,
            )
            intervals = simultaneous_intervals(
                gate_bootstrap_result.statistics.point,
                gate_bootstrap_result.statistics.samples,
                confidence=gate_bootstrap_result.statistics.confidence,
            )
            gate_bootstrap = {
                "rerank_within_every_draw": True,
                "threshold": 0.33,
                "draws": draws,
                "point": gate_bootstrap_result.statistics.point,
                "simultaneous_intervals": intervals,
            }
        except (ValueError, FloatingPointError) as error:
            gate_bootstrap = {
                "rerank_within_every_draw": True,
                "threshold": 0.33,
                "draws": draws,
                "error": f"{type(error).__name__}: {error}",
            }

    required_matrices = {
        *(f"background:{primitive}:predictor_residual" for primitive in _PRIMITIVE_NAMES),
        *(
            f"{point}:{target}"
            for point in ("background", "token_pre_readout", "step_post_readout", "decoder")
            for target in ("pb_raw_locator_miss", "prmb_pairwise_misorder")
        ),
        "answer_gate:gate_false_open",
        "answer_gate:gate_false_close",
        "answer_gate:pb_final",
    }
    predictor_keys = {
        f"background:{primitive}:predictor_residual" for primitive in _PRIMITIVE_NAMES
    }
    missing_matrices = sorted(
        required_matrices - (set(operational_matrices) | predictor_keys)
    )
    if missing_matrices:
        raise AtlasBackendError(
            "dependence matrix construction is incomplete: " + ", ".join(missing_matrices)
        )

    key_order = tuple(sorted(required_matrices))
    key_position = {key: position for position, key in enumerate(key_order)}
    rows_by_matrix: dict[str, list[dict]] = {}
    statuses_by_matrix: dict[str, Mapping[tuple[str, str], DependenceStatus]] = {}
    outer_statuses_by_fold: dict[
        int, dict[str, Mapping[tuple[str, str], DependenceStatus]]
    ] = {outer: {} for outer in range(5)}
    outer_rows_by_fold: dict[int, dict[str, list[dict]]] = {
        outer: {} for outer in range(5)
    }
    bootstrap_errors: dict[str, str] = {}
    matrix_ledger_rows: list[dict] = []
    seen_matrices: set[str] = set()

    def analyse_matrix(key: str, matrix: ErrorMatrix) -> None:
        if key in seen_matrices or key not in key_position:
            raise AtlasBackendError(f"unexpected or duplicate dependence matrix {key}")
        seen_matrices.add(key)
        position = key_position[key]
        matrix_ledger_rows.append({
            "name": key,
            "target": matrix.target.value,
            "resolution": matrix.resolution.value,
            "observations": matrix.shape[0],
            "methods": matrix.shape[1],
            "active_observations": int(matrix.active.sum()),
            "separate_target": True,
        })
        rows, statuses, error = _analyse_error_matrix(
            key, matrix, draws=int(draws), seed=int(seed) + position,
        )
        rows_by_matrix[key] = rows
        statuses_by_matrix[key] = statuses
        if error is not None:
            bootstrap_errors[key] = error
        if matrix.metadata.folds is None:
            raise AtlasBackendError(f"matrix {key} lacks source folds for nested compatibility")
        matrix_folds = np.asarray(matrix.metadata.folds)
        for outer in range(5):
            keep = matrix_folds != outer
            if not keep.any():
                raise AtlasBackendError(f"outer fold {outer} removes every row of {key}")
            subset = matrix.take(keep)
            local_rows, local_statuses, _ = _analyse_error_matrix(
                key, subset, draws=int(draws),
                seed=int(seed) + 10_000 + outer * 101 + position,
            )
            outer_rows_by_fold[outer][key] = local_rows
            outer_statuses_by_fold[outer][key] = local_statuses

    # Predictor matrices dominate memory.  Analyse all six backgrounds for one
    # primitive, including its five outer-training subsets, then release it
    # before materialising the next primitive.
    for primitive, predictor_matrix in _iter_predictor_residual_matrices(inputs):
        analyse_matrix(
            f"background:{primitive}:predictor_residual", predictor_matrix,
        )
        del predictor_matrix
    for key, matrix in sorted(operational_matrices.items()):
        analyse_matrix(key, matrix)
    missing_analyses = sorted(required_matrices - seen_matrices)
    if missing_analyses:
        raise AtlasBackendError(
            "dependence analysis is incomplete: " + ", ".join(missing_analyses)
        )
    pair_rows = [row for key in key_order for row in rows_by_matrix[key]]
    outer_pair_rows = [
        {"outer": outer, **row}
        for outer in range(5)
        for key in key_order
        for row in outer_rows_by_fold[outer][key]
    ]

    # Predictor-residual matrices are naturally named by background kind,
    # whereas downstream background experts are qualified by primitive and
    # readout.  Project the residual status onto compatible candidate pairs so
    # a background edge must pass both same-target predictor-residual and
    # downstream operational-error screens.
    background_names = [
        name for name, point in zip(consolidated.names, consolidated.points)
        if point == "background"
    ]
    kind_alias = {"bocpd_hazard_1_32": "bocpd"}
    for primitive in _PRIMITIVE_NAMES:
        matrix_key = f"background:{primitive}:predictor_residual"
        native_status = statuses_by_matrix.get(matrix_key, {})
        projected = {}
        parsed = {name: _parse_background_member(name) for name in background_names}
        local_names = [name for name, value in parsed.items() if value[0] == primitive]
        for left_index, left in enumerate(local_names):
            for right in local_names[left_index + 1:]:
                left_kind = kind_alias.get(parsed[left][1], parsed[left][1])
                right_kind = kind_alias.get(parsed[right][1], parsed[right][1])
                pair = (left_kind, right_kind)
                reverse = (right_kind, left_kind)
                if pair in native_status:
                    projected[(left, right)] = native_status[pair]
                elif reverse in native_status:
                    projected[(left, right)] = native_status[reverse]
        statuses_by_matrix[f"background:projected_predictor_residual:{primitive}"] = projected

    graph_payload: dict[str, dict] = {}
    group_rows: list[dict] = []
    group_preflights: list[dict] = []
    reference_by_point: dict[str, ErrorMatrix] = {}
    for key, matrix in operational_matrices.items():
        point = key.split(":", 1)[0]
        preference = (
            "pb_raw_locator_miss" in key
            or (point == "answer_gate" and key.endswith("pb_final"))
            or (point == "background" and "H0lim" in key)
        )
        if point not in reference_by_point or preference:
            reference_by_point[point] = matrix
    for point, reference in sorted(reference_by_point.items()):
        names = reference.method_names
        graph, combined = _combined_point_graph(point, names, statuses_by_matrix)
        for pair in tuple(combined):
            if not _semantic_group_is_legal(point, pair):
                combined[pair] = DependenceStatus.UNRESOLVED
        graph = build_compatibility_graph(names, combined)
        edges = [
            [left, right]
            for left in sorted(graph)
            for right in sorted(graph[left])
            if left < right
        ]
        graph_payload[point] = {
            "nodes": list(names),
            "edges": edges,
            "pair_statuses": [
                {"left": pair[0], "right": pair[1], "status": status.value}
                for pair, status in sorted(combined.items())
            ],
        }
        cliques = _enumerate_cliques_complete(
            graph, min_size=2, max_size=6, max_results=100_000,
        )
        if cliques:
            group_preflights.append({
                "scope": "full", "point": point,
                **clique_diagnostics_preflight(
                    len(cliques), int(draws),
                    int(np.unique(reference.metadata.groups).size),
                ),
            })
            conditioned = cross_fit_nuisance(reference, n_splits=5, seed=seed)
            for diagnostics in _common_group_diagnostics(
                reference, cliques, conditioned=conditioned,
                draws=int(draws), seed=int(seed),
            ):
                group_rows.append({"point": point, **_json_ready(diagnostics)})

    # Nested roster construction receives a separate compatibility graph for
    # every outer training partition.  The held outer fold is removed before
    # nuisance fitting, pair bootstrap, and group screening, so mutating held
    # labels cannot create or remove a candidate edge for that fold.
    outer_graph_payload: dict[str, dict[str, dict]] = {}
    outer_group_rows: list[dict] = []
    for outer in range(5):
        outer_statuses = outer_statuses_by_fold[outer]
        for primitive in _PRIMITIVE_NAMES:
            native_status = outer_statuses.get(
                f"background:{primitive}:predictor_residual", {},
            )
            projected = {}
            parsed = {name: _parse_background_member(name) for name in background_names}
            local_names = [name for name, value in parsed.items() if value[0] == primitive]
            for left_index, left in enumerate(local_names):
                for right in local_names[left_index + 1:]:
                    left_kind = kind_alias.get(parsed[left][1], parsed[left][1])
                    right_kind = kind_alias.get(parsed[right][1], parsed[right][1])
                    if (left_kind, right_kind) in native_status:
                        projected[(left, right)] = native_status[(left_kind, right_kind)]
                    elif (right_kind, left_kind) in native_status:
                        projected[(left, right)] = native_status[(right_kind, left_kind)]
            outer_statuses[f"background:projected_predictor_residual:{primitive}"] = projected

        fold_graphs = {}
        for point, full_reference in sorted(reference_by_point.items()):
            if full_reference.metadata.folds is None:
                raise AtlasBackendError(f"reference matrix for {point} lacks source folds")
            keep = np.asarray(full_reference.metadata.folds) != outer
            reference = full_reference.take(keep)
            graph, combined = _combined_point_graph(
                point, reference.method_names, outer_statuses,
            )
            for pair in tuple(combined):
                if not _semantic_group_is_legal(point, pair):
                    combined[pair] = DependenceStatus.UNRESOLVED
            graph = build_compatibility_graph(reference.method_names, combined)
            edges = [
                [left, right]
                for left in sorted(graph)
                for right in sorted(graph[left]) if left < right
            ]
            fold_graphs[point] = {
                "nodes": list(reference.method_names), "edges": edges,
                "pair_statuses": [
                    {"left": pair[0], "right": pair[1], "status": status.value}
                    for pair, status in sorted(combined.items())
                ],
                "held_outer_fold_excluded": outer,
            }
            cliques = _enumerate_cliques_complete(
                graph, min_size=2, max_size=6, max_results=100_000,
            )
            if cliques:
                group_preflights.append({
                    "scope": "outer_training", "outer": outer, "point": point,
                    **clique_diagnostics_preflight(
                        len(cliques), int(draws),
                        int(np.unique(reference.metadata.groups).size),
                    ),
                })
                conditioned = cross_fit_nuisance(reference, n_splits=5, seed=seed + outer)
                for diagnostics in _common_group_diagnostics(
                    reference, cliques, conditioned=conditioned,
                    draws=int(draws), seed=int(seed) + 20_000 + outer,
                ):
                    outer_group_rows.append({
                        "outer": outer, "point": point, **_json_ready(diagnostics),
                    })
        outer_graph_payload[str(outer)] = fold_graphs

    error_ledger = {
        "schema": "fusion-independence-atlas-v1/error-ledger-v1",
        "matrices": sorted(matrix_ledger_rows, key=lambda row: str(row["name"])),
        "gate_bootstrap": gate_bootstrap,
        "incumbent_gate": {
            "name": "equal_rank_tail15_digit_rate",
            "locator": "digit025",
            "available": incumbent_gate_error is None,
            "error": incumbent_gate_error,
        },
        "bootstrap_errors": bootstrap_errors,
        "unsupported": {},
        "five_point_inputs_ready": True,
        "labels_opened_only_downstream": True,
    }
    _write_json(output_root / "ERROR_LEDGER.json", error_ledger)
    _write_json(output_root / "PAIRWISE.json", {
        "schema": "fusion-independence-atlas-v1/pairwise-v1", "rows": pair_rows,
        "outer_training_rows": outer_pair_rows,
        "draws": int(draws), "simultaneous": True,
    })
    heatmap_cells: dict[str, list[dict]] = {}
    for row in pair_rows:
        heatmap_cells.setdefault(str(row["matrix"]), []).append({
            "left": row["left"], "right": row["right"],
            "conditional_phi": row.get("conditional_phi"),
            "odds_ratio": row.get("conditional_odds_ratio"),
            "score_spearman": row.get("score_spearman"),
            "topk_jaccard": row.get("topk_jaccard"),
            "peak_agreement": row.get("peak_agreement"),
            "status": row.get("status"),
        })
    _write_json(output_root / "HEATMAPS.json", {
        "schema": "fusion-independence-atlas-v1/pairwise-heatmap-data-v1",
        "matrices": heatmap_cells,
        "rendering": "symmetric; diagonal is identity and omitted",
    })
    _write_json(output_root / "COMPATIBILITY_GRAPH.json", {
        "schema": "fusion-independence-atlas-v1/compatibility-graph-v1",
        "graphs": graph_payload, "outer_training_graphs": outer_graph_payload,
        "maximum_clique_size": 6,
    })
    _write_json(output_root / "GROUPS.json", {
        "schema": "fusion-independence-atlas-v1/groups-v1", "rows": group_rows,
        "outer_training_rows": outer_group_rows,
        "preflight": group_preflights,
        "enumeration": "complete_or_fail_closed",
        "criteria": {"max_upper_correlation": 0.25, "min_lower_effective_rank_fraction": 0.70},
    })
    return {
        "status": "COMPLETE",
        "candidates": len(consolidated.names),
        "exact_duplicates": len(all_aliases),
        "matrices": len(matrix_ledger_rows),
        "pairwise_rows": len(pair_rows),
        "groups": len(group_rows),
        "gate_reranked_within_draw": bool(gate_bootstrap),
        "unsupported_points": error_ledger["unsupported"],
    }


def _block_row_plan(block_ids: np.ndarray) -> tuple[slice | np.ndarray, ...]:
    """Build a reusable exact row plan without scanning all rows per block."""
    blocks = _one_dimensional(block_ids, "block_ids")
    if not len(blocks):
        return ()
    boundaries = np.flatnonzero(blocks[1:] != blocks[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(blocks)]
    segment_keys = blocks[starts]
    # Answer-local rows are already contiguous in all token/step Atlas paths.
    # Preserve that zero-copy fast path while retaining exact support for
    # noncontiguous cell/block layouts used by gates and unit tests.
    if len(np.unique(segment_keys)) == len(segment_keys):
        return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
    _, inverse = np.unique(blocks, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    grouped = np.r_[0, np.cumsum(counts)]
    return tuple(
        order[int(grouped[index]):int(grouped[index + 1])]
        for index in range(len(counts))
    )


def _rank_one_fusion_column(
    scores: np.ndarray,
    active: np.ndarray,
    row_plan: Sequence[slice | np.ndarray],
    output: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Rank one column into ``output`` using a precomputed block plan."""
    values = _one_dimensional(scores, "scores")
    mask = _one_dimensional(active, "active", len(values)).astype(bool, copy=True)
    mask &= np.isfinite(values)
    output[:] = 0.0
    informative = False
    for block_rows in row_plan:
        local_active = mask[block_rows]
        if not local_active.any():
            continue
        if isinstance(block_rows, slice):
            start = int(block_rows.start or 0)
            rows = start + np.flatnonzero(local_active)
        else:
            rows = block_rows[np.flatnonzero(local_active)]
        output[rows] = (rankdata(values[rows], method="average") - 0.5) / len(rows)
        informative |= len(rows) > 1 and np.ptp(values[rows]) > 1e-12
    return mask, informative


def _rank_columns_for_fusion(
    scores: np.ndarray,
    active_mask: np.ndarray,
    block_ids: Sequence | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool).copy()
    if scores.ndim != 2 or active.shape != scores.shape:
        raise ValueError("fusion scores and active mask must be aligned matrices")
    ranks = np.zeros_like(scores)
    live = np.zeros(scores.shape[1], dtype=bool)
    if block_ids is None:
        blocks = np.zeros(len(scores), dtype=np.int64)
    else:
        blocks = _one_dimensional(block_ids, "block_ids", len(scores))
    row_plan = _block_row_plan(blocks)
    for column in range(scores.shape[1]):
        active[:, column], live[column] = _rank_one_fusion_column(
            scores[:, column], active[:, column], row_plan, ranks[:, column],
        )
    return ranks, active, live


def _prepared_fusion_ranks(
    scores: np.ndarray,
    active: np.ndarray,
    block_ids: Sequence | None,
    precomputed_ranks: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if precomputed_ranks is None:
        return _rank_columns_for_fusion(scores, active, block_ids)
    ranks = np.asarray(precomputed_ranks, dtype=np.float64)
    if ranks.shape != scores.shape:
        raise ValueError("precomputed ranks must align with fusion scores")
    ranked_active = np.asarray(active, dtype=bool).copy()
    ranked_active &= np.isfinite(scores) & np.isfinite(ranks)
    ranks = np.where(ranked_active, ranks, 0.0)
    # A constant block (including a singleton) has normalized midrank .5.
    # Any nonconstant block necessarily contains at least one other rank.
    live = np.any(ranked_active & (np.abs(ranks - 0.5) > 1e-12), axis=0)
    return ranks, ranked_active, live


def fit_label_free_fusion_weights(
    scores,
    active_mask,
    families: Sequence[str],
    head: str,
    *,
    block_ids: Sequence | None = None,
    precomputed_ranks=None,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Fit one fusion head from score covariance only.

    The API intentionally cannot receive benchmark annotations.  Nested OOF
    selection invokes it before opening the evaluator labels for that split.
    """
    values = np.asarray(scores, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    family = tuple(str(value) for value in families)
    if values.ndim != 2 or active.shape != values.shape or len(family) != values.shape[1]:
        raise ValueError("fusion training arrays/families are not aligned")
    ranks, ranked_active, live = _prepared_fusion_ranks(
        values, active, block_ids, precomputed_ranks,
    )
    weights = np.zeros(values.shape[1], dtype=np.float64)
    diagnostics: dict[str, object] = {"head": str(head), "live_members": int(live.sum())}
    if not live.any():
        return weights, {**diagnostics, "status": "NO_LIVE_MEMBERS"}
    if head == "singleton":
        if values.shape[1] != 1:
            raise ValueError("singleton head requires exactly one member")
        weights[0] = 1.0
    elif head == "equal_rank":
        weights[live] = 1.0 / int(live.sum())
    elif head == "family_equal":
        active_families = tuple(dict.fromkeys(family[index] for index in np.flatnonzero(live)))
        for name in active_families:
            indexes = np.asarray([live[index] and family[index] == name for index in range(len(family))])
            weights[indexes] = 1.0 / (len(active_families) * int(indexes.sum()))
    elif head == "nonnegative_shrunk_simplex":
        # Fit in the same answer/cell-local rank geometry used at application
        # time.  Calling the public registry head here would rerank globally
        # and silently discard ``block_ids``.
        from spectral_utils.fusion_signal_registry import (
            _ledoit_wolf_diagonal, _minimum_variance_simplex,
        )
        indexes = np.flatnonzero(live)
        if len(indexes) == 1:
            weights[indexes[0]] = 1.0
            diagnostics["shrinkage_alpha"] = 0.0
        else:
            z = np.zeros((len(values), len(indexes)), dtype=np.float64)
            for local, column in enumerate(indexes):
                rows = ranked_active[:, column]
                centered = ranks[rows, column] - ranks[rows, column].mean()
                z[rows, local] = centered / max(float(centered.std()), 1e-12)
            covariance = z.T @ z / max(len(z), 1)
            shrunk, alpha = _ledoit_wolf_diagonal(z, covariance)
            weights[indexes] = _minimum_variance_simplex(shrunk)
            diagnostics["shrinkage_alpha"] = float(alpha)
    elif head == "iu":
        indexes = np.flatnonzero(live)
        z = np.zeros((len(values), len(indexes)), dtype=np.float64)
        for local, column in enumerate(indexes):
            rows = ranked_active[:, column]
            centered = ranks[rows, column] - ranks[rows, column].mean()
            z[rows, local] = centered / max(float(centered.std()), 1e-12)
        covariance = z.T @ z / max(len(z), 1)
        inverse = np.linalg.pinv(covariance, rcond=1e-8)
        local_weights = inverse @ np.ones(len(indexes))
        denominator = float(local_weights.sum())
        if abs(denominator) <= 1e-12 or not np.isfinite(local_weights).all():
            local_weights = np.full(len(indexes), 1.0 / len(indexes))
            diagnostics["fallback"] = "equal_rank"
        else:
            local_weights /= denominator
        weights[indexes] = local_weights
        diagnostics.update({
            "negative_weights": int(np.sum(local_weights < -1e-9)),
            "condition_number": float(np.linalg.cond(covariance)) if len(indexes) > 1 else 1.0,
        })
    else:
        raise ValueError(f"unknown fusion head {head}")
    diagnostics["weights_sum"] = float(weights.sum())
    diagnostics["status"] = "FIT"
    return weights, diagnostics


def apply_label_free_fusion_weights(
    scores,
    active_mask,
    fitted_weights,
    *,
    block_ids: Sequence | None = None,
    families: Sequence[str] | None = None,
    head: str | None = None,
    precomputed_ranks=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply already-fitted weights to rank-normalized score columns."""
    values = np.asarray(scores, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    weights = np.asarray(fitted_weights, dtype=np.float64)
    if values.ndim != 2 or active.shape != values.shape or weights.shape != (values.shape[1],):
        raise ValueError("fusion application arrays are not aligned")
    ranks, ranked_active, _ = _prepared_fusion_ranks(
        values, active, block_ids, precomputed_ranks,
    )
    if head == "family_equal":
        if families is None or len(families) != values.shape[1]:
            raise ValueError("family_equal application requires aligned families")
        family_names = tuple(dict.fromkeys(str(value) for value in families))
        family_values = np.zeros((len(values), len(family_names)), dtype=np.float64)
        family_active = np.zeros_like(family_values, dtype=bool)
        for family_index, family in enumerate(family_names):
            columns = np.asarray([str(value) == family for value in families], dtype=bool)
            count = ranked_active[:, columns].sum(axis=1)
            family_active[:, family_index] = count > 0
            family_values[family_active[:, family_index], family_index] = (
                (ranks[:, columns] * ranked_active[:, columns]).sum(axis=1)[family_active[:, family_index]]
                / count[family_active[:, family_index]]
            )
        family_count = family_active.sum(axis=1)
        available = family_count > 0
        output = np.zeros(len(values), dtype=np.float64)
        output[available] = (
            (family_values * family_active).sum(axis=1)[available] / family_count[available]
        )
        return output, available
    denominator = ranked_active @ weights
    available = np.abs(denominator) > 1e-12
    output = np.zeros(len(values), dtype=np.float64)
    output[available] = (
        (ranks[available] * ranked_active[available] * weights).sum(axis=1)
        / denominator[available]
    )
    return output, available


def _evaluation_arrays(dependence_root: Path) -> Mapping[str, np.ndarray]:
    path = Path(dependence_root) / "EVALUATION.npz"
    if not path.is_file():
        raise AtlasBackendError("dependence EVALUATION.npz is missing")
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]).copy() for name in saved.files}


@dataclass(frozen=True)
class _NestedArtifacts:
    metadata: tuple[Mapping, ...]
    spans: np.ndarray
    token_values: np.ndarray
    token_active: np.ndarray
    token_candidate_map: Mapping[str, Mapping]
    token_answer_ids: np.ndarray
    backgrounds: _BackgroundArrays
    token_ranks: np.ndarray | None = None
    step_ranks: np.ndarray | None = None


@dataclass(frozen=True)
class _FusionGeometry:
    """Label-free row geometry exposed to fusion-weight fitting only."""

    offsets: np.ndarray
    cells: np.ndarray


def _load_nested_artifacts(
    dependence_root: Path,
    extraction_root: Path,
    evaluation: Mapping[str, np.ndarray],
) -> _NestedArtifacts:
    extraction_root = Path(extraction_root)
    manifest = _read_json(extraction_root / "MANIFEST.json")
    bundle_raw = manifest.get("bundle_root")
    if not bundle_raw:
        raise AtlasBackendError("extraction manifest does not bind bundle_root")
    bundle_root = Path(bundle_raw)
    if not bundle_root.is_absolute():
        bundle_root = extraction_root / bundle_root
    expected_freeze = manifest.get("bundle_freeze_sha256")
    if expected_freeze is not None:
        freeze_path = bundle_root / "FREEZE.json"
        if not freeze_path.is_file() or _file_sha256(freeze_path) != str(expected_freeze):
            raise AtlasBackendError("nested extraction/bundle FREEZE binding has drifted")
    metadata = tuple(_read_json(bundle_root / "METADATA.json"))
    if len(metadata) != len(evaluation["target"]):
        raise AtlasBackendError("nested bundle/evaluation answer count drift")
    spans = np.load(bundle_root / "step_spans.npy", mmap_mode="r", allow_pickle=False)
    token_index_path = dependence_root / "TOKEN_INDEX.json"
    if not token_index_path.is_file():
        raise AtlasBackendError("token-pre-readout search requires TOKEN_INDEX.json")
    token_index = _read_json(token_index_path)
    token_values = np.load(dependence_root / token_index["values"], mmap_mode="r", allow_pickle=False)
    token_active = np.load(dependence_root / token_index["active"], mmap_mode="r", allow_pickle=False)
    if token_values.shape != token_active.shape:
        raise AtlasBackendError("nested raw token value/mask axes drift")
    token_answer_ids = np.empty(len(token_values), dtype=np.int32)
    for answer_index, row in enumerate(metadata):
        start = int(row["offset"]); stop = start + int(row["tokens"])
        token_answer_ids[start:stop] = answer_index
    predictor_root = extraction_root.parent / "predictors"
    predictor_manifest_path = predictor_root / "MANIFEST.json"
    if predictor_manifest_path.is_file():
        predictor_manifest = _read_json(predictor_manifest_path)
        predictor_bundle = Path(predictor_manifest.get("bundle_root", bundle_root))
        if not predictor_bundle.is_absolute():
            predictor_bundle = predictor_root / predictor_bundle
        if predictor_bundle.resolve() != bundle_root.resolve():
            raise AtlasBackendError("nested predictor/extraction bundle binding disagrees")
        if expected_freeze is not None and str(
            predictor_manifest.get("bundle_freeze_sha256")
        ) != str(expected_freeze):
            raise AtlasBackendError("nested predictor/extraction FREEZE hashes disagree")
    proxy = type("_BackgroundInputProxy", (), {
        "bundle_root": bundle_root, "predictor_root": predictor_root,
    })()
    backgrounds = _open_background_arrays(proxy, require_inner=True)
    return _NestedArtifacts(
        metadata=metadata, spans=spans, token_values=token_values,
        token_active=token_active,
        token_candidate_map=token_index.get("candidate_map", {}),
        token_answer_ids=token_answer_ids, backgrounds=backgrounds,
    )


def _materialize_block_rank_cache(
    path: Path,
    values: np.ndarray,
    active: np.ndarray,
    block_ids: np.ndarray,
) -> np.ndarray:
    """Write exact float64 block-local midranks one column at a time."""
    values = np.asarray(values)
    active = np.asarray(active, dtype=bool)
    if values.ndim != 2 or active.shape != values.shape or len(block_ids) != len(values):
        raise AtlasBackendError("rank-cache inputs are not aligned")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    ranked = np.lib.format.open_memmap(
        temporary, mode="w+", dtype="float64", shape=values.shape,
    )
    row_plan = _block_row_plan(np.asarray(block_ids))
    for column in range(values.shape[1]):
        _rank_one_fusion_column(
            values[:, column], active[:, column], row_plan, ranked[:, column],
        )
    ranked.flush()
    del ranked
    os.replace(temporary, path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _attach_nested_rank_caches(
    artifacts: _NestedArtifacts,
    consolidated: _Consolidated,
    evaluation: Mapping[str, np.ndarray],
    output_root: Path,
) -> _NestedArtifacts:
    cache_root = Path(output_root) / "rank_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    token_path = cache_root / "TOKEN_RANKS.npy"
    step_path = cache_root / "STEP_RANKS.npy"
    token_ranks = _materialize_block_rank_cache(
        token_path,
        artifacts.token_values,
        artifacts.token_active,
        artifacts.token_answer_ids,
    )
    offsets = np.asarray(evaluation["offsets"], dtype=np.int64)
    step_blocks = np.repeat(
        np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets),
    )
    step_ranks = _materialize_block_rank_cache(
        step_path,
        consolidated.step_scores,
        consolidated.step_active,
        step_blocks,
    )
    _write_json(cache_root / "MANIFEST.json", {
        "schema": "fusion-independence-atlas-v1/nested-rank-cache-v1",
        "label_free": True,
        "dtype": "float64",
        "token_shape": token_ranks.shape,
        "step_shape": step_ranks.shape,
        "token_members": sorted({
            str(row.get("signal", "")) for row in artifacts.token_candidate_map.values()
        }),
        "step_members": list(consolidated.names),
    })
    return replace(artifacts, token_ranks=token_ranks, step_ranks=step_ranks)


def _answer_token_rows(metadata: Sequence[Mapping], answer_indexes: np.ndarray) -> np.ndarray:
    if not len(answer_indexes):
        return np.empty(0, dtype=np.int64)
    return np.concatenate([
        np.arange(
            int(metadata[index]["offset"]),
            int(metadata[index]["offset"]) + int(metadata[index]["tokens"]),
            dtype=np.int64,
        )
        for index in answer_indexes
    ])


def _step_subset_metrics(
    fused: np.ndarray,
    available: np.ndarray,
    answer_indexes: np.ndarray,
    evaluation: Mapping[str, np.ndarray],
    *,
    include_row_ledger: bool = False,
) -> dict[str, object]:
    offsets = np.asarray(evaluation["offsets"], dtype=np.int64)
    targets = np.asarray(evaluation["target"], dtype=np.int64)
    labels = np.asarray(evaluation["labels"], dtype=np.int8)
    cells = np.asarray(evaluation["cells"]).astype(str)
    fused = np.asarray(fused, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    expected = int(sum(offsets[index + 1] - offsets[index] for index in answer_indexes))
    if fused.shape != (expected,) or available.shape != fused.shape:
        raise AtlasBackendError("dynamic step output is not answer-aligned")
    peaks = np.full(len(answer_indexes), -1, dtype=np.int64)
    answer_active = np.zeros(len(answer_indexes), dtype=bool)
    cursor = 0
    within_sum = 0.0
    within_count = 0
    within_answers = []
    within_values = []
    for local, answer_index in enumerate(answer_indexes):
        width = int(offsets[answer_index + 1] - offsets[answer_index])
        local_active = available[cursor:cursor + width]
        if local_active.any():
            rows = np.flatnonzero(local_active)
            peaks[local] = int(rows[np.argmax(fused[cursor:cursor + width][rows])])
            answer_active[local] = True
        if cells[answer_index].startswith("prmbench_"):
            value = _risk_pair_auc(
                labels[offsets[answer_index]:offsets[answer_index + 1]],
                fused[cursor:cursor + width], local_active,
            )
            if np.isfinite(value):
                within_sum += float(value); within_count += 1
                within_answers.append(int(answer_index)); within_values.append(float(value))
        cursor += width
    gate = np.asarray(evaluation["incumbent_gate_open"], dtype=bool)[answer_indexes]
    prediction = np.where(gate, peaks, -1)
    decision_valid = (~gate) | answer_active
    result = {
        "pb_cells": {}, "within_sum": within_sum, "within_count": within_count,
        "available_answers": int(answer_active.sum()),
    }
    local_cells = cells[answer_indexes]
    local_targets = targets[answer_indexes]
    for cell in sorted(set(local_cells)):
        if not cell.startswith("pb_"):
            continue
        mask = local_cells == cell
        clean = mask & (local_targets == -1)
        error = mask & (local_targets >= 0)
        correct = decision_valid & (prediction == local_targets)
        result["pb_cells"][str(cell)] = {
            "clean_hits": int(np.sum(clean & correct)),
            "clean_total": int(clean.sum()),
            "error_hits": int(np.sum(error & correct)),
            "error_total": int(error.sum()),
            "valid": int(np.sum(mask & decision_valid)),
        }
    if include_row_ledger:
        result.update({
            "_answer_indexes": np.asarray(answer_indexes, dtype=np.int64),
            "_prediction": prediction,
            "_decision_valid": decision_valid,
            "_within_answer_indexes": np.asarray(within_answers, dtype=np.int64),
            "_within_values": np.asarray(within_values, dtype=np.float64),
        })
    return result


def _parse_background_member(name: str) -> tuple[str, str, str]:
    pieces = str(name).split("::")
    if len(pieces) != 4 or pieces[0] != "background":
        raise AtlasBackendError(f"malformed background member {name}")
    return pieces[1], pieces[2], pieces[3]


def _fit_natural_background_weights(
    residuals: np.ndarray,
    active: np.ndarray,
    families: Sequence[str],
    head: str,
) -> tuple[np.ndarray, Mapping[str, object]]:
    values = np.asarray(residuals, dtype=np.float64)
    mask = np.asarray(active, dtype=bool) & np.isfinite(values)
    member_count = values.shape[1]
    weights = np.zeros(member_count, dtype=np.float64)
    live = mask.any(axis=0)
    diagnostics: dict[str, object] = {
        "head": head, "geometry": "natural_primitive_residual",
        "live_members": int(live.sum()),
    }
    if not live.any():
        return weights, {**diagnostics, "status": "NO_LIVE_MEMBERS"}
    indexes = np.flatnonzero(live)
    if head == "singleton":
        if member_count != 1:
            raise ValueError("singleton head requires one background")
        weights[0] = 1.0
    elif head in {"equal_rank", "family_equal"}:
        if head == "equal_rank":
            weights[indexes] = 1.0 / len(indexes)
        else:
            live_families = tuple(dict.fromkeys(str(families[index]) for index in indexes))
            for family in live_families:
                local = np.asarray([
                    live[index] and str(families[index]) == family
                    for index in range(member_count)
                ], dtype=bool)
                weights[local] = 1.0 / (len(live_families) * int(local.sum()))
    elif head in {"nonnegative_shrunk_simplex", "iu"}:
        z = np.zeros((len(values), len(indexes)), dtype=np.float64)
        for local, column in enumerate(indexes):
            rows = mask[:, column]
            centered = values[rows, column] - values[rows, column].mean()
            z[rows, local] = centered / max(float(centered.std()), 1e-12)
        covariance = z.T @ z / max(len(z), 1)
        if head == "nonnegative_shrunk_simplex":
            from spectral_utils.fusion_signal_registry import (
                _ledoit_wolf_diagonal, _minimum_variance_simplex,
            )
            shrunk, alpha = _ledoit_wolf_diagonal(z, covariance)
            local_weights = _minimum_variance_simplex(shrunk)
            diagnostics["shrinkage_alpha"] = float(alpha)
        else:
            inverse = np.linalg.pinv(covariance, rcond=1e-8)
            local_weights = inverse @ np.ones(len(indexes))
            denominator = float(local_weights.sum())
            if abs(denominator) <= 1e-12 or not np.isfinite(local_weights).all():
                local_weights = np.full(len(indexes), 1.0 / len(indexes))
                diagnostics["fallback"] = "equal"
            else:
                local_weights /= denominator
            diagnostics["negative_weights"] = int(np.sum(local_weights < -1e-9))
        weights[indexes] = local_weights
    else:
        raise ValueError(f"unknown background head {head}")
    diagnostics.update({"weights_sum": float(weights.sum()), "status": "FIT"})
    return weights, diagnostics


def _background_training_matrix(
    artifacts: _NestedArtifacts,
    members: Sequence[str],
    answer_indexes: np.ndarray,
    *,
    inner_outer_fold: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    parsed = [_parse_background_member(name) for name in members]
    primitive, readout = parsed[0][0], parsed[0][2]
    if any(item[0] != primitive or item[2] != readout for item in parsed):
        raise AtlasBackendError("background fusion cannot cross primitives or readouts")
    residual_parts = []
    active_parts = []
    for answer_index in answer_indexes:
        row = artifacts.metadata[int(answer_index)]
        token_slice = slice(int(row["offset"]), int(row["offset"]) + int(row["tokens"]))
        columns = []
        masks = []
        for _, kind, _ in parsed:
            observed, prediction, active = _background_tokens(
                artifacts.backgrounds, token_slice, primitive, kind,
                inner_outer_fold=inner_outer_fold,
            )
            columns.append(observed - prediction); masks.append(active)
        residual_parts.append(np.column_stack(columns))
        active_parts.append(np.column_stack(masks))
    return np.concatenate(residual_parts), np.concatenate(active_parts)


def _apply_background_model(
    artifacts: _NestedArtifacts,
    members: Sequence[str],
    families: Sequence[str],
    head: str,
    weights: np.ndarray,
    answer_indexes: np.ndarray,
    *,
    inner_outer_fold: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    from spectral_utils.fusion_signal_registry import readout_steps

    parsed = [_parse_background_member(name) for name in members]
    primitive, readout = parsed[0][0], parsed[0][2]
    if any(item[0] != primitive or item[2] != readout for item in parsed):
        raise AtlasBackendError("background fusion cannot cross primitives or readouts")
    scores = []
    masks = []
    for answer_index in answer_indexes:
        row = artifacts.metadata[int(answer_index)]
        token_start = int(row["offset"]); token_stop = token_start + int(row["tokens"])
        predictions = []
        active_columns = []
        observed = None
        for _, kind, _ in parsed:
            observed, prediction, active = _background_tokens(
                artifacts.backgrounds, slice(token_start, token_stop), primitive, kind,
                inner_outer_fold=inner_outer_fold,
            )
            predictions.append(prediction); active_columns.append(active)
        prediction_matrix = np.column_stack(predictions)
        active_matrix = np.column_stack(active_columns)
        if head == "family_equal":
            family_names = tuple(dict.fromkeys(families))
            family_values = np.zeros((len(observed), len(family_names)))
            family_active = np.zeros_like(family_values, dtype=bool)
            for family_index, family in enumerate(family_names):
                columns = np.asarray([value == family for value in families], dtype=bool)
                count = active_matrix[:, columns].sum(axis=1)
                family_active[:, family_index] = count > 0
                live_rows = family_active[:, family_index]
                family_values[live_rows, family_index] = (
                    (prediction_matrix[:, columns] * active_matrix[:, columns]).sum(axis=1)[live_rows]
                    / count[live_rows]
                )
            count = family_active.sum(axis=1)
            token_active = count > 0
            fused_prediction = np.zeros(len(observed))
            fused_prediction[token_active] = (
                (family_values * family_active).sum(axis=1)[token_active] / count[token_active]
            )
        else:
            denominator = active_matrix @ weights
            token_active = np.abs(denominator) > 1e-12
            fused_prediction = np.zeros(len(observed))
            fused_prediction[token_active] = (
                (prediction_matrix * active_matrix * weights).sum(axis=1)[token_active]
                / denominator[token_active]
            )
        residual = np.asarray(observed) - fused_prediction
        global_spans = np.asarray(
            artifacts.spans[int(row["step_start"]):int(row["step_stop"])], dtype=np.int64,
        )
        local_spans = global_spans - token_start
        step_score, step_active = readout_steps(
            residual, local_spans, readout, active_mask=token_active,
        )
        scores.append(step_score); masks.append(step_active)
    return np.concatenate(scores), np.concatenate(masks)


def _token_training_matrix(
    artifacts: _NestedArtifacts,
    members: Sequence[str],
    answer_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    mappings = [artifacts.token_candidate_map.get(name) for name in members]
    if any(value is None for value in mappings):
        raise AtlasBackendError("token model member is absent from TOKEN_INDEX")
    readout = str(mappings[0]["readout"])
    if any(str(value["readout"]) != readout for value in mappings):
        raise AtlasBackendError("token fusion cannot cross readouts")
    columns = np.asarray([int(value["column"]) for value in mappings], dtype=np.int64)
    rows = _answer_token_rows(artifacts.metadata, answer_indexes)
    ranks = None
    if artifacts.token_ranks is not None:
        ranks = np.asarray(artifacts.token_ranks[np.ix_(rows, columns)], dtype=np.float64)
    values = (
        ranks
        if ranks is not None
        else np.asarray(artifacts.token_values[np.ix_(rows, columns)], dtype=np.float64)
    )
    return (
        values,
        np.asarray(artifacts.token_active[np.ix_(rows, columns)], dtype=bool),
        artifacts.token_answer_ids[rows],
        ranks,
    )


def _apply_token_model(
    artifacts: _NestedArtifacts,
    members: Sequence[str],
    families: Sequence[str],
    head: str,
    weights: np.ndarray,
    answer_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from spectral_utils.fusion_signal_registry import readout_steps

    mappings = [artifacts.token_candidate_map.get(name) for name in members]
    if any(value is None for value in mappings):
        raise AtlasBackendError("token model member is absent from TOKEN_INDEX")
    readout = str(mappings[0]["readout"])
    if any(str(value["readout"]) != readout for value in mappings):
        raise AtlasBackendError("token fusion cannot cross readouts")
    columns = np.asarray([int(value["column"]) for value in mappings], dtype=np.int64)
    scores = []
    masks = []
    for answer_index in answer_indexes:
        row = artifacts.metadata[int(answer_index)]
        start = int(row["offset"]); stop = start + int(row["tokens"])
        active = np.asarray(artifacts.token_active[start:stop, columns], dtype=bool)
        ranks = None
        if artifacts.token_ranks is not None:
            ranks = np.asarray(artifacts.token_ranks[start:stop, columns], dtype=np.float64)
        values = (
            ranks
            if ranks is not None
            else np.asarray(artifacts.token_values[start:stop, columns], dtype=np.float64)
        )
        fused, fused_active = apply_label_free_fusion_weights(
            values, active, weights, block_ids=np.zeros(stop - start, dtype=np.int8),
            families=families, head=head, precomputed_ranks=ranks,
        )
        global_spans = np.asarray(
            artifacts.spans[int(row["step_start"]):int(row["step_stop"])], dtype=np.int64,
        )
        local_spans = global_spans - start
        step_score, step_active = readout_steps(
            fused, local_spans, readout, active_mask=fused_active,
        )
        scores.append(step_score); masks.append(step_active)
    return np.concatenate(scores), np.concatenate(masks)


def _fusion_model_metrics(
    consolidated: _Consolidated,
    member_indexes: np.ndarray,
    weights: np.ndarray,
    evaluation: Mapping[str, np.ndarray],
    answer_indexes: np.ndarray,
    *,
    point: str,
    families: Sequence[str] | None = None,
    head: str | None = None,
    precomputed_step_ranks: np.ndarray | None = None,
    include_row_ledger: bool = False,
) -> dict[str, object]:
    offsets = np.asarray(evaluation["offsets"], dtype=np.int64)
    targets = np.asarray(evaluation["target"], dtype=np.int64)
    labels = np.asarray(evaluation["labels"], dtype=np.int8)
    cells = np.asarray(evaluation["cells"]).astype(str)
    step_rows = np.concatenate([
        np.arange(offsets[index], offsets[index + 1], dtype=np.int64) for index in answer_indexes
    ]) if len(answer_indexes) else np.empty(0, dtype=np.int64)
    result = {"pb_cells": {}, "within_sum": 0.0, "within_count": 0, "available_answers": 0}

    def record_pb(prediction: np.ndarray, valid: np.ndarray) -> None:
        local_cells = cells[answer_indexes]
        local_targets = targets[answer_indexes]
        for cell in sorted(set(local_cells)):
            if not str(cell).startswith("pb_"):
                continue
            mask = local_cells == cell
            clean = mask & (local_targets == -1)
            error = mask & (local_targets >= 0)
            correct = valid & (prediction == local_targets)
            result["pb_cells"][str(cell)] = {
                "clean_hits": int(np.sum(clean & correct)),
                "clean_total": int(clean.sum()),
                "error_hits": int(np.sum(error & correct)),
                "error_total": int(error.sum()),
                "valid": int(np.sum(mask & valid)),
            }
    if point == "answer_gate":
        # Gate percentiles are a full-population, within-cell label-free
        # transform.  Restricting the rank population to a held fold would
        # silently change the frozen .33 decision rule.
        fused_all, available_all = apply_label_free_fusion_weights(
            consolidated.answer_scores[:, member_indexes],
            consolidated.answer_active[:, member_indexes], weights,
            block_ids=cells, families=families, head=head,
        )
        opened_all, _, gate_active_all = rank_fused_gate_decisions(
            fused_all[:, None], ("fused",), cells, {"gate": ("fused",)},
            q=0.33, active=available_all[:, None],
        )
        peaks = np.asarray(evaluation["digit025_peaks"], dtype=np.int64)[answer_indexes]
        prediction = np.where(opened_all[answer_indexes, 0], peaks, -1)
        opened = opened_all[answer_indexes, 0]
        valid = (~opened) | (gate_active_all[answer_indexes, 0] & (peaks >= 0))
        record_pb(prediction, valid)
        result["available_answers"] = int(valid.sum())
        baseline_within = np.asarray(evaluation["digit025_within"], dtype=np.float64)
        local_within = baseline_within[answer_indexes]
        finite = np.isfinite(local_within)
        result["within_sum"] = float(local_within[finite].sum())
        result["within_count"] = int(finite.sum())
        if include_row_ledger:
            result.update({
                "_answer_indexes": np.asarray(answer_indexes, dtype=np.int64),
                "_prediction": prediction,
                "_decision_valid": valid,
                "_within_answer_indexes": np.asarray(answer_indexes[finite], dtype=np.int64),
                "_within_values": np.asarray(local_within[finite], dtype=np.float64),
            })
        return result

    block_ids = np.concatenate([
        np.full(int(offsets[index + 1] - offsets[index]), index, dtype=np.int64)
        for index in answer_indexes
    ]) if len(answer_indexes) else np.empty(0, dtype=np.int64)
    ranks = None
    if precomputed_step_ranks is not None:
        ranks = np.asarray(
            precomputed_step_ranks[np.ix_(step_rows, member_indexes)], dtype=np.float64,
        )
    values = (
        ranks
        if ranks is not None
        else consolidated.step_scores[np.ix_(step_rows, member_indexes)]
    )
    fused, available = apply_label_free_fusion_weights(
        values, consolidated.step_active[np.ix_(step_rows, member_indexes)], weights,
        block_ids=block_ids, families=families, head=head,
        precomputed_ranks=ranks,
    )
    cursor = 0
    peaks = np.full(len(answer_indexes), -1, dtype=np.int64)
    answer_available = np.zeros(len(answer_indexes), dtype=bool)
    for local, answer_index in enumerate(answer_indexes):
        width = int(offsets[answer_index + 1] - offsets[answer_index])
        local_active = available[cursor : cursor + width]
        if local_active.any():
            local_rows = np.flatnonzero(local_active)
            peaks[local] = int(local_rows[np.argmax(fused[cursor : cursor + width][local_rows])])
            answer_available[local] = True
        cursor += width
    gate = np.asarray(evaluation["incumbent_gate_open"], dtype=bool)[answer_indexes]
    prediction = np.where(gate, peaks, -1)
    decision_valid = (~gate) | answer_available
    record_pb(prediction, decision_valid)
    result["available_answers"] = int(answer_available.sum())
    cursor = 0
    within_answers = []
    within_values = []
    for local, answer_index in enumerate(answer_indexes):
        width = int(offsets[answer_index + 1] - offsets[answer_index])
        if cells[answer_index].startswith("prmbench_"):
            value = _risk_pair_auc(
                labels[offsets[answer_index] : offsets[answer_index + 1]],
                fused[cursor : cursor + width], available[cursor : cursor + width],
            )
            if np.isfinite(value):
                result["within_sum"] = float(result["within_sum"]) + value
                result["within_count"] = int(result["within_count"]) + 1
                within_answers.append(int(answer_index))
                within_values.append(float(value))
        cursor += width
    if include_row_ledger:
        result.update({
            "_answer_indexes": np.asarray(answer_indexes, dtype=np.int64),
            "_prediction": prediction,
            "_decision_valid": decision_valid,
            "_within_answer_indexes": np.asarray(within_answers, dtype=np.int64),
            "_within_values": np.asarray(within_values, dtype=np.float64),
        })
    return result


def _aggregate_metric_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells: dict[str, dict[str, int]] = {}
    for row in rows:
        for cell, counts in row.get("pb_cells", {}).items():
            target = cells.setdefault(str(cell), {
                "clean_hits": 0, "clean_total": 0, "error_hits": 0,
                "error_total": 0, "valid": 0,
            })
            for key in target:
                target[key] += int(counts[key])
    cell_metrics = {}
    f1_values = []
    for cell, counts in sorted(cells.items()):
        clean_accuracy = (
            counts["clean_hits"] / counts["clean_total"] if counts["clean_total"] else None
        )
        error_accuracy = (
            counts["error_hits"] / counts["error_total"] if counts["error_total"] else None
        )
        if clean_accuracy is None or error_accuracy is None:
            f1 = None
        else:
            denominator = clean_accuracy + error_accuracy
            f1 = 2.0 * clean_accuracy * error_accuracy / denominator if denominator else 0.0
            f1_values.append(f1)
        cell_metrics[cell] = {
            **counts, "clean_accuracy": clean_accuracy,
            "error_exact_accuracy": error_accuracy, "f1": f1,
        }
    within_sum = sum(float(row["within_sum"]) for row in rows)
    within_count = sum(int(row["within_count"]) for row in rows)
    pb_total = sum(value["clean_total"] + value["error_total"] for value in cells.values())
    pb_exact = sum(value["clean_hits"] + value["error_hits"] for value in cells.values())
    return {
        "pb": float(np.mean(f1_values)) if f1_values and len(f1_values) == len(cells) else None,
        "pb_cells": cell_metrics,
        "pb_correct": pb_exact,
        "pb_total": pb_total,
        "within": float(within_sum / within_count) if within_count else None,
        "within_count": within_count,
        "available_answers": sum(int(row["available_answers"]) for row in rows),
    }


def _pareto_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    access_precedence = {
        "answer_only": 0,
        "historical_artifact": 1,
        "source_excluded": 2,
        "pair_excluded": 3,
        "not_available": 4,
    }

    def access_burden(row: Mapping[str, object]) -> int:
        scopes = tuple(str(value) for value in row.get("access_scopes", ()))
        return max((access_precedence.get(value, 10) for value in scopes), default=0)

    output = []
    for row in rows:
        pb = row.get("pb")
        within = row.get("within")
        dominated = False
        for other in rows:
            if other is row or other.get("pb") is None:
                continue
            if pb is None:
                dominated = True; break
            within_ok = within is None or (
                other.get("within") is not None and float(other["within"]) >= float(within)
            )
            if (
                float(other["pb"]) >= float(pb)
                and within_ok
                and int(other["family_count"]) <= int(row["family_count"])
                and float(other["cost"]) <= float(row["cost"])
                and access_burden(other) <= access_burden(row)
                and (
                    float(other["pb"]) > float(pb)
                    or (within is not None and float(other["within"]) > float(within))
                    or int(other["family_count"]) < int(row["family_count"])
                    or float(other["cost"]) < float(row["cost"])
                    or access_burden(other) < access_burden(row)
                )
            ):
                dominated = True; break
        if not dominated:
            output.append(dict(row))
    return sorted(output, key=lambda row: (str(row["point"]), -float(row["pb"] or -1), str(row["name"])))


def _fit_nested_model(
    model: Mapping[str, object],
    answer_indexes: np.ndarray,
    *,
    outer: int | None,
    consolidated: _Consolidated,
    geometry: _FusionGeometry,
    artifacts: _NestedArtifacts,
) -> tuple[np.ndarray, Mapping[str, object]]:
    point = str(model["point"])
    members = tuple(str(value) for value in model["members"])
    families = tuple(str(value) for value in model["families"])
    head = str(model["head"])
    indexes = np.asarray(model["member_indexes"], dtype=np.int64)
    if point == "background":
        residuals, active = _background_training_matrix(
            artifacts, members, answer_indexes, inner_outer_fold=outer,
        )
        return _fit_natural_background_weights(residuals, active, families, head)
    if point == "token_pre_readout":
        values, active, block_ids, ranks = _token_training_matrix(
            artifacts, members, answer_indexes,
        )
        return fit_label_free_fusion_weights(
            values, active, families, head, block_ids=block_ids,
            precomputed_ranks=ranks,
        )
    if point == "answer_gate":
        cells = np.asarray(geometry.cells).astype(str)
        return fit_label_free_fusion_weights(
            consolidated.answer_scores[np.ix_(answer_indexes, indexes)],
            consolidated.answer_active[np.ix_(answer_indexes, indexes)],
            families, head, block_ids=cells[answer_indexes],
        )
    offsets = np.asarray(geometry.offsets, dtype=np.int64)
    step_rows = np.concatenate([
        np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
        for index in answer_indexes
    ])
    block_ids = np.concatenate([
        np.full(offsets[index + 1] - offsets[index], index, dtype=np.int64)
        for index in answer_indexes
    ])
    ranks = None
    if artifacts.step_ranks is not None:
        ranks = np.asarray(artifacts.step_ranks[np.ix_(step_rows, indexes)], dtype=np.float64)
    values = (
        ranks
        if ranks is not None
        else consolidated.step_scores[np.ix_(step_rows, indexes)]
    )
    return fit_label_free_fusion_weights(
        values, consolidated.step_active[np.ix_(step_rows, indexes)],
        families, head, block_ids=block_ids,
        precomputed_ranks=ranks,
    )


def _evaluate_nested_model(
    model: Mapping[str, object],
    weights: np.ndarray,
    answer_indexes: np.ndarray,
    *,
    outer: int,
    use_pair_excluded: bool,
    consolidated: _Consolidated,
    evaluation: Mapping[str, np.ndarray],
    artifacts: _NestedArtifacts,
    include_row_ledger: bool = False,
) -> dict[str, object]:
    point = str(model["point"])
    members = tuple(str(value) for value in model["members"])
    families = tuple(str(value) for value in model["families"])
    head = str(model["head"])
    if point == "background":
        scores, active = _apply_background_model(
            artifacts, members, families, head, weights, answer_indexes,
            inner_outer_fold=outer if use_pair_excluded else None,
        )
        return _step_subset_metrics(
            scores, active, answer_indexes, evaluation,
            include_row_ledger=include_row_ledger,
        )
    if point == "token_pre_readout":
        scores, active = _apply_token_model(
            artifacts, members, families, head, weights, answer_indexes,
        )
        return _step_subset_metrics(
            scores, active, answer_indexes, evaluation,
            include_row_ledger=include_row_ledger,
        )
    return _fusion_model_metrics(
        consolidated, np.asarray(model["member_indexes"], dtype=np.int64),
        weights, evaluation, answer_indexes, point=point,
        families=families, head=head, precomputed_step_ranks=artifacts.step_ranks,
        include_row_ledger=include_row_ledger,
    )


def _semantic_group_is_legal(point: str, members: Sequence[str]) -> bool:
    if point == "background":
        parsed = [_parse_background_member(name) for name in members]
        return len({value[0] for value in parsed}) == 1 and len({value[2] for value in parsed}) == 1
    if point == "token_pre_readout":
        parsed = [str(name).split("::") for name in members]
        return all(len(value) == 3 and value[0] == "token" for value in parsed) \
            and len({value[2] for value in parsed}) == 1 \
            and len({value[1] for value in parsed}) == len(parsed)
    return True


def _public_metric_row(row: Mapping[str, object]) -> dict[str, object]:
    """Drop private answer-level ledgers before serialising nested output."""
    return {
        str(key): _json_ready(value)
        for key, value in row.items()
        if not str(key).startswith("_")
    }


def _assemble_oof_ledger(
    rows: Sequence[Mapping[str, object]],
    answer_count: int,
) -> dict[str, np.ndarray]:
    prediction = np.full(answer_count, -1, dtype=np.int64)
    valid = np.zeros(answer_count, dtype=bool)
    assigned = np.zeros(answer_count, dtype=np.int8)
    within = np.full(answer_count, np.nan, dtype=np.float64)
    for row in rows:
        if "_answer_indexes" not in row:
            raise AtlasBackendError("OOF uncertainty requires the private answer ledger")
        indexes = np.asarray(row["_answer_indexes"], dtype=np.int64)
        if np.any(assigned[indexes]):
            raise AtlasBackendError("OOF answer ledger assigns an answer more than once")
        prediction[indexes] = np.asarray(row["_prediction"], dtype=np.int64)
        valid[indexes] = np.asarray(row["_decision_valid"], dtype=bool)
        assigned[indexes] += 1
        within_indexes = np.asarray(row["_within_answer_indexes"], dtype=np.int64)
        within_values = np.asarray(row["_within_values"], dtype=np.float64)
        if within_indexes.shape != within_values.shape:
            raise AtlasBackendError("OOF within-answer ledger is malformed")
        if np.isfinite(within[within_indexes]).any():
            raise AtlasBackendError("OOF within-answer ledger assigns an answer more than once")
        within[within_indexes] = within_values
    if not np.all(assigned == 1):
        missing = int(np.sum(assigned == 0))
        raise AtlasBackendError(f"OOF answer ledger is incomplete ({missing} answers missing)")
    return {"prediction": prediction, "valid": valid, "within": within}


def _ledger_metrics(
    ledger: Mapping[str, np.ndarray],
    indexes: np.ndarray,
    evaluation: Mapping[str, np.ndarray],
    *,
    required_pb_cells: Sequence[str] | None = None,
) -> tuple[float, float]:
    indexes = np.asarray(indexes, dtype=np.int64)
    sampled_cells = np.asarray(evaluation["cells"]).astype(str)[indexes]
    required = (
        tuple(sorted(str(value) for value in required_pb_cells))
        if required_pb_cells is not None else None
    )
    observed = tuple(sorted(set(sampled_cells[np.char.startswith(sampled_cells, "pb_")])))
    if required is not None and observed != required:
        # Canonical PB is an eight-cell macro.  A source-group draw that omits
        # a registered cell is not silently redefined as a smaller-cell macro.
        pb = float("nan")
    else:
        pb = _canonical_pb_score(
            np.asarray(evaluation["target"], dtype=np.int64)[indexes],
            np.asarray(ledger["prediction"], dtype=np.int64)[indexes],
            np.asarray(ledger["valid"], dtype=bool)[indexes],
            sampled_cells,
        )
    within = np.asarray(ledger["within"], dtype=np.float64)[indexes]
    finite = np.isfinite(within)
    return pb, float(within[finite].mean()) if finite.any() else float("nan")


def _bootstrap_gate_ledger(
    model: Mapping[str, object],
    oof_rows: Sequence[Mapping[str, object]],
    indexes: np.ndarray,
    *,
    consolidated: _Consolidated,
    evaluation: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Recompute member midranks, fusion and q=.33 inside one bootstrap draw."""
    indexes = np.asarray(indexes, dtype=np.int64)
    cells = np.asarray(evaluation["cells"]).astype(str)[indexes]
    folds = np.asarray(evaluation["folds"], dtype=np.int64)[indexes]
    member_indexes = np.asarray(model["member_indexes"], dtype=np.int64)
    families = tuple(str(value) for value in model["families"])
    head = str(model["head"])
    raw_scores = consolidated.answer_scores[np.ix_(indexes, member_indexes)]
    raw_active = consolidated.answer_active[np.ix_(indexes, member_indexes)]
    peaks = np.asarray(evaluation["digit025_peaks"], dtype=np.int64)[indexes]
    prediction = np.full(len(indexes), -1, dtype=np.int64)
    valid = np.zeros(len(indexes), dtype=bool)
    by_outer = {int(row["outer"]): row for row in oof_rows}
    if set(by_outer) != set(range(5)):
        raise AtlasBackendError("gate bootstrap requires one OOF weight vector per source fold")
    for outer in range(5):
        fused, fused_active = apply_label_free_fusion_weights(
            raw_scores,
            raw_active,
            np.asarray(by_outer[outer]["weights"], dtype=np.float64),
            block_ids=cells,
            families=families,
            head=head,
        )
        opened, _, gate_active = rank_fused_gate_decisions(
            fused[:, None], ("fused",), cells, {"gate": ("fused",)},
            q=0.33, active=fused_active[:, None],
        )
        local = folds == outer
        prediction[local] = np.where(opened[local, 0], peaks[local], -1)
        valid[local] = (~opened[local, 0]) | (
            gate_active[local, 0] & (peaks[local] >= 0)
        )
    return {
        "prediction": prediction,
        "valid": valid,
        "within": np.asarray(evaluation["digit025_within"], dtype=np.float64)[indexes],
    }


def _finalist_uncertainty(
    *,
    finalists: Mapping[str, Mapping[str, object]],
    model_by_name: Mapping[str, Mapping[str, object]],
    finalist_oof_rows: Mapping[str, Sequence[Mapping[str, object]]],
    consolidated: _Consolidated,
    evaluation: Mapping[str, np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Paired grouped OOF comparison with simultaneous five-point intervals."""
    if int(draws) <= 0:
        raise ValueError("uncertainty draws must be positive")
    points = tuple(sorted(finalists))
    answer_count = len(evaluation["target"])
    all_indexes = np.arange(answer_count, dtype=np.int64)
    all_cells = np.asarray(evaluation["cells"]).astype(str)
    required_pb_cells = tuple(sorted(set(
        all_cells[np.char.startswith(all_cells, "pb_")]
    )))
    baseline = {
        "prediction": np.where(
            np.asarray(evaluation["incumbent_gate_open"], dtype=bool),
            np.asarray(evaluation["digit025_peaks"], dtype=np.int64),
            -1,
        ),
        "valid": (
            ~np.asarray(evaluation["incumbent_gate_open"], dtype=bool)
            | (np.asarray(evaluation["digit025_peaks"], dtype=np.int64) >= 0)
        ),
        "within": np.asarray(evaluation["digit025_within"], dtype=np.float64),
    }
    candidate_ledgers = {
        point: _assemble_oof_ledger(finalist_oof_rows[point], answer_count)
        for point in points
    }
    baseline_point = _ledger_metrics(
        baseline, all_indexes, evaluation, required_pb_cells=required_pb_cells,
    )
    candidate_point = {
        point: _ledger_metrics(
            candidate_ledgers[point], all_indexes, evaluation,
            required_pb_cells=required_pb_cells,
        )
        for point in points
    }
    point_delta = np.asarray([
        value
        for point in points
        for value in (
            candidate_point[point][0] - baseline_point[0],
            candidate_point[point][1] - baseline_point[1],
        )
    ], dtype=np.float64)

    _, group_rows = _group_rows(np.asarray(evaluation["groups"]).astype(str))
    if len(group_rows) < 2:
        raise AtlasBackendError("finalist uncertainty requires at least two source groups")
    rng = np.random.default_rng(int(seed))
    samples = np.full((int(draws), len(point_delta)), np.nan, dtype=np.float64)
    for draw in range(int(draws)):
        indexes = _bootstrap_indices(group_rows, rng)
        baseline_draw = _ledger_metrics(
            baseline, indexes, evaluation, required_pb_cells=required_pb_cells,
        )
        for point_index, point in enumerate(points):
            if point == "answer_gate":
                local_ledger = _bootstrap_gate_ledger(
                    model_by_name[finalists[point]["name"]],
                    finalist_oof_rows[point],
                    indexes,
                    consolidated=consolidated,
                    evaluation=evaluation,
                )
                # The gate ledger is already in bootstrap-row coordinates.
                local_evaluation = {
                    key: np.asarray(evaluation[key])[indexes]
                    for key in ("target", "cells")
                }
                candidate_draw = _ledger_metrics(
                    local_ledger, np.arange(len(indexes)), local_evaluation,
                    required_pb_cells=required_pb_cells,
                )
            else:
                candidate_draw = _ledger_metrics(
                    candidate_ledgers[point], indexes, evaluation,
                    required_pb_cells=required_pb_cells,
                )
            samples[draw, 2 * point_index] = candidate_draw[0] - baseline_draw[0]
            samples[draw, 2 * point_index + 1] = candidate_draw[1] - baseline_draw[1]

    try:
        intervals = simultaneous_intervals(point_delta, samples, confidence=0.95)
        interval_error = None
    except (ValueError, FloatingPointError) as error:
        intervals = np.full((len(point_delta), 2), np.nan, dtype=np.float64)
        interval_error = f"{type(error).__name__}: {error}"
    rows = {}
    for point_index, point in enumerate(points):
        pb_delta = float(point_delta[2 * point_index])
        within_delta = float(point_delta[2 * point_index + 1])
        pb_interval = intervals[2 * point_index]
        within_interval = intervals[2 * point_index + 1]
        finite = np.isfinite(np.r_[pb_interval, within_interval]).all()
        pb_successor = (
            finite and pb_delta >= 0.01 and pb_interval[0] > 0.0
            and within_interval[0] >= -0.002
        )
        within_successor = (
            finite and within_delta >= 0.005 and within_interval[0] > 0.0
            and pb_interval[0] >= -0.01
        )
        point_promising = (
            (pb_delta >= 0.01 and within_delta >= -0.002)
            or (within_delta >= 0.005 and pb_delta >= -0.01)
        )
        if not finite:
            classification = "UNRESOLVED_UNCERTAINTY"
        elif pb_successor or within_successor:
            classification = "PERFORMANCE_SUCCESSOR"
        elif point_promising:
            classification = "RESEARCH_CANDIDATE"
        else:
            classification = "NO_SUCCESSOR"
        rows[point] = {
            "baseline": {"pb": baseline_point[0], "within": baseline_point[1]},
            "candidate": {
                "pb": candidate_point[point][0], "within": candidate_point[point][1],
            },
            "delta": {"pb": pb_delta, "within": within_delta},
            "simultaneous_95_ci": {
                "pb": pb_interval, "within": within_interval,
            },
            "finite_draws": {
                "pb": int(np.isfinite(samples[:, 2 * point_index]).sum()),
                "within": int(np.isfinite(samples[:, 2 * point_index + 1]).sum()),
            },
            "promotion": classification,
            "simplification": "UNRESOLVED_BASELINE_COMPLEXITY_NOT_BOUND",
        }
    return {
        "schema": "fusion-independence-atlas-v1/finalist-uncertainty-v1",
        "draws": int(draws),
        "seed": int(seed),
        "paired_source_group_draws": True,
        "simultaneous_across_insertion_points_and_metrics": True,
        "gate_reranked_within_every_draw": "answer_gate" in points,
        "interval_error": interval_error,
        "criteria": {
            "performance_pb": {
                "point_delta": 0.01, "lower_ci_above": 0.0,
                "within_noninferiority": -0.002,
            },
            "performance_within": {
                "point_delta": 0.005, "lower_ci_above": 0.0,
                "pb_noninferiority": -0.01,
            },
            "simplification": "requires a separately frozen incumbent family/stage complexity ledger",
        },
        "points": rows,
    }


def run_nested_fusion_search(
    *,
    dependence_root,
    extraction_root,
    output_root,
    folds: int = 5,
    family_cap: int = 2,
    maximum_size: int = 6,
    roster_stability: int = 4,
    heads: Sequence[str] = ("singleton", "equal_rank", "family_equal", "nonnegative_shrunk_simplex", "iu"),
    iu_eligible_only: bool = True,
    seed: int = 39_615,
):
    """Nested five-source-fold, label-separated fusion search.

    Labels rank rosters only in inner validation.  Every head is fit through
    :func:`fit_label_free_fusion_weights`, whose interface has no label input.
    """
    if folds != 5 or family_cap != 2 or maximum_size > 6 or roster_stability < 1:
        raise ValueError("nested Atlas contract requires folds=5, family_cap=2, maximum_size<=6")
    dependence_root = Path(dependence_root)
    extraction_root = Path(extraction_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    consolidated = _load_consolidated(dependence_root)
    evaluation = _evaluation_arrays(dependence_root)
    artifacts = _load_nested_artifacts(dependence_root, extraction_root, evaluation)
    artifacts = _attach_nested_rank_caches(
        artifacts, consolidated, evaluation, output_root,
    )
    geometry = _FusionGeometry(
        offsets=np.asarray(evaluation["offsets"], dtype=np.int64),
        cells=np.asarray(evaluation["cells"]).astype(str),
    )
    fold_vector = np.asarray(evaluation["folds"], dtype=np.int64)
    if set(np.unique(fold_vector)) != set(range(5)):
        raise AtlasBackendError("nested search requires five populated source folds 0..4")

    candidate_payload = _read_json(dependence_root / "CANDIDATES.json")
    selected_by_outer = {
        str(row["name"]): frozenset(int(value) for value in row.get("selected_outer_folds", ()))
        for row in candidate_payload["eligible"]
    }
    graph_document = _read_json(dependence_root / "COMPATIBILITY_GRAPH.json")
    outer_graphs = graph_document.get("outer_training_graphs")
    group_document = _read_json(dependence_root / "GROUPS.json")
    outer_group_rows = group_document.get("outer_training_rows")
    pairwise_document = _read_json(dependence_root / "PAIRWISE.json")
    outer_pair_rows = pairwise_document.get("outer_training_rows")
    if (
        not isinstance(outer_graphs, Mapping)
        or not isinstance(outer_group_rows, list)
        or not isinstance(outer_pair_rows, list)
    ):
        raise AtlasBackendError(
            "nested roster requires per-outer training-only compatibility graphs and group screens; "
            "per-outer pair diagnostics are also required"
        )
    compatible_by_outer = {
        outer: {
            (str(row["point"]), frozenset(str(value) for value in row["members"]))
            for row in outer_group_rows
            if int(row.get("outer", -1)) == outer
            and row.get("status") == DependenceStatus.INDEPENDENCE_COMPATIBLE.value
        }
        for outer in range(5)
    }
    name_to_index = {name: index for index, name in enumerate(consolidated.names)}
    points = ("background", "token_pre_readout", "step_post_readout", "decoder", "answer_gate")
    group_outer_folds: dict[tuple[str, tuple[str, ...]], set[int]] = {}
    noncompatible_pair_outer_folds: dict[tuple[str, tuple[str, str]], set[int]] = {}
    noncompatible_pair_screens: dict[tuple[str, tuple[str, str]], list[dict]] = {}
    supported_dependent_outer_folds: dict[tuple[str, tuple[str, str]], set[int]] = {}

    for row in outer_pair_rows:
        matrix_name = str(row.get("matrix", ""))
        point = matrix_name.split(":", 1)[0]
        if point not in points or row.get("status") != DependenceStatus.UNRESOLVED.value:
            continue
        if row.get("target") == ErrorTarget.PREDICTOR_RESIDUAL.value:
            continue
        members = tuple(sorted((str(row.get("left", "")), str(row.get("right", "")))))
        support_values = (
            row.get("left_successes"), row.get("left_failures"),
            row.get("right_successes"), row.get("right_failures"),
        )
        interval = row.get("interval")
        supported = (
            int(row.get("source_groups", 0)) >= 20
            and all(value is not None and float(value) >= 50 for value in support_values)
            and isinstance(interval, Mapping)
            and all(interval.get(key) is not None for key in (
                "phi_low", "phi_high", "odds_ratio_low", "odds_ratio_high",
            ))
        )
        if supported:
            supported_dependent_outer_folds.setdefault((point, members), set()).add(
                int(row["outer"])
            )
    for outer in range(5):
        fold_graphs = outer_graphs.get(str(outer), {})
        for point in points:
            graph_row = fold_graphs.get(point)
            if not graph_row:
                continue
            nodes = tuple(name for name in graph_row.get("nodes", ()) if name in name_to_index)
            for name in nodes:
                group_outer_folds.setdefault((point, (name,)), set()).add(outer)
            adjacency = {name: set() for name in nodes}
            for left, right in graph_row.get("edges", ()):
                if left in adjacency and right in adjacency:
                    adjacency[left].add(right); adjacency[right].add(left)
            for pair_row in graph_row.get("pair_statuses", ()):
                left = str(pair_row.get("left", "")); right = str(pair_row.get("right", ""))
                status = str(pair_row.get("status", DependenceStatus.UNRESOLVED.value))
                members = tuple(sorted((left, right)))
                if (
                    left in adjacency and right in adjacency
                    and status not in {
                        DependenceStatus.INDEPENDENCE_COMPATIBLE.value,
                        DependenceStatus.REDUNDANT.value,
                    }
                    and _semantic_group_is_legal(point, members)
                ):
                    key = (point, members)
                    noncompatible_pair_outer_folds.setdefault(key, set()).add(outer)
                    noncompatible_pair_screens.setdefault(key, []).append({
                        "outer": outer, **_json_ready(pair_row),
                    })
            for members in _enumerate_cliques_complete(
                adjacency, min_size=2, max_size=maximum_size, max_results=100_000,
            ):
                if (point, frozenset(members)) not in compatible_by_outer[outer]:
                    continue
                if _semantic_group_is_legal(point, members):
                    group_outer_folds.setdefault((point, tuple(members)), set()).add(outer)

    model_rows: list[dict] = []
    for (point, members), graph_outer_eligibility in sorted(group_outer_folds.items()):
        indexes = np.asarray([name_to_index[name] for name in members], dtype=np.int64)
        families = tuple(consolidated.families[index] for index in indexes)
        eligible_outer = set(graph_outer_eligibility)
        for member in members:
            eligible_outer &= set(selected_by_outer.get(member, ()))
        eligible_heads = ("singleton",) if len(members) == 1 else tuple(
            head for head in heads if head != "singleton"
        )
        for head in eligible_heads:
            if head == "iu" and len(members) < 3:
                continue
            model_rows.append({
                "name": f"{point}::{'+'.join(members)}::{head}",
                "point": point, "members": list(members),
                "member_indexes": indexes, "families": families, "head": head,
                "family_count": len(set(families)),
                "cost": float(consolidated.costs[indexes].sum()),
                "access_scopes": sorted({consolidated.access_scopes[index] for index in indexes}),
                "independence_status": (
                    "SINGLETON" if len(members) == 1
                    else DependenceStatus.INDEPENDENCE_COMPATIBLE.value
                ),
                "eligible_outer_folds": sorted(eligible_outer),
            })
    observed_points = {row["point"] for row in model_rows}
    if observed_points != set(points):
        missing = sorted(set(points) - observed_points)
        raise AtlasBackendError("five-point nested candidate construction is incomplete: " + ", ".join(missing))

    model_by_name = {row["name"]: row for row in model_rows}
    outer_results: dict[str, list[dict]] = {row["name"]: [] for row in model_rows}
    nested_selection: list[dict] = []
    iu_diagnostics: list[dict] = []

    def outer_evaluation(
        model: Mapping[str, object],
        outer: int,
        iu_diagnostic_only: bool,
        *,
        include_row_ledger: bool = False,
    ) -> dict:
        train_answers = np.flatnonzero(fold_vector != outer)
        test_answers = np.flatnonzero(fold_vector == outer)
        weights, diagnostics = _fit_nested_model(
            model, train_answers, outer=outer, consolidated=consolidated,
            geometry=geometry, artifacts=artifacts,
        )
        metrics = _evaluate_nested_model(
            model, weights, test_answers, outer=outer, use_pair_excluded=False,
            consolidated=consolidated, evaluation=evaluation, artifacts=artifacts,
            include_row_ledger=include_row_ledger,
        )
        return {
            "outer": outer, **metrics, "weights": weights.tolist(),
            "weight_diagnostics": _json_ready(diagnostics),
            "iu_diagnostic_only": bool(iu_diagnostic_only),
        }

    for outer in range(5):
        candidates: list[tuple[dict, dict, bool]] = []
        for model in model_rows:
            if outer not in model["eligible_outer_folds"]:
                continue
            inner_rows = []
            inner_signs = []
            for inner in range(5):
                if inner == outer:
                    continue
                train_answers = np.flatnonzero((fold_vector != outer) & (fold_vector != inner))
                validation_answers = np.flatnonzero(fold_vector == inner)
                weights, _ = _fit_nested_model(
                    model, train_answers, outer=outer, consolidated=consolidated,
                    geometry=geometry, artifacts=artifacts,
                )
                inner_signs.append(np.sign(weights).astype(np.int8).tolist())
                inner_rows.append(_evaluate_nested_model(
                    model, weights, validation_answers, outer=outer,
                    use_pair_excluded=(model["point"] == "background"),
                    consolidated=consolidated, evaluation=evaluation, artifacts=artifacts,
                ))
            inner_metric = _aggregate_metric_rows(inner_rows)
            iu_unstable = False
            if model["head"] == "iu":
                outer_train = np.flatnonzero(fold_vector != outer)
                outer_weights, _ = _fit_nested_model(
                    model, outer_train, outer=outer, consolidated=consolidated,
                    geometry=geometry, artifacts=artifacts,
                )
                all_signs = inner_signs + [np.sign(outer_weights).astype(np.int8).tolist()]
                signs = np.asarray(all_signs, dtype=np.int8)
                iu_unstable = bool(
                    np.any(outer_weights < -1e-9)
                    or np.any(signs < 0)
                    or np.any(np.ptp(signs, axis=0) > 0)
                )
                iu_diagnostics.append({
                    "outer": outer, "name": model["name"], "signs": all_signs,
                    "outer_weights": outer_weights.tolist(),
                    "diagnostic_only": iu_unstable,
                })
            candidates.append((model, inner_metric, iu_unstable))
        eligible = [item for item in candidates if not item[2]]
        selected_for_point: dict[str, tuple[dict, dict, bool]] = {}
        for point in points:
            local = [item for item in eligible if item[0]["point"] == point]
            if not local:
                raise AtlasBackendError(f"outer fold {outer} has no eligible {point} model")
            selected_for_point[point] = min(local, key=lambda item: (
                -float(item[1]["pb"] if item[1]["pb"] is not None else -np.inf),
                -float(item[1]["within"] if item[1]["within"] is not None else -np.inf),
                int(item[0]["family_count"]), float(item[0]["cost"]), str(item[0]["name"]),
            ))

        # Save OOF results for every screened model, not only the winner.  This
        # makes ALL_GROUPS and the Pareto frontier auditable.
        current_results: dict[str, dict] = {}
        for model, _, iu_unstable in candidates:
            result = outer_evaluation(model, outer, iu_unstable)
            outer_results[model["name"]].append(result)
            current_results[model["name"]] = result
        nested_selection.append({
            "outer": outer,
            "selections": {
                point: {
                    "selected": item[0]["name"], "members": item[0]["members"],
                    "head": item[0]["head"], "inner": item[1],
                    "outer_result": {
                        key: _json_ready(value)
                        for key, value in current_results[item[0]["name"]].items()
                        if not str(key).startswith("_")
                    },
                }
                for point, item in sorted(selected_for_point.items())
            },
            "training_folds": [fold for fold in range(5) if fold != outer],
            "held_labels_used_for_weights": False,
            "learned_background_axis": "pair_excluded_outer_fold_for_inner_selection",
        })

    selection_counts: dict[str, int] = {}
    for row in nested_selection:
        for selection in row["selections"].values():
            name = selection["selected"]
            selection_counts[name] = selection_counts.get(name, 0) + 1

    # A 4/5-stable roster still needs predictions on all five outer folds for
    # a canonical aggregate.  Fill only the missing folds of stable rosters;
    # family-cap eligibility affects selection, not held-fold measurement.
    stable_names = {
        name for name, count in selection_counts.items() if count >= roster_stability
    }
    for name in sorted(stable_names):
        present = {int(row["outer"]) for row in outer_results[name]}
        for outer in sorted(set(range(5)) - present):
            outer_results[name].append(outer_evaluation(model_by_name[name], outer, False))

    aggregate_rows = []
    aggregate_by_name = {}
    for name, rows in sorted(outer_results.items()):
        if len(rows) != 5:
            continue
        model = model_by_name[name]
        metrics = _aggregate_metric_rows(sorted(rows, key=lambda row: int(row["outer"])))
        aggregate = {
            "name": name, "point": model["point"], "members": model["members"],
            "head": model["head"], "family_count": model["family_count"],
            "cost": model["cost"], "access_scopes": model["access_scopes"],
            "independence_status": model["independence_status"],
            "outer_folds": sorted(int(row["outer"]) for row in rows), **metrics,
            "iu_diagnostic_only": bool(any(row["iu_diagnostic_only"] for row in rows)),
        }
        aggregate_rows.append(aggregate); aggregate_by_name[name] = aggregate

    finalists = {}
    for point in points:
        stable = [
            aggregate_by_name[name] for name in stable_names
            if name in aggregate_by_name and aggregate_by_name[name]["point"] == point
        ]
        if not stable:
            continue
        finalist = min(stable, key=lambda row: (
            -float(row["pb"] if row["pb"] is not None else -np.inf),
            -float(row["within"] if row["within"] is not None else -np.inf),
            int(row["family_count"]), float(row["cost"]), str(row["name"]),
        ))
        finalists[point] = {**finalist, "roster_stability": selection_counts[finalist["name"]]}

    # Freeze one executable, label-free full-development recipe per finalist.
    # These recipes are independently executable at their declared insertion
    # point; composition is assessed separately below rather than assumed.
    all_answers = np.arange(len(fold_vector), dtype=np.int64)
    finalist_recipes = {}
    for point, finalist in sorted(finalists.items()):
        model = model_by_name[finalist["name"]]
        weights, diagnostics = _fit_nested_model(
            model, all_answers, outer=None, consolidated=consolidated,
            geometry=geometry, artifacts=artifacts,
        )
        recipe = {
            "point": point, "model": model["name"], "members": model["members"],
            "families": list(model["families"]), "head": model["head"],
            "weights": weights.tolist(), "weight_fit": _json_ready(diagnostics),
            "labels_used_for_weights": False,
            "member_columns": np.asarray(model["member_indexes"], dtype=np.int64).tolist(),
            "tie_break": "lowest_step_index",
        }
        if point == "background":
            parsed = [_parse_background_member(name) for name in model["members"]]
            recipe.update({
                "primitive": parsed[0][0], "predictor_kinds": [value[1] for value in parsed],
                "readout": parsed[0][2],
                "execution": "natural-unit predictor fusion -> primitive residual -> step readout -> argmax",
                "callables": ["_background_tokens", "_apply_background_model"],
                "input_type": "token_primitive_level_and_predictor_backgrounds",
                "output_type": "step_risk_trajectory_and_peak",
                "predictor_artifacts": [
                    "predictors/fixed/backgrounds.npy",
                    "predictors/learned_oof/backgrounds.npy",
                ],
            })
        elif point == "token_pre_readout":
            mappings = [artifacts.token_candidate_map[name] for name in model["members"]]
            recipe.update({
                "token_signals": [value["signal"] for value in mappings],
                "readout": mappings[0]["readout"],
                "execution": "answer-local token rank fusion -> step readout -> argmax",
                "callables": ["apply_label_free_fusion_weights", "_apply_token_model"],
                "input_type": "aligned_token_signal_matrix",
                "output_type": "step_risk_trajectory_and_peak",
                "token_store": "dependence/TOKEN_VALUES.npy",
            })
        elif point == "step_post_readout":
            recipe.update({
                "execution": "per-view readout -> answer-local step rank fusion -> argmax",
                "callables": ["apply_label_free_fusion_weights", "_fusion_model_metrics"],
                "input_type": "aligned_step_readout_matrix",
                "output_type": "step_risk_trajectory_and_peak",
                "step_store": "dependence/CONSOLIDATED.npz::step_scores",
            })
        elif point == "decoder":
            recipe.update({
                "execution": "weighted vote over frozen member decisions",
                "callables": ["apply_label_free_fusion_weights", "_fusion_model_metrics"],
                "input_type": "aligned_one_hot_step_decisions",
                "output_type": "peak_decision",
                "decision_store": "dependence/CONSOLIDATED.npz::step_scores(one_hot)",
            })
        else:
            recipe.update({
                "threshold": 0.33,
                "execution": "full-population within-cell rank fusion -> rerank -> threshold",
                "frozen_locator": "digit025",
                "callables": ["apply_label_free_fusion_weights", "rank_fused_gate_decisions"],
                "input_type": "answer_detector_matrix_and_cell_ids",
                "output_type": "gate_open_decision",
                "detector_store": "dependence/CONSOLIDATED.npz::answer_scores",
            })
        finalist_recipes[point] = recipe
        finalists[point]["recipe"] = recipe

    # Registered order sensitivity for the post-readout finalist: use the
    # same raw signals, readout and head, but move fusion before the readout.
    # Cross-readout groups have no single legal pre-readout counterpart and
    # are reported unresolved rather than flattened.
    order_sensitivity: dict[str, object] = {
        "schema": "fusion-independence-atlas-v1/fusion-order-sensitivity-v1",
        "primary": "per-view readout then step fusion",
        "sensitivity": "token fusion then one shared readout",
        "status": "UNRESOLVED",
    }
    if "step_post_readout" in finalists:
        step_finalist = finalists["step_post_readout"]
        step_model = model_by_name[step_finalist["name"]]
        parsed = [str(member).split("::") for member in step_model["members"]]
        if not all(len(value) == 3 and value[0] == "step" for value in parsed):
            order_sensitivity["reason"] = "step finalist contains a native/non-token-derived expert"
        elif len({value[2] for value in parsed}) != 1:
            order_sensitivity["reason"] = "step finalist mixes readouts; no single pre-readout operator exists"
        else:
            signal_columns: dict[str, int] = {}
            for mapping in artifacts.token_candidate_map.values():
                signal = str(mapping["signal"]); column = int(mapping["column"])
                if signal in signal_columns and signal_columns[signal] != column:
                    raise AtlasBackendError("token store maps one signal to multiple columns")
                signal_columns[signal] = column
            missing = sorted({value[1] for value in parsed} - set(signal_columns))
            if missing:
                order_sensitivity["reason"] = (
                    "raw token columns were not retained for: " + ", ".join(missing)
                )
            else:
                readout = parsed[0][2]
                synthetic_names = tuple(
                    f"order_sensitivity::{value[1]}::{readout}" for value in parsed
                )
                extended_map = dict(artifacts.token_candidate_map)
                for name, value in zip(synthetic_names, parsed):
                    extended_map[name] = {
                        "signal": value[1], "column": signal_columns[value[1]],
                        "readout": readout,
                    }
                order_artifacts = replace(artifacts, token_candidate_map=extended_map)
                order_model = {
                    "name": "token_pre_readout::order_sensitivity",
                    "point": "token_pre_readout",
                    "members": list(synthetic_names),
                    "member_indexes": np.asarray(step_model["member_indexes"], dtype=np.int64),
                    "families": tuple(step_model["families"]),
                    "head": step_model["head"],
                }
                order_rows = []
                order_weights = []
                for outer in range(5):
                    train = np.flatnonzero(fold_vector != outer)
                    test = np.flatnonzero(fold_vector == outer)
                    weights, diagnostics = _fit_nested_model(
                        order_model, train, outer=outer, consolidated=consolidated,
                        geometry=geometry, artifacts=order_artifacts,
                    )
                    order_weights.append({
                        "outer": outer, "weights": weights.tolist(),
                        "diagnostics": _json_ready(diagnostics),
                    })
                    order_rows.append(_evaluate_nested_model(
                        order_model, weights, test, outer=outer,
                        use_pair_excluded=False, consolidated=consolidated,
                        evaluation=evaluation, artifacts=order_artifacts,
                    ))
                order_metrics = _aggregate_metric_rows(order_rows)
                order_sensitivity.update({
                    "status": "EVALUATED",
                    "members": list(step_model["members"]),
                    "shared_readout": readout,
                    "head": step_model["head"],
                    "primary_metrics": {
                        "pb": step_finalist.get("pb"), "within": step_finalist.get("within"),
                    },
                    "sensitivity_metrics": _public_metric_row(order_metrics),
                    "delta": {
                        "pb": (
                            None if step_finalist.get("pb") is None or order_metrics.get("pb") is None
                            else float(order_metrics["pb"]) - float(step_finalist["pb"])
                        ),
                        "within": (
                            None if step_finalist.get("within") is None or order_metrics.get("within") is None
                            else float(order_metrics["within"]) - float(step_finalist["within"])
                        ),
                    },
                    "outer_label_free_weights": order_weights,
                })
    else:
        order_sensitivity["reason"] = "no 4/5-stable step-post-readout finalist"

    # Re-evaluate only the five frozen finalists with compact, private
    # answer-level ledgers.  Keeping these ledgers out of the all-model pass
    # avoids O(models x answers) memory while still supporting a paired OOF
    # performance bootstrap.
    finalist_oof_rows: dict[str, list[dict]] = {}
    for point, finalist in sorted(finalists.items()):
        model = model_by_name[finalist["name"]]
        finalist_oof_rows[point] = [
            outer_evaluation(
                model, outer, False, include_row_ledger=True,
            )
            for outer in range(5)
        ]
    dependence_draws = int(_read_json(dependence_root / "PAIRWISE.json").get("draws", 10_000))
    uncertainty = _finalist_uncertainty(
        finalists=finalists,
        model_by_name=model_by_name,
        finalist_oof_rows=finalist_oof_rows,
        consolidated=consolidated,
        evaluation=evaluation,
        draws=dependence_draws,
        seed=int(seed) + 30_000,
    )
    for point, row in uncertainty["points"].items():
        finalists[point]["uncertainty"] = row

    # Independence failures are not silently equated with uselessness.  For
    # every semantically legal pair that was non-compatible in an outer
    # training screen, run one positive equal-rank OOF diagnostic.  A pair is
    # DEPENDENT_COMPLEMENTARY only when held-fold singleton successes are
    # genuinely unique and the fused pair improves an OOF headline metric.
    diagnostic_singletons: dict[tuple[str, str, int], dict] = {}
    dependent_pair_rows = []

    def singleton_diagnostic(point: str, member: str, outer: int) -> dict:
        key = (point, member, outer)
        if key not in diagnostic_singletons:
            index = name_to_index[member]
            model = {
                "name": f"{point}::{member}::singleton-diagnostic",
                "point": point,
                "members": [member],
                "member_indexes": np.asarray([index], dtype=np.int64),
                "families": (consolidated.families[index],),
                "head": "singleton",
            }
            diagnostic_singletons[key] = outer_evaluation(
                model, outer, False, include_row_ledger=True,
            )
        return diagnostic_singletons[key]

    for (point, members), observed_outer_folds in sorted(noncompatible_pair_outer_folds.items()):
        eligible_outer = set(observed_outer_folds)
        for member in members:
            eligible_outer &= set(selected_by_outer.get(member, ()))
        indexes = np.asarray([name_to_index[member] for member in members], dtype=np.int64)
        pair_model = {
            "name": f"{point}::{'+'.join(members)}::equal_rank-dependent-diagnostic",
            "point": point,
            "members": list(members),
            "member_indexes": indexes,
            "families": tuple(consolidated.families[index] for index in indexes),
            "head": "equal_rank",
        }
        pair_results = []
        left_results = []
        right_results = []
        unique_left = 0
        unique_right = 0
        for outer in sorted(eligible_outer):
            pair_result = outer_evaluation(
                pair_model, outer, False, include_row_ledger=True,
            )
            left_result = singleton_diagnostic(point, members[0], outer)
            right_result = singleton_diagnostic(point, members[1], outer)
            pair_results.append(pair_result)
            left_results.append(left_result)
            right_results.append(right_result)
            answer_indexes = np.asarray(pair_result["_answer_indexes"], dtype=np.int64)
            if not (
                np.array_equal(answer_indexes, left_result["_answer_indexes"])
                and np.array_equal(answer_indexes, right_result["_answer_indexes"])
            ):
                raise AtlasBackendError("dependent-pair held-fold ledgers are misaligned")
            target = np.asarray(evaluation["target"], dtype=np.int64)[answer_indexes]
            cells = np.asarray(evaluation["cells"]).astype(str)[answer_indexes]
            pb = np.char.startswith(cells, "pb_")
            left_correct = (
                np.asarray(left_result["_decision_valid"], dtype=bool)
                & (np.asarray(left_result["_prediction"], dtype=np.int64) == target)
            )
            right_correct = (
                np.asarray(right_result["_decision_valid"], dtype=bool)
                & (np.asarray(right_result["_prediction"], dtype=np.int64) == target)
            )
            unique_left += int(np.sum(pb & left_correct & ~right_correct))
            unique_right += int(np.sum(pb & right_correct & ~left_correct))
        if pair_results:
            pair_metrics = _aggregate_metric_rows(pair_results)
            left_metrics = _aggregate_metric_rows(left_results)
            right_metrics = _aggregate_metric_rows(right_results)
        else:
            pair_metrics = left_metrics = right_metrics = {
                "pb": None, "within": None, "pb_total": 0, "within_count": 0,
            }
        pb_values = [left_metrics.get("pb"), right_metrics.get("pb")]
        within_values = [left_metrics.get("within"), right_metrics.get("within")]
        best_single_pb = max((float(value) for value in pb_values if value is not None), default=float("nan"))
        best_single_within = max(
            (float(value) for value in within_values if value is not None), default=float("nan"),
        )
        pb_improved = (
            pair_metrics.get("pb") is not None and np.isfinite(best_single_pb)
            and float(pair_metrics["pb"]) > best_single_pb + 1e-12
        )
        within_improved = (
            pair_metrics.get("within") is not None and np.isfinite(best_single_within)
            and float(pair_metrics["within"]) > best_single_within + 1e-12
        )
        supported_outer = eligible_outer & supported_dependent_outer_folds.get(
            (point, members), set()
        )
        enough_outer_evidence = len(supported_outer) >= roster_stability
        complementary = (
            enough_outer_evidence
            and (unique_left + unique_right) > 0
            and (pb_improved or within_improved)
        )
        dependent_pair_rows.append({
            "name": pair_model["name"],
            "point": point,
            "members": list(members),
            "head": "equal_rank",
            "status": (
                DependenceStatus.DEPENDENT_COMPLEMENTARY.value
                if complementary else DependenceStatus.UNRESOLVED.value
            ),
            "diagnostic_only": True,
            "eligible_outer_folds": sorted(eligible_outer),
            "statistically_supported_outer_folds": sorted(supported_outer),
            "required_outer_folds": roster_stability,
            "outer_independence_screens": noncompatible_pair_screens[(point, members)],
            "held_fold_unique_successes": {
                "left": unique_left, "right": unique_right,
                "total": unique_left + unique_right,
            },
            "oof_fusion_improved": bool(pb_improved or within_improved),
            "improved_metric": [
                name for name, improved in (("pb", pb_improved), ("within", within_improved))
                if improved
            ],
            "oof_metrics": _public_metric_row(pair_metrics),
            "best_singleton": {"pb": best_single_pb, "within": best_single_within},
        })

    composition_contract = {
        "schema": "fusion-independence-atlas-v1/composition-contract-v1",
        "status": "BLOCKED_MISSING_TYPED_PIPELINE_COMPOSITION",
        "independent_recipes": finalist_recipes,
        "composable": False,
        "derivable_uniquely_from_current_incumbent": False,
        "judgment": (
            "The current artifacts define five independently evaluated interventions, "
            "not five serial ports. Background, token, step and decoder finalists each "
            "emit an alternative locator representation, so the incumbent code does not "
            "determine how one finalist replaces or augments the next."
        ),
        "missing_rules": [
            "routing of a background residual into the selected token-fusion member bank",
            "precedence or typed merge rule among background, token and post-readout step locators",
            "rule for reapplying the selected decoder to an upstream fused step trajectory",
        ],
        "smallest_material_user_decisions": [
            {
                "id": "locator_composition_topology",
                "question": (
                    "Are background, token, step and decoder finalists mutually exclusive "
                    "locator alternatives, or a serial augmentation pipeline?"
                ),
                "options": ["mutually_exclusive_locator", "serial_augmentation"],
            },
            {
                "id": "serial_replacement_port",
                "required_if": "locator_composition_topology=serial_augmentation",
                "question": (
                    "At each downstream stage, which registered member is replaced by the "
                    "upstream fused output (or is it appended as a new member), and are "
                    "downstream label-free weights refit?"
                ),
            },
        ],
        "minimal_implementable_resolution": (
            "Choose mutually_exclusive_locator and run four locator alternatives plus the "
            "gate as a two-stage experiment; a 2^5 factorial is not defined under that choice."
        ),
        "forbidden_fallback": "flat fusion of alternative targets or resolutions",
        "factorial_results_emitted": False,
    }

    # Leave-one-signal-out uses the same five outer folds and label-free fits.
    loso_rows = []
    for point, finalist in sorted(finalists.items()):
        model = model_by_name[finalist["name"]]
        if len(model["members"]) < 2:
            continue
        for removed in model["members"]:
            members = [value for value in model["members"] if value != removed]
            indexes = np.asarray([name_to_index[value] for value in members], dtype=np.int64)
            reduced = {
                **model,
                "name": f"{model['name']}::without::{removed}",
                "members": members,
                "member_indexes": indexes,
                "families": tuple(consolidated.families[index] for index in indexes),
                "head": "singleton" if len(members) == 1 else (
                    "equal_rank" if model["head"] == "iu" and len(members) < 3 else model["head"]
                ),
            }
            rows = [outer_evaluation(reduced, outer, False) for outer in range(5)]
            loso_rows.append({
                "point": point, "finalist": finalist["name"], "removed": removed,
                **_aggregate_metric_rows(rows),
            })

    enriched_groups = []
    for model in model_rows:
        row = {key: _json_ready(value) for key, value in model.items() if key != "member_indexes"}
        if model["name"] in aggregate_by_name:
            row["oof_metrics"] = aggregate_by_name[model["name"]]
        if len(model["members"]) > 1:
            member_set = frozenset(str(value) for value in model["members"])
            row["outer_group_screens"] = [
                item for item in outer_group_rows
                if str(item.get("point")) == str(model["point"])
                and frozenset(str(value) for value in item.get("members", ())) == member_set
            ]
        row["selected_outer_count"] = selection_counts.get(model["name"], 0)
        enriched_groups.append(row)
    enriched_groups.extend(dependent_pair_rows)
    pareto = _pareto_rows([
        row for row in aggregate_rows if not row.get("iu_diagnostic_only", False)
    ])
    missing_finalists = sorted(set(points) - set(finalists))
    unsupported = {}
    if missing_finalists:
        unsupported["roster_stability"] = (
            "no model recurred in at least " + str(roster_stability) + "/5 folds for: "
            + ", ".join(missing_finalists)
        )

    _write_json(output_root / "ALL_GROUPS.json", {
        "schema": "fusion-independence-atlas-v1/all-groups-v2",
        "rows": enriched_groups, "maximum_size": maximum_size,
        "iu_eligible_only": bool(iu_eligible_only),
        "group_level_compatibility_required": True,
        "dependent_complementary": {
            "evaluated_pairs": len(dependent_pair_rows),
            "classified": sum(
                row["status"] == DependenceStatus.DEPENDENT_COMPLEMENTARY.value
                for row in dependent_pair_rows
            ),
            "criterion": "held-fold unique singleton success plus OOF fusion improvement",
            "diagnostic_only": True,
        },
    })
    _write_json(output_root / "NESTED_SELECTION.json", {
        "schema": "fusion-independence-atlas-v1/nested-selection-v2",
        "folds": nested_selection, "roster_stability_required": f"{roster_stability}/5",
        "iu_diagnostics": iu_diagnostics,
        "labels_never_passed_to_weight_fit": True,
        "background_pair_excluded_inner_consumed": True,
    })
    _write_json(output_root / "PARETO.json", {
        "schema": "fusion-independence-atlas-v1/pareto-v2", "rows": pareto,
        "axes": ["pb", "within", "family_count", "cost", "access_scopes"],
        "baseline_independent": True,
    })
    _write_json(output_root / "LEAVE_ONE_SIGNAL_OUT.json", {
        "schema": "fusion-independence-atlas-v1/leave-one-signal-out-v1",
        "rows": loso_rows,
    })
    _write_json(output_root / "FUSION_ORDER_SENSITIVITY.json", order_sensitivity)
    _write_json(output_root / "UNCERTAINTY.json", uncertainty)
    _write_json(output_root / "DEPENDENT_COMPLEMENTARY.json", {
        "schema": "fusion-independence-atlas-v1/dependent-complementary-v1",
        "rows": dependent_pair_rows,
        "promotion_eligible": False,
        "reason": "these pairs failed the registered independence screen",
    })
    _write_json(output_root / "COMPOSITION_CONTRACT.json", composition_contract)
    _write_json(output_root / "FINALISTS.json", {
        "schema": "fusion-independence-atlas-v1/finalists-v2",
        "finalists": finalists, "unsupported_points": unsupported,
        "complete_five_point_factorial_ready": False,
        "composition_contract": "COMPOSITION_CONTRACT.json",
        "uncertainty": "UNCERTAINTY.json",
        "fusion_order_sensitivity": "FUSION_ORDER_SENSITIVITY.json",
    })
    summary = {
        "status": "PARTIAL_FAIL_CLOSED",
        "groups": len(model_rows), "pareto_rows": len(pareto),
        "dependent_complementary_pairs": sum(
            row["status"] == DependenceStatus.DEPENDENT_COMPLEMENTARY.value
            for row in dependent_pair_rows
        ),
        "finalists": finalists, "unsupported_points": unsupported,
        "baseline_independent": True,
        "factorial_results": [],
        "factorial_status": "UNRESOLVED_PIPELINE_COMPOSITION_NOT_FABRICATED",
        "composition_contract": composition_contract,
        "required_user_decisions": composition_contract["smallest_material_user_decisions"],
        "uncertainty": uncertainty,
        "fusion_order_sensitivity": order_sensitivity,
    }
    _write_json(output_root / "SUMMARY.json", summary)
    return summary


__all__ = [
    "AtlasBackendError",
    "BootstrapSamples",
    "ConditionedErrors",
    "DependenceStatus",
    "ErrorMatrix",
    "ErrorTarget",
    "FullPopulationGateBootstrap",
    "GateErrorMatrices",
    "GroupDiagnostics",
    "NuisanceDesign",
    "ObservationMetadata",
    "PairBootstrapResult",
    "PairDiagnostics",
    "PairInterval",
    "Resolution",
    "all_pair_diagnostics",
    "bootstrap_pair_diagnostics",
    "build_compatibility_graph",
    "classify_pair",
    "combine_error_matrices",
    "cross_fit_nuisance",
    "deterministic_group_folds",
    "enumerate_cliques",
    "fit_label_free_fusion_weights",
    "final_pb_error_matrix",
    "full_population_gate_bootstrap",
    "gate_error_matrices_from_scores",
    "gate_false_close_matrix",
    "gate_false_open_matrix",
    "group_diagnostics",
    "nuisance_design",
    "pair_diagnostics",
    "pb_raw_locator_miss_matrix",
    "predictor_residual_matrix",
    "prmb_pairwise_misorder_matrix",
    "rank_fused_gate_decisions",
    "run_atlas_dependence",
    "run_nested_fusion_search",
    "apply_label_free_fusion_weights",
    "simultaneous_intervals",
    "source_group_bootstrap",
]
