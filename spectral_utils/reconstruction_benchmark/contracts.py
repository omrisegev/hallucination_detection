"""Strict, label-free input and output contracts for reconstruction fits.

The reconstruction benchmark has one feature boundary.  Raw telemetry is
transformed by :mod:`spectral_utils.dufs_liu_feature_contract` exactly once;
the methods in this package receive only that frozen, confidence-oriented
matrix.  They never infer or change the direction of individual columns.

Every successful method returns one score with the benchmark convention
``higher_is_incorrect``.  This is only a *global* score conversion.  It does
not alter the prepared feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..dufs_liu_feature_contract import CONTRACT_VERSION
from ..specrage_views import FEATURE_TO_VIEW


PREPARED_MATRIX_SEMANTICS = "higher_is_confidence"
OUTPUT_SCORE_SEMANTICS = "higher_is_incorrect"
POSITIVE_CLASS = "incorrect"

# This exact record is copied into every ScoreResult, including failures.  A
# solver may work in confidence coordinates and negate its final score, or use
# the common risk view ``-X``.  Both are the same one-time *global* conversion;
# per-feature reorientation after preparation is forbidden.
SCORE_SEMANTICS_CONVERSION = MappingProxyType({
    "prepared_matrix_semantics": PREPARED_MATRIX_SEMANTICS,
    "output_score_semantics": OUTPUT_SCORE_SEMANTICS,
    "global_sign_anchor": "equal_family_mean_of_prepared_confidence_coordinates",
    "global_sign_rule": (
        "for a native confidence score with unresolved global sign, multiply by "
        "the sign of its Pearson correlation with the equal-family confidence "
        "anchor; require absolute correlation above 1e-6"
    ),
    "operation_after_global_sign_resolution": "confidence_to_risk_negation",
    "per_feature_reorientation_after_preparation": "forbidden",
})

_EPS = 1e-12
_STANDARDIZATION_ATOL = 1e-7


class FitStatus(str, Enum):
    """Machine-readable outcome for one method on one prepared cell."""

    OK = "OK"
    OK_FALLBACK = "OK_FALLBACK"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
    FIT_FAILED = "FIT_FAILED"
    INPUT_INVALID = "INPUT_INVALID"


def _canonical_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _canonical_jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not canonically serializable: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-like value using one deterministic representation."""

    payload = json.dumps(
        _canonical_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepared_matrix_sha256(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    row_ids: Sequence[str],
) -> str:
    """Hash the exact row order, feature order, contract, and float64 bytes."""

    values = np.ascontiguousarray(np.asarray(matrix, dtype="<f8"))
    header = {
        "contract": CONTRACT_VERSION,
        "dtype": "float64-le",
        "shape": list(values.shape),
        "feature_names": [str(name) for name in feature_names],
        "row_ids": [str(row_id) for row_id in row_ids],
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_canonical_feature_order(names: tuple[str, ...]) -> None:
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered feature(s): " + ", ".join(unknown))
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    # FEATURE_TO_VIEW is the dedicated label-free 30-column roster and its
    # insertion order is frozen.  Do not import subset_sweep here: that legacy
    # module also defines GOOD/LOCO label-selected references and therefore
    # must stay outside the physical fitting import graph.
    expected = tuple(name for name in FEATURE_TO_VIEW if name in set(names))
    if names != expected:
        raise ValueError(
            "feature_names must preserve the frozen label-free 30-feature order"
        )


def _validate_mixed_v2_standardization(matrix: np.ndarray) -> None:
    """Catch raw/wrong-contract inputs before any method sees them.

    Mixed-v2 ends with population z-scoring.  A constant raw column therefore
    becomes all zero; every other column has mean zero and population standard
    deviation one.  This numerical check cannot prove provenance on its own,
    so the exact preprocessing-step and hash checks remain mandatory.
    """

    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    if np.max(np.abs(means)) > _STANDARDIZATION_ATOL:
        raise ValueError("prepared matrix is not centered as required by mixed-v2")
    valid_scale = np.logical_or(
        np.abs(scales - 1.0) <= _STANDARDIZATION_ATOL,
        scales <= _EPS,
    )
    if not np.all(valid_scale):
        bad = np.flatnonzero(~valid_scale).tolist()
        raise ValueError(
            "prepared matrix is not mixed-v2 standardized; bad columns "
            + repr(bad)
        )


@dataclass(frozen=True)
class PreparedCell:
    """The only cell object accepted by reconstruction methods.

    There is deliberately no label, correctness, answer-key, judge, or metric
    field.  The strict :meth:`from_mapping` constructor rejects every unknown
    key so an upstream table cannot accidentally carry targets across the fit
    boundary.
    """

    population_id: str
    cell_id: str
    domain: str
    matrix: np.ndarray = field(repr=False)
    feature_names: tuple[str, ...]
    row_ids: tuple[str, ...]
    feature_contract: str = CONTRACT_VERSION
    preprocessing_steps: tuple[str, ...] = (CONTRACT_VERSION,)
    preprocessed: bool = True
    declared_matrix_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("population_id", "cell_id", "domain"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be nonempty")
        if self.feature_contract != CONTRACT_VERSION:
            raise ValueError(
                f"expected feature contract {CONTRACT_VERSION!r}, got "
                f"{self.feature_contract!r}"
            )
        if not self.preprocessed:
            raise ValueError("methods require an already-prepared mixed-v2 matrix")
        if tuple(self.preprocessing_steps) != (CONTRACT_VERSION,):
            raise ValueError(
                "preprocessing_steps must contain mixed-v2 exactly once; "
                "double or alternate preprocessing is forbidden"
            )

        values = np.array(self.matrix, dtype=float, order="C", copy=True)
        names = tuple(str(name) for name in self.feature_names)
        rows = tuple(str(row_id) for row_id in self.row_ids)
        if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 3:
            raise ValueError("matrix must have shape (n>=3, p>=3)")
        if values.shape != (len(rows), len(names)):
            raise ValueError("matrix, row_ids, and feature_names disagree")
        if not np.isfinite(values).all():
            raise ValueError("prepared matrix contains non-finite values")
        if any(not row_id for row_id in rows):
            raise ValueError("row_ids must be nonempty")
        if len(set(rows)) != len(rows):
            raise ValueError("row_ids must be unique")
        _validate_canonical_feature_order(names)
        _validate_mixed_v2_standardization(values)

        observed_hash = prepared_matrix_sha256(values, names, rows)
        if (
            self.declared_matrix_sha256 is not None
            and str(self.declared_matrix_sha256) != observed_hash
        ):
            raise ValueError("declared_matrix_sha256 does not match prepared matrix")

        values.setflags(write=False)
        object.__setattr__(self, "matrix", values)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "preprocessing_steps", tuple(self.preprocessing_steps))
        object.__setattr__(self, "declared_matrix_sha256", observed_hash)

    @property
    def matrix_sha256(self) -> str:
        return str(self.declared_matrix_sha256)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreparedCell":
        """Build from a strict payload and reject target-bearing extra keys."""

        allowed = {
            "population_id",
            "cell_id",
            "domain",
            "matrix",
            "feature_names",
            "row_ids",
            "feature_contract",
            "preprocessing_steps",
            "preprocessed",
            "declared_matrix_sha256",
        }
        extra = sorted(set(value) - allowed)
        if extra:
            raise ValueError(
                "PreparedCell payload contains forbidden/unknown fields: "
                + ", ".join(str(key) for key in extra)
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class MethodSpec:
    """Frozen identity and label-free configuration of one core arm."""

    method_id: str
    method_version_id: str
    display_name: str
    config: Mapping[str, Any]
    development_status: str
    source: str

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.config)


@dataclass(frozen=True)
class ScoreResult:
    """One fit result before labels or evaluation are opened."""

    method_id: str
    method_version_id: str
    config_sha256: str
    status: FitStatus
    score: np.ndarray | None
    population_id: str
    cell_id: str
    feature_contract: str
    prepared_matrix_sha256: str
    selected_features: tuple[str, ...] = ()
    fallback_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict, repr=False)
    score_semantics: str = OUTPUT_SCORE_SEMANTICS
    positive_class: str = POSITIVE_CLASS
    score_semantics_conversion: Mapping[str, str] = field(
        default_factory=lambda: dict(SCORE_SEMANTICS_CONVERSION)
    )

    def __post_init__(self) -> None:
        if self.score_semantics != OUTPUT_SCORE_SEMANTICS:
            raise ValueError("ScoreResult must use higher_is_incorrect semantics")
        if self.positive_class != POSITIVE_CLASS:
            raise ValueError("ScoreResult positive class must be incorrect")
        if dict(self.score_semantics_conversion) != dict(SCORE_SEMANTICS_CONVERSION):
            raise ValueError("score semantics conversion record changed")
        successful = self.status in (FitStatus.OK, FitStatus.OK_FALLBACK)
        if successful:
            if self.score is None:
                raise ValueError("successful ScoreResult requires a score")
            values = np.array(self.score, dtype=float, copy=True)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("successful score must be one finite vector")
            values.setflags(write=False)
            object.__setattr__(self, "score", values)
        elif self.score is not None:
            raise ValueError("failed/non-applicable ScoreResult cannot carry a score")
        if self.status == FitStatus.OK_FALLBACK and not self.fallback_reason:
            raise ValueError("OK_FALLBACK requires an explicit fallback_reason")

    @property
    def scores(self) -> np.ndarray | None:
        """Plural alias for callers that write prediction tables."""

        return self.score


__all__ = [
    "CONTRACT_VERSION",
    "FitStatus",
    "MethodSpec",
    "OUTPUT_SCORE_SEMANTICS",
    "POSITIVE_CLASS",
    "PREPARED_MATRIX_SEMANTICS",
    "PreparedCell",
    "SCORE_SEMANTICS_CONVERSION",
    "ScoreResult",
    "canonical_sha256",
    "prepared_matrix_sha256",
]
