"""Label-free core-method layer for reconstruction benchmark v1."""

from .contracts import (
    CONTRACT_VERSION,
    FitStatus,
    MethodSpec,
    OUTPUT_SCORE_SEMANTICS,
    POSITIVE_CLASS,
    PREPARED_MATRIX_SEMANTICS,
    PreparedCell,
    SCORE_SEMANTICS_CONVERSION,
    ScoreResult,
    canonical_sha256,
    prepared_matrix_sha256,
)
from .methods import (
    PRIMARY_METHOD_IDS,
    PRIMARY_METHOD_SPECS,
    MethodFitError,
    run_all_methods,
    run_method,
)

__all__ = [
    "CONTRACT_VERSION",
    "FitStatus",
    "MethodFitError",
    "MethodSpec",
    "OUTPUT_SCORE_SEMANTICS",
    "POSITIVE_CLASS",
    "PREPARED_MATRIX_SEMANTICS",
    "PRIMARY_METHOD_IDS",
    "PRIMARY_METHOD_SPECS",
    "PreparedCell",
    "SCORE_SEMANTICS_CONVERSION",
    "ScoreResult",
    "canonical_sha256",
    "prepared_matrix_sha256",
    "run_all_methods",
    "run_method",
]
