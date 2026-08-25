"""Evaluation-only reader for frozen RAG score archives.

This intentionally does not import the fit/scoring stack.  Post-freeze
evaluation only needs to parse the authenticated NPZ and validate its frozen
array schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .io import load_npz_no_pickle_bytes
from .rag_evidence_contract import (
    SCORE_SCHEMA,
    RagEvidenceContractError,
    read_bound_file_bytes,
)


SCORE_ARRAY_NAMES = (
    "schema_version",
    "rag_dev_response_id", "rag_dev_response_task", "rag_dev_response_score",
    "rag_dev_sentence_id", "rag_dev_sentence_score",
    "rag_dev_token_parent_id", "rag_dev_token_index", "rag_dev_token_score",
    "rag_test_response_id", "rag_test_response_task", "rag_test_response_score",
    "rag_test_sentence_id", "rag_test_sentence_score",
    "rag_test_token_parent_id", "rag_test_token_index", "rag_test_token_score",
    "gasp_sentence_id", "gasp_task", "gasp_threshold_score", "gasp_fixed_rag_score",
    "lettuce_unit_id", "lettuce_prediction", "lettuce_max_probability",
    "refchecker_unit_id", "refchecker_setting", "refchecker_nli_prediction",
    "refchecker_binary_score",
)


def validate_score_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(SCORE_ARRAY_NAMES):
        raise RagEvidenceContractError(
            "RAG score array roster drifted: "
            f"{sorted(set(arrays) ^ set(SCORE_ARRAY_NAMES))}"
        )
    if np.asarray(arrays["schema_version"]).tolist() != [SCORE_SCHEMA]:
        raise RagEvidenceContractError("RAG score schema array drifted")
    aligned = (
        ("rag_dev_response_id", "rag_dev_response_task", "rag_dev_response_score"),
        ("rag_dev_sentence_id", "rag_dev_sentence_score"),
        ("rag_dev_token_parent_id", "rag_dev_token_index", "rag_dev_token_score"),
        ("rag_test_response_id", "rag_test_response_task", "rag_test_response_score"),
        ("rag_test_sentence_id", "rag_test_sentence_score"),
        ("rag_test_token_parent_id", "rag_test_token_index", "rag_test_token_score"),
        ("gasp_sentence_id", "gasp_task", "gasp_threshold_score", "gasp_fixed_rag_score"),
        ("lettuce_unit_id", "lettuce_prediction", "lettuce_max_probability"),
        (
            "refchecker_unit_id", "refchecker_setting",
            "refchecker_nli_prediction", "refchecker_binary_score",
        ),
    )
    for names in aligned:
        lengths = {len(np.asarray(arrays[name])) for name in names}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise RagEvidenceContractError(
                f"RAG score alignment/coverage failed: {names}"
            )
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise RagEvidenceContractError(
                f"object dtype forbidden in RAG scores: {name}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise RagEvidenceContractError(f"non-finite RAG scores: {name}")
    for name in (
        "rag_dev_response_id", "rag_dev_sentence_id", "rag_test_response_id",
        "rag_test_sentence_id", "gasp_sentence_id", "lettuce_unit_id",
        "refchecker_unit_id",
    ):
        values = np.asarray(arrays[name]).astype(str).tolist()
        if len(values) != len(set(values)):
            raise RagEvidenceContractError(f"duplicate RAG score unit IDs: {name}")
    for split in ("dev", "test"):
        parents = np.asarray(arrays[f"rag_{split}_token_parent_id"]).astype(str)
        indexes = np.asarray(arrays[f"rag_{split}_token_index"], dtype=np.int64)
        if (indexes < 0).any():
            raise RagEvidenceContractError(
                f"negative RAG {split} scorer-token index"
            )
        lattice = list(zip(parents.tolist(), indexes.tolist(), strict=True))
        if len(lattice) != len(set(lattice)):
            raise RagEvidenceContractError(
                f"duplicate RAG {split} scorer-token lattice key"
            )


def load_scores_bytes(payload: bytes) -> dict[str, np.ndarray]:
    arrays = load_npz_no_pickle_bytes(payload)
    validate_score_arrays(arrays)
    return arrays


def load_scores(
    path: str | Path, *, expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    payload = read_bound_file_bytes(
        path, expected_sha256=expected_sha256, name="RAG score archive",
    )
    return load_scores_bytes(payload)


__all__ = [
    "SCORE_ARRAY_NAMES", "load_scores", "load_scores_bytes",
    "validate_score_arrays",
]
