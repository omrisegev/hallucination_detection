"""Target-free fitting and scoring for the separate RAG evidence panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from spectral_utils.fixed_application_pipelines import (
    CONTRACT_VERSION,
    aggregate_risk,
    fit_rag_evidence_head,
    fit_shared_mixed_transformer,
    rag_evidence_matrix,
    raw_token_feature_matrix,
)

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


def _raw_conditions(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(name): raw_token_feature_matrix(condition)
        for name, condition in row["conditions"].items()
    }


def _fit_rag_model(dev_rows: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any]:
    full_records, raw_by_id = [], {}
    for row in dev_rows:
        conditions = _raw_conditions(row)
        raw_by_id[str(row["unit_id"])] = conditions
        full_records.append((str(row["unit_id"]), conditions["full"]))
    transformer = fit_shared_mixed_transformer(full_records)
    noctx_records, loo_records = [], []
    for row in dev_rows:
        unit_id = str(row["unit_id"])
        conditions = raw_by_id[unit_id]
        matrix, names = rag_evidence_matrix(conditions, transformer, profile="noctx")
        noctx_records.append((unit_id, matrix, names))
        if any(name.startswith("loo_") for name in conditions):
            matrix, names = rag_evidence_matrix(conditions, transformer, profile="loo")
            loo_records.append((unit_id, matrix, names))
    if not loo_records:
        raise RagEvidenceContractError("RAG development fit has no LOO records")
    noctx_head = fit_rag_evidence_head(noctx_records, profile="noctx")
    loo_head = fit_rag_evidence_head(loo_records, profile="loo")
    return transformer, noctx_head, loo_head


def _risk(
    conditions: Mapping[str, np.ndarray], transformer: Any, noctx_head: Any, loo_head: Any
) -> np.ndarray:
    has_loo = any(name.startswith("loo_") for name in conditions)
    profile = "loo" if has_loo else "noctx"
    matrix, _ = rag_evidence_matrix(conditions, transformer, profile=profile)
    return (loo_head if has_loo else noctx_head).risk(matrix)


def _score_rag_split(
    rows: Sequence[Mapping[str, Any]], transformer: Any, noctx_head: Any, loo_head: Any
) -> dict[str, np.ndarray]:
    response_ids, response_tasks, response_scores = [], [], []
    sentence_ids, sentence_scores = [], []
    token_parents, token_indexes, token_scores = [], [], []
    for row in rows:
        unit_id = str(row["unit_id"])
        conditions = _raw_conditions(row)
        risk = _risk(conditions, transformer, noctx_head, loo_head)
        response_ids.append(unit_id)
        response_tasks.append(str(row["task_type"]))
        response_scores.append(aggregate_risk(risk, "mean"))
        token_parents.extend([unit_id] * len(risk))
        token_indexes.extend(range(len(risk)))
        token_scores.extend(risk.tolist())
        for sentence in row["sentence_windows"]:
            start, end = int(sentence["start"]), int(sentence["end"])
            if not 0 <= start < end <= len(risk):
                raise RagEvidenceContractError(f"invalid RAG sentence window: {sentence}")
            sentence_ids.append(str(sentence["unit_id"]))
            sentence_scores.append(aggregate_risk(risk[start:end], "mean"))
    return {
        "response_id": np.asarray(response_ids, dtype="U"),
        "response_task": np.asarray(response_tasks, dtype="U"),
        "response_score": np.asarray(response_scores, dtype=np.float64),
        "sentence_id": np.asarray(sentence_ids, dtype="U"),
        "sentence_score": np.asarray(sentence_scores, dtype=np.float64),
        "token_parent_id": np.asarray(token_parents, dtype="U"),
        "token_index": np.asarray(token_indexes, dtype=np.int32),
        "token_score": np.asarray(token_scores, dtype=np.float64),
    }


def _gasp_features(
    conditions: Mapping[str, Mapping[str, Any]], start: int, end: int
) -> dict[str, float]:
    full = conditions["full"]
    noctx = conditions["noctx"]
    full_nll = np.asarray(full["token_spilled_energies"], dtype=float)
    noctx_nll = np.asarray(noctx["token_spilled_energies"], dtype=float)
    if not 0 <= start < end <= len(full_nll):
        raise RagEvidenceContractError("invalid GASP sentence token window")
    noctx_jsd = np.asarray(noctx.get("token_jsd_vs_full"), dtype=float)
    if noctx_jsd.shape != full_nll.shape:
        raise RagEvidenceContractError("GASP exact no-context JSD is unavailable")
    loo_names = sorted(
        (name for name in conditions if name.startswith("loo_")),
        key=lambda name: int(name.split("_", 1)[1]),
    )
    if not loo_names:
        raise RagEvidenceContractError("GASP row has no leave-one-out conditions")
    drops, divergences = [], []
    for name in loo_names:
        row = conditions[name]
        values = np.asarray(row["token_spilled_energies"], dtype=float)
        exact = np.asarray(row.get("token_jsd_vs_full"), dtype=float)
        if values.shape != full_nll.shape or exact.shape != full_nll.shape:
            raise RagEvidenceContractError("GASP LOO trace/JSD alignment failed")
        drops.append(float(np.mean(values[start:end] - full_nll[start:end])))
        divergences.append(float(np.mean(exact[start:end])))
    return {
        "gasp_gap": float(np.mean(noctx_nll[start:end] - full_nll[start:end])),
        "gasp_jsd0": float(np.mean(noctx_jsd[start:end])),
        "gasp_drop": max(drops),
        "gasp_jsdloo": max(divergences),
    }


def _gasp_threshold_scores(rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    names = ("gasp_gap", "gasp_jsd0", "gasp_drop", "gasp_jsdloo")
    sensitivity = np.zeros(len(rows), dtype=np.float64)
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or values.std() <= 1e-12:
            raise RagEvidenceContractError(f"GASP feature is non-finite/degenerate: {name}")
        sensitivity += (values - values.mean()) / values.std()
    return -sensitivity


def _score_gasp(
    rows: Sequence[Mapping[str, Any]], transformer: Any, noctx_head: Any, loo_head: Any
) -> dict[str, np.ndarray]:
    identifiers, tasks, features, fixed_scores = [], [], [], []
    for row in rows:
        raw_conditions = _raw_conditions(row)
        risk = _risk(raw_conditions, transformer, noctx_head, loo_head)
        for sentence in row["sentence_windows"]:
            start, end = int(sentence["start"]), int(sentence["end"])
            identifiers.append(str(sentence["unit_id"]))
            tasks.append(str(row["task_type"]))
            features.append(_gasp_features(row["conditions"], start, end))
            fixed_scores.append(aggregate_risk(risk[start:end], "mean"))
    return {
        "sentence_id": np.asarray(identifiers, dtype="U"),
        "task": np.asarray(tasks, dtype="U"),
        "threshold_score": _gasp_threshold_scores(features),
        "fixed_rag_score": np.asarray(fixed_scores, dtype=np.float64),
    }


def compute_rag_evidence_scores(fit_input: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit on sanitized development telemetry and emit target-free scores."""

    panels = fit_input["panels"]
    dev_rows = panels["ragtruth"]["splits"]["dev"]
    transformer, noctx_head, loo_head = _fit_rag_model(dev_rows)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray([SCORE_SCHEMA], dtype="U"),
    }
    for split in ("dev", "test"):
        scored = _score_rag_split(
            panels["ragtruth"]["splits"][split], transformer, noctx_head, loo_head
        )
        arrays.update({f"rag_{split}_{key}": value for key, value in scored.items()})
    gasp = _score_gasp(
        panels["gasp"]["rows"], transformer, noctx_head, loo_head
    )
    arrays.update({f"gasp_{key}": value for key, value in gasp.items()})
    lettuce = panels["lettuce"]["rows"]
    arrays.update({
        "lettuce_unit_id": np.asarray([row["unit_id"] for row in lettuce], dtype="U"),
        "lettuce_prediction": np.asarray([row["binary_prediction"] for row in lettuce], dtype=np.uint8),
        "lettuce_max_probability": np.asarray(
            [row["maximum_token_probability"] for row in lettuce], dtype=np.float64
        ),
    })
    refchecker = panels["refchecker"]["rows"]
    ref_ids, ref_settings, nli_predictions, binary_scores = [], [], [], []
    for row in refchecker:
        conditions = _raw_conditions(row)
        risk = _risk(conditions, transformer, noctx_head, loo_head)
        ref_ids.append(str(row["unit_id"]))
        ref_settings.append(str(row["setting"]))
        nli_predictions.append(str(row["nli_prediction"]))
        binary_scores.append(aggregate_risk(risk, "mean"))
    arrays.update({
        "refchecker_unit_id": np.asarray(ref_ids, dtype="U"),
        "refchecker_setting": np.asarray(ref_settings, dtype="U"),
        "refchecker_nli_prediction": np.asarray(nli_predictions, dtype="U"),
        "refchecker_binary_score": np.asarray(binary_scores, dtype=np.float64),
    })
    validate_score_arrays(arrays, fit_input=fit_input)
    diagnostics = {
        "contract_version": CONTRACT_VERSION,
        "labels_seen_during_fit": False,
        "historical_scores_opened": False,
        "transform_fit_population": "RAGTruth development full-condition token traces",
        "noctx_head": noctx_head.diagnostics,
        "loo_head": loo_head.diagnostics,
        "panel_score_counts": {
            "rag_dev_response": len(arrays["rag_dev_response_id"]),
            "rag_test_response": len(arrays["rag_test_response_id"]),
            "rag_test_sentence": len(arrays["rag_test_sentence_id"]),
            "rag_test_token": len(arrays["rag_test_token_score"]),
            "gasp_sentence": len(arrays["gasp_sentence_id"]),
            "lettuce_example": len(arrays["lettuce_unit_id"]),
            "refchecker_claim": len(arrays["refchecker_unit_id"]),
        },
    }
    return arrays, diagnostics


def _expected_fit_score_rosters(
    fit_input: Mapping[str, Any],
) -> dict[str, list[Any]]:
    panels = fit_input["panels"]
    expected: dict[str, list[Any]] = {}
    for split in ("dev", "test"):
        rows = panels["ragtruth"]["splits"][split]
        expected[f"rag_{split}_response_id"] = [
            str(row["unit_id"]) for row in rows
        ]
        expected[f"rag_{split}_response_task"] = [
            str(row["task_type"]) for row in rows
        ]
        expected[f"rag_{split}_sentence_id"] = [
            str(window["unit_id"])
            for row in rows
            for window in row["sentence_windows"]
        ]
        token_lattice = [
            (str(row["unit_id"]), index)
            for row in rows
            for index in range(
                len(np.asarray(row["conditions"]["full"]["token_entropies"]))
            )
        ]
        expected[f"rag_{split}_token_lattice"] = token_lattice
    gasp_rows = panels["gasp"]["rows"]
    expected["gasp_sentence_id"] = [
        str(window["unit_id"])
        for row in gasp_rows
        for window in row["sentence_windows"]
    ]
    expected["gasp_task"] = [
        str(row["task_type"])
        for row in gasp_rows
        for _ in row["sentence_windows"]
    ]
    expected["lettuce_unit_id"] = [
        str(row["unit_id"]) for row in panels["lettuce"]["rows"]
    ]
    expected["refchecker_unit_id"] = [
        str(row["unit_id"]) for row in panels["refchecker"]["rows"]
    ]
    expected["refchecker_setting"] = [
        str(row["setting"]) for row in panels["refchecker"]["rows"]
    ]
    return expected


def validate_score_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    fit_input: Mapping[str, Any] | None = None,
) -> None:
    if set(arrays) != set(SCORE_ARRAY_NAMES):
        raise RagEvidenceContractError(
            f"RAG score array roster drifted: {sorted(set(arrays) ^ set(SCORE_ARRAY_NAMES))}"
        )
    if np.asarray(arrays["schema_version"]).tolist() != [SCORE_SCHEMA]:
        raise RagEvidenceContractError("RAG score schema array drifted")
    paired = (
        ("rag_dev_response_id", "rag_dev_response_task", "rag_dev_response_score"),
        ("rag_dev_sentence_id", "rag_dev_sentence_score"),
        ("rag_dev_token_parent_id", "rag_dev_token_index", "rag_dev_token_score"),
        ("rag_test_response_id", "rag_test_response_task", "rag_test_response_score"),
        ("rag_test_sentence_id", "rag_test_sentence_score"),
        ("rag_test_token_parent_id", "rag_test_token_index", "rag_test_token_score"),
        ("gasp_sentence_id", "gasp_task", "gasp_threshold_score", "gasp_fixed_rag_score"),
        ("lettuce_unit_id", "lettuce_prediction", "lettuce_max_probability"),
        ("refchecker_unit_id", "refchecker_setting", "refchecker_nli_prediction", "refchecker_binary_score"),
    )
    for names in paired:
        lengths = {len(np.asarray(arrays[name])) for name in names}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise RagEvidenceContractError(f"RAG score alignment/coverage failed: {names}")
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise RagEvidenceContractError(f"object dtype forbidden in RAG scores: {name}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise RagEvidenceContractError(f"non-finite RAG scores: {name}")
    for name in (
        "rag_dev_response_id", "rag_dev_sentence_id", "rag_test_response_id",
        "rag_test_sentence_id", "gasp_sentence_id", "lettuce_unit_id", "refchecker_unit_id",
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
    if fit_input is not None:
        expected = _expected_fit_score_rosters(fit_input)
        for name, expected_values in expected.items():
            if name.endswith("_token_lattice"):
                split = name.split("_")[1]
                observed_values = list(zip(
                    np.asarray(arrays[f"rag_{split}_token_parent_id"])
                    .astype(str).tolist(),
                    np.asarray(arrays[f"rag_{split}_token_index"], dtype=np.int64)
                    .tolist(),
                    strict=True,
                ))
            else:
                observed_values = np.asarray(arrays[name]).astype(str).tolist()
            if observed_values != expected_values:
                raise RagEvidenceContractError(
                    f"RAG score roster differs from registered fit input: {name}"
                )


def load_scores(
    path: str | Path,
    *,
    fit_input: Mapping[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    payload = read_bound_file_bytes(
        path,
        expected_sha256=expected_sha256,
        name="RAG score archive",
    )
    return load_scores_bytes(payload, fit_input=fit_input)


def load_scores_bytes(
    payload: bytes,
    *,
    fit_input: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    arrays = load_npz_no_pickle_bytes(payload)
    validate_score_arrays(arrays, fit_input=fit_input)
    return arrays


__all__ = [
    "SCORE_ARRAY_NAMES", "compute_rag_evidence_scores", "load_scores",
    "load_scores_bytes",
    "validate_score_arrays",
]
