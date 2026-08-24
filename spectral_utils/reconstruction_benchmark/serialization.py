"""Serialization of label-free scores and method artifacts."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from .contracts import ScoreResult
from .io import atomic_write_json, atomic_write_npz, sha256_file


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_token(value: str) -> str:
    output = _SAFE.sub("_", str(value)).strip("_")
    return output or "value"


def jsonable(value: Any, *, nonfinite_paths: list[str] | None = None, path: str = "$") -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), nonfinite_paths=nonfinite_paths, path=path)
    if isinstance(value, np.generic):
        return jsonable(value.item(), nonfinite_paths=nonfinite_paths, path=path)
    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value)
        return {
            "type": "csr_matrix",
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
        }
    if isinstance(value, Mapping):
        return {
            str(key): jsonable(item, nonfinite_paths=nonfinite_paths, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            jsonable(item, nonfinite_paths=nonfinite_paths, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite_paths is not None:
            nonfinite_paths.append(path)
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flatten_artifact(
    prefix: str,
    value: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    key = _safe_token(prefix)
    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value)
        arrays[f"{key}__data"] = np.asarray(matrix.data)
        arrays[f"{key}__indices"] = np.asarray(matrix.indices, dtype="<i8")
        arrays[f"{key}__indptr"] = np.asarray(matrix.indptr, dtype="<i8")
        arrays[f"{key}__shape"] = np.asarray(matrix.shape, dtype="<i8")
        metadata[key] = {"type": "csr_matrix", "shape": list(matrix.shape)}
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            metadata[key] = jsonable(value)
        else:
            arrays[key] = array
            metadata[key] = {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}
        return
    if isinstance(value, Mapping):
        for child, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            _flatten_artifact(f"{key}__{_safe_token(str(child))}", item, arrays, metadata)
        return
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except Exception:
            array = np.asarray([], dtype=float)
        if array.size and not array.dtype.hasobject:
            arrays[key] = array
            metadata[key] = {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}
        else:
            metadata[key] = jsonable(value)
        return
    if isinstance(value, np.generic):
        value = value.item()
    metadata[key] = jsonable(value)


def write_score_result(
    result: ScoreResult,
    row_ids: tuple[str, ...],
    out_dir: str | Path,
) -> dict:
    target = Path(out_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"score-result directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": "reconstruction-score-record-v1",
        "method_id": result.method_id,
        "method_version_id": result.method_version_id,
        "config_sha256": result.config_sha256,
        "status": result.status.value,
        "population_id": result.population_id,
        "cell_id": result.cell_id,
        "feature_contract": result.feature_contract,
        "prepared_matrix_sha256": result.prepared_matrix_sha256,
        "score_semantics": result.score_semantics,
        "positive_class": result.positive_class,
        "score_semantics_conversion": dict(result.score_semantics_conversion),
        "selected_features": list(result.selected_features),
        "fallback_reason": result.fallback_reason,
    }
    nonfinite_paths: list[str] = []
    record["diagnostics"] = jsonable(
        result.diagnostics,
        nonfinite_paths=nonfinite_paths,
        path="$.diagnostics",
    )
    record["nonfinite_diagnostic_paths"] = nonfinite_paths

    if result.score is not None:
        score_path = target / "score.npz"
        score_sha = atomic_write_npz(
            score_path,
            {
                "row_ids": np.asarray(row_ids, dtype="<U128"),
                "score": np.asarray(result.score, dtype="<f8"),
            },
        )
        record["score_path"] = score_path.name
        record["score_sha256"] = score_sha
        record["score_n"] = len(result.score)
    else:
        record["score_path"] = None
        record["score_sha256"] = None
        record["score_n"] = 0

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for key, value in sorted(result.artifacts.items(), key=lambda pair: str(pair[0])):
        _flatten_artifact(str(key), value, arrays, metadata)
    if arrays:
        artifacts_path = target / "artifacts.npz"
        record["artifacts_sha256"] = atomic_write_npz(artifacts_path, arrays)
        record["artifacts_path"] = artifacts_path.name
    else:
        record["artifacts_sha256"] = None
        record["artifacts_path"] = None
    atomic_write_json(target / "ARTIFACT_INDEX.json", metadata)
    record["artifact_index_sha256"] = sha256_file(target / "ARTIFACT_INDEX.json")
    record_sha = atomic_write_json(target / "RECORD.json", record)
    record["record_sha256"] = record_sha
    return record


__all__ = ["jsonable", "write_score_result"]
