"""Restricted fit logic for the frozen localization adapter.

Only the common token IU-PCR head is fitted.  The thirteen response risks have
already passed the independent external A/B reconstruction certificate and are
treated as immutable inputs.  No target, source-group, or error-family field is
accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io import atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file
from .localization_contract import (
    COMBINED_ADAPTER_ID,
    FIT_TOKEN_CAP,
    RESPONSE_ONLY_ADAPTER_ID,
    SCORE_SCHEMA_VERSION,
    TOKEN_CONTRACT_ID,
    TOKEN_ONLY_ADAPTER_ID,
    PreparedLocalizationCell,
    empirical_midrank,
    payload_sha256,
    primary_system_roster,
)
from ..upcr import upcr_fit


@dataclass(frozen=True)
class LocalizationScoreBundle:
    cell_id: str
    population_id: str
    dataset_id: str
    model_id: str
    slice_id: str
    row_ids: tuple[str, ...]
    segment_offsets: np.ndarray
    system_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    adapter_ids: tuple[str, ...]
    system_scores: np.ndarray
    token_step_score: np.ndarray
    token_fit_diagnostics: Mapping[str, Any]
    external_certificate_sha256: str
    external_score_bindings_sha256: str
    token_transform_sha256: str


def _fit_token_iu(cell: PreparedLocalizationCell) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(cell.token_confidence, dtype=np.float64)
    if len(values) > FIT_TOKEN_CAP:
        fit_indices = np.linspace(0, len(values) - 1, FIT_TOKEN_CAP, dtype=np.int64)
        fit = values[fit_indices]
    else:
        fit_indices = np.arange(len(values), dtype=np.int64)
        fit = values
    medians = np.median(fit, axis=0)
    clean_fit = np.where(np.isfinite(fit), fit, medians[None, :])
    scale = clean_fit.std(axis=0)
    keep = np.isfinite(medians) & np.isfinite(scale) & (scale > 1e-8)
    if int(keep.sum()) < 3:
        raise RuntimeError("token IU-PCR has fewer than three nondegenerate streams")
    mean = clean_fit[:, keep].mean(axis=0)
    std = clean_fit[:, keep].std(axis=0)
    standardized_fit = (clean_fit[:, keep] - mean[None, :]) / std[None, :]
    fitted = upcr_fit(
        standardized_fit.T,
        loss="l2",
        exclusion=False,
        difficulty_gate=False,
        simple_avg_fallback=False,
        recompute_after_exclusion=False,
        g2_projection_k=1,
        scale_ratio=0.25,
        n_components=2,
        auto_components=False,
    )
    weights = np.asarray(fitted.w, dtype=np.float64)
    anchor = standardized_fit.mean(axis=1)
    raw_fit_score = standardized_fit @ weights
    correlation = float(np.corrcoef(raw_fit_score, anchor)[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    if flipped:
        weights = -weights
    selected = values[:, keep]
    clean = np.where(np.isfinite(selected), selected, mean[None, :])
    standardized = (clean - mean[None, :]) / std[None, :]
    token_risk = -(standardized @ weights)
    if token_risk.shape != (len(values),) or not np.isfinite(token_risk).all():
        raise RuntimeError("token IU-PCR produced an invalid risk curve")
    diagnostics = {
        "schema_version": "localization-token-iu-fit-v1",
        "token_contract_id": TOKEN_CONTRACT_ID,
        "n_tokens": len(values),
        "n_fit_tokens": len(fit),
        "fit_token_cap": FIT_TOKEN_CAP,
        "fit_index_sha256": payload_sha256(fit_indices.tolist()),
        "n_input_streams": values.shape[1],
        "n_kept_streams": int(keep.sum()),
        "kept_stream_mask": keep.astype(int).tolist(),
        "components": 2,
        "scale_ratio": 0.25,
        "g2_hat": float(fitted.g2_hat),
        "projection_residual": float(fitted.proj_residual),
        "confidence_anchor_correlation": correlation,
        "orientation_flipped": flipped,
        "labels_seen_during_fit": False,
    }
    diagnostics["fit_sha256"] = payload_sha256(diagnostics)
    return token_risk, diagnostics


def _step_maxima(cell: PreparedLocalizationCell, token_risk: np.ndarray) -> np.ndarray:
    scores = np.empty(len(cell.segment_starts), dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(cell.segment_starts, cell.segment_ends)):
        scores[index] = float(np.max(token_risk[int(lo):int(hi)]))
    if not np.isfinite(scores).all():
        raise RuntimeError("token-to-step reducer produced non-finite risk")
    return scores


def fit_localization_cell(cell: PreparedLocalizationCell) -> LocalizationScoreBundle:
    token_risk, diagnostics = _fit_token_iu(cell)
    raw_step_score = _step_maxima(cell, token_risk)
    step_rank = empirical_midrank(raw_step_score)
    segment_counts = np.diff(cell.segment_offsets)
    systems = primary_system_roster(cell.method_ids)
    rows: list[np.ndarray] = []
    for method_index, method_id in enumerate(cell.method_ids):
        response_rank = empirical_midrank(cell.response_scores[method_index])
        expanded_response = np.repeat(response_rank, segment_counts)
        combined = np.sqrt(expanded_response * step_rank)
        if not np.isfinite(combined).all():
            raise RuntimeError(f"{method_id}: geometric localization adapter is non-finite")
        rows.append(combined)
    for method_index, _method_id in enumerate(cell.method_ids):
        response_rank = empirical_midrank(cell.response_scores[method_index])
        rows.append(np.repeat(response_rank, segment_counts))
    rows.append(step_rank)
    matrix = np.vstack(rows)
    if matrix.shape != (27, len(raw_step_score)):
        raise AssertionError("localization score bundle is not 27 x segments")
    return LocalizationScoreBundle(
        cell_id=cell.cell_id,
        population_id=cell.population_id,
        dataset_id=cell.dataset_id,
        model_id=cell.model_id,
        slice_id=cell.slice_id,
        row_ids=cell.row_ids,
        segment_offsets=np.asarray(cell.segment_offsets, dtype=np.int64),
        system_ids=tuple(row["system_id"] for row in systems),
        method_ids=tuple(row["method_id"] for row in systems),
        adapter_ids=tuple(row["adapter_id"] for row in systems),
        system_scores=matrix,
        token_step_score=raw_step_score,
        token_fit_diagnostics=diagnostics,
        external_certificate_sha256=cell.external_certificate_sha256,
        external_score_bindings_sha256=cell.external_score_bindings_sha256,
        token_transform_sha256=cell.token_transform_sha256,
    )


def write_localization_score_bundle(
    bundle: LocalizationScoreBundle,
    output_dir: str | Path,
) -> dict[str, Any]:
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"localization score output is not empty: {target}")
    target.mkdir(parents=True, exist_ok=False)
    score_path = target / "scores.npz"
    score_sha = atomic_write_npz(score_path, {
        "row_ids": np.asarray(bundle.row_ids, dtype="<U80"),
        "segment_offsets": np.asarray(bundle.segment_offsets, dtype="<i8"),
        "system_ids": np.asarray(bundle.system_ids, dtype="<U96"),
        "method_ids": np.asarray(bundle.method_ids, dtype="<U48"),
        "adapter_ids": np.asarray(bundle.adapter_ids, dtype="<U64"),
        "system_scores": np.asarray(bundle.system_scores, dtype="<f8"),
        "token_step_score": np.asarray(bundle.token_step_score, dtype="<f8"),
        "external_certificate_sha256": np.asarray(
            [bundle.external_certificate_sha256], dtype="<U64"
        ),
        "external_score_bindings_sha256": np.asarray(
            [bundle.external_score_bindings_sha256], dtype="<U64"
        ),
        "token_transform_sha256": np.asarray(
            [bundle.token_transform_sha256], dtype="<U64"
        ),
    })
    record = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "cell_id": bundle.cell_id,
        "population_id": bundle.population_id,
        "dataset_id": bundle.dataset_id,
        "model_id": bundle.model_id,
        "slice_id": bundle.slice_id,
        "n_rows": len(bundle.row_ids),
        "n_segments": len(bundle.token_step_score),
        "n_systems": len(bundle.system_ids),
        "system_ids": list(bundle.system_ids),
        "method_ids": list(bundle.method_ids),
        "adapter_ids": list(bundle.adapter_ids),
        "score_semantics": "higher_is_localization_risk",
        "combined_adapter_formula": "sqrt(empirical_midrank(response_risk) * empirical_midrank(step_risk))",
        "historical_075_025_blend_used": False,
        "response_scores_refit": False,
        "token_scores_fit_without_targets": True,
        "external_certificate_sha256": bundle.external_certificate_sha256,
        "external_score_bindings_sha256": bundle.external_score_bindings_sha256,
        "token_transform_sha256": bundle.token_transform_sha256,
        "token_fit_diagnostics": dict(bundle.token_fit_diagnostics),
        "score_path": score_path.name,
        "score_sha256": score_sha,
    }
    record["record_sha256"] = payload_sha256(record)
    atomic_write_json(target / "RECORD.json", record)
    return record


def load_localization_score_bundle(
    record_path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(record_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(record)
    recorded = payload.pop("record_sha256", None)
    if recorded != payload_sha256(payload):
        raise RuntimeError("localization score record hash failed")
    if record.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise RuntimeError("unexpected localization score schema")
    score_path = (path.parent / str(record["score_path"])).resolve()
    try:
        score_path.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise RuntimeError("localization score artifact escaped its record directory") from exc
    if sha256_file(score_path) != record.get("score_sha256"):
        raise RuntimeError("localization score artifact hash failed")
    arrays = load_npz_no_pickle(score_path)
    expected = {
        "row_ids", "segment_offsets", "system_ids", "method_ids", "adapter_ids",
        "system_scores", "token_step_score", "external_certificate_sha256",
        "external_score_bindings_sha256", "token_transform_sha256",
    }
    if set(arrays) != expected:
        raise RuntimeError("localization score artifact contains unknown arrays")
    if tuple(map(str, arrays["system_ids"].tolist())) != tuple(record["system_ids"]):
        raise RuntimeError("localization score system roster drifted")
    scores = np.asarray(arrays["system_scores"], dtype=np.float64)
    offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
    if (
        scores.shape != (int(record["n_systems"]), int(record["n_segments"]))
        or offsets.shape != (int(record["n_rows"]) + 1,)
        or offsets[-1] != scores.shape[1]
        or not np.isfinite(scores).all()
    ):
        raise RuntimeError("localization score matrix/offsets are malformed")
    return record, arrays


__all__ = [
    "LocalizationScoreBundle", "fit_localization_cell",
    "load_localization_score_bundle", "write_localization_score_bundle",
]
