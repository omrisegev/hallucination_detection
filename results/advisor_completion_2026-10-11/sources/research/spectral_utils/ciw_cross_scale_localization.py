"""Cross-scale innovation input layer for CIW localization.

The frozen localization input contains 29 confidence-oriented token streams
and one CIW-DEEM risk per complete response.  This module predicts each token
coordinate from two response-level coordinates: its response mean and the
CIW-DEEM response risk.  Out-of-fold predictability determines a bounded
coordinate gate, exactly as in CIW-DEEM.  The unchanged two-component IU-PCR
head then fuses the transformed token coordinates before step-wise maxima.

No targets, error positions, or comparator scores are accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from .reconstruction_benchmark.localization_contract import FIT_TOKEN_CAP
from .upcr import upcr_fit


RIDGE = 1.0
N_FOLDS = 5
MAX_GATE = 0.5
EPS = 1e-12


@dataclass(frozen=True)
class CrossScaleTokenResult:
    token_risk: np.ndarray
    step_risk: np.ndarray
    reliability: np.ndarray
    gate: np.ndarray
    diagnostics: dict[str, object]


def _folds(row_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([
        int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8], 16) % N_FOLDS
        for value in row_ids
    ], dtype=np.int64)


def _token_owners(token_offsets: np.ndarray, token_indices: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(token_offsets[1:], dtype=np.int64), token_indices, side="right")


def _ridge_coefficients(predictors: np.ndarray, target: np.ndarray) -> np.ndarray:
    predictors = np.asarray(predictors, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gram = predictors.T @ predictors / max(len(predictors), 1)
    rhs = predictors.T @ target / max(len(predictors), 1)
    return np.linalg.solve(gram + RIDGE * np.eye(predictors.shape[1]), rhs)


def fit_cross_scale_token_head(
    token_confidence: np.ndarray,
    token_offsets: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    row_ids: Sequence[str],
    ciw_response_risk: np.ndarray,
    *,
    max_gate: float = MAX_GATE,
    fusion: str = "iu",
) -> CrossScaleTokenResult:
    """Fit and apply the target-free cross-scale token input layer.

    ``max_gate=0`` is a mechanical alias of the frozen token-IU29 fit.
    """

    values = np.asarray(token_confidence, dtype=np.float64)
    offsets = np.asarray(token_offsets, dtype=np.int64)
    starts = np.asarray(segment_starts, dtype=np.int64)
    ends = np.asarray(segment_ends, dtype=np.int64)
    response_risk = np.asarray(ciw_response_risk, dtype=np.float64)
    rows = tuple(map(str, row_ids))
    if values.ndim != 2 or values.shape[1] != 29 or not np.isfinite(values).all():
        raise ValueError("token confidence must be a finite tokens-by-29 matrix")
    if offsets.shape != (len(rows) + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("token offsets are not aligned to response rows")
    if response_risk.shape != (len(rows),) or not np.isfinite(response_risk).all():
        raise ValueError("CIW response risk is not aligned to response rows")
    if starts.shape != ends.shape or np.any(ends <= starts):
        raise ValueError("step spans are malformed")
    if not 0.0 <= float(max_gate) <= 0.5:
        raise ValueError("max_gate must be in [0, 0.5]")
    if fusion not in {"iu", "su"}:
        raise ValueError("fusion must be 'iu' or 'su'")

    if len(values) > FIT_TOKEN_CAP:
        fit_indices = np.linspace(0, len(values) - 1, FIT_TOKEN_CAP, dtype=np.int64)
    else:
        fit_indices = np.arange(len(values), dtype=np.int64)
    fit = values[fit_indices]
    medians = np.median(fit, axis=0)
    clean_fit = np.where(np.isfinite(fit), fit, medians[None, :])
    scale = clean_fit.std(axis=0)
    keep = np.isfinite(medians) & np.isfinite(scale) & (scale > 1e-8)
    if int(keep.sum()) < 3:
        raise RuntimeError("fewer than three nondegenerate token streams remain")
    mean = clean_fit[:, keep].mean(axis=0)
    std = clean_fit[:, keep].std(axis=0)
    standardized_fit = (clean_fit[:, keep] - mean[None, :]) / std[None, :]

    # Whole-response context is built from the same coordinate's token mean
    # plus the separately frozen CIW response risk.  Only sampled fit tokens
    # participate in the OOF gate estimate, matching the frozen IU fit cap.
    response_means = np.vstack([
        np.mean(values[offsets[i]:offsets[i + 1]][:, keep], axis=0)
        for i in range(len(rows))
    ])
    response_means = (response_means - mean[None, :]) / std[None, :]
    risk_std = max(float(response_risk.std()), EPS)
    response_risk_z = (response_risk - float(response_risk.mean())) / risk_std
    owners = _token_owners(offsets, fit_indices)
    row_folds = _folds(rows)
    oof_prediction = np.zeros_like(standardized_fit)
    coefficient = np.zeros((standardized_fit.shape[1], 2), dtype=np.float64)

    for feature in range(standardized_fit.shape[1]):
        predictors = np.column_stack([
            response_means[owners, feature],
            response_risk_z[owners],
        ])
        for fold in range(N_FOLDS):
            donor = row_folds[owners] != fold
            held = ~donor
            if not donor.any() or not held.any():
                raise RuntimeError("cross-scale fold is empty")
            beta = _ridge_coefficients(predictors[donor], standardized_fit[donor, feature])
            oof_prediction[held, feature] = predictors[held] @ beta
        coefficient[feature] = _ridge_coefficients(predictors, standardized_fit[:, feature])

    oof_residual = standardized_fit - oof_prediction
    reliability = 1.0 - np.mean(np.square(oof_residual), axis=0) / np.maximum(
        np.var(standardized_fit, axis=0), EPS
    )
    reliability = np.clip(reliability, 0.0, 1.0)
    gate = float(max_gate) * reliability
    innovation_scale = np.maximum(oof_residual.std(axis=0), EPS)
    transformed_fit = (
        (1.0 - gate[None, :]) * standardized_fit
        + gate[None, :] * (oof_residual / innovation_scale[None, :])
    )

    if fusion == "iu":
        fitted = upcr_fit(
            transformed_fit.T,
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
        fusion_diagnostics = {
            "fusion": "iu_pcr",
            "g2_hat": float(fitted.g2_hat),
            "projection_residual": float(fitted.proj_residual),
        }
    else:
        from .dependency_fusion import sparse_upcr_fit

        fitted = sparse_upcr_fit(
            transformed_fit.T,
            scale_ratio=0.25,
            rank=2,
            n_components=2,
            g2_projection_components=1,
            g2_grid=300,
            threshold_multiplier=1.0,
            max_iter=100,
            inner_completion_iter=40,
            decomposition_tol=1e-8,
            max_sparse_fraction=None,
            target_condition=100.0,
        )
        weights = np.asarray(fitted.w_pcr, dtype=np.float64)
        fusion_diagnostics = {
            "fusion": "su_pcr_reproduction",
            "g2_hat": float(fitted.g2_hat),
            "projection_residual": float(fitted.projection_residual),
            "decomposition_converged": bool(fitted.decomposition.converged),
            "decomposition_iterations": int(fitted.decomposition.n_iter),
            "sparse_fraction": float(fitted.decomposition.sparse_fraction),
            "decomposition_relative_residual": float(
                fitted.decomposition.relative_residual
            ),
        }
    anchor = transformed_fit.mean(axis=1)
    raw_fit_score = transformed_fit @ weights
    correlation = float(np.corrcoef(raw_fit_score, anchor)[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    if flipped:
        weights = -weights

    # Apply in response-sized chunks, avoiding a second tokens-by-29 matrix for
    # the large PRMBench cell.
    token_risk = np.empty(len(values), dtype=np.float64)
    for row_index in range(len(rows)):
        lo, hi = map(int, offsets[row_index:row_index + 2])
        selected = values[lo:hi, keep]
        standardized = (selected - mean[None, :]) / std[None, :]
        predictors = np.column_stack([
            response_means[row_index],
            np.full(standardized.shape[1], response_risk_z[row_index]),
        ])
        prediction = np.sum(coefficient * predictors, axis=1)
        innovation = (standardized - prediction[None, :]) / innovation_scale[None, :]
        transformed = (1.0 - gate[None, :]) * standardized + gate[None, :] * innovation
        token_risk[lo:hi] = -(transformed @ weights)

    step_risk = np.asarray([
        float(np.max(token_risk[int(lo):int(hi)])) for lo, hi in zip(starts, ends)
    ], dtype=np.float64)
    if not np.isfinite(token_risk).all() or not np.isfinite(step_risk).all():
        raise RuntimeError("cross-scale token head produced non-finite scores")
    return CrossScaleTokenResult(
        token_risk=token_risk,
        step_risk=step_risk,
        reliability=reliability,
        gate=gate,
        diagnostics={
            "schema_version": "ciw-cross-scale-token-head-v1",
            "n_tokens": int(len(values)),
            "n_fit_tokens": int(len(fit_indices)),
            "n_responses": int(len(rows)),
            "n_steps": int(len(starts)),
            "n_input_streams": int(values.shape[1]),
            "n_kept_streams": int(keep.sum()),
            "n_folds": N_FOLDS,
            "ridge": RIDGE,
            "max_gate": float(max_gate),
            "mean_oof_r2": float(np.mean(reliability)),
            "median_oof_r2": float(np.median(reliability)),
            "mean_gate": float(np.mean(gate)),
            "max_observed_gate": float(np.max(gate)),
            "confidence_anchor_correlation": correlation,
            "orientation_flipped": flipped,
            "labels_seen_during_fit": False,
            **fusion_diagnostics,
        },
    )


__all__ = [
    "CrossScaleTokenResult", "MAX_GATE", "N_FOLDS", "RIDGE",
    "fit_cross_scale_token_head",
]
