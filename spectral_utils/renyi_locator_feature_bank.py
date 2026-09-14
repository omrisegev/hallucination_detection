"""Factorial Renyi/H1/Hinf locator banks with static label-free fusion."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .direct_probability_fusion import step_top_mean, zscore_columns, _orient
from .direct_probability_temporal import lw_alpha_memory_bounded
from .laplacian_upcr import IU_FIT_DEFAULTS
from .renyi_view_fusion import EPS, head_distribution
from .renyi_alpha_sweep import escort_varentropy
from .upcr import upcr_fit_covariance


ALL_FEATURES = ("H0lim_q15", "VE0_q15", "VE0.75_q15", "VE1_q15", "VE1_q50", "H1_native", "Hinf_q15")
SOLVERS = ("raw_step_equal", "scale_step_equal", "answer_z_local_iu")


@dataclass(frozen=True)
class BankSpec:
    name: str
    ve1_support: int
    add_h1: bool
    add_hinf: bool
    indices: tuple[int, ...]


def bank_specs() -> tuple[BankSpec, ...]:
    output = []
    for support in (15, 50):
        for add_h1 in (False, True):
            for add_hinf in (False, True):
                indices = [0, 1, 2, 3 if support == 15 else 4]
                if add_h1:
                    indices.append(5)
                if add_hinf:
                    indices.append(6)
                name = f"ve1q{support}__h1{int(add_h1)}__hinf{int(add_hinf)}"
                output.append(BankSpec(name, support, add_h1, add_hinf, tuple(indices)))
    return tuple(output)


BANKS = bank_specs()
BANK_BY_NAME = {bank.name: bank for bank in BANKS}
METHODS = tuple(f"{bank.name}__{solver}" for bank in BANKS for solver in SOLVERS)
BASELINE = "ve1q15__h10__hinf0__raw_step_equal"


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(left, right)[0, 1])


def feature_matrix(logprobs: np.ndarray, entropy: np.ndarray) -> dict:
    logprobs = np.asarray(logprobs, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    if logprobs.ndim != 2 or logprobs.shape[1] < 50 or len(logprobs) < 3:
        raise ValueError("need at least three tokens and 50 saved ranks")
    if entropy.shape != (len(logprobs),) or not np.isfinite(logprobs).all() or not np.isfinite(entropy).all():
        raise ValueError("invalid aligned log-probability/entropy input")
    q15, _ = head_distribution(logprobs, k=15)
    q50, _ = head_distribution(logprobs, k=50)
    anchor = escort_varentropy(q15, 1.0)
    raw = np.column_stack((
        np.log(q15 + EPS).mean(axis=1),
        escort_varentropy(q15, 0.0),
        escort_varentropy(q15, 0.75),
        anchor,
        escort_varentropy(q50, 1.0),
        entropy,
        -np.log(np.maximum(q15[:, 0], EPS)),
    ))
    correlations = np.asarray([_corr(raw[:, index], anchor) for index in range(5)])
    signs = np.ones(len(ALL_FEATURES), dtype=np.float64)
    signs[:5] = np.where(np.isfinite(correlations) & (correlations < 0), -1.0, 1.0)
    oriented = raw * signs
    if not np.isfinite(oriented).all():
        raise ValueError("nonfinite oriented feature bank")
    return {"matrix": oriented, "anchor": anchor, "signs": signs, "correlations": correlations}


def _local_iu_score(matrix: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, dict]:
    z, keep, mean, scale = zscore_columns(matrix)
    if len(z) < 3 or z.shape[1] < 3:
        raise ValueError("local IU needs at least three varying views")
    covariance = z.T @ z / len(z)
    diagonal = np.diag(np.diag(covariance))
    beta = float(lw_alpha_memory_bounded(z, covariance, diagonal))
    stabilized = (1.0 - beta) * covariance + beta * diagonal
    fit = upcr_fit_covariance(stabilized, **dict(IU_FIT_DEFAULTS))
    if fit.abstained or fit.used_simple_average:
        raise ValueError("local IU abstained or fell back")
    weight = np.asarray(fit.w, dtype=np.float64)
    score, flipped, correlation = _orient(z @ weight, anchor)
    return score, {
        "active": int(keep.sum()), "beta": beta, "condition": float(np.linalg.cond(stabilized)),
        "orientation_flipped": bool(flipped), "anchor_correlation": correlation,
        "mean": mean, "scale": scale,
    }


def score_bank(all_features: dict, spans: np.ndarray, bank: BankSpec, solver: str) -> tuple[np.ndarray, dict]:
    if solver not in SOLVERS:
        raise ValueError("unknown solver " + solver)
    matrix = np.asarray(all_features["matrix"][:, bank.indices], dtype=np.float64)
    spans = np.asarray(spans, dtype=np.int64)
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError("invalid step spans")
    if solver == "raw_step_equal":
        per_view = np.column_stack([step_top_mean(matrix[:, j], spans[:, 0], spans[:, 1], 10) for j in range(matrix.shape[1])])
        score = per_view.mean(axis=1)
        info = {"kind": "per-view Top10 then natural-unit equal", "feature_scale": matrix.std(axis=0)}
    elif solver == "scale_step_equal":
        scale = matrix.std(axis=0)
        if np.any(scale <= 1e-10):
            raise ValueError("scale-only bank contains a constant view")
        scaled = matrix / scale
        per_view = np.column_stack([step_top_mean(scaled[:, j], spans[:, 0], spans[:, 1], 10) for j in range(scaled.shape[1])])
        score = per_view.mean(axis=1)
        info = {"kind": "within-answer scale-only per-view Top10 equal", "feature_scale": scale}
    else:
        token, info = _local_iu_score(matrix, np.asarray(all_features["anchor"], dtype=np.float64))
        score = step_top_mean(token, spans[:, 0], spans[:, 1], 10)
        info["kind"] = "answer-z static local IU then Top10"
    if score.shape != (len(spans),) or not np.isfinite(score).all():
        raise ValueError("invalid fused step score")
    info.update(bank=bank.name, solver=solver, features=[ALL_FEATURES[index] for index in bank.indices])
    return score, info


__all__ = ["ALL_FEATURES", "BANKS", "BANK_BY_NAME", "BASELINE", "BankSpec", "METHODS", "SOLVERS", "bank_specs", "feature_matrix", "score_bank"]
