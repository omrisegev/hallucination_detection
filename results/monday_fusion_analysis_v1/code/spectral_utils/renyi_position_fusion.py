"""Label-free position-varying fusion of frozen Renyi/escort-varentropy views.

The representation bank is fixed to four token streams selected before this
experiment: H0lim, VE0, VE0.75 and VE1.  The position clock spans the complete
answer.  Step boundaries are used only by the frozen Top10 readout.

No function in this module accepts benchmark labels.  External position models
must be fitted by a caller that enforces source-group/fold exclusion.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import eigh

from .answer_position_fusion import centered_moments, position_overlap
from .direct_probability_fusion import step_top_mean
from .direct_probability_temporal import lw_alpha_memory_bounded
from .laplacian_upcr import IU_FIT_DEFAULTS
from .renyi_alpha_sweep import anchor_stream, sweep_matrix
from .upcr import upcr_fit_covariance


BINS = 16
BORROW_ALPHA = 0.25
EPS = 1e-12
FEATURE_NAMES = ("H0lim", "ve0", "ve0.75", "ve1")
SINGLE_METHODS = tuple("view__" + name for name in FEATURE_NAMES)
FUSION_METHODS = (
    "equal",
    "local_iu",
    "external_iu_static",
    "external_iu_position_mean",
    "external_iu_position",
    "external_iu_position_shuffled",
    "local_shrink_pooled",
    "local_shrink_position",
    "local_shrink_position_scale_only",
    "local_shrink_position_shuffled",
)
METHODS = SINGLE_METHODS + FUSION_METHODS
EXTERNAL_METHODS = tuple(name for name in FUSION_METHODS if name not in ("equal", "local_iu"))
ANCHOR_INDEX = FEATURE_NAMES.index("ve1")
PRIMARY = (
    ("external_iu_position", "external_iu_position_mean"),
    ("local_shrink_position", "local_shrink_position_scale_only"),
)
LABELS = {
    "view__H0lim": "Renyi H0 limit (mean log q)",
    "view__ve0": "Escort varentropy alpha 0",
    "view__ve0.75": "Escort varentropy alpha 0.75",
    "view__ve1": "Escort varentropy alpha 1 (varentropy15)",
    "equal": "Four-view oriented equal fusion",
    "local_iu": "Answer-local IU-PCR",
    "external_iu_static": "Other-answer stationary IU-PCR",
    "external_iu_position_mean": "Stationary IU weights plus position mean",
    "external_iu_position": "Other-answer position-varying IU-PCR",
    "external_iu_position_shuffled": "Position IU-PCR with shuffled assignments",
    "local_shrink_pooled": "Answer-local Shrinkage IU plus pooled prior",
    "local_shrink_position": "Answer-local Shrinkage IU plus position prior",
    "local_shrink_position_scale_only": "Position prior with baseline direction only",
    "local_shrink_position_shuffled": "Position prior with shuffled assignments",
}


def _correlation(x, y):
    if len(x) < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(x, y)[0, 1])


def feature_bank(logprobs):
    """Build frozen single streams and an answer-standardized fusion bank.

    Single streams exactly retain Stage-3b orientation: H0lim keeps its natural
    high-is-risk sign, while every VE stream is oriented to the answer's own
    varentropy15 anchor.  Fusion columns are then independently anchor-oriented,
    matching the earlier answer-local fusion convention, and standardized.
    """
    all_views, names = sweep_matrix(logprobs)
    indices = [names.index(name) for name in FEATURE_NAMES]
    raw = np.asarray(all_views[:, indices], dtype=np.float64)
    anchor = np.asarray(anchor_stream(logprobs), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE_NAMES) or len(raw) < 3:
        raise ValueError("four-view bank needs at least three tokens")
    if not np.isfinite(raw).all() or not np.isfinite(anchor).all():
        raise ValueError("nonfinite Renyi/varentropy view")

    single = {}
    single_signs = np.ones(len(FEATURE_NAMES), dtype=np.float64)
    single_corr = np.array([_correlation(raw[:, j], anchor) for j in range(raw.shape[1])])
    for j, name in enumerate(FEATURE_NAMES):
        if name.startswith("ve") and np.isfinite(single_corr[j]) and single_corr[j] < 0:
            single_signs[j] = -1.0
        single["view__" + name] = raw[:, j] * single_signs[j]

    oriented = raw * single_signs
    fusion_corr = np.array([_correlation(oriented[:, j], anchor) for j in range(oriented.shape[1])])
    fusion_signs = np.where(np.isfinite(fusion_corr) & (fusion_corr < 0), -1.0, 1.0)
    oriented = oriented * fusion_signs
    mean = oriented.mean(axis=0)
    scale = oriented.std(axis=0)
    if np.any(scale <= 1e-10):
        bad = [FEATURE_NAMES[j] for j in np.flatnonzero(scale <= 1e-10)]
        raise ValueError("near-constant frozen view(s): " + ", ".join(bad))
    z = (oriented - mean) / scale
    if not np.isfinite(z).all():
        raise ValueError("nonfinite standardized fusion bank")
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-10, rtol=0)
    return dict(
        z=z,
        singles=single,
        anchor=anchor,
        raw=raw,
        mean=mean,
        scale=scale,
        single_signs=single_signs,
        fusion_signs=fusion_signs,
        single_anchor_correlation=single_corr,
        fusion_anchor_correlation=fusion_corr,
    )


def regional_statistics(z, uid):
    """Per-answer whole-position first/second moments in common coordinates."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != len(FEATURE_NAMES) or not len(z):
        raise ValueError("invalid four-view answer")
    if not np.isfinite(z).all():
        raise ValueError("nonfinite four-view answer")
    result = {}
    for key, shuffled in (("real", False), ("shuffle", True)):
        overlap = position_overlap(len(z), uid, shuffled) * BINS / len(z)
        result[key + "_second"] = np.einsum("tj,tk,tl->jkl", overlap, z, z, optimize=True)
        result[key + "_mean"] = overlap.T @ z
    return result


def _fit_iu(second, mean, *, stationary, groups, anchor_index=ANCHOR_INDEX):
    """Canonical IU-PCR in training-region coordinates, returned in z space."""
    second = np.asarray(second, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if second.shape != (BINS, len(FEATURE_NAMES), len(FEATURE_NAMES)):
        raise ValueError("invalid regional second moments")
    if mean.shape != (BINS, len(FEATURE_NAMES)):
        raise ValueError("invalid regional means")
    if not np.isfinite(second).all() or not np.isfinite(mean).all():
        raise ValueError("nonfinite regional moments")
    pooled_second, pooled_mean = second.mean(axis=0), mean.mean(axis=0)
    region_fraction = 0.0 if stationary else float(groups) / (float(groups) + BINS)
    if stationary:
        seconds = pooled_second[None]
        means = pooled_mean[None]
    else:
        seconds = region_fraction * second + (1.0 - region_fraction) * pooled_second
        means = region_fraction * mean + (1.0 - region_fraction) * pooled_mean
    covariances = centered_moments(seconds, means)
    coefficients, intercepts, diagnostics = [], [], []
    for covariance, region_mean in zip(covariances, means):
        scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        active = scale > 1e-10
        if int(active.sum()) < 3:
            raise ValueError("IU needs at least three varying training views")
        correlation = covariance[np.ix_(active, active)] / np.outer(scale[active], scale[active])
        fit = upcr_fit_covariance(correlation, **IU_FIT_DEFAULTS)
        if fit.abstained or fit.used_simple_average:
            raise ValueError("IU abstained or used a fallback")
        weight = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        weight[active] = np.asarray(fit.w, dtype=np.float64) / scale[active]
        anchor_covariance = float(weight @ covariance[:, anchor_index])
        orientation = -1.0 if anchor_covariance < 0 else 1.0
        weight *= orientation
        coefficients.append(weight)
        intercepts.append(-float(region_mean @ weight))
        diagnostics.append(dict(
            active_columns=np.flatnonzero(active).tolist(),
            rho=np.asarray(fit.rho_hat).tolist(),
            g2=float(fit.g2_hat),
            components=int(fit.n_components_used),
            orientation=orientation,
            anchor_covariance=anchor_covariance,
            condition=float(np.linalg.cond(correlation)),
        ))
    coefficients = np.asarray(coefficients)
    intercepts = np.asarray(intercepts)
    if stationary:
        coefficients = np.repeat(coefficients, BINS, axis=0)
        intercepts = np.repeat(intercepts, BINS, axis=0)
    return dict(coefficients=coefficients, intercept=intercepts), dict(
        algorithm="canonical IU-PCR covariance API",
        stationary=bool(stationary),
        source_groups=int(groups),
        region_fraction=region_fraction,
        regions=diagnostics,
    )


def fit_external_model(statistics, group_count):
    """Fit all other-answer models from already exclusion-safe aggregates."""
    required = {"real_second", "real_mean", "shuffle_second", "shuffle_mean"}
    if set(statistics) != required:
        raise ValueError("external statistics roster mismatch")
    if int(group_count) <= 0:
        raise ValueError("positive source-group count required")
    static, static_info = _fit_iu(
        statistics["real_second"], statistics["real_mean"], stationary=True, groups=group_count
    )
    position, position_info = _fit_iu(
        statistics["real_second"], statistics["real_mean"], stationary=False, groups=group_count
    )
    shuffled, shuffled_info = _fit_iu(
        statistics["shuffle_second"], statistics["shuffle_mean"], stationary=False, groups=group_count
    )
    position_mean = {key: value.copy() for key, value in static.items()}
    shrunk_mean_fraction = float(group_count) / (float(group_count) + BINS)
    mean = (
        shrunk_mean_fraction * statistics["real_mean"]
        + (1.0 - shrunk_mean_fraction) * statistics["real_mean"].mean(axis=0)
    )
    position_mean["intercept"] = -np.sum(position_mean["coefficients"] * mean, axis=1)
    covariance = {
        "real": centered_moments(statistics["real_second"], statistics["real_mean"]),
        "shuffle": centered_moments(statistics["shuffle_second"], statistics["shuffle_mean"]),
    }
    for value in covariance.values():
        if not np.isfinite(value).all():
            raise ValueError("nonfinite external covariance prior")
    return dict(
        external_iu_static=static,
        external_iu_position_mean=position_mean,
        external_iu_position=position,
        external_iu_position_shuffled=shuffled,
        covariance=covariance,
    ), dict(
        external_iu_static=static_info,
        external_iu_position_mean=dict(
            algorithm="stationary coefficients with matched position-dependent training mean",
            source_groups=int(group_count),
            region_fraction=shrunk_mean_fraction,
        ),
        external_iu_position=position_info,
        external_iu_position_shuffled=shuffled_info,
    )


def prepare_local(z):
    """Freeze answer-local IU rho and its top-two covariance eigenspace."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != len(FEATURE_NAMES) or len(z) < 3:
        raise ValueError("local IU needs a finite T x 4 bank and at least three tokens")
    if not np.isfinite(z).all():
        raise ValueError("nonfinite local IU bank")
    covariance = z.T @ z / len(z)
    diagonal = np.diag(np.diag(covariance))
    beta = float(lw_alpha_memory_bounded(z, covariance, diagonal))
    base_covariance = (1.0 - beta) * covariance + beta * diagonal
    fit = upcr_fit_covariance(base_covariance, **IU_FIT_DEFAULTS)
    if fit.abstained or fit.used_simple_average:
        raise ValueError("local IU abstained or used a fallback")
    weight = np.asarray(fit.w, dtype=np.float64)
    if weight.shape != (len(FEATURE_NAMES),) or not np.isfinite(weight).all():
        raise ValueError("invalid local IU weights")
    _, eigvec = eigh(base_covariance, subset_by_index=[len(weight) - 2, len(weight) - 1])
    subspace = eigvec[:, ::-1]
    rho = subspace.T @ np.asarray(fit.rho_hat, dtype=np.float64)
    quadratic = subspace.T @ base_covariance @ subspace + EPS * np.eye(2)
    replay = subspace @ np.linalg.solve(quadratic, rho)
    np.testing.assert_allclose(replay, weight, atol=1e-10, rtol=1e-9)
    score = z @ weight
    corr = _correlation(score, z[:, ANCHOR_INDEX])
    sign = -1.0 if np.isfinite(corr) and corr < 0 else 1.0
    weight *= sign
    return dict(
        z=z,
        covariance=base_covariance,
        subspace=subspace,
        rho=rho,
        quadratic=quadratic,
        weight=weight,
        score=z @ weight,
        sign=sign,
        beta=beta,
        anchor_correlation=corr,
    )


def _conditional_weights(local, prior_covariance, *, alpha, shuffled=False, pooled=False, scale_only=False, uid=""):
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha < 1.0:
        raise ValueError("borrow alpha must be finite and in [0, 1)")
    z = local["z"]
    static = local["weight"]
    if alpha == 0.0:
        return np.broadcast_to(static, z.shape), dict(
            alpha=0.0, baseline_exact=True, direction_change_mean=0.0, gain_mean=1.0
        )
    covariance = np.asarray(prior_covariance["shuffle" if shuffled else "real"], dtype=np.float64)
    if covariance.shape != (BINS, len(FEATURE_NAMES), len(FEATURE_NAMES)):
        raise ValueError("invalid position covariance prior")
    projected = np.einsum(
        "pi,jpq,qk->jik", local["subspace"], covariance, local["subspace"], optimize=True
    )
    overlap = position_overlap(len(z), uid, shuffled)
    if pooled:
        target = np.broadcast_to(projected.mean(axis=0), (len(z), 2, 2))
    else:
        target = np.einsum("tj,jab->tab", overlap, projected, optimize=True)
    quadratic = (
        (1.0 - alpha) * (local["quadratic"] - EPS * np.eye(2))
        + alpha * target
        + EPS * np.eye(2)
    )
    quadratic = 0.5 * (quadratic + quadratic.transpose(0, 2, 1))
    if not np.isfinite(quadratic).all() or np.linalg.eigvalsh(quadratic).min() <= 0:
        raise ValueError("conditional quadratic is not positive definite")
    rhs = np.broadcast_to(local["rho"], (len(z), 2))
    theta = np.linalg.solve(quadratic, rhs[..., None])[..., 0]
    weights = local["sign"] * theta @ local["subspace"].T
    gain = weights @ static / float(static @ static)
    orthogonal = weights - gain[:, None] * static
    direction = float(np.linalg.norm(orthogonal, axis=1).mean() / np.linalg.norm(static))
    if scale_only:
        weights = gain[:, None] * static
        direction = 0.0
    return weights, dict(
        alpha=alpha,
        beta=local["beta"],
        pooled=bool(pooled),
        shuffled=bool(shuffled),
        scale_only=bool(scale_only),
        gain_mean=float(gain.mean()),
        gain_min=float(gain.min()),
        gain_max=float(gain.max()),
        direction_change_mean=direction,
        quadratic_condition_max=float(np.linalg.cond(quadratic).max()),
    )


def _external_token_scores(z, arrays, uid, *, shuffled=False):
    coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
    intercept = np.asarray(arrays["intercept"], dtype=np.float64)
    if coefficients.shape != (BINS, len(FEATURE_NAMES)) or intercept.shape != (BINS,):
        raise ValueError("invalid external coefficient map")
    overlap = position_overlap(len(z), uid, shuffled)
    weights = overlap @ coefficients
    bias = overlap @ intercept
    score = np.einsum("tp,tp->t", z, weights) + bias
    if not np.isfinite(score).all():
        raise ValueError("nonfinite external position scores")
    return score, weights


def score_answer(features, spans, external_model, uid, methods=METHODS, *, alpha=BORROW_ALPHA):
    """Score one answer; failures remain explicit and never substitute a baseline."""
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError("unknown methods: " + ", ".join(sorted(unknown)))
    z = np.asarray(features["z"], dtype=np.float64)
    spans = np.asarray(spans, dtype=int)
    if spans.ndim != 2 or spans.shape[1] != 2 or np.min(spans) < 0 or np.max(spans) > len(z):
        raise ValueError("invalid frozen step spans")
    local = None
    local_failure = None
    if any(method == "local_iu" or method.startswith("local_shrink_") for method in methods):
        try:
            local = prepare_local(z)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
            local_failure = f"{type(error).__name__}: {error}"
    scores, health, maps = {}, {}, {}
    for method in methods:
        started = time.perf_counter()
        try:
            if method in SINGLE_METHODS:
                token = np.asarray(features["singles"][method], dtype=np.float64)
                weight_map = np.zeros((BINS, len(FEATURE_NAMES)))
                weight_map[:, SINGLE_METHODS.index(method)] = 1.0
                info = dict(kind="frozen_single_view")
            elif method == "equal":
                weights = np.full_like(z, 1.0 / z.shape[1])
                token = z.mean(axis=1)
                weight_map = np.full((BINS, len(FEATURE_NAMES)), 1.0 / z.shape[1])
                info = dict(kind="answer-standardized oriented equal")
            elif method == "local_iu":
                if local is None:
                    raise ValueError("local IU preparation failed: " + str(local_failure))
                weights = np.broadcast_to(local["weight"], z.shape)
                token = local["score"]
                weight_map = np.broadcast_to(local["weight"], (BINS, z.shape[1])).copy()
                info = dict(kind="answer-local IU", beta=local["beta"], anchor_correlation=local["anchor_correlation"])
            elif method.startswith("external_iu_"):
                arrays = external_model[method]
                shuffled = method.endswith("shuffled")
                token, weights = _external_token_scores(z, arrays, uid, shuffled=shuffled)
                region = position_overlap(len(z), uid, shuffled) * BINS / len(z)
                weight_map = region.T @ weights
                info = dict(kind="other-answer position model", shuffled=shuffled)
            elif method.startswith("local_shrink_"):
                if local is None:
                    raise ValueError("local IU preparation failed: " + str(local_failure))
                pooled = method == "local_shrink_pooled"
                shuffled = method.endswith("shuffled")
                scale_only = method.endswith("scale_only")
                weights, info = _conditional_weights(
                    local,
                    external_model["covariance"],
                    alpha=alpha,
                    pooled=pooled,
                    shuffled=shuffled,
                    scale_only=scale_only,
                    uid=uid,
                )
                token = np.einsum("tp,tp->t", z, weights)
                if info.get("baseline_exact"):
                    weight_map = np.broadcast_to(local["weight"], (BINS, z.shape[1])).copy()
                else:
                    region = position_overlap(len(z), uid, shuffled) * BINS / len(z)
                    weight_map = region.T @ weights
                info["kind"] = "answer-local IU with external covariance prior"
            else:
                raise ValueError("unhandled method " + method)
            if token.shape != (len(z),) or not np.isfinite(token).all():
                raise ValueError("invalid token score vector")
            scores[method] = step_top_mean(token, spans[:, 0], spans[:, 1], 10)
            maps[method] = np.asarray(weight_map, dtype=np.float64)
            health[method] = dict(info, status="OK", seconds=time.perf_counter() - started)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
            scores[method] = np.full(len(spans), np.nan)
            maps[method] = np.full((BINS, len(FEATURE_NAMES)), np.nan)
            health[method] = dict(
                status="FAILED", reason=f"{type(error).__name__}: {error}", seconds=time.perf_counter() - started
            )
    return scores, health, maps


__all__ = [
    "ANCHOR_INDEX",
    "BINS",
    "BORROW_ALPHA",
    "EXTERNAL_METHODS",
    "FEATURE_NAMES",
    "FUSION_METHODS",
    "LABELS",
    "METHODS",
    "PRIMARY",
    "SINGLE_METHODS",
    "feature_bank",
    "fit_external_model",
    "prepare_local",
    "regional_statistics",
    "score_answer",
]
