"""Strict answer-only window pilot; use the source-bound CPU capsule.

Four consecutive blocks are resampling units within ONE answer. Omri explicitly
permits the existing feature-sign calibration. No other answers, correctness
labels or fitted fusion weights enter this API. Exact kernels use a capsule.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.stats import spearmanr

from .joint_lsml import (
    covariance_matrix, discover_loao_consensus_groups, fit_joint_lsml,
    raw_orientation_cell, regularized_joint_map_weights,
)
from .laplacian_upcr import IU_FIT_DEFAULTS
from .feature_contract import confidence_sign_vector
from .upcr import upcr_fit
from .window_localization import (
    build_window_matrix, make_window_plan, matrix_diagnostics, windows_to_tokens,
)

METHODS = ("joint_modelinv_lam0", "iu", "equal")
SETTINGS = {
    "width": 32, "stride": 32, "blocks": 4, "k_range": [3, 4, 6, 8],
    "minimum_group_size": 3, "minimum_held_admissible_fraction": 0.95,
    "joint_starts": 5, "joint_max_sweeps": 5000,
    "seed": 2026090601, "orientation": "existing_confidence_orientation_v1_user_authorized",
    "degree_filter": False, "target_condition": 1000.0,
    "stability_refits": ["omit_first_quarter", "omit_last_quarter"],
    "mapping": "covering_window_mean_then_official_span_max",
    "pooled_fallback": False,
}


def scaled_oriented_weight(w, fit, anchor):
    """The v2 SD=1 and row-mean/entropy sign boundary, on this answer only."""
    w = np.asarray(w, float).copy()
    score = fit @ w
    sd = float(score.std())
    if not np.isfinite(w).all() or not np.isfinite(sd) or sd < 1e-8:
        raise ValueError("DEGENERATE_SCORE")
    w /= sd
    rowmean = fit.mean(axis=1)
    corr = float(np.corrcoef(score, rowmean)[0, 1]) if rowmean.std() else np.nan
    rule = "rowmean_pearson"
    if not np.isfinite(corr) or abs(corr) < 0.02:
        corr = float(spearmanr(score, fit[:, anchor]).statistic)
        rule = "entropy_spearman_fallback"
    if not np.isfinite(corr):
        raise ValueError("ORIENTATION_UNDETERMINED")
    if corr < 0:
        w *= -1
    return w, {"score_sd_before": sd, "orientation_rule": rule,
               "orientation_correlation": corr, "flipped": bool(corr < 0)}


def fit_windows(values, feature_names, fit_indices):
    """Fit all three recipes, independently, on supplied windows of one answer."""
    values = np.asarray(values, float)
    fit_indices = np.asarray(fit_indices, int)
    fit_raw = values[fit_indices]
    active = np.isfinite(fit_raw).all(axis=0)
    for j in np.flatnonzero(active):
        active[j] = np.ptp(fit_raw[:, j]) > 1e-10 * max(1.0, np.max(np.abs(fit_raw[:, j])))
    names = [str(n) for n, yes in zip(feature_names, active) if yes]
    if len(names) < 3 or "epr" not in names:
        return {}, {m: {"status": "INSUFFICIENT_FEATURES"} for m in METHODS}, {}
    x = fit_raw[:, active]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    z = (values[:, active] - mu) / sd
    z -= z[fit_indices].mean(axis=0)
    anchor = names.index("epr")
    signs = confidence_sign_vector(names)
    z *= signs
    fit = z[fit_indices]
    scores, meta = {}, {}
    shared = {
        "active_features": names, "active_p": len(names), "fit_windows": len(fit),
        "feature_signs": signs.tolist(), "mean": mu.tolist(), "sd": sd.tolist(),
        "borrowed_calibration": "existing_feature_signs_only_explicitly_authorized",
        "rank": int(np.linalg.matrix_rank(fit - fit.mean(axis=0))),
    }

    def admit(method, w, detail):
        w, boundary = scaled_oriented_weight(w, fit, anchor)
        risk = -(z @ w)
        if not np.isfinite(risk).all():
            raise ValueError("NONFINITE_REPLAY")
        scores[method] = risk
        raw_map = np.zeros(values.shape[1])
        raw_map[active] = -w * signs / sd
        meta[method] = {"status": "OK", **detail, **boundary,
                        "weights_raw_coordinates": raw_map.tolist()}

    for method in ("equal", "iu"):
        try:
            if method == "equal":
                admit(method, np.ones(fit.shape[1]) / fit.shape[1], {})
            else:
                model = upcr_fit(fit.T, **dict(IU_FIT_DEFAULTS))
                admit(method, model.w, {"g2_hat": float(model.g2_hat),
                                       "abstained": bool(model.abstained)})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            meta[method] = {"status": "FAILED", "reason": str(exc)}
    try:
        # Adapt the existing consensus engine: owners here are temporal blocks,
        # explicitly not independent answers. Retain its eligibility rules.
        blocks = np.minimum(3, np.arange(len(fit)) * 4 // len(fit))
        grouping = discover_loao_consensus_groups(
            fit, blocks, k_range=tuple(SETTINGS["k_range"]), seed=SETTINGS["seed"],
            minimum_group_size=3, minimum_held_admissible_fraction=0.95,
            use_minimum_ari_tiebreak=True,
        )
        group_summary = {
            "grouping_status": grouping["status"],
            "resampling_unit": "four_contiguous_blocks_within_this_answer",
            "candidates": [{k: row.get(k) for k in
                ("K", "admissible", "group_sizes", "held_admissible_fraction", "median_ari", "rejection_reason")}
                for row in grouping.get("candidates", [])],
        }
        if grouping["status"] != "SELECTED":
            meta["joint_modelinv_lam0"] = {"status": "BLOCKED_NO_ADMISSIBLE_PARTITION", **group_summary}
        else:
            labels = np.asarray(grouping["labels"], int)
            fitted = fit_joint_lsml(covariance_matrix(fit), labels, anchor_index=anchor,
                                   seed=SETTINGS["seed"], starts=5, max_sweeps=5000)
            w, inverse = regularized_joint_map_weights(
                fit, fitted.model_covariance, fitted.global_loading,
                mode="liu", lam=0.0, target_condition=1000.0,
            )
            admit("joint_modelinv_lam0", w, {
                **group_summary, "K": int(grouping["K"]),
                "groups": labels.tolist(), "group_sizes": list(grouping["group_sizes"]),
                "grouping_median_ari": float(grouping["median_ari"]),
                "converged": bool(fitted.converged),
                "converged_starts": int(fitted.converged_starts),
                "multistart_status": fitted.multistart_audit["status"],
                "relative_offdiag_misfit": float(fitted.relative_offdiag_misfit),
                "inverse": inverse,
            })
            if not fitted.converged:
                meta["joint_modelinv_lam0"]["status"] = "FINITE_UNCONVERGED_DESCRIPTIVE"
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        meta["joint_modelinv_lam0"] = {"status": "FAILED", "reason": str(exc)}
    return scores, meta, shared


def fit_answer(raw, stream_names, step_starts, step_ends):
    start = time.monotonic()
    plan = make_window_plan(len(raw), 32, 32)
    matrix = build_window_matrix(raw, stream_names, plan)
    window_scores, metadata, shared = fit_windows(matrix.values, matrix.feature_names, plan.fit_indices)
    starts, ends = np.asarray(step_starts, int), np.asarray(step_ends, int)
    if np.any(starts < 0) or np.any(ends > len(raw)) or np.any(ends <= starts):
        raise ValueError("Invalid official span")
    arrays = {"window_starts": plan.starts, "window_ends": plan.ends,
              "feature_values": matrix.values, "step_starts": starts, "step_ends": ends}
    for method, risk in window_scores.items():
        token_risk = windows_to_tokens(plan, risk)
        arrays[method + "__window"] = risk
        arrays[method + "__token"] = token_risk
        arrays[method + "__step"] = np.asarray([token_risk[a:b].max() for a, b in zip(starts, ends)])
    # Two full refits, including own scaling/signs/group discovery. Agreement is
    # evaluated on the same complete trajectory, not just retained fit blocks.
    n = len(plan.fit_indices)
    cuts = (plan.fit_indices[n // 4:], plan.fit_indices[:n - n // 4])
    stability = {}
    for name, indices in zip(SETTINGS["stability_refits"], cuts):
        alt_scores, alt_meta, _ = fit_windows(matrix.values, matrix.feature_names, indices)
        stability[name] = {}
        for method in METHODS:
            entry = {"status": alt_meta.get(method, {}).get("status", "MISSING")}
            if method in alt_scores and method in window_scores:
                entry["score_spearman"] = float(spearmanr(window_scores[method], alt_scores[method]).statistic)
                a = np.asarray(metadata[method]["weights_raw_coordinates"])
                b = np.asarray(alt_meta[method]["weights_raw_coordinates"])
                entry["oriented_weight_cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            stability[name][method] = entry
    report = {"methods": metadata, "shared": shared, "geometry": matrix_diagnostics(matrix),
              "stability": stability, "labels_accessed": False,
              "elapsed_seconds": time.monotonic() - start}
    return arrays, report
