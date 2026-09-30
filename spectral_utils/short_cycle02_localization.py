"""Answer-only fixed-provenance-group diagnostic for localization cycle 2.

This module has no target/label API. It consumes one answer's frozen window
matrix and fits every parameter from that answer alone.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .feature_contract import confidence_sign_vector
from .fusion_utils import lsml_continuous
from .joint_lsml import (
    continuous_lsml_weight_vector,
    covariance_matrix,
    fit_joint_lsml,
    regularized_joint_map_weights,
)
from .short_cycle_localization import scaled_oriented_weight
from .window_localization import windows_to_tokens


METHODS = ("joint_fixed4_modelinv_lam0", "fixed4_continuous_lsml")
SETTINGS = {
    "groups": "fixed_feature_provenance_4",
    "joint_starts": 5,
    "joint_max_sweeps": 5000,
    "joint_seed": 2026090701,
    "target_condition": 1000.0,
    "stability_refits": ["omit_first_quarter", "omit_last_quarter"],
    "labels_accessed": False,
}

ENTROPY_FEATURES = frozenset({
    "epr", "trace_length", "spectral_entropy", "low_band_power",
    "high_band_power", "hl_ratio", "dominant_freq", "spectral_centroid",
    "stft_max_high_power", "stft_spectral_entropy", "rpdi", "sw_var_peak",
    "pe_mean", "hurst_exponent", "cusum_max", "cusum_shift_idx",
})
SPILLED_FEATURES = frozenset({
    "epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled", "min_spilled",
})
ENERGY_FEATURES = frozenset({
    "epr_energy", "min_energy", "sw_var_peak_energy", "cusum_max_energy",
})
DISTRIBUTION_FEATURES = frozenset({
    "mean_top1_logprob", "logprob_margin", "mean_logprob_entropy",
    "varentropy", "renyi_entropy_2", "topk_tail_mass",
})
GROUPS = (
    ("entropy_trace", ENTROPY_FEATURES),
    ("sampled_token_spilled", SPILLED_FEATURES),
    ("energy_series", ENERGY_FEATURES),
    ("next_token_distribution", DISTRIBUTION_FEATURES),
)


def fixed_provenance_labels(feature_names):
    """Return four fixed source labels for an active feature roster."""
    names = tuple(map(str, feature_names))
    labels = []
    group_names = []
    for name in names:
        matches = [index for index, (_, members) in enumerate(GROUPS) if name in members]
        if len(matches) != 1:
            raise ValueError(f"feature has no unique frozen provenance group: {name}")
        labels.append(matches[0])
    labels = np.asarray(labels, dtype=np.int64)
    present = tuple(int(value) for value in np.unique(labels))
    sizes = {GROUPS[value][0]: int(np.sum(labels == value)) for value in present}
    if present != (0, 1, 2, 3) or min(sizes.values()) < 3:
        raise RuntimeError(f"FIXED_GROUP_CONTRACT_UNAVAILABLE: sizes={sizes}")
    group_names = [GROUPS[value][0] for value in labels]
    return labels, group_names, sizes


def fit_fixed4_windows(values, feature_names, fit_indices):
    """Fit fixed-group Joint and continuous L-SML on one answer only."""
    values = np.asarray(values, dtype=np.float64)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    if values.ndim != 2 or fit_indices.ndim != 1 or len(fit_indices) < 2:
        raise ValueError("malformed window matrix or fit indices")
    fit_raw = values[fit_indices]
    active = np.isfinite(fit_raw).all(axis=0)
    for index in np.flatnonzero(active):
        scale = max(1.0, float(np.max(np.abs(fit_raw[:, index]))))
        active[index] = np.ptp(fit_raw[:, index]) > 1e-10 * scale
    names = [str(name) for name, keep in zip(feature_names, active) if keep]
    if len(names) < 12 or "epr" not in names:
        return {}, {method: {"status": "INSUFFICIENT_FEATURES"} for method in METHODS}, {}

    x = fit_raw[:, active]
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    z = (values[:, active] - mean) / sd
    z -= z[fit_indices].mean(axis=0)
    signs = confidence_sign_vector(names)
    z *= signs
    fit = z[fit_indices]
    anchor = names.index("epr")

    try:
        labels, group_names, group_sizes = fixed_provenance_labels(names)
    except (ValueError, RuntimeError) as error:
        return {}, {method: {"status": "FIXED_GROUP_CONTRACT_UNAVAILABLE", "reason": str(error)} for method in METHODS}, {
            "active_features": names,
        }

    scores = {}
    metadata = {}
    shared = {
        "active_features": names,
        "active_p": len(names),
        "fit_windows": len(fit),
        "feature_signs": signs.tolist(),
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "fixed_group_labels": labels.tolist(),
        "fixed_group_names": group_names,
        "fixed_group_sizes": group_sizes,
        "fit_scope": "one_answer",
        "labels_accessed": False,
    }

    def admit(method, weight, details):
        oriented, boundary = scaled_oriented_weight(weight, fit, anchor)
        risk = -(z @ oriented)
        if not np.isfinite(risk).all():
            raise RuntimeError("NONFINITE_REPLAY")
        raw_weight = np.zeros(values.shape[1], dtype=np.float64)
        raw_weight[active] = -oriented * signs / sd
        scores[method] = risk
        metadata[method] = {
            "status": "OK",
            **details,
            **boundary,
            "weights_raw_coordinates": raw_weight.tolist(),
        }

    try:
        fitted = fit_joint_lsml(
            covariance_matrix(fit), labels, anchor_index=anchor,
            seed=SETTINGS["joint_seed"], starts=SETTINGS["joint_starts"],
            max_sweeps=SETTINGS["joint_max_sweeps"],
        )
        weight, inverse = regularized_joint_map_weights(
            fit, fitted.model_covariance, fitted.global_loading,
            mode="liu", lam=0.0, target_condition=SETTINGS["target_condition"],
        )
        admit("joint_fixed4_modelinv_lam0", weight, {
            "groups": labels.tolist(),
            "group_sizes": group_sizes,
            "converged": bool(fitted.converged),
            "converged_starts": int(fitted.converged_starts),
            "multistart_status": fitted.multistart_audit["status"],
            "relative_offdiag_misfit": float(fitted.relative_offdiag_misfit),
            "inverse": inverse,
        })
        if not fitted.converged:
            metadata["joint_fixed4_modelinv_lam0"]["status"] = "FINITE_UNCONVERGED_DESCRIPTIVE"
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        metadata["joint_fixed4_modelinv_lam0"] = {
            "status": "FAILED", "reason": f"{type(error).__name__}: {error}",
        }

    try:
        fused, meta = lsml_continuous(
            *[fit[:, index] for index in range(fit.shape[1])],
            groups=labels, compute_score_matrix=False, small_m_guard=True,
        )
        weight = continuous_lsml_weight_vector(meta, fit.shape[1])
        if not np.allclose(fused, fit @ weight, atol=1e-10, rtol=1e-10):
            raise RuntimeError("continuous L-SML weight reconstruction drift")
        admit("fixed4_continuous_lsml", weight, {
            "groups": labels.tolist(),
            "group_sizes": group_sizes,
            "K": int(meta["K"]),
            "residual": float(meta["residual"]),
            "small_m_flags": [list(value) for value in meta["small_m_flags"]],
            "small_m_guarded": [list(value) for value in meta["small_m_guarded"]],
        })
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        metadata["fixed4_continuous_lsml"] = {
            "status": "FAILED", "reason": f"{type(error).__name__}: {error}",
        }
    return scores, metadata, shared


def score_fixed4_answer(values, feature_names, plan, step_starts, step_ends):
    """Fit both arms and map their window scores to the frozen official steps."""
    values = np.asarray(values, dtype=np.float64)
    starts = np.asarray(step_starts, dtype=np.int64)
    ends = np.asarray(step_ends, dtype=np.int64)
    if values.shape[0] != len(plan.starts):
        raise ValueError("window matrix does not match the frozen plan")
    if np.any(starts < 0) or np.any(ends > plan.token_count) or np.any(ends <= starts):
        raise ValueError("invalid official step span")

    window_scores, metadata, shared = fit_fixed4_windows(
        values, feature_names, plan.fit_indices,
    )
    arrays = {
        "window_starts": np.asarray(plan.starts),
        "window_ends": np.asarray(plan.ends),
        "step_starts": starts,
        "step_ends": ends,
    }
    for method, risk in window_scores.items():
        token_risk = windows_to_tokens(plan, risk)
        arrays[method + "__window"] = risk
        arrays[method + "__token"] = token_risk
        arrays[method + "__step"] = np.asarray([
            token_risk[lo:hi].max() for lo, hi in zip(starts, ends)
        ])

    count = len(plan.fit_indices)
    cuts = (
        plan.fit_indices[count // 4:],
        plan.fit_indices[:count - count // 4],
    )
    stability = {}
    for label, indices in zip(SETTINGS["stability_refits"], cuts):
        alternate_scores, alternate_meta, _ = fit_fixed4_windows(
            values, feature_names, indices,
        )
        stability[label] = {}
        for method in METHODS:
            entry = {"status": alternate_meta.get(method, {}).get("status", "MISSING")}
            if method in alternate_scores and method in window_scores:
                entry["score_spearman"] = float(spearmanr(
                    window_scores[method], alternate_scores[method],
                ).statistic)
                left = np.asarray(metadata[method]["weights_raw_coordinates"])
                right = np.asarray(alternate_meta[method]["weights_raw_coordinates"])
                denominator = np.linalg.norm(left) * np.linalg.norm(right)
                entry["oriented_weight_cosine"] = float(left @ right / denominator)
            stability[label][method] = entry
    return arrays, {
        "methods": metadata,
        "shared": shared,
        "stability": stability,
        "labels_accessed": False,
    }


__all__ = [
    "METHODS", "SETTINGS", "fixed_provenance_labels", "fit_fixed4_windows",
    "score_fixed4_answer",
]
