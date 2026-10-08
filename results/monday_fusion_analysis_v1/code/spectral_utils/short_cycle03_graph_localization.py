"""Exact answer-only graph test requested for localization short cycle 3.

The module has no target API. Every gate, graph, covariance and weight is fit
from one answer's frozen window matrix.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .adapted_dufs import adapted_dufs_soft_gates
from .feature_contract import confidence_sign_vector
from .joint_lsml import covariance_matrix, fit_joint_lsml, regularized_joint_map_weights
from .laplacian_upcr import build_graph_from_features, graph_diagnostics, permute_graph
from .short_cycle_localization import scaled_oriented_weight
from .window_localization import windows_to_tokens


METHODS = (
    "joint_modelinv_lam0_replay",
    "internal_joint_liu010_answer_only",
    "permctl_graph_internal_joint_liu010_answer_only",
)
SETTINGS = {
    "lambda": 0.1,
    "graph_k": 7,
    "gate_seeds": [0, 1, 2],
    "gate_epochs": 120,
    "joint_seed": 2026090601,
    "joint_starts": 5,
    "joint_max_sweeps": 5000,
    "target_condition": 1000.0,
    "permutation_seed_base": 2026090703,
    "labels_accessed": False,
}


def _cosine(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else float("nan")


def fit_graph_windows(
    values,
    feature_names,
    fit_indices,
    internal_groups,
    expected_active_features,
    *,
    permutation_seed,
):
    """Fit lambda zero, meaningful graph and relabeled-graph maps."""
    values = np.asarray(values, dtype=np.float64)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    fit_raw = values[fit_indices]
    active = np.isfinite(fit_raw).all(axis=0)
    for index in np.flatnonzero(active):
        scale = max(1.0, float(np.max(np.abs(fit_raw[:, index]))))
        active[index] = np.ptp(fit_raw[:, index]) > 1e-10 * scale
    names = [str(name) for name, keep in zip(feature_names, active) if keep]
    if names != list(map(str, expected_active_features)):
        raise RuntimeError("active feature roster does not reproduce short cycle 1")
    labels = np.asarray(internal_groups, dtype=np.int64)
    if labels.shape != (len(names),):
        raise RuntimeError("frozen internal partition does not match active features")

    x = fit_raw[:, active]
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    z = (values[:, active] - mean) / sd
    z -= z[fit_indices].mean(axis=0)
    signs = confidence_sign_vector(names)
    z *= signs
    fit = z[fit_indices]
    anchor = names.index("epr")

    fitted = fit_joint_lsml(
        covariance_matrix(fit), labels, anchor_index=anchor,
        seed=SETTINGS["joint_seed"], starts=SETTINGS["joint_starts"],
        max_sweeps=SETTINGS["joint_max_sweeps"],
    )
    convergence_status = "OK" if fitted.converged else "FINITE_UNCONVERGED_DESCRIPTIVE"
    scores = {}
    metadata = {}
    standardized_weights = {}

    def admit(method, weight, detail):
        oriented, boundary = scaled_oriented_weight(weight, fit, anchor)
        risk = -(z @ oriented)
        if not np.isfinite(risk).all():
            raise RuntimeError("NONFINITE_REPLAY")
        raw_weight = np.zeros(values.shape[1], dtype=np.float64)
        raw_weight[active] = -oriented * signs / sd
        standardized_weights[method] = oriented
        scores[method] = risk
        metadata[method] = {
            "status": convergence_status,
            **detail,
            **boundary,
            "weights_standardized_coordinates": oriented.tolist(),
            "weights_raw_coordinates": raw_weight.tolist(),
        }

    lambda0_weight, lambda0_inverse = regularized_joint_map_weights(
        fit, fitted.model_covariance, fitted.global_loading,
        mode="liu", lam=0.0, target_condition=SETTINGS["target_condition"],
    )
    admit("joint_modelinv_lam0_replay", lambda0_weight, {
        "lambda": 0.0,
        "inverse": lambda0_inverse,
    })

    gates, gate_diagnostics = adapted_dufs_soft_gates(
        fit.T, seeds=tuple(SETTINGS["gate_seeds"]), epochs=SETTINGS["gate_epochs"],
    )
    graph = build_graph_from_features(
        fit.T, gates=gates, k=SETTINGS["graph_k"],
    )
    graph_weight, graph_inverse = regularized_joint_map_weights(
        fit, fitted.model_covariance, fitted.global_loading,
        mode="liu", lam=SETTINGS["lambda"], gates=gates, graph=graph,
        graph_k=SETTINGS["graph_k"], target_condition=SETTINGS["target_condition"],
    )
    admit("internal_joint_liu010_answer_only", graph_weight, {
        "lambda": SETTINGS["lambda"],
        "inverse": graph_inverse,
        "graph": graph_diagnostics(graph),
    })

    permutation = np.random.default_rng(int(permutation_seed)).permutation(graph.shape[0])
    permuted = permute_graph(graph, permutation)
    permuted_weight, permuted_inverse = regularized_joint_map_weights(
        fit, fitted.model_covariance, fitted.global_loading,
        mode="liu", lam=SETTINGS["lambda"], gates=gates, graph=permuted,
        graph_k=SETTINGS["graph_k"], target_condition=SETTINGS["target_condition"],
    )
    admit("permctl_graph_internal_joint_liu010_answer_only", permuted_weight, {
        "lambda": SETTINGS["lambda"],
        "inverse": permuted_inverse,
        "graph": graph_diagnostics(permuted),
        "permutation_seed": int(permutation_seed),
        "permutation": permutation.tolist(),
    })

    reference = "joint_modelinv_lam0_replay"
    for method in METHODS[1:]:
        metadata[method]["versus_lambda0_score_spearman"] = float(
            spearmanr(scores[method], scores[reference]).statistic
        )
        metadata[method]["versus_lambda0_weight_cosine"] = _cosine(
            standardized_weights[method], standardized_weights[reference],
        )
    metadata["internal_joint_liu010_answer_only"]["versus_permuted_score_spearman"] = float(
        spearmanr(
            scores["internal_joint_liu010_answer_only"],
            scores["permctl_graph_internal_joint_liu010_answer_only"],
        ).statistic
    )
    metadata["internal_joint_liu010_answer_only"]["versus_permuted_weight_cosine"] = _cosine(
        standardized_weights["internal_joint_liu010_answer_only"],
        standardized_weights["permctl_graph_internal_joint_liu010_answer_only"],
    )

    gate_payload = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in gate_diagnostics.items()
    }
    shared = {
        "active_features": names,
        "active_p": len(names),
        "fit_windows": len(fit),
        "groups": labels.tolist(),
        "group_sizes": [int(np.sum(labels == group)) for group in np.unique(labels)],
        "joint_converged": bool(fitted.converged),
        "joint_converged_starts": int(fitted.converged_starts),
        "joint_multistart_status": fitted.multistart_audit["status"],
        "relative_offdiag_misfit": float(fitted.relative_offdiag_misfit),
        "gates": gates.tolist(),
        "gate_diagnostics": gate_payload,
        "fit_scope": "one_answer",
        "labels_accessed": False,
    }
    return scores, metadata, shared


def score_graph_answer(
    values,
    feature_names,
    plan,
    step_starts,
    step_ends,
    internal_groups,
    expected_active_features,
    *,
    permutation_seed,
):
    """Fit graph maps and apply the frozen span-maximum step readout."""
    starts = np.asarray(step_starts, dtype=np.int64)
    ends = np.asarray(step_ends, dtype=np.int64)
    window_scores, metadata, shared = fit_graph_windows(
        values, feature_names, plan.fit_indices, internal_groups,
        expected_active_features, permutation_seed=permutation_seed,
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
    return arrays, {
        "methods": metadata,
        "shared": shared,
        "labels_accessed": False,
    }


__all__ = ["METHODS", "SETTINGS", "fit_graph_windows", "score_graph_answer"]

