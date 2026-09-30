"""Label-free, within-answer representation experiment; no target API.

The leading-loading global sign has an explicit negative entropy anchor.
This is answer-fitted, not anchor-free. The legacy lane borrows calibrated
feature signs for continuity. Current kernels are loaded via the audited
capsule by the experiment runner.
"""
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture

from .adapted_dufs import adapted_dufs_soft_gates
from .joint_lsml import (covariance_matrix, discover_loao_consensus_groups,
                         fit_joint_lsml, raw_orientation_cell, regularized_joint_map_weights)
from .laplacian_upcr import IU_FIT_DEFAULTS, build_graph_from_features, graph_diagnostics, permute_graph
from .short_cycle_localization import fit_windows, scaled_oriented_weight
from .upcr import upcr_fit
from .window_localization import WindowPlan, build_window_matrix, make_window_plan, windows_to_tokens

STREAM_NAMES = (
    "trace_length_series", "entropy_series", "entropy_rolling_spectral_entropy",
    "entropy_rolling_low_band_power", "entropy_rolling_high_band_power",
    "entropy_rolling_hl_ratio", "entropy_rolling_dominant_freq",
    "entropy_rolling_spectral_centroid", "entropy_stft_high_series",
    "entropy_stft_frame_entropy", "entropy_rolling_tail_ratio",
    "entropy_sw_var_series", "entropy_pe_series", "entropy_rolling_rs_hurst",
    "entropy_cusum_abs_series", "spilled_series", "spilled_sw_var_series",
    "spilled_cusum_abs_series", "spilled_rolling_min", "energy_series",
    "energy_rolling_min", "energy_sw_var_series", "energy_cusum_abs_series",
    "top1_logprob_series", "logprob_margin_series", "topk_entropy_series",
    "topk_varentropy_series", "topk_renyi2_series", "topk_tail_mass_series",
)
PRIMITIVES = ("entropy_series", "spilled_series", "energy_series",
              "top1_logprob_series", "logprob_margin_series", "topk_entropy_series",
              "topk_varentropy_series", "topk_renyi2_series", "topk_tail_mass_series")
REPS = ("legacy_fixed32", "global30_local32", "moments27_local32", "moments27_local8")
BASE_METHODS = ("equal", "iu", "joint_lambda0")
GRAPH_METHODS = ("joint_graph010", "joint_graph_permuted")
ARM_IDS = tuple(f"{rep}__{method}" for rep in REPS
                for method in BASE_METHODS + (() if rep == "legacy_fixed32" else GRAPH_METHODS)) + ("entropy_mean_w8",)
MIN_WINDOWS = 8
JOINT_SEED = 2026090601


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def moment_plan(tokens, width):
    if width < 2 or tokens < width:
        raise ValueError("INSUFFICIENT_WINDOW_SUPPORT")
    full = np.arange(0, tokens - width + 1, width, dtype=np.int64)
    starts = np.unique(np.append(full, tokens - width))
    return WindowPlan(tokens, width, width, starts, starts + width, np.searchsorted(starts, full))


def moment_matrix(raw, plan):
    """Same three measurements on all nine primitive telemetry streams."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (plan.token_count, len(STREAM_NAMES)):
        raise ValueError("RAW_SCHEMA_MISMATCH")
    chosen = raw[:, [STREAM_NAMES.index(s) for s in PRIMITIVES]]
    coordinates = np.linspace(-.5, .5, plan.width)
    denominator = coordinates @ coordinates
    values = []
    for lo, hi in zip(plan.starts, plan.ends):
        chunk = chosen[lo:hi]
        mean, sd = chunk.mean(axis=0), chunk.std(axis=0)
        slope = coordinates @ (chunk - mean) / denominator
        values.append(np.column_stack((mean, sd, slope)).ravel())
    names = [f"{s}__{op}" for s in PRIMITIVES for op in ("level", "sd", "slope")]
    return np.asarray(values), names


def prepare_local(values, names, fit_indices):
    fit = values[fit_indices]
    active = np.isfinite(fit).all(axis=0)
    for j in np.flatnonzero(active):
        active[j] = np.ptp(fit[:, j]) > 1e-10 * max(1., float(np.max(np.abs(fit[:, j]))))
    indices = np.flatnonzero(active).tolist()
    if len(indices) < 3:
        raise ValueError("INSUFFICIENT_FEATURE_VARIATION")
    standardized = (fit[:, indices] - fit[:, indices].mean(axis=0)) / fit[:, indices].std(axis=0)
    standardized -= standardized.mean(axis=0)
    keep = []
    duplicates = {}
    for j in range(len(indices)):
        duplicate = next((k for k in keep if abs(float(np.corrcoef(standardized[:, j], standardized[:, k])[0, 1])) >= 1. - 1e-10), None)
        if duplicate is None:
            keep.append(j)
        else:
            duplicates[names[indices[j]]] = names[indices[duplicate]]
    indices = [indices[j] for j in keep]
    active_names = [names[j] for j in indices]
    entropy_name = "epr" if "epr" in active_names else "entropy_series__level"
    if len(indices) < 3 or entropy_name not in active_names:
        raise ValueError("ENTROPY_ANCHOR_OR_FEATURES_UNAVAILABLE")
    mean, sd = fit[:, indices].mean(axis=0), fit[:, indices].std(axis=0)
    z = (values[:, indices] - mean) / sd
    z -= z[fit_indices].mean(axis=0)
    anchor = active_names.index(entropy_name)
    orientation = raw_orientation_cell(z[fit_indices], entropy_index=anchor, tau=0.)
    signs = orientation["signs"]
    z *= signs
    singular = np.linalg.svd(z[fit_indices], compute_uv=False)
    eigen = singular ** 2
    summary = {"active_features": active_names, "active_p": len(indices),
               "n_fit_windows": len(fit), "exact_affine_duplicates_removed": duplicates,
               "feature_signs": signs, "mean": mean, "sd": sd,
               "orientation": "within_answer_offdiag_loading_negative_entropy_anchor",
               "borrowed_fitted_quantities": False,
               "rank": min(len(fit) - 1, int(np.linalg.matrix_rank(z[fit_indices]))),
               "participation_rank": float(eigen.sum() ** 2 / (eigen @ eigen)),
               "anchor_feature": entropy_name}
    return z, anchor, summary


def fit_local(values, names, plan, identity):
    z, anchor, shared = prepare_local(values, names, plan.fit_indices)
    fit = z[plan.fit_indices]
    scores, metadata = {}, {}

    def admit(name, weights, detail, valid=True):
        weights, boundary = scaled_oriented_weight(weights, fit, anchor)
        risk = -(z @ weights)
        if not np.isfinite(risk).all():
            raise ValueError("NONFINITE_RISK")
        scores[name] = risk
        metadata[name] = {**detail, **boundary, "valid": valid,
                          "status": "OK" if valid else "FIT_DIAGNOSTIC_ONLY",
                          "standardized_weights": weights}

    admit("equal", np.ones(fit.shape[1]) / fit.shape[1], {})
    try:
        iu = upcr_fit(fit.T, **dict(IU_FIT_DEFAULTS))
        admit("iu", iu.w, {"g2_hat": iu.g2_hat, "abstained": iu.abstained}, valid=not iu.abstained)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        metadata["iu"] = {"status": "FAILED", "valid": False, "reason": str(exc)}
    try:
        blocks = np.minimum(3, np.arange(len(fit)) * 4 // len(fit))
        grouping = discover_loao_consensus_groups(
            fit, blocks, k_range=(3, 4, 6, 8), seed=JOINT_SEED,
            minimum_group_size=3, minimum_held_admissible_fraction=.95,
            use_minimum_ari_tiebreak=True)
        shared["grouping"] = {k: grouping.get(k) for k in ("status", "K", "group_sizes", "median_ari", "candidates")}
        if grouping["status"] != "SELECTED":
            raise ValueError("BLOCKED_NO_ADMISSIBLE_PARTITION")
        joint = fit_joint_lsml(covariance_matrix(fit), grouping["labels"], anchor_index=anchor,
                               seed=JOINT_SEED, starts=5, max_sweeps=5000)
        jac = joint.jacobian_audit
        valid = bool(joint.converged and joint.multistart_audit["status"] == "PASS"
                     and jac["full_global_rank"] and np.isfinite(jac["condition_number"])
                     and jac["condition_number"] <= 1e8)
        shared["joint"] = {"groups": grouping["labels"], "converged": joint.converged,
                           "multistart": joint.multistart_audit, "jacobian": jac,
                           "diagonal": joint.diagonal_audit, "relative_offdiag_misfit": joint.relative_offdiag_misfit}
        w, detail = regularized_joint_map_weights(fit, joint.model_covariance, joint.global_loading,
                                                  mode="liu", lam=0., target_condition=1000.)
        admit("joint_lambda0", w, {"inverse": detail}, valid=valid)
        gates, gate_detail = adapted_dufs_soft_gates(fit.T, seeds=(0, 1, 2), epochs=120)
        graph = build_graph_from_features(fit.T, gates=gates, k=7)
        seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        permutation = np.random.default_rng(seed).permutation(len(fit))
        shared["gates"] = {"values": gates, "diagnostics": gate_detail,
                           "graph": graph_diagnostics(graph), "permutation": permutation, "seed": seed}
        for method, graph_used in (("joint_graph010", graph), ("joint_graph_permuted", permute_graph(graph, permutation))):
            w, detail = regularized_joint_map_weights(fit, joint.model_covariance, joint.global_loading,
                                                      mode="liu", lam=.1, gates=gates, graph=graph_used,
                                                      graph_k=7, target_condition=1000.)
            admit(method, w, {"inverse": detail}, valid=valid)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        for name in ("joint_lambda0", *GRAPH_METHODS):
            if name not in metadata:
                metadata[name] = {"status": "FAILED", "valid": False, "reason": str(exc)}
    return scores, metadata, shared


def mixture_readout(fit_risk, step_risk):
    """Fixed answer-only no-error/first-error baseline, not a calibrated posterior."""
    x = np.asarray(fit_risk, float).reshape(-1, 1)
    models = []
    with warnings.catch_warnings(record=True) as caught:
        for n in (1, 2):
            models.append(GaussianMixture(n_components=n, n_init=3, max_iter=300,
                                         reg_covar=1e-4, random_state=2026090705).fit(x))
    if not all(m.converged_ for m in models):
        raise ValueError("MIXTURE_NOT_CONVERGED")
    bic = [float(m.bic(x)) for m in models]
    split = bic[1] < bic[0]
    threshold = float(models[1].means_.mean()) if split else None
    candidates = np.flatnonzero(np.asarray(step_risk) > threshold) if split else np.array([], dtype=int)
    prediction = int(candidates[0]) if len(candidates) else -1
    return {"prediction": prediction, "bic": bic, "two_components_selected": split,
            "means": models[1].means_.ravel().tolist(), "threshold": threshold,
            "warnings": [str(w.message) for w in caught]}


def score_answer(raw, starts, ends, identity):
    raw = np.asarray(raw, dtype=float)
    starts, ends = np.asarray(starts, int), np.asarray(ends, int)
    if starts.shape != ends.shape or np.any(starts < 0) or np.any(ends > len(raw)) or np.any(ends <= starts):
        raise ValueError("INVALID_OFFICIAL_SPANS")
    arrays = {"step_starts": starts, "step_ends": ends}
    report = {"representations": {}, "methods": {}, "labels_accessed": False}
    global_matrix = None

    def save_arm(arm, risk, plan, detail):
        tokens = windows_to_tokens(plan, risk)
        steps = np.asarray([np.max(tokens[a:b]) for a, b in zip(starts, ends)])
        arrays[f"{arm}__window"] = risk
        arrays[f"{arm}__step"] = steps
        try:
            readout = mixture_readout(risk[plan.fit_indices], steps)
            detail["readout"] = readout
            detail["readout_valid"] = True
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            detail["readout_valid"] = False
            detail["readout_error"] = str(exc)
        report["methods"][arm] = detail

    for rep in REPS:
        started = time.monotonic()
        try:
            width = 8 if rep.endswith("local8") else 32
            if len(raw) // width < MIN_WINDOWS:
                raise ValueError("TOO_FEW_FIT_WINDOWS")
            if rep in ("legacy_fixed32", "global30_local32"):
                if global_matrix is None:
                    global_matrix = build_window_matrix(raw, STREAM_NAMES, make_window_plan(len(raw), 32, 32))
                plan, values, names = global_matrix.plan, global_matrix.values, list(global_matrix.feature_names)
            else:
                plan = moment_plan(len(raw), width)
                values, names = moment_matrix(raw, plan)
            arrays[f"{rep}__starts"], arrays[f"{rep}__ends"] = plan.starts, plan.ends
            arrays[f"{rep}__fit_indices"] = plan.fit_indices
            arrays[f"{rep}__features"] = values
            if rep == "legacy_fixed32":
                scores, meta, shared = fit_windows(values, names, plan.fit_indices)
                if "joint_modelinv_lam0" in scores:
                    scores["joint_lambda0"] = scores.pop("joint_modelinv_lam0")
                if "joint_modelinv_lam0" in meta:
                    meta["joint_lambda0"] = meta.pop("joint_modelinv_lam0")
                for detail in meta.values():
                    detail["valid"] = detail.get("status") == "OK"
            else:
                scores, meta, shared = fit_local(values, names, plan, identity + "/" + rep)
            for method, detail in meta.items():
                arm = f"{rep}__{method}"
                if method in scores:
                    save_arm(arm, scores[method], plan, detail)
                else:
                    report["methods"][arm] = detail
            report["representations"][rep] = {"status": "SCORED", "shared": shared,
                                              "seconds": time.monotonic() - started, "width": width,
                                              "n_fit_windows": len(plan.fit_indices)}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            report["representations"][rep] = {"status": "UNAVAILABLE", "reason": str(exc),
                                              "seconds": time.monotonic() - started}
            for arm in ARM_IDS:
                if arm.startswith(rep + "__"):
                    report["methods"].setdefault(arm, {"status": "UNAVAILABLE", "valid": False, "reason": str(exc)})
    plan = moment_plan(len(raw), 8)
    entropy = raw[:, STREAM_NAMES.index("entropy_series")]
    risk = np.array([np.mean(entropy[a:b]) for a, b in zip(plan.starts, plan.ends)])
    fit = risk[plan.fit_indices]
    if len(fit) >= MIN_WINDOWS and np.isfinite(risk).all() and np.std(fit) > 1e-12:
        risk = (risk - np.mean(fit)) / np.std(fit)
        save_arm("entropy_mean_w8", risk, plan, {"valid": True, "status": "OK"})
    else:
        report["methods"]["entropy_mean_w8"] = {"valid": False, "status": "UNAVAILABLE"}
    return arrays, json_safe(report)
