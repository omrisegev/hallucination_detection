"""Cross-fitted Continuous/Joint L-SML helpers for the Step-396 follow-up.

The estimators in this module never accept correctness labels.  Labels are
consumed only by the orchestration script when recipes are compared or scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from .fusion_utils import lsml_continuous
from .joint_lsml import (
    JointFitResult,
    _factor_model,
    _fit_one_start,
    _objective,
    _profiled_jacobian_audit,
    absolute_cosine,
    continuous_lsml_weight_vector,
    covariance_matrix,
    fit_joint_lsml,
    hierarchical_joint_weights,
    offdiag_relative_misfit,
    residual_affinity,
)


EPS = 1e-12


@dataclass(frozen=True)
class FusionRecipe:
    name: str
    members: tuple[str, ...]
    mode: str = "continuous"
    groups: tuple[int, ...] | None = None
    anchor: int = 0
    joint_members: tuple[int, ...] | None = None
    digit_member: int | None = None


def answer_standardize(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Columnwise z-score inside every answer; constants become zero."""
    x = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(offsets) - 1):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        block = x[sl]
        mean = block.mean(axis=0)
        sd = block.std(axis=0)
        out[sl] = np.divide(block - mean, sd, out=np.zeros_like(block), where=sd > EPS)
    return out


def cell_midranks(values: np.ndarray, cells: Sequence[str], mask: np.ndarray) -> np.ndarray:
    """Tie-aware [0,1] ranks, independently per cell and column."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    cell = np.asarray(cells, dtype=str)
    selected = np.asarray(mask, dtype=bool)
    out = np.full_like(x, np.nan, dtype=np.float64)
    for name in sorted(set(cell[selected].tolist())):
        ix = np.flatnonzero(selected & (cell == name))
        for j in range(x.shape[1]):
            out[ix, j] = (rankdata(x[ix, j], method="average") - 1.0) / max(len(ix) - 1, 1)
    return out


def _orient(values: np.ndarray, weight: np.ndarray, anchor: int) -> tuple[np.ndarray, dict[str, Any]]:
    w = np.asarray(weight, dtype=np.float64).copy()
    score = np.asarray(values, dtype=np.float64) @ w
    rho = float(spearmanr(score, np.asarray(values)[:, int(anchor)]).statistic)
    flipped = bool(np.isfinite(rho) and rho < 0)
    if flipped:
        w *= -1.0
        rho *= -1.0
    scale = float(np.abs(w).sum())
    if scale > EPS:
        w /= scale
    return w, {"anchor_spearman": rho, "anchor_flipped": flipped}


def fit_fusion_weights(values: np.ndarray, recipe: FusionRecipe, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one label-free recipe on already normalized observations."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(recipe.members) or not np.isfinite(x).all():
        raise ValueError("fusion matrix/recipe mismatch")
    if recipe.mode == "equal":
        weight = np.ones(x.shape[1], dtype=np.float64) / x.shape[1]
        return weight, {"mode": "equal"}
    if recipe.mode == "continuous":
        groups = None if recipe.groups is None else np.asarray(recipe.groups, dtype=np.int64)
        _, meta = lsml_continuous(
            *[x[:, j] for j in range(x.shape[1])], groups=groups,
            compute_score_matrix=False, small_m_guard=True,
        )
        raw = continuous_lsml_weight_vector(meta, x.shape[1])
        weight, orient = _orient(x, raw, recipe.anchor)
        return weight, {
            "mode": "continuous", "K": int(meta["K"]),
            "groups": np.asarray(meta["c"], dtype=int).tolist(),
            "residual": float(meta["residual"]),
            "small_m_guarded": [list(v) for v in meta["small_m_guarded"]],
            "small_m_flags": [list(v) for v in meta["small_m_flags"]], **orient,
        }
    if recipe.mode == "joint":
        if recipe.groups is None:
            raise ValueError("joint recipe requires structural groups")
        groups = np.asarray(recipe.groups, dtype=np.int64)
        cov = covariance_matrix(x)
        fit = fit_joint_lsml(cov, groups, anchor_index=recipe.anchor, seed=seed, starts=5)
        _, raw, joint_meta = hierarchical_joint_weights(
            x, groups, fit.global_loading, anchor_index=recipe.anchor, small_m_guard=True,
        )
        weight, orient = _orient(x, raw, recipe.anchor)
        return weight, {
            "mode": "joint", "groups": groups.tolist(),
            "relative_offdiag_misfit": float(fit.relative_offdiag_misfit),
            "converged": bool(fit.converged), "converged_starts": int(fit.converged_starts),
            "multistart": fit.multistart_audit["status"],
            "jacobian": fit.jacobian_audit, "joint_weight_meta": joint_meta, **orient,
        }
    if recipe.mode == "joint_then_continuous":
        if recipe.groups is None or recipe.joint_members is None or recipe.digit_member is None:
            raise ValueError("hierarchical gate recipe is incomplete")
        jm = np.asarray(recipe.joint_members, dtype=np.int64)
        jgroups = np.asarray(recipe.groups, dtype=np.int64)
        joint_x = x[:, jm]
        fit = fit_joint_lsml(
            covariance_matrix(joint_x), jgroups, anchor_index=0, seed=seed, starts=5,
        )
        unique = np.unique(jgroups)
        virtual = np.column_stack([
            joint_x[:, jgroups == group] @ fit.global_loading[jgroups == group]
            for group in unique
        ])
        outer = np.column_stack([virtual, x[:, int(recipe.digit_member)]])
        _, outer_meta = lsml_continuous(
            *[outer[:, j] for j in range(outer.shape[1])],
            compute_score_matrix=False, small_m_guard=True,
        )
        outer_w = continuous_lsml_weight_vector(outer_meta, outer.shape[1])
        raw = np.zeros(x.shape[1], dtype=np.float64)
        for pos, group in enumerate(unique):
            raw[jm[jgroups == group]] = fit.global_loading[jgroups == group] * outer_w[pos]
        raw[int(recipe.digit_member)] = outer_w[-1]
        weight, orient = _orient(x, raw, int(recipe.digit_member))
        return weight, {
            "mode": "joint_then_continuous", "joint_groups": jgroups.tolist(),
            "joint_converged": bool(fit.converged),
            "joint_multistart": fit.multistart_audit["status"],
            "joint_misfit": float(fit.relative_offdiag_misfit),
            "outer_K": int(outer_meta["K"]),
            "outer_groups": np.asarray(outer_meta["c"], dtype=int).tolist(), **orient,
        }
    raise ValueError(f"unknown fusion mode: {recipe.mode}")


def effective_rank(values: np.ndarray) -> float:
    eigen = np.linalg.eigvalsh(np.corrcoef(np.asarray(values, dtype=np.float64), rowvar=False))
    eigen = np.clip(eigen, 0.0, None)
    return float(eigen.sum() ** 2 / max(float(np.square(eigen).sum()), EPS))


def weight_diagnostics(weight: np.ndarray) -> Mapping[str, float]:
    w = np.abs(np.asarray(weight, dtype=np.float64))
    p = w / max(float(w.sum()), EPS)
    nz = p[p > EPS]
    return {
        "negative_fraction": float(np.mean(np.asarray(weight) < -EPS)),
        "entropy": float(-(nz * np.log(nz)).sum()),
        "effective_weight_count": float(np.exp(-(nz * np.log(nz)).sum())),
        "maximum_share": float(p.max(initial=0.0)),
    }


def normalized_residual_affinity(covariance: np.ndarray) -> np.ndarray:
    """No-K continuous mask from the maintained residual-dependence geometry."""
    affinity, _ = residual_affinity(np.asarray(covariance, dtype=np.float64))
    offdiag = affinity[~np.eye(len(affinity), dtype=bool)]
    scale = float(np.quantile(offdiag, .95)) if offdiag.size else 0.0
    output = np.clip(affinity / max(scale, EPS), 0.0, 1.0)
    output = .5 * (output + output.T)
    np.fill_diagonal(output, 0.0)
    return output


def fit_joint_mask(
    covariance: np.ndarray,
    mask: np.ndarray,
    *,
    anchor_index: int,
    seed: int,
    starts: int = 5,
) -> JointFitResult:
    """Experimental Joint factor fit for a symmetric hard or soft mask.

    This generalizes the maintained optimizer only at the mask seam.  It does
    not claim the hard-partition identifiability theorem for continuous masks.
    """
    observed = np.asarray(covariance, dtype=np.float64)
    group_mask = np.asarray(mask, dtype=np.float64).copy()
    if observed.ndim != 2 or observed.shape[0] != observed.shape[1] or group_mask.shape != observed.shape:
        raise ValueError("covariance/mask mismatch")
    if not np.isfinite(observed).all() or not np.isfinite(group_mask).all():
        raise ValueError("non-finite covariance/mask")
    if np.any(group_mask < 0.0) or np.any(group_mask > 1.0) or not np.allclose(group_mask, group_mask.T):
        raise ValueError("soft mask must be symmetric in [0,1]")
    np.fill_diagonal(group_mask, 0.0)
    rows = tuple(_fit_one_start(
        observed, group_mask, start=i, seed=int(seed), anchor_index=int(anchor_index),
        max_sweeps=5000, relative_tolerance=1e-10, consecutive_stable_sweeps=5,
        monotonicity_tolerance=1e-12,
    ) for i in range(int(starts)))
    converged = [row for row in rows if row.converged]
    selected = min(converged or list(rows), key=lambda row: row.objective_trace[-1])
    fitted = selected.fitted_offdiag
    component = np.outer(selected.global_loading, selected.global_loading) + group_mask * np.outer(
        selected.group_loading, selected.group_loading
    )
    diagonal_raw = np.diag(observed) - np.diag(component)
    model_covariance = component.copy()
    np.fill_diagonal(model_covariance, np.diag(component) + np.maximum(diagonal_raw, 0.0))
    comparisons = [{
        "start": int(row.start),
        "model_normalized_difference": float(np.linalg.norm(row.fitted_offdiag - fitted) / max(np.linalg.norm(fitted), EPS)),
        "global_loading_cosine": absolute_cosine(row.global_loading, selected.global_loading),
    } for row in converged]
    multistart = bool(
        len(converged) >= 4
        and all(row["model_normalized_difference"] <= 1e-5 for row in comparisons)
        and all(row["global_loading_cosine"] >= .999 for row in comparisons)
    )
    return JointFitResult(
        global_loading=selected.global_loading.copy(), group_loading=selected.group_loading.copy(),
        fitted_offdiag=fitted.copy(), model_covariance=model_covariance,
        relative_offdiag_misfit=offdiag_relative_misfit(observed, fitted),
        objective=_objective(observed, fitted), converged=bool(len(converged) >= 4),
        converged_starts=len(converged), selected_start=int(selected.start), starts=rows,
        multistart_audit={"status": "PASS" if multistart else "BLOCKED", "comparisons_to_selected": comparisons},
        jacobian_audit=_profiled_jacobian_audit(selected.global_loading, selected.group_loading, group_mask),
        diagonal_audit={"raw_residual": diagonal_raw, "clipped_count": int(np.sum(diagonal_raw < 0)),
                        "clipped_mass": float(-np.minimum(diagonal_raw, 0).sum())},
    )


def global_loading_weights(values: np.ndarray, fit: JointFitResult, anchor_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the global factor directly, without covariance inversion/IU."""
    return _orient(np.asarray(values, dtype=np.float64), fit.global_loading, int(anchor_index))


__all__ = [
    "FusionRecipe", "answer_standardize", "cell_midranks", "effective_rank",
    "fit_fusion_weights", "weight_diagnostics",
    "fit_joint_mask", "global_loading_weights", "normalized_residual_affinity",
]
