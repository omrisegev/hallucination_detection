"""Pair groups for Joint L-SML without arbitrary pair-loading variances.

A two-feature group's off-diagonal residual identifies u_i*u_j, not either
loading. For a feasible pair we choose equal fractions of the available
residual variances. This is a declared representative, not recovered latent
truth. Its model covariance has the observed diagonal, independently of the
original pair-loading ratio. Infeasible pairs fail explicitly.

The immutable legacy optimizer is reused. Minimum-three fits replay it
unchanged. Pair fits additionally require agreement of the final native
inverse maps across converged starts. This module has no label API.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np

from .joint_lsml import (
    JointFitResult, _fit_one_start, _objective, _profiled_jacobian_audit,
    absolute_cosine, canonicalize_labels, coassignment_from_labels,
    fit_joint_lsml, offdiag_relative_misfit, regularized_joint_map_weights,
)


@dataclass(frozen=True)
class PairJointFit:
    joint: JointFitResult
    pair_audit: dict[str, Any]
    native_map_audit: dict[str, Any]


class PairCovarianceError(ValueError):
    """An explicit infeasibility result with the offending variance budgets."""
    def __init__(self, reason, detail):
        super().__init__(reason)
        self.detail = detail


def pair_model_covariance(observed, labels, global_loading, group_loading):
    """Reconstruct a PSD factor covariance; reject infeasible pair budgets.

    For b_i=S_ii-v_i^2, r=u_i*u_j, feasibility is b_i,b_j >= 0 and
    r^2 <= b_i*b_j. We set u_i^2=|r| sqrt(b_i/b_j), and symmetrically for j.
    This preserves r and minimizes the largest fraction of either budget.
    Existing diagonal clipping remains unchanged for groups of size >= 3.
    Only tolerance-scale violations may be clamped and are recorded.
    """
    s = np.asarray(observed, dtype=float)
    v, u = np.asarray(global_loading, float), np.asarray(group_loading, float).copy()
    groups = canonicalize_labels(labels)
    p = len(groups)
    if s.shape != (p, p) or v.shape != (p,) or u.shape != (p,):
        raise ValueError('COVARIANCE_LOADING_SHAPE_MISMATCH')
    if not all(np.isfinite(x).all() for x in (s, v, u)) or not np.allclose(s, s.T, atol=1e-12, rtol=1e-12):
        raise ValueError('NONFINITE_OR_ASYMMETRIC_INPUT')
    sizes = [sum(groups == g) for g in np.unique(groups)]
    if len(sizes) < 3 or min(sizes) < 2:
        raise ValueError('REQUIRES_AT_LEAST_THREE_GROUPS_AND_MINIMUM_SIZE_TWO')
    tolerance = 1e-10 * max(1., float(np.max(np.abs(np.diag(s)))))
    pairs = []
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if len(ids) != 2:
            continue
        i, j = map(int, ids)
        budgets = np.diag(s)[ids] - v[ids]**2
        residual = float(u[i] * u[j])
        detail = {'indices': [i, j], 'residual_variance_budgets': budgets.tolist(),
                  'original_product': residual, 'tolerance': tolerance}
        if np.min(budgets) < -tolerance:
            raise PairCovarianceError('PAIR_NEGATIVE_RESIDUAL_VARIANCE', detail)
        safe = np.maximum(budgets, 0.)
        capacity = float(np.sqrt(safe[0]) * np.sqrt(safe[1]))
        if abs(residual) > capacity + tolerance:
            raise PairCovarianceError('PAIR_RESIDUAL_EXCEEDS_VARIANCE_CAPACITY', {**detail, 'capacity': capacity})
        fraction = min(abs(residual) / capacity, 1.) if capacity > 0 else 0.
        u[i], u[j] = np.sqrt(fraction * safe)
        u[j] *= -1. if residual < 0 else 1.
        pairs.append({'indices': [i, j], 'original_product': residual,
                      'residual_variance_budgets': budgets.tolist(),
                      'variance_fraction': fraction, 'product_after': float(u[i] * u[j]),
                      'product_roundoff_adjustment': float(u[i] * u[j] - residual)})
    same = coassignment_from_labels(groups)
    component = np.outer(v, v) + same * np.outer(u, u)
    raw_diagonal = np.diag(s) - np.diag(component)
    covariance = component + np.diag(np.maximum(raw_diagonal, 0.))
    return covariance, u, {
        'status': 'FEASIBLE', 'pair_count': len(pairs), 'pairs': pairs,
        'representative': 'equal_fraction_of_residual_variance_budgets',
        'latent_pair_loadings_identified': False if pairs else None,
        'tolerance': tolerance,
        'diagonal': {'raw_residual': raw_diagonal,
                     'clipped_count': int(np.sum(raw_diagonal < 0)),
                     'clipped_mass': float(-np.minimum(raw_diagonal, 0).sum())},
    }


def fit_joint_pairs(covariance, labels, *, anchor_index, seed=2026090601,
                    starts=5, max_sweeps=5000, target_condition=1000.):
    """Fit pairs with a covariance-defined native head and explicit guards.

    No score/label-dependent fallback occurs here. A caller may use the
    separately registered IU fallback. Pair infeasibility in ANY converged
    start rejects the fit; we do not cherry-pick a feasible local optimum.
    """
    observed = np.asarray(covariance, float)
    partition = canonicalize_labels(labels)
    sizes = [sum(partition == g) for g in np.unique(partition)]
    if len(sizes) < 3 or min(sizes) < 2:
        raise ValueError('REQUIRES_AT_LEAST_THREE_GROUPS_AND_MINIMUM_SIZE_TWO')
    if min(sizes) >= 3:
        legacy = fit_joint_lsml(observed, partition, anchor_index=anchor_index,
                               seed=seed, starts=starts, max_sweeps=max_sweeps)
        return PairJointFit(legacy, {'status': 'EXACT_LEGACY_REPLAY', 'pair_count': 0},
                            {'status': 'LEGACY_GUARDS', 'target_condition': target_condition})
    if observed.shape != (len(partition), len(partition)) or not np.isfinite(observed).all():
        raise ValueError('COVARIANCE_PARTITION_MISMATCH')
    if starts < 4:
        raise ValueError('AT_LEAST_FOUR_STARTS_REQUIRED')
    coassignment = coassignment_from_labels(partition)
    group_mask = coassignment.copy(); np.fill_diagonal(group_mask, 0.)
    results = tuple(_fit_one_start(observed, group_mask, start=i, seed=int(seed),
        anchor_index=int(anchor_index), max_sweeps=int(max_sweeps), relative_tolerance=1e-10,
        consecutive_stable_sweeps=5, monotonicity_tolerance=1e-12) for i in range(starts))
    converged = [x for x in results if x.converged]
    selected = min(converged or results, key=lambda x: x.objective_trace[-1])
    model, canonical_u, pair_audit = pair_model_covariance(
        observed, partition, selected.global_loading, selected.group_loading)
    # fit_values is unused at lambda zero; no score-dependent normalization.
    weight, inverse = regularized_joint_map_weights(None, model, selected.global_loading,
        mode='liu', lam=0., target_condition=target_condition)
    comparisons, map_comparisons = [], []
    for row in converged:
        c, _, _ = pair_model_covariance(observed, partition, row.global_loading, row.group_loading)
        w, _ = regularized_joint_map_weights(None, c, row.global_loading,
            mode='liu', lam=0., target_condition=target_condition)
        comparisons.append({'start': row.start,
            'model_normalized_difference': float(np.linalg.norm(row.fitted_offdiag-selected.fitted_offdiag) /
                                                max(np.linalg.norm(selected.fitted_offdiag), 1e-12)),
            'global_loading_cosine': absolute_cosine(row.global_loading, selected.global_loading)})
        map_comparisons.append({'start': row.start, 'weight_cosine': absolute_cosine(w, weight),
            'covariance_normalized_difference': float(np.linalg.norm(c-model)/max(np.linalg.norm(model), 1e-12))})
    legacy_pass = len(converged) >= 4 and all(x['model_normalized_difference'] <= 1e-5 and
                                              x['global_loading_cosine'] >= .999 for x in comparisons)
    map_pass = len(converged) >= 4 and all(x['weight_cosine'] >= .999 and
                                          x['covariance_normalized_difference'] <= 1e-5 for x in map_comparisons)
    joint = JointFitResult(global_loading=selected.global_loading.copy(), group_loading=canonical_u,
        fitted_offdiag=selected.fitted_offdiag.copy(), model_covariance=model,
        relative_offdiag_misfit=offdiag_relative_misfit(observed, selected.fitted_offdiag),
        objective=_objective(observed, selected.fitted_offdiag), converged=len(converged) >= 4,
        converged_starts=len(converged), selected_start=selected.start, starts=results,
        multistart_audit={'status': 'PASS' if legacy_pass and map_pass else 'BLOCKED',
            'required_converged_starts': 4, 'converged_starts': len(converged), 'comparisons_to_selected': comparisons},
        jacobian_audit=_profiled_jacobian_audit(selected.global_loading, canonical_u, group_mask),
        diagonal_audit=pair_audit['diagonal'])
    return PairJointFit(joint, pair_audit, {'status': 'PASS' if map_pass else 'BLOCKED',
        'target_condition': target_condition, 'comparisons_to_selected': map_comparisons,
        'selected_inverse': inverse})
