"""Cross-rank varentropy expansion banks with answer-local, label-free fusion.

With the frozen top-15 convention (``q = p/(sum p + 1e-12)``, ``s = -log(q+1e-12)``,
``H = sum q s``) the varentropy ``V = sum_i q_i (s_i - H)^2`` expands as

    V = sum_i q_i s_i^2 - sum_{i,j} q_i q_j s_i s_j
      = sum_i D_i - sum_i P_ii - 2 sum_{i<j} P_ij,

with ``D_i = q_i s_i^2`` (second-moment contributions), ``P_ii = (q_i s_i)^2`` and
``P_ij = q_i q_j s_i s_j`` (each unordered pair once).  This module exposes
those terms as separate fusion inputs, keeps the historical 15-contribution
bank by calling :mod:`varentropy_contribution_fusion` unchanged, and fits every
arm from the current answer alone.  No labels, no other answers, no hidden
fallback between arms: a failing arm is absent from ``fits`` and named in
``failures``.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import time

import numpy as np

from .direct_probability_fusion import _orient, logprob_matrix, zscore_columns
from .direct_probability_fusion_v2 import selected_surprisal
from .direct_probability_temporal import lw_alpha_memory_bounded
from .joint_lsml import (
    covariance_matrix,
    fit_joint_lsml,
    hard_lsml_misfit,
    regularized_joint_map_weights,
)
from .laplacian_upcr import IU_FIT_DEFAULTS
from .shrinkage_iu import shrink, target_matrix
from .upcr import upcr_fit, upcr_fit_covariance
from .varentropy_contribution_fusion import contributions
from .varentropy_contribution_fusion import fit_all as historical_fit_all

K = 15
WIDTH = 138
PAIRS = tuple(combinations(range(K), 2))          # 105 unordered pairs, i<j
BLOCKS = ((0, 5), (5, 10), (10, 15))               # rank blocks {1-5},{6-10},{11-15}
BLOCK_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
SOLVERS = ('equal_identity', 'equal_oriented', 'iu', 'shrink', 'joint')
HISTORICAL = ('B1_hist__raw', 'B1_hist__equal', 'B1_hist__iu')
IDENTITY_TOLERANCE = 1e-9
ANALYTIC_PAIR_THRESHOLD = 64   # upcr._fit_block switches to the closed-form pair inverse at m>=64


def _block_of(rank: int) -> int:
    for index, (lo, hi) in enumerate(BLOCKS):
        if lo <= rank < hi:
            return index
    raise ValueError(rank)


def _layout():
    names, coefficients, groups, signs, term = [], [], [], [], []
    for i in range(K):
        names.append(f'D_{i+1}'); coefficients.append(1.0); groups.append(_block_of(i)); signs.append(1.0); term.append('D')
    for i in range(K):
        names.append(f'Pd_{i+1}'); coefficients.append(-1.0); groups.append(3 + _block_of(i)); signs.append(-1.0); term.append('Pii')
    for i, j in PAIRS:
        names.append(f'P_{i+1}_{j+1}'); coefficients.append(-2.0)
        groups.append(6 + BLOCK_PAIRS.index((_block_of(i), _block_of(j)))); signs.append(-1.0); term.append('Pij')
    for name in ('sel_a', 'sel_a2', 'sel_a3'):
        names.append(name); coefficients.append(0.0); groups.append(12); signs.append(1.0); term.append('SEL')
    return (tuple(names), np.asarray(coefficients), np.asarray(groups, dtype=int),
            np.asarray(signs), tuple(term))


NAMES, IDENTITY_COEFFICIENTS, GROUPS, IDENTITY_SIGNS, TERM = _layout()
_D = np.arange(0, 15); _PII = np.arange(15, 30); _PIJ = np.arange(30, 135); _SEL = np.arange(135, 138)
BANK_COLUMNS = {
    'B2d_sel': np.concatenate([_D, _PII, _SEL]),          # 33 columns, primary
    'B2_sel': np.concatenate([_D, _PII, _PIJ, _SEL]),     # 138 columns, primary
    'B2d': np.concatenate([_D, _PII]),                    # 30 columns, secondary
    'B2': np.concatenate([_D, _PII, _PIJ]),               # 135 columns, secondary
}
BANKS = tuple(BANK_COLUMNS)
METHODS = HISTORICAL + tuple(f'{bank}__{solver}' for bank in BANKS for solver in SOLVERS)
EXPECTED_GROUP_COUNT = {'B2d_sel': 7, 'B2_sel': 13, 'B2d': 6, 'B2': 12}
assert len(NAMES) == WIDTH and len(PAIRS) == 105
assert all(len(np.unique(GROUPS[c])) == n for c, n in ((BANK_COLUMNS[b], EXPECTED_GROUP_COUNT[b]) for b in BANKS))


def seed_for(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(('varentropy-expansion-v1:' + uid).encode()).digest()[:4], 'little')


def expansion_columns(logprobs, chosen):
    """Return ``(X, names, coefficients)``: the 138-column bank for one answer.

    ``chosen`` is the cached selected-token ``-log p`` vector (``token_spilled_energies``);
    pass ``None`` for the 135-column bank without the selected block.  The identity
    coefficient vector is ``+1`` on D, ``-1`` on P_ii, ``-2`` on P_ij and ``0`` on SEL.
    """
    lp = logprob_matrix({'logprobs': logprobs}, k=K)
    p = np.exp(lp)
    q = p / (p.sum(axis=1, keepdims=True) + 1e-12)      # frozen epsilon convention
    s = -np.log(q + 1e-12)
    u = q * s
    D = q * s * s
    Pii = u * u
    Pij = np.column_stack([u[:, i] * u[:, j] for i, j in PAIRS])
    blocks = [D, Pii, Pij]
    if chosen is not None:
        a = selected_surprisal(chosen, len(lp))
        blocks.append(np.column_stack([a, a * a, a * a * a]))
    X = np.concatenate(blocks, axis=1)
    if not np.isfinite(X).all():
        raise ValueError('nonfinite expansion bank')
    width = X.shape[1]
    return X, NAMES[:width], IDENTITY_COEFFICIENTS[:width].copy()


def identity_fixed_fusion(X):
    """Fixed algebra-derived fusion of the RAW (unstandardized) bank: equals V."""
    X = np.asarray(X, float)
    return X[:, :135] @ IDENTITY_COEFFICIENTS[:135]


def identity_discrepancy(X, logprobs):
    """Max |sum D - sum P_ii - 2 sum P_ij - contributions(lp,15).sum(1)| over tokens."""
    return float(np.max(np.abs(identity_fixed_fusion(X) - contributions(logprobs, K).sum(axis=1))))


def anchor_series(logprobs):
    """Label-free orientation anchor: the answer's own raw top-15 varentropy."""
    return contributions(logprobs, K).sum(axis=1)


def group_admissibility(groups_kept):
    sizes = {int(g): int(np.sum(groups_kept == g)) for g in np.unique(groups_kept)}
    short = sorted(g for g, n in sizes.items() if n < 3)
    return sizes, short


def _finish(Xb, Z, keep, mean, scale, w, anchor, diagnostics):
    score, flipped, corr = _orient(Z @ w, anchor)
    w = np.asarray(w, float) * (-1.0 if flipped else 1.0)
    weights = np.zeros(Xb.shape[1]); weights[keep] = w
    effective = np.zeros(Xb.shape[1]); effective[keep] = w / scale[keep]
    intercept = -float(mean @ effective)
    np.testing.assert_allclose(Xb @ effective + intercept, score, atol=1e-8, rtol=1e-8,
                               err_msg='affine reconstruction of the standardized fusion failed')
    if not np.isfinite(score).all() or not np.isfinite(weights).all() or np.linalg.norm(w) == 0:
        raise ValueError('nonfinite or zero fusion (declared failure)')
    diagnostics.update(orientation_flipped=bool(flipped),
                       anchor_correlation=float(corr) if np.isfinite(corr) else None,
                       active_columns=int(keep.sum()),
                       tiny_scale_columns=int(np.sum(keep & (scale < 1e-6))),
                       n_tokens=int(len(Z)))
    return dict(score=score, weights=weights, effective=effective, intercept=intercept,
                state=dict(w=w.copy(), keep=keep.copy()), diagnostics=diagnostics)


def _iu_weights(fit):
    if fit.abstained or fit.used_simple_average:
        raise ValueError('IU abstention/simple-average fallback (declared failure)')
    return np.asarray(fit.w, float)


def _solve(solver, Z, keep, groups_b, signs_b, anchor, uid):
    n, p = Z.shape
    diag = {}
    if solver == 'equal_identity':
        w = signs_b[keep] / p
        diag.update(sign_rule='identity: +D, -Pii, -Pij, +SEL')
    elif solver == 'equal_oriented':
        sd = Z.std(axis=0)
        corr = (Z - Z.mean(axis=0)).T @ (anchor - anchor.mean()) / (n * np.where(sd > 0, sd, 1.0) * max(anchor.std(), 1e-300))
        sign = np.where(np.isfinite(corr) & (corr < 0), -1.0, 1.0)   # zero/NaN correlation -> +1 (declared)
        w = sign / p
        diag.update(sign_rule='sign of within-answer Pearson correlation with the anchor',
                    n_negative=int(np.sum(sign < 0)), n_zero_or_nan=int(np.sum(~np.isfinite(corr) | (corr == 0))))
    elif solver == 'iu':
        fit = upcr_fit(Z.T, **dict(IU_FIT_DEFAULTS))
        w = _iu_weights(fit)
        diag.update(g2_hat=float(fit.g2_hat), n_components=int(fit.n_components_used),
                    proj_residual=float(fit.proj_residual), analytic_pair_path=bool(p >= ANALYTIC_PAIR_THRESHOLD))
    elif solver == 'shrink':
        C = Z.T @ Z / n
        g = groups_b[keep]
        T = target_matrix(C, g, 'joint')
        alpha = lw_alpha_memory_bounded(Z, C, T)
        fit = upcr_fit_covariance(shrink(C, T, alpha), **dict(IU_FIT_DEFAULTS))
        w = _iu_weights(fit)
        diag.update(alpha=float(alpha), n_groups=int(len(np.unique(g))), target='joint',
                    g2_hat=float(fit.g2_hat), analytic_pair_path=bool(p >= ANALYTIC_PAIR_THRESHOLD))
    elif solver == 'joint':
        g = groups_b[keep]
        sizes, short = group_admissibility(g)
        if len(sizes) < 3:
            raise ValueError(f'joint needs K>=3 groups, kept groups={sorted(sizes)} (declared failure)')
        if short:
            raise ValueError(f'joint group(s) {short} have fewer than three varying columns; sizes={sizes} (declared failure)')
        cov = covariance_matrix(Z)
        sd = Z.std(axis=0)
        corr = (Z - Z.mean(axis=0)).T @ (anchor - anchor.mean()) / (n * np.where(sd > 0, sd, 1.0) * max(anchor.std(), 1e-300))
        corr = np.where(np.isfinite(corr), corr, 0.0)
        anchor_index = int(np.argmax(np.abs(corr)))
        fit = fit_joint_lsml(cov, g, anchor_index=anchor_index, seed=seed_for(uid), starts=5)
        # lam=0.0: the ungated model-inverse map (mode is irrelevant at lambda zero; no graph is built).
        w, mdiag = regularized_joint_map_weights(Z, fit.model_covariance, fit.global_loading, mode='diag', lam=0.0)
        hard = hard_lsml_misfit(cov, g)
        cos = [row['global_loading_cosine'] for row in fit.multistart_audit['comparisons_to_selected']]
        selected = fit.starts[fit.selected_start]
        diag.update(converged=bool(fit.converged), converged_starts=int(fit.converged_starts),
                    multistart_status=str(fit.multistart_audit['status']),
                    global_loading_cosine_min=float(min(cos)) if cos else None,
                    joint_relative_offdiag_misfit=float(fit.relative_offdiag_misfit),
                    hard_relative_offdiag_misfit=float(hard['relative_offdiag_misfit']),
                    joint_lower_misfit=bool(fit.relative_offdiag_misfit < hard['relative_offdiag_misfit']),
                    model_covariance_condition=float(np.linalg.cond(fit.model_covariance)),
                    map_ridge=float(mdiag['ridge']), map_condition_after=float(mdiag['condition_after']),
                    jacobian_full_global_rank=bool(fit.jacobian_audit['full_global_rank']),
                    jacobian_condition=float(fit.jacobian_audit['condition_number']),
                    selected_start_sweeps=int(selected.sweeps), selected_start=int(fit.selected_start),
                    failed_monotonicity_starts=int(sum(r.failed_monotonicity for r in fit.starts)),
                    diagonal_clipped_count=int(fit.diagonal_audit['clipped_count']),
                    anchor_index=anchor_index, n_groups=int(len(sizes)), group_sizes={str(k): v for k, v in sizes.items()},
                    seed=int(seed_for(uid)), lam=0.0)
    else:
        raise ValueError(solver)
    w = np.asarray(w, float)
    if w.shape != (p,) or not np.isfinite(w).all():
        raise ValueError('solver returned malformed weights (declared failure)')
    return w, diag


def fit_all(logprobs, chosen, *, uid):
    """Return ``(fits, failures, seconds)`` for every method on one answer.

    ``fits[m] = dict(score, weights, effective, intercept, state, diagnostics)`` with
    ``X_bank @ effective + intercept == score`` (checked at 1e-8).  Weights are in
    the bank's own column order (constant columns carry weight 0); the driver
    pads to the 138-column width with NaN.
    """
    fits, failures, seconds = {}, {}, {}
    hist_fits, hist_failures, hist_seconds = historical_fit_all(logprobs)
    for solver in ('raw', 'equal', 'iu'):
        old, new = f'k15__{solver}', f'B1_hist__{solver}'
        seconds[new] = hist_seconds[old]
        if old in hist_fits:
            f = hist_fits[old]
            fits[new] = dict(score=f['score'], weights=f['weights'], effective=f['effective'],
                             intercept=f['intercept'], state=dict(w=f['weights'].copy()),
                             diagnostics=dict(f['diagnostics'], source='varentropy_contribution_fusion.fit_all k15'))
        else:
            failures[new] = hist_failures[old]
    anchor = anchor_series(logprobs)
    started = time.perf_counter()
    X, _, _ = expansion_columns(logprobs, chosen)
    seconds['bank_build'] = time.perf_counter() - started
    for bank in BANKS:
        cols = BANK_COLUMNS[bank]
        Xb = X[:, cols]; groups_b = GROUPS[cols]; signs_b = IDENTITY_SIGNS[cols]
        Z, keep, mean, scale = zscore_columns(Xb)
        for solver in SOLVERS:
            name = f'{bank}__{solver}'; started = time.perf_counter()
            try:
                if len(Xb) < 3 or Z.shape[1] < 3:
                    raise ValueError('fewer than three tokens or varying columns (declared failure)')
                w, diag = _solve(solver, Z, keep, groups_b, signs_b, anchor, uid)
                diag.update(bank=bank, solver=solver, width=int(Xb.shape[1]))
                fits[name] = _finish(Xb, Z, keep, mean, scale, w, anchor, diag)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, AssertionError, RuntimeError) as error:
                failures[name] = f'{type(error).__name__}: {error}'
            seconds[name] = time.perf_counter() - started
    return fits, failures, seconds


DECLARED_FAILURE_MARKERS = ('(declared failure)', 'fewer than three tokens or varying contributions')


def is_declared_failure(reason: str) -> bool:
    return any(marker in reason for marker in DECLARED_FAILURE_MARKERS)


__all__ = [
    'ANALYTIC_PAIR_THRESHOLD', 'BANKS', 'BANK_COLUMNS', 'BLOCKS', 'BLOCK_PAIRS', 'EXPECTED_GROUP_COUNT',
    'GROUPS', 'HISTORICAL', 'IDENTITY_COEFFICIENTS', 'IDENTITY_SIGNS', 'IDENTITY_TOLERANCE', 'K', 'METHODS',
    'NAMES', 'PAIRS', 'SOLVERS', 'TERM', 'WIDTH', 'anchor_series', 'expansion_columns', 'fit_all',
    'group_admissibility', 'identity_discrepancy', 'identity_fixed_fusion', 'is_declared_failure', 'seed_for',
]
