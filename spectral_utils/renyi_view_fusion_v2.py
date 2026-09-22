"""Renyi-view fusion v2 (Stage 3): answer-local fusion of Renyi entropies of the top-15 head.

Design decision after the v1 prototype smoke (Omri, 2026-09-13): the question is
whether a COMBINATION of several Renyi orders localizes better than one entropy,
not whether some alternative single entropy does.  The v1 grid {0.5, 1, 2, 4, inf}
had only three distinct directions on the top-15 support (H2, H4, Hinf are
pairwise |rho| > 0.99 on every smoke answer, all dominated by q_1).  v2 therefore
spends the orders where the views differ, on the tail-sensitive side alpha < 1:

    alpha in {0.1, 0.25, 0.5, 1, 2, inf}

alpha = 4 is dropped (duplicate of H2/Hinf), alpha = 0 is constant (log 15) on
the retained support and stays a diagnostic only.  Every view is defined exactly
as in v1 (``renyi_view_fusion``): q = p/(sum p + 1e-12), s = -log(q + 1e-12),
H_alpha = log(sum q^alpha)/(1-alpha), H_1 = sum q s (== frozen entropy15),
H_inf = -log(max q + 1e-12).

Selected-token block SEL = [a, a^2, a^3], a = -log p(selected).

Banks: ``R6`` = six Renyi views; ``R6_sel`` = R6 + SEL (nine columns).
Solvers: ``equal`` (oriented z-scored mean), ``iu`` (two-component L2 IU-PCR,
IU_FIT_DEFAULTS), ``shrink`` (joint-target LW shrinkage; groups {Renyi} and
{SEL}; R6_sel only), ``joint`` (Joint L-SML, lambda 0, model-inverse map) with
the declared three-group partition

    tail  = {H0.1, H0.25, H0.5}   (orders < 1: sensitive to the head's tail mass)
    head  = {H1, H2, Hinf}        (orders >= 1: sensitive to top-mass concentration)
    sel   = {sel1, sel2, sel3}    (realized token, not the distribution)

This partition is a candidate model, not a validated one: all six Renyi views
are functions of the same q, so the group-conditional-independence premise is
only approximate.  The joint arm therefore reports its correlation-model fit
quality (relative off-diagonal misfit vs the hard-partition fit, multistart
audit, convergence) and numerical conditioning, and is judged against IU and
shrinkage, not only against equal weights.  ``R6__joint`` (two groups only) is
NOT APPLICABLE and never forced; ``R6__shrink`` is identical to ``R6__iu``.

Everything is fitted from the current answer alone.  No labels, no other
answers, no hidden fallback: a failed fit is reported, never substituted.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
from scipy.stats import rankdata

from .direct_probability_fusion import logprob_matrix, zscore_columns, _orient
from .direct_probability_fusion_v2 import selected_surprisal
from .direct_probability_temporal import lw_alpha_memory_bounded
from .joint_lsml import covariance_matrix, fit_joint_lsml, hard_lsml_misfit, regularized_joint_map_weights
from .laplacian_upcr import IU_FIT_DEFAULTS
from .shrinkage_iu import shrink, target_matrix
from .upcr import upcr_fit, upcr_fit_covariance
from .varentropy_contribution_fusion import contributions
from .renyi_view_fusion import head_distribution, renyi_entropy, EPS, K

ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0, np.inf)
VIEW_NAMES = ('H0.1', 'H0.25', 'H0.5', 'H1', 'H2', 'Hinf')
SEL_NAMES = ('sel1', 'sel2', 'sel3')
BANKS = {'R6': VIEW_NAMES, 'R6_sel': VIEW_NAMES + SEL_NAMES}
COLUMN_NAMES = VIEW_NAMES + SEL_NAMES            # weight coordinate system (width 9)
MAX_WEIGHTS = len(COLUMN_NAMES)
# Declared groups: 0 = tail orders (<1), 1 = head orders (>=1), 2 = selected token.
GROUP_OF = {'H0.1': 0, 'H0.25': 0, 'H0.5': 0, 'H1': 1, 'H2': 1, 'Hinf': 1, 'sel1': 2, 'sel2': 2, 'sel3': 2}
GROUP_NAMES = ('tail_orders_lt1', 'head_orders_ge1', 'selected_token')
SINGLE_VIEWS = VIEW_NAMES + ('sel1',)
FUSED_METHODS = ('R6__equal', 'R6__iu', 'R6_sel__equal', 'R6_sel__iu', 'R6_sel__shrink', 'R6_sel__joint')
METHODS = tuple('view__' + v for v in SINGLE_VIEWS) + FUSED_METHODS
NOT_APPLICABLE = {
    'R6__joint': 'Joint L-SML requires at least three groups of at least three columns; the six Renyi views '
                 'form only the two declared order groups (tail < 1, head >= 1); no third group is forced.',
    'R6__shrink': 'With a single group the joint target equals the empirical covariance, the Ledoit-Wolf alpha '
                  'is zero and the arm is identical to R6__iu.',
}
# Extended columns used only in the redundancy diagnostics.
DIAG_NAMES = COLUMN_NAMES + ('varentropy15', 'top1_logprob', 'renyi2_k50')


def seed_for(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(('renyi-view-fusion-v2:' + uid).encode()).digest()[:4], 'little')


def renyi_views(logprobs, k=K):
    """Return the ``[T,6]`` Renyi view matrix and its column names."""
    q, _ = head_distribution(logprobs, k)
    V = np.column_stack([renyi_entropy(q, a) for a in ALPHAS])
    if not np.isfinite(V).all():
        raise ValueError('nonfinite Renyi view')
    return V, VIEW_NAMES


def selected_block(chosen, n_tokens):
    a = selected_surprisal(chosen, n_tokens)
    return np.column_stack((a, a ** 2, a ** 3))


def bank_matrix(logprobs, chosen, bank):
    if bank not in BANKS:
        raise ValueError(bank)
    V, _ = renyi_views(logprobs)
    if bank == 'R6':
        return V, VIEW_NAMES
    X = np.column_stack((V, selected_block(chosen, len(V))))
    if not np.isfinite(X).all():
        raise ValueError('nonfinite bank')
    return X, COLUMN_NAMES


def anchor_stream(logprobs):
    """Label-free orientation anchor: raw K=15 varentropy of this answer."""
    return contributions(logprobs, K).sum(axis=1)


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _corr_matrices(M):
    with np.errstate(invalid='ignore', divide='ignore'):
        pearson = np.corrcoef(M, rowvar=False) if len(M) > 1 else np.full((M.shape[1],) * 2, np.nan)
        ranks = np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])
        spearman = np.corrcoef(ranks, rowvar=False) if len(M) > 1 else np.full((M.shape[1],) * 2, np.nan)
    return np.asarray(pearson, float), np.asarray(spearman, float)


def _condition_number(M):
    Z, keep, _, _ = zscore_columns(M)
    if Z.shape[1] < 2:
        return None
    C = Z.T @ Z / len(Z)
    return _finite_or_none(np.linalg.cond(C))


def _nan_to_none(matrix):
    return [[_finite_or_none(v) for v in row] for row in np.asarray(matrix, float)]


def view_diagnostics(logprobs, chosen):
    """Per-answer redundancy diagnostics over the views and reference scalars."""
    lp50 = np.asarray(logprobs, float)
    X, _ = bank_matrix(logprobs, chosen, 'R6_sel')
    q, _ = head_distribution(logprobs, K)
    hartley = renyi_entropy(q, 0.0)
    anchor = anchor_stream(logprobs)
    top1 = logprob_matrix({'logprobs': logprobs}, k=1)[:, 0]
    if lp50.shape[1] >= 50:
        p50 = np.exp(lp50[:, :50]); p50 = p50 / (p50.sum(axis=1, keepdims=True) + EPS)
        renyi2_50 = -np.log((p50 ** 2).sum(axis=1) + EPS)    # exact frozen topk_renyi2_series
    else:
        renyi2_50 = np.full(len(X), np.nan)
    M = np.column_stack((X, anchor, top1, renyi2_50))
    std = M.std(axis=0)
    pearson, spearman = _corr_matrices(M)
    _, keep, _, _ = zscore_columns(X)
    with np.errstate(invalid='ignore', divide='ignore'):
        anchor_corr = [np.corrcoef(X[:, j], anchor)[0, 1] if len(X) > 1 else np.nan for j in range(X.shape[1])]
        h2_vs_k50 = np.corrcoef(X[:, VIEW_NAMES.index('H2')], renyi2_50)[0, 1] if len(X) > 1 else np.nan
    return dict(
        columns=list(DIAG_NAMES), n_tokens=int(len(X)),
        std=[_finite_or_none(v) for v in std],
        near_constant=[bool(v <= 1e-10) for v in std[:MAX_WEIGHTS]],
        dropped_by_zscore=[bool(not v) for v in keep],
        pearson=_nan_to_none(pearson), spearman=_nan_to_none(spearman),
        condition_R6=_condition_number(X[:, :len(VIEW_NAMES)]),
        condition_R6_sel=_condition_number(X),
        condition_tail=_condition_number(X[:, :3]),
        hartley_value=float(hartley[0]), hartley_std=float(hartley.std()),
        hartley_constant=bool(hartley.std() <= 1e-12 and np.isclose(hartley[0], np.log(K))),
        anchor_correlation=[_finite_or_none(v) for v in anchor_corr],
        h2_k15_vs_renyi2_k50_pearson=_finite_or_none(h2_vs_k50),
    )


def _single_view(X, names, view, anchor):
    j = names.index(view)
    x = np.asarray(X[:, j], float)
    if not len(x):
        raise ValueError('empty view')
    sd = float(x.std())
    if sd <= 1e-10:
        raise ValueError(f'constant view {view}; no within-answer localization signal')
    weights = np.zeros(MAX_WEIGHTS); weights[COLUMN_NAMES.index(view)] = 1.0
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = float(np.corrcoef(x, anchor)[0, 1]) if len(x) > 1 and np.std(anchor) > EPS else np.nan
    return dict(score=x.copy(), weights=weights, effective=weights.copy(), intercept=0.0,
                state={}, diagnostics=dict(active_columns=1, orientation_flipped=False,
                                            anchor_correlation=_finite_or_none(corr), std=sd,
                                            natural_sign='high_is_risk'))


def _iu_weights(fit):
    if fit.abstained or fit.used_simple_average:
        raise ValueError('IU abstention/simple-average fallback (declared failure)')
    return np.asarray(fit.w, float)


def _fused(X, names, solver, anchor, uid):
    if len(X) < 3:
        raise ValueError('fewer than three tokens')
    Z, keep, mean, scale = zscore_columns(X)
    if Z.shape[1] < 3:
        raise ValueError('fewer than three varying columns')
    kept_names = [n for n, k in zip(names, keep) if k]
    with np.errstate(invalid='ignore', divide='ignore'):
        column_corr = np.array([np.corrcoef(Z[:, j], anchor)[0, 1] for j in range(Z.shape[1])])
    signs = np.where(np.isfinite(column_corr) & (column_corr < 0), -1.0, 1.0)
    Zo = Z * signs
    n = len(Zo)
    state = dict(column_signs=signs.copy(), column_anchor_correlation=column_corr.copy())
    diag = dict(kept_columns=kept_names, column_flips=int((signs < 0).sum()))
    groups = np.asarray([GROUP_OF[kn] for kn in kept_names], int)
    if solver == 'equal':
        w = np.full(Zo.shape[1], 1.0 / Zo.shape[1])
    elif solver == 'iu':
        f = upcr_fit(Zo.T, **dict(IU_FIT_DEFAULTS))
        w = _iu_weights(f)
        diag.update(g2_hat=float(f.g2_hat), n_components=int(f.n_components_used))
    elif solver == 'shrink':
        C = Zo.T @ Zo / n
        g = np.where(groups == 2, 1, 0)   # {Renyi} vs {SEL}
        if len(set(g.tolist())) < 2:
            raise ValueError('shrink needs both a Renyi group and a SEL group among varying columns')
        T = target_matrix(C, g, 'joint')
        alpha = lw_alpha_memory_bounded(Zo, C, T)
        f = upcr_fit_covariance(shrink(C, T, alpha), **dict(IU_FIT_DEFAULTS))
        w = _iu_weights(f)
        state['alpha'] = np.asarray(alpha)
        diag.update(alpha=float(alpha), g2_hat=float(f.g2_hat), groups=g.tolist())
    elif solver == 'joint':
        sizes = {int(g): int(np.sum(groups == g)) for g in np.unique(groups)}
        short = sorted(g for g, c in sizes.items() if c < 3)
        if len(sizes) < 3:
            raise ValueError(f'joint needs K>=3 groups, kept groups={sorted(sizes)} (declared failure)')
        if short:
            raise ValueError(f'joint group(s) {short} have fewer than three varying columns; sizes={sizes} (declared failure)')
        cov = covariance_matrix(Zo)
        corr = np.where(np.isfinite(column_corr), np.abs(column_corr), 0.0)
        anchor_index = int(np.argmax(corr))
        fit = fit_joint_lsml(cov, groups, anchor_index=anchor_index, seed=seed_for(uid), starts=5)
        # lam=0.0: the ungated model-inverse map (mode is irrelevant at lambda zero; no graph is built).
        w, mdiag = regularized_joint_map_weights(Zo, fit.model_covariance, fit.global_loading, mode='diag', lam=0.0)
        hard = hard_lsml_misfit(cov, groups)
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
                    anchor_index=anchor_index, n_groups=int(len(sizes)),
                    group_sizes={GROUP_NAMES[k]: v for k, v in sizes.items()}, seed=int(seed_for(uid)), lam=0.0)
    else:
        raise ValueError(solver)
    w = np.asarray(w, float)
    if w.shape != (Zo.shape[1],) or not np.isfinite(w).all() or np.linalg.norm(w) <= EPS:
        raise FloatingPointError('nonfinite, malformed or zero fusion weights')
    score, flipped, corr = _orient(Zo @ w, anchor)
    w_signed = w * signs * (-1.0 if flipped else 1.0)
    state['w'] = w.copy()
    full = np.zeros(len(names)); full[keep] = w_signed
    effective = np.zeros(len(names)); effective[keep] = w_signed / scale[keep]
    intercept = -float(mean[keep] @ effective[keep])
    np.testing.assert_allclose(X @ effective + intercept, score, atol=1e-8, rtol=1e-8)
    weights = np.zeros(MAX_WEIGHTS); eff = np.zeros(MAX_WEIGHTS)
    for j, nm in enumerate(names):
        weights[COLUMN_NAMES.index(nm)] = full[j]; eff[COLUMN_NAMES.index(nm)] = effective[j]
    diag.update(active_columns=int(keep.sum()), orientation_flipped=bool(flipped),
                anchor_correlation=_finite_or_none(corr))
    return dict(score=score, weights=weights, effective=eff, intercept=intercept, state=state, diagnostics=diag)


def fit_all(logprobs, chosen, methods=METHODS, *, uid='no-uid'):
    """Return pure fits and explicit failures; never a substitute score."""
    fits, failures, seconds = {}, {}, {}
    for name in methods:
        if name in NOT_APPLICABLE:
            raise ValueError(f'{name} is registered NOT APPLICABLE: {NOT_APPLICABLE[name]}')
        if name not in METHODS:
            raise ValueError(name)
    try:
        anchor = anchor_stream(logprobs)
    except (ValueError, FloatingPointError) as error:
        reason = f'{type(error).__name__}: anchor/head invalid: {error}'
        return {}, {name: reason for name in methods}, {name: 0.0 for name in methods}
    banks = {}
    for name in methods:
        started = time.perf_counter()
        try:
            prefix, solver = name.split('__')
            bank = 'R6_sel' if prefix == 'view' else prefix
            if bank not in banks:
                banks[bank] = bank_matrix(logprobs, chosen, bank)
            X, names = banks[bank]
            fit = (_single_view(X, list(names), solver, anchor) if prefix == 'view'
                   else _fused(X, list(names), solver, anchor, uid))
            if not np.isfinite(fit['score']).all() or not np.isfinite(fit['weights']).all():
                raise ValueError('nonfinite score or weights')
            fits[name] = fit
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
            failures[name] = f'{type(error).__name__}: {error}'
        seconds[name] = time.perf_counter() - started
    return fits, failures, seconds


__all__ = ['ALPHAS', 'BANKS', 'COLUMN_NAMES', 'DIAG_NAMES', 'EPS', 'K', 'MAX_WEIGHTS', 'METHODS', 'GROUP_OF',
           'GROUP_NAMES', 'NOT_APPLICABLE', 'SEL_NAMES', 'SINGLE_VIEWS', 'VIEW_NAMES', 'anchor_stream',
           'bank_matrix', 'fit_all', 'head_distribution', 'renyi_entropy', 'renyi_views', 'seed_for',
           'selected_block', 'view_diagnostics']
