"""Renyi-view fusion v1: answer-local fusion of Renyi entropies of the top-15 head.

Prototype for Stage 3 of the 2026-09-12 plan.  Its final design is deferred
until the cross-rank experiment is reviewed; this module fixes the view
definitions and the fitting mechanics only.

Views are computed on the renormalized top-15 head with exactly the frozen
token_feature_views epsilon convention (q = p/(sum p + 1e-12),
s = -log(q + 1e-12)):

    H_alpha = log(sum_i q_i^alpha + 1e-12) / (1 - alpha)    alpha in {0.5, 2, 4}
    H_1     = sum_i q_i s_i                                  (== frozen entropy15)
    H_inf   = min_i s_i = -log(max_i q_i + 1e-12)            (min-entropy)
    H_0     = log |{i : q_i > 0}| = log 15                   (Hartley; CONSTANT
              on the retained support, therefore excluded from every bank and
              only reported in the diagnostics)

Selected-token block SEL = [a, a^2, a^3] with a = -log p(selected token) taken
from the cached ``token_spilled_energies`` (validated by ``selected_surprisal``).

Banks: ``R5`` = five Renyi views; ``R5_sel`` = R5 + SEL (eight columns).
Solvers: ``equal`` (z-scored mean after per-column orientation), ``iu``
(two-component L2 IU-PCR, IU_FIT_DEFAULTS, abstention = declared failure),
``shrink`` (full-level joint-target covariance shrinkage with the memory-bounded
Ledoit-Wolf alpha; groups {Renyi} and {SEL}).  Joint L-SML needs at least three
groups of at least three columns; no admissible grouping exists for 5 + 3
columns, so the joint arms are registered as NOT APPLICABLE and never forced.

Orientation anchor: the answer's own raw K=15 varentropy (sum of the frozen
contribution matrix), label-free.  Fused columns are oriented by their
within-answer Pearson correlation to the anchor before solving; the fused
score is then passed through ``_orient`` once more and the flip is recorded.
Single-view arms are the raw views with their natural high-is-risk sign and
no data-driven flip, so ``view__H1`` is exactly entropy15.

Everything is fitted from the current answer alone.  No labels, no other
answers, no hidden fallback: a failed fit is reported, never substituted.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.stats import rankdata

from .direct_probability_fusion import logprob_matrix, zscore_columns, _orient
from .direct_probability_fusion_v2 import selected_surprisal
from .direct_probability_temporal import lw_alpha_memory_bounded
from .laplacian_upcr import IU_FIT_DEFAULTS
from .shrinkage_iu import shrink, target_matrix
from .upcr import upcr_fit, upcr_fit_covariance
from .varentropy_contribution_fusion import contributions

EPS = 1e-12
K = 15
ALPHAS = (0.5, 1.0, 2.0, 4.0, np.inf)
VIEW_NAMES = ('H0.5', 'H1', 'H2', 'H4', 'Hinf')
SEL_NAMES = ('sel1', 'sel2', 'sel3')
BANKS = {'R5': VIEW_NAMES, 'R5_sel': VIEW_NAMES + SEL_NAMES}
COLUMN_NAMES = VIEW_NAMES + SEL_NAMES            # weight coordinate system (width 8)
MAX_WEIGHTS = len(COLUMN_NAMES)
SINGLE_VIEWS = VIEW_NAMES + ('sel1',)
FUSED_METHODS = ('R5__equal', 'R5__iu', 'R5_sel__equal', 'R5_sel__iu', 'R5_sel__shrink')
METHODS = tuple('view__' + v for v in SINGLE_VIEWS) + FUSED_METHODS
NOT_APPLICABLE = {
    'R5__joint': 'Joint L-SML requires at least three groups of at least three columns; '
                 'the five Renyi views admit no such partition.',
    'R5_sel__joint': 'Joint L-SML requires at least three groups of at least three columns; '
                     'five Renyi views plus three SEL columns admit no such partition '
                     '(no grouping is forced).',
    'R5__shrink': 'With a single group the joint target equals the empirical covariance, '
                  'the Ledoit-Wolf alpha is zero and the arm is identical to R5__iu.',
}
# Extended columns used only in the redundancy diagnostics.
DIAG_NAMES = COLUMN_NAMES + ('varentropy15', 'top1_logprob', 'renyi2_k50')


def head_distribution(logprobs, k=K):
    """Frozen renormalized head ``q`` and surprisal ``s`` on the top-``k`` support."""
    lp = logprob_matrix({'logprobs': logprobs}, k=k)
    p = np.exp(lp)
    q = p / (p.sum(axis=1, keepdims=True) + EPS)
    return q, -np.log(q + EPS)


def renyi_entropy(q, alpha):
    """Renyi entropy of order ``alpha`` for each row of ``q`` (frozen epsilons)."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not q.shape[0] or not np.isfinite(q).all():
        raise ValueError('need a finite nonempty [T,K] head distribution')
    if alpha == 0:
        return np.log((q > 0).sum(axis=1).astype(float))
    if alpha == 1:
        s = -np.log(q + EPS)
        return (q * s).sum(axis=1)
    if np.isinf(alpha):
        return -np.log(q.max(axis=1) + EPS)
    if alpha < 0:
        raise ValueError('alpha must be nonnegative')
    # No epsilon here: on a renormalized K-support sum_i q_i^alpha >= K^(1-alpha) > 0
    # (alpha=4: >= 2.96e-4), whereas adding 1e-12 inside the log would dominate
    # the sum for large alpha (alpha=16 on a flat head) and break monotonicity.
    # The frozen +1e-12 enters only through s = -log(q + 1e-12) in H_1 and H_inf.
    return np.log((q ** float(alpha)).sum(axis=1)) / (1.0 - float(alpha))


def renyi_views(logprobs, k=K):
    """Return the ``[T,5]`` Renyi view matrix and its column names."""
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
    if bank == 'R5':
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
    """Pearson and Spearman correlation matrices; NaN where a column is constant."""
    with np.errstate(invalid='ignore', divide='ignore'):
        pearson = np.corrcoef(M, rowvar=False) if len(M) > 1 else np.full((M.shape[1],) * 2, np.nan)
        ranks = np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])
        spearman = np.corrcoef(ranks, rowvar=False) if len(M) > 1 else np.full((M.shape[1],) * 2, np.nan)
    return np.asarray(pearson, float), np.asarray(spearman, float)


def _condition_number(M):
    """Condition number of the correlation matrix of the varying columns."""
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
    X, _ = bank_matrix(logprobs, chosen, 'R5_sel')
    q, _ = head_distribution(logprobs, K)
    hartley = renyi_entropy(q, 0.0)
    anchor = anchor_stream(logprobs)
    top1 = logprob_matrix({'logprobs': logprobs}, k=1)[:, 0]
    renyi2_50 = None
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
        condition_R5=_condition_number(X[:, :len(VIEW_NAMES)]),
        condition_R5_sel=_condition_number(X),
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


def _fused(X, names, solver, anchor):
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
    state = dict(column_signs=signs.copy(), column_anchor_correlation=column_corr.copy())
    diag = dict(kept_columns=kept_names, column_flips=int((signs < 0).sum()))
    if solver == 'equal':
        w = np.full(Zo.shape[1], 1.0 / Zo.shape[1])
    elif solver == 'iu':
        f = upcr_fit(Zo.T, **dict(IU_FIT_DEFAULTS))
        if f.abstained or f.used_simple_average:
            raise ValueError('IU abstention/fallback')
        w = np.asarray(f.w, float)
        diag.update(g2_hat=float(f.g2_hat), n_components=int(f.n_components_used))
    elif solver == 'shrink':
        n = len(Zo); C = Zo.T @ Zo / n
        groups = np.asarray([0 if kn in VIEW_NAMES else 1 for kn in kept_names], int)
        if len(set(groups.tolist())) < 2:
            raise ValueError('shrink needs both a Renyi group and a SEL group among varying columns')
        T = target_matrix(C, groups, 'joint')
        alpha = lw_alpha_memory_bounded(Zo, C, T)
        f = upcr_fit_covariance(shrink(C, T, alpha), **dict(IU_FIT_DEFAULTS))
        if f.abstained or f.used_simple_average:
            raise ValueError('IU abstention/fallback on shrunk covariance')
        w = np.asarray(f.w, float)
        state['alpha'] = np.asarray(alpha)
        diag.update(alpha=float(alpha), g2_hat=float(f.g2_hat), groups=groups.tolist())
    else:
        raise ValueError(solver)
    if not np.isfinite(w).all() or np.linalg.norm(w) <= EPS:
        raise FloatingPointError('nonfinite or zero fusion weights')
    score, flipped, corr = _orient(Zo @ w, anchor)
    w_signed = w * signs * (-1.0 if flipped else 1.0)
    state['w'] = w.copy()
    full = np.zeros(len(names)); full[keep] = w_signed
    effective = np.zeros(len(names)); effective[keep] = w_signed / scale[keep]
    intercept = -float(mean[keep] @ effective[keep])
    np.testing.assert_allclose(X @ effective + intercept, score, atol=1e-8, rtol=1e-8)
    # Weights are reported in the fixed eight-column coordinate system.
    weights = np.zeros(MAX_WEIGHTS); eff = np.zeros(MAX_WEIGHTS)
    for j, nm in enumerate(names):
        weights[COLUMN_NAMES.index(nm)] = full[j]; eff[COLUMN_NAMES.index(nm)] = effective[j]
    diag.update(active_columns=int(keep.sum()), orientation_flipped=bool(flipped),
                anchor_correlation=_finite_or_none(corr))
    return dict(score=score, weights=weights, effective=eff, intercept=intercept, state=state, diagnostics=diag)


def fit_all(logprobs, chosen, methods=METHODS):
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
        # Invalid saved head: every arm fails explicitly with the same declared reason.
        reason = f'{type(error).__name__}: anchor/head invalid: {error}'
        return {}, {name: reason for name in methods}, {name: 0.0 for name in methods}
    banks = {}
    for name in methods:
        started = time.perf_counter()
        try:
            prefix, solver = name.split('__')
            bank = 'R5_sel' if prefix == 'view' else prefix
            if bank not in banks:
                banks[bank] = bank_matrix(logprobs, chosen, bank)
            X, names = banks[bank]
            fit = _single_view(X, list(names), solver, anchor) if prefix == 'view' else _fused(X, list(names), solver, anchor)
            if not np.isfinite(fit['score']).all() or not np.isfinite(fit['weights']).all():
                raise ValueError('nonfinite score or weights')
            fits[name] = fit
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
            failures[name] = f'{type(error).__name__}: {error}'
        seconds[name] = time.perf_counter() - started
    return fits, failures, seconds


__all__ = ['ALPHAS', 'BANKS', 'COLUMN_NAMES', 'DIAG_NAMES', 'EPS', 'K', 'MAX_WEIGHTS', 'METHODS',
           'NOT_APPLICABLE', 'SEL_NAMES', 'SINGLE_VIEWS', 'VIEW_NAMES', 'anchor_stream', 'bank_matrix',
           'fit_all', 'head_distribution', 'renyi_entropy', 'renyi_views', 'selected_block',
           'view_diagnostics']
