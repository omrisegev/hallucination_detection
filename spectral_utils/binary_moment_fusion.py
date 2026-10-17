"""Answer-local binary fusion of Top-15 entropy and varentropy contributions.

This is an empirical adaptation of the existing SML/L-SML solvers, not a
claim that thresholding establishes their conditional-independence model.
"""
import time
import numpy as np
from scipy.linalg import eigh
from .direct_probability_fusion import logprob_matrix, zscore_columns
from .direct_probability_fusion_v2 import selected_surprisal
from .fusion_utils import sml_fuse_signed, lsml_fuse, detect_dependent_groups

BANKS = ('var18', 'both33')
SOLVERS = ('continuous_equal', 'binary_equal', 'sml', 'lsml')
METHODS = tuple(f'{b}__{s}' for b in BANKS for s in SOLVERS)


def representation(logprobs, chosen):
    lp = logprob_matrix({'logprobs': logprobs}, k=15)
    p = np.exp(lp)
    q = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    s = -np.log(q + 1e-12)
    ec = q * s
    H = ec.sum(axis=1)
    vc = q * (s - H[:, None]) ** 2
    V = vc.sum(axis=1)
    a = selected_surprisal(chosen, len(lp))
    summaries = np.column_stack((H, V, a))
    return {'var18': np.column_stack((vc, summaries)),
            'both33': np.column_stack((ec, vc, summaries))}, H


def prepare(X, entropy):
    if len(X) < 3 or not np.isfinite(X).all():
        raise ValueError('need at least three finite token rows')
    Z, keep, means, scales = zscore_columns(X)
    indices = np.flatnonzero(keep)
    centered_anchor = entropy - entropy.mean()
    # A shared, label-free risk convention for equal and learned solvers.
    signs = np.where(Z.T @ centered_anchor < 0, -1., 1.)
    Z = Z * signs
    thresholds = np.median(X[:, indices] * signs, axis=0)
    B = np.where(X[:, indices] * signs > thresholds, 1., -1.)
    # Keep the first representative of exact duplicate binary votes.
    retained, seen = [], set()
    for j in range(B.shape[1]):
        if np.all(B[:, j] == B[0, j]):
            continue
        key = np.packbits(B[:, j] > 0).tobytes()
        if key not in seen:
            retained.append(j); seen.add(key)
    return Z, B[:, retained], indices, np.asarray(retained, int), signs, thresholds


def strict_signed(B):
    if B.shape[1] < 3:
        raise ValueError('fewer than three distinct binary detectors')
    C = np.cov(B.T); off = C - np.diag(np.diag(C))
    eigenvalues, _ = eigh(off)  # fail explicitly, before legacy fallback handler
    gap = float(eigenvalues[-1] - eigenvalues[-2])
    if gap <= 1e-12 * max(1., float(np.max(np.abs(eigenvalues)))):
        raise ValueError('nonunique leading spectral direction')
    score, w = sml_fuse_signed(*B.T, small_m_guard=False)
    if not np.isfinite(w).all() or np.sum(np.abs(w)) <= 0:
        raise ValueError('invalid signed spectral weights')
    return score, w, gap


def fit_all(logprobs, chosen, unused_entropy=None):
    banks, H = representation(logprobs, chosen)
    fits, failures, seconds = {}, {}, {}
    for bank, X in banks.items():
        try:
            Z, B, indices, retained, signs, thresholds = prepare(X, H)
            shared = dict(active_columns=len(indices), distinct_votes=B.shape[1],
                          retained_columns=indices[retained].tolist(),
                          signs=signs.tolist(), threshold_columns=indices.tolist(),
                          signed_thresholds=thresholds.tolist(),
                          removed_binary_columns=int(len(indices)-B.shape[1]))
        except (ValueError, np.linalg.LinAlgError) as e:
            for solver in SOLVERS:
                failures[f'{bank}__{solver}'] = str(e)
                seconds[f'{bank}__{solver}'] = 0.
            continue
        for solver in SOLVERS:
            name=f'{bank}__{solver}'; started=time.perf_counter(); diagnostic=dict(shared)
            weights=np.zeros(X.shape[1]); effective=np.zeros(X.shape[1]); intercept=0.
            try:
                if Z.shape[1] < 3: raise ValueError('fewer than three varying inputs')
                if solver == 'continuous_equal':
                    w=np.ones(Z.shape[1])/Z.shape[1]; score=Z@w
                    weights[indices]=w*signs
                elif solver == 'binary_equal':
                    if B.shape[1] < 3: raise ValueError('fewer than three distinct votes')
                    w=np.ones(B.shape[1])/B.shape[1]; score=B@w
                    weights[indices[retained]]=w
                elif solver == 'sml':
                    _, w, gap = strict_signed(B)
                    w=w/np.abs(w).sum(); score=B@w
                    weights[indices[retained]]=w
                    diagnostic['spectral_gap']=gap
                else:
                    if B.shape[1] < 5: raise ValueError('fewer than five distinct votes for group discovery')
                    K,c,residual,_,gdiag=detect_dependent_groups(
                        list(B.T), K_range=range(2,min(B.shape[1],9)),
                        method='residual',loading_scale='unit',return_diag=True)
                    if not np.isfinite(residual): raise ValueError('group discovery failed')
                    # Check eigensolver health before using the unchanged legacy solver.
                    for g in np.unique(c):
                        local=B[:,c==g]
                        if local.shape[1]>1:
                            C=np.cov(local.T); eigh(C-np.diag(np.diag(C)))
                    score,meta=lsml_fuse(*B.T,groups=c,method='residual',loading_scale='unit')
                    cw=np.asarray(meta['cross_weights']); score=score/np.abs(cw).sum()
                    diagnostic.update(groups=c.tolist(),group_count=int(K),grouping=gdiag,
                        residual=float(residual),cross_weights=(cw/np.abs(cw).sum()).tolist(),
                        within_weights=[dict(columns=indices[retained][idx].tolist(),weights=w.tolist())
                                        for idx,w in meta['group_weights']],
                        constant_virtual=int(np.sum(np.std(meta['virtual_classifiers'],axis=0)==0)))
                    weights[:]=np.nan  # nonlinear two-level vote has no raw-input weight vector
                if not np.isfinite(score).all(): raise ValueError('nonfinite fused scores')
                diagnostic['constant_score']=bool(np.std(score)<=1e-12)
                fits[name]=dict(score=score,weights=weights,effective=effective,
                               intercept=intercept,diagnostics=diagnostic)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
                failures[name]=f'{type(e).__name__}: {e}'
            seconds[name]=time.perf_counter()-started
    return fits,failures,seconds
