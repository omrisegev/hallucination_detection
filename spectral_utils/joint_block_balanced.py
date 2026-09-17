"""Joint off-diagonal factor fit with equal total weight per group-pair block.

Uses the legacy initialization and coordinate minimization, including weighted
nonnegative amplitude updates. Changes the loss, not the hierarchical readout.
Pair products receive the canonical feasibility and profiled-Jacobian checks.
"""
import numpy as np
from scipy.optimize import nnls
from .joint_lsml import (_initial_loadings, _factor_model, absolute_cosine,
                        coassignment_from_labels, offdiag_relative_misfit)
from .joint_pair_extension import pair_model_covariance
from .joint_pair_jacobian import profiled_pair_jacobian


def discover_sourcefold_groups(values, row_folds, *, seed):
    """One bounded K={3,4} stability check using four training-source folds.

    This is a declared block-deletion approximation, not historical LOAO.
    Every partition and the consensus exclude the outer test fold entirely.
    """
    from sklearn.metrics import adjusted_rand_score
    from .joint_lsml import (covariance_matrix, residual_affinity, canonicalize_labels,
                             _spectral_cluster_precomputed)
    covariances = [covariance_matrix(values[row_folds != f]) for f in np.unique(row_folds)]
    affinities = [residual_affinity(c)[0] for c in covariances]
    candidates = []
    for k in (3, 4):
        parts = [canonicalize_labels(_spectral_cluster_precomputed(a, k, seed=seed+1000*k+j))
                 for j, a in enumerate(affinities)]
        affinity = np.mean([coassignment_from_labels(p) for p in parts], axis=0)
        labels = canonicalize_labels(_spectral_cluster_precomputed(affinity, k, seed=seed+100000+k))
        ari = [adjusted_rand_score(labels, p) for p in parts]
        sizes = [int(np.sum(labels == g)) for g in np.unique(labels)]
        valid = len(sizes) == k and min(sizes) >= 2 and all(
            len(np.unique(p)) == k and min(np.bincount(p)) >= 2 for p in parts)
        candidates.append(dict(K=k, labels=labels, group_sizes=sizes, ari=ari,
            valid=valid, median_ari=float(np.median(ari)), mean_ari=float(np.mean(ari)), minimum_ari=float(min(ari))))
    valid = [c for c in candidates if c['valid']]
    if not valid: return dict(status='NO_ADMISSIBLE_PARTITION', candidates=candidates)
    best = sorted(valid, key=lambda c:(-c['median_ari'], -c['mean_ari'], -c['minimum_ari'], c['K']))[0]
    return dict(status='SELECTED', **best, candidates=candidates)


def block_pair_weights(labels):
    labels = np.asarray(labels); p = len(labels); w = np.zeros((p, p))
    left, right = np.triu_indices(p, 1)
    blocks = np.sort(np.column_stack((labels[left], labels[right])), axis=1)
    _, inverse, counts = np.unique(blocks, axis=0, return_inverse=True, return_counts=True)
    weights = 1/counts[inverse].astype(float)
    # Mean pair weight one preserves the optimizer's absolute tolerance scale.
    weights /= weights.mean(); w[left, right] = weights; w[right, left] = weights
    return w


def fit_weighted_start(observed, mask, weights, *, start, seed, anchor_index, max_sweeps=5000):
    loadings = _initial_loadings(observed, mask, start=start, seed=seed, anchor_index=anchor_index)
    masks = [np.ones_like(observed)-np.eye(len(observed)), mask.copy()]
    left, right = np.triu_indices(len(observed), 1); sqrtw = np.sqrt(weights[left, right])
    fitted = _factor_model(*loadings, mask)
    def objective(f): return float(np.sum(weights[left, right]*(observed[left, right]-f[left, right])**2))
    previous = objective(fitted); stable = 0; converged = False; monotone = True
    for used in range(1, max_sweeps+1):
        oldmodel = fitted.copy()
        for f, m in enumerate(masks):
            loading = loadings[f]
            residual = observed - (fitted - m*np.outer(loading, loading))
            for i in range(len(observed)):
                coefficients = m[i]*loading; coefficients[i] = 0
                denom = float((weights[i]*coefficients)@coefficients)
                value = 0. if denom <= 1e-12 else float((weights[i]*coefficients)@residual[i]/denom)
                old = float(loading[i]); loading[i] = value
                delta = m[i]*loading*(value-old); delta[i] = 0
                fitted[i] += delta; fitted[:, i] += delta
        basis = np.column_stack([(m*np.outer(v, v))[left, right] for m, v in zip(masks, loadings)])
        amplitude, _ = nnls(basis*sqrtw[:, None], observed[left, right]*sqrtw)
        loadings = [v*np.sqrt(max(float(a), 0)) for v, a in zip(loadings, amplitude)]
        fitted = _factor_model(*loadings, mask); current = objective(fitted)
        change = np.linalg.norm(fitted-oldmodel)/max(np.linalg.norm(oldmodel), 1e-12)
        if current > previous + 1e-12*max(1, abs(previous)):
            monotone = False; break
        stable = stable+1 if abs(current-previous)/max(1, abs(previous)) <= 1e-10 and change <= 1e-10 else 0
        previous = current
        if stable >= 5:
            converged = True; break
    v, u = loadings
    if v[anchor_index] < 0: v = -v
    return dict(v=v, u=u, fitted=fitted, objective=objective(fitted),
                converged=converged and monotone, sweeps=used, monotone=monotone, start=start)


def fit_block_balanced(observed, labels, *, anchor_index, seed, starts=5):
    observed = np.asarray(observed, float); labels = np.asarray(labels)
    if len(np.unique(labels)) < 3 or min(np.bincount(labels)) < 2:
        raise ValueError('requires >=3 groups with >=2 features')
    mask = coassignment_from_labels(labels); np.fill_diagonal(mask, 0)
    weights = block_pair_weights(labels)
    rows = [fit_weighted_start(observed, mask, weights, start=s, seed=seed,
                anchor_index=anchor_index) for s in range(starts)]
    converged = [r for r in rows if r['converged']]
    best = min(converged or rows, key=lambda r:r['objective'])
    covariance, u, pair = pair_model_covariance(observed, labels, best['v'], best['u'])
    agreements = []
    for r in converged:
        # Reject infeasible converged starts rather than cherry-picking one.
        pair_model_covariance(observed, labels, r['v'], r['u'])
        agreements.append(dict(start=r['start'], cosine=absolute_cosine(r['v'], best['v']),
            model_difference=float(np.linalg.norm(r['fitted']-best['fitted'])/max(np.linalg.norm(best['fitted']), 1e-12))))
    jac = profiled_pair_jacobian(best['v'], u, labels)
    stable = len(converged) >= 4 and all(r['cosine'] >= .999 and r['model_difference'] <= 1e-5 for r in agreements)
    audit = dict(valid=bool(stable and jac['full_global_rank'] and jac['condition_number'] <= 1e8),
        converged_starts=len(converged), multistart='PASS' if stable else 'BLOCKED',
        starts=[{k:r[k] for k in ('start','objective','converged','sweeps','monotone')} for r in rows],
        jacobian=jac, pair=pair, agreements=agreements, objective=best['objective'],
        relative_offdiag_misfit=offdiag_relative_misfit(observed, best['fitted']))
    return best['v'], u, covariance, audit
