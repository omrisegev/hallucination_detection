"""Diagnostics of existing fusion/readout; not an alternative detector.

Functions accepting targets are explicitly evaluation-only oracles/metrics.
No targets enter the mixture inspection or projection reconstruction.
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.mixture import GaussianMixture

ARMS = ('equal_parent', 'iu_parent', 'joint_parent', 'joint_graph010_parent',
        'joint_graph_permuted_parent', 'entropy_parent', 'iu__dufs_graph')
REP = 'moments27_local8'
PARENT_ARMS = dict(zip(ARMS[:6], tuple(REP+'__'+s for s in
    ('equal', 'iu', 'joint_lambda0', 'joint_graph010', 'joint_graph_permuted'))
    + ('entropy_mean_w8',)))


def projection(values, names, fit_indices, normal, weights):
    """Recover the exact centered map and an arbitrary-origin projection."""
    x = np.asarray(values, float)[:, [names.index(s) for s in normal['active_features']]]
    mean, sd, signs, w = (np.asarray(a, float) for a in
                         (normal['mean'], normal['sd'], normal['feature_signs'], weights))
    centered = (x-mean)/sd
    residual_mean = centered[np.asarray(fit_indices, int)].mean(axis=0)
    centered -= residual_mean
    centered_score = -((centered*signs) @ w)
    uncentered_score = -((x/sd*signs) @ w)
    offset = -float(((mean/sd + residual_mean)*signs) @ w)
    return centered_score, uncentered_score, offset


def inspect_mixture(fit_risk, step_risk):
    """Exact parent GMM configuration, with additional unlabeled diagnostics."""
    x = np.asarray(fit_risk, float).reshape(-1, 1)
    steps = np.asarray(step_risk, float)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(steps).all():
        raise ValueError('Invalid mixture data')
    with warnings.catch_warnings(record=True) as caught:
        models = [GaussianMixture(n_components=k, n_init=3, max_iter=300,
                  reg_covar=1e-4, random_state=2026090705).fit(x) for k in (1, 2)]
    if not all(m.converged_ for m in models):
        raise ValueError('MIXTURE_NOT_CONVERGED')
    bic = [float(m.bic(x)) for m in models]
    ll = [float(m.score_samples(x).sum()) for m in models]
    split = bic[1] < bic[0]
    threshold = float(models[1].means_.mean()) if split else None
    candidates = np.flatnonzero(steps > threshold) if split else []
    prediction = int(candidates[0]) if len(candidates) else -1
    delta = bic[0]-bic[1]
    duplicated_delta = 4*(ll[1]-ll[0])-3*np.log(2*len(x))
    return {'bic': bic, 'bic_gain': delta, 'log_likelihood': ll,
            'duplicated_fixed_parameter_bic_gain': float(duplicated_delta),
            'two_components_selected': split, 'threshold': threshold,
            'prediction': prediction, 'means': models[1].means_.ravel().tolist(),
            'variances': models[1].covariances_.ravel().tolist(),
            'weights': models[1].weights_.ravel().tolist(),
            'lag1_correlation': float(np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1])
                if len(x)>2 and np.std(x[:-1])>0 and np.std(x[1:])>0 else None,
            'warnings': [str(w.message) for w in caught]}


def comparison_parts(targets, scores):
    """Evaluation-only: pair-count exact decomposition of pooled binary AUC."""
    counts = {'within': 0, 'cross': 0}
    wins = {'within': 0., 'cross': 0.}
    within_auc = []
    for i, (yi, xi) in enumerate(zip(targets, scores)):
        yi, xi = np.asarray(yi), np.asarray(xi, float)
        positive = xi[yi == 1]
        for j, (yj, xj) in enumerate(zip(targets, scores)):
            negative = np.asarray(xj, float)[np.asarray(yj) == 0]
            n = len(positive)*len(negative)
            if not n:
                continue
            victories = float(np.sum(positive[:, None]>negative)+.5*np.sum(positive[:, None]==negative))
            key = 'within' if i==j else 'cross'
            counts[key] += n; wins[key] += victories
            if i==j:
                within_auc.append(victories/n)
    total = sum(counts.values())
    return {'total_pairs': total, 'within_pairs': counts['within'],
            'cross_pairs': counts['cross'],
            'cross_pair_fraction': counts['cross']/total if total else None,
            'pooled_auc': sum(wins.values())/total if total else None,
            'pair_weighted_within_auc': wins['within']/counts['within'] if counts['within'] else None,
            'cross_answer_auc': wins['cross']/counts['cross'] if counts['cross'] else None,
            'mean_within_answer_auc': float(np.mean(within_auc)) if within_auc else None,
            'mixed_answers': len(within_auc)}


def oracle_predictions(target, valid, gate_open, peak):
    """Evaluation-only: invalid fits fail all counterfactuals, even clean ones."""
    if not valid:
        return {k: None for k in ('actual', 'perfect_gate', 'perfect_locator', 'both_perfect')}
    error = target != -1
    return {'actual': int(peak) if gate_open else -1,
            'perfect_gate': int(peak) if error else -1,
            'perfect_locator': int(target) if gate_open and error else (int(peak) if gate_open else -1),
            'both_perfect': int(target)}


def pb_from_predictions(rows, key):
    """Evaluation-only: official clean/exact-error harmonic mean by subset."""
    cells = {}
    for cell in sorted({r['cell'] for r in rows}):
        records = [r for r in rows if r['cell']==cell]
        clean = [r for r in records if r['target']==-1]
        errors = [r for r in records if r['target']!=-1]
        cs = sum(r['predictions'][key]==-1 for r in clean)
        es = sum(r['predictions'][key]==r['target'] for r in errors)
        ca, ea = cs/len(clean) if clean else None, es/len(errors) if errors else None
        f1 = None if ca is None or ea is None else (2*ca*ea/(ca+ea) if ca+ea else 0.)
        cells[cell] = {'clean': len(clean), 'erroneous': len(errors), 'clean_successes': cs,
                       'error_exact_successes': es, 'clean_accuracy': ca, 'error_exact_accuracy': ea, 'f1': f1}
    f1s = [c['f1'] for c in cells.values()]
    return {'cells': cells, 'macro_f1': float(np.mean(f1s)) if f1s and None not in f1s else None}
