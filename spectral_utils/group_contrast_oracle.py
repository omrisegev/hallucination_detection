"""Supervised group compression and signed-weight diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location('group_contrast_shared', Path(__file__).with_name('contrast_span_oracle.py'))
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def group_means_matrix(groups):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    a = np.zeros((len(groups), len(unique)))
    for j, g in enumerate(unique):
        mask = groups == g
        a[mask, j] = 1. / mask.sum()
    return a


def nonnegative_lr(x, y):
    counts = np.bincount(y, minlength=2)
    if (counts == 0).any():
        raise ValueError('Both training classes required')
    weights = len(y) / (2. * counts[y])
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.r_[np.ones(x.shape[1]), 0.]
    def objective(theta):
        z = design @ theta
        value = weights @ (np.logaddexp(0, z) - y * z) + .5 * np.sum(penalty * theta**2)
        gradient = design.T @ (weights * (expit(z) - y)) + penalty * theta
        return value / len(y), gradient / len(y)
    solution = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method='L-BFGS-B',
                        bounds=[(0, None)] * x.shape[1] + [(None, None)],
                        options={'gtol': 1e-11, 'ftol': 1e-15, 'maxiter': 3000, 'maxls': 50})
    _, gradient = objective(solution.x)
    projected = gradient.copy()
    for j in range(x.shape[1]):
        if solution.x[j] <= 1e-10 and gradient[j] > 0:
            projected[j] = 0
    error = float(np.max(np.abs(projected)))
    if not solution.success or error > 1e-7:
        raise AssertionError(f'Nonnegative LR convergence: {solution.message}; projected gradient {error}')
    return solution.x, error


def fold_fit(x, z, y, folds, fold, groups):
    train = (folds != fold) & (y >= 0)
    test = folds == fold
    xs = StandardScaler().fit(x[train])
    zs = StandardScaler().fit(z[train])
    w, zg = xs.transform(x), zs.transform(z)
    a = group_means_matrix(groups)
    d, pairs = cs.differences(x.shape[1], groups)
    group_basis = cs.span(xs.scale_[:, None] * a)
    internal_basis = cs.span(xs.scale_[:, None] * d)
    complete_basis = cs.span(xs.scale_[:, None] * np.column_stack([a, d]))
    assert complete_basis.shape[1] == x.shape[1]
    between, _ = cs.differences(z.shape[1])
    between_basis = cs.span(np.column_stack([np.ones(z.shape[1]), between]))
    matrices = {
        'LR_GROUPS': zg,
        'LR_GROUP_MEAN_AND_CONTRASTS': zg @ between_basis,
        'LR_GROUPS_PLUS_WITHIN': np.column_stack([zg, w @ internal_basis]),
        'LR_RAW_GROUP_SPAN': w @ group_basis,
        'LR_RAW_GROUPS_PLUS_WITHIN': w @ complete_basis,
    }
    scores, records = {}, []
    for name, matrix in matrices.items():
        model = cs.lr_fit(matrix[train], y[train])
        scores[name] = model.decision_function(matrix[test])
        records.append(dict(method=name, n_columns=matrix.shape[1], coef=model.coef_.ravel().tolist(),
                            intercept=float(model.intercept_[0]),
                            n_negative=int(np.sum(model.coef_.ravel() < -1e-8))))
    theta, grad = nonnegative_lr(zg[train], y[train])
    scores['LR_GROUPS_NONNEG'] = zg[test] @ theta[:-1] + theta[-1]
    records.append(dict(method='LR_GROUPS_NONNEG', n_columns=zg.shape[1], coef=theta[:-1].tolist(),
                        intercept=float(theta[-1]), projected_gradient=grad))
    rotation_error = float(np.max(np.abs(scores['LR_GROUPS'] - scores['LR_GROUP_MEAN_AND_CONTRASTS'])))
    assert rotation_error <= 1e-5, rotation_error
    info = dict(fold=fold, n_train=int(train.sum()), n_test=int(test.sum()),
                n_groups=z.shape[1], n_channels=x.shape[1], within_rank=internal_basis.shape[1],
                n_within_pairs=len(pairs), rotation_error=rotation_error,
                x_mean=xs.mean_.tolist(), x_scale=xs.scale_.tolist(),
                z_mean=zs.mean_.tolist(), z_scale=zs.scale_.tolist(),
                group_basis=group_basis.tolist(), internal_basis=internal_basis.tolist(),
                complete_basis=complete_basis.tolist(), between_basis=between_basis.tolist(), fits=records)
    return test, scores, info


def independent_metrics(score, bank, pop):
    """Count positive-negative pairs by searchsorted, independent of rankdata."""
    auc = np.full(len(bank['answers']), np.nan)
    hit = np.full_like(auc, np.nan)
    for j, (i, a, b) in enumerate(zip(bank['answers'], bank['off'][:-1], bank['off'][1:])):
        s = score[a:b]
        if pop.prm[i] and pop.eligible[i]:
            y = pop.labels[pop.off[i]:pop.off[i + 1]].astype(bool)
            negative = np.sort(s[~y])
            positive = s[y]
            less = np.searchsorted(negative, positive, side='left')
            leq = np.searchsorted(negative, positive, side='right')
            auc[j] = np.sum((less + leq) / 2.) / (len(negative) * len(positive))
        elif not pop.prm[i] and pop.target[i] >= 0:
            top = np.max(s)
            chosen = next(k for k, value in enumerate(s) if value >= top - 8 * np.finfo(float).eps)
            hit[j] = float(chosen == pop.target[i])
    return auc, hit
