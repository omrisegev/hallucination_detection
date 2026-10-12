"""Development-only supervised contrast-span diagnostic; no label-free claim."""
from itertools import combinations
import warnings
import numpy as np
from scipy.stats import rankdata
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def span(a):
    a = np.asarray(a, float)
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, s > max(a.shape) * np.finfo(float).eps * (s[0] if len(s) else 1)]


def differences(m, groups=None):
    pairs = [(i, j) for i, j in combinations(range(m), 2)
             if groups is None or groups[i] == groups[j]]
    d = np.zeros((m, len(pairs)))
    for k, (i, j) in enumerate(pairs):
        d[i, k], d[j, k] = 1., -1.
    return d, pairs


def answer_std(x, off):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for a, b in zip(off[:-1], off[1:]):
        block = x[a:b]
        sd = block.std(axis=0)
        out[a:b] = np.divide(block - block.mean(axis=0), sd,
                             out=np.zeros_like(block), where=sd > 1e-8)
    return out


def lr_fit(x, y, legacy=False):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both classes required in training')
    model = LogisticRegression(C=1., class_weight='balanced', max_iter=3000,
                               tol=1e-4 if legacy else 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    if not legacy:
        # L-BFGS may stop on relative objective change before rotated solutions
        # agree. Newton-refine the SAME weighted logistic + C=1 L2 objective.
        # This changes neither the statistical model nor the registered gate.
        counts = np.bincount(y, minlength=2)
        weight = len(y) / (2. * counts[y])
        design = np.column_stack([x, np.ones(len(x))])
        theta = np.r_[model.coef_.ravel(), model.intercept_]
        penalty = np.r_[np.ones(x.shape[1]), 0.]
        for iteration in range(12):
            probability = expit(design @ theta)
            grad = design.T @ (weight * (probability - y)) + penalty * theta
            if np.max(np.abs(grad)) / weight.sum() < 1e-12:
                break
            variance = weight * probability * (1. - probability)
            hessian = design.T @ (variance[:, None] * design) + np.diag(penalty)
            theta -= np.linalg.solve(hessian, grad)
        else:
            raise AssertionError('Newton precision refinement did not converge')
        model.coef_ = theta[:-1][None, :]
        model.intercept_ = theta[-1:]
    return model


def fold_fit(x, y, folds, fold, groups, names, off):
    """Fit all transforms without using this fold's features or labels."""
    train = (folds != fold) & (y >= 0)
    train_unsup = folds != fold
    test = folds == fold
    scale = StandardScaler().fit(x[train])
    w = scale.transform(x)
    centered = w[train_unsup] - w[train_unsup].mean(axis=0)
    _, vec = np.linalg.eigh(centered.T @ centered / len(centered))
    p = vec[:, -1:]
    if p.sum() < 0:
        p = -p
    m = x.shape[1]
    single = None
    if 'realized_drv' in names and 'chosen_surprisal' in names:
        single = np.zeros((m, 1))
        single[names.index('realized_drv'), 0] = 1.
        single[names.index('chosen_surprisal'), 0] = -1.
    d, pairs = differences(m, groups)
    all_d, all_pairs = differences(m)
    # Raw X directions translated to standardized W coordinates.
    scaled = lambda a: scale.scale_[:, None] * a
    residual = np.eye(m) - p @ p.T
    bases = {
        'LR_PC1': p,
        'LR_PC1_within_group_linear': span(np.column_stack([p, scaled(d)])),
        'LR_mean_all_pair_linear': span(scaled(np.column_stack([np.ones(m), all_d]))),
        'LR_PC1_all_residuals_linear': span(np.column_stack([p, residual])),
        'LR_all_X': np.eye(m),
    }
    if single is not None:
        bases['LR_PC1_single_linear'] = span(np.column_stack([p, scaled(single)]))
    if bases['LR_mean_all_pair_linear'].shape[1] != m:
        raise AssertionError('Complete contrast span does not reconstruct X')
    predictions = {'PC1_score': (w[test] @ p).ravel()}
    records = []
    for name, basis in bases.items():
        z = w @ basis
        model = lr_fit(z[train], y[train])
        predictions[name] = model.decision_function(z[test])
        records.append({'method': name, 'fold': fold, 'rank': basis.shape[1],
                        'basis': basis.tolist(), 'coef': model.coef_.tolist(),
                        'effective_coef_W': (basis @ model.coef_.ravel()).tolist(),
                        'intercept': model.intercept_.tolist(),
                        'iterations': model.n_iter_.tolist()})
    for name, directions in [('LR_PC1_single_answer_std', single),
                              ('LR_PC1_within_group_answer_std', d)]:
        if directions is None:
            continue
        # Answer-only transform is applied independently to each whole answer.
        z = np.column_stack([w @ p, answer_std(x @ directions, off)])
        zs = StandardScaler().fit(z[train])
        model = lr_fit(zs.transform(z[train]), y[train])
        predictions[name] = model.decision_function(zs.transform(z[test]))
        records.append({'method': name, 'fold': fold, 'columns': z.shape[1],
                        'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
                        'z_mean': zs.mean_.tolist(), 'z_scale': zs.scale_.tolist()})
    legacy = lr_fit(w[train], y[train], legacy=True)
    predictions['LR_legacy_replay'] = legacy.decision_function(w[test])
    records.append({'method': 'LR_legacy_replay', 'fold': fold,
                    'coef': legacy.coef_.tolist(), 'intercept': legacy.intercept_.tolist()})
    checks = {}
    for name in ['LR_mean_all_pair_linear', 'LR_PC1_all_residuals_linear']:
        error = float(np.max(np.abs(predictions[name] - predictions['LR_all_X'])))
        if error > 1e-5:
            raise AssertionError(f'Full-space LR rotation mismatch {name}: {error}')
        checks[name + '_max_score_error'] = error
    info = {'fold': fold, 'n_train_labeled': int(train.sum()),
            'n_train_unlabeled': int(train_unsup.sum()), 'n_test': int(test.sum()),
            'n_within_group_pairs': len(pairs), 'n_all_pairs': len(all_pairs),
            'single_contrast_available': single is not None,
            'within_group_pairs': [[names[i], names[j]] for i, j in pairs],
            'scale_mean': scale.mean_.tolist(), 'scale_sd': scale.scale_.tolist(),
            'pc1': p.ravel().tolist(), 'checks': checks, 'fits': records}
    return test, predictions, info


def per_answer(score, bank, pop):
    auc = np.full(len(bank['answers']), np.nan)
    hit = np.full_like(auc, np.nan)
    for j, (i, a, b) in enumerate(zip(bank['answers'], bank['off'][:-1], bank['off'][1:])):
        s = score[a:b]
        if not len(s) or not np.isfinite(s).all():
            raise AssertionError('Empty or invalid score vector')
        am = int(np.flatnonzero(s >= s.max() - 8 * np.finfo(float).eps)[0])
        if pop.prm[i] and pop.eligible[i]:
            yy = pop.labels[pop.off[i]:pop.off[i + 1]].astype(bool)
            n1, n0 = int(yy.sum()), int((~yy).sum())
            auc[j] = (rankdata(s)[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        elif not pop.prm[i] and pop.target[i] >= 0:
            hit[j] = float(am == pop.target[i])
    return auc, hit


def bootstrap_differences(frame, pairs, draws=5000, seed=20261005):
    """Joint source-question resampling, retaining the eight-cell PB macro."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    result = []
    for bank in sorted(frame.bank.unique()):
        for panel in ['PRMB', 'PB']:
            f = frame[(frame.bank == bank) & (frame.panel == panel) & frame.value.notna()]
            wide = f.pivot(index=['cell', 'source_group', 'answer'], columns='method', values='value').reset_index()
            available_pairs = [(a, b) for a, b in pairs if a in wide and b in wide
                               and wide[a].notna().all() and wide[b].notna().all()]
            all_groups = sorted(wide.source_group.unique())
            all_cells = sorted(wide.cell.unique())
            gi = pd.Categorical(wide.source_group, categories=all_groups).codes
            ci = pd.Categorical(wide.cell, categories=all_cells).codes
            numerator = np.zeros((len(all_groups), len(all_cells), len(available_pairs)))
            denominator = np.zeros_like(numerator)
            for k, (a, b) in enumerate(available_pairs):
                v = (wide[a] - wide[b]).to_numpy()
                ok = np.isfinite(v)
                np.add.at(numerator[:, :, k], (gi[ok], ci[ok]), v[ok])
                np.add.at(denominator[:, :, k], (gi[ok], ci[ok]), 1)
            values = []
            for start in range(0, draws, 100):
                count = min(100, draws - start)
                weights = rng.multinomial(len(all_groups), np.full(len(all_groups), 1 / len(all_groups)), size=count)
                ns = np.einsum('bg,gcp->bcp', weights, numerator, optimize=True)
                ds = np.einsum('bg,gcp->bcp', weights, denominator, optimize=True)
                value = np.divide(ns, ds, out=np.full_like(ns, np.nan), where=ds > 0)
                # Undefined cell => undefined replicate, not a different macro.
                values.append(value.mean(axis=1))
            bs = np.concatenate(values)
            point = (numerator.sum(axis=0) / denominator.sum(axis=0)).mean(axis=0)
            for k, (a, b) in enumerate(available_pairs):
                valid = bs[:, k][np.isfinite(bs[:, k])]
                lo, hi = np.quantile(valid, [.025, .975])
                result.append(dict(bank=bank, panel=panel, method=a, reference=b,
                                   delta=point[k], lo95=lo, hi95=hi,
                                   valid_draws=len(valid), n_source_groups=len(all_groups)))
    return pd.DataFrame(result)
