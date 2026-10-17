"""S2 of the SSL / pseudo-label / residual localization plan (v1.1, section 8): prediction residual.

Ridge predicts z_token[t, :] from the 16 previous token vectors, 16 history-present bits and
t/max(T-1,1).  The prediction is made BEFORE the current value enters the context; the
whole-answer robust normalization stays offline, so nothing here is a causal-online claim.
Residual e = z_token - prediction; aux[s] = mean_c Top5Mean_c(e[t in step s]); the fused score
is BASE + 0.25 * sd_steps(BASE) * std_answer(aux)  (plan 5.4).  No label enters any fit.
"""
import numpy as np

LAGS = 16
RIDGE_ALPHA = 1.0
DOSE = 0.25


def robust_standardize_tokens(x):
    """Frozen cvf_v2.core.profiles normalization: median / (IQR/1.349), fallback SD, fallback 1."""
    x = np.asarray(x, float)
    med = np.median(x, axis=0); lo, hi = np.percentile(x, [25, 75], axis=0)
    scale = (hi - lo) / 1.349
    scale = np.where(scale > 1e-8, scale, np.where(x.std(0) > 1e-8, x.std(0), 1.))
    return (x - med) / scale


def top5_mean(v):
    """Per-channel mean of the 5 largest values (all values if fewer). v: (n, C)."""
    v = np.asarray(v, float); k = min(5, len(v))
    return np.partition(v, len(v) - k, axis=0)[-k:].mean(0)


def lag_features(z, lags=LAGS):
    """(T, C) -> (T, lags*C + lags + 1).  Row t holds z[t-1], ..., z[t-lags] (zeros when absent),
    lags presence bits, and t/max(T-1,1).  Uses strictly earlier tokens only."""
    z = np.asarray(z, float); T, C = z.shape
    X = np.zeros((T, lags * C + lags + 1))
    for L in range(1, lags + 1):
        if T > L:
            X[L:, (L - 1) * C:L * C] = z[:-L]
            X[L:, lags * C + (L - 1)] = 1.
    X[:, -1] = np.arange(T) / max(T - 1, 1)
    return X


def fit_ridge(X, Y, alpha=RIDGE_ALPHA, **kwargs):
    """Sum of squared errors + alpha*||W||_F^2, intercept unpenalized. Closed form."""
    banned = {'labels', 'label', 'y_true', 'error_steps', 'target', 'first_error', 'classification'} & set(kwargs)
    if banned: raise ValueError(f'fit_ridge does not accept label-like inputs: {sorted(banned)}')
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    xm = X.mean(0); ym = Y.mean(0); Xc = X - xm; Yc = Y - ym
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    b = ym - xm @ W
    return {'W': W, 'b': b, 'alpha': alpha, 'n': len(X), 'd': X.shape[1]}


def predict_ridge(z, model, lags=LAGS):
    return lag_features(z, lags) @ model['W'] + model['b']


def noreset_prediction(z):
    """prediction[t, c] = sum_{u<t} z[u, c] / (t + 1); answer-local, no external fit."""
    z = np.asarray(z, float); T = len(z)
    cs = np.cumsum(z, axis=0); prev = np.vstack([np.zeros((1, z.shape[1])), cs[:-1]])
    return prev / (np.arange(T) + 1)[:, None]


def step_aux(e, spans, offset=0):
    """aux[s] = mean over channels of the per-channel top5 mean of e over the step's tokens."""
    out = np.empty(len(spans))
    for s, (a, b) in enumerate(spans):
        out[s] = top5_mean(e[a - offset:b - offset]).mean()
    return out


def std_answer(v):
    v = np.asarray(v, float); sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-8 else np.zeros_like(v)


def corrected(base, aux, dose=DOSE):
    """Plan 5.4: base + dose * sd_steps(base) * std_answer(aux). dose 0 -> base exactly."""
    base = np.asarray(base, float)
    if dose == 0: return base.copy()
    return base + dose * base.std() * std_answer(aux)


def hierarchical_token_sample(rng, cells, groups, answer_ids, token_counts, n_draws):
    """Cell -> source group -> answer -> token, with replacement. Returns (answer_idx, token_idx) arrays.
    All arrays are aligned over the candidate answers."""
    cells = np.asarray(cells); groups = np.asarray(groups); answer_ids = np.asarray(answer_ids); token_counts = np.asarray(token_counts)
    ucells = np.unique(cells); by_cell = {c: np.flatnonzero(cells == c) for c in ucells}
    by_cell_group = {c: {g: idx[groups[idx] == g] for g in np.unique(groups[idx])} for c, idx in by_cell.items()}
    ai = np.empty(n_draws, int); ti = np.empty(n_draws, int)
    for k in range(n_draws):
        c = ucells[rng.integers(len(ucells))]; gs = list(by_cell_group[c]); g = gs[rng.integers(len(gs))]
        cand = by_cell_group[c][g]; j = cand[rng.integers(len(cand))]
        ai[k] = answer_ids[j]; ti[k] = rng.integers(token_counts[j])
    return ai, ti
