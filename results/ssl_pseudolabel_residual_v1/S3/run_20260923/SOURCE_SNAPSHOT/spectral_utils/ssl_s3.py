"""S3 of the SSL / pseudo-label / residual localization plan (v1.1, section 9): contribution residual.

Contributions h[s, c] sum to BASE exactly (PB: per-channel softmax mass / 11; PRMB: sigmoid(Z) / 11).
On the A rows, each contribution is residualized on the total b = sum_c h_c (weighted OLS), scaled
to unit SD, and a "neutral" direction is chosen in the active space: the eigenvector of the
cell-averaged covariance of U whose eigenvalue is closest to 1 (ties -> lower eigenvalue), oriented
so its component sum is positive.  aux[s] = U[s, :] . v; the fused score is plan 5.4's correction.
This is inspired by contribution-space / NRM, not an implementation of HARP.  No label enters.
"""
import numpy as np
from scipy.special import softmax, expit
from .ssl_s1 import answer_z, teacher_pb, teacher_prm

SD_FLOOR = 1e-8
GAP_FLOOR = 1e-6


def contributions(Z, task):
    """(S, C) -> h (S, C) with h.sum(1) == BASE."""
    Z = np.asarray(Z, float); C = Z.shape[1]
    return (softmax(Z, axis=0) if task == 'pb' else expit(Z)) / C


def fit_residualizer(H, b, w):
    """Weighted per-channel regression of h_c on b. Returns alpha, beta, sd of the raw residual, active mask."""
    w = np.asarray(w, float); w = w / w.sum(); mb = w @ b; vb = w @ (b - mb) ** 2
    mh = w @ H; cov = w @ ((H - mh) * (b - mb)[:, None]); beta = cov / max(vb, 1e-12); alpha = mh - beta * mb
    R = H - alpha - np.outer(b, beta); sd = np.sqrt(w @ R ** 2 - (w @ R) ** 2)
    return {'alpha': alpha, 'beta': beta, 'sd': sd, 'active': sd > SD_FLOOR}


def residualize(H, b, rz):
    R = H - rz['alpha'] - np.outer(b, rz['beta'])
    U = np.where(rz['active'], R / np.maximum(rz['sd'], SD_FLOOR), 0.)
    return R, U


def weighted_cov(U, w):
    w = np.asarray(w, float); w = w / w.sum(); mu = w @ U; D = U - mu
    return (D * w[:, None]).T @ D


def cell_averaged_cov(U, w, cells):
    cells = np.asarray(cells); uc = np.unique(cells)
    return np.mean([weighted_cov(U[cells == c], w[cells == c]) for c in uc], axis=0), len(uc)


def neutral_direction(cov, active):
    """Eigenvector (in the active subspace) whose eigenvalue is closest to 1; tie -> lower eigenvalue.
    Oriented so that the component sum is positive. Returns dict with vector in FULL coordinates."""
    idx = np.flatnonzero(active)
    if len(idx) == 0: return {'status': 'UNIDENTIFIED', 'reason': 'no active column', 'v': np.zeros(len(active))}
    vals, vecs = np.linalg.eigh(cov[np.ix_(idx, idx)])            # ascending
    dist = np.abs(vals - 1.); best = int(np.flatnonzero(dist <= dist.min() + 1e-15)[0])   # lowest eigenvalue among ties
    gap = float(np.min(np.abs(np.delete(vals, best) - vals[best]))) if len(vals) > 1 else np.inf
    v = np.zeros(len(active)); v[idx] = vecs[:, best] / np.linalg.norm(vecs[:, best]); s = v.sum()
    out = {'eigenvalues': vals, 'chosen': best, 'eigenvalue': float(vals[best]), 'eigengap': gap, 'component_sum': float(s), 'v': v}
    if gap <= GAP_FLOOR: out.update(status='UNIDENTIFIED', reason=f'eigengap {gap:.2e} <= {GAP_FLOOR}')
    elif abs(s) <= SD_FLOOR: out.update(status='UNIDENTIFIED', reason='component sum ~ 0: sign not identifiable without labels')
    else: out.update(status='OK', reason=''); out['v'] = v * np.sign(s); out['component_sum'] = float(abs(s))
    return out


def random_direction(active, seed):
    """Normalized Gaussian direction in the active subspace, same orientation rule."""
    idx = np.flatnonzero(active); rng = np.random.default_rng(np.random.SeedSequence([20260923, seed]))
    v = np.zeros(len(active)); g = rng.normal(size=len(idx)); v[idx] = g / np.linalg.norm(g); s = v.sum()
    if abs(s) <= SD_FLOOR: return {'status': 'UNIDENTIFIED', 'v': v, 'component_sum': float(s)}
    return {'status': 'OK', 'v': v * np.sign(s), 'component_sum': float(abs(s))}


def std_answer(v):
    v = np.asarray(v, float); sd = v.std()
    return (v - v.mean()) / sd if sd > SD_FLOOR else np.zeros_like(v)


def corrected(base, aux, dose=0.25):
    base = np.asarray(base, float)
    if dose == 0: return base.copy()
    return base + dose * base.std() * std_answer(aux)
