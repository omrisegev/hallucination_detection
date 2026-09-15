"""Label-blind numerical components for the frozen synthetic isolation gate.

Full-covariance simplex adaptation, NOT native two-component IU-PCR.
"""
from itertools import combinations
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


def covariance(x):
    x = np.asarray(x, dtype=float)
    z = x - x.mean(axis=0)
    return z.T @ z / len(z)


def iu_moments(C, var_y):
    """Batched canonical L2/full-pool IU moments; validated against upcr.py.

    Preserve scipy's leading-eigenvector convention and the300-point grid.
    Only grid projection arithmetic is vectorized, not the mathematical model.
    """
    C = np.atleast_3d(C) if np.asarray(C).ndim == 3 else np.asarray(C)[None]
    n, m, _ = C.shape
    pairs = list(combinations(range(m), 2))
    A = np.array([[int(k == i or k == j) for k in range(m)] for i, j in pairs])
    b = np.array([C[:, i, j] for i, j in pairs]).T
    rho0 = np.linalg.lstsq(A, b.T, rcond=None)[0].T
    v = np.array([eigh(c, subset_by_index=[m-1, m-1])[1][:, 0] for c in C])
    grid = np.linspace(0., float(var_y), 300)
    rho = rho0[:, None, :] + .5 * grid[None, :, None]
    proj = np.einsum('ngm,nm->ng', rho, v)[:, :, None] * v[:, None, :]
    error = np.linalg.norm(rho-proj, axis=2) / (np.linalg.norm(rho, axis=2)+1e-12)
    idx = error.argmin(axis=1)
    chosen = rho[np.arange(n), idx]
    fitted = chosen @ A.T - grid[idx, None]
    residual = np.linalg.norm(fitted-b, axis=1)/(np.linalg.norm(b, axis=1)+1e-12)
    return chosen, grid[idx], residual


def simplex_qp(Q, r):
    """Exact active-face enumeration, batched; raise on invalid KKT solution."""
    Q = np.asarray(Q, float); r = np.asarray(r, float)
    if Q.ndim == 2: Q = Q[None]
    if r.ndim == 1: r = np.broadcast_to(r, (len(Q), len(r)))
    n, m, _ = Q.shape
    best = np.full(n, np.inf); result = np.full((n, m), np.nan)
    for size in range(1, m+1):
        for face in combinations(range(m), size):
            ids = np.array(face)
            block = Q[:, ids[:, None], ids]
            system = np.zeros((n, size+1, size+1))
            system[:, :size, :size] = block
            system[:, :size, size] = 1.; system[:, size, :size] = 1.
            rhs = np.column_stack([r[:, ids], np.ones(n)])
            solution = np.linalg.solve(system, rhs[..., None])[..., 0]
            z = solution[:, :size]
            valid = np.all(z >= -1e-10, axis=1)
            w = np.zeros((n, m)); w[:, ids] = z
            value = .5*np.einsum('ni,nij,nj->n', w, Q, w)-np.sum(w*r, axis=1)
            update = valid & (value < best-1e-13)
            best[update] = value[update]; result[update] = w[update]
    if not np.isfinite(result).all(): raise ValueError('No feasible simplex solution')
    grad = np.einsum('nij,nj->ni', Q, result)-r
    level = np.sum(result*grad, axis=1)
    active = result > 1e-8
    violation = np.max(np.where(active, np.abs(grad-level[:, None]),
                                np.maximum(level[:, None]-grad, 0.)))
    if violation > 1e-7: raise ValueError(f'KKT residual {violation}')
    return result


def fusion_weights(C, scale, var_y, kind='full', rho_override=None, eta=.25):
    C = np.asarray(C, float)
    if C.ndim == 2: C = C[None]
    rho, g2, residual = iu_moments(C, var_y)
    if rho_override is not None: rho = np.broadcast_to(rho_override, rho.shape)
    B = np.asarray(scale)/np.mean(scale)
    Q = C * B[None, :, None] * B[None, None, :]
    r = rho*B
    if kind == 'minvar': r = np.zeros_like(r)
    if kind == 'vertex':
        w = np.eye(len(B))[np.argmax(r, axis=1)]
    elif kind == 'group':
        m = len(B); split = m//2
        u = np.r_[np.ones(split)/split, np.zeros(m-split)]
        v = np.r_[np.zeros(split), np.ones(m-split)/(m-split)]
        d = u-v
        denom = np.einsum('i,nij,j->n', d, Q, d)
        beta = np.clip((r@d-np.einsum('i,nij,j->n', d, Q, v))/denom, 0., 1.)
        w = v + beta[:, None]*d
    else: w = simplex_qp(Q, r)
    final = (1-eta)/len(B) + eta*w
    return final, {'g2':g2, 'rho':rho, 'residual':residual,
                   'a':final*B, 'raw_qp':w}


def local_moments(train_x, train_z, query_z, *, random_seed=None, k=64):
    """One row per independent synthetic source; same kernel masses for null."""
    distances = cdist(query_z, train_z)
    ids = np.argsort(distances, axis=1, kind='stable')[:, :k]
    ds = np.take_along_axis(distances, ids, axis=1)
    width = ds[:, -1]
    # Exactly tied oracle regimes define uniform neighborhoods.
    ratio = np.divide(ds, width[:, None], out=np.zeros_like(ds), where=width[:, None]>1e-12)
    kernel = np.exp(-.5*ratio**2)
    norm = kernel/kernel.sum(axis=1, keepdims=True)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        ids = np.array([rng.choice(len(train_x), k, replace=False) for _ in query_z])
    x = train_x[ids]; mean = np.einsum('nk,nkm->nm', norm, x)
    centered = x-mean[:, None, :]
    cov = np.einsum('nk,nki,nkj->nij', norm, centered, centered)
    neff = 1/np.sum(norm**2, axis=1)
    return mean, cov, neff


def regularize(C):
    C = np.asarray(C)
    return C + np.eye(C.shape[-1])*1e-8


def choose_alpha(train_x, train_z, val_x, val_z, *, random_seed=None):
    mu, cov, _ = local_moments(train_x, train_z, val_z, random_seed=random_seed)
    global_mu = train_x.mean(axis=0); global_cov = regularize(covariance(train_x))
    losses = {}
    for alpha in (0., .5, 1.):
        C = regularize((1-alpha)*cov+alpha*global_cov)
        e = val_x-((1-alpha)*mu+alpha*global_mu)
        sign, logdet = np.linalg.slogdet(C)
        if np.any(sign <= 0): raise ValueError('Non-positive Gaussian covariance')
        quadratic = np.einsum('ni,ni->n', e, np.linalg.solve(C, e[..., None])[..., 0])
        losses[alpha] = float(np.mean(logdet+quadratic))
    chosen = min(losses, key=lambda a:(losses[a], -a))
    return chosen, losses


class CCAContext:
    def fit(self, H, X, mode='linear'):
        self.mode = mode
        left, right = self.views(H, X)
        self.lmean = left.mean(axis=0); self.lsd = left.std(axis=0)
        self.rmean = right.mean(axis=0); self.rsd = right.std(axis=0)
        self.lsd[self.lsd<1e-8] = 1.; self.rsd[self.rsd<1e-8] = 1.
        L = (left-self.lmean)/self.lsd; R = (right-self.rmean)/self.rsd
        C = covariance(np.column_stack([L, R])); C = .9*C+.1*np.diag(np.diag(C))
        p = L.shape[1]
        def inverse_sqrt(c):
            ev, U = np.linalg.eigh(c)
            return (U*(1/np.sqrt(np.maximum(ev, 1e-8))))@U.T
        il = inverse_sqrt(C[:p, :p]); ir = inverse_sqrt(C[p:, p:])
        U, self.singular, Vt = np.linalg.svd(il@C[:p, p:]@ir, full_matrices=False)
        self.A = il@U[:, :2]; self.B = ir@Vt.T[:, :2]
        return self

    def views(self, H, X):
        flat = H.reshape(len(H), -1)
        if self.mode == 'linear': return flat, X
        if self.mode == 'history_square_only': return np.column_stack([flat, flat**2]), X
        if self.mode == 'second_moment': return flat**2, X**2
        raise ValueError(self.mode)

    def transform(self, H, X=None):
        if X is None: X = np.zeros((len(H), H.shape[-1]))
        L, _ = self.views(H, X)
        return ((L-self.lmean)/self.lsd)@self.A

    def diagnostics(self, H, X):
        _, R = self.views(H, X)
        z = self.transform(H); y = ((R-self.rmean)/self.rsd)@self.B
        corr = [float(np.corrcoef(z[:, j], y[:, j])[0, 1]) for j in range(2)]
        def summary_corr(summary):
            matrix = np.corrcoef(np.column_stack([z, summary]), rowvar=False)
            return float(np.nanmax(np.abs(matrix[:2, 2:])))
        return {'held_component_correlations':corr,
                'max_corr_history_mean':summary_corr(H.mean(axis=1)),
                'max_corr_history_energy':summary_corr((H**2).mean(axis=1))}
