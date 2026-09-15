"""Claude's own synthetic checks on the S0 framework (no real data, no labels in fitting).

A1  continuous difficulty gate (eta_t from g2/var_y) vs fixed eta, population covariance per regime.
A2  equal-variance informative world: energy context vs regime context.
B   temporal world with slow shared nuisance: IU on levels vs IU on innovations.
"""
import sys, numpy as np
ROOT = r'C:\Users\omris\TAU\hallucination_detection\.worktrees\temporal-research-20260915'
sys.path.insert(0, ROOT)
from scripts import dsp_contextual_iu_synthetic as old
from scripts.run_cca_iu_isolation_gate import population, history
from spectral_utils.cca_iu_isolation import covariance, regularize, fusion_weights, local_moments, choose_alpha
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.contextual_iu import DEFAULT_IU_FIT
from sklearn.metrics import roc_auc_score
np.set_printoptions(precision=4, suppress=True)
SEEDS = range(20); EPS_L = .1; G_HI = .5

def gated_eta(g2, var_y, eta_max):
    return eta_max * np.clip((g2 / var_y - EPS_L) / (G_HI - EPS_L), 0., 1.)

def blend(w_raw, eta):
    m = w_raw.shape[-1]; eta = np.asarray(eta, float).reshape(-1, 1)
    return (1 - eta) / m + eta * w_raw

# ------------------------------------------------------------------ A1
def part_a1():
    print('\n=== A1: continuous difficulty gate, population covariance per regime ===')
    table = {}
    for world in ('informative', 'coherent_nuisance', 'null'):
        rows = []
        for seed in SEEDS:
            train = old._world(seed, world, 320); test = old._world(seed + 100000, world, 640)
            X, Xt = train[0], test[0]; scale = X.std(axis=0); m = X.shape[1]
            Cg = regularize(covariance((X - X.mean(0)) / scale)); var_y = .25 * np.trace(Cg) / m
            covs = []
            for active in (np.r_[np.ones(3), np.zeros(3)], np.r_[np.zeros(3), np.ones(3)]):
                c, _ = population(world, active); covs.append(c / scale[:, None] / scale[None, :])
            popglobal = np.mean(covs, 0); idx = (~test[7][:, 0]).astype(int); pc = np.array(covs)[idx]
            out = {}
            w, _ = fusion_weights(popglobal, scale, var_y, 'full'); out['static_eta.25'] = (Xt * w).sum(1)
            w, t = fusion_weights(pc, scale, var_y, 'full'); out['context_eta.25'] = (Xt * w).sum(1)
            raw = t['raw_qp']; g2 = t['g2']
            out['context_eta1'] = (Xt * raw).sum(1)
            for eta_max in (1., .5):
                out[f'context_gated_max{eta_max:g}'] = (Xt * blend(raw, gated_eta(g2, var_y, eta_max))).sum(1)
            kw = dict(DEFAULT_IU_FIT); kw['var_y'] = var_y
            ng = upcr_fit_covariance(popglobal, **kw).w
            nl = np.array([upcr_fit_covariance(c, **kw).w for c in covs]); nl *= abs(ng).sum() / abs(nl).sum(1)[:, None]
            Zt = (Xt - X.mean(0)) / scale
            out['native_context'] = (Zt * nl[idx]).sum(1)
            # gated native: blend native context weights (normalised to sum 1 by abs) with equal
            nl_s = nl / nl.sum(1, keepdims=True)
            out['native_gated_max1'] = (Zt * blend(nl_s[idx], gated_eta(g2, var_y, 1.))).sum(1)
            rows.append({k: roc_auc_score(test[4], v) for k, v in out.items()} | {'g2_frac_mean': float(np.mean(g2 / var_y))})
        table[world] = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    names = list(next(iter(table.values())))
    print(f"{'arm':26s}" + ''.join(f'{w:>20s}' for w in table));
    for n in names: print(f'{n:26s}' + ''.join(f'{table[w][n]:20.4f}' for w in table))

# ------------------------------------------------------------------ A2
def world_equal_variance(seed, n):
    rng = np.random.default_rng(seed); regime = rng.integers(0, 2, n); target = rng.normal(size=n)
    labels = (target + rng.normal(scale=.15, size=n) > 0).astype(int)
    active = np.column_stack([np.where(regime == 0, f < 3, f >= 3) for f in range(6)])
    s_in = np.sqrt(1 + .45 ** 2)
    X = np.where(active, target[:, None] + rng.normal(scale=.45, size=(n, 6)), rng.normal(scale=s_in, size=(n, 6)))
    return X, labels, active

def history_equal_variance(active, seed):
    rng = np.random.default_rng(seed); n, m = active.shape; a = active[:, None, :]
    target = rng.normal(size=(n, 16, 1)); s_in = np.sqrt(1 + .45 ** 2)
    return np.where(a, target + rng.normal(scale=.45, size=(n, 16, m)), rng.normal(scale=s_in, size=(n, 16, m)))

def context_arm(Z, Cg, scale, var_y, tz, qz, inner_x, iz, val_x, vz, Xt, random_seed=None):
    alpha, _ = choose_alpha(inner_x, iz, val_x, vz, random_seed=random_seed)
    _, local, neff = local_moments(Z, tz, qz, random_seed=random_seed)
    C = np.broadcast_to(Cg, local.shape).copy() if alpha == 1 else regularize((1 - alpha) * local + alpha * Cg)
    w, t = fusion_weights(C, scale, var_y, 'full')
    return (Xt * w).sum(1), alpha, t

def part_a2():
    print('\n=== A2: informative world with EQUAL marginal variances (energy shortcut removed) ===')
    rows = []
    for seed in SEEDS:
        X, y, act = world_equal_variance(seed, 320); Xt, yt, act_t = world_equal_variance(seed + 100000, 640)
        scale = X.std(0); m = 6; Z = (X - X.mean(0)) / scale; Cg = regularize(covariance(Z)); var_y = .25 * np.trace(Cg) / m
        H = history_equal_variance(act, seed + 200000); Ht = history_equal_variance(act_t, seed + 300000)
        out = {}
        w, _ = fusion_weights(Cg, scale, var_y, 'full'); out['static'] = (Xt * w).sum(1)
        inner_mean, inner_scale = X[:240].mean(0), X[:240].std(0)
        inner_x = (X[:240] - inner_mean) / inner_scale; val_x = (X[240:] - inner_mean) / inner_scale
        # energy context
        e, et = (H ** 2).mean(1), (Ht ** 2).mean(1); sd = e.std(0); sd[sd < 1e-8] = 1
        tz, qz = (e - e.mean(0)) / sd, (et - e.mean(0)) / sd
        out['energy_context'], a_e, _ = context_arm(Z, Cg, scale, var_y, tz, qz, inner_x, tz[:240], val_x, tz[240:], Xt)
        # regime (oracle) context
        tz, qz = act[:, :1].astype(float), act_t[:, :1].astype(float)
        out['regime_context'], a_o, _ = context_arm(Z, Cg, scale, var_y, tz, qz, inner_x, tz[:240], val_x, tz[240:], Xt)
        # random neighbours
        out['random_context'], a_r, _ = context_arm(Z, Cg, scale, var_y, tz, qz, inner_x, tz[:240], val_x, tz[240:], Xt, random_seed=seed + 400000)
        # sanity: marginal variances of the two blocks in test data
        rows.append({k: roc_auc_score(yt, v) for k, v in out.items()} | {'alpha_energy': a_e, 'alpha_regime': a_o,
                     'var_active': float(Xt[act_t].var()), 'var_inactive': float(Xt[~act_t].var())})
    for k in rows[0]: print(f'{k:18s} {np.mean([r[k] for r in rows]):.4f}')

# ------------------------------------------------------------------ B
A = np.array([1, 1, 1, .3, .3, .3]); B_SLOW = np.array([.3, .3, .3, 1.6, 1.6, 1.6])

def temporal_world(seed, n_answers, T=64, phi=.97, b=B_SLOW, noise=.45, fast=False):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n_answers, T)); labels = (y + rng.normal(scale=.15, size=y.shape) > 0).astype(int)
    if fast: nuis = rng.normal(size=(n_answers, T))
    else:
        nuis = np.empty((n_answers, T)); nuis[:, 0] = rng.normal(size=n_answers)
        for t in range(1, T): nuis[:, t] = phi * nuis[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=n_answers)
    X = A * y[..., None] + b * nuis[..., None] + noise * rng.normal(size=(n_answers, T, 6))
    return X, y, labels

def windows(X, L=16):
    n, T, m = X.shape
    H = np.stack([X[:, t - L:t].reshape(n, -1) for t in range(L, T)], 1)   # n, T-L, L*m
    return H, X[:, L:]

def ridge_fit(H, Y, lam=1.):
    Hf, Yf = H.reshape(-1, H.shape[-1]), Y.reshape(-1, Y.shape[-1])
    mu, sd = Hf.mean(0), np.maximum(Hf.std(0), 1e-8); D = np.column_stack(((Hf - mu) / sd, np.ones(len(Hf))))
    P = np.eye(D.shape[1]) * lam; P[-1, -1] = 0
    W = np.linalg.solve(D.T @ D + P, D.T @ Yf)
    return lambda Hq: (np.column_stack(((Hq.reshape(-1, Hq.shape[-1]) - mu) / sd, np.ones(Hq.reshape(-1, Hq.shape[-1]).shape[0]))) @ W).reshape(Hq.shape[0], Hq.shape[1], -1)

def arms_for(train_x, test_x, y_train, labels_test):
    """train_x/test_x: (n,T',6) token matrices in some representation; y_train: latent for oracle rho only."""
    Xf = train_x.reshape(-1, 6); scale = Xf.std(0); mean = Xf.mean(0); m = 6
    Z = (Xf - mean) / scale; Cg = regularize(covariance(Z)); var_y = .25 * np.trace(Cg) / m
    Zt = (test_x.reshape(-1, 6) - mean) / scale; Xt = test_x.reshape(-1, 6)
    out = {}
    out['equal'] = Zt.mean(1)
    w, t = fusion_weights(Cg, scale, var_y, 'full'); out['simplex_eta.25'] = (Xt * w).sum(1)
    out['simplex_eta1'] = (Xt * t['raw_qp']).sum(1)
    kw = dict(DEFAULT_IU_FIT); kw['var_y'] = var_y
    out['native_iu'] = Zt @ upcr_fit_covariance(Cg, **kw).w
    rho_true = np.array([np.cov(Z[:, j], y_train.ravel())[0, 1] for j in range(6)]) / scale.mean() * 1  # standardized cov with unit-var y
    w, _ = fusion_weights(Cg, scale, var_y, 'full', rho_override=rho_true); out['oracle_rho_simplex'] = (Xt * w).sum(1)
    return {k: roc_auc_score(labels_test.ravel(), v) for k, v in out.items()}, t['rho'][0], rho_true

def part_b():
    print('\n=== B: temporal world, shared nuisance mostly on the weak block; IU on levels vs on innovations ===')
    for name, kw in (('slow nuisance phi=.97', dict()), ('fast (iid) nuisance', dict(fast=True)), ('no nuisance', dict(b=np.zeros(6)))):
        rows = []
        for seed in SEEDS:
            X, y, lab = temporal_world(seed, 200, **kw); Xt, yt, labt = temporal_world(seed + 100000, 200, **kw)
            H, Y = windows(X); Ht, Yt = windows(Xt)
            pred = ridge_fit(H, Y); R, Rt = Y - pred(H), Yt - pred(Ht)
            lev, rho_lev, rt_lev = arms_for(Y, Yt, y[:, 16:], labt[:, 16:])
            inn, rho_inn, rt_inn = arms_for(R, Rt, y[:, 16:], labt[:, 16:])
            rows.append({'levels_' + k: v for k, v in lev.items()} | {'innov_' + k: v for k, v in inn.items()}
                        | {'rho_lev_cos': rho_lev @ rt_lev / np.linalg.norm(rho_lev) / np.linalg.norm(rt_lev),
                           'rho_inn_cos': rho_inn @ rt_inn / np.linalg.norm(rho_inn) / np.linalg.norm(rt_inn)})
        print(f'-- {name}')
        for k in rows[0]: print(f'   {k:26s} {np.mean([r[k] for r in rows]):.4f}')

if __name__ == '__main__':
    part_a1(); part_a2(); part_b()
