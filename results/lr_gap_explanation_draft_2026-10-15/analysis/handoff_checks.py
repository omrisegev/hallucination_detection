"""Check two claims of docs/HANDOFF_within_group_fusion_2026-10-06.md on the N-banks (2026-10-16).
1) Non-negative LR (channels sign-oriented by training-fold mu) vs unconstrained LR: are negative weights needed?
2) Within-cluster conditioning (oracle block LDA) with L-SML clusters vs random clusters of the same sizes."""
import json, sys
import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize
from decomp import within, ROOT, lr

def nnlr(Z, y, l2=1e-4):
    n, m = Z.shape
    def f(p):
        w, b = p[:m], p[m]; s = Z @ w + b
        loss = np.logaddexp(0, s).sum() - y @ s
        g = 1 / (1 + np.exp(-s)) - y
        return loss / n + l2 * w @ w, np.r_[Z.T @ g / n + 2 * l2 * w, g.mean()]
    r = minimize(f, np.zeros(m + 1), jac=True, method='L-BFGS-B', bounds=[(0, None)] * m + [(None, None)],
                 options={'maxiter': 2000})
    return r.x[:m]

p = np.load(f'{ROOT}/inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off)); rng = np.random.default_rng(20261016)
for bank in sys.argv[1:]:
    X = np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); m = X.shape[1]
    sc = {k: np.full(len(y), np.nan) for k in ['LR', 'NNLR', 'BLOCK_LSML', 'BLOCK_RANDOM_mean']}
    rnd = np.zeros((20, len(y))); rnd[:] = np.nan; nz = []
    for o in range(5):
        L = json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G = np.asarray(L['diagnostic']['group']); tr, te = sf != o, sf == o
        mu0, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        Ztr, Zte = (X[tr] - mu0) / sd, (X[te] - mu0) / sd
        mu = Ztr[y[tr] == 1].mean(0) - Ztr[y[tr] == 0].mean(0); sg = np.where(mu >= 0, 1., -1.)
        sc['LR'][te] = lr(X[tr], y[tr], X[te])[0]
        w = nnlr(Ztr * sg, y[tr]); sc['NNLR'][te] = (Zte * sg) @ w; nz.append(int((w > 1e-6).sum()))
        S = np.cov(Ztr.T)
        def block(Gx):
            B = np.where(Gx[:, None] == Gx[None, :], S, 0.); return Zte @ solve(B + 0.1 * np.eye(m), mu)
        sc['BLOCK_LSML'][te] = block(G)
        for r in range(20): rnd[r, te] = block(rng.permutation(G))
    res = {k: round(within(v, off, y), 4) for k, v in sc.items() if k != 'BLOCK_RANDOM_mean'}
    ra = [within(rnd[r], off, y) for r in range(20)]
    res['BLOCK_RANDOM'] = f'{np.mean(ra):.4f} (range {min(ra):.4f}-{max(ra):.4f})'
    print(bank, f'm={m}', res, 'NNLR nonzero (mean over folds):', round(float(np.mean(nz)), 1), flush=True)
