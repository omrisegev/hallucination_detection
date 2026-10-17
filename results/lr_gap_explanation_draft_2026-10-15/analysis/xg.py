"""Label-free cross-group instrumented Fisher (XGF) on the L-SML groups.

For group g: w_g = (S_g + lam I)^-1 Cov(Z_g, s_-g), s_-g = L-SML score without group g.
If groups are conditionally independent given y, Cov(Z_g, s_-g) ∝ Cov(Z_g, y), so w_g
∝ the within-group Fisher/LR direction -> signed contrasts with no labels.
Group scores xi_g are z-scored and fused across groups (equal sum, or SML rank-1).
Fit on the training fold (label-free; labels never read in fitting), scored OOF.
"""
import json, sys
import numpy as np
from numpy.linalg import eigh, solve
from decomp import within, ROOT


def sml_w(Xi):
    R = np.cov(Xi.T); R = R - np.diag(np.diag(R))
    v = eigh(R)[1][:, -1]
    return v if (v > 0).sum() >= len(v) / 2 else -v


def xgf(Xtr, Xte, groups, wl, lam, iters=1):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    s = Xtr @ wl  # L-SML training score (oriented)
    G = np.unique(groups)
    contrib = np.column_stack([Xtr[:, groups == g] @ wl[groups == g] for g in G])
    for it in range(iters):
        W = np.zeros((len(G), Xtr.shape[1]))
        for a, g in enumerate(G):
            idx = groups == g
            inst = s - contrib[:, a]
            inst = (inst - inst.mean()) / inst.std()
            Zg = Ztr[:, idx]
            S = np.atleast_2d(np.cov(Zg.T)) + lam * np.eye(idx.sum())
            W[a, idx] = solve(S, Zg.T @ inst / len(inst))
        xi_tr = Ztr @ W.T
        m, sdev = xi_tr.mean(0), xi_tr.std(0)
        contrib = (xi_tr - m) / sdev
        s = contrib.sum(1)
    xi_te = (Zte @ W.T - m) / sdev
    return {'sum': xi_te.sum(1), 'sml': xi_te @ sml_w(contrib)}


def main(out, banks, lams=(0.01, 0.1, 1.0)):
    p = np.load(f'{ROOT}/inputs/population.npz')
    off = p['offsets'].astype(int); y = p['y'].astype(int)
    sf = np.repeat(p['fold'].astype(int), np.diff(off))
    rows = []
    for bank in banks:
        X = np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float)
        sc = {}
        for o in range(5):
            L = json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
            groups = np.asarray(L['diagnostic']['group']); wl = np.asarray(L['weight'])
            tr, te = sf != o, sf == o
            for lam in lams:
                for it in (1, 2):
                    r = xgf(X[tr], X[te], groups, wl, lam, it)
                    for k, v in r.items():
                        sc.setdefault(f'XGF_{k}_lam{lam}_it{it}', np.full(len(y), np.nan))[te] = v
        row = {'bank': bank, **{k: round(within(v, off, y), 4) for k, v in sc.items()}}
        rows.append(row); print(row, flush=True)
    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
