"""Does the L-SML clustering contain the subspace LR uses? Outer-fold OOF decomposition.

Per outer fold, using the TRAINING-fold L-SML groups saved in the advisor v11 checkpoints:
  LSML        : saved L-SML affine model (label-free reference)
  LR_FULL     : LR on all channels
  LR_LVL      : LR on K group-level components (leading eigvec per group) -> cross-group weights only
  LR_PC2/PC3  : LR on top-2 / top-3 within-group PCs per group
  LR_SPLIT_*  : full-LR score split into within-group-v1 part and within-group-orthogonal part
"""
import json, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

ROOT = '/Users/osegev/Desktop/hallucination_detection/results/advisor_completion_2026-10-11'
C = 10.0


def within(score, off, y):
    a = []
    for b, e in zip(off[:-1], off[1:]):
        ss, yy = score[b:e], y[b:e]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0:
            a.append((rankdata(ss)[yy.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(a))


def lr(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    m = LogisticRegression(C=C, max_iter=5000).fit((Xtr - mu) / sd, ytr)
    return m.decision_function((Xte - mu) / sd), m.coef_[0] / sd, mu


def group_pcs(Xtr, groups, k):
    """Per group: top-k eigvecs of the training correlation matrix (in raw-column coords)."""
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Z = (Xtr - mu) / sd
    blocks = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        R = np.atleast_2d(np.cov(Z[:, idx].T))
        ev, V = np.linalg.eigh(R); V = V[:, ::-1]
        for j in range(min(k, len(idx))):
            v = np.zeros(Xtr.shape[1]); v[idx] = V[:, j] / sd[idx]
            blocks.append((g, j, v))
    return blocks, mu


def main(banks):
    p = np.load(f'{ROOT}/inputs/population.npz')
    off = p['offsets'].astype(int); y = p['y'].astype(int)
    sf = np.repeat(p['fold'].astype(int), np.diff(off))
    rows = []
    for bank in banks:
        X = np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float)
        m = X.shape[1]
        sc = {k: np.full(len(y), np.nan) for k in
              ['LSML', 'LR_FULL', 'LR_LVL', 'LR_PC2', 'LR_PC3', 'SPLIT_V1', 'SPLIT_ORTH']}
        share = []
        for o in range(5):
            d = json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))
            L = d['selector_models']['LSML_CONT']
            groups = np.asarray(L['diagnostic']['group'])
            tr, te = sf != o, sf == o
            sc['LSML'][te] = X[te] @ np.asarray(L['weight']) + L['intercept']
            s, w, _ = lr(X[tr], y[tr], X[te]); sc['LR_FULL'][te] = s
            for name, k in [('LR_LVL', 1), ('LR_PC2', 2), ('LR_PC3', 3)]:
                blocks, mu = group_pcs(X[tr], groups, k)
                P = np.column_stack([b[2] for b in blocks])
                sc[name][te] = lr((X[tr] - mu) @ P, y[tr], (X[te] - mu) @ P)[0]
            # split full-LR weight into per-group v1 projection (in standardized metric) + orthogonal rest
            blocks, mu = group_pcs(X[tr], groups, 1)
            sd = X[tr].std(0); sd[sd == 0] = 1
            ws = w * sd  # standardized weights
            w1 = np.zeros(m)
            for g, _, v in blocks:
                vs = v * sd  # unit eigvec in standardized coords
                w1 += (ws @ vs) * vs
            wo = ws - w1
            Zte = (X[te] - X[tr].mean(0)) / sd
            sc['SPLIT_V1'][te] = Zte @ w1; sc['SPLIT_ORTH'][te] = Zte @ wo
            Ztr = (X[tr] - X[tr].mean(0)) / sd
            v1, vo = np.var(Ztr @ w1), np.var(Ztr @ wo)
            share.append(vo / (v1 + vo))
        r = {'bank': bank, 'm': m, 'K': int(L['diagnostic']['K'])}
        r.update({k: round(within(v, off, y), 4) for k, v in sc.items()})
        r['orth_var_share'] = round(float(np.mean(share)), 3)
        rows.append(r); print(r, flush=True)
    json.dump(rows, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[2:])
