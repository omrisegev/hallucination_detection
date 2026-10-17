"""Intra-cluster ZCA whitening before SML (Omri's idea, 2026-10-16).
Per cluster g: Z_g -> Z_g W_g, W_g = (S_g + eps I)^-1/2 (ZCA). Inside a whitened cluster all
off-diagonal covariances are 0, so the within-cluster SML step has nothing to estimate from.
  A  ZCA + lsml_continuous with the same clusters supplied (within-cluster SML on whitened data)
  B  ZCA, then ONE SML over all whitened features (loadings from cross-cluster covariance only),
     mapped back: w = W psi_z
  C  oracle ceiling: psi_z = true Cov(z, y) -> w = blockdiag(S_g)^-1 mu  (= within-cluster conditioning)
Train-fold fit, OOF scoring, within-answer AUC. Also: share of negative weights vs LR."""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection')
from spectral_utils.fusion_utils import lsml_continuous
from decomp import within, ROOT, lr
from xg import sml_w

def zca_blocks(Ztr, G, eps):
    m = Ztr.shape[1]; W = np.zeros((m, m))
    for g in np.unique(G):
        i = np.flatnonzero(G == g)
        S = np.atleast_2d(np.cov(Ztr[:, i].T)) + eps * np.eye(len(i))
        ev, V = np.linalg.eigh(S)
        W[np.ix_(i, i)] = V @ np.diag(ev ** -0.5) @ V.T
    return W

p = np.load(f'{ROOT}/inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off))
for bank in sys.argv[1:]:
    X = np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); sc = {}; neg = {}
    for o in range(5):
        L = json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G = np.asarray(L['diagnostic']['group']); wl = np.asarray(L['weight']); tr, te = sf != o, sf == o
        mu0, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        Ztr, Zte = (X[tr] - mu0) / sd, (X[te] - mu0) / sd
        ref = Ztr @ (wl * sd)
        mu = Ztr[y[tr] == 1].mean(0) - Ztr[y[tr] == 0].mean(0)
        _, wlr, _ = lr(X[tr], y[tr], X[te]); neg.setdefault('LR', []).append(int(((wlr * sd) < 0).sum()))
        for eps in (0.01, 0.1, 1.0):
            W = zca_blocks(Ztr, G, eps); Wtr = Ztr @ W
            # A: L-SML with clusters supplied, on whitened features
            try:
                _, meta = lsml_continuous(*[Wtr[:, j] for j in range(Wtr.shape[1])], groups=G, compute_score_matrix=False)
                wz = np.zeros(len(G))
                for cw, (idx, wg) in zip(meta['cross_weights'], meta['group_weights']): wz[np.asarray(idx)] = cw * np.asarray(wg)
                w = W @ wz; w = w if np.corrcoef(Ztr @ w, ref)[0, 1] > 0 else -w
                sc.setdefault(f'A_lsml_on_zca_eps{eps}', np.full(len(y), np.nan))[te] = Zte @ w
            except Exception as e:
                print('A failed', e)
            # B: single SML over all whitened features
            Ws = (Wtr - Wtr.mean(0)) / Wtr.std(0)
            v = sml_w(Ws); w = W @ (v / Wtr.std(0)); w = w if np.corrcoef(Ztr @ w, ref)[0, 1] > 0 else -w
            sc.setdefault(f'B_sml_all_whitened_eps{eps}', np.full(len(y), np.nan))[te] = Zte @ w
            neg.setdefault(f'B_eps{eps}', []).append(int((w < 0).sum()))
            # C: oracle loadings in whitened space
            wc = W @ (W.T @ mu)
            sc.setdefault(f'C_oracle_eps{eps}', np.full(len(y), np.nan))[te] = Zte @ wc
            neg.setdefault(f'C_eps{eps}', []).append(int((wc < 0).sum()))
    print(bank, 'm=', X.shape[1], {k: round(within(v, off, y), 4) for k, v in sc.items()},
          'neg weights (mean over folds):', {k: round(float(np.mean(v)), 1) for k, v in neg.items()}, flush=True)
