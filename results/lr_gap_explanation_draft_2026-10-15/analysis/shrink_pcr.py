"""Omri's two follow-ups (2026-10-16), train-fold fit, OOF within-answer AUC.
OPT1  shrinkage whitening inside each cluster: W_g = ((1-a) S_g + a I)^-1/2, then L-SML with the
      same clusters supplied, weights mapped back to original space. a=1 is plain L-SML.
OPT2  inside each cluster: PCA+ridge weights w_g = sum_{c<=k} (v_c.rho_g)/(lam_c+lam) v_c, then the
      usual SML across cluster scores (sml_fuse_signed). rho_g sources:
        XC  = Cov(Z_g, s_-g), s_-g = L-SML score without cluster g (label-free, cross-cluster)
        IU  = IU-PCR's own label-free rho_hat inside the cluster (as Step 293, no filtering)
        OR  = true mu_g (oracle ceiling for this pipeline)"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection')
from spectral_utils.fusion_utils import lsml_continuous, sml_fuse_signed
from spectral_utils.upcr import upcr_fit
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from decomp import within, ROOT

def mapback(meta, n):
    wz = np.zeros(n)
    for cw, (idx, wg) in zip(meta['cross_weights'], meta['group_weights']): wz[np.asarray(idx)] = cw * np.asarray(wg)
    return wz

def orient(w, Z, ref): return w if np.corrcoef(Z @ w, ref)[0, 1] > 0 else -w

p = np.load(f'{ROOT}/inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off))
for bank in sys.argv[1:]:
    X = np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); m = X.shape[1]; sc = {}; neg = {}
    def put(k, te, s, w=None):
        sc.setdefault(k, np.full(len(y), np.nan))[te] = s
        if w is not None: neg.setdefault(k, []).append(int((w < 0).sum()))
    for o in range(5):
        L = json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G = np.asarray(L['diagnostic']['group']); wl = np.asarray(L['weight']); tr, te = sf != o, sf == o
        mu0, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        Ztr, Zte = (X[tr] - mu0) / sd, (X[te] - mu0) / sd
        wl_s = wl * sd; ref = Ztr @ wl_s
        mu = Ztr[y[tr] == 1].mean(0) - Ztr[y[tr] == 0].mean(0)
        clusters = [np.flatnonzero(G == g) for g in np.unique(G)]
        # ---- OPT1
        for a in (0.5, 0.75, 0.9, 0.95, 1.0):
            W = np.zeros((m, m))
            for i in clusters:
                S = (1 - a) * np.atleast_2d(np.cov(Ztr[:, i].T)) + a * np.eye(len(i))
                ev, V = np.linalg.eigh(S); W[np.ix_(i, i)] = V @ np.diag(ev ** -0.5) @ V.T
            Wtr = Ztr @ W
            _, meta = lsml_continuous(*[Wtr[:, j] for j in range(m)], groups=G, compute_score_matrix=False)
            w = orient(W @ mapback(meta, m), Ztr, ref)
            put(f'OPT1_shrink_a{a}', te, Zte @ w, w)
        # ---- OPT2
        for src in ('XC', 'IU', 'OR'):
            for k in (1, 2, 'all'):
                for lam in (0.1, 1.0):
                    if src == 'IU' and (k != 2 or lam != 0.1): continue  # IU-PCR has its own fixed k=2 / scaling
                    W = np.zeros((len(clusters), m))
                    for a_, i in enumerate(clusters):
                        Zg = Ztr[:, i]
                        if len(i) == 1: W[a_, i] = 1.; continue
                        if src == 'IU':
                            if len(i) < 3:
                                _, wg = sml_fuse_signed(*[Zg[:, j] for j in range(len(i))]); W[a_, i] = wg; continue
                            W[a_, i] = upcr_fit(Zg.T, **IU_FIT_DEFAULTS).w; continue
                        if src == 'XC':
                            inst = ref - Zg @ wl_s[i]; inst = (inst - inst.mean()) / inst.std()
                            rho = Zg.T @ inst / len(inst)
                        else:
                            rho = mu[i]
                        ev, V = np.linalg.eigh(np.atleast_2d(np.cov(Zg.T))); ev, V = ev[::-1], V[:, ::-1]
                        kk = len(i) if k == 'all' else min(k, len(i))
                        W[a_, i] = sum((V[:, c] @ rho) / (ev[c] + lam) * V[:, c] for c in range(kk))
                    xi = Ztr @ W.T; mxi, sxi = xi.mean(0), xi.std(0); sxi[sxi == 0] = 1
                    _, cw = sml_fuse_signed(*[((xi - mxi) / sxi)[:, j] for j in range(xi.shape[1])])
                    w = orient(W.T @ (cw / sxi), Ztr, ref)
                    key = f'OPT2_{src}' + ('' if src == 'IU' else f'_k{k}_lam{lam}')
                    put(key, te, Zte @ w, w)
    res = {k: round(within(v, off, y), 4) for k, v in sc.items()}
    print(bank, json.dumps(res), 'neg:', json.dumps({k: round(float(np.mean(v)), 1) for k, v in neg.items()}), flush=True)
