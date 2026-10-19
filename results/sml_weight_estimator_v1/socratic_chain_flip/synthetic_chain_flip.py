"""Synthetic reproduction of the CHAIN inversion: two positively informative inputs, fused AUC < 0.5.

Answers of 8 steps; first error at tau, every step from tau on is an error (Socratic labelling).
zp = +a*pos + b*err + e ; telemetry groups g_k = -c*pos + d*err + u + e_k (u shared nuisance).
CHAIN = lsml_continuous([zp, az(g1..gK)], groups=arange) oriented by pooled corr with zp, as in ablation5_socratic.
"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from spectral_utils.fusion_utils import lsml_continuous
from scipy.stats import rankdata

rng = np.random.default_rng(0)


def az(v, o):
    out = np.empty_like(v)
    for i, j in zip(o[:-1], o[1:]):
        x = v[i:j]; out[i:j] = (x - x.mean()) / max(x.std(), 1e-8)
    return out


def wauc(s, y, o):
    r = []
    for i, j in zip(o[:-1], o[1:]):
        yy, ss = y[i:j], s[i:j]; p = yy.sum(); n = len(yy) - p
        if p and n: r.append((rankdata(ss)[yy].sum() - p * (p + 1) / 2) / (p * n))
    return float(np.mean(r))


def run(c, K=3, n=3000, T=8, a=1., b=1., d=.6, nu=.8, k=0.):
    tau = rng.integers(1, T + 1, n)  # tau == T means no error
    pos = np.tile(np.arange(T) / (T - 1), n); y = (np.arange(T)[None, :] >= tau[:, None]).ravel()
    o = np.arange(0, n * T + 1, T)
    m = rng.normal(size=n * T)  # label-free nuisance shared with opposite signs (e.g. step length)
    zp = az(a * pos + b * y + k * m + rng.normal(size=n * T), o)
    u = rng.normal(size=n * T)
    G = [az(-c * pos + d * y + nu * u - k * m + rng.normal(size=n * T), o) for _ in range(K)]
    VC = np.column_stack([zp] + G)
    s, meta = lsml_continuous(*VC.T, groups=np.arange(K + 1)); s = np.asarray(s, float)
    w = np.asarray(meta['cross_weights']); flip = np.corrcoef(s, zp)[0, 1] < 0
    so = -s if flip else s
    tele = np.mean(G, 0)
    return dict(c=c, k=k, auc_zp=wauc(zp, y, o), auc_tele=wauc(tele, y, o), corr_zp_tele=float(np.corrcoef(zp, tele)[0, 1]),
                w_sml=np.round(w, 3).tolist(), flipped=bool(flip), auc_chain=wauc(so, y, o),
                auc_chain_abs=wauc(VC @ np.abs(w), y, o), auc_half=wauc(.5 * zp + .5 * tele, y, o))


for c, k in ((0., 0.), (.5, 0.), (0., .5), (0., 1.), (.3, 1.), (.3, 1.5)):
    print(run(c, k=k))
