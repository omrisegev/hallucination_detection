"""Ceiling check (labels used for selection only): L-SML on the channels non-negative LR keeps. 2026-10-17.

Per outer fold: NNLR on the training fold (channels standardized and sign-oriented by training-fold
mu, w >= 0, as in feature_weights.py) -> kept = channels with w > 1e-6. Then the canonical label-free
fit spectral_utils.advisor_lr.fit_labelfree(X_train[:, kept], 'LSML_CONT') scores the held-out fold.
Compared with the saved LSML_FULL / LR_FULL out-of-fold rows in lr/METRICS.csv (same folds).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/osegev/Desktop/hallucination_detection')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_utils.advisor_lr import fit_labelfree
from spectral_utils.fusion_utils import lsml_continuous
from scipy.optimize import minimize
from scipy.stats import rankdata

def within(score, off, y):
    a = []
    for b, e in zip(off[:-1], off[1:]):
        ss, yy = score[b:e], y[b:e]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0: a.append((rankdata(ss)[yy.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(a))

def nnlr(Z, y, l2=1e-4):
    n, m = Z.shape
    def f(p):
        w, b = p[:m], p[m]; s = Z @ w + b
        g = 1 / (1 + np.exp(-s)) - y
        return (np.logaddexp(0, s).sum() - y @ s) / n + l2 * w @ w, np.r_[Z.T @ g / n + 2 * l2 * w, g.mean()]
    return minimize(f, np.zeros(m + 1), jac=True, method='L-BFGS-B',
                    bounds=[(0, None)] * m + [(None, None)], options={'maxiter': 2000}).x[:m]

BASE = ROOT / 'results/advisor_completion_2026-10-11'
OUT = Path(__file__).resolve().parent
p = np.load(BASE / 'inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off))
saved = {(r['bank'], r['arm']): float(r['within_answer_auc']) for r in csv.DictReader((BASE / 'lr/METRICS.csv').open())}
res = {}
for bank in sys.argv[1:] or ['N14', 'N16', 'N25', 'N43', 'N56']:
    X = np.load(BASE / f'inputs/bank_V_{bank}.npz')['V'].astype(float); m = X.shape[1]
    score = np.full(len(y), np.nan); kept_n = []; K = []
    for o in range(5):
        tr, te = sf != o, sf == o
        mu0, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        Z = (X[tr] - mu0) / sd
        sg = np.where(Z[y[tr] == 1].mean(0) - Z[y[tr] == 0].mean(0) >= 0, 1., -1.)
        kept = np.flatnonzero(nnlr(Z * sg, y[tr]) > 1e-6); kept_n.append(len(kept))
        try:
            model = fit_labelfree(X[tr][:, kept], 'LSML_CONT'); K.append(model.diagnostic['K'])
            score[te] = model.predict(X[te][:, kept])
        except RuntimeError:
            # Canonical fit refuses K<3; use the same lsml_continuous weights anyway and flag the fold.
            _, meta = lsml_continuous(*[X[tr][:, j] for j in kept])
            w = np.zeros(len(kept))
            for cross, (idx, wi) in zip(meta['cross_weights'], meta['group_weights']):
                w[np.asarray(idx, int)] = float(cross) * np.asarray(wi)
            K.append(f"{int(meta['K'])}*"); score[te] = X[te][:, kept] @ w
    res[bank] = {'channels': m, 'kept_mean': float(np.mean(kept_n)), 'lsml_K': K,
                 'LSML_on_NNLR_set': round(within(score, off, y), 4),
                 'LSML_FULL_saved': round(saved[(bank, 'LSML_FULL')], 4),
                 'LSML__LR_HALF_saved': round(saved[(bank, 'LSML__LR_HALF')], 4),
                 'LR_FULL_saved': round(saved[(bank, 'LR_FULL')], 4)}
    print(bank, res[bank], flush=True)
(OUT / 'lsml_on_nnlr_set.json').write_text(json.dumps(res, indent=1))
