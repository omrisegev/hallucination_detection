"""Law-of-total-covariance check: is corr(PRM, TELE) < 0 on Socratic Math-Shepherd a within-class effect?

Cov(X, Y) = E_y[Cov(X, Y | y)]  +  Cov(E[X|y], E[Y|y])
            (conditional dependence;   (signal term: p(1-p) * dX * dY,
             0 under SML's model)       > 0 when both are informative)
X = answer-z of the PRM risk (zp), Y = TELE_N13 (risk-oriented). Steps: non-empty and included.
"""
import json, sys
from pathlib import Path
import numpy as np

gold = {g['uid']: g for g in json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))}


def az(x, off):
    out = np.zeros_like(x)
    for a, b in zip(off[:-1], off[1:]):
        v = x[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out


for f in sys.argv[2:]:
    S = dict(np.load(f)); ne = S['nonempty'].astype(bool); lens = S['nonempty_lengths']
    starts = np.r_[0, np.cumsum(lens)]
    off = np.r_[0, np.cumsum(np.add.reduceat(ne.astype(int), starts[:-1]))]
    zp = az(np.asarray(S['score__PRM'], float), off); tele = np.asarray(S['score__TELE_N13'], float)
    err = np.zeros(len(zp), bool); keep = np.zeros(len(zp), bool); pos = np.zeros(len(zp))
    for i, u in enumerate(S['uids']):
        g = gold[str(u)]; a, b = starts[i], starts[i + 1]
        idx = np.flatnonzero(ne[a:b]); r0, r1 = off[i], off[i + 1]
        corr = np.asarray(g['correct'], bool)[idx]; inc = np.asarray(g['include'], bool)[idx]
        err[r0:r1] = ~corr; keep[r0:r1] = inc
        n = r1 - r0; pos[r0:r1] = (np.arange(n) + .5) / n
    X, Y, E, P = zp[keep], tele[keep], err[keep], pos[keep]
    X = (X - X.mean()) / X.std(); Y = (Y - Y.mean()) / Y.std()
    p = E.mean()
    cov_total = np.mean(X * Y)
    within = [np.cov(X[E == c], Y[E == c], bias=True)[0, 1] for c in (True, False)]
    w_term = p * within[0] + (1 - p) * within[1]
    dX = X[E].mean() - X[~E].mean(); dY = Y[E].mean() - Y[~E].mean()
    s_term = p * (1 - p) * dX * dY
    rc = lambda a, b, m: np.corrcoef(a[m], b[m])[0, 1]
    print(f'{Path(f).stem}: steps {len(X)}, error rate {p:.3f}')
    print(f'  corr(PRM, TELE) total {cov_total:+.3f} = within-class {w_term:+.3f} + signal {s_term:+.3f}'
          f'   (dPRM {dX:+.3f}, dTELE {dY:+.3f} SD units, error minus correct)')
    print(f'  corr(PRM, TELE) among error steps {rc(X, Y, E):+.3f}, among correct steps {rc(X, Y, ~E):+.3f}')
    print(f'  corr with position: PRM {rc(X, P, E):+.3f} (err) {rc(X, P, ~E):+.3f} (ok);'
          f' TELE {rc(Y, P, E):+.3f} (err) {rc(Y, P, ~E):+.3f} (ok); error vs position {np.corrcoef(E, P)[0, 1]:+.3f}')
    # remove position (linear) from both inside each class, then the within-class correlation again
    def resid(v, m):
        A = np.column_stack([np.ones(m.sum()), P[m]]); return v[m] - A @ np.linalg.lstsq(A, v[m], rcond=None)[0]
    print(f'  within-class corr after removing position: err {np.corrcoef(resid(X, E), resid(Y, E))[0, 1]:+.3f},'
          f' ok {np.corrcoef(resid(X, ~E), resid(Y, ~E))[0, 1]:+.3f}')
