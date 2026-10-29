"""Recovery test of fit_hds on data drawn from the model itself (no real data): rho, psi, phi, phi0, the position
prior and the calibration of P(has-error) must be recovered."""
import numpy as np
from scipy.special import expit
from ds_gate import fit_hds, relative_position

def simulate(n_ans=4000, rho=.65, a=-2.5, b=2.0, psi=(.75, .7, .6, .8, .65), phi=(.08, .1, .12, .06, .1), phi0=(.15, .2, .18, .12, .16), seed=1):
    rng = np.random.default_rng(seed); psi, phi, phi0 = map(np.array, (psi, phi, phi0))
    lens = rng.integers(3, 16, n_ans); off = np.r_[0, np.cumsum(lens)]; pos = relative_position(off)
    z = rng.random(n_ans) < rho; y = np.zeros(off[-1], bool)
    for j in np.flatnonzero(z):
        sl = slice(off[j], off[j + 1]); pi = expit(a + b * pos[sl])
        while True:
            yy = rng.random(lens[j]) < pi
            if yy.any(): break
        y[sl] = yy
    Z = np.repeat(z, lens)
    P = np.where(y[:, None], psi, np.where(Z[:, None], phi, phi0))
    G = (rng.random(P.shape) < P).astype(int)
    return G, off, pos, z, y, dict(rho=rho, a=a, b=b, psi=psi, phi=phi, phi0=phi0)

def check(**kw):
    G, off, pos, z, y, truth = simulate(**kw)
    fit = fit_hds(G, off, pos)
    pr = fit['params']
    w = fit['p_answer']; bins = np.quantile(w, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(w, bins[1:-1]), 0, 9)
    ece = sum(abs(w[idx == k].mean() - z[idx == k].mean()) * (idx == k).mean() for k in range(10))
    print('rho', round(pr['rho'], 3), 'truth', truth['rho'])
    for k in ('psi', 'phi', 'phi0'):
        print(k, np.round(pr[k], 3), 'truth', truth[k])
    print('a,b', round(pr['a'], 2), round(pr['b'], 2), 'truth', truth['a'], truth['b'])
    print('ECE(P has-error)', round(ece, 4), '| mean q', round(fit['q'].mean(), 4), 'true step rate', round(y.mean(), 4))
    print('likelihood drops', fit['ll_drops'], 'worst', fit['worst_drop'])
    ok = fit['ll_drops'] == 0 and abs(pr['a'] - truth['a']) < .3 and abs(pr['b'] - truth['b']) < .3 and abs(pr['rho'] - truth['rho']) < .02 and max(np.abs(np.array(pr['psi']) - truth['psi']).max(), np.abs(np.array(pr['phi']) - truth['phi']).max(),
                                                     np.abs(np.array(pr['phi0']) - truth['phi0']).max()) < .05 and ece < .03
    print('SYNTHETIC RECOVERY', kw, 'PASS' if ok else 'FAIL'); return ok


if __name__ == '__main__':
    ok = check() and check(seed=2, n_ans=6000) 
    raise SystemExit(0 if ok else 1)
