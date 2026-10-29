"""Checks of the binned-prior model on data drawn from the model itself (no real data):
(1) E-step P(has-error) and q against brute-force enumeration of every y configuration on tiny answers;
(2) recovery of rho, the 10 bin probabilities, psi/phi/phi0 and calibration, no likelihood drops."""
import itertools
import numpy as np
from scipy.special import expit
from ds_gate_b import fit_hds, e_step, relative_position, bin_design

PI = np.array([.05, .06, .08, .1, .12, .15, .2, .25, .3, .35])


def simulate(n_ans=4000, rho=.65, pi=PI, psi=(.75, .7, .6, .8, .65), phi=(.08, .1, .12, .06, .1), phi0=(.15, .2, .18, .12, .16),
             seed=1, lo=3, hi=16):
    rng = np.random.default_rng(seed); psi, phi, phi0 = map(np.array, (psi, phi, phi0))
    lens = rng.integers(lo, hi, n_ans); off = np.r_[0, np.cumsum(lens)]; D = bin_design(relative_position(off))
    pstep = D @ pi
    z = rng.random(n_ans) < rho; y = np.zeros(off[-1], bool)
    for j in np.flatnonzero(z):
        sl = slice(off[j], off[j + 1])
        while True:
            yy = rng.random(lens[j]) < pstep[sl]
            if yy.any(): break
        y[sl] = yy
    Z = np.repeat(z, lens)
    P = np.where(y[:, None], psi, np.where(Z[:, None], phi, phi0))
    G = (rng.random(P.shape) < P).astype(int)
    return G, off, D, z, y, dict(rho=rho, pi=pi, psi=psi, phi=phi, phi0=phi0)


def check(**kw):
    G, off, D, z, y, truth = simulate(**kw)
    fit = fit_hds(G, off, D); pr = fit['params']; w = fit['p_answer']
    idx = np.empty(len(w), int); idx[np.argsort(w, kind='stable')] = np.arange(len(w)) * 10 // len(w)
    ece = sum(abs(w[idx == k].mean() - z[idx == k].mean()) * (idx == k).mean() for k in range(10))
    pib = np.array(pr['pi_bins'])
    print('rho', round(pr['rho'], 3), 'truth', truth['rho'])
    print('pi_bins', np.round(pib, 3)); print('truth  ', truth['pi'])
    for k in ('psi', 'phi', 'phi0'): print(k, np.round(pr[k], 3), 'truth', truth[k])
    print('ECE', round(ece, 4), 'mean q', round(fit['q'].mean(), 4), 'true step rate', round(y.mean(), 4), 'drops', fit['ll_drops'])
    ok = (fit['ll_drops'] == 0 and abs(pr['rho'] - truth['rho']) < .02 and np.abs(pib - truth['pi']).max() < .04 and
          max(np.abs(np.array(pr[k]) - truth[k]).max() for k in ('psi', 'phi', 'phi0')) < .05 and ece < .03)
    print('SYNTHETIC RECOVERY', kw, 'PASS' if ok else 'FAIL'); return ok


def brute_force_check(seed=3):
    rng = np.random.default_rng(seed); K = 3
    lens = np.array([2, 3, 4, 1, 5]); off = np.r_[0, np.cumsum(lens)]; D = bin_design(relative_position(off))
    G = rng.integers(0, 2, (off[-1], K)); theta = rng.normal(-1, .7, 10)
    psi, phi, phi0 = rng.uniform(.5, .9, K), rng.uniform(.05, .3, K), rng.uniform(.1, .3, K); rho = .6
    w, q, W, ll = e_step(G, off, D, (rho, theta, psi, phi, phi0)); pi = expit(D @ theta)
    lik = lambda g, p: np.prod(np.where(g == 1, p, 1 - p))
    err, llb = 0., 0.
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        tot1 = 0.; qn = np.zeros(b - a); z0 = np.prod(1 - pi[a:b])
        for ys in itertools.product([0, 1], repeat=b - a):
            if not any(ys): continue
            pr = np.prod([pi[a + i] if v else 1 - pi[a + i] for i, v in enumerate(ys)]) / (1 - z0)
            l = pr * np.prod([lik(G[a + i], psi if v else phi) for i, v in enumerate(ys)])
            tot1 += l; qn += l * np.array(ys)
        tot0 = np.prod([lik(G[a + i], phi0) for i in range(b - a)])
        den = rho * tot1 + (1 - rho) * tot0; llb += np.log(den)
        err = max(err, abs(rho * tot1 / den - w[j]), np.abs(q[a:b] - rho * qn / den).max())
    err = max(err, abs(llb - ll))
    print('brute-force max abs error (P has-error, q, loglik)', err); return err < 1e-9


if __name__ == '__main__':
    ok = brute_force_check() and check() and check(seed=2, n_ans=6000) and check(seed=4, n_ans=6000, lo=2, hi=30)
    raise SystemExit(0 if ok else 1)
