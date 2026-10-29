"""Checks of the position-dependent-reliability model on data drawn from the model itself (no real data):
(1) E-step P(has-error), q and log-likelihood against brute-force enumeration of every y configuration on tiny answers;
(2) recovery of rho, the 10 prior bins, psi/phi/phi0 per position third, calibration, and no likelihood drops;
(3) the model reduces to ds_gate_b when the three thirds share their emissions (E-step identity)."""
import itertools, sys
from pathlib import Path
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_gate_e2 import fit_e2, e_step, third, B  # noqa: E402

PI = np.array([.05, .06, .08, .1, .12, .15, .2, .25, .3, .35])
# reliability that changes with position: groups 0-1 sharp early, groups 3-4 sharp late
PSI = np.array([[.85, .8, .6, .55, .5], [.7, .7, .65, .65, .65], [.5, .55, .6, .8, .85]])
PHI = np.array([[.06, .08, .12, .15, .15], [.1, .1, .1, .1, .1], [.15, .15, .12, .08, .06]])
PHI0 = np.array([[.12, .15, .18, .2, .2], [.15, .15, .15, .15, .15], [.2, .2, .18, .14, .12]])


def simulate(n_ans=6000, rho=.65, seed=1, lo=3, hi=16):
    rng = np.random.default_rng(seed)
    lens = rng.integers(lo, hi, n_ans); off = np.r_[0, np.cumsum(lens)]
    pos = B.relative_position(off); D = B.bin_design(pos); t = third(pos); pstep = D @ PI
    z = rng.random(n_ans) < rho; y = np.zeros(off[-1], bool)
    for j in np.flatnonzero(z):
        sl = slice(off[j], off[j + 1])
        while True:
            yy = rng.random(lens[j]) < pstep[sl]
            if yy.any(): break
        y[sl] = yy
    Z = np.repeat(z, lens)
    P = np.where(y[:, None], PSI[t], np.where(Z[:, None], PHI[t], PHI0[t]))
    G = (rng.random(P.shape) < P).astype(int)
    return G, off, D, t, z, y, rho


def check(**kw):
    G, off, D, t, z, y, rho = simulate(**kw)
    fit = fit_e2(G, off, D, t); pr = fit['params']; w = fit['p_answer']
    idx = np.empty(len(w), int); idx[np.argsort(w, kind='stable')] = np.arange(len(w)) * 10 // len(w)
    ece = sum(abs(w[idx == k].mean() - z[idx == k].mean()) * (idx == k).mean() for k in range(10))
    pib = np.array(pr['pi_bins'])
    err = {k: float(np.abs(np.array(pr[k]) - v).max()) for k, v in (('psi', PSI), ('phi', PHI), ('phi0', PHI0))}
    print('rho', round(pr['rho'], 3), 'truth', rho, '| pi max err', round(float(np.abs(pib - PI).max()), 3),
          '| emission max err', {k: round(v, 3) for k, v in err.items()}, '| ECE', round(ece, 4), '| drops', fit['ll_drops'])
    ok = (fit['ll_drops'] == 0 and abs(pr['rho'] - rho) < .02 and np.abs(pib - PI).max() < .05 and max(err.values()) < .06 and ece < .03)
    print('SYNTHETIC RECOVERY', kw, 'PASS' if ok else 'FAIL'); return ok


def brute_force_check(seed=3):
    rng = np.random.default_rng(seed); K = 3
    lens = np.array([2, 3, 4, 1, 5, 6]); off = np.r_[0, np.cumsum(lens)]
    pos = B.relative_position(off); D = B.bin_design(pos); t = third(pos)
    G = rng.integers(0, 2, (off[-1], K)); theta = rng.normal(-1, .7, 10)
    psi, phi, phi0 = rng.uniform(.5, .9, (3, K)), rng.uniform(.05, .3, (3, K)), rng.uniform(.1, .3, (3, K)); rho = .6
    w, q, W, ll = e_step(G, off, D, t, (rho, theta, psi, phi, phi0)); pi = expit(D @ theta)
    lik = lambda g, p: np.prod(np.where(g == 1, p, 1 - p))
    err, llb = 0., 0.
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        tot1 = 0.; qn = np.zeros(b - a); z0 = np.prod(1 - pi[a:b])
        for ys in itertools.product([0, 1], repeat=b - a):
            if not any(ys): continue
            pr = np.prod([pi[a + i] if v else 1 - pi[a + i] for i, v in enumerate(ys)]) / (1 - z0)
            l = pr * np.prod([lik(G[a + i], psi[t[a + i]] if v else phi[t[a + i]]) for i, v in enumerate(ys)])
            tot1 += l; qn += l * np.array(ys)
        tot0 = np.prod([lik(G[a + i], phi0[t[a + i]]) for i in range(b - a)])
        den = rho * tot1 + (1 - rho) * tot0; llb += np.log(den)
        err = max(err, abs(rho * tot1 / den - w[j]), np.abs(q[a:b] - rho * qn / den).max())
    err = max(err, abs(llb - ll))
    # (3) shared emissions -> identical to ds_gate_b's E-step
    s = (rho, theta, psi[0], phi[0], phi0[0]); w0, q0, _, ll0 = B.e_step(G, off, D, s)
    w1, q1, _, ll1 = e_step(G, off, D, t, (rho, theta, np.tile(psi[0], (3, 1)), np.tile(phi[0], (3, 1)), np.tile(phi0[0], (3, 1))))
    red = max(np.abs(w0 - w1).max(), np.abs(q0 - q1).max(), abs(ll0 - ll1))
    print('brute-force max abs error', err, '| reduction to ds_gate_b', red); return err < 1e-9 and red < 1e-9


if __name__ == '__main__':
    ok = brute_force_check() and check() and check(seed=2, n_ans=8000) and check(seed=4, n_ans=8000, lo=2, hi=30)
    raise SystemExit(0 if ok else 1)
