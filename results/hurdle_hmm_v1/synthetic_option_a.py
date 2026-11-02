"""Exploratory synthetic check for option A (not yet an amendment): anchor (mu0, s0) from the clean steps of m
labelled answers (sample mean and sd), adaptive quadrature, rest as v1. Compared with the naive estimate of pi from
the same m labels. N=1500 answers, lengths 4..15, seeds 0..1."""
import importlib.util, pathlib, time, numpy as np
sp = importlib.util.spec_from_file_location('hh', pathlib.Path(__file__).resolve().parents[2] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)
from scipy.optimize import minimize
print('pi,delta,m,seed,mu0_hat,s0_hat,pi_hat_model,pi_hat_naive,secs')
for pi in (0.10, 0.60, 0.95):
    for delta in (1.0, 2.0):
        for m in (50, 200):
            for seed in range(2):
                rng = np.random.default_rng(100 + seed); n = rng.integers(4, 16, 1500)
                x, off, y, A = H.simulate(n, pi=pi, e1=0.1, h=0.08, rho=0.4, delta=delta, sig1=1.0, f=0.3, seed=seed)
                lab = np.random.default_rng(7 + seed).choice(len(n), m, replace=False)
                cs = np.concatenate([x[off[a]:off[a + 1]][~y[off[a]:off[a + 1]]] for a in lab])
                mu0, s0 = cs.mean(), cs.std()
                X, M, _ = H.pad(x, off)
                nll = lambda th: -H._forward(X, M, H.unpack(th), mu0, s0).mean()
                t0 = time.time(); best = None
                for st in ([0.0, -2.0, -2.5, 0.0, 0.0, np.log(s0), 0.0], [1.5, -1.0, -2.0, 0.5, 0.4, np.log(s0), -1.0],
                           [-1.5, -2.5, -3.0, -0.5, -0.4, np.log(s0), 1.0]):
                    r = minimize(nll, st, method='L-BFGS-B', bounds=[(-8, 8)] * 4 + [(-4, 3), (-4, 3), (-6, 6)], options=dict(maxiter=300))
                    best = r if best is None or r.fun < best.fun else best
                print(f'{pi},{delta},{m},{seed},{mu0:.3f},{s0:.3f},{H.unpack(best.x)["pi"]:.3f},{A[lab].mean():.3f},{time.time()-t0:.0f}', flush=True)
