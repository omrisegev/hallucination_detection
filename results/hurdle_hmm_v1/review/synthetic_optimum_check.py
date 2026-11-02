"""Is a bad synthetic recovery an optimizer failure (truth has lower NLL than the fit) or a likelihood/anchor issue?
Re-creates the exact synthetic_recovery.py draw for given (pi, delta, seed), fits with the module, then evaluates
the module NLL at the true theta (with the anchored mu0, s0) and at the fit."""
import importlib.util, pathlib, sys, numpy as np
from scipy.special import logit
sp = importlib.util.spec_from_file_location('hh', pathlib.Path(__file__).resolve().parents[3] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)
pi, delta, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(100 + seed); n = rng.integers(4, 16, 1500)
x, off, y, A = H.simulate(n, pi=pi, e1=0.1, h=0.08, rho=0.4, delta=delta, sig1=1.0, f=0.3, seed=seed)
print('realised share A=1', A.mean().round(4), 'step error share', y.mean().round(4))
mu0, s0, t = H.anchor_fit(x, 0.30); print('anchor mu0 s0', round(mu0, 4), round(s0, 4), '(truth 0, 1)')
X, M, _ = H.pad(x, off)
def nll(th, m0=mu0, ss=s0): return -H._forward(X, M, H.unpack(th), m0, ss).sum() / len(n)
th_true = np.array([logit(pi), logit(.1), logit(.08), logit(.4), np.log(delta), np.log(1.0), logit(.3)])
print('NLL at truth (anchored mu0,s0):', nll(th_true), ' at truth with exact mu0=0,s0=1:', nll(th_true, 0.0, 1.0))
r = H.fit(x, off, q_anchor=0.30, max_fit_answers=1500, seed=seed)
print('fit params', {k: round(float(v), 3) for k, v in r['params'].items()}, 'NLL', r['nll'], 'converged', r['converged'])
r2 = H.fit(x, off, q_anchor=0.30, max_fit_answers=1500, seed=seed, th0=th_true)
print('fit from truth', {k: round(float(v), 3) for k, v in r2['params'].items()}, 'NLL', r2['nll'], 'converged', r2['converged'])
