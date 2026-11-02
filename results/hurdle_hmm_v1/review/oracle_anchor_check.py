"""Attribution: the synthetic draw (pi=.1, delta=2, seed=2) refitted with the TRUE clean law (mu0=0, s0=1)
in place of the stage-A anchor. If pi is then recovered, the failure is the anchor's sampling error."""
import importlib.util, pathlib, sys, numpy as np
sp = importlib.util.spec_from_file_location('hh', pathlib.Path(__file__).resolve().parents[3] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)
pi, delta, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(100 + seed); n = rng.integers(4, 16, 1500)
x, off, y, A = H.simulate(n, pi=pi, e1=0.1, h=0.08, rho=0.4, delta=delta, sig1=1.0, f=0.3, seed=seed)
H.anchor_fit = lambda x, q: (0.0, 1.0, float(np.quantile(x, q)))
r = H.fit(x, off, q_anchor=0.30, max_fit_answers=1500, seed=seed)
print('true-anchor fit', {k: round(float(v), 3) for k, v in r['params'].items()}, 'converged', r['converged'])
