"""Pre-label synthetic recovery test (stop rule in PREREG). Data from the model itself, N=1500 answers,
lengths drawn from 4..15 (CRV-like). Seeds 0..2. Reports pi_hat vs pi, and delta_hat."""
import importlib.util, pathlib, time, sys, numpy as np
sp = importlib.util.spec_from_file_location('hh', pathlib.Path(__file__).resolve().parents[2] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)
print('pi,delta,seed,pi_hat,delta_hat,f_hat,rho_hat,secs')
for pi in (0.10, 0.60, 0.95):
    for delta in (1.0, 2.0):
        for seed in range(3):
            rng = np.random.default_rng(100 + seed)
            n = rng.integers(4, 16, 1500)
            x, off, y, A = H.simulate(n, pi=pi, e1=0.1, h=0.08, rho=0.4, delta=delta, sig1=1.0, f=0.3, seed=seed)
            t0 = time.time(); r = H.fit(x, off, q_anchor=0.30, max_fit_answers=1500, seed=seed)
            P = r['params']
            print(f"{pi},{delta},{seed},{P['pi']:.3f},{P['delta']:.3f},{P['f']:.3f},{P['rho']:.3f},{time.time()-t0:.1f}", flush=True)
