"""Pre-label synthetic stop test for v2 (PREREG v2 section 7). Model data, N=1500, lengths 4..15, seeds 0..1.
Labelled set m=200 answers gives pi and the clean law; the model fits the rest label-free on all answers.
Compares step macro-F1 (all steps of the 1300 unlabelled answers) of HH-BAYES with GAP, SHARE, THR, GATE_GAP.
In synthetic data z = per-answer z-score of x (the analogue of the answer-z fused score)."""
import numpy as np, time
from common import H, calibrate, controls, macro_f1
print('pi,delta,seed,pi_cal,f1_hh,f1_gap,f1_share,f1_gate_gap,f1_thr,at_bound,secs')
for pi in (0.10, 0.60, 0.95):
    for delta in (1.0, 2.0):
        for seed in range(2):
            rng = np.random.default_rng(100 + seed); n = rng.integers(4, 16, 1500)
            x, off, y, A = H.simulate(n, pi=pi, e1=0.1, h=0.08, rho=0.4, delta=delta, sig1=1.0, f=0.3, seed=seed)
            keep = np.ones(len(x), bool)
            lab = np.sort(np.random.default_rng(7 + seed).choice(len(n), 200, replace=False))
            pi_c, st = calibrate(lab, off, y, keep, A)
            cs = x[st][~y[st]]; s_share = y[st].mean()
            z = np.concatenate([(x[a:b] - x[a:b].mean()) / max(x[a:b].std(), 1e-8) for a, b in zip(off[:-1], off[1:])])
            t0 = time.time()
            r = H.fit(x, off, anchor=(cs.mean(), cs.std()), fixed_pi=pi_c, max_fit_answers=1500, seed=seed)
            ll, pA, py = H.score(x, off, r)
            c = controls(x, z, off, st, y, pi_c, s_share)
            test = np.ones(len(x), bool); test[np.concatenate([np.arange(off[a], off[a + 1]) for a in lab])] = False
            th = r['theta']; atb = int(np.any(np.isclose(th[1:4], [-8, 8][0]) | np.isclose(th[1:4], 8)) or np.any(np.isclose(th[4:6], -4) | np.isclose(th[4:6], 3)) or np.any(np.isclose(th[6], [-6, 6])))
            f = {k: macro_f1(y[test], v[test]) for k, v in c.items()}
            print(f"{pi},{delta},{seed},{pi_c:.3f},{macro_f1(y[test], (py > .5)[test]):.4f},{f['GAP']:.4f},{f['SHARE']:.4f},{f['GATE_GAP']:.4f},{f['THR']:.4f},{atb},{time.time()-t0:.0f}", flush=True)
