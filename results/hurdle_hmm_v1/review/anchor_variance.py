"""Sampling spread of the stage-A anchor (truncated-normal MLE on the bottom 30%) on pure N(0,1) data,
at step counts like the synthetic test (~14k) and the real cells (9k-94k). Also with an answer effect
(f = .3), which makes steps dependent within answers."""
import importlib.util, pathlib, numpy as np
sp = importlib.util.spec_from_file_location('hh', pathlib.Path(__file__).resolve().parents[3] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)
rng = np.random.default_rng(7)
for N in (9000, 14000, 26000, 94000):
    for f in (0.0, 0.3):
        mus, sds = [], []
        for r in range(60):
            na = N // 9; b = rng.normal(0, np.sqrt(f), na)
            x = np.repeat(b, 9) + rng.normal(0, np.sqrt(1 - f), na * 9)
            m, s, _ = H.anchor_fit(x, 0.30); mus.append(m); sds.append(s)
        print(f'N={N:6d} f={f}: mu0 mean {np.mean(mus):+.3f} sd {np.std(mus):.3f} | s0 mean {np.mean(sds):.3f} sd {np.std(sds):.3f}')
