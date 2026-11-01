"""Synthetic: can answer-level DS / Gaussian latent class recover prevalence with K=4 voters, separation d,
and a shared answer-level nuisance factor (class-conditional correlation rho)? Seed 0."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from answer_level_ds_check import ds_binary
from sklearn.mixture import GaussianMixture
rng = np.random.default_rng(0); N = 3000; K = 4
print('true_pi,d,rho,dsb_median_pi,glc_pi')
for pi in (0.08, 0.21, 0.65):
    for d in (1.0, 2.0):
        for rho in (0.0, 0.2):
            out = []
            for r in range(5):
                y = rng.random(N) < pi
                f = rng.standard_normal(N)[:, None]
                X = d * y[:, None] + np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((N, K))
                V = (X > np.median(X, 0)).astype(float)
                _, pb, _, _ = ds_binary(V)
                gm = GaussianMixture(2, covariance_type='diag', n_init=5, random_state=r).fit(X)
                e = int(np.argmax(gm.means_.mean(1))); out.append((pb, gm.weights_[e]))
            m = np.median(np.array(out), 0)
            print(f'{pi},{d},{rho},{m[0]:.3f},{m[1]:.3f}')
