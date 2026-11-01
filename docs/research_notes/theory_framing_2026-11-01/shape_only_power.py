"""T3: power of the best shape-only (per-answer affine invariant) clean-answer test, Gaussian model.

One answer has n steps. Correct steps ~ N(0,1); k error steps ~ N(d,1). The invariant max test
(Grubbs statistic) is calibrated to 5% false alarms on clean answers. For contrast, the max test
with a KNOWN null (not invariant: it needs a calibrated level) is also given. Seed 0, 200k draws.
"""
import numpy as np

rng = np.random.default_rng(0); R = 200_000
def grubbs(x):
    return ((x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)).max(1)
print('n,d,k,power_invariant,power_known_null')
for n in (8, 12):
    c_inv = np.quantile(grubbs(rng.standard_normal((R, n))), .95)
    c_raw = np.quantile(rng.standard_normal((R, n)).max(1), .95)
    for d in (0.8, 1.0, 1.15, 2.3, 3.0):
        for k in (1, 2):
            x = rng.standard_normal((R, n)); x[:, :k] += d
            print(f'{n},{d},{k},{(grubbs(x) > c_inv).mean():.3f},{(x.max(1) > c_raw).mean():.3f}')
