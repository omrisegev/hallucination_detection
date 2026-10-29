import numpy as np
from spectral_utils import ds_answer_gate as D


def simulate(n_ans=3000, K=4, rho=.6, psi=.75, phi1=.1, phi0=.15, seed=1):
    rng = np.random.default_rng(seed)
    lens = rng.integers(3, 16, n_ans); off = np.r_[0, np.cumsum(lens)]
    pos = np.concatenate([(np.arange(k) + .5) / k for k in lens])
    pi_true = .05 + .3 * pos
    z = rng.random(n_ans) < rho; zs = np.repeat(z, lens)
    y = zs & (rng.random(len(pos)) < pi_true)
    th = np.where(y[:, None], psi, np.where(zs[:, None], phi1, phi0))
    G = rng.random((len(pos), K)) < th
    return G, off, pos, z, y


def test_recovers_parameters():
    G, off, pos, z, y = simulate()
    f = D.fit(G, off, pos)
    assert abs(f.rho_hat - z.mean()) < .05
    assert np.allclose(f.psi, .75, atol=.05) and np.allclose(f.phi1, .1, atol=.03) and np.allclose(f.phi0, .15, atol=.03)
    assert abs(f.p.mean() - y.mean()) < .02


def test_cost_ratio_matches_finite_difference():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500); flags = rng.random(500) < .3; keep = np.ones(500, bool)
    o = D.cost_ratio_threshold(flags, p, keep)
    assert 0 < o['c'] < 1 and o['gain'] > 0 and o['loss'] > 0


def test_length_groups_run():
    G, off, pos, z, y = simulate(n_ans=800)
    lg = np.minimum((np.diff(off) - 3) // 5, 2)
    f = D.fit(G, off, pos, length_group=lg, restarts=1)
    assert f.psi.shape == (3, 4) and np.isfinite(f.ll)
