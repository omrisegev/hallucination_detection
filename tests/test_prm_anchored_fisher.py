"""Synthetic checks of spectral_utils.prm_anchored_fisher (P5 end to end). No real data; labels only score."""
import os
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np

from lever_imports import ensure_spectral_package

ensure_spectral_package()
from spectral_utils.prm_anchored_fisher import anchored_fisher_fuse, anchored_fisher_weights  # noqa: E402


def auc(y, s):
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def answers(rng, n_ans, length, p, d0, groups, lam, per_group=3, lam0=0.):
    """Steps of n_ans answers. PRM = d0*y + lam0*eta + noise; each telemetry group g has per_group channels
    = groups[g]*y + lam[g]*eta + group noise + channel noise; answer effects on every view."""
    n = n_ans * length
    off = np.arange(0, n + 1, length)
    y = (rng.random(n) < p).astype(int)
    eta = rng.standard_normal(n)
    a = np.repeat(rng.standard_normal((n_ans, 1 + len(groups) * per_group)), length, axis=0)
    prm = d0 * y + lam0 * eta + rng.standard_normal(n) + a[:, 0]
    cols = []
    for g, (dg, lg) in enumerate(zip(groups, lam)):
        e = rng.standard_normal(n)
        for j in range(per_group):
            cols.append(dg * y + lg * eta + e + .5 * rng.standard_normal(n) + a[:, 1 + g * per_group + j])
    return y, prm, np.column_stack(cols), off


class AnchoredWeights(unittest.TestCase):
    def test_both_routes_recover_loading_ratios(self):
        """Signal and nuisance of comparable size, loadings not parallel: both routes identify u_k / u_0.
        (With telemetry signal much weaker than the nuisance, the rank profile is flat and the route abstains;
        that guard is fusion_theory's, kept as is.)"""
        rng = np.random.default_rng(1)
        n = 80_000; p = .2
        y = (rng.random(n) < p).astype(float); eta = rng.standard_normal(n)
        D = np.array([1.5, 1.0, .8, 1.2, .6, .9]); L = np.array([0., .3, .7, .2, .6, .45])
        X = y[:, None] * D + eta[:, None] * L + rng.standard_normal((n, 6))
        off = np.array([0, n])
        for route in ('rank', 'cumulant'):
            w, info = anchored_fisher_weights(X - X.mean(0), off, route)
            self.assertIsNotNone(w, (route, info))
            u = np.array([info['u'][k] for k in range(6)])
            np.testing.assert_allclose(u[1:] / u[0], D[1:] / D[0], atol=.08, err_msg=route)

    def test_negative_anchor_group_is_excluded(self):
        rng = np.random.default_rng(2)
        n = 40_000; y = (rng.random(n) < .2).astype(float); eta = rng.standard_normal(n)
        D = np.array([1.5, .4, .35, .3, .5, -.6]); L = np.array([0., .8, .7, .9, .6, .2])
        X = y[:, None] * D + eta[:, None] * L + rng.standard_normal((n, 6))
        w, info = anchored_fisher_weights(X - X.mean(0), np.array([0, n]), 'auto')
        self.assertEqual(info['excluded_groups'], [5])
        self.assertEqual(w[5], 0.)

    def test_not_identified_falls_back_to_prm(self):
        rng = np.random.default_rng(3)
        n = 5_000; X = rng.standard_normal((n, 4))                     # no signal, no structure
        w, info = anchored_fisher_weights(X - X.mean(0), np.array([0, n]), 'auto')
        self.assertIsNone(w)
        self.assertIn('reason', info)


class EndToEnd(unittest.TestCase):
    def test_strong_prm_with_telemetry_nuisance_beats_prm(self):
        """Socratic-like: a strong PRM and weak telemetry that shares a nuisance. The anchored fusion should not lose
        to the PRM, and should beat an equal-weight mean of the views."""
        rng = np.random.default_rng(4)
        y, prm, V, off = answers(rng, 3000, 10, .2, 1.5, [.35, .3, .4, .3, .35], [.9, .8, 1., .85, .9])
        out = anchored_fisher_fuse(prm, V, off, method='auto')
        self.assertTrue(out['fused'], out['info'])
        a_fused, a_prm = auc(y, out['score']), auc(y, out['views'][:, 0])
        a_mean = auc(y, out['views'].mean(1))
        self.assertGreater(a_fused, a_prm - .005)
        self.assertGreater(a_fused, a_mean)

    def test_strong_telemetry_gains_over_prm(self):
        """PRMBench-like: telemetry stronger than the PRM, mild nuisance: the fusion gains clearly over the PRM."""
        rng = np.random.default_rng(5)
        y, prm, V, off = answers(rng, 3000, 10, .2, .8, [.9, .8, 1., .7, .85], [.3, .25, .35, .3, .2])
        out = anchored_fisher_fuse(prm, V, off, method='auto')
        self.assertTrue(out['fused'], out['info'])
        self.assertGreater(auc(y, out['score']), auc(y, out['views'][:, 0]) + .03)


if __name__ == "__main__":
    unittest.main()
