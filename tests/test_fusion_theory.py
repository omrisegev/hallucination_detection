"""Synthetic checks of spectral_utils.fusion_theory (FRAMING_THEORY.md, P1-P5). No real data, no labels in fits."""
import os
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np
from scipy.stats import norm

from lever_imports import ensure_spectral_package

ensure_spectral_package()
from spectral_utils.fusion_theory import (anchored_loadings, fused_auc_gaussian, gate_decision, gate_statistic,  # noqa: E402
                                          improvement_threshold, two_view_separation, within_answer_center,
                                          within_answer_residualize, zero_diag_eigen_weights)


def auc(y, s):
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def latent_views(rng, n, p, d0, dt, lam, lam0=0.0, noise="gauss"):
    """PRM view 0 plus telemetry views with a shared nuisance eta (loadings lam; lam0 on the PRM)."""
    y = (rng.random(n) < p).astype(int)
    eta = rng.standard_normal(n)
    eps = (rng.exponential(1., (n, 1 + len(dt))) - 1.) if noise == "exp" else rng.standard_normal((n, 1 + len(dt)))
    D = np.r_[d0, dt]; L = np.r_[lam0, lam]
    return y, y[:, None] * D + eta[:, None] * L + eps, D, L


class PopulationFormulas(unittest.TestCase):
    def test_gaussian_auc_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        delta = np.array([1.2, .5, .3]); S = np.array([[1, .2, 0], [.2, 1, .4], [0, .4, 1.]])
        w = np.array([.6, .3, .1]); n = 200_000
        L = np.linalg.cholesky(S)
        x0 = rng.standard_normal((n, 3)) @ L.T; x1 = rng.standard_normal((n, 3)) @ L.T + delta
        y = np.r_[np.zeros(n), np.ones(n)]; s = np.r_[x0 @ w, x1 @ w]
        self.assertAlmostEqual(fused_auc_gaussian(w, delta, S), auc(y, s), delta=.003)

    def test_improvement_threshold_matches_brute_force(self):
        grid = np.linspace(0, .999, 2000)
        for d_p, d_t, rho in [(1.5, .7, 0.), (1.26, .5, .3), (1.2, 1.3, .3), (1.0, .2, .4), (.8, .6, -.2)]:
            a_min = improvement_threshold(d_p, d_t, rho)
            helps = np.array([two_view_separation(a, d_p, d_t, rho) > d_p for a in grid])
            if a_min >= 1.:
                self.assertFalse(helps.any())
            else:
                np.testing.assert_array_equal(helps[np.abs(grid - a_min) > 2e-3], grid[np.abs(grid - a_min) > 2e-3] > a_min)

    def test_zero_diag_eigenvector_closed_form(self):
        for u in ([.567, .136, .136, .136, .136, .136, .136], [.4, .3, .25, .1], [.3, .3, .3, .3]):
            u = np.array(u)
            R = np.outer(u, u) - np.diag(u ** 2)
            vals, vecs = np.linalg.eigh(R)
            v_num = np.abs(vecs[:, -1]); v_num /= v_num.sum()
            v_cf, lam1 = zero_diag_eigen_weights(u)
            np.testing.assert_allclose(v_cf, v_num, atol=1e-9)
            self.assertAlmostEqual(lam1, vals[-1], places=9)
        v_eq, _ = zero_diag_eigen_weights([.3, .3, .3, .3])
        np.testing.assert_allclose(v_eq, .25)


class AnswerStructure(unittest.TestCase):
    def test_centering_points_the_loading_at_the_error(self):
        rng = np.random.default_rng(3)
        J, n = 4000, 10; D = np.array([1.5, 1.0, .8, .6]); G = np.array([1.2, -.8, 1.0, .5])
        A = np.repeat(rng.standard_normal(J), n); y = (rng.random(J * n) < .15).astype(float)
        X = A[:, None] * G + y[:, None] * D + rng.standard_normal((J * n, 4))
        off = np.arange(0, J * n + 1, n)

        def loading(M):
            C = np.cov(M.T); v = np.full(4, .5)
            for _ in range(300):
                for i in range(4):
                    m = np.arange(4) != i
                    v[i] = C[i, m] @ v[m] / (v[m] @ v[m])
            return v / np.linalg.norm(v)

        cos = lambda a, b: abs(a @ b) / np.linalg.norm(b)
        self.assertLess(cos(loading(X), D), .8)
        self.assertGreater(cos(loading(within_answer_center(X, off)), D), .99)

    def test_residualize_removes_covariate(self):
        rng = np.random.default_rng(4)
        J, n = 2000, 8; off = np.arange(0, J * n + 1, n)
        pos = np.tile((np.arange(n) + .5) / n, J)
        X = np.column_stack([2 * pos + rng.standard_normal(J * n), -pos ** 2 + rng.standard_normal(J * n)])
        R = within_answer_residualize(X, np.column_stack([pos, pos ** 2, pos ** 3]), off)
        for k in range(2):
            self.assertLess(abs(np.corrcoef(R[:, k], pos)[0, 1]), 1e-8)


class AnchoredIdentification(unittest.TestCase):
    def test_cumulant_anchor_recovers_prm_loading_where_triads_fail(self):
        rng = np.random.default_rng(7)
        p = .15; y, X, D, L = latent_views(rng, 60_000, p, 1.5, [.6, .5, .4, .3, .25], [.8, .2, .7, .1, .6])
        u_true0 = np.sqrt(p * (1 - p)) * 1.5
        u, info = anchored_loadings(X, 0, "cumulant")
        self.assertIsNotNone(u)
        self.assertLess(abs(u[0] - u_true0) / u_true0, .1)
        C = info["C"]; triads = [C[0, a] * C[0, b] / C[a, b] for a in range(1, 6) for b in range(a + 1, 6)]
        self.assertLess(np.sqrt(np.median(triads)), .8 * u_true0)

    def test_negative_anchor_covariance_is_flagged(self):
        rng = np.random.default_rng(11)
        y, X, D, L = latent_views(rng, 30_000, .15, 1.5, [.6, .5, .4, .3, .25], [.8, .2, .7, .1, .6], lam0=-.9)
        _, info = anchored_loadings(X, 0, "cumulant")
        self.assertGreater(info["n_negative_anchor_cov"], 0)

    def test_gate_sign_with_true_loadings(self):
        rng = np.random.default_rng(5)
        for d0, dt, lam, helps in [(1.5, [.6, .5, .4, .3, .25], [.8, .2, .7, .1, .6], False),
                                   (1.0, [1.2, 1.0, .9, .8, .7], [.2, .1, .2, .05, .1], True)]:
            p = .15; y, X, D, L = latent_views(rng, 40_000, p, d0, dt, lam)
            X = X - X.mean(0); C = np.cov(X.T); u = np.sqrt(p * (1 - p)) * D
            w = 1. / np.sqrt(np.diag(C))                                   # equal weights on z-scored views
            g = gate_statistic(C, u, w)["G"]
            truth = auc(y, X @ w) - auc(y, X[:, 0])
            self.assertEqual(g > 0, helps); self.assertEqual(truth > 0, helps)

    def test_gate_decision_clear_regimes(self):
        rng = np.random.default_rng(9)
        n, J = 40_000, 4000; off = np.arange(0, n + 1, n // J)
        for d0, dt, lam, expect in [(1.5, [.6, .5, .4, .3, .25], [.8, .2, .7, .1, .6], {"prm", "abstain"}),
                                    (.8, [1.2, 1.0, .9, .8, .7], [.2, .1, .2, .05, .1], {"fuse"})]:
            y, X, D, L = latent_views(rng, n, .15, d0, dt, lam)
            w = 1. / X.std(0)
            dec = gate_decision(X - X.mean(0), off, w, method="cumulant", B=60, seed=1)
            self.assertIn(dec["decision"], expect)


if __name__ == "__main__":
    unittest.main()
