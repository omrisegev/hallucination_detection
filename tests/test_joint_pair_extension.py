"""Scientific invariants of a native Joint map with pair feature groups."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT / 'spectral_utils'))
from spectral_utils.joint_pair_extension import pair_model_covariance, fit_joint_pairs
from spectral_utils.joint_lsml import fit_joint_lsml, _profiled_jacobian_audit


def fixture(sizes=(2, 2, 2)):
    groups = np.repeat(np.arange(len(sizes)), sizes)
    n = len(groups)
    v = np.linspace(.3, .55, n)
    u = np.linspace(.2, .45, n); u[1] *= -1
    same = groups[:, None] == groups[None, :]
    s = np.outer(v, v) + same * np.outer(u, u)
    np.fill_diagonal(s, 1.)
    return s, groups, v, u


def old_covariance(s, groups, v, u):
    c = np.outer(v, v) + (groups[:, None] == groups[None, :]) * np.outer(u, u)
    return c + np.diag(np.maximum(np.diag(s)-np.diag(c), 0.))


def independent_inverse(c, v):
    values, vectors = np.linalg.eigh(c)
    psd = (vectors * np.maximum(values, 0.)) @ vectors.T
    lo, hi = np.linalg.eigvalsh(psd)[[0, -1]]
    ridge = max((hi-1000*lo)/999, hi*1e-10, 0.)
    return np.linalg.solve(psd+ridge*np.eye(len(v)), v)


class PairInvariants(unittest.TestCase):
    def test_arbitrary_scale_changes_legacy_head_but_not_offdiagonal(self):
        s, g, v, u = fixture(); altered = u.copy(); altered[0] *= 10; altered[1] /= 10
        left, right = old_covariance(s, g, v, u), old_covariance(s, g, v, altered)
        off = ~np.eye(len(g), dtype=bool)
        np.testing.assert_allclose(left[off], right[off], atol=1e-15)
        self.assertGreater(np.linalg.norm(independent_inverse(left, v)-independent_inverse(right, v)), .05)
        mask = (g[:, None] == g[None, :]).astype(float); np.fill_diagonal(mask, 0)
        self.assertTrue(_profiled_jacobian_audit(v, u, mask)['full_global_rank'])
        self.assertTrue(_profiled_jacobian_audit(v, altered, mask)['full_global_rank'])

    def test_pair_representative_preserves_covariance_under_scale_family(self):
        s, g, v, u = fixture()
        for scale in (.03, .2, 1., 4., 30.):
            altered = u.copy(); altered[::2] *= scale; altered[1::2] /= scale
            c, canonical, audit = pair_model_covariance(s, g, v, altered)
            np.testing.assert_allclose(c, s, atol=1e-14)
            np.testing.assert_allclose(canonical[::2]*canonical[1::2], u[::2]*u[1::2], atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(c)[0], -1e-14)
            budgets = 1-v*v
            np.testing.assert_allclose(canonical[::2]**2/budgets[::2], canonical[1::2]**2/budgets[1::2], atol=1e-14)
            self.assertFalse(audit['latent_pair_loadings_identified'])
            # Any fixed graph/sensitivity penalty leaves invariance intact.
            rng = np.random.default_rng(123); b = rng.normal(size=(6, 6)); penalty = b.T@b
            np.testing.assert_allclose(independent_inverse(c+.1*penalty, v), independent_inverse(s+.1*penalty, v), atol=1e-14)

    def test_zero_product_and_zero_budget(self):
        s, g, v, u = fixture(); u[:2] = 0.; v[0] = 1.
        s = old_covariance(np.eye(6), g, v, u)
        c, canonical, _ = pair_model_covariance(s, g, v, u)
        np.testing.assert_allclose(c, s, atol=1e-14)
        np.testing.assert_array_equal(canonical[:2], [0., 0.])

    def test_infeasible_variances_and_singleton_fail(self):
        s, g, v, u = fixture(); bad = v.copy(); bad[0] = 2.
        with self.assertRaisesRegex(ValueError, 'NEGATIVE_RESIDUAL_VARIANCE'):
            pair_model_covariance(s, g, bad, u)
        bad = u.copy(); bad[:2] = 2.
        with self.assertRaisesRegex(ValueError, 'EXCEEDS_VARIANCE_CAPACITY'):
            pair_model_covariance(s, g, v, bad)
        with self.assertRaisesRegex(ValueError, 'MINIMUM_SIZE_TWO'):
            pair_model_covariance(s, [0, 0, 1, 1, 2, 3], v, u)

    def test_feature_permutation_and_sign_equivariance(self):
        s, g, v, u = fixture(); c, _, _ = pair_model_covariance(s, g, v, u)
        order = np.array([4, 1, 3, 0, 5, 2]); signs = np.array([-1., 1., -1., -1., 1., 1.])
        changed = s[np.ix_(order, order)]*np.outer(signs, signs)
        cp, _, _ = pair_model_covariance(changed, g[order], v[order]*signs, u[order]*signs)
        np.testing.assert_allclose(cp, c[np.ix_(order, order)]*np.outer(signs, signs), atol=1e-14)
        np.testing.assert_allclose(independent_inverse(cp, v[order]*signs), independent_inverse(c, v)[order]*signs, atol=1e-14)

    def test_minimum_three_is_exact_legacy_replay(self):
        s, g, _, _ = fixture((3, 3, 3))
        old = fit_joint_lsml(s, g, anchor_index=0, seed=2026090601)
        new = fit_joint_pairs(s, g, anchor_index=0)
        self.assertEqual(new.pair_audit['status'], 'EXACT_LEGACY_REPLAY')
        np.testing.assert_array_equal(new.joint.global_loading, old.global_loading)
        np.testing.assert_array_equal(new.joint.model_covariance, old.model_covariance)
        self.assertEqual(new.joint.objective, old.objective)

    def test_genuine_pair_fit_recovers_native_covariance(self):
        s, g, v, _ = fixture()
        fit = fit_joint_pairs(s, g, anchor_index=0)
        self.assertTrue(fit.joint.converged)
        self.assertEqual(fit.joint.multistart_audit['status'], 'PASS')
        self.assertEqual(fit.native_map_audit['status'], 'PASS')
        self.assertTrue(fit.joint.jacobian_audit['full_global_rank'])
        np.testing.assert_allclose(fit.joint.global_loading, v, atol=1e-8)
        np.testing.assert_allclose(fit.joint.model_covariance, s, atol=1e-8)
        self.assertEqual(fit.pair_audit['pair_count'], 3)

    def test_full_parameter_jacobian_has_three_pair_scale_null_directions(self):
        _, g, v, u = fixture(); ids = np.triu_indices(6, 1)
        def prediction(theta):
            vv, uu = theta[:6], theta[6:]
            return (np.outer(vv, vv)+(g[:, None] == g[None, :])*np.outer(uu, uu))[ids]
        theta = np.r_[v, u]; step = 1e-6
        jac = np.column_stack([(prediction(theta+step*d)-prediction(theta-step*d))/(2*step) for d in np.eye(12)])
        self.assertEqual(np.linalg.matrix_rank(jac, tol=1e-7), 9)
        for i in (0, 2, 4):
            direction = np.zeros(12); direction[6+i] = u[i]; direction[6+i+1] = -u[i+1]
            np.testing.assert_allclose(jac@direction, 0., atol=1e-10)


if __name__ == '__main__':
    unittest.main()
