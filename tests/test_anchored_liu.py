import unittest
import numpy as np
from spectral_utils.anchored_liu import (
    covariance_target, liu_external_target, orient_to, pooled_zscore_features,
)
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS, laplacian_iu_path, self_tuning_knn_graph
from spectral_utils.upcr import upcr_fit


def _features(seed=3, n=400, m=7):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    nuisance = rng.normal(size=n)
    V = np.column_stack([y + .8 * rng.normal(size=n) for _ in range(4)]
                        + [nuisance + .3 * rng.normal(size=n) for _ in range(m - 4)])
    F, keep = pooled_zscore_features(V)
    return F, y


class ExternalTargetSolve(unittest.TestCase):
    def test_reproduces_laplacian_path_with_rho_hat(self):
        F, _ = _features()
        graph = self_tuning_knn_graph(F.T, k=7)
        ref = laplacian_iu_path(F, (0.1,), graph=graph)[0.1]
        out = liu_external_target(F, ref.baseline.rho_hat, graph=graph, lambda_=0.1)
        np.testing.assert_allclose(out.w, ref.w, atol=1e-10, rtol=1e-10)

    def test_lambda_zero_is_iu_pcr(self):
        F, _ = _features(seed=5)
        graph = self_tuning_knn_graph(F.T, k=7)
        base = upcr_fit(F, **IU_FIT_DEFAULTS)
        out = liu_external_target(F, base.rho_hat, graph=graph, lambda_=0.0)
        np.testing.assert_allclose(out.w, base.w, atol=1e-8, rtol=1e-8)

    def test_target_anchored_fit_equals_its_parts(self):
        from spectral_utils.laplacian_upcr import build_graph_from_features
        from spectral_utils.target_anchored_laplacian import positive_correlation_gates, target_anchored_laplacian_fit
        F, y = _features(seed=13)
        labels = (y > np.quantile(y, .8)).astype(float); idx = np.arange(0, 400, 2)
        ref = target_anchored_laplacian_fit(F, labels, idx, lambda_=0.1, k=7, baseline_kwargs=dict(IU_FIT_DEFAULTS))
        gate = positive_correlation_gates(F, labels, indices=idx, require_two_classes=True)
        graph = build_graph_from_features(F, gates=gate.gates, k=7)
        out = liu_external_target(F, upcr_fit(F, **IU_FIT_DEFAULTS).rho_hat, graph=graph, lambda_=0.1)
        np.testing.assert_allclose(out.w, ref.fit.w, atol=1e-10, rtol=1e-10)

    def test_split_weights_equal_laplacian_path_with_gates(self):
        from spectral_utils.anchored_liu import split_liu_weights
        F, _ = _features(seed=17)
        gates = np.linspace(0.2, 1.8, F.shape[0])
        ref = laplacian_iu_path(F, (0.1,), gates=gates, k=7)[0.1]
        w, graph, diag = split_liu_weights(F, gates)
        np.testing.assert_allclose(w, ref.w, atol=1e-10, rtol=1e-10)
        self.assertAlmostEqual(diag['weight_cosine_vs_iu'], ref.diagnostics['weight_cosine_vs_iu'], places=10)
        self.assertEqual(diag['n_components'], ref.diagnostics['n_components'])

    def test_covariance_target_reads_only_indices(self):
        F, y = _features(seed=7)
        idx = np.arange(0, 400, 3)
        y2 = y.copy(); y2[np.setdiff1d(np.arange(400), idx)] = 1e6
        np.testing.assert_array_equal(covariance_target(F, y, indices=idx), covariance_target(F, y2, indices=idx))

    def test_label_target_moves_weights_to_signal_block(self):
        F, y = _features(seed=11)
        graph = self_tuning_knn_graph(F.T, k=7)
        w = liu_external_target(F, covariance_target(F, y), graph=graph, lambda_=0.1).w
        self.assertGreater(np.abs(w[:4]).sum(), np.abs(w[4:]).sum())


class Helpers(unittest.TestCase):
    def test_constant_columns_dropped(self):
        V = np.column_stack([np.arange(10.), np.ones(10), np.arange(10.) ** 2, np.sin(np.arange(10.))])
        F, keep = pooled_zscore_features(V)
        self.assertEqual(keep.tolist(), [True, False, True, True])
        np.testing.assert_allclose(F.mean(axis=1), 0, atol=1e-12)
        np.testing.assert_allclose(F.std(axis=1), 1, atol=1e-12)

    def test_orient_to(self):
        s, flipped = orient_to(-np.arange(5.), np.arange(5.))
        self.assertTrue(flipped); np.testing.assert_array_equal(s, np.arange(5.))


if __name__ == '__main__':
    unittest.main()
