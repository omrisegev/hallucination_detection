"""Regression tests for scientific comparability and residual scale invariance."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("runtime_protocol", Path(__file__).resolve().parents[1] / "spectral_utils/runtime_fusion_protocol.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class ProtocolTests(unittest.TestCase):
    def test_residual_channels_are_scale_invariant_and_not_collapsed(self):
        rng = np.random.default_rng(73)
        x = rng.normal(size=(80, 4))
        spans = np.arange(0, 81, 10)
        spans = np.column_stack([spans[:-1], spans[1:]])
        a, _ = P.residual_step_bank(x, spans)
        b, _ = P.residual_step_bank(x * [1, 1000000, 3, 27] + [0, 100, -3, 2], spans)
        self.assertEqual(a.shape, (8, 4))
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(a.std(0), 1)

    def test_zero_predictor_retains_real_levels_not_zero_scores(self):
        x = np.arange(24.).reshape(12, 2)
        z, info = P.residual_step_bank(x - np.zeros_like(x), np.array([[0, 4], [4, 8], [8, 12]]))
        self.assertGreater(np.linalg.norm(z), 0)
        np.testing.assert_allclose(info["raw_step_profile"], [[3, 4], [11, 12], [19, 20]])

    def test_constant_and_single_step_are_zero(self):
        z, _ = P.residual_step_bank(np.ones((4, 3)), np.array([[0, 4]]))
        np.testing.assert_array_equal(z, np.zeros((1, 3)))

    def test_invalid_spans_fail(self):
        with self.assertRaises(ValueError):
            P.residual_step_bank(np.ones((4, 3)), np.array([[1, 1]]))

    def test_native_equal_drops_easy_fallback_rows_on_both_sides(self):
        mask = np.array([True, False, True])
        m = {"window_lsml_top10_native": {"valid": mask.copy()},
             "window_equal_top10": {"valid": np.ones(3, bool), "score": np.arange(3)}}
        pairs = P.add_native_controls(m, {"lsml": mask}, arms=("lsml",))
        np.testing.assert_array_equal(m[pairs[0][1]]["valid"], mask)
        self.assertTrue(m["window_equal_top10"]["valid"].all())
        m[pairs[0][1]]["valid"][0] = False
        self.assertTrue(mask[0])

    def test_paired_n_counts_observations_not_draws(self):
        m = P.paired_population(["a", "a", "b"], [1, 1, 0], [1, 1, 0], draws=100000, valid_draws=99999)
        self.assertEqual(m, {"paired_N": 2, "paired_groups": 1, "B": 100000, "valid_draws": 99999})
        with self.assertRaises(ValueError):
            P.paired_population(["a", "a"], [1, 0], [1, 1], draws=10, valid_draws=10)


if __name__ == "__main__":
    unittest.main()
