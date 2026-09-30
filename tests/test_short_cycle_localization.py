"""Standalone acceptance checks for the isolated short-cycle scientific capsule."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_cache/short_cycle01_code"))
spec = importlib.util.spec_from_file_location("spectral_utils.short_cycle_localization", ROOT / "spectral_utils/short_cycle_localization.py")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
from spectral_utils.joint_lsml import regularized_joint_map_weights, fit_joint_lsml
from spectral_utils.dependency_fusion import regularized_covariance_weights
from spectral_utils.window_localization import make_window_plan, windows_to_tokens


class PilotAcceptance(unittest.TestCase):
    def test_lambda_zero_identity_and_no_graph_construction(self):
        rng = np.random.default_rng(7)
        z = rng.normal(size=(60, 12))
        c = z.T @ z / len(z)
        v = rng.normal(size=12)
        expected, _ = regularized_covariance_weights(c, v, target_condition=1000)
        with patch("spectral_utils.laplacian_upcr.build_graph_from_features", side_effect=AssertionError("graph at lambda zero")):
            actual, _ = regularized_joint_map_weights(z, c, v, mode="liu", lam=0)
        np.testing.assert_array_equal(actual, expected)

    def test_group_size_guard_not_silently_relaxed(self):
        with self.assertRaises(ValueError):
            fit_joint_lsml(np.eye(10), [0, 0, 1, 1, 1, 2, 2, 2, 2, 2], anchor_index=0)

    def test_answer_only_normalization_excludes_held_windows(self):
        rng = np.random.default_rng(6)
        names = ["epr", "spectral_entropy", "low_band_power", "high_band_power", "hl_ratio", "dominant_freq",
                 "spectral_centroid", "stft_max_high_power", "stft_spectral_entropy", "rpdi", "sw_var_peak", "pe_mean"]
        x = rng.normal(size=(48, 12))
        fit = np.arange(32)
        with patch.object(core, "discover_loao_consensus_groups", return_value={"status": "BLOCKED", "candidates": []}):
            s1, m1, p1 = core.fit_windows(x, names, fit)
            changed = x.copy(); changed[32:] *= 1000
            s2, m2, p2 = core.fit_windows(changed, names, fit)
        self.assertEqual(p1, p2)
        for method in ("iu", "equal"):
            self.assertEqual(m1[method]["status"], "OK")
            np.testing.assert_array_equal(s1[method][:32], s2[method][:32])
            np.testing.assert_array_equal(m1[method]["weights_raw_coordinates"], m2[method]["weights_raw_coordinates"])
        self.assertEqual(m1["joint_modelinv_lam0"]["status"], "BLOCKED_NO_ADMISSIBLE_PARTITION")
        self.assertNotIn("joint_modelinv_lam0", s1)

    def test_tail_coverage_and_span_max(self):
        plan = make_window_plan(70, 32, 32)
        scores = np.array([1., 2., 4.])
        tokens = windows_to_tokens(plan, scores)
        np.testing.assert_array_equal(tokens[:32], 1.)
        self.assertEqual(tokens[-1], 4.)
        self.assertEqual(tokens[38], 3.)


if __name__ == "__main__": unittest.main()
