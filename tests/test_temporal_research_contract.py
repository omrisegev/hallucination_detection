import unittest
import numpy as np
from spectral_utils.pb_prediction_bundle import prediction_bundle
from spectral_utils.temporal_research_features import prefix_innovation, SUBSETS


class ReportingTests(unittest.TestCase):
    def test_gate_updates_entire_bundle(self):
        target = np.array([-1, -1, 0, 1, 2, 0])
        peaks = np.array([0, 0, 0, 1, 1, 0])
        cells = np.array(["pb_fixture_q4"] * 6)
        valid = np.ones(6, bool)
        gate = np.array([False, True, True, False, True, True])
        gate_valid = np.array([True, True, True, True, True, False])
        m, pred, decision = prediction_bundle(target, cells, peaks, valid, gate, gate_valid)
        self.assertEqual(m["pb_exact_count"], 3)
        self.assertEqual(m["pb_final_exact_count"], 1)
        self.assertEqual(m["pb_correct_peaks_suppressed"], 1)
        self.assertEqual(m["pb_correct_peaks_invalidated"], 1)
        self.assertEqual(m["pb_clean_accuracy"], .5)
        self.assertEqual(m["pb_error_exact_accuracy"], .25)
        self.assertAlmostEqual(m["pb_all8"], 1/3)
        self.assertEqual(m["pb_all8"], m["pb_cells"][cells[0]]["f1"])
        self.assertFalse(decision[-1])

    def test_invalid_closed_locator_not_clean_success(self):
        m, _, _ = prediction_bundle([-1, 0], ["pb_x_q8"] * 2, [-1, 0], [False, True], [False, True])
        self.assertEqual(m["pb_clean_accuracy"], 0)
        self.assertEqual(m["pb_invalid"], 1)

    def test_prefix_excludes_self_and_future(self):
        x = np.array([1., 3., 7., 10.])
        r, available = prefix_innovation(x)
        np.testing.assert_allclose(r, [0, 2, 5, 10 - 11/3])
        y = x.copy(); y[2:] = 999
        np.testing.assert_array_equal(prefix_innovation(y)[0][:2], r[:2])
        np.testing.assert_array_equal(available, [False, True, True, True])
        np.testing.assert_array_equal(prefix_innovation([5.])[0], [0.])
        self.assertEqual(len(SUBSETS), 15)


if __name__ == "__main__":
    unittest.main()
