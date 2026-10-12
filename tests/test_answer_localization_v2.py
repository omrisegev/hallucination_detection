"""Scientific-contract checks for the new representation and evaluator."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_cache/short_cycle01_code"))
import spectral_utils
spectral_utils.__path__.append(str(ROOT / "spectral_utils"))
from spectral_utils.answer_localization_v2 import (PRIMITIVES, STREAM_NAMES, moment_matrix, moment_plan,
                                                  mixture_readout, prepare_local)
from spectral_utils.window_localization import windows_to_tokens


class MeasurementContract(unittest.TestCase):
    def test_full_support_and_remainder_not_fit(self):
        plan = moment_plan(19, 8)
        np.testing.assert_array_equal(plan.starts, [0, 8, 11])
        np.testing.assert_array_equal(plan.fit_indices, [0, 1])
        scores = windows_to_tokens(plan, [1., 3., 5.])
        self.assertEqual(len(scores), 19)
        self.assertEqual(scores[0], 1.)
        self.assertEqual(scores[12], 4.)
        self.assertEqual(scores[-1], 5.)

    def test_local_measurements_have_known_units(self):
        raw = np.tile(np.arange(19, dtype=float)[:, None], (1, len(STREAM_NAMES)))
        values, names = moment_matrix(raw, moment_plan(19, 8))
        self.assertEqual(values.shape, (3, 27))
        self.assertEqual(len(set(names)), 27)
        for stream in PRIMITIVES:
            np.testing.assert_allclose(values[:, names.index(stream + "__level")], [3.5, 11.5, 14.5])
            np.testing.assert_allclose(values[:, names.index(stream + "__slope")], 7.)
            np.testing.assert_allclose(values[:, names.index(stream + "__sd")], np.sqrt(5.25))

    def test_affine_duplicates_and_constant_columns_are_removed(self):
        rng = np.random.default_rng(9)
        base = rng.normal(size=(30, 3))
        values = np.column_stack((base, 2*base[:, 1]+10, np.ones(30)))
        _, _, audit = prepare_local(values, ["epr", "b", "c", "b_copy", "constant"], np.arange(30))
        self.assertEqual(audit["active_features"], ["epr", "b", "c"])
        self.assertEqual(audit["exact_affine_duplicates_removed"], {"b_copy": "b"})
        self.assertFalse(audit["borrowed_fitted_quantities"])

    def test_readout_can_predict_clean_and_a_known_shift(self):
        rng = np.random.default_rng(31)
        one = rng.normal(size=1000)
        self.assertEqual(mixture_readout(one, [one.mean()])["prediction"], -1)
        two = np.r_[rng.normal(-3., .2, 40), rng.normal(3., .2, 40)]
        readout = mixture_readout(two, [-3., -2.8, 3., 3.1])
        self.assertEqual(readout["prediction"], 2)
        self.assertTrue(readout["two_components_selected"])

    def test_pb_fit_failures_never_earn_clean_success(self):
        spec = importlib.util.spec_from_file_location("pilot_runner_test", ROOT / "scripts/run_answer_localization_v2.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        rows = [dict(cell="pb_a", target=-1, predictions={"x": -1}, decision_valid={"x": False}),
                dict(cell="pb_a", target=2, predictions={"x": 2}, decision_valid={"x": True})]
        result = runner.pb_metric(rows, "x")
        self.assertEqual(result["cells"]["pb_a"]["clean_accuracy"], 0.)
        self.assertEqual(result["macro_f1"], 0.)
        self.assertIsNone(runner.pb_metric(rows[:1], "x")["macro_f1"])


if __name__ == "__main__":
    unittest.main()
