"""Synthetic checks of spectral_utils.risk_control (calibrated track, 2026-10-25)."""
import unittest

import numpy as np

from lever_imports import ensure_spectral_package

ensure_spectral_package()
from spectral_utils.risk_control import fixed_sequence_threshold, hb_pvalue  # noqa: E402


class RiskControl(unittest.TestCase):
    def test_pvalue_monotone_and_bounded(self):
        ps = [hb_pvalue(r, 1000, .05) for r in (0., .01, .03, .05, .1)]
        self.assertTrue(all(0 <= p <= 1 for p in ps))
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(ps, ps[1:])))
        self.assertEqual(hb_pvalue(.2, 1000, .05), 1.)

    def test_guarantee_holds_over_many_calibration_draws(self):
        # scores of erroneous units ~ U(0,1): false-clean risk at threshold t is exactly t
        rng = np.random.default_rng(0); alpha, delta, n = .05, .1, 600
        grid = np.linspace(0, .2, 201)
        viol = 0
        for _ in range(400):
            s = rng.random(n)
            risks = np.array([(s < t).mean() for t in grid])
            j, _ = fixed_sequence_threshold(grid, risks, n, alpha, delta)
            true_risk = 0. if j is None else grid[j]
            viol += true_risk > alpha
        self.assertLessEqual(viol / 400, delta)

    def test_fallback_when_nothing_is_safe(self):
        j, _ = fixed_sequence_threshold([1, 2, 3], [.3, .4, .5], 500, .05, .1)
        self.assertIsNone(j)


if __name__ == '__main__':
    unittest.main()
