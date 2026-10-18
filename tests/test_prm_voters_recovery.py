import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('voters', ROOT / 'scripts/run_prm_voters_prmbench.py')
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)


class Recovery(unittest.TestCase):
    def test_recovers_exact_per_step_errors_and_reports_gap(self):
        p_err = [.1, .3, .2]
        scores, cum = {}, 0.0
        for j, p in enumerate(p_err, 1):
            scores[j] = cum + math.log(p)
            cum += math.log(1 - p)
        scores[len(p_err) + 1] = cum
        lp, gap = V.recover_step_logp_err(scores)
        self.assertEqual(len(lp), 3)
        for a, b in zip(lp, p_err):
            self.assertAlmostEqual(math.exp(a), b, places=9)
        self.assertLess(gap, 1e-9)
        scores[4] += 0.002          # bf16-scale disagreement in S(T+1)
        self.assertAlmostEqual(V.recover_step_logp_err(scores)[1], 0.002, places=9)


if __name__ == '__main__':
    unittest.main()
