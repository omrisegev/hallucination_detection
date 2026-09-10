"""Mechanism tests for the readout controls and numeral provenance. No benchmark data."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils import provenance_readout as pr  # noqa: E402


class NumeralTests(unittest.TestCase):
    def test_runs_and_joiners(self):
        toks = ["x", " ", "1", "2", " =", " ", "3", ".", "5", " and", " ", "1", ",", "0", "0", "0", " ", "7"]
        runs = pr.numeral_runs(toks)
        self.assertEqual([(a, b, lit) for a, b, lit in runs], [(2, 4, "12"), (6, 9, "3.5"), (11, 16, "1000")])
        # single digit '7' dropped; leading-space digit starts a run
        toks2 = [" 4", "2", "x"]
        self.assertEqual(pr.numeral_runs(toks2), [(0, 2, "42")])
        # a joiner not followed by a digit ends the run
        self.assertEqual(pr.numeral_runs(["1", "2", ".", " x"]), [(0, 2, "12")])

    def test_given_literals(self):
        g = pr.given_literals("A class of 50 students, 1,000 apples, 3.5 hours, 7 cats")
        self.assertEqual(g, {"50", "1000", "3.5"})

    def test_origin_and_given(self):
        # step0: "50 students"; step1: "2 x 5 = 10"; step2: "50 - 10 = 40"; step3: "40 + 12"
        toks = ["50"[0], "50"[1], " s", "|", "2", " x", " ", "5", " =", " ", "1", "0", "|", " ", "5", "0", " -", " ", "1", "0", " =", " ", "4", "0", "|", " ", "4", "0", " +", " ", "1", "2"]
        starts = np.array([0, 4, 13, 25]); ends = np.array([4, 13, 25, len(toks)])
        nums = pr.build_numerals(toks, starts, ends, given={"50"})
        lits = [(x.literal, x.step, x.origin, x.given) for x in nums]
        self.assertEqual(lits, [("50", 0, 0, True), ("10", 1, 1, False), ("50", 2, 0, True),
                                ("10", 2, 1, False), ("40", 2, 2, False), ("40", 3, 2, False), ("12", 3, 3, False)])
        r = pr.reassign(nums, starts, ends, len(toks), "reassign")
        self.assertEqual(r.inherited_numerals, 2)      # "10" in step2, "40" in step3
        self.assertEqual(r.moved_tokens, 4)
        moved = {t: s for t, s in r.pairs}
        self.assertEqual(moved[18], 1); self.assertEqual(moved[19], 1)   # "10" tokens -> step1
        self.assertEqual(moved[26], 2); self.assertEqual(moved[27], 2)   # "40" tokens -> step2
        self.assertEqual(moved[14], 2)                                   # given "50" stays in step2
        d = pr.reassign(nums, starts, ends, len(toks), "duplicate")
        self.assertEqual(len(d.pairs), len(toks) + 4)
        s = pr.reassign(nums, starts, ends, len(toks), "shuffled", rng=np.random.default_rng(0))
        self.assertEqual(s.moved_tokens, 4)
        for t, dest in s.pairs:
            if t in (18, 19):
                self.assertLess(dest, 2)
            if t in (26, 27):
                self.assertLess(dest, 3)

    def test_step_scores_from_pairs_and_floor(self):
        tokens = np.arange(6, dtype=float)
        pairs = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1)]
        s = pr.step_scores_from_pairs(tokens, pairs, 3, k=2)
        np.testing.assert_allclose(s, [0.5, 4.5, -1.0])
        self.assertTrue(np.isfinite(s).all())


class ReadoutTests(unittest.TestCase):
    def test_top_k_matches_reference_rule(self):
        x = np.random.default_rng(1).normal(size=37)
        self.assertAlmostEqual(pr.top_k_mean(x), np.sort(x)[-10:].mean())
        self.assertAlmostEqual(pr.top_k_mean(x[:6]), x[:6].mean())

    def test_onset_rules(self):
        s = np.array([1.0, 1.1, 1.5, 1.52, 1.0])
        self.assertEqual(int(np.argmax(pr.rise_vs_history(s))), 2)
        f = pr.first_near_max(s, 0.25)
        self.assertEqual(int(np.argmax(f)), 2)                # earliest near-max (1.5 within .25 sd of 1.52)
        self.assertEqual(list(np.argsort(f)[:3]), list(np.argsort(s)[:3]))  # other ranks unchanged

    def test_controls(self):
        st = np.array([0, 5, 30]); en = np.array([5, 30, 31])
        np.testing.assert_array_equal(pr.length_scores(st, en), [5, 25, 1])
        self.assertEqual(int(np.argmax(pr.position_first_scores(4))), 0)
        r = pr.random_scores(4, np.random.default_rng(2026091101))
        self.assertEqual(r.shape, (4,))



class AttributionTests(unittest.TestCase):
    def _nums(self):
        # steps: 0 given "50"; 1 computes "10"; 2 uses "10" (from 1) and computes "40"; 3 uses "40" (from 2) and "10" (from 1)
        N = pr.Numeral
        return [N(0, 2, "50", 0, 0, True), N(4, 6, "10", 1, 1), N(8, 10, "10", 2, 1), N(12, 14, "40", 2, 2),
                N(16, 18, "40", 3, 2), N(20, 22, "10", 3, 1)]

    def test_dependency_counts(self):
        c = pr.dependency_counts(self._nums())
        self.assertEqual(c, {(2, 1): 1, (3, 2): 1, (3, 1): 1})

    def test_attribution_conservation_and_weights(self):
        z = np.array([0.0, -1.0, 0.5, 2.0])
        c = pr.dependency_counts(self._nums())
        out = pr.attribute_step_mass(z, c, alpha=0.5)
        # step2 sends 0.25 to step1; step3 sends 1.0 split 0.5/0.5 to steps 1 and 2
        np.testing.assert_allclose(out, [0.0, -1.0 + 0.25 + 0.5, 0.5 - 0.25 + 0.5, 2.0 - 1.0])
        self.assertAlmostEqual(out.sum(), z.sum())            # mass conserved
        full = pr.attribute_step_mass(z, c, alpha=1.0)
        np.testing.assert_allclose(full, [0.0, -1.0 + 0.5 + 1.0, 0.5 - 0.5 + 1.0, 0.0])
        self.assertEqual(int(np.argmax(full)), 2)            # attribution can move the argmax earlier
        uni = pr.attribute_step_mass(z, c, alpha=0.5, mode="uniform")
        self.assertAlmostEqual(uni.sum(), z.sum())
        np.testing.assert_allclose(uni, [0.0 + 0.125 + 1/3, -1.0 + 0.125 + 1/3, 0.5 - 0.25 + 1/3, 1.0])
        sh = pr.attribute_step_mass(z, c, alpha=0.5, mode="shuffled", rng=np.random.default_rng(3))
        self.assertAlmostEqual(sh.sum(), z.sum())
        self.assertAlmostEqual(sh[3], 1.0)                    # sender keeps (1-alpha) of its mass
        with self.assertRaises(ValueError):
            pr.attribute_step_mass(z, c, alpha=0.5, mode="shuffled")

    def test_zscore_argmax_identity(self):
        s = np.random.default_rng(5).normal(size=9)
        self.assertEqual(int(np.argmax(pr.zscore_steps(s))), int(np.argmax(s)))
        np.testing.assert_array_equal(pr.zscore_steps(np.ones(4)), np.zeros(4))
        out = pr.attribute_step_mass(pr.zscore_steps(s), {}, alpha=0.5)
        self.assertEqual(int(np.argmax(out)), int(np.argmax(s)))  # no parents -> unchanged

if __name__ == "__main__":
    unittest.main()
