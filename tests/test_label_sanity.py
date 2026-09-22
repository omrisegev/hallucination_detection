import unittest
import numpy as np
from spectral_utils import label_sanity as ls


class LabelSanityTests(unittest.TestCase):
    def test_healthy_cell_is_ok_and_unflagged(self):
        y = [1] * 120 + [0] * 80
        r = ls.check_labels(y, lengths=[100] * 200, max_new=1024)
        self.assertTrue(r.ok)
        self.assertEqual(r.flags, [])
        self.assertEqual(r.flag_string(), "")
        self.assertEqual((r.n_pos, r.n_neg), (120, 80))

    def test_three_positives_is_degenerate(self):
        y = [1] * 3 + [0] * 197
        r = ls.check_labels(y)
        self.assertFalse(r.ok)
        self.assertTrue(any("minority" in h for h in r.hard))
        self.assertTrue(r.flag_string().startswith("DEGENERATE:"))

    def test_single_class_is_degenerate(self):
        r = ls.check_labels([0] * 30)
        self.assertFalse(r.ok)
        self.assertIn("single-class", r.hard[0])

    def test_acc_099_is_degenerate_even_with_enough_rows(self):
        y = [1] * 990 + [0] * 10
        r = ls.check_labels(y)
        self.assertFalse(r.ok)   # minority 10 passes the hard count, acc 0.99 fails the band

    def test_acc_090_is_ceiling_flag_not_hard(self):
        y = [1] * 900 + [0] * 100
        r = ls.check_labels(y)
        self.assertTrue(r.ok)
        self.assertIn("CEILING", r.flags)
        self.assertEqual(r.flag_string(), "FLAG:CEILING")

    def test_thirty_percent_cap_pinned_with_label_leak_is_degenerate(self):
        # pinned traces are almost all wrong (the Step 168-172 pattern): 60 pinned, 5 correct
        y = [1] * 5 + [0] * 55 + [1] * 95 + [0] * 45
        lengths = [512] * 60 + [200] * 140
        r = ls.check_labels(y, lengths=lengths, max_new=512)
        self.assertFalse(r.ok)
        self.assertAlmostEqual(r.cap_pinned_frac, 0.30)
        self.assertLess(r.cap_leak_diff, -0.15)
        self.assertTrue(any("leaks" in h for h in r.hard))

    def test_cap_pinned_without_label_dependence_only_flags(self):
        # base-model QA cell: half the traces run to the cap, but correctness is unrelated
        y = ([1, 0] * 30) + ([1, 0] * 30)
        lengths = [256] * 60 + [40] * 60
        r = ls.check_labels(y, lengths=lengths, max_new=256)
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.cap_pinned_frac, 0.50)
        self.assertAlmostEqual(r.cap_leak_diff, 0.0)
        self.assertTrue(any("max_new" in f for f in r.flags))

    def test_small_cap_fraction_only_flags(self):
        y = [1] * 100 + [0] * 100
        lengths = [512] * 6 + [200] * 194
        r = ls.check_labels(y, lengths=lengths, max_new=512)
        self.assertTrue(r.ok)
        self.assertTrue(any("max_new" in f for f in r.flags))

    def test_check_pkl_walks_candidates(self):
        data = {i: {"candidates": [{"label": i % 2 == 0, "gen_token_ids": [0] * 50}]} for i in range(40)}
        r = ls.check_pkl(data, max_new=256)
        self.assertTrue(r.ok)
        self.assertEqual(r.n, 40)
        self.assertEqual(r.cap_pinned_frac, 0.0)

    def test_feasibility_tag(self):
        self.assertEqual(ls.feasibility_tag(86, 3400), "FEASIBILITY")
        self.assertEqual(ls.feasibility_tag(3400, 3400), "")
        self.assertEqual(ls.feasibility_tag(10, None), "")

    def test_gate_flag_matches_historical_contract(self):
        self.assertEqual(ls.gate_flag(0.5), "")
        self.assertEqual(ls.gate_flag(0.1), "FLOOR")
        self.assertEqual(ls.gate_flag(0.9), "CEILING")
        self.assertEqual(ls.gate_flag(None), "")
        self.assertEqual(ls.gate_flag(float("nan")), "")

    def test_nan_labels_are_dropped(self):
        y = np.array([1, 0] * 30 + [np.nan] * 5)
        r = ls.check_labels(y)
        self.assertEqual(r.n, 60)


if __name__ == "__main__":
    unittest.main()
