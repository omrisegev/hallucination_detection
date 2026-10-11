"""CPU test of spectral_utils.first_token_mask (the 2026-10-10 "mask1" rule)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spectral_utils.first_token_mask import MaskError, mask_cache, mask_first_token  # noqa: E402


def entry(n=6, spans=((0, 2), (2, 4), (4, 6))):
    return {
        "gen_token_ids": list(range(100, 100 + n)),
        "token_entropies": [float(i) for i in range(n)],
        "token_spilled_energies": np.arange(n, dtype=np.float64) + 10,
        "token_logsumexp": np.arange(n, dtype=np.float32) + 20,
        "top_k_logprobs": {"ids": np.arange(n * 3).reshape(n, 3).astype(np.int32),
                           "logprobs": -np.ones((n, 3), np.float32)},
        "step_token_spans": [None if s is None else tuple(s) for s in spans],
        "align_diag": {"step_token_lengths": [0 if s is None else s[1] - s[0] for s in spans],
                       "problems": []},
        "steps": ["a", "b", "c"],
    }


class TestMask1(unittest.TestCase):
    def test_fields_and_spans(self):
        e = entry()
        m = mask_first_token(e)
        self.assertEqual(m["gen_token_ids"], [101, 102, 103, 104, 105])
        self.assertEqual(m["token_entropies"], [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(m["token_spilled_energies"], np.arange(1, 6) + 10)
        np.testing.assert_array_equal(m["token_logsumexp"], np.arange(1, 6, dtype=np.float32) + 20)
        np.testing.assert_array_equal(m["top_k_logprobs"]["ids"], np.arange(18).reshape(6, 3)[1:])
        self.assertEqual(m["top_k_logprobs"]["logprobs"].shape, (5, 3))
        self.assertEqual(m["step_token_spans"], [(0, 1), (1, 3), (3, 5)])
        self.assertEqual(m["align_diag"]["step_token_lengths"], [1, 2, 2])
        self.assertEqual(m["first_token_mask"]["dropped_token_id"], 100)
        # spans still index the same tokens as before (shifted by one)
        for (lo, hi), (olo, ohi) in zip(m["step_token_spans"][1:], e["step_token_spans"][1:]):
            self.assertEqual(m["gen_token_ids"][lo:hi], e["gen_token_ids"][olo:ohi])

    def test_input_unchanged(self):
        e = entry()
        mask_first_token(e)
        self.assertEqual(len(e["gen_token_ids"]), 6)
        self.assertEqual(e["step_token_spans"][0], (0, 2))
        self.assertNotIn("first_token_mask", e)

    def test_empty_step0_is_explicit_failure(self):
        e = entry(spans=((0, 1), (1, 4), (4, 6)))
        with self.assertRaises(MaskError):
            mask_first_token(e)
        masked, failures = mask_cache({"x": e, "y": entry()})
        self.assertEqual(list(masked), ["y"])
        self.assertEqual(failures[0]["key"], "x")
        self.assertIn("step 0", failures[0]["error"])

    def test_length_mismatch_raises(self):
        e = entry()
        e["token_entropies"] = e["token_entropies"][:-1]
        with self.assertRaises(MaskError):
            mask_first_token(e)

    def test_unmapped_step_and_extra_field(self):
        e = entry(spans=((0, 3), None, (3, 6)))
        e["token_offsets"] = [(i, i + 1) for i in range(6)]
        e["my_extra"] = list(range(6))
        m = mask_first_token(e, extra_per_token=("my_extra",))
        self.assertEqual(m["step_token_spans"], [(0, 2), None, (2, 5)])
        self.assertEqual(m["token_offsets"][0], (1, 2))
        self.assertEqual(m["my_extra"], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
