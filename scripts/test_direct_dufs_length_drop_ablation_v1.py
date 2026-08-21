#!/usr/bin/env python3
import unittest

import numpy as np

from scripts.direct_dufs_length_drop_ablation_v1 import length_mask, summarize


class LengthDropTests(unittest.TestCase):
    def test_length_mask_removes_all_named_channels(self):
        mask = length_mask(
            ["full::epr", "full::trace_length", "loo::epr"],
            ["epr", "trace_length", "epr"],
        )
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_summary_uses_effect_size(self):
        rows = []
        for condition in ("original", "drop_length_fixed_gates", "drop_length_refit_gates"):
            rows.append({
                "lane": "global24", "condition": condition,
                "target_smoothness_effect": 0.2,
                "length_smoothness_effect": 0.7,
            })
        for lane in ("processbench", "ragtruth"):
            for condition in ("original", "drop_length_fixed_gates", "drop_length_refit_gates"):
                rows.append({
                    "lane": lane, "condition": condition,
                    "target_smoothness_effect": 0.4,
                    "length_smoothness_effect": 0.1,
                })
        result = summarize(rows)
        global_original = next(row for row in result if row["lane"] == "global24" and row["condition"] == "original")
        self.assertEqual(global_original["fraction_target_smoother_than_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
