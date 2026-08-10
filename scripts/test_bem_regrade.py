#!/usr/bin/env python3
"""Dependency-free unit tests for bem_regrade.py's cache logic."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("bem_regrade.py")
SPEC = importlib.util.spec_from_file_location("bem_regrade", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BemRegradeTest(unittest.TestCase):
    def test_expand_and_max_over_references(self):
        cache = {
            7: {
                "question": "Capital of France?",
                "gold_row": {"truthful_answers": ["Paris", "The city of Paris"]},
                "candidates": [{"full_text": "Paris", "label": False}],
            },
            8: {
                "question": "Two plus two?",
                "gold_row": {"truthful_answers": ["four"]},
                "candidates": [{"full_text": "4", "label": True}],
            },
        }
        candidates = list(MODULE.iter_candidates(cache))
        examples, groups = MODULE.expand_examples(candidates)
        self.assertEqual(len(examples), 3)
        self.assertEqual(MODULE.aggregate_max([0.8, 0.9, 0.7], groups), [0.9, 0.7])

    def test_restricted_pickle_round_trip_for_plain_cache(self):
        import pickle

        cache = {
            1: {
                "question": "q",
                "gold_row": {"truthful_answers": ["a"]},
                "candidates": [{"full_text": "a"}],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            path.write_bytes(pickle.dumps(cache))
            self.assertEqual(MODULE.load_cache(path), cache)

    def test_restricted_pickle_supports_numpy_arrays(self):
        import pickle
        import numpy as np

        cache = {1: {"values": np.asarray([1.0, 2.0], dtype=np.float32)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
            loaded = MODULE.load_cache(path)
            np.testing.assert_array_equal(loaded[1]["values"], cache[1]["values"])


if __name__ == "__main__":
    unittest.main()
