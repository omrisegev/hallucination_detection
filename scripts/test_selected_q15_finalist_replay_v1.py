"""Small mechanism tests for the selected q15 finalist replay."""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_selected_q15_finalist_replay_v1 import compose_q15_raw


def run():
    views = [
        np.asarray([1.0, 5.0, 9.0]),
        np.asarray([2.0, 6.0, 10.0]),
        np.asarray([3.0, 7.0, 11.0]),
        np.asarray([4.0, 8.0, 12.0]),
    ]
    np.testing.assert_array_equal(compose_q15_raw(views), np.asarray([2.5, 6.5, 10.5]))
    shifted = [value + offset for value, offset in zip(views, (10.0, 20.0, 30.0, 40.0))]
    np.testing.assert_array_equal(compose_q15_raw(shifted), np.asarray([27.5, 31.5, 35.5]))
    try:
        compose_q15_raw(views[:3])
    except ValueError:
        pass
    else:
        raise AssertionError("three-view input was not rejected")
    try:
        compose_q15_raw([*views[:3], np.asarray([np.nan, 1.0, 2.0])])
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite input was not rejected")
    return {"status": "PASS", "tests": 4}


if __name__ == "__main__":
    print(run())
