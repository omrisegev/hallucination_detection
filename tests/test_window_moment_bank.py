"""Item 4 (2026-09-23): the 8-token moment bank, its geometry, labels and the PR statistic."""
from __future__ import annotations

import re

import numpy as np
import pytest

from lever_imports import ensure_spectral_package

ROOT = ensure_spectral_package()
from spectral_utils import window_moment_bank as wb  # noqa: E402
from spectral_utils.window_localization import make_window_plan  # noqa: E402


def test_self_test():
    res = wb.self_test(0)
    assert res["fit_windows"] == 7


def test_width_guard_of_the_original_plan_is_left_alone():
    with pytest.raises(ValueError):
        make_window_plan(64, 8)
    plan = wb.window_plan(64, 8, 8)
    assert len(plan.starts) == 8 and np.array_equal(plan.starts, np.arange(0, 64, 8))


def test_short_trace_and_stride():
    with pytest.raises(ValueError):
        wb.window_plan(5, 8)
    plan = wb.window_plan(19, 8, 4)
    assert list(plan.starts) == [0, 4, 8, 11] and list(plan.fit_indices) == [0, 2]


def test_module_reads_no_labels_in_builders():
    src = (ROOT / "spectral_utils" / "window_moment_bank.py").read_text(encoding="utf8")
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
    # the only label-touching function is window_labels (a label carrier for the measurement)
    assert re.search(r"\b(target|error_steps|np\.load|pickle)\b", code) is None
