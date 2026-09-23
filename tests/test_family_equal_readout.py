"""Item 2 (2026-09-23): partition-then-equal readouts on planted families."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from lever_imports import ensure_spectral_package

ROOT = ensure_spectral_package()
from spectral_utils import family_equal_readout as fe  # noqa: E402


def test_self_test_recovers_planted_partition_and_identities():
    res = fe.self_test(seed=0)
    assert res["selected_k"] == 2


def test_fixed_weights_and_family_means():
    g = np.array([0, 0, 0, 0, 1, 1, 2])
    w = fe.fixed_partition_weights(g)
    assert np.isclose(w.sum(), 1.0)
    x = np.arange(21, dtype=float).reshape(3, 7)
    V = fe.family_means(x, g)
    assert V.shape == (3, 3) and np.allclose(V[:, 2], x[:, 6]) and np.allclose(V[:, 0], x[:, :4].mean(1))
    assert np.allclose(x @ w, V.mean(1))


def test_answer_rule_zeroes_constant_family_and_reports_counts():
    off = np.array([0, 3, 5])
    V = np.array([[1., 0.], [2., 0.], [4., 0.], [1., 5.], [3., 5.]])
    z, info = fe.answer_restandardize(V, off)
    assert np.allclose(z[:, 1], 0.0) and info["zeroed_family_answers"] == 2 and info["two_step_answers"] == 1
    assert np.allclose(np.abs(z[3:5, 0]), 1.0)


def test_module_never_reads_labels_or_data():
    src = (ROOT / "spectral_utils" / "family_equal_readout.py").read_text(encoding="utf8")
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"""', "*")))
    # no error labels, no first-error targets, no file reads: a pure function module
    assert re.search(r"\b(target|error_steps|np\.load|pickle|open\()", body) is None
