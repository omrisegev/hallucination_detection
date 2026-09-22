from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/joint_lsml_localization/evaluate_existing_v1_r1.py"


def _module():
    spec = importlib.util.spec_from_file_location("joint_eval_amendment_r1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixtures():
    scores = {
        "row_ids": np.asarray(["a", "b", "c", "d"]),
        "segment_offsets": np.asarray([0, 2, 5, 6, 8]),
    }
    labels = {
        "response_row_ids": np.asarray(["a", "c", "d"]),
        "step_offsets": np.asarray([0, 2, 3, 5]),
        "step_labels": np.asarray([0, 1, 0, 1, 0]),
    }
    return scores, labels


def test_canonical_subset_join_accepts_score_only_correct_control():
    module = _module()
    scores, labels = _fixtures()
    result = module.validate_prm_subset_join(
        scores, labels, expected_score_responses=4, expected_label_responses=3,
        expected_score_only_responses=1, expected_score_spans=8, expected_label_steps=5,
    )
    assert result["n_score_only_responses"] == 1
    np.testing.assert_array_equal(result["selected_indices"], [0, 2, 3])


def test_subset_join_rejects_missing_label_id():
    module = _module()
    scores, labels = _fixtures()
    labels["response_row_ids"] = np.asarray(["a", "missing", "d"])
    try:
        module.validate_prm_subset_join(
            scores, labels, expected_score_responses=4, expected_label_responses=3,
            expected_score_only_responses=1, expected_score_spans=8, expected_label_steps=5,
        )
    except module.AmendmentError:
        pass
    else:
        raise AssertionError("missing label ID was accepted")


def test_subset_join_rejects_step_count_drift():
    module = _module()
    scores, labels = _fixtures()
    labels["step_offsets"] = np.asarray([0, 2, 4, 5])
    try:
        module.validate_prm_subset_join(
            scores, labels, expected_score_responses=4, expected_label_responses=3,
            expected_score_only_responses=1, expected_score_spans=8, expected_label_steps=5,
        )
    except module.AmendmentError:
        pass
    else:
        raise AssertionError("step-count drift was accepted")


def test_subset_join_rejects_label_order_not_score_subsequence():
    module = _module()
    scores, labels = _fixtures()
    labels["response_row_ids"] = np.asarray(["c", "a", "d"])
    labels["step_offsets"] = np.asarray([0, 1, 3, 5])
    try:
        module.validate_prm_subset_join(
            scores, labels, expected_score_responses=4, expected_label_responses=3,
            expected_score_only_responses=1, expected_score_spans=8, expected_label_steps=5,
        )
    except module.AmendmentError:
        pass
    else:
        raise AssertionError("non-subsequence label order was accepted")

