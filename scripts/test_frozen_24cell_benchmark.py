#!/usr/bin/env python3
"""Contract tests for the frozen 24-cell fit/report boundary."""

import json
import os
import sys

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.frozen_24cell_benchmark import (  # noqa: E402
    ALL_GRAPH_ARMS,
    DEFAULT_BUNDLE,
    FROZEN_LAMBDA,
    LAMBDAS,
    _finalize_diagnostics,
    _jsonable,
    _nonfinite_paths,
    bundle_cells,
    lambda_token,
    score_key,
    validate_bundle,
)
from scripts.frozen_24cell_report import (  # noqa: E402
    HEADLINE_METHODS,
    diagnostic_availability_fields,
    headline_summary,
    lambda_summary,
    paired_comparisons,
    promotion_gates,
)
from scripts.inscope_cells import INSCOPE                 # noqa: E402


def main():
    data = np.load(DEFAULT_BUNDLE, allow_pickle=True)
    cells = validate_bundle(data)
    assert cells == tuple(INSCOPE)
    assert bundle_cells(data) == set(INSCOPE)
    assert len(cells) == 24
    assert lambda_token(0.1) == "0p1"
    assert score_key("dufs_liu", 0.1) == "dufs_liu__lambda_0p1"
    assert set(FROZEN_LAMBDA) == {
        "dufs_liu", "adapted_specrage_y_liu", "ca_specrage_alpha_liu"
    }
    unavailable = {
        "liu": {
            "micro__ca_specrage_y_liu": {
                "0": {"algebraic_connectivity": float("nan")}
            }
        }
    }
    expected_path = (
        "$.liu.micro__ca_specrage_y_liu.0.algebraic_connectivity"
    )
    assert _nonfinite_paths(unavailable) == [expected_path]
    finalized = _finalize_diagnostics(unavailable)
    assert finalized["nonfinite_diagnostic_paths"] == [expected_path]
    converted = _jsonable(finalized)
    assert converted["liu"]["micro__ca_specrage_y_liu"]["0"][
        "algebraic_connectivity"
    ] is None
    assert diagnostic_availability_fields(converted) == {
        "nonfinite_diagnostic_count": 1,
        "nonfinite_diagnostic_paths": json.dumps([expected_path]),
    }
    expected = {
        score_key(arm, lambda_) for arm in ALL_GRAPH_ARMS for lambda_ in LAMBDAS
    }
    assert score_key("dufs_liu", 0.1) in expected
    assert set(HEADLINE_METHODS) <= expected | {"deployed_upcr", "iu_pcr"}
    for cell in cells:
        labels = np.asarray(data[f"{cell}__labels"])
        matrix = np.asarray(data[f"{cell}__V"])
        assert labels.ndim == 1 and matrix.shape[0] == len(labels)
        assert len(np.unique(labels)) == 2
    mock_rows = []
    all_keys = expected | {"deployed_upcr", "iu_pcr"}
    headline_gain = {
        key: 0.01 * index for index, key in enumerate(HEADLINE_METHODS)
    }
    for cell in cells:
        for key in all_keys:
            value = 0.70 + headline_gain.get(key, 0.01)
            mock_rows.append({
                "cell": cell,
                "method_key": key,
                "auroc": value,
                "auprc": value - 0.05,
            })
    summary = headline_summary(mock_rows)
    comparisons = paired_comparisons(mock_rows)
    paths = lambda_summary(mock_rows)
    gates = promotion_gates(comparisons)
    assert len(summary) == 2 * len(HEADLINE_METHODS)
    assert len(paths) == len(ALL_GRAPH_ARMS) * len(LAMBDAS)
    assert len(gates) == 8
    assert any(row["reference_key"].startswith("manual__") for row in comparisons)
    print("FROZEN 24-CELL BENCHMARK CONTRACT PASS")


if __name__ == "__main__":
    main()
