#!/usr/bin/env python3
"""Register the executable Phase-2R reference details before result opening."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402


ROOT = REPO / "results/reasoning_localization_03662_v1"
REFERENCE = "P2R_A_TOPK5_REFERENCE"


def main() -> None:
    if (ROOT / "phase_2").exists():
        raise FileExistsError("Phase-2 output already exists; contract can no longer change")

    variant_path = ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variant_path.read_text(encoding="utf-8"))
    reference = next(row for row in variants["variants"] if row["variant_id"] == REFERENCE)
    if reference["execution_status"] != "PLANNED":
        raise RuntimeError("Phase-2R reference is no longer in design state")
    reference["parent_variant_ids"] = ["R1_ENTROPY_TOP5"]
    reference["reference_alias_contract"] = {
        "score_parent": "R1_ENTROPY_TOP5",
        "tolerance": 1e-12,
        "must_match": [
            "local scores", "combined scores", "fold assignments", "predictions",
            "panel metrics", "bootstrap samples",
        ],
        "threshold_export": "reconstruct the exact R1 fold thresholds once, then freeze them for every later reducer",
    }
    atomic_write_json(variant_path, variants)

    experiment_path = ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(
        row for row in experiments["experiments"]
        if row["experiment_id"] == "P2_REDUCER_STUDY"
    )
    if experiment["execution_status"] != "PLANNED":
        raise RuntimeError("Phase-2 reducer experiment is no longer in design state")
    experiment["reducer_contract"]["length_strata"] = (
        "model- and held-fold-specific NumPy-linear 1/3 and 2/3 quantiles of "
        "erroneous calibration rows' true-error-step token length; short <= q1, "
        "medium <= q2, long otherwise; descriptive stratum macro F1 combines "
        "stratum exact-error with unchanged full-cell clean abstention"
    )
    experiment["reducer_contract"]["threshold_rule"] = (
        "P2R_A_TOPK5_REFERENCE first reconstructs R1 fold thresholds and must "
        "alias R1 scores, folds, predictions, metrics, and bootstrap within 1e-12; "
        "all later candidates reuse those frozen thresholds with no candidate-specific rethresholding"
    )
    atomic_write_json(experiment_path, experiments)

    plot_path = ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(plot_path.read_text(encoding="utf-8"))
    length_plot = next(
        row for row in plots["plots"] if row["plot_id"] == "PLOT_P2_REDUCER_LENGTH_HEATMAP"
    )
    length_plot["selection"].pop("unit_type", None)
    length_plot["selection"]["population_id"] = "current_common_eight_qwen_step_length"
    atomic_write_json(plot_path, plots)


if __name__ == "__main__":
    main()
