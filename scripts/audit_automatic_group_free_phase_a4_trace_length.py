#!/usr/bin/env python3
"""Post-held adversarial trace-length diagnostic for frozen Phase A4.

This audit is deliberately diagnostic.  It reconstructs the already frozen
outer fits, never reads correctness or step-error targets, performs no refit or
reselection after removing ``trace_length``, and cannot change either A4
verdict.  Its purpose is to test the mechanistic interpretation of the held
result after the registered strongest baseline selected ``single:1`` in every
fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.automatic_group_free_phase_a4 import (  # noqa: E402
    DEFAULT_OUT,
    SEED,
    _outer_fit,
    load_and_verify_boundary,
    sha256_file,
    write_json,
)
from spectral_utils.paired_repeatability import (  # noqa: E402
    FEATURE_ROSTER,
    SUBSETS,
    correlation_summary,
    prepare_fold,
    safe_correlation,
)


VERSION = "automatic-group-free-iu-a4-trace-length-post-held-v1-2026-08-13"


def run(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    boundary, arrays = load_and_verify_boundary(output_dir)
    fitted = _outer_fit(arrays, transform_kind="mixed_v2", baselines=True)
    baseline_keys = [row["strongest_baseline"] for row in fitted["selections"]]
    if baseline_keys != ["single:1"] * 5:
        raise RuntimeError(f"unexpected frozen strongest baselines: {baseline_keys}")

    trace_index = FEATURE_ROSTER.index("trace_length")
    n_items = len(arrays["raw"])
    trace_only = np.full((n_items, 3), np.nan)
    trace_ablated = np.full((n_items, 3), np.nan)
    component_trace_correlations = []
    loading_diagnostics = []
    fold_zero_residual_check = None

    for fold, loading in enumerate(fitted["loadings"]):
        held = np.flatnonzero(arrays["outer_fold"] == fold)
        train = np.flatnonzero(arrays["outer_fold"] != fold)
        prepared = prepare_fold(
            arrays["raw"][train],
            arrays["covariates"][train],
            arrays["subset"][train],
            arrays["group_id"][train],
            arrays["raw"][held],
            arrays["covariates"][held],
            arrays["subset"][held],
            FEATURE_ROSTER,
            transform_kind="mixed_v2",
            split_seed=SEED + 3000 * fold,
        )
        trace_only[held] = (
            prepared.held_residuals[:, :, trace_index] * loading[trace_index]
        )
        ablated_loading = loading.copy()
        ablated_loading[trace_index] = 0.0
        trace_ablated[held] = prepared.held_residuals @ ablated_loading

        original_scores = prepared.held_residuals @ loading
        for view_index, view in enumerate(("qwen3_4b", "qwen3_8b", "llama31_8b")):
            component_trace_correlations.append({
                "fold": fold,
                "view": view,
                "correlation": safe_correlation(
                    original_scores[:, view_index], trace_only[held, view_index]
                ),
            })

        other = np.delete(np.abs(loading), trace_index)
        loading_diagnostics.append({
            "fold": fold,
            "trace_length_loading": loading[trace_index],
            "largest_other_absolute_loading": float(other.max()),
        })

        if fold == 0:
            fold_zero_residual_check = {
                "qwen_max_absolute_difference": {
                    "train_cross_fitted": float(np.max(np.abs(
                        prepared.train_residuals[:, 0, trace_index]
                        - prepared.train_residuals[:, 1, trace_index]
                    ))),
                    "train_full_fit": float(np.max(np.abs(
                        prepared.train_full_residuals[:, 0, trace_index]
                        - prepared.train_full_residuals[:, 1, trace_index]
                    ))),
                    "held": float(np.max(np.abs(
                        prepared.held_residuals[:, 0, trace_index]
                        - prepared.held_residuals[:, 1, trace_index]
                    ))),
                },
                "trace_residual_standard_deviation": {
                    "train_cross_fitted_qwen4": float(np.std(
                        prepared.train_residuals[:, 0, trace_index]
                    )),
                    "train_full_fit_qwen4": float(np.std(
                        prepared.train_full_residuals[:, 0, trace_index]
                    )),
                    "held_qwen4": float(np.std(
                        prepared.held_residuals[:, 0, trace_index]
                    )),
                    "held_llama": float(np.std(
                        prepared.held_residuals[:, 2, trace_index]
                    )),
                },
            }

    if not np.isfinite(trace_only).all() or not np.isfinite(trace_ablated).all():
        raise RuntimeError("post-held trace diagnostic predictions are incomplete")

    raw_trace = arrays["raw"][:, :, trace_index]
    trace_summary = correlation_summary(
        trace_only, arrays["outer_fold"], arrays["subset"]
    )
    strongest_summary = fitted["strongest_summary"]
    correlation_errors = [
        abs(left["correlation"] - right["correlation"])
        for key in ("qwen_cells", "llama_cells")
        for left, right in zip(trace_summary[key], strongest_summary[key])
    ]
    payload = {
        "version": VERSION,
        "status": "POST_HELD_ADVERSARIAL_DIAGNOSTIC_ONLY",
        "changes_frozen_verdict_or_selection": False,
        "correctness_or_step_targets_accessed": False,
        "audit_source_sha256": sha256_file(Path(__file__)),
        "boundary_sha256": sha256_file(output_dir / "A4_BOUNDARY.json"),
        "tensor_sha256": boundary["tensor"]["sha256"],
        "feature": {
            "index": trace_index,
            "name": FEATURE_ROSTER[trace_index],
            "registered_baseline_key": "single:1",
            "definition": "exact generated-token count",
            "selected_in_outer_folds": baseline_keys,
        },
        "raw_trace_length": {
            "qwen4_qwen8_max_absolute_difference": float(np.max(np.abs(
                raw_trace[:, 0] - raw_trace[:, 1]
            ))),
            "qwen_consensus_llama_correlation": safe_correlation(
                raw_trace[:, :2].mean(axis=1), raw_trace[:, 2]
            ),
        },
        "frozen_loading_diagnostics": loading_diagnostics,
        "component_trace_correlations_by_fold_view": component_trace_correlations,
        "component_trace_correlations_global_diagnostic": {
            view: safe_correlation(
                fitted["candidate_scores"][:, view_index], trace_only[:, view_index]
            )
            for view_index, view in enumerate(("qwen3_4b", "qwen3_8b", "llama31_8b"))
        },
        "registered_candidate": fitted["candidate_summary"],
        "trace_only": trace_summary,
        "trace_only_matches_registered_strongest_baseline_max_absolute_correlation_error": float(
            max(correlation_errors)
        ),
        "fixed_loading_trace_ablation_no_refit": correlation_summary(
            trace_ablated, arrays["outer_fold"], arrays["subset"]
        ),
        "fold_zero_residual_check": fold_zero_residual_check,
        "interpretation": (
            "The frozen CorrCA coordinate is dominated by an incompletely removed "
            "trace-length residual. The formal confound gate remains a registered "
            "PASS but cannot support a non-length interpretation because it reuses "
            "the same restricted nuisance basis. This post-held audit does not "
            "rescue, refit, or alter A4."
        ),
    }
    write_json(output_dir / "POST_HELD_TRACE_LENGTH_DIAGNOSTIC.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    payload = run(args.out)
    print({
        "trace_loading": [
            row["trace_length_loading"]
            for row in payload["frozen_loading_diagnostics"]
        ],
        "trace_only_qwen": payload["trace_only"]["qwen_macro"],
        "trace_only_llama": payload["trace_only"]["llama_macro"],
        "ablated_qwen": payload["fixed_loading_trace_ablation_no_refit"]["qwen_macro"],
        "ablated_llama": payload["fixed_loading_trace_ablation_no_refit"]["llama_macro"],
    })


if __name__ == "__main__":
    main()
