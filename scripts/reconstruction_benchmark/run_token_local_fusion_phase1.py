#!/usr/bin/env python3
"""Build the target-free Phase-1 token-local score freeze.

This process reads only the restricted localization input mount.  It does not
import the localization evaluator and has no path to ProcessBench or PRMBench
targets.  Label opening lives in a separate post-audit script.
"""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    empirical_midrank,
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from spectral_utils.reconstruction_benchmark.localization_fit import (  # noqa: E402
    _fit_token_iu,
)
from spectral_utils.token_local_fusion import (  # noqa: E402
    CONTROL_METHOD_IDS,
    LOCAL_IU29,
    PRIMARY_METHOD_IDS,
    fit_phase1_ladder,
    prepare_localization_cell,
    step_maxima,
)


PB_CELLS = (
    "processbench_gsm8k_qwen3_4b",
    "processbench_math_qwen3_4b",
    "processbench_olympiadbench_qwen3_4b",
    "processbench_omnimath_qwen3_4b",
    "processbench_gsm8k_qwen3_8b",
    "processbench_math_qwen3_8b",
    "processbench_olympiadbench_qwen3_8b",
    "processbench_omnimath_qwen3_8b",
)
PRM_CELL = "prmbench_response_qwen3_8b"
CELLS = PB_CELLS + (PRM_CELL,)
ALL_METHOD_IDS = PRIMARY_METHOD_IDS + CONTROL_METHOD_IDS
GLOBAL_PRIMARY_METHOD = "equal_feature_mean"
GLOBAL_CONTROL_METHOD = "iu_pcr"
SCHEMA_VERSION = "token-local-fusion-phase1-score-freeze-v1"

SOURCE_FILES = (
    "spectral_utils/token_local_fusion.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/adapted_dufs.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/upcr.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/fixed_application_pipelines.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/repeated_measurement_reliability.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/token_feature_views.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/localization_contract.py",
    "spectral_utils/reconstruction_benchmark/localization_fit.py",
    "scripts/reconstruction_benchmark/run_token_local_fusion_phase1.py",
    "scripts/test_token_local_fusion.py",
    "docs/experiments/TOKEN_LOCAL_FUSION_OPTIMIZATION_V1.md",
)


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _environment_snapshot() -> dict[str, Any]:
    """Return the deterministic runtime contract bound to every score fit."""

    value = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "torch",
                "pandas",
                "deem",
            )
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "LOKY_MAX_CPU_COUNT",
            )
        },
    }
    value["environment_sha256"] = _payload_sha(value)
    return value


def _fit_cell(cell: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    preparation = prepare_localization_cell(cell)
    ladder = fit_phase1_ladder(preparation, include_controls=True)
    if tuple(ladder) != ALL_METHOD_IDS:
        raise AssertionError("Phase-1 fit returned an unexpected method roster")

    incumbent_risk, _ = _fit_token_iu(cell)
    alias_risk = ladder[LOCAL_IU29].token_risk
    alias_byte_exact = bool(np.array_equal(alias_risk, incumbent_risk))
    alias_error = float(np.max(np.abs(alias_risk - incumbent_risk)))
    alias_within_tolerance = bool(alias_error <= 1e-12)
    if not alias_within_tolerance:
        raise RuntimeError(
            f"{cell.cell_id}: LOCAL_IU29 is not a numerical incumbent alias "
            f"(max error {alias_error})"
        )

    raw_step = np.vstack([
        step_maxima(
            ladder[method_id].token_risk,
            cell.segment_starts,
            cell.segment_ends,
        )
        for method_id in ALL_METHOD_IDS
    ])
    step_rank = np.vstack([empirical_midrank(row) for row in raw_step])
    segment_counts = np.diff(cell.segment_offsets)
    method_index = {name: index for index, name in enumerate(cell.method_ids)}
    if GLOBAL_PRIMARY_METHOD not in method_index or GLOBAL_CONTROL_METHOD not in method_index:
        raise RuntimeError("prepared response roster lacks equal-feature or IU global head")
    equal_response = np.asarray(
        cell.response_scores[method_index[GLOBAL_PRIMARY_METHOD]], dtype=np.float64
    )
    iu_response = np.asarray(
        cell.response_scores[method_index[GLOBAL_CONTROL_METHOD]], dtype=np.float64
    )
    equal_response_rank = empirical_midrank(equal_response)
    iu_response_rank = empirical_midrank(iu_response)
    expanded_equal = np.repeat(equal_response_rank, segment_counts)
    expanded_iu = np.repeat(iu_response_rank, segment_counts)
    primary_combined = np.sqrt(step_rank * expanded_equal[None, :])
    iu_global_iu_local = np.sqrt(
        step_rank[ALL_METHOD_IDS.index(LOCAL_IU29)] * expanded_iu
    )
    if (
        primary_combined.shape != raw_step.shape
        or iu_global_iu_local.shape != (raw_step.shape[1],)
        or not np.isfinite(primary_combined).all()
        or not np.isfinite(iu_global_iu_local).all()
    ):
        raise RuntimeError(f"{cell.cell_id}: combined localization scores are invalid")

    # Exact score reconstruction from frozen weights and the one preparation.
    reconstructed_steps = []
    for method_id in ALL_METHOD_IDS:
        result = ladder[method_id]
        reconstructed_token = preparation.token_risk(result.weights)
        if not np.array_equal(reconstructed_token, result.token_risk):
            raise RuntimeError(f"{cell.cell_id}/{method_id}: token score reconstruction failed")
        reconstructed_steps.append(step_maxima(
            reconstructed_token, cell.segment_starts, cell.segment_ends
        ))
    reconstruction_exact = bool(np.array_equal(
        np.vstack(reconstructed_steps), raw_step
    ))
    if not reconstruction_exact:
        raise RuntimeError(f"{cell.cell_id}: step score reconstruction failed")

    diagnostics = {
        "schema_version": "token-local-fusion-phase1-cell-fit-v1",
        "cell_id": cell.cell_id,
        "preparation": dict(preparation.diagnostics),
        "methods": {
            method_id: _jsonable(ladder[method_id].diagnostics)
            for method_id in ALL_METHOD_IDS
        },
        "local_iu_incumbent_alias_byte_exact": alias_byte_exact,
        "local_iu_incumbent_alias_within_1e_12": alias_within_tolerance,
        "local_iu_incumbent_alias_max_abs_error": alias_error,
        "score_reconstruction_exact": reconstruction_exact,
        "same_preparation_for_all_methods": True,
        "step_reducer": "maximum token risk in supplied half-open span",
        "primary_global_method": GLOBAL_PRIMARY_METHOD,
        "primary_combination": (
            "sqrt(midrank(equal_feature_mean response risk) * "
            "midrank(local token span maximum))"
        ),
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
    }
    diagnostics["fit_sha256"] = _payload_sha(diagnostics)
    arrays = {
        "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
        "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
        "method_ids": np.asarray(ALL_METHOD_IDS, dtype="<U48"),
        "kept_stream_mask": np.asarray(preparation.keep, dtype=np.int8),
        "method_weights": np.vstack([
            ladder[method_id].weights for method_id in ALL_METHOD_IDS
        ]).astype("<f8", copy=False),
        "token_step_scores": raw_step.astype("<f8", copy=False),
        "primary_combined_scores": primary_combined.astype("<f8", copy=False),
        "equal_response_score": equal_response.astype("<f8", copy=False),
        "iu_response_score": iu_response.astype("<f8", copy=False),
        "iu_global_iu_local_score": iu_global_iu_local.astype("<f8", copy=False),
    }
    return diagnostics, arrays


def fit(localization_release: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"score-freeze output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    input_root = localization_release / "build_A/localization/inputs"
    input_manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(input_manifest_path, input_root=input_root)
    environment = _environment_snapshot()
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    if not set(CELLS).issubset(by_cell):
        raise RuntimeError("localization input manifest lacks the eight Qwen cells or PRMBench")

    records = []
    for position, cell_id in enumerate(CELLS, start=1):
        input_record = by_cell[cell_id]
        input_path = input_root / str(input_record["artifact_path"])
        cell = load_prepared_localization_cell(input_path, input_record)
        diagnostics, arrays = _fit_cell(cell)
        target = output_root / "cells" / cell_id
        target.mkdir(parents=True, exist_ok=False)
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, arrays)
        record = {
            "schema_version": "token-local-fusion-phase1-cell-record-v1",
            "cell_id": cell_id,
            "population_id": cell.population_id,
            "dataset_id": cell.dataset_id,
            "model_id": cell.model_id,
            "slice_id": cell.slice_id,
            "n_rows": len(cell.row_ids),
            "n_tokens": len(cell.token_confidence),
            "n_steps": len(cell.segment_starts),
            "method_ids": list(ALL_METHOD_IDS),
            "primary_method_ids": list(PRIMARY_METHOD_IDS),
            "control_method_ids": list(CONTROL_METHOD_IDS),
            "prepared_input_sha256": sha256_file(input_path),
            "prepared_record_payload_sha256": _payload_sha(input_record),
            "token_transform_sha256": cell.token_transform_sha256,
            "environment": environment,
            "environment_sha256": environment["environment_sha256"],
            "fit_diagnostics": diagnostics,
            "score_path": score_path.name,
            "score_sha256": score_sha,
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
        }
        record["payload_sha256"] = _payload_sha(record)
        record_path = target / "RECORD.json"
        atomic_write_json(record_path, record)
        records.append({
            "cell_id": cell_id,
            "record_path": record_path.relative_to(output_root).as_posix(),
            "record_sha256": sha256_file(record_path),
            "score_sha256": score_sha,
        })
        print(f"fit {cell_id} ({position}/{len(CELLS)})", flush=True)

    freeze = {
        "schema_version": SCHEMA_VERSION,
        "phase": "linear_spectral_phase1",
        "expected_cells": list(CELLS),
        "processbench_cells": list(PB_CELLS),
        "prmbench_cell": PRM_CELL,
        "method_ids": list(ALL_METHOD_IDS),
        "primary_method_ids": list(PRIMARY_METHOD_IDS),
        "control_method_ids": list(CONTROL_METHOD_IDS),
        "primary_global_method": GLOBAL_PRIMARY_METHOD,
        "all_expected_scores_present": len(records) == len(CELLS),
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
        "independent_prelabel_audit_required": True,
        "environment": environment,
        "environment_sha256": environment["environment_sha256"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "source_snapshot": [
            {"path": name, "sha256": sha256_file(ROOT / name)}
            for name in SOURCE_FILES
        ],
        "records": records,
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(output_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    print(json.dumps({
        "status": "PASS", "stage": "target_free_score_freeze",
        "cells": len(records), "methods": len(ALL_METHOD_IDS),
        "payload_sha256": freeze["payload_sha256"],
    }, indent=2))
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--localization-release", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    fit(Path(args.localization_release).resolve(), Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
