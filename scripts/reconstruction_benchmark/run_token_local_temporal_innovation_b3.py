#!/usr/bin/env python3
"""Build the target-free Phase-2 B3/temporal-innovation score freeze.

This runner is deliberately score-only.  It validates the shared prepared
input manifest through the token-only contract and materializes token/span/row
members only; response-risk and evaluator/label modules are outside this
process.  A later evaluator may join these token scores to a separately frozen
response head by opaque row hashes after the independent pre-label audit.
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
    FIT_TOKEN_CAP,
    load_prepared_localization_token_cell,
    validate_fit_manifest,
)
from spectral_utils.token_temporal_innovation_b3 import (  # noqa: E402
    LOCAL_TOKEN_B3,
    LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL,
    LOCAL_TOKEN_B3_ROOK_ALL_INNOV,
    LOCAL_TOKEN_B3_ROOK_PSTG_INNOV,
    LOCAL_TOKEN_B3_SELF_INNOV,
    METHOD_IDS,
    fit_token_b3_ladder,
    innovation_map_record,
)
from spectral_utils.token_local_fusion import prepare_localization_cell, step_maxima  # noqa: E402


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
SCHEMA_VERSION = "token-local-temporal-innovation-b3-score-freeze-v1"

# Keep the closure explicit.  These are scientific/transitive modules used by
# the runner and B3 implementation, plus the generated initializer that Python
# executes on import.  No evaluator or raw/label loader belongs in this roster.
SOURCE_FILES = (
    "spectral_utils/__init__.py",
    "spectral_utils/io_utils.py",
    "spectral_utils/model_utils.py",
    "spectral_utils/data_loaders.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/contribution_subspace.py",
    "spectral_utils/contextual_stg.py",
    "spectral_utils/upcr_clustered.py",
    "spectral_utils/deem_adapter.py",
    "spectral_utils/diagnostics.py",
    "spectral_utils/baselines.py",
    "spectral_utils/judge_utils.py",
    "spectral_utils/bem_scorer.py",
    "spectral_utils/streaming_utils.py",
    "spectral_utils/anomaly_utils.py",
    "spectral_utils/agent_utils.py",
    "spectral_utils/adapted_dufs.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/temporal_models.py",
    "spectral_utils/token_temporal_innovation_b3.py",
    "spectral_utils/token_local_fusion.py",
    "spectral_utils/residual_graph_deem.py",
    "spectral_utils/fixed_application_pipelines.py",
    "spectral_utils/token_feature_views.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/repeated_measurement_reliability.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/upcr.py",
    "spectral_utils/graph_topology.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/reconstruction_benchmark/__init__.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/localization_contract.py",
    "scripts/reconstruction_benchmark/run_token_local_temporal_innovation_b3.py",
    "scripts/test_token_temporal_innovation_b3.py",
    "docs/experiments/TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1.md",
)

METHODS = (
    LOCAL_TOKEN_B3,
    LOCAL_TOKEN_B3_SELF_INNOV,
    LOCAL_TOKEN_B3_ROOK_ALL_INNOV,
    LOCAL_TOKEN_B3_ROOK_PSTG_INNOV,
    LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL,
)


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return sha256_bytes(array.tobytes(order="C"))


def _jsonable(value: Any, *, key: str | None = None) -> Any:
    """Canonicalize fit records and remove wall-clock-only diagnostics."""

    # Runtime is useful interactively but would make independent A/B freezes
    # differ for no scientific reason.
    if key == "runtime_seconds":
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(name): _jsonable(item, key=str(name))
            for name, item in value.items()
            if str(name) != "runtime_seconds"
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _environment_snapshot() -> dict[str, Any]:
    value = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "pandas", "deem")
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
            )
        },
    }
    value["environment_sha256"] = _payload_sha(value)
    return value


def _step_scores(cell: Any, token_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = [
        step_maxima(scores, cell.segment_starts, cell.segment_ends)
        for scores in (token_scores[method] for method in METHODS)
    ]
    output = np.vstack(rows).astype("<f8", copy=False)
    if output.shape != (len(METHODS), len(cell.segment_starts)):
        raise RuntimeError("Phase-2 token-to-step score shape drifted")
    return output


def _fit_cell(
    cell: Any, *, execution_workers: int = 1,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    preparation = prepare_localization_cell(cell)
    ladder = fit_token_b3_ladder(
        preparation, execution_workers=int(execution_workers)
    )
    if tuple(ladder) != METHODS:
        raise AssertionError("Phase-2 B3 method roster drifted")
    b3_spearman = float(ladder[LOCAL_TOKEN_B3].health["median_seed_spearman"])
    if not np.isfinite(b3_spearman) or b3_spearman < 0.90:
        raise RuntimeError(
            f"{cell.cell_id}: B3 median seed Spearman gate failed ({b3_spearman:.6f})"
        )
    token_scores = {
        method: np.asarray(ladder[method].token_risk, dtype="<f8")
        for method in METHODS
    }
    if any(scores.shape != (len(cell.token_confidence),) for scores in token_scores.values()):
        raise RuntimeError(f"{cell.cell_id}: incomplete token score vector")
    if any(not np.isfinite(scores).all() for scores in token_scores.values()):
        raise RuntimeError(f"{cell.cell_id}: non-finite token score vector")
    step_scores = _step_scores(cell, token_scores)
    result_records = {}
    for method in METHODS:
        result = ladder[method]
        result_records[method] = {
            "method_id": method,
            "schema_version": "token-local-temporal-innovation-b3-result-v1",
            "token_risk_sha256": _array_sha(result.token_risk),
            "per_seed_model_records": _jsonable(result.per_seed_model_records),
            "innovation_maps": [
                innovation_map_record(value) for value in result.innovation_maps
            ],
            "fold_diagnostics": _jsonable(result.fold_diagnostics),
            "health": _jsonable(result.health),
        }
    diagnostics = {
        "schema_version": "token-local-temporal-innovation-b3-cell-fit-v1",
        "cell_id": cell.cell_id,
        "preparation": _jsonable(preparation.diagnostics),
        "methods": result_records,
        "method_ids": list(METHODS),
        "fold_count": 5,
        "seed_roster": [0, 1, 2, 3, 4],
        "execution_workers": int(execution_workers),
        "fit_token_cap": FIT_TOKEN_CAP,
        "step_reducer": "maximum token risk in supplied half-open span",
        "token_scores_reconstruction_required": True,
        "response_scores_materialized": False,
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
    }
    diagnostics["fit_sha256"] = _payload_sha(diagnostics)
    arrays = {
        "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
        "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
        "segment_starts": np.asarray(cell.segment_starts, dtype="<i8"),
        "segment_ends": np.asarray(cell.segment_ends, dtype="<i8"),
        "method_ids": np.asarray(METHODS, dtype="<U64"),
        "token_scores": np.vstack([token_scores[method] for method in METHODS]),
        "token_step_scores": step_scores,
    }
    return diagnostics, arrays


def _validated_input_manifest(
    localization_release: Path, *, only_cells: tuple[str, ...] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    input_root = localization_release / "build_A/localization/inputs"
    input_manifest_path = input_root / "MANIFEST.json"
    # token_only=True validates archive names and hashes without materializing
    # the response_scores member.
    manifest = validate_fit_manifest(
        input_manifest_path, input_root=input_root, token_only=True,
        only_cells=only_cells,
    )
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    if not set(CELLS).issubset(by_cell):
        raise RuntimeError("localization input manifest lacks the Phase-2 cell roster")
    return input_root, input_manifest_path, manifest


def _cell_record(
    *, cell: Any, input_record: Mapping[str, Any], diagnostics: Mapping[str, Any],
    score_path: Path, score_sha: str, environment: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": "token-local-temporal-innovation-b3-cell-record-v1",
        "cell_id": cell.cell_id,
        "population_id": cell.population_id,
        "dataset_id": cell.dataset_id,
        "model_id": cell.model_id,
        "slice_id": cell.slice_id,
        "n_rows": len(cell.row_ids),
        "n_tokens": len(cell.token_confidence),
        "n_segments": len(cell.segment_starts),
        "method_ids": list(METHODS),
        "prepared_input_sha256": str(input_record["artifact_sha256"]),
        "prepared_record_payload_sha256": _payload_sha(input_record),
        "token_transform_sha256": cell.token_transform_sha256,
        "external_certificate_sha256": cell.external_certificate_sha256,
        "external_score_bindings_sha256": cell.external_score_bindings_sha256,
        "environment": dict(environment),
        "environment_sha256": environment["environment_sha256"],
        "fit_diagnostics": diagnostics,
        "score_path": score_path.name,
        "score_sha256": score_sha,
        "response_scores_materialized": False,
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
    }
    record["payload_sha256"] = _payload_sha(record)
    return record


def fit_cell_shard(
    localization_release: Path, output_root: Path, cell_id: str,
    *, execution_workers: int = 1,
) -> dict[str, Any]:
    """Fit one complete cell for a later deterministic assembly.

    Cell-level sharding changes only scheduling: every shard still performs all
    five outer folds, all five frozen arms, and all five seeds.  The final
    manifest is written only by :func:`assemble`, after it has checked the
    complete cell roster and every binding.
    """

    if cell_id not in CELLS:
        raise ValueError(f"unknown Phase-2 cell: {cell_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "cells" / cell_id
    if target.exists():
        raise FileExistsError(f"cell shard output already exists: {target}")
    input_root, _input_manifest_path, manifest = _validated_input_manifest(
        localization_release, only_cells=(cell_id,)
    )
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    input_record = by_cell[cell_id]
    input_path = input_root / str(input_record["artifact_path"])
    cell = load_prepared_localization_token_cell(input_path, input_record)
    diagnostics, arrays = _fit_cell(cell, execution_workers=execution_workers)
    pstg_folds = diagnostics["methods"][LOCAL_TOKEN_B3_ROOK_PSTG_INNOV]["fold_diagnostics"]
    pstg_counts = [
        int(fold["projected_stg"]["selected_edge_count"])
        for fold in pstg_folds
    ]
    if not pstg_counts or not all(0 < count < 36 for count in pstg_counts):
        raise RuntimeError(
            f"{cell_id}: Projected-STG support is empty or dense: {pstg_counts}"
        )
    environment = _environment_snapshot()
    target.mkdir(parents=True, exist_ok=False)
    score_path = target / "scores.npz"
    score_sha = atomic_write_npz(score_path, arrays)
    record = _cell_record(
        cell=cell, input_record=input_record, diagnostics=diagnostics,
        score_path=score_path, score_sha=score_sha, environment=environment,
    )
    record_path = target / "RECORD.json"
    atomic_write_json(record_path, record)
    print(json.dumps({
        "status": "PASS",
        "stage": "target_free_phase2_cell_shard",
        "cell_id": cell_id,
        "score_sha256": score_sha,
        "record_sha256": sha256_file(record_path),
        "pstg_edges_by_fold": pstg_counts,
        "response_scores_materialized": False,
    }, indent=2), flush=True)
    return record


def _record_payload_is_valid(record: Mapping[str, Any]) -> bool:
    unsigned = dict(record)
    expected = str(unsigned.pop("payload_sha256", ""))
    return bool(expected) and expected == _payload_sha(unsigned)


def assemble(localization_release: Path, output_root: Path) -> dict[str, Any]:
    """Validate all independently fitted cell shards and write the freeze."""

    freeze_path = output_root / "SCORE_FREEZE_MANIFEST.json"
    if freeze_path.exists():
        raise FileExistsError(f"score-freeze manifest already exists: {freeze_path}")
    _input_root, input_manifest_path, manifest = _validated_input_manifest(
        localization_release
    )
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    environment = _environment_snapshot()

    bindings = []
    pstg_support_cells = 0
    for position, cell_id in enumerate(CELLS, start=1):
        target = output_root / "cells" / cell_id
        record_path = target / "RECORD.json"
        if not record_path.is_file():
            raise FileNotFoundError(f"missing Phase-2 cell record: {record_path}")
        record = json.loads(record_path.read_text())
        if not _record_payload_is_valid(record):
            raise RuntimeError(f"{cell_id}: record payload hash mismatch")
        input_record = by_cell[cell_id]
        required = {
            "cell_id": cell_id,
            "method_ids": list(METHODS),
            "prepared_input_sha256": str(input_record["artifact_sha256"]),
            "prepared_record_payload_sha256": _payload_sha(input_record),
            "environment_sha256": environment["environment_sha256"],
            "response_scores_materialized": False,
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
        }
        for name, expected in required.items():
            if record.get(name) != expected:
                raise RuntimeError(
                    f"{cell_id}: shard binding mismatch for {name}: "
                    f"{record.get(name)!r} != {expected!r}"
                )
        if record.get("environment") != environment:
            raise RuntimeError(f"{cell_id}: shard environment snapshot mismatch")
        score_path = target / str(record["score_path"])
        score_sha = sha256_file(score_path)
        if score_sha != str(record["score_sha256"]):
            raise RuntimeError(f"{cell_id}: score artifact hash mismatch")
        diagnostics = record["fit_diagnostics"]
        pstg_folds = diagnostics["methods"][LOCAL_TOKEN_B3_ROOK_PSTG_INNOV]["fold_diagnostics"]
        pstg_counts = [
            int(fold["projected_stg"]["selected_edge_count"])
            for fold in pstg_folds
        ]
        if not pstg_counts or not all(0 < count < 36 for count in pstg_counts):
            raise RuntimeError(
                f"{cell_id}: Projected-STG support is empty or dense: {pstg_counts}"
            )
        if cell_id in PB_CELLS:
            pstg_support_cells += 1
        bindings.append({
            "cell_id": cell_id,
            "record_path": record_path.relative_to(output_root).as_posix(),
            "record_sha256": sha256_file(record_path),
            "score_sha256": score_sha,
        })
        print(f"verified {cell_id} ({position}/{len(CELLS)})", flush=True)

    if pstg_support_cells < 6:
        raise RuntimeError(
            "Projected-STG nonempty/non-dense support gate failed: "
            f"{pstg_support_cells}/{len(PB_CELLS)} ProcessBench cells"
        )

    freeze = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1",
        "expected_cells": list(CELLS),
        "processbench_cells": list(PB_CELLS),
        "prmbench_cell": PRM_CELL,
        "method_ids": list(METHODS),
        "primary_method_ids": list(METHODS),
        "control_method_ids": [LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL],
        "all_expected_scores_present": len(bindings) == len(CELLS),
        "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
        "response_scores_materialized": False,
        "independent_prelabel_audit_required": True,
        "environment": environment,
        "environment_sha256": environment["environment_sha256"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "source_snapshot": [
            {"path": name, "sha256": sha256_file(ROOT / name)} for name in SOURCE_FILES
        ],
        "records": bindings,
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(output_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    print(json.dumps({
        "status": "PASS",
        "stage": "target_free_phase2_score_freeze",
        "cells": len(bindings),
        "methods": len(METHODS),
        "response_scores_materialized": False,
        "payload_sha256": freeze["payload_sha256"],
    }, indent=2))
    return freeze


def fit(
    localization_release: Path, output_root: Path, *, execution_workers: int = 1,
) -> dict[str, Any]:
    """Sequential compatibility entrypoint; sharded scheduling uses the same fit."""

    if output_root.exists():
        raise FileExistsError(f"score-freeze output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    for cell_id in CELLS:
        fit_cell_shard(
            localization_release, output_root, cell_id,
            execution_workers=execution_workers,
        )
    return assemble(localization_release, output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--localization-release", required=True)
    parser.add_argument("--out-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cell", choices=CELLS)
    mode.add_argument("--assemble", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    localization_release = Path(args.localization_release).resolve()
    output_root = Path(args.out_dir).resolve()
    if args.cell is not None:
        fit_cell_shard(
            localization_release, output_root, args.cell,
            execution_workers=args.workers,
        )
    elif args.assemble:
        assemble(localization_release, output_root)
    else:
        fit(localization_release, output_root, execution_workers=args.workers)


if __name__ == "__main__":
    main()
