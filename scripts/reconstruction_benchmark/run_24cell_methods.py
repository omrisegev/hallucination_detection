#!/usr/bin/env python3
"""Fit the 13 registered methods on target-free mixed-v2 cell snapshots.

This executable has no label-bundle argument and never imports an evaluator.
It freezes score hashes before a separate program may open targets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.dufs_liu_feature_contract import CONTRACT_VERSION  # noqa: E402
from spectral_utils.reconstruction_benchmark import (  # noqa: E402
    PRIMARY_METHOD_IDS,
    PRIMARY_METHOD_SPECS,
    PreparedCell,
    prepared_matrix_sha256,
    run_method,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.preparation import (  # noqa: E402
    FORBIDDEN_FIELD_FRAGMENTS,
    _matrix_hash,
)
from spectral_utils.reconstruction_benchmark.serialization import (  # noqa: E402
    write_score_result,
)
from spectral_utils.reconstruction_benchmark.fit_validation import (  # noqa: E402
    validate_prepared_manifest,
    validate_score_record,
)


DEFAULT_RELEASE_ROOT = REPO / "results" / "reconstruction_benchmark_v1" / "releases"
METHOD_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "methods.json"
FEATURE_CONFIG = REPO / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"
CELL_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"
SUCCESS = {"OK", "OK_FALLBACK"}

SOURCE_SNAPSHOT_PATHS = (
    "configs/reconstruction_benchmark_v1/feature_contract.json",
    "configs/reconstruction_benchmark_v1/frozen24_cells.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "scripts/inscope_cells.py",
    "scripts/reconstruction_benchmark/run_24cell_methods.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
    "spectral_utils/reconstruction_benchmark/preparation.py",
    "spectral_utils/reconstruction_benchmark/serialization.py",
    "spectral_utils/reconstruction_benchmark/fit_validation.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/selectors/a2_groupfs.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/specrage_laplacian.py",
    "spectral_utils/fusion_aware_views.py",
    "spectral_utils/residual_graph_deem.py",
    "spectral_utils/contribution_subspace.py",
    "spectral_utils/graph_topology.py",
    "spectral_utils/specrage_views.py",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "torch",
                "deem",
                "pandas",
                "pyarrow",
                "duckdb",
            )
        },
    }


def _source_snapshot() -> dict:
    files = [
        {"path": relative, "sha256": sha256_file(REPO / relative)}
        for relative in SOURCE_SNAPSHOT_PATHS
    ]
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot record git/source provenance") from error
    if git_status.strip():
        raise RuntimeError(
            "scientific fitting requires a clean worktree; commit the frozen "
            "implementation and registries before launching"
        )
    payload = {
        "schema_version": "reconstruction-source-snapshot-v1",
        "git_head": git_head,
        "git_status_sha256": sha256_bytes(git_status.encode("utf-8")),
        "git_status_clean": True,
        "files": files,
    }
    payload["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _fit_source_snapshot_record(
    *,
    release_id: str,
    build_id: str,
    input_manifest: dict,
    source_snapshot: dict,
    method_ids: tuple[str, ...],
    cell_ids: tuple[str, ...],
) -> dict:
    """Bind every reusable fit artifact to the code seen *before* fitting.

    This record is deliberately written before the first worker starts.  A
    resumed process must reproduce it exactly; it may not discover old score
    files first and attribute them to the source tree visible at the end of a
    later invocation.
    """

    record = {
        "schema_version": "reconstruction-fit-source-snapshot-v1",
        "release_id": release_id,
        "build_id": build_id,
        "input_manifest_payload_sha256": input_manifest["manifest_payload_sha256"],
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": source_snapshot["snapshot_sha256"],
        "method_ids": list(method_ids),
        "cell_ids": list(cell_ids),
    }
    record["payload_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


def _freeze_or_verify_prefit_snapshot(
    *,
    fit_root: Path,
    expected: dict,
    resume: bool,
) -> dict:
    """Create the immutable pre-fit record or verify an exact resume."""

    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    if freeze_path.exists():
        raise RuntimeError(
            f"build is already scientifically frozen and cannot be rerun: {freeze_path}"
        )
    snapshot_path = fit_root / "FIT_SOURCE_SNAPSHOT.json"
    existing_entries = list(fit_root.iterdir()) if fit_root.exists() else []
    if snapshot_path.exists():
        if not resume:
            raise RuntimeError(
                "fit output already has a pre-fit snapshot; use --resume only for "
                "an interrupted build or choose a new release"
            )
        observed = json.loads(snapshot_path.read_text())
        observed_payload = dict(observed)
        recorded_hash = observed_payload.pop("payload_sha256", None)
        recomputed = sha256_bytes(canonical_json_bytes(observed_payload))
        if recorded_hash != recomputed:
            raise RuntimeError("existing FIT_SOURCE_SNAPSHOT.json failed its payload hash")
        if observed != expected:
            raise RuntimeError(
                "resume source/input/roster differs from the immutable pre-fit snapshot"
            )
        return observed
    if resume:
        raise RuntimeError(
            "--resume requires FIT_SOURCE_SNAPSHOT.json; legacy partial outputs "
            "cannot be attributed to the current source snapshot"
        )
    if existing_entries:
        raise RuntimeError(
            "fit output exists without an immutable pre-fit snapshot; choose a new release"
        )
    fit_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(snapshot_path, expected)
    return expected


def _validate_registry(method_ids: tuple[str, ...]) -> dict:
    registry = json.loads(METHOD_REGISTRY.read_text())
    registered = tuple(item["method_id"] for item in registry["methods"])
    if registered != PRIMARY_METHOD_IDS:
        raise RuntimeError(
            "machine-readable Method Guide and executable primary roster disagree: "
            f"{registered!r} versus {PRIMARY_METHOD_IDS!r}"
        )
    if any(method_id not in registered for method_id in method_ids):
        raise RuntimeError("requested method is absent from the frozen registry")
    by_id = {item["method_id"]: item for item in registry["methods"]}
    for method_id, spec in PRIMARY_METHOD_SPECS.items():
        row = by_id[method_id]
        if row.get("method_version_id") != spec.method_version_id:
            raise RuntimeError(f"{method_id}: registry/executable version mismatch")
        if row.get("config_sha256") != spec.config_sha256:
            raise RuntimeError(f"{method_id}: registry/executable config mismatch")
    if registry.get("runtime_labels_used") is not False:
        raise RuntimeError("method registry does not freeze runtime label isolation")
    if registry.get("preprocessing_selected_after_outcomes_were_opened") is not True:
        raise RuntimeError("method registry lost the D0 preprocessing boundary")
    return registry


def _target_free_member_check(path: Path) -> None:
    with np.load(path, allow_pickle=False) as bundle:
        for name in bundle.files:
            lowered = name.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
                raise RuntimeError(f"target-like array in fitting input {path}: {name}")
        allowed = {"X_confidence", "feature_names", "family_ids", "row_ids", "row_index"}
        if set(bundle.files) != allowed:
            raise RuntimeError(
                f"unexpected fitting arrays in {path}: {sorted(set(bundle.files) - allowed)}"
            )


def _load_cell(input_root: Path, cell_record: dict) -> PreparedCell:
    path = input_root / cell_record["artifact_path"]
    if sha256_file(path) != cell_record["artifact_sha256"]:
        raise RuntimeError(f"prepared input hash drifted: {path}")
    _target_free_member_check(path)
    arrays = load_npz_no_pickle(path)
    matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
    names = tuple(str(item) for item in arrays["feature_names"].tolist())
    rows = tuple(str(item) for item in arrays["row_ids"].tolist())
    if _matrix_hash(matrix, names) != cell_record["feature_matrix_sha256"]:
        raise RuntimeError(f"prepared matrix content drifted: {path}")
    observed = prepared_matrix_sha256(matrix, names, rows)
    return PreparedCell(
        population_id="frozen24_response_v1",
        cell_id=cell_record["cell_id"],
        domain=cell_record["domain"],
        matrix=matrix,
        feature_names=names,
        row_ids=rows,
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
        declared_matrix_sha256=observed,
    )


def _reusable_record(method_dir: Path, *, spec, cell: PreparedCell) -> dict | None:
    try:
        return validate_score_record(method_dir=method_dir, spec=spec, cell=cell)
    except Exception:
        return None


def _fit_cell(payload: dict) -> dict:
    input_root = Path(payload["input_root"])
    fit_root = Path(payload["fit_root"])
    cell_record = payload["cell_record"]
    method_ids = tuple(payload["method_ids"])
    resume = bool(payload["resume"])
    cell = _load_cell(input_root, cell_record)
    cell_output = fit_root / "cells" / cell.cell_id
    cell_output.mkdir(parents=True, exist_ok=True)
    records = []
    for method_id in method_ids:
        spec = PRIMARY_METHOD_SPECS[method_id]
        method_dir = cell_output / method_id
        reusable = _reusable_record(method_dir, spec=spec, cell=cell) if resume else None
        if reusable is not None:
            record = reusable
            records.append(record)
            continue
        if method_dir.exists() and any(method_dir.iterdir()):
            raise FileExistsError(
                f"non-reusable output exists for {cell.cell_id}/{method_id}; use a new release"
            )
        result = run_method(method_id, cell)
        record = write_score_result(result, cell.row_ids, method_dir)
        records.append(record)
    cell_manifest = {
        "schema_version": "reconstruction-cell-fit-manifest-v1",
        "cell_id": cell.cell_id,
        "population_id": cell.population_id,
        "domain": cell.domain,
        "n_rows": len(cell.row_ids),
        "n_features": len(cell.feature_names),
        "prepared_matrix_sha256": cell.matrix_sha256,
        "method_records": records,
    }
    cell_manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(cell_manifest))
    atomic_write_json(cell_output / "CELL_FIT_MANIFEST.json", cell_manifest)
    return cell_manifest


def _freeze_fit(
    *,
    build_id: str,
    input_manifest: dict,
    fit_root: Path,
    cell_manifests: list[dict],
    method_ids: tuple[str, ...],
    environment: dict,
    source_snapshot: dict,
    prefit_snapshot: dict,
    expected_cell_ids: tuple[str, ...],
) -> dict:
    records = []
    for cell in sorted(cell_manifests, key=lambda item: item["cell_id"]):
        for record in sorted(cell["method_records"], key=lambda item: item["method_id"]):
            records.append(
                {
                    "cell_id": cell["cell_id"],
                    "method_id": record["method_id"],
                    "method_version_id": record["method_version_id"],
                    "config_sha256": record["config_sha256"],
                    "status": record["status"],
                    "prepared_matrix_sha256": record["prepared_matrix_sha256"],
                    "score_sha256": record["score_sha256"],
                    "artifacts_sha256": record["artifacts_sha256"],
                    "artifact_index_sha256": record["artifact_index_sha256"],
                    "record_sha256": record["record_sha256"],
                }
            )
    actual_pairs = [(item["cell_id"], item["method_id"]) for item in records]
    expected_pairs = [
        (cell_id, method_id)
        for cell_id in expected_cell_ids
        for method_id in method_ids
    ]
    expected = 24 * len(PRIMARY_METHOD_IDS)
    complete = sorted(actual_pairs) == sorted(expected_pairs)
    complete = complete and len(actual_pairs) == len(set(actual_pairs)) == expected
    complete = complete and all(record["status"] in SUCCESS for record in records)
    complete = complete and all(record["score_sha256"] for record in records)
    manifest = {
        "schema_version": "reconstruction-score-freeze-v1",
        "build_id": build_id,
        "scientific_run": True,
        "feature_contract_id": CONTRACT_VERSION,
        "score_semantics": "higher_is_incorrect",
        "positive_class": "incorrect",
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "preprocessing_selected_after_outcomes_were_opened": True,
        "evidence_status": "D0_reused_development",
        "input_manifest_payload_sha256": input_manifest["manifest_payload_sha256"],
        "input_manifest_file_sha256": sha256_file(
            fit_root.parent / "inputs" / "MANIFEST.json"
        ),
        "n_cells": 24,
        "n_methods": len(method_ids),
        "n_records": len(records),
        "expected_records": expected,
        "all_headline_scores_present": complete,
        "method_ids": list(method_ids),
        "cell_ids": list(expected_cell_ids),
        "method_specs": {
            method_id: {
                "method_version_id": PRIMARY_METHOD_SPECS[method_id].method_version_id,
                "config": dict(PRIMARY_METHOD_SPECS[method_id].config),
                "config_sha256": PRIMARY_METHOD_SPECS[method_id].config_sha256,
            }
            for method_id in method_ids
        },
        "environment": environment,
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": source_snapshot["snapshot_sha256"],
        "prefit_snapshot_payload_sha256": prefit_snapshot["payload_sha256"],
        "prefit_snapshot_file_sha256": sha256_file(
            fit_root / "FIT_SOURCE_SNAPSHOT.json"
        ),
        "method_registry_sha256": sha256_file(METHOD_REGISTRY),
        "feature_config_sha256": sha256_file(FEATURE_CONFIG),
        "cell_registry_sha256": sha256_file(CELL_REGISTRY),
        "records": records,
    }
    manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    name = "SCORE_FREEZE_MANIFEST.json" if complete else "FIT_INCOMPLETE.json"
    atomic_write_json(fit_root / name, manifest)
    if not complete:
        failures = [
            (item["cell_id"], item["method_id"], item["status"])
            for item in records if item["status"] not in SUCCESS
        ]
        raise RuntimeError(f"fit is incomplete; no score freeze issued: {failures}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--method", action="append", dest="methods")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    method_ids = tuple(args.methods or PRIMARY_METHOD_IDS)
    _validate_registry(method_ids)
    release = args.release_root / args.release_id
    input_root = release / f"build_{args.build}" / "inputs"
    fit_root = release / f"build_{args.build}" / "fit"
    feature_config = json.loads(FEATURE_CONFIG.read_text())
    cell_registry = json.loads(CELL_REGISTRY.read_text())
    input_manifest = validate_prepared_manifest(
        input_root=input_root,
        build_id=args.build,
        repo=REPO,
        feature_config=feature_config,
        cell_registry=cell_registry,
    )
    cell_records = input_manifest["cells"]
    if args.cells:
        requested = set(args.cells)
        cell_records = [item for item in cell_records if item["cell_id"] in requested]
        if {item["cell_id"] for item in cell_records} != requested:
            raise KeyError("unknown --cell requested")
    selected_cell_ids = tuple(item["cell_id"] for item in cell_records)
    source_snapshot = _source_snapshot()
    prefit_snapshot = _fit_source_snapshot_record(
        release_id=args.release_id,
        build_id=args.build,
        input_manifest=input_manifest,
        source_snapshot=source_snapshot,
        method_ids=method_ids,
        cell_ids=selected_cell_ids,
    )
    _freeze_or_verify_prefit_snapshot(
        fit_root=fit_root,
        expected=prefit_snapshot,
        resume=args.resume,
    )
    payloads = [
        {
            "input_root": str(input_root),
            "fit_root": str(fit_root),
            "cell_record": item,
            "method_ids": method_ids,
            "resume": args.resume,
        }
        for item in cell_records
    ]
    cell_manifests = []
    if args.jobs == 1:
        for payload in payloads:
            cell_manifests.append(_fit_cell(payload))
            print(f"completed {payload['cell_record']['cell_id']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_fit_cell, payload): payload for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                cell_manifests.append(result)
                print(f"completed {result['cell_id']}", flush=True)

    expected_cell_ids = tuple(item["cell_id"] for item in cell_registry["cells"])
    is_full = (
        tuple(item["cell_id"] for item in cell_records) == expected_cell_ids
        and method_ids == PRIMARY_METHOD_IDS
    )
    if is_full:
        end_snapshot = _source_snapshot()
        if end_snapshot != source_snapshot:
            raise RuntimeError(
                "source tree changed during fitting; no scientific freeze was issued"
            )
        freeze = _freeze_fit(
            build_id=args.build,
            input_manifest=input_manifest,
            fit_root=fit_root,
            cell_manifests=cell_manifests,
            method_ids=method_ids,
            environment=_environment(),
            source_snapshot=source_snapshot,
            prefit_snapshot=prefit_snapshot,
            expected_cell_ids=expected_cell_ids,
        )
        print(json.dumps({
            "build_id": args.build,
            "n_records": freeze["n_records"],
            "payload_sha256": freeze["payload_sha256"],
        }, indent=2, sort_keys=True))
    else:
        atomic_write_json(
            fit_root / "DEBUG_RUN.json",
            {
                "scientific_run": False,
                "build_id": args.build,
                "cells": [item["cell_id"] for item in cell_records],
                "methods": list(method_ids),
                "label_arrays_accessed": False,
            },
        )


if __name__ == "__main__":
    main()
