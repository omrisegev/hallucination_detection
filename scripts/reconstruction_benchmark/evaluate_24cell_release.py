#!/usr/bin/env python3
"""Evaluate one fully verified 24x13 reconstruction release.

The program refuses to open correctness labels until both score builds and all
audited source-group sidecars have passed the strict target-free gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.evaluation import (  # noqa: E402
    BOOTSTRAP_DRAW_COUNT,
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    evaluate_verified_release,
    open_correctness_labels,
    prediction_snapshot_arrays,
    verify_release_before_labels,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


DEFAULT_RELEASE_ROOT = REPO / "results" / "reconstruction_benchmark_v1" / "releases"
DEFAULT_LABEL_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_CELL_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"
DEFAULT_METHOD_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "methods.json"
DEFAULT_FEATURE_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--label-bundle", type=Path, default=DEFAULT_LABEL_BUNDLE)
    parser.add_argument("--cell-registry", type=Path, default=DEFAULT_CELL_REGISTRY)
    parser.add_argument("--method-registry", type=Path, default=DEFAULT_METHOD_REGISTRY)
    parser.add_argument("--feature-registry", type=Path, default=DEFAULT_FEATURE_REGISTRY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <release>/evaluation (must not already exist)",
    )
    return parser.parse_args()


def _publish(
    output_dir: Path,
    evaluation: dict,
    bootstrap_arrays: dict,
    prediction_arrays: dict,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        evaluation_path = temporary / "EVALUATION.json"
        bootstrap_path = temporary / "BOOTSTRAP_DRAWS.npz"
        prediction_path = temporary / "PREDICTION_SNAPSHOT.npz"
        evaluation_sha = atomic_write_json(evaluation_path, evaluation)
        bootstrap_sha = atomic_write_npz(bootstrap_path, bootstrap_arrays)
        prediction_sha = atomic_write_npz(prediction_path, prediction_arrays)
        manifest = {
            "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
            "status": evaluation["status"],
            "headline_status": evaluation["headline_status"],
            "population_id": evaluation["population_id"],
            "n_cells": evaluation["n_cells"],
            "n_methods": evaluation["n_methods"],
            "bootstrap_draws": evaluation["bootstrap"]["draws"],
            "canonical_bootstrap_draws": BOOTSTRAP_DRAW_COUNT,
            "evaluation_path": evaluation_path.name,
            "evaluation_sha256": evaluation_sha,
            "bootstrap_path": bootstrap_path.name,
            "bootstrap_sha256": bootstrap_sha,
            "prediction_snapshot_path": prediction_path.name,
            "prediction_snapshot_sha256": prediction_sha,
            "prediction_snapshot_schema": "reconstruction-prediction-snapshot-v1",
            "evaluator_cli_sha256": sha256_file(Path(__file__)),
            "input_provenance": evaluation["provenance"],
        }
        manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        atomic_write_json(temporary / "EVALUATION_MANIFEST.json", manifest)
        if sha256_file(evaluation_path) != manifest["evaluation_sha256"]:
            raise RuntimeError("evaluation changed during publication")
        if sha256_file(bootstrap_path) != manifest["bootstrap_sha256"]:
            raise RuntimeError("bootstrap archive changed during publication")
        if sha256_file(prediction_path) != manifest["prediction_snapshot_sha256"]:
            raise RuntimeError("prediction snapshot changed during publication")
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    release = (args.release_root / args.release_id).resolve()
    output_dir = (args.output_dir or release / "evaluation").resolve()

    # This call is target-free and must finish before the next line can access
    # any ``<cell>__labels`` array.
    verified = verify_release_before_labels(
        release_root=release,
        cell_registry_path=args.cell_registry,
        method_registry_path=args.method_registry,
        feature_registry_path=args.feature_registry,
        label_bundle=args.label_bundle,
        group_manifest_path=args.group_manifest,
    )
    y_correct = open_correctness_labels(verified)
    evaluation, bootstrap_arrays = evaluate_verified_release(verified, y_correct)
    prediction_arrays = prediction_snapshot_arrays(verified, y_correct)
    manifest = _publish(output_dir, evaluation, bootstrap_arrays, prediction_arrays)
    print(json.dumps({
        "status": manifest["status"],
        "headline_status": manifest["headline_status"],
        "n_cells": manifest["n_cells"],
        "n_methods": manifest["n_methods"],
        "bootstrap_draws": manifest["bootstrap_draws"],
        "evaluation_sha256": manifest["evaluation_sha256"],
        "bootstrap_sha256": manifest["bootstrap_sha256"],
        "prediction_snapshot_sha256": manifest["prediction_snapshot_sha256"],
        "output_dir": str(output_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
