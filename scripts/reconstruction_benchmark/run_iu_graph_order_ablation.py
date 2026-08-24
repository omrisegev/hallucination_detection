#!/usr/bin/env python3
"""Fit one target-free build of the frozen IU graph-order ablation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.dufs_liu_feature_contract import CONTRACT_VERSION  # noqa: E402
from spectral_utils.reconstruction_benchmark.contracts import (  # noqa: E402
    PreparedCell,
    prepared_matrix_sha256,
)
from spectral_utils.reconstruction_benchmark.fit_validation import (  # noqa: E402
    validate_prepared_manifest,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.iu_graph_order_ablation import (  # noqa: E402
    config_sha256,
    expected_arm_ids,
    fit_cell,
    validate_config,
)
from spectral_utils.reconstruction_benchmark.preparation import _matrix_hash  # noqa: E402


SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/feature_contract.json",
    "configs/reconstruction_benchmark_v1/frozen24_cells.json",
    "configs/reconstruction_benchmark_v1/iu_graph_order_ablation_v1.json",
    "docs/experiments/IU_GRAPH_ORDER_ABLATION_V1.md",
    "scripts/reconstruction_benchmark/run_iu_graph_order_ablation.py",
    "spectral_utils/reconstruction_benchmark/iu_graph_order_ablation.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
    "spectral_utils/contribution_subspace.py",
    "spectral_utils/graph_topology.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/upcr.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--output-release", type=Path, required=True)
    parser.add_argument("--build-id", choices=("A", "B"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _source_snapshot() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Check only the exact scientific source closure.  The repository contains
    # large LFS artifacts whose clean filters are irrelevant to this fit.
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *SOURCE_FILES],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if changed:
        raise RuntimeError("scientific source closure is dirty: " + ", ".join(changed))
    files = [{"path": path, "sha256": sha256_file(REPO / path)} for path in SOURCE_FILES]
    payload = {
        "schema_version": "iu-graph-order-source-snapshot-v1",
        "git_head": head,
        "files": files,
    }
    payload["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _load_cell(input_root: Path, record: dict, population_id: str) -> PreparedCell:
    artifact = input_root / str(record["artifact_path"])
    if sha256_file(artifact) != record["artifact_sha256"]:
        raise RuntimeError(f"prepared artifact drifted: {record['cell_id']}")
    arrays = load_npz_no_pickle(artifact)
    if set(arrays) != {"X_confidence", "feature_names", "family_ids", "row_ids", "row_index"}:
        raise RuntimeError(f"prepared member roster drifted: {record['cell_id']}")
    matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
    names = tuple(str(value) for value in arrays["feature_names"].tolist())
    rows = tuple(str(value) for value in arrays["row_ids"].tolist())
    if _matrix_hash(matrix, names) != record["feature_matrix_sha256"]:
        raise RuntimeError(f"prepared matrix hash drifted: {record['cell_id']}")
    return PreparedCell(
        population_id=population_id,
        cell_id=str(record["cell_id"]),
        domain=str(record["domain"]),
        matrix=matrix,
        feature_names=names,
        row_ids=rows,
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
        declared_matrix_sha256=prepared_matrix_sha256(matrix, names, rows),
    )


def _fit_one(payload: dict) -> dict:
    input_root = Path(payload["input_root"])
    record = dict(payload["record"])
    config = dict(payload["config"])
    cell = _load_cell(input_root, record, str(payload["population_id"]))
    result = fit_cell(cell, config)
    arrays = {"row_ids": np.asarray(result.row_ids)}
    arrays.update({name: np.asarray(score, dtype="<f8") for name, score in result.scores.items()})
    return {
        "cell_id": cell.cell_id,
        "n_rows": len(cell.row_ids),
        "prepared_matrix_sha256": result.prepared_matrix_sha256,
        "arrays": arrays,
        "diagnostics": dict(result.diagnostics),
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    source_release = args.source_release.resolve()
    output_release = args.output_release.resolve()
    output = output_release / f"build_{args.build_id}"
    if output.exists():
        raise FileExistsError(f"output build already exists: {output}")

    config_path = REPO / "configs/reconstruction_benchmark_v1/iu_graph_order_ablation_v1.json"
    feature_path = REPO / "configs/reconstruction_benchmark_v1/feature_contract.json"
    cells_path = REPO / "configs/reconstruction_benchmark_v1/frozen24_cells.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lambdas = validate_config(config)
    feature_config = json.loads(feature_path.read_text(encoding="utf-8"))
    cell_registry = json.loads(cells_path.read_text(encoding="utf-8"))
    input_root = source_release / f"build_{args.build_id}" / "inputs"
    input_manifest = validate_prepared_manifest(
        input_root=input_root,
        build_id=args.build_id,
        repo=REPO,
        feature_config=feature_config,
        cell_registry=cell_registry,
    )
    source_snapshot = _source_snapshot()
    arm_ids = expected_arm_ids(lambdas)

    output_release.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".build_{args.build_id}.", dir=output_release))
    try:
        records = []
        payloads = [
            {
                "input_root": str(input_root),
                "record": record,
                "config": config,
                "population_id": cell_registry["population_id"],
            }
            for record in input_manifest["cells"]
        ]
        fitted: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fit_one, payload): payload["record"]["cell_id"] for payload in payloads}
            for future in as_completed(futures):
                cell_id = str(futures[future])
                fitted[cell_id] = future.result()
                print(f"FIT {args.build_id} {len(fitted):02d}/24 {cell_id}", flush=True)

        for source_record in input_manifest["cells"]:
            cell_id = str(source_record["cell_id"])
            row = fitted[cell_id]
            cell_dir = temporary / "cells" / cell_id
            score_path = cell_dir / "SCORES.npz"
            diagnostics_path = cell_dir / "DIAGNOSTICS.json"
            score_sha = atomic_write_npz(score_path, row["arrays"])
            diagnostics_sha = atomic_write_json(diagnostics_path, row["diagnostics"])
            records.append({
                "cell_id": cell_id,
                "domain": source_record["domain"],
                "n_rows": row["n_rows"],
                "prepared_matrix_sha256": row["prepared_matrix_sha256"],
                "score_path": score_path.relative_to(temporary).as_posix(),
                "score_sha256": score_sha,
                "diagnostics_path": diagnostics_path.relative_to(temporary).as_posix(),
                "diagnostics_sha256": diagnostics_sha,
            })
        manifest = {
            "schema_version": "iu-graph-order-score-freeze-v1",
            "experiment_id": config["experiment_id"],
            "build_id": args.build_id,
            "population_id": cell_registry["population_id"],
            "feature_contract_id": CONTRACT_VERSION,
            "runtime_labels_used": False,
            "evidence_status": config["evidence_status"],
            "config_path": config_path.relative_to(REPO).as_posix(),
            "config_sha256": config_sha256(config),
            "source_snapshot": source_snapshot,
            "source_input_manifest_sha256": sha256_file(input_root / "MANIFEST.json"),
            "source_input_manifest_payload_sha256": input_manifest["manifest_payload_sha256"],
            "source_prepared_ab_sha256": sha256_file(source_release / "PREPARED_AB_VERIFICATION.json"),
            "arm_ids": list(arm_ids),
            "n_arms": len(arm_ids),
            "n_cells": len(records),
            "n_rows": sum(int(row["n_rows"]) for row in records),
            "cells": records,
        }
        manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        atomic_write_json(temporary / "SCORE_FREEZE_MANIFEST.json", manifest)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "FROZEN",
        "build_id": args.build_id,
        "n_cells": 24,
        "n_arms": len(arm_ids),
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
