#!/usr/bin/env python3
"""Build an independent A/B target-free external final-answer input tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    ExternalContractError,
    PREPARED_SCHEMA_VERSION,
    load_external_registry,
    prepare_external_cell,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"

PREPARATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/feature_contract.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "scripts/reconstruction_benchmark/prepare_external_final_answer.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/repgrid_scoring.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/io.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    parser.add_argument(
        "--source-root", type=Path, default=REPO,
        help="Root containing the hash-frozen relative telemetry paths.",
    )
    parser.add_argument("--cell", action="append", dest="cells")
    args = parser.parse_args()

    registry = load_external_registry(
        repo=REPO,
        registry_path=args.registry,
        population_registry_path=args.populations,
    )
    source_snapshot = {
        "files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in PREPARATION_SOURCE_FILES
        ]
    }
    source_snapshot["snapshot_sha256"] = sha256_bytes(
        canonical_json_bytes(source_snapshot)
    )
    output = (
        args.release_root / args.release_id / f"build_{args.build}"
        / "external_final_answer" / "inputs"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"external input directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cell_dir = output / "cells"
    cell_dir.mkdir()
    requested = None if not args.cells else set(args.cells)
    unknown = set() if requested is None else requested - set(registry.by_cell)
    if unknown:
        raise KeyError(f"unknown external cells: {sorted(unknown)}")

    rows = []
    for spec in registry.cells:
        if requested is not None and spec.cell_id not in requested:
            continue
        base = {
            "cell_id": spec.cell_id,
            "population_id": spec.population_id,
            "fit_policy": spec.fit_policy,
            "expected_rows": spec.expected_rows,
        }
        if spec.fit_policy == "forbidden":
            rows.append({
                **base,
                "status": spec.configured_status,
                "reason": spec.status_reason,
                "prepared": False,
            })
            continue
        try:
            artifact = cell_dir / f"{spec.cell_id}.npz"
            record = prepare_external_cell(
                registry=registry,
                spec=spec,
                repo=args.source_root,
                output_path=artifact,
            )
            record = {**record, "artifact_path": artifact.relative_to(output).as_posix()}
            rows.append(record)
        except ExternalContractError as error:
            rows.append({
                **base,
                "status": error.status.value,
                "reason": str(error),
                "prepared": False,
            })

    runnable = [item for item in registry.cells if item.fit_policy == "run_if_compatible"]
    full_selection = requested is None
    prepared_ids = {row["cell_id"] for row in rows if row.get("status") == "ELIGIBLE"}
    terminal = {
        "ELIGIBLE",
        "INCOMPATIBLE_FEATURE_CONTRACT",
        "PROTOCOL_GATE_FAILED",
        "QUARANTINED",
    }
    applicability_complete = full_selection and all(row["status"] in terminal for row in rows)
    complete = applicability_complete and prepared_ids == {
        row["cell_id"] for row in rows if row["status"] == "ELIGIBLE"
    }
    manifest = {
        "schema_version": "reconstruction-external-target-free-build-v1",
        "prepared_cell_schema_version": PREPARED_SCHEMA_VERSION,
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full_build": bool(full_selection),
        "applicability_complete": bool(applicability_complete),
        "complete_eligible_roster": bool(complete),
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "source_root": str(args.source_root.resolve()),
        "preparation_source_snapshot": source_snapshot,
        "preparation_source_snapshot_sha256": source_snapshot["snapshot_sha256"],
        "feature_contract_id": registry.raw["feature_contract_id"],
        "mixed_v2_applied_exactly_once": True,
        "labels_opened": False,
        "historical_scores_opened": False,
        "n_registered_cells": len(registry.cells),
        "n_runnable_cells": len(runnable),
        "n_prepared_cells": len(prepared_ids),
        "cells": rows,
    }
    manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    atomic_write_json(output / "MANIFEST.json", manifest)
    print(json.dumps({
        "build": args.build,
        "applicability_complete": applicability_complete,
        "complete_eligible_roster": complete,
        "n_prepared_cells": len(prepared_ids),
        "manifest": str(output / "MANIFEST.json"),
        "statuses": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
