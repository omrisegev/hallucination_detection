#!/usr/bin/env python3
"""Build an independent A/B target-free external final-answer input tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    ExternalContractError,
    ID_CONTRACT_VERSION,
    PREPARED_SCHEMA_VERSION,
    external_id_contract_binding,
    fit_safe_external_cell_record,
    load_identity_key,
    load_external_registry,
    prepare_external_cell,
)
from spectral_utils.reconstruction_benchmark.external_fit_contract import (  # noqa: E402
    build_fit_row_identity_contract,
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
    "configs/reconstruction_benchmark_v1/fit_safe_feature_contract.json",
    "configs/reconstruction_benchmark_v1/fit_safe_feature_roster.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "scripts/reconstruction_benchmark/prepare_external_final_answer.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/repgrid_scoring.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/external_fit_contract.py",
    "spectral_utils/reconstruction_benchmark/external_fit_safe.py",
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
    parser.add_argument(
        "--identity-key",
        type=Path,
        help="Controller-only 32-byte release key; defaults outside releases/.",
    )
    args = parser.parse_args()

    registry = load_external_registry(
        repo=REPO,
        registry_path=args.registry,
        population_registry_path=args.populations,
    )
    controller_root = (
        args.release_root.parent / "private_control" / args.release_id
        / "external_final_answer"
    )
    identity_key_path = args.identity_key or (controller_root / "external-id-v2.key")
    resolved_key_path = identity_key_path.resolve()
    forbidden_key_roots = (
        args.release_root.resolve(),
        (
            controller_root / f"build_{args.build}"
            / "preparation_provenance"
        ).resolve(),
    )
    for forbidden_root in forbidden_key_roots:
        try:
            resolved_key_path.relative_to(forbidden_root)
        except ValueError:
            continue
        raise ValueError(
            "external identity key must be outside release, fit, input, and "
            "preparation-provenance trees"
        )
    try:
        resolved_key_path.relative_to(REPO.resolve())
    except ValueError:
        pass
    else:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(resolved_key_path)],
            cwd=REPO,
            check=False,
        )
        if ignored.returncode != 0:
            raise RuntimeError("in-repository external identity key path is not git-ignored")
    identity_key = load_identity_key(identity_key_path, create=True)
    identity_binding = external_id_contract_binding(
        registry, identity_key=identity_key
    )
    fit_identity_binding = build_fit_row_identity_contract(
        identity_binding, identity_key=identity_key
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
    provenance_output = controller_root / f"build_{args.build}" / "preparation_provenance"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"external input directory is not empty: {output}")
    if provenance_output.exists() and any(provenance_output.iterdir()):
        raise FileExistsError(
            f"external preparation-provenance directory is not empty: {provenance_output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    provenance_output.mkdir(parents=True, exist_ok=False)
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
            "dataset_id": spec.dataset_id,
            "model_id": spec.model_id,
            "slice_id": spec.slice_id,
            "domain": spec.domain,
            "comparison_group_id": spec.comparison_group_id,
            "panel_role": spec.panel_role,
            "adapter_id": spec.adapter_id,
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
                identity_key=identity_key,
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
        "schema_version": "reconstruction-external-target-free-build-v2",
        "prepared_cell_schema_version": PREPARED_SCHEMA_VERSION,
        "identity_contract": identity_binding,
        "fit_row_identity_contract": fit_identity_binding,
        "id_contract_version": ID_CONTRACT_VERSION,
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
    provenance_manifest_path = provenance_output / "MANIFEST.json"
    atomic_write_json(provenance_manifest_path, manifest)

    fit_manifest = {
        "schema_version": "reconstruction-external-fit-safe-build-v1",
        "prepared_cell_schema_version": PREPARED_SCHEMA_VERSION,
        "identity_contract": fit_identity_binding,
        "id_contract_version": ID_CONTRACT_VERSION,
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full_build": bool(full_selection),
        "applicability_complete": bool(applicability_complete),
        "complete_eligible_roster": bool(complete),
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "preparation_manifest_sha256": sha256_file(provenance_manifest_path),
        "preparation_manifest_payload_sha256": manifest["payload_sha256"],
        "preparation_attestation_sha256": source_snapshot["snapshot_sha256"],
        "feature_contract_id": registry.raw["feature_contract_id"],
        "mixed_v2_applied_exactly_once": True,
        "target_data_opened": False,
        "historical_scores_opened": False,
        "n_registered_cells": len(registry.cells),
        "n_runnable_cells": len(runnable),
        "n_prepared_cells": len(prepared_ids),
        "cells": [fit_safe_external_cell_record(row) for row in rows],
    }
    fit_manifest["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(fit_manifest)
    )
    fit_manifest_path = output / "MANIFEST.json"
    atomic_write_json(fit_manifest_path, fit_manifest)
    print(json.dumps({
        "build": args.build,
        "applicability_complete": applicability_complete,
        "complete_eligible_roster": complete,
        "n_prepared_cells": len(prepared_ids),
        "fit_safe_manifest": str(fit_manifest_path),
        "preparation_provenance_manifest": str(provenance_manifest_path),
        "statuses": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
