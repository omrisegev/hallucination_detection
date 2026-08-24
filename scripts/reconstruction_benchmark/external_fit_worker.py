#!/usr/bin/env python3
"""Restricted worker: fit prepared external cells and emit candidates only."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import sys


POLICY_ENV = "RECONSTRUCTION_EXTERNAL_FIT_POLICY_B64"
encoded_policy = os.environ.pop(POLICY_ENV, None)
if not encoded_policy:
    raise RuntimeError("external fit worker requires a controller audit policy")
policy = json.loads(base64.b64decode(encoded_policy).decode("utf-8"))

REPO = Path(__file__).resolve().parents[2]
# Importing this module through ``spectral_utils`` would execute that package's
# eager scientific imports before the hook exists.  Bootstrap the small,
# stdlib-only firewall directly from its frozen file instead.
firewall_path = REPO / "spectral_utils/reconstruction_benchmark/fit_firewall.py"
firewall_spec = importlib.util.spec_from_file_location(
    "_reconstruction_external_boot_firewall", firewall_path
)
if firewall_spec is None or firewall_spec.loader is None:
    raise RuntimeError("cannot load the external boot firewall")
boot_firewall = importlib.util.module_from_spec(firewall_spec)
firewall_spec.loader.exec_module(boot_firewall)
policy_sha256 = boot_firewall.install_fit_audit_hook(policy)
denial_probes = boot_firewall.run_forbidden_read_probes(policy)
del policy, encoded_policy

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse  # noqa: E402

from spectral_utils.reconstruction_benchmark.external_fit_safe import (  # noqa: E402
    validate_fit_safe_input_manifest,
)
from spectral_utils.reconstruction_benchmark.external_fit_contract import (  # noqa: E402
    ID_CONTRACT_VERSION,
    load_prepared_external_cell,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.methods import (  # noqa: E402
    PRIMARY_METHOD_IDS,
    PRIMARY_METHOD_SPECS,
    run_method,
)
from spectral_utils.reconstruction_benchmark.serialization import (  # noqa: E402
    write_score_result,
)


METHODS_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/methods.json"
SUCCESS = {"OK", "OK_FALLBACK"}


def _validate_methods(method_ids: tuple[str, ...]) -> None:
    registry = json.loads(METHODS_REGISTRY.read_text(encoding="utf-8"))
    rows = {str(item["method_id"]): item for item in registry["methods"]}
    if tuple(item["method_id"] for item in registry["methods"]) != PRIMARY_METHOD_IDS:
        raise RuntimeError("worker method registry is not the exact primary roster")
    for method_id in method_ids:
        spec = PRIMARY_METHOD_SPECS.get(method_id)
        if spec is None or method_id not in rows:
            raise KeyError(f"unregistered primary method: {method_id}")
        if (
            rows[method_id]["method_version_id"] != spec.method_version_id
            or rows[method_id]["config_sha256"] != spec.config_sha256
        ):
            raise RuntimeError(f"{method_id}: worker method contract mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--fit-root", required=True, type=Path)
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--method", action="append", dest="methods")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    fit_root = args.fit_root.resolve()
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_safe_input_manifest(
        manifest_path,
        repo=REPO,
        input_root=input_root,
        require_scientific=False,
    )
    if manifest.get("release_id") != args.release_id or manifest.get("build_id") != args.build:
        raise RuntimeError("worker release/build binding failed")
    method_ids = tuple(args.methods or PRIMARY_METHOD_IDS)
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("duplicate worker method request")
    _validate_methods(method_ids)
    records = [row for row in manifest["cells"] if row.get("status") == "ELIGIBLE"]
    requested = None if args.cells is None else set(args.cells)
    if requested is not None:
        available = {row["cell_id"] for row in records}
        if requested - available:
            raise KeyError(f"worker requested unavailable cells: {sorted(requested - available)}")
        records = [row for row in records if row["cell_id"] in requested]
    if not records:
        raise RuntimeError("worker has no eligible prepared cells")
    if fit_root.exists() and any(fit_root.iterdir()):
        raise FileExistsError(f"worker fit directory is not empty: {fit_root}")
    fit_root.mkdir(parents=True, exist_ok=False)

    identity_binding = manifest["identity_contract"]
    all_records: list[dict] = []
    for cell_record in records:
        cell = load_prepared_external_cell(
            artifact_path=input_root / cell_record["artifact_path"],
            record=cell_record,
            identity_contract=identity_binding,
        )
        cell_identity = {
            "identity_contract": identity_binding,
            "id_contract_version": ID_CONTRACT_VERSION,
            "id_contract_sha256": identity_binding["contract_sha256"],
            "identity_key_id": identity_binding["key_id"],
            "row_namespace_sha256": cell_record["row_namespace_sha256"],
            "row_roster_sha256": cell_record["row_roster_sha256"],
        }
        cell_root = fit_root / "cells" / cell.cell_id
        method_records = []
        for method_id in method_ids:
            result = run_method(method_id, cell)
            record = write_score_result(
                result,
                cell.row_ids,
                cell_root / method_id,
                identity_contract=cell_identity,
            )
            method_records.append(record)
            all_records.append({
                "cell_id": cell.cell_id,
                "population_id": cell.population_id,
                "method_id": method_id,
                "method_version_id": record["method_version_id"],
                "config_sha256": record["config_sha256"],
                "status": record["status"],
                "prepared_matrix_sha256": record["prepared_matrix_sha256"],
                "score_sha256": record["score_sha256"],
                "record_sha256": record["record_sha256"],
                "record_path": (cell_root / method_id / "RECORD.json").relative_to(fit_root).as_posix(),
                "score_path": (
                    (cell_root / method_id / "score.npz").relative_to(fit_root).as_posix()
                    if record["score_path"] else None
                ),
                "artifacts_sha256": record["artifacts_sha256"],
                "artifacts_path": (
                    (cell_root / method_id / "artifacts.npz").relative_to(fit_root).as_posix()
                    if record["artifacts_path"] else None
                ),
                "artifact_index_sha256": record["artifact_index_sha256"],
                "artifact_index_path": (
                    cell_root / method_id / "ARTIFACT_INDEX.json"
                ).relative_to(fit_root).as_posix(),
                **cell_identity,
            })
        atomic_write_json(cell_root / "CELL_FIT_MANIFEST.json", {
            "schema_version": "reconstruction-external-cell-fit-v3-worker-candidate",
            "cell_id": cell.cell_id,
            "population_id": cell.population_id,
            "prepared_matrix_sha256": cell.matrix_sha256,
            **cell_identity,
            "target_data_opened": False,
            "method_records": method_records,
        })

    expected = len(records) * len(method_ids)
    complete = (
        len(all_records) == expected
        and len({(row["cell_id"], row["method_id"]) for row in all_records}) == expected
        and all(row["status"] in SUCCESS and row["score_sha256"] for row in all_records)
    )
    violations = boot_firewall.fit_firewall_violations()
    if violations:
        raise RuntimeError(
            "external fit firewall recorded sticky violation(s): "
            + json.dumps(violations, sort_keys=True)
        )
    result_manifest = {
        "schema_version": "reconstruction-external-fit-worker-result-v1",
        "release_id": args.release_id,
        "build_id": args.build,
        "input_manifest_sha256": sha256_file(manifest_path),
        "input_manifest_payload_sha256": manifest["payload_sha256"],
        "audit_policy_sha256": policy_sha256,
        "denial_probes": denial_probes,
        "firewall_violations": [],
        "all_candidate_scores_present": bool(complete),
        "target_data_opened": False,
        "method_ids": list(method_ids),
        "cell_ids": [row["cell_id"] for row in records],
        "n_records": len(all_records),
        "expected_records": expected,
        "records": all_records,
    }
    result_manifest["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(result_manifest)
    )
    atomic_write_json(fit_root / "WORKER_RESULT_MANIFEST.json", result_manifest)
    if not complete:
        failed = []
        for row in all_records:
            if row["status"] in SUCCESS and row["score_sha256"]:
                continue
            score_record = json.loads(
                (fit_root / row["record_path"]).read_text(encoding="utf-8")
            )
            diagnostics = score_record.get("diagnostics", {})
            failed.append({
                "cell_id": row["cell_id"],
                "method_id": row["method_id"],
                "status": row["status"],
                "error_type": diagnostics.get("error_type"),
                "error": diagnostics.get("error"),
            })
        raise RuntimeError(
            "external fit worker candidates are incomplete: "
            + json.dumps(failed, sort_keys=True)
        )


if __name__ == "__main__":
    main()
