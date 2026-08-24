#!/usr/bin/env python3
"""Restricted worker for the target-free localization token head and adapters."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import sys


POLICY_ENV = "RECONSTRUCTION_LOCALIZATION_FIT_POLICY_B64"
encoded_policy = os.environ.pop(POLICY_ENV, None)
if not encoded_policy:
    raise RuntimeError("localization fit worker requires a controller audit policy")
policy = json.loads(base64.b64decode(encoded_policy).decode("utf-8"))

REPO = Path(__file__).resolve().parents[2]
firewall_path = REPO / "spectral_utils/reconstruction_benchmark/fit_firewall.py"
spec = importlib.util.spec_from_file_location("_localization_fit_firewall", firewall_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load localization fit firewall")
firewall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(firewall)
policy_sha256 = firewall.install_fit_audit_hook(policy)
denial_probes = firewall.run_forbidden_read_probes(policy)
del policy, encoded_policy

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse  # noqa: E402

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell,
    payload_sha256,
    validate_fit_manifest,
)
from spectral_utils.reconstruction_benchmark.localization_fit import (  # noqa: E402
    fit_localization_cell,
    write_localization_score_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--fit-root", required=True, type=Path)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    fit_root = args.fit_root.resolve()
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(
        manifest_path, input_root=input_root, require_scientific=False
    )
    if manifest["release_id"] != args.release_id or manifest["build_id"] != args.build:
        raise RuntimeError("localization worker release/build binding failed")
    if fit_root.exists() and any(fit_root.iterdir()):
        raise FileExistsError(f"localization fit root is not empty: {fit_root}")
    fit_root.mkdir(parents=True, exist_ok=False)

    records: list[dict] = []
    for cell_record in manifest["cells"]:
        cell = load_prepared_localization_cell(
            input_root / str(cell_record["artifact_path"]), cell_record
        )
        bundle = fit_localization_cell(cell)
        output = fit_root / "cells" / cell.cell_id
        record = write_localization_score_bundle(bundle, output)
        records.append({
            "cell_id": cell.cell_id,
            "population_id": cell.population_id,
            "dataset_id": cell.dataset_id,
            "model_id": cell.model_id,
            "slice_id": cell.slice_id,
            "record_path": (output / "RECORD.json").relative_to(fit_root).as_posix(),
            "record_file_sha256": sha256_file(output / "RECORD.json"),
            "record_sha256": record["record_sha256"],
            "score_path": (output / "scores.npz").relative_to(fit_root).as_posix(),
            "score_sha256": record["score_sha256"],
            "n_rows": record["n_rows"],
            "n_segments": record["n_segments"],
            "n_systems": record["n_systems"],
            "external_certificate_sha256": record["external_certificate_sha256"],
            "external_score_bindings_sha256": record["external_score_bindings_sha256"],
            "token_transform_sha256": record["token_transform_sha256"],
            "token_fit_sha256": record["token_fit_diagnostics"]["fit_sha256"],
        })
    violations = firewall.fit_firewall_violations()
    if violations:
        raise RuntimeError("localization fit firewall recorded a sticky violation")
    result = {
        "schema_version": "reconstruction-localization-fit-worker-result-v1",
        "release_id": args.release_id,
        "build_id": args.build,
        "input_manifest_sha256": sha256_file(manifest_path),
        "input_manifest_payload_sha256": manifest["payload_sha256"],
        "audit_policy_sha256": policy_sha256,
        "denial_probes": denial_probes,
        "firewall_violations": [],
        "target_data_opened": False,
        "response_scores_refit": False,
        "n_records": len(records),
        "records": records,
    }
    result["payload_sha256"] = payload_sha256(result)
    atomic_write_json(fit_root / "WORKER_RESULT_MANIFEST.json", result)


if __name__ == "__main__":
    main()
