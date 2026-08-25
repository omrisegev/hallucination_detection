#!/usr/bin/env python3
"""Restricted worker for target-free RAG evidence fitting and scoring."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys


POLICY_ENV = "RECONSTRUCTION_RAG_FIT_POLICY_B64"
encoded_policy = os.environ.pop(POLICY_ENV, None)
if not encoded_policy:
    raise RuntimeError("RAG fit worker requires a controller audit policy")
policy = json.loads(base64.b64decode(encoded_policy).decode("utf-8"))

CAPSULE = Path(__file__).resolve().parents[2]
firewall_path = CAPSULE / "spectral_utils/reconstruction_benchmark/fit_firewall.py"
spec = importlib.util.spec_from_file_location("_rag_fit_boot_firewall", firewall_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load RAG fit firewall")
firewall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(firewall)
policy_sha256 = firewall.install_fit_audit_hook(policy)
denial_probes = firewall.run_forbidden_read_probes(policy)
del policy, encoded_policy

if str(CAPSULE) not in sys.path:
    sys.path.insert(0, str(CAPSULE))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    sha256_bytes,
)
from spectral_utils.reconstruction_benchmark.rag_evidence_contract import (  # noqa: E402
    FIT_INPUT_SCHEMA,
    SCORE_FREEZE_SCHEMA,
    SCORES_FILENAME,
    add_payload_sha256,
    load_fit_input_handle,
    validate_artifact_identifier,
)
from spectral_utils.reconstruction_benchmark.rag_evidence_fit import (  # noqa: E402
    compute_rag_evidence_scores,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise RuntimeError("non-finite diagnostic in RAG fit worker")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-fd", required=True, type=int)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--forbidden-field", action="append", default=[])
    args = parser.parse_args()
    args.release_id = validate_artifact_identifier(
        args.release_id, name="RAG release ID"
    )

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"RAG worker output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    safe_registry = {
        "lane_id": args.lane_id,
        "fit_visibility": {"forbidden_fields": list(args.forbidden_field)},
    }
    input_stat = os.fstat(args.input_fd)
    if not stat.S_ISREG(input_stat.st_mode):
        raise RuntimeError("RAG worker inherited input is not a regular file")
    input_signature = (
        int(input_stat.st_dev), int(input_stat.st_ino), int(input_stat.st_size),
        int(input_stat.st_mtime_ns), int(input_stat.st_ctime_ns),
    )

    def held_input_sha256() -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(args.input_fd, 1024 * 1024, offset)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        return digest.hexdigest()

    if held_input_sha256() != args.expected_input_sha256:
        raise RuntimeError("RAG worker received another prepared input")
    with os.fdopen(os.dup(args.input_fd), "rb") as handle:
        fit_input = load_fit_input_handle(handle, safe_registry)
    end_stat = os.fstat(args.input_fd)
    end_signature = (
        int(end_stat.st_dev), int(end_stat.st_ino), int(end_stat.st_size),
        int(end_stat.st_mtime_ns), int(end_stat.st_ctime_ns),
    )
    if end_signature != input_signature or held_input_sha256() != args.expected_input_sha256:
        raise RuntimeError("RAG worker input changed during held-fd parsing")
    if fit_input.get("schema_version") != FIT_INPUT_SCHEMA:
        raise RuntimeError("RAG worker input schema drifted")
    arrays, diagnostics = compute_rag_evidence_scores(fit_input)
    score_sha = atomic_write_npz(output_root / SCORES_FILENAME, arrays)
    violations = firewall.fit_firewall_violations()
    if violations:
        raise RuntimeError(
            "RAG fit firewall recorded sticky violations: "
            + json.dumps(violations, sort_keys=True)
        )
    result = add_payload_sha256({
        "schema_version": "reconstruction-rag-evidence-worker-result-v1",
        "release_id": args.release_id,
        "build_id": args.build,
        "lane_id": args.lane_id,
        "input_sha256": args.expected_input_sha256,
        "score_schema": SCORE_FREEZE_SCHEMA,
        "score_path": SCORES_FILENAME,
        "score_sha256": score_sha,
        "audit_policy_sha256": policy_sha256,
        "denial_probes": denial_probes,
        "firewall_violations": [],
        "labels_opened_by_fit": False,
        "historical_scores_opened": False,
        "diagnostics": _jsonable(diagnostics),
    })
    atomic_write_json(output_root / "WORKER_RESULT.json", result)


if __name__ == "__main__":
    main()
