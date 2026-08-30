#!/usr/bin/env python3
"""Create the immutable Phase-2 report snapshot after acceptance checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_bytes, sha256_file  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import PROGRAM_ROOT  # noqa: E402


def main() -> None:
    target = PROGRAM_ROOT / "snapshots/phase_2"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable snapshot: {target}")
    experiments = {row["experiment_id"]:row for row in json.loads((PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json").read_text())["experiments"]}
    if experiments["P2_ATOMIC"]["execution_status"] != "COMPLETE" or experiments["P2_REDUCER_STUDY"]["execution_status"] != "COMPLETE":
        raise RuntimeError("Phase 2 is not terminal")
    target.mkdir(parents=True, exist_ok=False)
    for name in ("REPORT.html","REPORT_MANIFEST.json"):
        source = PROGRAM_ROOT / name
        atomic_write_bytes(target/name, source.read_bytes())
        if sha256_file(target/name) != sha256_file(source):
            raise RuntimeError(f"snapshot copy mismatch: {name}")
    print(json.dumps({"status":"IMMUTABLE_PHASE2_SNAPSHOT_CREATED","path":str(target),
                      "report_sha256":sha256_file(target/"REPORT.html"),
                      "manifest_sha256":sha256_file(target/"REPORT_MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    main()
