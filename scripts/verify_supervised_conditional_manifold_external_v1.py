#!/usr/bin/env python3
"""Verify hashes and deterministic rebuild of the external manifold audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/supervised_conditional_manifold_external_validation_v1"
RUNNER = ROOT / "scripts/validate_supervised_conditional_manifold_external_v2.py"
PLOTTER = ROOT / "scripts/plot_supervised_conditional_manifold_external_v1.py"
COMPARABLE = (
    "CELL_GRAPH_METRICS.csv", "DECISION.json", "FAMILY_SUMMARY.csv", "REPORT.md",
    "01_external_family_gates.png", "02_external_gate_ladder.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    definition = json.loads((args.results / "RUN_DEFINITION.json").read_text(encoding="utf-8"))
    source_checks = {
        name: (ROOT / name).exists() and sha256(ROOT / name) == expected
        for name, expected in definition["source_hashes"].items()
    }
    input_checks = {
        row["cell"]: Path(row["path"]).exists() and sha256(Path(row["path"])) == row["sha256"]
        for row in definition["inputs"]
    }
    candidate_ok = sha256(Path(definition["candidate"])) == definition["candidate_sha256"]
    manifest_ok = sha256(Path(definition["manifest"])) == definition["manifest_sha256"]
    with tempfile.TemporaryDirectory(prefix="external-manifold-rebuild-") as directory:
        rebuilt = Path(directory) / "run"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        environment["LOKY_MAX_CPU_COUNT"] = "8"
        environment["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
        subprocess.run(
            [sys.executable, str(RUNNER), "--candidate", definition["candidate"],
             "--manifest", definition["manifest"], "--out-dir", str(rebuilt)],
            cwd=ROOT, env=environment, check=True,
        )
        subprocess.run(
            [sys.executable, str(PLOTTER), "--results", str(rebuilt)],
            cwd=ROOT, env=environment, check=True,
        )
        rebuild_checks = {
            name: sha256(args.results / name) == sha256(rebuilt / name)
            for name in COMPARABLE
        }
    result = {
        "pass": bool(
            all(source_checks.values()) and all(input_checks.values())
            and candidate_ok and manifest_ok and all(rebuild_checks.values())
        ),
        "source_hashes_match": all(source_checks.values()),
        "source_checks": source_checks,
        "input_hashes_match": all(input_checks.values()),
        "input_checks": input_checks,
        "candidate_hash_match": candidate_ok,
        "manifest_hash_match": manifest_ok,
        "fresh_rebuild_match": all(rebuild_checks.values()),
        "rebuild_checks": rebuild_checks,
    }
    path = args.results / "REBUILD_VERIFICATION.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
