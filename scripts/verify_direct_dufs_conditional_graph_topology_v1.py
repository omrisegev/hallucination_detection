#!/usr/bin/env python3
"""Verify resume stability and an isolated fresh deterministic rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/direct_dufs_conditional_graph_topology_audit_v1.py"
DEFAULT_ROOT = ROOT / "results/direct_dufs_conditional_graph_topology_audit_v1"
FILES = (
    "RUN_DEFINITION.json",
    "CELL_GRAPH_METRICS.csv",
    "LANE_GRAPH_SUMMARY.csv",
    "PAIRED_INTERVALS.csv",
    "REPRESENTATION_DIAGNOSTICS.json",
    "CONTROL_CHECKS.json",
    "DECISION.json",
    "REPORT.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root: Path) -> dict[str, str]:
    output = {name: sha256(root / name) for name in FILES}
    for path in sorted((root / "checkpoints").glob("*.json")):
        output[f"checkpoints/{path.name}"] = sha256(path)
    return output


def invoke(out: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment.setdefault("LOKY_MAX_CPU_COUNT", "8")
    subprocess.run(
        [sys.executable, str(RUNNER), "--out", str(out)],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    before = manifest(root)
    invoke(root)
    after_resume = manifest(root)
    rebuild_root = Path(tempfile.mkdtemp(prefix="dufs_graph_rebuild_"))
    invoke(rebuild_root)
    rebuilt = manifest(rebuild_root)
    resume_match = before == after_resume
    rebuild_match = before == rebuilt
    differing_resume = sorted(
        key for key in set(before) | set(after_resume)
        if before.get(key) != after_resume.get(key)
    )
    differing_rebuild = sorted(
        key for key in set(before) | set(rebuilt)
        if before.get(key) != rebuilt.get(key)
    )
    payload = {
        "resume_match": resume_match,
        "fresh_rebuild_match": rebuild_match,
        "pass": bool(resume_match and rebuild_match),
        "canonical_root": str(root),
        "preserved_rebuild_root": str(rebuild_root),
        "canonical_manifest": before,
        "resume_differences": differing_resume,
        "fresh_rebuild_differences": differing_rebuild,
    }
    output = root / "REBUILD_VERIFICATION.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
