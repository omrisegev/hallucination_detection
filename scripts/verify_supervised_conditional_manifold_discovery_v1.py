#!/usr/bin/env python3
"""Verify provenance, resume determinism, and a fresh rebuild of discovery v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/supervised_conditional_manifold_discovery_v1"
RUNNER = ROOT / "scripts/supervised_conditional_manifold_discovery_v1.py"
SEMANTIC_OUTPUTS = (
    "RUN_DEFINITION.json",
    "OUTER_CELL_METRICS.csv",
    "OUTER_FAMILY_SUMMARY.csv",
    "WEIGHT_STABILITY.csv",
    "CANDIDATE_SUMMARY.csv",
    "WHOLE_SEARCH_NULL.json",
    "CONTROLS.json",
    "DECISION.json",
    "FROZEN_CANDIDATE.json",
    "REPORT.md",
    "01_conditional_geometry_and_linear_advantage.png",
    "02_discovered_feature_weights.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def hashes(directory: Path) -> dict[str, str | None]:
    return {
        name: sha256(directory / name) if (directory / name).exists() else None
        for name in SEMANTIC_OUTPUTS
    }


def checkpoints_match(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return False
        return all(np.array_equal(a[key], b[key], equal_nan=True) for key in a.files)


def run_rebuild(out: Path, definition: dict, env: dict[str, str]) -> None:
    command = [
        sys.executable,
        str(RUNNER),
        "--bundle", str(ROOT / definition["bundle"]),
        "--out-dir", str(out),
        "--permutations", str(definition["conditional_permutations"]),
        "--null-reruns", str(definition["whole_search_null_reruns"]),
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    definition_path = out / "RUN_DEFINITION.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))

    source_checks = {
        item["path"]: sha256(ROOT / item["path"]) == item["sha256"]
        for item in definition["sources"]
    }
    bundle_match = sha256(ROOT / definition["bundle"]) == definition["bundle_sha256"]
    expected_environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    environment_match = definition.get("environment") == expected_environment
    if not all(source_checks.values()) or not bundle_match or not environment_match:
        raise RuntimeError("canonical provenance no longer matches the frozen run definition")

    canonical_hashes = hashes(out)
    canonical_checkpoint = out / "checkpoints/whole_search_null.npz"
    if any(value is None for value in canonical_hashes.values()) or not canonical_checkpoint.exists():
        raise RuntimeError("canonical output is incomplete")

    with tempfile.TemporaryDirectory(prefix="supervised-manifold-rebuild-") as directory:
        scratch = Path(directory)
        resumed = scratch / "resumed"
        fresh = scratch / "fresh"
        (resumed / "checkpoints").mkdir(parents=True)
        shutil.copy2(canonical_checkpoint, resumed / "checkpoints/whole_search_null.npz")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["LOKY_MAX_CPU_COUNT"] = "8"
        env["MPLCONFIGDIR"] = str(scratch / "mpl")

        print("verification: checkpoint-resume rebuild", flush=True)
        run_rebuild(resumed, definition, env)
        resume_hashes = hashes(resumed)
        resume_match = resume_hashes == canonical_hashes and checkpoints_match(
            canonical_checkpoint, resumed / "checkpoints/whole_search_null.npz"
        )

        print("verification: isolated fresh rebuild", flush=True)
        run_rebuild(fresh, definition, env)
        fresh_hashes = hashes(fresh)
        fresh_match = fresh_hashes == canonical_hashes and checkpoints_match(
            canonical_checkpoint, fresh / "checkpoints/whole_search_null.npz"
        )

    payload = {
        "pass": bool(resume_match and fresh_match),
        "resume_match": bool(resume_match),
        "fresh_rebuild_match": bool(fresh_match),
        "source_hashes_match": bool(all(source_checks.values())),
        "source_checks": source_checks,
        "bundle_sha256_match": bool(bundle_match),
        "environment_match": bool(environment_match),
        "canonical_output_hashes": canonical_hashes,
        "run_fingerprint": definition["run_fingerprint"],
    }
    write_json(out / "REBUILD_VERIFICATION.json", payload)
    print(json.dumps({
        "pass": payload["pass"],
        "resume_match": payload["resume_match"],
        "fresh_rebuild_match": payload["fresh_rebuild_match"],
    }, sort_keys=True), flush=True)
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
