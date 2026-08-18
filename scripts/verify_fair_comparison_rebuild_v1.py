#!/usr/bin/env python3
"""Fail closed unless two independently written fair packages are byte-identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.fair_comparisons.registry import (  # noqa: E402
    canonical_json_bytes,
    verify_hash_manifest,
)


def _load(root: Path) -> tuple[dict, bytes]:
    path = root / "HASH_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{path} is not canonical JSON with one trailing newline")
    if value.get("scope") != "all-files-except-exclude":
        raise ValueError(f"{path} must cover the complete result tree")
    if value.get("excluded_paths") != ["HASH_MANIFEST.json"]:
        raise ValueError(f"{path} may exclude only HASH_MANIFEST.json")
    return value, raw


def verify_rebuild(reference: Path, candidate: Path) -> dict:
    reference = reference.resolve()
    candidate = candidate.resolve()
    if reference == candidate:
        raise ValueError("reference and candidate must be independently written directories")
    first, first_raw = _load(reference)
    second, second_raw = _load(candidate)
    first_check = verify_hash_manifest(reference, first)
    second_check = verify_hash_manifest(candidate, second)
    if not first_check["ok"] or not second_check["ok"]:
        raise RuntimeError(
            f"input package manifest failure: reference={first_check['problems']}; "
            f"candidate={second_check['problems']}"
        )
    if first != second or first_raw != second_raw:
        first_rows = {row["path"]: row for row in first["files"]}
        second_rows = {row["path"]: row for row in second["files"]}
        changed = sorted(
            path
            for path in set(first_rows) | set(second_rows)
            if first_rows.get(path) != second_rows.get(path)
        )
        if first_raw != second_raw and not changed:
            changed = ["HASH_MANIFEST.json"]
        raise RuntimeError(f"independent rebuild mismatch in {len(changed)} files: {changed[:20]}")
    return {
        "schema": "fair_comparison_rebuild_verification_v1",
        "directories_independent": True,
        "file_count": len(first["files"]),
        "tree_sha256": first["tree_sha256"],
        "byte_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_rebuild(args.reference, args.candidate), sort_keys=True))


if __name__ == "__main__":
    main()
