#!/usr/bin/env python3
"""Create the physically target-free input archive for geometry fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import DEFAULT_BUNDLE  # noqa: E402
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.graph_geometry_selection import (  # noqa: E402
    validate_physically_label_free_members,
)


VERSION = "graph-geometry-physical-label-isolation-v1-2026-08-23"
DEFAULT_OUT = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "label_free_input" / "cells_target_free.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    if value.dtype == object:
        digest.update(json.dumps(
            [str(item) for item in value.tolist()],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode())
    else:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    allowed = tuple(
        f"{cell}__{suffix}"
        for cell in INSCOPE
        for suffix in ("V", "pool", "hand_signs")
    )
    arrays = {}
    hashes = {}
    with np.load(args.source, allow_pickle=True) as source:
        source_registry = tuple(source.files)
        for key in allowed:
            value = np.asarray(source[key])
            arrays[key] = value
            hashes[key] = array_hash(value)
    np.savez_compressed(args.out, **arrays)
    with np.load(args.out, allow_pickle=True) as target_free:
        validate_physically_label_free_members(target_free.files, INSCOPE)
        for key in target_free.files:
            if array_hash(target_free[key]) != hashes[key]:
                raise RuntimeError(f"member changed during isolation: {key}")

    forbidden_source_members = sorted(set(source_registry) - set(allowed))
    manifest = {
        "version": VERSION,
        "source_path": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "output_path": str(args.out.resolve()),
        "output_sha256": sha256_file(args.out),
        "allowed_suffixes": ["V", "pool", "hand_signs"],
        "output_member_count": len(allowed),
        "output_members": list(allowed),
        "output_member_hashes": hashes,
        "source_registry_was_enumerated": True,
        "source_target_arrays_loaded": False,
        "source_forbidden_members_not_copied": forbidden_source_members,
        "output_contains_target_like_members": False,
    }
    write_json(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps({
        "status": "physically_target_free_bundle_created",
        "output": str(args.out),
        "sha256": manifest["output_sha256"],
        "members": len(allowed),
        "forbidden_source_members_not_copied": len(forbidden_source_members),
    }, indent=2))


if __name__ == "__main__":
    main()
