"""Deterministic filesystem I/O for the fair-comparison result package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .registry import canonical_json_bytes, sha256_file


PACKAGE_IO_REVISION = "fair_comparison_package_io_v1.0.0"


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def write_long_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a stable long-form CSV; structured extras use canonical JSON cells."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (dict, list, tuple)):
                    normalized[field] = canonical_json_bytes(value).decode("utf-8")
                elif value is None:
                    normalized[field] = ""
                elif isinstance(value, bool):
                    normalized[field] = "true" if value else "false"
                else:
                    normalized[field] = value
            writer.writerow(normalized)


def indexed_pickle_rows(run_dir: str | Path) -> dict[str, Any]:
    """Hash-check and read an indexed acquisition without positional fallbacks."""

    root = Path(run_dir).resolve()
    index_paths = sorted(root.glob("**/INDEX.jsonl"))
    if not index_paths:
        raise FileNotFoundError(f"no INDEX.jsonl below {root}")
    rows: list[dict[str, Any]] = []
    provenance = []
    shard_assets = []
    for index_path in index_paths:
        entries = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        index_count = 0
        for entry in entries:
            relative = str(entry["path"])
            posix = PurePosixPath(relative)
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"unsafe indexed shard path: {relative}")
            shard = index_path.parent.joinpath(*posix.parts)
            if not shard.is_file():
                raise FileNotFoundError(shard)
            actual_hash = sha256_file(shard)
            if actual_hash != entry["sha256"]:
                raise ValueError(f"indexed shard hash mismatch: {shard}")
            if shard.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"indexed shard byte-size mismatch: {shard}")
            with shard.open("rb") as handle:
                shard_rows = pickle.load(handle)
            if not isinstance(shard_rows, list) or len(shard_rows) != int(entry["n_traces"]):
                raise ValueError(f"indexed shard count mismatch: {shard}")
            declared_keys = entry.get("keys")
            if declared_keys is not None:
                observed_keys = [
                    row.get("trace_key", row.get("question_id")) for row in shard_rows
                ]
                if observed_keys != declared_keys:
                    raise ValueError(f"indexed shard key mismatch: {shard}")
            rows.extend(shard_rows)
            index_count += len(shard_rows)
            shard_assets.append(
                {
                    "path": str(shard.relative_to(root)),
                    "size_bytes": shard.stat().st_size,
                    "sha256": actual_hash,
                }
            )
        provenance.append(
            {
                "index": str(index_path.relative_to(root)),
                "index_sha256": sha256_file(index_path),
                "n_shards": len(entries),
                "n_rows": index_count,
            }
        )
    package_hash = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "rows": rows,
        "package_hash": package_hash,
        "provenance": provenance,
        "shard_assets": shard_assets,
    }


def tree_manifest(
    root: str | Path,
    *,
    exclude_names: Sequence[str] = (".git", "__pycache__", ".DS_Store"),
) -> dict[str, Any]:
    """Content-hash a tree without following symlinks or recording mtimes."""

    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    excluded = set(exclude_names)
    files = []
    for directory, dir_names, file_names in os.walk(base, followlinks=False):
        dir_names[:] = sorted(name for name in dir_names if name not in excluded)
        for name in sorted(file_names):
            if name in excluded:
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"tree manifest refuses symlink: {path}")
            if not path.is_file():
                continue
            files.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    files.sort(key=lambda row: row["path"])
    return {
        "schema": "content_tree_manifest_v1",
        "root_label": base.name,
        "exclusions": sorted(excluded),
        "file_count": len(files),
        "size_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
        "tree_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


__all__ = [
    "PACKAGE_IO_REVISION",
    "indexed_pickle_rows",
    "tree_manifest",
    "write_json",
    "write_jsonl",
    "write_long_csv",
]
