"""Authenticated immutable Pythia input contract for A6-S0b.

This module is stdlib-only.  It authenticates the exact official revision tree
and the five files consumed by the frozen prompt-only NLL audit before Torch,
Transformers, or the rest of the project package is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import unicodedata
from typing import Any


SCHEMA_VERSION = "a6-s0b-pythia-input-v1-2026-08-15"
REPOSITORY = "EleutherAI/pythia-410m-deduped"
REVISION = "c4fc8d586d62df497f1f9b69d66d3ca419992d3e"
OFFICIAL_API_URL = (
    "https://huggingface.co/api/models/EleutherAI/pythia-410m-deduped/"
    "revision/c4fc8d586d62df497f1f9b69d66d3ca419992d3e?blobs=true"
)
ALL_PATHS = (
    ".gitattributes", "README.md", "config.json", "model.safetensors",
    "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class FrozenFile:
    path: str
    size: int
    git_blob_sha1: str
    lfs_sha256: str | None = None


SELECTED_FILES = (
    FrozenFile("config.json", 570, "0425fa136ba3f95d9428d832cfd7bfd82c78bf1f"),
    FrozenFile(
        "model.safetensors", 911_373_632,
        "0938c13c297d010679cfed8ad81ab03a475cc84a",
        "e7ae132489f63d5d86009a8178a75c7d5d195410d067fca01a3160623e370fae",
    ),
    FrozenFile(
        "special_tokens_map.json", 99,
        "0204ed10c186a4c7c68f55dff8f26087a45898d6",
    ),
    FrozenFile(
        "tokenizer.json", 2_113_710,
        "f74dfbfab8f97770a87769c739fb080c21c8bacc",
    ),
    FrozenFile(
        "tokenizer_config.json", 396,
        "f1860edb10f80bcaf7b023fce47c68a23b724c23",
    ),
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    prefix = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(prefix + value).hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def validate_official_tree(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except Exception as error:
        raise RuntimeError("BLOCKED_PYTHIA_ACCESS: invalid official JSON") from error
    if value.get("id") != REPOSITORY or value.get("sha") != REVISION \
            or value.get("gated") is not False:
        raise RuntimeError("BLOCKED_PYTHIA_ACCESS: official identity mismatch")
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError("BLOCKED_PYTHIA_ACCESS: official tree missing")
    by_path: dict[str, dict[str, Any]] = {}
    for row in siblings:
        if not isinstance(row, dict) or not isinstance(row.get("rfilename"), str):
            raise RuntimeError("BLOCKED_PYTHIA_ACCESS: official tree row invalid")
        path = unicodedata.normalize("NFC", row["rfilename"])
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or path in by_path:
            raise RuntimeError("BLOCKED_PYTHIA_ACCESS: official path invalid")
        by_path[path] = row
    if tuple(sorted(by_path, key=lambda item: item.encode("utf-8"))) != ALL_PATHS:
        raise RuntimeError("BLOCKED_PYTHIA_ACCESS: official revision tree changed")
    for expected in SELECTED_FILES:
        row = by_path[expected.path]
        if row.get("blobId") != expected.git_blob_sha1 or row.get("size") != expected.size:
            raise RuntimeError(f"official Pythia object changed: {expected.path}")
        lfs = row.get("lfs")
        if expected.lfs_sha256 is None:
            if lfs is not None:
                raise RuntimeError(f"unexpected Pythia LFS object: {expected.path}")
        elif not isinstance(lfs, dict) or lfs.get("sha256") != expected.lfs_sha256 \
                or lfs.get("size") != expected.size:
            raise RuntimeError(f"official Pythia LFS identity changed: {expected.path}")
    return {
        "id": REPOSITORY, "sha": REVISION, "gated": False,
        "siblings": [
            {
                key: row[key] for key in ("rfilename", "blobId", "size", "lfs")
                if key in row
            }
            for row in sorted(siblings, key=lambda item: item["rfilename"].encode("utf-8"))
        ],
    }


def verify_selected_bytes(spec: FrozenFile, payload: bytes) -> dict[str, Any]:
    if len(payload) != spec.size:
        raise RuntimeError(f"Pythia payload size mismatch: {spec.path}")
    payload_sha = sha256_bytes(payload)
    if spec.lfs_sha256 is None:
        if git_blob_sha1(payload) != spec.git_blob_sha1:
            raise RuntimeError(f"Pythia Git object mismatch: {spec.path}")
    else:
        if payload_sha != spec.lfs_sha256:
            raise RuntimeError(f"Pythia LFS payload mismatch: {spec.path}")
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{spec.lfs_sha256}\nsize {spec.size}\n"
        ).encode("ascii")
        if git_blob_sha1(pointer) != spec.git_blob_sha1:
            raise RuntimeError(f"Pythia LFS pointer mismatch: {spec.path}")
    return {
        "path": spec.path, "size": spec.size, "sha256": payload_sha,
        "git_blob_sha1": spec.git_blob_sha1, "lfs_sha256": spec.lfs_sha256,
    }


__all__ = [
    "ALL_PATHS", "FrozenFile", "OFFICIAL_API_URL", "REPOSITORY", "REVISION",
    "SCHEMA_VERSION", "SELECTED_FILES", "canonical_json_bytes", "git_blob_sha1",
    "sha256_bytes", "validate_official_tree", "verify_selected_bytes",
]
