"""Deterministic, fail-closed artifact helpers for reconstruction benchmark v1."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_bytes(payload)


def atomic_write_json(path: str | Path, value) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _npy_bytes(array: np.ndarray) -> bytes:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object dtype is forbidden in reconstruction artifacts")
    stream = BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    return stream.getvalue()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray], *, compressed: bool = True) -> bytes:
    """Return a byte-stable NPZ archive.

    ``numpy.savez`` embeds current ZIP timestamps.  A fixed timestamp and sorted
    member order are required for the independent A/B byte-identity gate.
    """

    stream = BytesIO()
    mode = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(stream, mode="w", compression=mode, compresslevel=9) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"unsafe NPZ member name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIME)
            info.compress_type = mode
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    return stream.getvalue()


def atomic_write_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> str:
    payload = deterministic_npz_bytes(arrays)
    return atomic_write_bytes(path, payload)


def load_npz_no_pickle(
    path: str | Path,
    *,
    members: tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load NPZ members without pickle, optionally materializing a subset.

    The optional subset is important at target-free fit boundaries: an input
    archive may be a shared container that also carries response-risk arrays,
    but a fit worker must not materialize those arrays merely to validate or
    consume token telemetry.  Archive member names are still inspected, while
    only the requested ``members`` are read into memory.
    """

    with np.load(path, allow_pickle=False) as bundle:
        names = tuple(bundle.files) if members is None else tuple(members)
        unknown = sorted(set(names) - set(bundle.files))
        if unknown:
            raise KeyError(f"NPZ archive is missing requested members: {unknown}")
        return {name: np.asarray(bundle[name]) for name in names}


def load_npz_no_pickle_bytes(payload: bytes) -> dict[str, np.ndarray]:
    """Parse an NPZ from bytes already authenticated by a held file handle."""

    with np.load(BytesIO(payload), allow_pickle=False) as bundle:
        return {name: np.asarray(bundle[name]) for name in bundle.files}


def canonical_tree_manifest(root: str | Path) -> dict:
    base = Path(root)
    files = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"schema_version": "canonical-tree-manifest-v1", "files": files}
    payload["tree_sha256"] = sha256_bytes(canonical_json_bytes(files))
    return payload


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_npz",
    "canonical_json_bytes",
    "canonical_tree_manifest",
    "deterministic_npz_bytes",
    "load_npz_no_pickle",
    "load_npz_no_pickle_bytes",
    "sha256_bytes",
    "sha256_file",
]
