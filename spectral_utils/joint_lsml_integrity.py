"""Fail-closed pre-label file integrity for the existing v2 study.

Hashes bytes only. A late record binds the current artifacts; it does not prove
that a launch-time registry existed. Old manifests and scores are never edited.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Sequence


RECORD_NAME = "INTEGRITY_RECORD_V2.json"
SCHEMA = "joint-lsml-prelabel-integrity-v2"


class IntegrityError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise IntegrityError(f"unsafe manifest path: {relative}")
    path = (Path(root).resolve() / Path(*rel.parts)).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise IntegrityError(f"missing or escaping artifact: {relative}")
    return path


def expected_run_files(cells: Sequence[str], n_outer: int, n_inner: int, require_r1: bool) -> set[str]:
    if not cells or len(set(cells)) != len(cells) or n_outer < 1 or n_inner < 1:
        raise IntegrityError("nonempty unique cells and positive fold counts are required")
    paths = {"folds/folds.json", "folds/FOLDS_SHA256.txt", "AUDIT_PRELABEL_RECEIPT.json"}
    for cell in cells:
        if not cell or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in cell):
            raise IntegrityError("invalid cell identifier")
        paths.update((f"cells/{cell}.npz", f"labels/{cell}_labels.npz"))
        for k in range(n_outer):
            prefix = f"structure/{cell}/outer{k}"
            for name in ("COMPLETE.json", "scores_outer.npz", "meta_outer.json", "moduleb.npz", "moduleb_meta.json",
                         "CENTERED_DIAGNOSTIC_REPAIR.json"):
                paths.add(f"{prefix}/{name}")
            for j in range(n_inner):
                paths.update((f"{prefix}/inner{j}/scores_inner.npz", f"{prefix}/inner{j}/meta_inner.json"))
            if require_r1:
                for name in ("scores_continuity.npz", "moduleb_grid.npz", "MANIFEST_AMEND_R1.json"):
                    paths.add(f"{prefix}/{name}")
    return paths


def relative_manifest(directory: Path) -> dict[str, str]:
    """The collision-free replacement for a recursive basename dictionary."""
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): file_sha256(contained_file(directory, path.relative_to(directory).as_posix()))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }


def create_late_record(
    root: Path, cells: Sequence[str], *, producer_root: Path, amendment: Path,
    audit_root: Path | None = None, n_outer: int = 5, n_inner: int = 5, require_r1: bool = True,
) -> dict:
    """Create only after complete structure, with an explicit dated amendment.

    The amendment must explain late provenance and distinguish producer source
    from later audit repairs. Its existence is not treated as a scientific audit
    pass; the registered independent pre-label scientific audit is still needed.
    """
    root, producer_root, amendment = Path(root).resolve(), Path(producer_root).resolve(), Path(amendment).resolve()
    destination = root / RECORD_NAME
    if destination.exists():
        raise IntegrityError("integrity record already exists; never overwrite a freeze")
    if (root / "evaluation").exists():
        raise IntegrityError("evaluation directory already exists; cannot claim a pre-evaluation record")
    if not amendment.is_file() or not amendment.read_text(encoding="utf-8").strip():
        raise IntegrityError("an explicit late-provenance amendment is required")
    expected = expected_run_files(cells, n_outer, n_inner, require_r1)
    files = {name: file_sha256(contained_file(root, name)) for name in sorted(expected)}
    # Freeze the observed producer sources in the record. This is explicitly a
    # late snapshot; it is not an assertion about in-memory modules at launch.
    sources = {}
    for relative_root in ("spectral_utils", "scripts/joint_lsml_optimization_v2"):
        for path in sorted((producer_root / relative_root).rglob("*.py")):
            sources[path.relative_to(producer_root).as_posix()] = file_sha256(path)
    if not sources:
        raise IntegrityError("producer source snapshot is empty")
    audit_root = Path(audit_root if audit_root is not None else producer_root).resolve()
    audit_sha = file_sha256(audit_root / "scripts/joint_lsml_optimization_v2/audit_prelabel.py")
    _validate_audit_receipt(root, audit_sha)
    record = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_status": "LATE_PRELABEL_SNAPSHOT_NOT_LAUNCH_PROOF",
        "cells": list(cells), "n_outer": n_outer, "n_inner": n_inner, "require_r1": require_r1,
        "files": files, "producer_source_sha256": sources,
        "audit_source_sha256": audit_sha,
        "amendment": {"text": amendment.read_text(encoding="utf-8"), "sha256": file_sha256(amendment)},
        "labels_decoded": False, "scientific_audit_required_separately": True,
    }
    # Exclusive creation prevents accidental replacement of an existing record.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return record


def verify_prelabel_record(
    root: Path, cells: Sequence[str], *, n_outer: int = 5, n_inner: int = 5, require_r1: bool = True,
) -> dict:
    root = Path(root).resolve()
    path = root / RECORD_NAME
    if not path.is_file():
        raise IntegrityError(f"missing {RECORD_NAME}; evaluator must not load labels")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_run_files(cells, n_outer, n_inner, require_r1)
    if record.get("schema") != SCHEMA or record.get("cells") != list(cells):
        raise IntegrityError("integrity schema/roster mismatch")
    if (record.get("n_outer"), record.get("n_inner"), record.get("require_r1")) != (n_outer, n_inner, require_r1):
        raise IntegrityError("fold/amendment contract mismatch")
    if record.get("provenance_status") != "LATE_PRELABEL_SNAPSHOT_NOT_LAUNCH_PROOF":
        raise IntegrityError("late provenance must be disclosed")
    if not record.get("producer_source_sha256") or not record.get("amendment", {}).get("text", "").strip():
        raise IntegrityError("missing producer snapshot or provenance amendment")
    if set(record.get("files", {})) != expected:
        raise IntegrityError("manifest does not bind the exact required artifact set")
    for name in sorted(expected):
        if file_sha256(contained_file(root, name)) != record["files"][name]:
            raise IntegrityError(f"artifact hash changed: {name}")
    _validate_audit_receipt(root, record.get("audit_source_sha256"))
    return record


def _validate_audit_receipt(root: Path, audit_sha: str | None) -> None:
    receipt = json.loads((root / "AUDIT_PRELABEL_RECEIPT.json").read_text(encoding="utf-8"))
    checks = receipt.get("checks", [])
    if receipt.get("status") != "PASS" or not checks or any(row.get("passed") is not True for row in checks):
        raise IntegrityError("pre-label scientific audit receipt is absent, empty or failed")
    if not audit_sha or receipt.get("audit_source_sha256") != audit_sha:
        raise IntegrityError("audit receipt does not match the snapshotted audit source")
