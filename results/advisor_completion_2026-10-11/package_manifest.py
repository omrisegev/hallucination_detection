"""Hash the local delivery and verify pinned identities without network access."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
MANIFEST = PACKAGE / "PACKAGE_MANIFEST.json"
RUNTIME = [
    "spectral_utils/advisor_controls.py", "spectral_utils/advisor_lr.py",
    "spectral_utils/advisor_statistics.py", "scripts/run_advisor_v11_lr.py",
    "spectral_utils/fusion_utils.py", "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
]
SKIP_PARTS = {"__pycache__", "mpl_cache", ".pytest_cache"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role(path: Path) -> str:
    rel = path.relative_to(PACKAGE)
    if rel.parts[0] == "inputs":
        return "input"
    if rel.parts[0] == "sources":
        return "source"
    if path.suffix == ".py":
        return "code"
    if rel.parts[0] == "protocols":
        return "protocol"
    return "output"


def build() -> dict:
    files = []
    for path in sorted(PACKAGE.rglob("*")):
        if (not path.is_file() or path == MANIFEST or
                set(path.relative_to(PACKAGE).parts) & SKIP_PARTS):
            continue
        files.append({"path": str(path.relative_to(PACKAGE)),
                      "role": role(path), "size": path.stat().st_size,
                      "sha256": sha(path)})
    runtime = []
    for rel in RUNTIME:
        path = ROOT / rel
        archived = PACKAGE / "sources/runtime" / rel
        if not archived.is_file() or sha(path) != sha(archived):
            raise RuntimeError(f"Runtime source differs from delivered archive: {rel}")
        runtime.append({"repo_relative_path": rel, "size": path.stat().st_size,
                        "sha256": sha(path),
                        "archive": str(archived.relative_to(PACKAGE))})
    result = {
        "status": "HASHED_LOCAL_DELIVERY",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "output_date_requested": "2026-10-11",
        "python": {"executable": sys.executable, "version": sys.version},
        "versions": {n: importlib.metadata.version(n) for n in
                     ["numpy", "scipy", "scikit-learn", "matplotlib"]},
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "repo_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "publication": json.loads((PACKAGE / "PUBLICATION_STATUS.json").read_text()),
        "source_commits": {"research": "080877eb9a8369341d647f4115a3b475c040b46e",
                           "v10": "53ef62e36e80b90de30b83d3fee33e1fb85b82c3"},
        "exclusions": {"manifest_self": True, "runtime_caches": sorted(SKIP_PARTS)},
        "files": files, "runtime_files": runtime,
        "file_count": len(files), "total_bytes": sum(f["size"] for f in files),
    }
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify(full: bool = False) -> dict:
    result = json.loads(MANIFEST.read_text())
    if sys.version != result["python"]["version"]:
        raise RuntimeError("Python version differs from the recorded numerical environment")
    for name, version in result["versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Changed numerical dependency: {name}")
    checked = 0
    for entry in result["files"]:
        if not full and entry["role"] not in {"input", "source", "code", "protocol"}:
            continue
        path = PACKAGE / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"] or sha(path) != entry["sha256"]:
            raise RuntimeError(f"Changed/missing package identity: {entry['path']}")
        checked += 1
    for entry in result["runtime_files"]:
        if sha(ROOT / entry["repo_relative_path"]) != entry["sha256"]:
            raise RuntimeError(f"Changed runtime source: {entry['repo_relative_path']}")
        checked += 1
    return {"status": "PASS", "checked_files": checked, "full_outputs": full,
            "manifest_sha256": sha(MANIFEST)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.full), indent=2))
    else:
        receipt = build()
        print(json.dumps({k: receipt[k] for k in ["status", "file_count", "total_bytes"]}, indent=2))
