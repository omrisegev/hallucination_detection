#!/usr/bin/env python3
"""Run the strict localization fit capsule and freeze its 27-system scores."""

from __future__ import annotations

import argparse
import base64
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.fit_firewall import build_fit_audit_policy  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    canonical_tree_manifest,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    SCORE_FREEZE_SCHEMA_VERSION,
    payload_sha256,
    validate_fit_manifest,
)
from spectral_utils.reconstruction_benchmark.localization_fit import (  # noqa: E402
    load_localization_score_bundle,
)


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"

FIT_CAPSULE_CODE_ALLOWLIST = (
    "spectral_utils/fusion_utils.py",
    "spectral_utils/upcr.py",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/localization_contract.py",
    "spectral_utils/reconstruction_benchmark/localization_fit.py",
)

SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/localization.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "spectral_utils/fixed_application_pipelines.py",
    "spectral_utils/token_feature_views.py",
    "spectral_utils/repeated_measurement_reliability.py",
    "spectral_utils/reconstruction_benchmark/localization_contract.py",
    "spectral_utils/reconstruction_benchmark/localization_preparation.py",
    "spectral_utils/reconstruction_benchmark/localization_comparators.py",
    "spectral_utils/reconstruction_benchmark/localization_fit.py",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py",
    "scripts/reconstruction_benchmark/prepare_localization.py",
    "scripts/reconstruction_benchmark/run_localization_methods.py",
    "scripts/reconstruction_benchmark/localization_fit_worker.py",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _source_snapshot(*, allow_dirty_debug: bool) -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    clean = not status.strip()
    if not clean and not allow_dirty_debug:
        raise RuntimeError("scientific localization fitting requires a clean worktree")
    files = [{"path": name, "sha256": sha256_file(REPO / name)} for name in SOURCE_FILES]
    value = {
        "git_head": head,
        "git_clean": clean,
        "git_status_sha256": sha256_bytes(status.encode("utf-8")),
        "files": files,
    }
    value["snapshot_sha256"] = payload_sha256(value)
    return value


def _copy_capsule(root: Path) -> Path:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"localization fit capsule is not empty: {root}")
    code = root / "code"
    code.mkdir(parents=True, exist_ok=False)
    for relative in FIT_CAPSULE_CODE_ALLOWLIST:
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    (code / "spectral_utils/__init__.py").write_text(
        '"""Minimal target-free localization fit capsule."""\n', encoding="utf-8"
    )
    (code / "spectral_utils/reconstruction_benchmark/__init__.py").write_text(
        '"""Minimal reconstruction localization fit package."""\n', encoding="utf-8"
    )
    worker = code / "scripts/reconstruction_benchmark/localization_fit_worker.py"
    worker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REPO / "scripts/reconstruction_benchmark/localization_fit_worker.py", worker
    )
    closure = {
        "schema_version": "reconstruction-localization-fit-code-closure-v1",
        "source_files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in FIT_CAPSULE_CODE_ALLOWLIST
        ],
        "generated_initializers": [
            "spectral_utils/__init__.py",
            "spectral_utils/reconstruction_benchmark/__init__.py",
        ],
        "excluded_module_classes": [
            "raw_sources", "preparation", "response_method_fitters", "label_loaders",
            "comparators", "error_taxonomy", "evaluation", "reporting",
        ],
    }
    closure["payload_sha256"] = payload_sha256(closure)
    atomic_write_json(code / "FIT_CODE_CLOSURE.json", closure)
    return code


def _policy(
    *,
    code_root: Path,
    input_root: Path,
    fit_root: Path,
    forbidden: list[tuple[str, Path]],
) -> dict:
    runtime_roots = {
        Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
        Path("/usr").resolve(), Path("/System").resolve(),
        Path("/Library").resolve(), Path("/dev").resolve(),
        Path("/private/var/db/timezone/tz").resolve(),
    }
    temp_root = fit_root.parent / ".fit_worker_tmp"
    temp_root.mkdir(parents=True, exist_ok=False)
    return build_fit_audit_policy(
        allowed_read_roots=[code_root.resolve(), input_root.resolve(), *runtime_roots],
        allowed_read_files=[input_root / "MANIFEST.json", Path("/proc/self/maps")],
        allowed_write_roots=[fit_root.resolve(), temp_root.resolve()],
        allowed_native_roots=[
            Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
            Path("/usr").resolve(), Path("/System").resolve(), Path("/Library").resolve(),
        ],
        forbidden_probes=[
            {"probe_id": probe_id, "path": str(path.resolve())}
            for probe_id, path in forbidden
        ],
    )


def _launch(
    *,
    code_root: Path,
    input_root: Path,
    fit_root: Path,
    release_id: str,
    build_id: str,
    policy: dict,
) -> None:
    worker = code_root / "scripts/reconstruction_benchmark/localization_fit_worker.py"
    temp_root = fit_root.parent / ".fit_worker_tmp"
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(temp_root),
        "TMPDIR": str(temp_root),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RECONSTRUCTION_LOCALIZATION_FIT_POLICY_B64": base64.b64encode(
            canonical_json_bytes(policy)
        ).decode("ascii"),
    }
    completed = subprocess.run(
        [
            sys.executable, "-I", "-B", str(worker),
            "--release-id", release_id,
            "--build", build_id,
            "--input-root", str(input_root.resolve()),
            "--fit-root", str(fit_root.resolve()),
        ],
        cwd=code_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        close_fds=True,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "restricted localization fit worker failed closed\n"
            + completed.stdout[-4000:] + completed.stderr[-4000:]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--allow-dirty-debug", action="store_true")
    args = parser.parse_args()

    build_root = args.release_root / args.release_id / f"build_{args.build}" / "localization"
    input_root = build_root / "inputs"
    fit_root = build_root / "fit"
    capsule_root = build_root / "fit_capsule"
    manifest = validate_fit_manifest(
        input_root / "MANIFEST.json", input_root=input_root,
        require_scientific=not args.allow_dirty_debug,
    )
    if manifest["release_id"] != args.release_id or manifest["build_id"] != args.build:
        raise RuntimeError("localization controller release/build binding failed")
    if fit_root.exists() and any(fit_root.iterdir()):
        raise FileExistsError(f"localization fit root is not empty: {fit_root}")
    snapshot = _source_snapshot(allow_dirty_debug=args.allow_dirty_debug)
    code_root = _copy_capsule(capsule_root)
    controller_root = (
        args.release_root.parent / "private_control" / args.release_id / "localization"
    )
    forbidden = [
        ("localization_registry", REPO / "configs/reconstruction_benchmark_v1/localization.json"),
        ("external_registry", REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"),
        ("population_registry", REPO / "configs/reconstruction_benchmark_v1/populations.json"),
        ("preparation_provenance", controller_root / f"build_{args.build}" / "preparation_provenance/MANIFEST.json"),
        ("comparator_projections", build_root / "comparator_projections"),
        ("preparation_module", REPO / "spectral_utils/reconstruction_benchmark/localization_preparation.py"),
        ("comparator_module", REPO / "spectral_utils/reconstruction_benchmark/localization_comparators.py"),
        ("evaluation_module", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
    ]
    policy = _policy(
        code_root=code_root, input_root=input_root, fit_root=fit_root,
        forbidden=forbidden,
    )
    _launch(
        code_root=code_root, input_root=input_root, fit_root=fit_root,
        release_id=args.release_id, build_id=args.build, policy=policy,
    )
    worker_path = fit_root / "WORKER_RESULT_MANIFEST.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker_payload = dict(worker)
    if worker_payload.pop("payload_sha256", None) != payload_sha256(worker_payload):
        raise RuntimeError("localization worker manifest hash failed")
    if worker.get("audit_policy_sha256") != policy["policy_sha256"]:
        raise RuntimeError("localization worker used another firewall policy")
    expected_probes = [row["probe_id"] for row in policy["forbidden_probes"]]
    if worker.get("denial_probes") != [
        {"probe_id": probe, "read_denied": True} for probe in expected_probes
    ]:
        raise RuntimeError("localization worker did not pass all denial probes")
    if worker.get("target_data_opened") is not False or worker.get("response_scores_refit") is not False:
        raise RuntimeError("localization worker crossed its target/response-fit boundary")
    if int(worker.get("n_records", -1)) != len(manifest["cells"]):
        raise RuntimeError("localization worker did not score every cell")
    records = worker["records"]
    for row in records:
        record_path = fit_root / row["record_path"]
        if sha256_file(record_path) != row["record_file_sha256"]:
            raise RuntimeError("localization output record hash failed")
        record, _arrays = load_localization_score_bundle(record_path)
        if record["record_sha256"] != row["record_sha256"]:
            raise RuntimeError("localization output payload hash disagrees")
    freeze = {
        "schema_version": SCORE_FREEZE_SCHEMA_VERSION,
        "release_id": args.release_id,
        "external_release_id": manifest["external_release_id"],
        "build_id": args.build,
        "scientific_full": not args.allow_dirty_debug,
        "source_snapshot": snapshot,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "sklearn": _package_version("scikit-learn"),
        },
        "input_manifest_sha256": sha256_file(input_root / "MANIFEST.json"),
        "input_manifest_payload_sha256": manifest["payload_sha256"],
        "input_tree": canonical_tree_manifest(input_root),
        "external_certificate_sha256": manifest["external_certificate_sha256"],
        "capsule_tree": canonical_tree_manifest(capsule_root),
        "fit_code_closure_sha256": sha256_file(code_root / "FIT_CODE_CLOSURE.json"),
        "audit_policy": policy,
        "audit_policy_sha256": policy["policy_sha256"],
        "worker_result_sha256": sha256_file(worker_path),
        "pre_freeze_fit_tree": canonical_tree_manifest(fit_root),
        "preparation_provenance_sha256": sha256_file(
            controller_root / f"build_{args.build}"
            / "preparation_provenance/MANIFEST.json"
        ),
        "comparator_projection_manifest_sha256": sha256_file(
            build_root / "comparator_projections/MANIFEST.json"
        ),
        "comparator_projection_tree": canonical_tree_manifest(
            build_root / "comparator_projections"
        ),
        "target_data_opened": False,
        "response_scores_refit": False,
        "n_cells": len(records),
        "n_systems_per_cell": 27,
        "records": records,
    }
    freeze["payload_sha256"] = payload_sha256(freeze)
    atomic_write_json(fit_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    print(json.dumps({
        "release_id": args.release_id,
        "build": args.build,
        "n_cells": len(records),
        "n_systems_per_cell": 27,
        "score_freeze_sha256": sha256_file(fit_root / "SCORE_FREEZE_MANIFEST.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
