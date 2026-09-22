#!/usr/bin/env python3
"""Controller for the firewalled 13-method EDIS/AIME A/B fit."""

from __future__ import annotations

import argparse
import base64
import json
from importlib import metadata as importlib_metadata
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.edis_ab import (  # noqa: E402
    FIT_SOURCE_PATHS,
    load_private_provenance,
    verify_current_source_snapshot,
)
from spectral_utils.reconstruction_benchmark.edis_fit import (  # noqa: E402
    PREFIT_SCHEMA,
    SCORE_FREEZE_SCHEMA,
    WORKER_RESULT_SCHEMA,
    load_fit_registry,
    validate_fit_safe_feature_contract,
    validate_method_registry,
)
from spectral_utils.reconstruction_benchmark.edis_identity import (  # noqa: E402
    SharedEdisIdentityController,
    controller_key_path,
    load_edis_identity_controller,
)
from spectral_utils.reconstruction_benchmark.edis_preparation import (  # noqa: E402
    PREPARATION_SOURCE_PATHS,
    assert_expected_preparation_status_roster,
    load_preparation_registry,
    load_preparation_status,
)
from spectral_utils.reconstruction_benchmark.fit_firewall import (  # noqa: E402
    build_fit_audit_policy,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_tree_manifest,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.methods import (  # noqa: E402
    PRIMARY_METHOD_IDS,
)


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_PRIVATE_CONTROL = REPO / "results/reconstruction_benchmark_v1/private_control"
DEFAULT_TARGET_FREE_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/edis_target_free.json"
DEFAULT_POSTFREEZE_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/edis_postfreeze.json"
SUCCESS = {"OK", "OK_FALLBACK"}
# PyTorch performs this best-effort, process-local read while loading its
# native dependencies.  On macOS the file does not exist, but the attempted
# open still emits a Python audit event before PyTorch catches FileNotFoundError.
# Permit this one runtime file, never a /proc directory or subtree.
EDIS_RUNTIME_READ_FILES = (Path("/proc/self/maps"),)
FIT_CAPSULE_MODULES = (
    "dufs_liu_feature_contract.py",
    "feature_contract.py",
    "specrage_views.py",
    "fusion_utils.py",
    "upcr.py",
    "laplacian_upcr.py",
    "dependency_fusion.py",
    "specrage_laplacian.py",
    "fusion_aware_views.py",
    "residual_graph_deem.py",
    "contribution_subspace.py",
    "graph_topology.py",
)
FIT_CAPSULE_RECONSTRUCTION_MODULES = (
    "contracts.py",
    "edis_fit.py",
    "fit_firewall.py",
    "io.py",
    "methods.py",
    "serialization.py",
)


def _payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_and_validate_controller_identity(
    *,
    private_control_root: Path,
    release_id: str,
    release_root: Path,
    repo: Path,
    fit_registry: Mapping[str, Any],
    private_provenance: Mapping[str, Any],
) -> SharedEdisIdentityController:
    identity = load_edis_identity_controller(
        private_control_root=private_control_root,
        release_id=release_id,
        create=False,
        release_root=release_root,
        repo=repo,
    )
    if dict(identity.public_binding) != fit_registry.get("identity_contract"):
        raise RuntimeError("EDIS controller key does not match the fit-safe key binding")
    if dict(identity.private_identity_binding) != private_provenance.get(
        "private_identity_contract"
    ):
        raise RuntimeError("EDIS controller key/private identity contract binding differs")
    if identity.private_identity_commitment_sha256 != fit_registry.get(
        "private_identity_contract_commitment_sha256"
    ):
        raise RuntimeError("EDIS controller key/private identity commitment differs")
    return identity


def _resolve_requested_identity_key(
    *,
    requested: Path | None,
    private_control_root: Path,
    release_id: str,
) -> Path:
    canonical = controller_key_path(
        private_control_root=private_control_root,
        release_id=release_id,
    ).resolve()
    if requested is not None and requested.resolve() != canonical:
        raise ValueError(
            "--identity-key must resolve to the sealed controller key for this release"
        )
    return canonical


def _validate_preparation_preflight(
    *, private_provenance: Mapping[str, Any], repo: Path, build_id: str
) -> None:
    verify_current_source_snapshot(
        private_provenance.get("preparation_source_snapshot", {}),
        repo=repo,
        name=f"EDIS preparation build {build_id}",
        require_clean=False,
        expected_paths=PREPARATION_SOURCE_PATHS,
    )


def _source_snapshot(*, allow_dirty_debug: bool) -> Mapping[str, Any]:
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
        raise RuntimeError("scientific EDIS fitting requires a clean worktree")
    files = [
        {"path": relative, "sha256": sha256_file(REPO / relative)}
        for relative in FIT_SOURCE_PATHS
    ]
    payload: dict[str, Any] = {
        "git_head": head,
        "git_clean": clean,
        "git_status_sha256": sha256_bytes(status.encode("utf-8")),
        "files": files,
    }
    payload["snapshot_sha256"] = _payload_sha256(payload)
    return payload


def _copy_fit_capsule(target: Path) -> Path:
    """Copy a data-free code/config closure for the restricted worker."""

    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"EDIS fit capsule is not empty: {target}")
    code = target / "code"
    code.mkdir(parents=True, exist_ok=False)
    package = code / "spectral_utils"
    package.mkdir()
    atomic_write_bytes(
        package / "__init__.py",
        b'"""Minimal frozen fit capsule; no eager project imports."""\n',
    )
    for name in FIT_CAPSULE_MODULES:
        shutil.copy2(REPO / "spectral_utils" / name, package / name)
    selectors = package / "selectors"
    selectors.mkdir()
    atomic_write_bytes(
        selectors / "__init__.py",
        (
            b'"""Minimal selector registry for the EDIS fit capsule."""\n'
            b'_REGISTRY = {}\n'
            b'def register(name):\n'
            b'    def decorator(function):\n'
            b'        _REGISTRY[name] = function\n'
            b'        return function\n'
            b'    return decorator\n'
        ),
    )
    shutil.copy2(
        REPO / "spectral_utils/selectors/a2_groupfs.py",
        selectors / "a2_groupfs.py",
    )
    reconstruction = package / "reconstruction_benchmark"
    reconstruction.mkdir()
    atomic_write_bytes(
        reconstruction / "__init__.py",
        b'"""Minimal reconstruction fit package."""\n',
    )
    for name in FIT_CAPSULE_RECONSTRUCTION_MODULES:
        shutil.copy2(
            REPO / "spectral_utils/reconstruction_benchmark" / name,
            reconstruction / name,
        )
    config_root = code / "configs/reconstruction_benchmark_v1"
    config_root.mkdir(parents=True)
    for name in (
        "methods.json",
        "fit_safe_feature_contract.json",
        "fit_safe_feature_roster.json",
    ):
        shutil.copy2(
            REPO / "configs/reconstruction_benchmark_v1" / name,
            config_root / name,
        )
    script_root = code / "scripts/reconstruction_benchmark"
    script_root.mkdir(parents=True)
    shutil.copy2(
        REPO / "scripts/reconstruction_benchmark/edis_fit_worker.py",
        script_root / "edis_fit_worker.py",
    )
    return code


def _worker_policy(
    *,
    code_root: Path,
    input_root: Path,
    fit_root: Path,
    forbidden_paths: Sequence[tuple[str, Path]],
) -> Mapping[str, Any]:
    runtime_roots = {
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path("/usr").resolve(),
        Path("/System").resolve(),
        Path("/Library").resolve(),
        Path("/dev").resolve(),
        Path("/private/var/db/timezone/tz").resolve(),
    }
    temp_root = fit_root.parent / ".fit_worker_tmp"
    temp_root.mkdir(parents=True, exist_ok=False)
    return build_fit_audit_policy(
        allowed_read_roots=[code_root.resolve(), input_root.resolve(), *runtime_roots],
        allowed_read_files=[
            input_root / "FIT_REGISTRY.json",
            *EDIS_RUNTIME_READ_FILES,
        ],
        allowed_write_roots=[fit_root.resolve(), temp_root.resolve()],
        allowed_native_roots=[
            Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
            Path("/usr").resolve(), Path("/System").resolve(),
            Path("/Library").resolve(),
        ],
        forbidden_probes=[
            {"probe_id": probe_id, "path": str(path.resolve())}
            for probe_id, path in forbidden_paths
        ],
    )


def _launch_worker(
    *,
    code_root: Path,
    input_root: Path,
    fit_root: Path,
    release_id: str,
    build_id: str,
    policy: Mapping[str, Any],
    cells: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> None:
    worker = code_root / "scripts/reconstruction_benchmark/edis_fit_worker.py"
    command = [
        sys.executable,
        "-I",
        "-B",
        str(worker),
        "--release-id", release_id,
        "--build", build_id,
        "--input-root", str(input_root.resolve()),
        "--fit-root", str(fit_root.resolve()),
    ]
    for cell_id in cells or ():
        command.extend(("--cell", cell_id))
    for method_id in methods or ():
        command.extend(("--method", method_id))
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
        "RECONSTRUCTION_EDIS_FIT_POLICY_B64": base64.b64encode(
            canonical_json_bytes(policy)
        ).decode("ascii"),
    }
    completed = subprocess.run(
        command,
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
            "restricted EDIS fit worker failed closed\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )


def _load_worker_result(
    path: Path,
    *,
    fit_registry: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(result)
    recorded = payload.pop("payload_sha256", None)
    if recorded != _payload_sha256(payload):
        raise RuntimeError("EDIS fit worker-result hash failed")
    if result.get("schema_version") != WORKER_RESULT_SCHEMA:
        raise RuntimeError("unexpected EDIS fit worker-result schema")
    if result.get("audit_policy_sha256") != policy["policy_sha256"]:
        raise RuntimeError("EDIS fit worker used another audit policy")
    if result.get("firewall_violations") != []:
        raise RuntimeError("EDIS fit worker recorded a firewall violation")
    expected_probes = [
        {"probe_id": row["probe_id"], "read_denied": True}
        for row in policy["forbidden_probes"]
    ]
    if result.get("denial_probes") != expected_probes:
        raise RuntimeError("EDIS fit worker did not pass every denial probe")
    if result.get("fit_registry_payload_sha256") != fit_registry["payload_sha256"]:
        raise RuntimeError("EDIS fit worker bound another fit registry")
    if result.get("fit_safe_feature_contract_sha256") != sha256_file(
        REPO / "configs/reconstruction_benchmark_v1/fit_safe_feature_contract.json"
    ):
        raise RuntimeError("EDIS fit worker bound another fit-safe feature contract")
    expected_cells = [row["cell_id"] for row in fit_registry["cells"]]
    if (
        result.get("method_ids") != list(PRIMARY_METHOD_IDS)
        or result.get("cell_ids") != expected_cells
        or result.get("all_candidate_scores_present") is not True
        or result.get("scientific_full_build")
        != fit_registry["scientific_full_build"]
        or result.get("partial_descriptive_build")
        != fit_registry["partial_descriptive_build"]
        or result.get("headline_eligible") is not False
        or result.get("aggregate_metrics_allowed")
        != fit_registry["aggregate_metrics_allowed"]
        or result.get("preparation_status_commitment_sha256")
        != fit_registry["preparation_status_commitment_sha256"]
    ):
        raise RuntimeError("EDIS fit worker roster is incomplete or reordered")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--private-control-root", type=Path, default=DEFAULT_PRIVATE_CONTROL)
    parser.add_argument("--source-root", type=Path, default=REPO)
    parser.add_argument("--target-free-registry", type=Path, default=DEFAULT_TARGET_FREE_REGISTRY)
    parser.add_argument("--postfreeze-registry", type=Path, default=DEFAULT_POSTFREEZE_REGISTRY)
    parser.add_argument("--identity-key", type=Path)
    parser.add_argument("--allow-dirty-debug", action="store_true")
    args = parser.parse_args()

    lane_root = (
        args.release_root / args.release_id / f"build_{args.build}" / "edis"
    )
    input_root, fit_root = lane_root / "inputs", lane_root / "fit"
    capsule_root = lane_root / "fit_capsule"
    if fit_root.exists() and any(fit_root.iterdir()):
        raise FileExistsError(f"EDIS fit directory is not empty: {fit_root}")
    fit_registry_path = input_root / "FIT_REGISTRY.json"
    fit_registry = load_fit_registry(fit_registry_path)
    if (
        fit_registry.get("release_id") != args.release_id
        or fit_registry.get("build_id") != args.build
    ):
        raise RuntimeError("EDIS fit registry release/build binding failed")
    if fit_registry.get("preparation_registry_sha256") != sha256_file(args.target_free_registry):
        raise RuntimeError("EDIS fit registry is stale against the target-free registry")
    preparation_status_path = lane_root / "PREPARATION_STATUS.json"
    preparation_status = load_preparation_status(preparation_status_path)
    preparation_registry = load_preparation_registry(args.target_free_registry)
    assert_expected_preparation_status_roster(
        registry=preparation_registry, status=preparation_status
    )
    ready_status_ids = [
        row["cell_id"]
        for row in preparation_status["cells"]
        if row["status"] == "READY"
    ]
    if (
        preparation_status["status_commitment_sha256"]
        != fit_registry["preparation_status_commitment_sha256"]
        or preparation_status.get("release_id") != args.release_id
        or preparation_status.get("build_id") != args.build
        or ready_status_ids != [row["cell_id"] for row in fit_registry["cells"]]
        or preparation_status["scientific_full_build"]
        != fit_registry["scientific_full_build"]
        or preparation_status["partial_descriptive_build"]
        != fit_registry["partial_descriptive_build"]
    ):
        raise RuntimeError("EDIS preparation status differs from the fit registry")
    private_path = (
        args.private_control_root / args.release_id / "edis"
        / f"build_{args.build}" / "PREPARATION_PROVENANCE.json"
    )
    private = load_private_provenance(private_path)
    _validate_preparation_preflight(
        private_provenance=private,
        repo=REPO,
        build_id=args.build,
    )
    if private.get("identity_contract") != fit_registry.get("identity_contract"):
        raise RuntimeError("EDIS public/private identity binding differs before fit")
    if private.get("preparation_registry_sha256") != fit_registry.get("preparation_registry_sha256"):
        raise RuntimeError("EDIS public/private preparation registry binding differs")
    identity_key = _resolve_requested_identity_key(
        requested=args.identity_key,
        private_control_root=args.private_control_root,
        release_id=args.release_id,
    )
    _load_and_validate_controller_identity(
        private_control_root=args.private_control_root,
        release_id=args.release_id,
        release_root=args.release_root,
        repo=REPO,
        fit_registry=fit_registry,
        private_provenance=private,
    )

    validate_method_registry(REPO)
    fit_safe_feature_contract_sha = validate_fit_safe_feature_contract(REPO)
    before = _source_snapshot(allow_dirty_debug=args.allow_dirty_debug)
    clean_release = bool(before["git_clean"] and not args.allow_dirty_debug)
    scientific_full = bool(
        clean_release and fit_registry["scientific_full_build"]
    )
    descriptive_partial = bool(
        clean_release and fit_registry["partial_descriptive_build"]
    )
    code_root = _copy_fit_capsule(capsule_root)
    capsule_tree = canonical_tree_manifest(capsule_root)
    sentinel = (
        args.private_control_root / args.release_id / "edis" / "FORBIDDEN_FIT_SENTINEL.json"
    )
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    if not sentinel.exists():
        atomic_write_json(sentinel, {"controller_only": True})
    first_source = private["cells"][0]["source"]["path"]
    forbidden_paths = (
        ("target_free_registry", args.target_free_registry),
        ("postfreeze_registry", args.postfreeze_registry),
        ("controller_identity_key", identity_key),
        ("controller_preparation_provenance", private_path),
        ("controller_sentinel", sentinel),
        ("raw_telemetry_source", args.source_root / first_source),
        ("preparation_status", preparation_status_path),
    )
    policy = _worker_policy(
        code_root=code_root,
        input_root=input_root,
        fit_root=fit_root,
        forbidden_paths=forbidden_paths,
    )
    private_policy_path = private_path.parent / "FIT_AUDIT_POLICY.json"
    if private_policy_path.exists():
        raise FileExistsError(f"EDIS private fit policy already exists: {private_policy_path}")
    atomic_write_json(private_policy_path, policy)
    _launch_worker(
        code_root=code_root,
        input_root=input_root,
        fit_root=fit_root,
        release_id=args.release_id,
        build_id=args.build,
        policy=policy,
    )
    worker_path = fit_root / "WORKER_RESULT_MANIFEST.json"
    worker = _load_worker_result(
        worker_path,
        fit_registry=fit_registry,
        policy=policy,
    )
    records = list(worker["records"])
    expected = len(fit_registry["cells"]) * len(PRIMARY_METHOD_IDS)
    complete = (
        len(records) == expected
        and len({(row["cell_id"], row["method_id"]) for row in records}) == expected
        and all(row["status"] in SUCCESS and row["score_sha256"] for row in records)
    )
    prefit = {
        "schema_version": PREFIT_SCHEMA,
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full": scientific_full,
        "descriptive_partial": descriptive_partial,
        "headline_eligible": False,
        "aggregate_metrics_allowed": fit_registry["aggregate_metrics_allowed"],
        "preparation_status_commitment_sha256": fit_registry[
            "preparation_status_commitment_sha256"
        ],
        "fit_registry_sha256": sha256_file(fit_registry_path),
        "fit_registry_payload_sha256": fit_registry["payload_sha256"],
        "preparation_registry_sha256": fit_registry["preparation_registry_sha256"],
        "preparation_source_snapshot_sha256": fit_registry["preparation_source_snapshot_sha256"],
        "identity_contract": fit_registry["identity_contract"],
        "method_registry_sha256": validate_method_registry(REPO),
        "fit_safe_feature_contract_sha256": fit_safe_feature_contract_sha,
        "method_ids": list(PRIMARY_METHOD_IDS),
        "cell_ids": [row["cell_id"] for row in fit_registry["cells"]],
        "fit_source_snapshot": before,
        "fit_isolation_tier": "trusted_first_party_python_audit_hook_v1",
        "audit_policy_sha256": policy["policy_sha256"],
        "private_audit_policy_file_sha256": sha256_file(private_policy_path),
        "denial_probes": worker["denial_probes"],
        "firewall_violations": worker["firewall_violations"],
        "worker_result_sha256": sha256_file(worker_path),
        "worker_result_payload_sha256": worker["payload_sha256"],
        "capsule_tree_sha256": capsule_tree["tree_sha256"],
        "labels_opened": False,
        "raw_sources_opened": False,
        "group_structure_opened": False,
    }
    prefit["payload_sha256"] = _payload_sha256(prefit)
    atomic_write_json(fit_root / "FIT_SOURCE_SNAPSHOT.json", prefit)

    after = _source_snapshot(allow_dirty_debug=args.allow_dirty_debug)
    if after != before:
        raise RuntimeError("EDIS fit source tree changed during execution")
    freeze = {
        "schema_version": SCORE_FREEZE_SCHEMA,
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full": scientific_full,
        "descriptive_partial": descriptive_partial,
        "headline_eligible": False,
        "aggregate_metrics_allowed": fit_registry["aggregate_metrics_allowed"],
        "preparation_status_commitment_sha256": fit_registry[
            "preparation_status_commitment_sha256"
        ],
        "all_expected_scores_present": bool(complete),
        "fit_registry_sha256": sha256_file(fit_registry_path),
        "fit_registry_payload_sha256": fit_registry["payload_sha256"],
        "prefit_sha256": sha256_file(fit_root / "FIT_SOURCE_SNAPSHOT.json"),
        "preparation_registry_sha256": fit_registry["preparation_registry_sha256"],
        "preparation_source_snapshot_sha256": fit_registry["preparation_source_snapshot_sha256"],
        "identity_contract": fit_registry["identity_contract"],
        "method_registry_sha256": validate_method_registry(REPO),
        "fit_safe_feature_contract_sha256": fit_safe_feature_contract_sha,
        "method_ids": list(PRIMARY_METHOD_IDS),
        "cell_ids": [row["cell_id"] for row in fit_registry["cells"]],
        "n_records": len(records),
        "expected_records": expected,
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "raw_sources_opened_by_fit": False,
        "group_structure_opened_by_fit": False,
        "historical_scores_opened": False,
        "donors_used": False,
        "score_semantics": "higher_is_incorrect",
        "fit_source_snapshot_sha256": before["snapshot_sha256"],
        "fit_isolation_tier": "trusted_first_party_python_audit_hook_v1",
        "audit_policy_sha256": policy["policy_sha256"],
        "private_audit_policy_file_sha256": sha256_file(private_policy_path),
        "denial_probes": worker["denial_probes"],
        "firewall_violations": worker["firewall_violations"],
        "worker_result_sha256": sha256_file(worker_path),
        "worker_result_payload_sha256": worker["payload_sha256"],
        "capsule_tree_sha256": capsule_tree["tree_sha256"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "scipy", "scikit-learn", "torch", "deem")
            },
        },
        "records": records,
    }
    freeze["payload_sha256"] = _payload_sha256(freeze)
    if complete and (scientific_full or descriptive_partial):
        atomic_write_json(fit_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    else:
        atomic_write_json(fit_root / "FIT_INCOMPLETE_OR_DEBUG.json", freeze)
        raise RuntimeError(
            "EDIS fit is incomplete or debug-only; no score freeze was issued"
        )
    print(json.dumps({
        "release_id": args.release_id,
        "build_id": args.build,
        "n_records": len(records),
        "scientific_full": scientific_full,
        "descriptive_partial": descriptive_partial,
        "headline_eligible": False,
        "labels_opened_by_fit": False,
        "fit_isolation_tier": freeze["fit_isolation_tier"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
