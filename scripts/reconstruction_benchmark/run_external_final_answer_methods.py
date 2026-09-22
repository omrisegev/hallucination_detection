#!/usr/bin/env python3
"""Run the 13 target-free methods on applicable external final-answer cells."""

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


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    ID_CONTRACT_VERSION,
    SCORE_FREEZE_SCHEMA_VERSION,
    external_id_contract_binding,
    identity_key_id,
    load_identity_key,
    load_external_registry,
)
from spectral_utils.reconstruction_benchmark.external_fit_contract import (  # noqa: E402
    build_fit_row_identity_contract,
)
from spectral_utils.reconstruction_benchmark.external_ab import (  # noqa: E402
    assert_fit_safe_matches_preparation,
    current_feature_contract_bindings,
    validate_fit_safe_input_manifest,
    validate_scientific_input_manifest,
)
from spectral_utils.reconstruction_benchmark.fit_firewall import (  # noqa: E402
    build_fit_audit_policy,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    canonical_tree_manifest,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.methods import (  # noqa: E402
    PRIMARY_METHOD_IDS,
    PRIMARY_METHOD_SPECS,
)


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"
METHODS_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/methods.json"
SUCCESS = {"OK", "OK_FALLBACK"}

SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/fit_safe_feature_contract.json",
    "configs/reconstruction_benchmark_v1/fit_safe_feature_roster.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/external_ab.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/external_fit_contract.py",
    "spectral_utils/reconstruction_benchmark/external_fit_safe.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
    "spectral_utils/reconstruction_benchmark/serialization.py",
    "spectral_utils/reconstruction_benchmark/fit_validation.py",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/selectors/a2_groupfs.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/specrage_laplacian.py",
    "spectral_utils/fusion_aware_views.py",
    "spectral_utils/residual_graph_deem.py",
    "spectral_utils/contribution_subspace.py",
    "spectral_utils/graph_topology.py",
    "scripts/reconstruction_benchmark/run_external_final_answer_methods.py",
    "scripts/reconstruction_benchmark/external_fit_worker.py",
)

# Exact first-party Python closure visible to the restricted fit worker.  No
# preparation adapter, source registry, post-freeze evaluator, label loader,
# report builder, or error-taxonomy module is copied into the capsule.
FIT_CAPSULE_CODE_ALLOWLIST = (
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/fusion_aware_views.py",
    "spectral_utils/specrage_laplacian.py",
    "spectral_utils/residual_graph_deem.py",
    "spectral_utils/contribution_subspace.py",
    "spectral_utils/graph_topology.py",
    "spectral_utils/selectors/a2_groupfs.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/external_fit_contract.py",
    "spectral_utils/reconstruction_benchmark/external_fit_safe.py",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
    "spectral_utils/reconstruction_benchmark/serialization.py",
)
FIT_CAPSULE_CONFIG_ALLOWLIST = (
    "configs/reconstruction_benchmark_v1/methods.json",
    "configs/reconstruction_benchmark_v1/fit_safe_feature_contract.json",
    "configs/reconstruction_benchmark_v1/fit_safe_feature_roster.json",
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
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO,
        check=True, capture_output=True, text=True,
    ).stdout
    clean = not status.strip()
    if not clean and not allow_dirty_debug:
        raise RuntimeError("scientific external fitting requires a clean worktree")
    files = [{"path": value, "sha256": sha256_file(REPO / value)} for value in SOURCE_FILES]
    payload = {
        "git_head": head,
        "git_clean": clean,
        "git_status_sha256": sha256_bytes(status.encode("utf-8")),
        "files": files,
    }
    payload["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _validate_methods(method_ids: tuple[str, ...]) -> None:
    registry = json.loads(METHODS_REGISTRY.read_text(encoding="utf-8"))
    rows = {str(item["method_id"]): item for item in registry["methods"]}
    if tuple(item["method_id"] for item in registry["methods"]) != PRIMARY_METHOD_IDS:
        raise RuntimeError("executable and machine-readable 13-method rosters disagree")
    for method_id in method_ids:
        if method_id not in rows or method_id not in PRIMARY_METHOD_SPECS:
            raise KeyError(f"unregistered primary method: {method_id}")
        spec = PRIMARY_METHOD_SPECS[method_id]
        if rows[method_id]["method_version_id"] != spec.method_version_id:
            raise RuntimeError(f"{method_id}: version mismatch")
        if rows[method_id]["config_sha256"] != spec.config_sha256:
            raise RuntimeError(f"{method_id}: config hash mismatch")


def _load_input_manifest(path: Path, *, repo: Path, input_root: Path) -> dict:
    return validate_fit_safe_input_manifest(
        path,
        repo=repo,
        input_root=input_root,
        require_scientific=False,
    )


def _validate_input_cells(manifest: dict) -> bool:
    rows = manifest.get("cells", [])
    identifiers = [str(item.get("cell_id", "")) for item in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("external input manifest has empty or duplicate cell IDs")
    for row in rows:
        if row.get("status") == "ELIGIBLE" and not row.get("artifact_path"):
            raise RuntimeError(f"{row['cell_id']}: eligible row lacks a prepared artifact")
    return len(identifiers) == int(manifest.get("n_registered_cells", -1))


def _copy_fit_capsule(target: Path) -> Path:
    """Create a data-free code/config closure for the restricted worker."""

    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"external fit capsule is not empty: {target}")
    code = target / "code"
    code.mkdir(parents=True, exist_ok=False)
    for relative in FIT_CAPSULE_CODE_ALLOWLIST:
        source, destination = REPO / relative, code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # The repository package initializer eagerly imports model, judge, loader,
    # and plotting modules.  A fit worker needs none of those, and importing a
    # submodule would otherwise execute that broad surface.  Capsule-local
    # initializers are deliberately inert; scientific submodules are imported
    # explicitly after the audit hook is installed.
    (code / "spectral_utils/__init__.py").write_text(
        '"""Minimal target-free fit capsule package."""\n',
        encoding="utf-8",
    )
    (code / "spectral_utils/reconstruction_benchmark/__init__.py").write_text(
        '"""Minimal target-free reconstruction fit capsule."""\n',
        encoding="utf-8",
    )
    (code / "spectral_utils/selectors/__init__.py").write_text(
        '"""Minimal selector registry for the registered DUFS fit arm."""\n'
        "_REGISTRY = {}\n"
        "def register(name):\n"
        "    def decorate(function):\n"
        "        _REGISTRY[name] = function\n"
        "        return function\n"
        "    return decorate\n",
        encoding="utf-8",
    )
    for relative in FIT_CAPSULE_CONFIG_ALLOWLIST:
        source, destination = REPO / relative, code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (code / "scripts/reconstruction_benchmark").mkdir(parents=True)
    shutil.copy2(
        REPO / "scripts/reconstruction_benchmark/external_fit_worker.py",
        code / "scripts/reconstruction_benchmark/external_fit_worker.py",
    )
    closure = {
        "schema_version": "reconstruction-external-fit-code-closure-v1",
        "source_files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in FIT_CAPSULE_CODE_ALLOWLIST
        ],
        "config_files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in FIT_CAPSULE_CONFIG_ALLOWLIST
        ],
        "generated_initializers": [
            "spectral_utils/__init__.py",
            "spectral_utils/reconstruction_benchmark/__init__.py",
            "spectral_utils/selectors/__init__.py",
        ],
        "excluded_module_classes": [
            "preparation_adapters", "source_registries", "postfreeze_evaluation",
            "label_loaders", "reporting", "error_taxonomies",
        ],
    }
    closure["payload_sha256"] = sha256_bytes(canonical_json_bytes(closure))
    atomic_write_json(code / "FIT_CODE_CLOSURE.json", closure)
    return code


def _worker_policy(
    *,
    code_root: Path,
    input_root: Path,
    fit_root: Path,
    forbidden_paths: list[tuple[str, Path]],
) -> dict:
    runtime_roots = {
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path("/usr").resolve(),
        Path("/System").resolve(),
        Path("/Library").resolve(),
        Path("/dev").resolve(),
        Path("/private/var/db/timezone/tz").resolve(),
    }
    read_roots = [code_root.resolve(), input_root.resolve(), *runtime_roots]
    temp_root = fit_root.parent / ".fit_worker_tmp"
    temp_root.mkdir(parents=True, exist_ok=False)
    return build_fit_audit_policy(
        allowed_read_roots=read_roots,
        allowed_read_files=[
            input_root / "MANIFEST.json",
            # PyTorch probes this exact process-local runtime map on Linux.
            # The worker never maps controller secrets, raw sources, or labels.
            Path("/proc/self/maps"),
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
    cells: list[str] | None,
    methods: tuple[str, ...],
    policy: dict,
) -> None:
    worker = code_root / "scripts/reconstruction_benchmark/external_fit_worker.py"
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
    for method_id in methods:
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
        "LOKY_MAX_CPU_COUNT": "1",
        "RECONSTRUCTION_EXTERNAL_FIT_POLICY_B64": base64.b64encode(
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
            "restricted external fit worker failed closed\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )


def _load_worker_result(
    path: Path,
    *,
    input_manifest: dict,
    policy: dict,
    method_ids: tuple[str, ...],
    cell_ids: list[str],
) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(result)
    recorded = payload.pop("payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(payload)):
        raise RuntimeError("external fit worker-result hash failed")
    if result.get("schema_version") != "reconstruction-external-fit-worker-result-v1":
        raise RuntimeError("unexpected external fit worker-result schema")
    if result.get("audit_policy_sha256") != policy["policy_sha256"]:
        raise RuntimeError("external fit worker used another audit policy")
    if result.get("firewall_violations") != []:
        raise RuntimeError("external fit worker recorded a firewall violation")
    probes = result.get("denial_probes", ())
    expected_probe_ids = [row["probe_id"] for row in policy["forbidden_probes"]]
    if probes != [
        {"probe_id": probe_id, "read_denied": True}
        for probe_id in expected_probe_ids
    ]:
        raise RuntimeError("external fit worker did not pass every denial probe")
    if result.get("input_manifest_payload_sha256") != input_manifest["payload_sha256"]:
        raise RuntimeError("external fit worker bound another input manifest")
    if result.get("method_ids") != list(method_ids) or result.get("cell_ids") != cell_ids:
        raise RuntimeError("external fit worker roster differs from controller")
    if result.get("all_candidate_scores_present") is not True:
        raise RuntimeError("external fit worker candidates are incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    parser.add_argument(
        "--identity-key",
        type=Path,
        help="Controller-only key path used only as a worker denial probe.",
    )
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--allow-dirty-debug", action="store_true")
    args = parser.parse_args()

    registry = load_external_registry(
        repo=REPO,
        registry_path=args.registry,
        population_registry_path=args.populations,
    )
    method_ids = tuple(args.methods or PRIMARY_METHOD_IDS)
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("duplicate method request")
    _validate_methods(method_ids)
    release_build = args.release_root / args.release_id / f"build_{args.build}" / "external_final_answer"
    input_root = release_build / "inputs"
    fit_root = release_build / "fit"
    capsule_root = release_build / "fit_capsule"
    controller_root = (
        args.release_root.parent / "private_control" / args.release_id
        / "external_final_answer"
    )
    provenance_manifest_path = (
        controller_root / f"build_{args.build}" / "preparation_provenance"
        / "MANIFEST.json"
    )
    identity_key_path = args.identity_key or (controller_root / "external-id-v2.key")
    if fit_root.exists() and any(fit_root.iterdir()):
        raise FileExistsError(f"external fit directory is not empty: {fit_root}")
    input_manifest_path = input_root / "MANIFEST.json"
    input_manifest = _load_input_manifest(
        input_manifest_path, repo=REPO, input_root=input_root
    )
    full_input_roster = _validate_input_cells(input_manifest)
    records = [item for item in input_manifest["cells"] if item.get("status") == "ELIGIBLE"]
    requested = None if not args.cells else set(args.cells)
    if requested is not None:
        available = {item["cell_id"] for item in records}
        if requested - available:
            raise KeyError(f"requested cell is not prepared/eligible: {sorted(requested - available)}")
        records = [item for item in records if item["cell_id"] in requested]
    scientific_full = (
        requested is None
        and full_input_roster
        and method_ids == PRIMARY_METHOD_IDS
        and input_manifest.get("applicability_complete") is True
        and input_manifest.get("complete_eligible_roster") is True
        and not args.allow_dirty_debug
    )
    if scientific_full:
        input_manifest = validate_fit_safe_input_manifest(
            input_manifest_path, repo=REPO, input_root=input_root
        )
        preparation_manifest = validate_scientific_input_manifest(
            provenance_manifest_path,
            registry=registry,
            repo=REPO,
            input_root=input_root,
        )
        assert_fit_safe_matches_preparation(
            input_manifest,
            preparation_manifest,
            preparation_manifest_path=provenance_manifest_path,
        )
    else:
        preparation_manifest = None
    contract_bindings = current_feature_contract_bindings(REPO)
    identity_binding = input_manifest["identity_contract"]
    controller_key = load_identity_key(identity_key_path)
    if identity_key_id(controller_key) != identity_binding["key_id"]:
        raise RuntimeError("controller identity key does not match prepared opaque IDs")
    expected_fit_identity = build_fit_row_identity_contract(
        external_id_contract_binding(registry, identity_key=controller_key),
        identity_key=controller_key,
    )
    if identity_binding != expected_fit_identity:
        raise RuntimeError("fit manifest group-linkage commitment/key binding failed")
    del controller_key
    snapshot = _source_snapshot(allow_dirty_debug=args.allow_dirty_debug)
    code_root = _copy_fit_capsule(capsule_root)
    capsule_tree = canonical_tree_manifest(capsule_root)
    sentinel_path = controller_root / "FORBIDDEN_FIT_SENTINEL.json"
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    if not sentinel_path.exists():
        atomic_write_json(sentinel_path, {"controller_only": True})
    forbidden_paths = [
        ("full_external_registry", Path(args.registry)),
        ("population_registry", Path(args.populations)),
        ("controller_identity_key", Path(identity_key_path)),
        ("controller_provenance", provenance_manifest_path),
        ("controller_sentinel", sentinel_path),
        (
            "preparation_adapter_module",
            REPO / "spectral_utils/reconstruction_benchmark/external_final_answer.py",
        ),
        (
            "controller_ab_module",
            REPO / "spectral_utils/reconstruction_benchmark/external_ab.py",
        ),
        (
            "postfreeze_evaluation_module",
            REPO / "spectral_utils/reconstruction_benchmark/external_evaluation.py",
        ),
        (
            "error_taxonomy_module",
            REPO / "spectral_utils/residual_graph_deem_labels.py",
        ),
    ]
    if preparation_manifest is not None:
        first_source = next(
            (
                source
                for row in preparation_manifest["cells"]
                if row.get("status") == "ELIGIBLE"
                for source in row.get("source_files", ())
            ),
            None,
        )
        if first_source is None:
            raise RuntimeError("controller preparation manifest lacks a raw-source probe")
        forbidden_paths.append((
            "raw_telemetry_source",
            Path(preparation_manifest["source_root"]) / first_source["path"],
        ))
    policy = _worker_policy(
        code_root=code_root,
        input_root=input_root,
        fit_root=fit_root,
        forbidden_paths=forbidden_paths,
    )
    _launch_worker(
        code_root=code_root,
        input_root=input_root,
        fit_root=fit_root,
        release_id=args.release_id,
        build_id=args.build,
        cells=None if requested is None else sorted(requested),
        methods=method_ids,
        policy=policy,
    )
    worker_result_path = fit_root / "WORKER_RESULT_MANIFEST.json"
    worker_result = _load_worker_result(
        worker_result_path,
        input_manifest=input_manifest,
        policy=policy,
        method_ids=method_ids,
        cell_ids=[item["cell_id"] for item in records],
    )
    all_records = list(worker_result["records"])
    expected = len(records) * len(method_ids)
    complete = (
        bool(records)
        and len(all_records) == expected
        and all(item["status"] in SUCCESS and item["score_sha256"] for item in all_records)
    )
    prefit = {
        "schema_version": "reconstruction-external-prefit-snapshot-v3-controller-worker",
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full": scientific_full,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "input_manifest_payload_sha256": input_manifest["payload_sha256"],
        "external_registry_sha256": input_manifest["external_registry_sha256"],
        "population_registry_sha256": input_manifest["population_registry_sha256"],
        "preparation_attestation_sha256": input_manifest["preparation_attestation_sha256"],
        "preparation_manifest_sha256": input_manifest["preparation_manifest_sha256"],
        "feature_contract_bindings": contract_bindings,
        "identity_contract": identity_binding,
        "id_contract_version": ID_CONTRACT_VERSION,
        "method_ids": list(method_ids),
        "cell_ids": [item["cell_id"] for item in records],
        "source_snapshot": snapshot,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "fit_isolation_tier": "trusted_first_party_python_audit_hook_v1",
        "audit_policy": policy,
        "audit_policy_sha256": policy["policy_sha256"],
        "denial_probes": worker_result["denial_probes"],
        "firewall_violations": worker_result["firewall_violations"],
        "worker_result_sha256": sha256_file(worker_result_path),
        "worker_result_payload_sha256": worker_result["payload_sha256"],
        "capsule_tree_sha256": capsule_tree["tree_sha256"],
    }
    prefit["payload_sha256"] = sha256_bytes(canonical_json_bytes(prefit))
    atomic_write_json(fit_root / "FIT_SOURCE_SNAPSHOT.json", prefit)

    end_snapshot = _source_snapshot(allow_dirty_debug=args.allow_dirty_debug)
    if end_snapshot != snapshot:
        raise RuntimeError("source tree changed during external fitting")
    freeze = {
        "schema_version": SCORE_FREEZE_SCHEMA_VERSION,
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full": scientific_full,
        "all_expected_scores_present": bool(complete),
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "historical_scores_opened": False,
        "donors_used": False,
        "family_nrm_pgrd_regime": "A_within_cell_fully_unsupervised",
        "score_semantics": "higher_is_incorrect",
        "external_registry_sha256": input_manifest["external_registry_sha256"],
        "population_registry_sha256": input_manifest["population_registry_sha256"],
        "preparation_attestation_sha256": input_manifest["preparation_attestation_sha256"],
        "preparation_manifest_sha256": input_manifest["preparation_manifest_sha256"],
        "feature_contract_bindings": contract_bindings,
        "identity_contract": identity_binding,
        "id_contract_version": ID_CONTRACT_VERSION,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "input_manifest_payload_sha256": input_manifest["payload_sha256"],
        "prefit_snapshot_sha256": sha256_file(fit_root / "FIT_SOURCE_SNAPSHOT.json"),
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "fit_isolation_tier": "trusted_first_party_python_audit_hook_v1",
        "audit_policy": policy,
        "audit_policy_sha256": policy["policy_sha256"],
        "denial_probes": worker_result["denial_probes"],
        "firewall_violations": worker_result["firewall_violations"],
        "worker_result_sha256": sha256_file(worker_result_path),
        "worker_result_payload_sha256": worker_result["payload_sha256"],
        "capsule_tree_sha256": capsule_tree["tree_sha256"],
        "method_registry_sha256": sha256_file(METHODS_REGISTRY),
        "method_ids": list(method_ids),
        "cell_ids": [item["cell_id"] for item in records],
        "n_records": len(all_records),
        "expected_records": expected,
        "applicability_statuses": [
            {"cell_id": item["cell_id"], "status": item["status"]}
            for item in input_manifest["cells"]
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: _package_version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "deem")},
        },
        "records": all_records,
    }
    freeze["payload_sha256"] = sha256_bytes(canonical_json_bytes(freeze))
    if complete and scientific_full:
        atomic_write_json(fit_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    elif not complete:
        atomic_write_json(fit_root / "FIT_INCOMPLETE.json", freeze)
        failures = [(item["cell_id"], item["method_id"], item["status"]) for item in all_records if item["status"] not in SUCCESS]
        raise RuntimeError(f"external fit incomplete; no score freeze issued: {failures}")
    if complete and not scientific_full:
        atomic_write_json(fit_root / "DEBUG_RUN.json", {
            "scientific_run": False,
            "score_freeze_issued": False,
            "reason": "partial roster, selected methods, incomplete applicability, or dirty-debug override",
        })
    print(json.dumps({
        "all_expected_scores_present": complete,
        "n_records": len(all_records),
        "scientific_full": scientific_full,
        "score_freeze": (
            str(fit_root / "SCORE_FREEZE_MANIFEST.json")
            if scientific_full and complete else None
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
