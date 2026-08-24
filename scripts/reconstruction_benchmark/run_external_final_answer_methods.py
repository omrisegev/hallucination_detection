#!/usr/bin/env python3
"""Run the 13 target-free methods on applicable external final-answer cells."""

from __future__ import annotations

import argparse
import json
from importlib import metadata as importlib_metadata
import platform
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    SCORE_FREEZE_SCHEMA_VERSION,
    load_external_registry,
    load_prepared_external_cell,
)
from spectral_utils.reconstruction_benchmark.external_ab import (  # noqa: E402
    REQUIRED_PREPARATION_SOURCES,
    current_feature_contract_bindings,
    validate_scientific_input_manifest,
    verify_current_source_snapshot,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.methods import (  # noqa: E402
    PRIMARY_METHOD_IDS,
    PRIMARY_METHOD_SPECS,
    run_method,
)
from spectral_utils.reconstruction_benchmark.serialization import write_score_result  # noqa: E402


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"
METHODS_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/methods.json"
SUCCESS = {"OK", "OK_FALLBACK"}

SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/feature_contract.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/external_ab.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
    "spectral_utils/reconstruction_benchmark/serialization.py",
    "spectral_utils/reconstruction_benchmark/fit_validation.py",
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


def _load_input_manifest(path: Path, *, registry, repo: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    recorded = payload.pop("payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(payload)):
        raise RuntimeError("external input manifest payload hash failed")
    if manifest.get("external_registry_sha256") != registry.sha256:
        raise RuntimeError("external input manifest binds another registry")
    if manifest.get("population_registry_sha256") != registry.population_registry_sha256:
        raise RuntimeError("external input manifest binds another population registry")
    if manifest.get("labels_opened") is not False or manifest.get("historical_scores_opened") is not False:
        raise RuntimeError("external input manifest does not prove target isolation")
    if manifest.get("feature_contract_id") != current_feature_contract_bindings(repo)["feature_contract_id"]:
        raise RuntimeError("external input manifest binds another feature contract")
    if manifest.get("mixed_v2_applied_exactly_once") is not True:
        raise RuntimeError("external input manifest does not attest exactly one mixed-v2 pass")
    snapshot = manifest.get("preparation_source_snapshot", {})
    if manifest.get("preparation_source_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise RuntimeError("external preparation snapshot binding failed")
    verify_current_source_snapshot(
        snapshot,
        repo=repo,
        required_paths=REQUIRED_PREPARATION_SOURCES,
        name="external preparation",
    )
    return manifest


def _validate_input_cells(manifest: dict, registry) -> bool:
    rows = manifest.get("cells", [])
    identifiers = [str(item.get("cell_id", "")) for item in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("external input manifest has empty or duplicate cell IDs")
    unknown = set(identifiers) - set(registry.by_cell)
    if unknown:
        raise RuntimeError(f"external input manifest has unregistered cells: {sorted(unknown)}")
    for row in rows:
        spec = registry.by_cell[row["cell_id"]]
        if row.get("population_id") != spec.population_id:
            raise RuntimeError(f"{spec.cell_id}: input population binding changed")
        if spec.fit_policy == "forbidden" and row.get("status") != spec.configured_status:
            raise RuntimeError(f"{spec.cell_id}: terminal applicability status changed")
        if row.get("status") == "ELIGIBLE" and not row.get("artifact_path"):
            raise RuntimeError(f"{spec.cell_id}: eligible row lacks a prepared artifact")
    return set(identifiers) == set(registry.by_cell)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
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
    if fit_root.exists() and any(fit_root.iterdir()):
        raise FileExistsError(f"external fit directory is not empty: {fit_root}")
    input_manifest_path = input_root / "MANIFEST.json"
    input_manifest = _load_input_manifest(input_manifest_path, registry=registry, repo=REPO)
    full_input_roster = _validate_input_cells(input_manifest, registry)
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
        input_manifest = validate_scientific_input_manifest(
            input_manifest_path,
            registry=registry,
            repo=REPO,
            input_root=input_root,
        )
    contract_bindings = current_feature_contract_bindings(REPO)
    snapshot = _source_snapshot(allow_dirty_debug=args.allow_dirty_debug)
    fit_root.mkdir(parents=True, exist_ok=False)
    prefit = {
        "schema_version": "reconstruction-external-prefit-snapshot-v1",
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full": scientific_full,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "input_manifest_payload_sha256": input_manifest["payload_sha256"],
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "preparation_source_snapshot_sha256": input_manifest["preparation_source_snapshot_sha256"],
        "feature_contract_bindings": contract_bindings,
        "method_ids": list(method_ids),
        "cell_ids": [item["cell_id"] for item in records],
        "source_snapshot": snapshot,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
    }
    prefit["payload_sha256"] = sha256_bytes(canonical_json_bytes(prefit))
    atomic_write_json(fit_root / "FIT_SOURCE_SNAPSHOT.json", prefit)

    all_records = []
    for cell_record in records:
        artifact = input_root / cell_record["artifact_path"]
        cell, _ = load_prepared_external_cell(artifact_path=artifact, record=cell_record)
        cell_root = fit_root / "cells" / cell.cell_id
        method_records = []
        for method_id in method_ids:
            result = run_method(method_id, cell)
            record = write_score_result(result, cell.row_ids, cell_root / method_id)
            method_records.append(record)
            all_records.append({
                "cell_id": cell.cell_id,
                "population_id": cell.population_id,
                "method_id": method_id,
                "method_version_id": record["method_version_id"],
                "config_sha256": record["config_sha256"],
                "status": record["status"],
                "prepared_matrix_sha256": record["prepared_matrix_sha256"],
                "score_sha256": record["score_sha256"],
                "record_sha256": record["record_sha256"],
                "record_path": (cell_root / method_id / "RECORD.json").relative_to(fit_root).as_posix(),
                "score_path": (cell_root / method_id / "score.npz").relative_to(fit_root).as_posix()
                if record["score_path"] else None,
                "artifacts_sha256": record["artifacts_sha256"],
                "artifacts_path": (
                    (cell_root / method_id / "artifacts.npz").relative_to(fit_root).as_posix()
                    if record["artifacts_path"] else None
                ),
                "artifact_index_sha256": record["artifact_index_sha256"],
                "artifact_index_path": (
                    cell_root / method_id / "ARTIFACT_INDEX.json"
                ).relative_to(fit_root).as_posix(),
            })
        atomic_write_json(cell_root / "CELL_FIT_MANIFEST.json", {
            "schema_version": "reconstruction-external-cell-fit-v1",
            "cell_id": cell.cell_id,
            "population_id": cell.population_id,
            "prepared_matrix_sha256": cell.matrix_sha256,
            "labels_opened": False,
            "method_records": method_records,
        })
        print(f"completed {cell.cell_id}", flush=True)

    expected = len(records) * len(method_ids)
    complete = (
        bool(records)
        and len(all_records) == expected
        and len({(item["cell_id"], item["method_id"]) for item in all_records}) == expected
        and all(item["status"] in SUCCESS and item["score_sha256"] for item in all_records)
    )
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
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "preparation_source_snapshot_sha256": input_manifest["preparation_source_snapshot_sha256"],
        "feature_contract_bindings": contract_bindings,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "input_manifest_payload_sha256": input_manifest["payload_sha256"],
        "prefit_snapshot_sha256": sha256_file(fit_root / "FIT_SOURCE_SNAPSHOT.json"),
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
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
    if complete:
        atomic_write_json(fit_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    else:
        atomic_write_json(fit_root / "FIT_INCOMPLETE.json", freeze)
        failures = [(item["cell_id"], item["method_id"], item["status"]) for item in all_records if item["status"] not in SUCCESS]
        raise RuntimeError(f"external fit incomplete; no score freeze issued: {failures}")
    if not scientific_full:
        atomic_write_json(fit_root / "DEBUG_RUN.json", {
            "scientific_run": False,
            "reason": "partial roster, incomplete applicability, selected methods, or dirty-debug override",
        })
    print(json.dumps({
        "all_expected_scores_present": complete,
        "n_records": len(all_records),
        "scientific_full": scientific_full,
        "score_freeze": str(fit_root / "SCORE_FREEZE_MANIFEST.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
