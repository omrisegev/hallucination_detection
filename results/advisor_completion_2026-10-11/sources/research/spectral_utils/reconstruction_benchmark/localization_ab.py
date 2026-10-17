"""Fail-closed A/B attestation for localization inputs, scores, and projections."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from .external_ab import assert_external_ab_certificate
from .external_final_answer import load_external_registry
from .fit_firewall import validate_fit_audit_policy
from .io import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_tree_manifest,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .localization_comparators import (
    PROJECTION_MANIFEST_SCHEMA_VERSION,
    PROJECTION_SCHEMA_VERSION,
)
from .localization_contract import (
    SCORE_FREEZE_SCHEMA_VERSION,
    load_localization_registry,
    payload_sha256,
    primary_system_roster,
    validate_fit_manifest,
)
from .localization_fit import load_localization_score_bundle
from .methods import PRIMARY_METHOD_IDS


AB_CERTIFICATE_SCHEMA_VERSION = "reconstruction-localization-ab-certificate-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCALIZATION_REGISTRY = REPO_ROOT / "configs/reconstruction_benchmark_v1/localization.json"
DEFAULT_EXTERNAL_REGISTRY = REPO_ROOT / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATION_REGISTRY = REPO_ROOT / "configs/reconstruction_benchmark_v1/populations.json"
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "results/reconstruction_benchmark_v1/source_overlays/external_final_answer_v1"
)
EXPECTED_FIT_CAPSULE_SOURCES = (
    "spectral_utils/fusion_utils.py",
    "spectral_utils/upcr.py",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/localization_contract.py",
    "spectral_utils/reconstruction_benchmark/localization_fit.py",
)
EXPECTED_FIT_SOURCE_FILES = (
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
EXPECTED_DENIAL_PROBES = (
    "localization_registry", "external_registry", "population_registry",
    "preparation_provenance", "comparator_projections", "preparation_module",
    "comparator_module", "evaluation_module",
)


def _hashed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != payload_sha256(payload):
        raise RuntimeError(f"payload hash failed: {path}")
    return value


def _safe_child(root: Path, relative: Any, *, description: str) -> Path:
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{description} escaped its registered root") from exc
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tree_without(root: Path, excluded: set[str]) -> dict[str, Any]:
    tree = canonical_tree_manifest(root)
    files = [row for row in tree["files"] if row["path"] not in excluded]
    return {
        "schema_version": "canonical-tree-manifest-v1",
        "files": files,
        "tree_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def _validate_source_snapshot(snapshot: Mapping[str, Any], *, repo: Path) -> None:
    payload = dict(snapshot)
    recorded = payload.pop("snapshot_sha256", None)
    if (
        recorded != payload_sha256(payload)
        or snapshot.get("git_clean") is not True
        or snapshot.get("git_status_sha256") != sha256_bytes(b"")
    ):
        raise RuntimeError("localization source snapshot is partial or malformed")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if snapshot.get("git_head") != head:
        raise RuntimeError("localization source snapshot is from another commit")
    files = snapshot.get("files", ())
    paths = [str(row.get("path", "")) for row in files]
    if tuple(paths) != EXPECTED_FIT_SOURCE_FILES:
        raise RuntimeError("localization source snapshot file roster is malformed")
    for row in files:
        path = _safe_child(repo, row["path"], description="localization source snapshot")
        if sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"localization source changed after fitting: {row['path']}")


def _validate_fit_capsule(*, capsule_root: Path, repo: Path) -> None:
    code_root = capsule_root / "code"
    closure_path = code_root / "FIT_CODE_CLOSURE.json"
    closure = _hashed_json(closure_path)
    expected_sources = [
        {"path": relative, "sha256": sha256_file(repo / relative)}
        for relative in EXPECTED_FIT_CAPSULE_SOURCES
    ]
    if (
        closure.get("schema_version")
        != "reconstruction-localization-fit-code-closure-v1"
        or closure.get("source_files") != expected_sources
        or closure.get("generated_initializers") != [
            "spectral_utils/__init__.py",
            "spectral_utils/reconstruction_benchmark/__init__.py",
        ]
        or closure.get("excluded_module_classes") != [
            "raw_sources", "preparation", "response_method_fitters", "label_loaders",
            "comparators", "error_taxonomy", "evaluation", "reporting",
        ]
    ):
        raise RuntimeError("localization fit code closure is incomplete or stale")
    expected_files = {
        *EXPECTED_FIT_CAPSULE_SOURCES,
        "spectral_utils/__init__.py",
        "spectral_utils/reconstruction_benchmark/__init__.py",
        "scripts/reconstruction_benchmark/localization_fit_worker.py",
        "FIT_CODE_CLOSURE.json",
    }
    actual_files = {
        path.relative_to(code_root).as_posix()
        for path in code_root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError("localization fit capsule contains missing or unregistered code")
    for relative in EXPECTED_FIT_CAPSULE_SOURCES:
        if sha256_file(code_root / relative) != sha256_file(repo / relative):
            raise RuntimeError(f"localization capsule code differs from source: {relative}")
    worker = "scripts/reconstruction_benchmark/localization_fit_worker.py"
    if sha256_file(code_root / worker) != sha256_file(repo / worker):
        raise RuntimeError("localization capsule worker differs from current source")
    initializers = {
        "spectral_utils/__init__.py":
            b'\"\"\"Minimal target-free localization fit capsule.\"\"\"\n',
        "spectral_utils/reconstruction_benchmark/__init__.py":
            b'\"\"\"Minimal reconstruction localization fit package.\"\"\"\n',
    }
    for relative, payload in initializers.items():
        if (code_root / relative).read_bytes() != payload:
            raise RuntimeError("localization capsule generated initializer drifted")


def _validate_projection_artifact(
    *,
    projection_root: Path,
    source_root: Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if record.get("schema_version") != PROJECTION_SCHEMA_VERSION:
        raise RuntimeError("unexpected localization comparator projection schema")
    if record.get("target_fields_selected") is not False:
        raise RuntimeError("a comparator projection selected target fields")
    path = _safe_child(
        projection_root, record.get("artifact_path"), description="comparator projection"
    )
    if sha256_file(path) != record.get("artifact_sha256"):
        raise RuntimeError(f"comparator projection hash failed: {path}")
    raw_path = _safe_child(
        source_root, record.get("raw_source_path"), description="comparator raw source"
    )
    if sha256_file(raw_path) != record.get("raw_source_sha256"):
        raise RuntimeError(f"comparator raw source changed: {raw_path}")
    arrays = load_npz_no_pickle(path)
    expected = {"row_ids", "native_prediction", "score_offsets", "score", "coverage"}
    if set(arrays) != expected:
        raise RuntimeError("comparator projection contains unknown arrays")
    row_ids = tuple(map(str, arrays["row_ids"].tolist()))
    decisions = np.asarray(arrays["native_prediction"], dtype=np.int64)
    offsets = np.asarray(arrays["score_offsets"], dtype=np.int64)
    scores = np.asarray(arrays["score"], dtype=np.float64)
    coverage = np.asarray(arrays["coverage"], dtype=np.int8)
    if (
        not row_ids
        or row_ids != tuple(sorted(row_ids))
        or len(set(row_ids)) != len(row_ids)
        or any(not value.startswith("xridv2_") or len(value) != 71 for value in row_ids)
        or decisions.shape != (len(row_ids),)
        or coverage.shape != (len(row_ids),)
        or offsets.shape != (len(row_ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(scores)
        or np.any(np.diff(offsets) < 0)
        or not np.isfinite(scores).all()
        or not np.isin(coverage, (0, 1)).all()
    ):
        raise RuntimeError("comparator score-only projection arrays are malformed")
    exact = {
        "n_rows": len(row_ids),
        "n_scores": len(scores),
        "n_covered": int(coverage.sum()),
    }
    for field, expected_value in exact.items():
        if int(record.get(field, -1)) != expected_value:
            raise RuntimeError(f"comparator projection count drifted: {field}")
    return {
        "row_ids": row_ids,
        "artifact_file_sha256": sha256_file(path),
        "raw_source_file_sha256": sha256_file(raw_path),
    }


def _validate_projection_manifest(
    *,
    root: Path,
    source_root: Path,
    build_id: str,
    config: Mapping[str, Any],
    localization_registry_sha256: str,
    external_registry_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    projection_root = root / "comparator_projections"
    manifest = _hashed_json(projection_root / "MANIFEST.json")
    if (
        manifest.get("schema_version") != PROJECTION_MANIFEST_SCHEMA_VERSION
        or manifest.get("build_id") != build_id
        or manifest.get("localization_registry_sha256")
        != localization_registry_sha256
        or manifest.get("external_registry_sha256") != external_registry_sha256
        or manifest.get("target_fields_selected") is not False
        or manifest.get("fit_capsule_mount") is not False
    ):
        raise RuntimeError("localization comparator projection manifest is invalid")
    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for comparator in config["comparators"]:
        cells = (
            config["processbench"]["source_cells"]
            if comparator["dataset_id"] == "processbench"
            else [config["prmbench"]["source_cell"]]
        )
        for cell_id in cells:
            expected[(str(comparator["system_id"]), str(cell_id))] = comparator
    rows = manifest.get("records", ())
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("system_id", "")), str(row.get("cell_id", "")))
        if key in index:
            raise RuntimeError(f"duplicate localization comparator projection: {key}")
        index[key] = dict(row)
    if set(index) != set(expected):
        raise RuntimeError("localization comparator projection roster is incomplete")
    for key, row in index.items():
        comparator = expected[key]
        for field in ("dataset_id", "kind", "access_level", "fidelity"):
            if row.get(field) != comparator.get(field):
                raise RuntimeError(f"localization comparator contract drifted for {key}: {field}")
        cell_id = key[1]
        if comparator["dataset_id"] == "processbench":
            model_id = next(
                model for model in map(str, config["processbench"]["models"])
                if cell_id.endswith("_" + model)
            )
            slice_id = cell_id.removeprefix("processbench_").removesuffix("_" + model_id)
        else:
            model_id = str(config["prmbench"]["model_id"])
            slice_id = str(config["prmbench"]["source_slice_id"])
        expected_raw_path = str(comparator["path_template"]).format(slice_id=slice_id)
        exact = {
            "model_id": model_id,
            "slice_id": slice_id,
            "projection_contract": comparator["projection"],
            "raw_container_target_fields_co_located": True,
            "target_fields_selected": False,
            "raw_source_path": expected_raw_path,
            "artifact_path": f"{key[0]}/{cell_id}.npz",
        }
        for field, expected_value in exact.items():
            if row.get(field) != expected_value:
                raise RuntimeError(f"localization comparator projection drifted for {key}: {field}")
        validation = _validate_projection_artifact(
            projection_root=projection_root, source_root=source_root, record=row
        )
        row["artifact_file_sha256"] = validation["artifact_file_sha256"]
        row["raw_source_file_sha256"] = validation["raw_source_file_sha256"]
    return manifest, index


def _validate_freeze(
    *,
    root: Path,
    controller_root: Path,
    repo: Path,
    source_root: Path,
    build_id: str,
    manifest: Mapping[str, Any],
    expected_cells: set[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fit_root = root / "fit"
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = _hashed_json(freeze_path)
    if (
        freeze.get("schema_version") != SCORE_FREEZE_SCHEMA_VERSION
        or freeze.get("scientific_full") is not True
        or freeze.get("release_id") != manifest.get("release_id")
        or freeze.get("build_id") != build_id
        or freeze.get("external_release_id") != manifest.get("external_release_id")
        or freeze.get("target_data_opened") is not False
        or freeze.get("response_scores_refit") is not False
    ):
        raise RuntimeError("localization score freeze is partial or crossed its fit boundary")
    input_manifest_path = root / "inputs/MANIFEST.json"
    if (
        freeze.get("input_manifest_sha256") != sha256_file(input_manifest_path)
        or freeze.get("input_manifest_payload_sha256") != manifest.get("payload_sha256")
        or freeze.get("input_tree") != canonical_tree_manifest(root / "inputs")
        or freeze.get("capsule_tree") != canonical_tree_manifest(root / "fit_capsule")
        or freeze.get("fit_code_closure_sha256")
        != sha256_file(root / "fit_capsule/code/FIT_CODE_CLOSURE.json")
    ):
        raise RuntimeError("localization score freeze input/capsule binding failed")
    _validate_fit_capsule(capsule_root=root / "fit_capsule", repo=repo)
    _validate_source_snapshot(freeze.get("source_snapshot", {}), repo=repo)
    audit_policy = validate_fit_audit_policy(freeze.get("audit_policy", {}))
    if (
        audit_policy.get("policy_sha256") != freeze.get("audit_policy_sha256")
    ):
        raise RuntimeError("localization fit audit policy is incomplete or unbound")
    expected_probes = [
        {"probe_id": "localization_registry", "path": str(
            (repo / "configs/reconstruction_benchmark_v1/localization.json").resolve()
        )},
        {"probe_id": "external_registry", "path": str(
            (repo / "configs/reconstruction_benchmark_v1/external_final_answer.json").resolve()
        )},
        {"probe_id": "population_registry", "path": str(
            (repo / "configs/reconstruction_benchmark_v1/populations.json").resolve()
        )},
        {"probe_id": "preparation_provenance", "path": str(
            (controller_root / f"build_{build_id}/preparation_provenance/MANIFEST.json").resolve()
        )},
        {"probe_id": "comparator_projections", "path": str(
            (root / "comparator_projections").resolve()
        )},
        {"probe_id": "preparation_module", "path": str(
            (repo / "spectral_utils/reconstruction_benchmark/localization_preparation.py").resolve()
        )},
        {"probe_id": "comparator_module", "path": str(
            (repo / "spectral_utils/reconstruction_benchmark/localization_comparators.py").resolve()
        )},
        {"probe_id": "evaluation_module", "path": str(
            (repo / "spectral_utils/reconstruction_benchmark/localization_evaluation.py").resolve()
        )},
    ]
    if audit_policy.get("forbidden_probes") != expected_probes:
        raise RuntimeError("localization fit audit policy denial probes drifted")
    sensitive = (repo.resolve(), source_root.resolve(), root.parent.resolve())
    for allowed in map(Path, audit_policy.get("allowed_read_roots", ())):
        if any(
            _is_relative_to(target, allowed.resolve())
            for target in sensitive
        ):
            raise RuntimeError("localization fit audit policy mounted a broad project/source root")
    allowed_roots = [Path(value).resolve() for value in audit_policy["allowed_read_roots"]]
    for expected in (root / "fit_capsule/code", root / "inputs"):
        if not any(_is_relative_to(expected.resolve(), allowed) for allowed in allowed_roots):
            raise RuntimeError("localization fit audit policy omitted capsule/prepared inputs")
    worker_path = fit_root / "WORKER_RESULT_MANIFEST.json"
    worker = _hashed_json(worker_path)
    if (
        freeze.get("worker_result_sha256") != sha256_file(worker_path)
        or worker.get("release_id") != manifest.get("release_id")
        or worker.get("build_id") != build_id
        or worker.get("input_manifest_sha256") != freeze.get("input_manifest_sha256")
        or worker.get("input_manifest_payload_sha256") != freeze.get("input_manifest_payload_sha256")
        or worker.get("audit_policy_sha256") != freeze.get("audit_policy_sha256")
        or worker.get("target_data_opened") is not False
        or worker.get("response_scores_refit") is not False
        or worker.get("firewall_violations") != []
        or worker.get("denial_probes") != [
            {"probe_id": probe_id, "read_denied": True}
            for probe_id in EXPECTED_DENIAL_PROBES
        ]
        or freeze.get("pre_freeze_fit_tree")
        != _tree_without(fit_root, {"SCORE_FREEZE_MANIFEST.json"})
    ):
        raise RuntimeError("localization worker/audit/pre-freeze binding failed")
    provenance_path = controller_root / f"build_{build_id}/preparation_provenance/MANIFEST.json"
    provenance = _hashed_json(provenance_path)
    if (
        freeze.get("preparation_provenance_sha256") != sha256_file(provenance_path)
        or provenance.get("release_id") != manifest.get("release_id")
        or provenance.get("build_id") != build_id
        or provenance.get("target_values_selected") is not False
        or provenance.get("fit_manifest_sha256") != freeze.get("input_manifest_sha256")
    ):
        raise RuntimeError("localization preparation provenance binding failed")
    if freeze.get("comparator_projection_manifest_sha256") != sha256_file(
        root / "comparator_projections/MANIFEST.json"
    ) or freeze.get("comparator_projection_tree") != canonical_tree_manifest(
        root / "comparator_projections"
    ):
        raise RuntimeError("localization score freeze comparator binding failed")

    summaries = freeze.get("records", ())
    worker_summaries = worker.get("records", ())
    if summaries != worker_summaries:
        raise RuntimeError("localization freeze and worker score summaries differ")
    index: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        cell_id = str(summary.get("cell_id", ""))
        if not cell_id or cell_id in index:
            raise RuntimeError("localization score freeze has duplicate/empty cell summaries")
        record_path = _safe_child(
            fit_root, summary.get("record_path"), description="localization score record"
        )
        score_path = _safe_child(
            fit_root, summary.get("score_path"), description="localization score artifact"
        )
        if (
            sha256_file(record_path) != summary.get("record_file_sha256")
            or sha256_file(score_path) != summary.get("score_sha256")
        ):
            raise RuntimeError(f"{cell_id}: localization score summary hash failed")
        record, arrays = load_localization_score_bundle(record_path)
        exact = {
            "cell_id": record["cell_id"],
            "population_id": record["population_id"],
            "dataset_id": record["dataset_id"],
            "model_id": record["model_id"],
            "slice_id": record["slice_id"],
            "record_sha256": record["record_sha256"],
            "score_sha256": record["score_sha256"],
            "n_rows": record["n_rows"],
            "n_segments": record["n_segments"],
            "n_systems": record["n_systems"],
            "external_certificate_sha256": record["external_certificate_sha256"],
            "external_score_bindings_sha256": record["external_score_bindings_sha256"],
            "token_transform_sha256": record["token_transform_sha256"],
            "token_fit_sha256": record["token_fit_diagnostics"]["fit_sha256"],
        }
        for field, expected_value in exact.items():
            if summary.get(field) != expected_value:
                raise RuntimeError(f"{cell_id}: score record/summary differs at {field}")
        index[cell_id] = {
            "summary": dict(summary),
            "record": record,
            "arrays": arrays,
            "record_file_sha256": sha256_file(record_path),
            "score_file_sha256": sha256_file(score_path),
        }
    if (
        set(index) != expected_cells
        or int(freeze.get("n_cells", -1)) != len(expected_cells)
        or int(worker.get("n_records", -1)) != len(expected_cells)
        or int(freeze.get("n_systems_per_cell", -1)) != 27
    ):
        raise RuntimeError("localization score freeze is not the exact complete cell/system roster")
    return freeze, index


def _load_external_certificate_current(
    *,
    repo: Path,
    release_root: Path,
    external_release_id: str,
    external_registry_path: Path,
    population_registry_path: Path,
) -> tuple[dict[str, Any], Any, Path]:
    registry = load_external_registry(
        repo=repo,
        registry_path=external_registry_path,
        population_registry_path=population_registry_path,
    )
    path = release_root / external_release_id / "external_final_answer/AB_VERIFICATION.json"
    certificates = [
        assert_external_ab_certificate(
            path,
            release_id=external_release_id,
            release_root=release_root,
            selected_build=build_id,
            registry=registry,
            repo=repo,
        )
        for build_id in ("A", "B")
    ]
    if certificates[0] != certificates[1]:
        raise RuntimeError("external A/B certificate revalidation differs by selected build")
    return certificates[0], registry, path


def _external_response_bindings(
    certificate: Mapping[str, Any], *, expected_cells: set[str],
) -> dict[str, tuple[list[dict[str, Any]], str]]:
    records = certificate.get("comparison_records", ())
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in records:
        key = (str(row.get("cell_id", "")), str(row.get("method_id", "")))
        if key in index:
            raise RuntimeError("external certificate response comparison is duplicated")
        index[key] = row
    expected_pairs = {
        (cell_id, method_id)
        for cell_id in expected_cells for method_id in PRIMARY_METHOD_IDS
    }
    if not expected_pairs.issubset(index):
        raise RuntimeError("external certificate lacks a required localization response score")
    output = {}
    for cell_id in sorted(expected_cells):
        bindings = []
        for method_id in PRIMARY_METHOD_IDS:
            row = index[(cell_id, method_id)]
            if row.get("status") not in {"OK", "OK_FALLBACK"}:
                raise RuntimeError(f"{cell_id}/{method_id}: upstream response score is not successful")
            bindings.append({
                "cell_id": cell_id,
                "method_id": method_id,
                "method_version_id": row["method_version_id"],
                "config_sha256": row["config_sha256"],
                "record_sha256": row["record_sha256"],
                "score_sha256": row["score_sha256"],
                "row_roster_sha256": row["row_roster_sha256"],
            })
        output[cell_id] = (bindings, sha256_bytes(canonical_json_bytes(bindings)))
    return output


def verify_localization_ab(
    *,
    release_id: str,
    release_root: str | Path,
    output_path: str | Path | None = None,
    localization_registry_path: str | Path = DEFAULT_LOCALIZATION_REGISTRY,
    external_registry_path: str | Path = DEFAULT_EXTERNAL_REGISTRY,
    population_registry_path: str | Path = DEFAULT_POPULATION_REGISTRY,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    repo: str | Path = REPO_ROOT,
    _write_certificate: bool = True,
) -> dict[str, Any]:
    repo_path = Path(repo).resolve()
    current_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_path, check=True, capture_output=True, text=True,
    ).stdout
    if current_status.strip():
        raise RuntimeError("scientific localization A/B verification requires clean git")
    release_root_path = Path(release_root).resolve()
    source_root_path = Path(source_root).resolve()
    release = release_root_path / release_id
    config = load_localization_registry(localization_registry_path)
    expected_cells = {
        *map(str, config["processbench"]["source_cells"]),
        str(config["prmbench"]["source_cell"]),
    }
    expected_systems = primary_system_roster(PRIMARY_METHOD_IDS)
    expected_system_ids = [row["system_id"] for row in expected_systems]
    expected_method_ids = [row["method_id"] for row in expected_systems]
    expected_adapter_ids = [row["adapter_id"] for row in expected_systems]

    builds: dict[str, dict[str, Any]] = {}
    controller_root = (
        release_root_path.parent / "private_control" / release_id / "localization"
    )
    for build_id in ("A", "B"):
        root = release / f"build_{build_id}" / "localization"
        inputs = root / "inputs"
        manifest = validate_fit_manifest(
            inputs / "MANIFEST.json", input_root=inputs, require_scientific=True
        )
        if manifest.get("release_id") != release_id or manifest.get("build_id") != build_id:
            raise RuntimeError(f"localization build {build_id} input binding failed")
        cells = {str(row["cell_id"]): row for row in manifest["cells"]}
        if len(cells) != len(manifest["cells"]) or set(cells) != expected_cells:
            raise RuntimeError(f"localization build {build_id} input roster is incomplete")
        freeze, records = _validate_freeze(
            root=root,
            controller_root=controller_root,
            repo=repo_path,
            source_root=source_root_path,
            build_id=build_id,
            manifest=manifest,
            expected_cells=expected_cells,
        )
        projection, projections = _validate_projection_manifest(
            root=root,
            source_root=source_root_path,
            build_id=build_id,
            config=config,
            localization_registry_sha256=sha256_file(localization_registry_path),
            external_registry_sha256=sha256_file(external_registry_path),
        )
        builds[build_id] = {
            "root": root,
            "manifest": manifest,
            "cells": cells,
            "freeze": freeze,
            "records": records,
            "projection": projection,
            "projections": projections,
            "input_manifest_sha256": sha256_file(inputs / "MANIFEST.json"),
            "score_freeze_sha256": sha256_file(root / "fit/SCORE_FREEZE_MANIFEST.json"),
            "input_tree": canonical_tree_manifest(inputs),
            "fit_tree": canonical_tree_manifest(root / "fit"),
            "capsule_tree": canonical_tree_manifest(root / "fit_capsule"),
            "projection_tree": canonical_tree_manifest(root / "comparator_projections"),
            "preparation_tree": canonical_tree_manifest(
                controller_root / f"build_{build_id}/preparation_provenance"
            ),
        }

    left, right = builds["A"], builds["B"]
    manifest_exact = (
        "external_release_id", "external_certificate_sha256", "external_registry_sha256",
        "method_registry_sha256", "identity_contract", "id_contract_version",
        "token_contract_id", "token_mixed_v2_applied_exactly_once", "n_cells",
    )
    for field in manifest_exact:
        if left["manifest"].get(field) != right["manifest"].get(field):
            raise RuntimeError(f"localization A/B input manifest differs: {field}")

    external_certificate, external_registry, external_certificate_path = (
        _load_external_certificate_current(
            repo=repo_path,
            release_root=release_root_path,
            external_release_id=str(left["manifest"]["external_release_id"]),
            external_registry_path=Path(external_registry_path),
            population_registry_path=Path(population_registry_path),
        )
    )
    if (
        external_certificate.get("certificate_sha256")
        != left["manifest"].get("external_certificate_sha256")
        or external_registry.sha256 != left["manifest"].get("external_registry_sha256")
    ):
        raise RuntimeError("localization inputs do not bind the current external A/B certificate")
    response_bindings = _external_response_bindings(
        external_certificate, expected_cells=expected_cells
    )
    current_external_sha = str(external_certificate["certificate_sha256"])
    for build_id, build in builds.items():
        if (
            build["manifest"].get("external_certificate_sha256") != current_external_sha
            or build["freeze"].get("external_certificate_sha256") != current_external_sha
        ):
            raise RuntimeError(f"localization build {build_id} binds a stale external certificate")
        provenance_path = (
            controller_root / f"build_{build_id}/preparation_provenance/MANIFEST.json"
        )
        provenance = _hashed_json(provenance_path)
        provenance_cells = {
            str(row.get("cell_id", "")): row for row in provenance.get("cells", ())
        }
        if (
            len(provenance_cells) != len(provenance.get("cells", ()))
            or set(provenance_cells) != expected_cells
            or provenance.get("external_certificate_sha256") != current_external_sha
            or provenance.get("external_score_freeze_sha256")
            != external_certificate["builds"][build_id]["score_freeze_sha256"]
        ):
            raise RuntimeError(f"localization preparation {build_id} is not bound upstream")
        for cell_id in expected_cells:
            expected_rows, expected_digest = response_bindings[cell_id]
            input_row = build["cells"][cell_id]
            score_row = build["records"][cell_id]["record"]
            provenance_row = provenance_cells[cell_id]
            for candidate_name, candidate in (
                ("input", input_row), ("score", score_row),
                ("provenance", provenance_row),
            ):
                if (
                    candidate.get("external_certificate_sha256") != current_external_sha
                    or candidate.get("external_score_bindings_sha256") != expected_digest
                ):
                    raise RuntimeError(
                        f"{cell_id}: {candidate_name} response binding is stale/foreign"
                    )
            if provenance_row.get("response_score_bindings") != expected_rows:
                raise RuntimeError(
                    f"{cell_id}: preparation response records differ from upstream certificate"
                )

    input_comparisons = []
    for cell_id in sorted(expected_cells):
        lcell, rcell = left["cells"][cell_id], right["cells"][cell_id]
        if lcell.get("method_ids") != list(PRIMARY_METHOD_IDS):
            raise RuntimeError(f"{cell_id}: localization input response roster is not canonical")
        fields = (
            "artifact_sha256", "n_rows", "n_tokens", "n_segments", "method_ids",
            "row_roster_sha256", "external_certificate_sha256",
            "external_score_bindings_sha256", "token_transform_sha256",
        )
        unequal = [field for field in fields if lcell.get(field) != rcell.get(field)]
        if unequal:
            raise RuntimeError(f"localization A/B prepared input differs for {cell_id}: {unequal}")
        input_comparisons.append({"cell_id": cell_id, **{field: lcell[field] for field in fields}})

    score_comparisons = []
    for cell_id in sorted(expected_cells):
        lrow, rrow = left["records"][cell_id], right["records"][cell_id]
        for candidate in (lrow, rrow):
            record = candidate["record"]
            if (
                record.get("system_ids") != expected_system_ids
                or record.get("method_ids") != expected_method_ids
                or record.get("adapter_ids") != expected_adapter_ids
                or int(record.get("n_systems", -1)) != 27
            ):
                raise RuntimeError(f"{cell_id}: score roster is not the canonical 27 systems")
        fields = (
            "n_rows", "n_segments", "n_systems", "system_ids", "method_ids",
            "adapter_ids", "score_sha256", "record_sha256",
            "external_certificate_sha256", "external_score_bindings_sha256",
            "token_transform_sha256", "token_fit_diagnostics",
        )
        unequal = [
            field for field in fields
            if lrow["record"].get(field) != rrow["record"].get(field)
        ]
        if (
            unequal
            or lrow["record_file_sha256"] != rrow["record_file_sha256"]
            or lrow["score_file_sha256"] != rrow["score_file_sha256"]
        ):
            raise RuntimeError(f"localization A/B scores differ for {cell_id}: {unequal}")
        score_comparisons.append({
            "cell_id": cell_id,
            "record_file_sha256": lrow["record_file_sha256"],
            "score_file_sha256": lrow["score_file_sha256"],
            "record_sha256": lrow["record"]["record_sha256"],
            "score_sha256": lrow["record"]["score_sha256"],
            "token_fit_sha256": lrow["record"]["token_fit_diagnostics"]["fit_sha256"],
        })

    if set(left["projections"]) != set(right["projections"]):
        raise RuntimeError("localization A/B comparator projection rosters differ")
    projection_comparisons = []
    for key in sorted(left["projections"]):
        lrow, rrow = left["projections"][key], right["projections"][key]
        fields = (
            "artifact_sha256", "artifact_file_sha256", "raw_source_sha256",
            "raw_source_file_sha256", "n_rows", "n_scores", "n_covered",
            "access_level", "fidelity", "target_fields_selected",
        )
        unequal = [field for field in fields if lrow.get(field) != rrow.get(field)]
        if unequal:
            raise RuntimeError(f"localization A/B comparator projection differs for {key}: {unequal}")
        projection_comparisons.append({
            "system_id": key[0], "cell_id": key[1],
            **{field: lrow[field] for field in fields},
        })

    certificate = {
        "schema_version": AB_CERTIFICATE_SCHEMA_VERSION,
        "release_id": release_id,
        "localization_registry_sha256": sha256_file(localization_registry_path),
        "external_registry_sha256": sha256_file(external_registry_path),
        "population_registry_sha256": sha256_file(population_registry_path),
        "external_release_id": left["manifest"]["external_release_id"],
        "identity_contract": left["manifest"]["identity_contract"],
        "external_certificate_sha256": external_certificate["certificate_sha256"],
        "external_certificate_file_sha256": sha256_file(external_certificate_path),
        "external_response_bindings_sha256": payload_sha256({
            cell_id: digest for cell_id, (_rows, digest) in sorted(response_bindings.items())
        }),
        "status": "PASS",
        "scientific_full": True,
        "n_cells": len(expected_cells),
        "n_core_systems": 27,
        "target_data_opened": False,
        "response_scores_refit": False,
        "input_comparisons": input_comparisons,
        "input_comparisons_sha256": payload_sha256(input_comparisons),
        "score_comparisons": score_comparisons,
        "score_comparisons_sha256": payload_sha256(score_comparisons),
        "projection_comparisons": projection_comparisons,
        "projection_comparisons_sha256": payload_sha256(projection_comparisons),
        "builds": {
            build_id: {
                "input_manifest_sha256": builds[build_id]["input_manifest_sha256"],
                "score_freeze_sha256": builds[build_id]["score_freeze_sha256"],
                "input_tree_sha256": builds[build_id]["input_tree"]["tree_sha256"],
                "fit_tree_sha256": builds[build_id]["fit_tree"]["tree_sha256"],
                "capsule_tree_sha256": builds[build_id]["capsule_tree"]["tree_sha256"],
                "projection_tree_sha256": builds[build_id]["projection_tree"]["tree_sha256"],
                "preparation_tree_sha256": builds[build_id]["preparation_tree"]["tree_sha256"],
            }
            for build_id in ("A", "B")
        },
    }
    certificate["certificate_sha256"] = payload_sha256(certificate)
    if _write_certificate:
        target = (
            Path(output_path) if output_path is not None
            else release / "localization/AB_VERIFICATION.json"
        )
        atomic_write_json(target, certificate)
    return certificate


def assert_localization_ab_certificate(
    path: str | Path,
    *,
    release_id: str,
    release_root: str | Path,
    localization_registry_path: str | Path = DEFAULT_LOCALIZATION_REGISTRY,
    external_registry_path: str | Path = DEFAULT_EXTERNAL_REGISTRY,
    population_registry_path: str | Path = DEFAULT_POPULATION_REGISTRY,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    repo: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    certificate = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(certificate)
    recorded = payload.pop("certificate_sha256", None)
    if recorded != payload_sha256(payload):
        raise RuntimeError("localization A/B certificate hash failed")
    if (
        certificate.get("schema_version") != AB_CERTIFICATE_SCHEMA_VERSION
        or certificate.get("status") != "PASS"
        or certificate.get("scientific_full") is not True
        or certificate.get("release_id") != release_id
        or int(certificate.get("n_cells", -1)) != 13
        or int(certificate.get("n_core_systems", -1)) != 27
    ):
        raise RuntimeError("a passing complete scientific localization A/B certificate is required")
    current_registry_hashes = {
        "localization_registry_sha256": sha256_file(localization_registry_path),
        "external_registry_sha256": sha256_file(external_registry_path),
        "population_registry_sha256": sha256_file(population_registry_path),
    }
    for field, digest in current_registry_hashes.items():
        if certificate.get(field) != digest:
            raise RuntimeError(f"localization A/B certificate registry is stale: {field}")

    # Full verification is intentionally recomputed.  Comparing stored tree
    # hashes alone would let a structurally incomplete but self-consistent
    # certificate survive.
    recomputed = verify_localization_ab(
        release_id=release_id,
        release_root=release_root,
        localization_registry_path=localization_registry_path,
        external_registry_path=external_registry_path,
        population_registry_path=population_registry_path,
        source_root=source_root,
        repo=repo,
        _write_certificate=False,
    )
    if recomputed != certificate:
        raise RuntimeError("localization A/B certificate no longer matches full recomputation")
    return certificate


__all__ = [
    "AB_CERTIFICATE_SCHEMA_VERSION", "DEFAULT_EXTERNAL_REGISTRY",
    "DEFAULT_LOCALIZATION_REGISTRY", "DEFAULT_POPULATION_REGISTRY",
    "DEFAULT_SOURCE_ROOT", "assert_localization_ab_certificate",
    "verify_localization_ab",
]
