"""Controller for a code-capsuled, deny-default RAG evidence score stage."""

from __future__ import annotations

import base64
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .fit_firewall import build_fit_audit_policy
from .io import atomic_write_json, canonical_json_bytes, canonical_tree_manifest, sha256_bytes
from .rag_evidence_contract import (
    AtomicRagDirectory,
    BoundRagFile,
    FIT_INPUT_FILENAME,
    PREPARATION_MANIFEST_FILENAME,
    SCORE_FREEZE_SCHEMA,
    SCORE_MANIFEST_FILENAME,
    SCORES_FILENAME,
    RagEvidenceContractError,
    add_payload_sha256,
    load_fit_input,
    load_registry,
    read_bound_file_bytes,
    validate_artifact_identifier,
    verify_payload,
)
from .rag_evidence_fit import load_scores_bytes


FIT_CAPSULE_CODE_ALLOWLIST = (
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/fixed_application_pipelines.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/repeated_measurement_reliability.py",
    "spectral_utils/token_feature_views.py",
    "spectral_utils/upcr.py",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_contract.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_fit.py",
)
FIT_WORKER = "scripts/reconstruction_benchmark/rag_evidence_fit_worker.py"
FIT_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/rag_evidence.json",
    *FIT_CAPSULE_CODE_ALLOWLIST,
    "spectral_utils/reconstruction_benchmark/rag_evidence_ab.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_runner.py",
    FIT_WORKER,
)


class _BoundRagFitSourceClosure:
    """Hold the exact fit-source bytes from snapshot through worker completion."""

    def __init__(self, repo: Path, *, allow_dirty_debug: bool) -> None:
        self.repo = Path(repo).resolve(strict=True)
        self.allow_dirty_debug = bool(allow_dirty_debug)
        self.files: dict[str, BoundRagFile] = {}
        self.payloads: dict[str, bytes] = {}
        self.git_head = ""
        self.git_status = b""
        self.snapshot: dict[str, Any] = {}
        try:
            self.git_head, self.git_status = self._git_state()
            if self.git_status.strip() and not self.allow_dirty_debug:
                raise RagEvidenceContractError(
                    "scientific RAG fitting requires its registered source "
                    "closure to be clean"
                )
            rows: list[dict[str, str]] = []
            for relative in FIT_SOURCE_FILES:
                held = BoundRagFile(
                    self.repo / relative,
                    name=f"RAG fit source {relative}",
                )
                self.files[relative] = held
                payload = held.read_bytes()
                self.payloads[relative] = payload
                rows.append({"path": relative, "sha256": sha256_bytes(payload)})
            value = {
                "git_head": self.git_head,
                "registered_source_closure_clean": not self.git_status.strip(),
                "registered_status_sha256": sha256_bytes(self.git_status),
                "files": rows,
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": _package_version("numpy"),
                    "scipy": _package_version("scipy"),
                    "scikit_learn": _package_version("scikit-learn"),
                },
            }
            value["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(value))
            self.snapshot = value
            self.verify_stable()
        except Exception:
            self.close(verify=False)
            raise

    def _git_state(self) -> tuple[str, bytes]:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *FIT_SOURCE_FILES],
            cwd=self.repo, check=True, capture_output=True, text=True,
        ).stdout.encode("utf-8")
        return head, status

    def payload(self, relative: str) -> bytes:
        try:
            return self.payloads[relative]
        except KeyError as error:
            raise RagEvidenceContractError(
                f"RAG capsule source is outside the held closure: {relative}"
            ) from error

    def verify_stable(self) -> None:
        if not self.files:
            raise RuntimeError("RAG fit source closure is closed")
        for held in self.files.values():
            held.verify_stable()
        head, status = self._git_state()
        if head != self.git_head or status != self.git_status:
            raise RagEvidenceContractError(
                "RAG registered source closure git state changed during fit"
            )
        for relative, held in self.files.items():
            if held.read_bytes() != self.payloads[relative]:
                raise RagEvidenceContractError(
                    f"RAG held fit source bytes changed: {relative}"
                )

    def close(self, *, verify: bool = True) -> None:
        if not self.files:
            return
        failure: BaseException | None = None
        if verify:
            try:
                self.verify_stable()
            except BaseException as error:
                failure = error
        for held in reversed(tuple(self.files.values())):
            try:
                held.close(verify=False)
            except BaseException as error:
                if failure is None:
                    failure = error
        self.files.clear()
        self.payloads.clear()
        if failure is not None:
            raise failure

    def __enter__(self) -> "_BoundRagFitSourceClosure":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(verify=True)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _source_snapshot(repo: Path, *, allow_dirty_debug: bool) -> dict[str, Any]:
    with _BoundRagFitSourceClosure(
        repo, allow_dirty_debug=allow_dirty_debug
    ) as sources:
        return dict(sources.snapshot)


def _copy_capsule(
    repo: Path,
    target: Path,
    *,
    source_closure: _BoundRagFitSourceClosure | None = None,
) -> Path:
    owned_closure = source_closure is None
    sources = source_closure or _BoundRagFitSourceClosure(
        repo, allow_dirty_debug=True
    )
    if sources.repo != Path(repo).resolve(strict=True):
        if owned_closure:
            sources.close(verify=False)
        raise RagEvidenceContractError("RAG capsule/source repository binding differs")
    code = target / "code"
    try:
        code.mkdir(parents=True, exist_ok=False)
        expected_payloads: dict[str, bytes] = {}
        source_rows: list[dict[str, str]] = []
        for relative in FIT_CAPSULE_CODE_ALLOWLIST:
            payload = sources.payload(relative)
            destination = code / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            expected_payloads[relative] = payload
            source_rows.append({"path": relative, "sha256": sha256_bytes(payload)})
        initializers = {
            "spectral_utils/__init__.py":
                b'"""Minimal target-free RAG fit capsule."""\n',
            "spectral_utils/reconstruction_benchmark/__init__.py":
                b'"""Minimal target-free reconstruction package."""\n',
        }
        for relative, payload in initializers.items():
            destination = code / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            expected_payloads[relative] = payload
        worker = code / FIT_WORKER
        worker.parent.mkdir(parents=True, exist_ok=True)
        worker_payload = sources.payload(FIT_WORKER)
        worker.write_bytes(worker_payload)
        expected_payloads[FIT_WORKER] = worker_payload
        closure = add_payload_sha256({
            "schema_version": "reconstruction-rag-evidence-fit-code-closure-v1",
            "fit_source_snapshot_sha256": sources.snapshot["snapshot_sha256"],
            "source_files": source_rows,
            "worker": {"path": FIT_WORKER, "sha256": sha256_bytes(worker_payload)},
            "generated_inert_initializers": sorted(initializers),
            "excluded": [
                "raw assets", "private labels", "preparation adapters",
                "post-freeze evaluation", "historical result trees",
            ],
        })
        closure_payload = canonical_json_bytes(closure) + b"\n"
        atomic_write_json(code / "FIT_CODE_CLOSURE.json", closure)
        expected_payloads["FIT_CODE_CLOSURE.json"] = closure_payload
        expected_rows = [
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for relative, payload in sorted(expected_payloads.items())
        ]
        observed = canonical_tree_manifest(code)
        if observed["files"] != expected_rows:
            raise RagEvidenceContractError(
                "RAG capsule bytes/roster differ from authenticated source snapshot"
            )
        sources.verify_stable()
        return code
    finally:
        if owned_closure:
            sources.close(verify=True)


def _policy(
    *, code_root: Path, input_root: Path, output_root: Path,
    temp_root: Path, forbidden: list[tuple[str, Path]],
) -> dict[str, Any]:
    runtime_roots = {
        Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
        Path("/usr").resolve(), Path("/System").resolve(), Path("/Library").resolve(),
        Path("/dev").resolve(), Path("/private/var/db/timezone/tz").resolve(),
    }
    return build_fit_audit_policy(
        allowed_read_roots=[code_root.resolve(), input_root.resolve(), *runtime_roots],
        allowed_read_files=[Path("/proc/self/maps")],
        allowed_write_roots=[output_root.resolve(), temp_root.resolve()],
        allowed_native_roots=[
            Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
            Path("/usr").resolve(), Path("/System").resolve(), Path("/Library").resolve(),
        ],
        forbidden_probes=[
            {"probe_id": probe_id, "path": str(path.resolve())}
            for probe_id, path in forbidden
        ],
    )


def _launch_worker(
    *, code_root: Path, input_path: Path, input_sha256: str, output_root: Path,
    temp_root: Path, release_id: str, build_id: str, lane_id: str,
    forbidden_fields: list[str], policy: dict[str, Any],
) -> None:
    with BoundRagFile(
        input_path,
        expected_sha256=input_sha256,
        name="RAG worker fit input",
    ) as held_input:
        command = [
            sys.executable, "-I", "-B", str(code_root / FIT_WORKER),
            "--release-id", release_id, "--build", build_id,
            "--input", str(input_path.absolute()),
            "--input-fd", str(held_input.descriptor),
            "--expected-input-sha256", input_sha256,
            "--output-root", str(output_root.resolve()),
            "--lane-id", lane_id,
        ]
        for field in forbidden_fields:
            command.extend(("--forbidden-field", field))
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
            "RECONSTRUCTION_RAG_FIT_POLICY_B64": base64.b64encode(
                canonical_json_bytes(policy)
            ).decode("ascii"),
        }
        completed = subprocess.run(
            command, cwd=code_root, env=environment, check=False,
            capture_output=True, text=True, close_fds=True,
            pass_fds=(held_input.descriptor,), stdin=subprocess.DEVNULL,
        )
        held_input.verify_stable()
    if completed.returncode != 0:
        raise RagEvidenceContractError(
            "restricted RAG fit worker failed closed\n"
            + completed.stdout[-4000:] + completed.stderr[-4000:]
        )


def run_rag_evidence_methods(
    *,
    repo: str | Path,
    registry_path: str | Path,
    source_root: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    build_id: str,
    allow_dirty_debug: bool,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    if build_id not in {"A", "B"}:
        raise RagEvidenceContractError("RAG fit build must be A or B")
    repo_path = Path(repo).resolve(strict=True)
    registry = load_registry(registry_path)
    lane_root = Path(release_root) / release_id / "rag_evidence"
    build_root = lane_root / build_id
    preparation_path = build_root / PREPARATION_MANIFEST_FILENAME
    preparation_payload = read_bound_file_bytes(
        preparation_path, name="RAG preparation manifest"
    )
    preparation = json.loads(preparation_payload.decode("utf-8"))
    verify_payload(preparation, name="RAG preparation manifest")
    if preparation.get("release_id") != release_id or preparation.get("build_id") != build_id:
        raise RagEvidenceContractError("RAG fit/preparation release binding failed")

    # A score stage may proceed only after the independently rederived A/B prep
    # certificate exists and authenticates against all raw sources.
    from .rag_evidence_ab import authenticate_rag_evidence_preparation_certificate

    prep_certificate = authenticate_rag_evidence_preparation_certificate(
        repo=repo_path,
        registry_path=registry_path,
        source_root=source_root,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        require_scientific_full=not allow_dirty_debug,
    )
    fit_input_path = build_root / preparation["fit_input"]["path"]
    if prep_certificate["fit_input_sha256"] != preparation["fit_input"]["sha256"]:
        raise RagEvidenceContractError("RAG fit is underbound to preparation A/B")
    controller_fit_input = load_fit_input(
        fit_input_path,
        registry,
        expected_sha256=prep_certificate["fit_input_sha256"],
    )
    prep_certificate_payload = read_bound_file_bytes(
        lane_root / "PREPARATION_AB_VERIFICATION.json",
        name="RAG authenticated preparation certificate",
    )
    if json.loads(prep_certificate_payload.decode("utf-8")) != prep_certificate:
        raise RagEvidenceContractError(
            "RAG preparation certificate changed after authentication"
        )

    fit_sources = _BoundRagFitSourceClosure(
        repo_path, allow_dirty_debug=allow_dirty_debug
    )
    snapshot = dict(fit_sources.snapshot)
    fit_final = build_root / "fit"
    try:
        stage = AtomicRagDirectory(fit_final)
    except Exception:
        fit_sources.close(verify=False)
        raise
    try:
        stage.assert_path_binding()
        capsule_root = stage.path / "capsule"
        code_root = _copy_capsule(
            repo_path, capsule_root, source_closure=fit_sources
        )
        stage.assert_path_binding()
        candidate = stage.path / "candidate"
        temp_root = stage.path / "worker_tmp"
        temp_root.mkdir(parents=True, exist_ok=False)
        stage.assert_path_binding()
        private_label_path = Path(preparation["private_labels"]["path"])
        forbidden = [
            ("private_labels", private_label_path),
            ("full_registry", Path(registry_path)),
            ("preparation_adapter", repo_path / "spectral_utils/reconstruction_benchmark/rag_evidence_preparation.py"),
            ("postfreeze_evaluation", repo_path / "spectral_utils/reconstruction_benchmark/rag_evidence_evaluation.py"),
        ]
        for asset_id, item in registry["sources"].items():
            forbidden.append((f"raw_source::{asset_id}", Path(source_root) / item["path"]))
        policy = _policy(
            code_root=code_root, input_root=fit_input_path.parent,
            output_root=candidate, temp_root=temp_root, forbidden=forbidden,
        )
        stage.assert_path_binding()
        _launch_worker(
            code_root=code_root, input_path=fit_input_path,
            input_sha256=preparation["fit_input"]["sha256"], output_root=candidate,
            temp_root=temp_root, release_id=release_id, build_id=build_id,
            lane_id=registry["lane_id"],
            forbidden_fields=list(registry["fit_visibility"]["forbidden_fields"]),
            policy=policy,
        )
        stage.assert_path_binding()
        worker_path = candidate / "WORKER_RESULT.json"
        worker_payload = read_bound_file_bytes(
            worker_path, name="RAG fit worker result"
        )
        worker = json.loads(worker_payload.decode("utf-8"))
        verify_payload(worker, name="RAG fit worker result")
        expected_probes = [
            {"probe_id": row["probe_id"], "read_denied": True}
            for row in policy["forbidden_probes"]
        ]
        score_path = candidate / SCORES_FILENAME
        score_payload = read_bound_file_bytes(
            score_path,
            expected_sha256=worker.get("score_sha256"),
            name="RAG worker score archive",
        )
        load_scores_bytes(score_payload, fit_input=controller_fit_input)
        if (
            worker.get("release_id") != release_id
            or worker.get("build_id") != build_id
            or worker.get("input_sha256") != preparation["fit_input"]["sha256"]
            or worker.get("score_sha256") != sha256_bytes(score_payload)
            or worker.get("audit_policy_sha256") != policy["policy_sha256"]
            or worker.get("denial_probes") != expected_probes
            or worker.get("firewall_violations") != []
            or worker.get("labels_opened_by_fit") is not False
            or worker.get("historical_scores_opened") is not False
        ):
            raise RagEvidenceContractError("RAG worker result contract failed")
        fit_sources.verify_stable()
        end_snapshot = _source_snapshot(
            repo_path, allow_dirty_debug=allow_dirty_debug
        )
        if end_snapshot != snapshot:
            raise RagEvidenceContractError("RAG registered source closure changed during fit")
        capsule_tree = canonical_tree_manifest(capsule_root)
        stage.assert_path_binding()
        manifest = add_payload_sha256({
            "schema_version": SCORE_FREEZE_SCHEMA,
            "release_id": release_id,
            "build_id": build_id,
            "lane_id": registry["lane_id"],
            "scientific_full": bool(not allow_dirty_debug and preparation["scientific_full"]),
            "preparation_manifest_sha256": sha256_bytes(preparation_payload),
            "preparation_manifest_payload_sha256": preparation["payload_sha256"],
            "preparation_ab_sha256": sha256_bytes(prep_certificate_payload),
            "fit_input_sha256": preparation["fit_input"]["sha256"],
            "source_binding_sha256": preparation["source_binding_sha256"],
            "source_snapshot": snapshot,
            "capsule_tree_sha256": capsule_tree["tree_sha256"],
            "audit_policy_sha256": policy["policy_sha256"],
            "denial_probes": worker["denial_probes"],
            "firewall_violations": [],
            "worker_result": {
                "path": "candidate/WORKER_RESULT.json",
                "sha256": sha256_bytes(worker_payload),
                "payload_sha256": worker["payload_sha256"],
            },
            "scores": {
                "path": f"candidate/{SCORES_FILENAME}",
                "sha256": sha256_bytes(score_payload),
                "size_bytes": len(score_payload),
            },
            "labels_opened_by_fit": False,
            "historical_scores_opened": False,
            "all_registered_panels_scored": True,
        })
        stage.write_json(SCORE_MANIFEST_FILENAME, manifest)
        fit_sources.verify_stable()
        fit_sources.close(verify=True)
        stage.commit()
        return manifest
    finally:
        stage.cleanup()
        fit_sources.close(verify=False)


__all__ = [
    "FIT_CAPSULE_CODE_ALLOWLIST", "FIT_SOURCE_FILES", "run_rag_evidence_methods",
]
