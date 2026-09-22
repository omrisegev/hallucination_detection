"""Independent A/B certificates for RAG preparation, scores and evaluation."""

from __future__ import annotations

from contextlib import ExitStack
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .io import canonical_json_bytes, canonical_tree_manifest, sha256_bytes
from .rag_evidence_contract import (
    EVALUATION_AB_SCHEMA,
    EVALUATION_MANIFEST_FILENAME,
    EVALUATION_SCHEMA,
    FIT_INPUT_FILENAME,
    PANEL_IDS,
    PREPARATION_AB_SCHEMA,
    PREPARATION_MANIFEST_FILENAME,
    PREPARATION_SCHEMA,
    PRIVATE_LABEL_FILENAME,
    REFCHECKER_SETTINGS,
    SCORE_AB_SCHEMA,
    SCORE_FREEZE_SCHEMA,
    SCORE_MANIFEST_FILENAME,
    SCORES_FILENAME,
    BoundRagTree,
    RagEvidenceContractError,
    add_payload_sha256,
    assert_physical_tree_independence,
    load_fit_input_bytes,
    load_private_labels_bytes,
    load_registry,
    read_bound_file_bytes,
    validate_artifact_identifier,
    validate_source_binding,
    verify_payload,
    write_json_noreplace,
)
from .rag_evidence_fit import load_scores_bytes
from .rag_evidence_preparation import (
    PAIR_TRANSACTION_FILENAME,
    rag_evidence_pair_transaction,
    reconstruct_rag_evidence_preparation,
)


PREPARATION_CERTIFICATE = "PREPARATION_AB_VERIFICATION.json"
SCORE_CERTIFICATE = "SCORE_AB_VERIFICATION.json"
EVALUATION_CERTIFICATE = "EVALUATION_AB_VERIFICATION.json"


def _load_json_capture(
    path: Path,
    *,
    schema: str,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    payload = read_bound_file_bytes(
        path, expected_sha256=expected_sha256, name=name
    )
    value = json.loads(payload.decode("utf-8"))
    verify_payload(value, name=name)
    if value.get("schema_version") != schema:
        raise RagEvidenceContractError(f"unexpected {name} schema")
    return value, payload


def _load_json(path: Path, *, schema: str, name: str) -> dict[str, Any]:
    value, _ = _load_json_capture(path, schema=schema, name=name)
    return value


def _load_pass(path: Path, *, schema: str, name: str) -> dict[str, Any]:
    value = _load_json(path, schema=schema, name=name)
    if value.get("status") != "PASS":
        raise RagEvidenceContractError(f"{name} does not pass")
    return value


def _derive_preparation_certificate_bound(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    require_scientific_full: bool, verify_private_artifacts: bool,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    reconstruction = reconstruct_rag_evidence_preparation(
        repo=repo, registry_path=registry_path, source_root=source_root,
        include_payloads=False,
    )
    registry = reconstruction["registry"]
    lane_root = Path(release_root) / release_id / "rag_evidence"
    manifests: dict[str, dict[str, Any]] = {}
    manifest_payloads: dict[str, bytes] = {}
    manifest_hashes: dict[str, str] = {}
    for build_id in ("A", "B"):
        build_root = lane_root / build_id
        manifest_path = build_root / PREPARATION_MANIFEST_FILENAME
        manifest, manifest_payload = _load_json_capture(
            manifest_path, schema=PREPARATION_SCHEMA,
            name=f"RAG preparation {build_id}",
        )
        if (
            manifest.get("release_id") != release_id
            or manifest.get("build_id") != build_id
            or manifest.get("lane_id") != registry["lane_id"]
        ):
            raise RagEvidenceContractError(f"RAG preparation {build_id} release binding failed")
        if require_scientific_full and manifest.get("scientific_full") is not True:
            raise RagEvidenceContractError(f"RAG preparation {build_id} is not scientific-full")
        fit_descriptor = manifest.get("fit_input")
        private_descriptor = manifest.get("private_labels")
        pair_descriptor = manifest.get("pair_transaction")
        if (
            not isinstance(fit_descriptor, Mapping)
            or set(fit_descriptor)
            != {"path", "sha256", "size_bytes", "target_fields_present"}
            or not isinstance(private_descriptor, Mapping)
            or set(private_descriptor) != {"path", "sha256", "size_bytes"}
            or not isinstance(pair_descriptor, Mapping)
            or set(pair_descriptor)
            != {"path", "private_path", "sha256", "size_bytes"}
        ):
            raise RagEvidenceContractError(
                f"RAG preparation {build_id} file descriptors drifted"
            )
        fit_path = build_root / fit_descriptor["path"]
        private_path = Path(private_root) / release_id / "rag_evidence" / build_id / PRIVATE_LABEL_FILENAME
        pair_transaction = rag_evidence_pair_transaction(
            reconstruction=reconstruction,
            release_id=release_id,
            build_id=build_id,
        )
        pair_bytes = canonical_json_bytes(pair_transaction) + b"\n"
        public_pair_path = build_root / PAIR_TRANSACTION_FILENAME
        private_pair_path = private_path.parent / PAIR_TRANSACTION_FILENAME
        if str(Path(private_descriptor["path"]).absolute()) != str(private_path.absolute()):
            raise RagEvidenceContractError(f"RAG preparation {build_id} private path drifted")
        if (
            manifest.get("pair_transaction_id")
            != pair_transaction["payload_sha256"]
            or pair_descriptor.get("path") != PAIR_TRANSACTION_FILENAME
            or str(Path(pair_descriptor.get("private_path", "")).absolute())
            != str(private_pair_path.absolute())
            or pair_descriptor.get("sha256") != sha256_bytes(pair_bytes)
            or int(pair_descriptor.get("size_bytes", -1)) != len(pair_bytes)
            or read_bound_file_bytes(
                public_pair_path,
                expected_sha256=sha256_bytes(pair_bytes),
                expected_size=len(pair_bytes),
                name=f"RAG public pair marker {build_id}",
            ) != pair_bytes
            or read_bound_file_bytes(
                private_pair_path,
                expected_sha256=sha256_bytes(pair_bytes),
                expected_size=len(pair_bytes),
                name=f"RAG private pair marker {build_id}",
            ) != pair_bytes
        ):
            raise RagEvidenceContractError(
                f"RAG preparation {build_id} pair transaction binding failed"
            )
        fit_payload = read_bound_file_bytes(
            fit_path,
            expected_sha256=reconstruction["fit_input_sha256"],
            expected_size=reconstruction["fit_input_size_bytes"],
            name=f"RAG fit input {build_id}",
        )
        if (
            manifest["fit_input"].get("path") != f"inputs/{FIT_INPUT_FILENAME}"
            or manifest["fit_input"]["sha256"] != reconstruction["fit_input_sha256"]
            or int(manifest["fit_input"]["size_bytes"]) != reconstruction["fit_input_size_bytes"]
            or int(manifest["fit_input"]["size_bytes"]) != len(fit_payload)
            or manifest["fit_input"].get("target_fields_present") is not False
        ):
            raise RagEvidenceContractError(
                f"RAG preparation {build_id} differs from raw-source reconstruction"
            )
        load_fit_input_bytes(fit_payload, registry)
        if verify_private_artifacts:
            private_payload = read_bound_file_bytes(
                private_path,
                expected_sha256=reconstruction["private_label_sha256"],
                expected_size=reconstruction["private_label_size_bytes"],
                name=f"RAG private labels {build_id}",
            )
            if (
                manifest["private_labels"]["sha256"] != reconstruction["private_label_sha256"]
                or len(private_payload) != reconstruction["private_label_size_bytes"]
            ):
                raise RagEvidenceContractError(
                    f"RAG private labels {build_id} differ from raw-source reconstruction"
                )
            load_private_labels_bytes(private_payload, registry)
        if (
            manifest["private_labels"]["sha256"] != reconstruction["private_label_sha256"]
            or int(manifest["private_labels"]["size_bytes"]) != reconstruction["private_label_size_bytes"]
            or manifest["source_binding"] != reconstruction["source_binding"]
            or manifest["source_binding_sha256"] != reconstruction["source_binding"]["binding_sha256"]
            or manifest["source_snapshot"] != reconstruction["source_snapshot"]
            or manifest["rosters"] != reconstruction["rosters"]
            or manifest.get("labels_exposed_to_fit") is not False
            or manifest.get("historical_scores_opened") is not False
        ):
            raise RagEvidenceContractError(f"RAG preparation {build_id} provenance drifted")
        manifests[build_id] = manifest
        manifest_payloads[build_id] = manifest_payload
        manifest_hashes[build_id] = sha256_bytes(manifest_payload)
    comparisons = {
        "fit_input_byte_identity": manifests["A"]["fit_input"]["sha256"] == manifests["B"]["fit_input"]["sha256"],
        "private_label_byte_identity": manifests["A"]["private_labels"]["sha256"] == manifests["B"]["private_labels"]["sha256"],
        "source_binding_identity": manifests["A"]["source_binding_sha256"] == manifests["B"]["source_binding_sha256"],
        "source_snapshot_identity": manifests["A"]["source_snapshot"]["snapshot_sha256"] == manifests["B"]["source_snapshot"]["snapshot_sha256"],
        "roster_identity": manifests["A"]["rosters"] == manifests["B"]["rosters"],
    }
    if not all(comparisons.values()):
        raise RagEvidenceContractError("RAG A/B preparation identity failed")
    return add_payload_sha256({
        "schema_version": PREPARATION_AB_SCHEMA,
        "release_id": release_id,
        "status": "PASS",
        "scientific_full_required": bool(require_scientific_full),
        "comparisons": comparisons,
        "fit_input_sha256": reconstruction["fit_input_sha256"],
        "private_label_sha256": reconstruction["private_label_sha256"],
        "source_binding_sha256": reconstruction["source_binding"]["binding_sha256"],
        "source_asset_roster_sha256": reconstruction["source_binding"]["asset_roster_sha256"],
        "rosters": reconstruction["rosters"],
        "labels_exposed_to_fit": False,
        "historical_scores_copied": False,
        "independent_raw_source_reconstruction": True,
        "paired_public_private_recovery": True,
        "pair_transaction_ids": {
            build_id: manifests[build_id]["pair_transaction_id"]
            for build_id in ("A", "B")
        },
        "builds": {
            build_id: {
                "preparation_manifest_sha256": manifest_hashes[build_id],
                "preparation_manifest_payload_sha256": manifests[build_id]["payload_sha256"],
            }
            for build_id in ("A", "B")
        },
    })


def _derive_score_certificate(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    require_scientific_full: bool,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    repo_path = Path(repo).resolve(strict=True)
    lane_root = Path(release_root) / release_id / "rag_evidence"
    from .rag_evidence_runner import _BoundRagFitSourceClosure, _source_snapshot

    with ExitStack() as stack:
        fit_sources = stack.enter_context(_BoundRagFitSourceClosure(
            repo_path, allow_dirty_debug=not require_scientific_full
        ))
        builds = {
            build_id: stack.enter_context(BoundRagTree(
                lane_root / build_id,
                name=f"RAG score public build {build_id}",
            ))
            for build_id in ("A", "B")
        }
        assert_physical_tree_independence(
            builds["A"], builds["B"], name="RAG score public A/B builds"
        )
        result = _derive_score_certificate_bound(
            repo=repo_path,
            registry_path=registry_path,
            source_root=source_root,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            require_scientific_full=require_scientific_full,
            fit_sources=fit_sources,
        )
        for binding in builds.values():
            binding.verify_stable()
        fit_sources.verify_stable()
        if _source_snapshot(
            repo_path, allow_dirty_debug=not require_scientific_full
        ) != fit_sources.snapshot:
            raise RagEvidenceContractError(
                "RAG fit source closure changed during score certification"
            )
        return result


def _derive_preparation_certificate(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    require_scientific_full: bool, verify_private_artifacts: bool,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    lane_root = Path(release_root) / release_id / "rag_evidence"
    private_lane_root = Path(private_root) / release_id / "rag_evidence"
    with ExitStack() as stack:
        public = {
            build_id: stack.enter_context(BoundRagTree(
                lane_root / build_id,
                name=f"RAG preparation public build {build_id}",
            ))
            for build_id in ("A", "B")
        }
        assert_physical_tree_independence(
            public["A"], public["B"], name="RAG preparation public A/B builds"
        )
        private: dict[str, BoundRagTree] = {}
        if verify_private_artifacts:
            private = {
                build_id: stack.enter_context(BoundRagTree(
                    private_lane_root / build_id,
                    name=f"RAG preparation private build {build_id}",
                ))
                for build_id in ("A", "B")
            }
            assert_physical_tree_independence(
                private["A"], private["B"],
                name="RAG preparation private A/B builds",
            )
            for build_id in ("A", "B"):
                assert_physical_tree_independence(
                    public[build_id], private[build_id],
                    name=f"RAG preparation public/private build {build_id}",
                )
        result = _derive_preparation_certificate_bound(
            repo=repo,
            registry_path=registry_path,
            source_root=source_root,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            require_scientific_full=require_scientific_full,
            verify_private_artifacts=verify_private_artifacts,
        )
        for binding in (*public.values(), *private.values()):
            binding.verify_stable()
        return result


def verify_rag_evidence_preparation_ab(**kwargs: Any) -> dict[str, Any]:
    certificate = _derive_preparation_certificate(
        **kwargs, verify_private_artifacts=True
    )
    path = Path(kwargs["release_root"]) / kwargs["release_id"] / "rag_evidence" / PREPARATION_CERTIFICATE
    write_json_noreplace(path, certificate)
    return certificate


def authenticate_rag_evidence_preparation_certificate(**kwargs: Any) -> dict[str, Any]:
    validate_artifact_identifier(
        kwargs["release_id"], name="RAG release ID"
    )
    lane_root = Path(kwargs["release_root"]) / kwargs["release_id"] / "rag_evidence"
    observed = _load_pass(
        lane_root / PREPARATION_CERTIFICATE,
        schema=PREPARATION_AB_SCHEMA, name="RAG preparation A/B certificate",
    )
    derived = _derive_preparation_certificate(
        **kwargs, verify_private_artifacts=False
    )
    if observed != derived:
        raise RagEvidenceContractError(
            "RAG preparation certificate differs from transitive raw-source rederivation"
        )
    return observed


def _independently_reexecute_score_worker(
    *, repo: Path, registry_path: Path, source_root: Path,
    private_label_path: Path, fit_input_path: Path, fit_input_sha256: str,
    registry: Mapping[str, Any], release_id: str, fit_sources: Any | None = None,
) -> dict[str, Any]:
    """Build a third capsule and recompute scores without opening targets.

    This is the score certificate's trust anchor.  A/B equality alone is not
    evidence: both build trees may have been rewritten in the same way.
    """

    from .rag_evidence_runner import (
        _BoundRagFitSourceClosure,
        _copy_capsule,
        _launch_worker,
        _policy,
    )

    if fit_sources is None:
        with _BoundRagFitSourceClosure(
            repo, allow_dirty_debug=True
        ) as owned_sources:
            return _independently_reexecute_score_worker(
                repo=repo,
                registry_path=registry_path,
                source_root=source_root,
                private_label_path=private_label_path,
                fit_input_path=fit_input_path,
                fit_input_sha256=fit_input_sha256,
                registry=registry,
                release_id=release_id,
                fit_sources=owned_sources,
            )

    with tempfile.TemporaryDirectory(
        prefix="rag-score-cert-",
        dir=Path(tempfile.gettempdir()).resolve(strict=True),
    ) as temporary_name:
        root = Path(temporary_name)
        code_root = _copy_capsule(
            repo, root / "capsule", source_closure=fit_sources
        )
        worker_tmp = root / "worker_tmp"
        worker_tmp.mkdir()
        output_root = root / "candidate"
        forbidden = [
            ("private_labels", private_label_path),
            ("full_registry", registry_path),
            (
                "preparation_adapter",
                repo / "spectral_utils/reconstruction_benchmark/rag_evidence_preparation.py",
            ),
            (
                "postfreeze_evaluation",
                repo / "spectral_utils/reconstruction_benchmark/rag_evidence_evaluation.py",
            ),
        ]
        forbidden.extend(
            (f"raw_source::{asset_id}", source_root / item["path"])
            for asset_id, item in registry["sources"].items()
        )
        policy = _policy(
            code_root=code_root,
            input_root=fit_input_path.parent,
            output_root=output_root,
            temp_root=worker_tmp,
            forbidden=forbidden,
        )
        _launch_worker(
            code_root=code_root,
            input_path=fit_input_path,
            input_sha256=fit_input_sha256,
            output_root=output_root,
            temp_root=worker_tmp,
            release_id=release_id,
            build_id="A",
            lane_id=str(registry["lane_id"]),
            forbidden_fields=list(registry["fit_visibility"]["forbidden_fields"]),
            policy=policy,
        )
        worker_path = output_root / "WORKER_RESULT.json"
        worker, _ = _load_json_capture(
            worker_path,
            schema="reconstruction-rag-evidence-worker-result-v1",
            name="independent RAG score worker",
        )
        expected_probes = [
            {"probe_id": row["probe_id"], "read_denied": True}
            for row in policy["forbidden_probes"]
        ]
        score_path = output_root / SCORES_FILENAME
        independent_fit_payload = read_bound_file_bytes(
            fit_input_path,
            expected_sha256=fit_input_sha256,
            name="independent RAG fit input",
        )
        independent_fit_input = load_fit_input_bytes(
            independent_fit_payload, registry
        )
        worker_score_sha256 = worker.get("score_sha256")
        if not isinstance(worker_score_sha256, str):
            raise RagEvidenceContractError(
                "independent RAG worker omitted its score digest"
            )
        score_payload = read_bound_file_bytes(
            score_path,
            expected_sha256=worker_score_sha256,
            name="independent RAG score archive",
        )
        load_scores_bytes(score_payload, fit_input=independent_fit_input)
        if (
            worker.get("input_sha256") != fit_input_sha256
            or worker.get("score_sha256") != sha256_bytes(score_payload)
            or worker.get("audit_policy_sha256") != policy["policy_sha256"]
            or worker.get("denial_probes") != expected_probes
            or worker.get("firewall_violations") != []
            or worker.get("labels_opened_by_fit") is not False
            or worker.get("historical_scores_opened") is not False
        ):
            raise RagEvidenceContractError("independent RAG score worker failed its firewall")
        fit_sources.verify_stable()
        return {
            "score_bytes": score_payload,
            "score_sha256": sha256_bytes(score_payload),
            "diagnostics": worker["diagnostics"],
            "capsule_tree": canonical_tree_manifest(root / "capsule"),
            "denial_probes": expected_probes,
        }


def _assert_independent_score_match(
    *, independent: Mapping[str, Any], scores: Mapping[str, bytes],
    manifests: Mapping[str, Mapping[str, Any]],
    worker_diagnostics: Mapping[str, Any], capsule_trees: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    """Reject an identically coordinated A/B score or capsule rewrite."""

    checks = {
        "score_matches_A": independent["score_bytes"] == scores["A"],
        "score_matches_B": independent["score_bytes"] == scores["B"],
        "score_sha256_matches": independent["score_sha256"]
        == manifests["A"]["scores"]["sha256"],
        "fit_diagnostics_match": independent["diagnostics"] == worker_diagnostics["A"],
        "capsule_matches_A": independent["capsule_tree"]["tree_sha256"]
        == capsule_trees["A"]["tree_sha256"],
        "capsule_matches_B": independent["capsule_tree"]["tree_sha256"]
        == capsule_trees["B"]["tree_sha256"],
    }
    if not all(checks.values()):
        raise RagEvidenceContractError(
            "RAG coordinated score/capsule rewrite differs from independent re-execution"
        )
    return checks


def _derive_score_certificate_bound(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    require_scientific_full: bool, fit_sources: Any,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    registry = load_registry(registry_path)
    current_fit_snapshot = fit_sources.snapshot
    lane_root = Path(release_root) / release_id / "rag_evidence"
    prep = authenticate_rag_evidence_preparation_certificate(
        repo=repo, registry_path=registry_path, source_root=source_root,
        release_root=release_root, private_root=private_root, release_id=release_id,
        require_scientific_full=require_scientific_full,
    )
    prep_certificate, prep_certificate_payload = _load_json_capture(
        lane_root / PREPARATION_CERTIFICATE,
        schema=PREPARATION_AB_SCHEMA,
        name="authenticated RAG preparation A/B certificate",
    )
    if prep_certificate != prep:
        raise RagEvidenceContractError(
            "RAG preparation certificate changed after authentication"
        )
    canonical_fit_path = lane_root / "A" / "inputs" / FIT_INPUT_FILENAME
    canonical_fit_payload = read_bound_file_bytes(
        canonical_fit_path,
        expected_sha256=prep["fit_input_sha256"],
        name="canonical RAG fit input",
    )
    canonical_fit_input = load_fit_input_bytes(canonical_fit_payload, registry)
    expected_denial_probes = [
        {"probe_id": probe_id, "read_denied": True}
        for probe_id in (
            "private_labels",
            "full_registry",
            "preparation_adapter",
            "postfreeze_evaluation",
            *(f"raw_source::{asset_id}" for asset_id in registry["sources"]),
        )
    ]
    manifests, scores, worker_diagnostics, capsule_trees = {}, {}, {}, {}
    manifest_payloads: dict[str, bytes] = {}
    worker_payloads: dict[str, bytes] = {}
    preparation_manifests: dict[str, dict[str, Any]] = {}
    preparation_manifest_payloads: dict[str, bytes] = {}
    for build_id in ("A", "B"):
        build_root = lane_root / build_id
        prep_path = build_root / PREPARATION_MANIFEST_FILENAME
        score_path = build_root / "fit" / SCORE_MANIFEST_FILENAME
        manifest, manifest_payload = _load_json_capture(
            score_path, schema=SCORE_FREEZE_SCHEMA,
            name=f"RAG score freeze {build_id}",
        )
        if (
            manifest.get("release_id") != release_id
            or manifest.get("build_id") != build_id
            or manifest.get("lane_id") != registry["lane_id"]
        ):
            raise RagEvidenceContractError(f"RAG score {build_id} release binding failed")
        if require_scientific_full and manifest.get("scientific_full") is not True:
            raise RagEvidenceContractError(f"RAG score {build_id} is not scientific-full")
        preparation_manifest, preparation_manifest_payload = _load_json_capture(
            prep_path, schema=PREPARATION_SCHEMA,
            name=f"RAG preparation {build_id}",
        )
        if (
            manifest.get("preparation_manifest_sha256")
            != sha256_bytes(preparation_manifest_payload)
            or manifest.get("preparation_manifest_payload_sha256")
            != preparation_manifest["payload_sha256"]
            or manifest.get("preparation_ab_sha256")
            != sha256_bytes(prep_certificate_payload)
            or manifest.get("fit_input_sha256") != prep["fit_input_sha256"]
            or manifest.get("source_binding_sha256") != prep["source_binding_sha256"]
            or manifest.get("source_snapshot") != current_fit_snapshot
            or manifest.get("labels_opened_by_fit") is not False
            or manifest.get("historical_scores_opened") is not False
            or manifest.get("firewall_violations") != []
            or manifest.get("denial_probes") != expected_denial_probes
            or manifest.get("all_registered_panels_scored") is not True
        ):
            raise RagEvidenceContractError(f"RAG score {build_id} provenance/firewall failed")
        score_descriptor = manifest.get("scores")
        if (
            not isinstance(score_descriptor, Mapping)
            or set(score_descriptor) != {"path", "sha256", "size_bytes"}
            or score_descriptor.get("path") != f"candidate/{SCORES_FILENAME}"
        ):
            raise RagEvidenceContractError(f"RAG score {build_id} file descriptor drifted")
        frozen_path = build_root / "fit" / score_descriptor["path"]
        frozen_payload = read_bound_file_bytes(
            frozen_path,
            expected_sha256=str(score_descriptor["sha256"]),
            expected_size=int(score_descriptor["size_bytes"]),
            name=f"RAG score archive {build_id}",
        )
        scores[build_id] = frozen_payload
        load_scores_bytes(frozen_payload, fit_input=canonical_fit_input)
        worker_descriptor = manifest.get("worker_result")
        if (
            not isinstance(worker_descriptor, Mapping)
            or set(worker_descriptor) != {"path", "sha256", "payload_sha256"}
            or worker_descriptor.get("path") != "candidate/WORKER_RESULT.json"
        ):
            raise RagEvidenceContractError(
                f"RAG worker result {build_id} file descriptor drifted"
            )
        worker_path = build_root / "fit" / worker_descriptor["path"]
        worker, worker_payload = _load_json_capture(
            worker_path, schema="reconstruction-rag-evidence-worker-result-v1",
            name=f"RAG worker result {build_id}",
            expected_sha256=str(worker_descriptor["sha256"]),
        )
        if (
            worker["payload_sha256"] != manifest["worker_result"]["payload_sha256"]
            or worker.get("release_id") != release_id
            or worker.get("build_id") != build_id
            or worker.get("lane_id") != registry["lane_id"]
            or worker.get("input_sha256") != prep["fit_input_sha256"]
            or worker.get("score_sha256") != manifest["scores"]["sha256"]
            or worker.get("score_schema") != SCORE_FREEZE_SCHEMA
            or worker.get("score_path") != SCORES_FILENAME
            or worker.get("audit_policy_sha256") != manifest["audit_policy_sha256"]
            or worker.get("denial_probes") != expected_denial_probes
            or worker.get("labels_opened_by_fit") is not False
            or worker.get("historical_scores_opened") is not False
            or worker.get("firewall_violations") != []
        ):
            raise RagEvidenceContractError(f"RAG worker result {build_id} binding failed")
        validate_source_binding(
            preparation_manifest["source_binding"],
            source_root=source_root, registry=registry,
        )
        actual_capsule = canonical_tree_manifest(build_root / "fit" / "capsule")
        if actual_capsule["tree_sha256"] != manifest["capsule_tree_sha256"]:
            raise RagEvidenceContractError(f"RAG score {build_id} capsule tree was rewritten")
        manifests[build_id] = manifest
        manifest_payloads[build_id] = manifest_payload
        preparation_manifests[build_id] = preparation_manifest
        preparation_manifest_payloads[build_id] = preparation_manifest_payload
        worker_payloads[build_id] = worker_payload
        worker_diagnostics[build_id] = worker["diagnostics"]
        capsule_trees[build_id] = actual_capsule
    comparisons = {
        "score_byte_identity": scores["A"] == scores["B"],
        "score_sha256_identity": manifests["A"]["scores"]["sha256"] == manifests["B"]["scores"]["sha256"],
        "fit_input_identity": manifests["A"]["fit_input_sha256"] == manifests["B"]["fit_input_sha256"],
        "source_binding_identity": manifests["A"]["source_binding_sha256"] == manifests["B"]["source_binding_sha256"],
        "fit_diagnostic_identity": worker_diagnostics["A"] == worker_diagnostics["B"],
        "capsule_identity": capsule_trees["A"]["tree_sha256"] == capsule_trees["B"]["tree_sha256"],
    }
    if not all(comparisons.values()):
        raise RagEvidenceContractError("RAG A/B score identity failed")
    declared_canonical_fit_path = (
        lane_root / "A" / preparation_manifests["A"]["fit_input"]["path"]
    )
    if declared_canonical_fit_path != canonical_fit_path:
        raise RagEvidenceContractError("RAG canonical fit-input path drifted")
    independent = _independently_reexecute_score_worker(
        repo=Path(repo).resolve(strict=True),
        registry_path=Path(registry_path).resolve(strict=True),
        source_root=Path(source_root).resolve(strict=True),
        private_label_path=Path(preparation_manifests["A"]["private_labels"]["path"]),
        fit_input_path=canonical_fit_path,
        fit_input_sha256=prep["fit_input_sha256"],
        registry=registry,
        release_id=release_id,
        fit_sources=fit_sources,
    )
    fit_sources.verify_stable()
    rederivation = _assert_independent_score_match(
        independent=independent, scores=scores, manifests=manifests,
        worker_diagnostics=worker_diagnostics, capsule_trees=capsule_trees,
    )
    comparisons.update({f"independent_{key}": value for key, value in rederivation.items()})
    return add_payload_sha256({
        "schema_version": SCORE_AB_SCHEMA,
        "release_id": release_id,
        "status": "PASS",
        "scientific_full_required": bool(require_scientific_full),
        "preparation_ab_sha256": sha256_bytes(prep_certificate_payload),
        "preparation_ab_payload_sha256": prep["payload_sha256"],
        "comparisons": comparisons,
        "score_sha256": manifests["A"]["scores"]["sha256"],
        "fit_input_sha256": prep["fit_input_sha256"],
        "private_label_sha256": prep["private_label_sha256"],
        "source_binding_sha256": prep["source_binding_sha256"],
        "labels_opened_by_fit": False,
        "historical_scores_copied": False,
        "transitive_source_rederivation": True,
        "independent_score_reexecution": True,
        "independent_score_sha256": independent["score_sha256"],
        "independent_capsule_tree_sha256": independent["capsule_tree"]["tree_sha256"],
        "builds": {
            build_id: {
                "score_manifest_sha256": sha256_bytes(
                    manifest_payloads[build_id]
                ),
                "score_manifest_payload_sha256": manifests[build_id]["payload_sha256"],
            }
            for build_id in ("A", "B")
        },
    })


def verify_rag_evidence_score_ab(**kwargs: Any) -> dict[str, Any]:
    certificate = _derive_score_certificate(**kwargs)
    path = Path(kwargs["release_root"]) / kwargs["release_id"] / "rag_evidence" / SCORE_CERTIFICATE
    write_json_noreplace(path, certificate)
    return certificate


def authenticate_rag_evidence_score_certificate(**kwargs: Any) -> dict[str, Any]:
    validate_artifact_identifier(
        kwargs["release_id"], name="RAG release ID"
    )
    lane_root = Path(kwargs["release_root"]) / kwargs["release_id"] / "rag_evidence"
    observed = _load_pass(
        lane_root / SCORE_CERTIFICATE,
        schema=SCORE_AB_SCHEMA, name="RAG score A/B certificate",
    )
    derived = _derive_score_certificate(**kwargs)
    if observed != derived:
        raise RagEvidenceContractError(
            "RAG score certificate differs from transitive score/source rederivation"
        )
    return observed


def _assert_independent_evaluation_match(
    *, expected_payloads: Mapping[str, bytes],
    observed_payloads: Mapping[str, Mapping[str, bytes]],
    manifests: Mapping[str, Mapping[str, Any]],
    expected_panel_status: list[dict[str, Any]],
) -> dict[str, bool]:
    """Reject identically fabricated A/B reporting tables and status rows.

    This compares the evaluator's independently regenerated bytes, rather than
    trusting the hashes copied into either evaluation manifest.
    """

    checks: dict[str, bool] = {}
    for name, payload in expected_payloads.items():
        checks[f"{name}_matches_A"] = observed_payloads["A"].get(name) == payload
        checks[f"{name}_matches_B"] = observed_payloads["B"].get(name) == payload
    checks["panel_status_matches_A"] = manifests["A"].get("panel_status") == expected_panel_status
    checks["panel_status_matches_B"] = manifests["B"].get("panel_status") == expected_panel_status
    if not all(checks.values()):
        raise RagEvidenceContractError(
            "RAG coordinated evaluation rewrite differs from independent re-evaluation"
        )
    return checks


def _derive_evaluation_certificate_bound(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    require_scientific_full: bool,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    registry = load_registry(registry_path)
    from .rag_evidence_evaluation import EVALUATION_SOURCE_FILES

    current_evaluation_snapshot = {
        "files": [
            {
                "path": relative,
                "sha256": sha256_bytes(
                    read_bound_file_bytes(
                        Path(repo).resolve(strict=True) / relative,
                        name=f"RAG evaluation source {relative}",
                    )
                ),
            }
            for relative in EVALUATION_SOURCE_FILES
        ]
    }
    from .rag_evidence_contract import payload_sha256

    current_evaluation_snapshot["snapshot_sha256"] = payload_sha256(
        current_evaluation_snapshot
    )
    lane_root = Path(release_root) / release_id / "rag_evidence"
    score = authenticate_rag_evidence_score_certificate(
        repo=repo, registry_path=registry_path, source_root=source_root,
        release_root=release_root, private_root=private_root, release_id=release_id,
        require_scientific_full=require_scientific_full,
    )
    score_certificate, score_certificate_payload = _load_json_capture(
        lane_root / SCORE_CERTIFICATE,
        schema=SCORE_AB_SCHEMA,
        name="authenticated RAG score A/B certificate",
    )
    if score_certificate != score:
        raise RagEvidenceContractError(
            "RAG score certificate changed after authentication"
        )
    score_manifests: dict[str, dict[str, Any]] = {}
    score_manifest_payloads: dict[str, bytes] = {}
    for build_id in ("A", "B"):
        value, payload = _load_json_capture(
            lane_root / build_id / "fit" / SCORE_MANIFEST_FILENAME,
            schema=SCORE_FREEZE_SCHEMA,
            name=f"RAG score for evaluation {build_id}",
        )
        score_manifests[build_id] = value
        score_manifest_payloads[build_id] = payload
    manifests: dict[str, dict[str, Any]] = {}
    manifest_payloads: dict[str, bytes] = {}
    file_hashes: dict[str, dict[str, str]] = {}
    file_payloads: dict[str, dict[str, bytes]] = {}
    comparison_files = ("metrics.csv", "predictions.csv", "contrasts.csv", "panel_status.csv")
    expected_bootstrap = {
        "draws_requested": int(registry["evaluation"]["bootstrap"]["draws"]),
        "group": "panel-registered source group",
        "paired_contrasts": True,
        "seed": int(registry["evaluation"]["bootstrap"]["seed"]),
    }
    for build_id in ("A", "B"):
        root = lane_root / build_id / "evaluation"
        manifest_path = root / EVALUATION_MANIFEST_FILENAME
        manifest, manifest_payload = _load_json_capture(
            manifest_path, schema=EVALUATION_SCHEMA,
            name=f"RAG evaluation {build_id}",
        )
        if (
            manifest.get("release_id") != release_id
            or manifest.get("build_id") != build_id
            or manifest.get("lane_id") != registry["lane_id"]
        ):
            raise RagEvidenceContractError(f"RAG evaluation {build_id} release binding failed")
        if require_scientific_full and manifest.get("scientific_full") is not True:
            raise RagEvidenceContractError(f"RAG evaluation {build_id} is not scientific-full")
        if (
            manifest.get("score_sha256") != score["score_sha256"]
            or manifest.get("score_manifest_sha256")
            != sha256_bytes(score_manifest_payloads[build_id])
            or manifest.get("private_label_sha256") != score["private_label_sha256"]
            or manifest.get("source_binding_sha256") != score["source_binding_sha256"]
            or manifest.get("score_ab_certificate_sha256")
            != sha256_bytes(score_certificate_payload)
            or manifest.get("source_snapshot") != current_evaluation_snapshot
            or manifest.get("cross_panel_macro_computed") is not False
            or manifest.get("refchecker_settings_pooled") is not False
            or manifest.get("historical_scores_copied") is not False
            or manifest.get("bootstrap") != expected_bootstrap
            or [row["panel_id"] for row in manifest.get("panel_status", ())] != list(PANEL_IDS)
            or any(row["status"] != "PASS" for row in manifest.get("panel_status", ()))
        ):
            raise RagEvidenceContractError(f"RAG evaluation {build_id} panel/binding failed")
        declared_rows = manifest.get("files")
        if (
            not isinstance(declared_rows, list)
            or len(declared_rows) != len(comparison_files)
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"path", "sha256", "size_bytes"}
                for row in declared_rows
            )
        ):
            raise RagEvidenceContractError(
                f"RAG evaluation {build_id} file descriptors drifted"
            )
        declared = {row["path"]: row for row in declared_rows}
        if set(declared) != set(comparison_files):
            raise RagEvidenceContractError(f"RAG evaluation {build_id} file roster drifted")
        file_hashes[build_id] = {}
        file_payloads[build_id] = {}
        for name in comparison_files:
            path = root / name
            payload = read_bound_file_bytes(
                path,
                expected_sha256=str(declared[name]["sha256"]),
                expected_size=int(declared[name]["size_bytes"]),
                name=f"RAG evaluation {build_id}/{name}",
            )
            file_hashes[build_id][name] = declared[name]["sha256"]
            file_payloads[build_id][name] = payload
        metric_rows = list(csv.DictReader(StringIO(
            file_payloads[build_id]["metrics.csv"].decode("utf-8"), newline=""
        )))
        if not metric_rows or {row["panel_id"] for row in metric_rows} != set(PANEL_IDS):
            raise RagEvidenceContractError(f"RAG evaluation {build_id} metric panel coverage failed")
        panel_contract = {row["panel_id"]: row for row in registry["panels"]}
        for row in metric_rows:
            contract = panel_contract.get(row["panel_id"])
            if (
                contract is None
                or row["metric"] not in contract["metrics"]
                or row["method_id"] not in contract["methods"]
                or (
                    row["panel_id"].startswith("refchecker_")
                    and row["subgroup"] not in REFCHECKER_SETTINGS
                )
            ):
                raise RagEvidenceContractError(
                    f"RAG evaluation {build_id} contains a pooled/unregistered metric row"
                )
        contrast_rows = list(csv.DictReader(StringIO(
            file_payloads[build_id]["contrasts.csv"].decode("utf-8"), newline=""
        )))
        if not contrast_rows or any(
            row["panel_id"] != "gasp_protocol_sentence"
            or {row["left_method"], row["right_method"]}
            != {"gasp_threshold", "fixed_rag_iu_pcr_matched"}
            for row in contrast_rows
        ):
            raise RagEvidenceContractError(
                f"RAG evaluation {build_id} contains a cross-panel contrast"
            )
        status_rows = list(csv.DictReader(StringIO(
            file_payloads[build_id]["panel_status.csv"].decode("utf-8"), newline=""
        )))
        expected_status = [{key: str(value) for key, value in row.items()} for row in manifest["panel_status"]]
        if status_rows != expected_status:
            raise RagEvidenceContractError(
                f"RAG evaluation {build_id} status table differs from its manifest"
            )
        seen_prediction = False
        for row in csv.DictReader(StringIO(
            file_payloads[build_id]["predictions.csv"].decode("utf-8"), newline=""
        )):
            seen_prediction = True
            if (
                row["panel_id"] not in PANEL_IDS
                or (
                    row["panel_id"].startswith("refchecker_")
                    and row["subgroup"] not in REFCHECKER_SETTINGS
                )
            ):
                raise RagEvidenceContractError(
                    f"RAG evaluation {build_id} prediction panel semantics failed"
                )
        if not seen_prediction:
            raise RagEvidenceContractError(f"RAG evaluation {build_id} has no predictions")
        manifests[build_id] = manifest
        manifest_payloads[build_id] = manifest_payload
    # Only a fully authenticated, independently re-executed score certificate
    # permits this first private-label open.
    preparation_manifests: dict[str, dict[str, Any]] = {}
    for build_id in ("A", "B"):
        preparation_manifests[build_id], _ = _load_json_capture(
            lane_root / build_id / PREPARATION_MANIFEST_FILENAME,
            schema=PREPARATION_SCHEMA,
            name=f"RAG preparation for evaluation {build_id}",
        )
    private_paths = {
        build_id: Path(preparation_manifests[build_id]["private_labels"]["path"])
        for build_id in ("A", "B")
    }
    private_payloads = {
        build_id: read_bound_file_bytes(
            path,
            expected_sha256=score["private_label_sha256"],
            name=f"RAG private labels for evaluation certificate {build_id}",
        )
        for build_id, path in private_paths.items()
    }
    if private_payloads["A"] != private_payloads["B"]:
        raise RagEvidenceContractError(
            "RAG evaluation private labels differ across independent builds"
        )
    private = load_private_labels_bytes(private_payloads["A"], registry)
    score_paths = {
        build_id: lane_root / build_id / "fit" / score_manifests[build_id]["scores"]["path"]
        for build_id in ("A", "B")
    }
    score_payloads = {
        build_id: read_bound_file_bytes(
            path,
            expected_sha256=score["score_sha256"],
            name=f"RAG score input for evaluation certificate {build_id}",
        )
        for build_id, path in score_paths.items()
    }
    if score_payloads["A"] != score_payloads["B"]:
        raise RagEvidenceContractError(
            "RAG evaluation score inputs differ across independent builds"
        )
    scores = load_scores_bytes(score_payloads["A"])
    from .rag_evidence_evaluation import compute_rag_evidence_evaluation_tables

    independent = compute_rag_evidence_evaluation_tables(
        registry=registry,
        private=private,
        scores=scores,
        draws=expected_bootstrap["draws_requested"],
        seed=expected_bootstrap["seed"],
    )
    rederivation = _assert_independent_evaluation_match(
        expected_payloads=independent["file_payloads"],
        observed_payloads=file_payloads,
        manifests=manifests,
        expected_panel_status=independent["panel_status"],
    )
    comparisons = {
        name: file_hashes["A"][name] == file_hashes["B"][name]
        for name in comparison_files
    }
    comparisons.update({
        "source_snapshot_identity": manifests["A"]["source_snapshot"] == manifests["B"]["source_snapshot"],
        "bootstrap_identity": manifests["A"]["bootstrap"] == manifests["B"]["bootstrap"],
        "panel_status_identity": manifests["A"]["panel_status"] == manifests["B"]["panel_status"],
    })
    comparisons.update({f"independent_{key}": value for key, value in rederivation.items()})
    if not all(comparisons.values()):
        raise RagEvidenceContractError("RAG A/B evaluation identity failed")
    return add_payload_sha256({
        "schema_version": EVALUATION_AB_SCHEMA,
        "release_id": release_id,
        "status": "PASS",
        "scientific_full_required": bool(require_scientific_full),
        "score_ab_sha256": sha256_bytes(score_certificate_payload),
        "score_ab_payload_sha256": score["payload_sha256"],
        "comparisons": comparisons,
        "score_sha256": score["score_sha256"],
        "cross_panel_macro_computed": False,
        "refchecker_settings_pooled": False,
        "transitive_source_rederivation": True,
        "independent_postfreeze_reevaluation": True,
        "reporting_files": {
            name: sha256_bytes(file_payloads["A"][name])
            for name in comparison_files
        },
        "builds": {
            build_id: {
                "evaluation_manifest_sha256": sha256_bytes(
                    manifest_payloads[build_id]
                ),
                "evaluation_manifest_payload_sha256": manifests[build_id]["payload_sha256"],
            }
            for build_id in ("A", "B")
        },
    })


def _derive_evaluation_certificate(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    require_scientific_full: bool,
) -> dict[str, Any]:
    """Hold both public/private A/B roots through the complete certificate."""

    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    lane_root = Path(release_root) / release_id / "rag_evidence"
    private_lane_root = Path(private_root) / release_id / "rag_evidence"
    with ExitStack() as stack:
        public = {
            build_id: stack.enter_context(BoundRagTree(
                lane_root / build_id,
                name=f"RAG public build {build_id}",
            ))
            for build_id in ("A", "B")
        }
        private = {
            build_id: stack.enter_context(BoundRagTree(
                private_lane_root / build_id,
                name=f"RAG private build {build_id}",
            ))
            for build_id in ("A", "B")
        }
        assert_physical_tree_independence(
            public["A"], public["B"], name="RAG public A/B builds"
        )
        assert_physical_tree_independence(
            private["A"], private["B"], name="RAG private A/B builds"
        )
        for build_id in ("A", "B"):
            assert_physical_tree_independence(
                public[build_id],
                private[build_id],
                name=f"RAG public/private build {build_id}",
            )
        result = _derive_evaluation_certificate_bound(
            repo=repo,
            registry_path=registry_path,
            source_root=source_root,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            require_scientific_full=require_scientific_full,
        )
        for binding in (*public.values(), *private.values()):
            binding.verify_stable()
        return result


def verify_rag_evidence_evaluation_ab(**kwargs: Any) -> dict[str, Any]:
    certificate = _derive_evaluation_certificate(**kwargs)
    path = Path(kwargs["release_root"]) / kwargs["release_id"] / "rag_evidence" / EVALUATION_CERTIFICATE
    write_json_noreplace(path, certificate)
    return certificate


__all__ = [
    "EVALUATION_CERTIFICATE", "PREPARATION_CERTIFICATE", "SCORE_CERTIFICATE",
    "authenticate_rag_evidence_preparation_certificate",
    "authenticate_rag_evidence_score_certificate",
    "verify_rag_evidence_evaluation_ab", "verify_rag_evidence_preparation_ab",
    "verify_rag_evidence_score_ab",
]
