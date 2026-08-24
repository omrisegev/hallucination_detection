"""Release-preserving verifier for frozen localization evaluation builds.

The evaluation producer source snapshot is already embedded in each published
build manifest.  This verifier therefore runs the unchanged producer from a
separate clean checkout at that recorded commit, while independently binding
the newer verifier implementation that fixes the A/B provenance comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from . import localization_ab as _localization_ab_module
from . import localization_contract as _localization_contract_module
from . import localization_evaluation as _localization_evaluation_module
from . import localization_postfreeze as _localization_postfreeze_module
from . import localization_postfreeze_amendment as _amendment_module
from .io import sha256_file
from .localization_contract import payload_sha256
from .localization_evaluation import DEFAULT_BOOTSTRAP_DRAWS
from .localization_postfreeze import (
    EVALUATION_SOURCE_FILES,
    _hashed_json,
    _json_bytes,
    _repo_state,
    _source_snapshot,
    _validate_evaluation_build_against_derivation,
    _write_immutable_certificate,
    derive_localization_evaluation,
)


EVALUATION_AB_RELEASE_SCHEMA_VERSION = (
    "reconstruction-localization-evaluation-ab-v4"
)
EXPECTED_EVALUATION_PRODUCER_GIT_HEAD = (
    "660251b9b7aabe2119bfb4e0a469f4d284a07c2d"
)
EXPECTED_EVALUATION_PRODUCER_SNAPSHOT_SHA256 = (
    "7d9a0576c0a6e0e2cb3e45fcd9d18c969953dc838193210e467a490a4efcabba"
)
BUILD_SPECIFIC_MANIFEST_CORE_FIELDS = frozenset({
    "score_freeze_payload_sha256",
})
VERIFIER_SOURCE_FILES = (
    "spectral_utils/reconstruction_benchmark/"
    "localization_evaluation_ab_verifier.py",
    "scripts/reconstruction_benchmark/"
    "verify_localization_evaluation_ab_release.py",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _verifier_source_snapshot(repo: Path) -> dict[str, Any]:
    state = _repo_state(repo)
    if state["git_clean"] is not True:
        raise RuntimeError("localization evaluation A/B verifier repo must be clean")
    files = []
    for relative in VERIFIER_SOURCE_FILES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"verifier source file is absent or unsafe: {relative}")
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    value = {
        "repo_role": "localization_evaluation_ab_verifier",
        **{
            key: state[key]
            for key in ("git_head", "git_clean", "git_status_sha256")
        },
        "files": files,
    }
    value["snapshot_sha256"] = payload_sha256(value)
    return value


def _recorded_producer_snapshot(
    *, release_root: Path, release_id: str,
) -> dict[str, Any]:
    snapshots = []
    for build_id in ("A", "B"):
        manifest_path = (
            release_root / release_id / f"build_{build_id}"
            / "localization/evaluation/MANIFEST.json"
        )
        manifest = _load_json_object(manifest_path)
        manifest_payload = dict(manifest)
        claimed_payload_hash = manifest_payload.pop("payload_sha256", None)
        if (
            claimed_payload_hash != payload_sha256(manifest_payload)
            or manifest.get("release_id") != release_id
            or manifest.get("build_id") != build_id
            or manifest.get("status") != "PASS"
            or manifest.get("scientific_full") is not True
            or int(manifest.get("bootstrap_draws", -1))
            != DEFAULT_BOOTSTRAP_DRAWS
        ):
            raise RuntimeError(f"build {build_id} evaluation manifest is invalid")
        snapshot = manifest.get("evaluation_source_snapshot")
        if not isinstance(snapshot, Mapping):
            raise RuntimeError(
                f"build {build_id} lacks its frozen evaluation producer snapshot"
            )
        snapshots.append(dict(snapshot))
    if snapshots[0] != snapshots[1]:
        raise RuntimeError("A/B evaluation producer snapshots differ")
    snapshot_payload = dict(snapshots[0])
    claimed_snapshot_hash = snapshot_payload.pop("snapshot_sha256", None)
    if (
        claimed_snapshot_hash != payload_sha256(snapshot_payload)
        or claimed_snapshot_hash != EXPECTED_EVALUATION_PRODUCER_SNAPSHOT_SHA256
        or snapshots[0].get("git_head") != EXPECTED_EVALUATION_PRODUCER_GIT_HEAD
        or snapshots[0].get("git_clean") is not True
    ):
        raise RuntimeError("frozen evaluation producer snapshot anchor is invalid")
    return snapshots[0]


def _attest_repository_boundary(
    *, producer_repo: Path, verifier_repo: Path,
    recorded_producer_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    producer = producer_repo.resolve()
    verifier = verifier_repo.resolve()
    if producer == verifier:
        raise RuntimeError(
            "evaluation producer and A/B verifier require separate clean checkouts"
        )
    observed_producer = _source_snapshot(producer)
    if observed_producer.get("git_clean") is not True:
        raise RuntimeError("frozen localization evaluation producer repo must be clean")
    if observed_producer != dict(recorded_producer_snapshot):
        raise RuntimeError(
            "evaluation producer repo does not match the build-manifest snapshot"
        )

    recorded_files = observed_producer.get("files")
    if not isinstance(recorded_files, list) or tuple(
        row.get("path") if isinstance(row, Mapping) else None
        for row in recorded_files
    ) != EVALUATION_SOURCE_FILES:
        raise RuntimeError("frozen evaluation producer source roster is not exact")
    for row in recorded_files:
        relative = str(row["path"])
        verifier_path = verifier / relative
        if verifier_path.is_symlink() or not verifier_path.is_file():
            raise RuntimeError(
                f"verifier checkout lacks frozen producer source: {relative}"
            )
        if sha256_file(verifier_path) != row.get("sha256"):
            raise RuntimeError(
                f"executed producer source differs from frozen snapshot: {relative}"
            )

    module_paths = {
        "spectral_utils/reconstruction_benchmark/localization_ab.py": (
            _localization_ab_module
        ),
        "spectral_utils/reconstruction_benchmark/localization_contract.py": (
            _localization_contract_module
        ),
        "spectral_utils/reconstruction_benchmark/localization_evaluation.py": (
            _localization_evaluation_module
        ),
        "spectral_utils/reconstruction_benchmark/localization_postfreeze.py": (
            _localization_postfreeze_module
        ),
        "spectral_utils/reconstruction_benchmark/"
        "localization_postfreeze_amendment.py": _amendment_module,
    }
    for relative, module in module_paths.items():
        if Path(module.__file__).resolve() != (verifier / relative).resolve():
            raise RuntimeError(f"executed producer import escaped its repo: {relative}")
    expected_module = (
        verifier / "spectral_utils/reconstruction_benchmark/"
        "localization_evaluation_ab_verifier.py"
    ).resolve()
    if Path(__file__).resolve() != expected_module:
        raise RuntimeError("localization evaluation verifier import escaped its repo")
    return {
        "evaluation_producer_snapshot": observed_producer,
        "evaluation_ab_verifier_source_snapshot": _verifier_source_snapshot(verifier),
    }


def _split_manifest_core(
    manifest_core: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(manifest_core, Mapping):
        raise RuntimeError("localization evaluation manifest core is not a mapping")
    missing = BUILD_SPECIFIC_MANIFEST_CORE_FIELDS - set(manifest_core)
    if missing:
        raise RuntimeError(
            "localization evaluation manifest core lacks the sole build-specific "
            f"field: {sorted(missing)}"
        )
    shared = dict(manifest_core)
    build_specific = {}
    for field in BUILD_SPECIFIC_MANIFEST_CORE_FIELDS:
        value = shared.pop(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"invalid build-specific manifest hash: {field}")
        build_specific[field] = value
    return shared, build_specific


def _verify_score_freeze_payload_bindings(
    *, release_root: Path, release_id: str,
    score_ab_certificate_path: Path,
    derived: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    certificate = _load_json_object(score_ab_certificate_path)
    certificate_payload = dict(certificate)
    claimed_certificate_hash = certificate_payload.pop("certificate_sha256", None)
    if (
        claimed_certificate_hash != payload_sha256(certificate_payload)
        or certificate.get("release_id") != release_id
        or certificate.get("status") != "PASS"
        or not isinstance(certificate.get("builds"), Mapping)
    ):
        raise RuntimeError("localization score A/B certificate is invalid")
    certificate_file_sha256 = sha256_file(score_ab_certificate_path)

    output: dict[str, dict[str, str]] = {}
    for build_id in ("A", "B"):
        freeze_path = (
            release_root / release_id / f"build_{build_id}"
            / "localization/fit/SCORE_FREEZE_MANIFEST.json"
        )
        if freeze_path.is_symlink() or not freeze_path.is_file():
            raise RuntimeError(f"build {build_id} score freeze is absent or unsafe")
        freeze = _hashed_json(freeze_path)
        score_binding = certificate["builds"].get(build_id)
        if not isinstance(score_binding, Mapping):
            raise RuntimeError(f"score certificate lacks build {build_id}")
        freeze_file_sha256 = sha256_file(freeze_path)
        expected_payload = derived[build_id].manifest_core.get(
            "score_freeze_payload_sha256"
        )
        derived_certificate_payload = derived[build_id].manifest_core.get(
            "score_ab_certificate_sha256"
        )
        derived_certificate_file = derived[build_id].manifest_core.get(
            "score_ab_certificate_file_sha256"
        )
        if (
            freeze.get("release_id") != release_id
            or freeze.get("build_id") != build_id
            or freeze_file_sha256 != score_binding.get("score_freeze_sha256")
            or freeze.get("payload_sha256") != expected_payload
            or derived_certificate_payload != claimed_certificate_hash
            or derived_certificate_file != certificate_file_sha256
        ):
            raise RuntimeError(
                f"build {build_id} score-freeze payload is not certificate-bound"
            )
        output[build_id] = {
            "score_freeze_file_sha256": freeze_file_sha256,
            "score_freeze_payload_sha256": str(expected_payload),
        }
    return output


def verify_localization_evaluation_ab_release(
    *, release_id: str, release_root: str | Path,
    score_verifier_repo: str | Path,
    evaluation_producer_repo: str | Path,
    verification_repo: str | Path,
    identity_key_path: str | Path | None = None,
    output_path: str | Path | None = None,
    localization_ab_certificate_path: str | Path | None = None,
    localization_registry_path: str | Path,
    external_registry_path: str | Path,
    population_registry_path: str | Path,
    source_root: str | Path,
    localization_postfreeze_amendment_path: str | Path,
) -> dict[str, Any]:
    """Rederive frozen A/B builds and certify their exact shared science bytes."""

    release_root_path = Path(release_root).resolve()
    producer_repo = Path(evaluation_producer_repo).resolve()
    verifier_repo = Path(verification_repo).resolve()
    recorded_snapshot = _recorded_producer_snapshot(
        release_root=release_root_path, release_id=release_id,
    )
    boundary_before = _attest_repository_boundary(
        producer_repo=producer_repo, verifier_repo=verifier_repo,
        recorded_producer_snapshot=recorded_snapshot,
    )

    derived = {}
    validated = {}
    for build_id in ("A", "B"):
        derived[build_id] = derive_localization_evaluation(
            release_id=release_id,
            build_id=build_id,
            release_root=release_root_path,
            identity_key_path=identity_key_path,
            localization_ab_certificate_path=localization_ab_certificate_path,
            localization_registry_path=localization_registry_path,
            external_registry_path=external_registry_path,
            population_registry_path=population_registry_path,
            source_root=source_root,
            score_verifier_repo=score_verifier_repo,
            evaluation_repo=producer_repo,
            localization_postfreeze_amendment_path=(
                localization_postfreeze_amendment_path
            ),
            bootstrap_draws=DEFAULT_BOOTSTRAP_DRAWS,
        )
        root = (
            release_root_path / release_id / f"build_{build_id}"
            / "localization/evaluation"
        )
        validated[build_id] = _validate_evaluation_build_against_derivation(
            root=root,
            release_id=release_id,
            build_id=build_id,
            derived=derived[build_id],
        )

    shared_a, build_specific_a = _split_manifest_core(
        derived["A"].manifest_core
    )
    shared_b, build_specific_b = _split_manifest_core(
        derived["B"].manifest_core
    )
    if shared_a != shared_b:
        raise RuntimeError(
            "independent localization evaluation shared manifest cores differ"
        )
    if derived["A"].files != derived["B"].files:
        raise RuntimeError("independent localization evaluation derivations differ")
    if validated["A"]["artifact_sha256"] != validated["B"]["artifact_sha256"]:
        raise RuntimeError("localization evaluation A/B artifacts are not byte-identical")
    completeness = shared_a.get("completeness")
    if completeness != shared_b.get("completeness"):
        raise RuntimeError("localization evaluation A/B completeness differs")

    score_certificate_path = (
        Path(localization_ab_certificate_path).resolve()
        if localization_ab_certificate_path is not None
        else release_root_path / release_id / "localization/AB_VERIFICATION.json"
    )
    freeze_bindings = _verify_score_freeze_payload_bindings(
        release_root=release_root_path,
        release_id=release_id,
        score_ab_certificate_path=score_certificate_path,
        derived=derived,
    )
    if freeze_bindings["A"] != {
        **freeze_bindings["A"],
        **build_specific_a,
    } or freeze_bindings["B"] != {
        **freeze_bindings["B"],
        **build_specific_b,
    }:
        raise RuntimeError("score-freeze build-specific fields were not exact")

    boundary_after = _attest_repository_boundary(
        producer_repo=producer_repo, verifier_repo=verifier_repo,
        recorded_producer_snapshot=recorded_snapshot,
    )
    if boundary_after != boundary_before:
        raise RuntimeError("localization evaluation verifier sources drifted mid-run")

    certificate = {
        "schema_version": EVALUATION_AB_RELEASE_SCHEMA_VERSION,
        "release_id": release_id,
        "status": "PASS",
        "scientific_full": True,
        "bootstrap_draws_executed_per_execution": DEFAULT_BOOTSTRAP_DRAWS,
        "score_ab_certificate_sha256": shared_a["score_ab_certificate_sha256"],
        "score_ab_certificate_file_sha256": shared_a[
            "score_ab_certificate_file_sha256"
        ],
        "postfreeze_amendment": shared_a["postfreeze_amendment"],
        "score_verifier_repo_snapshot": shared_a[
            "score_verifier_repo_snapshot"
        ],
        "evaluation_source_snapshot": boundary_before[
            "evaluation_producer_snapshot"
        ],
        "evaluation_ab_verifier_source_snapshot": boundary_before[
            "evaluation_ab_verifier_source_snapshot"
        ],
        "build_specific_manifest_core_fields": sorted(
            BUILD_SPECIFIC_MANIFEST_CORE_FIELDS
        ),
        "completeness": completeness,
        "artifact_sha256": validated["A"]["artifact_sha256"],
        "builds": {
            build_id: {
                "manifest_file_sha256": validated[build_id][
                    "manifest_file_sha256"
                ],
                "tree_sha256": validated[build_id]["tree_sha256"],
                **freeze_bindings[build_id],
            }
            for build_id in ("A", "B")
        },
    }
    certificate["certificate_sha256"] = payload_sha256(certificate)
    target = (
        Path(output_path)
        if output_path is not None
        else release_root_path / release_id
        / "localization/EVALUATION_AB_VERIFICATION.json"
    )
    _write_immutable_certificate(target, _json_bytes(certificate))
    return certificate


__all__ = [
    "BUILD_SPECIFIC_MANIFEST_CORE_FIELDS",
    "EVALUATION_AB_RELEASE_SCHEMA_VERSION",
    "EXPECTED_EVALUATION_PRODUCER_GIT_HEAD",
    "EXPECTED_EVALUATION_PRODUCER_SNAPSHOT_SHA256",
    "VERIFIER_SOURCE_FILES",
    "verify_localization_evaluation_ab_release",
]
