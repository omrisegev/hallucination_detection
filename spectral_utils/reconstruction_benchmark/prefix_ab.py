"""Independent A/B preparation and score verification for the prefix lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_npz_no_pickle, sha256_file
from .prefix_contract import (
    METHOD_IDS,
    PREPARATION_AB_SCHEMA,
    SCORE_AB_SCHEMA,
    PrefixContractError,
    add_payload_sha256,
    load_registry,
    payload_sha256,
    validate_observation_arrays,
    verify_payload,
    write_json_noreplace,
)
from .prefix_fit import (
    SCORE_SOURCE_FILES,
    SCORES_FILENAME,
    SCORE_MANIFEST_FILENAME,
    load_score_manifest,
    recompute_prefix_scores,
)
from .prefix_preparation import (
    EXPECTED_SCORE_FILENAME,
    FIT_INPUT_FILENAME,
    PREPARATION_MANIFEST_FILENAME,
    PRIVATE_LABEL_FILENAME,
    load_fit_input,
    load_preparation_manifest,
    load_private_labels,
    reconstruct_prefix_preparation,
)


def _load_certificate(path: Path, *, schema: str, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    verify_payload(value, name=name)
    if value.get("schema_version") != schema or value.get("status") != "PASS":
        raise PrefixContractError(f"{name} is invalid")
    return value


def _derive_prefix_preparation_ab_certificate(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    source_root: str | Path,
    require_scientific_full: bool,
    verify_private_artifacts: bool,
) -> dict[str, Any]:
    reconstruction = reconstruct_prefix_preparation(
        repo=repo,
        registry_path=registry_path,
        source_root=source_root,
    )
    registry = reconstruction["registry"]
    lane_root = Path(release_root) / release_id / "prefix"
    manifests = {}
    labels = {}
    fit_inputs = {}
    manifest_hashes = {}
    for build_id in ("A", "B"):
        root = lane_root / build_id
        manifest_path = root / PREPARATION_MANIFEST_FILENAME
        manifest = load_preparation_manifest(manifest_path)
        if manifest.get("release_id") != release_id or manifest.get("build_id") != build_id:
            raise PrefixContractError(f"prefix preparation {build_id} release/build binding failed")
        if require_scientific_full and manifest.get("scientific_full_build") is not True:
            raise PrefixContractError(f"prefix preparation {build_id} is not scientific-full")
        if (
            manifest.get("lane_id") != registry["lane_id"]
            or manifest.get("task_id") != registry["task_id"]
            or manifest.get("population_id") != registry["population"]["population_id"]
            or manifest.get("claim_boundary") != registry["claim_boundary"]
            or manifest.get("execution_modes")
            != {
                row["method_id"]: row["execution_mode"]
                for row in registry["method_roster"]
            }
            or manifest.get("fit_input", {}).get("path")
            != f"inputs/{FIT_INPUT_FILENAME}"
            or manifest.get("expected_scores", {}).get("path")
            != f"inputs/{EXPECTED_SCORE_FILENAME}"
        ):
            raise PrefixContractError(f"prefix preparation {build_id} registry/roster binding failed")
        fit_path = root / manifest["fit_input"]["path"]
        expected_path = root / manifest["expected_scores"]["path"]
        private_path = Path(private_root) / release_id / "prefix" / build_id / PRIVATE_LABEL_FILENAME
        if (
            sha256_file(fit_path) != manifest["fit_input"]["sha256"]
            or sha256_file(expected_path) != manifest["expected_scores"]["sha256"]
            or fit_path.read_bytes() != reconstruction["fit_input_bytes"]
            or expected_path.read_bytes() != reconstruction["expected_scores_bytes"]
        ):
            raise PrefixContractError(
                f"prefix preparation {build_id} differs from registered-source reconstruction"
            )
        if verify_private_artifacts and (
            sha256_file(private_path) != manifest["private_labels"]["sha256"]
            or private_path.read_bytes() != reconstruction["private_labels_bytes"]
        ):
            raise PrefixContractError(
                f"prefix preparation {build_id} private labels differ from "
                "registered-source reconstruction"
            )
        if (
            int(manifest["fit_input"].get("size_bytes", -1)) != fit_path.stat().st_size
            or Path(manifest["private_labels"].get("path", "")).resolve()
            != private_path.resolve()
            or manifest["fit_input"].get("sha256")
            != reconstruction["fit_input_sha256"]
            or manifest["expected_scores"].get("sha256")
            != reconstruction["expected_scores_sha256"]
            or manifest["private_labels"].get("sha256")
            != reconstruction["private_labels_sha256"]
            or manifest["source_binding"] != reconstruction["source_binding"]
            or manifest["source_binding_sha256"]
            != payload_sha256(manifest["source_binding"])
        ):
            raise PrefixContractError(f"prefix preparation {build_id} provenance binding failed")
        fit_input = load_fit_input(fit_path, registry=registry)
        if (
            payload_sha256(fit_input["model_audit"])
            != reconstruction["model_audit_sha256"]
            or manifest["fit_model_audit_sha256"]
            != reconstruction["model_audit_sha256"]
            or fit_input["claim_boundary"] != manifest["claim_boundary"]
        ):
            raise PrefixContractError(f"prefix preparation {build_id} fit-input audit drifted")
        expected = load_npz_no_pickle(expected_path)
        validate_observation_arrays(expected, registry=registry, include_scores=True)
        label_bundle = (
            load_private_labels(private_path, registry=registry)
            if verify_private_artifacts
            else reconstruction["private_labels"]
        )
        label_by_id = {row["row_id"]: row for row in label_bundle["rows"]}
        observed_ids = np.asarray(expected["row_id"]).astype(str)
        observed_families = np.asarray(expected["family"]).astype(str)
        observed_budgets = np.asarray(expected["budget"], dtype=int)
        if set(observed_ids) != set(label_by_id):
            raise PrefixContractError(f"prefix preparation {build_id} score/label trace union drifted")
        for row_id, family, budget in zip(
            observed_ids, observed_families, observed_budgets, strict=True
        ):
            label_row = label_by_id[row_id]
            if label_row["family"] != family or int(label_row["final_length"]) <= int(budget):
                raise PrefixContractError(
                    f"prefix preparation {build_id} score/label eligibility drifted"
                )
        manifests[build_id] = manifest
        labels[build_id] = label_bundle
        fit_inputs[build_id] = fit_input
        manifest_hashes[build_id] = sha256_file(manifest_path)
    left, right = manifests["A"], manifests["B"]
    comparisons = {
        "source_binding_sha256": left["source_binding_sha256"] == right["source_binding_sha256"],
        "fit_input_sha256": left["fit_input"]["sha256"] == right["fit_input"]["sha256"],
        "expected_score_sha256": left["expected_scores"]["sha256"] == right["expected_scores"]["sha256"],
        "private_label_sha256": left["private_labels"]["sha256"] == right["private_labels"]["sha256"],
        "model_audit_sha256": left["fit_model_audit_sha256"] == right["fit_model_audit_sha256"],
        "execution_modes": left["execution_modes"] == right["execution_modes"],
        "claim_boundary": left["claim_boundary"] == right["claim_boundary"],
        "private_label_payload": labels["A"]["payload_sha256"] == labels["B"]["payload_sha256"],
        "fit_model_audit_payload": payload_sha256(fit_inputs["A"]["model_audit"])
        == payload_sha256(fit_inputs["B"]["model_audit"]),
    }
    if not all(comparisons.values()):
        raise PrefixContractError(
            f"prefix A/B preparation is not byte/contract identical: "
            f"{[name for name, ok in comparisons.items() if not ok]}"
        )
    certificate = add_payload_sha256(
        {
            "schema_version": PREPARATION_AB_SCHEMA,
            "release_id": release_id,
            "status": "PASS",
            "scientific_full_required": bool(require_scientific_full),
            "comparison": comparisons,
            "core_input_sha256": left["fit_input"]["sha256"],
            "expected_score_anchor_sha256": left["expected_scores"]["sha256"],
            "private_label_sha256": left["private_labels"]["sha256"],
            "source_binding_sha256": left["source_binding_sha256"],
            "source_asset_roster_sha256": reconstruction["source_binding"][
                "asset_roster_sha256"
            ],
            "observations": int(left["expected_scores"]["observations"]),
            "evaluation_traces": int(left["private_labels"]["rows"]),
            "labels_exposed_to_fit": False,
            "historical_scores_are_execution_substitute": False,
            "independent_source_reconstruction": True,
            "builds": {
                build_id: {
                    "preparation_manifest_sha256": manifest_hashes[build_id],
                    "preparation_manifest_payload_sha256": manifests[build_id]["payload_sha256"],
                }
                for build_id in ("A", "B")
            },
        }
    )
    return certificate


def verify_prefix_preparation_ab(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    source_root: str | Path,
    output_path: str | Path | None = None,
    require_scientific_full: bool,
) -> dict[str, Any]:
    certificate = _derive_prefix_preparation_ab_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=require_scientific_full,
        verify_private_artifacts=True,
    )
    lane_root = Path(release_root) / release_id / "prefix"
    output = Path(output_path) if output_path else lane_root / "PREPARATION_AB_VERIFICATION.json"
    write_json_noreplace(output, certificate)
    return certificate


def authenticate_prefix_preparation_certificate(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    source_root: str | Path,
    require_scientific_full: bool,
) -> dict[str, Any]:
    """Transitively rederive, then authenticate, the immutable prep certificate.

    The pre-label score stages deliberately do not open the published private
    label files.  Their registered-source reconstruction still derives the
    exact expected private-label hash, so a coordinated label/manifest/cert
    rewrite cannot become a new trust root.
    """

    lane_root = Path(release_root) / release_id / "prefix"
    path = lane_root / "PREPARATION_AB_VERIFICATION.json"
    observed = _load_certificate(
        path,
        schema=PREPARATION_AB_SCHEMA,
        name="prefix preparation A/B certificate",
    )
    derived = _derive_prefix_preparation_ab_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=require_scientific_full,
        verify_private_artifacts=False,
    )
    if observed != derived:
        raise PrefixContractError(
            "prefix preparation certificate is self-attested or differs from "
            "registered-source rederivation"
        )
    return observed


def _derive_prefix_score_ab_certificate(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    source_root: str | Path,
    require_scientific_full: bool,
) -> dict[str, Any]:
    repo_path = Path(repo).resolve()
    registry = load_registry(registry_path)
    lane_root = Path(release_root) / release_id / "prefix"
    preparation_certificate_path = lane_root / "PREPARATION_AB_VERIFICATION.json"
    preparation_certificate = authenticate_prefix_preparation_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=require_scientific_full,
    )
    if preparation_certificate.get("release_id") != release_id:
        raise PrefixContractError("prefix preparation certificate release binding failed")
    if (
        preparation_certificate.get("independent_source_reconstruction") is not True
        or preparation_certificate.get("labels_exposed_to_fit") is not False
        or preparation_certificate.get("historical_scores_are_execution_substitute") is not False
    ):
        raise PrefixContractError("prefix preparation certificate lacks source reconstruction")
    manifests = {}
    scores = {}
    manifest_hashes = {}
    for build_id in ("A", "B"):
        build_root = lane_root / build_id
        preparation_path = build_root / PREPARATION_MANIFEST_FILENAME
        preparation = load_preparation_manifest(preparation_path)
        prep_binding = preparation_certificate.get("builds", {}).get(build_id, {})
        if (
            preparation.get("release_id") != release_id
            or preparation.get("build_id") != build_id
            or prep_binding.get("preparation_manifest_sha256")
            != sha256_file(preparation_path)
            or prep_binding.get("preparation_manifest_payload_sha256")
            != preparation["payload_sha256"]
            or preparation["fit_input"]["sha256"]
            != preparation_certificate.get("core_input_sha256")
            or preparation["expected_scores"]["sha256"]
            != preparation_certificate.get("expected_score_anchor_sha256")
            or preparation["source_binding_sha256"]
            != preparation_certificate.get("source_binding_sha256")
            or preparation["source_binding"].get("asset_roster_sha256")
            != preparation_certificate.get("source_asset_roster_sha256")
            or preparation["private_labels"]["sha256"]
            != preparation_certificate.get("private_label_sha256")
        ):
            raise PrefixContractError(
                f"prefix score verifier lost preparation {build_id} certificate binding"
            )
        fit_input_path = build_root / preparation["fit_input"]["path"]
        expected_path = build_root / preparation["expected_scores"]["path"]
        if (
            sha256_file(fit_input_path) != preparation["fit_input"]["sha256"]
            or sha256_file(expected_path) != preparation["expected_scores"]["sha256"]
        ):
            raise PrefixContractError(f"prefix score {build_id} prepared inputs drifted")
        fit_input = load_fit_input(fit_input_path, registry=registry)
        expected = load_npz_no_pickle(expected_path)
        validate_observation_arrays(expected, registry=registry, include_scores=True)
        fit_root = build_root / "fit"
        manifest_path = fit_root / SCORE_MANIFEST_FILENAME
        manifest = load_score_manifest(manifest_path)
        if manifest.get("release_id") != release_id or manifest.get("build_id") != build_id:
            raise PrefixContractError(f"prefix score {build_id} release/build binding failed")
        if require_scientific_full and manifest.get("scientific_full_build") is not True:
            raise PrefixContractError(f"prefix score {build_id} is not scientific-full")
        if (
            manifest.get("lane_id") != registry["lane_id"]
            or manifest.get("task_id") != registry["task_id"]
            or manifest.get("claim_boundary") != registry["claim_boundary"]
            or manifest.get("preparation_manifest_sha256") != sha256_file(preparation_path)
            or manifest.get("preparation_ab_certificate_sha256")
            != sha256_file(preparation_certificate_path)
            or manifest.get("preparation_ab_certificate_payload_sha256")
            != preparation_certificate["payload_sha256"]
            or manifest.get("fit_input_sha256") != sha256_file(fit_input_path)
            or manifest.get("expected_score_anchor_sha256") != sha256_file(expected_path)
        ):
            raise PrefixContractError(f"prefix score {build_id} is not bound to preparation A/B")
        score_path = fit_root / manifest["score_artifact"]["path"]
        if sha256_file(score_path) != manifest["score_artifact"]["sha256"]:
            raise PrefixContractError(f"prefix score {build_id} artifact hash failed")
        arrays = load_npz_no_pickle(score_path)
        validate_observation_arrays(arrays, registry=registry, include_scores=True)
        recomputed, recomputation_audit = recompute_prefix_scores(
            fit_input, expected, registry
        )
        if set(recomputed) != set(arrays) or any(
            not np.array_equal(recomputed[name], arrays[name]) for name in recomputed
        ):
            raise PrefixContractError(
                f"prefix score {build_id} differs from independent CPU recomputation"
            )
        if manifest.get("recomputation_audit") != recomputation_audit:
            raise PrefixContractError(
                f"prefix score {build_id} recomputation audit is self-attested or stale"
            )
        snapshot = manifest.get("source_snapshot")
        if (
            not isinstance(snapshot, list)
            or tuple(str(row.get("path", "")) for row in snapshot) != SCORE_SOURCE_FILES
            or any(set(row) != {"path", "sha256"} for row in snapshot)
        ):
            raise PrefixContractError(f"prefix score {build_id} source snapshot roster drifted")
        for row in snapshot:
            relative = Path(str(row.get("path", "")))
            source_path = (repo_path / relative).resolve()
            try:
                source_path.relative_to(repo_path)
            except ValueError as error:
                raise PrefixContractError("prefix score source snapshot escapes repo") from error
            if relative.is_absolute() or sha256_file(source_path) != row.get("sha256"):
                raise PrefixContractError(f"prefix score source changed: {row['path']}")
        if manifest["source_snapshot_sha256"] != payload_sha256(manifest["source_snapshot"]):
            raise PrefixContractError(f"prefix score {build_id} source snapshot hash failed")
        anchors = recomputation_audit["anchors"]
        if set(anchors) != set(METHOD_IDS) or any(
            row.get("status") != registry["score_anchor"]["required_status"]
            or row.get("historical_score_rebind") is not False
            for row in anchors.values()
        ):
            raise PrefixContractError(f"prefix score {build_id} did not execute and anchor all methods")
        if (
            int(manifest["score_artifact"].get("observations", -1))
            != int(registry["population"]["expected_prefix_observations"])
            or int(manifest["score_artifact"].get("method_scores", -1))
            != int(registry["population"]["expected_prefix_observations"])
            * len(METHOD_IDS)
        ):
            raise PrefixContractError(f"prefix score {build_id} artifact counts drifted")
        manifests[build_id] = manifest
        scores[build_id] = arrays
        manifest_hashes[build_id] = sha256_file(manifest_path)
    comparison = {
        "score_artifact_sha256": manifests["A"]["score_artifact"]["sha256"]
        == manifests["B"]["score_artifact"]["sha256"],
        "fit_input_sha256": manifests["A"]["fit_input_sha256"]
        == manifests["B"]["fit_input_sha256"],
        "expected_score_anchor_sha256": manifests["A"]["expected_score_anchor_sha256"]
        == manifests["B"]["expected_score_anchor_sha256"],
        "recomputation_audit": manifests["A"]["recomputation_audit"]
        == manifests["B"]["recomputation_audit"],
        "source_snapshot": manifests["A"]["source_snapshot_sha256"]
        == manifests["B"]["source_snapshot_sha256"],
    }
    for name in scores["A"]:
        comparison[f"array::{name}"] = bool(np.array_equal(scores["A"][name], scores["B"][name]))
    if not all(comparison.values()):
        raise PrefixContractError(
            f"prefix A/B score verification failed: "
            f"{[name for name, ok in comparison.items() if not ok]}"
        )
    certificate = add_payload_sha256(
        {
            "schema_version": SCORE_AB_SCHEMA,
            "release_id": release_id,
            "status": "PASS",
            "scientific_full_required": bool(require_scientific_full),
            "preparation_certificate_sha256": sha256_file(preparation_certificate_path),
            "score_artifact_sha256": manifests["A"]["score_artifact"]["sha256"],
            "source_snapshot_sha256": manifests["A"]["source_snapshot_sha256"],
            "source_asset_roster_sha256": preparation_certificate[
                "source_asset_roster_sha256"
            ],
            "observations": int(manifests["A"]["score_artifact"]["observations"]),
            "method_scores": int(manifests["A"]["score_artifact"]["method_scores"]),
            "comparison": comparison,
            "execution_status": registry["score_anchor"]["required_status"],
            "causal_early_scoring_only": True,
            "stopping_claim_allowed": False,
            "cross_task_macro_allowed": False,
            "builds": {
                build_id: {
                    "score_manifest_sha256": manifest_hashes[build_id],
                    "score_manifest_payload_sha256": manifests[build_id]["payload_sha256"],
                    "preparation_manifest_sha256": manifests[build_id][
                        "preparation_manifest_sha256"
                    ],
                }
                for build_id in ("A", "B")
            },
        }
    )
    return certificate


def verify_prefix_score_ab(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    source_root: str | Path,
    output_path: str | Path | None = None,
    require_scientific_full: bool,
) -> dict[str, Any]:
    certificate = _derive_prefix_score_ab_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=require_scientific_full,
    )
    lane_root = Path(release_root) / release_id / "prefix"
    output = Path(output_path) if output_path else lane_root / "SCORE_AB_VERIFICATION.json"
    write_json_noreplace(output, certificate)
    return certificate


def authenticate_prefix_score_certificate(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    source_root: str | Path,
    require_scientific_full: bool,
) -> dict[str, Any]:
    """Transitively rederive prep and scores before trusting the score cert."""

    lane_root = Path(release_root) / release_id / "prefix"
    path = lane_root / "SCORE_AB_VERIFICATION.json"
    observed = _load_certificate(
        path,
        schema=SCORE_AB_SCHEMA,
        name="prefix score A/B certificate",
    )
    derived = _derive_prefix_score_ab_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=require_scientific_full,
    )
    if observed != derived:
        raise PrefixContractError(
            "prefix score certificate is self-attested or differs from "
            "transitive registered-source rederivation"
        )
    return observed


__all__ = [
    "authenticate_prefix_preparation_certificate",
    "authenticate_prefix_score_certificate",
    "verify_prefix_preparation_ab",
    "verify_prefix_score_ab",
]
