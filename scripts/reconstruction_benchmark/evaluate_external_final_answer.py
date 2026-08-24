#!/usr/bin/env python3
"""Open external labels only after the label-free score tree is frozen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    ID_CONTRACT_VERSION,
    assert_score_freeze,
    external_id_contract_binding,
    fit_safe_external_cell_record,
    load_identity_key,
    load_external_registry,
    load_labels_after_score_freeze,
    load_prepared_external_cell,
    write_label_vector,
)
from spectral_utils.reconstruction_benchmark.external_ab import (  # noqa: E402
    assert_fit_safe_matches_preparation,
    assert_external_ab_certificate,
    validate_fit_safe_input_manifest,
    validate_scientific_input_manifest,
    validate_scientific_score_freeze,
)
from spectral_utils.reconstruction_benchmark.external_evaluation import (  # noqa: E402
    grouped_paired_bootstrap,
    population_grouped_paired_bootstrap,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"
EVALUATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/feature_contract.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "scripts/reconstruction_benchmark/evaluate_external_final_answer.py",
    "spectral_utils/fair_comparisons/stopping.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/external_ab.py",
    "spectral_utils/reconstruction_benchmark/external_evaluation.py",
    "spectral_utils/reconstruction_benchmark/io.py",
)


def _verify_payload(value: dict, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(payload)):
        raise RuntimeError(f"{name} payload hash failed")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "comparison_group_id", "panel_role", "population_id", "cell_id",
        "dataset_id", "model_id", "slice_id", "method_id", "metric_id",
        "value", "ci_low", "ci_high", "status", "n", "n_incorrect",
        "n_correct", "bootstrap_unit", "bootstrap_draws", "bootstrap_valid_draws",
        "cohort_id", "score_sha256", "label_sha256",
        "record_level", "aggregate_weighting", "aggregate_interpretation",
        "linked_resampling", "stratified_by_label", "n_cells", "n_groups",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_contrasts_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "comparison_group_id", "panel_role", "population_id", "cell_id",
        "dataset_id", "model_id", "slice_id", "method_id", "reference_method_id",
        "metric_id", "delta", "ci_low", "ci_high", "probability_delta_le_zero",
        "higher_is_better", "bootstrap_unit", "bootstrap_draws",
        "bootstrap_valid_draws", "n", "n_groups", "cohort_id",
        "record_level", "aggregate_weighting", "aggregate_interpretation",
        "linked_resampling", "stratified_by_label", "n_cells", "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _restore_validated_score_freeze(
    validated_freeze: Mapping[str, Any],
    *,
    fit_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove exactly the verifier's known derived manifest augmentation.

    The scientific verifier validates the signed on-disk freeze, then attaches
    its already-validated fit manifest for controller callers.  The label gate
    must receive the original signed payload.  Accepting any other derived key,
    or an attachment that differs from the independently validated manifest,
    fails closed; all remaining fields are still checked by ``assert_score_freeze``.
    """

    derived_keys = {
        str(key) for key in validated_freeze if str(key).startswith("_")
    }
    if derived_keys != {"_validated_fit_manifest"}:
        raise RuntimeError(
            "scientific score verifier returned an unexpected derived-field roster"
        )
    embedded = validated_freeze.get("_validated_fit_manifest")
    if not isinstance(embedded, Mapping) or dict(embedded) != dict(fit_manifest):
        raise RuntimeError(
            "scientific score verifier attached another fit manifest"
        )
    restored = dict(validated_freeze)
    restored.pop("_validated_fit_manifest")
    return restored


def _remove_verified_empty_directory_tree(path: Path) -> None:
    """Remove only a verified file-free retry scaffold using ``rmdir``."""

    if path.is_symlink():
        raise FileExistsError(f"external evaluation target is not a directory: {path}")
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(f"external evaluation target is not a directory: {path}")
    descendants = sorted(
        path.rglob("*"), key=lambda item: len(item.parts), reverse=True,
    )
    unexpected = [
        item for item in descendants if item.is_symlink() or not item.is_dir()
    ]
    if unexpected:
        raise FileExistsError(
            "external evaluation directory contains material output: "
            + ", ".join(str(item) for item in unexpected[:5])
        )
    for directory in descendants:
        directory.rmdir()
    path.rmdir()


class _AtomicEvaluationStage:
    """Same-filesystem staging directory committed by one atomic rename."""

    def __init__(self, final_path: Path) -> None:
        self.final_path = final_path
        self.path = Path(tempfile.mkdtemp(
            prefix=f".{final_path.name}.staging-",
            dir=final_path.parent,
        ))
        self.committed = False

    def commit(self) -> None:
        if self.committed:
            raise RuntimeError("external evaluation stage was already committed")
        if self.final_path.exists() or self.final_path.is_symlink():
            raise FileExistsError(
                f"external evaluation directory already exists: {self.final_path}"
            )
        os.replace(self.path, self.final_path)
        self.committed = True

    def cleanup(self) -> None:
        if not self.committed and self.path.exists():
            shutil.rmtree(self.path)


_ACTIVE_EVALUATION_STAGE: _AtomicEvaluationStage | None = None


def _main() -> None:
    global _ACTIVE_EVALUATION_STAGE

    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    parser.add_argument(
        "--identity-key", type=Path,
        help="Controller-only sealed release key; defaults outside releases/.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument(
        "--ab-certificate", type=Path,
        help="Passing exact A/B certificate; defaults to the release-level external certificate.",
    )
    parser.add_argument(
        "--source-root", type=Path, default=REPO,
        help="Root containing the same hash-frozen telemetry and label paths used for preparation.",
    )
    args = parser.parse_args()

    registry = load_external_registry(
        repo=REPO,
        registry_path=args.registry,
        population_registry_path=args.populations,
    )
    root = args.release_root / args.release_id / f"build_{args.build}" / "external_final_answer"
    input_root, fit_root = root / "inputs", root / "fit"
    final_evaluation_root = root / "evaluation"
    controller_root = (
        args.release_root.parent / "private_control" / args.release_id
        / "external_final_answer"
    )
    identity_key = load_identity_key(
        args.identity_key or (controller_root / "external-id-v2.key")
    )
    certificate_path = args.ab_certificate or (
        args.release_root / args.release_id / "external_final_answer" / "AB_VERIFICATION.json"
    )
    certificate = assert_external_ab_certificate(
        certificate_path,
        release_id=args.release_id,
        release_root=args.release_root,
        selected_build=args.build,
        registry=registry,
        repo=REPO,
    )
    fit_manifest = validate_fit_safe_input_manifest(
        input_root / "MANIFEST.json",
        repo=REPO,
        input_root=input_root,
    )
    preparation_manifest_path = (
        controller_root / f"build_{args.build}" / "preparation_provenance"
        / "MANIFEST.json"
    )
    input_manifest = validate_scientific_input_manifest(
        preparation_manifest_path,
        registry=registry,
        repo=REPO,
        input_root=input_root,
    )
    assert_fit_safe_matches_preparation(
        fit_manifest,
        input_manifest,
        preparation_manifest_path=preparation_manifest_path,
    )
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    validated_freeze = validate_scientific_score_freeze(
        freeze_path,
        registry=registry,
        repo=REPO,
        input_root=input_root,
        fit_root=fit_root,
        input_manifest=fit_manifest,
    )
    freeze = _restore_validated_score_freeze(
        validated_freeze, fit_manifest=fit_manifest,
    )
    # Recheck the exact restored signed payload before creating output or
    # allowing any label adapter to run.  No hash fields are ignored here.
    assert_score_freeze(
        freeze, registry=registry, identity_key=identity_key,
    )
    if certificate.get("cell_ids") != freeze.get("cell_ids"):
        raise RuntimeError("A/B certificate and selected scientific freeze have different cell rosters")
    certified_build = certificate["builds"][args.build]
    if certified_build.get("score_freeze_payload_sha256") != freeze.get("payload_sha256"):
        raise RuntimeError("A/B certificate and selected score-freeze payload disagree")
    if certified_build.get("input_manifest_payload_sha256") != fit_manifest.get("payload_sha256"):
        raise RuntimeError("A/B certificate and selected input-manifest payload disagree")
    if args.bootstrap_draws <= 0:
        raise ValueError("--bootstrap-draws must be positive")
    if args.bootstrap_draws != 20_000:
        raise RuntimeError("scientific evaluation requires exactly 20,000 grouped paired draws")
    if freeze.get("input_manifest_sha256") != sha256_file(input_root / "MANIFEST.json"):
        raise RuntimeError("score freeze/input manifest binding failed")
    evaluation_source_snapshot = {
        "files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in EVALUATION_SOURCE_FILES
        ]
    }
    evaluation_source_snapshot["snapshot_sha256"] = sha256_bytes(
        canonical_json_bytes(evaluation_source_snapshot)
    )

    prepared = {
        item["cell_id"]: item for item in input_manifest["cells"]
        if item.get("status") == "ELIGIBLE"
    }
    fit_records: dict[str, list[dict]] = {}
    for record in freeze["records"]:
        fit_records.setdefault(record["cell_id"], []).append(record)
    if set(fit_records) != set(freeze["cell_ids"]) or set(fit_records) != set(prepared):
        raise RuntimeError("score-freeze and prepared eligible cell rosters differ")

    # All certificate, manifest, freeze, source, and roster checks above are
    # complete before touching the final output path.  A previous failed
    # preflight may have left directory-only scaffolding; remove only that
    # verified-empty tree.  Material output always blocks overwrite.
    _remove_verified_empty_directory_tree(final_evaluation_root)
    stage = _AtomicEvaluationStage(final_evaluation_root)
    _ACTIVE_EVALUATION_STAGE = stage
    evaluation_root = stage.path
    label_root = evaluation_root / "labels"
    label_root.mkdir()

    metrics: list[dict] = []
    contrasts: list[dict] = []
    label_records: list[dict] = []
    evaluation_cells: dict[str, dict] = {}
    for cell_id in freeze["cell_ids"]:
        spec = registry.by_cell[cell_id]
        input_record = prepared[cell_id]
        fit_input_record = fit_safe_external_cell_record(input_record)
        prepared_cell = load_prepared_external_cell(
            artifact_path=input_root / input_record["artifact_path"],
            record=fit_input_record,
            identity_contract=fit_manifest["identity_contract"],
        )
        labels = load_labels_after_score_freeze(
            registry=registry,
            spec=spec,
            repo=args.source_root,
            score_freeze=freeze,
            expected_row_ids=prepared_cell.row_ids,
            expected_group_roster_commitment_sha256=input_record[
                "sealed_group_roster_commitment_sha256"
            ],
            identity_key=identity_key,
        )
        group_ids = labels.group_ids
        label_path = label_root / f"{cell_id}.npz"
        label_record = write_label_vector(label_path, labels)
        label_records.append({**label_record, "artifact_path": label_path.relative_to(evaluation_root).as_posix()})
        label_sha = str(labels.provenance["row_label_sha256"])
        cohort_id = str(input_record["sealed_group_roster_commitment_sha256"])
        scores_by_method: dict[str, np.ndarray] = {}
        record_by_method: dict[str, dict] = {}
        for record in sorted(fit_records[cell_id], key=lambda item: item["method_id"]):
            if record["status"] not in {"OK", "OK_FALLBACK"}:
                raise RuntimeError(f"frozen record is not successful: {cell_id}/{record['method_id']}")
            score_path = fit_root / record["score_path"]
            record_path = fit_root / record["record_path"]
            if sha256_file(record_path) != record["record_sha256"]:
                raise RuntimeError(f"frozen method record hash mismatch: {record_path}")
            if sha256_file(score_path) != record["score_sha256"]:
                raise RuntimeError(f"frozen score hash mismatch: {score_path}")
            bundle = load_npz_no_pickle(score_path)
            if set(bundle) != {
                "row_ids", "score", "id_contract_version", "id_contract_sha256",
                "identity_key_id", "row_namespace_sha256", "row_roster_sha256",
            }:
                raise RuntimeError(f"unexpected score arrays: {score_path}")
            score_identity = {}
            for key in (
                "id_contract_version", "id_contract_sha256",
                "identity_key_id", "row_namespace_sha256", "row_roster_sha256",
            ):
                scalar = bundle[key]
                if scalar.shape != (1,):
                    raise RuntimeError(f"score identity member is not scalar: {score_path}/{key}")
                score_identity[key] = str(scalar.tolist()[0])
            expected_score_identity = {
                "id_contract_version": ID_CONTRACT_VERSION,
                "id_contract_sha256": fit_input_record["id_contract_sha256"],
                "identity_key_id": fit_input_record["identity_key_id"],
                "row_namespace_sha256": fit_input_record["row_namespace_sha256"],
                "row_roster_sha256": fit_input_record["row_roster_sha256"],
            }
            if score_identity != expected_score_identity:
                raise RuntimeError(
                    f"score identity contract mismatch: {cell_id}/{record['method_id']}"
                )
            row_ids = tuple(map(str, bundle["row_ids"].tolist()))
            if row_ids != labels.row_ids:
                raise RuntimeError(f"score/label row order mismatch: {cell_id}/{record['method_id']}")
            score = np.asarray(bundle["score"], dtype=float)
            if score.shape != labels.incorrect.shape or not np.isfinite(score).all():
                raise RuntimeError(f"invalid frozen score: {cell_id}/{record['method_id']}")
            scores_by_method[record["method_id"]] = score
            record_by_method[record["method_id"]] = record
        reference_method = "iu_pcr" if "iu_pcr" in scores_by_method else sorted(scores_by_method)[0]
        aggregate_rule = registry.raw["population_aggregates"][spec.population_id]
        stratified = aggregate_rule.get("bootstrap") == "source_group_stratified_by_label"
        seed_payload = f"{registry.sha256}:{cell_id}:grouped-paired-bootstrap-v1".encode("utf-8")
        seed = int(aggregate_rule.get("seed", int(sha256_bytes(seed_payload)[:8], 16)))
        interval = grouped_paired_bootstrap(
            labels=labels.incorrect,
            scores_by_method=scores_by_method,
            group_ids=group_ids,
            reference_method=reference_method,
            draws=args.bootstrap_draws,
            seed=seed,
            stratify_by_label=stratified,
        )
        evaluation_cells[cell_id] = {
            "labels": np.asarray(labels.incorrect, dtype=np.int8),
            "group_ids": tuple(map(str, group_ids)),
            "scores_by_method": scores_by_method,
            "score_hashes": {
                method_id: record_by_method[method_id]["score_sha256"]
                for method_id in sorted(record_by_method)
            },
            "method_statuses": {
                method_id: record_by_method[method_id]["status"]
                for method_id in sorted(record_by_method)
            },
            "label_sha256": label_sha,
            "cohort_id": cohort_id,
        }
        for method_id in sorted(scores_by_method):
            record = record_by_method[method_id]
            base = {
                "comparison_group_id": spec.comparison_group_id,
                "panel_role": spec.panel_role,
                "population_id": spec.population_id,
                "cell_id": spec.cell_id,
                "dataset_id": spec.dataset_id,
                "model_id": spec.model_id,
                "slice_id": spec.slice_id,
                "method_id": method_id,
                "n": len(labels.incorrect),
                "n_incorrect": int(labels.incorrect.sum()),
                "n_correct": int(len(labels.incorrect) - labels.incorrect.sum()),
                "cohort_id": cohort_id,
                "score_sha256": record["score_sha256"],
                "label_sha256": label_sha,
                "record_level": "cell",
                "stratified_by_label": stratified,
                "n_cells": 1,
                "n_groups": interval["n_groups"],
            }
            for metric_id, result in interval["metrics"][method_id].items():
                exact_group = "external_final_answer::cell::" + sha256_bytes(canonical_json_bytes({
                    "cell_id": spec.cell_id,
                    "cohort_id": cohort_id,
                    "metric_id": metric_id,
                    "positive_class": "incorrect",
                }))[:24]
                metrics.append({
                    **base,
                    "comparison_group_id": exact_group,
                    "metric_id": metric_id,
                    "value": result["value"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "status": record["status"],
                    "bootstrap_unit": interval["bootstrap_unit"],
                    "bootstrap_draws": interval["draws_requested"],
                    "bootstrap_valid_draws": result["valid_draws"],
                })
        common = {
            "comparison_group_id": spec.comparison_group_id,
            "panel_role": spec.panel_role,
            "population_id": spec.population_id,
            "cell_id": spec.cell_id,
            "dataset_id": spec.dataset_id,
            "model_id": spec.model_id,
            "slice_id": spec.slice_id,
            "n": len(labels.incorrect),
            "n_groups": interval["n_groups"],
            "cohort_id": cohort_id,
            "bootstrap_unit": interval["bootstrap_unit"],
            "bootstrap_draws": interval["draws_requested"],
            "record_level": "cell",
            "stratified_by_label": stratified,
            "n_cells": 1,
        }
        for method_id, method_contrasts in interval["contrasts"].items():
            for metric_id, result in method_contrasts.items():
                exact_group = "external_final_answer::cell::" + sha256_bytes(canonical_json_bytes({
                    "cell_id": spec.cell_id,
                    "cohort_id": cohort_id,
                    "metric_id": metric_id,
                    "positive_class": "incorrect",
                }))[:24]
                contrast_status = (
                    "OK_FALLBACK"
                    if "OK_FALLBACK" in {
                        record_by_method[method_id]["status"],
                        record_by_method[result["reference_method"]]["status"],
                    }
                    else "OK"
                )
                contrasts.append({
                    **common,
                    "comparison_group_id": exact_group,
                    "method_id": method_id,
                    "reference_method_id": result["reference_method"],
                    "metric_id": metric_id,
                    "delta": result["delta"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "probability_delta_le_zero": result["probability_delta_le_zero"],
                    "higher_is_better": result["higher_is_better"],
                    "bootstrap_valid_draws": result["valid_draws"],
                    "status": contrast_status,
                })

    # Population estimates are registry-driven equal-cell macros.  Missing or
    # incompatible cells keep the population visibly unaggregated; no method
    # gets a convenient smaller macro.
    application_statuses = {item["cell_id"]: item["status"] for item in input_manifest["cells"]}
    population_checks = []
    for population_id, expected in registry.raw.get("population_expectations", {}).items():
        population_cells = [item for item in registry.cells if item.population_id == population_id]
        statuses = {item.cell_id: application_statuses.get(item.cell_id, "MISSING") for item in population_cells}
        if not population_cells or not all(value == "ELIGIBLE" for value in statuses.values()):
            population_checks.append({
                "population_id": population_id,
                "status": "NOT_AGGREGATED_INCOMPLETE_OR_INAPPLICABLE",
                "cell_statuses": statuses,
            })
            continue
        selected = [row for row in label_records if registry.by_cell[row["cell_id"]].population_id == population_id]
        arrays = [load_npz_no_pickle(evaluation_root / row["artifact_path"])["incorrect"] for row in selected]
        labels = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.int8)
        observed = {
            "rows": int(len(labels)),
            "incorrect": int(labels.sum()),
            "correct": int(len(labels) - labels.sum()),
            "cells": len(selected),
        }
        expected_core = {key: int(expected[key]) for key in observed}
        if observed != expected_core:
            raise RuntimeError(f"population label totals failed for {population_id}: {observed} != {expected_core}")
        aggregate_rule = registry.raw["population_aggregates"][population_id]
        if aggregate_rule.get("enabled") is not True:
            population_checks.append({
                "population_id": population_id,
                "status": "AGGREGATE_DISABLED_BY_REGISTRY",
                "reason": aggregate_rule.get("reason"),
                "observed": observed,
            })
            continue
        cell_ids = [item.cell_id for item in population_cells]
        cell_payload = {
            cell_id: {
                "labels": evaluation_cells[cell_id]["labels"],
                "group_ids": evaluation_cells[cell_id]["group_ids"],
                "scores_by_method": evaluation_cells[cell_id]["scores_by_method"],
            }
            for cell_id in cell_ids
        }
        link_rule = aggregate_rule["link_cells_by"]
        if link_rule == "none":
            link_keys = None
        elif link_rule == "slice_id":
            link_keys = {cell_id: registry.by_cell[cell_id].slice_id for cell_id in cell_ids}
        elif link_rule == "all":
            link_keys = {cell_id: "all_registered_cells" for cell_id in cell_ids}
        else:  # load_external_registry already rejects this; keep the label gate fail-closed.
            raise RuntimeError(f"{population_id}: unknown linkage rule {link_rule!r}")
        stratified = aggregate_rule.get("bootstrap") == "source_group_stratified_by_label"
        seed_payload = f"{registry.sha256}:{population_id}:population-grouped-paired-bootstrap-v1".encode("utf-8")
        seed = int(aggregate_rule.get("seed", int(sha256_bytes(seed_payload)[:8], 16)))
        population_interval = population_grouped_paired_bootstrap(
            cells=cell_payload,
            link_keys=link_keys,
            reference_method="iu_pcr",
            draws=args.bootstrap_draws,
            seed=seed,
            weighting=str(aggregate_rule["weighting"]),
            stratify_by_label=stratified,
        )
        specs = [registry.by_cell[cell_id] for cell_id in cell_ids]
        datasets = sorted({item.dataset_id for item in specs})
        models = sorted({item.model_id for item in specs})
        slices = sorted({item.slice_id for item in specs})
        cohort_id = sha256_bytes(canonical_json_bytes([
            {"cell_id": cell_id, "cohort_id": evaluation_cells[cell_id]["cohort_id"]}
            for cell_id in cell_ids
        ]))
        label_sha = sha256_bytes(canonical_json_bytes([
            {"cell_id": cell_id, "label_sha256": evaluation_cells[cell_id]["label_sha256"]}
            for cell_id in cell_ids
        ]))
        common = {
            "comparison_group_id": f"population::{population_id}",
            "panel_role": specs[0].panel_role if len({item.panel_role for item in specs}) == 1 else "mixed",
            "population_id": population_id,
            "cell_id": "__population__",
            "dataset_id": datasets[0] if len(datasets) == 1 else "__multiple__",
            "model_id": models[0] if len(models) == 1 else "__multiple__",
            "slice_id": f"population::{aggregate_rule['interpretation']}",
            "n": population_interval["n_rows"],
            "n_groups": population_interval["n_resampling_groups"],
            "n_incorrect": observed["incorrect"],
            "n_correct": observed["correct"],
            "cohort_id": cohort_id,
            "label_sha256": label_sha,
            "bootstrap_unit": population_interval["bootstrap_unit"],
            "bootstrap_draws": population_interval["draws_requested"],
            "record_level": "population",
            "aggregate_weighting": aggregate_rule["weighting"],
            "aggregate_interpretation": aggregate_rule["interpretation"],
            "linked_resampling": population_interval["linked_resampling"],
            "stratified_by_label": population_interval["stratified_by_group_label"],
            "n_cells": population_interval["n_cells"],
        }
        for method_id in sorted(population_interval["metrics"]):
            aggregate_method_status = (
                "OK_FALLBACK"
                if any(
                    evaluation_cells[cell_id]["method_statuses"][method_id] == "OK_FALLBACK"
                    for cell_id in cell_ids
                )
                else "OK"
            )
            score_sha = sha256_bytes(canonical_json_bytes([
                {
                    "cell_id": cell_id,
                    "score_sha256": evaluation_cells[cell_id]["score_hashes"][method_id],
                }
                for cell_id in cell_ids
            ]))
            for metric_id, result in population_interval["metrics"][method_id].items():
                exact_group = "external_final_answer::population::" + sha256_bytes(canonical_json_bytes({
                    "population_id": population_id,
                    "cohort_id": cohort_id,
                    "metric_id": metric_id,
                    "positive_class": "incorrect",
                    "weighting": aggregate_rule["weighting"],
                    "interpretation": aggregate_rule["interpretation"],
                }))[:24]
                metrics.append({
                    **common,
                    "comparison_group_id": exact_group,
                    "method_id": method_id,
                    "metric_id": metric_id,
                    "value": result["value"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "status": aggregate_method_status,
                    "bootstrap_valid_draws": result["valid_draws"],
                    "score_sha256": score_sha,
                })
        for method_id, method_contrasts in population_interval["contrasts"].items():
            for metric_id, result in method_contrasts.items():
                exact_group = "external_final_answer::population::" + sha256_bytes(canonical_json_bytes({
                    "population_id": population_id,
                    "cohort_id": cohort_id,
                    "metric_id": metric_id,
                    "positive_class": "incorrect",
                    "weighting": aggregate_rule["weighting"],
                    "interpretation": aggregate_rule["interpretation"],
                }))[:24]
                candidate_fallback = any(
                    evaluation_cells[cell_id]["method_statuses"][method_id] == "OK_FALLBACK"
                    for cell_id in cell_ids
                )
                reference_fallback = any(
                    evaluation_cells[cell_id]["method_statuses"][result["reference_method"]] == "OK_FALLBACK"
                    for cell_id in cell_ids
                )
                contrasts.append({
                    **common,
                    "comparison_group_id": exact_group,
                    "method_id": method_id,
                    "reference_method_id": result["reference_method"],
                    "metric_id": metric_id,
                    "delta": result["delta"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "probability_delta_le_zero": result["probability_delta_le_zero"],
                    "higher_is_better": result["higher_is_better"],
                    "bootstrap_valid_draws": result["valid_draws"],
                    "status": "OK_FALLBACK" if candidate_fallback or reference_fallback else "OK",
                })
        population_checks.append({
            "population_id": population_id,
            "status": "OK_AGGREGATED",
            "observed": observed,
            "weighting": aggregate_rule["weighting"],
            "interpretation": aggregate_rule["interpretation"],
            "bootstrap_unit": population_interval["bootstrap_unit"],
            "seed": seed,
            "link_blocks": population_interval["link_blocks"],
        })

    metrics_path = evaluation_root / "metrics_long.csv"
    contrasts_path = evaluation_root / "contrasts_long.csv"
    end_evaluation_source_snapshot = {
        "files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in EVALUATION_SOURCE_FILES
        ]
    }
    end_evaluation_source_snapshot["snapshot_sha256"] = sha256_bytes(
        canonical_json_bytes(end_evaluation_source_snapshot)
    )
    if end_evaluation_source_snapshot != evaluation_source_snapshot:
        raise RuntimeError("evaluation source tree changed after labels were opened")
    _write_csv(metrics_path, metrics)
    _write_contrasts_csv(contrasts_path, contrasts)
    manifest = {
        "schema_version": "reconstruction-external-evaluation-v2",
        "release_id": args.release_id,
        "build_id": args.build,
        "scientific_full": freeze["scientific_full"],
        "ab_verification_status": certificate["status"],
        "ab_certificate_path": str(Path(certificate_path).resolve()),
        "ab_certificate_sha256": certificate["certificate_sha256"],
        "ab_certificate_file_sha256": sha256_file(certificate_path),
        "score_freeze_sha256": sha256_file(freeze_path),
        "score_freeze_payload_sha256": freeze["payload_sha256"],
        "external_registry_sha256": registry.sha256,
        "identity_contract": external_id_contract_binding(
            registry, identity_key=identity_key
        ),
        "id_contract_version": ID_CONTRACT_VERSION,
        "evaluation_source_snapshot": evaluation_source_snapshot,
        "evaluation_source_snapshot_sha256": evaluation_source_snapshot["snapshot_sha256"],
        "source_root": str(args.source_root.resolve()),
        "labels_opened_only_after_score_freeze": True,
        "score_semantics": "higher_is_incorrect",
        "positive_class": "incorrect",
        "metric_intervals": "registered per-cell and population grouped paired source-level bootstrap",
        "bootstrap_draws": args.bootstrap_draws,
        "n_metric_rows": len(metrics),
        "n_contrast_rows": len(contrasts),
        "metrics_path": metrics_path.name,
        "metrics_sha256": sha256_file(metrics_path),
        "contrasts_path": contrasts_path.name,
        "contrasts_sha256": sha256_file(contrasts_path),
        "label_records": label_records,
        "population_checks": population_checks,
        "applicability_statuses": [
            {"cell_id": item["cell_id"], "status": item["status"], "reason": item.get("reason")}
            for item in input_manifest["cells"]
        ],
    }
    manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    atomic_write_json(evaluation_root / "MANIFEST.json", manifest)
    stage.commit()
    _ACTIVE_EVALUATION_STAGE = None
    print(json.dumps({
        "n_metric_rows": len(metrics),
        "n_contrast_rows": len(contrasts),
        "metrics": str(final_evaluation_root / metrics_path.name),
        "population_checks": population_checks,
    }, indent=2, sort_keys=True))


def main() -> None:
    """Run evaluation and clean any uncommitted label-bearing stage on error."""

    global _ACTIVE_EVALUATION_STAGE

    try:
        _main()
    finally:
        stage = _ACTIVE_EVALUATION_STAGE
        _ACTIVE_EVALUATION_STAGE = None
        if stage is not None:
            stage.cleanup()


if __name__ == "__main__":
    main()
