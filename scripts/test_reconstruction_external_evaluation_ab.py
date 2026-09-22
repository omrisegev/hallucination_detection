#!/usr/bin/env python3
"""Adversarial tests for external final-answer Evaluation A/B certification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import spectral_utils.reconstruction_benchmark.external_evaluation_ab as evaluation_ab  # noqa: E402

from spectral_utils.reconstruction_benchmark.external_evaluation import (  # noqa: E402
    METRIC_IDS,
    binary_metric_values,
)
from spectral_utils.reconstruction_benchmark.external_evaluation_ab import (  # noqa: E402
    CONTRAST_FIELDS,
    DEFAULT_BOOTSTRAP_DRAWS,
    EVALUATION_SCHEMA_VERSION,
    EVALUATION_SOURCE_FILES,
    METRIC_FIELDS,
    VERIFICATION_SOURCE_FILES,
    _BuildContext,
    _LabelState,
    _comparison_group_id,
    _expected_population_checks,
    _label_sha256,
    _normalize_manifest,
    _payload_sha256,
    _population_metadata,
    _require_exact_ab_identity,
    _row_roster_sha256,
    _source_snapshot,
    _verify_evaluation_build,
    _write_immutable_certificate,
    verify_external_evaluation_ab,
)
from spectral_utils.reconstruction_benchmark.external_fit_contract import (  # noqa: E402
    fit_row_roster_sha256,
)
from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    ExternalCellSpec,
    ExternalRegistry,
    ID_CONTRACT_VERSION,
    LABEL_SCHEMA_VERSION,
    LabelVector,
    OpaqueIdentityRoster,
    sealed_group_roster_commitment,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.methods import (  # noqa: E402
    PRIMARY_METHOD_IDS,
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_table_and_manifest(
    *, context: _BuildContext, manifest: dict, table: str,
    fields: tuple[str, ...], rows: list[dict],
) -> None:
    path = context.root / f"evaluation/{table}.csv"
    _write_csv(path, fields, rows)
    manifest[f"n_{'metric' if table == 'metrics_long' else 'contrast'}_rows"] = len(rows)
    manifest[f"{'metrics' if table == 'metrics_long' else 'contrasts'}_sha256"] = sha256_file(path)
    manifest.pop("payload_sha256", None)
    manifest["payload_sha256"] = _payload_sha256(manifest)
    atomic_write_json(context.root / "evaluation/MANIFEST.json", manifest)


class ExternalEvaluationABFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.release_id = "synthetic_external_evaluation_ab"
        self.release_root = root / "releases"
        self.release = self.release_root / self.release_id
        self.source_root = REPO
        self.cell_id = "synthetic_cell"
        self.population_id = "synthetic_population"
        self.full_identity = {
            "version": ID_CONTRACT_VERSION,
            "digest_algorithm": "hmac-sha256-canonical-json-v1",
            "identity_key_contract_version": "reconstruction-external-identity-key-v1",
            "identity_key_bytes": 32,
            "opaque_row_id_prefix": "xridv2_",
            "opaque_group_id_prefix": "xgidv2_",
            "canonical_row_order": "lexicographic_opaque_row_id",
            "row_namespace_scope": "cell",
            "group_namespace_by_population": {
                "synthetic_population": "cell",
            },
            "key_id": "xkidv1_" + "1" * 64,
        }
        self.full_identity["contract_sha256"] = _payload_sha256(
            self.full_identity
        )
        self.fit_identity = {
            "schema_version": "reconstruction-external-fit-row-identity-v1",
            "version": ID_CONTRACT_VERSION,
            "digest_algorithm": "hmac-sha256-canonical-json-v1",
            "identity_key_contract_version": "reconstruction-external-identity-key-v1",
            "identity_key_bytes": 32,
            "opaque_row_id_prefix": "xridv2_",
            "canonical_row_order": "lexicographic_opaque_row_id",
            "row_namespace_scope": "cell",
            "key_id": self.full_identity["key_id"],
            "private_group_linkage_commitment": "xglcv1_" + "2" * 64,
        }
        self.fit_identity["contract_sha256"] = _payload_sha256(self.fit_identity)
        self.row_ids = tuple(f"xridv2_{index:064x}" for index in range(1, 5))
        self.group_ids = (
            "xgidv2_" + "a" * 64,
            "xgidv2_" + "a" * 64,
            "xgidv2_" + "b" * 64,
            "xgidv2_" + "b" * 64,
        )
        self.labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        self.score = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        self.row_namespace = "4" * 64
        self.group_namespace = "5" * 64
        self.label_row_roster = _row_roster_sha256(
            self.row_ids, identity_contract=self.full_identity,
            row_namespace_sha256=self.row_namespace,
        )
        self.fit_row_roster = fit_row_roster_sha256(
            self.row_ids, contract=self.fit_identity,
            row_namespace_sha256_value=self.row_namespace,
        )
        self.cohort_id = sealed_group_roster_commitment(OpaqueIdentityRoster(
            row_ids=self.row_ids,
            group_ids=self.group_ids,
            contract_binding=self.full_identity,
            row_namespace_sha256=self.row_namespace,
            group_namespace_sha256=self.group_namespace,
        ))
        self.cell_data = {
            self.cell_id: {
                "row_ids": self.row_ids,
                "group_ids": self.group_ids,
                "labels": self.labels,
                "score": self.score,
                "row_namespace": self.row_namespace,
                "group_namespace": self.group_namespace,
                "label_row_roster": self.label_row_roster,
                "fit_row_roster": self.fit_row_roster,
                "cohort_id": self.cohort_id,
            }
        }
        self.registry = ExternalRegistry(
            path=REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json",
            sha256="6" * 64,
            population_registry_path=REPO / "configs/reconstruction_benchmark_v1/populations.json",
            population_registry_sha256="7" * 64,
            raw={
                "population_expectations": {
                    self.population_id: {
                        "rows": 4, "incorrect": 2, "correct": 2, "cells": 1,
                    }
                },
                "population_aggregates": {
                    self.population_id: {
                        "enabled": False,
                        "reason": "synthetic aggregate disabled",
                        "weighting": "single_cell",
                        "interpretation": "overall",
                        "link_cells_by": "none",
                        "bootstrap": "source_group",
                    }
                },
            },
            cells=(ExternalCellSpec(
                cell_id=self.cell_id,
                population_id=self.population_id,
                dataset_id="synthetic_dataset",
                model_id="synthetic_model",
                slice_id="overall",
                domain="test",
                comparison_group_id="synthetic_group",
                expected_rows=4,
                adapter_id="synthetic_adapter",
                fit_policy="run_if_compatible",
                panel_role="application",
                source={"kind": "synthetic"},
                expected_incorrect=2,
                expected_correct=2,
                expected_group_count=2,
            ),),
        )
        self.certificate_path = self.release / "external_final_answer/AB_VERIFICATION.json"
        self.certificate_path.parent.mkdir(parents=True)
        atomic_write_json(self.certificate_path, {
            "schema_version": "synthetic-score-ab",
            "certificate_sha256": "8" * 64,
        })
        self.score_certificate = {
            "certificate_sha256": "8" * 64,
            "identity_contract": self.fit_identity,
            "method_ids": list(PRIMARY_METHOD_IDS),
            "cell_ids": [self.cell_id],
            "status": "PASS",
            "scientific_full": True,
            "builds": {"A": {}, "B": {}},
        }

    def reconfigure_population(
        self, *, population_id: str, group_scope: str,
        aggregate: dict, cells: list[dict], registry_sha256: str,
    ) -> None:
        """Replace the default single-cell population with a focused fixture."""

        identity = {
            key: value for key, value in self.full_identity.items()
            if key != "contract_sha256"
        }
        identity["group_namespace_by_population"] = {
            population_id: group_scope,
        }
        identity["contract_sha256"] = _payload_sha256(identity)
        self.full_identity = identity
        self.population_id = population_id
        specs: list[ExternalCellSpec] = []
        cell_data: dict[str, dict] = {}
        for index, configured in enumerate(cells, start=1):
            cell_id = str(configured["cell_id"])
            row_ids = tuple(map(str, configured["row_ids"]))
            group_ids = tuple(map(str, configured["group_ids"]))
            labels = np.asarray(configured["labels"], dtype=np.int8)
            score = np.asarray(configured["score"], dtype=np.float64)
            row_namespace = str(configured.get("row_namespace", f"{index + 3:x}" * 64))
            group_namespace = str(configured.get("group_namespace", "5" * 64))
            label_row_roster = _row_roster_sha256(
                row_ids, identity_contract=self.full_identity,
                row_namespace_sha256=row_namespace,
            )
            fit_roster = fit_row_roster_sha256(
                row_ids, contract=self.fit_identity,
                row_namespace_sha256_value=row_namespace,
            )
            cohort_id = sealed_group_roster_commitment(OpaqueIdentityRoster(
                row_ids=row_ids,
                group_ids=group_ids,
                contract_binding=self.full_identity,
                row_namespace_sha256=row_namespace,
                group_namespace_sha256=group_namespace,
            ))
            cell_data[cell_id] = {
                "row_ids": row_ids,
                "group_ids": group_ids,
                "labels": labels,
                "score": score,
                "row_namespace": row_namespace,
                "group_namespace": group_namespace,
                "label_row_roster": label_row_roster,
                "fit_row_roster": fit_roster,
                "cohort_id": cohort_id,
            }
            specs.append(ExternalCellSpec(
                cell_id=cell_id,
                population_id=population_id,
                dataset_id=str(configured.get("dataset_id", "synthetic_dataset")),
                model_id=str(configured.get("model_id", f"synthetic_model_{index}")),
                slice_id=str(configured.get("slice_id", "overall")),
                domain="test",
                comparison_group_id=str(configured.get("comparison_group_id", "synthetic_group")),
                expected_rows=len(row_ids),
                adapter_id="synthetic_adapter",
                fit_policy="run_if_compatible",
                panel_role=str(configured.get("panel_role", "application")),
                source={"kind": "synthetic"},
                expected_incorrect=int(labels.sum()),
                expected_correct=int(len(labels) - labels.sum()),
                expected_group_count=len(set(group_ids)),
            ))
        observed_labels = np.concatenate([cell_data[spec.cell_id]["labels"] for spec in specs])
        self.registry = ExternalRegistry(
            path=REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json",
            sha256=registry_sha256,
            population_registry_path=REPO / "configs/reconstruction_benchmark_v1/populations.json",
            population_registry_sha256="7" * 64,
            raw={
                "population_expectations": {
                    population_id: {
                        "rows": int(len(observed_labels)),
                        "incorrect": int(observed_labels.sum()),
                        "correct": int(len(observed_labels) - observed_labels.sum()),
                        "cells": len(specs),
                    }
                },
                "population_aggregates": {population_id: aggregate},
            },
            cells=tuple(specs),
        )
        self.cell_data = cell_data
        self.cell_id = specs[0].cell_id
        first = cell_data[self.cell_id]
        for key in (
            "row_ids", "group_ids", "labels", "score", "row_namespace",
            "group_namespace", "label_row_roster", "fit_row_roster", "cohort_id",
        ):
            setattr(self, key, first[key])
        self.score_certificate.update({
            "identity_contract": self.fit_identity,
            "cell_ids": [spec.cell_id for spec in specs],
        })

    def label_provenance(
        self, freeze_payload_sha256: str, cell_id: str | None = None,
    ) -> dict:
        data = self.cell_data[cell_id or self.cell_id]
        labels = data["labels"]
        return {
            "row_label_sha256": _label_sha256(data["row_ids"], labels),
            "positive_class": "incorrect",
            "n_incorrect": int(labels.sum()),
            "n_correct": int(len(labels) - labels.sum()),
            "score_freeze_payload_sha256": freeze_payload_sha256,
            "identity_contract": self.full_identity,
            "id_contract_version": ID_CONTRACT_VERSION,
            "id_contract_sha256": self.full_identity["contract_sha256"],
            "identity_key_id": self.full_identity["key_id"],
            "row_namespace_sha256": data["row_namespace"],
            "group_namespace_sha256": data["group_namespace"],
            "row_roster_sha256": data["label_row_roster"],
            "sealed_group_roster_commitment_sha256": data["cohort_id"],
        }

    def label_loader(self, context: _BuildContext):
        def load(**kwargs) -> LabelVector:
            cell_id = str(kwargs["spec"].cell_id)
            data = self.cell_data[cell_id]
            if Path(kwargs["repo"]) != self.source_root:
                raise RuntimeError("synthetic raw rederivation received another source root")
            if tuple(map(str, kwargs["expected_row_ids"])) != data["row_ids"]:
                raise RuntimeError("synthetic raw rederivation received another score cohort")
            if kwargs["expected_group_roster_commitment_sha256"] != data["cohort_id"]:
                raise RuntimeError("synthetic raw rederivation received another group commitment")
            return LabelVector(
                cell_id, data["row_ids"], data["group_ids"], data["labels"],
                self.label_provenance(context.freeze["payload_sha256"], cell_id),
            )
        return load

    def make_build(self, build_id: str) -> tuple[_BuildContext, dict]:
        root = self.release / f"build_{build_id}/external_final_answer"
        fit = root / "fit"
        evaluation = root / "evaluation"
        labels_root = evaluation / "labels"
        fit.mkdir(parents=True)
        labels_root.mkdir(parents=True)
        freeze = {
            "schema_version": "synthetic-freeze",
            "release_id": self.release_id,
            "build_id": build_id,
            "identity_contract": self.fit_identity,
        }
        freeze["payload_sha256"] = _payload_sha256(freeze)
        atomic_write_json(fit / "SCORE_FREEZE_MANIFEST.json", freeze)

        prepared_rows: list[dict] = []
        prepared_by_cell: dict[str, dict] = {}
        records: dict[tuple[str, str], dict] = {}
        for spec in self.registry.cells:
            data = self.cell_data[spec.cell_id]
            prepared = {
                "cell_id": spec.cell_id,
                "status": "ELIGIBLE",
                "reason": None,
                "identity_contract": self.full_identity,
                "fit_row_identity_contract": self.fit_identity,
                "id_contract_sha256": self.full_identity["contract_sha256"],
                "fit_row_id_contract_sha256": self.fit_identity["contract_sha256"],
                "identity_key_id": self.full_identity["key_id"],
                "row_roster_sha256": data["fit_row_roster"],
                "sealed_group_roster_commitment_sha256": data["cohort_id"],
                "row_namespace_sha256": data["row_namespace"],
                "group_namespace_sha256": data["group_namespace"],
                "group_count": len(set(data["group_ids"])),
            }
            prepared_rows.append(prepared)
            prepared_by_cell[spec.cell_id] = prepared
            for method_id in PRIMARY_METHOD_IDS:
                relative = f"cells/{spec.cell_id}/{method_id}/score.npz"
                score_path = fit / relative
                score_path.parent.mkdir(parents=True)
                score_sha = atomic_write_npz(score_path, {
                    "row_ids": np.asarray(data["row_ids"], dtype="<U80"),
                    "score": data["score"],
                    "id_contract_version": np.asarray([ID_CONTRACT_VERSION], dtype="<U64"),
                    "id_contract_sha256": np.asarray([self.fit_identity["contract_sha256"]], dtype="<U64"),
                    "identity_key_id": np.asarray([self.fit_identity["key_id"]], dtype="<U80"),
                    "row_namespace_sha256": np.asarray([data["row_namespace"]], dtype="<U64"),
                    "row_roster_sha256": np.asarray([data["fit_row_roster"]], dtype="<U64"),
                })
                records[(spec.cell_id, method_id)] = {
                    "cell_id": spec.cell_id,
                    "method_id": method_id,
                    "status": "OK",
                    "score_path": relative,
                    "score_sha256": score_sha,
                    "id_contract_version": ID_CONTRACT_VERSION,
                    "id_contract_sha256": self.fit_identity["contract_sha256"],
                    "identity_key_id": self.fit_identity["key_id"],
                    "row_namespace_sha256": data["row_namespace"],
                    "row_roster_sha256": data["fit_row_roster"],
                }
        context = _BuildContext(
            build_id=build_id,
            root=root,
            input_manifest={"cells": prepared_rows},
            preparation_manifest={
                "identity_contract": self.full_identity,
                "fit_row_identity_contract": self.fit_identity,
                "source_root": str(self.source_root),
                "cells": prepared_rows,
            },
            freeze=freeze,
            prepared_by_cell=prepared_by_cell,
            records_by_pair=records,
        )

        label_records: list[dict] = []
        label_states: dict[str, _LabelState] = {}
        point_metrics: dict[str, dict[str, dict[str, float]]] = {}
        for spec in self.registry.cells:
            cell_id = spec.cell_id
            data = self.cell_data[cell_id]
            label_path = labels_root / f"{cell_id}.npz"
            label_file_sha = atomic_write_npz(label_path, {
                "row_ids": np.asarray(data["row_ids"], dtype="<U80"),
                "group_ids": np.asarray(data["group_ids"], dtype="<U80"),
                "incorrect": data["labels"],
                "id_contract_version": np.asarray([ID_CONTRACT_VERSION], dtype="<U64"),
                "id_contract_sha256": np.asarray([self.full_identity["contract_sha256"]], dtype="<U64"),
                "identity_key_id": np.asarray([self.full_identity["key_id"]], dtype="<U80"),
                "row_namespace_sha256": np.asarray([data["row_namespace"]], dtype="<U64"),
                "group_namespace_sha256": np.asarray([data["group_namespace"]], dtype="<U64"),
            })
            label_sha = _label_sha256(data["row_ids"], data["labels"])
            provenance = self.label_provenance(freeze["payload_sha256"], cell_id)
            label_records.append({
                "schema_version": LABEL_SCHEMA_VERSION,
                "cell_id": cell_id,
                "n_rows": len(data["row_ids"]),
                "artifact_path": f"labels/{cell_id}.npz",
                "artifact_sha256": label_file_sha,
                "identity_contract": self.full_identity,
                "id_contract_version": ID_CONTRACT_VERSION,
                "id_contract_sha256": self.full_identity["contract_sha256"],
                "identity_key_id": self.full_identity["key_id"],
                "row_namespace_sha256": data["row_namespace"],
                "group_namespace_sha256": data["group_namespace"],
                "row_roster_sha256": data["label_row_roster"],
                "sealed_group_roster_commitment_sha256": data["cohort_id"],
                "provenance": provenance,
            })
            label_states[cell_id] = _LabelState(
                cell_id=cell_id,
                row_ids=data["row_ids"],
                group_ids=data["group_ids"],
                incorrect=data["labels"],
                artifact_sha256=label_file_sha,
                row_label_sha256=label_sha,
                cohort_id=data["cohort_id"],
                n_groups=len(set(data["group_ids"])),
            )
            points = binary_metric_values(data["labels"], data["score"])
            point_metrics[cell_id] = {
                method_id: dict(points) for method_id in PRIMARY_METHOD_IDS
            }

        metric_rows: list[dict] = []
        contrast_rows: list[dict] = []
        for spec in self.registry.cells:
            cell_id = spec.cell_id
            data = self.cell_data[cell_id]
            label_state = label_states[cell_id]
            stratified = (
                self.registry.raw["population_aggregates"][spec.population_id].get("bootstrap")
                == "source_group_stratified_by_label"
            )
            for method_id in sorted(PRIMARY_METHOD_IDS):
                record = records[(cell_id, method_id)]
                for metric_id in METRIC_IDS:
                    value = point_metrics[cell_id][method_id][metric_id]
                    metric_rows.append({
                        "comparison_group_id": _comparison_group_id(
                            level="cell", cell_id=cell_id, population_id=None,
                            cohort_id=label_state.cohort_id, metric_id=metric_id,
                        ),
                        "panel_role": spec.panel_role,
                        "population_id": spec.population_id,
                        "cell_id": cell_id,
                        "dataset_id": spec.dataset_id,
                        "model_id": spec.model_id,
                        "slice_id": spec.slice_id,
                        "method_id": method_id,
                        "metric_id": metric_id,
                        "value": value,
                        "ci_low": value,
                        "ci_high": value,
                        "status": "OK",
                        "n": len(data["labels"]),
                        "n_incorrect": int(data["labels"].sum()),
                        "n_correct": int(len(data["labels"]) - data["labels"].sum()),
                        "bootstrap_unit": "source_group",
                        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "bootstrap_valid_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "cohort_id": label_state.cohort_id,
                        "score_sha256": record["score_sha256"],
                        "label_sha256": label_state.row_label_sha256,
                        "record_level": "cell",
                        "stratified_by_label": stratified,
                        "n_cells": 1,
                        "n_groups": label_state.n_groups,
                    })
            for method_id in sorted(PRIMARY_METHOD_IDS):
                if method_id == "iu_pcr":
                    continue
                for metric_id in METRIC_IDS:
                    delta = (
                        point_metrics[cell_id][method_id][metric_id]
                        - point_metrics[cell_id]["iu_pcr"][metric_id]
                    )
                    contrast_rows.append({
                        "comparison_group_id": _comparison_group_id(
                            level="cell", cell_id=cell_id, population_id=None,
                            cohort_id=label_state.cohort_id, metric_id=metric_id,
                        ),
                        "panel_role": spec.panel_role,
                        "population_id": spec.population_id,
                        "cell_id": cell_id,
                        "dataset_id": spec.dataset_id,
                        "model_id": spec.model_id,
                        "slice_id": spec.slice_id,
                        "method_id": method_id,
                        "reference_method_id": "iu_pcr",
                        "metric_id": metric_id,
                        "delta": delta,
                        "ci_low": delta,
                        "ci_high": delta,
                        "probability_delta_le_zero": 1.0 if delta <= 0 else 0.0,
                        "higher_is_better": metric_id != "aurc_x1000",
                        "bootstrap_unit": "source_group",
                        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "bootstrap_valid_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "n": len(data["labels"]),
                        "n_groups": label_state.n_groups,
                        "cohort_id": label_state.cohort_id,
                        "record_level": "cell",
                        "stratified_by_label": stratified,
                        "n_cells": 1,
                        "status": "OK",
                    })

        population_checks = _expected_population_checks(
            registry=self.registry, context=context, labels=label_states,
        )
        for check in population_checks:
            if check["status"] != "OK_AGGREGATED":
                continue
            population_id = str(check["population_id"])
            metadata = _population_metadata(
                population_id=population_id,
                registry=self.registry,
                labels=label_states,
                point_metrics=point_metrics,
                context=context,
                check=check,
            )
            aggregate = metadata["aggregate"]
            observed = check["observed"]
            linked = any(bool(block["linked"]) for block in check["link_blocks"])
            stratified = aggregate.get("bootstrap") == "source_group_stratified_by_label"
            for method_id in sorted(PRIMARY_METHOD_IDS):
                for metric_id in METRIC_IDS:
                    value = metadata["points"][method_id][metric_id]
                    metric_rows.append({
                        "comparison_group_id": _comparison_group_id(
                            level="population", cell_id=None,
                            population_id=population_id,
                            cohort_id=metadata["cohort_id"], metric_id=metric_id,
                            aggregate=aggregate,
                        ),
                        "panel_role": metadata["panel_role"],
                        "population_id": population_id,
                        "cell_id": "__population__",
                        "dataset_id": metadata["dataset_id"],
                        "model_id": metadata["model_id"],
                        "slice_id": f"population::{aggregate['interpretation']}",
                        "method_id": method_id,
                        "metric_id": metric_id,
                        "value": value,
                        "ci_low": value,
                        "ci_high": value,
                        "status": metadata["statuses"][method_id],
                        "n": int(observed["rows"]),
                        "n_incorrect": int(observed["incorrect"]),
                        "n_correct": int(observed["correct"]),
                        "bootstrap_unit": check["bootstrap_unit"],
                        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "bootstrap_valid_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "cohort_id": metadata["cohort_id"],
                        "score_sha256": metadata["score_hashes"][method_id],
                        "label_sha256": metadata["label_sha256"],
                        "record_level": "population",
                        "aggregate_weighting": aggregate["weighting"],
                        "aggregate_interpretation": aggregate["interpretation"],
                        "linked_resampling": linked,
                        "stratified_by_label": stratified,
                        "n_cells": int(observed["cells"]),
                        "n_groups": metadata["n_groups"],
                    })
            for method_id in sorted(PRIMARY_METHOD_IDS):
                if method_id == "iu_pcr":
                    continue
                for metric_id in METRIC_IDS:
                    delta = (
                        metadata["points"][method_id][metric_id]
                        - metadata["points"]["iu_pcr"][metric_id]
                    )
                    contrast_rows.append({
                        "comparison_group_id": _comparison_group_id(
                            level="population", cell_id=None,
                            population_id=population_id,
                            cohort_id=metadata["cohort_id"], metric_id=metric_id,
                            aggregate=aggregate,
                        ),
                        "panel_role": metadata["panel_role"],
                        "population_id": population_id,
                        "cell_id": "__population__",
                        "dataset_id": metadata["dataset_id"],
                        "model_id": metadata["model_id"],
                        "slice_id": f"population::{aggregate['interpretation']}",
                        "method_id": method_id,
                        "reference_method_id": "iu_pcr",
                        "metric_id": metric_id,
                        "delta": delta,
                        "ci_low": delta,
                        "ci_high": delta,
                        "probability_delta_le_zero": 1.0 if delta <= 0 else 0.0,
                        "higher_is_better": metric_id != "aurc_x1000",
                        "bootstrap_unit": check["bootstrap_unit"],
                        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "bootstrap_valid_draws": DEFAULT_BOOTSTRAP_DRAWS,
                        "n": int(observed["rows"]),
                        "n_groups": metadata["n_groups"],
                        "cohort_id": metadata["cohort_id"],
                        "record_level": "population",
                        "aggregate_weighting": aggregate["weighting"],
                        "aggregate_interpretation": aggregate["interpretation"],
                        "linked_resampling": linked,
                        "stratified_by_label": stratified,
                        "n_cells": int(observed["cells"]),
                        "status": "OK",
                    })

        metrics_path = evaluation / "metrics_long.csv"
        contrasts_path = evaluation / "contrasts_long.csv"
        _write_csv(metrics_path, METRIC_FIELDS, metric_rows)
        _write_csv(contrasts_path, CONTRAST_FIELDS, contrast_rows)
        source_snapshot = _source_snapshot(REPO, EVALUATION_SOURCE_FILES)
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "release_id": self.release_id,
            "build_id": build_id,
            "scientific_full": True,
            "ab_verification_status": "PASS",
            "ab_certificate_path": str(self.certificate_path.resolve()),
            "ab_certificate_sha256": self.score_certificate["certificate_sha256"],
            "ab_certificate_file_sha256": sha256_file(self.certificate_path),
            "score_freeze_sha256": sha256_file(fit / "SCORE_FREEZE_MANIFEST.json"),
            "score_freeze_payload_sha256": freeze["payload_sha256"],
            "external_registry_sha256": self.registry.sha256,
            "identity_contract": self.full_identity,
            "id_contract_version": ID_CONTRACT_VERSION,
            "evaluation_source_snapshot": source_snapshot,
            "evaluation_source_snapshot_sha256": source_snapshot["snapshot_sha256"],
            "source_root": str(self.source_root),
            "labels_opened_only_after_score_freeze": True,
            "score_semantics": "higher_is_incorrect",
            "positive_class": "incorrect",
            "metric_intervals": "registered per-cell and population grouped paired source-level bootstrap",
            "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
            "n_metric_rows": len(metric_rows),
            "n_contrast_rows": len(contrast_rows),
            "metrics_path": "metrics_long.csv",
            "metrics_sha256": sha256_file(metrics_path),
            "contrasts_path": "contrasts_long.csv",
            "contrasts_sha256": sha256_file(contrasts_path),
            "label_records": label_records,
            "population_checks": population_checks,
            "applicability_statuses": [
                {"cell_id": spec.cell_id, "status": "ELIGIBLE", "reason": None}
                for spec in self.registry.cells
            ],
        }
        manifest["payload_sha256"] = _payload_sha256(manifest)
        atomic_write_json(evaluation / "MANIFEST.json", manifest)
        return context, manifest

    def verify_build(self, context: _BuildContext) -> dict:
        return _verify_evaluation_build(
            release_id=self.release_id,
            repo=REPO,
            certificate_path=self.certificate_path,
            registry=self.registry,
            score_certificate=self.score_certificate,
            context=context,
            identity_key=b"k" * 32,
            label_loader=self.label_loader(context),
        )


def _linked_aggregate_fixture(root: Path) -> ExternalEvaluationABFixture:
    fixture = ExternalEvaluationABFixture(root)
    shared_groups = (
        "xgidv2_" + "a" * 64,
        "xgidv2_" + "a" * 64,
        "xgidv2_" + "b" * 64,
        "xgidv2_" + "b" * 64,
    )
    fixture.reconfigure_population(
        population_id="synthetic_linked_population",
        group_scope="population",
        aggregate={
            "enabled": True,
            "weighting": "equal_cell",
            "interpretation": "equal_model_linked",
            "link_cells_by": "all",
            "bootstrap": "linked_source_question",
            "seed": 314159,
        },
        registry_sha256="9" * 64,
        cells=[
            {
                "cell_id": "linked_cell_a",
                "model_id": "model_a",
                "row_ids": tuple(f"xridv2_{index:064x}" for index in range(1, 5)),
                "group_ids": shared_groups,
                "labels": [0, 0, 1, 1],
                "score": [0.05, 0.15, 0.85, 0.95],
            },
            {
                "cell_id": "linked_cell_b",
                "model_id": "model_b",
                "row_ids": tuple(f"xridv2_{index:064x}" for index in range(11, 15)),
                "group_ids": shared_groups,
                "labels": [0, 0, 1, 1],
                "score": [0.10, 0.90, 0.20, 0.80],
            },
        ],
    )
    return fixture


def _hle_stratified_fixture(root: Path) -> ExternalEvaluationABFixture:
    fixture = ExternalEvaluationABFixture(root)
    groups = tuple(
        "xgidv2_" + digit * 64
        for digit in ("a", "a", "b", "b", "c", "c", "d", "d")
    )
    fixture.reconfigure_population(
        population_id="synthetic_hle_imbalanced_population",
        group_scope="cell",
        aggregate={
            "enabled": True,
            "weighting": "single_cell",
            "interpretation": "overall_hle_like",
            "link_cells_by": "none",
            "bootstrap": "source_group_stratified_by_label",
            "seed": 271828,
        },
        registry_sha256="a" * 64,
        cells=[{
            "cell_id": "hle_like_cell",
            "model_id": "qwen72b_like",
            "row_ids": tuple(f"xridv2_{index:064x}" for index in range(21, 29)),
            "group_ids": groups,
            "labels": [0, 0, 1, 1, 1, 1, 1, 1],
            "score": [0.05, 0.20, 0.40, 0.55, 0.65, 0.75, 0.85, 0.95],
        }],
    )
    return fixture


def _registry_conflict_fixture(root: Path) -> ExternalEvaluationABFixture:
    fixture = _linked_aggregate_fixture(root)
    fixture.registry.raw["population_expectations"][
        "synthetic_linked_population"
    ] = {
        "rows": 8,
        "incorrect": 5,
        "correct": 3,
        "cells": 2,
    }
    return fixture


class ExternalEvaluationABTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ExternalEvaluationABFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_synthetic_builds_are_exact_after_only_registered_normalization(self) -> None:
        # Real v3 has a full post-freeze identity, a distinct fit-only identity,
        # a full-v2 label roster hash, and a distinct fit-v3 prepared hash.
        self.assertNotEqual(
            self.fixture.full_identity["contract_sha256"],
            self.fixture.fit_identity["contract_sha256"],
        )
        self.assertNotEqual(
            self.fixture.label_row_roster,
            self.fixture.fit_row_roster,
        )
        context_a, _ = self.fixture.make_build("A")
        context_b, _ = self.fixture.make_build("B")
        audit_a = self.fixture.verify_build(context_a)
        audit_b = self.fixture.verify_build(context_b)
        _require_exact_ab_identity(audit_a, audit_b)
        self.assertEqual(audit_a["metrics_bytes"], audit_b["metrics_bytes"])
        self.assertEqual(audit_a["contrasts_bytes"], audit_b["contrasts_bytes"])
        self.assertEqual(audit_a["label_hashes"], audit_b["label_hashes"])

    def test_identically_missing_metric_row_is_rejected_as_partial(self) -> None:
        context, manifest = self.fixture.make_build("A")
        evaluation = context.root / "evaluation"
        metrics_path = evaluation / "metrics_long.csv"
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        _write_csv(metrics_path, METRIC_FIELDS, rows[:-1])
        manifest["n_metric_rows"] = len(rows) - 1
        manifest["metrics_sha256"] = sha256_file(metrics_path)
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = _payload_sha256(manifest)
        atomic_write_json(evaluation / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "exact ordered"):
            self.fixture.verify_build(context)

    def test_missing_contrast_row_is_rejected_even_when_manifest_is_rehashed(self) -> None:
        context, manifest = self.fixture.make_build("A")
        path = context.root / "evaluation/contrasts_long.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        _rewrite_table_and_manifest(
            context=context,
            manifest=manifest,
            table="contrasts_long",
            fields=CONTRAST_FIELDS,
            rows=rows[:-1],
        )
        with self.assertRaisesRegex(RuntimeError, "exact ordered"):
            self.fixture.verify_build(context)

    def test_metric_value_tamper_is_rejected_after_table_and_manifest_rehash(self) -> None:
        context, manifest = self.fixture.make_build("A")
        path = context.root / "evaluation/metrics_long.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        original = float(rows[0]["value"])
        replacement = original + 0.01 if original <= 0.98 else original - 0.01
        rows[0]["value"] = str(replacement)
        rows[0]["ci_low"] = str(replacement)
        rows[0]["ci_high"] = str(replacement)
        _rewrite_table_and_manifest(
            context=context,
            manifest=manifest,
            table="metrics_long",
            fields=METRIC_FIELDS,
            rows=rows,
        )
        with self.assertRaisesRegex(RuntimeError, "point estimate differs"):
            self.fixture.verify_build(context)

    def test_contrast_reference_tamper_is_rejected_after_rehash(self) -> None:
        context, manifest = self.fixture.make_build("A")
        path = context.root / "evaluation/contrasts_long.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["reference_method_id"] = "equal_feature_mean"
        _rewrite_table_and_manifest(
            context=context,
            manifest=manifest,
            table="contrasts_long",
            fields=CONTRAST_FIELDS,
            rows=rows,
        )
        with self.assertRaisesRegex(RuntimeError, "reference_method_id drifted"):
            self.fixture.verify_build(context)

    def test_contrast_delta_tamper_is_rejected_after_table_and_manifest_rehash(self) -> None:
        context, manifest = self.fixture.make_build("A")
        path = context.root / "evaluation/contrasts_long.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["delta"] = "0.01"
        rows[0]["ci_low"] = "0.01"
        rows[0]["ci_high"] = "0.01"
        rows[0]["probability_delta_le_zero"] = "0.0"
        _rewrite_table_and_manifest(
            context=context,
            manifest=manifest,
            table="contrasts_long",
            fields=CONTRAST_FIELDS,
            rows=rows,
        )
        with self.assertRaisesRegex(RuntimeError, "point delta differs"):
            self.fixture.verify_build(context)

    def test_tampered_source_snapshot_is_rejected_even_when_rehashed(self) -> None:
        context, manifest = self.fixture.make_build("A")
        snapshot = manifest["evaluation_source_snapshot"]
        snapshot["files"][0]["sha256"] = "f" * 64
        snapshot["snapshot_sha256"] = _payload_sha256({"files": snapshot["files"]})
        manifest["evaluation_source_snapshot_sha256"] = snapshot["snapshot_sha256"]
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = _payload_sha256(manifest)
        atomic_write_json(context.root / "evaluation/MANIFEST.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "source changed or is missing"):
            self.fixture.verify_build(context)

    def test_label_rederivation_source_root_must_match_certified_preparation(self) -> None:
        context, manifest = self.fixture.make_build("A")
        manifest["source_root"] = str(Path(self.temporary.name).resolve())
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = _payload_sha256(manifest)
        atomic_write_json(context.root / "evaluation/MANIFEST.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "differs from certified preparation"):
            self.fixture.verify_build(context)

    def test_real_v3_shaped_certified_source_overlay_is_accepted(self) -> None:
        overlay = (
            Path(self.temporary.name)
            / "results/reconstruction_benchmark_v1/source_overlays/external_final_answer_v1"
        )
        overlay.mkdir(parents=True)
        fixture = ExternalEvaluationABFixture(Path(self.temporary.name) / "overlay_release")
        fixture.source_root = overlay.resolve()
        context, manifest = fixture.make_build("A")
        self.assertEqual(
            manifest["source_root"], context.preparation_manifest["source_root"],
        )
        self.assertNotEqual(Path(manifest["source_root"]), REPO)
        audit = fixture.verify_build(context)
        self.assertEqual(audit["n_metric_rows"], len(PRIMARY_METHOD_IDS) * len(METRIC_IDS))

    def test_label_hash_difference_is_never_normalized(self) -> None:
        left = {
            "metrics_bytes": b"metrics",
            "contrasts_bytes": b"contrasts",
            "label_bytes": {"labels/cell.npz": b"labels"},
            "label_hashes": {"labels/cell.npz": "a" * 64},
            "normalized_manifest_bytes": b"manifest",
            "population_checks": [],
        }
        right = dict(left)
        right["label_hashes"] = {"labels/cell.npz": "b" * 64}
        with self.assertRaisesRegex(RuntimeError, "label artifact hashes differ"):
            _require_exact_ab_identity(left, right)

    def test_label_bytes_are_compared_directly_not_only_by_recorded_hash(self) -> None:
        left = {
            "metrics_bytes": b"metrics",
            "contrasts_bytes": b"contrasts",
            "label_bytes": {"labels/cell.npz": b"left"},
            "label_hashes": {"labels/cell.npz": "a" * 64},
            "normalized_manifest_bytes": b"manifest",
            "population_checks": [],
        }
        right = dict(left)
        right["label_bytes"] = {"labels/cell.npz": b"right"}
        with self.assertRaisesRegex(RuntimeError, "label artifacts are not byte-identical"):
            _require_exact_ab_identity(left, right)

    def test_coordinated_label_permutation_is_rejected_by_raw_rederivation(self) -> None:
        contexts = {}
        for build_id in ("A", "B"):
            context, manifest = self.fixture.make_build(build_id)
            contexts[build_id] = context
            label_path = context.root / f"evaluation/labels/{self.fixture.cell_id}.npz"
            arrays = load_npz_no_pickle(label_path)
            permuted = self.fixture.labels[::-1].copy()
            arrays["incorrect"] = permuted
            artifact_sha = atomic_write_npz(label_path, arrays)
            label_sha = _label_sha256(self.fixture.row_ids, permuted)
            record = manifest["label_records"][0]
            record["artifact_sha256"] = artifact_sha
            record["provenance"]["row_label_sha256"] = label_sha
            manifest.pop("payload_sha256")
            manifest["payload_sha256"] = _payload_sha256(manifest)
            atomic_write_json(context.root / "evaluation/MANIFEST.json", manifest)
        self.assertEqual(
            sha256_file(
                contexts["A"].root
                / f"evaluation/labels/{self.fixture.cell_id}.npz"
            ),
            sha256_file(
                contexts["B"].root
                / f"evaluation/labels/{self.fixture.cell_id}.npz"
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "independent registry/source rederivation"):
            self.fixture.verify_build(contexts["A"])

    def test_label_npz_scalar_tamper_is_rejected_after_artifact_rehash(self) -> None:
        context, manifest = self.fixture.make_build("A")
        label_path = context.root / f"evaluation/labels/{self.fixture.cell_id}.npz"
        arrays = load_npz_no_pickle(label_path)
        arrays["group_namespace_sha256"] = np.asarray(["f" * 64], dtype="<U64")
        artifact_sha = atomic_write_npz(label_path, arrays)
        manifest["label_records"][0]["artifact_sha256"] = artifact_sha
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = _payload_sha256(manifest)
        atomic_write_json(context.root / "evaluation/MANIFEST.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "label scalar binding failed"):
            self.fixture.verify_build(context)

    def test_enabled_two_cell_linked_population_roster_hashes_and_equal_cell_arithmetic(self) -> None:
        fixture = _linked_aggregate_fixture(Path(self.temporary.name) / "linked")
        context, manifest = fixture.make_build("A")
        audit = fixture.verify_build(context)
        check = audit["population_checks"][0]
        self.assertEqual(check["status"], "OK_AGGREGATED")
        self.assertEqual(check["observed"], {
            "rows": 8, "incorrect": 4, "correct": 4, "cells": 2,
        })
        self.assertEqual(check["bootstrap_unit"], "linked_source_group")
        block = check["link_blocks"][0]
        self.assertTrue(block["linked"])
        self.assertEqual(block["cell_ids"], ["linked_cell_a", "linked_cell_b"])
        roster = sorted(set(fixture.cell_data["linked_cell_a"]["group_ids"]))
        self.assertEqual(block["group_roster_sha256"], _payload_sha256(roster))
        self.assertEqual(block["group_member_counts_sha256"], _payload_sha256([
            {"group_id": group, "member_count": 2} for group in roster
        ]))

        metrics_path = context.root / "evaluation/metrics_long.csv"
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        population_row = next(
            row for row in rows
            if row["record_level"] == "population"
            and row["method_id"] == "equal_feature_mean"
            and row["metric_id"] == "auroc"
        )
        expected = float(np.mean([
            binary_metric_values(
                fixture.cell_data[cell_id]["labels"],
                fixture.cell_data[cell_id]["score"],
            )["auroc"]
            for cell_id in ("linked_cell_a", "linked_cell_b")
        ]))
        self.assertEqual(float(population_row["value"]), expected)
        population_row["value"] = str(expected - 0.01)
        population_row["ci_low"] = str(expected - 0.01)
        population_row["ci_high"] = str(expected - 0.01)
        _rewrite_table_and_manifest(
            context=context,
            manifest=manifest,
            table="metrics_long",
            fields=METRIC_FIELDS,
            rows=rows,
        )
        with self.assertRaisesRegex(RuntimeError, "equal-cell point estimate drifted"):
            fixture.verify_build(context)

    def test_hle_like_stratified_single_cell_population_has_exact_strata_and_hashes(self) -> None:
        fixture = _hle_stratified_fixture(Path(self.temporary.name) / "hle")
        context, manifest = fixture.make_build("A")
        audit = fixture.verify_build(context)
        check = audit["population_checks"][0]
        self.assertEqual(check["status"], "OK_AGGREGATED")
        self.assertEqual(check["bootstrap_unit"], "source_group_stratified_by_label")
        self.assertEqual(check["observed"], {
            "rows": 8, "incorrect": 6, "correct": 2, "cells": 1,
        })
        block = check["link_blocks"][0]
        self.assertFalse(block["linked"])
        self.assertEqual(block["groups_by_label"], {"0": 1, "1": 3})
        expected_labels = [
            {"group_id": "xgidv2_" + "a" * 64, "label": 0},
            {"group_id": "xgidv2_" + "b" * 64, "label": 1},
            {"group_id": "xgidv2_" + "c" * 64, "label": 1},
            {"group_id": "xgidv2_" + "d" * 64, "label": 1},
        ]
        self.assertEqual(block["group_labels_sha256"], _payload_sha256(expected_labels))

        manifest["population_checks"][0]["link_blocks"][0][
            "group_labels_sha256"
        ] = "0" * 64
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = _payload_sha256(manifest)
        atomic_write_json(context.root / "evaluation/MANIFEST.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "population roster/audit drifted"):
            fixture.verify_build(context)

    def test_registry_class_total_conflict_blocks_only_aggregate_with_exact_summaries(self) -> None:
        fixture = _registry_conflict_fixture(Path(self.temporary.name) / "conflict")
        context, manifest = fixture.make_build("A")
        audit = fixture.verify_build(context)
        self.assertEqual(audit["n_metric_rows"], 2 * len(PRIMARY_METHOD_IDS) * len(METRIC_IDS))
        self.assertEqual(
            audit["n_contrast_rows"],
            2 * (len(PRIMARY_METHOD_IDS) - 1) * len(METRIC_IDS),
        )
        check = audit["population_checks"][0]
        self.assertEqual(check, {
            "population_id": "synthetic_linked_population",
            "status": "AGGREGATE_BLOCKED_REGISTRY_CLASS_TOTAL_MISMATCH",
            "registered_summary": {
                "rows": 8, "incorrect": 5, "correct": 3, "cells": 2,
            },
            "atomic_expected": {
                "rows": 8, "incorrect": 4, "correct": 4, "cells": 2,
            },
            "observed": {
                "rows": 8, "incorrect": 4, "correct": 4, "cells": 2,
            },
        })
        with (context.root / "evaluation/metrics_long.csv").open(
            "r", encoding="utf-8", newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertFalse(any(row["record_level"] == "population" for row in rows))

        manifest["population_checks"][0]["atomic_expected"]["incorrect"] = 5
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = _payload_sha256(manifest)
        atomic_write_json(context.root / "evaluation/MANIFEST.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "population roster/audit drifted"):
            fixture.verify_build(context)

    def test_unknown_manifest_difference_is_not_normalized(self) -> None:
        context_a, manifest_a = self.fixture.make_build("A")
        context_b, manifest_b = self.fixture.make_build("B")
        manifest_b["source_root"] = str(REPO / "different_source_root")
        manifest_b.pop("payload_sha256")
        manifest_b["payload_sha256"] = _payload_sha256(manifest_b)
        left = {
            "metrics_bytes": b"metrics",
            "contrasts_bytes": b"contrasts",
            "label_bytes": {"labels/cell.npz": b"labels"},
            "label_hashes": {"labels/cell.npz": "a" * 64},
            "normalized_manifest_bytes": _normalize_manifest(
                manifest_a, context=context_a,
            ),
            "population_checks": [],
        }
        right = dict(left)
        right["normalized_manifest_bytes"] = _normalize_manifest(
            manifest_b, context=context_b,
        )
        with self.assertRaisesRegex(RuntimeError, "outside the explicit"):
            _require_exact_ab_identity(left, right)

    def test_verification_snapshot_covers_fit_contract_dependency(self) -> None:
        self.assertIn(
            "spectral_utils/reconstruction_benchmark/external_fit_contract.py",
            VERIFICATION_SOURCE_FILES,
        )

    def test_end_to_end_pass_certificate_is_self_hashed_idempotent_and_immutable(self) -> None:
        context_a, _ = self.fixture.make_build("A")
        context_b, _ = self.fixture.make_build("B")
        audits = {
            "A": self.fixture.verify_build(context_a),
            "B": self.fixture.verify_build(context_b),
        }
        contexts = {"A": context_a, "B": context_b}

        def context_for_build(**kwargs):
            return contexts[str(kwargs["build_id"])]

        def audit_for_build(**kwargs):
            return audits[str(kwargs["context"].build_id)]

        target = (
            self.fixture.release
            / "external_final_answer/EVALUATION_AB_VERIFICATION.json"
        )
        patchers = (
            mock.patch.object(
                evaluation_ab, "load_external_registry",
                return_value=self.fixture.registry,
            ),
            mock.patch.object(
                evaluation_ab, "assert_external_ab_certificate",
                return_value=self.fixture.score_certificate,
            ),
            mock.patch.object(
                evaluation_ab, "load_identity_key", return_value=b"k" * 32,
            ),
            mock.patch.object(
                evaluation_ab, "_validate_input_context",
                side_effect=context_for_build,
            ),
            mock.patch.object(
                evaluation_ab, "_verify_evaluation_build",
                side_effect=audit_for_build,
            ),
        )
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
            certificate = verify_external_evaluation_ab(
                release_id=self.fixture.release_id,
                release_root=self.fixture.release_root,
                registry_path="synthetic-registry.json",
                population_registry_path="synthetic-populations.json",
                repo=REPO,
            )
            self.assertEqual(certificate["status"], "PASS")
            self.assertTrue(certificate["scientific_full"])
            self.assertEqual(target.read_bytes(), canonical_json_bytes(certificate) + b"\n")
            recorded_sha = certificate["certificate_sha256"]
            unsigned = dict(certificate)
            unsigned.pop("certificate_sha256")
            self.assertEqual(recorded_sha, sha256_bytes(canonical_json_bytes(unsigned)))

            second = verify_external_evaluation_ab(
                release_id=self.fixture.release_id,
                release_root=self.fixture.release_root,
                registry_path="synthetic-registry.json",
                population_registry_path="synthetic-populations.json",
                repo=REPO,
            )
            self.assertEqual(second, certificate)

            tampered = dict(certificate)
            tampered["status"] = "FAIL"
            target.write_bytes(canonical_json_bytes(tampered) + b"\n")
            with self.assertRaisesRegex(FileExistsError, "target already differs"):
                verify_external_evaluation_ab(
                    release_id=self.fixture.release_id,
                    release_root=self.fixture.release_root,
                    registry_path="synthetic-registry.json",
                    population_registry_path="synthetic-populations.json",
                    repo=REPO,
                )

    def test_immutable_certificate_writer_rejects_symlink_target(self) -> None:
        directory = Path(self.temporary.name) / "immutable_symlink"
        directory.mkdir()
        destination = directory / "destination.json"
        destination.write_bytes(b"signed payload")
        target = directory / "certificate.json"
        target.symlink_to(destination)
        with self.assertRaisesRegex(FileExistsError, "target is unsafe"):
            _write_immutable_certificate(target, b"signed payload")

    def test_immutable_certificate_writer_detects_atomic_no_clobber_race(self) -> None:
        directory = Path(self.temporary.name) / "immutable_race"
        directory.mkdir()
        target = directory / "certificate.json"

        def claim_target(_source, destination, **_kwargs):
            Path(destination).write_bytes(b"coordinated tamper")
            raise FileExistsError(destination)

        with mock.patch.object(evaluation_ab.os, "link", side_effect=claim_target):
            with self.assertRaisesRegex(FileExistsError, "target already differs"):
                _write_immutable_certificate(target, b"signed payload")


if __name__ == "__main__":
    unittest.main()
