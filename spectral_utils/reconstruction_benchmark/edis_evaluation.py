"""Post-freeze grouped evaluation for the EDIS/AIME reconstruction lane."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .edis_ab import (
    assert_ab_certificate,
    canonical_evaluation_table_sha256,
    load_private_provenance,
    validate_score_freeze,
)
from .edis_bootstrap import (
    grouped_paired_bootstrap_auroc_auprc,
    population_grouped_paired_bootstrap_auroc_auprc,
)
from .edis_preparation import (
    EdisCellSpec,
    EdisPreparationRegistry,
    KeyedIdentityController,
    _raw_group_identity,
    _raw_identity,
    _question_fingerprint,
    _stable_problem_key,
    load_preparation_registry,
    verify_pinned_file,
)
from .io import atomic_write_json, atomic_write_npz, canonical_json_bytes, load_npz_no_pickle, sha256_bytes, sha256_file
from .methods import PRIMARY_METHOD_IDS, PRIMARY_METHOD_SPECS


POSTFREEZE_SCHEMA = "reconstruction-edis-postfreeze-registry-v1"
EVALUATION_SCHEMA = "reconstruction-edis-evaluation-v1"
ADAPTER_ID = "edis_saved_trace_identity_v1"
EVALUATOR_ID = "edis_grouped_postfreeze_v1"
ACCESS_CONTRACT_ID = "gray_box_multi_pass"
EVIDENCE_STATUS_DETAIL = (
    "DESCRIPTIVE_GATE_FAILED: the acquisition cell failed at least one frozen "
    "accuracy or minority-class gate; the numeric result is context-only."
)


def _payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def load_postfreeze_registry(path: str | Path, preparation: EdisPreparationRegistry) -> dict[str, Any]:
    target = Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    if raw.get("schema_version") != POSTFREEZE_SCHEMA or raw.get("lane_id") != preparation.lane_id:
        raise RuntimeError("unexpected EDIS post-freeze registry")
    if raw.get("positive_class") != "incorrect" or raw.get("metrics") != ["auroc", "auprc"]:
        raise RuntimeError("EDIS post-freeze estimand drifted")
    if raw.get("group_identity_contract") != {
        "unit": "source_question",
        "scope": "dataset_across_temperatures",
        "raw_identity": "dataset_id_plus_sha256_of_saved_question_text",
        "require_identical_ordered_question_roster_across_temperatures": True,
    }:
        raise RuntimeError("EDIS post-freeze group identity contract drifted")
    if raw.get("partial_preparation_evaluation") != {
        "required_status_roster_contract_id": (
            "edis-materialized-status-roster-v1-2026-08-24"
        ),
        "explicit_cli_opt_in_required": True,
        "open_labels_for_ready_cells_only": True,
        "blocked_cell_output": "coverage_status_row_per_method",
        "dataset_aggregates_allowed": False,
        "task_aggregates_allowed": False,
        "headline_eligible": False,
        "publication_eligible": False,
    }:
        raise RuntimeError("EDIS partial post-freeze evaluation contract drifted")
    bootstrap = raw.get("bootstrap", {})
    if bootstrap != {
        "draws": 20_000,
        "unit": "source_question",
        "paired": True,
        "reference_method_id": "iu_pcr",
    }:
        raise RuntimeError("EDIS post-freeze bootstrap contract drifted")
    evidence = raw.get("evidence_boundary", {})
    if (
        evidence.get("headline_eligible") is not False
        or evidence.get("status") != "DESCRIPTIVE_GATE_FAILED"
        or evidence.get("one_pass_leaderboard_comparison") != "forbidden"
        or evidence.get("track") != "multi_sample_inference"
    ):
        raise RuntimeError("EDIS descriptive evidence boundary drifted")
    rows = raw.get("cells")
    if not isinstance(rows, list):
        raise RuntimeError("EDIS post-freeze cell registry is absent")
    expected_ids = [cell.cell_id for cell in preparation.cells]
    observed_ids = [str(item.get("cell_id", "")) for item in rows]
    if observed_ids != expected_ids:
        raise RuntimeError("EDIS post-freeze cell roster/order differs from preparation")
    for item, spec in zip(rows, preparation.cells):
        correct = int(item.get("expected_correct", -1))
        incorrect = int(item.get("expected_incorrect", -1))
        if correct <= 0 or incorrect <= 0 or correct + incorrect != spec.expected_rows:
            raise RuntimeError(f"{spec.cell_id}: post-freeze class expectation is invalid")
        if item.get("class_status") != "TWO_CLASS" or item.get("gate_status") != "FAILED":
            raise RuntimeError(f"{spec.cell_id}: gate/class status drifted")
        if not item.get("gate_reasons"):
            raise RuntimeError(f"{spec.cell_id}: failed gate lacks reasons")
    return raw


def reconstruct_labels_and_groups(
    *, spec: EdisCellSpec, source_path: str | Path, identity: KeyedIdentityController
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, str, str]:
    """Open the embedded grader label only after A/B certification."""

    with Path(source_path).open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, Mapping):
        raise TypeError(f"{spec.cell_id}: label source is not a mapping")
    problem_keys = sorted((_stable_problem_key(key), key) for key in data)
    if [integer for integer, _ in problem_keys] != list(range(spec.expected_questions)):
        raise RuntimeError(f"{spec.cell_id}: post-freeze source-question roster drifted")
    row_namespace = {
        "lane_id": spec.lane_id,
        "scope": "dataset_temperature_cell",
        "cell_id": spec.cell_id,
    }
    group_namespace = {
        "lane_id": spec.lane_id,
        "scope": "dataset_across_temperatures",
        "dataset_id": spec.dataset_id,
    }
    rows: list[str] = []
    groups: list[str] = []
    incorrect: list[int] = []
    question_fingerprints: list[str] = []
    for problem, source_key in problem_keys:
        entry = data[source_key]
        if not isinstance(entry, Mapping):
            raise TypeError(f"{spec.cell_id}: label source-question entry is not a mapping")
        candidates = entry.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise TypeError(f"{spec.cell_id}: label source lacks candidates")
        if len(candidates) != spec.candidates_per_question:
            raise RuntimeError(f"{spec.cell_id}: post-freeze candidate multiplicity drifted")
        question_fingerprint = _question_fingerprint(
            spec.dataset_id, problem, entry.get("question")
        )
        question_fingerprints.append(question_fingerprint)
        group_id = identity.group_id(
            namespace=group_namespace,
            raw_identity=_raw_group_identity(spec.dataset_id, question_fingerprint),
        )
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping) or "label" not in candidate:
                raise RuntimeError(f"{spec.cell_id}: embedded frozen grader label is absent")
            value = candidate["label"]
            if type(value) not in (bool, int, np.bool_, np.int8, np.int16, np.int32, np.int64):
                raise TypeError(f"{spec.cell_id}: grader label is not binary")
            integer = int(value)
            if integer not in (0, 1):
                raise ValueError(f"{spec.cell_id}: grader label is not binary")
            rows.append(identity.row_id(
                namespace=row_namespace,
                raw_identity=_raw_identity(
                    spec.dataset_id,
                    spec.temperature,
                    problem,
                    question_fingerprint,
                    candidate_index,
                ),
            ))
            groups.append(group_id)
            incorrect.append(1 - integer)
    order = np.asarray(sorted(range(len(rows)), key=lambda index: rows[index]), dtype=np.int64)
    ordered_rows = tuple(rows[index] for index in order.tolist())
    ordered_groups = tuple(groups[index] for index in order.tolist())
    labels = np.asarray(incorrect, dtype=np.int8)[order]
    commitment = _payload_sha256([
        {"row_id": row_id, "group_id": group_id}
        for row_id, group_id in zip(ordered_rows, ordered_groups)
    ])
    return (
        ordered_rows,
        ordered_groups,
        labels,
        commitment,
        _payload_sha256(question_fingerprints),
    )


def _cohort_id(row_ids: Sequence[str], group_ids: Sequence[str]) -> str:
    return "cohort::" + _payload_sha256([
        {"row_id": row_id, "group_id": group_id}
        for row_id, group_id in sorted(zip(row_ids, group_ids))
    ])


def _aggregate_cohort_id(components: Sequence[tuple[str, str]]) -> str:
    return "cohort::" + _payload_sha256([
        {"component_id": component, "cohort_id": cohort}
        for component, cohort in sorted(components)
    ])


def _common(
    *, release_id: str, run_id: str, spec: EdisCellSpec | None,
    dataset_id: str, population_id: str, cell_id: str, slice_id: str,
    cohort_id: str, method_id: str, method_version_id: str,
    comparison_group_id: str, status_detail: str = EVIDENCE_STATUS_DETAIL,
) -> dict[str, Any]:
    return {
        "release_id": release_id,
        "run_id": run_id,
        "lane_id": "edis_aime_reconstruction_v1",
        "task_id": "final_answer_detection_on_multi_sample_traces",
        "dataset_id": dataset_id,
        "population_id": population_id,
        "cell_id": cell_id,
        "slice_id": slice_id,
        "cohort_id": cohort_id,
        "method_id": method_id,
        "method_version_id": method_version_id,
        "adapter_id": ADAPTER_ID,
        "system_id": f"{method_id}::{ADAPTER_ID}",
        "comparison_group_id": comparison_group_id,
        "feature_contract_id": "dufs-liu-mixed-v2-development-2026-08-07",
        "access_contract_id": ACCESS_CONTRACT_ID,
        "evaluator_id": EVALUATOR_ID,
        "evidence_grade": "context",
        "status": "CONTEXT_ONLY",
        "status_detail": status_detail,
    }


def _prediction_row(
    *,
    base: Mapping[str, Any],
    row_id: str,
    group_id: str,
    score: float,
    label: bool | int | np.bool_ | np.integer,
    fallback_used: bool,
    score_hash: str,
) -> dict[str, Any]:
    """Build one schema-exact binary EDIS prediction record.

    EDIS metric computation intentionally retains its signed ``int8`` label
    vectors.  The reporting prediction contract is different: ``label`` is a
    nullable Boolean column.  Validate the binary value before representing it
    as ``bool`` so CSV and Parquet serialize the same scientific value.
    """

    if (
        type(label) not in (bool, int, np.bool_)
        and not isinstance(label, np.integer)
    ):
        raise TypeError("EDIS prediction label must be a bool/integer binary value")
    integer_label = int(label)
    if integer_label not in (0, 1):
        raise ValueError("EDIS prediction label must be binary")
    return {
        **dict(base),
        "row_id": row_id,
        "group_id": group_id,
        "continuous_score": float(score),
        "discrete_prediction": None,
        "label": bool(integer_label),
        "eligible": True,
        "fallback_used": bool(fallback_used),
        "score_hash": score_hash,
    }


def _metric_rows(
    *, interval: Mapping[str, Any], base_by_method: Mapping[str, Mapping[str, Any]],
    aggregation_id: str, aggregation_level: str, component_ids: Sequence[str],
    n_rows: int, n_groups: int, n_positive: int, n_negative: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels = {"auroc": "AUROC", "auprc": "AUPRC"}
    for method_id in PRIMARY_METHOD_IDS:
        for metric_id in ("auroc", "auprc"):
            result = interval["metrics"][method_id][metric_id]
            output.append({
                **dict(base_by_method[method_id]),
                "aggregation_id": aggregation_id,
                "aggregation_level": aggregation_level,
                "metric_id": metric_id,
                "metric_label": labels[metric_id],
                "metric_unit": "probability",
                "positive_class": "incorrect",
                "better_direction": "higher",
                "value": result["value"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "n_rows": n_rows,
                "n_groups": n_groups,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "bootstrap_unit": str(interval["bootstrap_unit"]),
                "bootstrap_draws": int(interval["draws_requested"]),
                "is_primary": metric_id == "auroc",
                "fidelity": "local_common_saved_trace_replay_descriptive",
                "component_ids": list(component_ids),
            })
    return output


def _contrast_rows(
    *, interval: Mapping[str, Any], base_by_method: Mapping[str, Mapping[str, Any]],
    aggregation_id: str, aggregation_level: str, n_pairs: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method_id in PRIMARY_METHOD_IDS:
        if method_id == "iu_pcr":
            continue
        for metric_id in ("auroc", "auprc"):
            result = interval["contrasts"][method_id][metric_id]
            wins = ties = losses = 0
            cell_points = interval.get("cell_point_metrics")
            if isinstance(cell_points, Mapping):
                deltas = [
                    float(row[method_id][metric_id]) - float(row["iu_pcr"][metric_id])
                    for row in cell_points.values()
                ]
                wins = sum(delta > 0 for delta in deltas)
                ties = sum(delta == 0 for delta in deltas)
                losses = sum(delta < 0 for delta in deltas)
            elif n_pairs == 1:
                delta = float(result["delta"])
                wins, ties, losses = int(delta > 0), int(delta == 0), int(delta < 0)
            output.append({
                **dict(base_by_method[method_id]),
                "aggregation_id": aggregation_id,
                "aggregation_level": aggregation_level,
                "metric_id": metric_id,
                "metric_unit": "probability",
                "positive_class": "incorrect",
                "better_direction": "higher",
                "left_system_id": base_by_method[method_id]["system_id"],
                "right_system_id": base_by_method["iu_pcr"]["system_id"],
                "delta": result["delta"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "n_pairs": n_pairs,
                "bootstrap_unit": str(interval["bootstrap_unit"]),
                "bootstrap_draws": int(interval["draws_requested"]),
                "paired": True,
                "fidelity": "local_common_saved_trace_replay_descriptive",
            })
    return output


def _validate_partial_status_roster(
    *, certificate: Mapping[str, Any], preparation: EdisPreparationRegistry
) -> dict[str, str]:
    """Require the signed, preregistered 4-ready/8-short-cell roster."""

    expected = preparation.expected_status_by_cell
    contract = preparation.raw.get("target_free_status_roster_contract")
    if expected is None or not isinstance(contract, Mapping):
        raise RuntimeError("descriptive partial EDIS evaluation lacks a frozen status roster")
    rows = certificate.get("registered_cell_statuses")
    if not isinstance(rows, list):
        raise RuntimeError("descriptive partial EDIS certificate lacks cell statuses")
    observed_ids = [str(row.get("cell_id", "")) for row in rows]
    observed = {str(row.get("cell_id", "")): str(row.get("status", "")) for row in rows}
    ready = [cell_id for cell_id in observed_ids if observed[cell_id] == "READY"]
    blocked = [cell_id for cell_id in observed_ids if observed[cell_id] != "READY"]
    if (
        certificate.get("scientific_full") is not False
        or certificate.get("descriptive_partial") is not True
        or certificate.get("headline_eligible") is not False
        or certificate.get("aggregate_metrics_allowed") is not False
        or certificate.get("cell_ids") != ready
        or observed_ids != [cell.cell_id for cell in preparation.cells]
        or observed != expected
        or ready != list(contract["ready_cell_ids"])
        or blocked != [str(row["cell_id"]) for row in contract["blocked_cells"]]
        or int(certificate.get("registered_cell_count", -1)) != 12
        or int(certificate.get("ready_cell_count", -1)) != 4
        or int(certificate.get("blocked_cell_count", -1)) != 8
    ):
        raise RuntimeError("descriptive partial EDIS 4-ready/8-blocked roster drifted")
    return observed


def _remove_verified_empty_directory_tree(path: Path) -> None:
    """Remove only a file-free retry scaffold; preserve every material tree."""

    if path.is_symlink():
        raise FileExistsError(f"EDIS evaluation target is not a directory: {path}")
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(f"EDIS evaluation target is not a directory: {path}")
    descendants = sorted(
        path.rglob("*"), key=lambda item: len(item.parts), reverse=True,
    )
    unexpected = [
        item for item in descendants if item.is_symlink() or not item.is_dir()
    ]
    if unexpected:
        raise FileExistsError(
            "EDIS evaluation directory contains material output: "
            + ", ".join(str(item) for item in unexpected[:5])
        )
    for directory in descendants:
        directory.rmdir()
    path.rmdir()


class _AtomicEvaluationStage:
    """Same-filesystem unpublished tree committed by one atomic rename."""

    def __init__(self, final_path: Path) -> None:
        self.final_path = final_path
        self.path = Path(tempfile.mkdtemp(
            prefix=f".{final_path.name}.staging-",
            dir=final_path.parent,
        ))
        self.committed = False

    def commit(self) -> None:
        if self.committed:
            raise RuntimeError("EDIS evaluation stage was already committed")
        if self.final_path.exists() or self.final_path.is_symlink():
            raise FileExistsError(
                f"EDIS evaluation directory already exists: {self.final_path}"
            )
        try:
            os.replace(self.path, self.final_path)
        except OSError as error:
            # Close the check/rename race with another evaluator while keeping
            # the already-published winner and this unpublished stage intact.
            if self.final_path.exists() or self.final_path.is_symlink():
                raise FileExistsError(
                    f"EDIS evaluation directory already exists: {self.final_path}"
                ) from error
            raise
        self.committed = True

    def cleanup(self) -> None:
        if not self.committed and self.path.exists():
            shutil.rmtree(self.path)


def _persist_evaluation(
    *,
    release_id: str,
    build_id: str,
    output: Path,
    fit: Path,
    preparation: EdisPreparationRegistry,
    postfreeze_registry_path: str | Path,
    identity: KeyedIdentityController,
    certificate: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    label_artifacts: list[dict[str, Any]],
    postfreeze: Mapping[str, Any],
    descriptive_partial: bool,
) -> Mapping[str, Any]:
    from spectral_utils.reconstruction_reporting.io import write_parquet, write_tidy_csv

    artifacts: dict[str, Mapping[str, Any]] = {}
    canonical_table_hashes: dict[str, str] = {}
    for table, rows in (
        ("predictions", predictions),
        ("metrics", metrics),
        ("contrasts", contrasts),
        ("coverage", coverage),
    ):
        csv_name = "metrics_long.csv" if table == "metrics" else (
            "contrasts_long.csv" if table == "contrasts" else (
                "coverage_long.csv" if table == "coverage" else "predictions.csv"
            )
        )
        parquet_name = "metrics_long.parquet" if table == "metrics" else (
            "contrasts_long.parquet" if table == "contrasts" else (
                "coverage_long.parquet" if table == "coverage" else "predictions.parquet"
            )
        )
        csv_record = write_tidy_csv(output / csv_name, table, rows)
        parquet_record = write_parquet(output / parquet_name, table, rows)
        artifacts[table] = {"csv": csv_record, "parquet": parquet_record}
        canonical_table_hashes[table] = canonical_evaluation_table_sha256(
            table=table, rows=rows, release_id=release_id, build_id=build_id,
        )
    status_rows = list(certificate["registered_cell_statuses"])
    ready_ids = set(map(str, certificate["cell_ids"]))
    gate_audit = [
        dict(item)
        if not descriptive_partial or str(item["cell_id"]) in ready_ids
        else {
            "cell_id": str(item["cell_id"]),
            "status": "NOT_EVALUATED_PREPARATION_BLOCKED",
            "target_summary_source": (
                "frozen_postfreeze_registry; raw row-label artifact not opened"
            ),
        }
        for item in postfreeze["cells"]
    ]
    manifest = {
        "schema_version": EVALUATION_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "lane_id": preparation.lane_id,
        "ab_certificate_sha256": certificate["certificate_sha256"],
        "score_freeze_sha256": sha256_file(fit / "SCORE_FREEZE_MANIFEST.json"),
        "preparation_registry_sha256": preparation.sha256,
        "preparation_status_commitment_sha256": certificate[
            "preparation_status_commitment_sha256"
        ],
        "postfreeze_registry_sha256": sha256_file(postfreeze_registry_path),
        "identity_contract": dict(identity.public_binding),
        "scientific_full": not descriptive_partial,
        "descriptive_partial": descriptive_partial,
        "partial_evaluation_mode": descriptive_partial,
        "registered_cell_count": 12,
        "ready_cell_count": len(certificate["cell_ids"]),
        "blocked_cell_count": 12 - len(certificate["cell_ids"]),
        "registered_cell_statuses": status_rows,
        "raw_row_labels_opened_cell_ids": list(certificate["cell_ids"]),
        "blocked_raw_row_label_artifacts_opened": False,
        "postfreeze_target_summary_registry_loaded": True,
        "labels_opened_only_after_score_freeze_and_ab_pass": True,
        "historical_scores_copied": False,
        "bootstrap_draws": 20_000,
        "bootstrap_unit": "source_question",
        "positive_class": "incorrect",
        "metrics": ["auroc", "auprc"],
        "aggregation": (
            "per_temperature_only; dataset_and_task_aggregates_forbidden_in_partial_release"
            if descriptive_partial
            else (
                "per_temperature; equal_temperature_per_dataset; "
                "equal_dataset_after_equal_temperature"
            )
        ),
        "dataset_task_aggregates_emitted": not descriptive_partial,
        "track": "multi_sample_inference",
        "access_contract_id": ACCESS_CONTRACT_ID,
        "headline_eligible": False,
        "publication_eligible": False,
        "evidence_status": (
            "DESCRIPTIVE_PARTIAL_GATE_FAILED"
            if descriptive_partial else "DESCRIPTIVE_GATE_FAILED"
        ),
        "gate_audit": gate_audit,
        "one_pass_leaderboard_combination_forbidden": True,
        "evaluation_ab_certificate_required_for_release": True,
        "artifacts": artifacts,
        "canonical_table_sha256": canonical_table_hashes,
        "label_artifacts": label_artifacts,
    }
    manifest["payload_sha256"] = _payload_sha256(manifest)
    atomic_write_json(output / "MANIFEST.json", manifest)
    return manifest


def _evaluate_in_stage(
    *,
    release_id: str,
    build_id: str,
    release_root: str | Path,
    private_control_root: str | Path,
    source_root: str | Path,
    preparation_registry_path: str | Path,
    postfreeze_registry_path: str | Path,
    identity: KeyedIdentityController,
    repo: str | Path,
    output: Path,
    certificate_path: str | Path | None = None,
    allow_descriptive_partial: bool = False,
) -> Mapping[str, Any]:
    preparation = load_preparation_registry(preparation_registry_path)
    release = Path(release_root) / release_id
    certificate_target = Path(certificate_path) if certificate_path else release / "edis" / "AB_VERIFICATION.json"
    certificate = assert_ab_certificate(
        path=certificate_target,
        release_id=release_id,
        release_root=release_root,
        selected_build=build_id,
        preparation_registry_path=preparation_registry_path,
        private_control_root=private_control_root,
        repo=repo,
        allow_descriptive_partial=allow_descriptive_partial,
    )
    descriptive_partial = certificate.get("descriptive_partial") is True
    status_by_cell = (
        _validate_partial_status_roster(
            certificate=certificate, preparation=preparation
        )
        if descriptive_partial
        else {cell.cell_id: "READY" for cell in preparation.cells}
    )
    # The post-freeze registry contains target-derived class counts and gate
    # outcomes.  Do not even parse it until the exact A/B certificate passes.
    postfreeze = load_postfreeze_registry(postfreeze_registry_path, preparation)
    if dict(identity.public_binding) != certificate.get("identity_contract"):
        raise RuntimeError("post-freeze identity key does not match the A/B certificate")
    lane = release / f"build_{build_id}" / "edis"
    inputs, fit = lane / "inputs", lane / "fit"
    if not output.is_dir() or any(output.iterdir()):
        raise RuntimeError(f"EDIS evaluation staging directory is not empty: {output}")
    private_path = Path(private_control_root) / release_id / "edis" / f"build_{build_id}" / "PREPARATION_PROVENANCE.json"
    private_policy_path = private_path.parent / "FIT_AUDIT_POLICY.json"
    fit_registry, freeze = validate_score_freeze(
        fit_root=fit,
        input_root=inputs,
        expected_build=build_id,
        repo=repo,
        private_audit_policy_path=private_policy_path,
    )
    if certificate["builds"][build_id]["score_freeze_sha256"] != sha256_file(fit / "SCORE_FREEZE_MANIFEST.json"):
        raise RuntimeError("selected EDIS score freeze differs from the A/B certificate")
    private = load_private_provenance(private_path)
    if certificate["builds"][build_id]["private_provenance_sha256"] != sha256_file(private_path):
        raise RuntimeError("selected EDIS private provenance differs from the A/B certificate")
    private_by_cell = {row["cell_id"]: row for row in private["cells"]}
    fit_cells = {row["cell_id"]: row for row in fit_registry["cells"]}
    post_by_cell = {row["cell_id"]: row for row in postfreeze["cells"]}
    records_by_cell: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in freeze["records"]:
        records_by_cell[str(row["cell_id"])][str(row["method_id"])] = row

    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    evaluation_cells: dict[str, dict[str, Any]] = {}
    label_artifacts: list[dict[str, Any]] = []
    run_id = f"{release_id}::edis::{build_id}::postfreeze"
    root = Path(source_root).resolve()
    certificate_status_by_cell = {
        str(row["cell_id"]): row
        for row in certificate["registered_cell_statuses"]
    }
    for spec in preparation.cells:
        if status_by_cell[spec.cell_id] != "READY":
            status_row = certificate_status_by_cell[spec.cell_id]
            cohort = "cohort::" + _payload_sha256({
                "cell_id": spec.cell_id,
                "preparation_status": status_by_cell[spec.cell_id],
                "preparation_status_commitment_sha256": certificate[
                    "preparation_status_commitment_sha256"
                ],
            })
            slice_id = "temperature_" + str(spec.temperature).replace(".", "p")
            detail = (
                f"{status_by_cell[spec.cell_id]}: "
                f"{status_row.get('status_detail', 'target-free preparation blocked')}; "
                "raw labels were not opened and no row was dropped or imputed"
            )
            comparison = f"edis_partial_blocked::{spec.cell_id}::{cohort}"
            for method_id in PRIMARY_METHOD_IDS:
                base = _common(
                    release_id=release_id, run_id=run_id, spec=spec,
                    dataset_id=spec.dataset_id, population_id=spec.population_id,
                    cell_id=spec.cell_id, slice_id=slice_id, cohort_id=cohort,
                    method_id=method_id,
                    method_version_id=PRIMARY_METHOD_SPECS[method_id].method_version_id,
                    comparison_group_id=comparison, status_detail=detail,
                )
                coverage.append({
                    **base,
                    "status": "INPUT_INVALID",
                    "status_detail": detail,
                    "expected_n": spec.expected_rows,
                    "eligible_n": 0,
                    "scored_n": 0,
                    "fallback_n": 0,
                    "excluded_n": 0,
                    "failed_n": spec.expected_rows,
                    "coverage_fraction": 0.0,
                })
            continue
        verify_pinned_file(
            root=root, relative=spec.source_path,
            expected_sha256=spec.source_sha256,
            expected_size=spec.source_size_bytes,
        )
        rows, groups, labels, group_commitment, question_commitment = reconstruct_labels_and_groups(
            spec=spec,
            source_path=(root / spec.source_path).resolve(),
            identity=identity,
        )
        if group_commitment != private_by_cell[spec.cell_id]["group_membership_commitment_sha256"]:
            raise RuntimeError(f"{spec.cell_id}: post-freeze group membership differs from preparation")
        if question_commitment != private_by_cell[spec.cell_id]["question_roster_commitment_sha256"]:
            raise RuntimeError(f"{spec.cell_id}: saved question roster differs from preparation")
        if _payload_sha256(list(rows)) != fit_cells[spec.cell_id]["row_roster_sha256"]:
            raise RuntimeError(f"{spec.cell_id}: post-freeze row roster differs from preparation")
        expected = post_by_cell[spec.cell_id]
        n_incorrect = int(labels.sum())
        n_correct = int(len(labels) - n_incorrect)
        if n_incorrect != int(expected["expected_incorrect"]) or n_correct != int(expected["expected_correct"]):
            raise RuntimeError(f"{spec.cell_id}: frozen class counts differ from the post-freeze registry")
        label_path = output / "labels" / f"{spec.cell_id}.npz"
        label_sha = atomic_write_npz(label_path, {
            "row_ids": np.asarray(rows, dtype="<U80"),
            "group_ids": np.asarray(groups, dtype="<U80"),
            "incorrect": labels.astype("i1"),
        })
        label_artifacts.append({
            "cell_id": spec.cell_id,
            "artifact_path": label_path.relative_to(output).as_posix(),
            "artifact_sha256": label_sha,
        })
        scores: dict[str, np.ndarray] = {}
        versions: dict[str, str] = {}
        score_hashes: dict[str, str] = {}
        statuses: dict[str, str] = {}
        for method_id in PRIMARY_METHOD_IDS:
            record = records_by_cell[spec.cell_id][method_id]
            bundle = load_npz_no_pickle(fit / record["score_path"])
            if set(bundle) != {"row_ids", "score"}:
                raise RuntimeError(f"{spec.cell_id}/{method_id}: score artifact has unexpected members")
            if tuple(map(str, bundle["row_ids"].tolist())) != rows:
                raise RuntimeError(f"{spec.cell_id}/{method_id}: score/label row order mismatch")
            score = np.asarray(bundle["score"], dtype=float)
            if score.shape != labels.shape or not np.isfinite(score).all():
                raise RuntimeError(f"{spec.cell_id}/{method_id}: invalid frozen score")
            scores[method_id] = score
            versions[method_id] = str(record["method_version_id"])
            score_hashes[method_id] = str(record["score_sha256"])
            statuses[method_id] = str(record["status"])
        interval = grouped_paired_bootstrap_auroc_auprc(
            labels=labels,
            scores_by_method=scores,
            group_ids=groups,
            reference_method="iu_pcr",
            draws=20_000,
            seed=int(_payload_sha256({"release": release_id, "cell": spec.cell_id, "axis": "edis_cell"})[:8], 16),
        )
        cohort = _cohort_id(rows, groups)
        slice_id = "temperature_" + str(spec.temperature).replace(".", "p")
        comparison = f"edis_multi_sample_descriptive::{spec.cell_id}::{cohort}"
        gate_detail = (
            "DESCRIPTIVE_GATE_FAILED: class_status="
            + str(expected["class_status"])
            + "; gate_reasons="
            + "; ".join(map(str, expected["gate_reasons"]))
            + ". Numeric results are context-only."
        )
        base_by_method = {
            method_id: _common(
                release_id=release_id, run_id=run_id, spec=spec,
                dataset_id=spec.dataset_id, population_id=spec.population_id,
                cell_id=spec.cell_id, slice_id=slice_id, cohort_id=cohort,
                method_id=method_id, method_version_id=versions[method_id],
                comparison_group_id=comparison,
                status_detail=gate_detail,
            )
            for method_id in PRIMARY_METHOD_IDS
        }
        for method_id in PRIMARY_METHOD_IDS:
            base = base_by_method[method_id]
            for row_id, group_id, score, label in zip(rows, groups, scores[method_id], labels):
                predictions.append(_prediction_row(
                    base=base,
                    row_id=row_id,
                    group_id=group_id,
                    score=float(score),
                    label=label,
                    fallback_used=statuses[method_id] == "OK_FALLBACK",
                    score_hash=score_hashes[method_id],
                ))
            coverage.append({
                **base,
                "expected_n": len(labels),
                "eligible_n": len(labels),
                "scored_n": len(labels),
                "fallback_n": len(labels) if statuses[method_id] == "OK_FALLBACK" else 0,
                "excluded_n": 0,
                "failed_n": 0,
                "coverage_fraction": 1.0,
            })
        metrics.extend(_metric_rows(
            interval=interval,
            base_by_method=base_by_method,
            aggregation_id=f"cell::{spec.cell_id}",
            aggregation_level="cell",
            component_ids=[spec.cell_id],
            n_rows=len(labels), n_groups=int(interval["n_groups"]),
            n_positive=n_incorrect, n_negative=n_correct,
        ))
        contrasts.extend(_contrast_rows(
            interval=interval, base_by_method=base_by_method,
            aggregation_id=f"cell::{spec.cell_id}", aggregation_level="cell", n_pairs=1,
        ))
        evaluation_cells[spec.cell_id] = {
            "spec": spec,
            "labels": labels,
            "group_ids": groups,
            "scores_by_method": scores,
            "versions": versions,
            "score_hashes": score_hashes,
            "cohort_id": cohort,
        }

    if descriptive_partial:
        if (
            list(evaluation_cells) != list(certificate["cell_ids"])
            or [row["cell_id"] for row in label_artifacts]
            != list(certificate["cell_ids"])
            or len(coverage) != 12 * len(PRIMARY_METHOD_IDS)
        ):
            raise RuntimeError("descriptive partial EDIS evaluation roster drifted")
        return _persist_evaluation(
            release_id=release_id, build_id=build_id, output=output, fit=fit,
            preparation=preparation,
            postfreeze_registry_path=postfreeze_registry_path,
            identity=identity, certificate=certificate,
            predictions=predictions, metrics=metrics, contrasts=contrasts,
            coverage=coverage, label_artifacts=label_artifacts,
            postfreeze=postfreeze, descriptive_partial=True,
        )

    # Equal-temperature dataset aggregates.  Source-question draws are linked
    # across all three temperatures, and rows are never pooled.
    dataset_components: list[tuple[str, str]] = []
    for dataset_id in ("aime24", "amc23", "gsm8k", "math500"):
        cell_ids = [cell.cell_id for cell in preparation.cells if cell.dataset_id == dataset_id]
        cells = {
            cell_id: {
                "labels": evaluation_cells[cell_id]["labels"],
                "group_ids": evaluation_cells[cell_id]["group_ids"],
                "scores_by_method": evaluation_cells[cell_id]["scores_by_method"],
            }
            for cell_id in cell_ids
        }
        interval = population_grouped_paired_bootstrap_auroc_auprc(
            cells=cells,
            link_keys={cell_id: dataset_id for cell_id in cell_ids},
            reference_method="iu_pcr", draws=20_000,
            seed=int(_payload_sha256({"release": release_id, "dataset": dataset_id, "axis": "equal_temperature"})[:8], 16),
            weighting="equal_cell",
        )
        cohort = _aggregate_cohort_id([
            (cell_id, evaluation_cells[cell_id]["cohort_id"]) for cell_id in cell_ids
        ])
        dataset_components.append((dataset_id, cohort))
        first = evaluation_cells[cell_ids[0]]
        population_id = first["spec"].population_id
        comparison = f"edis_multi_sample_descriptive::dataset::{dataset_id}::{cohort}"
        base_by_method = {
            method_id: _common(
                release_id=release_id, run_id=run_id, spec=None,
                dataset_id=dataset_id, population_id=population_id,
                cell_id="__dataset__", slice_id="equal_temperature",
                cohort_id=cohort, method_id=method_id,
                method_version_id=first["versions"][method_id],
                comparison_group_id=comparison,
            )
            for method_id in PRIMARY_METHOD_IDS
        }
        n_rows = sum(len(evaluation_cells[cell_id]["labels"]) for cell_id in cell_ids)
        n_positive = sum(int(evaluation_cells[cell_id]["labels"].sum()) for cell_id in cell_ids)
        metrics.extend(_metric_rows(
            interval=interval, base_by_method=base_by_method,
            aggregation_id=f"dataset::{dataset_id}::equal_temperature",
            aggregation_level="dataset", component_ids=cell_ids,
            n_rows=n_rows, n_groups=int(interval["n_resampling_groups"]),
            n_positive=n_positive, n_negative=n_rows - n_positive,
        ))
        contrasts.extend(_contrast_rows(
            interval=interval, base_by_method=base_by_method,
            aggregation_id=f"dataset::{dataset_id}::equal_temperature",
            aggregation_level="dataset", n_pairs=len(cell_ids),
        ))

    # Equal-dataset after equal-temperature.  Because every dataset contributes
    # exactly three temperature cells, equal-cell here is algebraically the
    # registered equal-temperature then equal-dataset estimand.
    all_cells = {
        cell.cell_id: {
            "labels": evaluation_cells[cell.cell_id]["labels"],
            "group_ids": evaluation_cells[cell.cell_id]["group_ids"],
            "scores_by_method": evaluation_cells[cell.cell_id]["scores_by_method"],
        }
        for cell in preparation.cells
    }
    overall = population_grouped_paired_bootstrap_auroc_auprc(
        cells=all_cells,
        link_keys={cell.cell_id: cell.dataset_id for cell in preparation.cells},
        reference_method="iu_pcr", draws=20_000,
        seed=int(_payload_sha256({"release": release_id, "axis": "equal_dataset_after_temperature"})[:8], 16),
        weighting="equal_cell",
    )
    overall_cohort = _aggregate_cohort_id(dataset_components)
    first = evaluation_cells[preparation.cells[0].cell_id]
    comparison = f"edis_multi_sample_descriptive::all_datasets::{overall_cohort}"
    base_by_method = {
        method_id: _common(
            release_id=release_id, run_id=run_id, spec=None,
            dataset_id="__four_datasets__", population_id="edis_all_four_descriptive_v1",
            cell_id="__task__", slice_id="equal_dataset_after_equal_temperature",
            cohort_id=overall_cohort, method_id=method_id,
            method_version_id=first["versions"][method_id],
            comparison_group_id=comparison,
        )
        for method_id in PRIMARY_METHOD_IDS
    }
    n_rows = sum(len(value["labels"]) for value in evaluation_cells.values())
    n_positive = sum(int(value["labels"].sum()) for value in evaluation_cells.values())
    metrics.extend(_metric_rows(
        interval=overall, base_by_method=base_by_method,
        aggregation_id="task::edis_all_four::equal_dataset_after_equal_temperature",
        aggregation_level="task",
        component_ids=[dataset for dataset, _ in dataset_components],
        n_rows=n_rows, n_groups=int(overall["n_resampling_groups"]),
        n_positive=n_positive, n_negative=n_rows - n_positive,
    ))
    contrasts.extend(_contrast_rows(
        interval=overall, base_by_method=base_by_method,
        aggregation_id="task::edis_all_four::equal_dataset_after_equal_temperature",
        aggregation_level="task", n_pairs=len(preparation.cells),
    ))

    return _persist_evaluation(
        release_id=release_id, build_id=build_id, output=output, fit=fit,
        preparation=preparation,
        postfreeze_registry_path=postfreeze_registry_path,
        identity=identity, certificate=certificate,
        predictions=predictions, metrics=metrics, contrasts=contrasts,
        coverage=coverage, label_artifacts=label_artifacts,
        postfreeze=postfreeze, descriptive_partial=False,
    )


def evaluate(
    *,
    release_id: str,
    build_id: str,
    release_root: str | Path,
    private_control_root: str | Path,
    source_root: str | Path,
    preparation_registry_path: str | Path,
    postfreeze_registry_path: str | Path,
    identity: KeyedIdentityController,
    repo: str | Path,
    certificate_path: str | Path | None = None,
    allow_descriptive_partial: bool = False,
) -> Mapping[str, Any]:
    """Build an EDIS evaluation privately, then publish the complete tree once."""

    final_output = (
        Path(release_root) / release_id / f"build_{build_id}" / "edis" / "evaluation"
    )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    _remove_verified_empty_directory_tree(final_output)
    stage = _AtomicEvaluationStage(final_output)
    try:
        manifest = _evaluate_in_stage(
            release_id=release_id,
            build_id=build_id,
            release_root=release_root,
            private_control_root=private_control_root,
            source_root=source_root,
            preparation_registry_path=preparation_registry_path,
            postfreeze_registry_path=postfreeze_registry_path,
            identity=identity,
            repo=repo,
            output=stage.path,
            certificate_path=certificate_path,
            allow_descriptive_partial=allow_descriptive_partial,
        )
        stage.commit()
        return manifest
    finally:
        stage.cleanup()


__all__ = [
    "ACCESS_CONTRACT_ID",
    "ADAPTER_ID",
    "EVALUATION_SCHEMA",
    "EVALUATOR_ID",
    "POSTFREEZE_SCHEMA",
    "evaluate",
    "load_postfreeze_registry",
    "reconstruct_labels_and_groups",
]
