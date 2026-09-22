"""Outcome-isolated evaluation and grouped bootstrap for actual LEASH stopping."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import hashlib
from io import StringIO
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from spectral_utils.fair_comparisons.stopping import (
    AQUA_PARSER_REVISION,
    GSM8K_PARSER_REVISION,
    grade_aqua_option,
    score_stopping_records,
)
from spectral_utils.paper_exact import evaluator as paper_evaluator

from .io import canonical_json_bytes
from .leash_contract import (
    AtomicLeashDirectory,
    EVALUATION_AB_SCHEMA,
    EVALUATION_SCHEMA,
    FIDELITY,
    FIT_AB_SCHEMA,
    FIT_ALLOWED_FIELDS,
    FIT_SCHEMA,
    PREPARATION_AB_SCHEMA,
    PRIVATE_OUTCOME_SCHEMA,
    SEARCHABLE_TABLES,
    LeashContractError,
    add_payload_sha256,
    assert_no_symlinks,
    bound_json_sha256,
    bound_tree_manifest,
    canonical_jsonl_bytes,
    load_registry,
    leash_tree_manifest,
    leash_tree_write_bytes,
    leash_tree_write_json,
    payload_sha256,
    parse_json_bytes,
    parse_jsonl_bytes,
    read_bound_bytes,
    require_physically_disjoint_trees,
    validate_fit_row,
    verify_payload,
    write_json_noreplace,
)
from .leash_fit import FIT_MANIFEST_FILENAME, POLICY_LEDGER_FILENAME, derive_policy_ledger
from .leash_fit import derive_source_bound_fit_contract
from .leash_preparation import OUTCOME_MANIFEST_FILENAME, OUTCOMES_FILENAME


TABLE_SCHEMA_FILENAME = "TABLE_SCHEMA.json"
EVALUATION_MANIFEST_FILENAME = "EVALUATION_MANIFEST.json"


def _require_certificates(
    *,
    source_root: str | Path,
    registry_path: str | Path,
    preparation_dir: str | Path,
    fit_dir: str | Path,
    private_dir: str | Path,
    preparation_ab_certificate: str | Path,
    fit_ab_certificate: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate the entire chain from current raw source before private labels open."""

    source_contract = derive_source_bound_fit_contract(
        source_root=source_root,
        registry_path=registry_path,
        preparation_ab_certificate=preparation_ab_certificate,
    )
    for name, path, expected in (
        ("LEASH public preparation", preparation_dir, source_contract["preparation"]["public_tree"]),
        ("LEASH fit", fit_dir, source_contract["fit_tree"]),
    ):
        assert_no_symlinks(path, name=name)
        if bound_tree_manifest(path, name=name) != expected:
            raise LeashContractError(f"{name} differs from exact current-source rederivation")

    if Path(fit_ab_certificate).is_symlink():
        raise LeashContractError("LEASH fit A/B certificate is a symlink")
    expected_fit_certificate = source_contract["certificate"]
    canonical_fit_certificate = canonical_json_bytes(expected_fit_certificate) + b"\n"
    observed_fit_certificate_bytes = read_bound_bytes(
        fit_ab_certificate, name="LEASH fit A/B certificate"
    )
    if observed_fit_certificate_bytes != canonical_fit_certificate:
        raise LeashContractError("LEASH fit certificate bytes are not canonical/source-bound")

    # The private tree is touched only after source, public prep, fit, and both exact
    # canonical certificates have passed the transitive chain above.
    assert_no_symlinks(private_dir, name="LEASH private outcomes")
    if bound_tree_manifest(
        private_dir, name="LEASH private outcomes"
    ) != source_contract["preparation"]["private_tree"]:
        raise LeashContractError("private outcome tree differs from exact current-source rederivation")
    return (
        source_contract["preparation"]["certificate"],
        expected_fit_certificate,
        source_contract,
    )


def _load_policy_ledger(
    fit_dir: str | Path,
    *,
    registry: Mapping[str, Any],
    expected_tree: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(fit_dir)
    expected_files = {item["path"]: item for item in expected_tree["files"]}
    expected_manifest_file = expected_files.get(FIT_MANIFEST_FILENAME, {})
    manifest_bytes = read_bound_bytes(
        root / FIT_MANIFEST_FILENAME,
        name="LEASH fit manifest",
        expected_bytes=expected_manifest_file.get("bytes"),
        expected_sha256=expected_manifest_file.get("sha256"),
    )
    manifest = parse_json_bytes(manifest_bytes, name="LEASH fit manifest")
    if manifest != expected_manifest:
        raise LeashContractError("LEASH fit manifest differs from current-source rederivation")
    if manifest.get("schema_version") != FIT_SCHEMA:
        raise LeashContractError("unexpected LEASH fit manifest schema")
    verify_payload(manifest, name="LEASH fit manifest")
    if (
        manifest.get("lane_id") != registry["lane_id"]
        or manifest.get("fidelity") != FIDELITY
        or manifest.get("fit_visible_targets") is not False
        or manifest.get("policy_execution_evaluated") is not True
        or manifest.get("actual_stopping_claim_eligible_for_ready_leash_cells") is not False
        or manifest.get("proxy_stopping") is not False
        or manifest.get("paper_exact_claim") is not False
        or manifest.get("conceptual_objective_reproduced_as_equation") is not False
        or manifest.get("matched_accuracy_claim") is not False
    ):
        raise LeashContractError("LEASH fit claim boundary drifted")
    path = root / POLICY_LEDGER_FILENAME
    expected_ledger_file = expected_files.get(POLICY_LEDGER_FILENAME, {})
    ledger_bytes = read_bound_bytes(
        path,
        name="LEASH policy ledger",
        expected_bytes=expected_ledger_file.get("bytes"),
        expected_sha256=expected_ledger_file.get("sha256"),
    )
    if hashlib.sha256(ledger_bytes).hexdigest() != manifest.get("files", {}).get(POLICY_LEDGER_FILENAME):
        raise LeashContractError("LEASH policy ledger hash failed")
    ledger = parse_jsonl_bytes(ledger_bytes, name="LEASH policy ledger")
    if len(ledger) != int(manifest.get("n_rows", -1)):
        raise LeashContractError("LEASH policy ledger count drifted")
    base_rows: list[dict[str, Any]] = []
    for row in ledger:
        base = {field: row[field] for field in FIT_ALLOWED_FIELDS}
        validate_fit_row(base)
        expected_extra = {
            "forced_closure": bool(base["stopped_early"] and base["closure_generated"]),
            "policy_event_verified": bool(
                base["arm"] == "leash" and base["stopped_early"] and base["stop_reason"] == "policy"
            ),
            "actual_stopping_claim_eligible": False,
            "proxy_stopping": False,
        }
        if set(row) != set(FIT_ALLOWED_FIELDS) | set(expected_extra) or any(
            row[name] != value for name, value in expected_extra.items()
        ):
            raise LeashContractError(f"LEASH policy ledger derived-field drift: {row.get('row_id')}")
        base_rows.append(base)
    rederived, audit = derive_policy_ledger(base_rows)
    if rederived != ledger or audit != manifest.get("audit"):
        raise LeashContractError("LEASH policy ledger failed transitive rederivation")
    return ledger, manifest


def _load_private_outcomes(
    private_dir: str | Path,
    *,
    ledger: Sequence[Mapping[str, Any]],
    expected_tree: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(private_dir)
    expected_files = {item["path"]: item for item in expected_tree["files"]}
    expected_manifest_file = expected_files.get(OUTCOME_MANIFEST_FILENAME, {})
    manifest_bytes = read_bound_bytes(
        root / OUTCOME_MANIFEST_FILENAME,
        name="LEASH private outcome manifest",
        expected_bytes=expected_manifest_file.get("bytes"),
        expected_sha256=expected_manifest_file.get("sha256"),
    )
    manifest = parse_json_bytes(manifest_bytes, name="LEASH private outcome manifest")
    if manifest.get("schema_version") != PRIVATE_OUTCOME_SCHEMA:
        raise LeashContractError("unexpected LEASH private outcome schema")
    verify_payload(manifest, name="LEASH private outcome manifest")
    path = root / OUTCOMES_FILENAME
    expected_outcomes_file = expected_files.get(OUTCOMES_FILENAME, {})
    outcome_bytes = read_bound_bytes(
        path,
        name="LEASH private outcomes",
        expected_bytes=expected_outcomes_file.get("bytes"),
        expected_sha256=expected_outcomes_file.get("sha256"),
    )
    if hashlib.sha256(outcome_bytes).hexdigest() != manifest.get("files", {}).get(OUTCOMES_FILENAME):
        raise LeashContractError("LEASH private outcome hash failed")
    outcomes = parse_jsonl_bytes(outcome_bytes, name="LEASH private outcomes")
    if len(outcomes) != int(manifest.get("n_rows", -1)):
        raise LeashContractError("LEASH private outcome count drifted")
    row_ids = [row["row_id"] for row in outcomes]
    if row_ids != [row["row_id"] for row in ledger]:
        raise LeashContractError("LEASH private/fit row order or coverage drifted")
    if payload_sha256(row_ids) != manifest.get("row_order_sha256"):
        raise LeashContractError("LEASH private outcome row-order hash failed")
    base_rows = [{field: row[field] for field in FIT_ALLOWED_FIELDS} for row in ledger]
    if hashlib.sha256(canonical_jsonl_bytes(base_rows)).hexdigest() != manifest.get("public_fit_input_sha256"):
        raise LeashContractError("LEASH private outcomes do not bind the public fit input")
    return outcomes, manifest


def _grade(dataset: str, answer_text: Any, gold_answer: Any) -> dict[str, Any]:
    if dataset == "aqua":
        return grade_aqua_option(None if answer_text is None else str(answer_text), str(gold_answer))
    if dataset == "gsm8k":
        result = paper_evaluator.grade_math(str(answer_text or ""), str(gold_answer))
        return {**result, "parser_revision": GSM8K_PARSER_REVISION}
    raise LeashContractError(f"unsupported evaluation dataset {dataset!r}")


def _labeled_rows(
    ledger: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changes = 0
    parser_counts: dict[str, int] = defaultdict(int)
    for policy, outcome in zip(ledger, outcomes, strict=True):
        if (
            policy["row_id"] != outcome.get("row_id")
            or policy["trace_key"] != outcome.get("trace_key")
            or policy["source_artifact_sha256"] != outcome.get("source_artifact_sha256")
        ):
            raise LeashContractError(f"LEASH outcome join failed at {policy['row_id']}")
        grade = _grade(str(policy["dataset"]), outcome.get("answer_text"), outcome.get("gold_answer"))
        correct = bool(grade["correct"])
        stored_correct = bool(outcome.get("stored_correct"))
        changes += int(correct != stored_correct)
        parser_counts[str(grade["parse_status"])] += 1
        rows.append(
            {
                **dict(policy),
                "actual_stopping_claim_eligible": policy["arm"] == "leash",
                "gold_answer": str(grade["gold_answer"]),
                "prediction": grade["pred_answer"],
                "correct": correct,
                "parse_status": str(grade["parse_status"]),
                "parser_revision": str(grade["parser_revision"]),
                "parser_failure": grade["pred_answer"] is None,
                "stored_correct": stored_correct,
                "stored_prediction": outcome.get("stored_prediction"),
                "stored_parse_status": outcome.get("stored_parse_status"),
                "correctness_changed_from_stored": correct != stored_correct,
            }
        )
    return rows, {
        "n_rows_reparsed": len(rows),
        "n_correctness_changes_from_stored": changes,
        "parse_status_counts": dict(sorted(parser_counts.items())),
        "raw_answer_text_released": False,
        "gold_consulted_only_after_fit_ab_passed": True,
        "aqua_parser_revision": AQUA_PARSER_REVISION,
        "gsm8k_parser_revision": GSM8K_PARSER_REVISION,
    }


METRIC_GETTERS: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "pass_at_1": lambda row: float(bool(row["correct"])),
    "mean_reasoning_tokens": lambda row: float(row["n_reasoning_tokens"]),
    "mean_closure_tokens": lambda row: float(row["n_closure_tokens"]),
    "mean_total_tokens": lambda row: float(row["n_total_tokens"]),
    "mean_wall_s": lambda row: float(row["wall_s"]),
    "early_stop_rate": lambda row: float(bool(row["stopped_early"])),
    "forced_closure_rate": lambda row: float(bool(row["forced_closure"])),
    "parser_failure_rate": lambda row: float(bool(row["parser_failure"])),
}


def _percentile(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], *, n_boot: int, seed: int
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    output: list[dict[str, Any]] = []
    dataset_replicates: dict[tuple[str, str, str], np.ndarray] = {}
    dataset_points: dict[tuple[str, str, str], float] = {}
    dataset_contrast_replicates: dict[tuple[str, str, str], np.ndarray] = {}
    dataset_contrast_points: dict[tuple[str, str, str], float] = {}

    def append(
        *, scope: str, dataset: str | None, model: str | None, arm: str,
        metric: str, point: float, replicates: np.ndarray, n_groups: int,
        reference_arm: str | None = None,
    ) -> None:
        low, high = _percentile(replicates)
        output.append(
            {
                "scope": scope,
                "dataset": dataset,
                "model": model,
                "arm": arm,
                "reference_arm": reference_arm,
                "metric": metric,
                "point": float(point),
                "lo": low,
                "hi": high,
                "n_groups": int(n_groups),
                "n_boot": int(n_boot),
                "seed": int(seed),
                "grouping": "source_question_stratified_within_dataset_shared_across_arms_and_models",
            }
        )

    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        revisions = {str(row["dataset_revision"]) for row in dataset_rows}
        if len(revisions) != 1:
            raise LeashContractError(f"bootstrap dataset revision drift for {dataset}")
        revision = next(iter(revisions))
        question_ids = sorted({str(row["question_id"]) for row in dataset_rows})
        models = sorted({str(row["model"]) for row in dataset_rows})
        lookup = {
            (str(row["model"]), str(row["arm"]), str(row["question_id"])): row
            for row in dataset_rows
        }
        if len(lookup) != len(question_ids) * len(models) * 3:
            raise LeashContractError(f"bootstrap population is not rectangular for {dataset}")
        offset = int(hashlib.sha256(f"{revision}::{dataset}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng((int(seed) + offset) % (2**32))
        draws = rng.integers(0, len(question_ids), size=(int(n_boot), len(question_ids)))
        cell_rep: dict[tuple[str, str, str], np.ndarray] = {}
        cell_point: dict[tuple[str, str, str], float] = {}
        for model in models:
            for arm in ("cot", "leash", "nocot"):
                for metric, getter in METRIC_GETTERS.items():
                    vector = np.asarray(
                        [getter(lookup[(model, arm, question)]) for question in question_ids],
                        dtype=float,
                    )
                    replicates = vector[draws].mean(axis=1)
                    point = float(np.mean(vector))
                    cell_rep[(model, arm, metric)] = replicates
                    cell_point[(model, arm, metric)] = point
                    append(
                        scope="cell", dataset=dataset, model=model, arm=arm,
                        metric=metric, point=point, replicates=replicates,
                        n_groups=len(question_ids),
                    )
        for model in models:
            for arm in ("leash", "nocot"):
                for metric in METRIC_GETTERS:
                    reps = cell_rep[(model, arm, metric)] - cell_rep[(model, "cot", metric)]
                    point = cell_point[(model, arm, metric)] - cell_point[(model, "cot", metric)]
                    append(
                        scope="cell", dataset=dataset, model=model, arm=arm,
                        reference_arm="cot", metric=f"{metric}_delta_vs_cot", point=point,
                        replicates=reps, n_groups=len(question_ids),
                    )
                arm_tokens = cell_rep[(model, arm, "mean_total_tokens")]
                cot_tokens = cell_rep[(model, "cot", "mean_total_tokens")]
                reps = 1.0 - arm_tokens / cot_tokens
                point = 1.0 - cell_point[(model, arm, "mean_total_tokens")] / cell_point[(model, "cot", "mean_total_tokens")]
                append(
                    scope="cell", dataset=dataset, model=model, arm=arm,
                    reference_arm="cot", metric="token_reduction_vs_cot", point=point,
                    replicates=reps, n_groups=len(question_ids),
                )

        for arm in ("cot", "leash", "nocot"):
            for metric in METRIC_GETTERS:
                reps = np.mean([cell_rep[(model, arm, metric)] for model in models], axis=0)
                point = float(np.mean([cell_point[(model, arm, metric)] for model in models]))
                dataset_replicates[(dataset, arm, metric)] = reps
                dataset_points[(dataset, arm, metric)] = point
                append(
                    scope="equal_model_within_dataset", dataset=dataset, model=None,
                    arm=arm, metric=metric, point=point, replicates=reps,
                    n_groups=len(question_ids),
                )
        for arm in ("leash", "nocot"):
            for metric in METRIC_GETTERS:
                reps = dataset_replicates[(dataset, arm, metric)] - dataset_replicates[(dataset, "cot", metric)]
                point = dataset_points[(dataset, arm, metric)] - dataset_points[(dataset, "cot", metric)]
                dataset_contrast_replicates[(dataset, arm, metric)] = reps
                dataset_contrast_points[(dataset, arm, metric)] = point
                append(
                    scope="equal_model_within_dataset", dataset=dataset, model=None,
                    arm=arm, reference_arm="cot", metric=f"{metric}_delta_vs_cot",
                    point=point, replicates=reps, n_groups=len(question_ids),
                )
            arm_tokens = dataset_replicates[(dataset, arm, "mean_total_tokens")]
            cot_tokens = dataset_replicates[(dataset, "cot", "mean_total_tokens")]
            reps = 1.0 - arm_tokens / cot_tokens
            point = 1.0 - dataset_points[(dataset, arm, "mean_total_tokens")] / dataset_points[(dataset, "cot", "mean_total_tokens")]
            dataset_contrast_replicates[(dataset, arm, "token_reduction")] = reps
            dataset_contrast_points[(dataset, arm, "token_reduction")] = point
            append(
                scope="equal_model_within_dataset", dataset=dataset, model=None,
                arm=arm, reference_arm="cot", metric="token_reduction_vs_cot",
                point=point, replicates=reps, n_groups=len(question_ids),
            )

    datasets = sorted(by_dataset)
    total_groups = sum(len({str(row["question_id"]) for row in by_dataset[d]}) for d in datasets)
    for arm in ("cot", "leash", "nocot"):
        for metric in METRIC_GETTERS:
            reps = np.mean([dataset_replicates[(dataset, arm, metric)] for dataset in datasets], axis=0)
            point = float(np.mean([dataset_points[(dataset, arm, metric)] for dataset in datasets]))
            append(
                scope="equal_dataset_after_equal_model", dataset=None, model=None,
                arm=arm, metric=metric, point=point, replicates=reps, n_groups=total_groups,
            )
    for arm in ("leash", "nocot"):
        for metric in METRIC_GETTERS:
            reps = np.mean([dataset_contrast_replicates[(dataset, arm, metric)] for dataset in datasets], axis=0)
            point = float(np.mean([dataset_contrast_points[(dataset, arm, metric)] for dataset in datasets]))
            append(
                scope="equal_dataset_after_equal_model", dataset=None, model=None,
                arm=arm, reference_arm="cot", metric=f"{metric}_delta_vs_cot",
                point=point, replicates=reps, n_groups=total_groups,
            )
        reps = np.mean([dataset_contrast_replicates[(dataset, arm, "token_reduction")] for dataset in datasets], axis=0)
        point = float(np.mean([dataset_contrast_points[(dataset, arm, "token_reduction")] for dataset in datasets]))
        append(
            scope="equal_dataset_after_equal_model", dataset=None, model=None,
            arm=arm, reference_arm="cot", metric="token_reduction_vs_cot",
            point=point, replicates=reps, n_groups=total_groups,
        )
    return sorted(
        output,
        key=lambda row: (
            row["scope"], str(row["dataset"]), str(row["model"]), row["arm"],
            str(row["reference_arm"]), row["metric"],
        ),
    )


def _augment_metrics(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in metrics:
        n = int(row["n_questions"])
        output.append(
            {
                **dict(row),
                "mean_reasoning_tokens": float(row["reasoning_tokens"]) / n,
                "mean_closure_tokens": float(row["closure_tokens"]) / n,
                "mean_total_tokens": float(row["mean_tokens_per_question"]),
                "actual_stopping_claim_eligible": row["arm"] == "leash",
            }
        )
    return output


def _contrasts(cell_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["dataset"], row["model"], row["arm"]): row for row in cell_metrics}
    output = []
    for dataset, model, arm in sorted(lookup):
        if arm == "cot":
            continue
        row, cot = lookup[(dataset, model, arm)], lookup[(dataset, model, "cot")]
        output.append(
            {
                "dataset": dataset, "model": model, "cell_id": row["cell_id"],
                "arm": arm, "reference_arm": "cot", "contrast_direction": "arm_minus_cot",
                "pass_at_1_delta_vs_cot": row["pass_at_1"] - cot["pass_at_1"],
                "mean_reasoning_tokens_delta_vs_cot": row["mean_reasoning_tokens"] - cot["mean_reasoning_tokens"],
                "mean_closure_tokens_delta_vs_cot": row["mean_closure_tokens"] - cot["mean_closure_tokens"],
                "mean_total_tokens_delta_vs_cot": row["mean_total_tokens"] - cot["mean_total_tokens"],
                "token_reduction_vs_cot": 1.0 - row["mean_total_tokens"] / cot["mean_total_tokens"],
                "mean_wall_s_delta_vs_cot": row["mean_wall_s"] - cot["mean_wall_s"],
                "early_stop_rate_delta_vs_cot": row["early_stop_rate"] - cot["early_stop_rate"],
                "forced_closure_rate_delta_vs_cot": row["forced_closure_rate"] - cot["forced_closure_rate"],
                "parser_failure_rate_delta_vs_cot": row["parser_failure_rate"] - cot["parser_failure_rate"],
                "matched_accuracy_claim": False,
            }
        )
    return output


def _aggregate_metrics(bootstrap_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scope": row["scope"], "dataset": row["dataset"],
            "arm": row["arm"], "metric": row["metric"], "value": row["point"],
            "fidelity": FIDELITY,
        }
        for row in bootstrap_rows
        if row["scope"] != "cell" and row["reference_arm"] is None
    ]


def _coverage(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for spec in registry["source_contract"]["ready_runs"]:
        rows.append(
            {
                "run_id": spec["run_id"], "dataset": spec["dataset"], "model": spec["model"],
                "coverage_status": "READY", "n_expected": spec["expected_traces"],
                "n_finished": spec["expected_traces"], "n_failed": 0,
                "actual_policy_execution_observed": True, "usable_for_evaluation": True,
                "actual_stopping_claim_eligible": True,
                "n_leash_rows_replayed": spec["expected_questions"],
                "n_leash_policy_stops": spec["expected_leash_policy_stops"],
                "n_policy_replay_mismatches": 0,
                "fidelity": FIDELITY, "reason": None,
            }
        )
    for spec in registry["source_contract"]["blocked_runs"]:
        rows.append(
            {
                "run_id": spec["run_id"], "dataset": spec["dataset"], "model": spec["model"],
                "coverage_status": "PROTOCOL_GATE_FAILED", "n_expected": spec["expected_traces"],
                "n_finished": 0, "n_failed": spec["expected_failed"],
                "actual_policy_execution_observed": False, "usable_for_evaluation": False,
                "actual_stopping_claim_eligible": False,
                "fidelity": FIDELITY,
                "reason": "all expected traces failed because the base tokenizer had no chat template",
            }
        )
    return sorted(rows, key=lambda row: row["run_id"])


def derive_leash_evaluation(
    *,
    source_root: str | Path,
    registry_path: str | Path,
    preparation_dir: str | Path,
    fit_dir: str | Path,
    private_dir: str | Path,
    preparation_ab_certificate: str | Path,
    fit_ab_certificate: str | Path,
) -> dict[str, Any]:
    prep_cert, fit_cert, source_contract = _require_certificates(
        source_root=source_root,
        registry_path=registry_path,
        preparation_dir=preparation_dir,
        fit_dir=fit_dir,
        private_dir=private_dir,
        preparation_ab_certificate=preparation_ab_certificate,
        fit_ab_certificate=fit_ab_certificate,
    )
    registry = source_contract["registry"]
    ledger, fit_manifest = _load_policy_ledger(
        fit_dir,
        registry=registry,
        expected_tree=source_contract["fit_tree"],
        expected_manifest=source_contract["fit_manifest"],
    )
    if ledger != source_contract["ledger"] or fit_manifest != source_contract["fit_manifest"]:
        raise LeashContractError("LEASH fit changed after exact current-source verification")
    outcomes, outcome_manifest = _load_private_outcomes(
        private_dir,
        ledger=ledger,
        expected_tree=source_contract["preparation"]["private_tree"],
    )
    if bound_tree_manifest(fit_dir, name="LEASH fit final binding") != source_contract["fit_tree"]:
        raise LeashContractError("LEASH fit changed during bound ledger loading")
    if bound_tree_manifest(
        private_dir, name="LEASH private outcomes final binding"
    ) != source_contract["preparation"]["private_tree"]:
        raise LeashContractError("LEASH private outcomes changed during bound label loading")
    labeled, parser_audit = _labeled_rows(ledger, outcomes)
    bootstrap = registry["evaluation"]["bootstrap"]
    scored = score_stopping_records(
        labeled, n_boot=int(bootstrap["draws"]), seed=int(bootstrap["seed"])
    )
    cell_metrics = _augment_metrics(scored["cell_metrics"])
    grouped = _bootstrap_rows(
        labeled, n_boot=int(bootstrap["draws"]), seed=int(bootstrap["seed"])
    )
    per_question = [
        {
            key: row[key]
            for key in (
                "row_id", "group_id", "cell_id", "dataset_revision", "dataset", "question_id",
                "model", "model_revision", "arm", "method_id", "trace_key",
                "n_reasoning_tokens", "n_closure_tokens", "n_total_tokens", "wall_s",
                "stopped_early", "closure_generated", "forced_closure", "stop_reason",
                "policy_event_verified", "actual_stopping_claim_eligible", "gold_answer",
                "actual_policy_execution_observed", "policy_replay_verified",
                "policy_replay_fired", "policy_replay_stop_index", "closure_evidence_verified",
                "prediction", "correct", "parse_status", "parser_revision", "parser_failure",
                "stored_correct", "stored_prediction", "stored_parse_status",
                "correctness_changed_from_stored", "source_artifact_sha256", "fidelity",
            )
        }
        for row in labeled
    ]
    return {
        "tables": {
            "coverage": _coverage(registry),
            "per_question": per_question,
            "cell_metrics": cell_metrics,
            "contrasts": _contrasts(cell_metrics),
            "frontier": scored["accuracy_compute_frontier"],
            "aggregate_metrics": _aggregate_metrics(grouped),
            "bootstrap_intervals": grouped,
        },
        "parser_audit": parser_audit,
        "pairing_audit": scored["pairing_audit"],
        "fit_manifest": fit_manifest,
        "outcome_manifest": outcome_manifest,
        "preparation_certificate": prep_cert,
        "fit_certificate": fit_cert,
        "source_contract": source_contract,
    }


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (dict, list, tuple)):
        return canonical_json_bytes(value).decode("ascii")
    return value


def _normalize_table(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    if not rows:
        raise LeashContractError("searchable LEASH table cannot be empty")
    columns = sorted({str(key) for row in rows for key in row})
    normalized = [{column: _scalar(row.get(column)) for column in columns} for row in rows]
    return columns, normalized


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _parquet_bytes(
    columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> tuple[bytes, dict[str, str]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise LeashContractError("PyArrow is required by the LEASH searchable output contract") from error
    fields = []
    dtypes: dict[str, str] = {}
    for column in columns:
        values = [row[column] for row in rows if row[column] is not None]
        kinds = {
            "bool" if isinstance(value, bool)
            else "int" if isinstance(value, int)
            else "float" if isinstance(value, float)
            else "string" if isinstance(value, str)
            else type(value).__name__
            for value in values
        }
        if not kinds:
            raise LeashContractError(f"searchable column is entirely null: {column}")
        if kinds == {"bool"}:
            arrow_type, type_name = pa.bool_(), "bool"
        elif kinds <= {"int"}:
            arrow_type, type_name = pa.int64(), "int64"
        elif kinds <= {"int", "float"}:
            arrow_type, type_name = pa.float64(), "float64"
        elif kinds == {"string"}:
            arrow_type, type_name = pa.string(), "string"
        else:
            raise LeashContractError(f"mixed/unsupported searchable dtype for {column}: {sorted(kinds)}")
        fields.append(pa.field(column, arrow_type, nullable=any(row[column] is None for row in rows)))
        dtypes[column] = type_name
    schema = pa.schema(fields)
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table, sink, compression="zstd", use_dictionary=False, write_statistics=True,
        data_page_version="1.0",
    )
    return sink.getvalue().to_pybytes(), dtypes


def _write_evaluation_tree(
    *,
    stage_path: AtomicLeashDirectory,
    derived: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_path: str | Path,
) -> None:
    table_contract: dict[str, Any] = {}
    for table_name in SEARCHABLE_TABLES:
        rows = derived["tables"][table_name]
        columns, normalized = _normalize_table(rows)
        json_name, csv_name, parquet_name = (
            f"{table_name}.jsonl", f"{table_name}.csv", f"{table_name}.parquet"
        )
        parquet_bytes, dtypes = _parquet_bytes(columns, normalized)
        hashes = {
            json_name: leash_tree_write_bytes(
                stage_path, json_name, canonical_jsonl_bytes(normalized)
            ),
            csv_name: leash_tree_write_bytes(
                stage_path, csv_name, _csv_bytes(columns, normalized)
            ),
            parquet_name: leash_tree_write_bytes(stage_path, parquet_name, parquet_bytes),
        }
        table_contract[table_name] = {
            "row_count": len(rows), "columns": columns, "dtypes": dtypes, "files": hashes,
        }
    schema_bundle = add_payload_sha256(
        {
            "schema_version": "reconstruction-leash-searchable-tables-v1",
            "formats": ["jsonl", "csv", "parquet"],
            "tables": table_contract,
        }
    )
    schema_sha = leash_tree_write_json(stage_path, TABLE_SCHEMA_FILENAME, schema_bundle)
    manifest = add_payload_sha256(
        {
            "schema_version": EVALUATION_SCHEMA,
            "lane_id": registry["lane_id"],
            "fidelity": FIDELITY,
            "claim_status": "ACTUAL_POLICY_EXECUTION_EVALUATED_FOR_SIX_READY_CELLS",
            "claim_scope": "six ready model-by-dataset cells only; Mistral has coverage status only",
            "policy_execution_evaluated": True,
            "all_policy_stops_have_realized_closure": True,
            "proxy_stopping": False,
            "paper_exact_claim": False,
            "conceptual_objective_reproduced_as_equation": False,
            "matched_accuracy_claim": False,
            "frontier_interpretation": "accuracy versus realized total-token compute; no matched-accuracy claim",
            "aggregation": [
                "cell", "equal-model-within-dataset", "equal-dataset-after-equal-model"
            ],
            "cross_task_or_access_macro": False,
            "bootstrap": registry["evaluation"]["bootstrap"],
            "registry_sha256": bound_json_sha256(
                registry_path, registry, name="LEASH registry"
            ),
            "preparation_ab_certificate_sha256": hashlib.sha256(
                canonical_json_bytes(derived["preparation_certificate"]) + b"\n"
            ).hexdigest(),
            "preparation_ab_payload_sha256": derived["preparation_certificate"]["payload_sha256"],
            "fit_ab_certificate_sha256": hashlib.sha256(
                canonical_json_bytes(derived["fit_certificate"]) + b"\n"
            ).hexdigest(),
            "fit_ab_payload_sha256": derived["fit_certificate"]["payload_sha256"],
            "fit_manifest_payload_sha256": derived["fit_manifest"]["payload_sha256"],
            "outcome_manifest_payload_sha256": derived["outcome_manifest"]["payload_sha256"],
            "parser_audit": derived["parser_audit"],
            "policy_replay_audits": derived["source_contract"]["preparation"]["policy_replay_audits"],
            "pairing_audit": derived["pairing_audit"],
            "table_schema_sha256": schema_sha,
            "tables": table_contract,
        }
    )
    leash_tree_write_json(stage_path, EVALUATION_MANIFEST_FILENAME, manifest)


def evaluate_leash_build(
    *,
    source_root: str | Path,
    preparation_dir: str | Path,
    fit_dir: str | Path,
    private_dir: str | Path,
    preparation_ab_certificate: str | Path,
    fit_ab_certificate: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    derived = derive_leash_evaluation(
        source_root=source_root,
        registry_path=registry_path,
        preparation_dir=preparation_dir,
        fit_dir=fit_dir,
        private_dir=private_dir,
        preparation_ab_certificate=preparation_ab_certificate,
        fit_ab_certificate=fit_ab_certificate,
    )
    registry = load_registry(registry_path)
    stage = AtomicLeashDirectory(output_dir)
    try:
        _write_evaluation_tree(
            stage_path=stage,
            derived=derived,
            registry=registry,
            registry_path=registry_path,
        )
        tree = leash_tree_manifest(stage)
        stage.commit()
    finally:
        stage.cleanup()
    return {
        "status": "PASS",
        "output_dir": str(stage.final_path),
        "tree": tree,
        "rows_by_table": {name: len(rows) for name, rows in derived["tables"].items()},
    }


def verify_leash_evaluation_ab(
    *,
    source_root: str | Path,
    preparation_a: str | Path,
    fit_a: str | Path,
    private_a: str | Path,
    evaluation_a: str | Path,
    preparation_b: str | Path,
    fit_b: str | Path,
    private_b: str | Path,
    evaluation_b: str | Path,
    preparation_ab_certificate: str | Path,
    fit_ab_certificate: str | Path,
    registry_path: str | Path,
    certificate_path: str | Path,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    public_roots = {
        "LEASH preparation A": preparation_a,
        "LEASH fit A": fit_a,
        "LEASH evaluation A": evaluation_a,
        "LEASH preparation B": preparation_b,
        "LEASH fit B": fit_b,
        "LEASH evaluation B": evaluation_b,
    }
    for name, path in public_roots.items():
        assert_no_symlinks(path, name=name)
    physical_public = require_physically_disjoint_trees(public_roots)
    observed, expected = [], []
    row_counts = []
    for variant, preparation_dir, fit_dir, private_dir, evaluation_dir in (
        ("A", preparation_a, fit_a, private_a, evaluation_a),
        ("B", preparation_b, fit_b, private_b, evaluation_b),
    ):
        derived = derive_leash_evaluation(
            source_root=source_root,
            registry_path=registry_path,
            preparation_dir=preparation_dir,
            fit_dir=fit_dir,
            private_dir=private_dir,
            preparation_ab_certificate=preparation_ab_certificate,
            fit_ab_certificate=fit_ab_certificate,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"leash-eval-{variant}-rederive-",
            dir=Path(tempfile.gettempdir()).resolve(strict=True),
        ) as temporary:
            stage = AtomicLeashDirectory(Path(temporary) / "evaluation")
            try:
                _write_evaluation_tree(
                    stage_path=stage,
                    derived=derived,
                    registry=registry,
                    registry_path=registry_path,
                )
                actual_tree = physical_public[f"LEASH evaluation {variant}"]
                expected_tree = leash_tree_manifest(stage)
                if actual_tree != expected_tree:
                    raise LeashContractError(
                        f"LEASH evaluation {variant} differs from rederivation"
                    )
                observed.append(actual_tree)
                expected.append(expected_tree)
            finally:
                stage.cleanup()
        row_counts.append({name: len(rows) for name, rows in derived["tables"].items()})
    if observed[0] != observed[1] or expected[0] != expected[1] or row_counts[0] != row_counts[1]:
        raise LeashContractError("LEASH evaluation A/B outputs are not byte-identical")
    private_roots = {
        "LEASH private outcomes A": private_a,
        "LEASH private outcomes B": private_b,
    }
    for name, path in private_roots.items():
        assert_no_symlinks(path, name=name)
    physical_private = require_physically_disjoint_trees(private_roots)
    expected_private_tree = derived["source_contract"]["preparation"]["private_tree"]
    if any(tree != expected_private_tree for tree in physical_private.values()):
        raise LeashContractError(
            "LEASH private A/B tree changed before final certificate binding"
        )
    if require_physically_disjoint_trees(public_roots) != physical_public:
        raise LeashContractError("LEASH public A/B trees changed during verification")
    certificate = add_payload_sha256(
        {
            "schema_version": EVALUATION_AB_SCHEMA,
            "status": "PASS",
            "lane_id": registry["lane_id"],
            "registry_sha256": bound_json_sha256(
                registry_path, registry, name="LEASH registry"
            ),
            "preparation_ab_certificate_sha256": hashlib.sha256(
                canonical_json_bytes(derived["preparation_certificate"]) + b"\n"
            ).hexdigest(),
            "fit_ab_certificate_sha256": hashlib.sha256(
                canonical_json_bytes(derived["fit_certificate"]) + b"\n"
            ).hexdigest(),
            "evaluation_tree_sha256": {
                "A": observed[0]["tree_sha256"], "B": observed[1]["tree_sha256"],
            },
            "rederived_evaluation_tree_sha256": expected[0]["tree_sha256"],
            "rows_by_table": row_counts[0],
            "grouped_bootstrap_rederived": True,
            "private_outcomes_reparsed": True,
            "searchable_output_contract_verified": True,
            "byte_identical": True,
            "transitive_rederivation": True,
            "paper_exact_claim": False,
            "conceptual_objective_reproduced_as_equation": False,
            "matched_accuracy_claim": False,
        }
    )
    write_json_noreplace(certificate_path, certificate)
    return certificate


__all__ = [
    "EVALUATION_MANIFEST_FILENAME", "TABLE_SCHEMA_FILENAME", "derive_leash_evaluation",
    "evaluate_leash_build", "verify_leash_evaluation_ab",
]
