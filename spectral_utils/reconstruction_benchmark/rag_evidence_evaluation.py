"""Post-freeze evaluation for incomparable RAG evidence benchmark panels."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .io import canonical_json_bytes, sha256_bytes
from .rag_evidence_contract import (
    AtomicRagDirectory,
    EVALUATION_MANIFEST_FILENAME,
    EVALUATION_SCHEMA,
    PANEL_IDS,
    PREPARATION_MANIFEST_FILENAME,
    PRIVATE_LABEL_FILENAME,
    REFCHECKER_SETTINGS,
    SCORE_MANIFEST_FILENAME,
    RagEvidenceContractError,
    add_payload_sha256,
    load_private_labels,
    load_registry,
    payload_sha256,
    read_bound_file_bytes,
    validate_artifact_identifier,
    validate_source_binding,
    verify_payload,
)
from .rag_evidence_fit import load_scores, validate_score_arrays


EVALUATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/rag_evidence.json",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_contract.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_fit.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_ab.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_evaluation.py",
)
PREDICTION_COLUMNS = (
    "panel_id", "split", "subgroup", "method_id", "unit_id", "parent_id",
    "score", "prediction", "label", "bootstrap_group",
)
METRIC_COLUMNS = (
    "panel_id", "dataset", "unit", "access", "estimand", "split", "subgroup",
    "method_id", "metric", "value", "ci_low", "ci_high", "n", "n_groups",
    "positive_rate", "bootstrap_draws", "status",
)
CONTRAST_COLUMNS = (
    "panel_id", "split", "subgroup", "left_method", "right_method", "metric",
    "delta", "ci_low", "ci_high", "n", "n_groups", "bootstrap_draws", "status",
)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in columns})
    return stream.getvalue().encode("utf-8")


def _binary_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "auroc":
        return lambda y, s: float(roc_auc_score(y, s))
    if name == "auprc":
        return lambda y, s: float(average_precision_score(y, s))
    raise KeyError(name)


def _threeway_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    labels = ("Entailment", "Neutral", "Contradiction")
    if name == "accuracy":
        return lambda y, p: float(np.mean(y == p))
    if name == "macro_f1":
        return lambda y, p: float(f1_score(y, p, labels=labels, average="macro", zero_division=0))
    raise KeyError(name)


def grouped_interval(
    target: np.ndarray,
    value: np.ndarray,
    groups: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    draws: int,
    seed: int,
    require_two_classes: bool,
) -> dict[str, Any]:
    target = np.asarray(target)
    value = np.asarray(value)
    group_values = np.asarray(groups).astype(str)
    if len(target) != len(value) or len(target) != len(group_values) or not len(target):
        raise RagEvidenceContractError("grouped bootstrap input alignment failed")
    unique = np.unique(group_values)
    lookup = {group: np.flatnonzero(group_values == group) for group in unique}
    point_status = "OK"
    if require_two_classes and len(np.unique(target)) < 2:
        return {
            "value": "", "ci_low": "", "ci_high": "", "draws": 0,
            "status": "METRIC_UNDEFINED_SINGLE_CLASS",
        }
    point = metric(target, value)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(draws)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([lookup[group] for group in selected])
        if require_two_classes and len(np.unique(target[indexes])) < 2:
            continue
        result = float(metric(target[indexes], value[indexes]))
        if np.isfinite(result):
            samples.append(result)
    if not samples:
        return {
            "value": point, "ci_low": "", "ci_high": "", "draws": 0,
            "status": "BOOTSTRAP_UNDEFINED",
        }
    array = np.asarray(samples, dtype=float)
    return {
        "value": point,
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "draws": len(array),
        "status": point_status,
    }


def grouped_paired_delta(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *, draws: int, seed: int,
) -> dict[str, Any]:
    target = np.asarray(target)
    left, right = np.asarray(left, float), np.asarray(right, float)
    group_values = np.asarray(groups).astype(str)
    if not (len(target) == len(left) == len(right) == len(group_values)):
        raise RagEvidenceContractError("paired RAG contrast alignment failed")
    if len(np.unique(target)) < 2:
        return {"delta": "", "ci_low": "", "ci_high": "", "draws": 0, "status": "METRIC_UNDEFINED_SINGLE_CLASS"}
    unique = np.unique(group_values)
    lookup = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(draws)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([lookup[group] for group in selected])
        if len(np.unique(target[indexes])) < 2:
            continue
        samples.append(metric(target[indexes], left[indexes]) - metric(target[indexes], right[indexes]))
    array = np.asarray(samples, dtype=float)
    return {
        "delta": float(metric(target, left) - metric(target, right)),
        "ci_low": float(np.quantile(array, 0.025)) if len(array) else "",
        "ci_high": float(np.quantile(array, 0.975)) if len(array) else "",
        "draws": len(array),
        "status": "OK" if len(array) else "BOOTSTRAP_UNDEFINED",
    }


def _panel(registry: Mapping[str, Any], panel_id: str) -> Mapping[str, Any]:
    return next(row for row in registry["panels"] if row["panel_id"] == panel_id)


def _metric_row(
    *, registry: Mapping[str, Any], panel_id: str, split: str, subgroup: str,
    method_id: str, metric: str, summary: Mapping[str, Any], n: int,
    groups: Sequence[str], positive_rate: str | float,
) -> dict[str, Any]:
    panel = _panel(registry, panel_id)
    return {
        "panel_id": panel_id,
        "dataset": panel["dataset"],
        "unit": panel["unit"],
        "access": panel["access"],
        "estimand": panel["estimand"],
        "split": split,
        "subgroup": subgroup,
        "method_id": method_id,
        "metric": metric,
        "value": summary["value"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "n": n,
        "n_groups": len(set(map(str, groups))),
        "positive_rate": positive_rate,
        "bootstrap_draws": summary["draws"],
        "status": summary["status"],
    }


def _binary_panel(
    *, registry: Mapping[str, Any], panel_id: str, split: str, subgroup: str,
    method_id: str, ids: np.ndarray, labels: np.ndarray, scores: np.ndarray,
    groups: np.ndarray, parent_ids: np.ndarray | None, draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ids, groups = np.asarray(ids).astype(str), np.asarray(groups).astype(str)
    if not (len(ids) == len(labels) == len(scores) == len(groups)):
        raise RagEvidenceContractError(f"{panel_id}: score/label/group alignment failed")
    metrics = []
    for offset, metric_name in enumerate(("auroc", "auprc")):
        summary = grouped_interval(
            labels, scores, groups, _binary_metric(metric_name), draws=draws,
            seed=seed + offset, require_two_classes=True,
        )
        metrics.append(_metric_row(
            registry=registry, panel_id=panel_id, split=split, subgroup=subgroup,
            method_id=method_id, metric=metric_name, summary=summary, n=len(labels),
            groups=groups, positive_rate=float(labels.mean()),
        ))
    parents = ids if parent_ids is None else np.asarray(parent_ids).astype(str)
    predictions = [{
        "panel_id": panel_id, "split": split, "subgroup": subgroup,
        "method_id": method_id, "unit_id": unit_id, "parent_id": parent,
        "score": float(score), "prediction": "", "label": int(label),
        "bootstrap_group": group,
    } for unit_id, parent, score, label, group in zip(ids, parents, scores, labels, groups, strict=True)]
    return metrics, predictions


def _private_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output = {str(row["unit_id"]): row for row in rows}
    if len(output) != len(rows):
        raise RagEvidenceContractError("duplicate private RAG unit ID")
    return output


def _evaluate_ragtruth(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, predictions = [], []
    panel_by_unit = {
        "response": "ragtruth_evidence_contrast_answer",
        "sentence": "ragtruth_evidence_contrast_sentence",
        "token": "ragtruth_evidence_contrast_token",
    }
    for split_index, split in enumerate(("dev", "test")):
        private_rows = labels["splits"][split]
        response = _private_lookup(private_rows)
        response_ids = np.asarray(scores[f"rag_{split}_response_id"]).astype(str)
        if set(response_ids) != set(response):
            raise RagEvidenceContractError(f"RAGTruth {split} response roster drifted")
        response_y = np.asarray([response[item]["response_label"] for item in response_ids], int)
        response_groups = np.asarray([response[item]["source_id"] for item in response_ids], str)
        response_tasks = np.asarray(scores[f"rag_{split}_response_task"]).astype(str)
        expected_response_tasks = np.asarray(
            [response[item]["task_type"] for item in response_ids], str
        )
        if not np.array_equal(response_tasks, expected_response_tasks):
            raise RagEvidenceContractError(
                f"RAGTruth {split} response task/private binding drifted"
            )
        response_scores = np.asarray(scores[f"rag_{split}_response_score"], float)

        sentence_private = {
            sentence["unit_id"]: {**sentence, "source_id": row["source_id"], "task_type": row["task_type"]}
            for row in private_rows for sentence in row["sentence_labels"]
        }
        sentence_ids = np.asarray(scores[f"rag_{split}_sentence_id"]).astype(str)
        if set(sentence_ids) != set(sentence_private):
            raise RagEvidenceContractError(f"RAGTruth {split} sentence roster drifted")
        sentence_y = np.asarray([sentence_private[item]["label"] for item in sentence_ids], int)
        sentence_groups = np.asarray([sentence_private[item]["source_id"] for item in sentence_ids], str)
        sentence_tasks = np.asarray([sentence_private[item]["task_type"] for item in sentence_ids], str)
        sentence_scores = np.asarray(scores[f"rag_{split}_sentence_score"], float)

        token_parents = np.asarray(scores[f"rag_{split}_token_parent_id"]).astype(str)
        token_indexes = np.asarray(scores[f"rag_{split}_token_index"], int)
        observed_token_lattice = list(zip(
            token_parents.tolist(), token_indexes.tolist(), strict=True
        ))
        expected_token_lattice = [
            (str(row["unit_id"]), index)
            for row in private_rows
            for index in range(len(row["token_labels"]))
        ]
        if observed_token_lattice != expected_token_lattice:
            raise RagEvidenceContractError(
                f"RAGTruth {split} scorer-token lattice/private binding drifted"
            )
        token_ids = np.asarray([
            f"{parent}_t{index:05d}" for parent, index in zip(token_parents, token_indexes, strict=True)
        ], dtype="U")
        token_y = np.asarray([
            int(response[parent]["token_labels"][index])
            for parent, index in zip(token_parents, token_indexes, strict=True)
        ], int)
        token_groups = np.asarray([response[parent]["source_id"] for parent in token_parents], str)
        token_tasks = np.asarray([response[parent]["task_type"] for parent in token_parents], str)
        token_scores = np.asarray(scores[f"rag_{split}_token_score"], float)

        bundles = {
            "response": (response_ids, response_y, response_scores, response_groups, response_tasks, None),
            "sentence": (sentence_ids, sentence_y, sentence_scores, sentence_groups, sentence_tasks, None),
            "token": (token_ids, token_y, token_scores, token_groups, token_tasks, token_parents),
        }
        for unit_offset, (unit, bundle) in enumerate(bundles.items()):
            ids, target, values, groups, tasks, parents = bundle
            for subgroup_index, subgroup in enumerate(("all", *sorted(set(tasks.tolist())))):
                mask = np.ones(len(ids), bool) if subgroup == "all" else tasks == subgroup
                rows, preds = _binary_panel(
                    registry=registry, panel_id=panel_by_unit[unit], split=split,
                    subgroup=subgroup, method_id="fixed_rag_iu_pcr", ids=ids[mask],
                    labels=target[mask], scores=values[mask], groups=groups[mask],
                    parent_ids=None if parents is None else parents[mask], draws=draws,
                    seed=seed + split_index * 100 + unit_offset * 10 + subgroup_index,
                )
                metrics.extend(rows)
                # Keep a single tidy prediction roster; subgroup rows are
                # deterministic views of these same predictions.
                if subgroup == "all":
                    predictions.extend(preds)
    return metrics, predictions


def _evaluate_gasp(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    private = _private_lookup(labels["sentences"])
    ids = np.asarray(scores["gasp_sentence_id"]).astype(str)
    if set(ids) != set(private):
        raise RagEvidenceContractError("GASP score/private sentence roster drifted")
    target = np.asarray([private[item]["label"] for item in ids], int)
    groups = np.asarray([private[item]["source_id"] for item in ids], str)
    tasks = np.asarray(scores["gasp_task"]).astype(str)
    expected_tasks = np.asarray([private[item]["task_type"] for item in ids], str)
    if not np.array_equal(tasks, expected_tasks):
        raise RagEvidenceContractError("GASP task/private binding drifted")
    methods = {
        "gasp_threshold": np.asarray(scores["gasp_threshold_score"], float),
        "fixed_rag_iu_pcr_matched": np.asarray(scores["gasp_fixed_rag_score"], float),
    }
    metrics, predictions, contrasts = [], [], []
    for subgroup_index, subgroup in enumerate(("all", *sorted(set(tasks.tolist())))):
        mask = np.ones(len(ids), bool) if subgroup == "all" else tasks == subgroup
        for method_index, (method_id, values) in enumerate(methods.items()):
            rows, preds = _binary_panel(
                registry=registry, panel_id="gasp_protocol_sentence",
                split="local_400_response_sample", subgroup=subgroup,
                method_id=method_id, ids=ids[mask], labels=target[mask],
                scores=values[mask], groups=groups[mask], parent_ids=None,
                draws=draws, seed=seed + subgroup_index * 10 + method_index,
            )
            metrics.extend(rows)
            if subgroup == "all":
                predictions.extend(preds)
        for metric_index, metric_name in enumerate(("auroc", "auprc")):
            result = grouped_paired_delta(
                target[mask], methods["gasp_threshold"][mask],
                methods["fixed_rag_iu_pcr_matched"][mask], groups[mask],
                _binary_metric(metric_name), draws=draws,
                seed=seed + 100 + subgroup_index * 10 + metric_index,
            )
            contrasts.append({
                "panel_id": "gasp_protocol_sentence",
                "split": "local_400_response_sample", "subgroup": subgroup,
                "left_method": "gasp_threshold", "right_method": "fixed_rag_iu_pcr_matched",
                "metric": metric_name, "delta": result["delta"],
                "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                "n": int(mask.sum()), "n_groups": len(set(groups[mask].tolist())),
                "bootstrap_draws": result["draws"], "status": result["status"],
            })
    return metrics, predictions, contrasts


def _evaluate_lettuce(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    private = _private_lookup(labels["rows"])
    ids = np.asarray(scores["lettuce_unit_id"]).astype(str)
    if set(ids) != set(private):
        raise RagEvidenceContractError("Lettuce score/private example roster drifted")
    target = np.asarray([private[item]["label"] for item in ids], int)
    groups = np.asarray([private[item]["source_id"] for item in ids], str)
    tasks = np.asarray([private[item]["task_type"] for item in ids], str)
    prediction = np.asarray(scores["lettuce_prediction"], int)
    metrics, predictions = [], []
    functions = {
        "f1": lambda y, p: float(f1_score(y, p, zero_division=0)),
        "precision": lambda y, p: float(precision_score(y, p, zero_division=0)),
        "recall": lambda y, p: float(recall_score(y, p, zero_division=0)),
    }
    for subgroup_index, subgroup in enumerate(("all", *sorted(set(tasks.tolist())))):
        mask = np.ones(len(ids), bool) if subgroup == "all" else tasks == subgroup
        for metric_index, (metric_name, function) in enumerate(functions.items()):
            summary = grouped_interval(
                target[mask], prediction[mask], groups[mask], function, draws=draws,
                seed=seed + subgroup_index * 10 + metric_index, require_two_classes=False,
            )
            metrics.append(_metric_row(
                registry=registry, panel_id="lettucedetect_example", split="test",
                subgroup=subgroup, method_id="lettucedetect_large_modernbert",
                metric=metric_name, summary=summary, n=int(mask.sum()), groups=groups[mask],
                positive_rate=float(target[mask].mean()),
            ))
        if subgroup == "all":
            probabilities = np.asarray(scores["lettuce_max_probability"], float)
            predictions.extend({
                "panel_id": "lettucedetect_example", "split": "test", "subgroup": "all",
                "method_id": "lettucedetect_large_modernbert", "unit_id": unit_id,
                "parent_id": unit_id, "score": float(probability),
                "prediction": int(pred), "label": int(label), "bootstrap_group": group,
            } for unit_id, probability, pred, label, group in zip(
                ids, probabilities, prediction, target, groups, strict=True
            ))
    return metrics, predictions


def _evaluate_refchecker(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    private = _private_lookup(labels["rows"])
    ids = np.asarray(scores["refchecker_unit_id"]).astype(str)
    if set(ids) != set(private):
        raise RagEvidenceContractError("RefChecker score/private claim roster drifted")
    setting = np.asarray(scores["refchecker_setting"]).astype(str)
    expected_setting = np.asarray([private[item]["setting"] for item in ids], str)
    if not np.array_equal(setting, expected_setting):
        raise RagEvidenceContractError("RefChecker setting binding drifted")
    groups = np.asarray([private[item]["example_id"] for item in ids], str)
    gold_threeway = np.asarray([private[item]["human_label"] for item in ids], str)
    gold_binary = np.asarray([private[item]["label_unsupported"] for item in ids], int)
    nli = np.asarray(scores["refchecker_nli_prediction"]).astype(str)
    binary_score = np.asarray(scores["refchecker_binary_score"], float)
    metrics, predictions = [], []
    if set(setting.tolist()) != set(REFCHECKER_SETTINGS):
        raise RagEvidenceContractError("RefChecker required setting coverage failed")
    for setting_index, subgroup in enumerate(REFCHECKER_SETTINGS):
        mask = setting == subgroup
        for metric_index, metric_name in enumerate(("accuracy", "macro_f1")):
            summary = grouped_interval(
                gold_threeway[mask], nli[mask], groups[mask], _threeway_metric(metric_name),
                draws=draws, seed=seed + setting_index * 20 + metric_index,
                require_two_classes=False,
            )
            metrics.append(_metric_row(
                registry=registry, panel_id="refchecker_threeway",
                split="official_fixed_claims", subgroup=subgroup,
                method_id="refchecker_nli", metric=metric_name, summary=summary,
                n=int(mask.sum()), groups=groups[mask], positive_rate="",
            ))
        rows, _ = _binary_panel(
            registry=registry, panel_id="refchecker_binary_claim",
            split="official_fixed_claims", subgroup=subgroup,
            method_id="fixed_rag_iu_pcr_transfer", ids=ids[mask],
            labels=gold_binary[mask], scores=binary_score[mask], groups=groups[mask],
            parent_ids=None, draws=draws, seed=seed + setting_index * 20 + 10,
        )
        metrics.extend(rows)
    # One prediction row per method/claim.  There is deliberately no pooled
    # RefChecker metric row; setting is carried as the subgroup.
    for index, unit_id in enumerate(ids):
        predictions.append({
            "panel_id": "refchecker_threeway", "split": "official_fixed_claims",
            "subgroup": setting[index], "method_id": "refchecker_nli",
            "unit_id": unit_id, "parent_id": unit_id, "score": "",
            "prediction": nli[index], "label": gold_threeway[index],
            "bootstrap_group": groups[index],
        })
        predictions.append({
            "panel_id": "refchecker_binary_claim", "split": "official_fixed_claims",
            "subgroup": setting[index], "method_id": "fixed_rag_iu_pcr_transfer",
            "unit_id": unit_id, "parent_id": unit_id, "score": float(binary_score[index]),
            "prediction": "", "label": int(gold_binary[index]),
            "bootstrap_group": groups[index],
        })
    return metrics, predictions


def compute_rag_evidence_evaluation_tables(
    *, registry: Mapping[str, Any], private: Mapping[str, Any],
    scores: Mapping[str, np.ndarray], draws: int, seed: int,
) -> dict[str, Any]:
    """Deterministically re-evaluate frozen scores from isolated private labels."""

    if draws <= 0:
        raise RagEvidenceContractError("RAG evaluation needs positive bootstrap draws")
    validate_score_arrays(scores)
    metrics, predictions = _evaluate_ragtruth(
        registry=registry, labels=private["ragtruth"], scores=scores,
        draws=draws, seed=seed,
    )
    gasp_metrics, gasp_predictions, contrasts = _evaluate_gasp(
        registry=registry, labels=private["gasp"], scores=scores,
        draws=draws, seed=seed + 1000,
    )
    lettuce_metrics, lettuce_predictions = _evaluate_lettuce(
        registry=registry, labels=private["lettuce"], scores=scores,
        draws=draws, seed=seed + 2000,
    )
    ref_metrics, ref_predictions = _evaluate_refchecker(
        registry=registry, labels=private["refchecker"], scores=scores,
        draws=draws, seed=seed + 3000,
    )
    metrics.extend(gasp_metrics + lettuce_metrics + ref_metrics)
    predictions.extend(gasp_predictions + lettuce_predictions + ref_predictions)
    if any(
        row["panel_id"].startswith("refchecker_") and row["subgroup"] == "all"
        for row in metrics
    ):
        raise RagEvidenceContractError("RefChecker pooled metric escaped the evaluator")
    if any(row.get("panel_id") not in PANEL_IDS for row in metrics):
        raise RagEvidenceContractError("unregistered panel escaped the RAG evaluator")
    if {row["panel_id"] for row in metrics} != set(PANEL_IDS):
        raise RagEvidenceContractError("RAG evaluation did not cover every registered panel")

    panel_status = []
    for panel_id in PANEL_IDS:
        rows = [row for row in metrics if row["panel_id"] == panel_id]
        status = "PASS" if rows and all(row["status"] in {"OK", "METRIC_UNDEFINED_SINGLE_CLASS"} for row in rows) else "FAIL"
        panel_status.append({
            "panel_id": panel_id,
            "status": status,
            "metric_rows": len(rows),
            "prediction_rows": sum(row["panel_id"] == panel_id for row in predictions),
            "cross_panel_macro_contribution": "FORBIDDEN",
        })
    if any(row["status"] != "PASS" for row in panel_status):
        raise RagEvidenceContractError("one or more RAG panel status gates failed")

    file_payloads = {
        "metrics.csv": _csv_bytes(metrics, METRIC_COLUMNS),
        "predictions.csv": _csv_bytes(predictions, PREDICTION_COLUMNS),
        "contrasts.csv": _csv_bytes(contrasts, CONTRAST_COLUMNS),
        "panel_status.csv": _csv_bytes(
            panel_status,
            (
                "panel_id", "status", "metric_rows", "prediction_rows",
                "cross_panel_macro_contribution",
            ),
        ),
    }
    return {
        "file_payloads": file_payloads,
        "metrics": metrics,
        "predictions": predictions,
        "contrasts": contrasts,
        "panel_status": panel_status,
    }


def evaluate_rag_evidence_build(
    *, repo: str | Path, registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    build_id: str, draws_override: int | None = None,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    if build_id not in {"A", "B"}:
        raise RagEvidenceContractError("RAG evaluation build must be A or B")
    repo_path = Path(repo).resolve(strict=True)
    registry = load_registry(registry_path)
    lane_root = Path(release_root) / release_id / "rag_evidence"
    build_root = lane_root / build_id
    preparation_path = build_root / PREPARATION_MANIFEST_FILENAME
    score_manifest_path = build_root / "fit" / SCORE_MANIFEST_FILENAME
    preparation_payload = read_bound_file_bytes(
        preparation_path, name="RAG preparation manifest"
    )
    score_manifest_payload = read_bound_file_bytes(
        score_manifest_path, name="RAG score freeze"
    )
    preparation = json.loads(preparation_payload.decode("utf-8"))
    score_manifest = json.loads(score_manifest_payload.decode("utf-8"))
    verify_payload(preparation, name="RAG preparation manifest")
    verify_payload(score_manifest, name="RAG score freeze")
    from .rag_evidence_ab import authenticate_rag_evidence_score_certificate

    # This gate must complete before the first private-label file open.
    score_certificate = authenticate_rag_evidence_score_certificate(
        repo=repo_path, registry_path=registry_path, source_root=source_root,
        release_root=release_root, private_root=private_root,
        release_id=release_id, require_scientific_full=draws_override is None,
    )
    if (
        score_certificate["score_sha256"] != score_manifest["scores"]["sha256"]
        or score_certificate["private_label_sha256"]
        != preparation["private_labels"]["sha256"]
    ):
        raise RagEvidenceContractError("RAG evaluation is underbound to score A/B")
    score_certificate_path = lane_root / "SCORE_AB_VERIFICATION.json"
    score_certificate_payload = read_bound_file_bytes(
        score_certificate_path, name="RAG authenticated score certificate"
    )
    if json.loads(score_certificate_payload.decode("utf-8")) != score_certificate:
        raise RagEvidenceContractError(
            "RAG score certificate changed after authentication"
        )
    validate_source_binding(
        preparation["source_binding"], source_root=source_root, registry=registry
    )
    private_path = Path(preparation["private_labels"]["path"])
    private = load_private_labels(
        private_path,
        registry,
        expected_sha256=score_certificate["private_label_sha256"],
    )
    score_path = build_root / "fit" / score_manifest["scores"]["path"]
    scores = load_scores(
        score_path,
        expected_sha256=score_certificate["score_sha256"],
    )
    configured_draws = int(registry["evaluation"]["bootstrap"]["draws"])
    draws = configured_draws if draws_override is None else int(draws_override)
    if draws <= 0 or (draws_override is None and draws != 20_000):
        raise RagEvidenceContractError("RAG evaluation bootstrap draw contract failed")
    seed = int(registry["evaluation"]["bootstrap"]["seed"])
    derived = compute_rag_evidence_evaluation_tables(
        registry=registry, private=private, scores=scores, draws=draws, seed=seed
    )

    evaluation_final = build_root / "evaluation"
    stage = AtomicRagDirectory(evaluation_final)
    try:
        files = []
        for name, payload in derived["file_payloads"].items():
            digest = stage.write_bytes(name, payload)
            files.append({"path": name, "sha256": digest, "size_bytes": len(payload)})
        source_snapshot = {
            "files": [
                {
                    "path": relative,
                    "sha256": sha256_bytes(
                        read_bound_file_bytes(
                            repo_path / relative,
                            name=f"RAG evaluation source {relative}",
                        )
                    ),
                }
                for relative in EVALUATION_SOURCE_FILES
            ]
        }
        source_snapshot["snapshot_sha256"] = payload_sha256(source_snapshot)
        manifest = add_payload_sha256({
            "schema_version": EVALUATION_SCHEMA,
            "release_id": release_id,
            "build_id": build_id,
            "lane_id": registry["lane_id"],
            "scientific_full": draws_override is None,
            "score_ab_certificate_sha256": sha256_bytes(score_certificate_payload),
            "score_manifest_sha256": sha256_bytes(score_manifest_payload),
            "score_sha256": score_manifest["scores"]["sha256"],
            "private_label_sha256": preparation["private_labels"]["sha256"],
            "source_binding_sha256": preparation["source_binding_sha256"],
            "source_snapshot": source_snapshot,
            "bootstrap": {
                "draws_requested": draws,
                "group": "panel-registered source group",
                "paired_contrasts": True,
                "seed": seed,
            },
            "files": files,
            "panel_status": derived["panel_status"],
            "cross_panel_macro_computed": False,
            "refchecker_settings_pooled": False,
            "historical_scores_copied": False,
        })
        stage.write_json(EVALUATION_MANIFEST_FILENAME, manifest)
        stage.commit()
        return manifest
    finally:
        stage.cleanup()


__all__ = [
    "CONTRAST_COLUMNS", "EVALUATION_SOURCE_FILES", "METRIC_COLUMNS",
    "PREDICTION_COLUMNS", "compute_rag_evidence_evaluation_tables",
    "evaluate_rag_evidence_build", "grouped_interval", "grouped_paired_delta",
]
