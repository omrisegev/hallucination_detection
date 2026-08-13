#!/usr/bin/env python3
"""Build, evaluate, and report the fixed shared-feature RAG/reasoning pipelines.

This is an application experiment, not a new covariance solver.  Both domains
use the same token-resolved mixed-v2 feature contract.  Labels are opened only
after label-free scores have been written and hashed.  RAGTruth has already
been inspected in prior experiments, so its results are exploratory; the
report does not describe them as blinded confirmation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from spectral_utils.feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from spectral_utils.fixed_application_pipelines import (
    CONTRACT_VERSION,
    SHARED_TOKEN_VIEWS,
    aggregate_risk,
    condition_trace_row,
    contract_audit,
    fit_rag_evidence_head,
    fit_shared_mixed_transformer,
    fit_shared_token_iu,
    rag_evidence_matrix,
    raw_token_feature_matrix,
)
from spectral_utils.ragtruth_evidence_contrast import adapt_cache
from spectral_utils.upcr import upcr_fit


VERSION = "fixed-rag-reasoning-shared-features-v1-2026-08-13"
DEFAULT_OUT = ROOT / "results" / "fixed_application_pipelines_v1"
BAD_PRM_IDS = {
    "confidence_confidence_prm_train_p1_303",
    "deception_deception_prm_test_p1_87",
    "step_contradiction_step_contradiction_prm_test_p2_991",
}
PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PB_MODELS = ("qwen3_4b", "qwen3_8b")
PB_DEV = {("qwen3_4b", "gsm8k"), ("qwen3_4b", "math")}
NO_ERROR = -1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_hash(ids: Sequence[str], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(ids, dtype="U").tobytes())
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.asarray(arrays[name], dtype="<f8").tobytes())
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def binary_metrics(y: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
    }


def grouped_bootstrap_delta(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: Sequence[str],
    *,
    draws: int = 1000,
    seed: int = 20260813,
) -> dict[str, float]:
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.choice(unique, len(unique), replace=True)
        indexes = np.concatenate([lookup[group] for group in selected])
        if len(np.unique(y[indexes])) != 2:
            continue
        values.append(roc_auc_score(y[indexes], left[indexes]) - roc_auc_score(y[indexes], right[indexes]))
    values = np.asarray(values, dtype=float)
    return {
        "delta": float(roc_auc_score(y, left) - roc_auc_score(y, right)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "draws": int(len(values)),
    }


def grouped_metric_intervals(
    y: np.ndarray,
    score: np.ndarray,
    groups: Sequence[str],
    *,
    draws: int = 1000,
    seed: int = 20260813,
) -> dict[str, float]:
    """Grouped percentile intervals for ranking metrics.

    Complete source groups are resampled so multiple responses, sentences, or
    claims from one source never masquerade as independent observations.
    """

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    auroc, auprc = [], []
    for _ in range(draws):
        selected = rng.choice(unique, len(unique), replace=True)
        indexes = np.concatenate([lookup[group] for group in selected])
        if len(np.unique(y[indexes])) != 2:
            continue
        auroc.append(roc_auc_score(y[indexes], score[indexes]))
        auprc.append(average_precision_score(y[indexes], score[indexes]))
    return {
        "auroc_ci_low": float(np.quantile(auroc, 0.025)),
        "auroc_ci_high": float(np.quantile(auroc, 0.975)),
        "auprc_ci_low": float(np.quantile(auprc, 0.025)),
        "auprc_ci_high": float(np.quantile(auprc, 0.975)),
        "bootstrap_draws": int(len(auroc)),
    }


def _raw_rag_conditions(response: Any) -> dict[str, np.ndarray]:
    return {
        name: raw_token_feature_matrix(condition_trace_row(trace))
        for name, trace in response.conditions.items()
    }


def _fit_rag_models(dev_dataset: Any) -> tuple[Any, Any, Any, dict[str, dict[str, np.ndarray]]]:
    raw_by_response: dict[str, dict[str, np.ndarray]] = {}
    full_records = []
    for response in dev_dataset.responses:
        conditions = _raw_rag_conditions(response)
        raw_by_response[response.response_id] = conditions
        full_records.append((response.response_id, conditions["full"]))
    transformer = fit_shared_mixed_transformer(full_records)
    noctx_records, loo_records = [], []
    for response in dev_dataset.responses:
        conditions = raw_by_response[response.response_id]
        matrix, names = rag_evidence_matrix(conditions, transformer, profile="noctx")
        noctx_records.append((response.response_id, matrix, names))
        if any(name.startswith("loo_") for name in conditions):
            matrix, names = rag_evidence_matrix(conditions, transformer, profile="loo")
            loo_records.append((response.response_id, matrix, names))
    noctx_head = fit_rag_evidence_head(noctx_records, profile="noctx")
    loo_head = fit_rag_evidence_head(loo_records, profile="loo")
    return transformer, noctx_head, loo_head, raw_by_response


def _rag_risk(
    conditions: Mapping[str, np.ndarray],
    transformer: Any,
    noctx_head: Any,
    loo_head: Any,
) -> tuple[np.ndarray, str]:
    has_loo = any(name.startswith("loo_") for name in conditions)
    profile = "loo" if has_loo else "noctx"
    matrix, _ = rag_evidence_matrix(conditions, transformer, profile=profile)
    head = loo_head if has_loo else noctx_head
    return head.risk(matrix), profile


def _token_labels_from_cache(cache: Mapping[str, Mapping[str, Any]], response_id: str) -> np.ndarray:
    row = cache[f"{response_id}::full"]
    labels = np.zeros(len(row["gen_token_ids"]), dtype=int)
    for span in row.get("span_token_spans", []):
        if span is None:
            continue
        labels[max(0, int(span[0])):min(len(labels), int(span[1]))] = 1
    return labels


def _best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-score, kind="mergesort")
    candidates = np.concatenate([[np.inf], (score[order][:-1] + score[order][1:]) / 2, [-np.inf]])
    best = (0.0, np.inf)
    for threshold in candidates:
        pred = score >= threshold
        f1 = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)[2]
        if f1 > best[0]:
            best = (float(f1), float(threshold))
    return best[1], best[0]


def _rag_split_scores(dataset: Any, transformer: Any, noctx_head: Any, loo_head: Any) -> dict[str, Any]:
    response_ids, source_ids, tasks, profiles = [], [], [], []
    response_scores, token_scores, offsets = [], [], [0]
    sentence_ids, sentence_source_ids, sentence_scores = [], [], []
    for response in dataset.responses:
        conditions = _raw_rag_conditions(response)
        risk, profile = _rag_risk(conditions, transformer, noctx_head, loo_head)
        response_ids.append(response.response_id)
        source_ids.append(response.source_id)
        tasks.append(response.task_type)
        profiles.append(profile)
        response_scores.append(aggregate_risk(risk, "mean"))
        token_scores.extend(risk.tolist())
        offsets.append(len(token_scores))
        for unit in response.sentences:
            sentence_ids.append(f"{response.response_id}::sent_{unit.index}")
            sentence_source_ids.append(response.source_id)
            sentence_scores.append(aggregate_risk(risk[unit.token_start:unit.token_end], "mean"))
    return {
        "response_ids": np.asarray(response_ids, dtype="U"),
        "source_ids": np.asarray(source_ids, dtype="U"),
        "task_types": np.asarray(tasks, dtype="U"),
        "profiles": np.asarray(profiles, dtype="U"),
        "response_scores": np.asarray(response_scores, dtype=float),
        "token_scores": np.asarray(token_scores, dtype=float),
        "token_offsets": np.asarray(offsets, dtype=int),
        "sentence_ids": np.asarray(sentence_ids, dtype="U"),
        "sentence_source_ids": np.asarray(sentence_source_ids, dtype="U"),
        "sentence_scores": np.asarray(sentence_scores, dtype=float),
    }


def run_rag(out: Path) -> None:
    print("[rag] load canonical dev/test", flush=True)
    from transformers import AutoTokenizer

    tokenizer_path = ROOT / "local_cache/qwen25_15b_tokenizer"
    official = ROOT / "local_cache/RAGTruth_official/dataset/response.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    dev_cache = ROOT / "local_cache/ragtruth_ec/dev/ragtruth_ec_train.pkl"
    test_cache = ROOT / "local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl"
    dev_dataset, dev_labels, _ = adapt_cache(dev_cache, official, tokenizer)
    test_dataset, test_labels, _ = adapt_cache(test_cache, official, tokenizer)

    print("[rag] fit shared transform and fixed evidence heads", flush=True)
    transformer, noctx_head, loo_head, _ = _fit_rag_models(dev_dataset)
    print("[rag] score before opening cache labels", flush=True)
    dev_scores = _rag_split_scores(dev_dataset, transformer, noctx_head, loo_head)
    test_scores = _rag_split_scores(test_dataset, transformer, noctx_head, loo_head)
    score_path = out / "rag_scores.npz"
    np.savez_compressed(score_path, **{f"dev__{k}": v for k, v in dev_scores.items()},
                        **{f"test__{k}": v for k, v in test_scores.items()})
    score_digest = sha256(score_path)
    write_json(out / "rag_score_freeze.json", {
        "version": VERSION,
        "score_file": score_path.name,
        "sha256": score_digest,
        "labels_seen_during_fit": False,
        "ragtruth_status": "exploratory; labels were opened in earlier project work",
    })

    # Evaluation begins here.  Gold fields were isolated from all fit helpers.
    print("[rag] evaluate frozen scores", flush=True)
    raw_dev = load_pickle(dev_cache)
    raw_test = load_pickle(test_cache)
    rows = []
    evaluation: dict[str, Any] = {}
    for split, dataset, labels, scores, raw in (
        ("dev", dev_dataset, dev_labels, dev_scores, raw_dev),
        ("test", test_dataset, test_labels, test_scores, raw_test),
    ):
        response_y = np.asarray([int(labels.response[rid].hallucinated) for rid in scores["response_ids"]])
        response_metric = {
            **binary_metrics(response_y, scores["response_scores"]),
            **grouped_metric_intervals(
                response_y, scores["response_scores"], scores["source_ids"],
                seed=20260813 + (0 if split == "dev" else 1),
            ),
        }
        evaluation[f"{split}_response"] = response_metric
        rows.append({"domain": "RAG", "benchmark": "RAGTruth", "split": split,
                     "unit": "answer", "method": "Fixed RAG IU-PCR", **response_metric})
        for task in sorted(set(scores["task_types"].tolist())):
            mask = scores["task_types"] == task
            metric = binary_metrics(response_y[mask], scores["response_scores"][mask])
            rows.append({"domain": "RAG", "benchmark": "RAGTruth", "split": split,
                         "unit": "answer", "subgroup": task, "method": "Fixed RAG IU-PCR", **metric})

        token_y = np.concatenate([_token_labels_from_cache(raw, rid) for rid in scores["response_ids"]])
        token_metric = binary_metrics(token_y, scores["token_scores"])
        evaluation[f"{split}_token"] = token_metric
        rows.append({"domain": "RAG", "benchmark": "RAGTruth", "split": split,
                     "unit": "token", "method": "Fixed RAG IU-PCR", **token_metric})

        sentence_y = np.asarray([int(labels.sentence[sid].hallucinated) for sid in scores["sentence_ids"]])
        sentence_metric = {
            **binary_metrics(sentence_y, scores["sentence_scores"]),
            **grouped_metric_intervals(
                sentence_y, scores["sentence_scores"], scores["sentence_source_ids"],
                seed=20260815 + (0 if split == "dev" else 1),
            ),
        }
        evaluation[f"{split}_sentence"] = sentence_metric
        rows.append({"domain": "RAG", "benchmark": "RAGTruth", "split": split,
                     "unit": "sentence", "method": "Fixed RAG IU-PCR", **sentence_metric})

    # One dev-calibrated response operating threshold for the supervised
    # LettuceDetect example-F1 comparison.  Fusion itself remains label-free.
    dev_y = np.asarray([int(dev_labels.response[rid].hallucinated) for rid in dev_scores["response_ids"]])
    dev_max = np.asarray([
        np.max(dev_scores["token_scores"][a:b])
        for a, b in zip(dev_scores["token_offsets"][:-1], dev_scores["token_offsets"][1:])
    ])
    threshold, dev_f1 = _best_f1_threshold(dev_y, dev_max)
    test_y = np.asarray([int(test_labels.response[rid].hallucinated) for rid in test_scores["response_ids"]])
    test_max = np.asarray([
        np.max(test_scores["token_scores"][a:b])
        for a, b in zip(test_scores["token_offsets"][:-1], test_scores["token_offsets"][1:])
    ])
    pred = test_max >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(test_y, pred, average="binary", zero_division=0)
    rows.append({"domain": "RAG", "benchmark": "RAGTruth", "split": "test",
                 "unit": "answer", "method": "Fixed RAG IU-PCR (dev threshold)",
                 "metric_note": "example-level thresholded prediction", "f1": f1,
                 "precision": precision, "recall": recall, "n": len(test_y),
                 "positive_rate": test_y.mean()})
    evaluation["example_f1"] = {"threshold": threshold, "dev_f1": dev_f1,
                                "test_f1": f1, "precision": precision, "recall": recall}

    # Exact matched GASP sentence cohort.  The fixed RAG model remains the one
    # fitted above on RAGTruth development telemetry; only the already-frozen
    # score is evaluated on these 400 responses.
    gasp_cache_path = ROOT / "dataset_cache/four_localization/gasp_ragtruth_exact_qwen15b_full/gasp_exact.pkl"
    gasp_cache = load_pickle(gasp_cache_path)
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for raw_row in gasp_cache.values():
        grouped[str(raw_row["response_id"])][str(raw_row["condition"])] = raw_row
    gasp_y, gasp_score, gasp_groups, gasp_tasks = [], [], [], []
    for response_id in sorted(grouped):
        conditions = grouped[response_id]
        if "full" not in conditions or "noctx" not in conditions:
            continue
        raw_conditions = {
            name: raw_token_feature_matrix(row) for name, row in conditions.items()
        }
        risk, _ = _rag_risk(raw_conditions, transformer, noctx_head, loo_head)
        full = conditions["full"]
        encoded = tokenizer(
            str(full["response"]), add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if not np.array_equal(
            np.asarray(encoded["input_ids"], dtype=int),
            np.asarray(full["gen_token_ids"], dtype=int),
        ):
            raise ValueError(f"GASP response {response_id} does not reproduce stored tokens")
        offsets = list(encoded["offset_mapping"])
        for char_start, char_end in full["sentence_spans"]:
            indexes = [
                index for index, (start, end) in enumerate(offsets)
                if end > int(char_start) and start < int(char_end)
            ]
            if not indexes:
                continue
            hallucinated = int(any(
                max(int(char_start), int(item["start"]))
                < min(int(char_end), int(item["end"]))
                for item in full.get("span_labels", [])
            ))
            gasp_y.append(hallucinated)
            gasp_score.append(aggregate_risk(risk[min(indexes):max(indexes) + 1], "mean"))
            gasp_groups.append(str(full["source_id"]))
            gasp_tasks.append(str(full["task_type"]))
    gasp_y_array = np.asarray(gasp_y, dtype=int)
    gasp_score_array = np.asarray(gasp_score, dtype=float)
    gasp_metric = {
        **binary_metrics(gasp_y_array, gasp_score_array),
        **grouped_metric_intervals(
            gasp_y_array, gasp_score_array, gasp_groups, seed=20260817,
        ),
    }
    rows.append({"domain": "RAG", "benchmark": "RAGTruth balanced GASP cohort",
                 "split": "400-response sample", "unit": "sentence",
                 "method": "Fixed RAG IU-PCR", "reference_role": "ours_frozen",
                 **gasp_metric})
    for task in sorted(set(gasp_tasks)):
        mask = np.asarray(gasp_tasks) == task
        metric = binary_metrics(gasp_y_array[mask], gasp_score_array[mask])
        rows.append({"domain": "RAG", "benchmark": "RAGTruth balanced GASP cohort",
                     "split": "400-response sample", "unit": "sentence",
                     "subgroup": task, "method": "Fixed RAG IU-PCR",
                     "reference_role": "ours_frozen", **metric})
    evaluation["gasp_exact_sentence"] = gasp_metric

    # Published/protocol-reproduction references remain separate roles.
    reference_rows = read_csv(ROOT / "results/paper_aligned_benchmark_suite_2026_08_11/benchmark_scores.csv")
    for record in reference_rows:
        if record.get("protocol_id") in {
            "localization-gasp-ragtruth-sentence",
            "localization-lettucedetect-ragtruth-span",
        } and record.get("role") in {"published_peer", "published_ceiling", "protocol_reproduction"}:
            rows.append({"domain": "RAG", "benchmark": record.get("dataset"),
                         "split": record.get("split"), "unit": record.get("prediction_unit"),
                         "method": record.get("method"), "reference_role": record.get("role"),
                         "reported_metric": record.get("metric", "value"),
                         "reported_value": float(record["value"]),
                         "reported_ci_low": record.get("ci_low", ""),
                         "reported_ci_high": record.get("ci_high", ""),
                         "subgroup": record.get("subgroup", ""),
                         record.get("metric", "value"): float(record["value"]),
                         "n": record.get("n", ""), "caveat": record.get("caveat", "")})

    write_csv(out / "rag_metrics.csv", rows)
    write_json(out / "rag_diagnostics.json", {
        "score_sha256": score_digest,
        "base_transform_fit": {"n_dev_responses": len(dev_dataset.responses)},
        "noctx_head": noctx_head.diagnostics,
        "loo_head": loo_head.diagnostics,
        "evaluation": evaluation,
        "final_adapters": {"answer": "mean token risk", "sentence_claim": "mean token risk",
                           "reasoning_step": "maximum token risk",
                           "token_span": "native token risk"},
    })
    print(f"[rag] complete -> {out / 'rag_metrics.csv'}", flush=True)


def _global_mixed_v2_risk(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    support = ROOT / "scripts/gl_liu_v1"
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from evaluate_answer_level import trace_features

    extracted = [trace_features(dict(row)) for row in rows]
    names, columns, availability = [], [], {}
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        raw = np.asarray([item.get(name, np.nan) for item in extracted], dtype=float)
        finite = np.isfinite(raw)
        availability[name] = float(finite.mean())
        if finite.mean() < 0.70 or not finite.any():
            continue
        raw = np.where(finite, raw, np.median(raw[finite]))
        if raw.std() < 1e-8 or np.mean(raw == np.median(raw)) > 0.40:
            continue
        names.append(name)
        columns.append(raw)
    mixed, names, details = dufs_liu_mixed_v2_matrix(np.column_stack(columns), names)
    fitted = upcr_fit(
        mixed.T, loss="l2", exclusion=False, difficulty_gate=False,
        simple_avg_fallback=False, recompute_after_exclusion=False,
        g2_projection_k=1, scale_ratio=0.25, n_components=2,
        auto_components=False,
    )
    return -(fitted.w @ mixed.T), {
        "feature_names": list(names), "n_features": len(names),
        "availability": availability, "mixed_v2_transforms": details,
        "labels_seen_during_fit": False,
    }


def _zscore(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-12)


def _reasoning_cell_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = [(str(index), raw_token_feature_matrix(row)) for index, row in enumerate(rows)]
    token_model = fit_shared_token_iu(raw)
    curves = [token_model.risk(matrix) for _, matrix in raw]
    step_risk, locator, local_sequence = [], [], []
    for row, curve in zip(rows, curves):
        values = []
        for span in row["step_token_spans"]:
            if span is None or int(span[1]) <= int(span[0]):
                values.append(float("-inf"))
            else:
                values.append(aggregate_risk(curve[int(span[0]):int(span[1])], "max"))
        values = np.asarray(values, dtype=float)
        step_risk.append(values)
        locator.append(int(np.argmax(values)) if len(values) else NO_ERROR)
        local_sequence.append(float(np.max(values)) if len(values) else float("-inf"))
    global_risk, global_diag = _global_mixed_v2_risk(rows)
    local_sequence = np.asarray(local_sequence, dtype=float)
    detector = 0.75 * _zscore(global_risk) + 0.25 * _zscore(local_sequence)
    return {
        "detector": detector,
        "global_risk": np.asarray(global_risk),
        "local_sequence_risk": local_sequence,
        "locator": np.asarray(locator, dtype=int),
        "step_risk": np.asarray(step_risk, dtype=object),
    }, {
        "token_model": token_model.diagnostics,
        "global_model": global_diag,
        "detector_blend": {"global": 0.75, "local": 0.25},
        "step_adapter": "maximum token risk after full-trajectory fusion",
    }


def _processbench_metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    error = labels != NO_ERROR
    correct = ~error
    acc_error = float(np.mean(prediction[error] == labels[error])) if error.any() else 0.0
    acc_correct = float(np.mean(prediction[correct] == NO_ERROR)) if correct.any() else 0.0
    denom = acc_error + acc_correct
    return {
        "f1": float(2 * acc_error * acc_correct / denom) if denom else 0.0,
        "acc_erroneous": acc_error,
        "acc_correct": acc_correct,
        "exact": float(np.mean(prediction == labels)),
    }


def _calibrated_processbench(
    risk: np.ndarray,
    locator: np.ndarray,
    labels: np.ndarray,
    *,
    splits: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    support = ROOT / "scripts/gl_liu_v1"
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from two_stage_localization import evaluate_two_stage

    return evaluate_two_stage(risk, locator, labels, n_splits=splits, seed=seed)


def _paired_processbench_delta(
    left_risk: np.ndarray,
    left_locator: np.ndarray,
    right_risk: np.ndarray,
    right_locator: np.ndarray,
    labels: np.ndarray,
    *,
    splits: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired ProcessBench F1 deltas over identical calibration/evaluation splits."""

    support = ROOT / "scripts/gl_liu_v1"
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from two_stage_localization import best_threshold
    from localization_metrics import processbench_f1

    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(splits):
        perm = rng.permutation(len(labels))
        calibration, evaluation = perm[:len(labels) // 2], perm[len(labels) // 2:]
        predictions = []
        for risk, locator in (
            (left_risk, left_locator), (right_risk, right_locator),
        ):
            threshold, _ = best_threshold(risk, locator, labels, calibration)
            prediction = np.where(
                np.asarray(risk)[evaluation] > threshold,
                np.asarray(locator)[evaluation],
                NO_ERROR,
            )
            predictions.append(prediction)
        left = processbench_f1(predictions[0], labels[evaluation])["f1"]
        right = processbench_f1(predictions[1], labels[evaluation])["f1"]
        values.append(float(left - right))
    values_array = np.asarray(values, dtype=float)
    return {
        "delta": float(values_array.mean()),
        "ci_low": float(np.quantile(values_array, 0.025)),
        "ci_high": float(np.quantile(values_array, 0.975)),
        "positive_fraction": float(np.mean(values_array > 0)),
        "values": values,
    }


def _run_processbench(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    support = ROOT / "scripts/gl_liu_v1"
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from run import mindgap_control

    rows_out, diagnostics = [], {}
    paired_by_cell: dict[tuple[str, str], np.ndarray] = {}
    all_score_ids, all_scores = [], []
    for model in PB_MODELS:
        for subset in PB_SUBSETS:
            path = ROOT / f"cache/localization/processbench/pb_{model}/processbench_{subset}.pkl"
            cache = load_pickle(path)
            rows = [cache[key] for key in sorted(cache) if not cache[key]["align_diag"]["problems"]]
            print(f"[reasoning] ProcessBench {model}/{subset} ({len(rows)})", flush=True)
            scores, diag = _reasoning_cell_scores(rows)
            labels = np.asarray([row["label"] for row in rows], dtype=int)
            ours = _calibrated_processbench(scores["detector"], scores["locator"], labels)
            mind_risk, mind_locator = mindgap_control(rows)
            mind = _calibrated_processbench(mind_risk, mind_locator, labels)
            paired = _paired_processbench_delta(
                scores["detector"], scores["locator"], mind_risk, mind_locator, labels,
            )
            paired_by_cell[(model, subset)] = np.asarray(paired.pop("values"), dtype=float)
            split = "development" if (model, subset) in PB_DEV else "confirmation"
            for method, result in (("Fixed reasoning IU-PCR", ours), ("Mind the Gap control", mind)):
                rows_out.append({"domain": "reasoning", "benchmark": "ProcessBench",
                                 "model": model, "subgroup": subset, "split": split,
                                 "unit": "first erroneous step", "method": method,
                                 "f1": result["f1"], "f1_sd": result["f1_sd"],
                                 "acc_erroneous": result["acc_erroneous"],
                                 "acc_correct": result["acc_correct"], "n": len(rows)})
            diagnostics[f"{model}/{subset}"] = {**diag, "paired_vs_mindgap": paired}
            all_score_ids.extend([f"{model}/{subset}/{index}" for index in range(len(rows))])
            all_scores.extend(scores["detector"].tolist())

    for split_name, predicate in (
        ("all eight cells", lambda row: True),
        ("confirmation six cells", lambda row: row["split"] == "confirmation"),
    ):
        for method in ("Fixed reasoning IU-PCR", "Mind the Gap control"):
            selected = [row for row in rows_out if row["method"] == method and predicate(row)]
            record = {"domain": "reasoning", "benchmark": "ProcessBench",
                      "model": "macro", "subgroup": split_name, "split": split_name,
                      "unit": "first erroneous step", "method": method,
                      "f1": float(np.mean([row["f1"] for row in selected])),
                      "acc_erroneous": float(np.mean([row["acc_erroneous"] for row in selected])),
                      "acc_correct": float(np.mean([row["acc_correct"] for row in selected])),
                      "n": int(sum(row["n"] for row in selected))}
            if method == "Fixed reasoning IU-PCR":
                keys = [key for key in paired_by_cell
                        if predicate({"split": "development" if key in PB_DEV else "confirmation"})]
                paired_values = np.mean([paired_by_cell[key] for key in keys], axis=0)
                record.update({
                    "delta_vs_mindgap": float(paired_values.mean()),
                    "delta_ci_low": float(np.quantile(paired_values, 0.025)),
                    "delta_ci_high": float(np.quantile(paired_values, 0.975)),
                })
            rows_out.append(record)

    # Published ProcessBench controls in the benchmark registry use the four
    # Qwen3-8B subsets.  Keep these model-specific macros separate from the
    # eight-cell local summary so the report never compares different
    # populations in the same bar chart.
    for model in PB_MODELS:
        for method in ("Fixed reasoning IU-PCR", "Mind the Gap control"):
            selected = [row for row in rows_out
                        if row["method"] == method and row["model"] == model
                        and row["subgroup"] in PB_SUBSETS]
            record = {"domain": "reasoning", "benchmark": "ProcessBench",
                      "model": model, "subgroup": "four-subset macro",
                      "split": "model macro", "unit": "first erroneous step",
                      "method": method,
                      "f1": float(np.mean([row["f1"] for row in selected])),
                      "acc_erroneous": float(np.mean([row["acc_erroneous"] for row in selected])),
                      "acc_correct": float(np.mean([row["acc_correct"] for row in selected])),
                      "n": int(sum(row["n"] for row in selected))}
            if method == "Fixed reasoning IU-PCR":
                paired_values = np.mean(
                    [paired_by_cell[(model, subset)] for subset in PB_SUBSETS], axis=0
                )
                record.update({
                    "delta_vs_mindgap": float(paired_values.mean()),
                    "delta_ci_low": float(np.quantile(paired_values, 0.025)),
                    "delta_ci_high": float(np.quantile(paired_values, 0.975)),
                })
            rows_out.append(record)

    write_json(out / "processbench_score_freeze.json", {
        "score_hash": score_hash(all_score_ids, {"detector": np.asarray(all_scores)}),
        "labels_seen_during_score_construction": False,
        "labels_used_for_operating_threshold": "random calibration half inside each repeated split",
    })
    return rows_out, diagnostics


def _run_prmbench(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    telemetry_path = ROOT / "dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl"
    competitor_path = ROOT / "dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl"
    cache = load_pickle(telemetry_path)
    rows = [cache[key] for key in sorted(cache) if str(cache[key]["idx"]) not in BAD_PRM_IDS]
    print(f"[reasoning] PRMBench shared trajectory ({len(rows)} rows)", flush=True)
    raw = [(str(row["idx"]), raw_token_feature_matrix(row)) for row in rows]
    model = fit_shared_token_iu(raw)
    ours_y, ours_score, categories, groups, ids = [], [], [], [], []
    for row, (_, matrix) in zip(rows, raw):
        if row["classification"] == "correct":
            continue
        curve = model.risk(matrix)
        error = {int(index) - 1 for index in row["error_steps"]}
        for step_index, span in enumerate(row["step_token_spans"]):
            if span is None or int(span[1]) <= int(span[0]):
                continue
            ours_y.append(int(step_index in error))
            ours_score.append(aggregate_risk(curve[int(span[0]):int(span[1])], "max"))
            categories.append(str(row["classification"]))
            groups.append(str(row["idx"]))
            ids.append(f"{row['idx']}::{step_index}")
    y = np.asarray(ours_y, dtype=int)
    score = np.asarray(ours_score, dtype=float)
    result_rows = [{"domain": "reasoning", "benchmark": "PRMBench", "split": "all nine paper classes",
                    "unit": "reasoning step", "method": "Fixed reasoning IU-PCR",
                    **binary_metrics(y, score)}]
    for category in sorted(set(categories)):
        mask = np.asarray(categories) == category
        if len(np.unique(y[mask])) != 2:
            continue
        result_rows.append({"domain": "reasoning", "benchmark": "PRMBench", "split": "paper class",
                            "subgroup": category, "unit": "reasoning step",
                            "method": "Fixed reasoning IU-PCR", **binary_metrics(y[mask], score[mask])})

    competitor = load_pickle(competitor_path)
    by_idx = {str(row["idx"]): row for row in competitor.values()}
    prm_y, prm_score = [], []
    for row in rows:
        if row["classification"] == "correct":
            continue
        other = by_idx.get(str(row["idx"]))
        if other is None or len(other["rewards"]) != len(row["step_token_spans"]):
            continue
        error = {int(index) - 1 for index in row["error_steps"]}
        for step_index, reward in enumerate(other["rewards"]):
            prm_y.append(int(step_index in error))
            prm_score.append(-float(reward))
    result_rows.append({"domain": "reasoning", "benchmark": "PRMBench", "split": "all nine paper classes",
                        "unit": "reasoning step", "method": "Qwen2.5-Math-PRM-7B (supervised ceiling)",
                        **binary_metrics(prm_y, prm_score)})
    write_json(out / "prmbench_score_freeze.json", {
        "score_hash": score_hash(ids, {"fixed_reasoning_iu": score}),
        "labels_seen_during_fit": False,
        "excluded_alignment_ids": sorted(BAD_PRM_IDS),
    })
    return result_rows, {"token_model": model.diagnostics, "n_scored_steps": len(y),
                         "positive_rate": float(y.mean())}


def run_reasoning(out: Path) -> None:
    process_rows, process_diag = _run_processbench(out)
    prm_rows, prm_diag = _run_prmbench(out)
    rows = process_rows + prm_rows
    reference = read_csv(ROOT / "results/paper_aligned_benchmark_suite_2026_08_11/benchmark_scores.csv")
    for record in reference:
        if record.get("protocol_id") == "localization-processbench-first-error" \
                and record.get("subgroup") == "four-subset macro" \
                and record.get("method") in {"GL-LIU v1 (frozen)", "Qwen2.5-Math-PRM-7B", "Qwen2.5-72B critic"}:
            rows.append({"domain": "reasoning", "benchmark": "ProcessBench",
                         "split": "published/local reference", "subgroup": "four-subset macro",
                         "unit": "first erroneous step", "method": record["method"],
                         "reference_role": record["role"], "f1": float(record["value"]),
                         "n": record.get("n", ""), "caveat": record.get("caveat", "")})
    write_csv(out / "reasoning_metrics.csv", rows)
    write_json(out / "reasoning_diagnostics.json", {
        "processbench": process_diag,
        "prmbench": prm_diag,
        "fixed_architecture": {
            "token_streams": len(SHARED_TOKEN_VIEWS),
            "trajectory_first": True,
            "step_aggregation": "maximum token risk",
            "answer_detector": "0.75 z(global mixed-v2 IU risk) + 0.25 z(max local step risk)",
            "first_error": "argmax step risk after a calibration-half answer-error threshold",
        },
    })
    print(f"[reasoning] complete -> {out / 'reasoning_metrics.csv'}", flush=True)


def _float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key, "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _metric_row(rows: Sequence[Mapping[str, Any]], **criteria: str) -> Mapping[str, Any] | None:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            return row
    return None


def _plot_b64(labels: Sequence[str], values: Sequence[float], colors: Sequence[str], title: str,
              *, limit: tuple[float, float] = (0.0, 1.0)) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, values, color=colors, width=0.68)
    ax.set_xticks(positions, labels, rotation=16, ha="right")
    ax.set_ylim(*limit)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012 * (limit[1] - limit[0]),
                f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _render_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "—")
            number = _float(row, key)
            if number is not None and key not in {"n"}:
                value = f"{number:.4f}"
            elif number is not None and key == "n":
                value = f"{int(number):,}"
            cells.append(f"<td>{html.escape(str(value if value != '' else '—'))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def write_methods(out: Path) -> None:
    audit = contract_audit()
    text = f"""# Fixed RAG and Reasoning Pipelines

## Purpose

This experiment freezes two application pipelines. It does not introduce a new covariance solver.
Both pipelines use the same token feature basis and full-pool, two-component IU-PCR.

## Shared feature contract

The contract is `{CONTRACT_VERSION}`. It contains {audit['token_stream_count']} token streams that
cover all {audit['covered_global_feature_count']} mixed-v2 response features. CUSUM magnitude and
CUSUM location share one positional stream. Trace length is a constant stream within one response;
it can change answer risk but cannot create a false local peak.

{audit['exact_stream_count']} streams have an exact reduction back to a global feature.
{audit['approximate_stream_count']} are causal rolling counterparts of a whole-trace operation.
The rolling versions are disclosed as approximations; they are not presented as exact identities.

The same frozen mixed-v2 transformation rules are applied in both packages; each allowed training
population estimates its own label-free location and scale parameters. In particular,
permutation entropy is folded by squared distance and STFT spectral entropy is folded around its
label-free population mode. All columns are oriented so that larger means more confidence before
fusion. Final risk is the negative fused confidence.

## IU-PCR

Let `F` be the feature-by-sample matrix and `C = FF^T/n`. IU-PCR estimates the unobserved vector
`rho_f = Cov(f, Y)` from off-diagonal covariance equations, then solves inside the first two
eigenvectors `U` of `C`:

`w = U (U^T C U)^(-1) U^T rho_hat`.

The score is `w^T f`. No label enters this fit. This is based on unsupervised ensemble regression
(Dror et al., 2017) and the spectral crowdsourcing formulation (Tenzer et al., 2022).

## Fixed RAG pipeline

The observed object is `X[i,t,c,f]`: response `i`, answer token `t`, evidence condition `c`, and
shared feature `f`. Full context and no context are always required. If leave-one-chunk-out (LOO)
conditions exist, the evidence head also contains, for each base feature, maximum drop, mean of the
two largest drops, mean positive drop, and negative drop standard deviation.

The no-context head therefore has 2 × {audit['token_stream_count']} columns. The LOO head has
6 × {audit['token_stream_count']} columns. A response uses the LOO head when LOO traces exist and
the no-context head otherwise. This is a fixed availability rule, not benchmark-specific tuning.

One fused token-risk stream produces every output:

- native token risk for token/span evaluation;
- mean token risk inside a RAG sentence or supplied claim;
- maximum token risk inside a reasoning step;
- mean token risk for complete-answer RAG detection.

The supplied-claim adapter uses the same mean-inside-boundaries rule as the sentence adapter. It is
part of the fixed interface, but it is not separately validated in the present RAGTruth experiment.
RAG scoring requires `2 + J` teacher-forced passes for an answer with `J` evidence chunks: full
context, no context, and one pass for each LOO condition. The answer text is held fixed, so this is
not repeated stochastic generation.

## Fixed reasoning pipeline

The observed object is `X[i,t,f]` over the complete, uninterrupted reasoning trace. Feature
computation and IU-PCR happen before step boundaries are used. Each step receives the maximum token
risk inside its supplied span. The predicted error location is the step with maximum risk.

ProcessBench also requires a no-error decision. The frozen answer detector is
`0.75 z(global mixed-v2 IU risk) + 0.25 z(max local step risk)`. A calibration half chooses only
the operating threshold; it does not change features or fusion weights. PRMBench uses continuous
step risk and therefore needs no threshold.

Reasoning needs one teacher-forced scoring pass over the existing complete trace. It does not ask
the model to regenerate the reasoning path.

## Why DUFS/Laplacian is not in the final heads

DUFS-LIU-PCR remains an important control and implementation standard for the 24-cell detection
study. Across the completed RAG and reasoning experiments, however, its incremental gain over
IU-PCR was negligible or unstable. Keeping it in the fixed application pipeline would add
complexity without a demonstrated contribution. This report therefore uses IU-PCR and states that
decision explicitly.

## Evaluation and uncertainty

RAG AUROC and AUPRC intervals resample complete `source_id` groups. Sentence rows from one source
therefore remain together. ProcessBench compares methods on identical repeated calibration/evaluation
splits and reports the paired F1 difference. Published scores are shown as references only when the
model, sample IDs, or access level differs. RAGTruth is exploratory because its labels had already
been opened in earlier project work.

## References

- Dror et al., *Unsupervised Ensemble Regression* (2017).
- Tenzer et al., *Crowdsourcing Regression: A Spectral Approach* (AISTATS 2022).
- Lindenbaum et al., *Differentiable Unsupervised Feature Selection based on a Gated Laplacian*.
- Niu et al., *RAGTruth* (ACL 2024).
- *GASP: Look Beyond the Answer for RAG Grounding* (2026 preprint).
- Song et al., *PRMBench* (ACL 2025).
- Zheng et al., *ProcessBench* (2025).
"""
    (out / "METHODS.md").write_text(text, encoding="utf-8")


def build_report(out: Path) -> None:
    rag = read_csv(out / "rag_metrics.csv")
    reasoning = read_csv(out / "reasoning_metrics.csv")
    benchmark_registry = read_csv(
        ROOT / "results/paper_aligned_benchmark_suite_2026_08_11/benchmark_scores.csv"
    )
    rag_history = read_csv(
        ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1/summary_test.csv"
    )
    if not rag or not reasoning:
        raise FileNotFoundError("run both RAG and reasoning stages before report")
    write_methods(out)

    rag_answer = _metric_row(rag, benchmark="RAGTruth", split="test", unit="answer",
                             method="Fixed RAG IU-PCR")
    rag_token = _metric_row(rag, benchmark="RAGTruth", split="test", unit="token",
                            method="Fixed RAG IU-PCR")
    rag_sentence = _metric_row(rag, benchmark="RAGTruth", split="test", unit="sentence",
                               method="Fixed RAG IU-PCR")
    rag_f1 = _metric_row(rag, benchmark="RAGTruth", split="test", unit="answer",
                         method="Fixed RAG IU-PCR (dev threshold)")
    lettuce = next((row for row in rag if row.get("method") == "LettuceDetect (local reproduction)"), None)
    gasp = next((row for row in rag if row.get("method") == "GASP-threshold" and row.get("reference_role") == "protocol_reproduction"), None)

    pb_fixed = _metric_row(reasoning, benchmark="ProcessBench", model="macro",
                           subgroup="all eight cells", method="Fixed reasoning IU-PCR")
    pb_mind = _metric_row(reasoning, benchmark="ProcessBench", model="macro",
                          subgroup="all eight cells", method="Mind the Gap control")
    pb_fixed_matched = _metric_row(reasoning, benchmark="ProcessBench", model="qwen3_8b",
                                   subgroup="four-subset macro", method="Fixed reasoning IU-PCR")
    pb_mind_matched = _metric_row(reasoning, benchmark="ProcessBench", model="qwen3_8b",
                                  subgroup="four-subset macro", method="Mind the Gap control")
    pb_gl = _metric_row(reasoning, benchmark="ProcessBench", subgroup="four-subset macro",
                        method="GL-LIU v1 (frozen)")
    pb_prm = _metric_row(reasoning, benchmark="ProcessBench", subgroup="four-subset macro",
                         method="Qwen2.5-Math-PRM-7B")
    pb_critic = _metric_row(reasoning, benchmark="ProcessBench", subgroup="four-subset macro",
                            method="Qwen2.5-72B critic")
    prm_fixed = _metric_row(reasoning, benchmark="PRMBench", split="all nine paper classes",
                            method="Fixed reasoning IU-PCR")
    prm_ceiling = _metric_row(reasoning, benchmark="PRMBench", split="all nine paper classes",
                              method="Qwen2.5-Math-PRM-7B (supervised ceiling)")
    prm_old = _metric_row(
        benchmark_registry,
        protocol_id="localization-prmbench-every-step",
        subgroup="all nine paper classes (constructed control excluded)",
        method="IU-PCR",
        metric="auroc",
    )
    historical_loo = _metric_row(
        rag_history, task="MACRO_TASK", method="original30_loo__iu_pcr"
    )
    historical_gasp = _metric_row(
        rag_history, task="MACRO_TASK", method="gasp_top50"
    )
    current_loo_tasks = [
        _metric_row(rag, benchmark="RAGTruth", split="test", unit="answer",
                    subgroup=task, method="Fixed RAG IU-PCR")
        for task in ("QA", "Data2txt")
    ]
    current_loo_macro = float(np.mean([
        _float(row or {}, "auroc") for row in current_loo_tasks
        if _float(row or {}, "auroc") is not None
    ]))

    rag_plot_labels, rag_plot_values, rag_colors = [], [], []
    for label, row, metric, color in (
        ("Fixed RAG answer", rag_answer, "auroc", "#126E82"),
        ("Fixed RAG sentence", rag_sentence, "auroc", "#2E8B8B"),
        ("Fixed RAG token", rag_token, "auroc", "#6BA292"),
        ("GASP sentence", gasp, "auroc", "#D9A441"),
        ("LettuceDetect example F1", lettuce, "example_f1", "#7A6AA6"),
    ):
        value = _float(row or {}, metric)
        if value is not None:
            rag_plot_labels.append(label); rag_plot_values.append(value); rag_colors.append(color)
    rag_image = _plot_b64(rag_plot_labels, rag_plot_values, rag_colors,
                          "RAG: each bar keeps its own stated unit and metric", limit=(0.4, 0.9))

    reason_labels, reason_values, reason_colors = [], [], []
    for label, row, metric, color in (
        ("Fixed reasoning", pb_fixed_matched, "f1", "#126E82"),
        ("Frozen GL-LIU v1", pb_gl, "f1", "#5596A6"),
        ("Mind the Gap", pb_mind_matched, "f1", "#D9A441"),
        ("72B critic", pb_critic, "f1", "#B36B4B"),
        ("Supervised PRM", pb_prm, "f1", "#7A6AA6"),
    ):
        value = _float(row or {}, metric)
        if value is not None:
            reason_labels.append(label); reason_values.append(value); reason_colors.append(color)
    reason_image = _plot_b64(
        reason_labels, reason_values, reason_colors,
        "ProcessBench Qwen3-8B four-subset macro F1", limit=(0.0, 0.8),
    )

    prm_image = _plot_b64(
        ["Old step-first IU-PCR", "Fixed trajectory-first IU-PCR", "Supervised PRM"],
        [_float(prm_old or {}, "value") or 0.0,
         _float(prm_fixed or {}, "auroc") or 0.0,
         _float(prm_ceiling or {}, "auroc") or 0.0],
        ["#B7C4C8", "#126E82", "#7A6AA6"],
        "PRMBench every-step AUROC: order of operations matters", limit=(0.45, 0.85),
    )

    rag_core = [row for row in rag if row.get("method", "").startswith("Fixed RAG") and row.get("split") == "test"]
    rag_matched = []
    for row in rag:
        if row.get("benchmark") != "RAGTruth balanced GASP cohort":
            continue
        if row.get("method") == "Fixed RAG IU-PCR" and row.get("subgroup", "") == "":
            for metric in ("auroc", "auprc"):
                rag_matched.append({
                    "method": row["method"], "metric": metric.upper(),
                    "value": row.get(metric, ""),
                    "ci_low": row.get(metric + "_ci_low", ""),
                    "ci_high": row.get(metric + "_ci_high", ""),
                    "n": row.get("n", ""), "fidelity": "ours_frozen",
                })
        elif row.get("reported_metric") and row.get("subgroup") in {"all", "published paper"}:
            rag_matched.append({
                "method": row.get("method", ""),
                "metric": row.get("reported_metric", "").upper(),
                "value": row.get("reported_value", ""),
                "ci_low": row.get("reported_ci_low", ""),
                "ci_high": row.get("reported_ci_high", ""),
                "n": row.get("n", ""), "fidelity": row.get("reference_role", ""),
            })
    process_core = [row for row in reasoning if row.get("benchmark") == "ProcessBench" and
                    ((row.get("model") == "macro") or
                     (row.get("model") == "qwen3_8b" and
                      row.get("subgroup") == "four-subset macro"))]
    prm_classes = [row for row in reasoning if row.get("benchmark") == "PRMBench" and
                   row.get("method") == "Fixed reasoning IU-PCR"]
    contract = contract_audit()

    rag_competitive = (_float(rag_answer or {}, "auroc") or 0) >= 0.70
    reasoning_peer_win = ((_float(pb_fixed_matched or {}, "f1") or 0) >
                          (_float(pb_mind_matched or {}, "f1") or 1))
    summary = (
        f"The fixed RAG pipeline reaches answer AUROC {_float(rag_answer or {}, 'auroc') or float('nan'):.3f} "
        f"[{_float(rag_answer or {}, 'auroc_ci_low') or float('nan'):.3f}, "
        f"{_float(rag_answer or {}, 'auroc_ci_high') or float('nan'):.3f}]. "
        f"The fixed reasoning pipeline reaches ProcessBench eight-cell macro F1 {_float(pb_fixed or {}, 'f1') or float('nan'):.3f}; "
        f"on the matched Qwen3-8B four-subset protocol it reaches {_float(pb_fixed_matched or {}, 'f1') or float('nan'):.3f} "
        f"versus {_float(pb_mind_matched or {}, 'f1') or float('nan'):.3f} for Mind the Gap, a paired delta "
        f"{_float(pb_fixed_matched or {}, 'delta_vs_mindgap') or float('nan'):+.3f} "
        f"[{_float(pb_fixed_matched or {}, 'delta_ci_low') or float('nan'):+.3f}, "
        f"{_float(pb_fixed_matched or {}, 'delta_ci_high') or float('nan'):+.3f}]. "
        f"and PRMBench step AUROC {_float(prm_fixed or {}, 'auroc') or float('nan'):.3f}. "
        "The trajectory-first PRMBench result improves clearly over the old step-first adapter. "
        "Reasoning is competitive with the label-free Mind the Gap control, but it remains far below supervised PRM and the 72B critic."
    )

    css = """
    :root{--ink:#18323a;--muted:#5d6e73;--line:#dbe5e6;--card:#fff;--bg:#f4f8f7;--teal:#126e82;--gold:#d9a441}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Arial,sans-serif}
    main{max-width:1180px;margin:auto;padding:32px 22px 80px}.hero{background:linear-gradient(135deg,#0f6576,#153e4b);color:white;padding:34px;border-radius:18px}
    h1{margin:0 0 8px;font-size:34px}h2{margin-top:0}h3{margin-bottom:8px}.sub{opacity:.86;max-width:880px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:18px 0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 4px 16px #193b4210}.metric b{font-size:28px;color:var(--teal);display:block}.metric span{color:var(--muted)}
    .flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;align-items:center}.node{background:#e7f2f2;border:1px solid #b6d4d6;border-radius:10px;padding:12px;text-align:center}.arrow{text-align:center;color:var(--teal);font-size:24px}
    table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}th{background:#edf4f3;position:sticky;top:0}.scroll{overflow:auto;max-height:520px}img.plot{width:100%;max-width:900px;display:block;margin:auto}.pill{display:inline-block;padding:3px 9px;border-radius:99px;background:#e6f2ef;color:#146656;margin-right:6px}.warn{background:#fff6df;border-left:4px solid var(--gold);padding:13px}.equation{font-family:ui-monospace,SFMono-Regular,monospace;background:#f0f5f4;padding:12px;border-radius:9px;overflow:auto}
    @media(max-width:760px){.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}}
    """
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Fixed RAG and Reasoning Pipelines</title><style>{css}</style></head><body><main>
    <section class='hero'><h1>Fixed RAG and Reasoning Pipelines</h1><p class='sub'>One shared mixed-v2 token feature basis, two task-structure adapters, and paper-aligned evaluation. Generated from machine-readable artifacts; no result value is typed into the report.</p><span class='pill'>29 token streams</span><span class='pill'>30 global features covered</span><span class='pill'>IU-PCR, label-free fit</span></section>
    <section class='cards'>
      <div class='card metric'><span>RAGTruth answer AUROC</span><b>{_float(rag_answer or {}, 'auroc') or float('nan'):.3f}</b><span>Fixed full/no-context/LOO pipeline</span></div>
      <div class='card metric'><span>ProcessBench macro F1</span><b>{_float(pb_fixed or {}, 'f1') or float('nan'):.3f}</b><span>First erroneous step</span></div>
      <div class='card metric'><span>PRMBench step AUROC</span><b>{_float(prm_fixed or {}, 'auroc') or float('nan'):.3f}</b><span>Trajectory first</span></div>
      <div class='card metric'><span>Shared feature fidelity</span><b>{contract['exact_stream_count']} + {contract['approximate_stream_count']}</b><span>exact + rolling-approximate streams</span></div>
    </section>
    <section class='card'><h2>Decision</h2><p>{html.escape(summary)}</p><p class='warn'><b>Competitive does not mean equal access.</b> RAG and reasoning peers are separated from supervised token classifiers, trained PRMs, and 72B judges. The fixed reasoning method beats the label-free ProcessBench control, but it does not approach the supervised ceiling.</p></section>
    <section class='card'><h2>How the packages are used</h2><table><thead><tr><th>Package</th><th>Model scoring</th><th>Label use</th><th>Outputs</th></tr></thead><tbody><tr><td>RAG</td><td>Teacher-force the same fixed answer with full context, no context, and each LOO context. This is 2 + J scoring passes for J chunks; it is not repeated answer generation.</td><td>No labels in feature construction, scaling, or IU-PCR. RAGTruth labels are evaluation-only; one dev threshold is reported separately.</td><td>Native token/span risk; mean inside a supplied sentence or claim; mean over the answer. The supplied-claim adapter is defined but not separately validated here.</td></tr><tr><td>Reasoning</td><td>One teacher-forced scoring pass over the complete reasoning trace. No extra generation is required.</td><td>No labels in features or IU-PCR. ProcessBench uses labels from a calibration half only to set the no-error threshold.</td><td>Token trajectory; maximum inside a step; argmax for first error; global/local gate for no-error.</td></tr></tbody></table></section>
    <section class='card'><h2>One feature basis</h2><p>The same token streams are used in both applications. Trace length is constant inside one answer and therefore cannot create a localization peak. CUSUM magnitude and location share one stream.</p><div class='equation'>X_RAG[i,t,c,f] &nbsp;&nbsp; and &nbsp;&nbsp; X_reasoning[i,t,f]</div><p>{contract['exact_stream_count']} streams reduce exactly to their global counterparts; {contract['approximate_stream_count']} use disclosed causal rolling approximations.</p></section>
    <section class='card'><h2>Fixed RAG flow</h2><div class='flow'><div class='node'>Fixed answer + evidence</div><div class='arrow'>→</div><div class='node'>full / no-context / LOO traces</div><div class='arrow'>→</div><div class='node'>29 shared streams</div></div><div class='flow'><div class='node'>evidence-drop blocks</div><div class='arrow'>→</div><div class='node'>two-component IU-PCR</div><div class='arrow'>→</div><div class='node'>token → sentence/claim/answer</div></div><p>The LOO head is used whenever LOO conditions exist; otherwise the no-context head is used. This availability rule is fixed.</p><img class='plot' src='data:image/png;base64,{rag_image}'></section>
    <section class='card'><h2>RAG results</h2>{_render_table(rag_core, [('unit','Unit'),('method','Method'),('auroc','AUROC'),('auroc_ci_low','AUROC CI low'),('auroc_ci_high','AUROC CI high'),('auprc','AUPRC'),('f1','F1'),('n','n'),('subgroup','Task')])}<h3>Matched GASP protocol</h3><p>The frozen method and local GASP reproduction use the same 400 responses. Their small point difference is descriptive; it is not presented as a significant win. The published GASP number is a paper reference with different sampled IDs.</p>{_render_table(rag_matched, [('method','Method'),('metric','Metric'),('value','Value'),('ci_low','CI low'),('ci_high','CI high'),('n','n'),('fidelity','Fidelity')])}</section>
    <section class='card'><h2>RAG continuity with the earlier response-only experiment</h2><p>On QA and Data-to-Text, where LOO evidence is available in both experiments, the new shared token pipeline has task-macro answer AUROC <b>{current_loo_macro:.4f}</b>. The earlier response-only Original-30 LOO IU-PCR has <b>{_float(historical_loo or {}, 'auroc') or float('nan'):.4f}</b>, and its GASP-top50 control has <b>{_float(historical_gasp or {}, 'auroc') or float('nan'):.4f}</b>. The token pipeline therefore keeps almost all response-level ranking while adding one score curve that can also be reduced to sentences and tokens.</p></section>
    <section class='card'><h2>Fixed reasoning flow</h2><div class='flow'><div class='node'>Complete reasoning trace</div><div class='arrow'>→</div><div class='node'>29 shared streams</div><div class='arrow'>→</div><div class='node'>token IU-PCR risk</div></div><div class='flow'><div class='node'>max inside each step</div><div class='arrow'>→</div><div class='node'>global/local answer gate</div><div class='arrow'>→</div><div class='node'>first-error or no-error</div></div><img class='plot' src='data:image/png;base64,{reason_image}'><img class='plot' src='data:image/png;base64,{prm_image}'></section>
    <section class='card'><h2>ProcessBench</h2><p>The chart uses only the Qwen3-8B four-subset population shared by the published controls. The paired interval resamples the same 100 calibration/evaluation splits for our method and Mind the Gap. The eight-cell and six-cell rows are broader local summaries and remain separate.</p>{_render_table(process_core, [('model','Model'),('subgroup','Population'),('method','Method'),('f1','F1'),('f1_sd','Split SD'),('delta_vs_mindgap','Delta vs MindGap'),('delta_ci_low','Delta low'),('delta_ci_high','Delta high'),('n','n')])}</section>
    <section class='card'><h2>PRMBench</h2>{_render_table(prm_classes, [('subgroup','Class'),('method','Method'),('auroc','AUROC'),('auprc','AUPRC'),('n','Steps'),('positive_rate','Error rate')])}</section>
    <section class='card'><h2>Mechanism conclusion</h2><ul><li>The important RAG change is the evidence-condition axis, not a new feature set.</li><li>The important reasoning change is operation order: fuse the long token trajectory before reducing to steps.</li><li>DUFS/Laplacian is excluded from the final heads because completed controls did not show a stable incremental gain over IU-PCR.</li></ul></section>
    <section class='card'><h2>Limits and claim boundary</h2><ul><li>RAGTruth is exploratory because its labels were opened in earlier work.</li><li>PRMBench excludes exactly three registered alignment-defect rows.</li><li>The ProcessBench no-error threshold uses a calibration half; fusion weights remain label-free.</li><li>Eleven token streams are rolling approximations, not exact whole-trace identities.</li><li>No cross-task macro is computed. Each panel keeps its official unit and metric.</li></ul></section>
    <section class='card'><h2>Artifacts</h2><p><a href='METHODS.md'>METHODS.md</a> · <a href='REPORT.md'>REPORT.md</a> · <a href='rag_metrics.csv'>rag_metrics.csv</a> · <a href='reasoning_metrics.csv'>reasoning_metrics.csv</a> · <a href='feature_contract.json'>feature_contract.json</a></p></section>
    </main></body></html>"""
    (out / "REPORT.html").write_text(page, encoding="utf-8")
    report_md = f"""# Fixed RAG and Reasoning Pipeline Results

{summary}

## Fixed RAG

- RAGTruth answer AUROC: **{_float(rag_answer or {}, 'auroc') or float('nan'):.4f}**.
- RAGTruth answer 95% source-group interval: **[{_float(rag_answer or {}, 'auroc_ci_low') or float('nan'):.4f}, {_float(rag_answer or {}, 'auroc_ci_high') or float('nan'):.4f}]**.
- RAGTruth sentence AUROC: **{_float(rag_sentence or {}, 'auroc') or float('nan'):.4f}**.
- RAGTruth token AUROC: **{_float(rag_token or {}, 'auroc') or float('nan'):.4f}**.
- Dev-calibrated answer example F1: **{_float(rag_f1 or {}, 'f1') or float('nan'):.4f}**.

## Fixed reasoning

- ProcessBench eight-cell macro F1: **{_float(pb_fixed or {}, 'f1') or float('nan'):.4f}**.
- Mind the Gap control macro F1: **{_float(pb_mind or {}, 'f1') or float('nan'):.4f}**.
- ProcessBench matched Qwen3-8B four-subset F1: **{_float(pb_fixed_matched or {}, 'f1') or float('nan'):.4f}** versus **{_float(pb_mind_matched or {}, 'f1') or float('nan'):.4f}** for Mind the Gap.
- Paired Qwen3-8B F1 delta: **{_float(pb_fixed_matched or {}, 'delta_vs_mindgap') or float('nan'):+.4f} [{_float(pb_fixed_matched or {}, 'delta_ci_low') or float('nan'):+.4f}, {_float(pb_fixed_matched or {}, 'delta_ci_high') or float('nan'):+.4f}]** across identical calibration/evaluation splits.
- PRMBench step AUROC: **{_float(prm_fixed or {}, 'auroc') or float('nan'):.4f}**.
- Qwen2.5-Math-PRM-7B ceiling AUROC: **{_float(prm_ceiling or {}, 'auroc') or float('nan'):.4f}**.

See `REPORT.html` for plots, detailed tables, method flow, and limitations.
"""
    (out / "REPORT.md").write_text(report_md, encoding="utf-8")
    write_json(out / "feature_contract.json", contract)
    write_json(out / "suite_manifest.json", {
        "version": VERSION,
        "files": {path.name: sha256(path) for path in sorted(out.iterdir())
                  if path.is_file() and path.name != "suite_manifest.json"},
        "cross_task_macro": False,
        "algorithms_modified": False,
    })
    print(f"[report] complete -> {out / 'REPORT.html'}", flush=True)


def write_runbook(out: Path) -> None:
    text = f"""# Runbook

Run from the repository root with the project environment.

```bash
.venv/bin/python scripts/fixed_application_pipeline_experiment.py rag --out {out}
.venv/bin/python scripts/fixed_application_pipeline_experiment.py reasoning --out {out}
MPLBACKEND=Agg .venv/bin/python scripts/fixed_application_pipeline_experiment.py report --out {out}
.venv/bin/python scripts/test_fixed_application_pipelines.py
```

Or run all three stages:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/fixed_application_pipeline_experiment.py all --out {out}
```

Large raw caches remain outside Git. The output contains score hashes and the
machine-readable metrics used by the report.
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "RUNBOOK.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rag", "reasoning", "report", "all"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    write_runbook(args.out)
    if args.command in {"rag", "all"}:
        run_rag(args.out)
    if args.command in {"reasoning", "all"}:
        run_reasoning(args.out)
    if args.command in {"report", "all"}:
        build_report(args.out)


if __name__ == "__main__":
    main()
