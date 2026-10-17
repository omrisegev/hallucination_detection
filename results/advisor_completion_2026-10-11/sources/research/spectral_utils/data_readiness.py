"""Dataset-only integrity auditing for the project's collected artifacts.

This module deliberately has no dependency on any hallucination detector.  It
checks immutable raw caches, manifests, labels, trace alignment, and benchmark
coverage.  It may resolve a Git-LFS pointer to an object that is already in the
local object store, but it never changes or checks out the pointer file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


SCHEMA_VERSION = "data-readiness-v1-2026-08-11"
READY = "READY"
READY_WITH_LIMITATIONS = "READY_WITH_LIMITATIONS"
INCOMPLETE = "INCOMPLETE"
BLOCKED = "BLOCKED"


class RestrictedUnpickler(pickle.Unpickler):
    """Load project caches without permitting arbitrary Python globals."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked unsafe pickle global: {module}.{name}")


@dataclass(frozen=True)
class CanonicalUnit:
    """A label-free description of one stored benchmark unit.

    Large token arrays stay in their immutable source file.  ``artifact`` and
    ``row_key`` identify where they can be loaded; this record is the common
    metadata interface used by future dataset adapters.
    """

    dataset_id: str
    record_id: str
    source_id: str
    split: str
    task: str
    model_id: str
    artifact: str
    row_key: str
    condition: str = ""
    parent_id: str = ""


@dataclass(frozen=True)
class CanonicalLabel:
    """A label sidecar record, kept structurally separate from telemetry."""

    dataset_id: str
    record_id: str
    label_space: str
    value: Any
    provenance: str


@dataclass
class Audit:
    dataset_id: str
    title: str
    kind: str
    status: str
    source_paths: list[str] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)
    balance: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    file_hashes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_blocker(self, message: str, *, incomplete: bool = False) -> None:
        self.blockers.append(message)
        self.status = INCOMPLETE if incomplete and self.status != BLOCKED else BLOCKED

    def add_limitation(self, message: str) -> None:
        self.limitations.append(message)
        if self.status == READY:
            self.status = READY_WITH_LIMITATIONS

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_lfs_path(repo: Path, path: Path) -> tuple[Optional[Path], Optional[str]]:
    """Return the actual local content path and, when applicable, its LFS oid."""

    if not path.exists():
        return None, None
    if path.stat().st_size > 1024:
        return path, None
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return path, None
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return path, None
    match = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    if not match:
        return None, None
    oid = match.group(1)
    obj = repo / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    return (obj if obj.exists() else None), oid


def restricted_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return RestrictedUnpickler(handle).load()


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read physical JSONL records without splitting embedded Unicode separators."""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _finite(values: Any) -> bool:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(array)))


def _trace_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "gen_token_ids",
        "token_entropies",
        "token_spilled_energies",
        "token_logsumexp",
        "top_k_logprobs",
    )
    missing = [key for key in required if key not in row]
    if missing:
        return [f"missing trace fields: {', '.join(missing)}"]
    n = len(row["gen_token_ids"])
    if n == 0:
        errors.append("empty token trace")
    for key in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        if len(row[key]) != n:
            errors.append(f"{key} length {len(row[key])} != token length {n}")
        elif not _finite(row[key]):
            errors.append(f"{key} contains a non-finite value")
    topk = row["top_k_logprobs"]
    if not isinstance(topk, Mapping) or not {"ids", "logprobs"} <= set(topk):
        errors.append("top_k_logprobs does not contain ids and logprobs")
    else:
        ids = np.asarray(topk["ids"])
        logprobs = np.asarray(topk["logprobs"])
        if ids.shape != logprobs.shape or ids.ndim != 2 or ids.shape[0] != n:
            errors.append(
                f"top-k shape mismatch ids={ids.shape}, logprobs={logprobs.shape}, n={n}"
            )
        elif not _finite(logprobs):
            errors.append("top-k log probabilities contain a non-finite value")
    return errors


def _hash_sources(repo: Path, audit: Audit) -> None:
    for relative in audit.source_paths:
        path = repo / relative
        resolved, oid = resolve_lfs_path(repo, path)
        if resolved is None:
            audit.file_hashes[relative] = "MISSING"
            continue
        actual = sha256_file(resolved)
        audit.file_hashes[relative] = actual
        if oid is not None and actual != oid:
            audit.add_blocker(f"Git-LFS object hash disagrees for {relative}")


def audit_frozen_24cell(repo: Path) -> Audit:
    relative = Path("results/dependency_fusion_raw/cells.npz")
    manifest_relative = Path("results/dependency_fusion_raw/cells_manifest.csv")
    audit = Audit(
        "frozen_24cell",
        "Frozen 24-cell development collection",
        "answer-level feature matrices",
        READY,
        [str(relative), str(manifest_relative)],
        expected={"cells": 24, "rows": 48607},
        provenance={"role": "retrospective development collection"},
    )
    try:
        with np.load(repo / relative, allow_pickle=True) as bundle:
            cells = sorted(key[:-8] for key in bundle.files if key.endswith("__labels"))
            total = 0
            positives = 0
            finite = True
            shape_ok = True
            label_ok = True
            for cell in cells:
                required = [f"{cell}__{name}" for name in ("V", "F", "labels", "pool")]
                if not all(key in bundle.files for key in required):
                    shape_ok = False
                    continue
                V = np.asarray(bundle[f"{cell}__V"], dtype=float)
                F = np.asarray(bundle[f"{cell}__F"], dtype=float)
                labels = np.asarray(bundle[f"{cell}__labels"])
                pool = np.asarray(bundle[f"{cell}__pool"], dtype=object)
                shape_ok &= V.ndim == 2 and F.shape == V.T.shape and labels.shape == (V.shape[0],)
                shape_ok &= len(pool) == V.shape[1]
                finite &= bool(np.all(np.isfinite(V))) and bool(np.all(np.isfinite(F)))
                label_ok &= bool(np.all(np.isin(labels, [0, 1])))
                total += len(labels)
                positives += int(np.sum(labels == 1))
            audit.observed.update(
                cells=len(cells), rows=total, features_min=min(
                    np.asarray(bundle[f"{c}__V"]).shape[1] for c in cells
                ), features_max=max(
                    np.asarray(bundle[f"{c}__V"]).shape[1] for c in cells
                )
            )
            audit.balance = {
                "positive": positives,
                "negative": total - positives,
                "positive_rate": positives / total if total else None,
            }
            audit.checks = {
                "expected_cell_count": len(cells) == 24,
                "expected_row_count": total == 48607,
                "matrix_shapes": shape_ok,
                "finite_matrices": finite,
                "binary_labels": label_ok,
            }
    except Exception as exc:  # report corruption as data status, not a traceback-only failure
        audit.add_blocker(f"Could not validate 24-cell bundle: {exc}")
    if not all(audit.checks.values()):
        audit.add_blocker("One or more 24-cell structural checks failed")
    audit.add_limitation(
        "The cells use heterogeneous datasets and graders and were repeatedly used during method development."
    )
    _hash_sources(repo, audit)
    return audit


def audit_candidate_cache(
    repo: Path,
    dataset_id: str,
    title: str,
    relative: str,
    manifest_relative: str,
    expected_rows: int,
    *,
    hle: bool = False,
) -> Audit:
    audit = Audit(
        dataset_id,
        title,
        "answer-level generation telemetry",
        READY,
        [relative, manifest_relative],
        expected={"rows": expected_rows},
    )
    try:
        rows = restricted_pickle(repo / relative)
        manifest = _manifest(repo / manifest_relative)
        trace_failures = 0
        duplicate_questions = 0
        seen_questions: set[str] = set()
        labels: list[bool] = []
        for row in rows.values():
            question = str(row.get("question", ""))
            duplicate_questions += int(question in seen_questions)
            seen_questions.add(question)
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 1:
                trace_failures += 1
                continue
            candidate = candidates[0]
            trace_failures += int(bool(_trace_errors(candidate)))
            label_key = "label" if hle else "bem_correct"
            if label_key in candidate:
                labels.append(bool(candidate[label_key]))
        audit.observed = {
            "rows": len(rows),
            "duplicate_question_texts": duplicate_questions,
            "trace_failures": trace_failures,
        }
        audit.balance = {
            "positive": int(sum(labels)),
            "negative": len(labels) - int(sum(labels)),
            "positive_rate": float(np.mean(labels)) if labels else None,
        }
        audit.provenance = {
            "model": manifest.get("model"),
            "grader": "placeholder ROUGE-L" if hle else manifest.get("model"),
            "grader_threshold": manifest.get("threshold"),
            "input_sha256": manifest.get("input_sha256"),
        }
        audit.checks = {
            "expected_rows": len(rows) == expected_rows,
            "trace_integrity": trace_failures == 0,
            "labels_present": len(labels) == len(rows),
        }
    except Exception as exc:
        audit.add_blocker(f"Could not validate {dataset_id}: {exc}")
    if hle:
        audit.add_blocker(
            "The stored correctness label is a placeholder ROUGE-L decision; a real grader has not landed."
        )
    elif not all(audit.checks.values()):
        audit.add_blocker(f"One or more {dataset_id} structural checks failed")
    # SemGrad uses BEM as its primary correctness metric.  A separate human or
    # stronger-LLM audit can be useful as a robustness analysis, but it is not a
    # prerequisite for a paper-faithful SemGrad evaluation.
    _hash_sources(repo, audit)
    return audit


def validate_hle_judge_rows(
    queue_rows: list[Mapping[str, Any]],
    judged_rows: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    """Validate an interim HLE judgment set against the protected local queue."""

    required = {
        "row_key", "id", "extracted_final_answer", "reasoning", "correct", "confidence",
        "judge_model", "judge_reasoning_effort", "judge_protocol",
    }
    schema_ok = all(required <= set(row) for row in judged_rows)
    allowed_labels = all(row.get("correct") in {"yes", "no"} for row in judged_rows)
    confidence_ok = all(
        isinstance(row.get("confidence"), (int, float))
        and 0 <= float(row["confidence"]) <= 100
        for row in judged_rows
    )
    keys = [row.get("row_key") for row in judged_rows]
    ids = [str(row.get("id", "")) for row in judged_rows]
    aligned = len(queue_rows) == len(judged_rows) and all(
        queue.get("row_key") == judged.get("row_key")
        and str(queue.get("id")) == str(judged.get("id"))
        for queue, judged in zip(queue_rows, judged_rows)
    )
    labels = [row.get("correct") == "yes" for row in judged_rows]
    proxy = [bool(row.get("provisional_rouge_label")) for row in queue_rows]
    by_answer_type: dict[str, dict[str, Any]] = {}
    for answer_type in sorted({str(row.get("answer_type", "unknown")) for row in queue_rows}):
        indexes = [
            index for index, row in enumerate(queue_rows)
            if str(row.get("answer_type", "unknown")) == answer_type
        ]
        positives = sum(labels[index] for index in indexes) if aligned else 0
        by_answer_type[answer_type] = {
            "correct": int(positives),
            "total": len(indexes),
            "accuracy": positives / len(indexes) if indexes else None,
        }
    counts = {
        "rows": len(judged_rows),
        "correct": int(sum(labels)),
        "incorrect": len(labels) - int(sum(labels)),
        "accuracy": float(np.mean(labels)) if labels else None,
        "agreement_with_provisional_rouge": (
            float(np.mean(np.asarray(labels) == np.asarray(proxy))) if aligned else None
        ),
        "judge_correct_rouge_incorrect": (
            int(sum(label and not rough for label, rough in zip(labels, proxy))) if aligned else None
        ),
        "judge_incorrect_rouge_correct": (
            int(sum(not label and rough for label, rough in zip(labels, proxy))) if aligned else None
        ),
        "by_answer_type": by_answer_type,
    }
    provenance = {
        "judge_models": sorted({str(row.get("judge_model", "")) for row in judged_rows}),
        "reasoning_efforts": sorted({
            str(row.get("judge_reasoning_effort", "")) for row in judged_rows
        }),
        "protocols": sorted({str(row.get("judge_protocol", "")) for row in judged_rows}),
    }
    checks = {
        "exact_row_count": len(judged_rows) == len(queue_rows) == 2158,
        "source_alignment": aligned,
        "unique_row_keys": len(keys) == len(set(keys)),
        "deterministic_row_order": keys == list(range(len(keys))),
        "unique_source_ids": len(ids) == len(set(ids)),
        "required_fields": schema_ok,
        "allowed_labels": allowed_labels,
        "confidence_range": confidence_ok,
        "single_judge_model": provenance["judge_models"] == ["gpt-5.6-sol"],
        "single_reasoning_effort": provenance["reasoning_efforts"] == ["xhigh"],
    }
    return counts, provenance, checks


def audit_hle(repo: Path) -> Audit:
    """Audit HLE raw telemetry and prefer a fully validated interim judge sidecar."""

    audit = audit_candidate_cache(
        repo,
        "hle_qwen72b",
        "Humanity's Last Exam Qwen2.5-72B",
        "dataset_cache/four_localization/hle_full/raw_hle_T0.0.pkl",
        "dataset_cache/four_localization/hle_full/manifest.json",
        2158,
        hle=True,
    )
    queue_relative = "local_cache/data_readiness/hle_official_judge_queue.jsonl"
    judged_relative = "local_cache/data_readiness/hle_codex_5p6_sol_xhigh.jsonl"
    judged_manifest_relative = (
        "results/data_readiness_2026_08_11/hle_codex_5p6_sol_xhigh_manifest.json"
    )
    paths = [repo / queue_relative, repo / judged_relative, repo / judged_manifest_relative]
    if not all(path.exists() for path in paths):
        return audit
    try:
        # ``str.splitlines`` would also split Unicode separators that can occur
        # legitimately inside an HLE question.
        queue_rows = read_jsonl(paths[0])
        judged_rows = read_jsonl(paths[1])
        judge_manifest = _manifest(paths[2])
        counts, provenance, checks = validate_hle_judge_rows(queue_rows, judged_rows)
        manifest_hash = judge_manifest.get("hashes", {}).get("output_judgments_sha256")
        checks["judgment_hash"] = sha256_file(paths[1]) == manifest_hash
        checks["queue_hash"] = (
            sha256_file(paths[0])
            == judge_manifest.get("hashes", {}).get("input_queue_sha256")
        )
        checks["manifest_complete"] = (
            judge_manifest.get("completion", {}).get("status") == "complete"
        )
        audit.source_paths.extend([queue_relative, judged_relative, judged_manifest_relative])
        audit.observed.update({
            "interim_judge": counts,
            "interim_judge_manifest": str(paths[2].relative_to(repo)),
        })
        audit.checks.update({f"interim_{key}": value for key, value in checks.items()})
        if all(checks.values()):
            audit.blockers.clear()
            audit.status = READY
            audit.balance = {
                "correct": counts["correct"],
                "incorrect": counts["incorrect"],
                "accuracy": counts["accuracy"],
            }
            audit.provenance = {
                "generation_model": "Qwen/Qwen2.5-72B-Instruct",
                "grader": "gpt-5.6-sol",
                "grader_reasoning_effort": "xhigh",
                "grader_role": "interim independent Codex judge",
                "grader_protocol": provenance["protocols"],
                "paper_faithful_grader": False,
            }
            audit.add_limitation(
                "The complete labels come from an interim gpt-5.6-sol xhigh Codex judge, not the original paper's GPT-4o judge. Preserve both label sets when paper-faithful grading becomes available."
            )
        else:
            audit.add_blocker("The interim HLE judgment sidecar failed validation")
        _hash_sources(repo, audit)
    except Exception as exc:
        audit.add_blocker(f"Could not validate interim HLE judgments: {exc}")
    return audit


def audit_rag_conditions(
    repo: Path,
    dataset_id: str,
    title: str,
    relative: str,
    manifest_relative: str,
    expected_responses: int,
    expected_conditions: int,
    *,
    protocol_limitation: str = "",
) -> Audit:
    audit = Audit(
        dataset_id,
        title,
        "RAG evidence-condition telemetry",
        READY,
        [relative, manifest_relative],
        expected={"responses": expected_responses, "conditions": expected_conditions},
    )
    try:
        raw = restricted_pickle(repo / relative)
        manifest = _manifest(repo / manifest_relative)
        grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        trace_failures = 0
        duplicate_conditions = 0
        key_mismatches = 0
        for key, row in raw.items():
            response_id = str(row.get("response_id", ""))
            condition = str(row.get("condition", ""))
            expected_key = f"{response_id}::{condition}"
            key_mismatches += int(str(key) != expected_key)
            duplicate_conditions += int(condition in grouped[response_id])
            grouped[response_id][condition] = row
            trace_failures += int(bool(_trace_errors(row)))
        missing_pairs = 0
        token_mismatches = 0
        noncontiguous_loo = 0
        task_counts: Counter[str] = Counter()
        response_labels: list[bool] = []
        source_ids: set[str] = set()
        for response_id, rows in grouped.items():
            missing_pairs += int(not {"full", "noctx"} <= set(rows))
            full = rows.get("full")
            if full is None:
                continue
            reference = np.asarray(full["gen_token_ids"])
            task_counts[str(full.get("task_type", "unknown"))] += 1
            source_ids.add(str(full.get("source_id", response_id)))
            response_labels.append(bool(full.get("response_label", False)))
            indexes = sorted(
                int(name.split("_", 1)[1]) for name in rows if name.startswith("loo_")
            )
            noncontiguous_loo += int(bool(indexes) and indexes != list(range(max(indexes) + 1)))
            for row in rows.values():
                token_mismatches += int(not np.array_equal(reference, row["gen_token_ids"]))
        audit.observed = {
            "responses": len(grouped),
            "conditions": len(raw),
            "sources": len(source_ids),
            "task_counts": dict(sorted(task_counts.items())),
            "trace_failures": trace_failures,
            "missing_full_noctx_pairs": missing_pairs,
            "token_mismatches_across_conditions": token_mismatches,
            "duplicate_conditions": duplicate_conditions,
            "key_mismatches": key_mismatches,
            "noncontiguous_loo": noncontiguous_loo,
        }
        audit.balance = {
            "hallucinated": int(sum(response_labels)),
            "clean": len(response_labels) - int(sum(response_labels)),
            "hallucinated_rate": float(np.mean(response_labels)) if response_labels else None,
        }
        audit.provenance = {
            "model": manifest.get("model"),
            "paper": manifest.get("paper"),
            "fidelity_level": manifest.get("fidelity_level"),
            "written_utc": manifest.get("written_utc"),
        }
        audit.checks = {
            "expected_responses": len(grouped) == expected_responses,
            "expected_conditions": len(raw) == expected_conditions,
            "unique_conditions": duplicate_conditions == 0,
            "cache_keys": key_mismatches == 0,
            "full_noctx_complete": missing_pairs == 0,
            "condition_token_identity": token_mismatches == 0,
            "loo_indexes_contiguous": noncontiguous_loo == 0,
            "trace_integrity": trace_failures == 0,
        }
    except Exception as exc:
        audit.add_blocker(f"Could not validate {dataset_id}: {exc}")
    if audit.checks and not all(audit.checks.values()):
        audit.add_blocker(f"One or more {dataset_id} structural checks failed")
    if protocol_limitation:
        audit.add_limitation(protocol_limitation)
    _hash_sources(repo, audit)
    return audit


def audit_refchecker(repo: Path) -> Audit:
    relative = "dataset_cache/four_localization/refchecker_knowhalbench_open_full/refchecker_claim_telemetry.pkl"
    manifest_relative = "dataset_cache/four_localization/refchecker_knowhalbench_open_full/manifest.json"
    audit = Audit(
        "refchecker_claims",
        "RefChecker fixed human-labelled claims",
        "claim-level evidence telemetry",
        READY,
        [relative, manifest_relative],
        expected={"claims": 10733, "conditions": 21466},
    )
    try:
        raw = restricted_pickle(repo / relative)
        manifest = _manifest(repo / manifest_relative)
        grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        trace_failures = 0
        labels: Counter[str] = Counter()
        settings: Counter[str] = Counter()
        for key, row in raw.items():
            condition = str(row.get("condition", ""))
            parent = str(key).rsplit("::", 1)[0]
            grouped[parent][condition] = row
            trace_failures += int(bool(_trace_errors(row)))
            if condition == "full":
                labels[str(row.get("human_label", "missing"))] += 1
                settings[str(row.get("setting", "missing"))] += 1
        incomplete = sum(set(rows) != {"full", "noctx"} for rows in grouped.values())
        token_mismatch = sum(
            not np.array_equal(rows["full"]["gen_token_ids"], rows["noctx"]["gen_token_ids"])
            for rows in grouped.values() if set(rows) == {"full", "noctx"}
        )
        audit.observed = {
            "claims": len(grouped),
            "conditions": len(raw),
            "settings": dict(sorted(settings.items())),
            "trace_failures": trace_failures,
            "incomplete_pairs": incomplete,
            "token_mismatches_across_conditions": token_mismatch,
        }
        audit.balance = dict(sorted(labels.items()))
        audit.provenance = {
            "claim_set": manifest.get("claim_set"),
            "benchmark_data": manifest.get("benchmark_data"),
            "written_utc": manifest.get("written_utc"),
        }
        audit.checks = {
            "expected_claims": len(grouped) == 10733,
            "expected_conditions": len(raw) == 21466,
            "full_noctx_complete": incomplete == 0,
            "condition_token_identity": token_mismatch == 0,
            "trace_integrity": trace_failures == 0,
            "three_way_labels_complete": sum(labels.values()) == 10733,
        }
    except Exception as exc:
        audit.add_blocker(f"Could not validate RefChecker: {exc}")
    if audit.checks and not all(audit.checks.values()):
        audit.add_blocker("One or more RefChecker structural checks failed")
    audit.add_limitation(
        "The gold labels cover the fixed shipped claims. This is claim checking, not claim extraction."
    )
    _hash_sources(repo, audit)
    return audit


def _processbench_files(prefix: str) -> list[str]:
    return [f"{prefix}/processbench_{subset}.pkl" for subset in (
        "gsm8k", "math", "olympiadbench", "omnimath"
    )]


def audit_processbench_telemetry(repo: Path, model: str) -> Audit:
    prefix = f"dataset_cache/repgrid/pb_{model}"
    paths = _processbench_files(prefix)
    audit = Audit(
        f"processbench_{model}",
        f"ProcessBench telemetry ({model.replace('_', ' ')})",
        "first-error benchmark telemetry",
        READY,
        paths,
        expected={"rows": 3400, "subsets": 4},
        provenance={"dataset": "Qwen/ProcessBench", "labels": "official first-error labels"},
    )
    rows_total = 0
    error = 0
    correct = 0
    trace_failures = 0
    alignment_failures = 0
    missing_lfs: list[str] = []
    subset_counts: dict[str, int] = {}
    for relative in paths:
        path, _ = resolve_lfs_path(repo, repo / relative)
        if path is None:
            missing_lfs.append(relative)
            continue
        try:
            rows = restricted_pickle(path)
            subset = Path(relative).stem.replace("processbench_", "")
            subset_counts[subset] = len(rows)
            rows_total += len(rows)
            for row in rows.values():
                steps = row.get("steps") or []
                label = int(row.get("label", -999))
                if label == -1:
                    correct += 1
                else:
                    error += 1
                    alignment_failures += int(not (0 <= label < len(steps)))
                spans = row.get("step_token_spans") or []
                alignment_failures += int(len(spans) != len(steps))
                alignment_failures += int(not bool(row.get("align_diag", {}).get("ok", True)))
                trace_failures += int(bool(_trace_errors(row)))
        except Exception as exc:
            audit.warnings.append(f"{relative}: {exc}")
            alignment_failures += 1
    audit.observed = {
        "rows": rows_total,
        "subsets": len(subset_counts),
        "subset_counts": dict(sorted(subset_counts.items())),
        "missing_lfs_objects": missing_lfs,
        "trace_failures": trace_failures,
        "alignment_failures": alignment_failures,
    }
    audit.balance = {
        "first_error_present": error,
        "fully_correct": correct,
        "error_rate": error / rows_total if rows_total else None,
    }
    audit.checks = {
        "all_lfs_objects_local": not missing_lfs,
        "expected_rows": rows_total == 3400,
        "all_four_subsets": len(subset_counts) == 4,
        "trace_integrity": trace_failures == 0,
        "step_alignment": alignment_failures == 0,
    }
    if not all(audit.checks.values()):
        audit.add_blocker(f"One or more ProcessBench {model} data checks failed", incomplete=bool(missing_lfs))
    _hash_sources(repo, audit)
    return audit


def audit_processbench_competitor(
    repo: Path,
    dataset_id: str,
    title: str,
    directory: str,
    stem: str,
    *,
    manifest_required: bool = True,
) -> Audit:
    subsets = ("gsm8k", "math", "olympiadbench", "omnimath")
    paths = [f"{directory}/{stem}_{subset}.pkl" for subset in subsets]
    manifest = f"{directory}/manifest.json"
    source_paths = paths + ([manifest] if (repo / manifest).exists() else [])
    audit = Audit(
        dataset_id,
        title,
        "competitor prediction cache",
        READY,
        source_paths,
        expected={"rows": 3400, "subsets": 4, "manifest": manifest_required},
    )
    total = 0
    available = 0
    subset_counts: dict[str, int] = {}
    for subset, relative in zip(subsets, paths):
        if not (repo / relative).exists():
            continue
        try:
            rows = restricted_pickle(repo / relative)
            total += len(rows)
            available += 1
            subset_counts[subset] = len(rows)
        except Exception as exc:
            audit.warnings.append(f"{relative}: {exc}")
    has_manifest = (repo / manifest).exists()
    audit.observed = {
        "rows": total,
        "subsets": available,
        "subset_counts": dict(sorted(subset_counts.items())),
        "manifest": has_manifest,
    }
    audit.checks = {
        "expected_rows": total == 3400,
        "all_four_subsets": available == 4,
        "manifest_present": has_manifest or not manifest_required,
    }
    if not all(audit.checks.values()):
        audit.add_blocker(f"{title} is not a complete four-subset package", incomplete=True)
    _hash_sources(repo, audit)
    return audit


def audit_prmbench(repo: Path) -> list[Audit]:
    telemetry_relative = "dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl"
    telemetry_manifest = "dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/manifest.json"
    telemetry = Audit(
        "prmbench_qwen3_8b_telemetry",
        "PRMBench Qwen3-8B telemetry",
        "every-step benchmark telemetry",
        READY,
        [telemetry_relative, telemetry_manifest],
        expected={"rows": 6969, "step_spans": 94203, "official_error_class_steps": 83371},
    )
    try:
        rows = restricted_pickle(repo / telemetry_relative)
        manifest = _manifest(repo / telemetry_manifest)
        span_count = 0
        trace_failures = 0
        misaligned = 0
        misaligned_ids: list[str] = []
        classes: Counter[str] = Counter()
        for row in rows.values():
            steps = row.get("steps") or []
            spans = row.get("step_token_spans") or []
            span_count += len(spans)
            classes[str(row.get("classification", "missing"))] += 1
            problems = list(row.get("align_diag", {}).get("problems") or [])
            row_misaligned = len(spans) != len(steps) or bool(problems)
            misaligned += int(row_misaligned)
            if row_misaligned:
                misaligned_ids.append(str(row.get("idx", row.get("source_idx", "unknown"))))
            trace_failures += int(bool(_trace_errors(row)))
        official_steps = manifest.get("dataset_diagnostics", {}).get("n_steps_error_classes")
        telemetry.observed = {
            "rows": len(rows),
            "step_spans": span_count,
            "official_error_class_steps": official_steps,
            "classes": dict(sorted(classes.items())),
            "trace_failures": trace_failures,
            "misaligned_rows_or_span_sets": misaligned,
            "misaligned_ids": sorted(misaligned_ids),
        }
        telemetry.checks = {
            "expected_rows": len(rows) == 6969,
            "expected_span_count": span_count == 94203,
            "official_loader_count_recorded": official_steps == 83371,
            "trace_integrity": trace_failures == 0,
            "step_alignment": misaligned == 0,
        }
    except Exception as exc:
        telemetry.add_blocker(f"Could not validate PRMBench telemetry: {exc}")
    if telemetry.checks and not all(telemetry.checks.values()):
        telemetry.add_limitation(
            "Three rows were reported as misaligned; they must be identified and explicitly resolved or excluded before evaluation."
        )
    _hash_sources(repo, telemetry)

    prediction_relative = "dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl"
    prediction_manifest = "dataset_cache/four_localization/prmbench_qwen25math7b_full/manifest.json"
    prediction = Audit(
        "prmbench_qwen25math7b_predictions",
        "PRMBench Qwen2.5-Math-PRM predictions",
        "competitor prediction cache",
        READY,
        [prediction_relative, prediction_manifest],
        expected={"rows": 6969},
    )
    try:
        rows = restricted_pickle(repo / prediction_relative)
        label_count = sum(len(row.get("labels") or []) for row in rows.values())
        reward_count = sum(len(row.get("rewards") or []) for row in rows.values())
        prediction.observed = {"rows": len(rows), "labels": label_count, "rewards": reward_count}
        prediction.checks = {
            "expected_rows": len(rows) == 6969,
            "label_reward_lengths": label_count == reward_count,
        }
    except Exception as exc:
        prediction.add_blocker(f"Could not validate PRMBench predictions: {exc}")
    if prediction.checks and not all(prediction.checks.values()):
        prediction.add_blocker("PRMBench prediction structure failed validation")
    _hash_sources(repo, prediction)
    return [telemetry, prediction]


def audit_lettucedetect(repo: Path) -> Audit:
    relative = "dataset_cache/four_localization/ragtruth_lettuce_large_span_full/lettuce_spans_test.pkl"
    manifest_relative = "dataset_cache/four_localization/ragtruth_lettuce_large_span_full/manifest.json"
    audit = Audit(
        "ragtruth_lettucedetect_predictions",
        "RAGTruth LettuceDetect prediction package",
        "competitor span predictions",
        READY,
        [relative, manifest_relative],
        expected={"rows": 2700},
    )
    try:
        rows = restricted_pickle(repo / relative)
        ids = [str(row.get("response_id")) for row in rows.values()]
        truncated = sum(bool(row.get("truncated")) for row in rows.values())
        malformed_spans = 0
        hallucinated = 0
        for row in rows.values():
            text_len = len(str(row.get("response", "")))
            hallucinated += int(bool(row.get("gold_hallucinated")))
            for span in list(row.get("gold_spans") or []) + list(row.get("pred_spans") or []):
                if isinstance(span, Mapping):
                    start, end = span.get("start"), span.get("end")
                else:
                    start, end = span[:2]
                malformed_spans += int(not (
                    isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= text_len
                ))
        audit.observed = {
            "rows": len(rows),
            "unique_response_ids": len(set(ids)),
            "truncated": truncated,
            "malformed_spans": malformed_spans,
        }
        audit.balance = {
            "hallucinated": hallucinated,
            "clean": len(rows) - hallucinated,
            "hallucinated_rate": hallucinated / len(rows),
        }
        audit.checks = {
            "expected_rows": len(rows) == 2700,
            "unique_response_ids": len(set(ids)) == 2700,
            "span_bounds": malformed_spans == 0,
            "no_truncation": truncated == 0,
        }
    except Exception as exc:
        audit.add_blocker(f"Could not validate LettuceDetect package: {exc}")
    if audit.checks and not all(audit.checks.values()):
        audit.add_blocker("LettuceDetect package failed a structural check")
    _hash_sources(repo, audit)
    return audit


def audit_all(repo: Path) -> list[Audit]:
    audits = [
        audit_frozen_24cell(repo),
        audit_candidate_cache(
            repo,
            "semgrad_sciq",
            "SemGrad SciQ with BEM labels",
            "local_cache/semgrad_bem_regraded/raw_semgrad_sciq_T0.0_bem.pkl",
            "local_cache/semgrad_bem_regraded/raw_semgrad_sciq_T0.0_bem_manifest.json",
            1000,
        ),
        audit_candidate_cache(
            repo,
            "semgrad_truthfulqa",
            "SemGrad TruthfulQA with BEM labels",
            "local_cache/semgrad_bem_regraded/raw_semgrad_truthfulqa_T0.0_bem.pkl",
            "local_cache/semgrad_bem_regraded/raw_semgrad_truthfulqa_T0.0_bem_manifest.json",
            817,
        ),
        audit_hle(repo),
        audit_rag_conditions(
            repo,
            "ragtruth_full_evidence",
            "RAGTruth full test evidence-condition cache",
            "local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl",
            "local_cache/ragtruth_ec/test/manifest.json",
            2700,
            16200,
            protocol_limitation="RAGTruth labels have already been opened in earlier exploratory work.",
        ),
        audit_rag_conditions(
            repo,
            "gasp_ragtruth_400",
            "GASP-style balanced RAGTruth cohort",
            "dataset_cache/four_localization/gasp_ragtruth_exact_qwen15b_full/gasp_exact.pkl",
            "dataset_cache/four_localization/gasp_ragtruth_exact_qwen15b_full/manifest.json",
            400,
            2508,
            protocol_limitation=(
                "The paper did not publish its 400 response IDs or sentence splitter; this is a protocol-level reproduction."
            ),
        ),
        audit_refchecker(repo),
        audit_processbench_telemetry(repo, "qwen3_4b"),
        audit_processbench_telemetry(repo, "qwen3_8b"),
        audit_processbench_competitor(
            repo,
            "processbench_qwen25math7b_predictions",
            "ProcessBench Qwen2.5-Math-PRM predictions",
            "dataset_cache/four_localization/pb_prm_qwen25math7b_full",
            "pb_prm",
        ),
        audit_processbench_competitor(
            repo,
            "processbench_qwen3_8b_judge_control",
            "ProcessBench Qwen3-8B judge-control predictions",
            "dataset_cache/four_localization/pb_uprm_baseline_qwen3_8b_full",
            "pb_uprm_base",
        ),
        audit_processbench_competitor(
            repo,
            "processbench_qwen72b_critic",
            "ProcessBench Qwen2.5-72B critic predictions",
            "dataset_cache/four_localization/pb_critic_qwen72b_full",
            "pb_critic",
        ),
    ]
    audits.extend(audit_prmbench(repo))
    audits.append(audit_lettucedetect(repo))
    return audits


def registry_payload(repo: Path, audits: Iterable[Audit]) -> dict[str, Any]:
    audits = list(audits)
    fingerprint_material = json.dumps(
        [(audit.dataset_id, sorted(audit.file_hashes.items())) for audit in audits],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": str(repo),
        "registry_fingerprint": hashlib.sha256(fingerprint_material).hexdigest(),
        "status_counts": dict(sorted(Counter(audit.status for audit in audits).items())),
        "datasets": [audit.as_json() for audit in audits],
    }


def write_quality_csv(path: Path, audits: Iterable[Audit]) -> None:
    columns = (
        "dataset_id", "title", "kind", "status", "observed_rows", "positive_rate",
        "failed_checks", "limitations", "blockers",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for audit in audits:
            observed_rows = next(
                (audit.observed[key] for key in ("rows", "responses", "claims") if key in audit.observed),
                "",
            )
            positive_rate = next(
                (audit.balance[key] for key in (
                    "positive_rate", "hallucinated_rate", "error_rate"
                ) if key in audit.balance),
                "",
            )
            writer.writerow({
                "dataset_id": audit.dataset_id,
                "title": audit.title,
                "kind": audit.kind,
                "status": audit.status,
                "observed_rows": observed_rows,
                "positive_rate": positive_rate,
                "failed_checks": "; ".join(key for key, value in audit.checks.items() if not value),
                "limitations": " | ".join(audit.limitations),
                "blockers": " | ".join(audit.blockers),
            })


__all__ = [
    "Audit", "CanonicalLabel", "CanonicalUnit", "SCHEMA_VERSION", "READY",
    "READY_WITH_LIMITATIONS", "INCOMPLETE", "BLOCKED", "audit_all",
    "audit_candidate_cache", "audit_frozen_24cell", "audit_hle",
    "audit_processbench_telemetry",
    "audit_rag_conditions", "registry_payload", "resolve_lfs_path", "restricted_pickle",
    "read_jsonl", "sha256_file", "validate_hle_judge_rows", "write_quality_csv",
]
