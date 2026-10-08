"""CPU-only S2 LEASH stopping-lane loader and evaluator.

The S2 acquisition stores one real generation for each of three arms on the same
questions: full-chain ``cot|central``, early-stopping ``leash|central``, and the direct
answer control ``nocot|central``.  This module consumes those append-only artifacts
without modifying them.  It is deliberately strict about completeness, shard hashes,
trace keys, and arm-level question joins before it computes a number.

AQuA-RAT needs a lane-specific repair.  The acquisition called the frozen *numeric*
math parser even though AQuA gold labels are option letters A--E.  Consequently its
stored summaries report essentially zero accuracy.  ``parse_aqua_option`` is the frozen
common-protocol repair: it accepts only an explicit boxed option, an explicit
answer/option/choice phrase, or a leading option marker such as ``B)``.  Numeric-only
answers are intentionally unparsed and count wrong.  The raw shards and their stored
fields remain untouched, and every run audit reports the before/after result.

Bootstrap draws resample source questions within a dataset.  One draw is reused for all
three arms and all model copies, which preserves every pairing required by the fair
comparison contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from spectral_utils.paper_exact import evaluator as paper_evaluator
from spectral_utils.paper_exact.manifest import sha256_order

from .registry import ordered_id_sha256, validate_comparison_record


STOPPING_LANE_REVISION = "fair_s2_stopping_v1.0.0"
AQUA_PARSER_REVISION = "fair_aqua_option_parser_v1.0.0"
GSM8K_PARSER_REVISION = paper_evaluator.EVALUATOR_REVISION
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260818

S2_ARMS = ("cot", "leash", "nocot")
S2_SETTING = "central"
SUPPORTED_DATASETS = ("aqua", "gsm8k")
EXPECTED_FIDELITY = "paper-specified-partial"


class StoppingIntegrityError(ValueError):
    """A stopping artifact or identical-question join violates the frozen contract."""


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StoppingIntegrityError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StoppingIntegrityError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StoppingIntegrityError(f"{path} must contain a JSON object")
    return value


def _read_index(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise StoppingIntegrityError(f"missing required artifact: {path}") from exc
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StoppingIntegrityError(
                f"torn or invalid INDEX line {line_number} in {path}: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise StoppingIntegrityError(
                f"INDEX line {line_number} in {path} must be an object"
            )
        entries.append(entry)
    if not entries:
        raise StoppingIntegrityError(f"empty shard index: {path}")
    return entries


def _dataset_from_manifest(manifest: Mapping[str, Any]) -> str:
    source = str(manifest.get("dataset_source", ""))
    if source == "deepmind/aqua_rat":
        return "aqua"
    if source == "openai/gsm8k":
        return "gsm8k"
    raise StoppingIntegrityError(f"unsupported S2 dataset source {source!r}")


def canonical_s2_id(
    dataset_revision: str,
    dataset: str,
    question_id: str,
    model: str,
    arm: str,
) -> str:
    """Return ``<revision>::<dataset>::<question>::<model>::<arm>`` exactly."""

    parts = (dataset_revision, dataset, question_id, model, arm)
    if any(not isinstance(part, str) or not part for part in parts):
        raise StoppingIntegrityError("canonical S2 ID components must be non-empty strings")
    if dataset not in SUPPORTED_DATASETS:
        raise StoppingIntegrityError(f"unsupported S2 dataset {dataset!r}")
    if arm not in S2_ARMS:
        raise StoppingIntegrityError(f"unsupported S2 arm {arm!r}")
    return "::".join(parts)


def canonical_s2_group_id(dataset_revision: str, dataset: str, question_id: str) -> str:
    """Question-level uncertainty group shared by every model and arm copy."""

    if any(not isinstance(part, str) or not part for part in (dataset_revision, dataset, question_id)):
        raise StoppingIntegrityError("canonical S2 group components must be non-empty strings")
    return "::".join((dataset_revision, dataset, question_id))


# AQuA option parsing is intentionally conservative.  In particular, a bare number is
# not mapped to an option using the gold or the option text: doing that would introduce a
# new symbolic-equivalence evaluator and make the apparent repair depend on answer content.
_BOXED_OPTION = re.compile(
    r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?([A-E])(?:\s*\})?\s*\}", re.IGNORECASE
)
_SEMANTIC_OPTION_PATTERNS = (
    re.compile(
        r"(?:final\s+|correct\s+)?answer\s*(?:is|=|:|corresponds\s+to)?\s*"
        r"(?:\*\*|\\\(|\[|\()?\s*([A-E])(?=\s*(?:\)|\]|[).,:;*!]|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:final\s+|correct\s+)?(?:option|choice)\s*(?:is|=|:)?\s*"
        r"(?:\*\*|\\\(|\[|\()?\s*([A-E])(?=\s*(?:\)|\]|[).,:;*!]|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:corresponds?\s+to|matches)\s+(?:option|choice)\s*"
        r"(?:\*\*|\\\(|\[|\()?\s*([A-E])(?=\s*(?:\)|\]|[).,:;*!]|$))",
        re.IGNORECASE,
    ),
)
_LEADING_OPTION = re.compile(
    r"^\s*(?::\s*)?(?:\*\*)?(?:\\\()?\(?\s*([A-E])\s*"
    r"(?:\\\))?\)?\s*(?:\*\*)?\s*(?:[).,:;]|$)",
    re.IGNORECASE,
)


def parse_aqua_option(text: str | None) -> dict[str, Any]:
    """Parse one AQuA answer into an option letter without consulting its gold label.

    If several high-confidence mentions occur, the last mention wins.  This mirrors the
    usual final-answer convention while remaining independent of correctness labels.
    """

    value = text or ""
    semantic: list[tuple[int, str, str]] = []
    for match in _BOXED_OPTION.finditer(value):
        semantic.append((match.start(), match.group(1).upper(), "boxed_option"))
    for pattern in _SEMANTIC_OPTION_PATTERNS:
        for match in pattern.finditer(value):
            semantic.append((match.start(), match.group(1).upper(), "explicit_option"))
    if semantic:
        _, answer, status = max(semantic, key=lambda item: item[0])
        return {
            "answer": answer,
            "status": status,
            "parser_revision": AQUA_PARSER_REVISION,
        }

    leading = _LEADING_OPTION.match(value)
    if leading:
        return {
            "answer": leading.group(1).upper(),
            "status": "leading_option",
            "parser_revision": AQUA_PARSER_REVISION,
        }
    return {"answer": None, "status": "none", "parser_revision": AQUA_PARSER_REVISION}


def grade_aqua_option(text: str | None, gold_answer: str) -> dict[str, Any]:
    """Grade one AQuA option answer; unparsed outputs remain present and wrong."""

    gold = str(gold_answer).strip().upper()
    if gold not in set("ABCDE"):
        raise StoppingIntegrityError(f"AQuA gold must be A--E, got {gold_answer!r}")
    parsed = parse_aqua_option(text)
    return {
        "correct": parsed["answer"] == gold,
        "pred_answer": parsed["answer"],
        "gold_answer": gold,
        "parse_status": parsed["status"],
        "parser_revision": AQUA_PARSER_REVISION,
    }


def _rescore(record: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    if dataset == "aqua":
        return grade_aqua_option(record.get("answer_text"), str(record.get("gold_answer", "")))
    if dataset == "gsm8k":
        result = paper_evaluator.grade_math(
            str(record.get("answer_text", "")), str(record.get("gold_answer", ""))
        )
        return {**result, "parser_revision": GSM8K_PARSER_REVISION}
    raise StoppingIntegrityError(f"unsupported S2 dataset {dataset!r}")


def _require_nonnegative_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise StoppingIntegrityError(
            f"trace {record.get('trace_key')!r} has invalid {field}={value!r}"
        )
    return int(value)


def _validate_trace(
    record: Mapping[str, Any],
    *,
    dataset: str,
    expected_question_ids: set[str],
) -> None:
    required = (
        "trace_key",
        "question_id",
        "arm",
        "setting_label",
        "answer_text",
        "gold_answer",
        "correct",
        "pred_answer",
        "parse_status",
        "n_reasoning_tokens",
        "n_closure_tokens",
        "n_total_tokens",
        "stopped_early",
        "closure_generated",
        "wall_s",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise StoppingIntegrityError(
            f"trace {record.get('trace_key')!r} missing fields {missing}"
        )
    arm = record["arm"]
    question_id = record["question_id"]
    if arm not in S2_ARMS:
        raise StoppingIntegrityError(f"unsupported arm {arm!r}")
    if record["setting_label"] != S2_SETTING:
        raise StoppingIntegrityError(
            f"trace {record['trace_key']!r} is not the frozen central setting"
        )
    if question_id not in expected_question_ids:
        raise StoppingIntegrityError(f"unexpected question ID {question_id!r}")
    prefix = f"{dataset}:"
    if not str(question_id).startswith(prefix):
        raise StoppingIntegrityError(
            f"question ID {question_id!r} does not match dataset {dataset!r}"
        )
    expected_key = f"{arm}:{S2_SETTING}:{question_id}"
    if record["trace_key"] != expected_key:
        raise StoppingIntegrityError(
            f"trace key mismatch: expected {expected_key!r}, got {record['trace_key']!r}"
        )
    reasoning = _require_nonnegative_int(record, "n_reasoning_tokens")
    closure = _require_nonnegative_int(record, "n_closure_tokens")
    total = _require_nonnegative_int(record, "n_total_tokens")
    if total != reasoning + closure:
        raise StoppingIntegrityError(
            f"trace {record['trace_key']!r} total tokens do not equal reasoning+closure"
        )
    try:
        wall_s = float(record["wall_s"])
    except (TypeError, ValueError) as exc:
        raise StoppingIntegrityError(
            f"trace {record['trace_key']!r} has invalid wall_s"
        ) from exc
    if not math.isfinite(wall_s) or wall_s < 0:
        raise StoppingIntegrityError(
            f"trace {record['trace_key']!r} has invalid wall_s={wall_s!r}"
        )
    if not isinstance(record["stopped_early"], (bool, np.bool_)):
        raise StoppingIntegrityError(f"trace {record['trace_key']!r} has non-boolean stopped_early")
    if not isinstance(record["closure_generated"], (bool, np.bool_)):
        raise StoppingIntegrityError(
            f"trace {record['trace_key']!r} has non-boolean closure_generated"
        )


def _comparison_record(
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    dataset: str,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    rescored = _rescore(record, dataset)
    revision = str(manifest["dataset_revision"])
    model = str(manifest["model_id"])
    question_id = str(record["question_id"])
    arm = str(record["arm"])
    stopped_early = bool(record["stopped_early"])
    closure_generated = bool(record["closure_generated"])
    total_tokens = int(record["n_total_tokens"])
    comparison_record = {
        # This is the common lane interface, with stopping-specific realized-compute
        # fields carried as extras.  Stopping has no calibrated risk score or fold.
        "schema": "comparison_record_v1",
        "lane": "stopping",
        "population_id": f"s2_stopping::{revision}::{dataset}::{model}",
        "row_id": canonical_s2_id(revision, dataset, question_id, model, arm),
        "group_id": canonical_s2_group_id(revision, dataset, question_id),
        "cell_id": f"s2::{dataset}::{model}",
        "method_id": f"{arm}|{S2_SETTING}",
        "continuous_score": None,
        "discrete_prediction": rescored["pred_answer"],
        "label": rescored["gold_answer"],
        "budget": total_tokens,
        "fold": None,
        "calibration_hash": None,
        "source_artifact_hash": source_artifact_sha256,
        "dataset_revision": revision,
        "dataset": dataset,
        "question_id": question_id,
        "model": model,
        "model_revision": str(manifest["model_revision"]),
        "arm": arm,
        "setting_label": S2_SETTING,
        "gold_answer": rescored["gold_answer"],
        "prediction": rescored["pred_answer"],
        "correct": bool(rescored["correct"]),
        "parse_status": rescored["parse_status"],
        "parser_revision": rescored["parser_revision"],
        "parser_failure": rescored["pred_answer"] is None,
        "stored_correct": bool(record["correct"]),
        "stored_prediction": record.get("pred_answer"),
        "stored_parse_status": record.get("parse_status"),
        "n_reasoning_tokens": int(record["n_reasoning_tokens"]),
        "n_closure_tokens": int(record["n_closure_tokens"]),
        "n_total_tokens": total_tokens,
        "wall_s": float(record["wall_s"]),
        "stopped_early": stopped_early,
        "closure_generated": closure_generated,
        # A forced closure is realized only when the policy fired and its second-stage
        # answer was actually generated.  The CoT cap's routine answer stage is not an
        # early forced-closure event.
        "forced_closure": stopped_early and closure_generated,
        "stop_reason": record.get("stop_reason"),
        "trace_key": str(record["trace_key"]),
        "source_artifact_sha256": source_artifact_sha256,
        "acquisition_evaluator_revision": manifest.get("evaluator_revision"),
        "fidelity": manifest.get("fidelity"),
    }
    return validate_comparison_record(comparison_record)


def load_s2_run(run_dir: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Verify and load one completed model-by-dataset S2 run.

    Returns JSON-serializable per-question comparison records plus a compact audit.  The
    raw pickle dictionaries are never mutated or returned.
    """

    root = Path(run_dir)
    manifest = _read_json(root / "RUN_MANIFEST.json")
    status = _read_json(root / "STATUS.json")
    index = _read_index(root / "INDEX.jsonl")
    dataset = _dataset_from_manifest(manifest)

    if manifest.get("schema") != "paper_exact_acquisition_v1":
        raise StoppingIntegrityError(f"{root} has unsupported acquisition schema")
    if manifest.get("fidelity") != EXPECTED_FIDELITY:
        raise StoppingIntegrityError(
            f"{root} fidelity is {manifest.get('fidelity')!r}, expected {EXPECTED_FIDELITY!r}"
        )
    ids = manifest.get("dataset_example_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids):
        raise StoppingIntegrityError(f"{root} has invalid dataset_example_ids")
    if len(ids) != len(set(ids)):
        raise StoppingIntegrityError(f"{root} manifest has duplicate question IDs")
    if manifest.get("dataset_order_sha256") != sha256_order(ids):
        raise StoppingIntegrityError(f"{root} dataset order SHA-256 mismatch")
    arms = manifest.get("extra", {}).get("arms")
    if not isinstance(arms, list) or set(arms) != set(S2_ARMS) or len(arms) != len(S2_ARMS):
        raise StoppingIntegrityError(f"{root} must declare exactly the three S2 arms")
    if manifest.get("extra", {}).get("sweep") is not False:
        raise StoppingIntegrityError(f"{root} is not the frozen non-sweep full setting")
    expected_traces = len(ids) * len(S2_ARMS)
    if int(manifest.get("expected_traces", -1)) != expected_traces:
        raise StoppingIntegrityError(
            f"{root} manifest expected_traces does not equal questions x arms"
        )
    if status.get("complete") is not True:
        raise StoppingIntegrityError(f"{root} STATUS is not complete")
    if int(status.get("n_expected", -1)) != expected_traces:
        raise StoppingIntegrityError(f"{root} STATUS n_expected mismatch")
    if int(status.get("n_finished", -1)) != expected_traces:
        raise StoppingIntegrityError(f"{root} STATUS n_finished mismatch")
    if int(status.get("n_failed", -1)) != 0 or status.get("failures"):
        raise StoppingIntegrityError(f"{root} contains failed traces")
    if int(status.get("n_shards", -1)) != len(index):
        raise StoppingIntegrityError(f"{root} STATUS n_shards mismatch")

    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_shards: set[int] = set()
    seen_paths: set[str] = set()
    indexed_bytes = 0
    expected_id_set = set(ids)
    for entry_number, entry in enumerate(index):
        required = ("shard", "path", "n_traces", "bytes", "sha256", "keys", "question_ids")
        missing = [field for field in required if field not in entry]
        if missing:
            raise StoppingIntegrityError(f"INDEX entry {entry_number} missing fields {missing}")
        shard_number = entry["shard"]
        if isinstance(shard_number, bool) or not isinstance(shard_number, int):
            raise StoppingIntegrityError(f"INDEX entry {entry_number} has invalid shard number")
        if shard_number in seen_shards:
            raise StoppingIntegrityError(f"duplicate INDEX shard number {shard_number}")
        seen_shards.add(shard_number)
        relative = str(entry["path"])
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or not relative.startswith("shards/"):
            raise StoppingIntegrityError(f"unsafe shard path in INDEX: {relative!r}")
        if relative in seen_paths:
            raise StoppingIntegrityError(f"duplicate INDEX shard path {relative!r}")
        seen_paths.add(relative)
        shard_path = root.joinpath(*posix.parts)
        if not shard_path.is_file():
            raise StoppingIntegrityError(f"missing shard file {shard_path}")
        actual_bytes = shard_path.stat().st_size
        if actual_bytes != int(entry["bytes"]):
            raise StoppingIntegrityError(f"shard byte-size mismatch for {shard_path}")
        indexed_bytes += actual_bytes
        shard_sha256 = str(entry["sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", shard_sha256):
            raise StoppingIntegrityError(f"invalid shard SHA-256 for {shard_path}")
        if verify_hashes and _sha256_file(shard_path) != shard_sha256:
            raise StoppingIntegrityError(f"shard SHA-256 mismatch for {shard_path}")
        try:
            with shard_path.open("rb") as handle:
                shard_records = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - corruption should become one audit error
            raise StoppingIntegrityError(f"could not decode shard {shard_path}: {exc}") from exc
        if not isinstance(shard_records, list) or any(
            not isinstance(record, dict) for record in shard_records
        ):
            raise StoppingIntegrityError(f"shard {shard_path} must contain a list of records")
        keys = entry["keys"]
        if not isinstance(keys, list) or len(keys) != int(entry["n_traces"]):
            raise StoppingIntegrityError(f"INDEX count/key mismatch for {shard_path}")
        actual_keys = [record.get("trace_key") for record in shard_records]
        if len(shard_records) != int(entry["n_traces"]) or actual_keys != keys:
            raise StoppingIntegrityError(f"shard record/key mismatch for {shard_path}")
        duplicates = seen_keys.intersection(keys)
        if duplicates:
            raise StoppingIntegrityError(
                f"duplicate trace keys across shards: {sorted(duplicates)[:5]}"
            )
        seen_keys.update(keys)
        question_ids = sorted({str(record.get("question_id")) for record in shard_records})
        if question_ids != entry["question_ids"]:
            raise StoppingIntegrityError(f"INDEX question-ID coverage mismatch for {shard_path}")
        for raw_record in shard_records:
            _validate_trace(raw_record, dataset=dataset, expected_question_ids=expected_id_set)
            records.append(
                _comparison_record(
                    raw_record,
                    manifest=manifest,
                    dataset=dataset,
                    source_artifact_sha256=shard_sha256,
                )
            )

    if indexed_bytes != int(status.get("bytes_total", -1)):
        raise StoppingIntegrityError(f"{root} STATUS bytes_total mismatch")
    expected_keys = {
        f"{arm}:{S2_SETTING}:{question_id}" for arm in S2_ARMS for question_id in ids
    }
    missing_keys = expected_keys - seen_keys
    extra_keys = seen_keys - expected_keys
    if missing_keys or extra_keys:
        raise StoppingIntegrityError(
            f"{root} trace-key coverage mismatch: missing={sorted(missing_keys)[:5]}, "
            f"extra={sorted(extra_keys)[:5]}"
        )

    per_arm_seen: dict[str, set[str]] = defaultdict(set)
    for record in records:
        arm = record["arm"]
        question_id = record["question_id"]
        if question_id in per_arm_seen[arm]:
            raise StoppingIntegrityError(f"duplicate {arm} row for question {question_id!r}")
        per_arm_seen[arm].add(question_id)
    for arm in S2_ARMS:
        if per_arm_seen[arm] != expected_id_set:
            raise StoppingIntegrityError(f"{root} arm {arm!r} does not cover identical IDs")

    # Put rows into the canonical manifest-question / arm order, independent of shard cuts.
    record_lookup = {(record["question_id"], record["arm"]): record for record in records}
    records = [record_lookup[(question_id, arm)] for question_id in ids for arm in S2_ARMS]
    stored_correct = np.asarray([record["stored_correct"] for record in records], dtype=float)
    rescored_correct = np.asarray([record["correct"] for record in records], dtype=float)
    incompatible_stored = sum(
        record["stored_prediction"] is not None
        and str(record["stored_prediction"]).strip().upper() not in set("ABCDE")
        for record in records
    ) if dataset == "aqua" else 0
    parser_status_counts = Counter(record["parse_status"] for record in records)
    defect_detected = dataset == "aqua" and incompatible_stored > 0
    audit = {
        "schema": "s2_stopping_run_audit_v1",
        "run_id": manifest.get("run_id"),
        "dataset": dataset,
        "dataset_revision": manifest.get("dataset_revision"),
        "model": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
        "fidelity": manifest.get("fidelity"),
        "acquisition_evaluator_revision": manifest.get("evaluator_revision"),
        "rescoring_parser_revision": (
            AQUA_PARSER_REVISION if dataset == "aqua" else GSM8K_PARSER_REVISION
        ),
        "status_complete": True,
        "n_questions": len(ids),
        "n_traces": len(records),
        "n_shards": len(index),
        "bytes_total": indexed_bytes,
        "all_shard_hashes_verified": bool(verify_hashes),
        "identical_arm_question_ids": True,
        "dataset_order_sha256": manifest.get("dataset_order_sha256"),
        # This manifest-derived roster is the claim/paired-bootstrap population.
        # It is deliberately recorded independently of the observed arm rows.
        "registered_question_ids": list(ids),
        "registered_group_ids": [
            canonical_s2_group_id(
                str(manifest["dataset_revision"]), dataset, str(question_id)
            )
            for question_id in ids
        ],
        "paired_group_order_sha256": ordered_id_sha256(
            [
                canonical_s2_group_id(
                    str(manifest["dataset_revision"]), dataset, str(question_id)
                )
                for question_id in ids
            ]
        ),
        "stored_pass_at_1": float(np.mean(stored_correct)),
        "rescored_pass_at_1": float(np.mean(rescored_correct)),
        "n_correctness_changes": int(np.sum(stored_correct != rescored_correct)),
        "n_stored_predictions_incompatible_with_aqua_gold": incompatible_stored,
        "rescored_parser_status_counts": dict(sorted(parser_status_counts.items())),
        "n_rescored_parser_failures": int(sum(record["parser_failure"] for record in records)),
        "upstream_aqua_parser_defect": {
            "detected": defect_detected,
            "code": "numeric_math_parser_applied_to_aqua_option_letter_gold" if defect_detected else None,
            "raw_artifacts_mutated": False,
            "stored_summary_usable_for_accuracy": dataset != "aqua",
            "resolution": (
                "rescore raw answer_text with frozen option-letter parser; unparsed counts wrong"
                if dataset == "aqua"
                else "reuse frozen paper-exact math parser semantics"
            ),
        },
    }
    return {"manifest": manifest, "status": status, "records": records, "audit": audit}


def _validate_suite_pairing(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_dataset_model: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        key = (record["dataset_revision"], record["dataset"], record["model"])
        arm = record["arm"]
        question_id = record["question_id"]
        if question_id in by_dataset_model[key][arm]:
            raise StoppingIntegrityError(f"duplicate suite row for {key}, {arm}, {question_id}")
        by_dataset_model[key][arm].add(question_id)
    dataset_sets: dict[tuple[str, str], set[str]] = {}
    dataset_models: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, arm_sets in by_dataset_model.items():
        if set(arm_sets) != set(S2_ARMS):
            raise StoppingIntegrityError(f"cell {key} does not contain exactly three arms")
        first = arm_sets[S2_ARMS[0]]
        if any(arm_sets[arm] != first for arm in S2_ARMS[1:]):
            raise StoppingIntegrityError(f"cell {key} does not join identical IDs across arms")
        dataset_key = key[:2]
        if dataset_key in dataset_sets and dataset_sets[dataset_key] != first:
            raise StoppingIntegrityError(
                f"dataset {dataset_key} does not join identical IDs across model copies"
            )
        dataset_sets[dataset_key] = set(first)
        dataset_models[dataset_key].append(key[2])
    return {
        "identical_question_ids_across_arms": True,
        "identical_question_ids_across_model_copies": True,
        "datasets": {
            "::".join(key): {
                "n_questions": len(dataset_sets[key]),
                "models": sorted(dataset_models[key]),
            }
            for key in sorted(dataset_sets)
        },
    }


def load_s2_suite(
    cache_root: str | Path,
    *,
    verify_hashes: bool = True,
    require_six_complete_cells: bool = True,
) -> dict[str, Any]:
    """Load the complete local S2 suite and enforce model-copy pairing."""

    root = Path(cache_root)
    run_dirs = sorted(
        path for path in root.glob("s2_leash_*") if path.is_dir() and (path / "RUN_MANIFEST.json").is_file()
    )
    if not run_dirs:
        raise StoppingIntegrityError(f"no S2 LEASH runs under {root}")
    if require_six_complete_cells and len(run_dirs) != 6:
        raise StoppingIntegrityError(
            f"expected exactly six complete S2 cells, found {len(run_dirs)} under {root}"
        )
    loaded = [load_s2_run(path, verify_hashes=verify_hashes) for path in run_dirs]
    records = [record for run in loaded for record in run["records"]]
    pairing = _validate_suite_pairing(records)
    cell_keys = {(record["dataset"], record["model"]) for record in records}
    if require_six_complete_cells and (
        len(cell_keys) != 6
        or {dataset for dataset, _ in cell_keys} != set(SUPPORTED_DATASETS)
        or len({model for _, model in cell_keys}) != 3
    ):
        raise StoppingIntegrityError(
            "the complete S2 suite must be the 3-model x 2-dataset matrix"
        )
    return {
        "schema": "s2_stopping_suite_v1",
        "records": records,
        "run_audits": [run["audit"] for run in loaded],
        "suite_audit": {
            "n_runs": len(loaded),
            "n_cells": len(cell_keys),
            "n_question_arm_model_records": len(records),
            "all_status_complete": True,
            "all_indexes_and_shards_verified": bool(verify_hashes),
            **pairing,
        },
    }


def _cell_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_cell_arm: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell_arm[(row["dataset_revision"], row["dataset"], row["model"], row["arm"])].append(row)

    metrics: list[dict[str, Any]] = []
    for key in sorted(by_cell_arm):
        revision, dataset, model, arm = key
        arm_rows = by_cell_arm[key]
        n = len(arm_rows)
        reasoning = sum(int(row["n_reasoning_tokens"]) for row in arm_rows)
        closure = sum(int(row["n_closure_tokens"]) for row in arm_rows)
        total = reasoning + closure
        stopped = sum(bool(row["stopped_early"]) for row in arm_rows)
        forced = sum(bool(row["forced_closure"]) for row in arm_rows)
        parser_failures = sum(bool(row["parser_failure"]) for row in arm_rows)
        missing_closure = sum(
            bool(row["stopped_early"]) and not bool(row["closure_generated"])
            for row in arm_rows
        )
        metrics.append(
            {
                "schema": "s2_stopping_cell_metric_v1",
                "dataset_revision": revision,
                "dataset": dataset,
                "model": model,
                "cell_id": arm_rows[0]["cell_id"],
                "arm": arm,
                "method_id": f"{arm}|{S2_SETTING}",
                "n_questions": n,
                "pass_at_1": float(np.mean([bool(row["correct"]) for row in arm_rows])),
                "reasoning_tokens": reasoning,
                "closure_tokens": closure,
                "total_tokens": total,
                "mean_tokens_per_question": total / n,
                "total_wall_s": float(sum(float(row["wall_s"]) for row in arm_rows)),
                "mean_wall_s": float(np.mean([float(row["wall_s"]) for row in arm_rows])),
                "n_stopped_early": stopped,
                "early_stop_rate": stopped / n,
                "n_forced_closure": forced,
                "forced_closure_rate": forced / n,
                "n_parser_failures": parser_failures,
                "parser_failure_rate": parser_failures / n,
                "n_stopped_without_closure": missing_closure,
                "realized_savings_valid": missing_closure == 0,
                "fidelity": arm_rows[0]["fidelity"],
                "parser_revision": arm_rows[0]["parser_revision"],
            }
        )

    metric_lookup = {
        (row["dataset_revision"], row["dataset"], row["model"], row["arm"]): row
        for row in metrics
    }
    frontiers: list[dict[str, Any]] = []
    for metric in metrics:
        cell_prefix = (metric["dataset_revision"], metric["dataset"], metric["model"])
        cot = metric_lookup[(*cell_prefix, "cot")]
        competitors = [metric_lookup[(*cell_prefix, arm)] for arm in S2_ARMS]
        dominated_by = sorted(
            other["arm"]
            for other in competitors
            if other["arm"] != metric["arm"]
            and other["pass_at_1"] >= metric["pass_at_1"]
            and other["mean_tokens_per_question"] <= metric["mean_tokens_per_question"]
            and (
                other["pass_at_1"] > metric["pass_at_1"]
                or other["mean_tokens_per_question"] < metric["mean_tokens_per_question"]
            )
        )
        frontiers.append(
            {
                "schema": "s2_accuracy_compute_frontier_point_v1",
                "dataset_revision": metric["dataset_revision"],
                "dataset": metric["dataset"],
                "model": metric["model"],
                "cell_id": metric["cell_id"],
                "arm": metric["arm"],
                "pass_at_1": metric["pass_at_1"],
                "mean_tokens_per_question": metric["mean_tokens_per_question"],
                "mean_wall_s": metric["mean_wall_s"],
                "accuracy_delta_vs_cot": metric["pass_at_1"] - cot["pass_at_1"],
                "token_reduction_vs_cot": (
                    1.0 - metric["mean_tokens_per_question"] / cot["mean_tokens_per_question"]
                    if cot["mean_tokens_per_question"]
                    else None
                ),
                "pareto_efficient_within_cell": not dominated_by,
                "dominated_by": dominated_by,
            }
        )
    return metrics, frontiers


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_question_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Paired question bootstrap, sharing draws across arms and repeated model copies."""

    if int(n_boot) <= 0:
        raise ValueError("n_boot must be positive")
    _validate_suite_pairing(rows)
    by_dataset: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[(row["dataset_revision"], row["dataset"])].append(row)

    output: list[dict[str, Any]] = []
    for dataset_key in sorted(by_dataset):
        revision, dataset = dataset_key
        dataset_rows = by_dataset[dataset_key]
        question_ids = sorted({row["question_id"] for row in dataset_rows})
        models = sorted({row["model"] for row in dataset_rows})
        row_lookup = {
            (row["model"], row["arm"], row["question_id"]): row for row in dataset_rows
        }
        expected = len(question_ids) * len(models) * len(S2_ARMS)
        if len(row_lookup) != expected:
            raise StoppingIntegrityError(
                f"dataset {dataset_key} is not rectangular over question x model x arm"
            )
        # A dataset-specific deterministic stream means adding a different dataset does
        # not change this one's interval, while all model copies still share the draws.
        dataset_offset = int(
            hashlib.sha256(f"{revision}::{dataset}".encode("utf-8")).hexdigest()[:8], 16
        )
        rng = np.random.default_rng((int(seed) + dataset_offset) % (2**32))
        draws = rng.integers(0, len(question_ids), size=(int(n_boot), len(question_ids)))

        vectors: dict[tuple[str, str, str], np.ndarray] = {}
        fields = {
            "pass_at_1": lambda row: float(bool(row["correct"])),
            "mean_tokens_per_question": lambda row: float(row["n_total_tokens"]),
            "mean_wall_s": lambda row: float(row["wall_s"]),
            "early_stop_rate": lambda row: float(bool(row["stopped_early"])),
            "forced_closure_rate": lambda row: float(bool(row["forced_closure"])),
            "parser_failure_rate": lambda row: float(bool(row["parser_failure"])),
        }
        for model in models:
            for arm in S2_ARMS:
                for metric_name, getter in fields.items():
                    vectors[(model, arm, metric_name)] = np.asarray(
                        [getter(row_lookup[(model, arm, question_id)]) for question_id in question_ids],
                        dtype=float,
                    )

        for model in models:
            for arm in S2_ARMS:
                for metric_name in fields:
                    vector = vectors[(model, arm, metric_name)]
                    replicates = vector[draws].mean(axis=1)
                    point = float(np.mean(vector))
                    if metric_name == "mean_tokens_per_question":
                        # Total and mean are both required.  With a fixed-size question
                        # bootstrap, each replicate total is exactly N times its mean.
                        total_replicates = replicates * len(question_ids)
                        lo, hi = _percentile_interval(total_replicates)
                        output.append(
                            {
                                "schema": "s2_paired_question_interval_v1",
                                "dataset_revision": revision,
                                "dataset": dataset,
                                "model": model,
                                "arm": arm,
                                "metric": "total_tokens",
                                "point": float(np.sum(vector)),
                                "lo": lo,
                                "hi": hi,
                                "n_groups": len(question_ids),
                                "n_boot": int(n_boot),
                                "n_valid": int(n_boot),
                                "seed": int(seed),
                                "grouping": "source_question_with_all_arms_and_model_copies",
                            }
                        )
                    lo, hi = _percentile_interval(replicates)
                    output.append(
                        {
                            "schema": "s2_paired_question_interval_v1",
                            "dataset_revision": revision,
                            "dataset": dataset,
                            "model": model,
                            "arm": arm,
                            "metric": metric_name,
                            "point": point,
                            "lo": lo,
                            "hi": hi,
                            "n_groups": len(question_ids),
                            "n_boot": int(n_boot),
                            "n_valid": int(n_boot),
                            "seed": int(seed),
                            "grouping": "source_question_with_all_arms_and_model_copies",
                        }
                    )

                if arm == "cot":
                    continue
                def paired_delta(field: str, *, scale: float = 1.0):
                    arm_vector = vectors[(model, arm, field)]
                    cot_vector = vectors[(model, "cot", field)]
                    replicate_delta = (
                        arm_vector[draws].mean(axis=1)
                        - cot_vector[draws].mean(axis=1)
                    ) * scale
                    return (
                        float((np.mean(arm_vector) - np.mean(cot_vector)) * scale),
                        replicate_delta,
                    )

                token = vectors[(model, arm, "mean_tokens_per_question")]
                cot_token = vectors[(model, "cot", "mean_tokens_per_question")]
                contrasts = {
                    # Preserve the original public names while adding the explicit
                    # pass@1 name expected by the per-cell direct-comparison table.
                    "accuracy_delta_vs_cot": paired_delta("pass_at_1"),
                    "pass_at_1_delta_vs_cot": paired_delta("pass_at_1"),
                    "mean_token_delta_vs_cot": paired_delta("mean_tokens_per_question"),
                    "total_token_delta_vs_cot": paired_delta(
                        "mean_tokens_per_question", scale=float(len(question_ids))
                    ),
                    "mean_wall_s_delta_vs_cot": paired_delta("mean_wall_s"),
                    "total_wall_s_delta_vs_cot": paired_delta(
                        "mean_wall_s", scale=float(len(question_ids))
                    ),
                    "early_stop_rate_delta_vs_cot": paired_delta("early_stop_rate"),
                    "forced_closure_rate_delta_vs_cot": paired_delta(
                        "forced_closure_rate"
                    ),
                    "parser_failure_rate_delta_vs_cot": paired_delta(
                        "parser_failure_rate"
                    ),
                    "token_reduction_vs_cot": (
                        float(1.0 - np.mean(token) / np.mean(cot_token)),
                        1.0 - token[draws].mean(axis=1) / cot_token[draws].mean(axis=1),
                    ),
                }
                for metric_name, (point, replicates) in contrasts.items():
                    lo, hi = _percentile_interval(replicates)
                    output.append(
                        {
                            "schema": "s2_paired_question_interval_v1",
                            "dataset_revision": revision,
                            "dataset": dataset,
                            "model": model,
                            "arm": arm,
                            "reference_arm": "cot",
                            "contrast_direction": "arm_minus_cot",
                            "metric": metric_name,
                            "point": point,
                            "lo": lo,
                            "hi": hi,
                            "n_groups": len(question_ids),
                            "n_boot": int(n_boot),
                            "n_valid": int(n_boot),
                            "seed": int(seed),
                            "grouping": "source_question_with_all_arms_and_model_copies",
                        }
                    )
    return output


def score_stopping_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute stopping metrics, frontiers, and paired intervals for verified rows."""

    materialized = list(rows)
    if not materialized:
        raise StoppingIntegrityError("cannot score an empty stopping population")
    pairing = _validate_suite_pairing(materialized)
    metrics, frontier = _cell_metrics(materialized)
    if any(not metric["realized_savings_valid"] for metric in metrics):
        raise StoppingIntegrityError(
            "a stopped trace lacks a real closure; realized token savings are ineligible"
        )
    return {
        "schema": "s2_stopping_score_v1",
        "lane_revision": STOPPING_LANE_REVISION,
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(n_boot),
        "pairing_audit": pairing,
        "cell_metrics": metrics,
        "accuracy_compute_frontier": frontier,
        "paired_intervals": paired_question_bootstrap(
            materialized, n_boot=int(n_boot), seed=int(seed)
        ),
    }


def build_s2_stopping_lane(
    cache_root: str | Path,
    *,
    verify_hashes: bool = True,
    require_six_complete_cells: bool = True,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Verify, rescore, and evaluate the complete CPU-only S2 stopping lane."""

    suite = load_s2_suite(
        cache_root,
        verify_hashes=verify_hashes,
        require_six_complete_cells=require_six_complete_cells,
    )
    scores = score_stopping_records(suite["records"], n_boot=n_boot, seed=seed)
    return {
        **scores,
        "schema": "fair_s2_stopping_lane_package_v1",
        "lane_revision": STOPPING_LANE_REVISION,
        "aqua_parser_revision": AQUA_PARSER_REVISION,
        "gsm8k_parser_revision": GSM8K_PARSER_REVISION,
        "records": suite["records"],
        "run_audits": suite["run_audits"],
        "suite_audit": suite["suite_audit"],
    }


__all__ = [
    "AQUA_PARSER_REVISION",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "GSM8K_PARSER_REVISION",
    "S2_ARMS",
    "STOPPING_LANE_REVISION",
    "StoppingIntegrityError",
    "build_s2_stopping_lane",
    "canonical_s2_group_id",
    "canonical_s2_id",
    "grade_aqua_option",
    "load_s2_run",
    "load_s2_suite",
    "paired_question_bootstrap",
    "parse_aqua_option",
    "score_stopping_records",
]
