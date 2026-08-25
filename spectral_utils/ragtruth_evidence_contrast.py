"""Label-free evidence-contrast features for the RAGTruth experiment.

The cluster cache stores a fixed published response under several scoring
conditions.  ``token_spilled_energies`` is the target-token negative log
probability, so no generation is performed here.  Gold labels are deliberately
kept in :class:`RagLabelSet`; fitting functions accept only :class:`FeatureTable`.

The top-k Jensen-Shannon divergence is an explicit approximation.  Every known
top-k token is a category and all probability outside the stored union is one
shared ``OTHER`` category.  This is not the full-vocabulary GASP divergence.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np


NOCTX_FEATURES = (
    "mean_full_target_logprob",
    "negative_mean_full_top50_tail_entropy",
    "mean_full_margin",
    "negative_mean_full_tail_mass",
    "mean_context_gap",
    "q90_context_gap",
    "mean_noctx_jsd_top50",
    "mean_top50_tail_entropy_increase_noctx",
)

FULL_FEATURES = NOCTX_FEATURES + (
    "max_loo_mean_drop",
    "top2_loo_mean_drop",
    "mean_positive_loo_drop",
    "max_loo_mean_jsd_top50",
    "top2_loo_mean_jsd_top50",
    "fraction_tokens_positive_best_drop",
)

CONTRACT_VERSION = "ragtruth-ec-v1-top50-tail-2026-08-10"
MIN_SENTENCE_TOKENS = 3
EPS = 1e-12


class RestrictedUnpickler(pickle.Unpickler):
    """Load cluster caches without allowing arbitrary pickle globals."""

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
        raise pickle.UnpicklingError(
            f"Blocked unsafe pickle global: {module}.{name}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache_handle(handle: BinaryIO) -> dict[str, dict[str, Any]]:
    value = RestrictedUnpickler(handle).load()
    if not isinstance(value, dict) or not value:
        raise ValueError("RAGTruth cache root must be a non-empty dictionary")
    if not all(isinstance(key, str) and isinstance(row, dict)
               for key, row in value.items()):
        raise ValueError("RAGTruth cache contains an invalid key or row")
    return value


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        return load_cache_handle(handle)


@dataclass(frozen=True)
class ConditionTrace:
    condition: str
    prompt_len: int
    token_ids: np.ndarray
    target_logprob: np.ndarray
    entropy: np.ndarray
    logsumexp: np.ndarray
    top_ids: np.ndarray
    top_logprobs: np.ndarray


@dataclass(frozen=True)
class SentenceUnit:
    index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class RagResponse:
    response_id: str
    source_id: str
    task_type: str
    source: str
    generator_model: str
    quality: str
    response_text: str
    conditions: Mapping[str, ConditionTrace]
    sentences: tuple[SentenceUnit, ...]
    token_offsets: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class RagDataset:
    """Canonical data visible to feature construction and fitting: no labels."""

    responses: tuple[RagResponse, ...]
    cache_sha256: str
    tokenizer_name: str


@dataclass(frozen=True)
class UnitLabel:
    sample_id: str
    response_id: str
    source_id: str
    task_type: str
    hallucinated: bool
    label_types: tuple[str, ...]


@dataclass(frozen=True)
class RagLabelSet:
    response: Mapping[str, UnitLabel]
    sentence: Mapping[str, UnitLabel]


@dataclass(frozen=True)
class FeatureTable:
    name: str
    contract: str
    feature_names: tuple[str, ...]
    values: np.ndarray
    sample_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    task_types: tuple[str, ...]
    sources: tuple[str, ...]
    generator_models: tuple[str, ...]
    response_lengths: np.ndarray
    unit_lengths: np.ndarray
    chunk_counts: np.ndarray
    context_lengths: np.ndarray
    supporting_chunks: np.ndarray


def _as_1d(row: Mapping[str, Any], key: str, length: int) -> np.ndarray:
    value = np.asarray(row.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a finite vector of length {length}")
    return value


def _condition_trace(row: Mapping[str, Any]) -> ConditionTrace:
    condition = str(row.get("condition", ""))
    token_ids = np.asarray(row.get("gen_token_ids"), dtype=np.int64)
    if token_ids.ndim != 1 or len(token_ids) == 0:
        raise ValueError(f"{condition}: gen_token_ids must be a non-empty vector")
    n = len(token_ids)
    nll = _as_1d(row, "token_spilled_energies", n)
    entropy = _as_1d(row, "token_entropies", n)
    logsumexp = _as_1d(row, "token_logsumexp", n)
    if np.any(nll < -1e-7) or np.any(entropy < -1e-7):
        raise ValueError(f"{condition}: NLL and entropy must be nonnegative")
    top = row.get("top_k_logprobs")
    if not isinstance(top, dict):
        raise ValueError(f"{condition}: top_k_logprobs is missing")
    top_ids = np.asarray(top.get("ids"), dtype=np.int64)
    top_lp = np.asarray(top.get("logprobs"), dtype=np.float64)
    if (top_ids.ndim != 2 or top_ids.shape != top_lp.shape
            or top_ids.shape[0] != n or top_ids.shape[1] < 2):
        raise ValueError(f"{condition}: invalid top-k arrays")
    if not np.isfinite(top_lp).all() or np.any(top_lp > 1e-6):
        raise ValueError(f"{condition}: invalid top-k log-probabilities")
    mass = np.exp(top_lp).sum(axis=1)
    if np.any(mass > 1.0005) or np.any(mass < 0.0):
        raise ValueError(f"{condition}: top-k probability mass is invalid")
    if any(len(set(row_ids.tolist())) != len(row_ids) for row_ids in top_ids):
        raise ValueError(f"{condition}: duplicate token IDs within a top-k row")
    return ConditionTrace(
        condition=condition,
        prompt_len=int(row.get("prompt_len", 0)),
        token_ids=token_ids,
        target_logprob=-nll,
        entropy=entropy,
        logsumexp=logsumexp,
        top_ids=top_ids,
        top_logprobs=top_lp,
    )


def load_official_responses_handle(handle: BinaryIO) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw_line in handle:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        row = json.loads(line)
        response_id = str(row["id"])
        if response_id in output:
            raise ValueError(f"duplicate official response id: {response_id}")
        output[response_id] = row
    return output


def load_official_responses(path: Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        return load_official_responses_handle(handle)


def _sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Match GASP's punctuation-plus-whitespace sentence split."""
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(r"(?<=[.!?])\s+", text))
    boundaries.append(len(text))
    spans: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def _sentence_units(text: str, token_ids: np.ndarray, tokenizer: Any) -> tuple[SentenceUnit, ...]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    observed = np.asarray(encoded["input_ids"], dtype=np.int64)
    if not np.array_equal(observed, token_ids):
        raise ValueError("official response does not reproduce stored Qwen token IDs")
    offsets = list(encoded["offset_mapping"])
    units: list[SentenceUnit] = []
    for char_start, char_end in _sentence_char_spans(text):
        indexes = [index for index, (start, end) in enumerate(offsets)
                   if end > char_start and start < char_end]
        if len(indexes) < MIN_SENTENCE_TOKENS:
            continue
        token_start, token_end = min(indexes), max(indexes) + 1
        units.append(SentenceUnit(
            index=len(units),
            token_start=token_start,
            token_end=token_end,
            char_start=char_start,
            char_end=char_end,
            text=text[char_start:char_end],
        ))
    if not units and len(token_ids) >= MIN_SENTENCE_TOKENS:
        units.append(SentenceUnit(0, 0, len(token_ids), 0, len(text), text))
    return tuple(units)


def _token_offsets(
    text: str,
    token_ids: np.ndarray,
    tokenizer: Any,
) -> tuple[tuple[int, int], ...]:
    """Return exact character offsets after verifying the stored token IDs."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    observed = np.asarray(encoded["input_ids"], dtype=np.int64)
    if not np.array_equal(observed, token_ids):
        raise ValueError("official response does not reproduce stored Qwen token IDs")
    offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
    if len(offsets) != len(token_ids):
        raise ValueError("token offset count disagrees with stored Qwen token IDs")
    if any(start < 0 or end < start or end > len(text) for start, end in offsets):
        raise ValueError("token offsets are outside the official response text")
    return offsets


def adapt_cache(
    cache_path: Path,
    official_response_path: Path,
    tokenizer: Any,
    *,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    raw_cache: Mapping[str, dict[str, Any]] | None = None,
    official_responses: Mapping[str, dict[str, Any]] | None = None,
    sidecar_manifest: Mapping[str, Any] | None = None,
    cache_sha256: str | None = None,
    sidecar_manifest_sha256: str | None = None,
) -> tuple[RagDataset, RagLabelSet, dict[str, Any]]:
    """Split one raw cache into label-free canonical data and isolated labels."""
    raw = dict(raw_cache) if raw_cache is not None else load_cache(cache_path)
    official = (
        dict(official_responses)
        if official_responses is not None
        else load_official_responses(official_response_path)
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for cache_key, row in raw.items():
        response_id = str(row.get("response_id", ""))
        condition = str(row.get("condition", ""))
        if cache_key != f"{response_id}::{condition}":
            raise ValueError(f"cache key disagrees with row metadata: {cache_key}")
        if condition in grouped.setdefault(response_id, {}):
            raise ValueError(f"duplicate condition {cache_key}")
        grouped[response_id][condition] = row

    responses: list[RagResponse] = []
    response_labels: dict[str, UnitLabel] = {}
    sentence_labels: dict[str, UnitLabel] = {}
    task_condition_counts: dict[str, dict[str, int]] = {}
    for response_id in sorted(grouped, key=lambda value: int(value)):
        rows = grouped[response_id]
        if "full" not in rows or "noctx" not in rows:
            raise ValueError(f"{response_id}: full/noctx pair is incomplete")
        full_row = rows["full"]
        condition_names = set(rows)
        loo_indexes = sorted(
            int(name.split("_", 1)[1]) for name in condition_names
            if name.startswith("loo_")
        )
        if loo_indexes and loo_indexes != list(range(max(loo_indexes) + 1)):
            raise ValueError(f"{response_id}: LOO indexes are not contiguous")
        official_row = official.get(response_id)
        if official_row is None:
            raise ValueError(f"{response_id}: missing from official RAGTruth data")
        if str(official_row["source_id"]) != str(full_row["source_id"]):
            raise ValueError(f"{response_id}: source_id disagrees with official data")
        traces = {name: _condition_trace(row) for name, row in rows.items()}
        reference_ids = traces["full"].token_ids
        for name, trace in traces.items():
            if not np.array_equal(reference_ids, trace.token_ids):
                raise ValueError(f"{response_id}: tokens differ in condition {name}")
        metadata_keys = ("source_id", "task_type", "source", "generator_model", "quality")
        for key in metadata_keys:
            if len({str(row.get(key, "")) for row in rows.values()}) != 1:
                raise ValueError(f"{response_id}: {key} changes across conditions")
        text = str(official_row["response"])
        token_offsets = _token_offsets(text, reference_ids, tokenizer)
        sentences = _sentence_units(text, reference_ids, tokenizer)
        response = RagResponse(
            response_id=response_id,
            source_id=str(full_row["source_id"]),
            task_type=str(full_row["task_type"]),
            source=str(full_row["source"]),
            generator_model=str(full_row["generator_model"]),
            quality=str(full_row["quality"]),
            response_text=text,
            conditions=MappingProxyType(traces),
            sentences=sentences,
            token_offsets=token_offsets,
        )
        responses.append(response)

        spans = list(full_row.get("span_token_spans") or [])
        span_rows = list(full_row.get("span_labels") or [])
        if len(spans) != len(span_rows):
            raise ValueError(f"{response_id}: span labels and token spans disagree")
        types = tuple(sorted({str(item.get("label_type", "unknown")) for item in span_rows}))
        cached_label = bool(full_row.get("response_label", False))
        if cached_label != bool(spans):
            raise ValueError(f"{response_id}: response label disagrees with mapped spans")
        response_labels[response_id] = UnitLabel(
            sample_id=response_id,
            response_id=response_id,
            source_id=response.source_id,
            task_type=response.task_type,
            hallucinated=cached_label,
            label_types=types,
        )
        for unit in sentences:
            overlaps = [idx for idx, (start, end) in enumerate(spans)
                        if int(end) > unit.token_start and int(start) < unit.token_end]
            unit_types = tuple(sorted({
                str(span_rows[idx].get("label_type", "unknown")) for idx in overlaps
            }))
            sample_id = f"{response_id}::sent_{unit.index}"
            sentence_labels[sample_id] = UnitLabel(
                sample_id=sample_id,
                response_id=response_id,
                source_id=response.source_id,
                task_type=response.task_type,
                hallucinated=bool(overlaps),
                label_types=unit_types,
            )
        task_counts = task_condition_counts.setdefault(response.task_type, {})
        for name in condition_names:
            task_counts[name] = task_counts.get(name, 0) + 1

    dataset = RagDataset(
        responses=tuple(responses),
        cache_sha256=cache_sha256 or sha256_file(cache_path),
        tokenizer_name=tokenizer_name,
    )
    labels = RagLabelSet(
        response=MappingProxyType(response_labels),
        sentence=MappingProxyType(sentence_labels),
    )
    diagnostics = {
        "n_conditions": len(raw),
        "n_responses": len(responses),
        "n_sentences": len(sentence_labels),
        "task_condition_counts": task_condition_counts,
        "response_positive_count": sum(item.hallucinated for item in response_labels.values()),
        "sentence_positive_count": sum(item.hallucinated for item in sentence_labels.values()),
    }
    manifest_path = Path(cache_path).parent / "manifest.json"
    manifest = dict(sidecar_manifest) if sidecar_manifest is not None else None
    if manifest is None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest is not None:
        stats = manifest.get("stats") or {}
        by_condition: dict[str, int] = {}
        by_task: dict[str, int] = {}
        for task, counts in task_condition_counts.items():
            by_task[task] = int(sum(counts.values()))
            for condition, count in counts.items():
                by_condition[condition] = by_condition.get(condition, 0) + int(count)
        expected = {
            "n_items": len(raw),
            "n_responses": len(responses),
            "by_condition": by_condition,
            "by_task_type": by_task,
        }
        observed = {key: stats.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                "cache records disagree with the sidecar manifest: "
                f"observed={observed!r}, expected={expected!r}"
            )
        stored_k = int(manifest.get("logprob_top_k", 0))
        observed_k = int(responses[0].conditions["full"].top_ids.shape[1])
        if stored_k != observed_k:
            raise ValueError(
                f"top-k width disagrees with sidecar manifest: {observed_k} != {stored_k}"
            )
        diagnostics.update({
            "sidecar_manifest_validated": True,
            "sidecar_manifest_sha256": (
                sidecar_manifest_sha256 or sha256_file(manifest_path)
            ),
            "sidecar_manifest_split": manifest.get("split"),
        })
    else:
        diagnostics["sidecar_manifest_validated"] = False
    return dataset, labels, diagnostics


def topk_tail_mass(logprobs: np.ndarray) -> np.ndarray:
    mass = np.exp(np.asarray(logprobs, dtype=np.float64)).sum(axis=-1)
    return np.clip(1.0 - mass, 0.0, 1.0)


def topk_plus_tail_entropy(logprobs: np.ndarray) -> np.ndarray:
    """Entropy on saved token categories plus one aggregate tail category."""
    logprobs = np.asarray(logprobs, dtype=np.float64)
    probabilities = np.exp(logprobs)
    tail = topk_tail_mass(logprobs)
    entropy = -np.sum(probabilities * logprobs, axis=-1)
    positive_tail = tail > 0
    entropy[positive_tail] -= tail[positive_tail] * np.log(tail[positive_tail])
    return entropy


def approximate_topk_jsd(
    ids_p: Sequence[int], logp_p: Sequence[float],
    ids_q: Sequence[int], logp_q: Sequence[float],
) -> float:
    """JSD over the stored union plus one shared tail category."""
    p_map = {int(token): float(math.exp(lp)) for token, lp in zip(ids_p, logp_p)}
    q_map = {int(token): float(math.exp(lp)) for token, lp in zip(ids_q, logp_q)}
    if len(p_map) != len(ids_p) or len(q_map) != len(ids_q):
        raise ValueError("top-k token IDs must be unique")
    keys = set(p_map) | set(q_map)
    p = np.asarray([p_map.get(key, 0.0) for key in keys]
                   + [max(0.0, 1.0 - sum(p_map.values()))], dtype=np.float64)
    q = np.asarray([q_map.get(key, 0.0) for key in keys]
                   + [max(0.0, 1.0 - sum(q_map.values()))], dtype=np.float64)
    p /= max(float(p.sum()), EPS)
    q /= max(float(q.sum()), EPS)
    midpoint = 0.5 * (p + q)
    positive_p = p > 0
    positive_q = q > 0
    value = 0.5 * np.sum(p[positive_p] * np.log(p[positive_p] / midpoint[positive_p]))
    value += 0.5 * np.sum(q[positive_q] * np.log(q[positive_q] / midpoint[positive_q]))
    return float(np.clip(value, 0.0, math.log(2.0)))


def trace_jsd(full: ConditionTrace, other: ConditionTrace) -> np.ndarray:
    if not np.array_equal(full.token_ids, other.token_ids):
        raise ValueError("cannot compare conditions with different answer tokens")
    return np.asarray([
        approximate_topk_jsd(full.top_ids[index], full.top_logprobs[index],
                             other.top_ids[index], other.top_logprobs[index])
        for index in range(len(full.token_ids))
    ], dtype=np.float64)


def _top2_mean(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.mean(values[-min(2, len(values)):]))


def _feature_row(
    response: RagResponse, token_slice: slice, *, include_loo: bool,
    precomputed_jsd: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    full = response.conditions["full"]
    noctx = response.conditions["noctx"]
    index = np.arange(len(full.token_ids))[token_slice]
    if len(index) == 0:
        raise ValueError("feature unit contains no tokens")
    full_lp = full.target_logprob[index]
    noctx_lp = noctx.target_logprob[index]
    gap = full_lp - noctx_lp
    full_entropy = topk_plus_tail_entropy(full.top_logprobs[index])
    noctx_entropy = topk_plus_tail_entropy(noctx.top_logprobs[index])
    full_probs = np.exp(full.top_logprobs[index])
    margin = full_probs[:, 0] - full_probs[:, 1]
    tail = topk_tail_mass(full.top_logprobs[index])
    jsd = (precomputed_jsd or {}).get("noctx")
    if jsd is None:
        jsd = trace_jsd(full, noctx)
    values = [
        float(np.mean(full_lp)),
        -float(np.mean(full_entropy)),
        float(np.mean(margin)),
        -float(np.mean(tail)),
        float(np.mean(gap)),
        float(np.quantile(gap, 0.9)),
        float(np.mean(jsd[index])),
        float(np.mean(noctx_entropy - full_entropy)),
    ]
    supporting_chunk = -1
    if include_loo:
        loo_names = sorted(
            (name for name in response.conditions if name.startswith("loo_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if not loo_names:
            raise ValueError(f"{response.response_id}: LOO features requested but unavailable")
        drops = np.vstack([
            full_lp - response.conditions[name].target_logprob[index]
            for name in loo_names
        ])
        unit_drops = drops.mean(axis=1)
        supporting_chunk = int(loo_names[int(np.argmax(unit_drops))].split("_", 1)[1])
        positive = unit_drops[unit_drops > 0]
        loo_jsd = np.vstack([
            ((precomputed_jsd or {}).get(name)
             if (precomputed_jsd or {}).get(name) is not None
             else trace_jsd(full, response.conditions[name]))[index]
            for name in loo_names
        ])
        unit_jsd = loo_jsd.mean(axis=1)
        values.extend([
            float(np.max(unit_drops)),
            _top2_mean(unit_drops),
            float(np.mean(positive)) if len(positive) else 0.0,
            float(np.max(unit_jsd)),
            _top2_mean(unit_jsd),
            float(np.mean(np.max(drops, axis=0) > 0.0)),
        ])
    row = np.asarray(values, dtype=np.float64)
    if not np.isfinite(row).all():
        raise ValueError(f"{response.response_id}: non-finite evidence feature")
    return row, supporting_chunk


def build_feature_tables(dataset: RagDataset) -> dict[str, FeatureTable]:
    """Build response and sentence tables without accepting a label object."""
    if hasattr(dataset, "labels"):
        raise TypeError("RagDataset must not expose labels")
    builders: dict[str, list[Any]] = {
        "noctx_response": [], "full_response": [],
        "noctx_sentence": [], "full_sentence": [],
    }
    for response in dataset.responses:
        full = response.conditions["full"]
        loo_names = sorted(name for name in response.conditions if name.startswith("loo_"))
        jsd = {"noctx": trace_jsd(full, response.conditions["noctx"])}
        jsd.update({name: trace_jsd(full, response.conditions[name]) for name in loo_names})
        units = [(response.response_id, slice(0, len(full.token_ids)), len(full.token_ids), -1)]
        sentence_units = [
            (f"{response.response_id}::sent_{unit.index}",
             slice(unit.token_start, unit.token_end), unit.token_end - unit.token_start, unit.index)
            for unit in response.sentences
        ]
        for table_suffix, candidates in (("response", units), ("sentence", sentence_units)):
            for sample_id, token_slice, unit_length, _ in candidates:
                noctx_row, _ = _feature_row(
                    response, token_slice, include_loo=False, precomputed_jsd=jsd
                )
                builders[f"noctx_{table_suffix}"].append((
                    noctx_row, sample_id, response, unit_length, -1
                ))
                if loo_names:
                    full_row, supporting = _feature_row(
                        response, token_slice, include_loo=True, precomputed_jsd=jsd
                    )
                    builders[f"full_{table_suffix}"].append((
                        full_row, sample_id, response, unit_length, supporting
                    ))

    output: dict[str, FeatureTable] = {}
    for name, rows in builders.items():
        if not rows:
            continue
        feature_names = FULL_FEATURES if name.startswith("full_") else NOCTX_FEATURES
        rows.sort(key=lambda item: (
            int(item[2].response_id),
            int(item[1].split("::sent_")[1]) if "::sent_" in item[1] else -1,
        ))
        values = np.vstack([item[0] for item in rows])
        output[name] = FeatureTable(
            name=name,
            contract="EC-full-v1" if name.startswith("full_") else "EC-noctx-v1",
            feature_names=feature_names,
            values=values,
            sample_ids=tuple(item[1] for item in rows),
            response_ids=tuple(item[2].response_id for item in rows),
            source_ids=tuple(item[2].source_id for item in rows),
            task_types=tuple(item[2].task_type for item in rows),
            sources=tuple(item[2].source for item in rows),
            generator_models=tuple(item[2].generator_model for item in rows),
            response_lengths=np.asarray([len(item[2].conditions["full"].token_ids) for item in rows]),
            unit_lengths=np.asarray([item[3] for item in rows]),
            chunk_counts=np.asarray([
                sum(name.startswith("loo_") for name in item[2].conditions) for item in rows
            ]),
            context_lengths=np.asarray([item[2].conditions["full"].prompt_len for item in rows]),
            supporting_chunks=np.asarray([item[4] for item in rows]),
        )
    return output


def label_vector(table: FeatureTable, labels: RagLabelSet) -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
    mapping = labels.response if table.name.endswith("response") else labels.sentence
    missing = [sample_id for sample_id in table.sample_ids if sample_id not in mapping]
    if missing:
        raise ValueError(f"labels missing for {len(missing)} samples")
    return (
        np.asarray([mapping[sample_id].hallucinated for sample_id in table.sample_ids], dtype=bool),
        tuple(mapping[sample_id].label_types for sample_id in table.sample_ids),
    )


def standardize_features(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    keep = scale >= 1e-8
    if int(keep.sum()) < 3:
        raise ValueError("fewer than three nonconstant evidence features remain")
    return (values[:, keep] - mean[keep]) / scale[keep], keep, mean, scale


__all__ = [
    "CONTRACT_VERSION", "FULL_FEATURES", "NOCTX_FEATURES",
    "ConditionTrace", "FeatureTable", "RagDataset", "RagLabelSet",
    "RagResponse", "SentenceUnit", "UnitLabel", "adapt_cache",
    "approximate_topk_jsd", "build_feature_tables", "label_vector",
    "load_cache", "load_cache_handle", "load_official_responses",
    "load_official_responses_handle", "sha256_file", "standardize_features", "topk_plus_tail_entropy",
    "topk_tail_mass",
    "trace_jsd",
]
