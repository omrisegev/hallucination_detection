"""Frozen A6-S0b shortcut and matching primitives.

This module consumes only the public, mechanically verified A6-S0a quartet
records plus prompt-only Pythia NLL values.  It has no response-generation,
telemetry, correctness-sidecar, benchmark-label, or PopQA-content API.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .a6_interventions import (
    DOMAINS,
    MUTATIONS,
    RENDERINGS,
    RESPONSE_GRAMMARS,
    TaskAST,
    changed_node_details,
    contains_answer_atom,
    task_complexity,
)


S0B_VERSION = "a6-s0b-core-v1-2026-08-15"
POPULATIONS = ("qwen-source", "llama-audit")
SCORERS_BY_POPULATION = {
    "qwen-source": ("qwen3-4b", "qwen3-8b"),
    "llama-audit": ("llama31-8b",),
}
TOKENIZER_SCORERS = ("qwen3-4b", "qwen3-8b", "llama31-8b")
RIDGES = (0.01, 0.1, 1.0, 10.0)

CONTINUOUS_COLUMNS = (
    "prompt_char_length", "prompt_word_length",
    "qwen4_prompt_tokens", "qwen8_prompt_tokens", "llama_prompt_tokens",
    "response_char_length", "response_word_length",
    "qwen4_response_span_tokens", "qwen8_response_span_tokens",
    "llama_response_span_tokens",
    "ast_node_count", "solution_depth", "changed_node_count",
    "prompt_levenshtein_distance", "prompt_response_token_jaccard",
    "answer_atom_in_prompt", "numeric_rarity_mean", "numeric_rarity_max",
    "entity_rarity_mean", "entity_rarity_max", "pythia_prompt_mean_nll",
)
CATEGORICAL_COLUMNS = (
    "domain", "mutation_family", "response_grammar", "rendering_family",
    "changed_node_type", "template_bank_id", "template_id", "donor_id",
)
MATCHING_CATEGORICAL_COLUMNS = (
    "domain", "mutation_family", "response_grammar", "rendering_family",
    "changed_node_type",
)

_LEXICAL_TOKEN = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", flags=re.UNICODE)
_NUMERIC_ATOM = re.compile(r"[-+]?\d+(?:/[1-9]\d*)?")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def first64(value: bytes) -> int:
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big", signed=False)


def _require_finite(values: Any, name: str) -> None:
    finite = np.isfinite(values.data).all() if sparse.issparse(values) \
        else np.isfinite(values).all()
    if values.ndim != 2 or not finite:
        raise ValueError(f"{name} must be a finite matrix")


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + int(left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _lexical_jaccard(left: str, right: str) -> float:
    left_tokens = set(_LEXICAL_TOKEN.findall(unicodedata.normalize("NFKC", left).lower()))
    right_tokens = set(_LEXICAL_TOKEN.findall(unicodedata.normalize("NFKC", right).lower()))
    union = left_tokens | right_tokens
    return 1.0 if not union else len(left_tokens & right_tokens) / len(union)


def _utf8_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=lambda value: value.encode("utf-8")))


def _task_from_public(value: Mapping[str, Any]) -> TaskAST:
    if not isinstance(value, Mapping):
        raise ValueError("public task must be a mapping")
    data = dict(value)
    data["records"] = tuple(tuple(row) for row in data["records"])
    return TaskAST(**data)


@dataclass(frozen=True)
class ShortcutRow:
    row_id: str
    population_id: str
    group_id: str
    outer_fold: int
    scorer_id: str
    rendering_family: str
    prompt_world: str
    response_world: str
    prompt_sha256: str
    response_sha256: str
    target: int
    continuous: tuple[float, ...]
    categorical: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.population_id not in POPULATIONS:
            raise ValueError("unknown S0b population")
        if self.scorer_id not in SCORERS_BY_POPULATION[self.population_id]:
            raise ValueError("scorer is not registered for this population")
        if self.rendering_family not in RENDERINGS:
            raise ValueError("unknown rendering")
        if self.prompt_world not in {"A", "B"} or self.response_world not in {"A", "B"}:
            raise ValueError("world must be A or B")
        if self.target != int(self.prompt_world != self.response_world):
            raise ValueError("shortcut target violates reciprocal truth")
        if len(self.continuous) != len(CONTINUOUS_COLUMNS) \
                or not np.isfinite(np.asarray(self.continuous, dtype=np.float64)).all():
            raise ValueError("shortcut continuous row is invalid")
        if len(self.categorical) != len(CATEGORICAL_COLUMNS) \
                or any(not isinstance(value, str) or not value for value in self.categorical):
            raise ValueError("shortcut categorical row is invalid")
        if not 0 <= self.outer_fold < 5:
            raise ValueError("outer fold is invalid")

    def continuous_dict(self) -> dict[str, float]:
        return dict(zip(CONTINUOUS_COLUMNS, self.continuous))

    def categorical_dict(self) -> dict[str, str]:
        return dict(zip(CATEGORICAL_COLUMNS, self.categorical))


@dataclass(frozen=True)
class FrozenVocabulary:
    values: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if tuple(name for name, _ in self.values) != CATEGORICAL_COLUMNS:
            raise ValueError("categorical vocabulary column order changed")
        for _, values in self.values:
            if values != _utf8_sorted(values):
                raise ValueError("categorical vocabulary is not canonical")

    def as_mapping(self) -> dict[str, tuple[str, ...]]:
        return dict(self.values)


@dataclass(frozen=True)
class LogisticFit:
    ridge: float
    weight: tuple[float, ...]
    intercept: float
    objective: float
    gradient_inf: float
    iterations: int


@dataclass(frozen=True)
class OofBundle:
    population_id: str
    ridge: float
    scores: tuple[float, ...]
    fold_auc: tuple[float, ...]
    fits: tuple[LogisticFit, ...]

    @property
    def macro_auc(self) -> float:
        return float(np.mean(self.fold_auc))


@dataclass(frozen=True)
class ShortcutGateBootstrap:
    population_id: str
    gate_name: str
    observed_max_macro_auc: float
    selected_ridge: float
    upper_95: float
    bootstrap_max_macro_auc: tuple[float, ...]

    @property
    def gate_pass(self) -> bool:
        return self.upper_95 <= 0.60


@dataclass(frozen=True)
class GroupMatchingRecord:
    group_id: str
    outer_fold: int
    null_stratum_id: str
    source_record_id: str
    donor_id: str
    template_bank_id: str


@dataclass(frozen=True)
class MatchingFreeze:
    group_ids: tuple[str, ...]
    vector_sha256: str
    caliper: float
    unordered_pool_size: int
    directed_eligible_edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ControlSchedule:
    family: int
    draw: int
    seed_u64: int
    assignments: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    schedule_sha256: str
    outer_held_sha256: str


def population_rarity_from_public_groups(
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Recompute the frozen per-population AST-atom rarity sidecars."""
    by_population: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        by_population[str(group["population_id"])].append(group)
    if set(by_population) != set(POPULATIONS):
        raise ValueError("rarity requires both frozen quartet populations")

    output: dict[str, dict[str, float]] = {}
    for population_id in POPULATIONS:
        numeric_counts: Counter[str] = Counter()
        entity_counts: Counter[str] = Counter()
        atoms: dict[str, tuple[list[str], list[str]]] = {}
        for group in by_population[population_id]:
            numeric: list[str] = []
            entities: list[str] = []
            for task_value in (group["task_a"], group["task_b"]):
                task = _task_from_public(task_value)
                for key, value, _ in task.records:
                    numeric.extend(_NUMERIC_ATOM.findall(f"{key}\0{value}"))
                    if task.domain == "relational":
                        entities.append(key)
            numeric_counts.update(numeric)
            entity_counts.update(entities)
            atoms[str(group["group_id"])] = (numeric, entities)
        numeric_total = sum(numeric_counts.values())
        entity_total = sum(entity_counts.values())
        for group_id, (numeric, entities) in atoms.items():
            numeric_rarity = [
                -math.log(numeric_counts[value] / numeric_total) for value in numeric
            ] if numeric_total else []
            entity_rarity = [
                -math.log(entity_counts[value] / entity_total) for value in entities
            ] if entity_total else []
            output[group_id] = {
                "numeric_rarity_mean": float(np.mean(numeric_rarity)) if numeric_rarity else 0.0,
                "numeric_rarity_max": float(max(numeric_rarity)) if numeric_rarity else 0.0,
                "entity_rarity_mean": float(np.mean(entity_rarity)) if entity_rarity else 0.0,
                "entity_rarity_max": float(max(entity_rarity)) if entity_rarity else 0.0,
            }
    return output


def _condition_index(render_index: int, prompt_world: str, response_world: str) -> int:
    return (0 if response_world == "A" else 8) \
        + (0 if prompt_world == "A" else 4) + render_index


def build_shortcut_rows(
    quartet_payloads: Sequence[Mapping[str, Any]],
    pythia_nll_by_prompt_sha256: Mapping[str, float],
) -> tuple[ShortcutRow, ...]:
    """Build the exact crossed S0b rows from public S0a checkpoint payloads."""
    groups = [payload["group"] for payload in quartet_payloads]
    rarity = population_rarity_from_public_groups(groups)
    rows: list[ShortcutRow] = []
    seen_groups: set[str] = set()

    for payload in quartet_payloads:
        if set(payload) != {
            "slot", "group", "contextual_evidence", "attempt_ledger",
            "placeholder_token_evidence_persisted",
        } or payload["placeholder_token_evidence_persisted"] is not False:
            raise ValueError("S0a public quartet payload schema changed")
        group = payload["group"]
        group_id = str(group["group_id"])
        if group_id in seen_groups:
            raise ValueError("duplicate group in S0b input")
        seen_groups.add(group_id)
        population_id = str(group["population_id"])
        scorers = SCORERS_BY_POPULATION.get(population_id)
        if scorers is None:
            raise ValueError("unknown quartet population")
        context = payload["contextual_evidence"]
        # Canonical JSON sorts mapping keys, so mapping insertion order is not a
        # scientific contract.  The registered scorer *set* is; every later
        # access uses the explicit TOKENIZER_SCORERS order.
        if set(context) != set(TOKENIZER_SCORERS):
            raise ValueError("contextual scorer roster changed")
        if any(len(context[name]) != 16 for name in TOKENIZER_SCORERS):
            raise ValueError("contextual evidence must contain 16 crossed rows")

        task_a = _task_from_public(group["task_a"])
        task_b = _task_from_public(group["task_b"])
        complexity_a = task_complexity(task_a)
        complexity_b = task_complexity(task_b)
        if complexity_a[:2] != complexity_b[:2]:
            raise ValueError("A/B task complexity differs")
        change = changed_node_details(task_a, task_b)
        node_type = "/".join(sorted(
            str(value["node_type"]) for value in change["changed_nodes"]
        ))
        if not node_type:
            raise ValueError("shortcut changed-node category is empty")
        group_rarity = rarity[group_id]

        prompts_by_world = {"A": group["prompts_a"], "B": group["prompts_b"]}
        responses_by_world = {"A": group["response_text_a"], "B": group["response_text_b"]}
        response_ast_by_world = {"A": group["response_a"], "B": group["response_b"]}
        response_sha_by_world = {"A": group["response_sha256_a"], "B": group["response_sha256_b"]}

        for render_index, rendering in enumerate(RENDERINGS):
            prompt_edit = _levenshtein(
                prompts_by_world["A"][render_index],
                prompts_by_world["B"][render_index],
            )
            for prompt_world in ("A", "B"):
                prompt = str(prompts_by_world[prompt_world][render_index])
                for response_world in ("A", "B"):
                    response = str(responses_by_world[response_world])
                    index = _condition_index(render_index, prompt_world, response_world)
                    evidence_by_scorer = {
                        name: context[name][index] for name in TOKENIZER_SCORERS
                    }
                    prompt_hashes = {
                        value["prompt_sha256"] for value in evidence_by_scorer.values()
                    }
                    if len(prompt_hashes) != 1:
                        raise ValueError("prompt hashes differ across tokenizer evidence")
                    prompt_sha = next(iter(prompt_hashes))
                    if prompt_sha not in pythia_nll_by_prompt_sha256:
                        raise ValueError("Pythia NLL is missing for a frozen prompt")
                    nll = float(pythia_nll_by_prompt_sha256[prompt_sha])
                    if not math.isfinite(nll):
                        raise ValueError("Pythia NLL is nonfinite")
                    answer = str(response_ast_by_world[response_world]["answer"])
                    continuous = (
                        float(len(prompt)), float(len(prompt.split())),
                        float(len(evidence_by_scorer["qwen3-4b"]["prefix_ids"])),
                        float(len(evidence_by_scorer["qwen3-8b"]["prefix_ids"])),
                        float(len(evidence_by_scorer["llama31-8b"]["prefix_ids"])),
                        float(len(response)), float(len(response.split())),
                        float(len(evidence_by_scorer["qwen3-4b"]["response_ids"])),
                        float(len(evidence_by_scorer["qwen3-8b"]["response_ids"])),
                        float(len(evidence_by_scorer["llama31-8b"]["response_ids"])),
                        float(complexity_a[0]), float(complexity_a[1]),
                        float(change["changed_node_count"]), float(prompt_edit),
                        float(_lexical_jaccard(prompt, response)),
                        float(contains_answer_atom(prompt, answer)),
                        group_rarity["numeric_rarity_mean"],
                        group_rarity["numeric_rarity_max"],
                        group_rarity["entity_rarity_mean"],
                        group_rarity["entity_rarity_max"], nll,
                    )
                    categorical = (
                        str(group["domain"]), str(group["mutation_family"]),
                        str(group["response_grammar"]), rendering, node_type,
                        str(group["template_bank_id"]), str(group["template_id"]),
                        str(group["donor_id"]),
                    )
                    for scorer_id in scorers:
                        row_id = ":".join((
                            group_id, scorer_id, rendering, prompt_world, response_world,
                        ))
                        rows.append(ShortcutRow(
                            row_id=row_id, population_id=population_id,
                            group_id=group_id, outer_fold=int(group["outer_fold"]),
                            scorer_id=scorer_id, rendering_family=rendering,
                            prompt_world=prompt_world, response_world=response_world,
                            prompt_sha256=prompt_sha,
                            response_sha256=str(response_sha_by_world[response_world]),
                            target=int(prompt_world != response_world),
                            continuous=continuous, categorical=categorical,
                        ))

    expected_groups = {population: 900 for population in POPULATIONS}
    for population, expected in expected_groups.items():
        group_count = len({row.group_id for row in rows if row.population_id == population})
        row_count = sum(row.population_id == population for row in rows)
        expected_rows = expected * len(SCORERS_BY_POPULATION[population]) * 16
        if group_count != expected or row_count != expected_rows:
            raise ValueError("S0b population cardinality changed")
    return tuple(rows)


def freeze_vocabulary(qwen_rows: Sequence[ShortcutRow]) -> FrozenVocabulary:
    if not qwen_rows or any(row.population_id != "qwen-source" for row in qwen_rows):
        raise ValueError("vocabulary must be fit on Qwen rows only")
    values = []
    for index, column in enumerate(CATEGORICAL_COLUMNS):
        values.append((column, _utf8_sorted(row.categorical[index] for row in qwen_rows)))
    return FrozenVocabulary(tuple(values))


def _categorical_matrix(
    rows: Sequence[ShortcutRow], vocabulary: FrozenVocabulary,
    *, columns: Sequence[str] = CATEGORICAL_COLUMNS, sparse_output: bool = False,
) -> Any:
    vocab = vocabulary.as_mapping()
    offsets: dict[str, int] = {}
    size = 0
    for column in columns:
        offsets[column] = size
        size += len(vocab[column])
    row_column = {name: index for index, name in enumerate(CATEGORICAL_COLUMNS)}
    value_index = {
        name: {value: index for index, value in enumerate(vocab[name])}
        for name in columns
    }
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, row in enumerate(rows):
        for column in columns:
            value = row.categorical[row_column[column]]
            index = value_index[column].get(value)
            if index is not None:
                row_indices.append(row_index)
                column_indices.append(offsets[column] + index)
    output = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.float64), (row_indices, column_indices)),
        shape=(len(rows), size), dtype=np.float64,
    )
    return output if sparse_output else output.toarray()


def design_matrices(
    train_rows: Sequence[ShortcutRow], held_rows: Sequence[ShortcutRow],
    vocabulary: FrozenVocabulary,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    continuous_train = np.asarray([row.continuous for row in train_rows], dtype=np.float64)
    continuous_held = np.asarray([row.continuous for row in held_rows], dtype=np.float64)
    _require_finite(continuous_train, "training continuous data")
    _require_finite(continuous_held, "held continuous data")
    # NumPy reductions are not bitwise invariant to row order.  The boundary
    # identifies rows by immutable row_id, so compute fold statistics in that
    # canonical order while returning matrices in the caller's original order.
    canonical_train = continuous_train[np.argsort(
        np.asarray([row.row_id.encode("utf-8") for row in train_rows], dtype="S"),
        kind="stable",
    )]
    mean = np.mean(canonical_train, axis=0)
    std = np.std(canonical_train, axis=0, ddof=0)
    safe_std = np.where(std > 0.0, std, 1.0)
    standardized_train = (continuous_train - mean) / safe_std
    standardized_held = (continuous_held - mean) / safe_std
    standardized_train[:, std == 0.0] = 0.0
    standardized_held[:, std == 0.0] = 0.0
    train_cat = _categorical_matrix(train_rows, vocabulary, sparse_output=True)
    held_cat = _categorical_matrix(held_rows, vocabulary, sparse_output=True)
    return (
        sparse.hstack((sparse.csr_matrix(standardized_train), train_cat), format="csr"),
        sparse.hstack((sparse.csr_matrix(standardized_held), held_cat), format="csr"),
        mean,
        std,
    )


def logistic_objective_gradient(
    parameters: np.ndarray, design: np.ndarray, labels: np.ndarray, ridge: float,
) -> tuple[float, np.ndarray]:
    _require_finite(design, "logistic design")
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (design.shape[0],) or set(np.unique(labels)) != {-1.0, 1.0}:
        raise ValueError("logistic labels must contain exactly -1 and +1")
    if ridge not in RIDGES:
        raise ValueError("unregistered shortcut ridge")
    weight, intercept = parameters[:-1], float(parameters[-1])
    counts = {value: int(np.sum(labels == value)) for value in (-1.0, 1.0)}
    omega = np.where(labels > 0, 1.0 / (2.0 * counts[1.0]), 1.0 / (2.0 * counts[-1.0]))
    margin = labels * (design @ weight + intercept)
    objective = float(np.sum(omega * np.logaddexp(0.0, -margin)) + 0.5 * ridge * weight @ weight)
    # sigmoid(-margin), evaluated without overflow.
    factor = np.empty_like(margin)
    positive = margin >= 0.0
    factor[positive] = np.exp(-margin[positive]) / (1.0 + np.exp(-margin[positive]))
    factor[~positive] = 1.0 / (1.0 + np.exp(margin[~positive]))
    derivative = -omega * labels * factor
    gradient = np.concatenate((
        design.T @ derivative + ridge * weight,
        np.asarray([np.sum(derivative)], dtype=np.float64),
    ))
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise ValueError("logistic objective became nonfinite")
    return objective, gradient


def fit_shortcut_logistic(
    design: np.ndarray, target: Sequence[int], ridge: float,
) -> LogisticFit:
    labels = 2.0 * np.asarray(target, dtype=np.float64) - 1.0
    initial = np.zeros(design.shape[1] + 1, dtype=np.float64)

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        return logistic_objective_gradient(value, design, labels, ridge)

    result = minimize(
        objective, initial, method="L-BFGS-B", jac=True,
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-12},
    )
    value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    gradient_inf = float(np.max(np.abs(gradient)))
    if not result.success or not np.isfinite(result.x).all() or gradient_inf > 1e-8:
        raise RuntimeError(
            f"shortcut logistic is unusable: {result.message}; gradient={gradient_inf}"
        )
    return LogisticFit(
        ridge=float(ridge), weight=tuple(float(item) for item in result.x[:-1]),
        intercept=float(result.x[-1]), objective=value,
        gradient_inf=gradient_inf, iterations=int(result.nit),
    )


def binary_auc(target: Sequence[int], score: Sequence[float]) -> float:
    target_array = np.asarray(target, dtype=np.int8)
    score_array = np.asarray(score, dtype=np.float64)
    if target_array.shape != score_array.shape or target_array.ndim != 1 \
            or not np.isfinite(score_array).all() or set(np.unique(target_array)) != {0, 1}:
        raise ValueError("AUC inputs are invalid")
    order = np.argsort(score_array, kind="mergesort")
    sorted_score = score_array[order]
    sorted_target = target_array[order]
    negative_before = 0
    favorable = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        block = sorted_target[start:stop]
        positives = int(np.sum(block == 1))
        negatives = int(np.sum(block == 0))
        favorable += positives * (negative_before + 0.5 * negatives)
        negative_before += negatives
        start = stop
    n_positive = int(np.sum(target_array == 1))
    n_negative = int(np.sum(target_array == 0))
    return favorable / (n_positive * n_negative)


def weighted_binary_auc(
    target: Sequence[int], score: Sequence[float], weight: Sequence[float],
) -> float:
    """Mann--Whitney AUROC with exact half credit for tied weighted rows."""
    target_array = np.asarray(target, dtype=np.int8)
    score_array = np.asarray(score, dtype=np.float64)
    weight_array = np.asarray(weight, dtype=np.float64)
    if target_array.shape != score_array.shape or target_array.shape != weight_array.shape \
            or target_array.ndim != 1 or not np.isfinite(score_array).all() \
            or not np.isfinite(weight_array).all() or np.any(weight_array < 0.0) \
            or set(np.unique(target_array)) != {0, 1}:
        raise ValueError("weighted AUC inputs are invalid")
    positive_total = float(np.sum(weight_array[target_array == 1]))
    negative_total = float(np.sum(weight_array[target_array == 0]))
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("weighted AUC has an empty resampled class")
    order = np.argsort(score_array, kind="mergesort")
    sorted_score = score_array[order]
    sorted_target = target_array[order]
    sorted_weight = weight_array[order]
    negative_before = 0.0
    favorable = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        block_target = sorted_target[start:stop]
        block_weight = sorted_weight[start:stop]
        positive = float(np.sum(block_weight[block_target == 1]))
        negative = float(np.sum(block_weight[block_target == 0]))
        favorable += positive * (negative_before + 0.5 * negative)
        negative_before += negative
        start = stop
    return favorable / (positive_total * negative_total)


def fit_oof_bundles(
    rows: Sequence[ShortcutRow], vocabulary: FrozenVocabulary,
) -> tuple[OofBundle, ...]:
    if not rows:
        raise ValueError("shortcut audit has no rows")
    populations = {row.population_id for row in rows}
    if len(populations) != 1:
        raise ValueError("Qwen and Llama shortcut audits cannot be pooled")
    population_id = next(iter(populations))
    ordered_rows = tuple(rows)
    bundles = []
    for ridge in RIDGES:
        scores = np.full(len(ordered_rows), np.nan, dtype=np.float64)
        fold_auc = []
        fits = []
        for fold in range(5):
            train_indices = [i for i, row in enumerate(ordered_rows) if row.outer_fold != fold]
            held_indices = [i for i, row in enumerate(ordered_rows) if row.outer_fold == fold]
            train_rows = [ordered_rows[i] for i in train_indices]
            held_rows = [ordered_rows[i] for i in held_indices]
            train_design, held_design, _, _ = design_matrices(train_rows, held_rows, vocabulary)
            fit = fit_shortcut_logistic(
                train_design, [row.target for row in train_rows], ridge,
            )
            held_score = held_design @ np.asarray(fit.weight) + fit.intercept
            scores[held_indices] = held_score
            fold_auc.append(binary_auc([row.target for row in held_rows], held_score))
            fits.append(fit)
        if not np.isfinite(scores).all():
            raise AssertionError("OOF scores are incomplete")
        bundles.append(OofBundle(
            population_id=population_id, ridge=ridge,
            scores=tuple(float(value) for value in scores),
            fold_auc=tuple(fold_auc), fits=tuple(fits),
        ))
    return tuple(bundles)


def gate_row_mask(rows: Sequence[ShortcutRow], gate_name: str) -> np.ndarray:
    if gate_name == "overall":
        return np.ones(len(rows), dtype=bool)
    prefix, separator, value = gate_name.partition(":")
    if not separator:
        raise ValueError("invalid shortcut gate name")
    if prefix == "domain" and value in DOMAINS:
        return np.asarray([row.categorical[0] == value for row in rows])
    if prefix == "cell":
        parts = value.split(":")
        if len(parts) == 2 and parts[0] in DOMAINS and parts[1] in MUTATIONS:
            return np.asarray([
                row.categorical[0] == parts[0] and row.categorical[1] == parts[1]
                for row in rows
            ])
    if prefix == "grammar" and value in RESPONSE_GRAMMARS:
        return np.asarray([row.categorical[2] == value for row in rows])
    if prefix == "render" and value in RENDERINGS:
        return np.asarray([row.rendering_family == value for row in rows])
    raise ValueError("invalid shortcut gate name")


def gate_names() -> tuple[str, ...]:
    return (
        "overall",
        *(f"domain:{value}" for value in DOMAINS),
        *(f"cell:{domain}:{mutation}" for domain in DOMAINS for mutation in MUTATIONS),
        *(f"grammar:{value}" for value in RESPONSE_GRAMMARS),
        *(f"render:{value}" for value in RENDERINGS),
    )


def gate_macro_auc(
    rows: Sequence[ShortcutRow], scores: Sequence[float], gate_name: str,
) -> float:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(rows),) or not np.isfinite(score_array).all():
        raise ValueError("gate score vector is invalid")
    mask = gate_row_mask(rows, gate_name)
    fold_values = []
    for fold in range(5):
        indices = np.flatnonzero(mask & np.asarray([row.outer_fold == fold for row in rows]))
        fold_values.append(binary_auc(
            [rows[index].target for index in indices], score_array[indices],
        ))
    return float(np.mean(fold_values))


def bootstrap_group_multiplicities(
    rows: Sequence[ShortcutRow], gate_name: str, *, n_draws: int = 20_000,
) -> tuple[tuple[str, ...], np.ndarray, int]:
    """Draw the exact frozen grouped-bootstrap multiplicity matrix.

    All domain x mutation x grammar strata are traversed even when ``gate_name``
    later filters most of them.  This keeps one multiplicity for each source
    group and reuses it across scorers, renderings, cells, and ridge bundles.
    """
    if n_draws <= 0:
        raise ValueError("bootstrap draw count must be positive")
    populations = {row.population_id for row in rows}
    if len(populations) != 1:
        raise ValueError("bootstrap populations cannot be pooled")
    population_id = next(iter(populations))
    if gate_name not in gate_names():
        raise ValueError("unregistered shortcut gate")
    group_cell: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        cell = row.categorical[:3]
        previous = group_cell.setdefault(row.group_id, cell)
        if previous != cell:
            raise ValueError("group changes shortcut bootstrap stratum")
    by_stratum: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for group_id, cell in group_cell.items():
        by_stratum[cell].append(group_id)
    ordered_strata = sorted(
        by_stratum,
        key=lambda cell: b"\0".join(value.encode("utf-8") for value in cell),
    )
    seed_payload = (
        b"a6-s0b-shortcut-v1\0" + population_id.encode("utf-8") + b"\0"
        + gate_name.encode("utf-8")
    )
    seed = first64(seed_payload)
    generator = np.random.Generator(np.random.PCG64(seed))
    ordered_groups: list[str] = []
    blocks: list[np.ndarray] = []
    for stratum in ordered_strata:
        group_ids = list(_utf8_sorted(by_stratum[stratum]))
        n_groups = len(group_ids)
        if n_groups == 0:
            raise AssertionError("empty bootstrap stratum")
        draws = generator.integers(
            0, n_groups, size=(n_draws, n_groups), dtype=np.int64,
        )
        counts = np.zeros((n_draws, n_groups), dtype=np.uint8)
        np.add.at(
            counts,
            (np.repeat(np.arange(n_draws), n_groups), draws.reshape(-1)),
            1,
        )
        if not np.all(np.sum(counts, axis=1) == n_groups):
            raise AssertionError("bootstrap multiplicity row changed size")
        ordered_groups.extend(group_ids)
        blocks.append(counts)
    return tuple(ordered_groups), np.concatenate(blocks, axis=1), seed


def _weighted_auc_draws(
    target: np.ndarray, score: np.ndarray, group_columns: np.ndarray,
    multiplicities: np.ndarray,
) -> np.ndarray:
    """Vectorized grouped weighted AUROC for one held fold and score vector."""
    if target.shape != score.shape or target.ndim != 1 \
            or group_columns.shape != target.shape or not np.isfinite(score).all():
        raise ValueError("bootstrap held inputs are invalid")
    order = np.argsort(score, kind="mergesort")
    target = target[order]
    score = score[order]
    group_columns = group_columns[order]
    n_draws = multiplicities.shape[0]
    favorable = np.zeros(n_draws, dtype=np.float64)
    negative_before = np.zeros(n_draws, dtype=np.float64)
    positive_total = np.zeros(n_draws, dtype=np.float64)
    negative_total = np.zeros(n_draws, dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and score[stop] == score[start]:
            stop += 1
        columns = group_columns[start:stop]
        block_target = target[start:stop]
        positive_columns = columns[block_target == 1]
        negative_columns = columns[block_target == 0]
        positive = np.sum(multiplicities[:, positive_columns], axis=1, dtype=np.float64) \
            if len(positive_columns) else np.zeros(n_draws, dtype=np.float64)
        negative = np.sum(multiplicities[:, negative_columns], axis=1, dtype=np.float64) \
            if len(negative_columns) else np.zeros(n_draws, dtype=np.float64)
        favorable += positive * (negative_before + 0.5 * negative)
        negative_before += negative
        positive_total += positive
        negative_total += negative
        start = stop
    denominator = positive_total * negative_total
    if np.any(denominator <= 0.0):
        raise RuntimeError("CLOSE_S0B_BOOTSTRAP_EMPTY_CLASS")
    return favorable / denominator


def shortcut_gate_bootstrap(
    rows: Sequence[ShortcutRow], bundles: Sequence[OofBundle], gate_name: str,
    *, n_draws: int = 20_000,
) -> ShortcutGateBootstrap:
    """Recompute the maximum-over-ridges five-fold macro in every group draw."""
    if {bundle.ridge for bundle in bundles} != set(RIDGES) or len(bundles) != len(RIDGES):
        raise ValueError("shortcut bootstrap requires all four ridge bundles")
    populations = {row.population_id for row in rows}
    if len(populations) != 1 or any(bundle.population_id not in populations for bundle in bundles):
        raise ValueError("shortcut bundle population mismatch")
    population_id = next(iter(populations))
    group_ids, multiplicities, _ = bootstrap_group_multiplicities(
        rows, gate_name, n_draws=n_draws,
    )
    group_column = {group_id: index for index, group_id in enumerate(group_ids)}
    row_group = np.asarray([group_column[row.group_id] for row in rows], dtype=np.int64)
    mask = gate_row_mask(rows, gate_name)
    fold_masks = [
        mask & np.asarray([row.outer_fold == fold for row in rows], dtype=bool)
        for fold in range(5)
    ]
    ridge_draws: list[np.ndarray] = []
    observed: list[tuple[float, float]] = []
    target_all = np.asarray([row.target for row in rows], dtype=np.int8)
    for bundle in sorted(bundles, key=lambda item: item.ridge):
        score_all = np.asarray(bundle.scores, dtype=np.float64)
        fold_draws = []
        for fold_mask in fold_masks:
            fold_draws.append(_weighted_auc_draws(
                target_all[fold_mask], score_all[fold_mask], row_group[fold_mask],
                multiplicities,
            ))
        ridge_draws.append(np.mean(np.stack(fold_draws, axis=0), axis=0))
        observed.append((gate_macro_auc(rows, score_all, gate_name), bundle.ridge))
    max_draw = np.max(np.stack(ridge_draws, axis=0), axis=0)
    # The artifact-name tie rule chooses the larger ridge.
    observed_max, selected_ridge = max(observed, key=lambda item: (item[0], item[1]))
    sorted_draw = np.sort(max_draw)
    # Contract: ascending order statistic 19,501 of 20,000 in one-based
    # indexing, i.e. numpy's method="higher" at 0.975 -> zero-based
    # ceil(0.975 * (n - 1)).  ceil(0.975 * n) - 1 lands one statistic lower
    # exactly when 0.975 * n is an integer (including the registered 20,000),
    # which is anti-conservative for a gate that must catch the upper tail.
    upper_index = math.ceil(0.975 * (n_draws - 1))
    return ShortcutGateBootstrap(
        population_id=population_id, gate_name=gate_name,
        observed_max_macro_auc=float(observed_max),
        selected_ridge=float(selected_ridge), upper_95=float(sorted_draw[upper_index]),
        bootstrap_max_macro_auc=tuple(float(value) for value in max_draw),
    )


def marginal_prevalence_audit(rows: Sequence[ShortcutRow]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, field in (("prompt", "prompt_sha256"), ("response", "response_sha256")):
        values: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            values[getattr(row, field)].append(row.target)
        prevalence = {key: float(np.mean(target)) for key, target in values.items()}
        deviation = max(abs(value - 0.5) for value in prevalence.values())
        output[f"n_{name}_hashes"] = len(prevalence)
        output[f"max_{name}_prevalence_deviation"] = deviation
        output[f"{name}_pass"] = deviation <= 1e-12
    output["pass"] = bool(output["prompt_pass"] and output["response_pass"])
    return output


def matching_vectors(
    qwen_rows: Sequence[ShortcutRow], vocabulary: FrozenVocabulary,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the exact full-Qwen, 32-row flattened matching vector per group."""
    if not qwen_rows or any(row.population_id != "qwen-source" for row in qwen_rows):
        raise ValueError("matching vectors require Qwen rows only")
    # Canonicalize before every floating reduction and subsequent flattening;
    # otherwise reversing an equivalent input changes low-order bits of mean
    # and variance and therefore the frozen matching graph.
    ordered_rows = tuple(sorted(qwen_rows, key=lambda row: row.row_id.encode("utf-8")))
    continuous = np.asarray([row.continuous for row in ordered_rows], dtype=np.float64)
    _require_finite(continuous, "matching continuous data")
    mean = np.mean(continuous, axis=0)
    std = np.std(continuous, axis=0, ddof=0)
    safe_std = np.where(std > 0.0, std, 1.0)
    continuous = (continuous - mean) / safe_std
    continuous[:, std == 0.0] = 0.0
    categorical = _categorical_matrix(
        ordered_rows, vocabulary, columns=MATCHING_CATEGORICAL_COLUMNS,
    )
    row_matrix = np.concatenate((continuous, categorical), axis=1)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered_rows):
        by_group[row.group_id].append(index)
    group_ids = _utf8_sorted(by_group)
    vectors = []
    scorer_rank = {name: index for index, name in enumerate(SCORERS_BY_POPULATION["qwen-source"])}
    rendering_rank = {name: index for index, name in enumerate(RENDERINGS)}
    world_rank = {"A": 0, "B": 1}
    for group_id in group_ids:
        indices = sorted(by_group[group_id], key=lambda index: (
            scorer_rank[ordered_rows[index].scorer_id],
            rendering_rank[ordered_rows[index].rendering_family],
            world_rank[ordered_rows[index].prompt_world],
            world_rank[ordered_rows[index].response_world],
        ))
        if len(indices) != 32:
            raise ValueError("Qwen group does not contain exactly 32 shortcut rows")
        vectors.append(row_matrix[indices].reshape(-1))
    output = np.asarray(vectors, dtype=np.float64)
    _require_finite(output, "matching vectors")
    return group_ids, output


def normalized_euclidean(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() \
            or not np.isfinite(right).all() or left.size == 0:
        raise ValueError("matching distance inputs are invalid")
    return float(np.linalg.norm(left - right) / math.sqrt(left.size))


def distance_caliper(distances: Sequence[float]) -> float:
    values = sorted(float(value) for value in distances)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("caliper distances are invalid")
    return values[math.ceil(0.75 * len(values)) - 1]


def group_matching_records(
    quartet_payloads: Sequence[Mapping[str, Any]],
    null_strata: Sequence[Mapping[str, Any]],
) -> tuple[GroupMatchingRecord, ...]:
    """Join Qwen public group identities to the frozen S0a null strata."""
    group_to_stratum: dict[str, str] = {}
    for record in null_strata:
        if record.get("population_id") != "qwen-source":
            continue
        for group_id, stratum_id in record.get("group_to_stratum", ()):
            group_id, stratum_id = str(group_id), str(stratum_id)
            previous = group_to_stratum.setdefault(group_id, stratum_id)
            if previous != stratum_id:
                raise ValueError("Qwen group has conflicting null strata")
    output = []
    seen: set[str] = set()
    for payload in quartet_payloads:
        group = payload["group"]
        if group["population_id"] != "qwen-source":
            continue
        group_id = str(group["group_id"])
        if group_id in seen or group_id not in group_to_stratum:
            raise ValueError("Qwen matching group/stratum roster is invalid")
        seen.add(group_id)
        output.append(GroupMatchingRecord(
            group_id=group_id, outer_fold=int(group["outer_fold"]),
            null_stratum_id=group_to_stratum[group_id],
            source_record_id=str(group["source_record_id"]),
            donor_id=str(group["donor_id"]),
            template_bank_id=str(group["template_bank_id"]),
        ))
    if len(output) != 900 or seen != set(group_to_stratum):
        raise ValueError("Qwen matching roster must contain exactly 900 groups")
    return tuple(sorted(output, key=lambda item: item.group_id.encode("utf-8")))


def freeze_matching_graph(
    qwen_rows: Sequence[ShortcutRow], vocabulary: FrozenVocabulary,
    records: Sequence[GroupMatchingRecord],
) -> MatchingFreeze:
    """Freeze the global Qwen matching vectors, q75 caliper, and edge pool."""
    group_ids, vectors = matching_vectors(qwen_rows, vocabulary)
    record_by_group = {record.group_id: record for record in records}
    if set(group_ids) != set(record_by_group) or len(record_by_group) != len(records):
        raise ValueError("matching row and metadata rosters differ")
    distances: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(group_ids):
        left_record = record_by_group[left]
        for right_index in range(left_index + 1, len(group_ids)):
            right = group_ids[right_index]
            right_record = record_by_group[right]
            if left_record.null_stratum_id != right_record.null_stratum_id \
                    or left_record.source_record_id == right_record.source_record_id \
                    or left_record.donor_id == right_record.donor_id \
                    or left_record.template_bank_id == right_record.template_bank_id:
                continue
            distances.append((
                left, right,
                normalized_euclidean(vectors[left_index], vectors[right_index]),
            ))
    if not distances:
        raise RuntimeError("CLOSE_S0B_CONTROL3_EMPTY_CALIPER_POOL")
    caliper = distance_caliper([value for _, _, value in distances])
    directed = []
    for left, right, distance in distances:
        if distance <= caliper:
            directed.extend(((left, right), (right, left)))
    if not directed:
        raise RuntimeError("CLOSE_S0B_CONTROL3_EMPTY_ELIGIBLE_GRAPH")
    return MatchingFreeze(
        group_ids=group_ids,
        vector_sha256=sha256_bytes(np.asarray(vectors, dtype="<f8").tobytes(order="C")),
        caliper=float(caliper), unordered_pool_size=len(distances),
        directed_eligible_edges=tuple(sorted(directed)),
    )


def canonical_partition_memberships(
    records: Sequence[GroupMatchingRecord], inner_assignments: Sequence[Sequence[Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Materialize all 60 registered Qwen outer/inner partition rosters."""
    record_by_group = {record.group_id: record for record in records}
    if len(record_by_group) != 900:
        raise ValueError("partition roster must contain 900 Qwen groups")
    inner: dict[tuple[int, str], int] = {}
    for row in inner_assignments:
        if len(row) != 3:
            raise ValueError("inner-fold row schema changed")
        outer_fold, group_id, inner_fold = int(row[0]), str(row[1]), int(row[2])
        if group_id not in record_by_group:
            continue
        if not 0 <= outer_fold < 5 or not 0 <= inner_fold < 5 \
                or record_by_group[group_id].outer_fold == outer_fold:
            raise ValueError("inner-fold assignment is invalid")
        key = (outer_fold, group_id)
        if key in inner:
            raise ValueError("duplicate inner-fold assignment")
        inner[key] = inner_fold
    expected_inner = 5 * 720
    if len(inner) != expected_inner:
        raise ValueError("Qwen inner-fold assignment roster changed")
    output: list[tuple[str, tuple[str, ...]]] = []
    for outer_fold in range(5):
        train = _utf8_sorted(
            group_id for group_id, record in record_by_group.items()
            if record.outer_fold != outer_fold
        )
        held = _utf8_sorted(
            group_id for group_id, record in record_by_group.items()
            if record.outer_fold == outer_fold
        )
        output.extend(((f"outer:{outer_fold}:train", train), (f"outer:{outer_fold}:held", held)))
        for inner_fold in range(5):
            inner_train = _utf8_sorted(
                group_id for group_id in train if inner[(outer_fold, group_id)] != inner_fold
            )
            validation = _utf8_sorted(
                group_id for group_id in train if inner[(outer_fold, group_id)] == inner_fold
            )
            output.extend((
                (f"outer:{outer_fold}:inner:{inner_fold}:train", inner_train),
                (f"outer:{outer_fold}:inner:{inner_fold}:validation", validation),
            ))
    if len(output) != 60:
        raise AssertionError("canonical partition count changed")
    return tuple(output)


def control_seed(family: int, draw: int) -> tuple[int, bytes]:
    if family not in {2, 3} or not 0 <= draw < 200:
        raise ValueError("unregistered S0b control seed")
    payload = b"a6-s0b-control-v1\0" + str(family).encode("ascii") \
        + b"\0" + str(draw).encode("ascii")
    value = first64(payload)
    return value, value.to_bytes(8, "big")


def materialize_control_schedule(
    family: int, draw: int,
    partitions: Sequence[tuple[str, tuple[str, ...]]],
    records: Sequence[GroupMatchingRecord],
    directed_eligible_edges: Sequence[tuple[str, str]],
) -> ControlSchedule:
    """Materialize one complete split-local Control-2 or Control-3 schedule."""
    seed_u64, seed_bytes = control_seed(family, draw)
    stratum_by_group = {record.group_id: record.null_stratum_id for record in records}
    if len(stratum_by_group) != len(records):
        raise ValueError("control record roster contains duplicate groups")
    eligible = set(directed_eligible_edges)
    assignments = []
    for partition_id, partition_groups in partitions:
        if tuple(partition_groups) != _utf8_sorted(partition_groups) \
                or any(group_id not in stratum_by_group for group_id in partition_groups):
            raise ValueError("control partition roster is invalid")
        if family == 2:
            by_stratum: dict[str, list[str]] = defaultdict(list)
            for group_id in partition_groups:
                by_stratum[stratum_by_group[group_id]].append(group_id)
            mapping = []
            for stratum_id in _utf8_sorted(by_stratum):
                mapping.extend(control2_derangement(
                    by_stratum[stratum_id], seed_bytes, partition_id, stratum_id,
                ))
            partition_mapping = tuple(sorted(mapping))
        elif family == 3:
            partition_set = set(partition_groups)
            partition_edges = {
                (left, right) for left, right in eligible
                if left in partition_set and right in partition_set
            }
            partition_mapping = control3_matching(
                partition_groups, partition_edges, seed_bytes, partition_id,
            )
        else:
            raise ValueError("unregistered S0b control family")
        if {left for left, _ in partition_mapping} != set(partition_groups) \
                or {right for _, right in partition_mapping} != set(partition_groups) \
                or any(left == right for left, right in partition_mapping):
            raise AssertionError("control schedule is not a fixed-point-free bijection")
        assignments.append((partition_id, partition_mapping))
    assignment_tuple = tuple(assignments)
    schedule_bytes = canonical_json_bytes([
        [partition_id, [list(pair) for pair in mapping]]
        for partition_id, mapping in assignment_tuple
    ])
    outer_held_bytes = canonical_json_bytes([
        [partition_id, [list(pair) for pair in mapping]]
        for partition_id, mapping in assignment_tuple if partition_id.endswith(":held")
    ])
    return ControlSchedule(
        family=family, draw=draw, seed_u64=seed_u64,
        assignments=assignment_tuple,
        schedule_sha256=sha256_bytes(schedule_bytes),
        outer_held_sha256=sha256_bytes(outer_held_bytes),
    )


def control2_derangement(
    group_ids: Sequence[str], seed_bytes: bytes, partition_id: str, stratum_id: str,
) -> tuple[tuple[str, str], ...]:
    ordered = tuple(sorted(group_ids, key=lambda value: value.encode("utf-8")))
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("Control-2 stratum IDs are invalid")
    for attempt in range(10_000):
        payload = (
            seed_bytes + b"\0" + partition_id.encode("utf-8") + b"\0"
            + stratum_id.encode("utf-8") + b"\0attempt:" + str(attempt).encode("ascii")
        )
        generator = np.random.Generator(np.random.PCG64(first64(payload)))
        permuted = list(ordered)
        for index in range(len(permuted) - 1, 0, -1):
            swap = int(generator.integers(0, index + 1, dtype=np.int64))
            permuted[index], permuted[swap] = permuted[swap], permuted[index]
        if all(left != right for left, right in zip(ordered, permuted)):
            return tuple(zip(ordered, permuted))
    raise RuntimeError("CLOSE_S0B_CONTROL2_DERANGEMENT_EXHAUSTED")


def hungarian_exact(costs: Sequence[Sequence[int | None]]) -> tuple[int, ...]:
    """Exact shortest-augmenting-path assignment with arbitrary-precision costs."""
    n = len(costs)
    if n == 0 or any(len(row) != n for row in costs):
        raise ValueError("Hungarian cost matrix must be nonempty and square")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for row_to_add in range(1, n + 1):
        p[0] = row_to_add
        min_value: list[int | None] = [None] * (n + 1)
        used = [False] * (n + 1)
        column = 0
        while True:
            used[column] = True
            active_row = p[column]
            delta: int | None = None
            next_column = -1
            for candidate in range(1, n + 1):
                if used[candidate]:
                    continue
                cost = costs[active_row - 1][candidate - 1]
                # Relaxation applies only along existing edges, but the delta
                # scan must see every pending column: min_value[candidate] may
                # have been set by an earlier tree row even when the active
                # row has no edge to it.  Fusing the two behind a missing-edge
                # `continue` produced false NO_PERFECT_MATCHING and silently
                # suboptimal assignments on sparse eligibility graphs.
                if cost is not None:
                    reduced = int(cost) - u[active_row] - v[candidate]
                    if min_value[candidate] is None or reduced < min_value[candidate]:
                        min_value[candidate] = reduced
                        way[candidate] = column
                if min_value[candidate] is not None and (
                    delta is None or min_value[candidate] < delta
                ):
                    delta = min_value[candidate]
                    next_column = candidate
            if delta is None or next_column < 0:
                raise RuntimeError("CLOSE_S0B_CONTROL3_NO_PERFECT_MATCHING")
            for candidate in range(n + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                elif min_value[candidate] is not None:
                    min_value[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * n
    for column in range(1, n + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    if any(value < 0 or costs[row][value] is None for row, value in enumerate(assignment)) \
            or len(set(assignment)) != n:
        raise RuntimeError("CLOSE_S0B_CONTROL3_NO_PERFECT_MATCHING")
    return tuple(assignment)


def control3_matching(
    group_ids: Sequence[str], eligible_edges: Iterable[tuple[str, str]],
    seed_bytes: bytes, partition_id: str,
) -> tuple[tuple[str, str], ...]:
    ordered = tuple(sorted(group_ids, key=lambda value: value.encode("utf-8")))
    index = {value: position for position, value in enumerate(ordered)}
    eligible = set(eligible_edges)
    n = len(ordered)
    if n < 2 or any(left not in index or right not in index for left, right in eligible):
        raise ValueError("Control-3 eligible graph is invalid")
    if any((right, left) not in eligible for left, right in eligible):
        raise ValueError("Control-3 eligibility must be symmetric")
    primary_seen: set[int] = set()
    base = (n + 1) ** n
    costs: list[list[int | None]] = [[None] * n for _ in range(n)]
    for left, right in sorted(eligible):
        if left == right:
            continue
        row, column = index[left], index[right]
        payload = (
            seed_bytes + b"\0" + partition_id.encode("utf-8") + b"\0"
            + left.encode("utf-8") + b"\0" + right.encode("utf-8")
        )
        primary = int.from_bytes(hashlib.sha256(payload).digest(), "big", signed=False)
        if primary in primary_seen:
            raise RuntimeError("CLOSE_S0B_CONTROL3_HASH_COLLISION")
        primary_seen.add(primary)
        costs[row][column] = primary * base + column * (n + 1) ** (n - 1 - row)
    # Eligibility is block diagonal by the frozen null stratum.  Solve its
    # undirected connected components independently, but retain global row/
    # column indices and the global B/secondary powers in ``costs``.  This is
    # algebraically the same global assignment while avoiding a cubic solve on
    # hundreds of vertices whose cross-stratum edges are all absent.
    adjacency = {value: set() for value in ordered}
    for left, right in eligible:
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    unseen = set(ordered)
    components: list[tuple[str, ...]] = []
    while unseen:
        start = min(unseen, key=lambda value: value.encode("utf-8"))
        stack = [start]
        component: set[str] = set()
        while stack:
            value = stack.pop()
            if value in component:
                continue
            component.add(value)
            stack.extend(adjacency[value] - component)
        unseen -= component
        components.append(tuple(sorted(component, key=lambda value: value.encode("utf-8"))))
    assignment_list = [-1] * n
    for component in components:
        component_indices = [index[value] for value in component]
        local_costs = [
            [costs[row][column] for column in component_indices]
            for row in component_indices
        ]
        local_assignment = hungarian_exact(local_costs)
        for local_row, local_column in enumerate(local_assignment):
            assignment_list[component_indices[local_row]] = component_indices[local_column]
    assignment = tuple(assignment_list)
    if any(value < 0 for value in assignment):
        raise RuntimeError("CLOSE_S0B_CONTROL3_NO_PERFECT_MATCHING")
    return tuple((left, ordered[assignment[row]]) for row, left in enumerate(ordered))


def brute_force_assignment(costs: Sequence[Sequence[int | None]]) -> tuple[int, ...]:
    """Development oracle for tiny matching graphs."""
    n = len(costs)
    best: tuple[int, tuple[int, ...]] | None = None
    for assignment in itertools.permutations(range(n)):
        selected = [costs[row][column] for row, column in enumerate(assignment)]
        if any(value is None for value in selected):
            continue
        candidate = (sum(int(value) for value in selected if value is not None), assignment)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("CLOSE_S0B_CONTROL3_NO_PERFECT_MATCHING")
    return best[1]


def pythia_prompt_mean_nll(model: Any, tokenizer: Any, prompt: str) -> float:
    """Compute the frozen CPU/float64-reduction whole-prompt mean next-token NLL."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Pythia prompt must be nonempty text")
    import torch
    from scipy.special import logsumexp

    encoded = tokenizer(
        prompt, add_special_tokens=False, padding=False, truncation=False,
        return_attention_mask=False, return_tensors="pt",
    )
    if set(encoded) != {"input_ids"}:
        raise ValueError("Pythia tokenizer output must contain only input_ids")
    input_ids = encoded["input_ids"]
    if tuple(input_ids.shape)[0] != 1 or tuple(input_ids.shape)[1] < 2:
        raise ValueError("CLOSE_S0B_PYTHIA_INPUT_TOO_SHORT")
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    logits_array = logits[0, :-1].detach().cpu().to(torch.float64).numpy()
    targets = input_ids[0, 1:].detach().cpu().numpy().astype(np.int64, copy=False)
    if logits_array.shape[0] != targets.shape[0] or not np.isfinite(logits_array).all():
        raise ValueError("Pythia logits are invalid")
    log_normalizer = logsumexp(logits_array, axis=1)
    selected = logits_array[np.arange(len(targets)), targets]
    value = float(np.mean(log_normalizer - selected))
    if not math.isfinite(value):
        raise ValueError("Pythia prompt NLL is nonfinite")
    return value


__all__ = [
    "CATEGORICAL_COLUMNS", "CONTINUOUS_COLUMNS", "ControlSchedule",
    "FrozenVocabulary", "GroupMatchingRecord", "LogisticFit", "MatchingFreeze",
    "MATCHING_CATEGORICAL_COLUMNS", "OofBundle", "POPULATIONS",
    "RIDGES", "S0B_VERSION", "SCORERS_BY_POPULATION", "ShortcutRow",
    "ShortcutGateBootstrap",
    "binary_auc", "brute_force_assignment", "build_shortcut_rows",
    "bootstrap_group_multiplicities", "canonical_json_bytes",
    "canonical_partition_memberships", "control_seed",
    "control2_derangement", "control3_matching",
    "design_matrices", "distance_caliper", "first64", "fit_oof_bundles",
    "fit_shortcut_logistic", "freeze_vocabulary", "gate_macro_auc", "gate_names",
    "gate_row_mask", "hungarian_exact", "logistic_objective_gradient",
    "freeze_matching_graph", "group_matching_records",
    "marginal_prevalence_audit", "matching_vectors",
    "materialize_control_schedule", "normalized_euclidean",
    "population_rarity_from_public_groups", "pythia_prompt_mean_nll",
    "sha256_bytes", "shortcut_gate_bootstrap", "weighted_binary_auc",
]
