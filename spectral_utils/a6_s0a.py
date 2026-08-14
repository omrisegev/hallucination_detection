"""Fail-closed A6-S0a construction, tokenizer, and manifest contracts.

This module implements only the pre-telemetry S0a boundary frozen in
``AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md``.  It may construct
mechanical reciprocal tasks, prompt-only natural manifests, contextual
tokenizer evidence, folds, and null strata.  It has no model-generation,
telemetry, correctness-sidecar, benchmark-row, or PopQA-content API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from fnmatch import fnmatch
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from .a6_interventions import (
    DOMAINS,
    MUTATIONS,
    RENDERINGS,
    RESPONSE_GRAMMARS,
    ConstructionAttempt,
    ReciprocalGroup,
    ResponseAST,
    TaskAST,
    canonical_answer,
    construct_reciprocal_attempt,
    construct_task_pair_from_seed,
    contains_answer_atom,
    evaluate_generator,
    render_response,
    render_task,
    semantic_task_sha256,
    task_complexity,
    task_sha256,
)


CONTRACT_VERSION = "a6-s0a-execution-v1-2026-08-14"
QUARTET_POPULATIONS = ("qwen-source", "llama-audit")
NATURAL_COHORTS = (
    "qwen3-4b-natural", "qwen3-8b-natural", "llama31-8b-natural",
)
SCORER_IDS = ("qwen3-4b", "qwen3-8b", "llama31-8b")
MAX_ATTEMPTS_PER_SLOT = 10_000
POPQA_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
POPQA_ROWS = 14_267
POPQA_TEMPLATE = (
    "Answer the following question with one short answer.\n"
    "Question: {question}\nAnswer:"
)
POPQA_TEMPLATE_SHA256 = "97cc05f94fecfc2e30dd3751c2e800039d196b149d538376777aa837c5123963"


@dataclass(frozen=True)
class ModelIdentity:
    role: str
    scorer_id: str
    repository: str
    revision: str
    template_kind: str


MODEL_IDENTITIES = (
    ModelIdentity(
        "qwen_source_1", "qwen3-4b", "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c", "qwen",
    ),
    ModelIdentity(
        "qwen_source_2", "qwen3-8b", "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218", "qwen",
    ),
    ModelIdentity(
        "held_llama", "llama31-8b", "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659", "llama",
    ),
    ModelIdentity(
        "s0b_prompt_nll", "pythia-410m", "EleutherAI/pythia-410m-deduped",
        "c4fc8d586d62df497f1f9b69d66d3ca419992d3e", "none",
    ),
)
TOKENIZER_IDENTITIES = MODEL_IDENTITIES[:3]

_TOKENIZER_LITERAL_FILES = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "tokenizer.model",
}
_TOKENIZER_GLOBS = ("sentencepiece*.model", "chat_template*.jinja")
_FORBIDDEN_NATURAL_KEYS = (
    "answer", "solution", "correct", "label", "target", "response", "feature",
    "sidecar", "alias", "possible_answer", "object",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first64(payload: bytes) -> int:
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sha256_lsb(payload: bytes) -> int:
    """Least-significant bit of the complete SHA-256 integer."""
    return hashlib.sha256(payload).digest()[-1] & 1


def _ids_sha256(ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes([int(value) for value in ids]))


def _selected_tokenizer_path(relative: str) -> bool:
    name = Path(relative).name
    return name in _TOKENIZER_LITERAL_FILES or any(
        fnmatch(name, pattern) for pattern in _TOKENIZER_GLOBS
    )


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SnapshotManifest:
    role: str
    scorer_id: str
    repository: str
    revision: str
    content_sha256: str
    repository_tree: tuple[str, ...]
    files: tuple[SnapshotFile, ...]


def _source_selected_files(source: Path) -> tuple[tuple[str, Path], ...]:
    if not source.is_dir():
        raise ValueError(f"snapshot source is not a directory: {source}")
    selected = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source).as_posix()
        if _selected_tokenizer_path(relative):
            if path.is_symlink():
                target = Path(os.readlink(path))
                target = target if target.is_absolute() else path.parent / target
                if target.is_symlink():
                    raise ValueError(f"tokenizer source has a multi-hop link: {relative}")
                resolved = target.resolve(strict=True)
            else:
                resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"tokenizer source is not a regular file: {relative}")
            selected.append((relative, resolved))
    if not selected:
        raise ValueError("tokenizer snapshot contains no allowlisted files")
    if len({relative for relative, _ in selected}) != len(selected):
        raise ValueError("tokenizer snapshot has duplicate relative paths")
    return tuple(selected)


def prepare_content_addressed_tokenizer_snapshot(
    source: str | Path,
    destination_root: str | Path,
    identity: ModelIdentity,
    *,
    resolved_revision: str,
) -> tuple[Path, SnapshotManifest]:
    """Copy one pre-resolved tokenizer snapshot into immutable regular files.

    This function never downloads or imports Transformers.  The caller must
    supply the revision resolved by the repository client; a mismatch fails
    before any copied boundary input is accepted.
    """
    if identity not in TOKENIZER_IDENTITIES:
        raise ValueError("S0a accepts only the three frozen tokenizer identities")
    if resolved_revision != identity.revision:
        raise ValueError("resolved tokenizer revision differs from frozen revision")
    selected = _source_selected_files(Path(source))
    repository_tree = tuple(
        path.relative_to(Path(source)).as_posix()
        for path in sorted(Path(source).rglob("*")) if path.is_file() or path.is_symlink()
    )
    file_rows = tuple(
        SnapshotFile(relative, path.stat().st_size, sha256_file(path))
        for relative, path in selected
    )
    content_sha = sha256_bytes(canonical_json_bytes([
        asdict(value) for value in file_rows
    ]))
    destination = Path(destination_root) / f"{identity.scorer_id}-{content_sha}"
    if destination.exists():
        raise FileExistsError(f"refusing to reuse snapshot destination: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for relative, source_path in selected:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_path.open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
        manifest = SnapshotManifest(
            identity.role, identity.scorer_id, identity.repository,
            identity.revision, content_sha, repository_tree, file_rows,
        )
        verify_content_addressed_snapshot(destination, manifest)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination, manifest


def verify_content_addressed_snapshot(
    snapshot: str | Path, manifest: SnapshotManifest,
) -> None:
    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise ValueError("content-addressed tokenizer snapshot is missing")
    actual_paths = []
    actual_directories = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            actual_directories.append(path.relative_to(snapshot).as_posix())
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("content-addressed tokenizer input must contain regular files")
        actual_paths.append(path.relative_to(snapshot).as_posix())
    expected_paths = [value.path for value in manifest.files]
    allowed_directories = sorted({
        parent.as_posix()
        for relative in expected_paths
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    })
    if actual_directories != allowed_directories:
        raise ValueError("tokenizer snapshot directory set changed after freeze")
    if actual_paths != expected_paths:
        raise ValueError("tokenizer snapshot path set changed after freeze")
    rows = []
    for expected in manifest.files:
        path = snapshot / expected.path
        actual = SnapshotFile(expected.path, path.stat().st_size, sha256_file(path))
        if actual != expected:
            raise ValueError(f"tokenizer snapshot file changed: {expected.path}")
        rows.append(asdict(actual))
    if sha256_bytes(canonical_json_bytes(rows)) != manifest.content_sha256:
        raise ValueError("tokenizer snapshot content hash mismatch")


def load_verified_fast_tokenizer(
    snapshot: str | Path, manifest: SnapshotManifest,
):
    """Verify bytes before importing Transformers and loading offline only."""
    verify_content_addressed_snapshot(snapshot, manifest)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer  # imported only after byte verification

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("A6 S0a requires a fast tokenizer with offsets")
    return tokenizer


@dataclass(frozen=True)
class ContextualInputEvidence:
    scorer_id: str
    prompt_sha256: str
    response_sha256: str
    prefix_text_sha256: str
    full_text_sha256: str
    prefix_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    full_ids_sha256: str
    prefix_ids_sha256: str
    response_ids_sha256: str
    suffix_ids_sha256: str
    prefix_character_length: int
    response_character_length: int


@dataclass(frozen=True)
class NaturalTokenizerEvidence:
    scorer_id: str
    repository: str
    revision: str
    prefix_text_sha256: str
    input_ids: tuple[int, ...]
    input_ids_sha256: str
    input_length: int
    attention_mask_sha256: str
    generation_seed: int
    generation_parameters: tuple[tuple[str, Any], ...]


def _chat_kwargs(identity: ModelIdentity) -> dict[str, Any]:
    if identity.template_kind == "qwen":
        return {"enable_thinking": False}
    if identity.template_kind == "llama":
        return {}
    raise ValueError("identity does not define a chat-template contract")


def _extract_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError("batched tokenizer output is not permitted")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("tokenizer did not return a token-ID sequence")
    ids = tuple(int(item) for item in value)
    if not ids or any(item < 0 for item in ids):
        raise ValueError("tokenizer returned empty or negative token IDs")
    return ids


def _extract_offsets(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Mapping) or "offset_mapping" not in value:
        raise TypeError("fast tokenizer did not return offset_mapping")
    offsets = value["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], Sequence) and len(offsets[0]) == 1 \
            and isinstance(offsets[0][0], Sequence):
        offsets = offsets[0]
    return tuple((int(start), int(end)) for start, end in offsets)


def build_contextual_input_evidence(
    tokenizer: Any,
    identity: ModelIdentity,
    prompt: str,
    response: str,
) -> ContextualInputEvidence:
    """Audit one fixed response in its exact teacher-forced chat context."""
    if identity not in TOKENIZER_IDENTITIES:
        raise ValueError("unknown frozen tokenizer identity")
    if not prompt or not response:
        raise ValueError("prompt and response must be nonempty")
    kwargs = _chat_kwargs(identity)
    user = [{"role": "user", "content": prompt}]
    conversation = user + [{"role": "assistant", "content": response}]
    prefix_text = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, **kwargs,
    )
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **kwargs,
    )
    if not isinstance(prefix_text, str) or not isinstance(full_text, str):
        raise TypeError("chat template must return strings when tokenize=False")
    start = len(prefix_text)
    stop = start + len(response)
    if not full_text.startswith(prefix_text) or full_text[start:stop] != response:
        raise ValueError("assistant response does not start at the frozen prefix boundary")
    if full_text.rfind(response) != start:
        raise ValueError("assistant response is not its final exact occurrence")
    encoded = tokenizer(
        full_text, add_special_tokens=False, padding=False, truncation=False,
        return_offsets_mapping=True,
    )
    full_ids = _extract_ids(encoded)
    offsets = _extract_offsets(encoded)
    if len(offsets) != len(full_ids):
        raise ValueError("offset and token-ID lengths differ")
    previous_start = previous_end = 0
    covered = [False] * len(response)
    prefix_ids, response_ids, suffix_ids = [], [], []
    for token_id, (left, right) in zip(full_ids, offsets):
        if left < 0 or right < left or right > len(full_text):
            raise ValueError("token offset is outside the contextual text")
        if left < previous_start or right < previous_end:
            raise ValueError("token offsets are not monotone")
        previous_start, previous_end = left, right
        intersects = right > start and left < stop
        if intersects:
            if left == right == 0:
                raise ValueError("special zero offset assigned to response span")
            response_ids.append(token_id)
            for index in range(max(left, start), min(right, stop)):
                covered[index - start] = True
        elif right <= start:
            prefix_ids.append(token_id)
        elif left >= stop:
            suffix_ids.append(token_id)
        else:
            raise ValueError("token offset has an unclassified boundary relation")
    if not response_ids or not all(covered):
        raise ValueError("contextual token offsets do not cover the response")
    template_ids = _extract_ids(tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, **kwargs,
    ))
    if template_ids != full_ids:
        raise ValueError("chat-template token IDs differ from full-text tokenization")
    return ContextualInputEvidence(
        identity.scorer_id, sha256_text(prompt), sha256_text(response),
        sha256_text(prefix_text), sha256_text(full_text), tuple(prefix_ids),
        tuple(response_ids), tuple(suffix_ids), _ids_sha256(full_ids),
        _ids_sha256(prefix_ids), _ids_sha256(response_ids), _ids_sha256(suffix_ids),
        len(prefix_text), len(response),
    )


def build_natural_tokenizer_evidence(
    tokenizer: Any,
    identity: ModelIdentity,
    prompt: str,
    generation_seed: int,
) -> NaturalTokenizerEvidence:
    if identity not in TOKENIZER_IDENTITIES:
        raise ValueError("unknown frozen tokenizer identity")
    kwargs = _chat_kwargs(identity)
    messages = [{"role": "user", "content": prompt}]
    prefix_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs,
    )
    template_ids = _extract_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **kwargs,
    ))
    encoded_ids = _extract_ids(tokenizer(
        prefix_text, add_special_tokens=False, padding=False, truncation=False,
    ))
    if template_ids != encoded_ids:
        raise ValueError("natural prompt chat IDs differ from prefix tokenization")
    attention_hash = _ids_sha256((1,) * len(template_ids))
    parameters = (
        ("do_sample", False), ("num_beams", 1), ("max_new_tokens", 150),
        ("batch_size", 1), ("attention_mask_dtype", "int64"),
        ("custom_stopping_strings", False), ("custom_logits_processors", False),
    )
    return NaturalTokenizerEvidence(
        identity.scorer_id, identity.repository, identity.revision,
        sha256_text(prefix_text), template_ids, _ids_sha256(template_ids),
        len(template_ids), attention_hash, int(generation_seed), parameters,
    )


@dataclass(frozen=True)
class QuartetSlot:
    slot_id: str
    population_id: str
    seed: int
    outer_fold: int
    within_fold: int
    source_record_id: str
    donor_id: str
    template_bank_id: str
    template_id: str
    domain: str
    mutation_family: str
    response_grammar: str
    subdomain: str


@dataclass(frozen=True)
class NaturalSlot:
    slot_id: str
    cohort_id: str
    cohort_index: int
    scorer_id: str
    outer_fold: int
    within_cell: int
    domain: str
    mutation_family: str
    subdomain: str
    item_id: str
    source_record_id: str
    donor_id: str
    template_bank_id: str
    template_id: str


@dataclass(frozen=True)
class S0aAttemptRecord:
    attempt_index: int
    attempt_seed: int
    status: str
    reason: str
    semantic_task_sha256_a: str | None
    semantic_task_sha256_b: str | None
    prompt_content_sha256_a: tuple[str, ...] = ()
    prompt_content_sha256_b: tuple[str, ...] = ()
    contextual_evidence_sha256: str | None = None
    task_sha256_a: str | None = None
    task_sha256_b: str | None = None


@dataclass(frozen=True)
class QuartetRecord:
    slot: QuartetSlot
    group: ReciprocalGroup
    contextual_evidence: tuple[tuple[str, tuple[ContextualInputEvidence, ...]], ...]
    attempt_ledger: tuple[S0aAttemptRecord, ...]


@dataclass(frozen=True)
class NaturalPromptRow:
    item_id: str
    cohort_id: str
    scorer_id: str
    outer_fold: int
    domain: str
    mutation_family: str
    subdomain: str
    attempt_index: int
    prompt_text: str
    prompt_sha256: str
    semantic_task_sha256: str
    source_record_id: str
    donor_id: str
    template_bank_id: str
    template_id: str
    complete_prompt_id: str
    tokenizer_evidence: NaturalTokenizerEvidence


@dataclass(frozen=True)
class NaturalPromptRecord:
    slot: NaturalSlot
    row: NaturalPromptRow
    attempt_ledger: tuple[S0aAttemptRecord, ...]


def frozen_quartet_slots() -> tuple[QuartetSlot, ...]:
    slots = []
    for population in QUARTET_POPULATIONS:
        for domain in DOMAINS:
            for mutation in MUTATIONS:
                for grammar in RESPONSE_GRAMMARS:
                    for fold in range(5):
                        for within_fold in range(10):
                            slot_id = (
                                f"{population}:{domain}:{mutation}:{grammar}:"
                                f"fold{fold}:{within_fold:02d}"
                            )
                            seed = first64(b"a6-s0-slot-v2\0" + slot_id.encode("utf-8"))
                            bank = (
                                f"{population}:fold{fold}:template-bank:{domain}:"
                                f"{mutation}:{grammar}:{within_fold % 5}"
                            )
                            slots.append(QuartetSlot(
                                slot_id, population, seed, fold, within_fold,
                                f"{population}:fold{fold}:source:{slot_id}",
                                f"{population}:fold{fold}:donor:{slot_id}",
                                bank, f"{bank}:instance:{slot_id}", domain, mutation,
                                grammar,
                                "derived-answer" if domain == "arithmetic"
                                and within_fold < 2 else "general",
                            ))
    return tuple(slots)


def _rotated_natural_cells(cohort_index: int, fold: int) -> tuple[tuple[str, str], ...]:
    cells = tuple((domain, mutation) for domain in DOMAINS for mutation in MUTATIONS)
    shift = (cohort_index + 2 * fold) % len(cells)
    return cells[shift:] + cells[:shift]


def frozen_natural_slots() -> tuple[NaturalSlot, ...]:
    scorer_by_cohort = dict(zip(NATURAL_COHORTS, SCORER_IDS))
    slots = []
    for cohort_index, cohort in enumerate(NATURAL_COHORTS):
        for fold in range(5):
            for cell_index, (domain, mutation) in enumerate(
                _rotated_natural_cells(cohort_index, fold)
            ):
                count = 45 if cell_index < 4 else 44
                for within_cell in range(count):
                    slot_id = (
                        f"{cohort}:fold{fold}:{domain}:{mutation}:{within_cell:03d}"
                    )
                    bank = f"{cohort}:fold{fold}:template-bank:{within_cell % 20}"
                    slots.append(NaturalSlot(
                        slot_id, cohort, cohort_index, scorer_by_cohort[cohort], fold,
                        within_cell, domain, mutation,
                        "derived-answer" if domain == "arithmetic"
                        and within_cell < 10 else "general",
                        sha256_text("a6-natural-item-v1\0" + slot_id),
                        f"{cohort}:fold{fold}:source:{slot_id}",
                        f"{cohort}:fold{fold}:donor:{slot_id}", bank,
                        f"{bank}:instance:{slot_id}",
                    ))
    return tuple(slots)


def inner_fold_manifest(
    records: Sequence[QuartetRecord],
) -> tuple[tuple[int, str, int], ...]:
    output = []
    for population in QUARTET_POPULATIONS:
        population_records = [value for value in records if value.slot.population_id == population]
        for outer_fold in range(5):
            training = [value for value in population_records if value.slot.outer_fold != outer_fold]
            for domain in DOMAINS:
                for mutation in MUTATIONS:
                    for grammar in RESPONSE_GRAMMARS:
                        cell = [
                            value for value in training
                            if value.slot.domain == domain
                            and value.slot.mutation_family == mutation
                            and value.slot.response_grammar == grammar
                        ]
                        if len(cell) != 40:
                            raise ValueError("outer-training semantic cell must contain 40 groups")
                        ordered = sorted(cell, key=lambda value: hashlib.sha256(
                            b"a6-s0-inner-v1\0" + str(outer_fold).encode("ascii")
                            + b"\0" + value.group.group_id.encode("utf-8")
                        ).digest())
                        output.extend(
                            (outer_fold, value.group.group_id, index % 5)
                            for index, value in enumerate(ordered)
                        )
    manifest = tuple(sorted(output))
    if len(manifest) != 2 * 5 * 18 * 40:
        raise AssertionError("inner-fold manifest has the wrong cardinality")
    return manifest


def _derived_pair_pass(task_a, task_b, prompts_a: Sequence[str], prompts_b: Sequence[str]) -> bool:
    answers = (
        canonical_answer(evaluate_generator(task_a)),
        canonical_answer(evaluate_generator(task_b)),
    )
    return not any(
        contains_answer_atom(prompt, answer)
        for answer in answers for prompt in (*prompts_a, *prompts_b)
    )


def _placeholder_ids(_text: str) -> tuple[int, ...]:
    """Disable Step-265 standalone token filtering; never boundary evidence."""
    return (1,) * 50


_PLACEHOLDER_TOKENIZERS = {"llama": _placeholder_ids, "qwen": _placeholder_ids}


def contextual_quartet_evidence(
    group: ReciprocalGroup,
    tokenizers: Mapping[str, Any],
) -> tuple[tuple[str, tuple[ContextualInputEvidence, ...]], ...]:
    if tuple(tokenizers) != SCORER_IDS:
        raise ValueError(f"tokenizer order must be exactly {SCORER_IDS}")
    identity_by_scorer = {value.scorer_id: value for value in TOKENIZER_IDENTITIES}
    result = []
    for scorer_id in SCORER_IDS:
        evidence = []
        identity = identity_by_scorer[scorer_id]
        for prompts, response in (
            (group.prompts_a, group.response_text_a),
            (group.prompts_b, group.response_text_a),
            (group.prompts_a, group.response_text_b),
            (group.prompts_b, group.response_text_b),
        ):
            evidence.extend(
                build_contextual_input_evidence(
                    tokenizers[scorer_id], identity, prompt, response,
                )
                for prompt in prompts
            )
        # The order above is A/RA, B/RA, A/RB, B/RB, each over four renders.
        for response_block in (evidence[:8], evidence[8:]):
            if len({value.response_ids for value in response_block}) != 1:
                raise ValueError("contextual response IDs drift across prompt/render cells")
            if len({value.suffix_ids for value in response_block}) != 1:
                raise ValueError("contextual suffix IDs drift across prompt/render cells")
        for render_index in range(4):
            crossed_lengths = {
                len(evidence[offset + render_index].prefix_ids)
                for offset in (0, 4, 8, 12)
            }
            if len(crossed_lengths) != 1:
                raise ValueError(
                    "crossed contextual-prefix token counts differ by prompt/response world"
                )
        if group.response_grammar == "certificate" and any(
            not 40 <= len(value.response_ids) <= 80 for value in evidence
        ):
            raise ValueError("contextual certificate response is outside [40,80]")
        result.append((scorer_id, tuple(evidence)))
    return tuple(result)


class _CollisionRegistry:
    def __init__(self) -> None:
        self.semantic_tasks: set[str] = set()
        self.prompt_contents: set[str] = set()
        self.source_ids: set[str] = set()
        self.donor_ids: set[str] = set()
        self.template_ids: set[str] = set()
        self.complete_prompt_ids: set[str] = set()
        self.template_bank_owners: dict[str, tuple[str, int]] = {}

    def rejection_reason(
        self, *, semantic_tasks: Iterable[str], prompt_contents: Iterable[str],
        source_id: str, donor_id: str, template_bank_id: str, template_id: str,
        complete_prompt_ids: Iterable[str], owner: tuple[str, int],
    ) -> str | None:
        values = tuple(semantic_tasks)
        prompts = tuple(prompt_contents)
        complete = tuple(complete_prompt_ids)
        checks = (
            (any(value in self.semantic_tasks for value in values), "global_semantic_ast_collision"),
            (any(value in self.prompt_contents for value in prompts), "global_prompt_content_collision"),
            (source_id in self.source_ids, "global_source_collision"),
            (donor_id in self.donor_ids, "global_donor_collision"),
            (template_id in self.template_ids, "global_template_instance_collision"),
            (any(value in self.complete_prompt_ids for value in complete), "global_complete_prompt_collision"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        previous = self.template_bank_owners.get(template_bank_id)
        if previous is not None and previous != owner:
            return "global_template_bank_owner_collision"
        return None

    def admit(
        self, *, semantic_tasks: Iterable[str], prompt_contents: Iterable[str],
        source_id: str, donor_id: str, template_bank_id: str, template_id: str,
        complete_prompt_ids: Iterable[str], owner: tuple[str, int],
    ) -> None:
        self.semantic_tasks.update(semantic_tasks)
        self.prompt_contents.update(prompt_contents)
        self.source_ids.add(source_id)
        self.donor_ids.add(donor_id)
        self.template_ids.add(template_id)
        self.complete_prompt_ids.update(complete_prompt_ids)
        self.template_bank_owners.setdefault(template_bank_id, owner)


def _admit_quartet_record(
    registry: _CollisionRegistry, record: QuartetRecord,
) -> None:
    group, slot = record.group, record.slot
    reason = registry.rejection_reason(
        semantic_tasks=(semantic_task_sha256(group.task_a), semantic_task_sha256(group.task_b)),
        prompt_contents=tuple(sha256_text(value) for value in (*group.prompts_a, *group.prompts_b)),
        source_id=group.source_record_id, donor_id=group.donor_id,
        template_bank_id=slot.template_bank_id, template_id=group.template_id,
        complete_prompt_ids=(*group.complete_prompt_ids_a, *group.complete_prompt_ids_b),
        owner=(slot.population_id, slot.outer_fold),
    )
    if reason is not None:
        raise ValueError(f"restored quartet checkpoint collides: {reason}")
    registry.admit(
        semantic_tasks=(semantic_task_sha256(group.task_a), semantic_task_sha256(group.task_b)),
        prompt_contents=tuple(sha256_text(value) for value in (*group.prompts_a, *group.prompts_b)),
        source_id=group.source_record_id, donor_id=group.donor_id,
        template_bank_id=slot.template_bank_id, template_id=group.template_id,
        complete_prompt_ids=(*group.complete_prompt_ids_a, *group.complete_prompt_ids_b),
        owner=(slot.population_id, slot.outer_fold),
    )


def _admit_natural_record(
    registry: _CollisionRegistry, record: NaturalPromptRecord,
) -> None:
    row, terminal = record.row, record.attempt_ledger[-1]
    if (
        terminal.status != "ACCEPTED"
        or terminal.semantic_task_sha256_a is None
        or terminal.semantic_task_sha256_b is None
    ):
        raise ValueError("restored natural checkpoint lacks its accepted identities")
    semantic = (terminal.semantic_task_sha256_a, terminal.semantic_task_sha256_b)
    prompts = (*terminal.prompt_content_sha256_a, *terminal.prompt_content_sha256_b)
    reason = registry.rejection_reason(
        semantic_tasks=semantic, prompt_contents=prompts,
        source_id=row.source_record_id, donor_id=row.donor_id,
        template_bank_id=row.template_bank_id, template_id=row.template_id,
        complete_prompt_ids=(row.complete_prompt_id,),
        owner=(row.cohort_id, row.outer_fold),
    )
    if reason is not None:
        raise ValueError(f"restored natural checkpoint collides: {reason}")
    registry.admit(
        semantic_tasks=semantic, prompt_contents=prompts,
        source_id=row.source_record_id, donor_id=row.donor_id,
        template_bank_id=row.template_bank_id, template_id=row.template_id,
        complete_prompt_ids=(row.complete_prompt_id,),
        owner=(row.cohort_id, row.outer_fold),
    )


def _attempt_record_from_group(
    attempt_index: int,
    construction: ConstructionAttempt,
    group: ReciprocalGroup | None,
    *,
    status: str | None = None,
    reason: str | None = None,
    contextual_evidence: Any = None,
) -> S0aAttemptRecord:
    if group is None:
        return S0aAttemptRecord(
            attempt_index, construction.attempt_seed,
            status or construction.status, reason or construction.reason,
            None, None,
            task_sha256_a=construction.ast_sha256_a,
            task_sha256_b=construction.ast_sha256_b,
        )
    prompts_a = tuple(sha256_text(value) for value in group.prompts_a)
    prompts_b = tuple(sha256_text(value) for value in group.prompts_b)
    context_sha = None if contextual_evidence is None else sha256_bytes(
        canonical_json_bytes([
            [scorer_id, [asdict(item) for item in evidence]]
            for scorer_id, evidence in contextual_evidence
        ])
    )
    return S0aAttemptRecord(
        attempt_index, construction.attempt_seed, status or construction.status,
        reason or construction.reason, semantic_task_sha256(group.task_a),
        semantic_task_sha256(group.task_b), prompts_a, prompts_b, context_sha,
        group.ast_sha256_a, group.ast_sha256_b,
    )


def _validate_and_admit_quartet_record(
    registry: _CollisionRegistry,
    record: QuartetRecord,
    tokenizers: Mapping[str, Any],
    expected_slot: QuartetSlot | None = None,
) -> None:
    """Replay every scheduled attempt before restoring one checkpoint."""
    slot = record.slot if expected_slot is None else expected_slot
    replayed_ledger = []
    for attempt_index, observed in enumerate(record.attempt_ledger):
        construction, candidate = construct_reciprocal_attempt(
            seed=slot.seed, attempt_index=attempt_index,
            population_id=slot.population_id, outer_fold=slot.outer_fold,
            source_record_id=slot.source_record_id, donor_id=slot.donor_id,
            template_id=slot.template_id, domain=slot.domain,
            mutation_family=slot.mutation_family,
            response_grammar=slot.response_grammar,
            tokenizers=_PLACEHOLDER_TOKENIZERS,
        )
        if candidate is None:
            expected = _attempt_record_from_group(attempt_index, construction, None)
        elif slot.subdomain == "derived-answer" and not _derived_pair_pass(
            candidate.task_a, candidate.task_b, candidate.prompts_a, candidate.prompts_b,
        ):
            expected = _attempt_record_from_group(
                attempt_index, construction, candidate,
                status="REJECTED", reason="derived_answer_copy",
            )
        else:
            try:
                contextual = contextual_quartet_evidence(candidate, tokenizers)
            except (TypeError, ValueError) as error:
                expected = _attempt_record_from_group(
                    attempt_index, construction, candidate,
                    status="REJECTED", reason=f"contextual_tokenizer:{error}",
                )
            else:
                semantic = (
                    semantic_task_sha256(candidate.task_a),
                    semantic_task_sha256(candidate.task_b),
                )
                prompt_hashes = tuple(
                    sha256_text(value)
                    for value in (*candidate.prompts_a, *candidate.prompts_b)
                )
                complete_ids = (
                    *candidate.complete_prompt_ids_a, *candidate.complete_prompt_ids_b,
                )
                reason = registry.rejection_reason(
                    semantic_tasks=semantic, prompt_contents=prompt_hashes,
                    source_id=slot.source_record_id, donor_id=slot.donor_id,
                    template_bank_id=slot.template_bank_id,
                    template_id=slot.template_id, complete_prompt_ids=complete_ids,
                    owner=(slot.population_id, slot.outer_fold),
                )
                if reason is not None:
                    expected = _attempt_record_from_group(
                        attempt_index, construction, candidate,
                        status="REJECTED_GLOBAL", reason=reason,
                        contextual_evidence=contextual,
                    )
                else:
                    expected = _attempt_record_from_group(
                        attempt_index, construction, candidate,
                        contextual_evidence=contextual,
                    )
                    if attempt_index != len(record.attempt_ledger) - 1 \
                            or record.contextual_evidence != contextual:
                        raise ValueError("quartet checkpoint accepted before its terminal row")
        if observed != expected:
            raise ValueError("quartet checkpoint attempt ledger does not replay")
        replayed_ledger.append(expected)
        if expected.status == "ACCEPTED":
            expected_record = QuartetRecord(
                slot, candidate, contextual, tuple(replayed_ledger),
            )
            if canonical_json_bytes(public_quartet_record(record)) != \
                    canonical_json_bytes(public_quartet_record(expected_record)):
                raise ValueError("quartet checkpoint bytes do not match frozen replay")
            _admit_quartet_record(registry, record)
            return
    raise ValueError("quartet checkpoint ledger has no replayed acceptance")


def build_s0a_quartets(
    tokenizers: Mapping[str, Any],
    *,
    registry: _CollisionRegistry | None = None,
    slots: Sequence[QuartetSlot] | None = None,
    existing_records: Sequence[QuartetRecord] = (),
    on_record: Callable[[int, QuartetRecord], None] | None = None,
) -> tuple[QuartetRecord, ...]:
    """Build the exact first 1,800 S0a groups with contextual token gates."""
    if tuple(tokenizers) != SCORER_IDS:
        raise ValueError(f"tokenizer order must be exactly {SCORER_IDS}")
    registry = _CollisionRegistry() if registry is None else registry
    slots = frozen_quartet_slots() if slots is None else tuple(slots)
    if slots != frozen_quartet_slots()[:len(slots)]:
        raise ValueError("quartet slots must be a canonical frozen prefix")
    records = list(existing_records)
    if len(records) > len(slots) or any(
        record.slot != slots[index] for index, record in enumerate(records)
    ):
        raise ValueError("existing quartet records are not the frozen prefix")
    for index, record in enumerate(records):
        _validate_and_admit_quartet_record(
            registry, record, tokenizers, slots[index],
        )
    for slot in slots[len(records):]:
        ledger = []
        accepted_group = accepted_evidence = None
        for attempt_index in range(MAX_ATTEMPTS_PER_SLOT):
            construction, candidate = construct_reciprocal_attempt(
                seed=slot.seed, attempt_index=attempt_index,
                population_id=slot.population_id, outer_fold=slot.outer_fold,
                source_record_id=slot.source_record_id, donor_id=slot.donor_id,
                template_id=slot.template_id, domain=slot.domain,
                mutation_family=slot.mutation_family,
                response_grammar=slot.response_grammar,
                tokenizers=_PLACEHOLDER_TOKENIZERS,
            )
            if candidate is None:
                ledger.append(_attempt_record_from_group(
                    attempt_index, construction, None,
                ))
                continue
            if slot.subdomain == "derived-answer" and not _derived_pair_pass(
                candidate.task_a, candidate.task_b,
                candidate.prompts_a, candidate.prompts_b,
            ):
                ledger.append(_attempt_record_from_group(
                    attempt_index, construction, candidate,
                    status="REJECTED", reason="derived_answer_copy",
                ))
                continue
            try:
                contextual = contextual_quartet_evidence(candidate, tokenizers)
            except (TypeError, ValueError) as error:
                ledger.append(_attempt_record_from_group(
                    attempt_index, construction, candidate,
                    status="REJECTED", reason=f"contextual_tokenizer:{error}",
                ))
                continue
            semantic = (
                semantic_task_sha256(candidate.task_a),
                semantic_task_sha256(candidate.task_b),
            )
            prompt_hashes = tuple(
                sha256_text(value) for value in (*candidate.prompts_a, *candidate.prompts_b)
            )
            complete_ids = (*candidate.complete_prompt_ids_a, *candidate.complete_prompt_ids_b)
            reason = registry.rejection_reason(
                semantic_tasks=semantic, prompt_contents=prompt_hashes,
                source_id=slot.source_record_id, donor_id=slot.donor_id,
                template_bank_id=slot.template_bank_id, template_id=slot.template_id,
                complete_prompt_ids=complete_ids,
                owner=(slot.population_id, slot.outer_fold),
            )
            if reason is not None:
                ledger.append(_attempt_record_from_group(
                    attempt_index, construction, candidate,
                    status="REJECTED_GLOBAL", reason=reason,
                    contextual_evidence=contextual,
                ))
                continue
            registry.admit(
                semantic_tasks=semantic, prompt_contents=prompt_hashes,
                source_id=slot.source_record_id, donor_id=slot.donor_id,
                template_bank_id=slot.template_bank_id, template_id=slot.template_id,
                complete_prompt_ids=complete_ids,
                owner=(slot.population_id, slot.outer_fold),
            )
            ledger.append(_attempt_record_from_group(
                attempt_index, construction, candidate,
                contextual_evidence=contextual,
            ))
            accepted_group, accepted_evidence = candidate, contextual
            break
        if accepted_group is None or accepted_evidence is None:
            raise RuntimeError(f"CLOSE_INVALID_INTERVENTION_BOUNDARY:{slot.slot_id}")
        record = QuartetRecord(
            slot, accepted_group, accepted_evidence, tuple(ledger),
        )
        records.append(record)
        if on_record is not None:
            on_record(len(records) - 1, record)
    return tuple(records)


def _natural_attempt_seed(slot: NaturalSlot, attempt_index: int) -> int:
    payload = (
        "a6-natural-attempt-v1\0" + slot.cohort_id + "\0" + slot.slot_id
        + "\0" + str(attempt_index)
    ).encode("utf-8")
    return first64(payload)


def _natural_generation_seed(slot: NaturalSlot) -> int:
    return first64((
        "a6-natural-generation-v1\0" + slot.cohort_id + "\0" + slot.item_id
    ).encode("utf-8"))


def _natural_prompt(task, template_id: str) -> str:
    from .a6_interventions import render_task

    canonical = render_task(task, "canonical", template_id)
    return (
        "Answer the following task with one short answer.\n"
        f"Task: {canonical}\nAnswer:"
    )


def _validate_and_admit_natural_record(
    registry: _CollisionRegistry,
    record: NaturalPromptRecord,
    tokenizers: Mapping[str, Any],
    expected_slot: NaturalSlot | None = None,
) -> None:
    """Replay the complete prompt-only attempt prefix before resume."""
    slot = record.slot if expected_slot is None else expected_slot
    identity_by_scorer = {value.scorer_id: value for value in TOKENIZER_IDENTITIES}
    replayed_ledger = []
    for attempt_index, observed in enumerate(record.attempt_ledger):
        attempt_seed = _natural_attempt_seed(slot, attempt_index)
        pair = construct_task_pair_from_seed(
            attempt_seed=attempt_seed, domain=slot.domain,
            mutation_family=slot.mutation_family, template_id=slot.template_id,
        )
        if pair.status != "ACCEPTED" or pair.task_a is None or pair.task_b is None:
            expected = S0aAttemptRecord(
                attempt_index, attempt_seed, "REJECTED", pair.reason,
                None if pair.task_a is None else semantic_task_sha256(pair.task_a),
                None if pair.task_b is None else semantic_task_sha256(pair.task_b),
                task_sha256_a=None if pair.task_a is None else task_sha256(pair.task_a),
                task_sha256_b=None if pair.task_b is None else task_sha256(pair.task_b),
            )
        elif slot.subdomain == "derived-answer" and not _derived_pair_pass(
            pair.task_a, pair.task_b, (pair.prompts_a[0],), (pair.prompts_b[0],),
        ):
            expected = S0aAttemptRecord(
                attempt_index, attempt_seed, "REJECTED", "derived_answer_copy",
                semantic_task_sha256(pair.task_a), semantic_task_sha256(pair.task_b),
                (sha256_text(pair.prompts_a[0]),), (sha256_text(pair.prompts_b[0]),),
                task_sha256_a=task_sha256(pair.task_a),
                task_sha256_b=task_sha256(pair.task_b),
            )
        else:
            side = sha256_lsb(("a6-natural-side-v1\0" + slot.slot_id).encode("utf-8"))
            selected_task = pair.task_a if side == 0 else pair.task_b
            prompt = _natural_prompt(selected_task, slot.template_id)
            complete_id = sha256_text(
                "a6-natural-prompt-v1\0" + slot.cohort_id + "\0" + prompt
            )
            evidence = build_natural_tokenizer_evidence(
                tokenizers[slot.scorer_id], identity_by_scorer[slot.scorer_id],
                prompt, _natural_generation_seed(slot),
            )
            semantic = (
                semantic_task_sha256(pair.task_a), semantic_task_sha256(pair.task_b),
            )
            prompt_hashes = (
                sha256_text(pair.prompts_a[0]), sha256_text(pair.prompts_b[0]),
            )
            reason = registry.rejection_reason(
                semantic_tasks=semantic, prompt_contents=prompt_hashes,
                source_id=slot.source_record_id, donor_id=slot.donor_id,
                template_bank_id=slot.template_bank_id,
                template_id=slot.template_id, complete_prompt_ids=(complete_id,),
                owner=(slot.cohort_id, slot.outer_fold),
            )
            if reason is not None:
                expected = S0aAttemptRecord(
                    attempt_index, attempt_seed, "REJECTED_GLOBAL", reason,
                    semantic[0], semantic[1], (prompt_hashes[0],),
                    (prompt_hashes[1],), task_sha256_a=task_sha256(pair.task_a),
                    task_sha256_b=task_sha256(pair.task_b),
                )
            else:
                expected_row = NaturalPromptRow(
                    slot.item_id, slot.cohort_id, slot.scorer_id, slot.outer_fold,
                    slot.domain, slot.mutation_family, slot.subdomain, attempt_index,
                    prompt, sha256_text(prompt), semantic_task_sha256(selected_task),
                    slot.source_record_id, slot.donor_id, slot.template_bank_id,
                    slot.template_id, complete_id, evidence,
                )
                expected = S0aAttemptRecord(
                    attempt_index, attempt_seed, "ACCEPTED", "accepted",
                    semantic[0], semantic[1], (prompt_hashes[0],),
                    (prompt_hashes[1],),
                    sha256_bytes(canonical_json_bytes(asdict(evidence))),
                    task_sha256(pair.task_a), task_sha256(pair.task_b),
                )
                if attempt_index != len(record.attempt_ledger) - 1 \
                        or record.row != expected_row:
                    raise ValueError("natural checkpoint accepted before its terminal row")
        if observed != expected:
            raise ValueError("natural checkpoint attempt ledger does not replay")
        replayed_ledger.append(expected)
        if expected.status == "ACCEPTED":
            expected_record = NaturalPromptRecord(
                slot, expected_row, tuple(replayed_ledger),
            )
            if canonical_json_bytes(public_natural_prompt_record(record)) != \
                    canonical_json_bytes(public_natural_prompt_record(expected_record)):
                raise ValueError("natural checkpoint bytes do not match frozen replay")
            _admit_natural_record(registry, record)
            return
    raise ValueError("natural checkpoint ledger has no replayed acceptance")


def build_s0a_natural_prompts(
    tokenizers: Mapping[str, Any],
    *,
    registry: _CollisionRegistry | None = None,
    slots: Sequence[NaturalSlot] | None = None,
    existing_records: Sequence[NaturalPromptRecord] = (),
    on_record: Callable[[int, NaturalPromptRecord], None] | None = None,
) -> tuple[NaturalPromptRecord, ...]:
    """Build three prompt-only cohorts; answers never enter the public row."""
    if tuple(tokenizers) != SCORER_IDS:
        raise ValueError(f"tokenizer order must be exactly {SCORER_IDS}")
    registry = _CollisionRegistry() if registry is None else registry
    identity_by_scorer = {value.scorer_id: value for value in TOKENIZER_IDENTITIES}
    slots = frozen_natural_slots() if slots is None else tuple(slots)
    if slots != frozen_natural_slots()[:len(slots)]:
        raise ValueError("natural slots must be a canonical frozen prefix")
    records = list(existing_records)
    if len(records) > len(slots) or any(
        record.slot != slots[index] for index, record in enumerate(records)
    ):
        raise ValueError("existing natural records are not the frozen prefix")
    for index, record in enumerate(records):
        _validate_and_admit_natural_record(
            registry, record, tokenizers, slots[index],
        )
    for slot in slots[len(records):]:
        ledger = []
        accepted = None
        for attempt_index in range(MAX_ATTEMPTS_PER_SLOT):
            attempt_seed = _natural_attempt_seed(slot, attempt_index)
            pair = construct_task_pair_from_seed(
                attempt_seed=attempt_seed, domain=slot.domain,
                mutation_family=slot.mutation_family, template_id=slot.template_id,
            )
            if pair.status != "ACCEPTED" or pair.task_a is None or pair.task_b is None:
                ledger.append(S0aAttemptRecord(
                    attempt_index, attempt_seed, "REJECTED", pair.reason,
                    None if pair.task_a is None else semantic_task_sha256(pair.task_a),
                    None if pair.task_b is None else semantic_task_sha256(pair.task_b),
                    task_sha256_a=None if pair.task_a is None else task_sha256(pair.task_a),
                    task_sha256_b=None if pair.task_b is None else task_sha256(pair.task_b),
                ))
                continue
            canonical_prompts_a = (pair.prompts_a[0],)
            canonical_prompts_b = (pair.prompts_b[0],)
            if slot.subdomain == "derived-answer" and not _derived_pair_pass(
                pair.task_a, pair.task_b, canonical_prompts_a, canonical_prompts_b,
            ):
                ledger.append(S0aAttemptRecord(
                    attempt_index, attempt_seed, "REJECTED", "derived_answer_copy",
                    semantic_task_sha256(pair.task_a), semantic_task_sha256(pair.task_b),
                    (sha256_text(pair.prompts_a[0]),), (sha256_text(pair.prompts_b[0]),),
                    task_sha256_a=task_sha256(pair.task_a),
                    task_sha256_b=task_sha256(pair.task_b),
                ))
                continue
            side = sha256_lsb(("a6-natural-side-v1\0" + slot.slot_id).encode("utf-8"))
            selected_task = pair.task_a if side == 0 else pair.task_b
            prompt = _natural_prompt(selected_task, slot.template_id)
            prompt_sha = sha256_text(prompt)
            complete_id = sha256_text(
                "a6-natural-prompt-v1\0" + slot.cohort_id + "\0" + prompt
            )
            identity = identity_by_scorer[slot.scorer_id]
            evidence = build_natural_tokenizer_evidence(
                tokenizers[slot.scorer_id], identity, prompt,
                _natural_generation_seed(slot),
            )
            semantic = (
                semantic_task_sha256(pair.task_a), semantic_task_sha256(pair.task_b),
            )
            prompt_hashes = (sha256_text(pair.prompts_a[0]), sha256_text(pair.prompts_b[0]))
            reason = registry.rejection_reason(
                semantic_tasks=semantic, prompt_contents=prompt_hashes,
                source_id=slot.source_record_id, donor_id=slot.donor_id,
                template_bank_id=slot.template_bank_id, template_id=slot.template_id,
                complete_prompt_ids=(complete_id,), owner=(slot.cohort_id, slot.outer_fold),
            )
            if reason is not None:
                ledger.append(S0aAttemptRecord(
                    attempt_index, attempt_seed, "REJECTED_GLOBAL", reason,
                    semantic[0], semantic[1], (prompt_hashes[0],), (prompt_hashes[1],),
                    task_sha256_a=task_sha256(pair.task_a),
                    task_sha256_b=task_sha256(pair.task_b),
                ))
                continue
            registry.admit(
                semantic_tasks=semantic, prompt_contents=prompt_hashes,
                source_id=slot.source_record_id, donor_id=slot.donor_id,
                template_bank_id=slot.template_bank_id, template_id=slot.template_id,
                complete_prompt_ids=(complete_id,), owner=(slot.cohort_id, slot.outer_fold),
            )
            row = NaturalPromptRow(
                slot.item_id, slot.cohort_id, slot.scorer_id, slot.outer_fold,
                slot.domain, slot.mutation_family, slot.subdomain, attempt_index,
                prompt, prompt_sha, semantic_task_sha256(selected_task),
                slot.source_record_id, slot.donor_id, slot.template_bank_id,
                slot.template_id, complete_id, evidence,
            )
            ledger.append(S0aAttemptRecord(
                attempt_index, attempt_seed, "ACCEPTED", "accepted",
                semantic[0], semantic[1], (prompt_hashes[0],), (prompt_hashes[1],),
                sha256_bytes(canonical_json_bytes(asdict(evidence))),
                task_sha256(pair.task_a), task_sha256(pair.task_b),
            ))
            accepted = NaturalPromptRecord(slot, row, tuple(ledger))
            break
        if accepted is None:
            raise RuntimeError(f"CLOSE_INVALID_INTERVENTION_BOUNDARY:{slot.slot_id}")
        records.append(accepted)
        if on_record is not None:
            on_record(len(records) - 1, accepted)
    return tuple(records)


def build_full_s0a_population(
    tokenizers: Mapping[str, Any],
    *,
    on_quartet: Callable[[int, QuartetRecord], None] | None = None,
    on_natural: Callable[[int, NaturalPromptRecord], None] | None = None,
    existing_quartets: Sequence[QuartetRecord] = (),
    existing_natural: Sequence[NaturalPromptRecord] = (),
) -> tuple[tuple[QuartetRecord, ...], tuple[NaturalPromptRecord, ...]]:
    registry = _CollisionRegistry()
    quartets = build_s0a_quartets(
        tokenizers, registry=registry, existing_records=existing_quartets,
        on_record=on_quartet,
    )
    natural = build_s0a_natural_prompts(
        tokenizers, registry=registry, existing_records=existing_natural,
        on_record=on_natural,
    )
    return quartets, natural


def public_quartet_record(record: QuartetRecord) -> dict[str, Any]:
    """Serialize authoritative contextual evidence, never placeholder tokens."""
    group = record.group
    return {
        "slot": asdict(record.slot),
        "group": {
            "group_id": group.group_id,
            "population_id": group.population_id,
            "outer_fold": group.outer_fold,
            "source_record_id": group.source_record_id,
            "donor_id": group.donor_id,
            "template_bank_id": record.slot.template_bank_id,
            "template_id": group.template_id,
            "seed": group.seed,
            "domain": group.domain,
            "mutation_family": group.mutation_family,
            "response_grammar": group.response_grammar,
            "subdomain": record.slot.subdomain,
            "task_a": asdict(group.task_a),
            "task_b": asdict(group.task_b),
            "response_a": asdict(group.response_a),
            "response_b": asdict(group.response_b),
            "prompts_a": list(group.prompts_a),
            "prompts_b": list(group.prompts_b),
            "response_text_a": group.response_text_a,
            "response_text_b": group.response_text_b,
            "ast_sha256_a": group.ast_sha256_a,
            "ast_sha256_b": group.ast_sha256_b,
            "semantic_task_sha256_a": semantic_task_sha256(group.task_a),
            "semantic_task_sha256_b": semantic_task_sha256(group.task_b),
            "response_sha256_a": group.response_sha256_a,
            "response_sha256_b": group.response_sha256_b,
            "complete_prompt_ids_a": list(group.complete_prompt_ids_a),
            "complete_prompt_ids_b": list(group.complete_prompt_ids_b),
            "mechanical_truth": [[1, 0], [0, 1]],
        },
        "contextual_evidence": {
            scorer_id: [asdict(value) for value in evidence]
            for scorer_id, evidence in record.contextual_evidence
        },
        "attempt_ledger": [asdict(value) for value in record.attempt_ledger],
        "placeholder_token_evidence_persisted": False,
    }


def public_natural_prompt_record(record: NaturalPromptRecord) -> dict[str, Any]:
    payload = asdict(record.row)
    sanitized = sanitize_natural_prompt_row(payload)
    return {
        "row": asdict(sanitized),
        "attempt_ledger": [asdict(value) for value in record.attempt_ledger],
    }


def _attempt_record_from_public(value: Mapping[str, Any]) -> S0aAttemptRecord:
    expected_fields = {field.name for field in fields(S0aAttemptRecord)}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("attempt-ledger record schema mismatch")
    data = dict(value)
    if type(data["attempt_index"]) is not int or data["attempt_index"] < 0 \
            or type(data["attempt_seed"]) is not int or data["attempt_seed"] < 0:
        raise ValueError("attempt-ledger integer fields are invalid")
    if data["status"] not in {"REJECTED", "REJECTED_GLOBAL", "ACCEPTED"} \
            or not isinstance(data["reason"], str) or not data["reason"]:
        raise ValueError("attempt-ledger status/reason is invalid")
    for name in ("prompt_content_sha256_a", "prompt_content_sha256_b"):
        raw = data.get(name, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("attempt-ledger prompt hashes use an invalid schema")
        if any(
            not isinstance(item, str) or len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for item in raw
        ):
            raise ValueError("attempt-ledger prompt hash is invalid")
        data[name] = tuple(raw)
    for name in (
        "semantic_task_sha256_a", "semantic_task_sha256_b",
        "contextual_evidence_sha256", "task_sha256_a", "task_sha256_b",
    ):
        item = data[name]
        if item is not None and (
            not isinstance(item, str) or len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
        ):
            raise ValueError(f"attempt-ledger {name} is invalid")
    return S0aAttemptRecord(**data)


def quartet_record_from_public(payload: Mapping[str, Any]) -> QuartetRecord:
    """Reconstruct and strictly validate one immutable resume checkpoint."""
    if set(payload) != {
        "slot", "group", "contextual_evidence", "attempt_ledger",
        "placeholder_token_evidence_persisted",
    } or payload["placeholder_token_evidence_persisted"] is not False:
        raise ValueError("quartet checkpoint public schema mismatch")
    slot = QuartetSlot(**payload["slot"])
    group_value = payload["group"]
    if not isinstance(group_value, Mapping):
        raise TypeError("quartet checkpoint group must be a mapping")
    task_a_value, task_b_value = dict(group_value["task_a"]), dict(group_value["task_b"])
    task_a_value["records"] = tuple(tuple(row) for row in task_a_value["records"])
    task_b_value["records"] = tuple(tuple(row) for row in task_b_value["records"])
    response_a_value = dict(group_value["response_a"])
    response_b_value = dict(group_value["response_b"])
    for value in (response_a_value, response_b_value):
        value["source_facts"] = tuple(value.get("source_facts", ()))
        value["selected_values"] = tuple(value.get("selected_values", ()))
    ledger = tuple(_attempt_record_from_public(value) for value in payload["attempt_ledger"])
    if not ledger or ledger[-1].status != "ACCEPTED":
        raise ValueError("quartet checkpoint has no accepted terminal attempt")
    if tuple(value.attempt_index for value in ledger) != tuple(range(len(ledger))):
        raise ValueError("quartet checkpoint attempt schedule is not contiguous")
    group = ReciprocalGroup(
        group_id=group_value["group_id"], population_id=group_value["population_id"],
        outer_fold=group_value["outer_fold"],
        source_record_id=group_value["source_record_id"],
        donor_id=group_value["donor_id"], template_id=group_value["template_id"],
        seed=group_value["seed"], domain=group_value["domain"],
        mutation_family=group_value["mutation_family"],
        response_grammar=group_value["response_grammar"],
        task_a=TaskAST(**task_a_value), task_b=TaskAST(**task_b_value),
        response_a=ResponseAST(**response_a_value), response_b=ResponseAST(**response_b_value),
        prompts_a=tuple(group_value["prompts_a"]), prompts_b=tuple(group_value["prompts_b"]),
        response_text_a=group_value["response_text_a"],
        response_text_b=group_value["response_text_b"],
        ast_sha256_a=group_value["ast_sha256_a"], ast_sha256_b=group_value["ast_sha256_b"],
        response_sha256_a=group_value["response_sha256_a"],
        response_sha256_b=group_value["response_sha256_b"],
        attempt_index=ledger[-1].attempt_index,
        complete_prompt_ids_a=tuple(group_value["complete_prompt_ids_a"]),
        complete_prompt_ids_b=tuple(group_value["complete_prompt_ids_b"]),
        prompt_token_counts_a=(), prompt_token_counts_b=(),
        response_token_ids_a=(), response_token_ids_b=(),
    )
    expected_group_id = hashlib.sha256(
        "\0".join((
            "a6-group", slot.population_id, str(slot.outer_fold),
            slot.source_record_id, slot.donor_id, slot.template_id,
            str(slot.seed), slot.domain, slot.mutation_family,
            slot.response_grammar,
        )).encode("utf-8")
    ).hexdigest()
    if (
        group.group_id != expected_group_id
        or group.population_id != slot.population_id
        or group.outer_fold != slot.outer_fold
        or group.source_record_id != slot.source_record_id
        or group.donor_id != slot.donor_id
        or group.template_id != slot.template_id
        or group.seed != slot.seed
        or group.domain != slot.domain
        or group.mutation_family != slot.mutation_family
        or group.response_grammar != slot.response_grammar
        or group.ast_sha256_a != task_sha256(group.task_a)
        or group.ast_sha256_b != task_sha256(group.task_b)
        or group.response_sha256_a != sha256_text(group.response_text_a)
        or group.response_sha256_b != sha256_text(group.response_text_b)
        or group.response_text_a != render_response(group.response_a)
        or group.response_text_b != render_response(group.response_b)
        or group.prompts_a != tuple(
            render_task(group.task_a, rendering, slot.template_id)
            for rendering in RENDERINGS
        )
        or group.prompts_b != tuple(
            render_task(group.task_b, rendering, slot.template_id)
            for rendering in RENDERINGS
        )
    ):
        raise ValueError("quartet checkpoint does not match its frozen slot")
    contextual = tuple(
        (
            scorer_id,
            tuple(ContextualInputEvidence(**{
                **dict(value),
                "prefix_ids": tuple(value["prefix_ids"]),
                "response_ids": tuple(value["response_ids"]),
                "suffix_ids": tuple(value["suffix_ids"]),
            }) for value in payload["contextual_evidence"][scorer_id]),
        )
        for scorer_id in SCORER_IDS
    )
    record = QuartetRecord(slot, group, contextual, ledger)
    terminal = ledger[-1]
    expected_context_sha = sha256_bytes(canonical_json_bytes([
        [scorer_id, [asdict(item) for item in evidence]]
        for scorer_id, evidence in contextual
    ]))
    terminal_construction, _ = construct_reciprocal_attempt(
        seed=slot.seed, attempt_index=terminal.attempt_index,
        population_id=slot.population_id, outer_fold=slot.outer_fold,
        source_record_id=slot.source_record_id, donor_id=slot.donor_id,
        template_id=slot.template_id, domain=slot.domain,
        mutation_family=slot.mutation_family,
        response_grammar=slot.response_grammar,
        tokenizers=_PLACEHOLDER_TOKENIZERS,
    )
    if (
        terminal.attempt_seed != terminal_construction.attempt_seed
        or terminal.semantic_task_sha256_a != semantic_task_sha256(group.task_a)
        or terminal.semantic_task_sha256_b != semantic_task_sha256(group.task_b)
        or terminal.task_sha256_a != task_sha256(group.task_a)
        or terminal.task_sha256_b != task_sha256(group.task_b)
        or terminal.prompt_content_sha256_a != tuple(
            sha256_text(value) for value in group.prompts_a
        )
        or terminal.prompt_content_sha256_b != tuple(
            sha256_text(value) for value in group.prompts_b
        )
        or terminal.contextual_evidence_sha256 != expected_context_sha
    ):
        raise ValueError("quartet terminal ledger hashes do not bind its record")
    if json.loads(canonical_json_bytes(public_quartet_record(record))) != payload:
        raise ValueError("quartet checkpoint does not round-trip canonically")
    return record


def natural_record_from_public(payload: Mapping[str, Any]) -> NaturalPromptRecord:
    """Reconstruct and strictly validate one prompt-only resume checkpoint."""
    if set(payload) != {"row", "attempt_ledger"}:
        raise ValueError("natural checkpoint public schema mismatch")
    row = sanitize_natural_prompt_row(payload["row"])
    slot = _frozen_natural_slot_by_source().get(row.source_record_id)
    if slot is None:
        raise ValueError("natural checkpoint is outside the frozen schedule")
    ledger = tuple(_attempt_record_from_public(value) for value in payload["attempt_ledger"])
    if not ledger or ledger[-1].status != "ACCEPTED":
        raise ValueError("natural checkpoint has no accepted terminal attempt")
    if tuple(value.attempt_index for value in ledger) != tuple(range(len(ledger))) \
            or any(
                value.attempt_seed != _natural_attempt_seed(slot, value.attempt_index)
                for value in ledger
            ) or row.attempt_index != ledger[-1].attempt_index:
        raise ValueError("natural checkpoint attempt schedule is not contiguous")
    pair = construct_task_pair_from_seed(
        attempt_seed=ledger[-1].attempt_seed, domain=slot.domain,
        mutation_family=slot.mutation_family, template_id=slot.template_id,
    )
    side = sha256_lsb(("a6-natural-side-v1\0" + slot.slot_id).encode("utf-8"))
    selected = pair.task_a if side == 0 else pair.task_b
    if pair.status != "ACCEPTED" or selected is None \
            or row.prompt_text != _natural_prompt(selected, slot.template_id) \
            or row.semantic_task_sha256 != semantic_task_sha256(selected):
        raise ValueError("natural checkpoint prompt does not replay from its frozen attempt")
    terminal = ledger[-1]
    expected_context_sha = sha256_bytes(canonical_json_bytes(asdict(row.tokenizer_evidence)))
    if (
        terminal.semantic_task_sha256_a != semantic_task_sha256(pair.task_a)
        or terminal.semantic_task_sha256_b != semantic_task_sha256(pair.task_b)
        or terminal.task_sha256_a != task_sha256(pair.task_a)
        or terminal.task_sha256_b != task_sha256(pair.task_b)
        or terminal.prompt_content_sha256_a != (sha256_text(pair.prompts_a[0]),)
        or terminal.prompt_content_sha256_b != (sha256_text(pair.prompts_b[0]),)
        or terminal.contextual_evidence_sha256 != expected_context_sha
    ):
        raise ValueError("natural terminal ledger hashes do not bind its record")
    record = NaturalPromptRecord(slot, row, ledger)
    if json.loads(canonical_json_bytes(public_natural_prompt_record(record))) != payload:
        raise ValueError("natural checkpoint does not round-trip canonically")
    return record


def _contains_forbidden_key(key: str) -> bool:
    lowered = key.casefold()
    return any(fragment in lowered for fragment in _FORBIDDEN_NATURAL_KEYS)


def _reject_nested_target_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        for key in keys:
            if not isinstance(key, str) or _contains_forbidden_key(key):
                raise ValueError("target-like nested key reached the natural firewall")
        for key in keys:
            _reject_nested_target_keys(value[key])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_nested_target_keys(item)


def _canonical_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{field_name} is not canonical SHA-256")
    return value


_NATURAL_ROW_FIELDS = tuple(field.name for field in fields(NaturalPromptRow))
_TOKENIZER_EVIDENCE_FIELDS = tuple(
    field.name for field in fields(NaturalTokenizerEvidence)
)


@lru_cache(maxsize=1)
def _frozen_natural_slot_by_source() -> dict[str, NaturalSlot]:
    return {value.source_record_id: value for value in frozen_natural_slots()}


def sanitize_natural_prompt_row(payload: Mapping[str, Any]) -> NaturalPromptRow:
    """Copy exactly the public prompt-only allowlist without touching extras."""
    keys = tuple(payload.keys())
    if set(keys) != set(_NATURAL_ROW_FIELDS):
        raise ValueError("natural prompt row does not match the exact public allowlist")
    if any(_contains_forbidden_key(key) for key in keys):
        raise ValueError("target-like key reached the natural prompt firewall")
    _reject_nested_target_keys(payload)
    evidence_payload = payload["tokenizer_evidence"]
    if not isinstance(evidence_payload, Mapping):
        raise TypeError("tokenizer_evidence must be its fixed mapping schema")
    evidence_keys = tuple(evidence_payload.keys())
    if set(evidence_keys) != set(_TOKENIZER_EVIDENCE_FIELDS):
        raise ValueError("tokenizer evidence schema mismatch")
    if any(_contains_forbidden_key(key) for key in evidence_keys):
        raise ValueError("target-like key reached tokenizer evidence")
    evidence_values = {key: evidence_payload[key] for key in _TOKENIZER_EVIDENCE_FIELDS}
    raw_ids = evidence_values["input_ids"]
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)) \
            or not raw_ids or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in raw_ids
            ):
        raise ValueError("natural input IDs must be nonnegative integers")
    evidence_values["input_ids"] = tuple(raw_ids)
    raw_parameters = evidence_values["generation_parameters"]
    if not isinstance(raw_parameters, Sequence) or isinstance(
        raw_parameters, (str, bytes)
    ) or any(
        not isinstance(pair, Sequence) or isinstance(pair, (str, bytes))
        or len(pair) != 2 or not isinstance(pair[0], str)
        for pair in raw_parameters
    ):
        raise ValueError("generation parameters must use the exact pair schema")
    evidence_values["generation_parameters"] = tuple(
        (pair[0], pair[1]) for pair in raw_parameters
    )
    evidence = NaturalTokenizerEvidence(**evidence_values)
    values = {key: payload[key] for key in _NATURAL_ROW_FIELDS if key != "tokenizer_evidence"}
    values["tokenizer_evidence"] = evidence
    row = NaturalPromptRow(**values)
    expected_slot = _frozen_natural_slot_by_source().get(row.source_record_id)
    if expected_slot is None:
        raise ValueError("natural prompt row is outside the frozen slot schedule")
    frozen_fields = (
        "item_id", "cohort_id", "scorer_id", "outer_fold", "domain",
        "mutation_family", "subdomain", "source_record_id", "donor_id",
        "template_bank_id", "template_id",
    )
    if any(getattr(row, name) != getattr(expected_slot, name) for name in frozen_fields):
        raise ValueError("natural prompt row does not match its frozen slot")
    if not isinstance(row.prompt_text, str) or not row.prompt_text \
            or _canonical_sha256(row.prompt_sha256, "prompt_sha256") != sha256_text(row.prompt_text):
        raise ValueError("natural prompt hash mismatch")
    if isinstance(row.attempt_index, bool) or not isinstance(row.attempt_index, int) \
            or not 0 <= row.attempt_index < MAX_ATTEMPTS_PER_SLOT:
        raise ValueError("natural attempt index is invalid")
    if not row.source_record_id.startswith(f"{row.cohort_id}:fold{row.outer_fold}:source:"):
        raise ValueError("natural source-record owner mismatch")
    slot_id = row.source_record_id.split(":source:", 1)[1]
    if row.item_id != sha256_text("a6-natural-item-v1\0" + slot_id):
        raise ValueError("natural item ID mismatch")
    if row.donor_id != f"{row.cohort_id}:fold{row.outer_fold}:donor:{slot_id}":
        raise ValueError("natural donor ID mismatch")
    if not row.template_bank_id.startswith(
        f"{row.cohort_id}:fold{row.outer_fold}:template-bank:"
    ) or row.template_id != f"{row.template_bank_id}:instance:{slot_id}":
        raise ValueError("natural template ownership mismatch")
    expected_complete = sha256_text(
        "a6-natural-prompt-v1\0" + row.cohort_id + "\0" + row.prompt_text
    )
    if row.complete_prompt_id != expected_complete:
        raise ValueError("natural complete-prompt ID mismatch")
    _canonical_sha256(row.semantic_task_sha256, "semantic_task_sha256")
    identity = {value.scorer_id: value for value in TOKENIZER_IDENTITIES}[row.scorer_id]
    if (
        evidence.scorer_id != row.scorer_id
        or evidence.repository != identity.repository
        or evidence.revision != identity.revision
    ):
        raise ValueError("natural tokenizer identity mismatch")
    _canonical_sha256(evidence.prefix_text_sha256, "prefix_text_sha256")
    _canonical_sha256(evidence.input_ids_sha256, "input_ids_sha256")
    _canonical_sha256(evidence.attention_mask_sha256, "attention_mask_sha256")
    if not evidence.input_ids or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in evidence.input_ids
    ):
        raise ValueError("natural input IDs are empty or invalid")
    if evidence.input_length != len(evidence.input_ids) \
            or evidence.input_ids_sha256 != _ids_sha256(evidence.input_ids):
        raise ValueError("natural input-ID evidence mismatch")
    if evidence.attention_mask_sha256 != _ids_sha256((1,) * len(evidence.input_ids)):
        raise ValueError("natural attention-mask evidence mismatch")
    expected_seed = first64((
        "a6-natural-generation-v1\0" + row.cohort_id + "\0" + row.item_id
    ).encode("utf-8"))
    if evidence.generation_seed != expected_seed:
        raise ValueError("natural generation seed mismatch")
    expected_parameters = (
        ("do_sample", False), ("num_beams", 1), ("max_new_tokens", 150),
        ("batch_size", 1), ("attention_mask_dtype", "int64"),
        ("custom_stopping_strings", False), ("custom_logits_processors", False),
    )
    if evidence.generation_parameters != expected_parameters:
        raise ValueError("natural generation parameters changed")
    return row


@dataclass(frozen=True)
class FutureLlamaSidecarSchema:
    key_fields: tuple[str, ...]
    future_fields: tuple[str, ...]


def future_llama_sidecar_schema() -> FutureLlamaSidecarSchema:
    return FutureLlamaSidecarSchema(
        ("cohort_id", "item_id", "response_sha256"),
        (
            "prompt_token_ids", "prompt_token_ids_sha256", "generation_seed",
            "generated_token_ids", "generated_token_ids_sha256", "stop_reason",
            "decoded_response_sha256", "contextual_response_span_evidence",
            "mechanical_correctness",
        ),
    )


def assert_no_a6_llama_payloads(paths: Sequence[str | Path]) -> None:
    forbidden = ("response", "feature", "correct", "sidecar", "target", "label")
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix().casefold()
            if any(fragment in relative for fragment in forbidden):
                raise RuntimeError(f"A6 Llama future payload exists before S2: {relative}")


def popqa_opaque_reservation() -> dict[str, Any]:
    if sha256_text(POPQA_TEMPLATE) != POPQA_TEMPLATE_SHA256:
        raise AssertionError("frozen PopQA prompt-template hash drift")
    return {
        "dataset": "akariasai/PopQA",
        "revision": POPQA_REVISION,
        "split": "test",
        "expected_rows": POPQA_ROWS,
        "opaque_row_ids": [f"popqa:test:{index}" for index in range(POPQA_ROWS)],
        "prompt_template_sha256": POPQA_TEMPLATE_SHA256,
        "later_validation_schema": {
            "required": ["row_index", "prompt_sha256", "response_sha256"],
            "forbidden_before_s4": [
                "question", "object", "aliases", "possible_answers", "response", "label",
            ],
        },
        "dataset_content_accessed": False,
    }


def _context_prefix_lengths(record: QuartetRecord) -> tuple[int, ...]:
    lengths = []
    for _, evidence in record.contextual_evidence:
        # Every crossed prompt/response cell is required to have this same
        # rendering-specific count by ``contextual_quartet_evidence``.
        lengths.extend(len(value.prefix_ids) for value in evidence[:4])
    if len(lengths) != 12:
        raise AssertionError("quartet prefix-length evidence must contain 4x3 values")
    return tuple(lengths)


def _ranks_and_bins(
    values: Sequence[tuple[float, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    if len(values) != 50 or len({group_id for _, group_id in values}) != 50:
        raise ValueError("null-bin ranking requires exactly 50 unique groups")
    ordered = sorted(values, key=lambda value: (value[0], value[1]))
    ranks = {group_id: rank for rank, (_, group_id) in enumerate(ordered)}
    bins = {
        group_id: min(3, math.floor(4 * rank / 50))
        for group_id, rank in ranks.items()
    }
    return ranks, bins


@dataclass(frozen=True)
class NullMergeOperation:
    left: tuple[tuple[int, int], ...]
    right: tuple[tuple[int, int], ...]
    merged: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NullStratumCell:
    population_id: str
    domain: str
    mutation_family: str
    response_grammar: str
    group_ranks: tuple[tuple[str, float, int, int, int, int, int], ...]
    occupied_bins: tuple[tuple[int, int], ...]
    empty_bins: tuple[tuple[int, int], ...]
    merges: tuple[NullMergeOperation, ...]
    group_to_stratum: tuple[tuple[str, str], ...]
    final_partition_counts: tuple[tuple[str, str, int], ...]


def _partition_memberships(
    cell: Sequence[QuartetRecord],
    inner_manifest: Mapping[tuple[int, str], int],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for outer in range(5):
        held = {value.group.group_id for value in cell if value.slot.outer_fold == outer}
        train = {value.group.group_id for value in cell if value.slot.outer_fold != outer}
        output[f"outer:{outer}:held"] = held
        output[f"outer:{outer}:train"] = train
        for inner in range(5):
            validation = {
                group_id for group_id in train
                if inner_manifest[(outer, group_id)] == inner
            }
            output[f"outer:{outer}:inner:{inner}:validation"] = validation
            output[f"outer:{outer}:inner:{inner}:train"] = train - validation
    return output


def _component_key(component: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(component))


def _component_distance(
    left: frozenset[tuple[int, int]], right: frozenset[tuple[int, int]],
) -> int:
    return min(
        abs(a - c) + abs(b - d) for a, b in left for c, d in right
    )


def build_null_strata(
    records: Sequence[QuartetRecord],
    inner_folds: Sequence[tuple[int, str, int]],
) -> tuple[NullStratumCell, ...]:
    inner_lookup = {(outer, group_id): inner for outer, group_id, inner in inner_folds}
    output = []
    for population in QUARTET_POPULATIONS:
        for domain in DOMAINS:
            for mutation in MUTATIONS:
                for grammar in RESPONSE_GRAMMARS:
                    cell = [
                        value for value in records
                        if value.slot.population_id == population
                        and value.slot.domain == domain
                        and value.slot.mutation_family == mutation
                        and value.slot.response_grammar == grammar
                    ]
                    if len(cell) != 50:
                        raise ValueError("null-stratum cell must contain exactly 50 groups")
                    length_values = [
                        (sum(_context_prefix_lengths(value)) / 12, value.group.group_id)
                        for value in cell
                    ]
                    complexity_values = [
                        (task_complexity(value.group.task_a)[0]
                         + 2 * task_complexity(value.group.task_a)[1],
                         value.group.group_id)
                        for value in cell
                    ]
                    length_ranks, length_bins = _ranks_and_bins(length_values)
                    complexity_ranks, complexity_bins = _ranks_and_bins(complexity_values)
                    length_lookup = {group_id: value for value, group_id in length_values}
                    complexity_lookup = {
                        group_id: int(value) for value, group_id in complexity_values
                    }
                    group_bins = {
                        value.group.group_id: (
                            length_bins[value.group.group_id],
                            complexity_bins[value.group.group_id],
                        ) for value in cell
                    }
                    occupied = tuple(sorted(set(group_bins.values())))
                    all_bins = {(left, right) for left in range(4) for right in range(4)}
                    empty = tuple(sorted(all_bins - set(occupied)))
                    components = [frozenset((value,)) for value in occupied]
                    partitions = _partition_memberships(cell, inner_lookup)

                    def component_groups(component):
                        return {
                            group_id for group_id, bin_value in group_bins.items()
                            if bin_value in component
                        }

                    merges = []
                    while True:
                        deficient = [
                            component for component in components
                            if any(
                                len(component_groups(component) & members) < 4
                                for members in partitions.values()
                            )
                        ]
                        if not deficient:
                            break
                        left = min(deficient, key=_component_key)
                        partners = [value for value in components if value != left]
                        if not partners:
                            raise RuntimeError("CLOSE_INVALID_INTERVENTION_BOUNDARY:null_stratum")
                        distance = min(_component_distance(left, value) for value in partners)
                        right = min(
                            (value for value in partners if _component_distance(left, value) == distance),
                            key=_component_key,
                        )
                        merged = left | right
                        merges.append(NullMergeOperation(
                            _component_key(left), _component_key(right), _component_key(merged),
                        ))
                        components = [value for value in components if value not in (left, right)]
                        components.append(merged)
                        components.sort(key=_component_key)
                    mapping = []
                    counts = []
                    for component in sorted(components, key=_component_key):
                        component_id = sha256_bytes(canonical_json_bytes({
                            "population_id": population, "domain": domain,
                            "mutation_family": mutation, "response_grammar": grammar,
                            "member_bins": _component_key(component),
                        }))
                        groups = component_groups(component)
                        mapping.extend((group_id, component_id) for group_id in sorted(groups))
                        for partition_id, members in sorted(partitions.items()):
                            count = len(groups & members)
                            if count < 4:
                                raise RuntimeError(
                                    "CLOSE_INVALID_INTERVENTION_BOUNDARY:null_stratum_count"
                                )
                            counts.append((component_id, partition_id, count))
                    output.append(NullStratumCell(
                        population, domain, mutation, grammar,
                        tuple(sorted(
                            (
                                group_id, length_lookup[group_id],
                                complexity_lookup[group_id], length_ranks[group_id],
                                complexity_ranks[group_id], length_bins[group_id],
                                complexity_bins[group_id],
                            )
                            for group_id in group_bins
                        )), occupied, empty, tuple(merges), tuple(sorted(mapping)),
                        tuple(sorted(counts)),
                    ))
    return tuple(output)


def verify_null_strata(
    records: Sequence[QuartetRecord],
    inner_folds: Sequence[tuple[int, str, int]],
    frozen: Sequence[NullStratumCell],
) -> None:
    if tuple(frozen) != build_null_strata(records, inner_folds):
        raise ValueError("S0a null-stratum replay differs from frozen artifact")


__all__ = [
    "CONTRACT_VERSION", "MAX_ATTEMPTS_PER_SLOT", "MODEL_IDENTITIES",
    "NATURAL_COHORTS", "POPQA_REVISION", "POPQA_ROWS", "POPQA_TEMPLATE",
    "POPQA_TEMPLATE_SHA256", "QUARTET_POPULATIONS", "SCORER_IDS",
    "ContextualInputEvidence", "FutureLlamaSidecarSchema", "ModelIdentity",
    "NaturalPromptRecord", "NaturalPromptRow", "NaturalSlot",
    "NaturalTokenizerEvidence", "NullMergeOperation", "NullStratumCell",
    "QuartetRecord", "QuartetSlot", "S0aAttemptRecord", "SnapshotFile",
    "SnapshotManifest", "assert_no_a6_llama_payloads", "build_contextual_input_evidence",
    "build_full_s0a_population", "build_natural_tokenizer_evidence",
    "build_null_strata", "build_s0a_natural_prompts", "build_s0a_quartets",
    "canonical_json_bytes", "first64", "frozen_natural_slots",
    "frozen_quartet_slots", "future_llama_sidecar_schema", "inner_fold_manifest",
    "load_verified_fast_tokenizer", "popqa_opaque_reservation",
    "public_natural_prompt_record", "public_quartet_record",
    "natural_record_from_public", "quartet_record_from_public",
    "prepare_content_addressed_tokenizer_snapshot", "sanitize_natural_prompt_row",
    "sha256_bytes", "sha256_file", "sha256_text",
    "sha256_lsb",
    "verify_content_addressed_snapshot", "verify_null_strata",
]
