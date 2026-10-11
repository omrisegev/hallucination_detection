"""Deterministic A6-S0 population schedules and global rejection ledger.

Local semantic/token rejections are produced by :mod:`a6_interventions`.
This layer additionally rejects locally legal attempts that collide with any
earlier AST, prompt, group, donor, source, or template boundary constraint.
The combined Qwen+Llama schedule must be built/audited as one object so their
disjointness is structural rather than checked after selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import Counter
import hashlib
import math
import re
from typing import Callable, Mapping, Sequence

from .a6_interventions import (
    DOMAINS,
    MUTATIONS,
    RESPONSE_GRAMMARS,
    ConstructionAttempt,
    ReciprocalConstruction,
    ReciprocalGroup,
    audit_reciprocal_group,
    construct_reciprocal_attempt,
    public_group_record,
    semantic_task_sha256,
)


QWEN_POPULATION_ID = "qwen-source"
LLAMA_POPULATION_ID = "llama-audit"
QWEN_SEED_NAMESPACE = 640_000
LLAMA_SEED_NAMESPACE = 650_000


@dataclass(frozen=True)
class PopulationSlot:
    slot_id: str
    population_id: str
    seed: int
    outer_fold: int
    source_record_id: str
    donor_id: str
    template_id: str
    domain: str
    mutation_family: str
    response_grammar: str


@dataclass(frozen=True)
class PopulationConstruction:
    slots: tuple[PopulationSlot, ...]
    groups: tuple[ReciprocalGroup, ...]
    attempt_ledgers: tuple[tuple[ConstructionAttempt, ...], ...]


def _stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def frozen_population_slots(population_id: str, seed_namespace: int) -> tuple[PopulationSlot, ...]:
    """Create the exact 900-slot balanced schedule for one scorer population."""
    if not population_id or not isinstance(seed_namespace, int):
        raise ValueError("population_id and integer seed_namespace are required")
    slots = []
    for domain in DOMAINS:
        for mutation in MUTATIONS:
            for within_stratum in range(100):
                grammar = RESPONSE_GRAMMARS[within_stratum // 50]
                slot_id = f"{population_id}:{domain}:{mutation}:{within_stratum:03d}"
                seed = _stable_int("a6-s0-slot", seed_namespace, slot_id)
                outer_fold = _stable_int("a6-s0-fold", slot_id) % 5
                owner = f"{population_id}:fold{outer_fold}"
                slots.append(PopulationSlot(
                    slot_id, population_id, seed, outer_fold,
                    f"{owner}:source:{within_stratum:03d}:{domain}:{mutation}",
                    f"{owner}:donor:{within_stratum:03d}:{domain}:{mutation}",
                    f"{owner}:template:{within_stratum:03d}:{domain}:{mutation}:{grammar}",
                    domain, mutation, grammar,
                ))
    return tuple(slots)


def frozen_combined_slots() -> tuple[PopulationSlot, ...]:
    return (
        frozen_population_slots(QWEN_POPULATION_ID, QWEN_SEED_NAMESPACE)
        + frozen_population_slots(LLAMA_POPULATION_ID, LLAMA_SEED_NAMESPACE)
    )


def _validate_slot_manifest(slots: Sequence[PopulationSlot]) -> None:
    if not slots or len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("population slot IDs must be nonempty and unique")
    ownership: dict[tuple[str, str], tuple[str, int]] = {}
    for slot in slots:
        if slot.domain not in DOMAINS or slot.mutation_family not in MUTATIONS:
            raise ValueError("population slot has an invalid semantic stratum")
        if slot.response_grammar not in RESPONSE_GRAMMARS:
            raise ValueError("population slot has an invalid response grammar")
        owner = (slot.population_id, slot.outer_fold)
        for kind, identifier in (
            ("source", slot.source_record_id), ("donor", slot.donor_id),
            ("template", slot.template_id),
        ):
            key = (kind, identifier)
            previous = ownership.setdefault(key, owner)
            if previous != owner:
                raise ValueError(f"{kind} identity crosses population or outer fold")


def _globally_legal_reason(
    group: ReciprocalGroup,
    *, semantic_ast_hashes: set[str], prompt_content_hashes: set[str],
    group_ids: set[str],
) -> str | None:
    semantic = tuple(semantic_task_sha256(task) for task in (group.task_a, group.task_b))
    if any(value in semantic_ast_hashes for value in semantic):
        return "global_semantic_ast_collision"
    prompt_content = tuple(
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in (*group.prompts_a, *group.prompts_b)
    )
    if any(value in prompt_content_hashes for value in prompt_content):
        return "global_prompt_content_collision"
    if group.group_id in group_ids:
        return "global_group_collision"
    return None


def build_population(
    slots: Sequence[PopulationSlot],
    tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
    *,
    max_attempts_per_slot: int = 10_000,
) -> PopulationConstruction:
    """Build a combined population and preserve local plus global rejections."""
    slots = tuple(slots)
    _validate_slot_manifest(slots)
    semantic_ast_hashes: set[str] = set()
    prompt_content_hashes: set[str] = set()
    group_ids: set[str] = set()
    groups, ledgers = [], []
    for slot in slots:
        ledger = []
        accepted = None
        for attempt_index in range(int(max_attempts_per_slot)):
            record, candidate = construct_reciprocal_attempt(
                seed=slot.seed, attempt_index=attempt_index,
                population_id=slot.population_id, outer_fold=slot.outer_fold,
                source_record_id=slot.source_record_id, donor_id=slot.donor_id,
                template_id=slot.template_id, domain=slot.domain,
                mutation_family=slot.mutation_family,
                response_grammar=slot.response_grammar, tokenizers=tokenizers,
            )
            if candidate is None:
                ledger.append(record)
                continue
            global_reason = _globally_legal_reason(
                candidate, semantic_ast_hashes=semantic_ast_hashes,
                prompt_content_hashes=prompt_content_hashes, group_ids=group_ids,
            )
            if global_reason is not None:
                ledger.append(replace(
                    record, status="REJECTED_GLOBAL", reason=global_reason
                ))
                continue
            ledger.append(record)
            accepted = candidate
            semantic_ast_hashes.update(
                semantic_task_sha256(task) for task in (candidate.task_a, candidate.task_b)
            )
            prompt_content_hashes.update(
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in (*candidate.prompts_a, *candidate.prompts_b)
            )
            group_ids.add(candidate.group_id)
            break
        if accepted is None:
            raise RuntimeError(f"A6 population slot exhausted: {slot.slot_id}")
        groups.append(accepted)
        ledgers.append(tuple(ledger))
    return PopulationConstruction(slots, tuple(groups), tuple(ledgers))


def audit_population(
    population: PopulationConstruction,
    tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
    *,
    require_frozen_allocation: bool = False,
) -> dict:
    """Replay the full combined schedule and fail on any ledger or global drift."""
    _validate_slot_manifest(population.slots)
    allocation_pass = True
    if require_frozen_allocation:
        allocation_pass = population.slots == frozen_combined_slots()
        population_ids = sorted({slot.population_id for slot in population.slots})
        allocation_pass = allocation_pass and len(population_ids) == 2
        for population_id in population_ids:
            selected = [slot for slot in population.slots if slot.population_id == population_id]
            allocation_pass = allocation_pass and len(selected) == 900
            for domain in DOMAINS:
                for mutation in MUTATIONS:
                    cell = [
                        slot for slot in selected
                        if slot.domain == domain and slot.mutation_family == mutation
                    ]
                    allocation_pass = allocation_pass and len(cell) == 100
                    allocation_pass = allocation_pass and all(
                        sum(slot.response_grammar == grammar for slot in cell) == 50
                        for grammar in RESPONSE_GRAMMARS
                    )
    try:
        replay = build_population(
            population.slots, tokenizers,
            max_attempts_per_slot=max(len(ledger) for ledger in population.attempt_ledgers),
        )
        replay_pass = replay == population
    except (RuntimeError, TypeError, ValueError):
        replay_pass = False
    lengths_pass = (
        len(population.slots) == len(population.groups) == len(population.attempt_ledgers)
        and all(ledger and ledger[-1].status == "ACCEPTED" for ledger in population.attempt_ledgers)
    )
    group_pass = lengths_pass and all(
        audit_reciprocal_group(group, tokenizers)["pass"] for group in population.groups
    )
    checks = {
        "allocation_pass": bool(allocation_pass),
        "lengths_pass": bool(lengths_pass),
        "group_pass": bool(group_pass),
        "replay_pass": bool(replay_pass),
        "n_groups": len(population.groups),
        "n_local_rejections": sum(
            record.status == "REJECTED"
            for ledger in population.attempt_ledgers for record in ledger
        ),
        "n_global_rejections": sum(
            record.status == "REJECTED_GLOBAL"
            for ledger in population.attempt_ledgers for record in ledger
        ),
    }
    checks["pass"] = bool(all(value for key, value in checks.items() if key.endswith("_pass")))
    return checks


def public_population_record(population: PopulationConstruction) -> dict:
    rarity = population_rarity_sidecars(population)
    return {
        "slots": [asdict(slot) for slot in population.slots],
        "groups": [public_group_record(group) for group in population.groups],
        "attempt_ledgers": [
            [asdict(record) for record in ledger] for ledger in population.attempt_ledgers
        ],
        "rarity_sidecars": rarity,
    }


def population_rarity_sidecars(population: PopulationConstruction) -> dict[str, dict]:
    """Empirical AST-atom surprisal, computed separately inside each population.

    Rarity is `-log(count(atom)/total_atom_occurrences)` with natural logarithm.
    It is a forbidden-fit shortcut sidecar and never enters PTNI coordinates.
    """
    result = {}
    for population_id in sorted({group.population_id for group in population.groups}):
        groups = [group for group in population.groups if group.population_id == population_id]
        numeric_counts, entity_counts = Counter(), Counter()
        group_atoms = {}
        for group in groups:
            numeric, entities = [], []
            for task in (group.task_a, group.task_b):
                for key, value, _ in task.records:
                    # Count each canonical AST atom once, before rendering.  Keys
                    # can themselves be visible numeric universe identifiers in
                    # finite tasks, so inspecting only record values is incomplete.
                    numeric.extend(
                        re.findall(r"[-+]?\d+(?:/[1-9]\d*)?", f"{key}\0{value}")
                    )
                    if task.domain == "relational":
                        entities.append(key)
            numeric_counts.update(numeric)
            entity_counts.update(entities)
            group_atoms[group.group_id] = (numeric, entities)
        numeric_total = sum(numeric_counts.values())
        entity_total = sum(entity_counts.values())
        for group in groups:
            numeric, entities = group_atoms[group.group_id]
            numeric_values = [
                -math.log(numeric_counts[atom] / numeric_total) for atom in numeric
            ] if numeric_total else []
            entity_values = [
                -math.log(entity_counts[atom] / entity_total) for atom in entities
            ] if entity_total else []
            result[group.group_id] = {
                "population_id": population_id,
                "numeric_rarity_mean": float(sum(numeric_values) / len(numeric_values))
                if numeric_values else 0.0,
                "numeric_rarity_max": float(max(numeric_values)) if numeric_values else 0.0,
                "entity_rarity_mean": float(sum(entity_values) / len(entity_values))
                if entity_values else 0.0,
                "entity_rarity_max": float(max(entity_values)) if entity_values else 0.0,
                "numeric_atom_count": len(numeric),
                "entity_atom_count": len(entities),
            }
    return result


__all__ = [
    "LLAMA_POPULATION_ID", "LLAMA_SEED_NAMESPACE", "PopulationConstruction",
    "PopulationSlot", "QWEN_POPULATION_ID", "QWEN_SEED_NAMESPACE",
    "audit_population", "build_population", "frozen_combined_slots",
    "frozen_population_slots", "population_rarity_sidecars", "public_population_record",
]
