"""Mechanical, target-firewalled construction primitives for A6 PTNI-IU.

This module deliberately contains no model loading, telemetry extraction, or
benchmark labels.  It builds reciprocal task pairs, renders them through four
reversible prompt grammars, constructs deterministic response ASTs, and checks
the complete 2x2 truth matrix with two independent exact evaluators.

The full 900+900 construction is performed only by a later append-only S0
runner.  The functions here are development/source-boundary code and accept
token-count callbacks so tokenizer equality can be checked without importing a
model runtime into the semantic generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import re
import sqlite3
from typing import Callable, Mapping, Sequence
import unicodedata


DOMAINS = ("arithmetic", "relational", "finite_logic")
MUTATIONS = ("value_leaf", "relation_operator", "constraint_condition")
RENDERINGS = ("canonical", "paraphrase", "layout", "notation")
RESPONSE_GRAMMARS = ("short", "certificate")
ANSWER_WRAPPERS = (
    "{answer}",
    "Answer: {answer}",
    "The answer is {answer}.",
    "The final answer is {answer}.",
)
TOKENIZER_FAMILIES = ("llama", "qwen")
NATURAL_ANSWER_KINDS = ("integer", "rational", "finite_set", "entity", "relation")

_ENTITY_NAMES = ("Ada", "Bela", "Cora", "Dion")


@dataclass(frozen=True)
class TaskAST:
    """One exact finite task.

    ``records`` has a common three-field schema so serialization, hashing, and
    complexity checks are domain-independent:

    - arithmetic: ``(name, integer-string, "number")``;
    - relational: ``(entity, integer-string, group)``;
    - finite logic: ``(universe-element, membership-code, "membership")`` where
      bit 0 denotes set A and bit 1 denotes set B.
    """

    domain: str
    mutation_family: str
    records: tuple[tuple[str, str, str], ...]
    operator: str
    constraint: str
    output_kind: str = "exact_integer"
    unit: str = ""

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unsupported A6 domain: {self.domain}")
        if self.mutation_family not in MUTATIONS:
            raise ValueError(f"unsupported A6 mutation: {self.mutation_family}")
        if not self.records or any(len(row) != 3 for row in self.records):
            raise ValueError("task records must be a nonempty tuple of triples")
        forbidden = re.compile(r"[,;~|\[\]\r\n]")
        if any(
            not isinstance(field, str) or not field
            or forbidden.search(field) is not None
            for row in self.records for field in row
        ):
            raise ValueError("task record fields contain an empty or reserved delimiter")
        keys = tuple(row[0] for row in self.records)
        if len(set(keys)) != len(keys):
            raise ValueError("task record keys must be unique")
        if self.domain == "arithmetic":
            if any(
                marker != "number"
                or re.fullmatch(r"v\d+", name) is None
                or canonical_answer(parse_answer_atom(value)) != value
                for name, value, marker in self.records
            ):
                raise ValueError("arithmetic records violate the canonical schema")
        elif self.domain == "relational":
            if any(
                re.fullmatch(r"[A-Z][A-Za-z0-9_]*", entity) is None
                or re.fullmatch(r"-?(?:0|[1-9]\d*)", value) is None
                or group not in ("red", "blue")
                for entity, value, group in self.records
            ):
                raise ValueError("relational records violate the canonical schema")
        elif any(
            re.fullmatch(r"[1-9]\d*", element) is None
            or code not in ("0", "1", "2", "3")
            or marker != "membership"
            for element, code, marker in self.records
        ):
            raise ValueError("finite-logic records violate the canonical schema")
        operators = {
            "arithmetic": {"sum", "product", "maximum"},
            "relational": {"sum", "maximum", "count", "lookup_first", "lookup_last"},
            "finite_logic": {"union", "intersection", "difference"},
        }
        constraints = {
            "arithmetic": {"all", "first_three", "last_three"},
            "relational": {"red", "blue"},
            "finite_logic": {"all", "even", "odd"},
        }
        if self.operator not in operators[self.domain]:
            raise ValueError("task operator is outside the registered domain grammar")
        if self.constraint not in constraints[self.domain]:
            raise ValueError("task constraint is outside the registered domain grammar")
        if self.output_kind not in ("exact_integer", "exact_rational"):
            raise ValueError("A6 tasks require exact_integer or exact_rational")
        if self.domain != "arithmetic" and self.output_kind != "exact_integer":
            raise ValueError("only arithmetic tasks may use exact_rational")
        if (
            self.domain == "arithmetic" and self.output_kind == "exact_integer"
            and any("/" in value for _, value, _ in self.records)
        ):
            raise ValueError("exact-integer arithmetic cannot contain rational records")
        if self.unit:
            raise ValueError("reciprocal construction tasks do not admit unit-bearing outputs")


@dataclass(frozen=True)
class ResponseAST:
    grammar: str
    answer: str
    domain: str
    source_facts: tuple[str, ...] = ()
    selected_values: tuple[str, ...] = ()
    operation: str = ""

    def __post_init__(self) -> None:
        if self.grammar not in RESPONSE_GRAMMARS:
            raise ValueError(f"unsupported response grammar: {self.grammar}")
        if self.grammar == "short" and (
            self.source_facts or self.selected_values or self.operation
        ):
            raise ValueError("short response may contain only its answer assertion")


@dataclass(frozen=True)
class ReciprocalGroup:
    group_id: str
    population_id: str
    outer_fold: int
    source_record_id: str
    donor_id: str
    template_id: str
    seed: int
    domain: str
    mutation_family: str
    response_grammar: str
    task_a: TaskAST
    task_b: TaskAST
    response_a: ResponseAST
    response_b: ResponseAST
    prompts_a: tuple[str, ...]
    prompts_b: tuple[str, ...]
    response_text_a: str
    response_text_b: str
    ast_sha256_a: str
    ast_sha256_b: str
    response_sha256_a: str
    response_sha256_b: str
    attempt_index: int
    complete_prompt_ids_a: tuple[str, ...]
    complete_prompt_ids_b: tuple[str, ...]
    prompt_token_counts_a: tuple[tuple[str, tuple[int, ...]], ...]
    prompt_token_counts_b: tuple[tuple[str, tuple[int, ...]], ...]
    response_token_ids_a: tuple[tuple[str, tuple[int, ...]], ...]
    response_token_ids_b: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class ConstructionAttempt:
    attempt_index: int
    attempt_seed: int
    status: str
    reason: str
    ast_sha256_a: str | None
    ast_sha256_b: str | None
    prompt_token_counts_a: tuple[tuple[str, tuple[int, ...]], ...] = ()
    prompt_token_counts_b: tuple[tuple[str, tuple[int, ...]], ...] = ()
    response_token_counts_a: tuple[tuple[str, int], ...] = ()
    response_token_counts_b: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class ReciprocalConstruction:
    group: ReciprocalGroup
    attempts: tuple[ConstructionAttempt, ...]


@dataclass(frozen=True)
class TaskPairCandidate:
    """One response-free reciprocal task-pair attempt.

    This type is the S0a natural-prompt firewall: it contains only typed task
    worlds and their reversible prompt renderings.  It never constructs a
    response AST, response text, answer sidecar, or tokenizer evidence.
    """

    attempt_seed: int
    status: str
    reason: str
    task_a: TaskAST | None = None
    task_b: TaskAST | None = None
    prompts_a: tuple[str, ...] = ()
    prompts_b: tuple[str, ...] = ()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def task_sha256(task: TaskAST) -> str:
    return hashlib.sha256(_canonical_json(asdict(task)).encode("utf-8")).hexdigest()


def semantic_task_sha256(task: TaskAST) -> str:
    """Hash task meaning while excluding intervention-family bookkeeping."""
    payload = {
        "domain": task.domain,
        "records": task.records,
        "operator": task.operator,
        "constraint": task.constraint,
        "output_kind": task.output_kind,
        "unit": task.unit,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _numbers(task: TaskAST) -> tuple[int | Fraction, ...]:
    return tuple(parse_answer_atom(row[1]) for row in task.records)


def _arithmetic_selected(task: TaskAST) -> tuple[int | Fraction, ...]:
    values = _numbers(task)
    if task.constraint == "all":
        return values
    if task.constraint == "first_three":
        return values[:3]
    if task.constraint == "last_three":
        return values[-3:]
    raise ValueError(f"invalid arithmetic constraint: {task.constraint}")


def _logic_sets(task: TaskAST) -> tuple[set[int], set[int]]:
    left, right = set(), set()
    for name, code_text, marker in task.records:
        if marker != "membership":
            raise ValueError("invalid finite-logic record marker")
        value, code = int(name), int(code_text)
        if code & 1:
            left.add(value)
        if code & 2:
            right.add(value)
    return left, right


def evaluate_generator(task: TaskAST) -> int | Fraction:
    """Generator-side exact evaluator (direct recursive/domain operations)."""
    if task.domain == "arithmetic":
        selected = _arithmetic_selected(task)
        if task.operator == "sum":
            return sum(selected)
        if task.operator == "product":
            result = 1
            for value in selected:
                result *= value
            return result
        if task.operator == "maximum":
            return max(selected)
        raise ValueError(f"invalid arithmetic operator: {task.operator}")

    if task.domain == "relational":
        selected = [int(value) for _, value, group in task.records
                    if group == task.constraint]
        if not selected:
            raise ValueError("relational task selected no rows")
        if task.operator == "lookup_first":
            return selected[0]
        if task.operator == "lookup_last":
            return selected[-1]
        if task.operator == "sum":
            return sum(selected)
        if task.operator == "maximum":
            return max(selected)
        if task.operator == "count":
            return len(selected)
        raise ValueError(f"invalid relational operator: {task.operator}")

    left, right = _logic_sets(task)
    if task.operator == "union":
        selected = left | right
    elif task.operator == "intersection":
        selected = left & right
    elif task.operator == "difference":
        selected = left - right
    else:
        raise ValueError(f"invalid finite-logic operator: {task.operator}")
    if task.constraint == "even":
        selected = {value for value in selected if value % 2 == 0}
    elif task.constraint == "odd":
        selected = {value for value in selected if value % 2 == 1}
    elif task.constraint != "all":
        raise ValueError(f"invalid finite-logic constraint: {task.constraint}")
    return len(selected)


def evaluate_verifier(task: TaskAST) -> int:
    """Independent exact verifier-side evaluator.

    Arithmetic uses a small stack machine, relational tasks use an isolated
    in-memory SQL engine, and finite logic exhaustively evaluates membership
    predicates over the declared universe.
    """
    if task.domain == "arithmetic":
        values = list(_numbers(task))
        keep = [True] * len(values)
        if task.constraint == "first_three":
            keep = [index < 3 for index in range(len(values))]
        elif task.constraint == "last_three":
            keep = [index >= len(values) - 3 for index in range(len(values))]
        elif task.constraint != "all":
            raise ValueError("verifier rejected arithmetic constraint")
        stack = [value for value, include in zip(values, keep) if include]
        if task.operator == "sum":
            accumulator = 0
            while stack:
                accumulator = accumulator + stack.pop()
            return accumulator
        if task.operator == "product":
            accumulator = 1
            while stack:
                accumulator = accumulator * stack.pop()
            return accumulator
        if task.operator == "maximum":
            accumulator = stack.pop()
            while stack:
                candidate = stack.pop()
                accumulator = candidate if candidate > accumulator else accumulator
            return accumulator
        raise ValueError("verifier rejected arithmetic operator")

    if task.domain == "relational":
        if task.operator in ("lookup_first", "lookup_last"):
            ordering = "ASC" if task.operator == "lookup_first" else "DESC"
            query = f"SELECT value FROM facts WHERE grp=? ORDER BY rowid {ordering} LIMIT 1"
        else:
            aggregate = {"sum": "SUM", "maximum": "MAX", "count": "COUNT"}.get(
                task.operator
            )
            if aggregate is None:
                raise ValueError("verifier rejected relational operator")
            query = f"SELECT {aggregate}(value) FROM facts WHERE grp=?"  # nosec B608
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE TABLE facts(entity TEXT, value INTEGER, grp TEXT)")
            connection.executemany(
                "INSERT INTO facts(entity,value,grp) VALUES(?,?,?)",
                [(entity, int(value), group) for entity, value, group in task.records],
            )
            row = connection.execute(
                query, (task.constraint,),  # nosec B608
            ).fetchone()
        finally:
            connection.close()
        if row is None or row[0] is None:
            raise ValueError("verifier relational query returned no answer")
        return int(row[0])

    membership = {int(name): int(code) for name, code, marker in task.records
                  if marker == "membership"}
    if len(membership) != len(task.records):
        raise ValueError("verifier rejected finite-logic records")
    count = 0
    for value in sorted(membership):
        in_left = bool(membership[value] & 1)
        in_right = bool(membership[value] & 2)
        if task.operator == "union":
            chosen = in_left or in_right
        elif task.operator == "intersection":
            chosen = in_left and in_right
        elif task.operator == "difference":
            chosen = in_left and not in_right
        else:
            raise ValueError("verifier rejected finite-logic operator")
        if task.constraint == "even":
            chosen = chosen and value % 2 == 0
        elif task.constraint == "odd":
            chosen = chosen and value % 2 == 1
        elif task.constraint != "all":
            raise ValueError("verifier rejected finite-logic constraint")
        count += int(chosen)
    return count


def _selected_values(task: TaskAST) -> tuple[int | Fraction, ...]:
    if task.domain == "arithmetic":
        return _arithmetic_selected(task)
    if task.domain == "relational":
        selected = tuple(
            int(value) for _, value, group in task.records if group == task.constraint
        )
        if task.operator == "lookup_first":
            return selected[:1]
        if task.operator == "lookup_last":
            return selected[-1:]
        return selected
    left, right = _logic_sets(task)
    if task.operator == "union":
        selected = left | right
    elif task.operator == "intersection":
        selected = left & right
    else:
        selected = left - right
    if task.constraint == "even":
        selected = {value for value in selected if value % 2 == 0}
    elif task.constraint == "odd":
        selected = {value for value in selected if value % 2 == 1}
    return tuple(sorted(selected))


def _source_facts(task: TaskAST) -> tuple[str, ...]:
    # `~` separates typed fields and `|` separates records.  The TaskAST
    # constructor admits neither delimiter, so rational `/` remains unambiguous.
    if task.domain == "arithmetic":
        return tuple("~".join(row[:2]) for row in task.records)
    if task.domain == "finite_logic":
        return tuple("~".join(row[:2]) for row in task.records)
    return tuple("~".join(row) for row in task.records)


def build_response_ast(task: TaskAST, grammar: str) -> ResponseAST:
    answer = canonical_answer(evaluate_generator(task))
    if grammar == "short":
        return ResponseAST("short", answer, task.domain)
    if grammar != "certificate":
        raise ValueError(f"unsupported response grammar: {grammar}")
    operation = task.operator if task.domain != "finite_logic" else f"{task.operator}_then_count"
    return ResponseAST(
        "certificate", answer, task.domain, _source_facts(task),
        tuple(canonical_answer(value) for value in _selected_values(task)), operation,
    )


def render_response(response: ResponseAST) -> str:
    if response.grammar == "short":
        return f"The final answer is {response.answer}."
    facts = "|".join(response.source_facts)
    selected = ",".join(response.selected_values)
    return (
        f"CERT[d={response.domain};f={facts};s={selected};o={response.operation};"
        f"r={response.answer}]. The final answer is {response.answer}."
    )


_CERTIFICATE_RE = re.compile(
    r"^CERT\[d=(?P<domain>[a-z_]+);f=(?P<facts>[^;]+);"
    r"s=(?P<selected>[-+]?\d+(?:/[1-9]\d*)?"
    r"(?:,[-+]?\d+(?:/[1-9]\d*)?)*);o=(?P<operation>[a-z_]+);"
    r"r=(?P<computed>[-+]?\d+(?:/[1-9]\d*)?)\]\. "
    r"The final answer is (?P<answer>[-+]?\d+(?:/[1-9]\d*)?)\.$"
)


def parse_response(text: str, grammar: str) -> ResponseAST:
    if grammar == "short":
        value = parse_closed_answer(text)
        return ResponseAST("short", canonical_answer(value), "unknown")
    match = _CERTIFICATE_RE.fullmatch(text)
    if match is None:
        raise ValueError("certificate response is outside the closed grammar")
    if match.group("computed") != match.group("answer"):
        raise ValueError("certificate computed result and final answer disagree")
    facts = tuple(match.group("facts").split("|"))
    selected = tuple(
        canonical_answer(parse_answer_atom(value))
        for value in match.group("selected").split(",")
    )
    return ResponseAST(
        "certificate", match.group("answer"), match.group("domain"), facts,
        selected, match.group("operation"),
    )


def canonical_answer(value: int | Fraction) -> str:
    if isinstance(value, Fraction) and value.denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    return str(int(value))


def parse_answer_atom(text: str) -> int | Fraction:
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?\d+/[1-9]\d*", text):
        numerator, denominator = text.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    raise ValueError("answer atom is outside the exact integer/rational grammar")


def parse_closed_answer(text: str) -> int | Fraction:
    """Construction-task numeric parser; not the full S0 natural-answer parser."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    patterns = (
        r"(?P<a>[-+]?\d+(?:/[1-9]\d*)?)",
        r"Answer: (?P<a>[-+]?\d+(?:/[1-9]\d*)?)",
        r"The answer is (?P<a>[-+]?\d+(?:/[1-9]\d*)?)\.",
        r"The final answer is (?P<a>[-+]?\d+(?:/[1-9]\d*)?)\.",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, normalized)
        if match is not None:
            return parse_answer_atom(match.group("a"))
    raise ValueError("response is outside the closed natural-answer grammar")


def canonicalize_natural_answer_atom(
    atom: str,
    *,
    kind: str,
    registered_atoms: Sequence[str] = (),
    required_unit: str = "",
) -> str:
    """Canonicalize one complete natural-cohort answer atom.

    Entity and relation vocabularies are task-local, mechanically registered
    strings.  A required unit is part of the task contract rather than inferred
    from the response.  This helper never performs correctness matching.
    """
    if kind not in NATURAL_ANSWER_KINDS:
        raise ValueError("unregistered natural-answer kind")
    normalized = unicodedata.normalize("NFKC", atom).strip()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError("natural answer atom is empty or multiline")
    value_text = normalized
    if required_unit:
        suffix = " " + required_unit
        if not value_text.endswith(suffix):
            raise ValueError("natural answer has a missing or unmatched unit")
        value_text = value_text[:-len(suffix)]
        if not value_text:
            raise ValueError("natural answer unit has no value")
    elif re.search(r"\s(?:kg|m|s|items)$", value_text):
        raise ValueError("natural answer contains an unregistered unit")

    registered = tuple(
        unicodedata.normalize("NFKC", value).strip() for value in registered_atoms
    )
    if len(set(registered)) != len(registered) or any(not value for value in registered):
        raise ValueError("registered natural atoms must be unique and nonempty")
    if kind == "integer":
        parsed = parse_answer_atom(value_text)
        canonical = str(parsed)
        if "/" in value_text or not isinstance(parsed, int) or value_text != canonical:
            raise ValueError("natural integer answer is not an integer")
    elif kind == "rational":
        parsed = parse_answer_atom(value_text)
        canonical = canonical_answer(parsed)
        if value_text != canonical:
            raise ValueError("natural rational answer is not in canonical form")
    elif kind in ("entity", "relation"):
        if value_text not in registered:
            raise ValueError(f"natural {kind} answer is outside its registered vocabulary")
        canonical = value_text
    else:
        if not registered:
            raise ValueError("natural finite-set answers require a registered universe")
        match = re.fullmatch(r"\{(?P<members>[^{}]*)\}", value_text)
        if match is None:
            raise ValueError("natural finite-set answer is outside canonical braces")
        members = () if not match.group("members") else tuple(
            match.group("members").split(",")
        )
        if any(not member or member != member.strip() for member in members):
            raise ValueError("natural finite-set members use noncanonical spacing")
        allowed = set(registered)
        canonical_members = []
        for member in members:
            if member in allowed:
                canonical_members.append(member)
            else:
                raise ValueError("natural finite-set member is unregistered")
        if len(set(canonical_members)) != len(members):
            raise ValueError("natural finite-set answer contains duplicate members")
        canonical_members = sorted(canonical_members, key=registered.index)
        canonical = "{" + ",".join(canonical_members) + "}"
        if value_text != canonical:
            raise ValueError("natural finite-set answer is not in canonical order")
    return canonical + ((" " + required_unit) if required_unit else "")


def parse_closed_natural_answer(
    text: str,
    *,
    kind: str,
    registered_atoms: Sequence[str] = (),
    required_unit: str = "",
) -> str:
    """Parse exactly one of the four frozen complete-string wrappers."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    if normalized.startswith("The final answer is "):
        if not normalized.endswith("."):
            raise ValueError("response is outside the closed natural-answer grammar")
        candidate = normalized[len("The final answer is "):-1]
    elif normalized.startswith("The answer is "):
        if not normalized.endswith("."):
            raise ValueError("response is outside the closed natural-answer grammar")
        candidate = normalized[len("The answer is "):-1]
    elif normalized.startswith("Answer: "):
        candidate = normalized[len("Answer: "):]
    else:
        candidate = normalized
    return canonicalize_natural_answer_atom(
        candidate, kind=kind, registered_atoms=registered_atoms,
        required_unit=required_unit,
    )


def contains_answer_atom(text: str, canonical_atom: str) -> bool:
    """Return whether a canonical atom occurs at exact alphanumeric boundaries."""
    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_atom = unicodedata.normalize("NFKC", canonical_atom)
    if not normalized_atom:
        raise ValueError("canonical answer atom must be nonempty")
    pattern = rf"(?<![A-Za-z0-9_/]){re.escape(normalized_atom)}(?![A-Za-z0-9_/])"
    return re.search(pattern, normalized_text) is not None


def _parse_fact_rows(
    facts: Sequence[str], domain: str,
) -> tuple[tuple[str, str, str], ...]:
    decoded = tuple(tuple(fact.split("~")) for fact in facts)
    expected_fields = 3 if domain == "relational" else 2
    if any(
        len(row) != expected_fields or any(not value for value in row) for row in decoded
    ):
        raise ValueError("certificate contains a malformed source fact")
    if domain == "arithmetic":
        return tuple((*row, "number") for row in decoded)
    if domain == "finite_logic":
        return tuple((*row, "membership") for row in decoded)
    return decoded  # type: ignore[return-value]


def _verifier_selected_values(task: TaskAST) -> tuple[int | Fraction, ...]:
    """Recompute the selected values without generator selection helpers."""
    if task.domain == "arithmetic":
        values = tuple(parse_answer_atom(row[1]) for row in task.records)
        indices = tuple(range(len(values)))
        if task.constraint == "first_three":
            indices = tuple(index for index in indices if index < 3)
        elif task.constraint == "last_three":
            indices = tuple(index for index in indices if index >= len(values) - 3)
        elif task.constraint != "all":
            raise ValueError("certificate verifier rejected arithmetic selection")
        return tuple(values[index] for index in indices)
    if task.domain == "relational":
        selected = tuple(
            int(value) for _, value, group in task.records if group == task.constraint
        )
        if task.operator == "lookup_first":
            return selected[:1]
        if task.operator == "lookup_last":
            return selected[-1:]
        return selected
    output = []
    for name, code_text, marker in task.records:
        if marker != "membership":
            raise ValueError("certificate verifier rejected logic record")
        value, code = int(name), int(code_text)
        left, right = bool(code & 1), bool(code & 2)
        if task.operator == "union":
            selected = left or right
        elif task.operator == "intersection":
            selected = left and right
        elif task.operator == "difference":
            selected = left and not right
        else:
            raise ValueError("certificate verifier rejected logic operation")
        if task.constraint == "even":
            selected = selected and value % 2 == 0
        elif task.constraint == "odd":
            selected = selected and value % 2 == 1
        elif task.constraint != "all":
            raise ValueError("certificate verifier rejected logic constraint")
        if selected:
            output.append(value)
    return tuple(output)


def _certificate_result(task: TaskAST, selected: Sequence[int | Fraction]) -> int | Fraction:
    if not selected:
        raise ValueError("certificate selected no values")
    if task.domain == "finite_logic":
        return len(selected)
    if task.operator == "sum":
        return sum(selected)
    if task.operator == "product":
        result: int | Fraction = 1
        for value in selected:
            result *= value
        return result
    if task.operator == "maximum":
        return max(selected)
    if task.operator == "count":
        return len(selected)
    if task.operator in ("lookup_first", "lookup_last"):
        if len(selected) != 1:
            raise ValueError("certificate lookup requires exactly one selected value")
        return selected[0]
    raise ValueError("certificate verifier rejected operation")


def verify_response(task: TaskAST, response: ResponseAST) -> bool:
    """Verify the whole parsed response through verifier-only operations."""
    try:
        expected_answer = canonical_answer(evaluate_verifier(task))
        if response.answer != expected_answer:
            return False
        if response.grammar == "short":
            return True
        if response.domain != task.domain:
            return False
        if _parse_fact_rows(response.source_facts, response.domain) != task.records:
            return False
        selected = tuple(parse_answer_atom(value) for value in response.selected_values)
        if selected != _verifier_selected_values(task):
            return False
        expected_operation = (
            task.operator if task.domain != "finite_logic" else f"{task.operator}_then_count"
        )
        if response.operation != expected_operation:
            return False
        return canonical_answer(_certificate_result(task, selected)) == response.answer
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _records_payload(task: TaskAST) -> str:
    return ";".join(",".join(row) for row in task.records)


def _domain_schema(domain: str) -> str:
    if domain == "arithmetic":
        return (
            "Each record is name,value,number. Select all, first_three, or "
            "last_three in listed order; sum, product, and maximum use exact arithmetic."
        )
    if domain == "relational":
        return (
            "Each record is entity,integer,group. Select rows whose group equals "
            "the rule, then return their sum, maximum, count, first listed value "
            "(lookup_first), or last listed value (lookup_last)."
        )
    if domain == "finite_logic":
        return (
            "Each record is element,membership-code,membership: 0 means neither, "
            "1 A only, 2 B only, and 3 both. Apply union, intersection, or A-minus-B, "
            "then keep all, even, or odd elements and return the exact cardinality."
        )
    raise ValueError("unsupported domain schema")


def _template_variant(template_id: str, rendering: str) -> int:
    if not template_id:
        raise ValueError("template_id must be nonempty")
    return _stable_seed("a6-template", template_id, rendering) % 2


def render_task(task: TaskAST, rendering: str, template_id: str) -> str:
    if rendering not in RENDERINGS:
        raise ValueError(f"unsupported prompt rendering: {rendering}")
    records = _records_payload(task)
    output = "one exact integer" if task.output_kind == "exact_integer" else "one exact rational"
    variant = _template_variant(template_id, rendering)
    finish = lambda body: body + "\nSchema: " + _domain_schema(task.domain)  # noqa: E731
    if rendering == "canonical":
        if variant == 0:
            return finish(
                f"Task domain: {task.domain}.\nData records: {records}.\n"
                f"Selection rule: {task.constraint}.\nOperation: {task.operator}.\n"
                f"Return {output}."
            )
        return finish(
            f"Domain: {task.domain}.\nRecords: {records}.\n"
            f"Condition: {task.constraint}.\nOperator: {task.operator}.\n"
            f"Respond with {output}."
        )
    if rendering == "paraphrase":
        if variant == 0:
            return finish(
                f"Within the {task.domain} task, use the records {records}. "
                f"Keep the entries specified by {task.constraint}, apply "
                f"{task.operator}, and reply with {output}."
            )
        return finish(
            f"For this {task.domain} problem, consider {records}. Select according "
            f"to {task.constraint}, compute {task.operator}, and provide {output}."
        )
    if rendering == "layout":
        labels = ("DOMAIN", "RECORDS", "FILTER", "AGGREGATE", "OUTPUT")
        if variant:
            labels = ("TASK", "DATA", "SELECT", "COMPUTE", "RESPONSE")
        return finish(
            f"{labels[0]} [{task.domain}]\n{labels[1]} [{records}]\n"
            f"{labels[2]} [{task.constraint}]\n{labels[3]} [{task.operator}]\n"
            f"{labels[4]} [{output}]"
        )
    if variant == 0:
        return finish(
            f"Solve ( domain = {task.domain} ; records = {records} ; "
            f"select = {task.constraint} ; op = {task.operator} ; "
            f"output = {task.output_kind} )."
        )
    return finish(
        f"Evaluate {{ domain : {task.domain} | records : {records} | "
        f"condition : {task.constraint} | operator : {task.operator} | "
        f"output : {task.output_kind} }}."
    )


def _parse_records(payload: str) -> tuple[tuple[str, str, str], ...]:
    rows = tuple(tuple(part.split(",")) for part in payload.split(";"))
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("rendered task has malformed records")
    return rows  # type: ignore[return-value]


def parse_task(
    text: str, rendering: str, mutation_family: str, template_id: str,
) -> TaskAST:
    try:
        text, schema = text.rsplit("\nSchema: ", 1)
    except ValueError as exc:
        raise ValueError("prompt is missing its explicit domain schema") from exc
    variant = _template_variant(template_id, rendering)
    if rendering == "canonical":
        pattern = re.compile((
            r"^Task domain: (?P<domain>[a-z_]+)\.\nData records: (?P<records>.+)\.\n"
            r"Selection rule: (?P<constraint>[a-z_]+)\.\nOperation: (?P<operator>[a-z_]+)\.\n"
            r"Return one (?P<output>exact (?:integer|rational))\.$"
        ) if variant == 0 else (
            r"^Domain: (?P<domain>[a-z_]+)\.\nRecords: (?P<records>.+)\.\n"
            r"Condition: (?P<constraint>[a-z_]+)\.\nOperator: (?P<operator>[a-z_]+)\.\n"
            r"Respond with one (?P<output>exact (?:integer|rational))\.$"
        ))
    elif rendering == "paraphrase":
        pattern = re.compile((
            r"^Within the (?P<domain>[a-z_]+) task, use the records (?P<records>.+)\. "
            r"Keep the entries specified by (?P<constraint>[a-z_]+), apply "
            r"(?P<operator>[a-z_]+), and reply with one (?P<output>exact (?:integer|rational))\.$"
        ) if variant == 0 else (
            r"^For this (?P<domain>[a-z_]+) problem, consider (?P<records>.+)\. "
            r"Select according to (?P<constraint>[a-z_]+), compute (?P<operator>[a-z_]+), "
            r"and provide one (?P<output>exact (?:integer|rational))\.$"
        ))
    elif rendering == "layout":
        labels = ("DOMAIN", "RECORDS", "FILTER", "AGGREGATE", "OUTPUT")
        if variant:
            labels = ("TASK", "DATA", "SELECT", "COMPUTE", "RESPONSE")
        pattern = re.compile(
            rf"^{labels[0]} \[(?P<domain>[a-z_]+)\]\n"
            rf"{labels[1]} \[(?P<records>.+)\]\n"
            rf"{labels[2]} \[(?P<constraint>[a-z_]+)\]\n"
            rf"{labels[3]} \[(?P<operator>[a-z_]+)\]\n"
            rf"{labels[4]} \[one (?P<output>exact (?:integer|rational))\]$"
        )
    elif rendering == "notation":
        pattern = re.compile((
            r"^Solve \( domain = (?P<domain>[a-z_]+) ; records = (?P<records>.+) ; "
            r"select = (?P<constraint>[a-z_]+) ; op = (?P<operator>[a-z_]+) ; "
            r"output = (?P<output>exact_(?:integer|rational)) \)\.$"
        ) if variant == 0 else (
            r"^Evaluate \{ domain : (?P<domain>[a-z_]+) \| records : (?P<records>.+) \| "
            r"condition : (?P<constraint>[a-z_]+) \| operator : (?P<operator>[a-z_]+) \| "
            r"output : (?P<output>exact_(?:integer|rational)) \}\.$"
        ))
    else:
        raise ValueError(f"unsupported prompt rendering: {rendering}")
    match = pattern.fullmatch(text)
    if match is None:
        raise ValueError("prompt is outside its frozen rendering grammar")
    output_text = match.group("output").replace(" ", "_")
    task = TaskAST(
        match.group("domain"), mutation_family, _parse_records(match.group("records")),
        match.group("operator"), match.group("constraint"), output_text,
    )
    if schema != _domain_schema(task.domain):
        raise ValueError("prompt domain schema does not match its task domain")
    return task


def task_complexity(task: TaskAST) -> tuple[int, int, tuple[str, ...]]:
    """Frozen AST shape summary used for A/B matching."""
    node_types = (
        "task", "domain", *("record" for _ in task.records),
        "operator", "constraint", "output_kind",
    )
    return 5 + len(task.records), 3, tuple(node_types)


def changed_fields(left: TaskAST, right: TaskAST) -> tuple[str, ...]:
    fields = []
    for name in ("domain", "records", "operator", "constraint", "output_kind", "unit"):
        if getattr(left, name) != getattr(right, name):
            fields.append(name)
    return tuple(fields)


def changed_node_details(left: TaskAST, right: TaskAST) -> dict:
    """Return the exact typed AST edit rather than the enclosing dataclass field."""
    changed = changed_fields(left, right)
    if changed == ("records",):
        identities = []
        for record_index, (left_row, right_row) in enumerate(zip(left.records, right.records)):
            for field_index, (left_value, right_value) in enumerate(zip(left_row, right_row)):
                if left_value != right_value:
                    identities.append({
                        "node_type": ("record_key", "record_value", "record_kind")[field_index],
                        "record_index": record_index,
                        "from": left_value,
                        "to": right_value,
                    })
    else:
        identities = [
            {
                "node_type": field,
                "record_index": None,
                "from": getattr(left, field),
                "to": getattr(right, field),
            }
            for field in changed
        ]
    return {"changed_node_count": len(identities), "changed_nodes": identities}


def _pair_candidate(domain: str, mutation: str, seed: int) -> tuple[TaskAST, TaskAST]:
    # All values are derived by hash, never Python's process-randomized hash().
    offset = _stable_seed("a6-pair", domain, mutation, seed)
    values = tuple(2 + ((offset >> (5 * index)) % 7) for index in range(8))

    if domain == "arithmetic":
        rational = bool(offset & (1 << 55))
        number_texts = tuple(
            canonical_answer(Fraction(values[index], 2)) if rational else str(values[index])
            for index in range(4)
        )
        output_kind = "exact_rational" if rational else "exact_integer"
        base_records = tuple(
            (f"v{index}", number_texts[index], "number") for index in range(4)
        )
        if mutation == "value_leaf":
            changed = list(base_records)
            old_value = parse_answer_atom(changed[2][1])
            changed[2] = (
                changed[2][0], canonical_answer(old_value + 1), changed[2][2]
            )
            left = TaskAST(domain, mutation, base_records, "sum", "all", output_kind)
            right = TaskAST(
                domain, mutation, tuple(changed), "sum", "all", output_kind
            )
        elif mutation == "relation_operator":
            left = TaskAST(domain, mutation, base_records, "sum", "all", output_kind)
            right = TaskAST(domain, mutation, base_records, "product", "all", output_kind)
        else:
            left = TaskAST(
                domain, mutation, base_records, "sum", "first_three", output_kind
            )
            right = TaskAST(
                domain, mutation, base_records, "sum", "last_three", output_kind
            )

    elif domain == "relational":
        groups = ("red", "red", "blue", "blue")
        base_records = tuple(
            (_ENTITY_NAMES[index], str(values[index]), groups[index]) for index in range(4)
        )
        lookup = bool(offset & (1 << 54))
        if mutation == "value_leaf":
            changed = list(base_records)
            changed_index = 0 if lookup else 1
            changed[changed_index] = (
                changed[changed_index][0], str(int(changed[changed_index][1]) + 1),
                changed[changed_index][2],
            )
            operator = "lookup_first" if lookup else "sum"
            left = TaskAST(domain, mutation, base_records, operator, "red")
            right = TaskAST(domain, mutation, tuple(changed), operator, "red")
        elif mutation == "relation_operator":
            left_operator, right_operator = (
                ("lookup_first", "lookup_last") if lookup else ("sum", "maximum")
            )
            left = TaskAST(domain, mutation, base_records, left_operator, "red")
            right = TaskAST(domain, mutation, base_records, right_operator, "red")
        else:
            operator = "lookup_first" if lookup else "sum"
            left = TaskAST(domain, mutation, base_records, operator, "red")
            right = TaskAST(domain, mutation, base_records, operator, "blue")

    else:
        # The full 4^8 code space keeps the 200 required Qwen/Llama groups
        # constructible. Equal-answer pairs are rejected by the attempt schedule.
        codes = tuple((offset >> (3 * index)) % 4 for index in range(1, 9))
        base_records = tuple((str(index + 1), str(codes[index]), "membership")
                             for index in range(8))
        if mutation == "value_leaf":
            changed = list(base_records)
            old = int(changed[3][1])
            changed[3] = (changed[3][0], str(0 if old else 3), changed[3][2])
            left = TaskAST(domain, mutation, base_records, "union", "all")
            right = TaskAST(domain, mutation, tuple(changed), "union", "all")
        elif mutation == "relation_operator":
            left = TaskAST(domain, mutation, base_records, "union", "all")
            right = TaskAST(domain, mutation, base_records, "intersection", "all")
        else:
            left = TaskAST(domain, mutation, base_records, "union", "even")
            right = TaskAST(domain, mutation, base_records, "union", "odd")
    return left, right


def _pair_is_legal(left: TaskAST, right: TaskAST, mutation: str) -> bool:
    if task_complexity(left) != task_complexity(right):
        return False
    if evaluate_generator(left) != evaluate_verifier(left):
        return False
    if evaluate_generator(right) != evaluate_verifier(right):
        return False
    if evaluate_generator(left) == evaluate_generator(right):
        return False
    expected = {
        "value_leaf": ("records",),
        "relation_operator": ("operator",),
        "constraint_condition": ("constraint",),
    }[mutation]
    return changed_fields(left, right) == expected


def construct_task_pair_from_seed(
    *, attempt_seed: int, domain: str, mutation_family: str, template_id: str,
) -> TaskPairCandidate:
    """Build one response-free task pair from an already frozen attempt seed.

    The caller owns the seed namespace.  This matters for A6 natural prompts,
    whose attempt-seed bytes are fixed independently from the older reciprocal
    response constructor.  All semantic and reversible-rendering checks are
    completed here; response rendering and tokenization are structurally
    impossible through this API.
    """
    if not isinstance(attempt_seed, int) or attempt_seed < 0:
        raise ValueError("attempt_seed must be a nonnegative integer")
    if domain not in DOMAINS or mutation_family not in MUTATIONS:
        raise ValueError("invalid A6 domain/mutation")
    if not isinstance(template_id, str) or not template_id:
        raise ValueError("template_id must be a nonempty string")
    left, right = _pair_candidate(domain, mutation_family, attempt_seed)
    if not _pair_is_legal(left, right, mutation_family):
        return TaskPairCandidate(
            attempt_seed, "REJECTED", "semantic_pair_invariant", left, right,
        )
    if not _selected_values(left) or not _selected_values(right):
        return TaskPairCandidate(
            attempt_seed, "REJECTED", "empty_certificate_selection", left, right,
        )
    prompts_left = tuple(
        render_task(left, rendering, template_id) for rendering in RENDERINGS
    )
    prompts_right = tuple(
        render_task(right, rendering, template_id) for rendering in RENDERINGS
    )
    if any(
        parse_task(text, rendering, mutation_family, template_id) != left
        for text, rendering in zip(prompts_left, RENDERINGS)
    ) or any(
        parse_task(text, rendering, mutation_family, template_id) != right
        for text, rendering in zip(prompts_right, RENDERINGS)
    ):
        return TaskPairCandidate(
            attempt_seed, "REJECTED", "prompt_roundtrip", left, right,
            prompts_left, prompts_right,
        )
    return TaskPairCandidate(
        attempt_seed, "ACCEPTED", "accepted", left, right,
        prompts_left, prompts_right,
    )


def _validate_construction_inputs(
    *, population_id: str, outer_fold: int, source_record_id: str,
    donor_id: str, template_id: str,
    tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
) -> None:
    for name, value in (
        ("population_id", population_id), ("source_record_id", source_record_id),
        ("donor_id", donor_id), ("template_id", template_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if not isinstance(outer_fold, int) or not 0 <= outer_fold < 5:
        raise ValueError("outer_fold must be an integer in [0,4]")
    if tuple(sorted(tokenizers)) != TOKENIZER_FAMILIES:
        raise ValueError(
            f"tokenizers must contain exactly the frozen families {TOKENIZER_FAMILIES}"
        )


def _token_ids(
    texts: Sequence[str], tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    output = []
    for name in TOKENIZER_FAMILIES:
        encoded = []
        for text in texts:
            raw = tokenizers[name](text)
            if isinstance(raw, (str, bytes, int)):
                raise TypeError("tokenizer callback must return a token-ID sequence")
            ids = tuple(int(token_id) for token_id in raw)
            if not ids or any(token_id < 0 for token_id in ids):
                raise ValueError("tokenizer callback returned empty or invalid token IDs")
            encoded.append(ids)
        output.append((name, tuple(encoded)))
    return tuple(output)


def _token_counts(
    evidence: Sequence[tuple[str, Sequence[Sequence[int]]]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(len(ids) for ids in sequences)) for name, sequences in evidence)


def _attempt_record(
    *, attempt: int, attempt_seed: int, status: str, reason: str,
    left: TaskAST | None = None, right: TaskAST | None = None,
    left_counts: tuple[tuple[str, tuple[int, ...]], ...] = (),
    right_counts: tuple[tuple[str, tuple[int, ...]], ...] = (),
    response_ids_left: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = (),
    response_ids_right: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = (),
) -> ConstructionAttempt:
    return ConstructionAttempt(
        int(attempt), int(attempt_seed), status, reason,
        None if left is None else task_sha256(left),
        None if right is None else task_sha256(right),
        left_counts, right_counts,
        tuple((name, len(sequences[0])) for name, sequences in response_ids_left),
        tuple((name, len(sequences[0])) for name, sequences in response_ids_right),
    )


def construct_reciprocal_attempt(
    *, seed: int, attempt_index: int, population_id: str, outer_fold: int,
    source_record_id: str, donor_id: str, template_id: str, domain: str,
    mutation_family: str, response_grammar: str,
    tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
) -> tuple[ConstructionAttempt, ReciprocalGroup | None]:
    """Construct exactly one prehashed attempt; never advances the schedule."""
    if domain not in DOMAINS or mutation_family not in MUTATIONS:
        raise ValueError("invalid A6 domain/mutation")
    if response_grammar not in RESPONSE_GRAMMARS:
        raise ValueError("invalid A6 response grammar")
    if not isinstance(attempt_index, int) or attempt_index < 0:
        raise ValueError("attempt_index must be a nonnegative integer")
    tokenizers = dict(tokenizers)
    _validate_construction_inputs(
        population_id=population_id, outer_fold=outer_fold,
        source_record_id=source_record_id, donor_id=donor_id,
        template_id=template_id, tokenizers=tokenizers,
    )
    attempt_seed = _stable_seed(
        "a6-attempt", seed, population_id, outer_fold, source_record_id, donor_id,
        template_id, domain, mutation_family, attempt_index,
    )
    left, right = _pair_candidate(domain, mutation_family, attempt_seed)
    if not _pair_is_legal(left, right, mutation_family):
        return _attempt_record(
            attempt=attempt_index, attempt_seed=attempt_seed, status="REJECTED",
            reason="semantic_pair_invariant", left=left, right=right,
        ), None
    if not _selected_values(left) or not _selected_values(right):
        return _attempt_record(
            attempt=attempt_index, attempt_seed=attempt_seed, status="REJECTED",
            reason="empty_certificate_selection", left=left, right=right,
        ), None
    if _stable_seed("a6-ab", seed, population_id, source_record_id, attempt_index) & 1:
        left, right = right, left
    prompts_left = tuple(render_task(left, rendering, template_id) for rendering in RENDERINGS)
    prompts_right = tuple(render_task(right, rendering, template_id) for rendering in RENDERINGS)
    if any(
        parse_task(text, rendering, mutation_family, template_id) != left
        for text, rendering in zip(prompts_left, RENDERINGS)
    ) or any(
        parse_task(text, rendering, mutation_family, template_id) != right
        for text, rendering in zip(prompts_right, RENDERINGS)
    ):
        return _attempt_record(
            attempt=attempt_index, attempt_seed=attempt_seed, status="REJECTED",
            reason="prompt_roundtrip", left=left, right=right,
        ), None
    prompt_ids_left = _token_ids(prompts_left, tokenizers)
    prompt_ids_right = _token_ids(prompts_right, tokenizers)
    left_counts, right_counts = _token_counts(prompt_ids_left), _token_counts(prompt_ids_right)
    if left_counts != right_counts:
        return _attempt_record(
            attempt=attempt_index, attempt_seed=attempt_seed, status="REJECTED",
            reason="prompt_token_count_mismatch", left=left, right=right,
            left_counts=left_counts, right_counts=right_counts,
        ), None
    response_left = build_response_ast(left, response_grammar)
    response_right = build_response_ast(right, response_grammar)
    text_left, text_right = render_response(response_left), render_response(response_right)
    parsed_left = parse_response(text_left, response_grammar)
    parsed_right = parse_response(text_right, response_grammar)
    if not verify_response(left, parsed_left) or not verify_response(right, parsed_right):
        return _attempt_record(
            attempt=attempt_index, attempt_seed=attempt_seed, status="REJECTED",
            reason="response_verifier", left=left, right=right,
            left_counts=left_counts, right_counts=right_counts,
        ), None
    response_ids_left = _token_ids((text_left,), tokenizers)
    response_ids_right = _token_ids((text_right,), tokenizers)
    if response_grammar == "certificate":
        counts = [
            len(sequences[0])
            for _, sequences in (*response_ids_left, *response_ids_right)
        ]
        if any(count < 40 or count > 80 for count in counts):
            return _attempt_record(
                attempt=attempt_index, attempt_seed=attempt_seed, status="REJECTED",
                reason="certificate_token_band", left=left, right=right,
                left_counts=left_counts, right_counts=right_counts,
                response_ids_left=response_ids_left,
                response_ids_right=response_ids_right,
            ), None
    group_id = hashlib.sha256(
        "\0".join((
            "a6-group", population_id, str(outer_fold), source_record_id, donor_id,
            template_id, str(seed), domain, mutation_family, response_grammar,
        )).encode("utf-8")
    ).hexdigest()
    prompt_id = lambda world, rendering, text: hashlib.sha256(  # noqa: E731
        "\0".join(("a6-prompt", group_id, world, rendering,
                    hashlib.sha256(text.encode("utf-8")).hexdigest())).encode("utf-8")
    ).hexdigest()
    complete_left = tuple(
        prompt_id("A", rendering, text)
        for rendering, text in zip(RENDERINGS, prompts_left)
    )
    complete_right = tuple(
        prompt_id("B", rendering, text)
        for rendering, text in zip(RENDERINGS, prompts_right)
    )
    group = ReciprocalGroup(
        group_id, population_id, outer_fold, source_record_id, donor_id, template_id,
        int(seed), domain, mutation_family, response_grammar, left, right,
        response_left, response_right, prompts_left, prompts_right, text_left, text_right,
        task_sha256(left), task_sha256(right),
        hashlib.sha256(text_left.encode("utf-8")).hexdigest(),
        hashlib.sha256(text_right.encode("utf-8")).hexdigest(), attempt_index,
        complete_left, complete_right, left_counts, right_counts,
        tuple((name, sequences[0]) for name, sequences in response_ids_left),
        tuple((name, sequences[0]) for name, sequences in response_ids_right),
    )
    record = _attempt_record(
        attempt=attempt_index, attempt_seed=attempt_seed, status="ACCEPTED",
        reason="accepted", left=left, right=right, left_counts=left_counts,
        right_counts=right_counts, response_ids_left=response_ids_left,
        response_ids_right=response_ids_right,
    )
    return record, group


def build_reciprocal_group(
    *, seed: int, population_id: str, outer_fold: int, source_record_id: str,
    donor_id: str, template_id: str, domain: str, mutation_family: str,
    response_grammar: str,
    tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
    max_attempts: int = 10_000,
) -> ReciprocalConstruction:
    """Build one group and preserve every locally rejected schedule attempt."""
    attempts = []
    for attempt in range(int(max_attempts)):
        record, group = construct_reciprocal_attempt(
            seed=seed, attempt_index=attempt, population_id=population_id,
            outer_fold=outer_fold, source_record_id=source_record_id,
            donor_id=donor_id, template_id=template_id, domain=domain,
            mutation_family=mutation_family, response_grammar=response_grammar,
            tokenizers=tokenizers,
        )
        attempts.append(record)
        if group is not None:
            return ReciprocalConstruction(group, tuple(attempts))
    raise RuntimeError("A6 attempt schedule exhausted before a legal reciprocal group")


def audit_reciprocal_group(
    group: ReciprocalGroup,
    tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
) -> dict:
    """Recompute every derived field from persisted semantic text and ASTs."""
    tokenizers = dict(tokenizers)
    _validate_construction_inputs(
        population_id=group.population_id, outer_fold=group.outer_fold,
        source_record_id=group.source_record_id, donor_id=group.donor_id,
        template_id=group.template_id, tokenizers=tokenizers,
    )
    tasks = (group.task_a, group.task_b)
    expected_attempt_seed = _stable_seed(
        "a6-attempt", group.seed, group.population_id, group.outer_fold,
        group.source_record_id, group.donor_id, group.template_id, group.domain,
        group.mutation_family, group.attempt_index,
    )
    expected_left, expected_right = _pair_candidate(
        group.domain, group.mutation_family, expected_attempt_seed
    )
    if _stable_seed(
        "a6-ab", group.seed, group.population_id, group.source_record_id,
        group.attempt_index,
    ) & 1:
        expected_left, expected_right = expected_right, expected_left
    parsed_responses = (
        parse_response(group.response_text_a, group.response_grammar),
        parse_response(group.response_text_b, group.response_grammar),
    )
    expected_response_asts = (
        build_response_ast(group.task_a, group.response_grammar),
        build_response_ast(group.task_b, group.response_grammar),
    )
    expected_response_texts = tuple(render_response(value) for value in expected_response_asts)
    truth_by_rendering = tuple(
        tuple(
            tuple(bool(verify_response(parsed_task, response)) for response in parsed_responses)
            for parsed_task in (
                parse_task(group.prompts_a[index], rendering, group.mutation_family,
                           group.template_id),
                parse_task(group.prompts_b[index], rendering, group.mutation_family,
                           group.template_id),
            )
        )
        for index, rendering in enumerate(RENDERINGS)
    )
    truth = tuple(
        tuple(bool(verify_response(task, response)) for response in parsed_responses)
        for task in tasks
    )
    expected = ((True, False), (False, True))
    prompt_polarities = {
        "A": (int(truth[0][0]), int(truth[0][1])),
        "B": (int(truth[1][0]), int(truth[1][1])),
    }
    response_polarities = {
        "A": (int(truth[0][0]), int(truth[1][0])),
        "B": (int(truth[0][1]), int(truth[1][1])),
    }
    roundtrip = all(
        parse_task(text, rendering, group.mutation_family, group.template_id) == task
        for task, prompts in zip(tasks, (group.prompts_a, group.prompts_b))
        for text, rendering in zip(prompts, RENDERINGS)
    )
    expected_group_id = hashlib.sha256(
        "\0".join((
            "a6-group", group.population_id, str(group.outer_fold),
            group.source_record_id, group.donor_id, group.template_id, str(group.seed),
            group.domain, group.mutation_family, group.response_grammar,
        )).encode("utf-8")
    ).hexdigest()
    prompt_id = lambda world, rendering, text: hashlib.sha256(  # noqa: E731
        "\0".join(("a6-prompt", expected_group_id, world, rendering,
                    hashlib.sha256(text.encode("utf-8")).hexdigest())).encode("utf-8")
    ).hexdigest()
    expected_prompt_ids_a = tuple(
        prompt_id("A", rendering, text)
        for rendering, text in zip(RENDERINGS, group.prompts_a)
    )
    expected_prompt_ids_b = tuple(
        prompt_id("B", rendering, text)
        for rendering, text in zip(RENDERINGS, group.prompts_b)
    )
    prompt_token_ids_a = _token_ids(group.prompts_a, tokenizers)
    prompt_token_ids_b = _token_ids(group.prompts_b, tokenizers)
    prompt_counts_a = _token_counts(prompt_token_ids_a)
    prompt_counts_b = _token_counts(prompt_token_ids_b)
    response_token_ids_a = tuple(
        (name, sequences[0])
        for name, sequences in _token_ids((group.response_text_a,), tokenizers)
    )
    response_token_ids_b = tuple(
        (name, sequences[0])
        for name, sequences in _token_ids((group.response_text_b,), tokenizers)
    )
    certificate_band = group.response_grammar != "certificate" or all(
        40 <= len(ids) <= 80
        for _, ids in (*response_token_ids_a, *response_token_ids_b)
    )
    stored_response_asts_match = (
        group.response_a.grammar == parsed_responses[0].grammar
        and group.response_a.answer == parsed_responses[0].answer
        and group.response_a.domain == group.domain
        and group.response_b.grammar == parsed_responses[1].grammar
        and group.response_b.answer == parsed_responses[1].answer
        and group.response_b.domain == group.domain
    )
    if group.response_grammar == "certificate":
        stored_response_asts_match = stored_response_asts_match and (
            group.response_a == parsed_responses[0] and group.response_b == parsed_responses[1]
        )
    checks = {
        "truth_matrix": truth,
        "truth_matrix_pass": truth == expected,
        "attempt_preimage_pass": tasks == (expected_left, expected_right),
        "top_level_semantics_pass": (
            group.domain in DOMAINS
            and group.mutation_family in MUTATIONS
            and group.response_grammar in RESPONSE_GRAMMARS
            and all(task.domain == group.domain for task in tasks)
            and all(task.mutation_family == group.mutation_family for task in tasks)
            and all(response.grammar == group.response_grammar
                    for response in (group.response_a, group.response_b))
            and _pair_is_legal(group.task_a, group.task_b, group.mutation_family)
        ),
        "truth_all_renderings_pass": all(value == expected for value in truth_by_rendering),
        "prompt_marginals_pass": all(sorted(value) == [0, 1]
                                     for value in prompt_polarities.values()),
        "response_marginals_pass": all(sorted(value) == [0, 1]
                                       for value in response_polarities.values()),
        "independent_evaluators_pass": all(
            evaluate_generator(task) == evaluate_verifier(task) for task in tasks
        ),
        "answers_differ": evaluate_generator(tasks[0]) != evaluate_generator(tasks[1]),
        "complexity_match": task_complexity(tasks[0]) == task_complexity(tasks[1]),
        "changed_fields": changed_fields(tasks[0], tasks[1]),
        "render_roundtrip_pass": roundtrip,
        "group_id_pass": group.group_id == expected_group_id,
        "ast_hashes_pass": (
            group.ast_sha256_a == task_sha256(group.task_a)
            and group.ast_sha256_b == task_sha256(group.task_b)
        ),
        "response_text_hashes_pass": (
            group.response_sha256_a
            == hashlib.sha256(group.response_text_a.encode("utf-8")).hexdigest()
            and group.response_sha256_b
            == hashlib.sha256(group.response_text_b.encode("utf-8")).hexdigest()
        ),
        "response_ast_text_pass": stored_response_asts_match,
        "deterministic_response_construction_pass": (
            (group.response_a, group.response_b) == expected_response_asts
            and (group.response_text_a, group.response_text_b) == expected_response_texts
        ),
        "complete_prompt_ids_pass": (
            group.complete_prompt_ids_a == expected_prompt_ids_a
            and group.complete_prompt_ids_b == expected_prompt_ids_b
        ),
        "prompt_token_evidence_pass": (
            group.prompt_token_counts_a == prompt_counts_a
            and group.prompt_token_counts_b == prompt_counts_b
            and prompt_counts_a == prompt_counts_b
        ),
        "response_token_evidence_pass": (
            group.response_token_ids_a == response_token_ids_a
            and group.response_token_ids_b == response_token_ids_b
        ),
        "certificate_token_band_pass": certificate_band,
        "response_hashes_distinct": group.response_sha256_a != group.response_sha256_b,
    }
    checks["pass"] = bool(all(
        value for key, value in checks.items()
        if key.endswith("_pass") or key in (
            "answers_differ", "complexity_match", "response_hashes_distinct"
        )
    ))
    return checks


def audit_reciprocal_construction(
    construction: ReciprocalConstruction,
    tokenizers: Mapping[str, Callable[[str], Sequence[int]]],
) -> dict:
    """Verify the exact contiguous local attempt prefix through first acceptance."""
    group = construction.group
    attempts = construction.attempts
    expected_indices = tuple(range(group.attempt_index + 1))
    prefix_pass = (
        tuple(record.attempt_index for record in attempts) == expected_indices
        and len(attempts) == group.attempt_index + 1
        and bool(attempts)
        and attempts[-1].status == "ACCEPTED"
        and all(record.status == "REJECTED" for record in attempts[:-1])
    )
    recomputed_records = []
    recomputed_final = None
    earlier_acceptance = False
    for attempt_index in expected_indices:
        record, candidate = construct_reciprocal_attempt(
            seed=group.seed, attempt_index=attempt_index,
            population_id=group.population_id, outer_fold=group.outer_fold,
            source_record_id=group.source_record_id, donor_id=group.donor_id,
            template_id=group.template_id, domain=group.domain,
            mutation_family=group.mutation_family,
            response_grammar=group.response_grammar, tokenizers=tokenizers,
        )
        recomputed_records.append(record)
        if candidate is not None:
            if attempt_index != group.attempt_index:
                earlier_acceptance = True
            else:
                recomputed_final = candidate
    group_audit = audit_reciprocal_group(group, tokenizers)
    checks = {
        "group_audit": group_audit,
        "group_pass": bool(group_audit["pass"]),
        "contiguous_prefix_pass": prefix_pass,
        "attempt_records_pass": tuple(recomputed_records) == attempts,
        "no_skipped_local_acceptance_pass": not earlier_acceptance,
        "final_group_pass": recomputed_final == group,
    }
    checks["pass"] = bool(all(value for key, value in checks.items() if key.endswith("_pass")))
    return checks


def _levenshtein(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        next_row = [left_index]
        for right_index, right_char in enumerate(right, 1):
            next_row.append(min(
                next_row[-1] + 1, row[right_index] + 1,
                row[right_index - 1] + int(left_char != right_char),
            ))
        row = next_row
    return row[-1]


def construction_shortcut_sidecar(group: ReciprocalGroup) -> dict:
    """Construction-only forbidden-fit diagnostics; prompt perplexity joins in S0b."""
    def lexical_overlap(prompt: str, response: str) -> float:
        prompt_words = set(re.findall(r"\w+", prompt.casefold()))
        response_words = set(re.findall(r"\w+", response.casefold()))
        union = prompt_words | response_words
        return 0.0 if not union else len(prompt_words & response_words) / len(union)

    return {
        "source_record_id": group.source_record_id,
        "donor_id": group.donor_id,
        "template_id": group.template_id,
        "prompt_character_lengths_a": [len(value) for value in group.prompts_a],
        "prompt_character_lengths_b": [len(value) for value in group.prompts_b],
        "prompt_word_lengths_a": [len(value.split()) for value in group.prompts_a],
        "prompt_word_lengths_b": [len(value.split()) for value in group.prompts_b],
        "prompt_token_counts_a": dict(group.prompt_token_counts_a),
        "prompt_token_counts_b": dict(group.prompt_token_counts_b),
        "prompt_edit_distances": [
            _levenshtein(left, right)
            for left, right in zip(group.prompts_a, group.prompts_b)
        ],
        "ast_size_depth": task_complexity(group.task_a)[:2],
        "changed_node_details": changed_node_details(group.task_a, group.task_b),
        "prompt_response_overlap_a": [
            lexical_overlap(prompt, group.response_text_a) for prompt in group.prompts_a
        ],
        "prompt_response_overlap_b": [
            lexical_overlap(prompt, group.response_text_b) for prompt in group.prompts_b
        ],
        "answer_in_prompt_a": [
            contains_answer_atom(prompt, group.response_a.answer)
            for prompt in group.prompts_a
        ],
        "answer_in_prompt_b": [
            contains_answer_atom(prompt, group.response_b.answer)
            for prompt in group.prompts_b
        ],
        "numeric_atoms_a": re.findall(r"[-+]?\d+(?:/[1-9]\d*)?", " ".join(group.prompts_a)),
        "numeric_atoms_b": re.findall(r"[-+]?\d+(?:/[1-9]\d*)?", " ".join(group.prompts_b)),
        "response_character_lengths": [len(group.response_text_a), len(group.response_text_b)],
        "response_word_lengths": [
            len(group.response_text_a.split()), len(group.response_text_b.split())
        ],
        "response_token_counts_a": {
            name: len(ids) for name, ids in group.response_token_ids_a
        },
        "response_token_counts_b": {
            name: len(ids) for name, ids in group.response_token_ids_b
        },
    }


def public_group_record(group: ReciprocalGroup) -> dict:
    """Serialize only mechanical construction data; benchmark targets are impossible."""
    return {
        "group_id": group.group_id,
        "population_id": group.population_id,
        "outer_fold": group.outer_fold,
        "source_record_id": group.source_record_id,
        "donor_id": group.donor_id,
        "template_id": group.template_id,
        "seed": group.seed,
        "domain": group.domain,
        "mutation_family": group.mutation_family,
        "response_grammar": group.response_grammar,
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
        "semantic_ast_sha256_a": semantic_task_sha256(group.task_a),
        "semantic_ast_sha256_b": semantic_task_sha256(group.task_b),
        "prompt_content_sha256_a": [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in group.prompts_a
        ],
        "prompt_content_sha256_b": [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in group.prompts_b
        ],
        "response_sha256_a": group.response_sha256_a,
        "response_sha256_b": group.response_sha256_b,
        "attempt_index": group.attempt_index,
        "complete_prompt_ids_a": list(group.complete_prompt_ids_a),
        "complete_prompt_ids_b": list(group.complete_prompt_ids_b),
        "prompt_token_counts_a": dict(group.prompt_token_counts_a),
        "prompt_token_counts_b": dict(group.prompt_token_counts_b),
        "response_token_ids_a": dict(group.response_token_ids_a),
        "response_token_ids_b": dict(group.response_token_ids_b),
        "shortcut_sidecar": construction_shortcut_sidecar(group),
        "mechanical_truth": [[1, 0], [0, 1]],
    }


def public_construction_record(construction: ReciprocalConstruction) -> dict:
    """Canonical local attempt ledger plus its first accepted group."""
    return {
        "attempts": [asdict(record) for record in construction.attempts],
        "group": public_group_record(construction.group),
    }


__all__ = [
    "ANSWER_WRAPPERS", "DOMAINS", "MUTATIONS", "NATURAL_ANSWER_KINDS", "RENDERINGS",
    "RESPONSE_GRAMMARS", "TOKENIZER_FAMILIES", "ConstructionAttempt",
    "ReciprocalConstruction", "ReciprocalGroup", "ResponseAST", "TaskAST",
    "TaskPairCandidate",
    "audit_reciprocal_construction", "audit_reciprocal_group",
    "build_reciprocal_group", "build_response_ast",
    "canonical_answer", "canonicalize_natural_answer_atom", "changed_fields",
    "changed_node_details", "evaluate_generator",
    "construct_reciprocal_attempt", "construct_task_pair_from_seed",
    "construction_shortcut_sidecar",
    "contains_answer_atom", "evaluate_verifier", "parse_answer_atom", "parse_closed_answer",
    "parse_closed_natural_answer",
    "parse_response", "parse_task", "public_construction_record",
    "public_group_record", "render_response",
    "render_task", "semantic_task_sha256", "task_complexity", "task_sha256",
    "verify_response",
]
