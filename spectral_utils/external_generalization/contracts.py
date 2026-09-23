"""Immutable scoring inputs, exact spans and evaluator-only annotations."""
from dataclasses import dataclass, asdict
import hashlib
import json
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def question_key(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


@dataclass(frozen=True)
class Answer:
    uid: str
    benchmark: str
    source_id: str
    question: str
    original_question: str
    steps: tuple[str, ...]
    generator: str | None = None

    def __post_init__(self):
        if not self.uid or not self.question.strip() or not self.steps or any(not isinstance(s, str) for s in self.steps):
            raise ValueError("empty answer identity/question or invalid steps")

    @property
    def group(self):
        return question_key(self.original_question)

    def chain(self):
        text, spans = "", []
        for step in self.steps:
            if text:
                text += "\n\n"
            start = len(text)
            text += step
            spans.append((start, len(text)))
        return text, spans


@dataclass(frozen=True)
class Annotation:
    uid: str
    correct: tuple[bool, ...]
    include: tuple[bool, ...]
    category: str
    out_of_range_error_indices: tuple[int, ...] = ()


def adapt_hard2verify(rows):
    """Input is decrypted by the pinned author's decrypt_sample, never inferred."""
    answers, gold = [], []
    for row in rows:
        steps, labels = row["model_response_by_step"], row["human_labels"]
        if not isinstance(steps, list) or not isinstance(labels, list) or len(steps) != len(labels):
            raise ValueError("decryption/schema/label length failure")
        if any(type(v) not in (int, bool) or v not in (0, 1) for v in labels):
            raise ValueError("Hard2Verify labels must be binary correctness")
        uid = "hard2verify:" + str(row["unique_id"])
        answers.append(Answer(uid, "hard2verify", str(row.get("question_id", "")),
                              row["question"], row["question"], tuple(steps), row.get("model")))
        gold.append(Annotation(uid, tuple(map(bool, labels)), (True,) * len(steps), "all"))
    validate_corpus(answers, gold)
    return answers, gold


def adapt_socratic(directory):
    answers, gold = [], []
    for path in sorted(Path(directory).rglob("test_*.jsonl")):
        for line in path.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            steps, errors = row["modified_process"], row["error_steps"]
            if not isinstance(steps, list) or any(type(x) != int for x in errors):
                raise ValueError("Socratic schema/index failure")
            uid = "socratic:" + path.stem + ":" + str(row["idx"])
            answers.append(Answer(uid, "socratic", str(row["idx"]), row["modified_question"],
                                  row["original_question"], tuple(steps), row.get("generator")))
            # PRMEval's one-based membership semantics retain out-of-range indices as inert.
            gold.append(Annotation(uid, tuple(i + 1 not in errors for i in range(len(steps))),
                                   (True,) * len(steps), row["classification"],
                                   tuple(i for i in errors if not 1 <= i <= len(steps))))
    validate_corpus(answers, gold)
    return answers, gold


def validate_corpus(answers, gold):
    if not answers or len(answers) != len(gold):
        raise ValueError("empty or unmatched corpus")
    ids = [a.uid for a in answers]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate answer IDs; never silently deduplicate")
    for a, g in zip(answers, gold):
        if a.uid != g.uid or len(a.steps) != len(g.correct) or len(g.correct) != len(g.include):
            raise ValueError("annotation alignment failure")


def tokenize_answer(answer, tokenizer, prompt_ids, max_context):
    text, chars = answer.chain()
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(prompt_ids) + len(ids) > max_context:
        raise ValueError(f"context overflow: {answer.uid}; truncation prohibited")
    spans = []
    for left, right in chars:
        if left == right:
            insertion = next((i for i, (a, b) in enumerate(offsets) if a >= left), len(ids))
            spans.append((insertion, insertion))
            continue
        indices = [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("unmapped/noncontiguous step")
        start, stop = indices[0], indices[-1] + 1
        if spans and start < spans[-1][1]:
            raise ValueError("one token crosses two steps; ambiguous alignment")
        if offsets[start][0] > left or offsets[stop - 1][1] < right:
            raise ValueError("step text not fully represented by tokens")
        spans.append((start, stop))
    return {"uid": answer.uid, "prompt_ids": prompt_ids, "gen_ids": ids,
            "token_offsets": offsets, "step_char_spans": chars, "step_token_spans": spans}


def overlap_manifest(corpora):
    """Union source identity and original/evaluated exact text, including variants."""
    entries = [(name, a) for name, rows in corpora.items() for a in rows]
    parent = list(range(len(entries)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, (name, a) in enumerate(entries):
        keys = [("text", question_key(a.original_question)), ("text", question_key(a.question))]
        if a.source_id:
            keys.append(("source:" + name, a.source_id))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    components = {}
    for i, (name, a) in enumerate(entries):
        components.setdefault(root(i), []).append((name, a.uid))
    groups, overlap = {}, []
    for rows in components.values():
        key = digest(sorted(rows))
        groups.update({uid: key for _, uid in rows})
        if len({name for name, _ in rows}) > 1:
            overlap.append({"group": key, "datasets": sorted({name for name, _ in rows}), "uids": [u for _, u in rows]})
    return {"groups": groups, "overlap": overlap, "paraphrases_checked": False}
