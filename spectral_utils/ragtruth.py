"""
ragtruth.py — loader, evidence-condition prompt builder, and span-alignment helpers for
the RAGTruth corpus (Niu et al. 2024, ACL 2024, "RAGTruth: A Hallucination Corpus for
Developing Trustworthy Retrieval-Augmented Language Models"), vendored under
`data/ragtruth_protocol/` (provenance: `data/ragtruth_protocol/PROVENANCE.md`).

Used by the Evidence-Contrast U-PCR/DUFS-LIU campaign
(`cluster/run_conditional_rescore.py`): keep each published RAGTruth response FIXED and
teacher-force-rescore it under several evidence conditions (full context, no context,
leave-one-chunk-out), building dependent per-token telemetry traces that a downstream
Laplacian fusion can use for response-level grounding detection and span localization.
See `docs/research_notes/ragtruth_ec_preregistration_v1.md` for the frozen arm roster.

PROMPT SURGERY, NOT TEMPLATE RECONSTRUCTION
--------------------------------------------------------------------------------------
`source_info.jsonl`'s `prompt` field is the exact, original generation prompt used to
produce each response, and the evidence text it embeds is always an exact substring of
that prompt:

- QA:       `source_info["passages"]` (three `"passage N:"` blocks) is a substring.
- Summary:  `source_info["source_info"]` (the raw document string) is a substring.
- Data2txt: `str(source_info["source_info"])` (a Python dict's `str()` form) is a
            substring.

(All three verified exactly against the real corpus before this module was written.)
Every evidence condition is therefore built by locating this evidence substring inside
`prompt` and replacing it — never by reconstructing the surrounding instruction text from
scratch. This keeps every condition identical to the original prompt modulo the evidence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data" / "ragtruth_protocol"

TASK_TYPES = ("QA", "Summary", "Data2txt")

_PASSAGE_PATTERN = re.compile(r"passage \d+:")
_PARA_SEP_PATTERN = re.compile(r"\n\n+")

# Preregistered chunking parameters (docs/research_notes/ragtruth_ec_preregistration_v1.md).
SUMMARY_MIN_CHARS = 200
SUMMARY_MAX_CHUNKS = 8


# ── loading ──────────────────────────────────────────────────────────────────────────

def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _context_text(source_row: dict) -> str:
    """The exact evidence substring embedded verbatim in `source_row['prompt']`."""
    task_type = source_row["task_type"]
    if task_type == "QA":
        return source_row["source_info"]["passages"]
    if task_type == "Summary":
        return source_row["source_info"]
    if task_type == "Data2txt":
        return str(source_row["source_info"])
    raise ValueError(f"unknown RAGTruth task_type: {task_type!r}")


def load_ragtruth(split="test", task_types=None, n=None, seed=0, data_dir=None):
    """Join `response.jsonl` and `source_info.jsonl` on `source_id`.

    Returns `(rows, diagnostics)`. Each row: `response_id, source_id, task_type, source,
    model, quality, response, labels, context, prompt`. `context` is the exact evidence
    substring `condition_prompt`/`chunk_source` operate on; `prompt` is the original
    full-context generation prompt.

    A row is dropped (and counted in `diagnostics`) if either of two integrity checks
    fails: (a) every label's `response[start:end] == text`, catching corpus/text drift;
    (b) `context in prompt`, the precondition every downstream prompt-surgery function
    requires.
    """
    data_dir = Path(data_dir) if data_dir else _DATA_DIR
    responses = _read_jsonl(data_dir / "response.jsonl")
    sources = {row["source_id"]: row for row in _read_jsonl(data_dir / "source_info.jsonl")}

    wanted_types = set(task_types) if task_types else set(TASK_TYPES)
    kept, dropped_mismatch, dropped_no_source = [], 0, 0
    for row in responses:
        if row["split"] != split:
            continue
        src = sources.get(row["source_id"])
        if src is None:
            dropped_no_source += 1
            continue
        if src["task_type"] not in wanted_types:
            continue

        response_text = row["response"]
        spans_ok = all(
            response_text[label["start"]:label["end"]] == label["text"]
            for label in row["labels"]
        )
        if not spans_ok:
            dropped_mismatch += 1
            continue

        context = _context_text(src)
        if context not in src["prompt"]:
            dropped_mismatch += 1
            continue

        kept.append({
            "response_id": row["id"],
            "source_id": row["source_id"],
            "task_type": src["task_type"],
            "source": src["source"],
            "model": row["model"],
            "quality": row["quality"],
            "response": response_text,
            "labels": row["labels"],
            "context": context,
            "prompt": src["prompt"],
            # Raw parsed source_info, carried through for _field_chunks (Data2txt only —
            # a dict there; a {question,passages} dict for QA; a plain str for Summary).
            "_source_info_dict": src["source_info"],
        })

    if n is not None and n < len(kept):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(kept), size=n, replace=False))
        kept = [kept[i] for i in idx]

    diagnostics = {
        "split": split, "task_types": sorted(wanted_types),
        "n_kept": len(kept),
        "n_dropped_span_mismatch": dropped_mismatch,
        "n_dropped_no_source": dropped_no_source,
    }
    return kept, diagnostics


# ── chunking: the leave-one-out unit, as char-offset spans into `context` ──────────────

def chunk_source(row: dict):
    """Preregistered per-task-type chunk spans `[(start, end), ...]` into `row['context']`.

    Offsets, not substrings, so a leave-one-chunk-out condition can splice the original
    context string directly (`context[:s] + context[e:]`) with no reconstruction risk.
    """
    task_type = row["task_type"]
    context = row["context"]
    if task_type == "QA":
        return _qa_chunks(context)
    if task_type == "Summary":
        return _paragraph_chunks(context)
    if task_type == "Data2txt":
        return _field_chunks(row)
    raise ValueError(f"unknown RAGTruth task_type: {task_type!r}")


def _qa_chunks(passages: str):
    marks = list(_PASSAGE_PATTERN.finditer(passages))
    spans = []
    for i, match in enumerate(marks):
        start = match.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(passages)
        spans.append((start, end))
    return spans


def _split_paragraph_spans(doc: str):
    spans = []
    prev_end = 0
    for match in _PARA_SEP_PATTERN.finditer(doc):
        start, end = match.span()
        if start > prev_end:
            spans.append((prev_end, start))
        prev_end = end
    if prev_end < len(doc):
        spans.append((prev_end, len(doc)))
    return spans


def _paragraph_chunks(doc: str, min_chars=SUMMARY_MIN_CHARS, max_chunks=SUMMARY_MAX_CHUNKS):
    spans = _split_paragraph_spans(doc)
    if not spans:
        return [(0, len(doc))]

    # Merge fragments shorter than `min_chars` into their running buffer (fallback arm:
    # a document with no `\n\n` breaks at all collapses to the single (0, len(doc)) span,
    # handled correctly by this loop with spans == [(0, len(doc))]).
    merged, buf_start, buf_end = [], None, None
    for start, end in spans:
        if buf_start is None:
            buf_start = start
        buf_end = end
        if buf_end - buf_start >= min_chars:
            merged.append((buf_start, buf_end))
            buf_start = None
    if buf_start is not None:
        if merged:
            merged[-1] = (merged[-1][0], buf_end)
        else:
            merged.append((buf_start, buf_end))

    while len(merged) > max_chunks:
        sizes = [e - s for s, e in merged]
        pair_sizes = [sizes[i] + sizes[i + 1] for i in range(len(merged) - 1)]
        j = int(np.argmin(pair_sizes))
        merged[j] = (merged[j][0], merged[j + 1][1])
        del merged[j + 1]
    return merged


def _field_chunks(row: dict):
    """Leave-one-top-level-JSON-field-out spans, located in `str(dict)` order.

    Each span covers `repr(key): value` through just before the next key's anchor (so it
    also owns the trailing `, ` separator, keeping removal a single contiguous deletion).
    Assumes no top-level key name recurs inside a nested value's own keys — verified for
    every Data2txt row in the corpus (`test_ragtruth.py`).
    """
    source_info = row["_source_info_dict"]
    text = row["context"]
    keys = list(source_info.keys())
    starts, cursor = [], 0
    for key in keys:
        anchor = repr(key) + ": "
        pos = text.index(anchor, cursor)
        starts.append(pos)
        cursor = pos + len(anchor)
    boundaries = starts + [len(text) - 1]  # -1: stop before the closing '}'
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(keys))]


# ── condition prompts ────────────────────────────────────────────────────────────────

def condition_prompt(row: dict, condition: str) -> str:
    """`condition in {"full", "noctx", "loo_0", ..., "loo_{m-1}"}`.

    Every condition is `row['prompt']` with only the evidence substring edited — the
    instruction text, question, and trailing output cue are always byte-identical to the
    original, per the module docstring's prompt-surgery design.
    """
    if condition == "full":
        return row["prompt"]

    context = row["context"]
    if condition == "noctx":
        new_context = ""
    elif condition.startswith("loo_"):
        j = int(condition[len("loo_"):])
        spans = chunk_source(row)
        if not (0 <= j < len(spans)):
            raise ValueError(f"condition {condition!r} out of range for {len(spans)} chunks")
        start, end = spans[j]
        new_context = context[:start] + context[end:]
    else:
        raise ValueError(f"unknown condition: {condition!r}")

    prompt = row["prompt"]
    idx = prompt.index(context)  # existence guaranteed by load_ragtruth's integrity gate
    return prompt[:idx] + new_context + prompt[idx + len(context):]


def n_chunks(row: dict) -> int:
    return len(chunk_source(row))


def all_conditions(row: dict):
    return ["full", "noctx"] + [f"loo_{j}" for j in range(n_chunks(row))]


def distinct_conditions(row: dict):
    """`all_conditions`, minus any `loo_j` pass whose prompt is byte-identical to another
    already-listed condition.

    Measured on the vendored corpus (2,700/2,700 test rows, zero exceptions): every
    Summary document has no `\\n\\n` break at all, so `chunk_source` returns exactly one
    chunk spanning the whole document and `loo_0` degenerates to `noctx` — this is the
    documented fallback case in the module docstring / preregistration, not a rare edge
    case. Skipping the duplicate here is a compute optimization (one fewer teacher-forced
    pass per Summary response); it changes no condition's definition.
    """
    context = row["context"]
    seen_prompts = set()
    kept = []
    for condition in all_conditions(row):
        if condition.startswith("loo_"):
            j = int(condition[len("loo_"):])
            start, end = chunk_source(row)[j]
            if (start, end) == (0, len(context)):
                continue  # identical to noctx
        prompt = condition_prompt(row, condition)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        kept.append(condition)
    return kept


# ── response tokenization + span alignment ──────────────────────────────────────────

def response_token_spans(tokenizer, response: str, labels):
    """Tokenize `response` once (`add_special_tokens=False`, char offsets), then map each
    annotated char span onto token index ranges via the same overlap rule
    `spectral_utils.processbench.step_token_spans` uses: a token `[x, y)` belongs to an
    annotated span `[a, b)` iff `x < b and y > a` (any character overlap).

    Returns `(gen_token_ids, token_offsets, span_token_spans, align_diag)`, where
    `span_token_spans[i]` is `(first_token_idx, last_token_idx_exclusive)` for `labels[i]`,
    or `None` if the label produced no overlapping token (should not occur once
    `load_ragtruth`'s text-equality gate has passed, but recorded rather than assumed).
    """
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    gen_token_ids = encoded["input_ids"]

    span_token_spans = []
    n_unmapped = 0
    for label in labels:
        a, b = label["start"], label["end"]
        first, last = None, None
        for token_idx, (x, y) in enumerate(offsets):
            if x < b and y > a:
                if first is None:
                    first = token_idx
                last = token_idx
        if first is None:
            span_token_spans.append(None)
            n_unmapped += 1
        else:
            span_token_spans.append((first, last + 1))

    align_diag = {
        "n_labels": len(labels), "n_unmapped": n_unmapped,
        "n_tokens": len(gen_token_ids),
    }
    return gen_token_ids, offsets, span_token_spans, align_diag


# ── grouped splits (group by source_id, never by response row) ─────────────────────

def group_split(rows, frac=0.5, seed=0):
    """Partition `rows` into two lists, grouped by `source_id` so responses that share a
    source never land on both sides (required for any calibration/evaluation split and
    for grouped bootstrap intervals — RAGTruth has up to 6 responses per source)."""
    source_ids = sorted({row["source_id"] for row in rows})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(source_ids))
    cut = int(round(frac * len(source_ids)))
    left_ids = {source_ids[i] for i in perm[:cut]}
    left = [row for row in rows if row["source_id"] in left_ids]
    right = [row for row in rows if row["source_id"] not in left_ids]
    return left, right


# ── known-answer tests ───────────────────────────────────────────────────────────────

def smoke() -> None:
    # Handmade rows, one per task type, exercising the prompt-surgery contract without
    # touching the vendored corpus (so this test never depends on data/ being present).
    qa_passages = "passage 1:AAA BBB\n\npassage 2:CCC DDD\n\npassage 3:EEE FFF\n"
    qa_prompt = f"Answer the question:\nwhat is it\nPassages:\n{qa_passages}\noutput:"
    qa_row = {
        "task_type": "QA", "context": qa_passages, "prompt": qa_prompt,
        "response": "It is X.", "labels": [],
    }
    spans = chunk_source(qa_row)
    assert len(spans) == 3, spans
    for start, end in spans:
        assert qa_passages[start:end].startswith("passage ")
    full = condition_prompt(qa_row, "full")
    assert full == qa_prompt
    noctx = condition_prompt(qa_row, "noctx")
    assert "passage" not in noctx and "Answer the question" in noctx and "output:" in noctx
    loo1 = condition_prompt(qa_row, "loo_1")
    assert "passage 1:" in loo1 and "passage 3:" in loo1 and "CCC DDD" not in loo1

    doc = ("Para one is short.\n\n"
           + "Para two is quite a bit longer than the first one, well over the floor. " * 3
           + "\n\nPara three is also reasonably long, over the floor by itself here too. " * 3)
    summ_prompt = f"Summarize:\n{doc}\n\noutput:"
    summ_row = {"task_type": "Summary", "context": doc, "prompt": summ_prompt,
                "response": "A summary.", "labels": []}
    sp = chunk_source(summ_row)
    assert len(sp) >= 2, sp
    assert sp[0][1] - sp[0][0] >= SUMMARY_MIN_CHARS or len(sp) == 1
    loo0 = condition_prompt(summ_row, "loo_0")
    assert "Para one is short." not in loo0
    assert "Summarize:" in loo0 and "output:" in loo0

    d = {"name": "Acme", "hours": {"Mon": "9-5", "Tue": "9-5"}, "stars": 4.5}
    d2t_context = str(d)
    d2t_prompt = f"Write an overview.\nStructured data:\n{d2t_context}\nOverview:"
    d2t_row = {"task_type": "Data2txt", "context": d2t_context, "prompt": d2t_prompt,
               "response": "Overview text.", "labels": [], "_source_info_dict": d}
    fp = chunk_source(d2t_row)
    assert len(fp) == 3, fp
    loo_hours = condition_prompt(d2t_row, "loo_1")  # 'hours' is the 2nd key
    assert "'name': 'Acme'" in loo_hours
    assert "'stars': 4.5" in loo_hours
    assert "'hours'" not in loo_hours
    assert "Write an overview." in loo_hours and "Overview:" in loo_hours

    # A degenerate no-break Summary document still yields one whole-doc chunk, and
    # loo_0 on it must equal the noctx condition (the only chunk IS the whole context).
    flat_doc = "one long paragraph with no double newlines at all in it whatsoever."
    flat_row = {"task_type": "Summary", "context": flat_doc,
                "prompt": f"Summarize:\n{flat_doc}\n\noutput:",
                "response": "x", "labels": []}
    flat_spans = chunk_source(flat_row)
    assert flat_spans == [(0, len(flat_doc))]
    assert condition_prompt(flat_row, "loo_0") == condition_prompt(flat_row, "noctx")

    # group_split never splits one source_id across both sides.
    rows = [{"source_id": str(i % 5), "response_id": str(i)} for i in range(30)]
    left, right = group_split(rows, frac=0.4, seed=1)
    left_ids = {r["source_id"] for r in left}
    right_ids = {r["source_id"] for r in right}
    assert not (left_ids & right_ids), (left_ids, right_ids)
    assert len(left) + len(right) == 30

    print("ragtruth.smoke: PASS (6 checks: QA/Summary/Data2txt prompt surgery, "
          "degenerate-Summary fallback, group_split)")


if __name__ == "__main__":
    smoke()
