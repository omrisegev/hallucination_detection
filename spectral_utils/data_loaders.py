"""
Dataset loaders and grading functions for all benchmarks used in the project.

Supported datasets:
  - GSM8K       (LapEigvals Listing 5 prompt; exact-match grading)
  - MATH-500    (competition math; boxed answer extraction)
  - GPQA Diamond (graduate-level MCQ; letter extraction)
  - HotpotQA    (multi-hop QA; substring match)
  - 2WikiMultiHopQA (multi-hop QA; normalized to Hotpot-style context)
  - TriviaQA    (rc.nocontext; normalized alias exact-match grading)
  - WebQ        (WebQuestions; normalized alias exact-match grading)
"""
import re
import string
from typing import Any

import numpy as np


# ── GSM8K ─────────────────────────────────────────────────────────────────────

# LapEigvals (arXiv:2502.17598) Listing 5 — verbatim
_LAPEI_PROMPT = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" '
    "where [answer] is the response to the problem."
)


def load_gsm8k(split: str = "test") -> list[dict]:
    """Load GSM8K from HuggingFace datasets. Returns list of {question, answer} dicts."""
    from datasets import load_dataset
    ds    = load_dataset("openai/gsm8k", "main", split=split)
    items = [{"question": ds[i]["question"], "answer": ds[i]["answer"]}
             for i in range(len(ds))]
    print(f"Loaded {len(items)} GSM8K {split} problems.")
    return items


def gsm8k_prompt(question: str) -> str:
    """Format a GSM8K question using the LapEigvals Listing 5 prompt template."""
    return _LAPEI_PROMPT.format(question=question)


def extract_gold_gsm8k(answer_field: str) -> str:
    """Extract the numeric answer from the '#### N' field of a GSM8K answer."""
    m = re.search(r"####\s*(.+)", answer_field)
    return m.group(1).strip() if m else answer_field.strip()


def extract_model_answer_gsm8k(text: str):
    """
    Extract the model's answer from 'The final answer is [X]'.

    Tries the end of the text first (most reliable), then falls back to any
    occurrence. Returns None if the phrase is absent (rejected response).
    """
    m = re.search(
        r"[Tt]he final answer is\s*\[?([^\]\n\.]{1,50}?)\]?[\.,!\s]*$",
        text, re.MULTILINE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"[Tt]he final answer is\s*\[?([^\]\n\.]{1,50}?)\]?", text)
    return m.group(1).strip() if m else None


def _normalize_gsm8k(s) -> float | str | None:
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"[\$,%]", "", s).replace(",", "").rstrip(".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return s.lower().strip()


def is_correct_gsm8k(gen: str, gold_answer: str) -> bool:
    """Exact-match grading for GSM8K (numeric comparison with 1e-6 tolerance)."""
    gold_norm  = _normalize_gsm8k(extract_gold_gsm8k(gold_answer))
    model_norm = _normalize_gsm8k(extract_model_answer_gsm8k(gen))
    if model_norm is None:
        return False
    if isinstance(gold_norm, float) and isinstance(model_norm, float):
        return abs(gold_norm - model_norm) < 1e-6
    return str(gold_norm) == str(model_norm)


# ── MATH-500 ──────────────────────────────────────────────────────────────────

def load_math500(n_samples: int = 300) -> list:
    """
    Load MATH-500 problems. Tries several HuggingFace paths in order.
    Raises RuntimeError if none succeed.
    """
    from datasets import load_dataset
    attempts = [
        ("lighteval/MATH_500",         {},                  "test"),
        ("HuggingFaceH4/MATH-500",     {},                  "test"),
        ("EleutherAI/hendrycks_math",  {"name": "all"},     "test"),
        ("EleutherAI/hendrycks_math",  {"name": "algebra"}, "test"),
    ]
    for path, kwargs, split in attempts:
        try:
            ds      = load_dataset(path, split=split, **kwargs)
            samples = [ds[i] for i in range(min(n_samples, len(ds)))]
            print(f"Loaded {len(samples)} MATH problems from {path}.")
            return samples
        except Exception as e:
            print(f"  {path} failed: {e}")
    raise RuntimeError("Could not load MATH-500 from any source.")


def math_prompt(row: dict) -> str:
    """Format a MATH-500 row as a chain-of-thought prompt."""
    q = row.get("problem", row.get("query", row.get("question", "")))
    return (
        "Solve the following competition math problem. "
        "Show all your work step by step, then give your final answer in \\boxed{}.\n\n"
        + q
    )


def _extract_math_answer(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        val = re.sub(r"[^\d\.\-\/\(\)]", "", m.group(1).replace(",", ""))
        if val:
            return val
    nums = re.findall(r"[\-\d]+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def is_correct_math(gen: str, gold_row: dict) -> bool:
    """Compare extracted boxed answers numerically, fall back to string match."""
    sol = gold_row.get("solution", gold_row.get("answer", gold_row.get("output", "")))
    p, g = _extract_math_answer(gen), _extract_math_answer(sol)
    if not p or not g:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return p.strip() == g.strip()


# ── GPQA Diamond ──────────────────────────────────────────────────────────────

def load_gpqa() -> list:
    """Load all GPQA Diamond problems from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    print(f"Loaded {len(ds)} GPQA Diamond problems.")
    return list(ds)


def gpqa_prompt_and_answer(row: dict, idx: int) -> tuple[str, str]:
    """
    Shuffle the four answer choices (deterministically per idx) and return
    (prompt, correct_letter).
    """
    rng     = np.random.default_rng(idx)
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    order          = rng.permutation(4)
    letters        = ["A", "B", "C", "D"]
    shuffled       = [choices[i] for i in order]
    correct_letter = letters[int(np.where(order == 0)[0][0])]

    opts   = "\n".join(f"{l}) {t}" for l, t in zip(letters, shuffled))
    prompt = (
        "Answer the following graduate-level science question by selecting the best answer. "
        "Think through your reasoning carefully.\n\n"
        f"{row['Question']}\n\n{opts}\n\n"
        "Provide your reasoning, then state your final answer as a single letter: A, B, C, or D."
    )
    return prompt, correct_letter


def extract_gpqa_answer(text: str) -> str:
    """Extract the final answer letter (A/B/C/D) from a model response."""
    patterns = [
        r"(?:answer is|answer:|the answer|final answer)[^A-Da-d]*([A-Da-d])\b",
        r"\b([A-Da-d])\s*(?:is correct|is the best|is the answer)",
        r"^\s*([A-Da-d])[\)\.]?\s*$",
    ]
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r"\b([A-D])\b", text.upper())
    return matches[-1] if matches else ""


def is_correct_gpqa(gen: str, correct_letter: str) -> bool:
    return extract_gpqa_answer(gen) == correct_letter.upper()


# ── HotpotQA ──────────────────────────────────────────────────────────────────

def load_hotpotqa(n_samples: int = 200) -> list:
    """Load HotpotQA fullwiki validation split."""
    from datasets import load_dataset
    ds      = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
    samples = [ds[i] for i in range(min(n_samples, len(ds)))]
    print(f"Loaded {len(samples)} HotpotQA samples (fullwiki, validation).")
    return samples


def hotpotqa_prompt(row: dict) -> str:
    return (
        "You will answer a multi-hop question that requires finding two pieces "
        "of information and connecting them.\n"
        "Think step by step: first identify the intermediate fact, then use it "
        "to answer the main question. Show your reasoning clearly.\n\n"
        f"Question: {row['question']}\n\n"
        "Provide your reasoning, then give your final answer on the last line."
    )


def _normalize_hotpotqa(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def is_correct_hotpotqa(gen: str, gold: str) -> bool:
    return _normalize_hotpotqa(gold) in _normalize_hotpotqa(gen)


def _normalize_supporting_facts(sf: Any) -> dict:
    titles: list[str] = []
    sent_ids: list[int] = []

    if isinstance(sf, dict):
        titles = [str(t) for t in sf.get("title", [])]
        raw_sent_ids = sf.get("sent_id", sf.get("sent_idx", sf.get("sentence_id", [])))
        sent_ids = []
        for sid in raw_sent_ids:
            try:
                sent_ids.append(int(sid))
            except (TypeError, ValueError):
                sent_ids.append(-1)
    elif isinstance(sf, list):
        for item in sf:
            if isinstance(item, dict):
                title = item.get("title", item.get("paragraph_title", ""))
                sent_id = item.get("sent_id", item.get("sent_idx", item.get("sentence_id", -1)))
            elif isinstance(item, (list, tuple)) and item:
                title = item[0]
                sent_id = item[1] if len(item) > 1 else -1
            else:
                title = str(item)
                sent_id = -1
            titles.append(str(title))
            try:
                sent_ids.append(int(sent_id))
            except (TypeError, ValueError):
                sent_ids.append(-1)

    if len(sent_ids) < len(titles):
        sent_ids.extend([-1] * (len(titles) - len(sent_ids)))

    return {
        "title": titles,
        "sent_id": sent_ids[:len(titles)],
        "pairs": [(titles[i], sent_ids[i]) for i in range(min(len(titles), len(sent_ids)))],
    }


def _normalize_hotpot_style_context(context: Any) -> dict:
    if isinstance(context, dict) and "title" in context and "sentences" in context:
        titles = [str(t) for t in context.get("title", [])]
        raw_sentences = context.get("sentences", [])
        sentences = []
        for sent_block in raw_sentences:
            if isinstance(sent_block, str):
                sentences.append([sent_block.strip()])
            elif isinstance(sent_block, (list, tuple)):
                sentences.append([str(s).strip() for s in sent_block])
            else:
                sentences.append([str(sent_block).strip()])
        return {"title": titles, "sentences": sentences}

    if isinstance(context, dict):
        for key in ("paragraphs", "contexts", "documents", "docs"):
            if key in context:
                return _normalize_hotpot_style_context(context[key])

    titles: list[str] = []
    sentences: list[list[str]] = []
    if isinstance(context, (list, tuple)):
        for idx, item in enumerate(context):
            title = f"Passage {idx+1}"
            sent_block: Any = ""
            if isinstance(item, dict):
                title = str(item.get("title", item.get("name", item.get("paragraph_title", title))))
                sent_block = item.get("sentences", item.get("text", item.get("paragraph", "")))
            elif isinstance(item, (list, tuple)) and item:
                title = str(item[0])
                sent_block = item[1] if len(item) > 1 else ""
            else:
                sent_block = item

            if isinstance(sent_block, str):
                sentence_list = [sent_block.strip()] if sent_block.strip() else []
            elif isinstance(sent_block, (list, tuple)):
                sentence_list = [str(s).strip() for s in sent_block if str(s).strip()]
            else:
                sentence_list = [str(sent_block).strip()] if str(sent_block).strip() else []

            titles.append(title)
            sentences.append(sentence_list)

    return {"title": titles, "sentences": sentences}


def normalize_agentic_multihop_row(row: dict, dataset: str) -> dict:
    dataset_key = dataset.lower()
    if dataset_key in ("2wiki", "2wikimultihopqa"):
        context = _normalize_hotpot_style_context(
            row.get("context", row.get("paragraphs", row.get("contexts", row.get("documents", []))))
        )
        supporting_facts = _normalize_supporting_facts(
            row.get("supporting_facts", row.get("supporting_sentences", row.get("evidences", [])))
        )
        return {
            "id": row.get("_id", row.get("id", row.get("qid", ""))),
            "dataset": "2wikimultihopqa",
            "question": row.get("question", row.get("query", "")),
            "answer": row.get("answer", row.get("gold_answer", "")),
            "context": context,
            "supporting_facts": supporting_facts,
            "type": row.get("type", row.get("question_type", "")),
            "raw_row": dict(row),
        }

    if dataset_key == "hotpotqa":
        return {
            "id": row.get("id", row.get("_id", "")),
            "dataset": "hotpotqa",
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "context": _normalize_hotpot_style_context(row.get("context", {})),
            "supporting_facts": _normalize_supporting_facts(row.get("supporting_facts", {})),
            "type": row.get("type", ""),
            "raw_row": dict(row),
        }

    raise ValueError(f"Unsupported agentic multi-hop dataset: {dataset!r}")


def load_hotpotqa_agentic(n_samples: int = 200) -> list[dict]:
    rows = load_hotpotqa(n_samples=n_samples)
    return [normalize_agentic_multihop_row(row, "hotpotqa") for row in rows]


def load_2wikimultihopqa(n_samples: int = 200) -> list[dict]:
    from datasets import load_dataset

    attempts = [
        ("framolfese/2WikiMultihopQA", {}, "validation"),
        ("framolfese/2WikiMultihopQA", {}, "dev"),
        ("framolfese/2WikiMultihopQA", {"trust_remote_code": True}, "validation"),
        ("framolfese/2WikiMultihopQA", {"trust_remote_code": True}, "dev"),
        ("xanhho/2WikiMultihopQA", {}, "validation"),
        ("xanhho/2WikiMultihopQA", {"trust_remote_code": True}, "validation"),
    ]
    last_error = None
    for path, kwargs, split in attempts:
        try:
            ds = load_dataset(path, split=split, **kwargs)
            samples = [normalize_agentic_multihop_row(ds[i], "2wikimultihopqa")
                       for i in range(min(n_samples, len(ds)))]
            print(f"Loaded {len(samples)} 2WikiMultihopQA samples from {path} ({split}).")
            return samples
        except Exception as ex:
            last_error = ex
            print(f"  {path} ({split}) failed: {ex}")

    raise RuntimeError(f"Could not load 2WikiMultihopQA from any source. Last error: {last_error}")


def load_agentic_multihop_dataset(dataset: str, n_samples: int = 200) -> list[dict]:
    dataset_key = dataset.lower()
    if dataset_key == "hotpotqa":
        return load_hotpotqa_agentic(n_samples=n_samples)
    if dataset_key in ("2wiki", "2wikimultihopqa"):
        return load_2wikimultihopqa(n_samples=n_samples)
    raise ValueError(f"Unsupported agentic dataset: {dataset!r}")


# ── TriviaQA ──────────────────────────────────────────────────────────────────

def load_trivia_qa(n_samples: int = 300, split: str = "validation") -> list[dict]:
    """
    Load TriviaQA rc.nocontext from HuggingFace.

    Returns list of {question, answer_value, aliases} dicts.
    aliases is the full list of valid answer strings (from the dataset).
    """
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    items = []
    for i in range(min(n_samples, len(ds))):
        row = ds[i]
        items.append({
            "question":     row["question"],
            "answer_value": row["answer"]["value"],
            "aliases":      row["answer"]["aliases"],
        })
    print(f"Loaded {len(items)} TriviaQA {split} samples.")
    return items


def trivia_qa_prompt(question: str) -> str:
    """Direct-answer prompt matching the EPR paper setup (no CoT, short answer)."""
    return (
        f"Answer the following question with a short, direct answer.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def _normalize_qa(s: str) -> str:
    """Normalize for QA exact-match: lowercase, strip articles, strip punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def is_correct_trivia_qa(gen: str, item: dict) -> bool:
    """
    Normalized exact-match against all aliases (TriviaQA standard evaluation).

    Extracts the first line / text before a newline as the model's answer,
    then checks if the normalized form matches any normalized alias.
    """
    pred = gen.strip().split("\n")[0].strip()
    pred_norm = _normalize_qa(pred)
    return any(_normalize_qa(a) == pred_norm for a in item["aliases"])


# ── WebQuestions ──────────────────────────────────────────────────────────────

def load_webq(n_samples: int = 300, split: str = "test") -> list[dict]:
    """
    Load WebQuestions from HuggingFace.

    Returns list of {question, answers} dicts where answers is the list of
    gold answer strings.
    """
    from datasets import load_dataset
    ds = load_dataset("web_questions", split=split)
    items = []
    for i in range(min(n_samples, len(ds))):
        row = ds[i]
        items.append({
            "question": row["question"],
            "answers":  row["answers"],
        })
    print(f"Loaded {len(items)} WebQuestions {split} samples.")
    return items


def webq_prompt(question: str) -> str:
    """Direct-answer prompt for WebQuestions."""
    return (
        f"Answer the following question with a short, direct answer.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def is_correct_webq(gen: str, item: dict) -> bool:
    """
    Normalized exact-match against all gold answers (WebQ standard evaluation).
    """
    pred = gen.strip().split("\n")[0].strip()
    pred_norm = _normalize_qa(pred)
    return any(_normalize_qa(a) == pred_norm for a in item["answers"])


# ── HumanEval ─────────────────────────────────────────────────────────────────

def load_humaneval(n_samples: int = 164) -> list[dict]:
    """Load HumanEval from HuggingFace. Returns list of {task_id, prompt, test, entry_point}."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for i in range(min(n_samples, len(ds))):
        row = ds[i]
        items.append({
            "task_id":     row["task_id"],
            "prompt":      row["prompt"],
            "test":        row["test"],
            "entry_point": row["entry_point"],
        })
    print(f"Loaded {len(items)} HumanEval problems.")
    return items


def humaneval_prompt(row: dict, error_context: str = "") -> str:
    """Instruction prompt for HumanEval. Optionally includes prior error for retry attempts."""
    base = (
        "Complete the following Python function. "
        "Return ONLY the function body (indented), no explanation, no markdown fences.\n\n"
        f"{row['prompt']}"
    )
    if error_context:
        base += f"\n\n# Previous attempt failed with:\n# {error_context}\n# Fix the implementation:"
    return base


def is_correct_humaneval(row: dict, full_code: str) -> bool:
    """Test whether full_code passes the HumanEval unit tests for this row."""
    from spectral_utils.agent_utils import execute_python_solution
    passed, _ = execute_python_solution(full_code, row["test"], row["entry_point"])
    return passed


# ── L-CiteEval ────────────────────────────────────────────────────────────────

_LCITEEVAL_CONFIG_MAP = {
    "hotpotqa":          "L-CiteEval-Data_hotpotqa",
    "natural_questions": "L-CiteEval-Data_natural_questions",
    "narrativeqa":       "L-CiteEval-Data_narrativeqa",
    "2wikimultihopqa":   "L-CiteEval-Data_2wikimultihopqa",
}


def _normalize_lciteeval_docs(row: dict) -> list:
    """
    Extract passage list from any L-CiteEval row format, returning [{title, text}].

    Handles three input shapes:
      1. row["docs"] is a list of dicts with title/text fields
      2. row["docs"] is a list of strings (ALCE format: "Title\\nText" or just text)
      3. row["context"] is a single string with passages separated by \\n\\n
    """
    docs_raw = row.get("docs", [])

    if isinstance(docs_raw, list) and docs_raw:
        out = []
        for i, d in enumerate(docs_raw):
            if isinstance(d, dict):
                out.append({
                    "title": str(d.get("title", f"Passage {i+1}")),
                    "text":  str(d.get("text", "")),
                })
            elif isinstance(d, str):
                # ALCE-style: first line is title, rest is body
                s = d.strip()
                if "\n" in s:
                    title, _, text = s.partition("\n")
                    out.append({"title": title.strip(), "text": text.strip()})
                else:
                    out.append({"title": f"Passage {i+1}", "text": s})
            else:
                out.append({"title": f"Passage {i+1}", "text": str(d)})
        return out

    if "context" in row:
        ctx = row["context"]
        if isinstance(ctx, str) and ctx:
            parts = [p.strip() for p in ctx.split("\n\n") if p.strip()]
            return [{"title": f"Passage {i+1}", "text": p} for i, p in enumerate(parts)]

    return []


def load_lciteeval(task: str = "hotpotqa", n_samples: int = 100) -> list:
    """
    Load L-CiteEval from HuggingFace, normalized to {question, docs, answers, raw_row}.

    docs: list of {title, text} dicts (1-indexed in the prompt).
    answers: list of acceptable gold answer strings.
    raw_row: original HF row (used by lciteeval_grounding_label for supporting_facts).
    """
    from datasets import load_dataset
    config_name = _LCITEEVAL_CONFIG_MAP.get(task, task)
    ds      = load_dataset("Jonaszky123/L-CiteEval", config_name, split="test")
    samples = [ds[i] for i in range(min(n_samples, len(ds)))]

    out = []
    for row in samples:
        docs    = _normalize_lciteeval_docs(row)
        q       = row.get("question", row.get("input", ""))
        answers = row.get("answers", [row.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]
        out.append({"question": q, "docs": docs, "answers": answers, "raw_row": dict(row)})

    print(f"Loaded {len(out)} L-CiteEval samples ({task}, config={config_name}).")
    return out


def lciteeval_prompt(row: dict, max_chars_per_doc: int = 600,
                     max_docs: int = 15) -> str:
    """
    Format a normalized L-CiteEval row for citation-grounded generation.

    Passages are numbered [1] … [N]; model must cite each statement.
    Truncates each passage to max_chars_per_doc to keep prompts manageable.
    """
    docs = row["docs"][:max_docs]
    passages = ""
    for i, d in enumerate(docs, 1):
        text = d["text"][:max_chars_per_doc].rstrip()
        passages += f"[{i}] {d['title']}\n{text}\n\n"

    return (
        "Read the following passages carefully. "
        "Answer the question with clear statements. "
        "After EACH statement, cite the passage(s) that support it using [number] format "
        "(e.g. 'Paris is the capital of France [1]. It has 2.1 million residents [2, 3].').\n\n"
        f"Passages:\n{passages}"
        f"Question: {row['question']}\n\n"
        "Your answer (include a citation after every statement):"
    )


def lciteeval_grounding_label(citation_ids: list, row: dict) -> int:
    """
    Label a parsed statement as grounded (1) or ungrounded (0).

    Primary: for HotpotQA sub-task, a statement is grounded if any cited passage
    title appears in the gold supporting_facts.
    Fallback: check if gold answer is a substring of any cited passage text.

    Args:
        citation_ids: 1-based passage indices from the model's citation markers.
        row: normalized row from load_lciteeval.
    """
    docs = row["docs"]
    raw  = row["raw_row"]

    # HotpotQA supporting_facts: [[title, sent_idx], ...] or HF dict format
    sf = raw.get("supporting_facts", [])
    sf_titles: set = set()
    if isinstance(sf, dict):
        sf_titles = set(sf.get("title", []))
    elif isinstance(sf, list) and sf:
        sf_titles = set(
            (x[0] if isinstance(x, (list, tuple)) else str(x))
            for x in sf
        )

    if sf_titles:
        for cid in citation_ids:
            idx = cid - 1
            if 0 <= idx < len(docs) and docs[idx].get("title", "") in sf_titles:
                return 1
        return 0

    # Fallback: gold answer substring in cited passages.
    # answers can be list[str] (HotpotQA) or list[list[str]] (NaturalQuestions,
    # NarrativeQA) — flatten to a single list of strings before matching.
    raw_answers = row.get("answers", [])
    answers: list = []
    for a in raw_answers:
        if isinstance(a, list):
            answers.extend(a)
        elif isinstance(a, str):
            answers.append(a)
    for cid in citation_ids:
        idx = cid - 1
        if 0 <= idx < len(docs):
            chunk_lower = docs[idx].get("text", "").lower()
            for ans in answers:
                if ans and isinstance(ans, str) and ans.lower().strip() in chunk_lower:
                    return 1
    return 0
