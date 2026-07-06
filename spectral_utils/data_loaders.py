"""
Dataset loaders and grading functions for all benchmarks used in the project.

Supported datasets:
  - GSM8K       (LapEigvals Listing 5 prompt; exact-match grading)
  - MATH-500    (competition math; boxed answer extraction)
  - AMC23       (AMC 2023 competition; numeric/letter grading)
  - AIME24      (AIME 2024 competition; integer exact-match grading)
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


def gsm8k_prompt(question) -> str:
    """Format a GSM8K question asking for \\boxed{} answer (matches Qwen-Math output format)."""
    if isinstance(question, dict):
        question = question.get("question", "")
    return (
        "Solve the following problem. "
        "Show all your work step by step, then give your final answer in \\boxed{}.\n\n"
        "Problem: " + question
    )


def gsm8k_prompt_with_conf(question) -> str:
    """GSM8K prompt that asks for answer + confidence in one generation pass."""
    if isinstance(question, dict):
        question = question.get("question", "")
    return (
        "Solve the following problem. "
        "Show all your work step by step, then give your final answer in \\boxed{}. "
        "After the boxed answer, on a new line write exactly: "
        "Confidence: X  (where X is an integer 0-100 reflecting how sure you are).\n\n"
        "Problem: " + question
    )


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


def normalize_gsm8k(s) -> float | str | None:
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"[\$,%]", "", s).replace(",", "").rstrip(".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return s.lower().strip()


def is_correct_gsm8k(gen: str, gold_answer) -> bool:
    """Exact-match grading for GSM8K. Tries \\boxed{} first (Qwen-Math style),
    then falls back to 'The final answer is [X]' (LapEigvals style)."""
    if isinstance(gold_answer, dict):
        gold_answer = gold_answer.get("answer", "")
    gold_norm = normalize_gsm8k(extract_gold_gsm8k(gold_answer))

    # Primary: boxed answer (Qwen-Math and most modern math models).
    # _extract_boxed handles nested braces; the old first-'}' regex truncated them.
    boxed = _extract_boxed(gen)
    if boxed is not None:
        model_norm = normalize_gsm8k(re.sub(r"[^\d\.\-\/]", "", _normalize_math_answer(boxed)))
        if model_norm is not None:
            if isinstance(gold_norm, float) and isinstance(model_norm, float):
                if abs(gold_norm - model_norm) < 1e-6:
                    return True
            elif str(gold_norm) == str(model_norm):
                return True

    # Fallback: "The final answer is [X]" (LapEigvals / older prompt format)
    model_norm = normalize_gsm8k(extract_model_answer_gsm8k(gen))
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


def _extract_boxed(text: str):
    """
    Return the balanced-brace content of the LAST \\boxed{...} in text, or None.

    Replaces the old regex r"\\boxed\\{([^}]*)\\}" which stopped at the first '}'
    and truncated nested LaTeX (\\boxed{\\frac{1}{2}} -> "\\frac{1") — the Phase-13
    7.7%-accuracy grading bug. The LAST boxed is used because models sometimes box
    intermediate results; the final answer is the last one.
    """
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    depth, out = 1, []
    for c in text[start + len("\\boxed{"):]:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
    # depth > 0 = generation truncated mid-answer; return best-effort content
    return "".join(out) if out else None


def _normalize_math_answer(val: str) -> str:
    """Normalize a boxed LaTeX answer for numeric/string comparison."""
    val = val.replace(",", "").replace("$", "").strip()
    val = re.sub(r"\\text(?:bf|it|rm)?\{([^{}]*)\}", r"\1", val)
    val = re.sub(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", val)
    val = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ ", "", val).strip()
    # pure numeric fraction -> decimal so is_correct_math's float path can compare
    m = re.fullmatch(r"\((-?\d+(?:\.\d+)?)\)/\((-?\d+(?:\.\d+)?)\)", val)
    if m and float(m.group(2)) != 0:
        return repr(float(m.group(1)) / float(m.group(2)))
    cleaned = re.sub(r"[^\d\.\-\/\(\)]", "", val)
    return cleaned if cleaned else val


def _extract_math_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed is not None:
        val = _normalize_math_answer(boxed)
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


# ── AMC 2023 ──────────────────────────────────────────────────────────────────

def load_amc23(n_samples: int = 40) -> list[dict]:
    """
    Load AMC 2023 competition problems from HuggingFace.

    Tries several parquet-backed sources (no trust_remote_code needed).
    Each source contains exactly the 40-problem canonical AMC23 eval set.
    Returns list of {question, answer} dicts.
    """
    from datasets import load_dataset
    # (path, kwargs, split) — all parquet-backed, verified 2026-06-03
    attempts = [
        ("math-ai/amc23",                  {}, "test"),   # 40 rows; question, answer, url
        ("zwhe99/amc23",                   {}, "test"),   # 40 rows; answer is float-string e.g. "27.0"
        ("knoveleng/AMC-23",               {}, "train"),  # 40 rows; problem/question + answer
        ("meoconxinhxan/eval_math_amc23",  {}, "train"),  # 40 rows; problem, answer
    ]
    for path, kwargs, split in attempts:
        try:
            ds   = load_dataset(path, split=split, **kwargs)
            cols = set(ds.column_names)
            items: list[dict] = []
            for i in range(len(ds)):
                row = ds[i]
                q = row.get("question", row.get("problem", row.get("Problem", row.get("query", ""))))
                a = row.get("answer", row.get("Answer", row.get("solution", "")))
                items.append({"question": str(q).strip(), "answer": str(a).strip()})
                if len(items) >= n_samples:
                    break
            if items:
                print(f"Loaded {len(items)} AMC23 problems from {path}.")
                return items
            print(f"  {path}: 0 matching rows (columns: {sorted(cols)[:8]})")
        except Exception as e:
            print(f"  {path} ({split}) failed: {e}")
    raise RuntimeError("Could not load AMC23 from any HF source.")


def amc23_prompt(row: dict) -> str:
    """Format an AMC23 row as a chain-of-thought math prompt."""
    q = row.get("question", row.get("problem", ""))
    return (
        "Solve the following AMC competition math problem. "
        "Show all your work step by step, then give your final answer in \\boxed{}.\n\n"
        + q
    )


def is_correct_amc23(gen: str, gold_row: dict) -> bool:
    """Grade an AMC23 response. Tries numeric boxed-answer match, then letter match."""
    if is_correct_math(gen, gold_row):
        return True
    ans = gold_row.get("answer", "").strip().upper()
    if ans in "ABCDE" and len(ans) == 1:
        last_line = gen.strip().split("\n")[-1].upper()
        return ans in last_line
    return False


# ── AIME 2024 ─────────────────────────────────────────────────────────────────

def load_aime24(n_samples: int = 30) -> list[dict]:
    """
    Load AIME 2024 competition problems from HuggingFace.

    Tries several sources; AIME I + II combined gives ~30 problems.
    Returns list of {question, answer} dicts.
    """
    from datasets import load_dataset
    attempts = [
        ("Maxwell-Jia/AIME_2024", {}, "train"),   # 30 rows; Problem, Answer, Solution — verified 2026-06-03
        ("open-r1/AIME2024",      {}, "test"),
        ("math-ai/AIME2024",      {}, "test"),
    ]
    for path, kwargs, split in attempts:
        try:
            ds   = load_dataset(path, split=split, **kwargs)
            cols = set(ds.column_names)
            items: list[dict] = []
            for i in range(len(ds)):
                row = ds[i]
                q = row.get("Problem", row.get("problem", row.get("question", row.get("query", ""))))
                a = row.get("Answer", row.get("answer", row.get("solution", "")))
                items.append({"question": str(q).strip(), "answer": str(a).strip()})
                if len(items) >= n_samples:
                    break
            if items:
                print(f"Loaded {len(items)} AIME24 problems from {path}.")
                return items
            print(f"  {path}: 0 matching rows (columns: {sorted(cols)[:8]})")
        except Exception as e:
            print(f"  {path} ({split}) failed: {e}")
    raise RuntimeError("Could not load AIME24 from any HF source.")


def aime24_prompt(row: dict) -> str:
    """Format an AIME24 row as a chain-of-thought math prompt."""
    q = row.get("question", row.get("problem", ""))
    return (
        "Solve the following AIME competition math problem. "
        "Your final answer must be an integer from 0 to 999. "
        "Show all your work step by step, then give your final answer in \\boxed{}.\n\n"
        + q
    )


def is_correct_aime24(gen: str, gold_row: dict) -> bool:
    """Grade an AIME24 response. Answers are integers 0–999; uses boxed answer extraction."""
    return is_correct_math(gen, gold_row)


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
                     max_docs: int = 15, variant: int = 0) -> str:
    """
    Format a normalized L-CiteEval row for citation-grounded generation.

    Passages are numbered [1] … [N]; model must cite each statement.
    Truncates each passage to max_chars_per_doc to keep prompts manageable.

    variant=0 (baseline): direct answer with citations.
    variant=1: explicit reasoning preamble before answer.
    variant=2: "Think through" framing to encourage CoT.
    variant=3: explain why each citation supports the claim.
    variant=4: evaluate passage relevance before answering.
    """
    docs = row["docs"][:max_docs]
    passages = ""
    for i, d in enumerate(docs, 1):
        text = d["text"][:max_chars_per_doc].rstrip()
        passages += f"[{i}] {d['title']}\n{text}\n\n"

    base = (
        "Read the following passages carefully. "
        "After EACH statement, cite the passage(s) that support it using [number] format "
        "(e.g. 'Paris is the capital of France [1]. It has 2.1 million residents [2, 3].').\n\n"
        f"Passages:\n{passages}"
        f"Question: {row['question']}\n\n"
    )

    if variant == 0:
        instruction = (
            "Answer the question with clear statements. "
        )
        closing = "Your answer (include a citation after every statement):"

    elif variant == 1:
        # Explicit reasoning preamble → longer traces
        instruction = (
            "Answer the question with clear statements, "
            "starting with your reasoning process and ending with the answer. "
        )
        closing = "Your reasoning and answer (include a citation after every statement):"

    elif variant == 2:
        # "Think through" framing
        instruction = (
            "Think through the question step by step, then provide your answer with clear statements. "
        )
        closing = "Your step-by-step reasoning and answer (include a citation after every statement):"

    elif variant == 3:
        # Explain why each citation supports the claim
        instruction = (
            "Answer the question with clear statements, "
            "briefly explaining why each cited passage supports your claim before stating it. "
        )
        closing = "Your answer (explain each citation, then state each supported claim):"

    elif variant == 4:
        # Evaluate passage relevance first
        instruction = (
            "Consider whether the passages clearly answer the question, then answer "
            "with clear statements. "
        )
        closing = "Your answer (note which passages are most relevant, then answer with citations):"

    else:
        raise ValueError(f"Unknown lciteeval_prompt variant: {variant}. Choose 0–4.")

    return base + instruction + "\n" + closing


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
