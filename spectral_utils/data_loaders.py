"""
Dataset loaders and grading functions for all benchmarks used in the project.

Supported datasets:
  - GSM8K       (LapEigvals Listing 5 prompt; exact-match grading)
  - MATH-500    (competition math; boxed answer extraction)
  - GPQA Diamond (graduate-level MCQ; letter extraction)
  - HotpotQA    (multi-hop QA; substring match)
"""
import re
import string

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
