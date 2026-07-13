#!/usr/bin/env python
"""
smoke_preset.py — CPU-only pre-submit validation for a cluster preset.

Catches the *pure-CPU* pilot bugs (Qwen3 empty-`<think>`, OPT-30B few-shot/raw_prompt,
multimodal list-content `fmt_prompt`, grader / judge-parse) offline in seconds — so an
N=30 cluster pilot only ever fails on a genuine GPU/model issue, not on prompt/label logic.
Four of the six Step-163 pilot bugs were exactly this kind and each cost a full GPU round-trip.

It runs the preset's REAL `prompt_fn` / `grader` / judge helpers (imported from the same
source of truth the driver uses — `run_inference.DATASETS`, `spectral_utils.judge_utils`)
on hand-made fixtures. It never loads the model and never touches the dataset. The tokenizer
group is best-effort (needs `transformers` + model access); the grader and judge-parse groups
are hard and network-free.

Gate order (see CLAUDE.md): local smoke  ->  N=30 pilot  ->  full N.

Usage:
    python scripts/smoke_preset.py <preset_id> [<preset_id> ...]
    python scripts/smoke_preset.py --all
Exit code is nonzero iff a HARD check fails.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cluster"))  # so `presets` / `run_inference` resolve

from presets import PRESETS                                   # cluster/presets.py (pure data)
from run_inference import DATASETS                            # source-of-truth dataset -> triple
from spectral_utils.judge_utils import (                      # pure-python (no transformers)
    judge_prompt, _parse_decision, gold_answers_from_row,
)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


# ── Fixtures ──────────────────────────────────────────────────────────────────
# Grader fixtures are chosen so the expected label is grader-AGNOSTIC: exact-match
# (is_correct_trivia_qa) and ROUGE-L>0.3 (is_correct_trivia_qa_rougel) agree on all five,
# so the same table validates every trivia_qa-family preset. Each row exercises a real
# Step-163 failure mode.
def _trivia_row(alias, q="What is the capital of France?"):
    return {"question": q, "aliases": [alias], "answer_value": alias}


def _coqa_row(gold, q="What color was the cat?"):
    return {"question": q, "answers": [gold]}


GRADER_FIXTURES = {
    "trivia_qa_family": [
        ("Paris",                               _trivia_row("Paris"),  True,  "exact match"),
        ("London",                              _trivia_row("Paris"),  False, "wrong answer"),
        ("<think>\n\n</think>\n\nParis",         _trivia_row("Paris"),  True,  "Qwen3 empty-<think> stripped"),
        ("Ross Bagdasarian\n\nQuestion: Who created Alvin?",
         _trivia_row("Ross Bagdasarian", "Who created the Chipmunks?"), True,  "OPT-30B ramble -> first line only"),
        ("",                                    _trivia_row("Paris"),  False, "empty generation"),
    ],
    # GSM8K exact-match (is_correct_gsm8k): gold row carries "#### N"; boxed OR "final answer is"
    # both count. The <think>-then-boxed case guards the R1-Distill / Qwen3 reasoning cells.
    "gsm8k_family": [
        (r"The answer is \boxed{42}",           {"answer": "steps... #### 42"}, True,  "boxed correct"),
        (r"\boxed{7}",                          {"answer": "#### 42"},          False, "boxed wrong"),
        ("The final answer is 42",              {"answer": "#### 42"},          True,  "final-answer-is fallback (LapEigvals prompt)"),
        ("<think>maybe 10</think>\n\nThe answer is \\boxed{42}",
                                                {"answer": "#### 42"},          True,  "R1/Qwen3 <think> then boxed"),
        ("",                                    {"answer": "#### 42"},          False, "empty generation"),
    ],
    # MATH-500 (is_correct_math): gold row carries the boxed solution; numeric/frac equivalence.
    "math_family": [
        (r"Work...\boxed{42}",                  {"solution": r"...\boxed{42}"}, True,  "boxed correct"),
        (r"\boxed{7}",                          {"solution": r"\boxed{42}"},    False, "boxed wrong"),
        (r"<think>..</think>\boxed{\frac{1}{2}}", {"solution": r"\boxed{0.5}"}, True,  "<think> + \\frac numeric normalization"),
        ("",                                    {"solution": r"\boxed{42}"},    False, "empty generation"),
    ],
    # AMC23 (is_correct_amc23): gold row is {question, answer} — numeric boxed match via
    # is_correct_math first, falls back to a single-letter match on the last line only
    # when the gold answer itself is A-E (rare for AMC23's mostly-numeric answers).
    "amc23_family": [
        (r"Work...\boxed{25}",                  {"answer": "25"}, True,  "boxed correct (numeric)"),
        (r"\boxed{7}",                          {"answer": "25"}, False, "boxed wrong"),
        (r"<think>..</think>\boxed{25}",        {"answer": "25"}, True,  "<think> + boxed"),
        ("The answer is C",                     {"answer": "C"},  True,  "letter-match fallback (no boxed)"),
        ("",                                    {"answer": "25"}, False, "empty generation"),
    ],
    # AIME24 (is_correct_aime24 == is_correct_math): gold row is {question, answer}, integer
    # 0-999 boxed match.
    "aime24_family": [
        (r"Work...\boxed{204}",                 {"answer": "204"}, True,  "boxed correct"),
        (r"\boxed{7}",                          {"answer": "204"}, False, "boxed wrong"),
        (r"<think>..</think>\boxed{204}",       {"answer": "204"}, True,  "<think> + boxed"),
        ("",                                    {"answer": "204"}, False, "empty generation"),
    ],
    # CoQA (is_correct_coqa): ROUGE-L>0.3 vs item["answers"][0], first line only. Hand-verified
    # against rouge_l's LCS/F1 formula + _normalize_qa (lowercase, strip articles/punctuation):
    # "The cat was black." -> normalized "cat was black" vs "black" gives LCS=1, P=1/3, R=1,
    # F1=0.5 > 0.3 (true positive from a full-sentence answer, not just exact match).
    "coqa_family": [
        ("black",                               _coqa_row("black"), True,  "exact match"),
        ("The cat was black.",                  _coqa_row("black"), True,  "full-sentence answer, ROUGE-L 0.5>0.3"),
        ("white",                                _coqa_row("black"), False, "wrong answer, zero token overlap"),
        ("black\n\nQuestion: What did it eat?", _coqa_row("black"), True,  "multi-line ramble -> first line only"),
        ("",                                    _coqa_row("black"), False, "empty generation"),
    ],
}

# Judge-output parse fixtures (grader-agnostic). The critical one is 'incorrect', which
# CONTAINS 'correct' — the parser must test the negative first (_parse_decision does).
PARSE_FIXTURES = [
    ("Final Decision: correct",                    True,  "plain correct"),
    ("Final Decision: incorrect",                  False, "'incorrect' contains 'correct' (ordering guard)"),
    ("The answer is wrong.\nFinal Decision: incorrect", False, "reasoned incorrect"),
    ("Final Decision: Correct",                    True,  "capitalized"),
    ("I am not sure.",                             False, "unparseable -> conservative False"),
]


def _fixture_family(dataset):
    if dataset.startswith("trivia_qa"):
        return "trivia_qa_family"
    if dataset == "gsm8k":
        return "gsm8k_family"
    if dataset in ("math500", "math"):
        return "math_family"
    if dataset == "amc23":
        return "amc23_family"
    if dataset == "aime24":
        return "aime24_family"
    if dataset == "coqa":
        return "coqa_family"
    return None


# ── Check groups ──────────────────────────────────────────────────────────────
def check_grader(preset):
    """HARD: the preset's real grader classifies the fixtures correctly."""
    ds = preset["dataset"]
    fam = _fixture_family(ds)
    if fam is None:
        return [("grader", ds, SKIP, "no fixtures for this dataset family — add one to GRADER_FIXTURES")]
    _, _, grader = DATASETS[ds]
    out = []
    for gen, row, expected, desc in GRADER_FIXTURES[fam]:
        try:
            got = bool(grader(gen, row))
            status = PASS if got == expected else FAIL
            out.append(("grader", desc, status, f"grader={got} expected={expected}"))
        except Exception as e:
            out.append(("grader", desc, FAIL, f"raised {type(e).__name__}: {e}"))
    return out


def check_judge(preset):
    """HARD (only when the preset uses an LLM judge): prompt build strips <think> and
    embeds the gold; _parse_decision maps every canned verdict correctly."""
    if preset.get("judge") is None:
        return [("judge", "n/a", SKIP, "preset has no LLM judge (lexical/ROUGE-L grader)")]
    out = []
    q, gold = "What is the capital of France?", ["Paris"]
    think_answer = "<think>\nlet me think, it might be Lyon\n</think>\n\nParis"
    try:
        p = judge_prompt(q, think_answer, gold)
        checks = {
            "non-empty": bool(p and p.strip()),
            "contains gold 'Paris'": "Paris" in p,
            "<think> stripped from answer": "<think>" not in p,
            "has 'Final Decision' instruction": "Final Decision" in p,
        }
        for name, ok in checks.items():
            out.append(("judge.prompt", name, PASS if ok else FAIL, ""))
    except Exception as e:
        out.append(("judge.prompt", "build", FAIL, f"raised {type(e).__name__}: {e}"))
    for text, expected, desc in PARSE_FIXTURES:
        try:
            got = bool(_parse_decision(text))
            out.append(("judge.parse", desc, PASS if got == expected else FAIL,
                        f"parsed={got} expected={expected}"))
        except Exception as e:
            out.append(("judge.parse", desc, FAIL, f"raised {type(e).__name__}: {e}"))
    # gold extraction from a stored gold_row (dataset-agnostic path used at label time)
    try:
        g = gold_answers_from_row(_trivia_row("Paris"))
        out.append(("judge.gold", "gold_answers_from_row", PASS if "Paris" in g else FAIL, f"got={g}"))
    except Exception as e:
        out.append(("judge.gold", "gold_answers_from_row", FAIL, f"raised {type(e).__name__}: {e}"))
    return out


def check_prompt(preset):
    """SOFT: mirror the driver's prompt construction (run_temp + generate_full). Needs
    `transformers` + tokenizer access; SKIP (not FAIL) if unavailable — the load itself is
    what the N=30 pilot exists to test. When the tokenizer DOES load, a raised exception here
    is a real fmt_prompt / raw_prompt bug and fails."""
    ds = preset["dataset"]
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        return [("prompt", "tokenizer", SKIP, f"transformers unavailable locally ({type(e).__name__})")]
    try:
        tok = AutoTokenizer.from_pretrained(preset["model"])
    except Exception as e:
        return [("prompt", "tokenizer", SKIP, f"tokenizer load failed (gated/offline?): {type(e).__name__}")]
    try:
        _, prompt_fn, _ = DATASETS[ds]
        row = _trivia_row("Paris") if _fixture_family(ds) else {"question": "What is 2+2?"}
        msg = prompt_fn(row)
        if preset.get("prompt_suffix"):
            msg = f"{msg}{preset['prompt_suffix']}"
        if preset.get("raw_prompt"):
            prompt = msg                                     # base LM: no chat template
        else:
            from spectral_utils.model_utils import fmt_prompt
            prompt = fmt_prompt(tok, msg)
        ok = isinstance(prompt, str) and bool(prompt.strip())
        mode = "raw_prompt" if preset.get("raw_prompt") else "fmt_prompt"
        return [("prompt", mode, PASS if ok else FAIL, f"{len(prompt)} chars")]
    except Exception as e:
        return [("prompt", "build", FAIL, f"raised {type(e).__name__}: {e}")]


def smoke_one(preset_id):
    if preset_id not in PRESETS:
        print(f"  UNKNOWN preset '{preset_id}' — known: {', '.join(sorted(PRESETS))}")
        return False
    preset = PRESETS[preset_id]
    print(f"\n=== {preset_id} | {preset['model']} | dataset={preset['dataset']} | "
          f"judge={preset.get('judge')} | raw_prompt={preset.get('raw_prompt')} ===")
    rows = check_grader(preset) + check_judge(preset) + check_prompt(preset)
    hard_fail = 0
    for group, name, status, detail in rows:
        soft = group.startswith("prompt")                    # prompt group is best-effort
        if status == FAIL and not soft:
            hard_fail += 1
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[status]
        print(f"  [{mark}] {group:14s} {name:44s} {detail}")
    verdict = "PASS" if hard_fail == 0 else f"FAIL ({hard_fail} hard)"
    print(f"  ---> {preset_id}: {verdict}")
    return hard_fail == 0


def main():
    ap = argparse.ArgumentParser(description="CPU-only pre-submit validation for a cluster preset.")
    ap.add_argument("presets", nargs="*", help="preset id(s) to smoke-test")
    ap.add_argument("--all", action="store_true", help="smoke-test every preset")
    args = ap.parse_args()

    ids = sorted(PRESETS) if args.all else args.presets
    if not ids:
        ap.error("give one or more preset ids, or --all")
    ok_all = all([smoke_one(pid) for pid in ids])
    print(f"\n{'ALL PRESETS PASS' if ok_all else 'SOME PRESETS FAILED'} ({len(ids)} tested)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
