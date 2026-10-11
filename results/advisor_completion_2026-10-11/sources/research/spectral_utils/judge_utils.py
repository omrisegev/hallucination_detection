"""
LLM-as-judge correctness labeling for the replication grid.

Several competitor papers grade answer correctness with an LLM judge, not lexical match:
  - EPR              -> Gemma-3-12b-it
  - Semantic Energy  -> TIGER-Lab/general-verifier

To make "same scenario, only the method differs" hold, we label our re-run generations
with the SAME judge the paper used, so our AUROC (X) and the paper's published AUROC (Y)
share a correctness definition. This module produces LABELS (ground truth) — it does NOT
run any hallucination-detection method.

Runs on the cluster as a second in-job pass: the target model generates + captures, is
freed, then the judge is loaded and labels every candidate. Resumable: candidates already
marked ``label_judged`` are skipped, so a preempted+requeued job continues cleanly.
"""
import re

import torch

from .model_utils import load_model


def load_judge(model_id: str):
    """Load a judge model + tokenizer.

    Handles multimodal checkpoints (e.g. Gemma-3-12b-it is Gemma3ForConditionalGeneration,
    not a plain CausalLM): if the CausalLM path fails, fall back to the image-text-to-text
    class and drive it text-only. The text tokenizer + text-only generate work regardless.
    """
    try:
        return load_model(model_id)
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"[judge] load_model({model_id}) failed ({type(e).__name__}: {e}); "
              f"trying the multimodal text-judge path", flush=True)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            from transformers import AutoModelForImageTextToText as _AM
        except ImportError:
            from transformers import AutoModelForCausalLM as _AM
        mdl = _AM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto",
                                  attn_implementation="eager")
        mdl.eval()
        print(f"[judge] loaded {model_id} via {_AM.__name__} (text-only judging)", flush=True)
        return mdl, tok


def _judge_render(tok, text: str) -> str:
    """Apply the judge's chat template robustly. Tries plain-string content (most models),
    then multimodal list-content (Gemma-3 / processors), then a raw fallback."""
    for content in (text, [{"type": "text", "text": text}]):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            continue
    return f"{text}\n"


# One judge instruction for all papers — a factual-equivalence check that outputs a single
# parseable decision line. Kept deliberately hint-free (no chain-of-thought, no dataset cues)
# so the label reflects only answer correctness.
JUDGE_TEMPLATE = (
    "You are judging whether a student's answer to a question is correct.\n"
    "The student's answer is CORRECT if it states the same factual answer as ANY of the "
    "reference answers, even if phrased differently, abbreviated, or with extra words. "
    "Otherwise it is INCORRECT.\n\n"
    "Question: {question}\n"
    "Reference answer(s): {refs}\n"
    "Student's answer: {answer}\n\n"
    "Respond with exactly one line, either:\n"
    "Final Decision: correct\n"
    "or\n"
    "Final Decision: incorrect"
)


def gold_answers_from_row(row) -> list:
    """Best-effort gold-answer list from a stored gold_row (dataset-agnostic)."""
    if not isinstance(row, dict):
        return []
    for key in ("aliases", "answers"):
        v = row.get(key)
        if v:
            return [str(x) for x in v]
    for key in ("answer_value", "answer", "value"):
        v = row.get(key)
        if v:
            return [str(v)]
    return []


def judge_prompt(question: str, answer: str, gold_answers) -> str:
    from .data_loaders import first_answer_line  # strips <think> blocks (Qwen3) too
    refs = "; ".join(str(g) for g in (gold_answers or []) if str(g).strip()) or "(none)"
    ans = first_answer_line(answer or "")[:300] or "(empty)"
    return JUDGE_TEMPLATE.format(question=str(question)[:500], refs=refs[:500], answer=ans)


def _parse_decision(text: str) -> bool:
    """Parse the judge output to a bool. NOTE: 'incorrect' contains 'correct', so the
    negative must be tested first. Unparseable -> False (conservative incorrect)."""
    t = (text or "").lower()
    m = re.search(r"final decision\s*:?\s*(.*)", t, flags=re.DOTALL)
    tail = m.group(1) if m else t
    for probe in (tail, t):
        if "incorrect" in probe or "not correct" in probe or "wrong" in probe:
            return False
        if "correct" in probe or re.search(r"\byes\b", probe):
            return True
    return False


@torch.no_grad()
def judge_correct(mdl, tok, question: str, answer: str, gold_answers,
                  max_new_tokens: int = 8) -> bool:
    """Greedy one-line judge verdict -> bool. Lightweight (no capture)."""
    prompt = _judge_render(tok, judge_prompt(question, answer, gold_answers))
    enc = tok(prompt, return_tensors="pt").to(next(mdl.parameters()).device)
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_decision(gen)


def judge_label_cache(cache, mdl, tok, gold_fn=gold_answers_from_row,
                      stop_flag=None, checkpoint=None, checkpoint_every=50,
                      on_progress=None) -> int:
    """
    Relabel every candidate in a run cache with the judge.

    For each candidate: keep the prior lexical label as ``label_lexical``, set ``label`` to
    the judge verdict, and mark ``label_judged=True``. Idempotent — already-judged
    candidates are skipped, so a requeued job resumes. Honors ``stop_flag['flag']`` (SIGTERM)
    by checkpointing and returning early.

    Args:
        cache:       {idx: {question, gold_row, candidates:[...]}} run cache (mutated in place).
        gold_fn:     gold_row -> list[str] gold answers.
        checkpoint:  zero-arg callable that persists `cache` (e.g. save_cache_atomic partial).
    Returns number of candidates labeled this call.
    """
    n = 0
    for idx in sorted(cache.keys()):
        entry = cache[idx]
        q = entry.get("question", "")
        gold = gold_fn(entry.get("gold_row", {}))
        for c in entry["candidates"]:
            if c.get("label_judged"):
                continue
            if stop_flag is not None and stop_flag.get("flag"):
                if checkpoint:
                    checkpoint()
                return n
            c["label_lexical"] = bool(c.get("label", False))
            c["label"] = bool(judge_correct(mdl, tok, q, c.get("full_text", ""), gold))
            c["label_judged"] = True
            n += 1
            if on_progress:
                on_progress(idx, n)
            if checkpoint and (n % checkpoint_every == 0):
                checkpoint()
    if checkpoint:
        checkpoint()
    return n
