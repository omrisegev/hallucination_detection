"""
answer_span.py — crop a generation's token-aligned series to the span the GRADER reads.

WHY THIS EXISTS (Step 216)
--------------------------
`seiclr_triviaqa_opt30b` (OPT-30B, a *base* model) has no learned EOS for the
few-shot QA format, so it never stops on its own: 99.7% of its 5,000 generations
sit at exactly `max_new_tokens=64`, and 99.8% run past the answer and start
inventing a fresh `Question:` block. The median answer is 3 tokens — 4.7% of the
trace. `inside_coqa_llama7b` (LLaMA-7B base) has the same defect: a 2-token
answer inside a 256-token trace.

Every feature in the pool is an aggregate over the token series, so on those cells
~95% of what each feature measures is the model's fabricated continuation rather
than its answer. No fusion, selector, or orientation rule can recover from that —
it is a generation defect upstream of the method.

The asymmetry that makes this a *bug* rather than a design choice: the grader
already crops. `is_correct_trivia_qa_rougel` scores `first_answer_line(gen)`, so
the LABEL is computed on the answer while the FEATURES were computed on all 64
tokens. This module removes that inconsistency by giving the feature extractor the
same span the grader uses.

WHAT THIS IS NOT
----------------
Not semantic. `answer_span` is a pure text-boundary rule — strip a `<think>`
block, take the first non-empty line — identical to `first_answer_line`, applied
to character offsets instead of to a string. Nothing reads the answer's meaning,
so the detector keeps its probability-only property.

NO RE-GENERATION IS NEEDED. Decoding is autoregressive, so `token_entropies[i]`
conditions only on tokens < i. Truncating a suffix offline is therefore
bit-identical to having passed a stop sequence at generation time — the retained
prefix's per-token distributions do not depend on tokens that came after it.
"""
import re

import numpy as np

# Token-aligned per-step series in the raw pkl schema. `top_k_logprobs` is handled
# separately (it is a dict of two [T, K] arrays, not a flat list).
_TOKEN_SERIES = ("token_entropies", "token_spilled_energies", "token_logsumexp",
                 "gen_token_ids")

# ── which cells get cropped ───────────────────────────────────────────────────
# A cell qualifies only if all three hold (measured over its raw pkl by
# `scripts/answer_span_audit.py`):
#   (a) SHORT-ANSWER FORMAT — its grader reads `first_answer_line`, so the answer
#       is the first line by construction. Cells that reason first and answer last
#       (every math cell, and `losnet_hotpotqa_mistral7b`) fail this: cropping to
#       the first line would keep the preamble and throw the ANSWER away.
#   (b) THE ANSWER IS A MINORITY OF THE TRACE — answer tokens < 50% of the trace.
#       `truthfulqa_llama8b` sits at 95.7%; there is nothing to crop.
#   (c) THE CROPPED SPAN IS ACTUALLY AN ANSWER on nearly every row. This is the
#       one that matters. `inside_coqa_llama7b` passes (a) and (b) but 45.1% of
#       its first lines are `[/INST]` echoes, fabricated `Question:` turns, raw
#       passage text, or empty — a Mistral chat template applied to a LLaMA-7B
#       *base* checkpoint (the preset never set `raw_prompt=True`). That is a
#       prompt defect upstream of anything an offline crop can reach, so the cell
#       is NOT cropped; it needs re-generation. See HISTORY Step 216.
#
# Measured rates for (c): seiclr_triviaqa_opt30b 0.0%, epr_triviaqa_mistral24b
# 0.0%, semenergy_triviaqa_qwen3_8b 0.0%, truthfulqa_llama8b 0.0%,
# inside_coqa_llama7b 45.1%.
RUNON_CELLS = ("seiclr_triviaqa_opt30b",)

# Cells with the run-on symptom whose defect cropping cannot fix -> re-generate.
UNREPAIRABLE_CELLS = {
    "inside_coqa_llama7b":
        "chat template applied to a base checkpoint (raw_prompt not set); 45.1% of "
        "first lines are [/INST] echoes / fabricated Question: turns / empty",
}

_UNUSABLE_FIRST_LINE = re.compile(r"^\s*$|\[/?INST\]|^\s*(Question|Q)\s*:|^\s*#{2,}|^\s*\[")


def answer_span(text):
    """(start, end) character offsets of `first_answer_line(text)` inside `text`.

    Mirrors `spectral_utils.data_loaders.first_answer_line` exactly, but returns
    offsets rather than the substring so the token series can be cut with it.
    Returns an empty span (start == end) when there is no non-empty line.
    """
    t = text or ""
    s, e = 0, len(t)
    # `strip_think` drops <think>...</think>; on these models the block is a
    # prefix, so the answer starts after the last close tag.
    if "</think>" in t:
        s = t.rindex("</think>") + len("</think>")
    elif "<think>" in t:
        e = t.index("<think>")

    pos = s
    for line in t[s:e].split("\n"):
        if line.strip():
            a = pos + (len(line) - len(line.lstrip()))
            return a, pos + len(line.rstrip())
        pos += len(line) + 1
    return s, s


def answer_token_slice(cand):
    """(lo, hi) token indices covering the answer span, or None if unmappable.

    Unmappable means no `token_offsets` (older caches) or an empty answer — in
    both cases the caller should leave the candidate alone rather than guess.
    """
    off = cand.get("token_offsets")
    if not off:
        return None
    a, b = answer_span(cand.get("full_text", ""))
    if b <= a:
        return None
    idx = [i for i, (x, y) in enumerate(off) if x < b and y > a]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


def crop_candidate(cand, min_tokens=1):
    """Candidate dict with every token-aligned series cut to the answer span.

    Returns the candidate UNCHANGED when the span is unmappable or shorter than
    `min_tokens`, so a cell can be cropped without silently losing rows to a
    parsing miss. `full_text` and the labels are left as they are — this only
    changes what the FEATURES are computed over.
    """
    sl = answer_token_slice(cand)
    if sl is None:
        return cand
    lo, hi = sl
    if hi - lo < min_tokens:
        return cand

    out = dict(cand)
    for k in _TOKEN_SERIES:
        v = cand.get(k)
        if v is not None and len(v) >= hi:
            out[k] = list(np.asarray(v)[lo:hi])
    tk = cand.get("top_k_logprobs")
    if isinstance(tk, dict) and "logprobs" in tk:
        lp = np.asarray(tk["logprobs"])
        if lp.ndim == 2 and lp.shape[0] >= hi:
            out["top_k_logprobs"] = {"ids": np.asarray(tk["ids"])[lo:hi],
                                     "logprobs": lp[lo:hi]}
    out["_cropped"] = (lo, hi)
    return out


def runon_stats(cand, cap=None):
    """Per-candidate diagnostics behind criteria (b) and (c) above."""
    n = len(cand.get("token_entropies") or [])
    sl = answer_token_slice(cand)
    ans = (sl[1] - sl[0]) if sl else None
    text = cand.get("full_text", "")
    a, b = answer_span(text)
    first = text[a:b]
    return {"n_tokens": n, "n_answer": ans,
            "ans_frac": (ans / n) if (ans is not None and n) else None,
            "at_cap": (n >= cap) if cap else None,
            "unusable": bool(_UNUSABLE_FIRST_LINE.match(first)) or not first.strip()}
