"""
uprm_baseline.py — the "LLM-as-a-Judge" control from "Unsupervised Process Reward Models"
(Gadetsky et al., EPFL, arXiv:2605.10158), Eq. (6). This is NOT uPRM itself — uPRM requires
training a new LoRA-tuned reward model via a bespoke RL objective (~44 GPU-hours on 8xH200,
gradient estimator only described in their Appendix B). This module reproduces the paper's own
cheap, non-joint, no-training control that uPRM is measured against in their Table 1
(49.8/42.8/29.4/26.6 F1 on GSM8K/MATH/OlympiadBench/Omni-MATH vs uPRM's 58.3/52.6/42.7/39.8).

THE MARKER SYMBOLS AND PROMPT FRAMING ARE OURS, NOT THEIRS (pre-registered, no code released)
------------------------------------------------------------------------------------------------
The paper defines the score abstractly (Eq 4/6: interleave steps with correctness markers "+"/
"-", read the LLM's next-token probabilities of those markers, renormalize over {+,-}) but
publishes no code and no exact prompt/marker surface form. We use the literal single-character
markers "+"/"-" (matching the paper's own notation exactly) and a short system message explaining
the convention (unavoidable — a raw solution with no framing gives the model no reason to treat
those characters as evaluative). Report any number from this module as "our LLM-as-a-Judge
reproduction," never as the paper's own reported score.

ONLY ONE FORWARD PASS PER TRAJECTORY IS NEEDED
------------------------------------------------
Eq (4)'s marked sequence s(tau,j) truncates right after step j; steps after j are not included.
But steps BEFORE j are always marked "+" for EVERY candidate j (Eq 4 marks y1..y_{j-1} as "+"
regardless of which j is being scored) — so the model's next-token distribution over {+,-} at
step boundary t is identical across every candidate j > t. One forward pass over the all-"+"
(fully-correct) sequence yields p+_t and p-_t at every boundary t=1..T simultaneously (reading
the {+,-} logits at each boundary; which token was actually fed next does not change what the
model predicted there), and S(j) for every candidate j=1..T+1 is a cumulative sum over those T
values — no additional forward passes needed.

THE MARKER IS NOT ALWAYS A SINGLE TOKEN — CAUGHT BY THIS MODULE'S OWN CHECK, NOT ASSUMED
-------------------------------------------------------------------------------------------
Verified empirically for Qwen3-8B (2026-08-10): " +" tokenizes alone as exactly one token (488),
but embedded before the step separator "\n\n" it BPE-merges into a single DIFFERENT token
(" +\n\n" -> 59454, not [488, "\n\n"...]), and " -\n\n" merges analogously to its own token
(21974). The two markers therefore need a *context-specific* (pos_id, neg_id) pair: one for a
step followed by STEP_SEP ("mid"), another for the last step (followed by whatever the chat
template renders after the assistant turn — which did NOT merge with the marker in this same
test). Both pairs are derived from the ACTUAL following text in the rendered conversation, never
hardcoded to one chat template, and every marker position's real token id is checked against its
context's expected id — a mismatch raises loudly rather than silently mis-scoring.
"""
import numpy as np

MARKER_POS = "+"
MARKER_NEG = "-"

SYSTEM_PROMPT = (
    "You will be shown a math problem and a proposed step-by-step solution. After each step, a "
    "marker has been inserted: \"+\" means the step is correct, \"-\" means the step is "
    "incorrect. Assess each step in order and predict the marker that belongs there."
)


def build_marked_chain(steps, sep: str = None):
    """All steps marked '+' (the j=T+1 / fully-correct sequence, Eq 5). Returns (text,
    marker_char_spans) where marker_char_spans[i] is the (start, end) of step i's MARKER
    CHARACTER (not the step text) inside `text`."""
    from .processbench import STEP_SEP
    sep = STEP_SEP if sep is None else sep
    parts, spans, pos = [], [], 0
    for i, s in enumerate(steps):
        if i:
            pos += len(sep)
        piece = f"{s} {MARKER_POS}"
        marker_offset = len(piece) - 1  # the '+' is the last character of piece
        spans.append((pos + marker_offset, pos + marker_offset + 1))
        parts.append(piece)
        pos += len(piece)
    return sep.join(parts), spans


def _marker_ids_for_context(tok, following_text: str):
    """Tokenize ' +'/' -' immediately followed by `following_text` (the real text that comes
    right after this marker in the rendered conversation) and take the FIRST resulting token of
    each — i.e. whatever the marker itself merges into (alone, or fused with some of the
    following characters), not the whole probe string. That first token is exactly the thing
    whose probability score_candidates reads off at the marker's position; anything the probe
    tokenizes to AFTER that first token is irrelevant here (it belongs to `following_text`, not
    to the marker), so the two probes are allowed to produce different total lengths."""
    pos_ids = tok.encode(f" {MARKER_POS}{following_text}", add_special_tokens=False)
    neg_ids = tok.encode(f" {MARKER_NEG}{following_text}", add_special_tokens=False)
    if not pos_ids or not neg_ids:
        raise ValueError(f"empty tokenization for following-context {following_text!r}")
    return pos_ids[0], neg_ids[0]


def score_candidates(mdl, tok, problem: str, steps: list) -> dict:
    """One forward pass -> {j: S(j)} for every candidate j in 1..len(steps)+1 (Eq 6)."""
    import torch
    from .processbench import STEP_SEP

    text, marker_spans = build_marked_chain(steps)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": text},
    ]
    conv = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tok(conv, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # The chat-template string is built from `text`, so a naive re-tokenization's offsets are
    # relative to the FULL rendered conversation, not `text` alone — locate `text` inside `conv`
    # first so marker_spans (relative to `text`) line up with `offsets` (relative to `conv`).
    text_start = conv.rfind(text)
    if text_start < 0:
        raise ValueError("rendered chat template does not contain the marked chain verbatim — "
                          "the tokenizer's chat template must not be transforming assistant "
                          "content (e.g. stripping/escaping) for this scorer to be valid")
    text_end = text_start + len(text)

    # Two following-contexts in practice: every non-final marker is followed by STEP_SEP; the
    # final marker is followed by whatever the chat template renders after the assistant turn.
    # 32 chars of real tail is ample for a short-range BPE merge (observed merges are 1-2 chars).
    tail = conv[text_end: text_end + 32]
    mid_pos_id, mid_neg_id = _marker_ids_for_context(tok, STEP_SEP)
    final_pos_id, final_neg_id = _marker_ids_for_context(tok, tail)

    marker_token_idx, marker_ctx = [], []
    for i, (a, b) in enumerate(marker_spans):
        a2, b2 = a + text_start, b + text_start
        idx = [k for k, (x, y) in enumerate(offsets) if x < b2 and y > a2 and y > x]
        if not idx:
            raise ValueError(f"marker span {(a, b)} did not map to any token")
        tidx = idx[-1]
        is_final = (i == len(steps) - 1)
        expected_pos = final_pos_id if is_final else mid_pos_id
        if input_ids[tidx] != expected_pos:
            raise ValueError(
                f"marker token at step {i} = {input_ids[tidx]} != expected {expected_pos} for "
                f"its context ({'final' if is_final else 'mid'}) — tokenization assumption "
                "broke for this tokenizer/template; do not trust this row's score"
            )
        marker_token_idx.append(tidx)
        marker_ctx.append((final_pos_id, final_neg_id) if is_final else (mid_pos_id, mid_neg_id))

    ids_t = torch.tensor([input_ids], device=mdl.device)
    with torch.no_grad():
        logits = mdl(input_ids=ids_t).logits[0]  # [T, V]

    log_p_pos, log_p_neg = [], []
    for tidx, (pos_id, neg_id) in zip(marker_token_idx, marker_ctx):
        # a model predicts the token AT tidx from the hidden state after token tidx-1
        pair_logits = logits[tidx - 1, [pos_id, neg_id]]
        logp = torch.log_softmax(pair_logits, dim=-1)
        log_p_pos.append(float(logp[0]))
        log_p_neg.append(float(logp[1]))

    T = len(steps)
    scores, cum_pos = {}, 0.0
    for j in range(1, T + 1):
        scores[j] = log_p_neg[j - 1] + cum_pos
        cum_pos += log_p_pos[j - 1]
    scores[T + 1] = cum_pos
    return scores


def localize_first_error(scores: dict) -> int:
    """argmax_j S(j) (paper's own decision rule); returns a 0-indexed step, or NO_ERROR (-1)
    when the T+1 ('all correct') candidate wins."""
    from .processbench import NO_ERROR
    n_candidates = len(scores)  # T+1 candidates, keys 1..T+1
    best_j = max(scores, key=lambda j: scores[j])
    return NO_ERROR if best_j == n_candidates else best_j - 1


def smoke() -> None:
    text, spans = build_marked_chain(["abc", "de"])
    assert text == "abc +\n\nde +", repr(text)
    # "abc +\n\nde +": '+' at index 4 (end of "abc +") and index 10 (end of "de +" starting at 7)
    assert spans == [(4, 5), (10, 11)], spans
    for (a, b) in spans:
        assert text[a:b] == "+", text[a:b]

    # localize_first_error: argmax over a hand-built score dict, both branches.
    assert localize_first_error({1: -0.1, 2: -5.0, 3: -3.0}) == 0        # j=1 wins -> step 0
    assert localize_first_error({1: -5.0, 2: -0.1, 3: -3.0}) == 1        # j=2 wins -> step 1
    assert localize_first_error({1: -5.0, 2: -3.0, 3: -0.1}) == -1       # j=T+1 wins -> NO_ERROR

    print("uprm_baseline.smoke: PASS (3 checks)")


if __name__ == "__main__":
    smoke()
