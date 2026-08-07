"""
worked_example.py — extract one reasoning trace, marked up by the detector, for the report figure.

This is the thing Omri asked to see: a real answer rendered as text, with each step tinted by the
fused risk and the token-level evidence curve beneath it, drops annotated on the tokens they land
on — the anatomy of "Mind the Gap"'s Figure 4.

WHAT IT CAN AND CANNOT SHOW, PER SOURCE
---------------------------------------
* **ProcessBench rows** carry `label` = the index of the first *annotated* erroneous step, so the
  figure can show the detector's pick beside the ground truth and the reader can see a hit or a
  miss. That is the primary panel.
* **Our own generated answers** (the `evdrop_*` cells) are labelled only at the answer level —
  correct or not — with no per-step annotation anywhere. So for these the figure can honestly
  show *where the detector fires*, and contrast a wrong answer against a right one, but it cannot
  show a ground-truth step. Saying otherwise would be inventing an annotation. These are the
  secondary panels and are labelled as such.

THE STEPS ARE THE SAME CONSTRUCT IN BOTH CASES
----------------------------------------------
ProcessBench ships pre-segmented steps joined by `processbench.STEP_SEP` ("\\n\\n"). A generated
answer is segmented by splitting on the same separator, so "step" means one paragraph of
reasoning either way, and the token->step mapping goes through `token_offsets` rather than being
re-tokenized (re-tokenizing a substring does not reproduce the offsets it had in context).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from our_arm import CANONICAL_POOL, step_feature_rows, upcr_arm_fit  # noqa: E402
from spectral_utils.processbench import STEP_SEP                     # noqa: E402
from token_trace import trace_row, worst_drops                       # noqa: E402

MIN_STEP_TOKENS = 8      # compute_spectral_features' floor; below this a step is unmeasurable


def split_steps(text, sep: str = STEP_SEP):
    """(steps, char_spans) for a generated answer, splitting on the ProcessBench separator.

    Empty segments are dropped but their characters are still accounted for, so the returned
    spans stay exact offsets into `text` rather than offsets into a rejoined copy.
    """
    steps, spans, pos = [], [], 0
    for piece in text.split(sep):
        start = pos
        pos += len(piece) + len(sep)
        if piece.strip():
            steps.append(piece)
            spans.append((start, start + len(piece)))
    return steps, spans


def char_to_token_spans(token_offsets, char_spans, n_tokens=None):
    """Map character spans onto half-open token ranges using the saved offsets.

    `token_offsets` is one (start, end) pair per token and is documented to run 1-2 short of the
    token arrays (measured here: 283 offsets against 284 entropies), so the last span is extended
    to `n_tokens` rather than truncating the final step off the figure.
    """
    starts = np.array([o[0] for o in token_offsets], dtype=int)
    ends = np.array([o[1] for o in token_offsets], dtype=int)
    n_off = len(starts)
    out = []
    for c0, c1 in char_spans:
        lo = int(np.searchsorted(ends, c0, side="right"))
        hi = int(np.searchsorted(starts, c1, side="left"))
        lo, hi = max(0, min(lo, n_off)), max(0, min(hi, n_off))
        out.append((lo, hi) if hi > lo else None)
    if out and n_tokens and out[-1] is not None:
        lo, _ = out[-1]
        out[-1] = (lo, int(n_tokens))
    return out


def as_row(cand, sep: str = STEP_SEP):
    """Turn a generated-answer candidate into the row shape the step/token machinery consumes."""
    text = cand["full_text"]
    steps, char_spans = split_steps(text, sep)
    n_tok = len(cand["token_entropies"])
    spans = char_to_token_spans(cand["token_offsets"], char_spans, n_tokens=n_tok)
    return {
        "problem": cand.get("question", ""),
        "full_text": text,
        "steps": steps,
        "char_spans": char_spans,
        "step_token_spans": spans,
        "label": -1,                       # generated answers carry NO step annotation
        "answer_correct": bool(cand.get("label", False)),
        "token_entropies": cand["token_entropies"],
        "token_spilled_energies": cand.get("token_spilled_energies"),
        "top_k_logprobs": cand.get("top_k_logprobs"),
    }


def fit_step_arm(rows):
    """U-PCR fitted on the pooled steps of these rows — the SAME construction ProcessBench uses.

    Fitting on pooled steps rather than on whole answers is deliberate: it makes the figure's
    detector procedurally identical to `score_processbench.py`'s step arm, so the picture built
    from our own generations previews exactly what the ProcessBench table will report.
    """
    per_row = [step_feature_rows(r, feat_names=CANONICAL_POOL) for r in rows]
    flat = [s for r in per_row for s in r]
    fd = {f: np.array([s.get(f, np.nan) for s in flat], dtype=float) for f in CANONICAL_POOL}
    return upcr_arm_fit(fd, labels=None), per_row


def build_example(row, arm, W=32, stride=1, ema_span=5, n_drops=3):
    """Everything one figure panel needs, with nothing left to be recomputed at render time."""
    tr = trace_row(row, arm, W=W, stride=stride, ema_span=ema_span)
    risk = np.asarray(tr["step_risk"], dtype=float)
    finite = np.isfinite(risk)
    drops = worst_drops(tr, k=n_drops)

    # Which step each annotated drop lands in — attribution rule 2 (the step containing token j+1).
    drop_steps = []
    for j, val in drops:
        hit = None
        for i, span in enumerate(row["step_token_spans"]):
            if span and span[0] <= j + 1 < span[1]:
                hit = i
                break
        drop_steps.append({"token": int(j), "flux": float(val), "step": hit})

    return {
        "text": row["full_text"],
        "steps": row["steps"],
        "char_spans": row["char_spans"],
        "step_token_spans": row["step_token_spans"],
        "step_risk": risk,
        "peak_step": int(np.nanargmax(risk)) if finite.any() else None,
        "label": int(row["label"]),
        "answer_correct": bool(row.get("answer_correct", False)),
        "evidence": tr["evidence_filled"],
        "smoothed": tr["smoothed"],
        "flux": tr["flux"],
        "drops": drop_steps,
        "params": tr["params"],
    }


# ── known-answer tests ───────────────────────────────────────────────────────

def smoke() -> None:
    # 1. Splitting is exact: every returned span indexes back to its own step text.
    text = "First part.\n\nSecond part here.\n\n\n\nThird."
    steps, spans = split_steps(text)
    assert steps == ["First part.", "Second part here.", "Third."], steps
    for s, (a, b) in zip(steps, spans):
        assert text[a:b] == s, (s, text[a:b])

    # 2. Char -> token mapping on a hand-countable tokenizer: one token per character.
    offs = [(i, i + 1) for i in range(len(text))]
    tspans = char_to_token_spans(offs, spans, n_tokens=len(text))
    assert tspans[0] == (0, 11), tspans[0]
    assert tspans[1][0] == 13, tspans[1]
    assert tspans[-1][1] == len(text), tspans[-1]
    # contiguous and non-overlapping, in order
    for (a0, a1), (b0, b1) in zip(tspans, tspans[1:]):
        assert a1 <= b0, (a1, b0)

    # 3. The documented offsets/entropies skew does not truncate the last step.
    short = offs[:-2]
    t2 = char_to_token_spans(short, spans, n_tokens=len(text))
    assert t2[-1][1] == len(text), \
        "the last step must extend to n_tokens, not stop at the shorter offsets array"

    # 4. End to end on a synthetic candidate, with a planted entropy burst in a known step.
    rng = np.random.default_rng(0)
    para = ["word " * 40, "word " * 40, "word " * 40, "word " * 40]
    body = STEP_SEP.join(p.strip() for p in para)
    n_tok = len(body)
    ent = np.full(n_tok, 0.3)
    _, cs = split_steps(body)
    bad = 2
    ent[cs[bad][0]:cs[bad][1]] = rng.uniform(2.0, 3.0, cs[bad][1] - cs[bad][0])
    cand = {"full_text": body, "token_entropies": ent.tolist(),
            "token_spilled_energies": None, "token_offsets": [(i, i + 1) for i in range(n_tok)],
            "label": False}
    row = as_row(cand)
    assert len(row["steps"]) == 4, row["steps"]
    assert all(s is not None for s in row["step_token_spans"]), row["step_token_spans"]

    from token_trace import _toy_arm
    ex = build_example(row, _toy_arm())
    assert ex["peak_step"] == bad, f"planted the burst in step {bad}, peak risk at {ex['peak_step']}"
    assert ex["drops"] and ex["drops"][0]["flux"] < 0, ex["drops"]
    assert len(ex["evidence"]) == n_tok, (len(ex["evidence"]), n_tok)
    assert ex["label"] == -1, "a generated answer must not claim a step annotation"

    print(f"worked_example.smoke: PASS (4 checks)  [peak step {ex['peak_step']}/4, "
          f"worst drop at token {ex['drops'][0]['token']}]")


if __name__ == "__main__":
    smoke()
