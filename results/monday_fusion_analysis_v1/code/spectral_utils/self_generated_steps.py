"""
Deterministic step segmentation for a model's OWN free-generated math answers.

ProcessBench / PRMBench answers arrive pre-split into steps. An answer the scoring model wrote
itself (the `evdrop_*` Qwen3 cells) does not, so a step boundary has to be defined here before
any judge labels it. The rule is fixed, text-only and label-free, and every step is an exact
character span of the original `full_text`, so a step maps onto the saved `token_offsets`
without re-tokenizing.

Qwen3 (non-thinking) writes Markdown: `### Step k:` headings, `---` separators and display
equations in their own paragraphs. Splitting on blank lines alone would make a `---` line or a
bare heading a "step". The rule, applied to blank-line-separated blocks in order:

  * SEP  (`---`, `***`, `___` alone)            -> belongs to no step (carries no content)
  * HEAD (one line, `#...` or fully bold)       -> opens the next step, joined with what follows
  * MATH (display block `$$...$$` / `\\[...\\]`)  -> appended to the current step
  * TEXT                                        -> appended if the current step's last block
                                                   ends with ':' (a colon-introduced list),
                                                   otherwise starts a new step

An empty leading `<think>\\n\\n</think>` block (Qwen3 non-thinking) is excluded from every step.
A non-empty think block is NOT stripped: `segment_answer` raises instead, because it would mean
the cell was not generated in non-thinking mode.
"""
import re

_THINK = re.compile(r"\A\s*<think>(.*?)</think>", re.S)
_BLANK = re.compile(r"\n[ \t]*\n")
_SEP = re.compile(r"(-{3,}|\*{3,}|_{3,})")
_BOLD_LINE = re.compile(r"\*\*[^*\n]+\*\*:?")

SEGMENTATION_RULE_ID = "selfgen-markdown-blocks-v1"


def _blocks(text: str, start: int):
    """(s, e) spans of non-blank blocks in text[start:], trimmed of surrounding whitespace."""
    out = []
    pos = start
    for m in list(_BLANK.finditer(text, start)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            trail = len(seg) - len(seg.rstrip())
            out.append((pos + lead, end - trail))
        if m:
            pos = m.end()
    return out


def _kind(s: str) -> str:
    if _SEP.fullmatch(s):
        return "SEP"
    if "\n" not in s and (s.startswith("#") or _BOLD_LINE.fullmatch(s)):
        return "HEAD"
    if (s.startswith("$$") and s.endswith("$$") and len(s) >= 4) or \
       (s.startswith("\\[") and s.endswith("\\]")):
        return "MATH"
    return "TEXT"


def segment_answer(text: str):
    """Return (body_start, [(start, end), ...]) step spans over `text`.

    Raises ValueError on a non-empty think block or an answer with no content.
    """
    body_start = 0
    m = _THINK.match(text)
    if m:
        if m.group(1).strip():
            raise ValueError("non-empty <think> block: cell was not generated in non-thinking mode")
        body_start = m.end()
    steps, pending = [], []
    for s, e in _blocks(text, body_start):
        kind = _kind(text[s:e])
        if kind == "SEP":
            continue
        if kind == "HEAD":
            pending.append((s, e))
            continue
        if pending:
            steps.append(pending + [(s, e)])
            pending = []
        elif steps and (kind == "MATH" or text[steps[-1][-1][0]:steps[-1][-1][1]].endswith(":")):
            steps[-1].append((s, e))
        else:
            steps.append([(s, e)])
    if pending:
        steps.append(pending)
    if not steps:
        raise ValueError("answer has no content after the think block")
    return body_start, [(blk[0][0], blk[-1][1]) for blk in steps]


def smoke() -> None:
    """Known-answer checks (no data needed)."""
    t = ("<think>\n\n</think>\n\nWe are given:\n\n- a = 1\n- b = 2\n\nWe want a+b.\n\n---\n\n"
         "### Step 1: Add\n\nAdding them:\n\n$$\na+b=3\n$$\n\n---\n\n### Final Answer\n\n"
         "$$\n\\boxed{3}\n$$\n\nSo the answer is 3.")
    b, spans = segment_answer(t)
    got = [t[s:e] for s, e in spans]
    assert b == t.index("</think>") + len("</think>"), b
    assert got[0] == "We are given:\n\n- a = 1\n- b = 2", got[0]
    assert got[1] == "We want a+b.", got[1]
    assert got[2].startswith("### Step 1: Add\n\nAdding them:") and got[2].endswith("a+b=3\n$$"), got[2]
    assert got[3].startswith("### Final Answer") and got[3].endswith("\\boxed{3}\n$$"), got[3]
    assert got[4] == "So the answer is 3.", got[4]
    assert len(got) == 5, got
    assert all("---" not in g for g in got)
    try:
        segment_answer("<think>reasoning</think>\n\nx")
        raise AssertionError("non-empty think block must raise")
    except ValueError:
        pass
    print("self_generated_steps smoke OK")


if __name__ == "__main__":
    smoke()
