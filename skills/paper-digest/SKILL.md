---
name: paper-digest
description: Cache the artifacts of reading a paper under papers/ (full text, summary, datasets/models, compared methods, experiment scores) so the next deep-dive starts from the cache instead of from scratch. Use whenever a paper in papers/ needs to be read, re-read, or looked up.
---

# Paper Digest

This skill has no dependency on either tool's internals — it's just Python plus markdown
files committed in this repo. It works identically whether you're Claude Code or
antigravity/Gemini CLI.

## Why this exists

Papers under `papers/` get re-read repeatedly across sessions (EPR, the SML/ensemble-learning
line, U-PCR, FUSE, Semantic/Spilled Energy, EDIS, and various benchmarks). Each re-read used to
start cold: full PDF read, re-derive the summary, re-find the datasets/baselines/scores. This
skill caches that work once per paper so future sessions read a small markdown card instead.

Scope: this is a **per-paper deep-dive cache**, not a cross-paper literature-survey tool. For
surveying many candidate papers at once (pre-emption checks, benchmark comparisons across
dozens of items), use the `/research` family of skills and
`docs/research_notes/research_phase10_rag/` instead — different job, heavier schema.

There is deliberately no semantic/RAG search layer here. With ~20 papers, `papers/index.md`
plus grep is enough. Revisit only if the paper count grows a lot.

## Workflow

### Step 1 — Check the cache first

Read `papers/index.md`. Find the row for the paper (match on filename/title).

- **`status: digested`** — read `papers/digests/<slug>.md` directly. Pull
  `papers/extracted/<slug>.md` only if you need an exact quote or page reference. Stop here —
  no PDF read needed.
- **`status: extracted`** — full text is already pulled but no digest exists yet. Skip to
  Step 3.
- **`status: raw`, or the paper isn't in the index at all** — go to Step 2.

### Step 2 — Extract (mechanical, no reading required)

```bash
python skills/paper-digest/scripts/extract_pdf_text.py "papers/<Paper Name>.pdf"
```

This is a zero-judgment mechanical step — PyMuPDF pulls the text, no summarization happens
here. It writes `papers/extracted/<slug>.md` and prints the slug it used.

- Idempotent: if the extract already exists, it no-ops and tells you so. Pass `--force` to
  regenerate (e.g. after replacing the source PDF with a newer version).
  Idempotency check is by-slug (does `papers/extracted/<slug>.md` exist), not deep content
  hashing — this project's papers are static references, not living documents.
- If a page comes back near-empty, the script flags it inline in the output as
  `[EMPTY — possible scanned image]` and warns on stdout. There's no OCR step here — that page's
  text just won't be available; note the gap in the digest if it matters.

### Step 3 — Write the digest

Read the extracted markdown from Step 2 (not the original PDF, and not from memory/prior
knowledge of the paper). Copy `skills/paper-digest/references/digest_template.md` to
`papers/digests/<slug>.md` and fill it in. Keep it tight — this is a lookup card, not a
re-statement of the whole paper.

**Grounding rule — every field must trace to a specific line in the extracted markdown, not
to what you already "know" about a well-known paper:**
- `authors`, `arxiv_id`, `venue`, `year` — copy these verbatim from the first page of the
  extract (they're almost always right there: title block, `arXiv:XXXX.XXXXXv1 [cs.XX] date`
  line, footer). If a field genuinely isn't findable in the extract, write
  `not found in extract` — never guess, and never fall back to a generic placeholder like
  "Research Team et al." or a plausible-sounding but unverified name.
- The **scores table** must contain literal numbers copied from the paper's own
  results/tables section of the extract, not qualitative paraphrases like "outperforms
  baseline" or "strong gains." If you can't find a numeric result in the extract, say so
  explicitly rather than writing a vague qualitative substitute.
- **Datasets/models/baselines** must be the ones actually named in the extract — grep the
  extract for the terms before writing them down. A paper's abstract almost always lists its
  actual benchmarks; don't substitute benchmarks from a similar-sounding paper you recall.

This matters more here than in most digest-writing: getting a paper's own facts wrong in a
hallucination-detection project's paper cache is the exact failure mode the project studies.
Before finishing a digest, grep the extract for anything you're about to assert (a dataset
name, a score, an author) and confirm it's actually there.

Section guidance:

- **Summary** — 2-4 sentences: the core method or finding.
- **Datasets & models used** — exactly what was evaluated on / with.
- **Methods it compared itself against** — the baselines the paper benchmarks against.
- **Experiments — methodology & scores** — how the experiment was run (splits, metrics,
  sample sizes) and the headline numbers, as a table where possible.
- **Connection to our pipeline** — how it relates to the spectral/L-SML/EPR work in this repo
  (carries forward the requirement from CLAUDE.md's old "Research papers" rule — don't drop
  this even though the rest of the workflow changed).
- **Notes / open questions** — anything unresolved, surprising, or worth a follow-up.

### Step 4 — Update the index

In `papers/index.md`, flip that paper's row to `status: digested`, fill in the slug (if not
already set from Step 2), a one-line takeaway, and today's date.

### Step 5 — HISTORY.md (only if it matters)

A **cache hit** (Step 1 stopped early) never needs a HISTORY.md entry — nothing new happened.

A **substantive new read** that changes understanding of the roadmap is worth a short
HISTORY.md step: title + 2-3 sentence takeaway + a pointer to the digest file for detail. Don't
restate the whole digest inline — that's what `papers/digests/<slug>.md` is for. Follow the
existing HISTORY.md format (`### Step N — <title>`, `**What**/**Why**/**Result**`) and the
"one step per logical investigation" rule — a five-paper batch read is one step with
sub-headers, not five steps.

If the paper changes the thesis roadmap, also update `PROGRESS.md` and
`Research_Directions.md` as usual.
