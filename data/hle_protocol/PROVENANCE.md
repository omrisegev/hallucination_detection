# HLE (Humanity's Last Exam) protocol assets — provenance

Vendored 2026-08-08 for the HLE pilot (see `cluster/manifests/hle_pilot_v1.json`). Sources:
official repo [github.com/centerforaisafety/hle](https://github.com/centerforaisafety/hle)
(`hle_eval/run_model_predictions.py`, `hle_eval/run_judge_results.py`, `README.md`), the
paper (arXiv:2501.14249), and [HLE-Verified](https://github.com/SKYLENAGE-AI/HLE-Verified)
(arXiv:2602.13964).

## Dataset

- HF `cais/hle`, gated ("auto"-approved, our existing HF_TOKEN already has access — verified
  2026-08-08), test split only, 2500 rows, 12 fields (`id, question, image, image_preview,
  answer, answer_type, author_name, rationale, rationale_image, raw_subject, category, canary`).
- **HLE is a rolling dataset** (documented per-ID changes Feb 2026 and Jul 2026 in the repo's
  `hle-rolling-changes.txt`) — **pin revision `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`**
  (matches what `inspect_evals` pins and what our HF_TOKEN currently resolves to). Do not use
  `macabdul9/hle_text_only` — it's a stale Jan-2025 snapshot (2370 rows, wrong counts).
- `answer_type` is exactly `exactMatch` or `multipleChoice` (camelCase). Multiple-choice options
  are inlined into the `question` string itself (`"...\n\nAnswer Choices:\nA. ...\nB. ..."`) —
  there is no separate options field.
- **Text-only filter**: `bool(row["image"])` — `""` means text-only. 2158/2500 (86.3%) are
  text-only, matching the paper's own stated ~14% image rate.

## HLE-Verified tiers — `hle_verified_tiers.csv`

HLE-Verified re-annotates all 2500 items into three quality tiers (Gold 668 / Revision 1143 /
Uncertain 689, disjoint, sum to 2500). The tier-assignment code itself is not published in their
repo — treat the tiers as given data, not re-derivable.

The official repo ships the tiers only as three full-record JSONL dumps (~340 MB total,
Git-LFS, contain the actual question/answer/rationale text). We do **not** vendor those —
HLE's own dataset card explicitly asks not to "publicly share, re-upload, or distribute" the
benchmark content, to protect it from training-data contamination. `hle_verified_tiers.csv`
instead carries only non-content metadata per id: `id, tier, answer_type, category, has_image`
— no question/answer/rationale text — so it carries zero contamination risk while still letting
`load_hle()` join a tier label onto each row for later slice analysis (Gold vs Revision vs
Uncertain), per `external_data_collection_plan_2026.md`'s "retain HLE-Verified status" instruction.

**Caveat**: tier assignment is confounded with subject — the verification pipeline is math-centric
(Revision is 100% Math/Physics, Uncertain absorbs almost all Humanities/Engineering/Other/CS).
Gold ∩ text-only = 575 items, skewed toward Math/CS. Not every current `cais/hle` id
(post-rolling-update) is necessarily in this 2500-row index — a row with no match gets `tier=None`
in `load_hle()` (untiered, not an error).

## Prompt — verified from the CURRENT official code, not the original paper text

**System message** (`hle_eval/run_model_predictions.py::SYSTEM_PROMPT`, verbatim):
```
Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```
**User message**: the raw `question` string, nothing else appended (no few-shot, no extra
instructions — official `format_message()` sends exactly `question_text` as the only content).

**Known discrepancy, decided explicitly**: the paper (Appendix C.1.1) originally specified TWO
system prompts, keyed on `answer_type` (a separate "Exact Answer:" wording for exact-match
questions). Commit `67b32511` ("Unify system prompt", 2025-06-06) removed this split in the
current repo. The removed code compared `answer_type == 'exact_match'` (snake_case) against a
dataset whose real values are camelCase (`exactMatch`) — meaning that branch could never fire
even before its removal, so the single unified prompt was already the operative behavior in
practice. **We use the current unified single system prompt** (matches what the live official
code actually does today), not the paper's original dual-prompt text. This is the same kind of
"reproduce the current official artifact, not a superseded paper description" call as SemGrad's
Qwen3 leading-slash-bug correction.

`o1`-family models get no system role (`system_role = "user" if "o1" in args.model else
"system"`) — not applicable to any open-weight model we'd run.

## Decoding

Paper's stated policy (Appendix C.5): **temperature 0.0** "when configurable." (The official
code's own `--temperature` flag is dead — the line is commented out, a documented bug, not a
protocol choice — so we follow the paper's stated intent, not the buggy script.)
`max_completion_tokens`: README recommends ≥8192 for reasoning models; not applicable to a
non-reasoning instruct model (see Model below).

## Grading — DEFERRED (Omri, 2026-08-08)

Official grading uses an LLM judge (`o3-mini-2025-01-31` via the OpenAI API, previously
`gpt-4o-2024-08-06`), verbatim `JUDGE_PROMPT` reproduced below for when grading is picked back
up. This project has no OpenAI API key set up anywhere. Decision: **do not grade at generation
time beyond a rough placeholder** — this pilot's job is to collect the raw generation + telemetry
only. Real grading is deferred to a second pass, "maybe Claude or Gemini... separately" (Omri).

The inline `is_correct_hle_provisional` grader (see `spectral_utils/data_loaders.py`) is a
ROUGE-L proxy against the `answer` field extracted from the response's `Answer:` line — good
enough to drive `run_inference.py`'s per-cell sanity print, **not trustworthy as a real label**.
Given HLE's difficulty (frontier models <25% accuracy) the accuracy-band gate may well come back
REJECT on this cell — expected and not a failure, per this project's existing convention for
known-hard cells (e.g. `AIME24 x Qwen-1.5B` in `cluster/presets.py`'s own docstring).

Official judge prompt, kept verbatim for the future regrade pass:
```
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
```
The judge also extracts the model's stated confidence in the same call (defaults to 100 if
absent) — no separate confidence parser exists officially.

**`<think>` warning for a future regrade pass**: if the eventual judge (Claude/Gemini or
otherwise) is shown `full_text` for a reasoning-style model, strip the `<think>...</think>` block
first (`spectral_utils.data_loaders.strip_think`, already used elsewhere in this project) — the
official repo has no handling for this at all (unmerged PR #18) and would otherwise let a judge
extract a "final answer" from mid-reasoning text. Not an issue for this pilot's model (see below,
not a reasoning-tuned checkpoint), but will matter if a DeepSeek-R1-Distill class model is used
on HLE later.

## Model — "strongest model used in this project" (Omri, 2026-08-08)

`Qwen/Qwen2.5-72B-Instruct` (bf16, full precision) — the single largest model anywhere in this
repo's presets (tied with `meta-llama/Llama-3.3-70B-Instruct` at ~70-72B, but Qwen is larger,
non-gated, and already proven on this exact cluster path: `gpqa_qwen72b` preset, "bf16 fits a
192GB B200" per `cluster/presets.py`'s own comment). Not a reasoning-tuned checkpoint (no
`<think>` chain-of-thought training), so `max_new_tokens` is set to 2048, not the 8192 the
README recommends for reasoning models.

## New infra note — system message support

Every existing prompt in this project's `DATASETS` registry is a single user-turn (the official
protocol for every dataset used so far never separates system/user). HLE's official protocol
genuinely needs a system turn distinct from the user turn (the format instructions are systemic,
the user turn is only ever the bare question). `spectral_utils/model_utils.py`'s `fmt_prompt`/
`generate_full` and `cluster/presets.py`'s `_preset()` gained a new optional `system_message`
field for this — see `cluster/run_inference.py` diff. This is now available to any future preset
that needs a real system turn, not an HLE-only hack.
