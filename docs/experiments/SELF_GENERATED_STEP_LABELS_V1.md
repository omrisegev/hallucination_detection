# Self-generated step labels v1 (stage A: existing data, no GPU)

Status: packet frozen 2026-09-28, before any judge output. Branch
`claude/self-generated-step-labels-v1`. Authorized by Omri on 2026-09-28.

## Why

Every localization number so far comes from teacher-forced scoring of text the scoring model
did not write: ProcessBench answers were written by 12 other models (none is Qwen3), and
PRMBench errors were planted into correct solutions by another LLM. Part of what the method
detects may therefore be "text I would not have written" rather than "a place where I got
confused". The recent strongest channels (written-token surprise, CT7's 7th view) are the ones
most exposed to this. Teacher forcing itself is not the confound: on the model's own text it
reproduces generation telemetry (Gate B, `cluster/run_teacher_forced.py`). The confound is
off-policy text. This experiment builds step labels for answers the model wrote itself.

## Data (existing, no new inference)

Four `evdrop_*` cells (`dataset_cache/repgrid/`): Qwen3-4B and Qwen3-8B, greedy (T=0),
non-thinking (`/no_think`), GSM8K test (1,319) and full-MATH test sample (1,500); rich save
with `token_offsets`, top-50 log-probs and `gen_token_ids`. 5,638 answers, 825 graded wrong
at inference time.

Excluded: 271 answers truncated at the token cap (262 of them wrong MATH answers, mostly long
root-search loops). They have no final answer; they are a separate failure class, not labeled
in v1.

## Step segmentation (fixed, label-free)

`spectral_utils/self_generated_steps.py`, rule id `selfgen-markdown-blocks-v1`: blank-line
blocks; `---` separators belong to no step; a one-line heading opens the next step; display
math joins the current step; a block after one ending with ':' joins it. Every step is an exact
character span of `full_text` (stored in the item key), so it maps onto `token_offsets`.
ProcessBench items keep their official steps.

## Packet (what the judges label)

`scripts/self_generated_labels/build_packet.py`, seed 20260928, 1,198 items in 48 shards of 25:

- all 558 wrong, non-truncated own answers (GSM8K 4B 114 / 8B 78; MATH 4B 190 / 8B 176);
- 100 random correct, non-truncated own answers per cell (400);
- judge validation: 240 ProcessBench items with human first-error labels, 60 per
  (GSM8K, MATH) x (error, no error).

Every item has the problem, the reference solution and final answer (GSM8K test / MATH test,
matched by exact whitespace-normalized question text: 1,400/1,400 ProcessBench GSM8K+MATH
questions match), and the candidate steps. Items are shuffled together under opaque ids, so
the judges cannot tell own answers from validation items, and any prefix of shards is a random
sample of the packet. The key (source, cell, labels, step spans) is in `private/ITEM_KEY.jsonl`;
the judge-visible manifest carries no composition.

## Judges

Two independent judges, same instructions (`packet/JUDGE_INSTRUCTIONS.md`), same packet:

- `claude-opus-5.5`: Claude Opus 5.5 subagents (effort medium), run in this session, several
  instances over disjoint shard ranges;
- `gpt-6-sol`: GPT-6 SOL, run by Omri with the same launch prompt.

Neither judge sees the other's labels, the key, or any repository file outside the packet.
Mind the Gap and the dedicated PRMs are NOT labelers: they are comparators to be evaluated on
these labels. The format reference given to the second judge is the synthetic example in the
packet, not the first judge's labels.

## Analysis plan (declared before any label exists)

1. Validity: each judge against ProcessBench human labels on the 240 validation items: exact
   first-error-step accuracy on error items, -1 accuracy on no-error items, and their
   harmonic mean (ProcessBench F1), per subset.
2. Agreement on the 958 own answers: exact first-error agreement, agreement on -1, Cohen's kappa
   on error/no-error, and the step-distance distribution when both flag an error.
3. Final-answer check: judge equivalence calls against the inference-time grader; disagreements
   listed by item.
4. Primary label = both judges agree exactly. The adjudication procedure for disagreements is
   decided after steps 1-3 and before any detection method is scored on these labels.
5. No detection-method quality is looked at until the labels are frozen.

Not decided here: the bias-measurement analysis on the existing teacher-forced data, which
methods to score, and the new-inference stage (sampling, OlympiadBench / Omni-MATH).
