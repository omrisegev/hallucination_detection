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

## Technical debt recorded 2026-09-29 (deferred by Omri)

Both judges call 85 of the 558 grader-wrong own answers correct (81 MATH, 22% of the MATH
grader-wrong class): `spectral_utils.data_loaders.is_correct_math` misgrades equivalent forms
(`\frac32` vs `3/2`, `3/2` vs gold `1.5`, `\pm 3i` vs `3i,-3i`, `D` vs `\text{(B)}`, pmatrix vs
tuple). Benchmark-labelled localization results (ProcessBench, PRMBench, Hard2Verify, Socratic)
are not affected. Every self-generated MATH cell graded at inference time may carry this label
noise (math500 cells via `cluster/run_inference.py`, AMC/AIME wrappers, `evdrop_math_*`); the
Colab-era MATH-500 headline cells are not yet verified to use the same grader. To do later:
regrade offline with a real equivalence checker (CPU), count flips per cell, rescore old MATH
results only if material. Omri chose to test the leading method on self-generated data first.

## Frozen-method evaluation (declared 2026-09-29, before the evaluation was run)

Omri's decision: test the methods locked for the external study directly on the generation-time
telemetry of the self-generated answers; no cluster run. Arms: the seven locked arms of
`spectral_utils/external_generalization` (frozen bank11 L-SML / equal / partition-equal,
answer-local L-SML / equal / partition-equal, CT7) with the frozen source bundle
(`BUNDLE.json`, sha b96939fc...), no refit and no recalibration. Code hashes match
METHOD_FREEZE.json (15/15, line endings normalized).

Telemetry check before scoring (40 GSM8K answers): generation `top_k_logprobs` equals the raw
top-50, saved entropy equals top-15 entropy of the raw log-probs (max diff 2.1e-7), saved
surprisal equals the raw chosen-token log-prob, greedy token is always raw top-1, and the locked
`validate_telemetry` passes. Token-to-step rule: a token belongs to the step containing its first
character. Scores are sealed (`scoring/SCORES.jsonl`) before labels are read.

Labels: consensus of the two judges (identical first-error step, including -1); the 102
disagreements are excluded pending adjudication. Metrics (`scripts/self_generated_labels/
evaluate_selfgen.py`): ProcessBench F1 at the frozen thresholds; within-answer AUC of the first
error step against the steps before it; argmax accuracy. Reference with the same code: the
same arms' cross-fitted source scores on ProcessBench GSM8K + MATH (Qwen3-8B and Qwen3-4B
telemetry over other models' text). Bootstrap 2000 draws over question groups. The comparison
is between different answer populations (own greedy answers vs ProcessBench answers), not a
paired test; the reference also uses per-fold rather than deployed thresholds.

## B16 per-dataset behaviour check (declared 2026-09-29, before the evaluation was run)

Omri's question: does the leading method WITH digits behave the same on the model's own answers
as under teacher forcing? Method (Omri's choice): B16 (bank11 + realized_z + realized_drv + three
digit channels), DS filter + plain average (BASE) and DS filter + grouped DS-MLE weights (GRP),
fitted per model per dataset, label-free, on all non-truncated answers of each new cell
(`scripts/self_generated_labels/b16_fit.py`, mirroring `per_dataset_fit_run.py::fit_arms`,
Step 462). Hard stops before any new fit: the 16 channels recomputed from raw ProcessBench
telemetry must equal the stored source arrays, and the fit must reproduce the Step 462 per-dataset
B16 BASE/GRP scores (<= 1e-9) on two ProcessBench cells.

Comparison (`b16_eval.py`): each own cell against the ProcessBench cell with the same model and
dataset (Step 462 per-dataset scores). Erroneous answers; first-error hit (earliest argmax) and
within-answer AUC (first-error step against earlier steps). Success = behaviour preserved:
|own - ProcessBench| < 0.03 within-AUC with the 95% interval of the difference containing 0
(independent bootstrap over question groups, 2000 draws). The first-error hit is reported beside it
but depends on answer length (own MATH answers are ~3x longer), so it is not the equivalence test.

## Round 2: resolving the 118 round-1 disagreements (declared 2026-09-29, before round-2 labels)

Omri's design. The 102 own + 16 ProcessBench items on which the two round-1 judges disagree are
re-packed blind (`round2/packet`, new K-ids, new seeded order, same content, instructions and
checker; the mapping is in `round2/private/ROUND2_KEY.jsonl`). Two new independent judges label all
118 from scratch: `fable-5.1` (Claude Fable 5.1 subagent, step-judge definition, effort medium) and
`astra-6` (GPT Astra 6, run by Omri with the same launch prompt). Neither sees round-1 labels.

Decision rule (four votes per item): RESOLVED if the two round-2 judges give the same first-error
step AND it equals one of the two round-1 answers (3 of 4). Everything else (round-2 judges
disagree, or agree on a third answer) goes to a blind debate with votes, designed next. The rule's
accuracy is measured on the 16 ProcessBench items against their human labels.
