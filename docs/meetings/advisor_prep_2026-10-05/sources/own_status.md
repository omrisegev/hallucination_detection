# Line status: claude/self-generated-step-labels-v1

Status: PAUSED-FOR-DECISION

Owner / last updated: Claude session e533cab8 ("Data collection without teacher forcing"), 2026-09-30

## Steps on this branch

- Step 463 [Claude, self-generated step labels] - Two independent frontier judges (Claude Opus 5.5 and GPT-6 SOL, with a second round by Fable 5.1 and Astra 6) labelled the first erroneous step of 958 answers that Qwen3-4B/8B wrote themselves (greedy, non-thinking, GSM8K and MATH); their accuracy on 240 hidden ProcessBench human labels is ProcessBench F1 88.7 and 86.2 (90.9 where they agree), and 890 answers received a final label. Source: `results/self_generated_step_labels_v1/analysis/AGREEMENT_V1.json`, `results/self_generated_step_labels_v1/FINAL_LABELS_V1.jsonl`.
- Step 463 (same step, second result) - The leading method B16 (with digit channels; filter then plain average, and filter then grouped weights), fitted separately for each model and dataset without labels, ranks the first erroneous step as well on the model's own text as on teacher-forced ProcessBench: pooled within-answer AUC 0.757 vs 0.763 (plain average) and 0.739 vs 0.720 (grouped), both inside the declared equivalence margin. Source: `results/self_generated_step_labels_v1/analysis/B16_BEHAVIOUR_V1_FINAL890.json`.
- Step 463 (same step, third result) - The frozen bank11 L-SML sent to the external benchmarks behaves the same way: within-answer AUC 0.732 on own answers vs 0.718 on ProcessBench. Source: `results/self_generated_step_labels_v1/analysis/FROZEN_METHODS_V1.json`.

Step-number collision: another `Step 463 [Claude, estimator provenance]` exists on
`origin/claude/estimator-provenance-collection-2026-09-29`. Both keep the number 463; this one is
tagged `[Claude, self-generated step labels]`, per the tag-never-renumber rule in CLAUDE.md.

## Decision needed from Omri

Question: the behaviour check passed when all cells are pooled, but per cell it is underpowered (57 to 143 erroneous
answers per cell) and one cell is borderline: GSM8K with Qwen3-4B, plain average, 0.712 on own answers vs 0.801 on
ProcessBench (upper end of the 95% interval at 0.000). Do we collect more of the model's own errors to settle the per-cell
question, or close the line here?

Options:
1. Close the line as done: behaviour preserved on the pooled comparison; record the GSM8K / Qwen3-4B cell as an open caveat.
2. Collect sampled answers (temperature 1, about 3 per question, same prompts, Qwen3-4B and Qwen3-8B) on AIRCC for GSM8K and
   MATH, label them with the same two-judge protocol and packet tooling, and rerun the same B16 per-dataset check. Needs:
   AIRCC preflight, a timing run of at most 12 examples per cell, Omri's budget decision, and judge runs on the GPT side.
3. A paired test on the existing answers: score each answer with the model that wrote it and with the other model (Qwen3-8B
   answers with Qwen3-4B telemetry and the reverse), so the only difference is who wrote the text. Needs a short AIRCC
   teacher-forced run (minutes of GPU).

Also recorded, deferred by Omri: technical debt in `spectral_utils/data_loaders.py:is_correct_math`, which marks
equivalent MATH answers wrong (81 of 366 non-truncated grader-wrong MATH own answers are correct according to both judges).
Old self-generated MATH final-answer results are unverified until regraded. See the "Technical debt" section of
`docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md`.

## Outside git but needed

- `C:\Users\omris\TAU\hallucination_detection\dataset_cache\repgrid\evdrop_{gsm8k,math}_qwen3_{4b,8b}\raw_*.pkl`
  (about 2.7 GB in total): the raw self-generated answers and telemetry. Git LFS objects; restore with
  `git lfs pull --include=<path>` in the main checkout.
- `results/self_generated_step_labels_v1/private/own_answers.jsonl` (15 MB, gitignored): regenerable with
  `scripts/self_generated_labels/extract_own_answers.py` from the pickles above.
- `results/self_generated_step_labels_v1/b16/FEATURES.npz` (5.4 MB, untracked): regenerable with
  `scripts/self_generated_labels/b16_fit.py` (about 65 minutes on CPU).
- An untracked copy of `.claude/agents/step-judge.md` sits in the main checkout; the tracked copy is on this branch.
- Other worktrees read (not written) by `b16_fit.py`: `.worktrees/ssl-pseudolabel-residual-v1` (per-dataset fit code and
  Step 462 scores), `.worktrees/readout-quickest-detection-v1`, `.worktrees/token-probability-fusion-v1` and
  `.worktrees/cumulative-vote-fusion-v2` (stored source arrays used only for the reproduction checks).
