# Readout controls and provenance rollback (v1) — frozen protocol

User authorized Stage 0 and Stage 1 on 2026-09-11 with the simple token streams
(entropy, varentropy). Base commit 490f4b6c (`codex/binary-moment-fusion-v1`).
Branch/worktree `claude/readout-provenance-v1` / `.worktrees/readout-provenance-v1`.
Frozen inputs are read from the main checkout (`--source-root`); nothing frozen is
rewritten. No new inference, no fitting, no labels in any score.

## Why

Scratchpad diagnostics on the 4,442 erroneous ProcessBench answers (2026-09-11,
recorded in PROGRESS/HISTORY with this run) found that (a) `argmax(step length)`
alone reaches 29.7% raw exact peaks versus 31.5% for the frozen entropy top-10
readout and 15.5% for a random step, so every step readout in the PB tables
inherits a step-length prior that has never had an explicit control row; and
(b) the dominant miss is LATE (predicted after the true step in 43% of erroneous
answers, before it in 25%). Stage 0 adds the missing controls to the frozen
benchmark table. Stage 1 tests one fixed, label-free, cache-only attribution rule
against the late bias, with a shuffled-attribution control that separates
"moving mass earlier" from "moving mass to the right step".

## Shared contract (unchanged from Step 334 / 339)

Full frozen roster: 13,769 answers (PB 6,800 in 8 cells; PRMB 6,969), canonical
v3 labels, v2 source groups and outer folds, saved token-to-step spans, the
frozen mean-entropy q0.3 fold gate from `fusion_fixed_gate_v1` (`dual__iu`),
first-max tie rule (`np.argmax`), PRMScore with the existing per-method q0.8
held-source-group calibration. Evaluation code is imported unchanged from
`scripts/run_direct_probability_temporal.py` (`evaluate_arrays`,
`paired_bootstrap`) so that every number is directly comparable with the
Step 334–341 tables. All cached data are development, not untouched test.

Token streams: `entropy_series` (column 1) and `topk_varentropy_series`
(column 26, K=50) of the frozen benchmark telemetry
(`results/localization_full_benchmark_v3/inputs/<cell>/raw.npy`, PRMB from the
same release). Reference readout: top-10 token mean per step. Reproduction
gates: entropy must give 0.354444 / 0.7301113611 / 0.6254255392 and varentropy
0.356755 / 0.7424645484 / 0.6327768739 (PB all-8 / PRMB within / PRMScore), else
the run aborts.

## Stage 0 — simple controls and stratified reporting (no new method)

Arms (step scores):

| arm | step score | role |
|---|---|---|
| `entropy_top10`, `varentropy_top10` | frozen top-10 token mean | references |
| `length` | token count of the step | simple control |
| `random_step` | i.i.d. uniform(0,1) per step, seed 2026091101 | chance row |
| `position_first` | `-k` (always the first step) | positional control |

Reporting for EVERY arm in Stage 0 and Stage 1, in addition to the standard
metrics: raw exact and within-one on erroneous PB answers, split by whether the
true step is the longest step of its answer; counts of early / exact / late
predictions and the histogram of `predicted - true` clipped to [-4, +4];
per-cell versions of the same. The analytic chance level `mean(1/K)` is
recorded beside `random_step`.

## Stage 1 — provenance rollback of surprise (fixed rule, label-free)

Definitions, per answer, per stream:

- Token strings are the scoring model's tokenizer decoding of the saved
  `gen_token_ids` (Qwen3-4B for `*_q4`, Qwen3-8B for `*_q8` and PRMB; local
  files only). Qwen splits every digit into its own token (verified: 18,063
  single-digit tokens, 0 multi-digit, in `pb_gsm8k_q4`).
- A numeral is a maximal run of digit tokens, optionally joined by one "." or ","
  token between digit runs; its literal is the concatenated digits (commas
  removed). Numerals of a single digit are ignored (too common to carry
  provenance). Leading/trailing whitespace tokens are not part of the numeral.
- Given literals are the numerals of the problem statement (`problem` /
  `question` text, same regex on characters). Using a given is not an error
  source, so given numerals never roll back.
- `origin(v)` of a non-given literal `v` is the earliest step in which `v`
  appears as a numeral. A numeral in step `k` with `origin(v) < k` is
  "inherited"; otherwise it is "native".
- `provenance_reassign_top10`: every token of an inherited numeral is moved from
  its own step to `origin(v)`; the frozen top-10 readout then runs on the
  reassigned token sets (a step's set = its native tokens plus tokens attributed
  to it). Steps left with no tokens score `-inf` (never selected; recorded).
- `provenance_duplicate_top10`: as above, but the inherited tokens stay in their
  own step AND are copied to `origin(v)`.
- `provenance_shuffled_top10` (control): the same inherited tokens are moved to a
  uniformly random earlier step (seed 2026091102, per answer), so the amount and
  direction of moved mass match `reassign` while the target is uninformative.
- `rise_vs_history_top10` and `first_near_max_top10` (onset-style fixed
  readouts on the frozen step scores): `argmax_k (s_k - mean(s_<k))` and the
  earliest step with `s_k >= max(s) - 0.25 * sd(s)`. These are the cheap
  "choose earlier" comparators from the diagnostics.

Twelve Stage 1 arms = two streams × {reassign, duplicate, shuffled,
rise_vs_history, first_near_max} plus the two references.

Primary contrasts (paired canonical-source bootstrap, 10,000 draws, 97.5% CI,
same routine as Steps 338–341): per stream, `reassign - reference` and
`reassign - shuffled` on PB all-8 macro F1 and PRMB within-answer AUC.
Everything else is exploratory (95%). Decision question: does provenance
reassignment reduce the late/early ratio without lowering exact peaks, and does
it beat the shuffled control? No promotion threshold; no historical24 run.

## Preflight

Tokenizer round-trip: for every answer, `decode(gen_token_ids[span])` must
match the saved step text after whitespace normalization; mismatches abort.
Numeral extraction is unit-tested on synthetic token lists (digit runs,
decimals, commas, given-exclusion, origin, single-digit exclusion). A 27-answer
smoke (shortest / median / 95th-percentile trace per cell) runs all arms and
reports counts of inherited numerals; it gives feasibility only.

## Outputs

`results/readout_length_control_and_provenance_v1/`: `METRICS.json`
(per-arm standard metrics + stratified/late-early panels + contrasts),
`SCORES.npz` (step scores, predictions, validity per arm), `SUMMARY.csv`,
`PROVENANCE_STATS.json` (numerals, inherited counts, moved tokens per cell),
`MANIFEST.json` (input hashes, code hash, tokenizer ids), `RUN_STATE.json`.
Chat-first results; no HTML unless asked.
