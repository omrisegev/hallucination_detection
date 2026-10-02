# Line status proposal: codex/claude-feature-bank-token-lsml-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- The branch: SUPERSEDED (by `claude/token-probability-fusion-v1`, which carries its one own commit byte-identically and continued the analysis in Steps 422-427)
- Token-level L-SML on the eleven-channel bank, as implemented here (pooled donor-fold token standardization, fusion before a Top-10 step readout): IMPLEMENTATION-NEGATIVE (direction open; most of its margin over equal weighting was later traced to a redundant channel pair, and the CT7-stream and external versions also lost to equal)
- The eleven-channel feature bank module (`spectral_utils/claude_feature_bank_v1.py`): TOOL-ONLY (the "bank11" channel definitions used by later lines)

Owner / last updated: no owner session. Tip 890459866 (2026-09-18), identical on origin. No worktree. The four earlier commits on the branch (Codex B3 / CIW-DEEM research, 2026-08-25/26) are on `origin/master` and many other refs.

## Steps on this branch

- Commit 890459866 "Add token-level Claude feature bank experiment" (Codex; no HISTORY step of its own; the experiment is reported in Step 422 [Claude] on the ssl/token-probability lines) - eleven risk-oriented token channels (`q15_H1`, `q15_VE1`, `chosen_surprisal`, `logprob_margin`, `true_tail50`, `energy_level`, `energy_innovation`, `top15_turnover`, `top50_js`, `dominant_freq16`, `bocpd_p0`), continuous-residual L-SML fitted on up to 60,000 pooled donor-fold tokens, window 16, five source folds, LOCO-5 gate at 0.33. Gated ProcessBench macro F1 0.3408 (L-SML) against 0.3234 (equal mean); PRMBench within-answer AUC 0.7532 against 0.7311. Source: `results/claude_feature_bank_token_lsml_v1/RESULTS.json`, `docs/experiments/CLAUDE_FEATURE_BANK_TOKEN_LSML_V1.md`.

## Evidence

- `results/claude_feature_bank_token_lsml_v1/RESULTS.json` @ `codex/claude-feature-bank-token-lsml-v1`: N = 13,769 answers / 145,597 steps; ProcessBench F1 over 8 cells with 5,391 of 6,800 answers opened by the gate; within-answer AUC on 6,030 PRMBench answers; L-SML found K = 6 groups in every fold; `development_only: true`.
- `results/claude_feature_bank_token_lsml_v1/GATE_ISOLATION.json` @ `claude/ssl-pseudolabel-residual-v1` (Step 422, separating gate from locator): gate-free localization 35.92 (L-SML) against 32.59 (equal), +3.33 [+1.90, +4.75] over 10,000 paired source-group draws.
- `HISTORY.md` Step 423 [Claude] @ `claude/token-probability-fusion-v1`: CT7's gate-free localization is 39.89, +3.96 [+2.28, +5.59] above this arm.
- `results/token_probability_fusion_v1/ENERGY_PAIR_ABLATION.json` @ `claude/token-probability-fusion-v1` (Steps 425 and 427): with the redundant energy pair repaired, the L-SML margin shrinks to +1.23 [+0.26, +2.23] or +1.00 [+0.12, +1.89]; shrinkage -2.10 [-3.42, -0.83]. The margin stays positive.
- Later versions of the same idea: token-level L-SML on CT7's streams below equal, -0.83 [-1.93, +0.25] (Step 436, `results/ct7_token_lsml_v1/UNCERTAINTY.json` @ `lsml-ct7-levers-run`); answer-local token L-SML below token equal on both Socratic backbones (-1.285 and -1.087 points, `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md` @ `codex/lsml-external-generalization-v1`).

## What failed (implementation) vs what is still open (direction)

Failed or qualified:
- This implementation's headline advantage over equal weighting was mostly L-SML repairing a gauge-redundant pair (`energy_level` / `energy_innovation`), which equal weighting cannot do; the remaining +1.0 to +1.2 points is small and below CT7.
- Protocol deviation recorded in Step 422: fitting on a pooled donor-fold token matrix with no answer-local standardization while the readout is an argmax within the answer. The matched answer-local versus pooled comparison on one fixed bank "was specified three times and has never been run" (Steps 421-422 PROGRESS block).

Still open: that matched answer-local versus pooled comparison; token-level fusion on a bank with more than about two effective independent views. The ct7-levers status file treats token-level L-SML on CT7's streams as saturated; this branch's bank is a different (wider) roster, but the same one-pass output distribution.

## Reopening conditions

A new token-level channel that is not another function of the same one-pass entropy and probability block, or the pre-specified answer-local versus pooled comparison, run with the redundant energy level removed or replaced by one explicit difference channel (Step 425 proposal: ten identified channels).

## Dependencies other lines have on it

- `spectral_utils/claude_feature_bank_v1.py` is the source of the eleven channel definitions ("bank11"); Step 424 on the token-axis line copied it unchanged, and the token-probability line's `TOKEN_MATRICES.npz` and `DERIVATIVE_CHANNELS.npz` are built on these channels, which the ssl, decision-rule, estimator and self-generated-label lines read.
- `GATE_ISOLATION.json` (Step 422) is on the ssl and rescue branches but not on this branch.

## Outside git but needed

None: `OOF_SCORES.npz`, `RESULTS.json` and `SMOKE.json` are committed. The `source.frozen_roster` field in `RESULTS.json` points to a Codex machine path (`/Users/osegev/Desktop/hallucination_detection/.worktrees/fusion-independence-atlas-v1/results/localization_full_benchmark_v3`), which is provenance, not a local dependency.
