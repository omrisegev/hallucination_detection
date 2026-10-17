# Line status proposal: codex/digit-alternative-probability-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- The branch as a development line: SUPERSEDED (by `claude/ssl-pseudolabel-residual-v1`, which contains every commit of this branch and adopted its three digit columns as the "+d" channels of the label-free banks)
- The three decoding-independent digit features (alternative digit probability, digit spread, alternative-probability innovation): TOOL-ONLY inputs, kept in the label-free line by Omri's decision of 2026-09-30

Owner / last updated: no owner session. Tip efc0527b0 (2026-09-28); no origin ref of its own, but the tip is contained in `origin/claude/ssl-pseudolabel-residual-v1`. Worktree `.worktrees/digit-alternative-probability-v1`.

## Steps on this branch

On top of `codex/lsml-external-generalization-v1` (bef0b1de3):
- Step 447 [Codex, digit alternative probability] - The probability of the second most likely single digit (among digits seen in the top 50, no use of the written token) carries signal but is not a consistently better standalone replacement: against the old digit-disagreement signal it wins on PRMBench within-answer AUC (+0.07352 [+0.06473, +0.08242]) and ties on ProcessBench (-0.89 points [-3.91, +2.07]); against entropy with the same step readout it wins on ProcessBench (+6.91 [+3.45, +10.23]) and ties on PRMBench. Source: `results/digit_alternative_probability_v1/REPORT_HE.md`. (Collides with `Step 447 [Claude, partition ceiling]` and `Step 447 [Claude, PB tail weights]`; keep all, tagged.)
- Step 448 [Codex, numeric-family extension] - Adding three digit columns to bank11 under L-SML raises PRMScore by +0.3491 points [+0.0837, +0.6420] over the original bank11, and L-SML finds them as a separate group, but the gain over adding one column (-0.1340 [-0.3419, +0.0753]), over three identical copies (+0.0597 [-0.0060, +0.1270]) and over the plain average of the same bank (+0.1494 [-0.2688, +0.5588]) is not established. Source: `results/digit_family_extension_v1/REPORT_HE.md`. (Collides with `Step 448 [Claude, Mind the Gap code audit]`.)

## Evidence

- `results/digit_alternative_probability_v1/REPORT_HE.md` @ `codex/digit-alternative-probability-v1`: single-signal table on the full source population, 13,769 answers / 145,597 steps / 6,968,779 tokens; ProcessBench exact first-error localization on 4,442 erroneous answers over 8 cells; PRMBench within-answer AUC on 6,030 answers; 5,000 paired source-question draws, 98.75% Bonferroni intervals over 4 primary contrasts. Two digits are missing from the top 50 at 4,930,517 of 6,968,779 positions; both bounds were kept (the score is a lower bound on the full-vocabulary probability).
- `results/digit_family_extension_v1/REPORT_HE.md` @ same branch: L-SML with one digit column 64.6554 PRMScore x100, within-answer AUC 0.76946; with three columns 64.5214 / 0.76799; original bank11 L-SML 64.1723 / 0.76455. PRMScore on 6,211 non-control answers; 707 source groups.
- Later use, `results/algorithm_decisions_v1/SUMMARY.md` @ `claude/ssl-pseudolabel-residual-v1` (Step 457): adding the three digit features raises the plain average on all 8 banks (+0.0093 to +0.0152 within-answer AUC; ProcessBench +0.022 to +0.038), on 6,030 PRMBench answers.
- Omri, 2026-09-30 (recorded in CLAUDE.md on the ssl branch, commit f7ef911e8, and in `docs/consolidation/DECISIONS_2026-10-01.md` item 4): the digit features stay in the label-free fusion line. This does not reopen digit gates, digit anchors or digit-oriented signs (the 2026-09-17 exclusion of digit disagreement with the scoring model's preferred token still applies to those).

## What failed (implementation) vs what is still open (direction)

Not established (not a failure of the features): the alternative-digit probability as a standalone replacement for entropy or for the old digit signal, and the three-column family as better than one column or than averaging.

Still open:
- The features were measured only on teacher-forced telemetry with a censored top 50. They are computable during greedy generation, but behaviour on self-generated answers was not tested on this branch. (The self-generated-labels line, Step 463, ran the leading method with digit channels on the model's own answers and found pooled behaviour preserved; per-cell power is limited.)
- Whether the digit gain is content beyond step position (Step 457 says "content beyond position" for part of it; see the ssl SUMMARY).

## Reopening conditions

- A standalone-replacement claim would need a new pre-declared contrast that beats both the old digit signal and entropy on both benchmarks; the current result says it does not.
- A full-vocabulary digit probability (not top-50 censored) would need new inference and is outside the current no-new-inference scope.

## Dependencies other lines have on it

- `claude/ssl-pseudolabel-residual-v1`: `scripts/run_digit_family_extension_v1.py`, `scripts/complete_digit_family_equal_control.py`, `scripts/experiments/algorithm_decisions_run.py`, `scripts/experiments/lsml_merge_step_run.py` and `scripts/verify_external_banks_v4_source.py` use the digit columns; every "+d" bank of Steps 457-462 depends on them.
- `claude/self-generated-step-labels-v1` (B16 with digit channels) and `claude/decision-rule-v1` consume ssl outputs built with them.

## Outside git but needed

- None found in this check: the worktree's ignored files are only Python caches, and the commit "Preserve compact replay arrays and sealed artifact byte conventions" (911abc5f9) placed the replay arrays in git.
- The features are extracted from the source telemetry (top-50 log-probabilities of the 13,769 answers), the same raw inputs as every other bank; no separate copy.
