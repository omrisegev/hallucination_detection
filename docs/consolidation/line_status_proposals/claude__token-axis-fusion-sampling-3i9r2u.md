# Line status proposal: claude/token-axis-fusion-sampling-3i9r2u (origin ref only)

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- The branch as a development line: SUPERSEDED (its three proposed L-SML levers were implemented and run on `lsml-ct7-levers-run`, whose status file marks them SATURATED; its cumulative-vote design was run on the full population on `codex/cumulative-vote-fusion-v2`)
- Cumulative-vote fusion of first-error localizers (Steps 423-424 of this line): SATURATED for the tested implementations (reproduces the incumbent; the branch's own diagnosis note says "closed for weighting", which this proposal reads as a statement about these inputs, not a closed direction)

Owner / last updated: no owner session. Exists only as `origin/claude/token-axis-fusion-sampling-3i9r2u`, tip 2cddcfff8 (2026-09-23); no local branch, no worktree.

## Steps on this branch

Own commits on top of the ssl base 72d8235b4:
- Step 423 [Claude] - Cumulative-vote fusion of first-error localizers ("is the first error at step <= n?" votes, fused by SML / L-SML / Dawid-Skene) reproduces the incumbent on the ProcessBench x Llama-3.1-8B fair-comparison lane (3,400 questions, 5 localizers): incumbent 30.12 against Dawid-Skene mode 30.08 (-0.05 [-0.27, +0.18]); the long-chain deficit is a shared late bias, not a weighting problem. Source: `results/cumulative_vote_fusion_v1/`, `docs/experiments/CUMULATIVE_VOTE_FUSION_V1.md`. (Collides with `Step 423 [Claude]` on the token-probability line and `Step 423 [Codex cross-branch audit]` on the ssl line; keep all, tagged.)
- Step 424 [Claude] - Raw-channel readouts plus soft cumulative-vote fusion: pipeline built and checked on 30-answer pilot caches only (feasibility, no comparative claim); Mind-the-Gap replay numbers verified (23.32% / 50.88% on 2,221 erroneous Llama answers equal the frozen package's own values). Source: `results/raw_channel_readout_fusion_v1/`, `results/cumulative_vote_fusion_v1/MIND_THE_GAP_VERIFICATION.md`. (Collides with `Step 424 [Claude]` on the token-probability line.)
- Documentation commits (2026-09-21 to 09-23, no step numbers): cumulative-vote clarifications and PRMBench extension design; a review of `codex/cumulative-vote-fusion-v2`; FUSE-style boundary search proposal and its diagnostics summary; `docs/research_notes/ct7_anatomy_2026-09-23.md`; `docs/research_notes/lsml_against_ct7_failures_2026-09-23.md`; a pointer to the implementation branch of the three levers.

## Evidence

- `HISTORY.md` Step 423 block @ `origin/claude/token-axis-fusion-sampling-3i9r2u`: overall localization incumbent 30.12, Dawid-Skene mode 30.08, L-SML mode 29.90, SML weighted median 29.76, median of positions 29.40; the label-free consensus is the incumbent (Dawid-Skene "not late" 0.968 / "not early" 0.945). Population: 3,400 ProcessBench questions, Llama-3.1-8B, out-of-fold on five source folds.
- Full-population run of the same design: `HISTORY.md` "Cumulative vote fusion v2 [Codex]" and `Step 421 [Codex]` @ `codex/cumulative-vote-fusion-v2`: selected soft L-SML 36.184 against CT7 39.886 gate-free localization on 13,769 answers; learned fusion adds +0.152 over selected soft equal (interval includes 0); on CT7's own seven profiles +0.109 [-0.600, +0.829], and PRMBench within-answer AUC -0.003765 [-0.004880, -0.002680].
- `docs/research_notes/lsml_against_ct7_failures_2026-09-23.md` section 2 @ `origin/claude/token-axis-fusion-sampling-3i9r2u`: "Argmax competition ... cumulative-vote L-SML / DS was the fusion designed for this decision and reproduced the incumbent ... closed for weighting".
- The levers proposed in that note (family-equal partition of CT7's views, token-level L-SML on CT7's streams, window representation) were run as Steps 434 and 436 on `lsml-ct7-levers-run`: family-equal null on ProcessBench (+0.36 [-3.13, +3.75]), token-level L-SML -0.83 [-1.93, +0.25] against equal, window arms 10.19-10.59 points below CT7 (see that line's status file). The answer-level L-SML gate lever in the same table was marked "never run".

## What failed (implementation) vs what is still open (direction)

Exhausted: re-weighting first-error decisions of localizers that are fusions of the same one-pass streams (binary or soft, SML, L-SML, Dawid-Skene, hierarchical EM); all five localizers drift the same way on long chains, so there are no independent errors to exploit.

Still open (named in the diagnosis note and not run anywhere found in this check): an answer-level L-SML gate over whole-answer readouts (6,800 ProcessBench answers, all inputs cached, no inference). The decision-rule line's Step 460 tested the project's existing answer-level detectors as gates, which is related but not the same construction.

## Reopening conditions

Cumulative-vote fusion: only with localizers whose errors are not driven by the same late bias (for example a localizer built on a genuinely new view). The answer-level gate lever needs Omri's go as part of the gate decision already open on `claude/decision-rule-v1`.

## Dependencies other lines have on it

- None by path. Its two result directories (`results/cumulative_vote_fusion_v1/`, `results/raw_channel_readout_fusion_v1/`) are carried with identical trees on `claude/readout-quickest-detection-v1` (merged at an earlier tip through `consolidation/fusion-2026-09-22`), and the cumulative-vote design continues on `codex/cumulative-vote-fusion-v2`, whose `cvf_v2` code the ssl line imports.
- `docs/research_notes/ct7_anatomy_2026-09-23.md` is byte-identical on `lsml-ct7-levers-run`; `lsml_against_ct7_failures_2026-09-23.md` there differs only by the four-line implementation pointer added on this ref.
- The last five documentation commits (26ef3db83 to 2cddcfff8) are on no other ref; their content is the two notes above plus that pointer.

## Outside git but needed

None produced on this line: Step 424's full cells were not run here (its HISTORY block says the telemetry was on Drive/AIRCC and the Drive connector capped downloads at 10 MB); only 30-answer pilot outputs are in git.
