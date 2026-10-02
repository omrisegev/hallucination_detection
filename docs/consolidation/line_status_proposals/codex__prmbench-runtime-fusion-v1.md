# Line status proposal: codex/prmbench-runtime-fusion-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- The branch: SUPERSEDED (by `lsml-ct7-levers-run`, of which it is a strict prefix; that line's owner-written status file already reports every step here)
- Combining the supervised PRM with this project's label-free method (motivated by Step 437's complementarity): DEFERRED, not closed (Omri, 2026-10-02, `docs/consolidation/DECISIONS_2026-10-01.md` item 18; nothing built)

Owner / last updated: no owner session. Tip b016ef6a1 (2026-09-23). No origin ref of its own, but contained in `origin/claude/lsml-ct7-levers-v1`. No worktree of its own. Despite its name, the branch does not hold the PRMBench runtime-fusion plan document (`docs/experiments/PRMBENCH_RUNTIME_FUSION_PLAN_HE.md` lives on the token-local / ssl / rescue refs).

## Steps on this branch

On top of `codex/cumulative-vote-fusion-v2` (all also on `lsml-ct7-levers-run`, summarized in `docs/line_status/lsml-ct7-levers-run.md`):
- Sync commit 49787d468 - Steps 428-432 code, protocols, tables and ledgers placed on the cumulative-vote tip (no new LFS objects).
- Step 433 [Claude][ct7-levers] - Three L-SML levers against CT7's failures built with protocols and synthetic tests; no real-data number. (Collides with the external line's untagged Step 433.)
- Step 434 [local][ct7-levers] - Family-equal weighting of CT7's seven views: ProcessBench null (+0.36 [-3.13, +3.75]), PRMBench within-answer AUC +0.77 points. Source: `results/ct7_family_equal_v1/`.
- Step 436 [local][ct7-levers] (with gate correction A1) - Token-level L-SML on CT7's streams below equal (-0.83 [-1.93, +0.25]); window representation passes its participation-ratio gate but its learned weights buy nothing, 10.19-10.59 points below CT7. Source: `results/ct7_token_lsml_v1/`, `results/window_representation_b3_v1/`.
- Step 437 [local][prm-measure] - The supervised Qwen2.5-Math-PRM-7B measured beside CT7 on PRMBench with matched endpoints: the PRM leads on within-answer AUC (+2.88 points [+2.14, +3.60]) and PRMScore at q80 (0.6804 against 0.6457) but trails on the argmax first-error hit (57.78% against 61.14%), and the two hit sets overlap on only 39.02% of answers. Source: `results/prm_vs_ct7_prmbench_v1/MEASUREMENT.json`, `PRMSCORE.json`.

## Evidence

- `results/prm_vs_ct7_prmbench_v1/MEASUREMENT.json` and `PRMSCORE.json` @ `codex/prmbench-runtime-fusion-v1`: 6,035 erroneous PRMBench answers for the first-error hit; 94,203 labelled PRMBench steps for the conditional-independence measurements; measurement only, nothing fitted on labels, no fusion of the PRM with CT7.
- `docs/line_status/lsml-ct7-levers-run.md` @ `lsml-ct7-levers-run` (owner-written): the error-type split measured on 2026-09-27 puts this project's methods ahead of the supervised PRM on redundancy errors and the PRM ahead on semantic errors (published page https://claude.ai/artifact/EArjBd9q2aXnvwoAimYPWr, not in git); "that would be a new line, not a continuation of this one".

## What failed (implementation) vs what is still open (direction)

Nothing new failed on this branch beyond what the ct7-levers status file reports (CT7 weighting levers SATURATED).

Deferred, not closed (Omri, 2026-10-02): combining the supervised PRM with the label-free method. Omri's decision keeps two items separate: the measured error-type split (Step 437 and the 2026-09-27 page) and the combination idea. A combination need not be an error-type router; plain score combination is another option. The open question is whether it improves measured performance, and under what information budget. No general claim about "structural errors" beyond the measured PRMBench categories. The 2026-10-01 consolidation review keeps it as an open proposal "with matched access and error-type controls" and warns that complementary hit sets do not establish a working label-free router. Omri's 2026-09-23 note (CLAUDE.md) also cautions that the CT7-versus-PRM hit advantage does not establish overall superiority to supervised PRMs. Any such line changes the access level (a supervised model's scores enter), so it must be reported in a separate panel.

## Reopening conditions

Omri lifting the deferral, then a new, separately labelled line with: matched access stated up front, the information budget stated, and the PRM-alone and label-free-alone rows as mandatory references. If the combination is a router by error type, the error-type variable must be available without labels at decision time (or the analysis carries an explicit oracle label).

## Dependencies other lines have on it

None of its own: every file is on `lsml-ct7-levers-run` and `origin/claude/lsml-ct7-levers-v1`. Step 437 reads `results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy` and the PRM scores in `prmbench_prm.pkl` (see the ct7-levers status file for their locations).

## Outside git but needed

Nothing beyond what the ct7-levers status file lists for Steps 434-437 (main-checkout copy of `profiles.npy`, `CT7_TOKEN_MATRICES.npz`, the bootstrap and decision files of 0.1 to 4.8 MB). The error-type analysis exists only as the published page named above; its scripts were not located in git in this check.
