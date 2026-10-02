# Line status proposal: codex/family15-tail20-transfer-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status: SUPERSEDED (by `codex/lsml-external-generalization-v1`, which carries every code and result file of this branch byte-identically and evaluated the packaged candidate externally; the family/tail recipe itself is SATURATED per that line's status)

Owner / last updated: no owner session. One own commit, df8371f3c (2026-09-24), on top of 715e586d5; identical on origin. Worktree `.worktrees/family15-tail20-transfer-v1`.

## Steps on this branch

- 2026-09-24 [Codex] Preserve family15 tail20 for external-evaluation handoff (untagged HISTORY block, no step number) - Packaged Claude's `F15_tailtie_lsml` candidate (15 families, top-ceil(20%) step marks with fractional boundary ties, answer centering, no extra pooled normalization of marks, continuous L-SML backend) with its frozen deployment lock and a portable implementation. Full source replay passes on 13,769 answers / 145,597 steps over all 5 fit/calibration/evaluation rotations: maximum score difference 2.22e-16; deployment weights, scores and threshold replay exactly. No external evaluation on this branch. Source: `results/family_tail_transfer_v1/PORT_REPLAY.json`, `TRANSFER_LOCK_V1.json`, `docs/experiments/FAMILY15_TAIL20_AGENT_HANDOFF_HE.md`.

## Evidence

- Source quality of the packaged candidate: PRMScore 0.645210 against 0.643323 for the family-equal average and 0.641723 for bank11 L-SML, on 6,211 non-control PRMBench answers / 83,371 steps; within-answer AUC 0.768533 on 6,030 answers; its edge over the average and over bank11 is not established after multiplicity correction. File: `results/family_tail_transfer_v1/source_snapshot/METRICS.csv` @ `codex/family15-tail20-transfer-v1`, summarized in the handoff document.
- External result of the same candidate (evaluated on the later branch, Codex family-tail external V1): Hard2Verify balanced F1 41.020, Socratic PRMScore 59.921 / 62.690 (Qwen3-8B / QwQ-32B), against bank11 L-SML 43.670 / 63.221 / 64.238; Socratic losses supported by the adjusted paired intervals. N = 6,190 records / 53,970 steps. File: `results/family_tail_external_v1/METRICS.json` and `CONTRASTS.json` @ `codex/lsml-external-generalization-v1`.
- The packaged lock collapses to K=2 groups because unstandardized centred marks were fed to the scale-sensitive L-SML group criterion (ssl Step 443); the corrected row (Step 444) reaches only the family-equal level externally (42.28 / 61.00 / 62.30). File: `results/family_tail_external_v2/REPORT_HE.md` @ `codex/lsml-external-generalization-v1`.
- Containment check (done 2026-10-01): all 24 code, document and result files of df8371f3c have the same blob on `codex/lsml-external-generalization-v1` (bef0b1de3). Not carried there: this commit's HISTORY block, its PROGRESS block, two LESSONS entries ("Preserve executed tail preprocessing, not an inconsistent method description"; "Check blob bytes before fixing apparent changes in a fresh worktree"), and two `.gitattributes` lines (`spectral_utils/family_*.py text eol=lf`, `tests/test_family_*.py text eol=lf`).

## What failed (implementation) vs what is still open (direction)

Failed: this specific family-tail recipe (V1 lock, K=2) and its corrected version transfer below bank11 L-SML on Socratic. The packaging and portable port succeeded; nothing failed in the port.

Still open: nothing that is specific to this branch. The broader questions are carried in the `codex/lsml-external-generalization-v1` status file: the source-side transfer proxy is approved as one bounded exploratory test, and "why bank11 transfers" is deferred, not closed (Omri, 2026-10-02).

## Reopening conditions

As for the family/tail recipes in the external line: only with a new idea outside tail-threshold tuning, per-family thresholds, 16 families or within-family SML, token-level tail fusion and FUSE-style consistency objectives (`docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` section 3 item 6).

## Dependencies other lines have on it

- None live: the external line already holds `spectral_utils/family_tail_transfer.py`, `scripts/verify_family_tail_transfer.py`, `tests/test_family_tail_transfer.py` and `results/family_tail_transfer_v1/` byte-identically.
- To preserve when consolidating: the HISTORY and PROGRESS blocks, the two LESSONS entries and the two `.gitattributes` lines listed above (the LF rule protects the port's recorded source hash across checkout).

## Outside git but needed

- None produced here: the worktree's ignored files are only Python caches; the large telemetry and source feature matrix were intentionally not copied into git (handoff document, first section).
- The upstream source-snapshot scripts contain absolute paths of a Claude session and are historical evidence, not a runnable pipeline (handoff document).
