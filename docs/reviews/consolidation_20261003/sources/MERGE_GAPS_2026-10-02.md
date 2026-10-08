# Remaining gaps before the merge, 2026-10-02

Omri asked (2026-10-01) for the remaining gaps to be reported before the merge proceeds. This file
lists them. Nothing here is deleted, merged or pushed by writing it. The research handoff is
`RESEARCH_HANDOFF_2026-10-01.md`, the backup state is `RECONCILIATION_2026-10-01.md`, the historical
archive index is `HISTORICAL_INDEX_2026-10-01.md`, and the ownerless-line status proposals are in
`line_status_proposals/`.

## A. Gaps that block deleting originals (not the merge)

1. **Three items are not on Drive yet** (403 quota on rclone's shared client_id; retry running):
   `results/fusion_multiwidth_dense_v1/` (main checkout), off-git files of `.worktrees/er-generality-v1`
   (37 files) and of `.worktrees/lsml-ct7-levers-run` (43 files). Their local copies are complete. Keep
   them until verified. Durable fix: a personal rclone client_id (Omri, Google Cloud Console), which is
   needed anyway because the shared one is being retired during 2026.
2. **LFS publication without GitHub LFS upload is not yet restore-tested.** Decision 9 requires every
   needed LFS object verified on Drive and a tested restore procedure. The bundle restore was tested
   (`git bundle verify`, 7 heads) and the 12 LFS objects of the bundled branches pass `rclone check`.
   Still to do: a written restore procedure (clone with `GIT_LFS_SKIP_SMUDGE=1`, place objects from Drive
   into `.git/lfs/objects/`, `git lfs checkout`) and one test on a scratch clone, plus a check that the
   63 backup-pre-lfs-fix objects are all reachable on Drive (61 matched by SHA-256 in older Drive folders,
   2 uploaded in pass 2).
3. **Codex's documentation branch `codex/advisor-prep-2026-10-05` (tip 096b17128, 4 commits, 70 files) is
   local only.** Codex is still committing to it. Its handoff
   (`docs/meetings/advisor_prep_2026-10-05/MERGE_HANDOFF.md` on that branch) asks for the whole series
   to be integrated, not one commit. It needs a backup push once Codex stops writing to it.
4. **The main checkout's working copy changed after the rescue commit.** `CLAUDE.md`, `HISTORY.md`,
   `LESSONS.md`, `PROGRESS.md` and `Research_Directions.md` differ from the rescue branch (last modified
   2026-10-02 12:49 and 22:31). Part of this is Codex's preparation series; Codex's handoff says other
   pre-existing edits were deliberately left out of its commits. At merge time these five files must be
   snapshotted again and union-merged, then checked by counting blocks on each side.

## B. Gaps the merge itself must resolve

Trial merges into the newest main line (`origin/claude/ssl-pseudolabel-residual-v1`, which already
contains `codex/lsml-external-generalization-v1`), run with `git merge-tree` (no ref changed):

| Branch | Commits ahead | Conflicted files |
|---|---|---|
| `claude/estimator-provenance-collection-2026-09-29` | 4 | PROGRESS |
| `claude/decision-rule-v1` | 12 | HISTORY, LESSONS, PROGRESS |
| `claude/self-generated-step-labels-v1` | 15 | HISTORY, LESSONS, PROGRESS |
| `claude/whitebox-layer-views-v1` | 18 | none |
| `lsml-ct7-levers-run` (= `origin/claude/lsml-ct7-levers-v1`) | 19 | .gitattributes, .gitignore, CLAUDE.md, PROGRESS, Research_Directions, docs/HANDOFF_TOKEN_PROBABILITIES.md, papers/index.md, scripts/test_token_local_fusion.py, spectral_utils/fusion_utils.py, spectral_utils/token_local_fusion.py |
| `claude/readout-quickest-detection-v1` | 279 | .gitattributes, .gitignore, CLAUDE.md, HISTORY, PROGRESS, Research_Directions, papers/index.md, spectral_utils/fusion_utils.py |
| `rescue/main-checkout-loose-files-2026-10-01` | 5 | LESSONS |
| `codex/advisor-prep-2026-10-05` | 4 | HISTORY, LESSONS, PROGRESS, Research_Directions |

5. **Prose logs** (HISTORY, PROGRESS, LESSONS, Research_Directions): union by block, never
   `--ours`/`--theirs`; count blocks on both sides against the result (worked example: `cd423ab`).
   Step-number collisions are tagged, not renumbered (e.g. two Step 463 entries; a Step 459 collision
   was reported in the handoff check).
6. **Code conflicts:**
   - `spectral_utils/fusion_utils.py`: the SSL line converted the file's line endings (whole-file diff;
     only 9 real line insertions with CR ignored); the old line made a 70-line edit. Resolve by applying
     the 70-line edit on the normalized file, then run its tests.
   - `spectral_utils/token_local_fusion.py` and `scripts/test_token_local_fusion.py`: added on both lines
     independently; they differ by 20 and 59 changed lines. Take the superset after review and run
     the tests.
   - Hash-locked CRLF files (`spectral_utils/external_generalization/_bank11/__init__.py`,
     `chosen_token_calibration.py`) must not be renormalized by the `.gitattributes` merge.
7. **CLAUDE.md:** the SSL copy carries the 2026-09-30 (digits stay) and 2026-10-01 (averaging not final;
   Joint stopped) decisions; other branches still open with the 2026-09-17 / 09-24 text. Keep all
   blocks, newest decisions on top, and state which older instructions they supersede (the 09-17 digit
   exclusion still governs gates, anchors and signs).
8. **HISTORY content outside the main line:** Steps 298-353 were removed by commit `9215ae69d` (a full
   copy exists at `48f9ef291`, 54 of those headings); Steps 336-421 exist only on the readout-quickest /
   CT7-levers lines (for Steps 390-421: 34 headings there, 7 on the SSL, decision-rule and rescue tips);
   Steps 283-289 (graph line) exist only on `origin/codex/graph-geometry-selection-v1`. The merged
   HISTORY must contain all of them, tagged by line.
9. **Two branches are not contained anywhere else:** `claude/token-axis-fusion-sampling-3i9r2u` and
   `codex/claude-feature-bank-token-lsml-v1`. They are indexed in the historical index and their status proposals;
   they enter the consolidated tree as archive refs, with any needed code brought over.

## C. Reproducibility gaps (do not block the merge; block re-running old experiments)

10. **Missing pool inputs.** `pool_z.npy` and `pool_names.json` are gone from the temporary scratchpad
    that `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json` points to. A hash-verified
    copy is in the SSL worktree `results/algorithm_decisions_v1/inputs_backup/` and on Drive. Runners
    that hash-check them will stop until the files are restored or a new manifest version points to the
    backup (frozen manifests are not edited in place).
11. **Runners read other worktrees by absolute path** (30 files define the main checkout path; details
    in the handoff appendix). After the merge, the worktrees they read must stay in place, or new
    runners must use repo-relative paths. Removing a worktree breaks these runners even when its files
    are on Drive.
12. **`scripts/build_advisor_update_aug21_2026.py`** depends on files that exist only on the
    graph-geometry ref.
13. **DEEM-B3 run outputs** exist only in the second computer's `local_cache`, not here and not on
    Drive as far as checked. This needs Omri or that machine.
14. **`codex/combined-fusion-v1`** is named in records but was never created on this machine; whether it
    exists on the other machine is unverified.
15. **MATH grader debt** (deferred by Omri, 2026-09-29): old self-generated MATH final-answer results are
    unverified.

## D. Research follow-ups accounted for, not started

Per decision 21, no experiment starts automatically. The handoff lists every direction with its tag.
Those approved to run after preservation and the merge: the V2/V3 family-tail audits (verification of
existing work) and one bounded transfer-proxy test whose scope, protocol and success criterion are
written first. The SML/MoM fusion direction is Omri's stated main direction and has no protocol yet.

## Proposed merge order (for Omri's go)

On a new branch `claude/consolidation-2026-10-01` from `origin/claude/ssl-pseudolabel-residual-v1`:
estimator-provenance, decision-rule, self-generated labels, whitebox, rescue, CT7-levers, readout-quickest,
then Codex's advisor-prep series once it is final. One merge commit per branch, each followed by
block counts on the logs and the test suite. All original branches, worktrees and data stay until the
checks in A pass.
