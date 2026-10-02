# Integration handoff for the Monday advisor preparation

User-authorized documentation work by Codex, 2026-10-02. Worktree creation was optional; the user explicitly waived it given limited disk space. The preparation lives in the existing main checkout. A dedicated Git reference `codex/advisor-prep-2026-10-05` preserves only this documentation change without switching the checkout, changing its branch tip, or staging other sessions' edits.

## Files to preserve

- Entire `docs/meetings/advisor_prep_2026-10-05/`: plan, Hebrew preparation page, source manifest, 27 small source snapshots, dated consolidation status, validation and this handoff.
- `docs/line_status/codex__advisor-prep-2026-10-05.md`.
- The dated 2026-10-02 preparation blocks added to `CLAUDE.md`, `PROGRESS.md`, `Research_Directions.md`, `HISTORY.md`, and the new documentation-preservation lesson in `LESSONS.md`.

The source snapshot manifest records original repository path, originating commit and SHA256. `matches_commit_blob=false` explicitly identifies a working-copy source. In particular the latest consolidation approvals were not yet in the rescue branch commit when copied; preserve their exact snapshot, not only the older commit.

## Integration procedure after the consolidation is ready

1. Review `git show --stat codex/advisor-prep-2026-10-05` and the full documentation diff. This is a single documentation-only commit on the observed main-checkout base `bef0b1de3e185a2cf48d1c12a9ee974e2a2acc1d`.
2. On the user-authorized consolidation branch, integrate that commit. Resolve canonical-document conflicts by retaining both lines' substantive text, with the dated preparation blocks intact. Do not replace whole logs with one side.
3. Check all files in the commit are present; revalidate source snapshot hashes, local links and the preparation status against the latest Claude fact sheet. Reconcile differences against the cited artifacts; do not silently use the larger number.
4. Keep historical operational snapshots dated. Mark live upload/merge status separately; do not turn an old status paragraph into a current assertion.
5. Update the preparation checklist when actual slide work and claim verification finish. Do not mark slides, V2/V3 audits or the proxy complete merely because the documents exist.

No merge, push, experiment, upload or cleanup was performed by this documentation commit. The main checkout's pre-existing edits to HISTORY, PROGRESS and LESSONS are not included in the dedicated commit. Do not discard them; they belong to earlier work. Keep this branch and all preparation files until integration is checked.
