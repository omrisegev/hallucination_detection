# Decisions accepted during the 2026-10-01 consolidation

Source: Omri, in the Claude consolidation session `ae2dd164`, after the review
`docs/reviews/CONSOLIDATION_REVIEW_20261001.md`. These are Omri's words translated and condensed;
nothing here is a Claude proposal unless marked so.

## Research decisions (also recorded in CLAUDE.md and Research_Directions.md on
`claude/ssl-pseudolabel-residual-v1`, commit 680c537c6)

1. **The plain average is not acceptable as the final fusion step.** The frozen candidate (Dawid-Skene
   filter, then plain average) is still reported as the current result, also to the advisors. Next
   direction: replace the plain average with a combination that uses the label-free SML or
   method-of-moments estimates. The documented starting point is the Step 457 runner-up: filter,
   label-free partition plus one merge step, latent-group EM (HEM) weights within groups and
   Dawid-Skene weights between groups (8-bank mean 0.7670 vs 0.7656; `results/algorithm_decisions_v1/SUMMARY.md`).
2. **Joint L-SML is discontinued** for new arms. Its historical results stay as records.
3. **Removing 2-3 level-family channels stays an open experiment** to run later (Omri, 2026-09-27),
   with Omri's caveat that going from 5 to 3 channels need not leave two level groups.
4. **Digit features stay** in the label-free fusion line (2026-09-30; commit f7ef911e8). This does not
   reopen digit gates, anchors or digit-oriented signs.

## Consolidation decisions

5. **Historical branches:** an archive with an index is acceptable. The index must record what each
   line accomplished, its evidence, and any unfinished or superseded next steps, not just branch
   names. Code, tables and dependencies needed by the current research are brought into the
   consolidated tree. (Not all eight uncovered refs are from August: one carries a September review.)
6. **backup-pre-lfs-fix:** check the existing Drive backup and upload only missing files; verify
   coverage by path, size and checksum. "Probably backed up" is not sufficient.
7. **scratch/external_generalization_private:** check the existing cluster archives on Drive and
   upload anything missing. Research inputs needed for reproduction must not depend on a local-only copy.
8. **Ownerless-line statuses:** written explicitly as proposals, with evidence and reopening
   conditions. Not a blanket approval to close those directions. Distinguish "this implementation
   failed" from "the research direction is closed".
9. **Publishing without uploading LFS objects to GitHub** is acceptable once all required LFS objects
   are verified on Drive and the restore instructions have been tested. Git history is preserved.
10. **Safety rejection:** no mode switch or new session solely to bypass it; preserve the exact
    message and blocked action. (Omri later switched to the regular mode himself.)
11. **The handoff** adopts the review's complete research-direction checklist, each entry with source,
    current status, next step and dependencies, preserved in the repository.
12. **Backups:** process completion is not backup success. A final reconciliation report separates
    completed-and-verified, failed, not-yet-uploaded and covered-by-a-previously-verified-backup.
    Replacement uploads never write to the same destination concurrently with another upload.
13. **Retention:** keep all original branches, worktrees and data until preservation, restore and
    dependency checks pass. Do not declare everything consolidated until missing material and
    research follow-ups are explicitly accounted for. Report remaining gaps before the merge.

## Scope decision 2026-10-02

14. **The whitebox answer-gate stage and the whitebox step-length follow-up are out of scope for
    now** (Omri). They stay indexed with their evidence (`RESULTS_STAGE_1A.md`, `RESULTS_STAGE_1B.md`
    on `claude/whitebox-layer-views-v1`); they are not closed and not next steps.

## Approvals 2026-10-02 (after Codex's review of the follow-up list)

15. **Preservation and consolidation proceed** under the retention rules (decision 13). This does not
    authorize deleting original branches, worktrees or data before preservation, restore and
    dependency checks pass.
16. **V2/V3 independent audits** of the family-tail external results are approved, as verification of
    existing work (`HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` section 3 item 3).
17. **Transfer proxy: one bounded exploratory test** (same handoff, section 3 item 1). Its exact scope,
    protocol and success criterion are documented before it runs. It is not expanded into a method
    sweep and does not block the SML/MoM direction. Hard2Verify and Socratic are exposed, so matching
    their ranking is exploratory evidence, not proof of transfer; final confirmation stays on unexposed data.
18. **Deferred, not closed** (evidence and next questions preserved):
    - why bank11 L-SML transfers (section 3 item 2);
    - combining the supervised PRM with the label-free method. The measured error-type split
      (Step 437) and the combination idea are kept as two separate items. A combination need not be an
      error-type router (score combination is another option); the open question is whether it improves
      measured performance and under what information budget. No general claim about "structural
      errors" beyond the measured PRMBench categories.
19. **Monday advisor material** (research summary and presentation) is prepared in parallel, without
    waiting for the merge or the proxy.
20. **The handoff** verifies findings, decisions, next steps and dependencies against their sources, not
    only headings, and separates completed findings, Omri's decisions, proposed experiments and
    unresolved verification gaps.
21. No experiment starts automatically.

## Earlier consolidation decisions (2026-09-30 / 10-01)

- Backup pushes of local-only branches and a rescue branch for loose files: approved.
- Upload to Drive of the LFS-blocked branches (as a bundle plus their LFS objects) and of the
  main-checkout loose results: approved.
