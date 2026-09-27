# Authorized inactive worktree cleanup — 2026-09-23

Omri explicitly authorized removing inactive worktrees backed by GitHub branches.
Inspected actual remote heads with `git ls-remote --heads origin`, tracked changes,
all untracked files, ignored files, LFS materialization, path resolution and locks.
No branches, shared Git objects, LFS object store or experiment caches were deleted.

| Removed worktree | Commit | Verified GitHub backing | Local-only audit |
|---|---|---|---|
| `.worktrees/competition-sync` | `49787d4683b0e13bf18626e43cb304fc66e95d45` | Exact head of `claude/competition-sync-2026-09-23` | Clean; no untracked or ignored files; all104 LFS files pointers only |
| `.worktrees/review-token-axis-20260921` | `2b321fa3aaf3da20ee5ba938077631d02d580e83` | Ancestor of verified `claude/token-axis-fusion-sampling-3i9r2u` head `2cddcfff84332ae6966d42d8b76d14fdda242573` | Clean; only seven ignored Python bytecode files; all104 LFS files pointers only |

The first was a completed publication/synchronization checkout, with16,677 tracked
files totalling1,972,573,443 logical bytes. The second was Codex's completed review
checkout with a stale `initializing` lock dated September21; unlocked it before
normal `git worktree remove`. No force removal was used. Current experiment input
worktrees were retained. Process inspection found no worktree-specific experiment
command targeting these paths; agent process names alone were not treated as
proof that every other worktree was inactive.

Disk available before:0 bytes. After first removal:2,625,314,816 bytes. After both
removals and guide restoration:3,936,321,536 bytes (~3.94GB decimal /3.67GiB).
These are volume observations; other processes may also change free space.

## Retained

- `consolidation-fusion-2026-09-22`: clean except rebuildable bytecode, but no
  remote-tracking branch containing its commit was found. Not proven backed up.
- `depth-feature-fusion-v1`: four untracked source/result files plus ignored score
  arrays. No remote-containing branch found. Preserve local results.
- `whitebox-layer-views-v1`: untracked provenance and ~815MB ignored material,
  including step/answer arrays. No remote-containing branch found.
- `a6-s0b`:207 tracked changes, including206 deletions and a modified cluster README.
- Root and `cumulative-vote-fusion-v2`, `token-probability-fusion-v1`,
  `readout-quickest-detection-v1`, `ssl-pseudolabel-residual-v1`,
  `lsml-ct7-levers-run`: current work and/or inputs for the requested research audit.

## Recovery and documentation

The earlier disk-full write had truncated root CLAUDE.md. Its original73,787 bytes
were restored exactly (SHA256 `80ae91476c848725cf792c0b505b0fad235c6f59cb841356ff74c5bd9a858630`).
After space became available, replaced the emergency hardlink with an independent
copy and verified link count1 BEFORE updating project instructions. The source
CLAUDE.md in the SSL worktree was never edited. The latest CPU/L-SML/PRMBench
requirements are now recorded in canonical CLAUDE.md and the revised plans.

Before this authorized worktree cleanup, only a disposable Codex file-list inventory
and two old browser crash dumps under scratch had been removed in recovery attempts;
those attempts did not resolve disk exhaustion. Scientific artifacts were preserved.
