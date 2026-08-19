# Research consolidation verification report

**Integration date:** 2026-08-19

**Final verification date:** 2026-08-20

**Branch:** `codex/consolidate-research-2026-08-19`

## Scope and lineage

The branch was created from local `master` at `7ad92c9` and preserves the
following heads as ancestors:

| lineage | head at integration | integration commit |
|---|---|---|
| remote master | `cd423ab` | `441a3f4` |
| paper-exact acquisition | `79ee28e` | `f98e8d9` |
| corrected remote white-box capture | `85149a0` | `4332b28` |
| local white-box analysis | `7cdc39a` | `f34332c` |

Fair-comparison `bc296ff`, Unified `d3ca3a4`, and three-way `ef3154e` were
verified ancestors and were not merged again. `git merge-base --is-ancestor`
passed for these seven lineage refs plus the local base (eight commits total).

The opening inventory records 366 untracked files, three modified files,
content sizes, mtimes, SHA-256 values, refs, three stashes, and the current
worktree plus five linked worktree records. The stashes and worktree records
remain present.

## Mechanical and focused tests

All commands used `.venv/bin/python` (Python 3.11) because the system Python
lacks the numerical packages required by the project.

| group | result |
|---|---|
| `compileall` over `spectral_utils`, `scripts`, and `cluster` | PASS |
| A6 feature/intervention/PTNI/S0/S0a/S0b/tokenizer suites | PASS, 150 tests |
| Fair-comparison suites | PASS, 146 tests with one expected opt-in 406-MB cache test skipped |
| Paper-exact evaluator | PASS, 108/108 checks |
| Paper-aligned benchmark suite | PASS, 9 tests |
| Local/Online and fixed application pipelines | PASS |
| Contextual IU and c-STG | PASS |
| Data readiness | PASS, 8 tests |
| Corrected layer capture/reference/resume smoke tests | PASS |
| White-box depth, fusion, report, full-data, NRM, matched gray-box, and organic suites | PASS |
| Cross-dataset manifold diagnostic | PASS; 336 expected metric rows and source/protocol hashes verified |

Three scripts that import `scripts` as a package initially failed when invoked
as filesystem paths without the repository on `PYTHONPATH`. Re-running them
with `PYTHONPATH=.` passed: A6-S0a (18 tests), A6-S0b (37 tests), and the
white/gray matched comparison (5 tests). This was an invocation issue, not a
code failure.

The organic-NRM integrity test initially found three source-hash drifts. The
155 prepared inputs (146 MiB) were located intact in the preserved
`hallucination_detection_whitebox_layer_fusion` worktree. A label-free fit was
rebuilt into `/private/tmp`; all ten score archives and ten diagnostic files
were byte-identical to the frozen canonical artifacts. The provenance hashes
were advanced to the integrated sources, including the backward-compatible
`upcr_fit_covariance` extension, and the dependent score/report manifests were
rehash-bound. The full 10-test organic suite then passed. No evaluation label,
metric, score array, or research conclusion changed.

GPU inference, live two-cell white-box capture, the new architecture-fidelity
pilot, and the active A6-S0b numerical execution were not run. No
`S0B_COMPLETE.json` or `S0B_CLOSED.json` exists, so the research status remains
open exactly as recorded in the canonical status note.

## Repository and artifact integrity

- `git diff --cached --check`: PASS after LF-normalizing the new generated CSVs.
- Conflict-marker scan: PASS.
- `git fsck --full --strict`: exit 0. It reports dangling objects/commits only;
  these are retained recovery material and were not pruned.
- Current-tree largest ordinary Git blob: 80,883,712 bytes
  (`scripts/runai-cli-amd64.exe`), below GitHub's 100-MB hard limit. The new
  manifold assets are all below 150 KiB.
- Curated consolidation payload audit: PASS for all 23 pickle objects. Each
  local payload and its canonical Drive copy matched under `rclone check
  --one-way --checksum`; eleven mapped directories reported zero differences
  and 35 matching payload/metadata files. The 36-file source inventory totals
  3,351,678,118 bytes, and its 13 compact metadata files remain ordinary Git
  objects. The unpublished history was rewritten to omit only the 23 payloads;
  `git lfs push --dry-run` then reported no new LFS objects relative to fetched
  GitHub refs. The prior head is retained in a local backup ref.
- Repository-wide `git lfs fsck` cannot be fully green in this partial clone:
  numerous older LFS objects are intentionally absent from the local object
  store. The command reported them as unavailable and could not create its
  optional `.git/lfs/bad` repair directory in the sandbox. This does not affect
  the Drive-backed consolidation payloads or older LFS objects already present
  on GitHub.
- Canonical status-versus-artifact audit: PASS for the fair-comparison,
  Family-NRM, Localization/RAG, contextual decisions, white-box values,
  manifold decision, A6 gate state, and preservation manifests.

## Drive preservation

The non-destructive archive remains at
`gdrive:hallucination_detection/consolidated_results/integration_2026-08-19/pre_merge_untracked/`.
A fresh `rclone check --one-way --checksum` on 2026-08-20 reported zero
differences and 225 matching files. Remote size is 225 objects and 162,152,300
bytes. Local sources were not removed.

The only operational warning is that the configured remote still uses
rclone's shared Google Drive client ID, which rclone says is being retired
during 2026. It does not affect this checksum result, but a project-owned
client ID should be configured before the shared ID stops working.

Separately, all 23 consolidation-only dataset payloads were checked against
their canonical `cluster_results` directories with zero differences before
they were omitted from Git. Their recovery map is
`dataset_cache/DRIVE_BACKUP_2026_08_20.json`; no Drive or local object was
deleted.

## GitHub publication readiness

GitHub CLI 2.97.0 arm64 was installed from the official release and its archive
matched the official SHA-256 checksum. Authentication is `omrisegev` with
`repo` and `workflow` scope. Repository permission is ADMIN. The `master`
branch had no branch-protection rule and the repository exposed no ruleset at
the opening check. A fresh fetch and protection check are required immediately
before publication; no force-push is permitted.

The first branch push was rejected before a remote ref was created because the
repository's LFS budget was exhausted. After the verified Drive-backed payload
removal, publication requires no new LFS upload. The local integration commits
were rewritten only because the branch had never reached GitHub; all merged
lineage heads remain direct ancestors.

## Independent review

A separate read-only agent audited the staged delta, history, recovery points,
LFS/Drive preservation, research claims, and test evidence. It found one P1:
the organic-NRM report manifest still bound the prior run-definition hash while
the score-freeze manifest already bound the integrated one. The input hash was
corrected, and the regression test was strengthened to verify all nine report
inputs as well as all six generated outputs. Independent recheck verified 15
hashes with zero mismatches, reran the 10-test organic suite successfully, and
reported no remaining P0--P2 findings. Its two P3 wording corrections are
incorporated above.
