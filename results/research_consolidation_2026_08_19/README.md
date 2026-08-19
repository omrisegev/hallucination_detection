# Research consolidation audit trail — 2026-08-19

This directory records the non-destructive consolidation that started from local
`master` at `7ad92c98c73c56387e4c1d2a9f5e098df6bc07da`.

## Opening state

- Local master: `7ad92c98c73c56387e4c1d2a9f5e098df6bc07da`
- Remote master: `cd423ab041b1ddb0ba62b5324740722d6f8ab238`
- Remote paper-exact: `79ee28e81cae93805331cbebf40a47594ce73301`
- Remote white-box: `85149a07dc1d04356479556bd50aa6a7b722c2b5`
- Local white-box: `7cdc39a81004c8ba1a536c019177da3525d3b24c`
- Modified tracked files: 3
- Untracked files: 366
- Stashes: 3 (preserved)
- Worktrees: 5 (preserved, including prunable records)

`opening_state_manifest.json` contains the complete ref inventory and the size,
mtime, and SHA-256 of every modified or untracked file present before
consolidation tooling was introduced.

## Non-destructive Drive archive

Remote:
`gdrive:hallucination_detection/consolidated_results/integration_2026-08-19/pre_merge_untracked/`

- Selected files: 225
- Selected bytes: 162,152,300
- Transfer command: `rclone copy` (no local deletion and no remote deletion)
- Verification: `rclone check --one-way --checksum`
- Verification result: 225 matching files, 0 differences
- Remote size result: 225 files, 162,152,300 bytes

`pre_merge_untracked_drive_manifest.json` contains each archived path, its
selection reason, size, mtime, and opening-state SHA-256. The ignored
`pre_merge_untracked_files.txt` is a generated transport list; the JSON manifest
is the canonical, tracked inventory.

The local source files remain in place. Narrow `.gitignore` rules exclude only
the archived tokenizer materializations, per-cell/cells payloads, per-question,
partial, warning, and per-trace intermediates. Compact reports, aggregates,
decisions, and provenance remain eligible for Git.
