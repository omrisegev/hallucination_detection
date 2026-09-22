# Agent operating contract

Applies to every coding agent working in this repository (Claude Code, Codex, any
future tool). `CLAUDE.md` and `AGENTS.md` both point here. This file is the *how we
operate* layer; research decisions stay in `CLAUDE.md`. Structure follows the
AGENTS.md principles Karpathy has argued for (permission boundary, reversible or
auditable, fail loudly and stop, memory file as source of truth) and the four
`claude.md` habits (think before coding, simplicity first, surgical changes,
goal-driven execution). Every rule below is tied to an incident in `LESSONS.md`;
none is decorative.

## 1. Permission boundary

```
READ   : everything in the repo, cache/, results/, papers/, the AIRCC results tree, gdrive: (read-only rclone)
WRITE  : spectral_utils/**, scripts/**, cluster/**, tests/**, docs/**, results/<new experiment dir>/**,
         HISTORY.md, PROGRESS.md, LESSONS.md, .claude/**
NEVER  : frozen score files inside an existing results/<dir>/ (edit = new versioned dir),
         results/localization_prm_label_audit_v1/RELEASE_V3.json,
         results/localization_source_group_audit_v1/FOLDS_V2.json,
         any file holding a token or credential (cluster/submit_inference.sbatch HF_TOKEN, .env*, ~/.ssh),
         another agent's worktree under .worktrees/ (read yes, write no),
         Drive data via delete/move/sync (rclone copy only, and only cluster -> gdrive:)
HUMAN_CHECKPOINT (ask, then wait):
         sbatch / any GPU job, rclone copy to Drive, git push, git merge into master,
         every command .claude/hooks/guard_git.py blocks, advisor-facing email or report,
         changing a frozen benchmark contract (labels, folds, IDs, metrics)
```

The hook enforces the git part mechanically. The rest is a promise; breaking it goes
in `LESSONS.md`.

## 2. Reversible or auditable

- Anything irreversible is a HUMAN_CHECKPOINT above. Everything else must be replayable.
- A cluster job records `SYNC_COMMIT.json` (commit, dirty flag, time). A dirty tree
  is a smoke-only run, never a full run.
- A results table must carry, per row or in a sidecar: the source file (pkl/npz) and
  its hash, the exact command, `n_checked`, `n_total`, and the `flag` column from
  `spectral_utils.label_sanity`. A number without a source file is a rumour.
- A number computed on fewer rows than the registered population carries the
  `FEASIBILITY` tag (`label_sanity.feasibility_tag`) and never appears in a headline.
- Session transcripts are retained (`cleanupPeriodDays` 3650) so `/insights` and
  retrospectives can be rerun over the whole project.

## 3. Fail loudly and stop

- A gate or assert that fires is reported verbatim and the run stops. Never relax a
  gate to reach a number (`spectral_utils/paper_exact/gates.py` policy applies desk-wide).
- A bash parse error (`unexpected EOF`) means **nothing** ran, including earlier parts
  of the chain. Verify the filesystem before reporting anything as written.
- One ssh timeout is not an outage. Probe twice, once with `ConnectTimeout=30`, and
  report the observation ("two probes timed out"), not the inferred cause ("VPN down").
  A Slurm error can be a missing `SLURM_CONF_SERVER`, not a cluster outage.
- When a task leaves the boundary above, or two readings of the request lead to
  materially different work, stop and ask. Improvising a workaround is the failure mode.

## 4. Memory file: `LESSONS.md`

- Read the top section of `LESSONS.md` at every session start (`/session-start` does it).
- At session end, or immediately after a mistake is caught, append an entry
  (`/update-docs` Step 7). Format:

  ```
  ## YYYY-MM-DD — <one-line mistake or lesson>
  What happened: ...
  Why: ...
  Rule: ...
  Enforced by: code | command | prose   (name the file)
  ```

- "Enforced by: prose" is a known-weak rule. The retrospective found that 6 of 9
  correction categories recurred after a prose rule and 0 of 2 recurred after a code
  rule. When a prose rule fires a second time, the fix is to move it into code or a
  command, not to rewrite the prose.
- Claude Code's private memory directory is a cache for Claude only. Anything both
  agents must know goes in `LESSONS.md`, `CLAUDE.md`, or `PROGRESS.md`.

## 5. Think before coding

- Before any change wider than one file, write down: the assumption being made, the
  success criterion (a number, a test, a file), and what would falsify it.
- Which method / arm / entry point to score is asked, never inferred (`CLAUDE.md`
  "Which method to evaluate"). Then mirror the script that produced the last reported
  number for it.
- Price the ceiling before building the method (Steps 189, 220, 221 closed whole
  families without a build; Steps 198 and 213 built first and returned no-ops).
- Orient before reading code: `git fetch --all --prune`, branches by date,
  `git worktree list`, and name the live branch. Three sessions were lost to an
  archived branch.

## 6. Simplicity first, surgical changes

- One variant, one discussion, then build ("tailor, never transplant", Step 225).
  No batch-built families reported as a table.
- Prefer an offline-derived constant to a runtime adaptive mechanism.
- Touch only the files the task needs. No drive-by refactors, no renamed steps, no
  "while I was here" edits in `HISTORY.md` or `PROGRESS.md`.
- A new helper lives in `spectral_utils`, never inline in a notebook or script.

## 7. Goal-driven execution: tests before claims

- Name the verifiable check before building: a unit test, `smoke_preset.py`,
  `inspect_cell.py` plus the label-sanity gate, or a `GATE.json`.
- Gate order for anything that touches a GPU: local CPU smoke (`/preflight`) → N=30
  pilot → full N. No sbatch without a `/preflight` PASS in the same session.
- "Done" means the check ran and its output is quoted in the message. A claim
  without the N actually examined ("checked 3 of 55") is not a claim.
- Any headline number goes through `/red-team` before it enters an advisor document.
- A negative result is a deliverable: `/negative-result` writes it in the fixed format.

## 8. Communication

- First: one plain sentence saying what happened. Second: the single most important
  number and what counts as good or bad for it. Third: the next step. Then stop;
  methodology and per-arm detail only on request.
- No shorthand labels (R1/T4/arm B). Name each thing by what it does.
- Ours-vs-theirs is one grid: columns per method, rows per metric, winner marked.
- A required non-conclusive wording exists for interim stages
  (`feedback_fusion_framing_and_controls`): never "weights do not matter".
