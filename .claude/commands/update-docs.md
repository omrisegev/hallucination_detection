---
description: Draft and write HISTORY.md + PROGRESS.md updates after completing an experiment, fix, or milestone. Stages and commits the result. Use whenever the session produced a meaningful outcome worth recording.
---

Run the following documentation update sequence.

## Step 1 — Find the next step number
Read the tail of `HISTORY.md` (last 60 lines). Find the last `### Step N` heading. The new step is N+1.

## Step 2 — Draft the HISTORY.md entry
Using context from the current conversation (what was done, why, what the result was), draft a step entry in this exact format:

```
### Step N — <one-line title describing the action>

**What**: <what was done or discovered — 2-4 sentences>
**Why**: <why this was needed or what triggered it>
**Result**: <outcome: numbers (AUROCs, sample counts, file sizes), status, or key finding>

**Files changed**:
- `file1` — reason
- `file2` — reason

---
```

Rules:
- Title must be a verb phrase ("Fix X", "Add Y", "Diagnose Z", not "X was fixed")
- Numbers go in Result, not What
- If multiple failed attempts: group under one step with `**Attempt N**` sub-headers
- Do NOT create a new step for every retry — one step per logical investigation

Ask the user to confirm or edit the draft before writing it.

## Step 3 — Append to HISTORY.md
Once confirmed, append the step to the bottom of `HISTORY.md`.

## Step 4 — Check for PROGRESS.md updates
Scan PROGRESS.md for sections that are now stale based on what was done:
- Did a new AUROC beat the "Best results" table? Update it.
- Did an experiment complete? Update its status (e.g., "READY TO RUN" → "COMPLETE").
- Did a blocker get resolved? Remove or update it.
- Does "Immediate next actions" need reordering?

List the proposed PROGRESS.md changes and ask the user to confirm before writing.

## Step 5 — Write PROGRESS.md changes
Apply the confirmed changes. Update the `**Last updated**` date at the top to today's date.

## Step 6 — Commit
Stage `HISTORY.md` and `PROGRESS.md` (and any other files changed this session that haven't been committed yet). Commit with:
```
git commit -m "Step N: <same title as HISTORY.md entry>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

Then push to `origin feature/meta-agentic-integration`.
