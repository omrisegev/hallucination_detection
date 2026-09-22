---
description: Initialize a work session. Reads PROGRESS.md, shows git status and last commits, surfaces last 3 HISTORY steps, and identifies the highest-priority next action. Run this at the start of every session before doing anything else.
---

Run the following session initialization sequence in order. Do not skip steps.

## Step 0 — Orient: which code is live?
```
git fetch --all --prune
git branch -a --sort=-committerdate | head -15
git worktree list
git log --oneline -5
```
State in one plain sentence which branch and worktree you believe is the live pipeline
and **wait for Omri to confirm before reading any code**. Three sessions were lost to
analysing an archived branch (LESSONS.md). Then read `docs/AGENT_OPERATING_CONTRACT.md`
and the "Standing rules" table at the top of `LESSONS.md`.

## Step 1 — Read PROGRESS.md
Read `PROGRESS.md` in full. Summarize the current state as exactly 5 bullets:
- **Branch**: which git branch is active and why
- **Active experiments**: what is currently running or partially done on Colab
- **Blockers**: anything that needs to happen before the next experiment can run
- **Drive cache status**: which Phase caches are complete vs missing
- **Next actions**: the numbered list from "Immediate next actions"

## Step 2 — Git state
Run `git log --oneline -5` and `git status`. Report:
- Current branch name
- How many commits ahead of origin
- Any uncommitted changes (staged or unstaged)

## Step 3 — Last 3 HISTORY steps
Read the tail of `HISTORY.md` (last ~80 lines). Extract and print the titles and one-line Results of the last 3 steps.

## Step 4 — Session checklist
Print this checklist as-is:
```
Before-session checklist:
[ ] Read PROGRESS.md ✓ (just done)
[ ] Confirm git branch and worktree with Omri (Step 0) — never assume from PROGRESS.md alone
[ ] List Drive folders — which Phase caches exist?
[ ] Check last 5 HISTORY steps for open failures or pending fixes
[ ] Verify CLAUDE.md gptqmodel install order if planning a GPU session
```

## Step 5 — Priority action
Based on PROGRESS.md "Immediate next actions", print a single bold line:

**PRIORITY THIS SESSION: <action #1 from the list, paraphrased in one sentence>**

Then stop and wait for the user to direct the session.
