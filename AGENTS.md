# AGENTS.md — Codex project instructions

## Session start and canonical rules

- Before doing project work, read `CLAUDE.md` completely and follow it as the canonical detailed project guide.
- Read `PROGRESS.md` at the start of every session before relying on git history or choosing the next action.
- Treat this file as a Codex adapter, not a second copy of the project rules. If project guidance changes, update `CLAUDE.md`; update this adapter only when Codex-specific discovery or translation changes.
- Translate Claude-specific mechanisms to the available Codex equivalents. For example, when a workflow refers to a slash command, read the matching file under `.claude/commands/` and perform that workflow with the available tools. Do not assume Claude-only commands, agent types, or tool names exist in Codex.
- Follow active user, system, and developer instructions when they impose stricter requirements than the repository guidance.
- Read `docs/AGENT_OPERATING_CONTRACT.md` (permission boundary, reversible-or-auditable, fail loudly and stop, memory file, plan/test/simplicity) and the "Standing rules" table at the top of `LESSONS.md` at every session start. Append a `LESSONS.md` entry when a mistake is caught and at the end of any session that had one; both agents share that file. Before any cluster/GPU submission, perform the workflow in `.claude/commands/preflight.md` and refuse to submit without its PASS. The destructive-git commands listed in `.claude/hooks/guard_git.py` need a human in every tool.

## Local Python runtime

- For local project scripts, tests, and numerical experiments, use the repository virtual environment explicitly: `/Users/osegev/Desktop/hallucination_detection/.venv/bin/python` (Python 3.11; currently 3.11.9). Do not use the system `python` or `python3` for this repository.
- Invoke Python tooling through that interpreter as well, for example `.venv/bin/python -m pytest` or `.venv/bin/python -m pip`. Worktrees should reuse the main checkout's virtual environment rather than expecting a separate `.venv` inside each worktree.
- Before a numerical run, verify the interpreter and required imports with `.venv/bin/python -c 'import sys, numpy, scipy; print(sys.executable, sys.version, numpy.__version__, scipy.__version__)'`. A failed shell glob or worktree-local lookup is not evidence that the environment is missing.
- Do not create a replacement environment or install/upgrade packages unless the user explicitly authorizes it.

## Google Drive data

- Large experiment data, inference caches, and cluster results are stored in Google Drive under the configured rclone remote `gdrive:`.
- The main project prefix is `gdrive:hallucination_detection/`; important subpaths include `gdrive:hallucination_detection/cluster_results/` and `gdrive:hallucination_detection/consolidated_results/`.
- Access Google Drive with the already-configured `rclone` CLI. Prefer read-only discovery commands such as `rclone lsd`, `rclone lsf`, `rclone lsjson`, `rclone size`, and `rclone cat` when inspecting data.
- Before downloading or copying a Drive artifact, inspect its path, size, modification time, and accompanying manifest/report. Large pickle files can be hundreds of MB or more.
- Do not delete, overwrite, move, or bulk-sync Drive data unless the user explicitly authorizes that mutation and the exact source/destination paths have been verified.
- For cluster-to-Drive transfers, follow the AIRCC rules in `CLAUDE.md` and `cluster/README.md`; large results should move directly between the cluster and `gdrive:` rather than through the local machine.
