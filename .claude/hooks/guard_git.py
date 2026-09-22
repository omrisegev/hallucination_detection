#!/usr/bin/env python
"""
guard_git.py — Claude Code PreToolUse hook that BLOCKS destructive git / tree commands.

Why this exists (LESSONS.md, four distinct incidents):
  2026-07-07  concurrent sessions aborted an in-progress merge twice
  2026-09-13  a "dry-run" merge script ran `git reset --hard master` in the root
              checkout and destroyed six uncommitted Codex edits
  2026-09-17  `git sparse-checkout add` wiped Codex's local-only npz results
  2026-09-18  a worktree materialised 29 GB of LFS objects and starved two sessions

Behaviour (Omri's choice, 2026-09-22): block and explain. The hook never stashes or
runs anything on the user's behalf. It prints what is dirty in every worktree and
tells the agent to hand the command to a human.

Wiring (.claude/settings.json):
    "hooks": {"PreToolUse": [{"matcher": "Bash|PowerShell",
              "hooks": [{"type": "command",
                         "command": "python \"$CLAUDE_PROJECT_DIR/.claude/hooks/guard_git.py\""}]}]}

Exit codes: 0 = allow, 2 = block (stderr is shown to the agent).
Self-test:  python .claude/hooks/guard_git.py --self-test
"""
import json
import os
import re
import subprocess
import sys

# Each pattern is matched case-insensitively against the whole command string, so a
# blocked command hiding inside `cd x && git reset --hard` or a heredoc is still caught.
BLOCK_PATTERNS = [
    (r"\bgit\b[^|;&\n]*\breset\b[^|;&\n]*--hard", "git reset --hard discards uncommitted edits"),
    (r"\bgit\b[^|;&\n]*\bclean\b[^|;&\n]*-[a-zA-Z]*[fdxX]", "git clean deletes untracked files (local-only results live there)"),
    (r"\bgit\b[^|;&\n]*\bcheckout\b[^|;&\n]*\s--\s+\.", "git checkout -- . discards every working-tree edit"),
    (r"\bgit\b[^|;&\n]*\brestore\b(?![^|;&\n]*--staged)[^|;&\n]*\s\.", "git restore . discards working-tree edits"),
    (r"\bgit\b[^|;&\n]*\bsparse-checkout\b[^|;&\n]*\b(add|set|reapply|disable)\b", "sparse-checkout changes delete ignored files in the worktree (2026-09-17 incident)"),
    (r"\bgit\b[^|;&\n]*\bworktree\b[^|;&\n]*\b(remove|prune)\b", "worktree remove/prune can delete another agent's uncommitted work"),
    (r"\bgit\b[^|;&\n]*\bbranch\b[^|;&\n]*\s(?-i:-D)\b", "branch -D force-deletes an unmerged branch"),
    (r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*(--force\b|-f\b|--force-with-lease)", "force push rewrites shared history"),
    (r"\bgit\b[^|;&\n]*\bstash\b[^|;&\n]*\b(drop|clear)\b", "stash drop/clear destroys saved edits"),
    (r"\bgit\b[^|;&\n]*\bfilter-(branch|repo)\b", "history rewrite"),
    (r"\brm\b[^|;&\n]*-[a-zA-Z]*[rR][a-zA-Z]*\s+[^|;&\n]*\.worktrees", "rm -r on .worktrees deletes other agents' checkouts"),
    (r"Remove-Item\b[^|;&\n]*-Recurse[^|;&\n]*\.worktrees", "Remove-Item -Recurse on .worktrees"),
]

# Commands that only *look* dangerous. Checked first; if one matches, allow.
ALLOW_PATTERNS = [
    r"\bgit\b[^|;&\n]*\breset\b(?![^|;&\n]*--hard)",          # git reset (soft/mixed), git reset HEAD file
    r"\bgit\b[^|;&\n]*\bclean\b[^|;&\n]*\s-[a-zA-Z]*n",         # git clean -n (dry run)
    r"\bgit\b[^|;&\n]*\bworktree\b\s+(list|add|lock|unlock)\b",
    r"\bgit\b[^|;&\n]*\bstash\b\s+(push|list|show|pop|apply)\b",
    r"\bgit\b[^|;&\n]*\bsparse-checkout\b\s+list\b",
    r"--self-test",
]


def classify(command: str):
    """Return (blocked: bool, reason: str)."""
    cmd = command or ""
    for pat in ALLOW_PATTERNS:
        if re.search(pat, cmd, flags=re.IGNORECASE):
            # An allow pattern only exempts the *same* segment; still scan for other blocks.
            pass
    for pat, why in BLOCK_PATTERNS:
        if re.search(pat, cmd, flags=re.IGNORECASE):
            # Dry-run clean and non-hard reset are explicitly allowed even if a block pattern grazes them.
            if why.startswith("git clean") and re.search(r"\bclean\b[^|;&\n]*\s-[a-zA-Z]*n", cmd):
                continue
            return True, why
    return False, ""


def worktree_status():
    """Read-only summary of every worktree's dirty-file count. Never modifies anything."""
    lines = []
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception as exc:  # git missing, not a repo, timeout
        return [f"(could not list worktrees: {exc})"]
    paths = [l.split(" ", 1)[1] for l in out.splitlines() if l.startswith("worktree ")]
    for p in paths:
        try:
            st = subprocess.run(["git", "-C", p, "status", "--porcelain"], capture_output=True,
                                text=True, timeout=20).stdout
            n = len([l for l in st.splitlines() if l.strip()])
            br = subprocess.run(["git", "-C", p, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
            lines.append(f"  {n:4d} dirty  {br:45s} {p}")
        except Exception as exc:
            lines.append(f"  ??  {p} ({exc})")
    return lines


def main():
    if "--self-test" in sys.argv:
        return self_test()
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # no JSON => not a tool call we understand; never block on a parse error
    tool = payload.get("tool_name", "")
    if tool not in ("Bash", "PowerShell"):
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    blocked, why = classify(command)
    if not blocked:
        return 0
    msg = [
        "BLOCKED by .claude/hooks/guard_git.py: destructive git/tree command.",
        f"  reason : {why}",
        f"  command: {command.strip()[:300]}",
        "",
        "Worktrees and their uncommitted files right now (read-only):",
        *worktree_status(),
        "",
        "Do NOT retry or work around this. Report the exact command to Omri and ask him to run it",
        "in his own terminal after stashing or committing every dirty worktree above.",
        "A parallel Codex session may own those edits (LESSONS.md 2026-09-13, 2026-09-17).",
    ]
    sys.stderr.write("\n".join(msg) + "\n")
    return 2


def self_test():
    cases = [
        ("git reset --hard master", True),
        ("cd .worktrees/x && git reset --hard HEAD~1", True),
        ("git clean -fd", True),
        ("git clean -fdx", True),
        ("git clean -n", False),
        ("git reset HEAD HISTORY.md", False),
        ("git reset --soft HEAD~1", False),
        ("git checkout -- .", True),
        ("git checkout -b feature/x", False),
        ("git restore --staged HISTORY.md", False),
        ("git restore .", True),
        ("git sparse-checkout add results/foo", True),
        ("git sparse-checkout list", False),
        ("git worktree remove .worktrees/foo", True),
        ("git worktree prune", True),
        ("git worktree list", False),
        ("git worktree add .worktrees/y branch-y", False),
        ("git branch -D old-branch", True),
        ("git branch -d merged-branch", False),
        ("git push --force origin main", True),
        ("git push -f", True),
        ("git push origin master", False),
        ("git stash drop", True),
        ("git stash push -u -m wip", False),
        ("rm -rf .worktrees/foo", True),
        ("rm -rf build/", False),
        ("Remove-Item -Recurse -Force .worktrees\\foo", True),
        ("git status --porcelain && git log --oneline -3", False),
        ("python scripts/score_repgrid.py --cells x", False),
    ]
    bad = []
    for cmd, expect_block in cases:
        got, why = classify(cmd)
        if got != expect_block:
            bad.append(f"  {'BLOCK' if expect_block else 'ALLOW'} expected, got {'BLOCK' if got else 'ALLOW'}: {cmd}  ({why})")
    if bad:
        print("guard_git self-test FAILED:\n" + "\n".join(bad))
        return 1
    print(f"guard_git self-test PASS ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
