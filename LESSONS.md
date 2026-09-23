# LESSONS.md — the agents' memory of their own mistakes

**Read the "Standing rules" section at the start of every session. Append an entry the
moment a mistake is caught, and at the end of any session that had one.** This is the
memory file from `docs/AGENT_OPERATING_CONTRACT.md` §4. Both Claude Code and Codex read
and write it; it is the one place a lesson survives a context reset, a branch switch, or
a change of tool.

Seeded 2026-09-22 from `docs/research_notes/PROJECT_RETROSPECTIVE_2026-09-22.md` and
the 23 dated corrections in Claude's private memory. Full evidence, step numbers and
counts are in the retrospective; this file keeps only the rule and what enforces it.

Entry format:

```
## YYYY-MM-DD — <one-line mistake or lesson>
What happened: ...
Why: ...
Rule: ...
Enforced by: code | command | prose   (name the file)
```

---

## 2026-09-23 — Disk-full writes can truncate canonical documentation

What happened: apply_patch truncated CLAUDE.md after the volume filled. The original
content was restored exactly; a temporary emergency hardlink was detached and its
link count verified as1 before editing either copy. User-authorized removal of two
clean GitHub-backed inactive worktrees restored space. No experiment data lost.
Why: a small free-space reading was treated as enough for subsequent writes while
other activity exhausted the remaining space; the edit did not fail atomically.
Rule: after disk exhaustion, require a useful free-space margin before rewriting
canonical files. Prefer write-to-temp, verify and replace. Never edit an emergency
hardlink shared with another worktree; detach it first. Check ignored files and
actual remote ancestry, not just clean git status, before deleting a worktree.
Enforced by: prose and recorded recovery checks in docs/reviews/WORKTREE_CLEANUP_20260923.md.

## Standing rules (read these)

| Rule | Enforced by | Recurred after the rule? |
|---|---|---|
| Never `reset --hard`, `clean -fd`, `sparse-checkout add`, `worktree remove` without a human | **code**: `.claude/hooks/guard_git.py` (2026-09-22) | 4 incidents while it was prose |
| No AUROC on < 10 minority rows, acc outside [0.05, 0.98], or > 10 % cap-pinned traces | **code**: `spectral_utils/label_sanity.py` in `score_repgrid.py`, `score_ubaselines.py`, `inspect_cell.py` (2026-09-22) | 8 label bugs while unchecked |
| A number on fewer rows than the full population is `FEASIBILITY`, never a headline | **code**: `label_sanity.feasibility_tag` | Step 327 (30.16 % pilot vs 20.12 % rest) |
| No sbatch without a same-session `/preflight` PASS | **command**: `.claude/commands/preflight.md`, `aircc-submit.md` Step 0 | 5+ multi-hour jobs died on smoke-catchable bugs |
| Orient on branches/worktrees before reading code | **command**: `session-start.md` Step 0 | 3 sessions on an archived branch |
| Ask which method/arm/entry point to score; mirror the canonical scorer | **prose**: `CLAUDE.md` "Which method to evaluate" | yes, 3× (07-30, 08-01, 08-03) |
| One timeout is not an outage; probe twice, report the observation | **prose**: contract §3 | yes (09-17, 09-19) |
| A bash parse error means nothing ran; check the filesystem before "done" | **prose**: contract §3 | new (09-18) |
| State the N actually checked with every claim | **prose**: contract §7, `/red-team` | yes (3 of 55 selectors; PR verdict 2×) |
| Plain-language first: one sentence, one number, next step | **prose**: `CLAUDE.md` Communication style | yes (2× "REWRITE IN SIMPLE ENGLISH") |
| Terminology bans: not "Nadler", not "MV_EPR", not "recommended", not "CONT" | prose, `feedback_terminology` | no |
| Notebook JSON is edited with NotebookEdit / json.dump, never str.replace | **code**: `.git/hooks/pre-commit` (untracked!) | no |

---

## 2026-09-22 — HISTORY.md in the main worktree is a Claude-lane-only lineage
What happened: the retrospective read `HISTORY.md` on `codex/token-local-fusion-optimization-v1` and found Steps 336-355, 360-398 and 413-428 missing. They exist on `master` and `consolidation/fusion-2026-09-22` (`.worktrees/readout-quickest-detection-v1/HISTORY.md`, 57,727 lines, three concatenated copies of the log).
Why: the branch forked from the Claude log before the Codex blocks were merged; three lines appending to one prose log diverge by construction.
Rule: before summarising history, check which lineage the file is (`grep -c '^# MV_EPR Project History'` > 1 means concatenated copies; compare step coverage with `master`). Omri decides whether to merge the log back.
Enforced by: prose (this entry); open decision for Omri.

## 2026-09-19 — Declared the VPN down from one 5-second ssh timeout
What happened: one probe timed out; several turns were spent planning around an outage that did not exist.
Why: inferred cause reported instead of the observation.
Rule: probe twice, second with `ConnectTimeout=30`; report "two probes timed out", not "VPN is down".
Enforced by: prose, contract §3.

## 2026-09-18 — Reported a document section as written when a nested heredoc had failed to parse
What happened: `unexpected EOF` from a `$(cat <<EOF)` inside a heredoc; nothing in the chain ran; the write was reported as done. Omri's reviewing agent caught it.
Why: a parse error was treated like a runtime error of one step.
Rule: write files with the Write tool, commit with `-F`, and `ls`/`head` the file before reporting.
Enforced by: prose, contract §3.

## 2026-09-18 — A participation-ratio result was proposed as a go/no-go gate
What happened: PR rose 2.46 → 2.78 (largest addition ever) while quality fell 0.73 pp; two independent sessions failed at the same comparability point.
Why: a descriptive statistic was given verdict authority.
Rule: a statistic is not a verdict; write in advance what a stage's output can and cannot close. Pick the reference bank by coverage, not by incumbent status.
Enforced by: prose, `feedback_pr_not_a_quality_gate`.

## 2026-09-18 — A worktree materialised 28 GB of LFS objects and starved two sessions
Rule: `GIT_LFS_SKIP_SMUDGE=1 git worktree add`; `git status --porcelain` inside before removing one.
Enforced by: prose + hook blocks `worktree remove`.

## 2026-09-17 — `git sparse-checkout add` deleted Codex's local-only npz results
Rule: back local-only results up to Drive first; use `git add --sparse`, never `sparse-checkout add/reapply`.
Enforced by: **code**, `guard_git.py`.

## 2026-09-17 — A Slurm failure read as an outage was a missing `SLURM_CONF_SERVER`
Rule: `export SLURM_CONF_SERVER=controller-primary` or `bash -lc`; check env before declaring the cluster down.
Enforced by: prose, `project_aircc_slurm_conf_server`.

## 2026-09-13 — A dry-run merge script ran `git reset --hard master` in the root checkout
What happened: six uncommitted Codex edits across seven files destroyed.
Rule: no destructive git without a human; stash-or-commit every worktree first.
Enforced by: **code**, `guard_git.py`.

## 2026-09-12 — A plan draft asserted "weights do not matter" and broadcast token labels
Rule: required non-conclusive wording for interim stages; interim ≠ complete; bind replay checks to K, function and file.
Enforced by: prose, `feedback_fusion_framing_and_controls`.

## 2026-09-07 — PRMBench `error_steps` are one-based; `flags[step]` shifted every label (Step 313)
What happened: 6,035 target arrays wrong; every prior PRMB conclusion withdrawn; 207 comparisons re-bridged.
Rule: raw-annotation verification must be independent of the derived NPZ; use `spectral_utils.prm_label_contract`.
Enforced by: **code**, `prm_label_contract.py` + its tests.

## 2026-09-07 — Pilot cohort numbers set direction, then regressed on the full population (Step 327)
Rule: subsets are feasibility checks only; never rank directions or promote a candidate from one.
Enforced by: **code**, `label_sanity.feasibility_tag` (2026-09-22); policy in `CLAUDE.md`.

## 2026-08-05 — 21 published feature-selection keep rules transplanted faithfully; all 111 variants lost (Step 224)
Rule: a published metric is inspiration, not a specification. One variant, one discussion, then build.
Enforced by: prose, `CLAUDE.md` "Borrowing from a paper".

## 2026-08-03 — Built an arm around GOOD_6 because CLAUDE.md said so; ran the legacy `upcr_pipeline` and called it U-PCR (Step 219)
Rule: ask which method to score; find the script that produced the last reported number and mirror it.
Enforced by: prose, `CLAUDE.md` "Which method to evaluate". Recurred 3×.

## 2026-08-03 — Shorthand labels (R2/T4/Rb) in chat, twice in one day; tables split across sections
Rule: name each experiment by what it does; one grid for ours-vs-theirs, direction in the row label, winner marked.
Enforced by: prose, contract §8.

## 2026-08-02 — Gemini's transform sweep optimised on zero in-scope cells; our own `nonmono_gain` used `max(p, 1−p)`
Rule: verify a delivered result's objective definition: print the resolved cell list; read the code behind any derived column. `max(a, 1−a)` on a supervised score is a sign oracle and only ever inflates (found in five places).
Enforced by: prose; candidates for a lint rule.

## 2026-08-01 — Anchored an analysis on a cell the repo had already flagged as length-leaked (6 positives)
Rule: grep the repo for a cell key before making it an exemplar; a top AUROC is suspicion, not selection.
Enforced by: **code** (2026-09-22): 6 positives now fails `label_sanity`.

## 2026-07-30 — Read the `a2.dufs` column as U-PCR; wrong numbers into HISTORY
Rule: do not infer an arm from a column name.
Enforced by: prose.

## 2026-07-13 — Gemini fabricated authors, venues and results in 5 of 5 spot-checked paper digests, twice in a day
Rule: Gemini does analysis only; cross-check every checkable claim against `papers/extracted/`.
Enforced by: prose, `feedback_gemini_role`. No recurrence.

## 2026-07-12 — Advisor report headlined an in-house baseline (seq-logprob) instead of the cited roster
Rule: lead with the published-paper scoreboard; audit baselines are appendix.
Enforced by: prose, `feedback_published_roster_headlines`.

## 2026-07-11 — Three bugs from writing `spectral_utils` signatures from memory (`boot_auc` arg order, 4-tuple return, hand-typed subsets)
Rule: read and mirror `repgrid_scoring.score_subset` before any new scorer.
Enforced by: prose, `feedback_read_canonical_scorer_first`.

## 2026-07-11 — Benchmark cells left in limbo
Rule: every cell ends scored-in-CSV or documented-REJECT; judge-regrade floors before REJECT.
Enforced by: **code**, `report_figs.gate_flag` + `REJECT_REGISTRY`. No recurrence.

## 2026-07-01 — "5-feat L-SML is always K=2 with 0.5/0.5" restated as fact, never verified from disk
Rule: verify from disk before restating a structural claim; 16-feat is the primary clustering test.
Enforced by: prose.

## 2026-06-25 — Method called "Nadler"/"MV_EPR"; supervised `best_nadler_on` beside unsupervised rows; "recommended" in advisor mail
Rule: method = L-SML; never those names; never compare to the supervised comparator.
Enforced by: prose, `feedback_terminology`. No recurrence.

## 2026-06-01 — Proposed runtime adaptive anchors and a four-step prompt template
Rule: offline-derived constants over runtime mechanisms; subtle prompt clauses, "no need to exaggerate".
Enforced by: prose.

## 2026-05-27 — `str.replace` on notebook JSON produced an unopenable notebook
Rule: NotebookEdit or json.load/dump only.
Enforced by: **code**, `.git/hooks/pre-commit` (untracked; recreate on a fresh clone). No recurrence.

## 2026-05-22 — End-only save lost 11/16 cells on a Colab disconnect
Rule: loops > 5 keys or > 30 s save after every key with partial resume.
Enforced by: prose + `save_cache_atomic` pattern. No recurrence.

## 2026-05-12 — Retry loop on an oversized notebook edit; "are you working? stuck?"
Rule: after two failed retries, write a paste-in fix document instead.
Enforced by: prose.
