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

## 2026-09-24 — Do not rely on a requested shell working directory for project reads

What happened: the first session-start read used relative paths with a requested working directory, but PowerShell resolved them under `C:\` and did not read the project files.
Why: the command assumed the runner would honor `workdir` without verifying the resolved location.
Rule: for required project reads, use verified absolute paths (or first print the working directory) before relying on the output.
Enforced by: prose (AGENTS.md session-start procedure).

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
| z-score every matrix given to `lsml_continuous`; rerun at another input scale before explaining a surprising K | **code**: `tail_calib_common.lsml_fit_scaled` + scale test (2026-09-24) | new (09-24: K=2 artefact locked and run externally) |

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

## 2026-09-24 — A new object fed to `lsml_continuous` without z-scoring collapsed K to 2, and the artefact was reported as a finding
What happened: step-level tail-mark L-SML (Steps 441-442) passed centred 0/1 marks (variance ~0.18) straight to `lsml_continuous`. K=2 came out on every fold and bank. It was reported and locked as "tail learning collapses to K=2" with a mechanistic story (a side conversation blamed CUSUM end-of-answer marks). Step 443 showed K=2 is produced by input scale: the same continuous features multiplied by 0.4 also give K=2, and standardized marks give a stable K=4. The locked V1 candidate and its external run carried the artefact.
Why: `lsml_continuous` documents z-scored inputs, but nothing enforces it. Its default `loading_scale='unit'` Eq.14 K criterion is not scale invariant. `METHOD_NOTE.md` said "standardize" while the code did not.
Rule: z-score every matrix passed to `lsml_continuous` (or use `loading_scale='complete'`). Before explaining a surprising K, run the scale control: rerun at another input scale.
Enforced by: code, `tail_calib_common.lsml_fit_scaled(standardize=True)` with the scale-invariance test in `scripts/experiments/test_tail_calib_common.py`; prose for other callers.

## 2026-09-24 — A write-once test passed on its own "not caught" sentinel
What happened: the ScoreBundle overwrite test did `try: second_write(); raise AssertionError('overwrite not caught') except AssertionError as e: assert 'overwrite' in str(e)`. The sentinel text contains "overwrite", so the test passed while `_init` re-created the arrays on every write (a key check on `m` against a dict keyed `(m, role)`). The full run then failed at finalize after 7 minutes of fits.
Why: the same `except` catches the sentinel, and string matching cannot tell it from the real error.
Rule: set a flag only on the specific real message and assert the flag after the `try`. Also test the positive path end to end.
Enforced by: code, `test_calfix_common.py` and `test_tail_calib_common.py` (flag pattern).

## 2026-09-27 — The evaluation/calibration score overwrite recurred because a runner was copied from a defective template
What happened: `expectation_realization_run.py` was built by copying `declared_joint_run.py`, whose fold loop writes each model's scores into ONE array for both roles (`for rr in (ev_rows, cal_rows): scores[...][rr] = ...`). In five folds every fold is written twice, so fold 0 ends up scored by the fold-4 model and the PRMScore threshold of folds 0-3 comes from a model fitted on the evaluation fold (label-free, but not the declared contract); a failed fit would also silently keep the previous model's scores. Step 442 had already fixed this exact defect for the family/tail runs, yet the Step 438, 439 and 440 runners (`bank20_lsml_run.py`, `indbank_lsml_run.py`, `declared_joint_run.py`) still carry it. A one-fold smoke cannot show it. An independent review subagent caught it before the full run.
Why: the fix lived in one module (`calfix_common.ScoreBundle`) and was never propagated to the older runners that serve as templates; within-answer AUC stays out-of-fold, so no number looked wrong.
Rule: every fold-k model writes ONLY its evaluation rows; its threshold is computed inside the fold loop from the same model's calibration-fold scores; assert write-once per row. Never copy a runner as a template without checking its fold-role handling, and run a multi-fold smoke (or a write-order toy replay) before trusting role separation.
Enforced by: code, the `put()` write-once assert in `scripts/experiments/expectation_realization_run.py` and `calfix_common.ScoreBundle`. Update 2026-09-27 (Step 453): the three named runners AND three more found by grep (`error_cluster_lsml_run.py`, `core_virtual_lsml_run.py`, `partition_ceiling_run.py`) now use the same `put()` (write-once `RuntimeError`, same-model calibration array saved as `CAL_SCORES.npz`) and refuse to write into an existing run folder; re-scored in `run_20260927_calfix`, checked by `scripts/experiments/old_runners_calfix_verify.py`.

## 2026-09-27 — "Averaging" was a handicapped control, and no position-only reference was reported
What happened: expectation_realization_v1 compared L-SML and block-equal fusion against plain equal weighting of 13 channels, with the protocol stating that every column is risk-oriented. Two change channels (energy_innovation 0.407, top50_js 0.458 single-channel within-AUC) are anti-oriented within answer, as known since Step 438. The red team showed that equal weighting without those two ties block_equal (0.7802 vs 0.7805), so the "block structure" gain was a down-weighting of two wrong-signed channels. It also found that step index alone scores 0.6617 within answer on PRMBench, that part of the new derivative channel's signal is that positional prior (its margin over CT7 falls from +0.0081 to +0.0026 once the positional profile is removed), and that the single post-hoc channel (0.7957) beats every fused arm.
Why: controls were chosen by label-free form (equal, block-equal) without checking which inputs are anti-oriented, and no trivial position baseline was in the design.
Rule: in every PRMBench within-answer comparison, report (a) a position-only reference (step index) and (b) the equal control with and without channels whose single-channel within-AUC is below 0.5, labelled as a label-informed diagnostic. Never call a column risk-oriented without its single-channel AUC in the table.
Enforced by: prose (this entry, RED_TEAM.md of expectation_realization_v1).

## 2026-09-27 — A within-answer gain passed the shuffle null but was reproduced by position alone
What happened: expectation_realization_v1 stage B found that dropping the channels flagged by label-free Dawid-Skene estimates and averaging the rest beats averaging all 13 channels on PRMBench (+0.0053 within-AUC, z 4.5 against a within-answer label shuffle). The red team's whole-answer label swap (each answer gets the labels of another answer with the same step count) gave a null mean of +0.0063, at or above the observed gain: the dropped channel (energy_innovation) falls along the answer, and removing it helps wherever errors sit late, whatever the content. A content gain of +0.0043 survives only on a position-adjusted bank. Separately, the frozen protocol changed the tie handling of the 20% marks relative to stage A without re-checking, label-free, that the partition motivating the design still appeared; it did not, and an amendment was needed after the smoke.
Why: the null family tested "are labels informative at all" but not "is the gain more than the error-position prior"; and a design premise (the family partition) was carried over from a different input recipe without re-verification.
Rule: every within-answer PRMBench gain gets BOTH nulls: within-answer label shuffle AND whole-answer label swap within the same step count, and the position-adjusted replicate is reported beside the headline. Before freezing a protocol, re-run every label-free premise (partitions, survivor sets) on the exact frozen input recipe.
Enforced by: prose (this entry, results/expectation_realization_v1/run_20260927_stage_b/RED_TEAM.md).

## 2026-09-27 — A preliminary report read three artefacts as evidence
What happened: in the stage-B2 preliminary report (before the red team) I presented (1) "the estimated weights beat equal group weights" as value from the binary estimates, (2) a +0.0155 gain on the position-adjusted bank as a strengthening, and (3) a drop of the clean-step group correlation from 0.52 to 0.31 as reduced dependence. The red team showed that (1) equal group weights are a weak reference (below plain averaging of the same channels) beaten simply by down-weighting one incoherent group, (2) the position bank split that group in two, making the reference weaker still, and (3) 0.52 was the correlation of the two level sub-groups that the merge deletes by construction.
Why: I compared the new rule with the protocol's matched control only, not with the strongest simple control on the same channels, and read a summary statistic without checking what changes it by construction.
Rule: a new weighting is always compared with the plain average of the SAME channels, not only with a structured control; a diagnostic whose value can change by construction (a deleted pair, a changed partition) is reported with that mechanism, never as evidence; a preliminary report says which reference each number is relative to.
Enforced by: prose (this entry, results/expectation_realization_v1/run_20260927_stage_b2/RED_TEAM.md).

## 2026-09-27 — A hard-coded family list was reported as if the method had found it
What happened: stage B2 of expectation_realization_v1 merged "the two level groups" and removed "two of the five level channels" through a fixed list `LEVEL` in the runner. The protocol justified the list by the stable discovered partition, but the code did not discover it, and the first report presented the merge rule (0.7818) as a result of the method. Omri's goal was explicitly to avoid predeclared families. The same report also added gain components computed on two different paths (merge and removal).
Why: the list was convenient for a mechanism test and its provenance was argued in the protocol text, so its non-algorithmic status was never stated next to the number.
Rule: any row that depends on a fixed channel list, a manual merge or a post-hoc choice carries that label in every table and summary ("mechanism test, not algorithmic"); a gain decomposition uses one path of paired comparisons whose parts sum to the total, and names the pair behind each part.
Enforced by: prose (this entry, docs/HANDOFF_EXPECTATION_REALIZATION_2026-09-27.md section 5).

## 2026-09-27 — A lesson named three affected runners; a grep found three more
What happened: the overwrite entry above listed the Step 438-440 runners as "the three older runners" still carrying the defect. When they were fixed (Step 453), the independent reviewer grepped the pattern `for rr in (ev_rows, cal_rows)` and found it in three more runners with reported results: `error_cluster_lsml_run.py` and `core_virtual_lsml_run.py` (Step 441 numbers that the Step 442 calfix never re-scored, although Step 441's title says its evaluation was superseded) and `partition_ceiling_run.py`. Most numbers moved by less than 0.002 and no verdict changed; the largest change, err50 L-SML within-AUC 0.5733 -> 0.5902 (still far below its equal control 0.7028), sat in a runner nobody had listed.
Why: the affected set was written from memory of which runners had been copied, not from a search of the code.
Rule: when a defect pattern is identified, grep the whole `scripts/` tree for it and list every hit with its results folder before claiming a scope; "superseded by step N" must name which arms step N actually re-scored.
Enforced by: prose (this entry); `grep -rn "for rr in (" scripts/` now returns nothing.

## 2026-09-27 — Python `write_text` on Windows rewrote LF files as CRLF, turning four small edits into whole-file diffs
What happened: Step 453 edited HISTORY.md, PROGRESS.md, LESSONS.md, a handoff and three runners with `Path.read_text()` / `write_text()`. On Windows `write_text` translates every newline (LF) to CR+LF, so the commit showed 36,633 changed lines in HISTORY.md alone. The repo has no `* text=auto` and `core.autocrlf` is false, so git stored the CRLF bytes. It was caught from `git show --stat` and amended before any push; two runs had already hashed the CRLF working copies.
Why: text-mode writes on Windows convert newlines, and the scripted edits never looked at the byte-level diff.
Rule: scripted edits read and write bytes (`read_bytes().decode()` / `write_bytes(...encode())`) or pass `newline=''`; before every commit check `git diff --cached --stat` against the expected size of the change.
Enforced by: prose (this entry).

## 2026-09-27 — A PRMScore decomposition reported panels without their null baselines
What happened: prmscore_decomposition_v1 (frozen protocol, all numbers exact) reported paired contrasts, a clean-answer false-alarm panel and a first-error-position panel. The red team showed three readings needed baselines the design lacked: (1) about 70% of the filter gain and the fused row's apparent tie with realized_drv are reproduced by a whole-answer label swap between answers of equal length (position structure); (2) under our answer-z rule, 0.0140 of the 0.0168 gap to the PRM survives a within-answer label permutation (it is flag-count allocation, not placement); (3) "99% of clean answers get a false flag" is produced by random scores under the same rule; (4) the position gradient mirrors class composition.
Why: the standing rule to run both nulls was applied to within-answer AUC gains in earlier stages but not carried into a new kind of analysis (a decomposition of a threshold metric); stratifications by label-derived variables were reported without their class mix.
Rule: every paired contrast in a decomposition carries the within-answer permutation AND the whole-answer swap null residual; every rate-under-a-rule panel carries a random-score baseline under the same rule; every stratification by a label-derived variable reports its composition by class before any effect is read.
Enforced by: prose (this entry; results/expectation_realization_v1/prmscore_decomposition_v1/RED_TEAM.md).

## 2026-09-28 — A secondary bank's gain was called "content" although it had no null behind it
What happened: in er_generality_v1 the runner computed permutation nulls only for the primary banks (B20, B32). My preliminary report nevertheless said the filter's gain on the secondary bank B51 (+0.0128) was content; the red team ran the whole-answer same-length swap there and ~70% of it is position. The same report proposed a mechanism for the simple filter's failure (a level-dominated mean) that the red team refuted, and omitted that the B32 gain comes 86% from 61 answers.
Why: I extrapolated the primary banks' null result to a bank without one, and wrote a mechanism as an explanation before testing it.
Rule: never characterise a gain as content or position without that bank's own null; a proposed mechanism is labelled a hypothesis until a test supports it; every headline gain carries its concentration (share from the top 1% of answers, trimmed mean).
Enforced by: prose (this entry, results/er_generality_v1/run_20260927/RED_TEAM.md).

### 2026-09-24 AIRCC external collection startup

Job265833 failed to resolve package-index DNS inside Pyxis before model load;
265834/265835 cancelled while pending. Current account is cycle3/owner_940;
old shared data and model cache remain accessible. Use pinned offline wheels,
preserve NGC torch AND numpy, and prepare missing models on CPU compute nodes.
Never assume login-node DNS proves container network access.

## 2026-09-24 - Validate the deployed archive, not only the checkout

What happened: QwQ timing job265865 failed after loading weights because a manually
selected code archive omitted cluster/backfill_specs.py, imported indirectly by
the scorer. No telemetry was collected;105 allocated GPU seconds were consumed.
Why: the five-example CPU smoke ran in the full checkout, hiding a packaging gap.
Rule: extract the exact Git archive into a separate directory and execute its
five-example CPU collection/save/resume smoke before uploading it. Include the
complete committed cluster and spectral_utils trees. Never overwrite a failed
snapshot; record packaging revision and archive SHA256.
Enforced by: tests/test_external_generalization.py --cpu-smoke executed from
scratch/external_generalization_private/snapshot_956_complete; JOBS.json.

## 2026-09-24 - Verify precision and numerical fallback paths before transfer
What happened: review of the new external adapter found an extra float32 cast after
CT7 BOCPD Top10, and the legacy L-SML backend could hide a numerical failure as equal
weights. Both were caught before full external scoring or quality inspection.
Why: the bank storage precision was incorrectly generalized to the residual stream;
outer estimator exception handling could not see exceptions consumed by its backend.
Rule: replay each historical view's full dtype chain and instrument internal numerical
fallbacks; verify all source scores under the strict observer before method freeze.
Enforced by: code in external_generalization/{ct7,fusion}.py, backend failure observer,
test_external_scoring.py and verify_external_source_strict.py.

## 2026-09-24 - Preserve frozen byte identities across Windows checkouts
What happened: line-ending normalization required explicit old/new hash lineage;
ordinary autocrlf checkout could otherwise change a scientifically unchanged scorer.
Why: executable source files had not been assigned stable checkout line endings.
Rule: pin LF before a method/analysis freeze and verify staged bytes against frozen
hashes. Test discovery must also explicitly set the repository import path because
the tool working directory did not reliably select it in this Windows session.
Enforced by: .gitattributes, ANALYSIS_FREEZE.json, method identity checks and the
absolute-path unittest discovery command (19 tests passed after correcting invocation).

## 2026-09-24 - Scope external fusion claims to their matched controls
What happened: the completed comparison supports frozen L-SML on Socratic,
but Hard2 versus ordinary equal remains inconclusive and answer-local L-SML
loses to that control on both Socratic backbones. A36-answer feature perturbation
cannot establish the mechanism over6,190 records; exact-match exclusions cannot
establish absence of semantic source overlap.
Why: stronger-than-CT7 performance, positive point deltas and bounded sensitivity
checks answer different questions from incremental learned-weight superiority.
Rule: retain corrected matched-control intervals, native/fallback accounting,
null prevalence effects and FEASIBILITY labels; record the negative implementation
result alongside the positive transfer result.
Enforced by: CONTRASTS.json, RED_TEAM.md, NEGATIVE_RESULT.md and independent audits.

## 2026-09-27 — Replaying digit innovation requires its opportunity mask
What happened: a new extraction initially subtracted the historical token-clock mean at all tokens. The canonical registry emits only at provided-digit positions after token zero and masks other positions during step readout. The extraction was stopped before scores or quality were saved, corrected and restarted.
Why: the method-card shorthand omitted activity-mask semantics; the earlier conversation also simplified this detail.
Rule: verify both values and masks against the canonical registry before claiming a historical feature replay. Keep no-opportunity steps explicitly distinguishable from verified low risk.
Enforced by: code (tests/test_digit_alternative_probability.py::test_historical_innovation_mask_is_not_all_tokens; spectral_utils/digit_alternative_probability.py::digit_innovation_step_max).

## 2026-09-27 — Disable LFS smudge before creating a code-only experiment worktree
What happened: worktree creation started materializing large cached datasets. Stopping its specific LFS process caused Git to roll back the incomplete worktree; it was recreated with GIT_LFS_SKIP_SMUDGE=1. Existing raw caches remained intact.
Why: the checkout inherited default LFS smudge despite the experiment already having read-only access to source caches. A recovery restore also assumed the rolled-back directory still existed and failed harmlessly.
Rule: use GIT_LFS_SKIP_SMUDGE=1 for code-only worktrees, verify directory existence after failed checkout, and use absolute Python paths because this Windows runner can ignore workdir or deny Set-Location even when file access succeeds.
Enforced by: command (the worktree creation and experiment commands recorded in this session).

## 2026-09-27 — Do not assume frozen reasoning steps have disjoint token spans
What happened: the new digit extraction stopped at a disjoint-token-span assertion on PRMB before quality evaluation. Three existing answers have ten shared boundary tokens. Independent re-tokenization exactly reproduced every affected ID/span, and all 6,969 label rows matched corrected v3.
Why: character-disjoint steps need not map to token-disjoint spans; the original producer records these with strict=False.
Rule: preserve frozen spans and validate exact audited overlaps rather than adding an incompatible invariant. Any unregistered overlap still fails; report an exclusion sensitivity for the three known rows.
Enforced by: code (scripts/audit_digit_source_spans.py, SPAN_AUDIT.json, extraction identity checks).

## 2026-09-27 — Verify sealed arrays and bound whitespace-check output before committing
What happened: the first experiment commit omitted two compact NPZ files because of the repository-wide result-array ignore rule. The staged whitespace check also emitted a large CRLF-only warning stream for sealed JSON; the command sequence did not gate the commit on its exit status.
Why: artifact coverage was checked on disk rather than against the staged file list; Windows JSON writes preserved CRLF under results/** -text.
Rule: compare manifest members to tracked paths, explicitly include small replay arrays, use the existing cr-at-eol attribute for sealed Windows artifacts, and inspect a bounded check result before committing.
Enforced by: code/config (.gitignore exact exceptions and .gitattributes rule for digit_alternative_probability_v1); command (manifest-versus-git validation).

## 2026-09-27 — Match the executed control roster to every bank in the protocol
What happened: the numeric-family extension protocol specified equal fusion for each bank, but the runner omitted equal fusion on the duplicate-feature bank. After primary outcomes, the missing predeclared control was completed in a separate sealed subdirectory; no primary predictions or contrasts were changed.
Why: the learned-arm roster and equal-control roster were written separately and one explicitly excluded the duplication bank.
Rule: verify the Cartesian product of registered banks and requested control types before fitting; disclose late completion instead of silently rewriting a frozen run.
Enforced by: command/artifact (duplicate_equal_completion/SEAL.json and the report's explicit timing disclosure).

## 2026-09-28 — A loss was attributed to a step that never fired on that bank, and a derived rule was called size-free without the algebra
What happened: in lsml_merge_step_v1 my preliminary report put the B32 loss (-0.0239) under "L-SML with the merge step", although the step merged nothing on B32 in any fold (the loss is the binary partition plus L-SML weights). It reported the central losses as significant on 4 of 5 banks without saying that the same-length swap null explains most of them, and it did not say that on B13 L-SML's own 3-unit guard turns the "learned" weights into equal group weights. Earlier, in the clustering diagnosis and to Omri, I called the absorption ratio "size-free"; the red team derived rho = (1-r)/(1+(m-1)r) for halves of one factor, which depends on the smaller half's size.
Why: I read the contrast table by arm name instead of checking, per bank, whether the component under test was active and which estimator branch (guard or eigen-solve) produced the weights; and I described a rule's behaviour from examples instead of deriving it.
Rule: before attributing a contrast to a component, report per bank and fold whether that component changed anything (e.g. merge fired / partition changed / guard fired) and exclude inactive banks from the attribution; every loss, like every gain, carries its bank's position null; any property claimed for a new rule ("size-free", "scale-invariant") needs a one-line derivation or a test in the helper's test file.
Enforced by: code (lsml_merge_step_run.py records merged_* flags and small_m_guarded per fit; tests/test_lsml_merge_step.py::test_halves_of_one_factor_rho_formula) + prose (this entry, results/lsml_merge_step_v1/run_20260928/RED_TEAM.md).

## 2026-09-28 — A mechanism was written into the red-team claims without looking at the recorded weights, and an activity claim was repeated after it had already failed
What happened: in algorithm_decisions_v1 I stated as the mechanism of the DS-estimate group weights that they "give ~0 weight to chance groups"; the recorded weights (GROUPS.csv) show 0 of 226 zero weights under mean or hem within-group weights and no group at chance - what the weights do is up-weight the level group with ratios stretched by inflated estimates. I also claimed "the merge fired wherever the level family was split" for all learning banks, although Step 456 had shown it fails on position-adjusted partitions (here 10/14), and "the level group is the most influential on every bank", which holds for one of two influence measures.
Why: I wrote the explanation from the aggregate scores before opening the per-group records the runner had saved for exactly this purpose, and I restated an earlier claim without its known exception.
Rule: before stating a mechanism for a weighting, read its recorded per-group weights and state what they are (zeros, shares, ranking vs truth); a claim that already failed in a scope keeps that scope in every later statement; an "every bank" claim names the measure.
Enforced by: prose (this entry, results/algorithm_decisions_v1/run_20260928/RED_TEAM.md); the runner saves GROUPS.csv and ACTIVITY.jsonl for these checks.

## 2026-09-28 — A leave-one-bank-out rule selection treated 8 banks as independent although they were 4 twin families
What happened: in partition_switch_v1 the stopping rule for bank b was chosen on the other banks (and other folds, to avoid label leakage). But the 8 banks are 4 families (each bank with and without the same three digit channels), so B32's rule was trained on B35 and vice versa. The frozen success criterion was met (+0.0028, no bank loss); with the family held out the rule never switched on and the gain was 0. The runner's pre-run review also had to catch a 95-minute slowdown from indexing an npz inside a loop, a rule already in this file.
Why: I guarded against the leak I had seen before (shared answers across banks) and not against shared structure (shared channels); I did not ask what the independent units of the leave-out design are.
Rule: in any leave-out selection across feature sets, first list which units share inputs (channels, answers, families) and hold out the whole sharing group; report the leave-family-out result beside the declared one. Load every npz array once into memory before any loop.
Enforced by: prose (this entry, results/partition_switch_v1/run_20260928/RED_TEAM.md).

## 2026-09-29 — An external protocol compared methods without a step-index baseline row or a position null
What happened: in algorithm_external_v1 the frozen protocol compared the candidates with ct7 and the earlier L-SML on the official metric and within-answer AUC, but had no step-position reference and no swap null. The red team found that the step index alone beats every method on the external benchmarks (within-AUC 0.73 Socratic, 0.86 Hard2Verify) and that most of the candidate's advantage over ct7, and all of L-SML's advantage over plain averaging, is a tilt toward later steps. Part of the PRMScore gaps was also threshold transfer (different predicted-correct shares).
Why: the development protocols carried step_index and position nulls, but I did not carry them into the external protocol, and I compared thresholded metrics without checking the predicted-correct shares.
Rule: every external or new-benchmark protocol includes the step-index row, the same-length swap null for each headline contrast, and an equal-share (same fraction predicted correct) version of every thresholded comparison.
Enforced by: prose (this entry, results/algorithm_external_v1/run_20260929/RED_TEAM.md).

## 2026-09-29 — A label-free diagnostic was computed on a different row set than the method it diagnoses, and a keep decision was read as support without a floor
What happened: in position_channel_v1 the logged orientation diagnostic (Spearman of POS vs the base score) used all fit rows, which include ProcessBench steps (35% of them), while the Dawid-Skene filter and the evaluation use PRMBench only. It came out negative on 3 of 4 banks and two red-team agents repeated it as "the label-free sign contradicts the prior"; on PRMBench fit rows it is positive on 4 of 4. The protocol also read "DS keeps POS in 20/20" as support, but random mark channels pass the same pi_hat > 0.5 rule (59/60); only the flipped-channel control (dropped 20/20) shows the direction. And its principle sentence said the machinery decides "how much" position counts, while in a plain average an added channel weighs exactly 1/(p-1) of the base.
Why: I wrote the diagnostic against the existing `fit_rows` variable instead of the rows the decision uses (`pf`), and I specified the keep rule's evidence without asking what a meaningless channel would score.
Rule: compute every diagnostic on exactly the rows the diagnosed step uses, and report it per benchmark when rows mix benchmarks. Any "channel kept by the filter" statement carries a random-channel floor and a flipped (or order-permuted) control. Before claiming an estimate sets a weight, write the weight's algebra.
Enforced by: prose (this entry, results/position_channel_v1/run_20260929/RED_TEAM.md).

## 2026-09-29 — An "estimated" weight was presented as principled without a dose check, and a model built from the same channels was used to calibrate them
What happened: in position_prior_v1 the protocol said the position weight "is estimated from the model rather than set by the channel count" and treated that as the principled answer. The red team's post-hoc dose check showed the estimated weight is 2-5x below what PRMBench rewards in 40/40 folds, because the slope a that converts content into the model's log-odds is inflated 2-3.6x: the Dawid-Skene latent class is a consensus of the same channels that make up the content score, its posteriors are nearly hard, and content looks more separating than it is. The review had named this circularity as a declared limit, but the protocol had no test of its size. The 10-bin shape was also described as the mechanism; a straight line with the same weight does as well.
Why: I checked that the weight was label-free and different from 1/(p-1), not whether it was near what works; and I did not separate "shape" from "size" of the added term.
Rule: any claim that a label-free procedure sets a weight "correctly" carries a post-hoc dose curve (score + lambda * term) reported beside it, marked descriptive; a calibration that uses a latent class defined by the same inputs it calibrates must be cross-fitted (disjoint channel halves) or declared circular with its measured inflation; when a structured term (bins, shape) is added, compare it with a scalar term of the same size.
Enforced by: prose (this entry, results/position_prior_v1/run_20260929/RED_TEAM.md).

## 2026-09-29 — Weights were pooled across datasets against a standing instruction, and a failure mechanism was written from the protocol's own expectation
What happened: Omri has repeatedly said the method must be fitted per model per dataset on that dataset's own unlabeled answers. My runners (Steps 457-461) fitted the Dawid-Skene machinery on PRMBench learning folds and applied it to the ProcessBench cells, and I then reported that "no single position weight suits both benchmarks" - a problem created by the pooling. In per_dataset_fit_v1 I also explained the failure of the first-error readout by "too sharp posteriors"; the red team's slope shrinkage refuted it (the cause is the high per-step latent prevalence in the product), and "position dropped in 32/32" was stated as "no ProcessBench cost from position" although it held only for the plain average (31/32).
Why: I carried the fold design of earlier supervised-style evaluations into label-free methods without asking what the deployment data is; and I wrote mechanisms from expectations instead of testing them.
Rule: every label-free method is fitted per (model, dataset) on that dataset's own answers (transductive is legitimate without labels); never share fitted quantities across datasets or frame a result as "one weight for both". A stated failure mechanism needs the intervention that would test it (e.g. shrink the slope) before it is written. A readout that turns step posteriors into a first-error distribution gets a label-free check of its implied step-0 share in the smoke.
Enforced by: prose (this entry; memory feedback_fit_per_model_per_dataset; results/per_dataset_fit_v1/run_20260929/RED_TEAM.md).

## 2026-09-29 — An estimator name ("DS estimate") was used for two weeks without a source, and a glossary entry cited the wrong paper
What happened: Steps 450-462 reported "DS estimates" and "the DS filter" without naming the paper, equation or code path. When Omri asked which paper it implements, the answer was Dawid-Skene EM (1979) imported from `cvf_v2/em.py`, not the Jaffe-Nadler-Kluger (2015) method the advisor had pointed to, which had never been implemented. The trace also found `GLOSSARY.md` (`a1_residual`) attributing the Eq. 14 residual to the 2015 paper; it is Eq. 14 of the 2016 paper, as `fusion_utils.py` already said. A first answer in the session was also based on `main`, whose `core/lsml.py` is an unrelated early skeleton, before the live branch was identified.
Why: estimator names were carried from run to run as labels, and attributions were copied from memory instead of the extracted paper text.
Rule: every estimator named in a protocol or report carries paper + equation + code path (`module.function`) at first use; attributions are checked against `papers/extracted/` before they enter GLOSSARY or an advisor text; before answering "what does the code do", name the branch the answer is based on (the default checkout may be `main`).
Enforced by: prose (this entry; `docs/research_notes/ESTIMATOR_PROVENANCE_2026-09-29_HE.md` section 2 as the reference table).

## 2026-09-29 — `git stash` in a checkout with permanently-modified hash-locked files
What happened: two files under `spectral_utils/external_generalization/_bank11/` are committed with CRLF while `.gitattributes` says `text eol=lf`, so a Linux checkout shows them modified forever (line endings only). A stash/pop round trip was refused because of them, and `guard_git` blocked the `stash drop`; the stashed edits were restored with `git checkout stash@{0} -- <paths>`.
Why: the files are byte-identity locked (external-evaluation rule), so they must not be renormalized, and stash does not tolerate them.
Rule: do not stash in such a checkout; commit work in progress on your own branch instead. Never commit a renormalized copy of a hash-locked file; `git update-index --assume-unchanged` quiets them locally.
Enforced by: command (`.claude/hooks/guard_git.py` blocks stash drop/clear) + prose (this entry; `docs/HANDOFF_COLLECTION_2026-09-29.md` section 6).
## 2026-09-29 — A decision-rule experiment pooled the benchmarks in its estimator and misread its clean-answer panel
What happened: decision_rule_v1 (frozen protocol, numbers exact) fitted Dawid-Skene for the estimate-based threshold rules on marks pooled over ProcessBench and PRMBench fit steps, unlike stage B (PRMBench only). Several channels fire far more on ProcessBench steps, so the latent "error" class partly encoded benchmark membership; the stored truth also used OOF_STEP_SCORES labels, which mark EVERY ProcessBench step True (a placeholder: PB uses `target`). Separately, the clean-answer panel read "controls flagged 99% -> 57%" as recognizing clean answers, but the controls are the benchmark's unmodified originals: modified-but-correct answers stayed flagged like erroneous ones (93% vs 90%). And the label nulls were read as "length structure / placement", although both nulls scramble the ranking; direct controls showed a length-only count rule is worth about 0.
Why: the estimator's row set was chosen to match the global standardization instead of the stage it was meant to reproduce; the PB label convention in the OOF file is not documented next to the array; the clean-answer comparison had no modified-but-clean reference.
Rule: estimators of step-level error (DS, SML, EM) are fitted on one benchmark's steps unless pooling is the question; never compute step-level truth for ProcessBench from OOF_STEP_SCORES['labels'] (use `target`); a clean-answer panel compares modified-but-clean answers (multi_solutions, length-matched) with erroneous ones, not only the unmodified controls; a count rule is tested with direct controls (rate-matched reference, length-only counts, counts shuffled among same-length answers), not only with label permutations.
Enforced by: prose (this entry; results/decision_rule_v1/RED_TEAM.md, AMENDMENT_A1.json).

## 2026-09-30 — PRMBench construction priors and a style-confounded "fair" reference passed unnoticed into a frozen design
What happened: answer_gate_v1 (numbers exact) declared PRMBench erroneous vs multi_solutions the "fair" comparison and read 0.92-0.98 AUROC as correctness detection. The red team showed multi_solutions are a different generation process (verbose "Step N" format, twice the tokens per step) that the detectors separate from the correct controls almost as well. It also showed that PRMBench inserts about 2 errors per answer regardless of length, so the error share falls from 0.33 to 0.10 with length: a length-only control scored the best PRMScore of the stage (0.6822), and flagging the top-2 steps in every answer, with no answer-level score, gives 0.6808 (above R2, level with the raw-scale PRM). On ProcessBench a random gate with the same unflagged share already produced 76% of the F1 gain, and the fused answer detectors' edge over mean entropy was answer length.
Why: the design checked label-free facts about the features but not the benchmark's construction (errors per answer, the provenance and format of each answer class), and it had no random-gate or fixed-count baseline.
Rule: before any PRMBench decision-rule result is read, report the fixed-count baseline (top-k steps per answer) and the error-count-per-answer structure; never use multi_solutions as a clean reference without a style check; every ProcessBench gate result is shown next to a random gate with the same unflagged share; every answer-level detector AUROC is shown within length deciles.
Enforced by: prose (this entry; results/answer_gate_v1/RED_TEAM.md).
## 2026-09-29 - Parallel subagents share one session scratchpad
What happened: seven step-judge subagents ran in parallel from one session; two reported that another instance overwrote
their helper script in the shared scratchpad (one produced empty printouts before noticing). Labels were unaffected (the
checker verifies ids and order; the helper scripts only serialized hand-written labels), but only because the judges noticed.
Also: a new `.claude/agents/*.md` definition is NOT loaded mid-session (it needed a restart).
Why: every subagent of a session gets the same scratchpad path and picks generic file names.
Rule: every parallel-agent launch prompt names a private scratch subfolder (e.g. `<scratchpad>/<judge_id>_<range>/`);
write agent definitions before the session that needs them.
Enforced by: prose in the round-2 launch prompts (results/self_generated_step_labels_v1/round2).

## 2026-09-29 - An inference-time grade is not ground truth
What happened: the MATH grade saved at generation time (`spectral_utils.data_loaders.is_correct_math`) marked 81 of 366
non-truncated "wrong" Qwen3 MATH answers wrong although both independent judges found the final answer equivalent
(`\frac32` vs `3/2`, `3/2` vs gold `1.5`, `\pm 3i`, `\text{D}`, pmatrix vs tuple).
Why: ad-hoc normalization plus float/string comparison; the `label` field was never audited against an independent check.
Rule: before using a self-generated cell's `label` as ground truth, regrade with a real equivalence checker or an
independent judge; treat old self-generated MATH results as unverified until regraded (technical debt, deferred).
Enforced by: prose; docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md "Technical debt".
## 2026-09-24 - Use absolute paths for Windows tool reads
What happened: this results-summary session's initial relative-path reads resolved under C:\, including a retry with an explicit tool workdir. No project files were changed by those failed reads.
Why: the shell did not honor the expected working directory; the earlier unittest invocation lesson also applies to document reads.
Rule: pass absolute -LiteralPath arguments and git -C paths in this tool environment; verify the resolved location before any relative-path mutation.
Enforced by: corrected absolute-path command invocations in this session; no existing research code was changed.

## 2026-09-24 - Preserve fitted-model identity for calibration and verify CT7 from its canonical implementation
What happened: read-only review of the tail-L-SML discussion found that tail_lsml_banks_run.py writes evaluation and calibration scores into one shared array; later folds overwrite them, mixing fitted models in four of five calibration/evaluation pairs. The tail protocol also describes frozen CT7 as token fusion with 4/2/1 weights, whereas frozen_locator_ct7.py defines seven step views averaged with equal 1/7 weights.
Why: predictions were keyed only by answer rather than outer fitted model and role, and the baseline description was not checked against its canonical scorer.
Rule: retain calibration and evaluation predictions separately for every fitted model and verify model identity before thresholding. Replay the exact canonical reference before attributing a gain to weight learning or operation order. Existing tail within-answer rankings are not automatically invalidated by the calibration issue; PRMScore needs a corrected replay.
Enforced by: source inspection of tail_lsml_banks_run.py lines 102-116 and 139-142, and frozen_locator_ct7.py lines 4-18 and 65-67. Experimental files remain unchanged; an automated model-identity assertion is still needed in the owning lane.

## 2026-09-24 - Re-audit calibration when an experiment moves to another harness
What happened: the new ct7_token_tail_lsml_v1 run in the levers worktree has no prior eval/cal overwrite bug, but its learned OOF scores enter the fixed-reference PRMScore branch because rosters is empty. Thresholds mix fitted-model identities; inner threshold selection uses source labels; raw CT7 and answer-z candidates have different final normalization. These findings were recorded in docs/reviews/CLAUDE_TOKEN_TAIL_LSML_CORRECTION_HANDOFF_20260924_HE.md without modifying experimental artifacts.
Why: a reusable reference-score evaluator was treated as a nested learned-model evaluator, and the previous run's findings could not safely be carried over by experiment name. During document inspection, one shell read also incorrectly combined Get-Content -Raw with -Tail; it was corrected to -Tail only.
Rule: inspect the current driver-to-evaluator route, preserve model identity across calibration and evaluation, label threshold-fitting access, match normalization, and compute uncertainty for the actual primary metric. State that current CT7 replay checks saved scores rather than regenerating its full pipeline.
Enforced by: source/hash review of six frozen source/config entries and the linked correction handoff. Same-model calibration assertions and paired PRMScore intervals remain required implementation work in the owning lane.

## 2026-09-24 - Separate feature orientation, filtering and family fusion in attribution
What happened: review of named_group_fusion_v1 showed that the direct L-SML recovery is largely filtering, whereas the equal-fusion path improves most from label-chosen orientation before filtering. The recorded bank is 48 channels, reduced to 28. Residual-correlation groups and their re-standardization do not certify conditional independence.
Why: comparing only the original broad bank with the final family pipeline combines multiple interventions and changes label access.
Rule: compare original-sign, source-oriented, filtered and family controls on the same population. Describe conditional-independence as an assumption unless supported beyond pooled residual correlation; disclose all label-guided design before freezing transfer candidates.
Enforced by: the matched source-table audit and section 8 of docs/reviews/CLAUDE_TOKEN_TAIL_LSML_CORRECTION_HANDOFF_20260924_HE.md. Frozen experiments were not modified.

## 2026-09-24 - Do not convert unknown PB step labels into error truth
What happened: the independent confidence-experiment reviewer initially cast JOINED PB step labels to bool and tried to equate the first positive with the first-error target. Those labels are sentinel -2; the assertion was the audit's mistake, not a scoring defect. All6,800 PB rows were then checked using the explicit canonical target, with4,442 erroneous answers scored.
Why: a mixed-benchmark array was assumed to have one label contract throughout.
Rule: validate benchmark-specific sentinel and label conventions before casting. PRMB uses one-based error_steps; PB localization uses its explicit target. Never infer PB error truth from sentinel labels.
Enforced by: results/lsml_group_confidence_v1/audit_recompute.py and AUDIT_RECOMPUTE.json.

## 2026-09-24 - Confidence ablations and latent accuracies need narrow claims
What happened: independent math review clarified that averaging group log evidence changes both slope and intercept, and that singleton continuous contributions do not saturate. Exact duplicate invariance only covers within-family copies. Model-implied sensitivities/specificities differ materially from gold-label accuracies on the held-out PRMB folds.
Why: language about group confidence can accidentally imply pure cardinality attribution, universal robustness, or calibrated correctness.
Rule: identify the exact intervention, latent reference variable, continuous extension and duplication scope. Sum-versus-average is an evidence-temperature ablation; synthetic independent-measurement success does not establish benchmark improvement.
Enforced by: binary enumeration/duplicate tests, AUDIT_MATH_NULL.json, RED_TEAM.md and the group-confidence results report. Method and frozen outputs were not changed after quality evaluation.

## 2026-09-24 - Verification scripts also need smoke checks and bounded I/O
What happened: Windows Set-Location and pytest ancestor-directory discovery were denied; tests were instead run by absolute-path runpy with explicit sys.path, and all5 test functions passed. A synthetic fixture initially indexed a probability table with a Boolean class array; converting it to integer class indices fixed the fixture before the benchmark run. A review script repeatedly decompressed NPZ arrays inside an answer loop; only that audit was stopped, arrays were cached once, and unchanged checks completed.
Why: filesystem traversal, NumPy indexing semantics and compressed-array access were assumed to behave like their simpler alternatives. One broad file inventory also returned excessive output before being narrowed.
Rule: use absolute paths in this environment, smoke-test fixtures before scientific fitting, cache NPZ members outside per-answer loops, and constrain file inventories to the metadata being sought. Report the actual test runner rather than calling an unexecuted pytest invocation successful.
Enforced by: scripts/run_lsml_group_confidence.py, tests/test_lsml_group_confidence.py and results/lsml_group_confidence_v1/audit_math_null.py.

## 2026-09-24 - Generic binary-family method names hide different fitting recipes
What happened: the user compared our family15_binary_lsml PRMScore0.641834 with Claude's corrected F15_tailtie_lsml0.645210. Reading both implementations showed our strict empirical-q80 binary indicators with pooled standardization differ from Claude's top-ceil quota, fractional boundary ties, within-answer centering and continuous-anchor orientation. The common descriptive name was insufficient to establish estimator identity.
Why: prior communication named the representation and broad method, but omitted consequential threshold, tie and centering rules.
Rule: compare exact frozen recipes, not display names. Do not attribute the difference to calibration or tie handling alone without a matched ablation. Fractional tie marks are not hard binary classifier outputs. Distinguish the separate negative latent-tree EM confidence experiment from both spectral family variants.
Enforced by: direct comparison of spectral_utils/lsml_group_confidence.py and spectral_utils/lsml_group_confidence_experiment.py with .worktrees/ssl-pseudolabel-residual-v1/scripts/experiments/calfix_common.py; frozen scores remain unchanged.

## 2026-09-24 - Normalize filesystem paths before Git object lookup
What happened: the post-commit family-tail archive check passed a Windows backslash destination to git show HEAD:path, which failed to find the committed file. Converting only the Git object path to forward slashes fixed the check; all18 archived inputs and the portable implementation match their recorded hashes in commit df8371f3c.
Why: historical provenance records filesystem paths, while Git tree paths use forward slashes on every platform.
Rule: normalize separators at the Git object-lookup boundary; do not reinterpret a lookup error as missing committed evidence before inspecting the path syntax.
Enforced by: the successful committed-byte verification for codex/family15-tail20-transfer-v1. The handoff worktree is committed and clean.

## 2026-09-24 - Require a positive test count when checking transferred code
What happened: unittest discovery returned NO TESTS RAN for two plain pytest-style functions in the family-tail handoff. No test success was inferred; the two functions were then explicitly loaded and executed, and both passed.
Why: the selected discovery framework did not collect plain test functions.
Rule: inspect the test style and assert a positive expected test count before reporting validation.
Enforced by: the review command asserts len(tests)==2 before executing both functions; results/family_tail_handoff_review_v1/PROVENANCE.json records 2/2 actual tests.

## 2026-09-24 - Verify actual feature precision and freeze before launching workers
What happened: the initial family-tail review repeated a historical comment describing the CT7 token cache as float32. Inspecting the actual saved array showed float64; the extractor and review were corrected to preserve actual source values before full source parity and external scoring. The source-run code hash observation was recorded after workers started, so its timing limitation is explicit rather than claimed as a prelaunch attestation.
Why: prose about cache storage and the existence of a frozen method lock do not establish the exact bytes executed by a new extractor.
Rule: inspect array dtype and every rounding point; require full raw-source parity, and hash extraction code and dependencies before worker launch. If an observation is late, preserve its real timing and verify unchanged hashes/mtimes at completion. Windows command-size failures must be retried as file patches, not assumed to have written output.
Enforced by: scripts/verify_family_external_source.py, source-code observation/completion artifacts and IMPLEMENTATION_FREEZE.json; external scoring refuses a missing full source gate or changed frozen dependency.

## 2026-09-27 - Absolute paths before session-start reads (recurrence)
What happened: relative document reads resolved under C:\ despite an explicit workdir; Set-Location was also denied. Absolute-path reads with reviewed escalation succeeded.
Why: the existing Windows runner lesson was not applied before the first read.
Rule: use absolute file paths from the first tool call and keep document reads bounded to avoid output truncation.
Enforced by: corrected absolute-path calls in this Joint L-SML review; no experimental files changed.

## 2026-09-27 — Session read repeated the PowerShell working-directory pitfall
What happened: initial relative documentation reads resolved under C:\ even with workdir supplied; absolute UTF-8 reads succeeded. No scientific files changed.
Why: the shell runner's requested directory was trusted before the existing lesson was read.
Rule: start required reads with absolute paths and login=false; bound long documentation output to avoid truncation.
Enforced by: command (absolute-path Get-Content with login=false used for subsequent reads in this session).

## 2026-09-27 — Session-start relative-path read repeated the known working-directory failure
What happened: required documentation reads resolved under C:\, including a retry with workdir supplied; absolute-path reads succeeded.
Why: the existing 2026-09-24 lesson was not available until the required files were read.
Rule: bootstrap project reads with absolute paths from the supplied environment context; do not retry relative reads merely by setting workdir.
Enforced by: command (absolute LiteralPath/path arguments used for the remaining session operations).

## 2026-10-01 — Verify ignored artifacts and historical refs before declaring consolidation complete
What happened: the consolidation review found 36,143 ignored main-result files outside the October 1 upload lists, five orphan-file versions outside the planned sources' reachable history, and historical remote sources with unique files absent from all proposed merge tips. The loose-code rescue itself passed its 268-file content check.
Why: recent-branch ancestry, nonignored loose-file lists, and regenerability were treated as stronger preservation evidence than they provide.
Rule: enumerate ignored as well as ordinary loose files, audit every ref with an explicit disposition, compare complete content rather than step counts alone, and verify restore/dependency coverage before cleanup. Keep saturated-implementation verdicts distinct from direction closure.
Enforced by: code (results/consolidation_review_20261001/audit_inventory.py) and the acceptance criteria in docs/reviews/CONSOLIDATION_REVIEW_20261001.md; archive restore checks remain required.

## 2026-10-01 — Repeat of shell working-directory and output-encoding assumptions
What happened: this review initially resolved relative reads under C:\ despite the requested project workdir; a transcript extraction also stopped on the default Windows stdout encoding. Required reads were subsequently repeated with absolute paths and UTF-8 output. No project source was changed by those failed reads.
Why: the runner's requested workdir and Python's console encoding were assumed reliable despite an existing working-directory lesson.
Rule: executable audit scripts must use an explicit absolute repository root, pass it to every Git call, and configure UTF-8 output; check return codes before dependent verification commands.
Enforced by: code (results/consolidation_review_20261001/audit_inventory.py uses ROOT, explicit git -C and checked subprocess status).
## 2026-09-24 - Preserve executed tail preprocessing, not an inconsistent method description
What happened: while packaging Claude's F15_tailtie_lsml, METHOD_NOTE said to standardize pooled tail marks, but calfix_common.lsml_fit and the deployment builder feed answer-centered marks directly into the frozen backend. A portable implementation following the prose could silently create a different method. The full source port now exactly replays the executed recipe. A replay initially failed because upstream saved answer_uid as an object array; the verifier explicitly enables pickle only for this trusted local source artifact.
Why: display names and prose did not specify all normalization details, and the upstream NPZ mixed numeric arrays with object identifiers.
Rule: preserve exact executable preprocessing and verify all source folds plus deployment weights/scores/threshold. State fractional ties and the absence of extra pooled normalization. Validate source identifiers using the actual storage format rather than skipping their alignment check.
Enforced by: spectral_utils/family_tail_transfer.py, scripts/verify_family_tail_transfer.py, tests/test_family_tail_transfer.py and results/family_tail_transfer_v1/PORT_REPLAY.json (13,769answers/145,597steps).

## 2026-09-24 - Check blob bytes before fixing apparent changes in a fresh worktree
What happened: two pinned bank11 files appeared modified immediately after checkout even though their disk bytes exactly equaled HEAD. Their committed CRLF conflicted with the blanket text eol=lf attribute. Added two specific -text overrides on this new branch, preserving every byte rather than normalizing sealed dependencies. A first attribute patch had a missing-context error and made no change; the corrected patch then applied.
Why: Git clean normalization can report a difference even when worktree bytes match a non-normalized committed blob.
Rule: compare raw HEAD and disk bytes before changing a pinned dependency; use narrowly scoped attributes to preserve byte identity. Explicit LF attributes preserve the new port's recorded source hash across checkout.
Enforced by: this branch's .gitattributes and PORT_REPLAY.json; the pinned dependency blobs are unchanged.

## 2026-10-02 — Explicit UTF-8 is required when piping Hebrew source through PowerShell
What happened: the first local advisor-preparation HTML was written with question marks instead of Hebrew because PowerShell encoded the Python stdin script before Python could configure stdout. The encoding check caught this before delivery; the exact original Unicode generator was replayed and all Hebrew text and 80 links were checked.
Why: setting Python stdout to UTF-8 does not fix PowerShell's input-pipe encoding.
Rule: set PowerShell $OutputEncoding to a UTF-8 encoding before piping non-ASCII source to Python; verify the saved title and non-ASCII character count before delivering generated prose.
Enforced by: prose and the document's encoding/link validation in this session.

## 2026-10-02 - Graphify review repeated the absolute-path and bounded-output mistakes
What happened: initial relative session-start reads resolved under C:\, including a retry with workdir supplied. A subsequent combined absolute read exceeded the output limit. Absolute UTF-8 reads in bounded sections recovered the required guidance.
Why: the runner working-directory assumption was retried before applying the existing lesson, and large handoff documents were requested together.
Rule: start with absolute LiteralPath arguments, explicit UTF-8, and bounded output; never retry relative reads merely by supplying workdir. Treat truncated output as incomplete evidence.
Enforced by: command (absolute-path bounded Get-Content calls used for the rest of this review); no experiment or integration files changed.

## 2026-10-02 — Check ignored paths when preserving a complete documentation source set
What happened: the first isolated-index staging of the advisor preparation stopped because the repository ignores .txt files, including the deliberately preserved own_eval.py.txt code excerpt. The shared index and checkout branch were unchanged, and no commit was created.
Why: the evidence snapshot extension matched a broad ignore rule.
Rule: compare the staged path set against the exact intended manifest; explicitly include an ignored evidence file only after checking its content and destination. Never assume a directory add preserved every cited source.
Enforced by: code in this session's exact staged-path and committed snapshot-hash checks.


## 2026-10-02 - Source-faithful copying does not validate equivalence or preserve updated decisions
What happened: the consolidation handoff retained all 21 headings yet still contradicted approved scope decisions, while the advisor slides repeated an equivalence claim and omitted the paired own-answer option.
Why: draft content and source summaries were copied without applying later decisions or checking the boolean behind equivalent_within; its code tests a small point difference plus a confidence interval containing zero.
Rule: reconcile each next step with the latest decision and inspect the actual statistical criterion before repeating equivalence claims. Preserve frozen artifacts; correct interpretation in separately versioned documentation.
Enforced by: review evidence and completion checklist in docs/meetings/advisor_prep_2026-10-05/CONSOLIDATION_REVIEW_20261002_AFTERNOON.md.


## 2026-10-02 - Distinguish feature-bank names from complete algorithm versions
What happened: Omri pointed out that summaries switched between CT7, bank11 and family counts without explaining that experiments changed features, fusion, fitting and readout over time.
Why: historical aliases were treated like directly comparable methods, obscuring which component produced the gain and which candidate a comparator analysis actually tested.
Rule: identify each recipe by inputs, feature list, preprocessing, filtering, grouping, estimator/fusion, readout, decision and fitting contract. Compare a component against matched controls and rerun candidate-specific analyses only after selecting that candidate; do not transfer CT7 findings automatically.
Enforced by: docs/research_notes/RESEARCH_SYNTHESIS_SCOPE_20261002_HE.md recipe schema, evidence-map requirements and comparison matrix.

## 2026-10-02 — Integration instructions must track the full documentation series
What happened: the advisor-preparation merge handoff still described a single commit after two follow-up commits added reviews and research scope. Following it literally could omit the later work. The live upload also transitioned to retry during review, so a new partial manifest could be mistaken for the completed previous run.
Why: integration instructions and live operational filenames did not carry the same version boundary as the artifacts they summarized.
Rule: review the full base-to-current-tip documentation range, preserve every substantive addition and source snapshot, and name/snapshot the completed upload run before interpreting a restarted manifest.
Enforced by: prose and evidence — docs/meetings/advisor_prep_2026-10-05/MERGE_HANDOFF.md and docs/research_notes/research_synthesis_20261002/HANDOFF_EVIDENCE_20261002_EVENING.json.


## 2026-10-03 - Preserve arithmetic layout in exact zero-removal replay
What happened: the new cell-local LEVEL runner recomputed centered tie-aware marks for all columns; 40 entries in pb_gsm8k_q4/B32 differed from the historical path by at most 1.1102230246251565e-16. Values and binary marks were exact. The hard gate stopped before new arms.
Why: historical sign changes recomputed only a column slice, whose memory layout changes the floating-point reduction order relative to the new full C-order matrix.
Rule: preserve original mark columns and recompute changed-sign slices exactly as the historical implementation; never relax an exact replay gate to hide an implementation difference.
Enforced by: code (spectral_utils/monday_fusion_execution.py zero_removal; ZERO_DIAGNOSIS.json).


## 2026-10-04 ? repeated relative-path tool reads during delegated work
What happened: a delegated panel task initially read repository rules relative to the shell's actual directory and failed; absolute reads recovered, with no mutation.
Why: the tool working directory was assumed despite the earlier session lesson.
Rule: task entrypoints derive ROOT from their own resolved file path; tool reads use explicit absolute project paths.
Enforced by: code in scripts/monday_candidate_guard.py and spectral_utils/monday_workflow_guard.py for the new guardian; absolute-path tool calls remain required for interactive reads.


## 2026-10-04 ? helper entrypoint retained old module name after rename
What happened: parent review caught the new LR helper entrypoint importing monday_fusion_lr_helper after the module was renamed monday_lr_helper. The helper had not launched and no result changed.
Why: unit tests imported the module directly and did not exercise its CLI.
Rule: after renaming a helper, test the actual entrypoint with --help before launching a long run, alongside the module tests.
Enforced by: command, python -B <absolute monday_fusion_lr_helper_run.py> --help, in the round-two helper verification.


## 2026-10-04 ? Hebrew text lost through PowerShell stdin encoding
What happened: the newly appended plan amendment contained question marks after a Python here-string traversed the shell pipeline; a read-back caught it and apply_patch restored only the new text. No experiment file changed.
Why: Python UTF-8 file output cannot recover characters already replaced by the PowerShell pipe encoding.
Rule: use apply_patch for Unicode prose; for piped Python source use ASCII with explicit Unicode escapes, and verify the saved text.
Enforced by: command, apply_patch plus UTF-8 Get-Content read-back for the amendment.
