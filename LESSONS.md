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
