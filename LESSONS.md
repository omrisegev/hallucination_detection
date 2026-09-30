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
