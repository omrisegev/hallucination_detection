# Codex → Claude: research handoff, 12 September 2026

נבדק מחדש בבוקר 12.9, בערך 08:46–08:50 שעון ישראל. זהו מסמך ההעברה העדכני; רשומות מוקדמות יותר על ריצות פעילות הן היסטוריות.

**DUFS הושלם ונבדק. גם variance, capacity ו־temporal הושלמו על כל האוכלוסייה ועברו review. התוכנית כולה עדיין לא הושלמה: stability נשאר בבדיקת היתכנות, ו־depth נכשל בבדיקת היתכנות. בבדיקת התהליכים לא נמצאו תהליכי Python פעילים. לא חידשתי ניסויים בזמן הכנת ה־handoff.**

## 1. Where to work and what to read

Project root (`ROOT`): `C:/Users/omris/TAU/hallucination_detection`.

Current implementation worktree (`WT`):
`ROOT/.worktrees/rbm-literature-completion-v1`

- Branch: `codex/rbm-literature-completion-v1`.
- Base: `de237a3622c776f2cfd866b39e0b0de3ca30fe90`.
- Latest implementation/documentation commit before this handoff: `1949e7788`.
- Code-only sparse worktree; input caches are read from other worktrees. Do not copy the large caches into this checkout.
- Results (`PROGRAM`): `WT/results/rbm_literature_completion_v1/`.
- Frozen plan: `WT/docs/experiments/RBM_LITERATURE_COMPLETION_V1.md`.
- Read `CLAUDE.md`, `SUPERVISED_ORACLE_CORRECTION.md`, then **this handoff and WT/PROGRESS.md**. Root PROGRESS alone does not describe the latest RBM work.

Start with these artifacts:

| Artifact under PROGRAM | Purpose |
|---|---|
| `HANDOFF_STATUS_20260912.json` | Fresh states, smoke failures, capacity convergence counts, process observation |
| `RBM_FUSION_COMPARISON.csv` | 106 readable completed-method/control rows, with source paths, hashes, access and readout |
| `RBM_FUSION_COMPARISON_COVERAGE.json` | What the comparison includes and omits |
| `PRIOR_METHODS.csv`, `PRIOR_EXPERIMENTS.json` | 317 earlier result aliases and their source experiments; not 317 unique algorithms |
| `REQUIREMENT_LEDGER.json` | Refreshed completion inventory; distinguish full states from smoke states |
| `variance/`, `capacity/`, `temporal/` | Full reviewed results, saved states, metrics, per-cell CSV and changed predictions |
| `stability/`, `depth/` | Partial smoke artifacts only |
| `QUEUE_STATE.json` and queue logs | Why the execution queue stopped |

`COMPARISON.csv` only covers completed suites in this completion program. Use `RBM_FUSION_COMPARISON.csv` when comparing older RBM/B3/IU variants too. No top-level PROGRAM/REPORT.md is produced; each completed suite has its own REPORT.md.

Many generated results are untracked; the current comparison CSV/coverage JSON have changed since commit 1949e7788. This handoff and fresh status/docs are also local changes. **A checkout of that commit alone does not contain all final artifacts.** Do not clean untracked files, reset the worktree, or use `git add -A`. Do not upload large SQLite/cache files indiscriminately.

## 2. User intent and fixed research contract

- Fusion remains the method's core. Graphs, temporal models and feature selection should support it.
- Primary learning uses one complete answer only, without its labels or other answers. Full evaluation does not mean pooled fitting.
- Evaluate all **13,769 model-answer records**: ProcessBench 6,800; PRMBench 6,969. Within-answer AUC has 6,030 eligible answers. Canonical source grouping has 3,483 groups.
- Same identities, source folds, labels and token-to-step spans. No silent dropping of failures.
- Fixed mean-entropy `q=0.3` gate, externally calibrated. PRMScore uses `q=0.8` on the other folds for each score configuration. Do not describe this as an entirely calibration-free method.
- Current RBM readout: mean of the highest 10 token scores in each step, then original earliest argmax. Short steps use all their tokens.
- **Do not adopt `first_near_max` for RBM.** It remains a diagnostic/historical row. The positive Claude result did not transfer uniformly.
- Posterior and Logit before Top10 are different scoring configurations, even with the same model. A monotone token transform can change the ordering of step averages.
- Report PB macro over eight cells, Q4/Q8 and per-cell results; PRMB pooled and within-answer AUC plus PRMScore. Do not call pooled-AUC-only gains localization gains.
- Each mechanism's two primary bank contrasts use 10,000 paired source-group bootstrap draws, 97.5% intervals. Other contrasts are descriptive 95%. This does not correct for all prior research selection.
- Small subsets check mechanics/runtime only. All performance conclusions below use the full benchmark.
- Chat first, clear tables, short documentation and CSV/JSON. No new HTML requested. No arbitrary improvement threshold or forced winner.
- These are development findings, not untouched confirmation or a complete comparison to all published leaders.

## 3. What the matrices and models mean

Each answer supplies rows of tokens. The continuous columns are normalized within that answer. The bank uses the existing top-15 probability representation; this round did not change probability capture, token windows or moment orders.

- Bank 6: `H15, V15, m3, a, a², a³`, where `a` is selected-token surprisal.
- Bank 12 adds `m4, a⁴, m5, a⁵, m6, a⁶`.
- The first three and subsequent distribution moments are distinct from selected-token powers. Preserve the frozen extractor and column order.
- Gaussian-visible RBM; hidden units are binary. H1 has exact likelihood. H4 enumerates all 16 hidden states for exact likelihood.
- Risk orientation uses the original mean-six-feature anchor without labels.
- H4 fixed readout averages the oriented hidden posterior streams. The second-layer proposal learns a fusion above those same streams.
- DUFS selects **six feature columns out of twelve**, not six tokens. Tokens are the observations used to build its graph.

## 4. Completed DUFS experiment

Worktree: `ROOT/.worktrees/dufs-moment-selection-v1`.
Results: `results/dufs_moment_selection_v1/` within that worktree.
Frozen implementation was preserved; resumed after the earlier disk-full interruption. It completed at **04:46:51 Israel time on 12 September**.

`RUN_STATE.json`: COMPLETE, 13,769/13,769. `PIPELINE_STATE.json`: COMPLETE. `RESULT_REVIEW.json` and `STATE_REVIEW.json`: PASS. Saved-state review checked 110,152 model records; 27 real-token replay cases covered 216 model-step replays, max error about 1.40e-14. This is same-session arithmetic/code review, not external scientific replication; the DUFS optimizer was not independently reimplemented.

All eight arms below use **Posterior + Top10 + original argmax**:

| Feature bank | Model | PB macro F1 % | PRMB within AUC | PRMScore |
|---|---|---:|---:|---:|
| All 12 | Initial, before learning | 36.2946 | 0.746325 | 0.605313 |
| All 12 | Trained RBM | 36.3750 | 0.738702 | 0.629276 |
| Original 6 | Initial | 35.9662 | 0.744068 | 0.613085 |
| Original 6 | Trained RBM | 36.2017 | 0.735982 | 0.630749 |
| DUFS-selected 6 | Initial | 35.9569 | 0.744191 | 0.607197 |
| DUFS-selected 6 | Trained RBM | 36.1235 | 0.740974 | 0.626847 |
| Low-correlation-selected 6 | Initial | 35.5307 | 0.745312 | 0.600512 |
| Low-correlation-selected 6 | Trained RBM | 36.9930 | 0.740490 | 0.630604 |

Primary paired comparisons, 97.5% CI:

- DUFS trained minus all12 trained: PB **−0.2516 pp [−0.8346, +0.3144]**; within AUC **+0.002273 [+0.000990, +0.003543]**. PB successes gained/lost: 82/98.
- DUFS trained minus low-correlation trained: PB **−0.8695 pp [−1.8668, +0.1303]**; within AUC **+0.000484 [−0.001577, +0.002564]**. PB successes gained/lost: 183/241.

Interpretation: DUFS shows a modest local-ranking gain over the full trained bank, but no demonstrated advantage over the simpler six-column control and no PB gain. It does **not** justify automatic integration as the new default. Low-correlation selection has a better PB point estimate, not a verified universal win. Trained DUFS has lower within-answer AUC than its own initialization, while PB/PRMScore behave differently. Do not conceal that learning tradeoff.

DUFS selected the six distribution columns very frequently: H15 95.56%, V15 97.50%, m3 98.90%, m4 99.15%, m5 99.47%, m6 98.38%. Selected-token surprisal and its powers were selected in only 0.49–4.84% of answers. This is a property of the unsupervised selector, not proof that selected-token information is useless for the task. It motivates inspecting whether the graph objective is selecting redundant distribution structure. No new ablation was run to prove that explanation.

Per-cell values: `SUMMARY.csv`. Complete metrics, selected columns, seed agreement: `METRICS.json`. `ERROR_CASES.json` contains paired changes. The frozen finish script has an old September 11 note date; filesystem/PIPELINE UTC establish actual September 12 completion.

## 5. New literature-linked experiments: full results

### V — shared versus separate state variance: complete and reviewed

Two unnamed Gaussian latent components, initialized from the saved RBM. Means and mixture mass refitted in both arms. Shared diagonal variance versus a separate diagonal variance per state, with matched shrinkage at equality and floor 0.05. They are not known correct/error classes.

| Configuration | PB % | Within AUC | PRMScore |
|---|---:|---:|---:|
| Original RBM6, Posterior | 36.2017 | 0.735982 | 0.630749 |
| Shared variance, bank6, Posterior | 35.8081 | 0.732299 | 0.625092 |
| Separate variance, bank6, Posterior | 36.0834 | 0.737466 | 0.624875 |
| Original RBM12, Logit | 36.2712 | 0.745204 | 0.622215 |
| Shared variance, bank12, Logit | 36.1817 | 0.747256 | 0.617775 |
| Separate variance, bank12, Logit | 21.0920 | 0.698596 | 0.589463 |
| Shared variance, bank12, Posterior | 36.8106 | 0.739357 | 0.628349 |
| Separate variance, bank12, Posterior | 35.4398 | 0.737512 | 0.633079 |

Separate/shared bank12 Logit: PB −15.0897 pp, CI97.5 [−17.4746, −12.7660]. It gained 283 successes and lost 934; 925 losses were early, nine late, none due to the fixed gate. Bank6 separate/shared improves within AUC, but superiority over the original RBM6 is not established. No overall replacement was promoted.

Additional completed forensics:

- 55,076 saved model-answer fits inspected for likelihood and covariance.
- Bank12 separate variance improves mean data NLL more than shared variance (gain 4.574 versus 1.371 against original), but off-diagonal covariance error remains about 0.72. Better density fit is not better localization.
- All 1,217 changed bank12 PB cases decomposed into linear and quadratic contributions. In 922 of 934 losses the linear component favored the true step over the selected wrong step, but the quadratic component reversed this **pairwise** ordering.
- This does not mean deleting quadratic terms would recover 922 cases; another wrong step could win, and refitting changes the model.
- 26 bank6 and one bank12 separate-variance fits flagged nonconvergence; finite improving fits retained, no silent fallback.

Evidence: `variance/MODEL_MECHANISMS.csv`, `MODEL_MECHANISM_SUMMARY.csv`, `LINEAR_QUADRATIC_COEFFICIENTS.csv`, `VARIANCE_LOSS_DECOMPOSITION.{json,csv}`, `DECISION.json`, review JSONs.

### C — exact versus CD; one versus four hidden units: complete and reviewed

H1/H4 share the registered input, initialization scheme and fixed readout. Exact optimization uses L-BFGS-B, maxiter 100. CD uses actual Bernoulli/Gaussian Gibbs sampling, CD-10, 100 epochs, fixed decaying learning rate, final epoch only. No label-based checkpoint selection.

Representative retained-bank comparisons:

| Configuration | PB % | Within AUC | PRMScore |
|---|---:|---:|---:|
| Exact H1, bank6, Posterior | 36.2017 | 0.735982 | 0.630749 |
| CD H1, bank6, Posterior | 35.9925 | 0.742042 | 0.594137 |
| Exact H4, bank6, Posterior | 25.3205 | 0.710100 | 0.616757 |
| CD H4, bank6, Posterior | 34.4456 | 0.734925 | 0.580452 |
| Exact H1, bank12, Logit | 36.2712 | 0.745204 | 0.622215 |
| CD H1, bank12, Logit | 30.5433 | 0.719634 | 0.510910 |
| Exact H4, bank12, Logit | 28.6553 | 0.721693 | 0.601725 |
| CD H4, bank12, Logit | 30.5041 | 0.718182 | 0.507889 |

Exact H4/H1: PB −10.8812 pp for bank6 Posterior, CI97.5 [−13.0988, −8.7371]; −7.6159 pp for bank12 Logit, CI97.5 [−9.4943, −5.7564]. Within AUC also falls in both.

**Major interpretation limitation found in this handoff:** `FIT_HEALTH.json` flags **13,769/13,769 bank12 H4 fits and 13,768/13,769 bank6 H4 fits as not converged**. H1 counts are zero and one respectively. CD runs a fixed epoch budget and has no convergence flag; do not describe it as converged. The full scores are valid reviewed results of the registered finite-budget implementation, but they do not isolate the effect of representational capacity from optimization difficulty. Do not conclude that four hidden units are inherently worse.

H1 provenance: same C-contiguous inputs yield bit-identical parameters with current and historical optimizers. F-contiguous input reproduces the original saved parameters. A 27-answer mechanics audit localized tiny differences to numeric reductions, not a changed objective. Full H1 benchmark metrics reproduce the original rows.

All 16 new score configurations, controls and intervals are in `capacity/REPORT.md`, `METRICS.json`, `COMPARISON.csv`. Full replay checked 220,304 step vectors and 29 metric bundles. `analyze_rbm_completion_mechanisms.py --suite capacity` is implemented but its full post-completion analysis remains outstanding.

Do not describe CD as uniformly worse. H1 CD with **Posterior** improves within-answer AUC over exact H1 by +.006060 for bank6 (descriptive CI95 [.003875, .008279]) and +.005003 for bank12 ([.003121, .006841]). PB differences are inconclusive and PRMScore falls substantially. Bank12 CD-H1 Posterior is 36.2766% / .743705 / .558246. The next interpretation must preserve this metric/readout dependence.

### T — temporal fusion on actual token adjacency: complete and reviewed

Saved RBM emissions held fixed. A two-state Markov transition matrix learned within each answer, with offline forward/backward smoothing. Three arms: full token sequence across step boundaries; reset at step starts; shuffled token order with inverse mapping before the unchanged readout. No artificial cross-answer adjacency.

| Configuration | PB % | Within AUC | PRMScore |
|---|---:|---:|---:|
| RBM6 baseline, Posterior | 36.2017 | 0.735982 | 0.630749 |
| Full token sequence, bank6, Posterior | 35.9052 | 0.732364 | 0.624977 |
| Reset at steps, bank6, Posterior | 35.9380 | 0.732526 | 0.624631 |
| Shuffled control, bank6, Posterior | 36.2394 | 0.735925 | 0.630824 |
| RBM12 baseline, Logit | 36.2712 | 0.745204 | 0.622215 |
| Full token sequence, bank12, Logit | 35.8688 | 0.742263 | 0.618424 |
| Reset at steps, bank12, Logit | 35.9113 | 0.742120 | 0.617794 |
| Shuffled control, bank12, Logit | 36.1991 | 0.745076 | 0.622643 |

Full sequence minus shuffled: bank6 PB −0.3342 pp CI97.5 [−0.9826, +0.3015], within −0.003561 [−0.005372, −0.001824]; bank12 PB −0.3303 pp [−0.9615, +0.3041], within −0.002814 [−0.003966, −0.001703].

The measured residual serial dependence did not become a benchmark benefit under this Markov mechanism. No automatic temporal promotion. This does not prove that all token-boundary or temporal methods fail. Full review replayed 165,228 step vectors and 25 metric bundles, plus separate short-sequence Markov checks.

## 6. Why the queue stopped, and exactly what remains

The controller stopped at **05:49:35 Israel time**. `QUEUE_STATE.json` has `status=FAILED`, `active=[]`. An elevated Win32_Process query at this handoff found no Python processes. Old PIDs 18724/20464 (DUFS), 29404 (queue), 17352 (capacity) are historical, not live handles.

| Remaining suite | Actual state | Next required action |
|---|---|---|
| Stability: three exact starts, H1/H4, answer-NLL selection | 27 smoke answers scored, `SMOKE.json=PASS`; no separate smoke review and no full run | Review saved smoke, then full evaluation and review under the frozen contract |
| Depth: fixed exact-H4 representation → learned Gaussian H1 layer | `SMOKE.json=FAIL`; 27 smoke answers, six answers affected, 14 failed model records | Diagnose collapsed/constant hidden streams and specify a versioned handling rule before full execution |
| Capacity mechanism analysis | Full task scores reviewed; analysis driver exists | Analyze NLL, gradients, nonconvergence, score/peak changes and effective hidden representation; no new fits needed first |
| Final synthesis | Tables exist; this handoff interprets the completed suites | Add remaining results and decisions after completion; do not claim the entire request is done |

Depth failure text: `ValueError: fewer than three varying hidden views`. Code computes the four oriented H4 posteriors, calls `zscore_columns`, and requires at least three surviving columns. Some fitted H4 posteriors do not satisfy this requirement. This is a concrete representation/coverage failure, not a disk-full exception and not an unexplained scheduler crash.

Evidence: `depth/SMOKE.json`, `depth/SMOKE.sqlite`, `depth/queue_smoke_20260912_054905.log`; driver branch `elif suite=='depth'` in `scripts/run_rbm_literature_completion.py`. The queue correctly saw the failed smoke and stopped scheduling new stages; therefore the independent stability smoke review was not launched.

**Do not simply restart the whole queue:** the failed depth checkpoint will reproduce the same failure and halt it again. Do not silently drop failed answers, lower the minimum-view rule or add a fallback under the old manifest. Preserve the original smoke, diagnose its values/precision, then record an explicit implementation/protocol amendment if changing the handling. The weak/nonconverged H4 first layer is itself relevant before spending on a second layer.

## 7. Earlier work in this long conversation: what is retained

The following is historical context, not work newly run during this handoff. Exact rows and source worktrees are indexed in the two comparison ledgers.

| Research line | Outcome or lesson retained |
|---|---|
| Historical Joint/graphs and benchmark consolidation | Established the need for fixed labels, canonical groups, calibration access and full populations; original numbers from different contracts must not be merged |
| Claude's entropy gate | A large PB recovery came from replacing the old no-error gate, not from better peaks; raw entropy q=.3 remains the shared gate |
| Direct top-probability fusion | Tested raw rank information and selected-token information; vocabulary probability is not a soft hallucination decision for native DEEM |
| Varentropy contributions, IU-PCR, equal weights | Strong simple references; do not omit them or equate equal weights with learned fusion |
| Moments and selected-token powers | Explored through order six; bank6/bank12 both retained. Raw 48-column rank-power RBM was poor in its tested form; not evidence that all moments fail |
| Binary moment fusion and B3 | Existing source experiment rows indexed. Six-moment B3 completed; native DEEM/direct-probability B3 did not have a full comparable result in the audited artifact |
| RBM shrinkage / shared diagonal variance | Shrinkage improved within-answer ranking modestly but no clear general PB win; diagonal changes not a universal improvement |
| Posterior/Logit/first_near_max | Showed readout can alter rankings independently of weights. Original argmax retained for RBM; near-max is not silently reintroduced |
| Answer-local position-dependent weights | Did not improve the retained baseline. Shuffled-position control also did not support a useful position mechanism |
| Supervised step-mean diagnostic | Answered a narrow question about step means; not a matched upper bound on token Top10 RBM |
| Matched supervised versus unlabeled correction | Same frozen token-level base and external source-fold access in both correction arms. PB point gain lacked clear primary interval support; PRMScore fell |
| First-error training objective | Did not improve the matched supervised step-BCE alternative; PRMB rows were inherited, not a fresh PRMB run |
| Residual/model-assumption diagnostics | Corrected old-readout attribution and bootstrap grouping; real residual correlation is not proof that modeling it improves task decisions |

Useful reference numbers (PB % / within AUC / PRMScore):

- Entropy: 35.4444 / .730111 / .625426.
- Varentropy15: 35.9610 / .737786 / .625781; Varentropy50: 35.6755 / .742465 / .632777.
- Varentropy15 contributions + IU-PCR: 35.3498 / .746824 / .622689; equal contributions: 35.5980 / .746980 / .612812.
- B3 six moments: 35.9537 / .743983 / .613277; before learning 36.0148 / .744052 / .613257.
- RBM6 shrinkage: 35.8711 / .738630 / .629544.
- Answer-local position correction, RBM12 Logit: 35.5837 / .742490 / .622079; shuffled-position control 35.6343 / .743963 / .621543.
- Matched supervised correction: 37.2042 / .747301 / .599189; baseline/unlabeled correction 36.2712 / .745204 / .622215. Primary PB difference +.9330 pp CI97.5 [−.2162, +2.0836], within +.002096 [−.000251, +.004439]. Different training access from the answer-only primary method.
- First-error supervised objective PB 35.6451 versus step-BCE 37.2042. Do not sell this as progress.

The 106-row readable table has separate answer-local, readout-diagnostic, simple-control and other-answer-diagnostic panels. It is not a 106-way claim of comparable unlabeled candidates or a complete literature leaderboard. Historical 24-cell global detection is a separate task and must remain separate from localization.

## 8. Code, audits, cleanup and commits in the latest completion session

Implemented the V/C/S/D/T model suite in `spectral_utils/rbm_literature_completion.py` and the checkpoint/evaluation driver `scripts/run_rbm_literature_completion.py`. Supporting scripts:

- `review_rbm_literature_completion_v2.py`: saved-state replay, separate Top10/PB/AUC arithmetic and held-fold PRMScore checks.
- `run_rbm_completion_queue.py`: dependency scheduling, process-creation identity, one suite while DUFS runs then up to two suites, two workers each.
- `analyze_rbm_completion_mechanisms.py`, `diagnose_state_variance_losses.py`: saved-model and failure analysis.
- `correct_literature_diagnostic_attribution.py`: corrected bank/readout attribution and canonical-group bootstrap.
- `audit_rbm_h1_provenance.py`: historical optimizer/memory-layout audit.
- `index_rbm_research_evidence.py`, `build_rbm_literature_summary.py`, `build_rbm_fusion_comparison.py`: source-indexed comparisons and readable labels.

Core mechanics tests passed (11), stability/depth worker tests (2), reviewer-unit tests (2), queue/process-identity tests (4). Those tests do not replace the real-data smoke: depth later exposed a representation failure on saved H4 fits.

Two implementation/review defects were corrected transparently:

1. JSON tuple/list manifest equality prevented resumption. Canonical JSON equality and connection closing on rejected checkpoints were fixed before continuing.
2. The original reviewer compared PB percentages to stored fractions. A separately named reviewer v2 fixes units; original reviewer and failed review state are retained. `REVIEW_UNIT_AMENDMENT.json` records unchanged scientific checkpoint, scores and metrics hashes. No scientific score correction was hidden inside that fix.

Cleanup removed **159 SHA256-identical duplicate cache files, 19.2099 GiB of logical file bytes**: 104 in the failed literature checkout, 55 in an inactive shrinkage checkout. Original source files were hash-checked and preserved. Audits: `ROOT/scratch/cleanup_20260912/verified_duplicates.jsonl`, `shrinkage_duplicate_candidates.jsonl`, `shrinkage_removed.jsonl`. Count only actual `Action=removed` rows. Disk free at handoff: 36,608,270,336 bytes (about 34.1 GiB); free-space changes also reflect other allocation, not only this cleanup. No unique result/code or Drive data was deleted.

Commit map:

| Commit | Change |
|---|---|
| `0042d13db` | Registered full variance/capacity/CD/depth/token chronology implementation and tests |
| `a0d23c8e3` | Resume manifest equality and rejected-checkpoint closing |
| `62aa581b5` | Evidence audit and reviewer unit amendment |
| `c1006b921` | Reviewed variance and verified cleanup notes |
| `28b15d646` | Dependency queue and variance loss-margin analysis |
| `40999d714` | Variance mechanism interpretation |
| `cc4d917fe` | Clear method display names |
| `038985abb` | H1 historical numerical provenance |
| `1949e7788` | Readable consolidated RBM comparison |

## 9. Recommended continuation for Claude

1. Verify fresh process state and the source manifests before opening any writer. No original DUFS or V/C/T rerun is needed.
2. Explain DUFS as a partial ranking gain, not a breakthrough/default. Inspect selection stability and cheaper low-correlation control before proposing integration. No further DUFS fit is needed to state the present conclusion.
3. Complete the saved-state capacity analysis first: near-universal H4 nonconvergence and collapsed hidden views are measured problems. Separate optimization, representation and readout effects before claiming additional units fail.
4. Finish the already-registered stability smoke review/full run if its contract remains applicable. Its minimum-NLL restart selection is label-free but does not guarantee better hallucination ranking. Keep all starts for diagnosis.
5. Diagnose depth's six failed smoke answers. Any new constant/saturated-view handling must be named and recorded, with original failed artifacts preserved. Do not silently call the depth suite completed.
6. Integrate the outcomes, update the 106-row table and short research docs, and return the full picture before another parameter/model sweep.

Do not automatically expand moment order, graph strength, temporal models or CD schedules. Higher moments beyond six, DUFS token selection, global 24-cell transfer, LOCA, Diverging Flows and KalmanNet are distinct backlog items. Earlier requests for them are not evidence that they have been completed in this registered localization program.

Paper mapping: Shaham et al. (ICML 2016) and `ushaham/RBMpaper` motivated multi-unit/CD/layer tests. Their binary Dawid–Skene equivalence is not a Gaussian-moment theorem. CoNAL/common-noise and instance-dependent reliability motivated diagnostics and the earlier position experiments; their full algorithms were not reproduced here. Sequential/networked ensemble work motivated the tested token-chain mechanism, whose negative result is specific to that mechanism.

**Current research conclusion:** simpler H1 fusion and strong scalar/equal-weight references remain essential. More expressive density models, graph selection and temporal dependence have not produced a clear overall localization winner in these completed tests. Some improve one metric while harming another; H4 is additionally limited by the tested optimizer budget. Finish the explicit open items and diagnose that limitation before expanding the search.
