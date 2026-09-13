# Stage 1 account — completing the Codex RBM / sampling program (Claude, 2026-09-12)

Status line: **Stage 1 is COMPLETE (2026-09-13 10:13).** The RBM suites (stability, amended depth, capacity
interpretation) are complete and reviewed, and the full window-sampling run reached
`COMPLETE_REVIEWED_FULL_SAMPLING` with the consolidation supervisor's evaluation, review and Hebrew reflection
(`docs/reviews/research_consolidation_2026-09-08.html`, `results/research_consolidation_v1/LEDGER.json`; see §6). Stage 2 (cross-rank varentropy-expansion fusion) and Stage 3 (Rényi views) remain drafts
and do not start before Omri reviews this account and the sampling run reaches its reviewed end.

Framing (Omri, 2026-09-12): *We have not yet demonstrated a consistent overall advantage from
learned fusion. The contributions of representation, optimization, normalization and readout remain
partly entangled.* Learning changed performance in both directions; the tables below keep the
losses visible beside the gains and never select a winner.

Contract for every row: 13,769 answers (ProcessBench 6,800 in 8 cells; PRMBench 6,969), v3 labels,
v2 canonical source groups and folds (3,483 groups), top-10 token-mean step readout with earliest
argmax, external mean-entropy q=0.3 fold gate (calibrated without labels on other answers; not
label-free method selection), PRMScore with q=0.8 held-fold thresholds, 10,000 paired source-group
bootstrap draws (97.5% for pre-registered primaries, 95% descriptive otherwise). Development
evidence only; no untouched confirmation. Tables: `results/rbm_literature_completion_v1/STAGE1_COMPARISON_TABLE.md`
(hash-bound rows) and `STAGE1_CONTRASTS.md` (primary contrasts verbatim).

## 1. What was completed in this stage

| Item | State before | Action | State now |
|---|---|---|---|
| Full window-sampling run (`localization_full_sampling_v3`, 56 arms) | stalled 3,547/13,769 since 09-09 (disk incident) | independent checkpoint/manifest review PASS (`research_consolidation_v1/RESUME_20260912_REVIEW.json`: 357 hashes match, 3,547 records verified, 0 bad); unchanged supervisor relaunched (driver pid 14584) | COMPLETE_REVIEWED_FULL_SAMPLING (§6) |
| RBM stability suite (3 exact starts, H1/H4, min-NLL selection) | smoke PASS, no full run | program runner: smoke → smoke review → full → full review → summary, unchanged protocol | COMPLETE, review PASS (§4) |
| RBM capacity interpretation | scored/reviewed; 13,769/13,769 exact-H4 "nonconverged" unexplained | `analyze_rbm_completion_mechanisms.py --suite capacity` + new `analyze_rbm_capacity_convergence.py` (+ 27-answer maxiter probe) | `capacity/CAPACITY_INTERPRETATION.md`, `CAPACITY_CONVERGENCE.{csv,json}`, `CAPACITY_MAXITER_PROBE.json` |
| RBM depth suite | smoke FAIL (6/27 answers, 14 records) | diagnosis → amendment doc → separate driver (original untouched) → smoke PASS_WITH_DECLARED_FAILURES (94 non-failing original model records over 21 answers replay exactly, 188 = 94 × 2 readouts; 14 failures renamed) → smoke review PASS (404 vectors) → full run + review | COMPLETE, review PASS (§3) |

Nothing in DUFS, variance, capacity scoring or temporal scoring was rerun; their reviewed results
are reused as saved.

## 2. Capacity: optimization limitation, representation behaviour, negative result

- **Optimization limitation.** All exact-H4 fits stopped at the registered L-BFGS-B cap
  (`maxiter=100`): 13,768/13,769 (bank6) and 13,769/13,769 (bank12), median final gradient 0.031 /
  0.072 against gtol 1e-6. H4 still fits the density better than H1 on every answer (median gain
  0.98 / 1.78 nats per token). A feasibility probe on the 27 smoke answers (54 fits, maxiter 1000,
  ≤ 1.4 s each) converges 48/54 with a further median NLL decrease of 0.47 and moves the top-10
  peak in 7–8 of 54 fits. So the H4 result is conditional on the budget; a larger-budget run would
  be cheap but is a new registered experiment, not a conclusion.
- **Representation behaviour.** No dead units; duplicate units in 0.4% of fits; **sigmoid
  saturation** (logit varies, posterior constant to machine precision) in one unit for 9.9% /
  16.6% of fits and two units for 1.0% / 2.6% (bank6 / bank12). Two-view answers: 140 / 353.
- **Negative result at the registered budget.** Exact H4 loses to exact H1 on both benchmarks
  (bank6 posterior −10.88 pp [−13.10, −8.74]; bank12 logit −7.62 pp [−9.49, −5.76]; within-answer
  AUC also lower). Losses are heavier in the least-optimized gradient quartile (bank6: 114 gained /
  333 lost) but the best-optimized quartile still loses 2:1 (24 / 48); per-answer loss rates are
  *lower* for fits with saturated units (bank6 12.8% / 7.6% / 0.8% for 0 / 1 / 2 saturated units),
  so saturation is not the mechanism of the H4 deficit. Nothing here isolates capacity from optimization; nothing supports
  "more units are inherently worse or better".
- **Implementation failure:** none (review PASS; H1 provenance bit-identical to the historical
  optimizer).

## 3. Depth: diagnosis, amendment, measurement

Diagnosis (`depth/SMOKE_DIAGNOSIS.json`): every one of the 14 failed records is condition (a),
numerical sigmoid saturation of two of the four first-layer units (logit std 5–45, posterior std
1e-13 to 1e-43, posterior mean 0.000); no duplicate or dead units among the failures. Expected
full-population coverage of the original second layer: 98.98% (bank6) / 97.44% (bank12).

Amendment (`docs/experiments/RBM_DEPTH_AMENDMENT_20260912.md`): (1) the original variants are
measured on the full population with `COLLAPSED_HIDDEN_VIEWS` recorded as a named per-answer
failure — counted as a missed decision in full-population metrics, reported beside conditional
metrics on covered answers; this is a measurement, not a fix, and does not claim the layer works on
every answer; (2) because the diagnosis is saturation, registered logit-input variants
`layer2_logit_{exact,cd}` feed the oriented hidden logits instead of posteriors. They remove the
saturation collapses (0 failures in the smoke) but change the representation for every answer, so
they are compared beside the original, never in its place. The original driver file and the original
failed smoke artifacts are untouched; the amended run lives in `depth_amended/`.

Results — COMPLETE, review PASS (`depth_amended/RESULT_REVIEW.json`: 13,769 answers, 218,332 step
vectors replayed, 37 metric bundles). Declared failures: exactly the predicted 140 (bank6) and 353
(bank12) `COLLAPSED_HIDDEN_VIEWS` answers for the original posterior-input variants, none of any
other kind; the logit-input variants have full coverage.

| Configuration (retained readout) | PB all-8 % full population | PB all-8 % covered answers | Coverage | PRMB within (n) | PRMScore |
|---|---:|---:|---:|---:|---:|
| exact H1 first layer (reference) bank6 / bank12 | 36.20 / 36.27 | — | 1.000 | 0.7360 / 0.7452 | 0.631 / 0.622 |
| exact H4 first layer (reference) bank6 / bank12 | 25.32 / 28.66 | — | 1.000 | 0.7101 / 0.7217 | 0.617 / 0.602 |
| original: exact second layer on H4 posteriors, bank6 | 19.27 | 19.64 | 0.990 | 0.6215 (6,022) | 0.467 (conditional) |
| original: CD-10 second layer on H4 posteriors, bank6 | 28.16 | 28.76 | 0.990 | 0.7269 (6,022) | 0.599 (conditional) |
| original: exact second layer on H4 posteriors, bank12 | 22.10 | 22.96 | 0.974 | 0.5795 (5,914) | 0.487 (conditional) |
| original: CD-10 second layer on H4 posteriors, bank12 | 20.31 | 21.13 | 0.974 | 0.6568 (5,914) | 0.536 (conditional) |
| amendment: exact second layer on H4 logits, bank6 | 31.69 | 31.69 | 1.000 | 0.6895 (6,030) | 0.569 |
| amendment: CD-10 second layer on H4 logits, bank6 | 32.32 | 32.32 | 1.000 | 0.7374 (6,030) | 0.610 |
| amendment: exact second layer on H4 logits, bank12 | 32.78 | 32.78 | 1.000 | 0.7151 (6,030) | 0.596 |
| amendment: CD-10 second layer on H4 logits, bank12 | 22.97 | 22.97 | 1.000 | 0.6735 (6,030) | 0.543 |

Primaries (original exact second layer minus the exact-H4 first layer, 97.5%): bank6 posterior
**−6.05 pp [−7.42, −4.72]** (within −0.089 [−0.097, −0.081]); bank12 logit **−6.56 pp [−7.95, −5.26]**
(within −0.142 [−0.154, −0.132]). On the common covered answers the deficit is the same size
(−5.66 pp [−7.06, −4.32]; −5.54 pp [−6.97, −4.18]), so it is not a coverage artefact. The logit
input removes every collapse and raises the exact second layer by +12.4 pp / +10.7 pp over its
posterior-input counterpart (confirming saturation as the collapse mechanism), but the stacked
model still sits below the single-unit first layer on 15 of 16 endpoints (the exception: bank6 CD-10
on logits, within-AUC 0.7374 vs 0.7360), and its CD variant on bank12 is far below under the retained
logit readout (22.97) while the same fitted model read out as a posterior gives 34.68 / 0.7344 — the
readout interaction of §5 is present here too. Loss-category correction (2026-09-13, `depth_amended/LOSS_BREAKDOWN_CORRECTION_20260913.{json,csv}`): the driver's
early/late/failure counts overlapped because a failed answer (peak = −1) also satisfied peak < target; with
mutually exclusive categories the bank12 exact primary loses 249 = 27 failures + 0 gate + 209 early + 13 late
(the saved table said 236 early), bank6 229 = 5 + 0 + 218 + 6; 16 of 40 comparisons were affected, no point
score changed (all 37 PB macros re-derived from saved predictions match), no other suite affected (full coverage).
Classification: the collapse was a representation property (fixed by the amendment
for coverage); the stacked second layer over the maxiter-100 H4 layer is a **negative result at
the registered budget**, entangled with the H4 optimization limitation of §2. No promotion.

## 4. Stability (three exact starts, lowest answer NLL) — COMPLETE, review PASS

Full 13,769; saved-state replay and separate metric arithmetic PASS
(`stability/RESULT_REVIEW.json`); all 13 reference rows reproduce to 1e-12; 0 failures.

| Configuration | PB all-8 % | PRMB within | PRMScore |
|---|---:|---:|---:|
| exact H1, bank6, posterior (capacity start) | 36.2017 | 0.735982 | 0.630749 |
| best-of-3 exact H1, bank6, posterior | 36.2017 | 0.735954 | 0.630789 |
| exact H4, bank6, posterior (capacity start) | 25.3205 | 0.710100 | 0.616757 |
| best-of-3 exact H4, bank6, posterior | 24.7061 | 0.708662 | 0.618526 |
| exact H1, bank12, logit (capacity start) | 36.2712 | 0.745204 | 0.622215 |
| best-of-3 exact H1, bank12, logit | 36.2325 | 0.745024 | 0.622175 |
| exact H4, bank12, logit (capacity start) | 28.6553 | 0.721693 | 0.601725 |
| best-of-3 exact H4, bank12, logit | 27.3262 | 0.721817 | 0.612483 |

Primaries (best-of-3 H4 minus the capacity single start, 97.5%): bank6 posterior **−0.61 pp
[−1.32, +0.08]**, within −0.0014 [−0.0027, −0.0002]; bank12 logit **−1.33 pp [−2.17, −0.51]**,
within +0.0001 [−0.0016, +0.0019]. Lost successes move early (bank12: 114 early / 19 late of 133).

What the saved starts show:
- **H1 is start-invariant.** All three exact starts converge (99.99% / 100%) to the same NLL
  (spread 0 at the median, ≈ 0.01 at p95) and the same top-10 peak in 13,768 / 13,705 answers; the
  chosen start is a three-way tie. Multiple initialization is not a lever for the single-unit model.
- **H4 restarts do not converge at the registered budget** (0.004% / 0.0% of restarts), the three
  starts end at different NLL (median spread 0.056 / 0.23 nats per token; the best start improves
  on the capacity start by 0.009 / 0.070 at the median) and disagree on the peak in 18% / 33% of
  answers. **Selecting the lowest-NLL start lowers ProcessBench** while leaving within-answer AUC
  flat: under this budget, better density fit is not selecting task-useful H4 solutions.
- Classification: optimization limitation (H4 budget) plus a negative result for min-NLL start
  selection as a label-free selector; no implementation failure (review PASS).

## 5. What worked, what did not, what is unresolved (completed RBM program, all suites)

Rows are the completed, reviewed configurations; classification follows Omri's three categories.
Full table with coverage and sources: `STAGE1_COMPARISON_TABLE.md`.

| Finding | Evidence | Class |
|---|---|---|
| The single-unit Gaussian RBM on the moment bank (RBM6 36.20 / 0.7360 / 0.631; RBM12 logit 36.27 / 0.7452 / 0.622) has higher PB points than the raw varentropy references (35.96 / 35.68); on PRMScore RBM6 posterior (0.631) is above varentropy15 (0.626) but below varentropy50 (0.633), and RBM12 logit (0.622) is below both; both have lower within-AUC than the equal-weight contributions (0.7470); every primary interval against a matched reference includes zero (correction 2026-09-13: the earlier wording claimed a PRMScore advantage for both) | moment-rbm, higher-moment, logit-readout suites | no winner |
| Learning inside the RBM family changed results in both directions: shared-variance bank12 36.81; low-correlation-6 RBM 36.99 (not a registered primary); separate-variance bank12 21.09 (−15.09 pp [−17.47, −12.77]); 48-column rank-power RBM 19.44 (−16.94 pp below its own initialization); exact H4 25.3 / 28.7 | variance, DUFS, rbm-m3-powers, capacity | mixed; the losses are real learned-model failures, the gains are point estimates |
| Posterior versus logit readout of the same weights changes PB by up to 1.5 pp and within-AUC by 0.01 for the trained RBM6/RBM12; for other fitted models the readout is decisive: separate-variance bank12 reads 21.09 (logit) vs 35.44 (posterior); CD-10 H1 bank6 27.78 / bank12 30.54 (logit) vs 35.99 / 36.28 (posterior) (correction 2026-09-13: 27.78 was previously attributed to bank12); depth CD bank12 22.97 (logit) vs 34.68 (posterior). Several primaries are therefore model × readout results, not model results alone | rbm-logit-readout-v1; variance, capacity, depth_amended METRICS | readout confound inside the primaries; disclosed, not resolved |
| DUFS column selection versus a greedy low-correlation control: PB −0.87 pp [−1.87, +0.13], within +0.0005 [−0.0016, +0.0026] | dufs-moment-selection-v1 | no winner; the unsupervised selector is not better than the trivial filter |
| Within-answer token order (two-state Markov on fixed emissions): actual order loses within-AUC to the shuffled control in both banks (−0.0028 [−0.0040, −0.0017]; −0.0036 [−0.0054, −0.0018]); PB intervals include zero | temporal suite | negative result for that mechanism |
| Position-conditioned weights: −0.69 pp [−1.25, −0.13], within −0.0027 | rbm-position-fusion-v1 | negative result |
| Supervised step-BCE correction (labels, other answers) 37.20 / 0.7473 / 0.599: +0.93 pp [−0.22, +2.08] over the unlabeled update; PRMScore falls | rbm-supervision-matched-v1 | diagnostic; not an answer-only method, not a ceiling |
| Learned rows are not consistently above their own untrained initializations on within-answer AUC (RBM12 0.7463 initial vs 0.7387 / 0.7452 trained; RBM6 0.7441 vs 0.7360; DUFS-6 0.7442 vs 0.7410; low-corr-6 0.7453 vs 0.7405); the highest within-AUC among answer-local rows is the fixed equal-weight varentropy contributions (0.7470) | RBM_FUSION_COMPARISON.csv | learned fusion has not shown a consistent overall advantage |
| Exact H4 at maxiter 100 | capacity, §2 | optimization limitation + negative result at that budget |
| Depth second layer on saturated posteriors | §3 | representation property of the first layer; measured with declared failures; logit amendment registered |
| Full window-sampling run | §6 | open obligation, in progress |

Unresolved after Stage 1: where the small PB point gains of the learned rows come from
(representation vs normalization vs readout is not separated by any completed contrast); whether a
converged H4 changes the capacity conclusion (feasible, not run); the late bias in exact-step misses
and the strong step-length prior (Claude Steps 354–355) that every row in this table shares; an
untouched confirmation cohort for any candidate.

## 6. Full window-sampling run — COMPLETE, review PASS (2026-09-13)

Resumed 2026-09-12 through the unchanged supervisor; the driver died once (Windows file-replace race on its
status file, `research_consolidation_v1/SAMPLING_INCIDENT_20260912.json`, checkpoints verified intact) and was
relaunched under an execution-only loop; scoring finished 2026-09-13 10:13, `evaluation/REVIEW.json` PASS
(13,769 records, 97 arms, 1,000-draw paired source-group intervals on 3,483 groups). This run keeps its ORIGINAL
frozen per-answer GMM/BIC gate (pre-declared; not the shared entropy-q0.3 gate), so its ProcessBench scale is the
old one: answer-only arms 16.4–21.3, pooled historical arms 34.0–34.6.

Observation-selection replication (seven selectors × seven fusion cores, 56 arms; IU core vs the full grid, 95%):

| Selector (IU core) minus full grid | PB all-8 pp | within-answer AUC | pooled AUC |
|---|---:|---:|---:|
| uniform | +0.01 [−0.79, +0.83] | −0.0012 [−0.0027, +0.0002] | −0.0011 [−0.0025, +0.0005] |
| risk-top (highest mean entropy) | −0.06 [−0.84, +0.67] | −0.0015 [−0.0028, −0.0001] | **+0.0301 [+0.0269, +0.0331]** |
| entropy tails | −0.26 [−0.86, +0.39] | +0.0002 [−0.0008, +0.0012] | **+0.0262 [+0.0234, +0.0288]** |
| entropy quantiles | +0.24 [−0.58, +1.07] | −0.0009 [−0.0024, +0.0005] | +0.0023 [+0.0009, +0.0035] |
| DUFS transposed | −0.04 [−0.95, +0.88] | −0.0016 [−0.0033, +0.0000] | +0.0034 [+0.0008, +0.0059] |
| DUFS permuted (control) | −0.10 [−0.89, +0.67] | +0.0003 [−0.0011, +0.0018] | −0.0015 [−0.0032, +0.0002] |
| window diffusion | +0.15 [−0.57, +0.83] | −0.0012 [−0.0025, +0.0002] | +0.0035 [+0.0016, +0.0053] |

Reading: no selector changes ProcessBench exact localization or within-answer ranking on the full population
(risk-top is marginally negative on within-AUC); the pooled-AUC gains of risk-top and entropy-tails are the
answer-level scale effect identified in the 110-answer pilot (Step 317) and do not translate into local ranking or
decisions. The pilot's pooled jump replicates as a pooled-only effect. Graph cores (joint0 / graph010 / graph_perm)
sit within 0.7 pp of each other with the permuted-graph control at the top of the PB points. Full arm table:
`results/localization_full_sampling_v3/evaluation/METRICS.csv`; intervals `INTERVALS.json`.

## 7. The letter's four ideas against the record (coverage, not closure)

| Idea | Tested as | Outcome (full 13,769) |
|---|---|---|
| Per-rank p, log p, powers | direct-probability v1/v2 (rank risks), surprisal powers d1–d3 (16/32/48 columns), 48-column RBM | all below token entropy; RBM on 48 columns collapses below its initialization |
| Rényi / Tsallis alpha grid | only α=2 exists as one of 29 streams; moment grid m3–m6 is the closest cousin | not tested (Stage 3 draft) |
| Gating test p_i (log p_i)² + H² | varentropy contributions k15 (equal / IU) | within +0.009 [+0.005, +0.013]; PB −0.6 [−1.7, +0.4] (IU vs raw) |
| Cross-rank products p_i p_j log p_i log p_j | never as explicit columns; only quadratic evidence is the separate-variance mixture (quadratic term reversed 922 of 934 lost cases) | not tested (Stage 2 draft; primary contrast B2_sel vs B2d_sel) |
| Cross-position products / autocorrelation | lag8 concatenation (−2.1 pp), chain-LIU (nil), within-answer Markov (order worse than shuffle), BOCPD (within −0.014 to −0.066) | consistently negative for the tested mechanisms |
| Derivatives along position | delta bank (within +0.004 [+0.002, +0.006]; PB +0.3 n.s.), C7/C8 onset/innovation (null), rise-vs-history readout (over-corrects early) | the only mildly positive temporal result; still below token entropy on PB |

## 7b. Parallel track status (Omri's revised schedule, 2026-09-12 evening)

- **Cross-rank varentropy-expansion fusion v1** (Stage 2): protocol frozen
  (`.worktrees/varentropy-expansion-fusion-v1/docs/experiments/VARENTROPY_EXPANSION_FUSION_V1.md`),
  14 unit tests pass, identity discrepancy 8.5e-12, Step-339 historical bank replays exactly, smoke
  27/27 PASS with independent replay (621 arm checks), pre-run review READY WITH CAVEATS (fixes
  applied, re-smoke PASS). Full run launched with 3 workers (Joint arms dominate; ≈14 h estimate).
  Disclosures required by the reviewer: LW shrinkage alpha clips to 1 on nearly every answer
  (shrink ≡ IU on the joint target); Joint model covariance condition up to 1e19, absorbed by the
  analytic ridge; non-converged Joint fits scored and flagged; identity-sign equal weights are
  anchor-flipped on the pair banks; n<p fits on the 138-column banks for short answers.
- **Rényi-view fusion v1** (Stage 3): implemented and tested (14 tests), DRAFT protocol only; smoke
  15/27 (stopped to free memory). Redundancy already visible: H2, H4 and H∞ are one view
  (|r| ≥ 0.99, ≈ −log p₁), only H0.5 carries distinct information, and IU-PCR sits at its residual
  ceiling on the 5-view bank. Final design waits for the Stage-2 review.

Independent figure review: `results/rbm_literature_completion_v1/STAGE1_REVIEW_FIGURES/` (7 PNG figures,
`REVIEW.md`, `CHECKS.json`: 260 recomputed numbers matched; the 4 wording mismatches it found are corrected above).

## 8. Provenance and housekeeping

- New code (worktree `codex/rbm-literature-completion-v1`): `scripts/analyze_rbm_capacity_convergence.py`,
  `scripts/run_rbm_depth_amended.py`, `scripts/review_rbm_depth_amended.py`,
  `scripts/build_stage1_account_tables.py`; summary builders extended for the amended suite.
- Original drivers, checkpoints and the failed depth smoke are unchanged. Root HISTORY.md and
  PROGRESS.md remain fragmented across worktrees (root ends at Step 335); this account and the
  PROGRESS blocks point to the worktree files rather than merging the logs.
