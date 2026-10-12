# Localization: benchmark continuity and the next experiment

**Current mandate, September 7:** Omri has authorized the expanded research
program in [LOCALIZATION_RESEARCH_MANDATE_20260907.md](LOCALIZATION_RESEARCH_MANDATE_20260907.md).
Joint/graph representation and hyperparameters remain active, alongside IU,
chronological HMM/BOCPD and token/window sampling. Preserve the history and
benchmark dependencies below. The old stop-Joint and IU-only next-stage
recommendations are superseded. A different cohort of previously evaluated
data is development data, not untouched confirmation.

**Execution preference, clarified September 6:** this document is a backlog.
Omri wants one short stage, a report of its findings, and a new decision before
the next stage. Do not launch this entire program as one experiment chain.
Short cycles 1–3 are complete. The direct graph test requested by Omri finds no
attributable benefit from the answer-only Joint-LIU lambda-0.1 graph: it reaches
0.69188 versus 0.69148 for lambda zero, 0.69578 for a node-permuted graph and
0.72072 for IU on the common strict answers. The current next stage is a fresh,
disjoint 30-answer confirmation of IU versus equal under the unchanged width-32
span-max contract. See
`../reviews/localization_experiment_review_2026-09-07.md`.

Date: September 6, 2026. Status: continuation plan, not a frozen experiment
contract. No new fit, evaluation or cluster job was launched by this planning
step. This consolidates Omri's existing choices and the work completed while
waiting for Claude. The standing comparison preference is in `CLAUDE.md`.

## 1. Where the work stands

At **22:36 Israel time**, Claude's work was not confirmed complete. His latest
message, at 22:16, said that evaluation had reached **Module B**, the last stage.
The live `evaluation/` directory still contained only `headline.json` and
`inner_selection.json`; no Module B output or final report was present.
This is a chat/artifact observation, not a fresh verification of process health.

**Update, 22:50:** `moduleb.json` was written at 22:43:38, after that earlier
check. Module-B primary evaluation is now available. Claude's 22:50 message
reports a 37-arm table and additional contrast bootstraps still in progress;
the complete report/audit is not yet closed. The two-axis results and remaining
coverage gaps are recorded in Section 3A below.

The main checkout's older August handoff is not the current experiment state.
Use these separate sources:

| Source | Purpose |
|---|---|
| `C:/Users/omris/TAU/hd_jlsml_v2_wt`, branch `claude/joint-lsml-optimization-v2` | Claude's current v2 code, amendments, frozen scores and evaluation |
| `C:/Users/omris/TAU/hd_per_answer_wt`, branch `codex/per-answer-localization-v1` | Our window extractor, feasibility evidence and prepared continuation |
| This main checkout | Publication/visual reviews and this cross-worktree handoff |

The live result root is `results/joint_lsml_optimization_v2/` in Claude's
worktree. The latest reviewed scientific/documentation revision was `d15a3267`.
The separate window branch's latest recorded commit was `be34742`. Branch
names are discovery pointers; capture exact source and artifact hashes before
the next experiment. Preserve later uncommitted changes, including Claude's
tuple-key JSON serialization repair to the evaluator.

### Preliminary results that affect the next decision

| Row | ProcessBench macro-F1 | PRMBench step AUROC |
|---|---:|---:|
| Nested-selected L-SML/Joint family | 0.343705 | 0.672394 |
| Nested-selected IU family | 0.349282 | 0.666523 |
| Fixed `internal_joint` reference | 0.128903 | 0.610973 |
| Fixed `internal_cont` reference | 0.134278 | 0.588512 |

These values come from `evaluation/headline.json`, written at 21:10. Claude
reported that the registered ProcessBench activation-floor guard classifies
`internal_joint` as **CATASTROPHE on four cells** and `internal_cont` on **two**.
Verify the guard against its exact registered definition in the final audit.
A failed guard is a result; it is not, by itself, proof of an implementation bug.

On PRMBench the inner selector chose `internal_joint_liu010` in all five folds.
However, its shuffled-graph control scores **0.673177**, slightly above the
selected method. The observed improvement over IU therefore does **not yet
establish that meaningful graph structure caused the gain**. On ProcessBench,
the selected L-SML family does not beat selected IU. Do not announce one
universal winner or treat nested selection as a fixed label-free method.

## 2. What is already done, and what is still open

| Subject | Completed evidence | Remaining work |
|---|---|---|
| Project/advisor history | Actual August 28 email and later thread reconciled with subsequent Joint development; HTML project review and Joint visual guide created | Update the claim table after final v2 interpretation |
| v2 audit repairs | Relative-path integrity manifests, evaluator preflight and per-answer centering checks prepared and applied; 45 centering artifacts, 22 audit checks and 876 recorded paths reported/inspected; R3 has a separate freeze record | Verify final source closure, amendment bindings, exclusions and evaluator outputs; an honestly late integrity record cannot become a launch-time freeze retroactively |
| Window extraction | Recompute original feature definitions per window; explicit short/constant-feature handling, fit/scoring grids and official-step mapping utilities | Integrate the canonical Joint recipes and lock the final primary readout |
| Window feasibility | AIRCC job 247840: all 400 Qwen3-4B/GSM8K answers, no extraction errors; 10.815 seconds of computation on eight CPUs | Repeat feature/stability checks on other cells; fitting speed and localization quality remain unmeasured |
| Numerical verification | Corrected centered-rank diagnostic; local/cluster agreement; focused test suite reached 29 passing tests after repair integration | Meaningful tests for the new single-answer fitting and fallback contracts |
| Disk space | About 10.94 GiB of disposable caches removed earlier; scientific data preserved | Inventory obsolete code and redundant artifacts separately, with provenance checks before removal |
| Stable benchmark | Existing frozen-24, fair-comparison and reconstruction infrastructure already exists | Reconcile localization contracts and add the continuity/comparator registry; do not build another unrelated evaluator |

The earlier window plan contains historical snapshots saying repairs were
not yet applied. The later repair status above supersedes those statements.
No Joint window-fusion accuracy experiment has run.

## 3. First establish a benchmark that survives the next experiment

**A new experiment should add rows, not redefine the yardstick.**

1. Inventory the previous localization releases, exact populations, scorers,
   features, folds, fitting scope, reducers, thresholds and budgets. In
   particular, v2 uses nested fitting/selection, whereas the earlier
   fair-comparison package cross-fitted calibration around frozen fits. A
   matching dataset name alone does not make those procedures identical.
2. Recover the actual code that produced each important historical row:
   IU-PCR, LIU-PCR/DUFS-LIU, L-SML, dedicated localizers and other competitive
   project methods. Names are insufficient: the old LIU projected solve and
   the new Joint model-covariance inverse are different implementations.
3. Create one machine-readable release manifest with exact source-question
   IDs and label version; fixed outer/inner folds and a benchmark namespace
   independent of experiment names; model/tokenizer/raw-data hashes; access
   level and fitting scope; feature definitions and orientation; official-step
   alignment; no-error calibration; primary metrics/aggregation; uncertainty
   procedure; selection budget; source and environment versions.
4. Establish two linked comparisons. **Fusion comparison:** same feature
   matrix, change the fuser. **End-to-end localization comparison:** same raw
   answers, labels, held-out IDs and evaluation, allow token or window feature
   representations. A fixed response head/readout in the first window ablation
   isolates the local representation change.
5. Run a bridge using common historical anchors under both contracts where
   a valid old implementation can be recovered. Show changes attributable to
   protocol separately from changes attributable to method. Preserve old
   releases and explicitly supersede invalid results. Do not force an old bug
   into the new benchmark merely to reproduce a number.
6. Make future reports verify benchmark identity automatically. Every result
   needs coverage/failure status and paired contrasts against mandatory
   references. Rank metrics remain fold-wise before aggregation; bootstrap
   uncertainty respects source-question groups. Define any shared eligible
   population before ranking, and report exclusions from the full population.

Reuse the contracts and tools under
`configs/reconstruction_benchmark_v1/` and
`spectral_utils/fair_comparisons/`, and read
`docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md` and
`docs/experiments/FROZEN_24_CELL_BENCHMARK.md`. Their completed releases remain
immutable; response detection, first-error localization, every-step ranking,
causal-prefix prediction and stopping remain different targets.

### Comparator coverage is broader than our graph variants

Maintain a registry of **all relevant historical and current leading
candidates**, plus strong simple controls. Start with the existing project
registry and `papers/index.md`, then verify relevant leading methods against
official papers/code. This planning step has not completed that fresh survey.

Each entry must identify the exact method/version and scorer, evidence for
relevance, target, required observables, training/calibration labels, inference
budget, implementation fidelity and status (`READY`, `NEEDS_ADAPTER`,
`BLOCKED_ASSET`, or a reasoned exclusion). The initial bridge must include
canonical IU-PCR, historical LIU/L-SML and the dedicated localization
incumbent, alongside entropy/simple averaging and the new Joint references.
Consider other historical reconstruction contenders according to their actual
localization evidence, not only their response-level ranking.

Include relevant PRMs/critics or other methods with additional access as
explicit comparison panels with their extra requirements and costs visible.
They cannot silently disappear, and a paper's reported number cannot stand
in for a matched local reproduction. A new adaptation is labelled as such.

## 3A. Both fusion axes: what was actually tested

R1 Section 3 registers a 3-by-3 grid. Feature fusion combines 23 token streams
into token risks. Trajectory fusion then combines the ten largest token risks
inside each official step, sorted by score rather than chronological position.
The feature substrates are deployed IU, `internal_cont` and `internal_joint`;
the trajectory fusers are SML, IU and Joint. Both stages use multiple training
answers. The word Joint describes the shared/group covariance fit; it does not
mean both pipeline axes are optimized jointly.

Measured PRMBench step AUROC from `evaluation/moduleb.json`:

| Feature fusion | Trajectory SML | Trajectory IU | Trajectory Joint |
|---|---:|---:|---|
| Deployed IU | 0.663423 | 0.654497 | Blocked, all five folds |
| `internal_cont` | 0.602877 | 0.600225 | Blocked, all five folds |
| `internal_joint` | 0.642823 | 0.649164 | Blocked, all five folds |

The 15 Joint-trajectory metadata entries all report
`BLOCKED_NO_ADMISSIBLE_PARTITION`. This is a structural failure to produce an
admissible model, not a measured losing AUROC. The grid selector chose IU plus
trajectory-SML in every fold, but that row lost to the **same-substrate top-10
mean**: 0.663423 versus 0.666855; paired bootstrap delta -0.003429 with 95% CI
[-0.004285, -0.002604], registered decision `HARM`.

The separate B2a readout is a promising development candidate: deployed IU
plus 0.5*maximum + 0.5*top-10 mean scores 0.672838 versus its **span-max** B0
at 0.666523. Alpha was selected using other training answers' labels in all
five folds. It is not a strict single-answer, label-free result. B3, supervised
logistic regression over order statistics, scores 0.672576; positional-bin
B2b scores 0.656506. Keep the grid's top-10 B0 and the original Module-B
span-max B0 distinct in every report: they are different controls.

The tested grid does **not** cross the new gate/LIU/diagonal feature variants
with trajectory fusers. In particular, the 0.672394 `Joint-LIU 0.1` headline
uses the existing PRMB span-max readout; it is not evidence for Joint-LIU plus
B2a. The protocol's B4 composition row and the Module-B ProcessBench secondary
results are not present in the inspected output. Verify/complete this coverage
before calling the full two-axis study finished. There is presently no
established winning learned combination across both localization targets.

For our continuation, add a **bounded feature-fuser by trajectory-rule
comparison** explicitly. Preserve the registered grid and complete its missing
reporting first; new selected-variant combinations get their own development
registration. In the single-answer experiment, both fitted stages must obey
the single-answer boundary. Specify the trajectory units, observations and
readout: chronological windows, position bins and sorted top-k risks are
different representations. Keep a sequence of local scores and official-step
locations; collapsing the whole trajectory to one scalar is insufficient for
localization. Selecting the strongest axis separately does not establish that
their composition will win.

Implementation evidence in Claude's worktree:
`second_pass_amendments.py:106,142,183` under
`scripts/joint_lsml_optimization_v2/`, `evaluate_v2.py:569,624`,
`spectral_utils/trajectory_reducer.py:48`, and the per-fold
`moduleb_grid_meta.json` files.

## 4. The agreed Joint window study

**Omri's primary objective is learning from the current answer alone.** The
matrix is **N windows from that answer by P feature definitions**. Make this
the leading experimental arm wherever its features and fitting are stable.
Pooling windows from multiple answers is a secondary comparison and disclosed
fallback, as Omri allowed; it must not silently replace the primary objective.
Use identical window features when comparing the two fitting scopes.

For the strict arm, compute scaling, groups, gates, covariance, feature-fusion
weights and any learned trajectory reducer within that answer. Width/lambda
selection and stability rules must be fixed engineering choices or use only
that answer's unlabeled observations. Any borrowed groups, priors, globally
tuned parameters or correctness thresholds define a separately named hybrid
or calibrated arm. Shared algorithm definitions are allowed; external fitted
quantities are not silently allowed. Check the fit provenance in both axes.
Keep unrelated answers/labels out of the answer-only fitting API and verify
that changing them cannot change its localizer output.

A long trace makes adequate window counts more plausible. It does not ensure
that windows vary enough, are sufficiently independent, or distinguish error
from correct reasoning. Test those questions rather than declaring a fixed
length sufficient. The strict no-error decision remains a required design
item: an externally calibrated threshold would not satisfy this strict arm.

### Candidate family already chosen

| Candidate | Role in the window study |
|---|---|
| `internal_joint` | Joint hierarchical reference; diagnose its current failure before transferring it |
| `internal_cont` | Staged continuous L-SML reference |
| `internal_joint_gate050`, `internal_joint_gate100` | Soft-gating variants; test whether feature reliability helps |
| `internal_joint_liu010`, `internal_joint_liu050` | Graph-regularized Joint model-covariance inverse |
| `internal_joint_diag010`, `internal_joint_diag050` | Diagonal-regularized inverse, controlling for the source of regularization benefit |
| Joint model inverse at lambda zero | Required reference to isolate regularization from the change of readout |
| Canonical IU, historical LIU and simple controls | Matched benchmark anchors; exact adapters are checked before use |

Keep meaningful shuffled-graph and diagonal controls. The diagonal penalty
does not use graph edges directly, but its DUFS-derived weights still have
graph dependence upstream. Provenance variants and other current v2 rows
remain in the result registry; the table above is the user-selected window
development family, not a claim that no other contender matters.

### Resolve width, feature validity and fitting together

- Start feasibility at widths **32, 48, 64, 96, 128** tokens. Width 32 is the
  current STFT computability floor, not an accuracy optimum. On GSM8K the
  counts with at least eight full, nonoverlapping fit windows were respectively
  **220, 66, 13, 2, 2 out of 400**. Eight is only an exploratory floor.
- At width 32, median N is 8, active P is 29 and centered rank is 7. Keep
  nominal feature identities but explicitly handle constant/unavailable
  columns: `trace_length` is constant at fixed width and `min_spilled` was
  constant in 166/400 answers. Do not silently substitute placeholders.
- Evaluate spectral-estimate reliability, rank/conditioning, score stability
  under boundary perturbation and held blocks, and fit/fallback coverage.
  Overlapping scoring windows improve coverage but do not provide independent
  fitting observations. Do not require N>P mechanically or assume N<P is safe.
- Adapt the fitting mechanics deliberately. Existing leave-one-answer-out
  grouping needs multiple answer owners; the strict one-answer arm needs a
  registered within-answer blocked stability rule. Shared training components
  belong only to the separate hybrid comparison. A k=7
  nearest-neighbor graph on eight rows is complete, so copying token settings
  can remove the very locality being tested.
- In the strict arm, freeze engineering rules or select width/fallback from
  the current answer's label-free diagnostics. Selection learned across
  development answers is a separate hybrid arm; accuracy-based selection
  belongs inside its training/inner folds with a stated budget. Report the
  resulting tradeoff, not an unsupported universal optimum.
- For insufficient or unstable one-answer fits, compare the disclosed pooled
  alternative. Extremely short traces can still lack computable features;
  report that status and coverage rather than silently predicting all-correct.

### Preserve localization and the no-error decision

Lock window-to-token and token-to-official-step handling, final-window rules,
and scoring stride. The existing mean-to-step utility is not automatically
the primary benchmark reducer. First retain the incumbent response head,
calibration and primary reducer to isolate the window change; test a readout
change as a separately named ablation.

An answer-relative risk curve is not enough to decide that an entire answer
is correct. Keep the absolute no-error decision explicit. Develop a fixed or
within-answer label-free rule for the strict arm; any cross-answer calibration
is confined to the separate calibrated comparison's allowed training folds.
Check brief error peaks and step-boundary offsets.
Report ProcessBench first-error/all-correct performance and PRMBench step
ranking separately, plus length/domain/model slices, stability, coverage,
fallback rate and runtime. Full-answer fitting supports offline localization;
it does not by itself demonstrate causal online detection.

## 5. Follow-ups after the matched window baseline

**Graph/lambda decision:** closed for the current answer-only representation.
The meaningful lambda-0.1 graph does not beat lambda zero and trails its
node-permuted control; Claude's pooled experiment also degrades from lambda zero
to 0.1 to 0.5. Do not add a larger graph lambda. `target_condition` is a
different inverse-conditioning parameter and remains a possible separate
pooled-Joint study with nested selection and untouched confirmation data; it is
not the next answer-only localization stage.

**Alternatives to more graph changes:** after the representation comparison,
consider one small readout study using persistence or a change-point/sequence
model against the fixed reducer. HMMs, change points, centering and reducer
search have historical tests, but a changed window representation can make a
new, precisely stated test meaningful. Do not claim they were already tested
in this exact setting. Prefer a focused hypothesis about sustained error onset,
short peaks or absolute calibration over another unrestricted variant sweep.
Sample-point/stride selection must also be training-only or label-free.

## 5A. Separate transfer experiment: the historical 24 final-answer cells

Omri separately requests testing the fusion recipe that leads localization on
the historical **24-cell final-answer hallucination-detection benchmark**.
This is an additional experiment, not a replacement for single-answer
localization. It can begin once the localization candidate and transfer
contract are frozen; it need not await a new confirmation-data collection.

- Select/freeze the recipe using the localization study only. If ProcessBench
  and PRMBench lead to different candidates, preserve separately identified
  transfer rows under a declared selection rule. Do not choose which one to
  report by inspecting the 24-cell scores.
- **Primary: fusion-rule transfer.** Apply the selected canonical fusion
  mechanism to the frozen historical answer-by-feature matrix, preserving the
  24-cell population, feature inventory, orientation, fit scope, metric,
  aggregation and comparator contract. Register any necessary feature/group
  adaptation; do not silently carry a 23-token-stream subset into a different
  historical feature inventory. The answer-only fit restriction is the primary
  localization goal; this transfer arm follows the historical final-answer
  fitting contract so that the old candidates remain comparable.
- **If the claimed winner includes a trajectory reducer:** specify a separate
  complete-pipeline transfer using cached traces, window/step mapping and an
  answer-level readout, with a fixed input-access and compute contract. Check
  telemetry availability across all 24 cells first. Do not claim that testing
  the feature fuser alone tested the winning two-axis pipeline. Any unavailable
  full-pipeline comparison gets a visible coverage status.
- Add the new row(s) to the existing historical candidate table, including
  canonical IU/LIU/L-SML, the actual historical leading methods and simple
  references. Reuse source-bound scores where contracts match; otherwise run
  an explicit bridge. Preserve all existing releases.
- Report the historical primary aggregation and uncertainty, per-cell and
  domain results, paired gains/losses, coverage and runtime. A single method
  need not lead both tasks. These extensively inspected cells supply
  retrospective transfer evidence, not untouched confirmation or an excuse
  to tune the localizer on the final-answer benchmark.

Deliverable: a separate transfer protocol, coverage manifest and 24-cell
comparison table linked to the exact frozen localization recipe. No transfer
fit was launched by this planning update.

## 5B. Backlog idea: DUFS-style selection of token/window sampling points

Omri proposed applying the original DUFS idea, which was used here for
feature selection, to observations along one answer. A parameter-free graph
over tokens, non-overlapping windows or candidate official-step points could
learn survival/gate values and retain a compact set of informative points
before feature or trajectory fusion. This is a backlog idea, not part of the
completed pilot and not evidence that DUFS will improve localization.

The smallest future test would keep one answer-only IU reference and compare
uniform windows, top-risk windows and DUFS-selected windows on the same trace
and fixed feature/readout contract. The selector must be label-free, fit only
from that answer in the strict arm, preserve chronological coordinates and
return an explicit unsupported status when it removes all usable points.
Graph similarity and temporal adjacency are different relations and must be
reported separately. A selected point can be locally similar yet be the only
short error peak; therefore evaluate boundary/first-error recall, selected-set
stability under block perturbation, coverage, runtime and step localization,
not just final-answer AUROC. Apply this only after discussing the completed
answer-only IU/Joint pilot; do not expand the current experiment.

## 6. Execution order, outputs and decision gates

| Order | Work | Concrete output / completion condition |
|---|---|---|
| 1 — prepare now | Comparator inventory and historical protocol map | Coverage registry, exact incumbent scorer paths, list of differences requiring a bridge |
| 2 — after Claude finishes | Independent final v2 audit and interpretation | Full report including Module B, guards, exclusions, amendment/source bindings and graph controls; explain the S1/S2 failures before promoting or discarding their mechanisms |
| 3 | Freeze the continuing localization benchmark and reproduce anchors | Manifest, reusable fold IDs, matched tables and old/new bridge; stop if an anchor discrepancy is unexplained |
| 4 | Extend label-free window feasibility, then pilot canonical fitting | Width/feature/stability/fallback contract across cells, tested single-answer and pooled implementations, measured fitting runtime |
| 5 | Run the selected feature/trajectory combinations under the shared benchmark | Strict single-answer primary, pooled/hybrid comparisons explicit; paired results, honest coverage and compute; no silent best-width selection on test labels |
| 6 — conditional | Larger-lambda or one sequence/readout study | A specific unresolved mechanism and a bounded registered test; retain relevant controls |
| Separate track after localization recipe selection | Transfer the frozen fusion recipe to the historical 24 final-answer cells | Same historical benchmark/comparators; distinguish fusion-rule and full-pipeline transfer; no 24-cell tuning |
| 7 | Publication confirmation and advisor report | Freeze the chosen recipe, evaluate on untouched evidence with strong comparators, then state the supported contribution and limitations |

Steps 1 and the documentation work need not wait for Claude. Final method
interpretation and the dependent comparison should wait for his completed
artifacts. Do not stop his run merely to prepare the benchmark.

Continue implementation in an isolated, source-only worktree and share large
inputs read-only; preserve Claude's current source/output namespaces. Before
resuming, reconcile the latest repairs and verify worktree access. The current
session can write the main checkout only, so this planning step did not modify
either experiment worktree.

For AIRCC: use a local smoke test, a small fitting pilot and then full cells.
The previous eight-CPU feature job was fast, but it did not measure Joint
fitting speed. Measure CPU/memory and queue costs before assigning resources;
parallelize independent cells/folds with capped BLAS threads, atomic
checkpoints and independent outputs. The prior run used
`/shared/cycle3_tau_averbuch_prj/omrisegev1/`; verify the current account/path
before submission. Existing cycle2 data stays read-only. No GPU is needed for
the existing NumPy/SciPy window feature stage. Move large finished results
directly from AIRCC to `gdrive:` under the established cluster procedure.

Cleanup is a separate maintenance task: identify active entry points, mark
superseded scripts and archive reproducibility code before considering removal.
Check hashes/backups and active use before deleting redundant artifacts; do not
delete raw traces, historical results or worktrees based only on their age.
Large Drive mutations still require exact authorized paths. Consolidating the
canonical scorer registry is also how we prevent old code from being selected
accidentally.

For publication, the current inspected cells are development evidence. A new
experiment name does not make them untouched. The confirmation plan must
specify fresh questions/populations or a genuinely held-out transfer set,
primary contrasts, uncertainty and a stopping rule before evaluation. A clean
negative mechanism result may still inform the paper; a universal graph/SOTA
claim is not supported by the current preliminary table.

## Evidence to reopen at the next session

- Main checkout: `docs/reviews/project_publication_review_2026-09-05.html`,
  `docs/reviews/joint_lsml_visual_guide_2026-09-06.html`.
- Window branch: `docs/experiments/PER_ANSWER_LOCALIZATION_V1.md`,
  `docs/reviews/window_feasibility_2026-09-06.json`,
  `docs/reviews/advisor_thread_reconciliation_2026-09-06.md`,
  `docs/reviews/post_email_method_lineage_2026-09-06.md`.
- Claude worktree: `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md`,
  its R1/R2/R3 amendments, and the result root's `AUDIT_PRELABEL_RECEIPT.json`,
  `INTEGRITY_RECORD_V2.json`, `AMENDMENT_R3_FREEZE.json` and `evaluation/`.
- Latest inspected Claude conversation:
  `C:/Users/DELL/.claude/projects/C--Users-omris-TAU-hallucination-detection/50cfe0dd-cccd-410a-a2f4-58cedbd12474.jsonl`.
  Read only the relevant project messages; prefer completed artifacts for
  scientific claims.
