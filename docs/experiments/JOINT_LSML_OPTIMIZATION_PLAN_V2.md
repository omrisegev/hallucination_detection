# Joint L-SML localization optimization plan v2

Status: `REGISTERED_RETROSPECTIVE_DEVELOPMENT_STUDY`
Date: 2026-09-05
Branch: `claude/joint-lsml-optimization-v2` (worktree copy of
`codex/joint-lsml-localization-eval-v1`; frozen result namespaces untouched)
Author: Claude (protocol), under Omri's 2026-09-05 directives; Codex reviews via the pushed
branch.

Supersedes: `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V1.md` (DRAFT, never registered or
run). Every supersession of previously frozen text is listed in Section 0 and mirrored in
`PRIOR_ORDER_AUDIT_JOINT_LSML_OPTIMIZATION_V2.md`.

## 0. Authority, scope, and explicit supersessions

All outcomes on the opened Qwen ProcessBench/PRMBench populations are
`RETROSPECTIVE_OPENED_DEVELOPMENT`. Confirmation/generalization requires the separately
registered fresh-population experiment (Section 9); nothing here is a promotion or new-leader
claim.

User-authorized supersessions (Omri, 2026-09-05), recorded per project convention:

1. **v1 plan superseded**: (a) the fusion-order axis {token-fuse-then-step-reduce,
   per-feature-step-reduce-then-fuse} is removed from the factorial (Step-253 precedent); one
   diagnostic row remains (Section 6.4). (b) DUFS moves from "separately registered second
   stage" into the core roster. (c) The primary development readout becomes symmetric
   tuned-vs-tuned with **16-vs-16** configuration budgets (v1 was 8-vs-8).
2. **One-DUFS-mechanism default superseded**: Steps 225/343 established "one variant, one
   discussion" and the Step-349 DUFS slot named a single soft-affinity mechanism. Omri's
   explicit directive includes **four** gate-integration mechanisms in this study (Hooks 1, 2,
   3a, 3b below). Compensating controls so any win remains attributable (Section 7.3): exact
   lambda=0 identity rows per mechanism, permutation negative controls per mechanism,
   dose-response rows, and the mechanism-attribution rule (a gated win counts as mechanism
   evidence only if its permutation control fails the same gate).
3. **Step-347 all-eight ProcessBench rule superseded**: whole-panel `STRUCTURAL_NO_SCORE` on
   one blocked cell is replaced by the registered per-lane provenance fallback (Section 3.4),
   with the strict `STRUCTURAL_NO_SCORE` aggregation co-reported as a sensitivity bound. The
   SD=1 invariant removes the scale-splice hazard that made mixing dangerous in Step 348.
4. **No closure clause**: the opened Qwen populations remain open for development after this
   study (Omri: "we are not publishing anything yet; generalization is done separately next").
   The cumulative exposure ledger is still appended — bookkeeping, not a gate.
5. **Freeze rule**: the tuned winner (nested-CV selection) may be frozen for the fresh-data
   experiment. This is disclosed as label-selected development; the pre-fixed label-free rows
   S1/S2 are co-reported throughout so a selection-untouched line always exists.

Non-negotiables that do NOT change: ProcessBench and PRMBench metrics are never averaged;
separate selection and gates per panel; intersection-union for any "support" claim; frozen
reducers as incumbents (PB top-min(10, step_length) step mean + max-token answer detector;
PRMB official-span max) — Module B (Section 6B) *studies* the reducer with the incumbent as
control, it does not replace it; DUFS never selects K; DUFS gates are signed — never rank by
|mu|; `results/joint_lsml_*` namespaces are immutable; all new artifacts go under
`results/joint_lsml_optimization_v2/` only.

## 1. Prior-step conflict table (mandatory)

| Axis in v2 | HISTORY steps | Binding verdict for v2 |
|---|---|---|
| Score-scale convention | 347/348/349 | SD=1 donor-score invariant on every arm; unit-L2 kept only as a diagnostic |
| Grouping: INTERNAL vs provenance | 347 (every fitted lane chose K=3; `processbench_math_qwen3_4b` blocked), 349 ("data do not yet causally separate group discovery from the hierarchical map") | 2x2 core of the roster; K rule unchanged from Step 347 (comparability) |
| Weight map: CONT vs HIERARCHICAL_JOINT | 347 PRMB 0.669063 vs IU 0.671539 vs fixed 0.672619; 348 `HARM__NO_PROMOTION` | JOINT re-tested only under SD=1 + guards; the frozen Step-347 artifact is never re-scored |
| DUFS as per-feature ranker / transplanted keep rules | 222 (-0.70pp), 224 (-0.96pp Holm 0.0072) | CLOSED; not reopened (gates here weight coefficients, they never rank/prune by gate order) |
| DUFS family sweeps at localization | 343 stop (deltas +0.000168 / -0.000065, "do not open another DUFS family") | Reopened by explicit user override (Section 0.2), with per-mechanism attribution controls |
| DUFS detection ties | 207/216 (0.7507 vs 0.7551 p=0.059; 0.7687 vs 0.7741 p=0.067; point estimate always behind) | Motivates full-dose lambda=1.0 rows + the inertness futility guard |
| Small-m SML degeneracy | 203/205 (Eq.15 exactly zero at m=3; "numerically undetermined at 4 views") | DOF-aware guard (Section 3.3) |
| Fusion order | 253 (step-first 0.6136 vs trajectory-first 0.6711) | Axis dropped; exactly one FR_SF diagnostic row, PRMB-only |
| Reducer choice | 312-326 (fixed-reducer ladder; top-10 mean frozen) | Closed "which hand-picked rule", NOT "structure-derived weights" — Module B's control is the frozen top-10 mean; flat ladder = small-headroom prior |
| Threshold pooling / fallback splice | 348 flat-SML splice; 349 activation collapse 9.6%/14.5% vs 67.0%/53.8% | Provenance fallback + SD=1 + activation-floor catastrophe guard |
| Adaptive-K | 198 (all label-free size rules at chance) | K stays inside the frozen stability rule; no gate-derived K |
| Supervised baseline hygiene | SUPERVISED_ORACLE_CORRECTION.md | Module-B LR row: class_weight='balanced', fold-scoped fits, no cross_val_predict calibration |

## 2. Populations, panels, folds

- ProcessBench: the 8 opened Qwen cells (4 subsets x {Qwen3-4B, Qwen3-8B}), full N.
  Local telemetry: `dataset_cache/repgrid/pb_qwen3_4b/*.pkl` and
  `dataset_cache/four_localization/pb_uprm_baseline_qwen3_8b_full/*.pkl` (main checkout).
- PRMBench: the opened Qwen3-8B response population
  (`dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/`), official spans, the
  three registered alignment exclusions unchanged.
- Folds: deterministic namespaced grouped assignment (existing
  `fair_comparisons.folds.assign_group_folds`), namespace `joint_lsml_optimization_v2`;
  5 outer folds; 5 inner folds inside each outer-train, namespace
  `joint_lsml_optimization_v2/outer{k}/inner`. ProcessBench q4/q8 copies of a source question
  share folds (existing pairing asserts). PRMBench folds are problem-level.
- Fit scope: imputation medians, mean/SD standardization, DUFS gates, covariance, grouping,
  weights, and the SD=1 scalar are computed on outer-train rows only (inner-train rows during
  tuning); held rows are projection-only. Required because `upcr_fit` consumes uncentered
  C = FF^T/n — valid only when the fit population equals the standardization population.

## 3. Estimator invariants (every learned arm)

### 3.1 Donor fused-score SD=1
Every (23,) weight vector is rescaled by `1 / SD(Z_fit @ w)` on its fit population before any
cross-cell threshold or score freeze. Floor 1e-8, fail-closed (arm inadmissible on that lane).
Unit-L2 re-normalization is computed as a scale-sensitivity diagnostic only. The fixed-family
exact-reconstruction assertion keeps the scalar applied to both sides.

### 3.2 One orientation anchor
All arms unified on the standardized-rowmean Pearson rule (the current controls' rule).
Registered fallback when |corr| < 0.02: entropy_series Spearman. Both undefined →
`ORIENTATION_UNDETERMINED`, arm inadmissible on that lane. Applied after SD normalization.

### 3.3 Small-m degeneracy guard (Steps 203/205)
Any SML eigen-stage over m=3 units (within-group or cross-group) is replaced by equal weights
over SD-standardized units; m=4 is retained but flagged `small_m_flag`. m<=2 unchanged.
Registration-time assertion: the guard is a no-op on the provenance-family CONT arms per cell
(provenance family sizes are validated then; a size-3 family would be registered as an explicit
exemption, never silently absorbed). Interpretation limit: under K=3 the INTERNAL "grouping
effect" includes the guarded (equal) cross stage; the map axis remains cleanly isolated.

### 3.4 Blocked-cell policy
INTERNAL grouping with no admissible partition on a (cell x fold) lane → provenance groups with
the SAME map type (CONT stays CONT, JOINT stays JOINT), same invariants; logged per lane;
fallback-rate table reported. Co-reported sensitivity aggregation scores blocked lanes as
`STRUCTURAL_NO_SCORE`. Fragility cap: fallback on >10 of 40 PB lanes or >1 of 5 PRMB lanes →
INTERNAL family flagged `STRUCTURALLY_FRAGILE` (bars S1 fresh-data eligibility; development
readouts still reported). Flat-SML is never spliced into a scoring panel.

## 4. Rosters (16 vs 16) and controls

### 4.1 IU family (16) — full cross
{n_components 1,2} x {scale_ratio 0.25,0.10} x {loss l2,l1} x {exclusion+refit off,on}; all
other `upcr_fit` gates stay off/default. Row 1 = deployed `IU_CONFIG` (2, 0.25, l2, off).
Roster order (pre-registered tie-break): components-major descending exactly as enumerated in
`configs/joint_lsml_optimization_v2.json`.

### 4.2 Joint/L-SML family (16) — roster order = selection tie-break

| # | id | grouping | map | mechanism |
|---|---|---|---|---|
| R1 | prov5_cont | provenance | CONT_LSML | none (lambda=0 anchor) |
| R2 | prov5_joint | provenance | HIER_JOINT | none (scale-repaired joint on stable groups) |
| R3 | internal_cont | INTERNAL | CONT_LSML | none (= S2) |
| R4 | internal_joint | INTERNAL | HIER_JOINT | none (= S1, the repaired contribution) |
| R5 | prov5_cont_gate050 | provenance | CONT | Hook 2 within-group, lambda=0.5 |
| R6 | prov5_cont_gate100 | provenance | CONT | Hook 2 within-group, lambda=1.0 |
| R7 | internal_cont_gate100 | INTERNAL | CONT | Hook 2 within-group, lambda=1.0 |
| R8 | internal_joint_gate050 | INTERNAL | JOINT | Hook 2 on the joint fit: covariance replaced by diag(q_l) Sigma_off diag(q_l), factors pulled back through diag(q_l); gated cross stage; lambda=0.5 |
| R9 | internal_joint_gate100 | INTERNAL | JOINT | same, lambda=1.0 |
| R10 | internal_joint_liu010 | INTERNAL | joint model-inverse | Hook 3a (LIU transplant): w = solve(Psi(C_model) + gamma I + lambda R_bar, v), R_bar = trace-matched Z^T L Z / n from the DUFS-gated sample kNN graph; lambda=0.1 (LIU's deployed value) |
| R11 | internal_joint_liu050 | INTERNAL | joint model-inverse | Hook 3a, lambda=0.5 |
| R12 | internal_joint_diag010 | INTERNAL | joint model-inverse | Hook 3b: + lambda diag(1/q^2) prior (dense-graph-free; labeled a distinct mechanism, not a LIU port); lambda=0.1 |
| R13 | internal_joint_diag050 | INTERNAL | joint model-inverse | Hook 3b, lambda=0.5 |
| R14 | internal_gaff_cont | INTERNAL on diag(q) affinity diag(q) | CONT | Hook 1 (grouping only; coefficients untouched) |
| R15 | internal_gaff_joint | same gated grouping | JOINT | Hook 1 |
| R16 | dufs_pf_lsml | provenance survivors | CONT (unmodified) | historical hard selector: `dufs_pf_gates`, signed mu>0 (never |mu|), fail-closed below 9 survivors; emptied families dropped, singletons weight 1 |

Gates `q`: `adapted_dufs_soft_gates(standardized_fit.T)` (frozen constants, seeds (0,1,2),
RMS-normalized nonnegative), computed once per (cell x fold population) and shared by every
gated row. Effective gates `q_lambda = (1-lambda)*1 + lambda*q`; lambda=0 (or all-ones gates)
takes the unmodified code path verbatim → exact identity, mirrored from
`laplacian_upcr.py`'s `lambda_==0` branch. Hook 3a/3b act on the joint **model covariance**
map — the only head family that uses the fitted group factors u, directly answering the
Step-349 objective/head mismatch.

### 4.3 Named controls (outside the budgets; SD=1; outer-refit)
1. deployed IU-PCR (`IU_CONFIG`);
2. deployed U-PCR port: exclusion+refit on (the Step-341 P3D1 recipe at token level; equals
   IU roster row (2, .25, l2, on) — reported as a named control row as well);
3. equal-all23;
4. equal-family;
5. fixed-family continuous L-SML (= R1's estimator untuned; ledger counts it once).

Historical anchors (context only, never comparators): PB `0.3662328342` (different H2/H3
contract), Step-347/348 frozen numbers.

## 5. Estimand and reporting

### 5.1 Primary development readout: tuned-vs-tuned
Per task, per outer fold: each family's configuration is selected by the 5-inner-fold mean of
the task-native metric (PB: macro-F1 under inner-cross-fitted thresholds fit on inner-train
only; PRMB: step AUROC); ties break by roster order. The selected configuration is refit on the
full outer-train and scored once on outer-test; PB outer thresholds via the existing
cross-fitted threshold machinery on outer-train only. Report per task: tuned-IU, tuned-Joint,
and Delta = tuned-Joint - tuned-IU with paired grouped bootstrap 95% CI (in-replicate threshold
refit; >=2,000 draws PB / 10,000 PRMB; frozen seeds; source-question groups on PB, problems on
PRMB). Registered approximation: the CI conditions on the per-fold selected configurations;
selection variability is reported as a selection-frequency table.

### 5.2 Co-primary label-free rows
S1 = `internal_joint` (R4) and S2 = `internal_cont` (R3), scored through the same outer folds
with no inner selection; paired contrasts vs deployed IU-PCR.

### 5.3 Panels and claims
PB and PRMB always separate; `SUPPORTED (development)` requires the support gate on BOTH tasks
(intersection-union); one task → `PARTIAL_TASK_SUPPORT`.

## 6. Diagnostics

1. Unit-L2 scale-sensitivity rescore of S1, S2, and the per-task tuned winners.
2. Gate-permutation negative control (Hook 2 rows): feature-permuted q at lambda=1.0, fixed
   seed, both panels.
3. Graph-permutation negative control (Hook 3a rows): node-relabeled Laplacian
   (existing `permute_graph`), fixed seed.
4. Gated-vs-ungated weight cosine per lane (mechanism inertness; Section 7.4).
5. INTERNAL-vs-provenance ARI per lane; K/group-size/fallback/`small_m_flag` tables.
6. Per-arm weight-map agreement: Spearman vs the fixed-family control per lane, floor 0.50 —
   per-arm flags only, never a whole-panel block; a lane blocks only if the control itself is
   degenerate.
7. Dose-response summaries over lambda in {0, 0.5, 1.0} (Hook 2 on provenance substrate;
   {0, 0.1, 0.5} for Hooks 3a/3b) — descriptive monotonicity.
8. Exactly one fusion-order diagnostic row: per-feature-step-reduce-then-fuse variant of S1,
   PRMB-only, threshold-free (Step-253 conflict entry).
9. Conditional IU-on-DUFS-support row (only if R16 is selected by any outer fold).
10. Leave-one-subset-pair-out PB scale-transfer audit, fixed arms only (S1, S2, deployed IU,
    fixed-family control).

### 6B. Module B — learned trajectory-axis reducer

Idea (Omri 2026-09-05): apply the same label-free weighting principle along the trajectory
axis. Raw token positions are not aligned units, but **order statistics are**: the sorted
top-k token risks within a step ("largest", "2nd largest", ..., "k-th") are exchangeable views
of the step's risk level. Module B learns weights over the k=10 order statistics instead of
the incumbent equal-weight top-10 mean. Substrate: the **deployed IU token score** (fixed →
clean attribution; decoupled from Module A). One mechanism, two axes.

| id | arm |
|---|---|
| B0 | control: frozen top-10 mean (PB) / span max (PRMB) |
| B1 | label-free learned weights over the k=10 order statistics (`sml_fuse_signed` over order-stat views; samples = outer-train steps; orientation known-positive; SD=1) |
| B2a | learned max-vs-mean blend: alpha*OS1 + (1-alpha)*top10mean, alpha inner-CV over {0,.25,.5,.75,1} (peaked-vs-diffuse; alpha=0 reproduces B0 exactly) |
| B2b | positional bins: 5 relative-position bins per step, learned weights (where-in-the-step) |
| B3 | supervised LR competitor over the same k=10 order statistics; class_weight='balanced'; outer-train labels only; PB step labels: pre-error steps=clean, first-error step=error, post-error steps EXCLUDED; PRMB official per-step labels; labeled SUPERVISED |
| B4 | composition row: B-winner reducer on the Module-A tuned winner's token score (descriptive only) |

Pre-registered mechanism diagnostic (tail-vs-bulk): B1's label-free objective reconstructs the
shared step-level factor and is expected to drift toward trimmed/bulk weights; if the
discriminative signal is tail-concentrated, B1 down-weights exactly the informative slots. The
module's headline mechanism figure compares B1's weight profile against B3's LR coefficient
profile over the same 10 slots, per task per fold, regardless of gate outcomes. A
within-answer-centered covariance fit is co-reported as a weight-profile diagnostic (no scoring
arm); the scored B1 uses raw values.

Readout: PRMB step AUROC is the PRIMARY Module-B gate (the reducer IS the score there); PB
macro-F1 is SECONDARY (the reducer only touches the locator; the PB answer detector stays
incumbent max token risk). Fitting: steps with >=10 tokens; replay rule for shorter steps/spans
= renormalized truncation of the weight vector. k=10 → no small-m issue. B1/B2b are label-free
at application; B2a is inner-CV-tuned (development row); B3 is supervised (competitor/ceiling;
freezable for fresh data only with SUPERVISED disclosure).

Module-B gates: paired vs B0, PRMB primary / PB secondary; SUPPORT = CI lower > 0 on PRMB (no
extra floor; exploratory module); any B row is fresh-data-eligible only if non-inferior to B0
on both tasks and superior on PRMB.

## 7. Decision gates (frozen before any label access)

### 7.1 Development gates (tuned-vs-tuned)
PB: SUPPORT iff the 95% CI lower bound of Delta > 0 AND point Delta >= +0.010 macro-F1; HARM
iff CI upper < 0; else NULL. PRMB: same with floor +0.005 AUROC. Calibration: +0.005 is ~2x the
Step-347 harm magnitude and ~25x Step-343 tie noise; +0.010 is ~1/6 of the diagnosed Step-349
net panel harm. Qualifier `UNSTABLE_SELECTION` when no configuration reaches 3/5 outer-fold
selections in a family.

### 7.2 Fresh-data freeze rule
The per-task tuned winner is frozen for the fresh-population experiment (if the two tasks
select different configurations, both are frozen, one per task, disclosed). S1 and S2 are
always carried. Non-inferiority floor for ANY frozen object vs deployed IU: CI lower bound
> -0.005 (PB macro-F1) / > -0.0025 (PRMB AUROC) — a successor may not be as bad as the failure
it replaces.

### 7.3 Mechanism attribution rule
A gated row's win counts as evidence FOR its mechanism only if the matching permutation control
(gate-permuted for Hook 2, graph-permuted for Hook 3a) fails the same gate on the same task.
Otherwise the win is reported as `MECHANISM_UNATTRIBUTED` (still a valid development result).

### 7.4 Catastrophe guards (mechanical)
- PB activation floor: per (cell x fold), out-of-fold error-side activation >= max(0.10,
  0.5 x fixed-family control's activation on the same lane); >=2 violating cells of 8 → arm
  `CATASTROPHE`, excluded from freezing regardless of aggregate F1 (calibrated to catch the
  Step-349 failure: 9.6%/14.5% vs 67.0%/53.8%).
- Per-arm map agreement (Section 6.6) violated on >=2 PB lanes or any PRMB fold → freeze
  barred.
- Cross-fold map cosine >= 0.5 (mean pairwise, oriented maps), else `UNSTABLE_MAP`, freeze
  barred.
- Gate inertness futility: cosine(gated, ungated) >= 0.995 on every lane → the gated row is
  `MECHANISM_INERT` (reported; auto-non-frozen).
- Gate seed stability: `mean_seed_std` <= 0.15 on every lane, else `GATE_UNSTABLE`, gated rows
  non-freezable.
- Fail-closed everywhere: SD floor, orientation undefined, non-finite values → lane
  inadmissible, never silently repaired.

### 7.5 Study-level aborts (pre-label)
Any lambda=0 / all-ones identity failure; the v1 bit-exact regression fixture failure;
fold-hash mismatch; any write into an immutable namespace; AST label-firewall failure; the
small-m/provenance no-op assertion failure; INTERNAL blocked-lane rate above the Section-3.4
cap on the structure run (requires a registered amendment before labels).

## 8. Execution order

1. Registration: this document + the prior-order audit committed; source hashes + config
   hash-bound in `results/joint_lsml_optimization_v2/EXECUTION_REGISTRY.json`.
2. Implementation (Section 10 file map) + unit tests.
3. Pre-label audit suite green (identities, regression fixture, guard triggers, fold goldens,
   immutability scan, firewall).
4. Synthetic 2-cell end-to-end smoke (including bootstrap machinery).
5. Real-telemetry pre-label structure run: fits, gates census, fallback rates, score freeze,
   independent pre-label audit. No label decode.
6. Label evaluation (separate evaluator process; firewalled imports) + report + plots.
7. Independent post-label re-computation pass (project convention).
8. HISTORY step + PROGRESS update; commit and push `claude/joint-lsml-optimization-v2`.

## 9. Fresh-population stub (registered, execution deferred)

Reserved scorer: `Microsoft/Phi-4-reasoning-plus` (immutable revision resolved at preflight).
Frozen objects: the per-task tuned winners (weights-generation recipe, not weights) + S1 + S2 +
any Module-B row passing its gate. Same invariants, panels, reducers (or the Module-B winner as
a disclosed second reducer row), gates. Named eligible additions on positive v2 evidence:
none required — all four hook mechanisms already ran here. Generation is not authorized by this
document.

## 10. Implementation file map

New (this branch): `spectral_utils/joint_lsml_v2_localization.py` (SD/orient invariant, 16-row
roster, fallback restructure, gate cache), `spectral_utils/trajectory_reducer.py` (Module B),
`scripts/joint_lsml_optimization_v2/run_v2.py` (nested runner + freeze),
`scripts/joint_lsml_optimization_v2/audit_prelabel.py`,
`scripts/joint_lsml_optimization_v2/evaluate_v2.py` (label stage),
`configs/joint_lsml_optimization_v2.json`, `tests/test_joint_lsml_v2.py`,
`tests/test_trajectory_reducer.py`.

Modified (additive, default-preserving; bit-exact when new arguments are unset):
`spectral_utils/fusion_utils.py` (`sml_fuse_signed(..., gates=None)`;
`lsml_continuous(..., gates=None, small_m_guard=False)`),
`spectral_utils/joint_lsml.py` (new wrapper functions for the gated joint fit and the
regularized model-inverse maps; `fit_joint_lsml` internals untouched),
`spectral_utils/joint_lsml_localization.py` (`prepare_active23(..., fit_row_mask=None)`).

Immutable: `results/joint_lsml_*`, existing `configs/joint_lsml_*.json`, HISTORY entries,
`spectral_utils/joint_lsml_processbench_amendment.py`, `spectral_utils/dependency_fusion.py`
(Hook 3a/3b wrap `regularized_covariance_weights`; they do not edit it).
