# Joint L-SML localization evaluation v1

Status: `DRAFT_PROTOCOL__NOT_REGISTERED__NO_SCORING_AUTHORITY`

This document records the completed Agent-B work and defines the next
ProcessBench/PRMBench experiment. It does not register a run, open a label,
create a score array, authorize cluster work, or support a promotion claim.

## 1. Decision in one paragraph

Joint L-SML is a credible **structural** candidate, but it is not yet an
efficacy result. The next valid test changes only the token-fusion part of the
localization pipeline, keeps the task adapters fixed, and compares one frozen
Joint L-SML candidate with three matched controls. The already opened Qwen and
Llama ProcessBench/PRMBench populations will not be rescored. The planned run
uses a new scorer-model population, freezes all scores before the evaluator can
load labels, and reports ProcessBench and PRMBench as separate estimands.

## 2. What Agent B actually completed

### 2.1 Lineage and scope

- Worktree: `local_cache/worktrees/og_sml_agent_b_v1`
- Branch: `codex/og-sml-agent-b-v1`
- Base SHA: `250e092e1a0f5b2e460e2fd0221bcbded28069dc`
- Git publication: no staging, commit, push, or pull was performed.
- Outcome boundary: no benchmark label, target, AUROC, F1, AURC, or persisted
  fused-score array was accessed by the Agent-B structural run.

The attached OG-SML proposal first required a retrospective falsification test,
T0. T0 failed its frozen prediction: none of the three prior C-v2 primary-gate
passes was graph-admissible, while six of fifteen failures were admissible, and
their selection-J values did not separate. Therefore the proposed overlapping
OG-SML Steps 0--6 were not implemented. This does **not** falsify the graph
theorems; it falsifies the claim that they explain the previous C-v2 gate
pattern.

Agent B then implemented the narrower, directly testable **Joint disjoint
L-SML** estimator. It uses one automatically learned hard partition and jointly
fits the shared loading and group loadings. Overlap is outside v1.

Canonical T0 evidence:

- `results/og_sml_agent_b_v1/T0_REPORT.md`
- terminal state: `T0_FALSIFIED_STOP_BEFORE_STEPS_0_6`

Canonical Joint L-SML evidence:

- `results/joint_lsml_v1_r2/REPORT.md`
- `results/joint_lsml_v1_r2/SUMMARY.json`
- `results/joint_lsml_v1_r2/JOINT_STRUCTURAL_LEDGER.json`
- `results/joint_lsml_v1_r2/INDEPENDENT_AUDIT.md`
- `results/joint_lsml_v1_r2/COMPLETE.json`

### 2.2 Frozen orientation and roster

The 29-field absolute raw-domain orientation registry is compatible with the
strict V2 loader. `trace_length_series` is nuisance-only and cannot enter the
fit. Five of the 28 token-varying streams are removed globally, leaving a
23-stream active roster:

1. `entropy_rolling_hl_ratio` -- weak and degree-rejected;
2. `entropy_pe_series` -- weak, sign-unstable, and degree-rejected;
3. `spilled_cusum_abs_series` -- degree-rejected;
4. `spilled_rolling_min` -- weak and degree-rejected;
5. `energy_cusum_abs_series` -- sign-unstable and degree-rejected.

Two streams are not unstable; they are stable orientation corrections and stay
active:

- `entropy_rolling_spectral_entropy`;
- `energy_sw_var_series`.

Frozen files and file hashes:

- `results/joint_lsml_v1_r2/V2_ABSOLUTE_ORIENTATION_REGISTRY.json`
  (`65e765da9feb5f9969b9cd825cce2d3c4b2e5e551e4b63a423bd2ce18b77e5ad`)
- `results/joint_lsml_v1_r2/V2_GLOBAL_PRUNED_ROSTER.json`
  (`cd90d253ba228d06d74c1fd7f4f3fd4afdd0d65e72f89eeebc3388c5cb20558d`)

### 2.3 Structural result

The run contained nine target-free donor cells, each with an active-23 lane and
an H2 reference lane, for 18 lane-cells total.

- Ridge-condition donor-score stability passed in 18/18 lanes. The minimum
  pairwise Spearman across target conditions `1e2`, `1e3`, and `1e4` ranged from
  `0.9933622788` to `0.9999503673`.
- Joint L-SML produced an admissible fit in 16/18 lanes.
- Both blocked artifacts were the `v2_active28` ProcessBench lanes after global
  pruning to 23 fitted streams: `math/Qwen3-4B` and
  `omnimath/Qwen3-8B`.
- All 16 fitted lanes converged from all five starts, passed the multistart and
  profiled-Jacobian checks, and produced finite weights.
- Joint off-diagonal misfit was lower than historical hard continuous L-SML in
  all 16 fitted lanes. Absolute improvement ranged from `0.02649` to `0.08338`.
- The four candidate weight maps were not interchangeable. Their per-lane
  minimum donor-score Spearman ranged from `0.708240` to `0.976135`.
- Diagonal-residual clipping occurred in 14/16 fitted lanes, at no more than two
  coordinates per lane and with maximum clipped mass `0.05452`.

The independent audit reconstructed all 18 ridge diagnostics, all 16 fitted
covariances, all saved score-ranking comparisons, the orientation, roster, and
hash inventories. Its verdict was PASS for the bounded label-free structural
claims.

### 2.4 What V2 contributed after the handoff

The parallel Within-Answer Graph Structure Discovery V2 pilot used the Agent-B
orientation and roster, but its result was negative:

- Q1 selected `NONE__NO_ELIGIBLE_CAPACITY`; no innovation coefficients were
  exported.
- Independent inspection of the Q2 exports bound by the manifest found every
  positive one-lag affinity matrix exactly zero. The apparent ARI of 1.0 was a
  deterministic tie-break on a zero matrix, not learned structure.
- No V2 weight, innovation column, LAG partition, overlap family, or replay is
  eligible for the localization scoring experiment.

Therefore the next experiment has exactly one new candidate and uses only Joint
L-SML's contemporaneous residual grouping.

Canonical V2 evidence:

- `/Users/osegev/Desktop/hallucination_detection/local_cache/worktrees/reasoning_localization_dynamics_sufficiency_v1/results/within_answer_graph_structure_discovery_v2_og_sml_handoff_v1_r2/run_v2/REPORT.md`
  (`aa2982683ff6306f20ffba338f7e21dd08c356257ab32608933947aab1937fdc`)
- `/Users/osegev/Desktop/hallucination_detection/local_cache/worktrees/reasoning_localization_dynamics_sufficiency_v1/results/within_answer_graph_structure_discovery_v2_og_sml_handoff_v1_r2/run_v2/COMPLETE.json`
  (`806bb57bf2ba8f454d09575fdbfe020998ab79ca7164b7ed38192486155c2e8a`)
- `/Users/osegev/Desktop/hallucination_detection/local_cache/worktrees/reasoning_localization_dynamics_sufficiency_v1/results/within_answer_graph_structure_discovery_v2_og_sml_handoff_v1_r2/run_v2/Q2_LAG_COASSIGNMENT_MANIFEST.json`
  (`a8fa1bda18f9d6044919af7f46b0c07b323dea0b04862844cf69ef9d4d358467`),
  which binds the inspected Q2 sidecars.

## 3. Why a localization test is justified -- and what it can prove

The evidence is sufficient to justify a score-frozen efficacy test because the
candidate is deterministic, target-free, independently audited, and explains
the donor covariance better than the historical two-stage L-SML fit whenever it
is admissible. It is **not** sufficient to predict a localization win: the
weight maps disagree, covariance fit is not an outcome metric, and active-23
failed to produce an admissible partition in two donor cells.

The planned experiment can answer:

> When the feature matrix, orientation, fit rows, token-to-step adapter, and
> threshold protocol are held fixed, does Joint L-SML rank erroneous tokens or
> steps better than the current matched fusion controls?

It cannot establish a new-data confirmation claim because the project has
already inspected the official ProcessBench and PRMBench labels. Using a new,
previously unscored model family makes this a prospective **new-scorer transfer
test**, not a sealed-benchmark confirmation.

## 4. Candidate frozen for the next protocol

Candidate ID: `joint_lsml23_hierarchical_local_top10_v1`

Only this candidate may be evaluated.

### 4.1 Input and preprocessing

1. Extract the existing 29 raw token streams from each complete teacher-forced
   reasoning trace.
2. Exclude `trace_length_series` and the five streams in the frozen global
   pruned-roster registry.
3. Apply the 29-entry absolute confidence-orientation registry, then select the
   23 active streams. Larger values must mean greater confidence.
4. Use the current deterministic token cap of 60,000 positions per cell,
   selected by evenly spaced indices over canonical response order.
5. Estimate imputation medians, means, and standard deviations only on those
   target-free fit tokens; replay the parameters on every token in the cell.
6. Do not apply rank transforms, SUMMA prevalence, DUFS, graph-degree
   normalization, Katz centrality, extra monotonic folds, or V2 innovations.

All three controls must consume the same oriented, imputed, standardized
23-column matrix and the same fit-token indices.

### 4.2 Group discovery

- Unit of leave-one-out: a complete answer, never a token.
- Affinity: absolute residual covariance after one masked rank-one fit.
- Candidate K: `{3,4,6,8}`, restricted to `K < p`.
- A candidate is admissible only when the consensus partition and every held
  answer partition have all group sizes at least three.
- Selection: maximum median LOAO-to-consensus ARI, then maximum mean ARI, then
  smaller K.
- Residual misfit and outcome labels never select K.

There is no hidden fallback. Every ProcessBench subset cell and the PRMBench
cell must produce an admissible partition before label access. A blocked cell
ends its entire benchmark panel as `STRUCTURAL_NO_SCORE`; it is not silently
replaced by H2, fixed families, or ordinary L-SML.

### 4.3 Joint fit and primary weight map

For the selected hard partition, fit on off-diagonal covariance entries:

`R_off = v v^T + sum_g I_g (u_g u_g^T)`.

Use the existing deterministic five-start Gauss-Seidel estimator and its frozen
convergence, multistart, and profiled-Jacobian checks.

The only scored weight map is `hierarchical_joint`:

1. within each learned group, form a virtual classifier `X_g v_g` from the
   jointly estimated shared loading;
2. run ordinary signed SML across the virtual classifiers;
3. compose feature weights as `w_i = v_i * a_group(i)`;
4. orient the fused confidence by the entropy anchor;
5. export token risk as the negative fused confidence.

This choice is frozen before labels because it is the closest direct extension
of continuous L-SML and does not depend on the frequently clipped fitted
diagonal. `model_inverse_1e3` and `sample_inverse_1e3` remain structural
diagnostics only and must not become hidden scoring candidates.

## 5. The three matched controls

The candidate receives exactly three score-producing controls:

1. `matched_active23_iu_pcr`: the repository IU-PCR configuration on the same
   23 columns and fit rows: `loss=l2`, no exclusion/difficulty gate/fallback,
   `g2_projection_k=1`, `scale_ratio=0.25`, two components, and no automatic
   component selection;
2. `equal_family_active23`: equal averaging within the frozen provenance
   families followed by equal averaging across nonempty families;
3. `fixed_family_continuous_lsml_active23`: the maintained continuous L-SML
   implementation with frozen provenance-family groups on the same 23 columns.

After global pruning, the six frozen provenance families contain
`1, 3, 8, 2, 3, 6` active streams. The maintained continuous-L-SML code can
compute singleton and two-stream groups, but those two small groups do not meet
the `>=3` structural-identifiability condition imposed on the learned Joint
partition. This arm is therefore retained only as the required exact
fixed-family algorithmic control; no theorem-valid hard-L-SML interpretation is
claimed for it.

The existing `family6 + level + step_top5mean` ProcessBench result (macro-F1
`0.326141`) and fixed trajectory-first IU PRMBench result (step AUROC
`0.671149`) remain historical system-of-record anchors. They are displayed for
context but are not used to select a weight map, roster, grouping, threshold
rule, or candidate after the new scores are visible.

The same-partition continuous-L-SML score is retained as a pre-label
score-agreement diagnostic only. It is not a fourth efficacy arm. Consequently,
a Joint win would validate the complete automatic-grouping plus joint-fitting
package; target efficacy cannot be attributed to only one of those two changes.

## 6. Fresh scoring population

The default reserved scorer is `Microsoft/Phi-4-reasoning-plus`; its immutable
model revision must be resolved and written into the registry at preflight. No
model or data download is authorized by this document.

The planned population is:

- ProcessBench: all 3,400 official solutions, kept as four separate subset
  cells (`gsm8k`, `math`, `olympiadbench`, `omnimath`);
- PRMBench: the official preview population after only the three already
  registered alignment exclusions;
- one teacher-forced pass over each supplied solution with the same scorer
  model and raw telemetry contract;
- no answer regeneration and no alternative prompt search.

Before acquisition, the registry must pin model revision, tokenizer revision,
dataset revisions, ordered IDs, prompts, chat template, maximum sequence length,
precision, batching, telemetry schema, and exact alignment exclusions.

Because the benchmark outcomes are historically opened, the label firewall
must be physical. Orientation, preprocessing, group discovery, and fusion
fitting receive raw telemetry and row IDs only. An outcome-free reducer stage
may separately receive a hash-bound step-span sidecar after the token curves
are frozen; its schema permits only row/step IDs and token start/end offsets and
must reject `first_error`, `error_steps`, `classification`, correctness, reward,
or other outcome fields. A separate evaluator receives the frozen reduced-score
artifact and label sidecar only after all hashes and structural gates pass.

## 7. Fixed localization adapters

Joint L-SML replaces **only token fusion**. It does not introduce a new
localization reducer or answer-error detector.

### 7.1 ProcessBench first-error panel

For each response:

1. compute the full token-risk curve before reading step boundaries;
2. define each step score as the mean of its largest `min(10, n_step)` finite
   token risks;
3. select the step with the largest step score as the locator;
4. use maximum token risk as the continuous no-error detector score;
5. use the existing fixed five-fold ProcessBench threshold cross-fit, optimizing
   equal-subset macro-F1 on four folds and applying once to the held-out fold.

Primary metric: official equal-subset macro-F1.

Co-reported metrics: per-subset F1, erroneous-trace exact first-step accuracy,
within-one-step accuracy, clean-trace abstention accuracy, and alignment
coverage. These cannot replace the primary metric.

### 7.2 PRMBench every-step panel

For each supplied reasoning step, use maximum token risk inside its official
token span. Positive class is an erroneous step. No operating threshold is
fitted.

Primary metric: step AUROC on the registered common population.

Co-reported metrics: error AUPRC, error prevalence, per-paper-class AUROC/AUPRC,
alignment coverage, and the official native accuracy fields as context. The
three registered alignment exclusions remain unchanged.

ProcessBench macro-F1 and PRMBench step AUROC are never averaged.

## 8. Execution order and gates

### Phase 0 -- implementation and preregistration

Create a new clean worktree only after explicit authorization, proposed path
`local_cache/worktrees/joint_lsml_localization_v1`, on the then-current approved
source SHA. Import the audited Joint L-SML source and bind its hash. Do not edit
the R2 result namespace.

The preregistry must bind:

- candidate and three-control IDs;
- all source hashes and direct imports;
- orientation and roster file hashes;
- input revisions and ordered-ID hashes;
- 60,000-token sampling and preprocessing rules;
- group-discovery and fit settings;
- PB/PRM adapters, metrics, folds, bootstrap seeds, SESOIs, and decision states;
- one shared candidate/multiplicity ledger.

The registry must bind the exact fixed-top-ten behavior of
`scripts/reasoning_localization/run_phase2_reducer.py::reduce_steps`, with an
equivalence test extracted into the outcome-free reducer boundary. Thresholding
and evaluation must bind the implementation and hash from
`spectral_utils.fair_comparisons.localization` and
`spectral_utils.fair_comparisons.evaluator`; it must not silently substitute the
separate 20,000-draw reconstruction evaluator. The PRM reducer must bind the
maximum-inside-official-span behavior in
`scripts/fixed_application_pipeline_experiment.py` and a registered grouped
bootstrap evaluator using an explicit `draws=2000` override.

### Phase 1 -- telemetry-only smoke

With separate authorization, score 30 deterministically selected responses per
benchmark for schema, memory, throughput, alignment, and numerical telemetry
only. Do not read outcome fields and do not compute a performance metric.

### Phase 2 -- full target-free fit and structural gate

Generate all telemetry, fit all four arms, and require for the Joint candidate:

- an admissible partition in all four ProcessBench subset cells and in the
  PRMBench cell;
- five finite starts, at least four converged starts, monotone objective traces,
  the existing multistart agreement pass, full-rank profiled Jacobian, and
  finite primary weights;
- finite token, response, and step scores with complete official ID coverage;
- the same-partition hard-L-SML misfit and score-rank diagnostic reported but
  never used to relax a gate;
- diagonal clipping count and mass reported, never silently discarded.

Any failure produces `STRUCTURAL_NO_SCORE` for that benchmark panel. There is
no post-failure algorithm edit inside the registered namespace.

### Phase 3 -- score freeze

Persist and hash only the minimum evaluation interface:

- ProcessBench: row ID, subset, continuous detector score, locator, and arm ID;
- PRMBench: source/problem ID, step ID, continuous step risk, and arm ID;
- fit ledgers, structural diagnostics, preprocessing parameters, partitions,
  and weights;
- a separate visualization sidecar for row IDs selected deterministically by a
  pre-label ID hash, containing their token-risk curves and outcome-free step
  spans only;
- no benchmark label in the score artifact.

The completion manifest must bind raw input hashes, registry, code, tests,
fit artifacts, score artifacts, plot code, and the independent pre-label audit.

### Phase 4 -- label opening and paired evaluation

Only after Phase 3 passes may the separate evaluator import labels.

- ProcessBench uncertainty: 2,000 paired group bootstraps over official question
  IDs, stratified by subset; the threshold is re-fitted inside each bootstrap
  using the same five-fold rule for every arm.
- PRMBench uncertainty: 2,000 paired bootstraps over complete source/problem IDs,
  keeping all steps from one problem together. This is an explicit override of
  the historical helper's 1,000-draw default.
- All arms and all contrasts within a panel use the same frozen resampled-group
  index stream. PB and PRM use separate registered seeds because their sampling
  units differ.
- The confirmatory family contains exactly two contrasts: Joint L-SML minus
  matched IU-PCR for ProcessBench macro-F1 and for PRMBench step AUROC. Each uses
  a paired two-sided 95% percentile-bootstrap interval. Promotion is an
  intersection-union decision: both component lower bounds must exceed zero and
  both point estimates must reach their panel-specific SESOI. No alpha split is
  required for this conjunction because failure of either component prevents
  rejection of the composite null.
- The four Joint-minus-equal-family and Joint-minus-fixed-family contrasts
  across the two panels receive nominal paired 95% percentile intervals for
  descriptive diagnosis only. They carry no familywise claim and cannot trigger
  promotion.

Provisional engineering SESOIs to freeze after a power-ledger check are
`+0.010` ProcessBench macro-F1 and `+0.005` PRMBench AUROC. The values may be
changed only before registration and before telemetry acquisition, with a
written power rationale.

Decision states:

- `PROMOTE_TO_FRESH_SYSTEM_INTEGRATION`: both primary point improvements reach
  their SESOI and both paired 95% interval lower bounds exceed zero;
- `RETAIN_RESEARCH_CANDIDATE`: neither panel shows clear harm, but the dual
  promotion gate is not met;
- `REJECT_FOR_LOCALIZATION`: either primary paired 95% interval is wholly below
  zero;
- `STRUCTURAL_NO_SCORE`: a pre-label structural or coverage gate fails.

Even `PROMOTE_TO_FRESH_SYSTEM_INTEGRATION` promotes only the local fusion head
to a later system test. It is not a new-leader claim on the historically opened
benchmarks.

## 9. How it would enter the localization system

The clean integration path is deliberately narrow:

`raw token telemetry`
` -> frozen absolute orientation`
` -> frozen active-23 roster`
` -> Joint L-SML token confidence`
` -> token risk`
` -> unchanged task reducer`
` -> unchanged threshold/evaluator`.

For ProcessBench, Stage 1 uses the local-only evaluation boundary: the Joint
curve supplies both the maximum-risk no-error score and the frozen fixed
top-ten-mean locator. The separate Global detector and the complete
Global/Local architecture are not evaluated or modified in this stage. This
keeps a win or loss attributable to the local fusion package. For PRMBench, the
same token-risk curve directly replaces the IU token curve before the existing
maximum-inside-step adapter.

If the dual-panel gate passes, a second, separately registered experiment may
insert the new Local head into the complete Global+Local system and compare it
with the current `family6 + level + step_top5mean` incumbent on another fresh
population. That later experiment must not reuse the present population for
architecture or blend-weight selection.

## 10. Required presentation

The report must be generated from signed JSON/CSV rather than typed numbers.
PB and PRM receive separate panels.

Pre-label plots:

1. retained/removed orientation heatmap with per-cell loading opacity;
2. selected K, group sizes, LOAO ARI, and blocked-cell dashboard;
3. observed covariance, rank-one residual, and Joint fitted residual;
4. five-start objective traces and Jacobian/conditioning diagnostics;
5. Joint versus same-partition hard-L-SML misfit and score Spearman.

Post-label plots:

1. ProcessBench per-subset F1 and paired candidate-minus-control deltas;
2. ProcessBench clean versus erroneous accuracy trade-off;
3. PRMBench AUROC/AUPRC by paper class and grouped paired deltas;
4. deterministic example traces selected by row-ID hash before labels, with
   token risk, official step boundaries, and first-error annotation;
5. a gate dashboard that states the exact terminal decision.

Every plot carries three short captions: Observation, Inference, Limitation.

## 11. Prior-order audit

This plan preserves the explicit project boundaries:

- no new scoring arm is run on the already opened Qwen/Llama caches;
- no label or outcome enters orientation, pruning, grouping, fitting, or score
  construction;
- ProcessBench and PRMBench remain separate;
- one candidate, three baselines, and the confirmatory/secondary contrast rules
  frozen in Section 8 define the shared multiplicity budget;
- no prevalence component, rank transform, DUFS, Katz, graph-normalized SML,
  extra monotonic transform, multi-threshold expansion, LAG, innovation, or
  overlapping factor is reintroduced;
- fixed-family and equal-family methods remain controls, not learned winners;
- the negative V2 result is consumed as a veto, not reinterpreted as a grouping;
- historical opened populations cannot establish promotion or a new leader;
- no commit, push, download, cluster run, or label opening is authorized here.

## 12. Historical reasons for this exact design

- Continuous L-SML previously improved binary L-SML by 4.9 percentage points
  macro, so the candidate remains continuous and never reintroduces
  binarization.
- The historical broad-28 local pool fell to 29.03% ProcessBench F1, so the
  current experiment uses the globally pruned 23-stream roster rather than
  treating feature count as automatically beneficial.
- Exhaustive subset work exposed large label-based selection optimism, so no
  outcome selects streams, K, groups, weight maps, or reducers.
- The opened reducer ladder found fixed top-ten raw-best: ProcessBench macro-F1
  `0.358163` versus `0.345227` for fixed top-five, a delta of `+0.012936` with
  the final closed eleven-contrast simultaneous interval
  `[+0.001444,+0.025225]`. It was not promoted because
  the lower bound missed the preregistered `+0.005` practical-benefit bound.
  The H2/H3 and deployed-U-PCR development ladders subsequently retained this
  top-ten boundary. It is therefore frozen prospectively for every arm in the
  new-scorer test; top-five remains a historical incumbent anchor, not a second
  reducer candidate. Fixed top-ten must not be confused with top-10%-of-step,
  which was harmful.
- Current application evidence therefore makes the trajectory-first token
  curve and fixed-top-ten/maximum-inside-step adapters the correct fixed
  integration boundary.
- Graph-normalized SML, DUFS+L-SML, and the V2 lag program did not produce a
  transferable gain that justifies another scoring arm.

## 13. Authorization still required

Before implementation or execution, the user must separately approve:

1. creation of the new scoring worktree and exact base SHA;
2. the pinned scorer/model revision and telemetry acquisition contract;
3. the final power/multiplicity ledger and SESOIs;
4. cluster execution, if required;
5. label opening after the score-freeze audit passes.
