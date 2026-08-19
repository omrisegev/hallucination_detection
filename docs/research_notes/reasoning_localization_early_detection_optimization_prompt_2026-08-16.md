# Prompt for the next research agent: optimize joint reasoning localization and early detection

Use the text below as the next agent's task prompt.

---

You are continuing a research project on gray-box, single-pass, label-free
hallucination scoring for LLM reasoning traces. Your task is to optimize one
efficient method for two co-primary applications:

1. **Reasoning hallucination localization:** decide whether a reasoning trace
   contains an error and locate the first erroneous token/step.
2. **Early/online final-error detection:** from a causal unfinished prefix,
   estimate whether the completed answer will be wrong and determine when that
   decision is stable enough to declare.

## Single goal

Find the **smallest causal label-free Global-Local-Online IU architecture on
the joint performance/compute Pareto frontier** across both tasks. Improve the
strongest existing method on at least one task under grouped uncertainty,
remain non-inferior on the other under a margin frozen before selection, and
remove every feature, graph, reducer, or head whose incremental value does not
pay for its compute, memory, and redundancy.

Do not optimize a single benchmark in isolation. Every model change must be
evaluated on both localization and early detection. Keep the two result panels
separate; never average them into one leaderboard score.

## First actions and canonical reading order

Before changing code or interpreting results:

1. Read `CLAUDE.md` completely and read the current `PROGRESS.md`.
2. Read the Step-269 application override and current priority order in
   `Research_Directions.md`.
3. Read `SUPERVISED_ORACLE_CORRECTION.md` for leakage, split, and oracle-reporting
   rules.
4. Inspect `git status` and preserve every unrelated or untracked artifact.
5. Read these method/protocol/result artifacts:
   - `spectral_utils/feature_contract.py`
   - `docs/methods/iu_pcr.md`
   - `docs/methods/gl_liu_v1.md`
   - `docs/experiments/EARLY_ONLINE_EXISTING_DATA_V1.md`
   - `docs/experiments/EARLY_ONLINE_LOCALIZATION_MODELS_V1.md`
   - `results/early_online_existing_data_v1/REPORT.md`
   - `results/early_online_localization_models_v1/REPORT.md`
   - `results/ours_only_localization_v1/REPORT.md`
   - `results/gl_liu_factorial_v2/REPORT.md`
   - `results/gl_liu_external_v1/llama31_8b/REPORT.md`, if present
   - `results/fixed_application_pipelines_v1/REPORT.html`
   - `docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md`
   - `docs/research_notes/early_online_hallucination_detection_phase1_checkpoint_2026-08-16.md`
6. Survey the broader experiment history relevant to CUSUM, sliding-window
   variance, spilled energy, feature orientation, non-monotone transforms,
   trajectory-first fusion, GL-LIU, ProcessBench, PRMBench, Phase-15, the
   replication grid, temperature variation, and model/dataset transfer. Use
   `rg` over `HISTORY.md`, `PROGRESS.md`, `Research_Directions.md`, `results/`,
   and `docs/`; do not read thousands of unrelated lines blindly.

Produce a compact evidence inventory before proposing a new method. For every
relevant experiment record: data family, model/generator, temperature,
generation protocol, number of independent questions, telemetry available at
token level, whether causal prefix replay is valid, label exposure status,
selection/confirmation role, method, metric, cost, and conclusion. Distinguish
new data from the same questions rescored by another model.

## Non-negotiable scientific boundaries

### The operational monotonicity assumption

Do not state that every raw feature is intrinsically monotone. The usable
scorer inputs follow the frozen `CONFIDENCE_FEATURE_SIGNS_V1` contract: larger
oriented values mean greater confidence, and risk uses the negative direction.
The four recurrently non-monotone or unstable raw views are quarantined under
the stable contract or replaced by the frozen mixed-v2 transforms. A raw view
and its transformed version may not coexist. Never re-estimate feature signs,
choose transformations, or reverse a score using target labels in an
evaluation cell.

Some causal trajectory summaries, such as a running maximum or accumulated
CUSUM evidence, are monotone in elapsed time. Treat that as useful structure
but also as a possible length confound. Compare raw accumulation with
elapsed-length-normalized versions and a length-only control. Final trace
length is not causal. The primary online contract is the 28-stream no-final-
length adapter; elapsed prefix length is a declared 29th-stream ablation only.

### Label and confirmation boundary

The scorer fit, feature orientation, graph, component count, feature selection,
and online aggregation must be label-free. Labels may be used only for a
declared development selection, threshold calibration, and evaluation split.
Say precisely that this is an **unsupervised scorer with calibrated decision
policies**, not a wholly label-free policy.

Existing original cells, the current ProcessBench cells, PRMBench, the 11-cell
online screen, and the already opened localization suites are retrospective
development assets. They can support architecture selection and falsification,
but not a new confirmation claim. Preserve question/model sharing in splits
and bootstrap units. Do not tune on tokens as if they were independent rows.

### Causality and access

At prefix budget `b`, use only telemetry produced at or before `b`. Rebuild
rolling/spectral features causally rather than slicing a final feature vector.
Add tests showing that two traces with the same prefix and different suffixes
produce identical prefix scores. Never use final answer text, final length,
future step boundaries, or future correctness to score a prefix.

For localization, reasoning-step spans may map a frozen token prediction to a
benchmark label only after scoring. They may not define features unless a
separate step-aware arm is explicitly declared.

### Frozen A6 boundary

The separate A6/PTNI core-method program is frozen. Do not edit, rescue, delay,
reinterpret, or use this application program to select its stages. If A6 work
is requested separately, follow its existing preregistration exactly.

## Current evidence you must reproduce before extending

Treat the following as regression anchors, not as fresh confirmation:

- Existing-cache early screen: 11 cells, five dataset families, no new
  inference. IU28 AUROC is 0.648 at 64 tokens and 0.694 at 128; DeepConf
  entropy-w64 proxy is 0.616 and 0.671. Equal-family IU-minus-DeepConf deltas
  are +0.024 [-0.005,+0.056] and +0.014 [-0.031,+0.058]. This is promising
  parity, not established superiority.
- IU28 convergence: prefix/final Spearman is 0.417/0.659/0.817 at
  64/128/512; final-decision agreement is 0.640/0.739/0.880.
- Early declarations are not yet reliable enough: IU28 coverage 0.366,
  ever-wrong 0.137, only 5/11 cells at or below the 10% target. `sw_var_peak`
  has coverage 0.348, ever-wrong 0.121, and 6/11 cells passing.
- Causal localization-model screen: at 64 tokens global/fused/`sw_var_peak`/
  IU28/DeepConf-w64 AUROC is 0.638/0.635/0.643/0.648/0.616; at 128 it is
  0.679/0.678/0.679/0.694/0.671. The completed-trace CUSUM+`sw_var` arm
  reaches 0.798. At 512, fused Global-Local beats IU28 by equal-family +0.066
  [+0.043,+0.089].
- GL-LIU v1 reaches 31.36% ProcessBench F1 versus 25.71% for the reproduced
  Mind the Gap control. The later trajectory-first IU package reaches 30.70%
  across eight cells and 30.35% versus 24.96% on matched Qwen3-8B. PRMBench
  step AUROC is 0.6711 versus 0.6136 for the older step-first adapter.
- The Laplacian contribution is small or absent in matched existing
  localization components: global ordinary mixed IU 0.791369 versus DUFS-LIU
  0.793561 (+0.002193); local ordinary top-five IU 0.723303 versus DUFS
  0.723881 (+0.000578). Temporal Laplacian transfer is worse. Do not attribute
  the application strength to the Laplacian without a clean causal ablation.

If your reproduction differs, stop and diagnose data membership, orientation,
grouping, split hashes, and causal prefix construction before adding a model.

## Task definitions: keep four lanes distinct

1. **Prefix correctness detection:** rank whether the eventual completed
   answer will be wrong from an unfinished prefix. This is the primary early
   ranking task.
2. **First-error localization:** place the onset of the first reasoning error
   while also abstaining on clean traces. This is the primary localization
   task.
3. **Single-trace declaration/stopping:** decide when the prefix verdict is
   reliable enough to expose or stop. This is a calibrated policy layered on
   the unsupervised score.
4. **Multi-trace adaptive compute:** allocate samples or reasoning budget
   across traces/questions. Keep this outside the primary architecture claim;
   compare later with REFRAIN/DeepConf only after lanes 1–3 are strong.

Do not use performance in one lane as if it directly proves another.

## Working architecture hypothesis

Start from the simplest shared causal model:

```text
frozen confidence-aligned token telemetry
    -> causal rolling/spectral feature state
    -> ordinary IU-PCR shared representation (lambda = 0)
       -> global error-presence head
       -> local first-onset/token-risk head
       -> online convergence/declaration head
```

The primary baseline is **Global-Local IU-PCR without a Laplacian**. Frozen
GL-LIU, DUFS feature-graph IU, uniform-graph IU, and temporal-graph IU are
controls. Evidence may overturn this baseline, but complexity receives no
presumption of value.

The strongest next hypothesis is that early information is in the *evolution*
of CUSUM and `sw_var`, not just their final or running maximum. Candidate
causal summaries include:

- current oriented value and running maximum;
- EWMA at a small predeclared set of time constants;
- area above an unlabeled baseline, normalized by elapsed prefix length;
- recent slope and slope stability;
- persistence/run length above an unlabeled reference level;
- CUSUM magnitude, onset/location normalized by elapsed time, and recovery;
- change-point magnitude and time since change;
- dispersion or disagreement between global and local risk;
- convergence rate and stability of the predicted final score/decision.

Prefer online recurrences with `O(1)` or amortized `O(1)` work per new token.
Avoid multiple highly correlated encodings of the same accumulator. Include
drop-one, add-one, and correlation/redundancy diagnostics; if two variants are
statistically tied, keep the cheaper and more interpretable one.

## Required experimental program

### Phase 0 — Audit and preregister

1. Build the broad cache/evidence inventory described above, including
   temperatures and generation protocols not used by the initial 11-cell
   screen.
2. Mark caches as `causal-prefix-valid`, `localization-only`, `final-only`, or
   `unusable`, with a reason.
3. Freeze the development/validation roles, grouping keys, primary metrics,
   candidate roster, compute measurements, and non-inferiority margins before
   reading candidate performance. Derive margins from prior repeated-split or
   bootstrap variability, not from the candidate result.
4. Write a protocol under `docs/experiments/` before implementing a search.

### Phase 1 — Build one cross-task regression harness

1. Reproduce the anchor numbers above from canonical artifacts.
2. Expose a single method interface that returns:
   - whole-trace/global risk;
   - token-level/local risk and first-onset prediction;
   - prefix score trajectory;
   - calibrated declaration trajectory;
   - runtime, peak memory, feature count, and state size.
3. Run the same candidate identity on both the localization and early panels.
4. Add deterministic tests for suffix invariance, feature-order invariance,
   no-label fitting, repeated-run identity, missing-feature handling, and
   exact `lambda=0` reproduction.

### Phase 2 — Clean architecture ablations

Use the same feature matrix, normalization, IU subspace, reducer, questions,
splits, and calibration for all arms:

1. ordinary IU-PCR (`lambda=0`);
2. uniform graph;
3. DUFS feature graph;
4. temporal graph only for token/trajectory heads;
5. global-only, local-only, naive fixed fusion, and learned-but-nested fusion;
6. no-final-length primary versus elapsed-prefix-length ablation;
7. core-five token curves versus the registered shared causal stream set;
8. CUSUM only, `sw_var` only, fixed equal combination, and dynamic summaries.

Do not compare IU28 with GL-LIU and call it a Laplacian ablation; their feature
contracts and heads differ. Do not search graph hyperparameters on the same
labels later used to claim transfer.

### Phase 3 — Dynamic causal optimization

Start with small, mechanism-motivated candidate sets. Use nested question-level
selection inside development data. Evaluate whether magnitude, persistence,
slope, onset/change point, and convergence stability add information beyond
current maxima. Inspect per-family gains and failures, especially short traces,
long traces, temperature shifts, model-family shifts, and scorer/generator
mismatch.

Use supervised logistic or nonlinear models only as clearly labeled diagnostic
ceilings. They may reveal missing target information or interaction capacity,
but cannot choose or orient the deployable unsupervised scorer.

### Phase 4 — Pareto selection and cross-task gate

For every candidate report:

- localization metrics and paired deltas;
- early ranking, convergence, and declaration metrics and paired deltas;
- equal-family and per-family estimates with grouped confidence intervals;
- coverage and failure modes by token budget;
- feature count, fit/update complexity, wall time, peak memory, and saved state;
- add-one/drop-one incremental value and redundancy;
- sensitivity to missing streams and trace length.

Promote a candidate only if it has grouped evidence of improvement on at least
one co-primary task, stays within the preregistered non-inferiority margin on
the other, and lies on the empirical performance/compute Pareto frontier.
Within uncertainty, choose the simpler method. A late-only gain does not imply
an early-declaration gain; a better locator does not imply a better global
detector.

### Phase 5 — Fresh confirmation and exact paper benchmarking

Only after the existing-cache program identifies a promising frozen
architecture, propose a new-data/GPU gate. Prefer a new dataset family and a
new model/generator family with native token telemetry. Freeze score identity,
threshold protocol, and grouping before labels open.

Then compare access-matched methods:

- **Localization:** Mind the Gap as the measured label-free peer; ordinary
  entropy and IU baselines; uPRM only if its real training procedure is run and
  never conflated with the cheap LLM-as-a-Judge reconstruction; supervised PRM
  and large critic models as separate ceilings.
- **Early/online:** DeepConf official processor/protocol after raw-logit
  equivalence is verified; the current w32/w64 entropy arms remain proxies, not
  an exact reproduction. REFRAIN is a stopping/adaptive-compute comparison,
  with unreleased constants/code recorded as fidelity limits. Streaming
  Hallucination Detection is a supervised ceiling if its assets become
  available.

Do not spend GPU time merely to copy a paper table before the existing-cache
architecture is convincing. Any new inference, cluster job, Drive write, or
large download requires a separate explicit approval and a verified source,
destination, cost, and manifest.

## Primary metrics

### Localization panel

- global answer-error AUROC and AUPRC;
- exact first-error token/step localization;
- SLA within one reasoning step;
- clean-trace abstention/accuracy;
- ProcessBench F1;
- PRMBench every-step AUROC/F1 as a separate task, not averaged with
  ProcessBench.

### Early panel

- equal-family AUROC and AUPRC at 16/32/64/128/256/512 tokens, restricted to
  at-risk unfinished traces;
- Spearman correlation with the method's own completed-trace score;
- agreement with its own final binary decision;
- declaration coverage at a frozen held-out ever-wrong target;
- ever-wrong rate over the full monitoring horizon, not per-time false-alarm
  rate;
- selective error and realized token savings only when actual forced-closure
  branches are executed.

Always show eligible cell/question counts at each budget. A 512-token macro is
not directly comparable with a 64-token macro if the eligible family set has
changed.

## Deliverables

At the end of the first research cycle, produce:

1. a cache/evidence inventory and independence map;
2. a frozen experimental protocol;
3. a reusable two-panel regression harness with causal/no-label tests;
4. a candidate ledger recording hypothesis, incremental cost, result, and
   keep/drop decision;
5. machine-readable per-question/per-cell results and grouped intervals;
6. one report with separate localization, early ranking, convergence,
   declaration, and efficiency panels;
7. an explicit decision: promote the simplest Pareto-optimal candidate, retain
   the baseline, or close a mechanism with reasons;
8. updates to `HISTORY.md`, `PROGRESS.md`, and `Research_Directions.md`.

Do not stage, commit, push, mutate Google Drive, or launch inference unless the
user explicitly asks. Begin with read-only inventory and CPU-only reuse of
existing caches. Report blockers rather than weakening causality, label, or
confirmation boundaries.

---

The intended research contribution is not “we added another graph.” It is a
compact unsupervised trajectory method whose shared dynamic evidence supports
both **where reasoning first goes wrong** and **how early the final failure can
be known**, with measurable efficiency and honest access-matched comparisons.
