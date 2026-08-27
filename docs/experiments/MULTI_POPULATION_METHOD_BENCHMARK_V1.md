# Multi-population method benchmark v1

**Status:** inclusive design draft; no experiment is authorized by this file.

## 1. Why this benchmark has several leaderboards

We have one broad question: which of our methods works, where, and under what
access assumptions? It cannot be answered by one average. The collected data
contain different prediction tasks:

- one score for a complete answer;
- the first wrong reasoning step;
- prediction from a causal prefix;
- an actual stop/continue policy;
- unsupported RAG text at answer, sentence, token, or claim level; and
- white-box scores that require internal model states.

These tasks have different units and metrics. We therefore use one method
registry and one population registry, but a separate leaderboard for every
lane. We may later summarize whether a method improves over a matched IU-PCR
control across panels, but we will not average response AUROC, localization F1,
prefix AUROC, and token savings into one number.

The machine-readable inputs to this design are:

- `configs/multi_population_benchmark_v1_method_registry.csv`;
- `configs/multi_population_benchmark_v1_population_registry.csv`; and
- `configs/multi_population_benchmark_v1_compatibility.csv`.

The compatibility CSV is currently a method-group by lane scaffold. Before a
run it must be expanded to one row for every selected method-population pair,
using `configs/multi_population_benchmark_v1_pair_registry_schema.csv`.

All lists are deliberately broad. An uncertain candidate remains in the
registry until the pre-run review decides whether it is a main arm, an
ablation, a ceiling, or a historical control.

## 2. Method groups

### 2.1 Static response fusion

The main 24-cell comparison should include the linear covariance methods
(U-PCR, U-PCR with estimated polarity, and IU-PCR), DUFS-LIU, SU-PCR,
balanced-atomic CA-SpecRaGE, continuous additive DEEM-B3, and the new
within-cell Family-NRM and PGRD variants. Continuous L-SML and the earlier
DUFS/GroupFS selector pipelines remain in the roster so that the new benchmark
does not erase the feature-selection path that led here.

Family-NRM and PGRD use the same three selection regimes:

- **A — within-cell unsupervised:** target cell only, no donors or labels;
- **B — donor-unsupervised:** target-free donor data, no labels; and
- **C — donor-label selection:** donor labels may select the rule while the
  target dataset family is held out.

Only A belongs in the main label-free 24-cell leaderboard. B is a secondary
transfer/stability ablation and C is a supervised-selection ceiling. The
existing donor-free A versions are new, unrun ablations; they must not be
confused with the completed historical Family-NRM/PGRD implementations.

DEEM-B3 is also kept distinct from Residual-Graph DEEM. DEEM-B3 completed all
24 cells on its own full-present feature contract and passed its registered
noninferiority decision. Residual-Graph DEEM stopped at a synthetic gate and
has no natural-data leaderboard result.

### 2.2 Task adapters and application methods

A response scorer is not automatically a localizer or an early detector. A
new method enters another lane only after the adapter is fixed. Examples are
the token-to-step aggregation and no-error threshold for localization, the
causal feature set at every prefix budget, and the evidence-removal matrix for
RAG. Stopping is stricter still: it requires a real generation policy and
cannot be inferred from cached AUROC.

The application roster therefore also retains the dedicated six-family
ProcessBench localizer, GL-LIU, the Step-272 global/local prefix head, the
evidence-contrast RAG pipelines, Unified-28, white-box depth fusion, and LEASH.
Published methods appear as access- and fidelity-labelled comparators, not as
if all of them ran on the same rows.

## 3. Benchmark lanes

### R0 — Frozen 24-cell response detection

- **Population:** 48,607 answers in 24 dataset-model cells (9 QA and 15 math).
- **Primary metric:** equal-cell macro AUROC; one cell is one dataset-model
  pair.
- **Required views:** a 24-by-method table, per-cell heatmap, paired deltas to
  the exact matched IU-PCR arm, QA/math and dataset-family summaries, AUPRC,
  runtime, and method disagreement.
- **Status:** retrospective development evidence. It can select a candidate
  for later validation, but it is not a fresh generalization test.

The detailed method contract is in
`docs/experiments/GLOBAL_24CELL_METHOD_BENCHMARK_V2.md`. DEEM-B3, IU/DUFS/CA,
and older selector scores cannot be ranked until rows, feature inventory,
orientation, IU implementation, and macro are identical.

Seven cells contain ten generations for each source question. Their point
estimates remain usable, but a row bootstrap would treat correlated
generations as independent. Recover source-question IDs from the raw artifacts
and group per-cell uncertainty by question. Until then, report the registered
cell point estimates and cell/dataset-family-blocked comparisons only.

### R1 — External response transfer and stress

Use separate panels rather than one pooled macro:

- ProcessBench complete-trace correctness, using the same 3,400 solutions
  scored through Qwen3-4B, Qwen3-8B, and Llama-3.1-8B telemetry;
- SemGrad SciQ and TruthfulQA on new Qwen responses;
- PRMBench response correctness on 6,966 valid traces;
- HLE on 2,158 Qwen2.5-72B answers;
- four Evidence-Drop answer cells: Qwen3-4B/Qwen3-8B on GSM8K and the full
  MATH test (5,638 responses), after the raw LFS artifacts are retrieved and a
  common feature matrix is rebuilt;
- the central CoT answer from each of three AQuA model runs; and
- CoQA only as a quarantined appendix because the old base-model prompt was
  malformed.

These labels have already been inspected during development. They test
retrospective transfer and reveal counterexamples; they are not an unopened
confirmation set. HLE must show class counts and AUPRC because only 68 answers
are currently judged correct, and its labels come from an interim judge.

Report two modes separately wherever they exist: target-cell unsupervised fit
and frozen donor transfer. Do not call the latter necessary for Family-NRM or
PGRD.

### C0 — Published response comparisons

Use the accepted paper-aligned package as the comparison ledger. There are
three distinct evidence levels:

1. **exact local rows:** methods rerun on identical answer IDs;
2. **same protocol/model but different generated answers:** useful external
   anchor, not a paired comparison; and
3. **same dataset only or different access:** context only.

The published catalog includes EPR/WEPR/HalluDetect/SelfCheckGPT, semantic
entropy and energy methods, AttentionScore/LapEigvals, LOS-Net, ARS/CCS,
INSIDE, and internal-state methods. Every row must show its access level,
number of generations or scorer passes, supervision, and fidelity. Never
construct a 24-cell published macro from unmatched paper numbers.

### L0 — First-error localization

Keep two ProcessBench panels:

- the identity-proven Llama-3.1-8B population: 3,400 solutions in four
  subsets; and
- the historical Qwen3-4B/Qwen3-8B panel: eight scorer-subset cells over the
  same 3,400 questions.

The primary metric is the official ProcessBench macro-F1, with first-error
accuracy, clean-trace abstention, and within-one-step accuracy as supporting
metrics. Because scorer models reuse the same questions, uncertainty is
clustered by source question rather than treating every scorer-row as
independent.

PRMBench every-step evaluation is a second localization table: 6,966 valid
solutions and 83,280 evaluated steps after the three registered alignment
exclusions. Its step AUROC/AUPRC is not combined with ProcessBench F1 or with
the PRMBench response-level result.

Response-level Family-NRM, PGRD, and DEEM need a prespecified token/step
adapter before they can enter either localization table. Until then their
status is `ADAPTER_NEEDED`, not a missing score or a loss.

### P0 — Causal prefix / early detection

The fair population is the Llama ProcessBench evaluation half: 1,717 traces
across four subsets, evaluated at 16, 32, 64, 128, 256, and 512 tokens. The
primary summary is mean AUROC at 64 and 128 tokens; report AUPRC, eligible N,
recovery toward the final-trace score, and warning coverage at calibrated false
positive rates at every budget.

Only information available by the prefix boundary is allowed. A method that
uses final length, a future token, or a full-trace normalization is ineligible.
The broader 12 scorer-subset panel remains a retrospective architecture study,
not twelve independent question sets.

### S0 — Adaptive stopping

The current stopping population is AQuA and GSM8K with Qwen, Llama, and Phi:
six model-dataset cells. Report pass@1, realized total tokens including forced
closure, latency, parser failures, and the pass@1-versus-token Pareto frontier.

LEASH is a partial reproduction and remains the current policy comparator. A
static detector can enter this lane only after a threshold, closure rule,
calibration set, and compute budget are frozen and generation is rerun. Prefix
AUROC alone is not stopping performance.

### G0 — Retrieval-grounded detection and localization

RAGTruth must produce separate answer, sentence, and token tables. The same
2,700 responses generate repeated evidence conditions, so uncertainty is
grouped by source/response. RefChecker is a separate fixed-claim checking
table; its official three-way score and our binary unsupported-claim AUROC are
different columns. The GASP 400-response cohort is a protocol-level
reproduction, not an independent subset or paper-exact sample.

The evidence-contrast methods use full-context, no-context, and leave-one-
evidence-out teacher-forced passes. They therefore have a different access and
cost contract from the one-pass 24-cell methods. Applying DEEM, Family-NRM, or
PGRD here requires a frozen evidence-condition adapter.

### W0 — White-box access

Compare white-box depth fusion and gray-box output-probability methods only on
the 31,440 exact-common answers from 13 dataset-model cells. Show AUROC and
AUPRC beside a separate coverage table (42,238 scorable white-box rows versus
31,467 gray-box rows). The existing method roster and hybrid were chosen
retrospectively; neither is validated on untouched data. CoQA remains an
invalid appendix.

### M0 — Repeated generations

Retain the five-generation MATH-500/Qwen2.5-Math-7B experiment as its own
compute tier. It contains 200 common questions and shows a large historical
same-temperature fusion gain, but only one cell. It is a promising replication
target, not evidence that the single-pass method universally improves. Do not
confuse repeated generations with block-resampling one saved token trace.

### X0 — Negative and limitation panels

GPQA and the old LCiteEval-style RAG collection remain explicit negative
stress tests: GPQA results were weak and unstable across models, and only HotpotQA carried stable
signal in the older 20-cell/4,400-response RAG collection. The IU-HMM localizer and moving-block
reliability study are mechanism negatives on existing ProcessBench data, not
new populations. Reporting these panels prevents a positive-only selection of
datasets without corrupting the main leaderboards.

## 4. Generalization claims

Use three evidence labels in every result:

- **D0 — reused development:** the frozen 24 cells and any population used to
  design the feature or method;
- **D1 — retrospective transfer/stress:** a different population whose labels
  were already opened before the current comparison; and
- **D2 — sealed confirmation:** method, adapter, row rules, and score files
  frozen before labels are opened.

There is currently no D2 population for the new full roster. A future sealed
dataset/model-family test is required for a confirmation claim. Existing D1
panels are still valuable: they show which gains transfer, which reverse, and
which require task-specific adapters.

## 5. Shared validity rules

1. Freeze row IDs/hashes, label source, feature contract, target polarity,
   orientation, access tier, and method configuration before evaluation.
2. Fit label-free scores from label-free bundles. Join target labels only after
   score hashes are written.
3. Use correctness-positive AUPRC in response tables unless a metric is
   explicitly named hallucination-positive. Never compare opposite AUPRC
   conventions.
4. Bootstrap the independent source: question for ProcessBench/AQuA, source or
   response for RAGTruth, problem for PRMBench, and dataset/model cell for the
   24-cell macro.
5. Keep full-feature, fixed-stable, 16-feature harmonized, causal-prefix,
   evidence-condition, and white-box contracts in different columns or
   panels. A score from one contract cannot silently enter another ranking.
6. Threshold-based F1 needs a disjoint or cross-fitted calibration split.
   Ranking metrics remain primary for a label-free score.
7. Expand the group-level compatibility scaffold and record `DIRECT`,
   `REBUILD`, `ADAPTER_NEEDED`, `SEPARATE_ACCESS`, `CONTEXT_ONLY`,
   `QUARANTINED`, or `INELIGIBLE` for every selected method-population pair. A
   blank cell is not allowed in the executable registry.

## 6. Required outputs

1. A master applicability matrix and a data/claim ledger.
2. A 24-cell leaderboard, per-cell heatmap, paired-delta forest plot, and
   method-disagreement/graph-diagnostic appendix.
3. Separate external-response tables, including worst-panel change and sign
   of the matched delta to IU-PCR.
4. ProcessBench and PRMBench localization tables, with per-subset/error-family
   breakdowns.
5. Prefix budget curves and warning trade-offs.
6. A stopping Pareto plot rather than an AUROC substitution.
7. Separate RAG answer/sentence/token/claim tables.
8. White-box exact-row accuracy and coverage tables.
9. A published-comparator table with fidelity and access labels.
10. A short failure-analysis package linking method disagreement to feature
    families, graph edges/roughness, trace length, class balance, and dataset
    family. These diagnostics explain results; they do not choose a winner
    after labels are opened.

## 7. Decisions before any run

The next review should decide only the contracts, not the winner:

1. the primary 24-cell feature contract and a separate sensitivity contract;
2. the final inclusive method rows in each lane;
3. which response methods receive new localization, prefix, and RAG adapters;
4. the paired uncertainty and promotion/noninferiority rules; and
5. whether to reserve compute for one genuinely sealed D2 validation after the
   retrospective benchmark is complete.

No evaluation or inference should start until these decisions are frozen and
Omri authorizes the run.
