# Handoff: Paper-Exact Localization and Early/Online Comparison

**Date:** 2026-08-16

**Audience:** Claude on the AIRCC-connected machine

**Status:** implementation and execution plan; no new paper-reproduction result is claimed here
**Goal:** acquire once on the cluster, then compute every fair comparison offline

## 0. Read this first

Before implementation or submission:

1. Read `CLAUDE.md` and `PROGRESS.md` completely.
2. Read `docs/research_notes/early_online_hallucination_detection_phase1_checkpoint_2026-08-16.md`.
3. Read `papers/PAPER_EXACT_SOURCES.md`, then the seven relevant cards under
   `papers/digests/`; the exact PDFs and extracted text are committed under `papers/`.
4. Read the Step-269--273 code and contracts listed in §2. Do not recreate them from memory.
5. Inspect `git status`; preserve unrelated work. Record the checked-out commit in every run.
6. Verify TAU VPN and `ssh aircc`. A connectivity check on 2026-08-16 at about 23:08 failed
   DNS resolution, so do not interpret a missing Slurm job as a code failure.

The independent protocol audit has already been applied to this plan. In particular:

- opened historical caches are retrospective evaluation assets, not locked confirmation;
- the causal primary is the 28-stream no-final-length adapter, not the literal historical
  29-stream contract whose final-length stream leaks the future;
- a repeatedly inspected alarm is calibrated on the maximum over its whole monitoring horizon;
- REFRAIN carries bandit state across questions;
- no cache is called exact DeepConf until its raw-logit calculation matches a pinned official
  implementation;
- Streaming Hallucination Detection is blocked on official assets, not reconstructed by guesses.

## 1. Non-negotiable scientific structure

There are four result lanes. Never rank numbers from different lanes in one leaderboard.

| Lane | Question | Native outputs |
|---|---|---|
| Localization | Where is the first erroneous reasoning step, or is the trace clean? | error accuracy, clean accuracy, ProcessBench F1; Mind-the-Gap SLA separately |
| Prefix detection | From tokens available now, will this one trace finish wrong? | AUROC/AUPRC at absolute budgets, causal alarm performance |
| Single-trace stopping | Should this trace stop reasoning and force an answer now? | pass@1, generated tokens, latency, accuracy-compute frontier |
| Multi-trace adaptive compute | Which traces should finish/vote, and should sampling continue? | vote accuracy versus total sampled tokens |

For every row assign one fidelity label:

- `official-exact`: official data, checkpoint, code commit, prompt, decoding, parser, and metric;
- `paper-specified`: implemented from a sufficiently detailed paper when runnable code is absent;
- `paper-specified-partial`: declared choices or sensitivity grid fill omissions in the paper;
- `adapted-common-protocol`: concept applied to our shared rows/closure/task;
- `published-context-only`: number quoted, not rerun;
- `blocked-assets`: required official data/checkpoint/split is unavailable.

Published values are regression targets, never data-quality promotion gates. Smoke-to-pilot-to-full
promotion depends only on schema, hashes, causality, parser coverage, determinism, checkpoint/resume,
and resource safety—not on whether a method wins.

## 2. Existing work that must travel with this handoff

The Step-269--273 files are relevant and are part of the implementation context. They define the
current feature inventory, causal models, comprehensive ablations, and transfer decision. Read:

- experiment contracts:
  `docs/experiments/EARLY_ONLINE_EXISTING_DATA_V1.md`,
  `docs/experiments/EARLY_ONLINE_LOCALIZATION_MODELS_V1.md`,
  `docs/experiments/GLOBAL_LOCAL_ONLINE_IU_V1.md`,
  `docs/experiments/GLOBAL_LOCAL_ONLINE_ARCHITECTURE_V2.md`, and
  `docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.md`;
- current summaries:
  `results/early_online_existing_data_v1/REPORT.md`,
  `results/early_online_localization_models_v1/REPORT.md`,
  `results/global_local_online_iu_v1/REPORT.md`,
  `results/global_local_online_architecture_v2/REPORT.md`, and
  `results/local_online_comprehensive_v1/REPORT.md`;
- implementations:
  `spectral_utils/global_local_online.py`,
  `spectral_utils/local_online_comprehensive.py`,
  `spectral_utils/comprehensive_fusion.py`,
  `spectral_utils/multitask_trajectory.py`,
  `spectral_utils/online_convergence.py`, and
  `spectral_utils/online_localization_fusion.py`;
- the matching `run_*`, `summarize_*`, `finalize_*`, and `test_*` scripts under `scripts/`.

Git contains compact reports, aggregates, decisions, diagnostics, and manifests. Per-question tables,
pickled checkpoints, raw telemetry, dataset caches, and `.beagle-retrieval-diag` are deliberately not
transported through Git. Locate large artifacts through their committed inventories/manifests and
`gdrive:hallucination_detection/`; never regenerate a large cache merely because it is absent locally.

## 3. One acquisition contract for every new model run

Implement and validate schema `paper_exact_acquisition_v1` before a full submission. Cluster work is
limited to model-dependent operations: generation, teacher forcing, hidden-state extraction, forced
closure, and unavoidable competitor training. Feature construction, calibration, bootstraps, resampling,
plots, and reports run offline on CPU from the saved acquisition.

### 3.1 Run manifest

Every run directory must contain immutable `RUN_MANIFEST.json` with:

- run ID, UTC creation time, repository commit, dirty-tree flag, container image digest;
- paper title, committed PDF path and SHA-256, fidelity label, official-code URL and commit;
- dataset source/revision/hash, ordered example IDs and an order hash;
- model/tokenizer/checkpoint revisions, chat-template hash, prompt text/hash;
- decoding configuration, seed policy, maximum length, stopping/EOS behavior;
- exact signal definitions, raw-versus-post-warper status, selected hidden-state layers;
- expected/finished/failed trace counts, shard index, software/GPU versions;
- native metric/parser revision and all declared deviations from the paper.

Do not fetch a floating branch, dataset, model, or PDF inside a full job. Resolve and pin it in a
prefetch/audit job first.

### 3.2 Per-question and per-trace record

Preserve at least:

- stable question/source ID, gold answer, model prompt text and token IDs;
- generated token IDs, decoded text, answer parse, correctness, token count, timing;
- token entropy over the full vocabulary, log-sum-exp, sampled-token probability/logprob,
  spilled energy, pmax, and top-two logprob margin;
- exact DeepConf top-k confidence produced by the pinned function;
- raw and post-warper top-50 IDs/logprobs in smoke/audit shards and wherever the native method
  requires them; never label post-warper telemetry as raw;
- blank-line and sentence step spans, with the segmentation implementation/hash;
- every natural stop, paper stop, forced-closure point, and closure output as separate fields;
- labels plus provenance: benchmark human label, semantic parser, external judge, or trained probe.

For the full DeepConf pool, retain the native confidence scalar and the four frozen project channels
for every token; full raw top-50 arrays may be limited to a deterministic audit sample after equality
to the pinned official function is proved. This keeps the reusable full pool near tens of GB rather
than 0.6--1.2 TB. If future development genuinely needs full distributions, create a separately
approved `rich_top50` acquisition—do not silently expand the run.

For Streaming, store only official selected-layer step summaries unless the official code requires
token-level states. At 200k steps, one fp16 4096-d summary is about 1.6 GB before metadata; a
0.1--0.5 TB estimate applies to token/all-layer states, not step summaries.

### 3.3 Sharding, checkpoints, and transfer

- Shard at no more than 64 traces or 1 GB, whichever comes first.
- Write a shard atomically, then update `INDEX.jsonl`, `STATUS.json`, and SHA-256 checksums.
- Catch SIGTERM, flush the active shard, and resume by stable trace key without duplicates.
- Use `$SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1`, the NGC
  `nvcr.io/nvidia/pytorch:25.01-py3` image, and the repository's preemption pattern.
- Large results move directly between `$SHARED/results/` and
  `gdrive:hallucination_detection/cluster_results/paper_exact_v1/` with manifests; do not route
  them through the laptop.
- Before any projected retained footprint above 8 TB, stop and verify Drive and shared-disk quotas.

## 4. Required jobs, in execution order

Implement each job with deterministic `--smoke`, `--pilot`, `--full`, `--resume`, and
`--report-only` modes. Smoke uses 2-5 questions and a short cap. Pilot validates at least one clean
and one error/incorrect case. Full begins automatically only after the integrity checklist passes.

### P0 — environment, assets, and frozen-code audit (no GPU or tiny sandbox)

1. Verify VPN/SSH, Slurm partitions/QoS, shared/Drive space, container, and preemption/resume.
2. Verify all seven committed PDF hashes against `papers/PAPER_EXACT_SOURCES.md`.
3. Clone official repositories into a read-only source cache, pin commits, and record licenses:
   ProcessBench, Mind the Gap, DeepConf, and REFRAIN. Do not treat REFRAIN's placeholder README
   as runnable official code.
4. Re-audit the anonymous Streaming endpoint for data, splits, prompts, probe checkpoints, and code.
   If any are absent, emit `BLOCKED_ASSETS.json`; do not book generation or Claude-labeling compute.
5. Freeze a single evaluator library for exact answer parsing, ProcessBench F1, SLA, AUROC/AUPRC,
   pass@1, token count, and grouped bootstrap.

### P1 — regression and causal integrity (CPU only)

Run the existing Step-269--273 regression suite plus:

- identical prefixes with arbitrary different suffixes produce identical prefix scores;
- tokenwise and chunked causal replay agree;
- the primary excludes final response length; elapsed prefix length is a declared ablation only;
- equal-problem/equal-trace/equal-budget fitting prevents long traces dominating;
- alarm thresholds are calibrated on `max_t score(t)` over the full registered horizon;
- `AUROC95` uses `0.5 + 0.95*(AUROC_full-0.5)`, not `0.95*AUROC_full`.

Historical `lsml16` and opened rich caches are regression anchors/retrospective reanalyses only.
Reserve a genuinely untouched model/dataset cell for confirmation after score hashes are frozen.

### L0 — shared ProcessBench table from existing artifacts (CPU)

Rebuild one 3,400-row-by-method table with stable IDs and the official earliest-step-or-`-1`
evaluator. Include our frozen family-six/step-top-five locator, max-entropy+top-5, Mind the Gap,
existing official PRM and critic scores, and every available published/checkpoint baseline. Emit
error accuracy, clean accuracy, harmonic F1 per subset, macro F1, paired question bootstrap, access
tier, and fidelity. Keep Mind-the-Gap native SLA in a second panel because SLA excludes clean traces.

Expected regression anchors, not acceptance gates: ours 0.3662 macro F1; max entropy 0.3614;
shared-protocol Mind the Gap 0.2646; Qwen2.5-Math PRM ceiling 0.7280; critic ceiling 0.5895.

### L1 — uPRM paper's exact cheap control (Qwen2.5-14B, full ProcessBench)

Implement uPRM Eq. 6 independent `LLM-as-a-Judge`, not the trained uPRM, using
Qwen2.5-14B-Instruct and all 3,400 official rows. Construct the candidate marked sequences exactly
as the paper card specifies, teacher-force marker probabilities, and predict the first error/clean
class. Smoke each marker-token/tokenizer edge case, then run all four subsets. This is the first
overnight localization GPU priority because it creates a fair same-backbone inference-only control.

Compare with the paper's 49.8/42.8/29.4/26.6 F1 targets and label deviations. Do not rename this
row uPRM.

### L2 — ProcessBench official ceilings and Mind-the-Gap native reproduction

On the same 3,400 ordered rows:

- run official released PRM checkpoints with the official scorer/evaluator;
- run the selected official critic checkpoint/prompt with the paper's eight-sample majority vote;
- run Mind the Gap under its native Qwen3 teacher-forced protocol and native SLA;
- score our frozen locator and max entropy on the identical telemetry rows.

Prefer released checkpoints. Retraining Qwen2.5-Math-7B-PRM800K is conditional because the PDF
does not fully specify optimization; if retrained, call it paper-specified and preserve contamination
removal. Do not compare high-access PRMs/critics as though they were label-free single-pass methods.

### L3 — full trained uPRM (conditional, separate project-size job)

Only after L1 is correct, implement the paper's joint in-context scoring, LoRA, custom RL estimator,
packing, and degenerate-solution correction. Run unit tests against every appendix equation, then
train on PRM800K text with Qwen2.5-14B and evaluate ProcessBench. The paper reports about 44 H200
GPU-hours. With no audited official code/checkpoint and some implementation degrees of freedom,
the honest ceiling is `paper-specified`, not guaranteed `official-exact`.

### S1 — REFRAIN Qwen3-8B/MATH-500 (primary single-trace stopping)

Implement Appendix algorithms and trigger vocabulary exactly. Freeze P0, official thinking mode,
temperature 0.6, top-p 0.95, top-k 20, 16,384-token cap, seed 42, SBERT
`all-MiniLM-L6-v2`, threshold grid 0.60--0.80 by 0.05, `C=1`, `W=100`, `lambda=0.2`, and
`0.0001*L` first-round penalty. Freeze MATH-500 order, bandit reset scope, arm initialization,
ties, running-length update, reward timing, tokenizer/chat template, and native forced closure.

Run vanilla and REFRAIN. A 30-question pilot is an implementation check only. Then run all 500.
Published targets: 91.40% / 2.64M tokens vanilla and 91.20% / 1.61M REFRAIN. Also acquire our
four channels so our frozen causal method can be evaluated on the same traces; its forced-closure
policy is a separate adapted-common-protocol experiment.

### S2 — LEASH native matrix (secondary single-trace sensitivity)

Implement the paper's full-vocabulary entropy slope, margin plateau, saturation exclusion,
entropy-drop gate, majority vote, disabled rationale EOS, and second-stage answer. Reproduce native
GSM8K-300 and AQuA-RAT test for the four published models if capacity permits. Freeze the disclosed
values `k=8`, `L=5`, `epsilon_H=.005`, `delta_M=.05`, `m=64`, `M=320`, temperature 0.7,
top-p .95, and greedy closure.

The paper omits `B`, `tau_p`, `w`, `gamma`, exact prompts, and GSM8K seed. Pre-register a small
factorial sensitivity grid on pilot IDs, freeze one central choice before full evaluation, and label
every row `paper-specified-partial`. Never call the best post-hoc grid point the reproduction.

### M1 — DeepConf official-code pilot

Pin the official DeepConf commit and paper-pinned vLLM commit
`31f09c615f4f067dba765ce5fe7d00d880212a6d`. First prove row-level equality of our saved
confidence with the pinned function using raw logits. Run Qwen3-8B/AIME24 with the paper prompt,
temperature .6, top-p .95, top-k 20, 32k cap, native warm-up `N_init=16`, and pilot K=32/64.
Validate answer normalization, 2,048-token overlapping group confidence, percentile direction,
positive vote weights, `beta>=.95`, and budget termination. This pilot is not a paper-table result.

### M2 — DeepConf full acquisition and offline reproduction

Acquire 4,096 complete traces for each of the 30 AIME24 questions. Save native confidence plus
our four frozen scalar channels for every token, and raw top-k audit shards. Perform all 64 fresh
resampling repetitions and K/filter/budget variants offline without more generation.

Regression targets include majority@512 80.0% at 2.32e8 tokens and DeepConf-low online 86.5%
at 0.90e8 tokens (-61.1%). Estimated generation is about 1.8B tokens, 75--150 B200-equivalent
GPU-hours, and 20--60 GB under the scalar-rich retention contract. Full top-50 retention would be
about 0.6--1.2 TB and requires a separate storage decision.

### W1 — Streaming official-asset reproduction (conditional)

Do not generate/label a substitute corpus. If P0 recovers the official trajectories, Claude labels,
split files, layer choices, probe code/checkpoints, and evaluator:

1. verify hashes and reproduce one released checkpoint on a smoke subset;
2. reproduce step AUC, prefix `Local` AUC, and prefix `Final` AUC separately for each model;
3. run our causal/logprob scores only on rows with aligned telemetry and identical labels;
4. report the supervised hidden-state row in its own access tier.

Published AUC targets are documented in the digest. If assets or exact training configuration remain
missing, publish a precise `blocked-assets` row rather than an approximate number.

### C1 — untouched causal confirmation (after all method hashes freeze)

Use a genuinely untouched family, recommended gpt-oss-20B/CommonsenseQA under the REFRAIN
protocol. Freeze the method, budgets `{16,32,64,128,256,512}`, feature/order hash, calibration,
alarm horizon, and analysis script before correctness labels are opened. This—not the repeatedly
used caches—is where a new transfer/generalization claim may be made.

## 5. Apples-to-apples analysis contracts

### Localization table

One ordered ProcessBench population and official evaluator. Report each subset and macro values for
our method, max entropy, Mind the Gap, uPRM control/full uPRM, official PRMs, and critics. Include
access, labels/training, model passes, scorer size, fidelity, and grouped intervals. Put native SLA in
a visibly separate subtable.

### Prefix-detection table

At absolute token budgets only, report AUROC/AUPRC, prevalence-normalized AP, fraction of
above-chance full-trace discrimination recovered, time to reliable warning, detection at fixed
trace-level ever-alarm FPR, false alarms on correct traces, length bands, and paired question
bootstrap. Fractional trace budgets are retrospective diagnostics only.

### Single-trace stopping table

For vanilla, REFRAIN, LEASH, our forced-closure policy, and fixed budgets, report pass@1, total
reasoning+answer tokens, wall time, parser failures, forced-closure rate, and the full
accuracy-compute frontier. A policy saves tokens only after the closure is actually generated and
graded; truncation-only estimates are not realized savings.

### Multi-trace table

For majority, official DeepConf variants, and our score used only as a declared trace-filtering or
vote-weight adaptation, report accuracy against total generated tokens and number of completed/
aborted traces. Never compare multi-trace total compute to a one-trace detector without the access
tier next to the number.

## 6. Automated gates and failure handling

Each stage emits `GATE.json` with machine-checkable pass/fail reasons:

- manifest fields complete and hashes verified;
- expected IDs and deterministic order match;
- no duplicate/resumed trace keys;
- token/logit arrays align and values are finite;
- raw/post-warper distinction verified;
- suffix-invariance and no-future-length tests pass;
- native parser coverage and metric unit tests pass;
- smoke reproducibility passes on a second invocation;
- projected disk use remains under the declared budget.

On model/OOM failure, reduce batch size only; do not change model, max length, quantization,
prompt, or decoding silently. On preemption, resume the same run ID. On an outcome that differs from
the paper, preserve the result and diagnose provenance—never tune toward the published number on
evaluation labels.

## 7. Deliverables for tomorrow and final completion

### Tomorrow's advisor packet

Produce even if long jobs are still running:

1. `STATUS.md`: commit, cluster connectivity, job IDs/states, completed counts, ETA, storage;
2. a protocol/fidelity matrix for all seven papers;
3. the L0 shared ProcessBench table from existing artifacts;
4. completed P0/P1 audits and any L1/S1/M1 smoke/pilot results, explicitly labelled preliminary;
5. no headline conclusion based on a partial or outcome-selected run.

### Final package

- four separate comparison tables from §5;
- machine-readable per-question results, manifests, hashes, and grouped bootstrap intervals;
- AUROC-versus-budget and accuracy-versus-compute plots;
- a self-contained HTML report showing access, passes, labels, fidelity, win/tie/loss, and cost;
- a Markdown advisor summary and a pass/fail panel for every preregistered claim;
- exact links from each table row to its paper digest, run manifest, code commit, and evaluator.

Do not update `HISTORY.md`, `PROGRESS.md`, or `Research_Directions.md` with new conclusions until
the full result identities and report hashes are frozen. Do not commit raw cluster data.

## 8. Codex review addendum — execution decisions after the 2026-08-17 interim results

This addendum was requested by Omri after review of
`STATUS_FOR_CODEX_2026-08-17.md`. Treat it as the current execution direction unless Omri
explicitly supersedes it. It resolves Q1--Q5 from that status report. Preserve the fidelity labels
and the rule that outcomes never determine promotion.

### Immediate safety: do not perturb active runs

1. Continue the already submitted S1 REFRAIN, S2 LEASH, and M2 DeepConf jobs under their original
   manifests and stamped commit.
2. Do **not** sync the three post-`d0d4df1` reporting/offline commits into the shared cluster code
   while an active job or a future requeue may resume from that code tree. A changed repository
   stamp can invalidate resume provenance even if the driver itself did not change. Analyze from an
   isolated checkout if necessary; sync only at a clean run boundary.
3. Keep LEASH's preregistered central constants
   `{B: 30.0, tau_p: 0.95, w: 16, gamma: 0.10}`. The pilot sweep remains sensitivity analysis;
   never promote a post-hoc winning cell as the reproduction. This resolves Q4.
4. Refresh the stale P0 gate after the model prefetch state is visible, but do not let the stale
   Slurm/P0 bookkeeping interrupt valid GPU jobs.

### M2 DeepConf: mandatory balanced operational checkpoint

M2 is scientifically valuable but now projects to roughly 1,120 B200 GPU-hours. Add a read-only
operational checkpoint as soon as the pool contains a balanced milestone, preferably at least 256
completed traces for every AIME24 question. The checkpoint must report:

- expected/observed trace keys by question and shard, duplicates, missing IDs, and hash verification;
- parser coverage, finite/aligned telemetry, raw-logit audit coverage, and answer-normalization
  failures;
- trace-length distribution, tokens/s, storage growth, updated GPU-hour/ETA estimate, and requeue
  requirements;
- an offline smoke reproduction of majority and the named DeepConf variants on the balanced
  partial pool, clearly labelled partial and not entered into a findings table;
- agreement of the retained DeepConf scalar with the already-passed raw-top-50 equality audit.

This is a `continue` versus `repair provenance/schema/resource failure` checkpoint. Performance
relative to the paper is **not** a stop criterion. If the operational checks pass, continue the full
pool even when a preliminary score is disappointing. Do not open or optimize our causal method on
these partial outcomes before its registry is frozen.

### Q1 decision: preserve the uPRM reconstruction; do not tune on ProcessBench

Keep L1's 0.2265 macro-F1 as the honest result of the first declared, paper-specified-partial
reconstruction. The nearly constant predicted-clean rate is evidence about marker/prompt
sensitivity, not permission to select a new prompt using the 3,400 evaluation labels.

Do not book another uPRM GPU run for tomorrow's packet. A second reconstruction is allowed later
only as a preregistered sensitivity analysis whose marker/prompt is chosen without ProcessBench
labels (for example by an external label-free calibration corpus). Evaluate it once on untouched
IDs and retain both rows. Never replace the first row or call the second one official-exact.

### Q2 decision: build L0 now; separate direct competitors from ceilings

Wire L0 immediately using the available artifacts:

- ours, maximum entropy, shared-protocol and native Mind-the-Gap rows where available;
- the new L1 uPRM-control row;
- existing released-PRM and critic ceiling rows.

Inventory and hash every source. Mark earlier artifacts `pre-contract provenance`; do not fabricate
a retroactive immutable run manifest. This labelled table is acceptable for the interim advisor
packet.

For final paper-exact work, prioritize rerunning **Mind the Gap native** under the current contract,
because it is the direct localization competitor. Do not spend scarce GPU capacity rerunning the
PRM/critic ceilings before the advisor packet; they are different-access supervised ceilings. A
later rerun is warranted only when missing provenance materially prevents verification of their
metric or population. Keep Mind-the-Gap native SLA separate from the all-trace ProcessBench F1
panel.

### Q5 decision: freeze and implement the prefix-detection lane now

This is the highest-priority missing implementation and is CPU/offline work over acquired
telemetry. Before inspecting M2 outcome comparisons, create the versioned registry at
`docs/experiments/PAPER_EXACT_CLAIM_REGISTRY_V1.yaml` and freeze:

- question-grouped development/calibration/audit splits and stable IDs;
- absolute budgets `{16,32,64,128,256,512}`;
- `iu28_no_length` as the primary current method, with elapsed prefix length as an ablation only;
- feature names/order/signs and code hash;
- trace-level alarms calibrated on `max_t score(t)` over the complete registered horizon;
- AUROC, AUPRC, prevalence-normalized AP, fraction of above-chance final discrimination recovered,
  time-to-warning, and fixed ever-alarm-FPR metrics;
- question-grouped paired bootstrap and the claim/non-inferiority margins already frozen by the
  phase-1 plan.

Machine-check **all ten** nulls already preregistered by the phase-1 checkpoint rather than choosing
a favorable subset. The generated gate must at minimum make suffix invariance, tokenwise-versus-
chunked replay, endpoint handling, label permutation, feature-order perturbation, sign/orientation,
length-only leakage, split isolation, alarm-horizon calibration, and grouped-resampling validity
explicit. If the canonical list uses different names, map these checks to it in the registry rather
than silently creating an eleventh protocol. Emit a frozen JSON rendering plus hashes before opening
comparison results.

### Q3 decision: keep C1 untouched

Do not submit C1 until all of the following are true:

1. S1 full has landed and passed its non-outcome gates;
2. the prefix method, budgets, feature/order hash, alarm horizon, calibration, registry, and analysis
   script are frozen and committed;
3. no C1 correctness labels or comparative outputs have been opened.

Only then run the recommended gpt-oss-20B/CommonsenseQA confirmation once. This resolves Q3 and
preserves the only clean transfer/generalization claim in the cycle.

### W1 and tomorrow's decision-oriented packet

Leave Streaming Hallucination Detection at `blocked-assets`. Do not create a substitute corpus,
labels, prompt, split, probe, or checkpoint and present it as a reproduction.

For tomorrow, prioritize artifacts that remain useful while long jobs run: the provenance-labelled
L0 table, the fidelity/access matrix, the frozen prefix registry and driver/gates, the balanced M2
operational audit if its milestone is available, REFRAIN/LEASH progress, and explicit ETAs. Keep
localization, prefix detection, single-trace stopping, and multi-trace adaptive compute in four
separate panels. Every partial number must say `preliminary`; no partial outcome may become a
headline claim.
