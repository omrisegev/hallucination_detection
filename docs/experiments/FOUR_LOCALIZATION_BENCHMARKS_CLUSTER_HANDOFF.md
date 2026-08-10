# Four Hallucination-Localization Benchmarks: Cluster and Reporting Handoff

**Date:** 2026-08-10  
**Audience:** Claude or another agent continuing the cluster work  
**Status:** execution handoff; no result is claimed by this document  
**Git rule:** do not commit or push until Omri reviews the final benchmark report

## 1. Goal

Build one advisor-facing benchmark report that shows what our spectral method can
and cannot do across four different hallucination-localization tasks:

1. locate hallucinated tokens or character spans in a RAG answer;
2. identify an unsupported sentence or claim in a RAG answer;
3. score the correctness of every reasoning step;
4. locate the first erroneous reasoning step, or abstain when all steps are
   correct.

These are four different prediction problems. They do not share one label space
or one official metric. Never average their scores into one leaderboard value.
The report should instead contain four separate, apples-to-apples panels.

The final report must also include the existing 24-cell answer-level detection
result as background. Humanity's Last Exam should be added later as a separate
answer-level detection transfer test. Neither belongs to the four localization
panels.

## 2. Relationship to the earlier 2026 reasoning-localization plan

This plan extends, rather than replaces,
`docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md`.

The earlier document separated answer detection, first-error localization,
every-step scoring, streaming detection, and agent failure localization. It made
ProcessBench the primary benchmark for our existing first-error claim. That
remains unchanged.

The relationship is:

| Earlier decision | Status in this plan |
|---|---|
| ProcessBench is the primary first-error benchmark. | Unchanged. Our full ProcessBench score already exists. We only need stronger full-size competitors. |
| PRMBench is not a replacement for ProcessBench. | Unchanged. PRMBench is used here for the different every-step task. |
| Add supervised PRM and critic ceilings in separate access categories. | Continued through Qwen2.5-Math-PRM-7B and QwQ-32B-Preview. |
| Audit uPRM first because it is the closest label-free peer. | The audit was attempted. The repository's current uPRM-like pilot is only an independent LLM-as-judge reconstruction. The real uPRM requires unavailable training details/code and was not reproduced. Do not scale or rename the reconstruction. |
| Do not mix metrics from different tasks. | Strengthened: the final report has four visibly separate panels. |
| Freeze method choices before evaluation labels are used. | Required for new RAGTruth, KnowHalBench, and PRMBench scoring. Existing ProcessBench results are retrospective/calibrated and must be labelled that way. |

The broader RAG work is new scope relative to the reasoning-only map. It does
not alter the earlier reasoning conclusions.

## 3. What “our method” means

The common algorithmic family is **DUFS-LIU**:

- **DUFS** is based on Lindenbaum et al., *Differentiable Unsupervised Feature
  Selection based on a Gated Laplacian*. It learns feature gates without
  correctness labels and uses them to define a sample graph.
- **LIU** means Laplacian IU-PCR. IU-PCR is the two-component extension of
  U-PCR based on Tenzer et al., *Crowdsourcing Regression: A Spectral
  Approach*. The Laplacian penalty is our addition.

The solver and frozen graph settings stay common:

| Setting | Frozen value |
|---|---:|
| IU-PCR components | 2 |
| graph neighbours | 7 |
| global Laplacian strength | 0.1 |
| local Laplacian strength | 0.3 |
| DUFS seeds | 11, 23, 37 |
| DUFS epochs | 80 |

The observation unit changes by task. Therefore the feature adapter must also
change by task. This is not a contradiction and must not be hidden:

- answer detection uses the full-trace mixed-v2 feature contract;
- token and step localization use short-sequence, token-resolved views;
- RAG tasks may add evidence-condition contrasts because the same fixed answer
  is scored under different contexts.

Do not claim that all 30 response-level mixed-v2 summaries are unchanged token
features. Several spectral summaries are undefined or unstable on short token
windows. The previous broad-28 local experiment already showed that direct
expansion hurt ProcessBench localization.

The current local reasoning contract contains five curves:

1. token entropy;
2. sliding-window entropy variance;
3. absolute entropy CUSUM;
4. sliding-window spilled-energy variance;
5. absolute spilled-energy CUSUM.

At `lambda=0`, every DUFS-LIU implementation must reproduce the matching
IU-PCR score exactly. Always report IU-PCR next to DUFS-LIU so that any claimed
Laplacian contribution is visible.

## 4. The apples-to-apples rule

Within each panel, our method and the competitor must use:

- the same benchmark version and split;
- the same example identifiers;
- the same fixed answer or reasoning trace;
- the same gold annotations;
- the official metric of that benchmark;
- the same bootstrap groups and resamples.

For every result, record:

- dataset and revision;
- paper and official-code revision;
- model and checkpoint revision;
- prompt, tokenizer, truncation, and decoding settings;
- whether the method saw generator internals;
- whether it used a separately trained evaluator;
- whether it required another generation or only teacher forcing;
- whether training used human labels, synthetic labels, or no labels;
- whether a decision threshold used development labels;
- runtime, accelerator count, peak memory, and number of model passes.

Every competitor result must receive one fidelity label:

1. **Exact reproduction:** official data, model, prompt, inference, parser, and
   metric all match.
2. **Protocol reproduction:** official task and metric match, but a declared
   model or infrastructure component differs.
3. **Adaptation:** a published concept is applied under a different input or
   score contract.
4. **Published context only:** the paper's number is quoted but was not
   reproduced locally.

Never describe levels 2--4 as an exact reproduction.

## 5. Current completed work: do not rerun it

### 5.1 Answer-level 24-cell detection

The current descriptive implementation standard is full-pool mixed-v2
DUFS-LIU:

- macro AUROC: **0.776562**;
- matching IU-PCR: **0.776087**;
- deployed U-PCR: **0.773528**.

The DUFS-LIU versus IU-PCR difference is small and its uncertainty includes
zero. Present DUFS-LIU as the current implementation standard, not as a proven
large Laplacian gain. Source:
`results/hard_filter_dufs_liu_24cell/REPORT.md`.

### 5.2 First-error localization with our method

Our ProcessBench inference and scoring are complete for Qwen3-4B and Qwen3-8B
over GSM8K, MATH, OlympiadBench, and OmniMath. There are 3,400 underlying
ProcessBench examples and two scorer/model families, producing eight
model-by-dataset cells.

Existing results:

- unified global/local core-five DUFS-LIU: **31.72% ProcessBench F1**;
- frozen GL-LIU v1: **31.36%**;
- shared-protocol Mind the Gap control: **25.71%**.

Do not rerun our ProcessBench inference. The missing work is the full-size PRM
and critic comparison and, if needed, regeneration of row-level joined score
files from the existing caches.

### 5.3 Existing RAGTruth work

The repository already contains fixed-answer Qwen2.5-1.5B telemetry for:

- 900 development responses;
- 2,700 test responses;
- full-context and no-context conditions for all tasks;
- leave-one-chunk-out conditions for QA and Data-to-Text;
- exact answer token IDs and character offsets;
- target-token probabilities and top-50 distributions.

The current GASP comparison is only a top-50-plus-tail approximation. The
original-30 evidence-aware experiment is response-level, not localization.
Neither should be renamed as the exact experiments requested below.

The existing LettuceDetect run used the base checkpoint and retained only
counts/overlap summaries. It reproduced example-level performance reasonably
well but discarded predicted span coordinates. It is not sufficient for a
span-localization comparison.

## 6. Benchmark 1: token or character-span localization

### Published anchor

**LettuceDetect**, evaluated on the RAGTruth test split.

- Paper: <https://arxiv.org/abs/2502.17125>
- Benchmark: RAGTruth, <https://aclanthology.org/2024.acl-long.585/>
- Target checkpoint: the paper-reported large ModernBERT checkpoint. Verify the
  exact Hugging Face ID and revision in the manifest before submission; the
  expected repository name is
  `KRLabsOrg/lettucedect-large-modernbert-en-v1`.

### Why this job is required

LettuceDetect is a supervised token classifier that outputs localized spans.
It supplies a strong trained ceiling for the exact token/span task. Our
current base-checkpoint JSON contains only response-level decisions and span
counts, so no official span F1 can be reconstructed from it.

### Cluster job

Create `ragtruth_lettuce_large_span_full`.

Run the official preprocessing and large checkpoint on all 2,700 RAGTruth test
responses. Save, for every response:

- response ID and task;
- predicted character start/end coordinates;
- predicted token start/end coordinates when available;
- raw token probabilities or logits when exposed by the checkpoint;
- threshold and merging rule;
- inference time and truncation diagnostics.

The paper merges consecutive tokens above its fixed threshold. Reproduce that
rule exactly before adding any sensitivity analysis.

### Our score on the same examples

Do not run another language-model inference job unless the existing telemetry
fails alignment validation. Build a frozen RAG local adapter from the saved
telemetry:

- use the five stable local curves;
- align full-context and no-context target tokens exactly;
- use both the full-context curve and its evidence-removal change;
- keep LOO-only diagnostics separate because Summary has no comparable LOO
  conditions;
- fit IU-PCR/DUFS-LIU without labels;
- select any binary threshold using grouped development sources only;
- freeze the score and threshold before test labels are opened.

### Primary result

Character-overlap micro precision, recall, and F1 on the same 2,700 responses.
Also report token micro F1, span IoU, example-level F1, and QA/Summary/Data-to-
Text separately.

### Fidelity gate

Do not promote the run unless the official LettuceDetect preprocessing and
metric reproduce the paper's large-checkpoint result within an explained
numerical tolerance. If the model card and paper use different splits or
metrics, report both and name the discrepancy.

## 7. Benchmark 2: unsupported sentence or claim localization

This panel has two sub-benchmarks because sentence localization and semantic
claim checking are not identical.

### 7.1 GASP sentence localization on RAGTruth

#### Published anchor

**GASP**, arXiv:2607.04223:
<https://arxiv.org/abs/2607.04223>.

GASP keeps an answer fixed and measures how its target-token distribution
changes when evidence is removed. It is the closest direct competitor to our
evidence-aware spectral fusion.

#### Exact protocol to reproduce

Pin the official GASP repository and released response IDs before coding. The
paper-compatible RAGTruth experiment uses:

- 400 balanced Summary and Data-to-Text responses;
- the paper's seed and exact sample IDs;
- Qwen2.5-1.5B as the scorer;
- answer cap 200 tokens;
- context cap 700 tokens;
- full context;
- no context;
- five sentence-grouped leave-one-chunk-out conditions;
- full-vocabulary Jensen-Shannon divergence;
- the paper's sentence splitting and AUROC computation.

The existing top-50 approximation does not satisfy this protocol.

#### Cluster job

Create `gasp_ragtruth_exact_qwen15b_full`.

Compute full-vocabulary JSD online during the forward pass. Do not save a dense
vocabulary distribution for every token. Save the final per-token and
per-sentence likelihood/JSD components, response IDs, chunk definitions, and
all truncation/alignment diagnostics.

#### Our score

On the exact same sentence rows, run the already defined Evidence-Contrast
IU-PCR and DUFS-LIU arms with frozen graph settings. Report both. Do not claim
that DUFS helped unless it beats IU-PCR on the same rows with paired grouped
uncertainty.

#### Primary result

Sentence AUROC and AUPRC, grouped by source ID, with 1,000 paired bootstrap
samples. Report Summary and Data-to-Text separately.

### 7.2 RefChecker claim checking on KnowHalBench

#### Published anchor

Hu et al., **Knowledge-Centric Hallucination Detection**, introducing
RefChecker and KnowHalBench:
<https://aclanthology.org/2024.emnlp-main.395/>.

RefChecker extracts claim triplets and classifies each as supported,
contradicted, or unverifiable relative to a reference/context.

#### Why this job is required

GASP treats a sentence as the localized unit. RefChecker tests whether our
method also works when the unit is an explicit semantic claim. It also reveals
an important limitation: our current scalar risk score naturally separates
supported from unsupported claims, but it does not automatically distinguish
contradiction from missing evidence.

#### Cluster jobs

First create an N=30 alignment and metric pilot. Pin the official RefChecker
repository, KnowHalBench release, extractor/checker configuration, and
evaluation code.

The pilot must verify:

- stable example and claim IDs;
- exact Zero, Noisy, and Accurate Context definitions;
- reproducibility of the official three-way metric;
- availability of the strongest fully public configuration;
- whether extracted claims are text units, triplets, or contiguous answer
  spans.

If the pilot passes, create `refchecker_knowhalbench_open_full` and reproduce
the strongest accessible open configuration. If the strongest paper result
uses an unavailable proprietary model, quote it only as published context.

For our method, teacher-force the fixed claim text or a deterministic textual
rendering of the official triplet under each official context condition using
one pinned scorer model. Save rich token telemetry. This is an adaptation and
must be labelled as such. Do not claim that a separately scored claim is an
unchanged full-response token span.

#### Primary result

For the exact RefChecker reproduction, report official three-way accuracy and
macro-F1 for supported, contradicted, and unverifiable claims, separately for
Zero, Noisy, and Accurate Context.

For our scalar score, collapse contradicted and unverifiable into
**unsupported** and report AUROC, AUPRC, and development-thresholded macro-F1
against RefChecker under the same binary collapse. Do not place the binary
score in the paper's three-way column.

## 8. Benchmark 3: correctness of every reasoning step

### Published anchor

Song et al., **PRMBench: A Fine-grained and Challenging Benchmark for
Process-Level Reward Models**:
<https://arxiv.org/abs/2501.03124>.

PRMBench contains 6,216 problems and 83,456 step labels. It evaluates
simplicity, soundness, sensitivity, and their published subcategories.

### Why this job is required

ProcessBench labels the first wrong step. It does not independently certify
every later step, so it cannot be used to measure an every-step classifier.
PRMBench provides the missing per-step ground truth.

### Cluster jobs

Create two jobs over the exact same official PRMBench traces.

1. `prmbench_qwen25math7b_full`: run
   `Qwen/Qwen2.5-Math-PRM-7B` with its official tokenizer, chat template,
   separator tokens, reward extraction, and evaluation recipe.
2. `prmbench_qwen3_8b_telemetry_full`: teacher-force the same fixed reasoning
   traces through a pinned Qwen3-8B causal language model and save the rich raw
   telemetry required by our five local curves.

Before full execution:

- pin the exact official PRMBench data/code release; the earlier draft called
  it “v5”, but the manifest must verify the real release name and hash;
- run local schema fixtures;
- run N=30 problems through both jobs;
- verify step boundaries, token offsets, reward counts, and official metrics;
- freeze the token-to-step aggregation before reading evaluation labels.

For our primary step score, use the maximum token risk inside each official
step, matching the current first-error locator's maximum-risk convention.
Report the 95th-percentile token risk as a fixed sensitivity analysis and
measure correlation with step length. Do not select between aggregations using
test labels.

### Primary result

Report:

- the official PRMBench score;
- simplicity, soundness, and sensitivity;
- every published subcategory;
- step-level AUROC and AUPRC;
- macro-F1 only with a threshold frozen on the official development split or
  another declared non-test calibration set.

Bootstrap complete problems, never individual steps.

## 9. Benchmark 4: first erroneous reasoning step

### Published anchor

Zheng et al., **ProcessBench: Identifying Process Errors in Mathematical
Reasoning**:
<https://arxiv.org/abs/2412.06559>.

The task is to return the first erroneous step or `-1` when the entire
solution is correct.

### What is already complete

Our full ProcessBench result and the shared-protocol Mind the Gap control are
complete. Do not repeat their language-model inference.

The existing Qwen2.5-Math-PRM-7B, Qwen2.5-72B critic, and reconstructed judge
artifacts are only N=30-per-subset pilots. The Qwen2.5-Math-PRM pilot is a
faithful checkpoint implementation. The reconstructed judge is not uPRM and
must not be scaled.

### Required full PRM job

The existing cluster driver is ready. On the cluster, generate the gitignored
live `submit_pb_prm.sbatch` from its tracked template as described in the
template header. Then submit without `--n-samples`:

```bash
sbatch -p power-gpu --qos=owner_880 cluster/submit_pb_prm.sbatch \
  --out /shared/cycle2_tau_averbuch_prj/omrisegev1/results/pb_prm_qwen25math7b_full
```

This runs one supervised PRM forward pass for all 3,400 rows.

### Required critic job

The ProcessBench paper's stronger open critic is QwQ-32B-Preview. The existing
critic driver supports the official ProcessBench prompt and parser, but QwQ
requires its own frozen manifest and 32,768-token generation limit.

Run the pilot first:

```bash
sbatch -p power-gpu --qos=owner_880 cluster/submit_pb_critic.sbatch \
  --model Qwen/QwQ-32B-Preview \
  --n-samples 30 \
  --max-new 32768 \
  --out /shared/cycle2_tau_averbuch_prj/omrisegev1/results/pb_critic_qwq32b_pilot
```

Promote to full size only when each subset contains both classes, parse failure
is a small minority, and truncation is not systematic. The full command removes
`--n-samples` and writes to `pb_critic_qwq32b_full`.

If exact QwQ inference is technically unavailable, the already validated
Qwen2.5-72B-Instruct critic may be scaled instead, but it must be labelled
“ProcessBench protocol reproduction with a different critic model.”

### Primary result

Report on GSM8K, MATH, OlympiadBench, OmniMath, and their macro:

- official ProcessBench F1;
- erroneous-trace first-error accuracy;
- correct-trace abstention accuracy;
- exact localization;
- within-one-step localization;
- runtime, model size, and additional inference calls.

Our current threshold is selected on calibration labels and evaluated on a
held-out half. State this clearly. Do not describe the final decision policy as
fully label-free.

## 10. Cluster execution rules

Before doing anything, read `PROGRESS.md`, this file, and the relevant method
report. Audit `origin/master` and existing result directories so completed work
is not rerun.

The repository may contain local uncommitted research work. Do not overwrite
or discard it. Record the exact source commit in every experiment manifest.

Every new job follows this promotion sequence:

1. pinned paper, code, model, and dataset manifest;
2. local CPU/schema fixture;
3. N=30 cluster pilot;
4. pilot validation report;
5. full submission only after all fidelity and health gates pass.

All long jobs must:

- checkpoint atomically;
- resume idempotently after requeue;
- trap the cluster's preemption signal;
- save the richest raw output rather than only aggregate metrics;
- keep labels structurally separate from our score-fitting functions;
- save per-row IDs so paired comparisons are possible;
- record failed parses and truncation rather than dropping rows silently.

Do not put Hugging Face tokens or other credentials in tracked files. Live
sbatch files containing credentials remain gitignored.

## 11. Score freezing and evaluation

For new experiments, fitting code must not receive evaluation labels. The safe
order is:

1. validate inputs and align units;
2. construct features without labels;
3. fit IU-PCR and DUFS-LIU without labels;
4. write per-unit scores;
5. hash the score file and feature contract;
6. load labels in a separate evaluation command;
7. compute metrics and paired grouped uncertainty.

Threshold-free ranking metrics are primary whenever the benchmark supports
them. If a binary threshold is required, select it only from an official
development split or explicitly declared calibration sources, then freeze it
before test evaluation.

RAGTruth bootstrap samples must group by `source_id`. PRMBench and ProcessBench
bootstrap samples must group complete problems. Never treat evidence
conditions, sentences, claims, or steps from the same source as independent
bootstrap observations.

## 12. Required final report

Create one self-contained English `REPORT.html`, with a generated `REPORT.md`
and machine-readable CSV/JSON sources. Use simple English and define every
metric before showing it.

Required structure:

1. executive summary and current claim boundary;
2. the four localization tasks, shown as separate definitions;
3. existing 24-cell answer-detection background;
4. token/span RAGTruth panel;
5. sentence GASP and claim RefChecker panel;
6. every-step PRMBench panel;
7. first-error ProcessBench panel;
8. supervision, access, compute, and inference-pass table;
9. reproduction-fidelity checklist;
10. limitations, failed gates, and missing comparisons;
11. numeric provenance and artifact hashes.

For each panel show:

- dataset coverage;
- the competitor under its official protocol;
- IU-PCR and DUFS-LIU on the identical examples;
- paired method differences;
- per-task or per-subcategory breakdown;
- representative localized examples;
- mechanism diagnostics;
- runtime and access cost.

Do not create one macro score across the four localization tasks. The “macro
summary” is a status table containing each task's own primary metric, not an
average of incompatible metrics.

## 13. Acceptance criteria

The advisor report may call a field **measured** only when:

- the full intended split completed;
- row IDs align exactly across methods;
- the official metric implementation passed a known-answer or paper-
  reproduction check;
- our score was frozen before evaluation labels;
- supervision and inference access are visible beside every method;
- per-unit predictions and manifests are saved;
- uncertainty groups the correct independent source unit.

If a field fails these requirements, show **pending** or **protocol mismatch**.
Do not fill the cell with a published number from another model or split.

## 14. Separate HLE follow-up

Humanity's Last Exam is not a localization benchmark. The current full HLE
generation should finish; do not submit a duplicate generation job. After it
finishes, run the official grader and compare mixed-v2 IU-PCR/DUFS-LIU with the
model's verbalized confidence on the same answers.

Agentic Confidence Calibration and FUSE may be cited as contextual HLE work,
but their published numbers use different trajectories or best-of-50
selection protocols. Do not place them in the same same-output AUROC bar unless
their complete protocol is separately reproduced.

## 15. Bottom line for Claude

Do not develop another DUFS or Laplacian variant during this benchmark cycle.
The purpose is measurement, not method selection.

Complete the missing data in this order:

1. full ProcessBench Qwen2.5-Math-PRM-7B;
2. QwQ ProcessBench pilot, then full run if healthy;
3. LettuceDetect-large span-preserving RAGTruth run;
4. exact GASP RAGTruth rescore;
5. RefChecker N=30 fidelity/alignment pilot, then full open configuration;
6. PRMBench N=30 dual-adapter pilot, then both full jobs.

Use the existing telemetry for our RAGTruth token head whenever validation
permits. Do not spend cluster time regenerating data that already contains the
required aligned token statistics.

The scientific question is not whether our score beats every supervised
ceiling. It is whether a single-pass or teacher-forced, label-free spectral
score provides useful localization under much lower supervision and inference
cost. The final report must make that trade-off visible.
