# Reasoning Hallucination Localization: Methods and Benchmarks

**Date:** 2026-08-09
**Status:** Literature map and benchmark decision. No new experiment is reported here.

## Why this document exists

The current ProcessBench report compares our method with Mind the Gap. That is
not a complete view of the field. Earlier research in this repository covered
other process-error datasets and process reward models, but the information was
spread across several files and mixed tasks that are not directly comparable.

This document gives one clear map. It separates:

1. detecting whether a complete answer is wrong;
2. finding the first wrong reasoning step;
3. scoring every reasoning step;
4. detecting a changing hallucination state while text is generated; and
5. locating failures in tool-using agents.

Our current localization claim concerns item 2. Results from the other tasks
can be useful evidence, but they must not be placed in the same result column
without explaining the difference.

## The task and its metrics

For a reasoning trace with steps `1, ..., T`, the target is either the index of
the earliest wrong step or `T + 1` when all steps are correct.

- **ProcessBench F1:** the harmonic mean of exact first-error accuracy on
  erroneous traces and correct abstention on clean traces. This is the primary
  end-to-end metric in our current report.
- **Exact localization:** the percentage of erroneous traces for which the
  predicted step is the annotated first wrong step.
- **Within-one localization:** the percentage within one annotated step. This
  is a useful diagnostic, not a replacement for the official metric.
- **Clean-trace accuracy:** the percentage of fully correct traces for which
  the method predicts no error.
- **AUROC for error presence:** how well a global score ranks erroneous traces
  above clean traces before choosing a decision threshold.

Threshold calibration must be described separately from score fitting. Our
score construction is unsupervised, but the current ProcessBench decision
threshold uses calibration labels. The correct claim is therefore
**calibrated unsupervised scoring**, not a fully label-free decision rule.

## Benchmarks and datasets

| Resource | What it measures | Annotation or supervision | Direct fit for our current claim |
|---|---|---|---|
| **ProcessBench** | Earliest erroneous step, or all steps correct, in 3,400 mathematical solutions | Human expert error locations | **Yes.** This is our primary benchmark. |
| **MR-GSM8K** | Meta-reasoning over GSM8K solutions, including judging the reasoning process | Constructed meta-reasoning evaluation data | **Partial.** Useful external transfer, but its task and MR-Score are not identical to ProcessBench F1. |
| **PRMBench** | Fine-grained process-reward behavior across simplicity, soundness, and sensitivity | 6,216 problems and 83,456 step labels | **Partial.** Strong stress test for per-step evaluators, not a drop-in first-error benchmark. |
| **Socratic-PRMBench** | Errors under six reasoning patterns | 2,995 flawed reasoning paths | **Partial.** Useful for mechanism stress tests, but it changes the evaluation taxonomy. |
| **ReasonEval** | Step validity and redundancy | Labeled meta-evaluation data and trained evaluators | **No direct score match.** Useful as a supervised evaluation baseline. |
| **PRM800K** | Human process supervision for mathematical solutions | About 800,000 filtered step labels over about 75,000 solutions | **Training resource, not a held-out benchmark.** It supports supervised PRM ceilings. |

ProcessBench remains the cleanest benchmark for the exact claim we currently
make. MR-GSM8K is the most relevant transfer candidate if its data contract can
be aligned with our token telemetry. PRMBench is valuable for asking *which
types* of errors a detector misses, but it should not replace ProcessBench.

## Method families beyond Mind the Gap

### 1. Process reward models trained with human labels

PRMs assign a score to each reasoning step. The PRM800K work showed that
process supervision can outperform outcome-only supervision. Later systems,
including Qwen mathematical PRMs, R-PRM, and generative PRMs, evaluate each
step with a separate trained model.

These are important performance ceilings. They are not direct peers of our
method because they use process labels, trained evaluator models, or both. A
fair table should place them in a **supervised external-evaluator** category.

### 2. Automatically supervised PRMs

Math-Shepherd and OmegaPRM create process labels automatically from rollouts or
search. This reduces human annotation, but the final detector is still trained
on generated process targets. These methods test whether scalable synthetic
supervision can replace human labels. They do not meet our stricter rule that
labels and pseudo-labels stay outside score fitting.

### 3. Unsupervised Process Reward Models (uPRM)

The 2026 uPRM paper is the most important missing comparison in the previous
localization summary. It defines a score from next-token probabilities and
jointly evaluates candidate first-error positions across a batch of reasoning
traces. The paper reports no human step labels and no final-answer labels for
training, and evaluates first-error localization on ProcessBench.

This is much closer to our scientific setting than a supervised PRM. It is not
automatically comparable yet: we must reproduce its input contract, candidate
construction, threshold rule, and ProcessBench metric before placing its
number beside ours. It should be the **first new published baseline audited**.

### 4. Critic language models

ProcessBench also evaluates general LLMs prompted to critique a solution step
by step. These models can identify the first wrong step without a task-specific
PRM, but they add a second large model and additional inference. They are a
useful high-cost reference, not a single-pass gray-box peer.

### 5. Streaming hallucination detection

Streaming Hallucination Detection in Long Chain-of-Thought Reasoning treats
hallucination as an evolving latent state. It combines local step judgments
with a cumulative prefix-level signal. This is conceptually close to our
global/local decomposition, but it asks for online state tracking rather than
only the first wrong step. It is relevant for future onset detection and early
stopping. Its published score should not be copied into a ProcessBench F1 table
unless the official code is run under our exact protocol.

### 6. Spectral and trajectory detectors

HALT treats top-token log-probabilities as time series but trains a GRU, so it
is a supervised answer-level ceiling rather than a localization peer. The Graph
Signal Processing framework uses attention-induced graphs and spectral energy
patterns, but its reported task is answer-level hallucination detection. These
papers support the use of trajectory structure; they do not yet establish
first-error localization under our benchmark.

### 7. Mind the Gap

Mind the Gap is currently the only external published method reproduced in our
exact ProcessBench run. It is therefore a valid shared-protocol control, but it
is not the only relevant method in the literature.

### 8. Our method

Our leading simple candidate uses mixed-v2 DUFS-LIU at both resolutions:

- a global head fuses complete-trace summaries to estimate whether an error is
  present;
- a local head fuses five token-resolved curves to estimate where the error
  begins.

It reaches 31.72% ProcessBench F1 in the current eight-cell study. This number
is internal evidence from four dataset families across two model sizes. It is
not a claim of state of the art against supervised PRMs or critic models.

## Fair comparison categories

Every localization result should be assigned to one access category:

| Category | Examples | Main cost or supervision |
|---|---|---|
| Single-pass, unsupervised gray-box | DUFS-LIU localization | Generator token statistics; no label-trained evaluator |
| Unsupervised probability method | uPRM | Next-token probabilities; exact batching and inference cost must be reproduced |
| Published shared-protocol control | Mind the Gap | Reproduced under our ProcessBench protocol |
| Automatically supervised PRM | Math-Shepherd, OmegaPRM | Rollouts or search create training targets |
| Human-supervised PRM | PRM800K-trained and Qwen PRMs | Step labels and a separate evaluator model |
| Critic model | Prompted general LLM | Additional large-model inference, often step by step |

This grouping prevents a cheap label-free score from being described as if it
used the same information and compute as a trained 72B PRM.

## What should be added to the benchmark

The next benchmark revision should add the following in this order:

1. **uPRM reproduction.** It is the closest newly identified label-free peer.
2. **Transparent token baselines.** Maximum entropy, minimum token
   probability, entropy CUSUM, a simple change-point score, random step, and
   last step. These reveal whether a complex locator beats obvious rules.
3. **Published supervised PRM ceilings.** Use official ProcessBench outputs or
   a faithful run, but report them in a separate category.
4. **Critic-model ceiling.** Include only if the model, prompt, and inference
   budget are fixed and reported.
5. **External transfer.** Prefer MR-GSM8K if the exact first-error and trace
   contracts can be aligned. Otherwise add a new ProcessBench-compatible model
   family before changing the method again.

All method choices must be frozen before the evaluation labels are opened. A
new benchmark should report per-dataset results, grouped uncertainty, runtime,
model calls, token budget, and whether a threshold used labels.

## Decision

The previous statement that Mind the Gap is the only external method in the
field was too broad. The correct statement is:

> Mind the Gap is the only external published method measured in our existing
> shared-protocol artifact. The most important missing label-free peer is uPRM;
> supervised PRMs and critic models are required ceilings, while streaming and
> spectral trajectory methods are related but not direct first-error results.

No localization variant should be developed merely to fill this literature
gap. First expand the benchmark and see which failure mode remains.

## Primary sources

- ProcessBench: <https://arxiv.org/abs/2412.06559>
- MR-GSM8K: <https://arxiv.org/abs/2312.17080>
- PRMBench: <https://arxiv.org/abs/2501.03124>
- Socratic-PRMBench: <https://arxiv.org/abs/2505.23474>
- ReasonEval: <https://arxiv.org/abs/2404.05692>
- Let's Verify Step by Step and PRM800K: <https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf>
- Math-Shepherd: <https://arxiv.org/abs/2312.08935>
- OmegaPRM: <https://arxiv.org/abs/2406.06592>
- Unsupervised Process Reward Models: <https://arxiv.org/abs/2605.10158>
- Streaming Hallucination Detection: <https://arxiv.org/abs/2601.02170>
- HALT: <https://arxiv.org/abs/2602.02888>
- Graph Signal Processing framework: <https://arxiv.org/abs/2510.19117>
- Qwen PRM development lessons: <https://arxiv.org/abs/2501.07301>

## Earlier repository research retained for audit

- `papers/State of the Art in LLM Hallucination Detection for Reasoning Tasks (as of July 2026) A Benchmarking Guide for Unsupervised Gray-Box Methods.md`
- `docs/research_notes/CoT and Agentic Hallucination Detection.md`
- `docs/research_notes/AI Agent Hallucination Benchmark Analysis.md`
- `docs/research_notes/localization_research_handoff_2026-08-08.md`

The older files remain useful source notes. This document is the current
decision-oriented map.
