# RAG Hallucination Localization: Methods and Benchmarks

**Date:** 2026-08-09
**Status:** Literature map for the planned Evidence-Contrast U-PCR study. No
RAG result from our method is claimed here.

## Why this document exists

The repository already contains a Phase-10 RAG research folder and a proposal
for Evidence-Contrast U-PCR. The raw research is useful, but it mixes
response-level hallucination detection, span localization, citation quality,
long-context grounding, and agent trajectories.

This document narrows the question:

> Given a fixed answer and retrieved evidence, can a method decide whether the
> answer is grounded and locate the unsupported claim, span, or citation?

This is the RAG analogue of the global/local decomposition used in our
reasoning work.

## The tasks are different

- **Response detection:** does the answer contain any unsupported content?
- **Sentence or claim detection:** which statements are unsupported?
- **Span localization:** which exact answer tokens or character spans are
  unsupported or contradictory?
- **Citation correctness:** does each citation support its associated claim?
- **Citation completeness:** are the claims that need evidence actually cited?
- **Retrieval diagnosis:** was the required evidence retrieved at all?

A method can do well on response detection and still fail to localize a span.
It can also identify an unsupported answer even when the retrieval system, not
the generator, caused the failure. These outputs must be reported separately.

## Metrics

- **Response AUROC:** ranking quality for answers with at least one
  hallucination.
- **Response AUPRC:** precision-recall quality when hallucinations are
  imbalanced.
- **Token or span AUROC:** ranking quality at token or character resolution.
- **Span F1 or overlap F1:** overlap between predicted and annotated spans.
- **Citation precision:** fraction of predicted citations that support their
  claims.
- **Citation recall or completeness:** fraction of claims needing support that
  receive adequate citations.
- **Latency and scorer passes:** required because evidence perturbation may
  rescore one answer many times.

Splits and bootstrap intervals must follow the source document or query, not
individual responses, when several responses share the same source.

## Benchmark map

| Benchmark | Annotation level | Main strength | Important limitation | Role for us |
|---|---|---|---|---|
| **RAGTruth** | Response and word/span annotations | Nearly 18,000 naturally generated RAG responses across QA, summarization, and data-to-text | Contexts are not a full long-document stress test; later re-annotation found missed spans | **Primary span benchmark** and first EC-U-PCR study |
| **RAGTruth++** | 865 spans in 408 re-annotated examples | Denser labels expose false negatives in the original subset | Dataset card and commercial re-annotation, not a peer-reviewed benchmark paper; small and high-prevalence | Confirmation and label-sensitivity audit, not primary selection |
| **RAGTruth-Enhance** | Re-annotated response and span labels | Used by RT4CHART and reports more missed hallucinations | Tied to one recent method paper; availability and split contract must be checked | Optional external confirmation |
| **TofuEval** | Sentence-level factual consistency and explanations | Expert annotations over topic-focused dialogue summaries | Context-grounded summarization, not standard retrieved-chunk RAG | Positive transfer test for evidence-sensitive methods |
| **RAGBench** | Explainable RAG quality labels across about 100,000 examples | Broad domains and task types | Many answers are short; not primarily a span-localization benchmark | Explicit failure test for evidence perturbation |
| **TRIVIA+** | Human grounded-hallucination labels plus controlled noisy labels | Long RAG contexts and realistic label-noise stress tests | New benchmark; exact span/citation contract must be checked before adoption | Strong modern confirmation candidate |
| **L-CiteEval** | Citation and long-context faithfulness metrics | Tests whether long-context models use and cite supplied context across tasks | Citation evaluation is not the same as hallucination-span detection | Citation-specific benchmark after the first span study |
| **FACTS Grounding** | Long-form response grounding | Strong long-context groundedness evaluation | Primarily response-level; does not directly provide RAG span localization | Response-level robustness benchmark |

RAGTruth is still the practical first benchmark because it supplies fixed
responses, contexts, response labels, and exact spans. TRIVIA+ is important
because it addresses two weaknesses of older data: long contexts and label
noise. L-CiteEval is important only when the project makes an explicit
**citation** claim rather than a general unsupported-span claim.

## Method families

### 1. GASP: evidence perturbation

GASP is the closest direct competitor to our planned idea. It keeps the answer
fixed, rescores it with full context, no context, and each chunk removed, and
uses likelihood changes and Jensen-Shannon divergence as grounding signals.
It reports about 0.73 response AUROC and 0.67 span AUROC on RAGTruth. The signal
transfers to TofuEval but fails on short-answer RAGBench.

This result matters for novelty: removing evidence is not our contribution.
The possible contribution is unsupervised fusion of many dependent evidence
contrasts with one U-PCR/DUFS-LIU family at response and span resolution.

### 2. Claim decomposition and external verification

RT4CHART decomposes the answer into claims, checks each claim against context,
and maps the decisions back to answer spans. It reports response F1 of 0.776 on
RAGTruth++ and span F1 of 47.5% on RAGTruth-Enhance. It provides interpretable
evidence, but depends on external claim decomposition and verification. It is a
strong verifier-based ceiling, not a gray-box peer.

### 3. Supervised token classifiers

The original RAGTruth work trains detectors on its labels. LettuceDetect uses a
ModernBERT token classifier over context-question-answer triples and reports
example-level F1 of 79.22% on RAGTruth. These methods directly localize tokens,
but their label-trained detector gives them a different access contract from
our proposed label-free score fitting.

### 4. Mechanistic context-versus-memory detectors

ReDeEP separates the contribution of external context from parametric
knowledge using Knowledge FFNs and Copying Heads. SEReDeEP adds semantic
entropy through trained linear probes. Lookback Lens uses ratios of attention
to context versus generated tokens with a linear classifier.

These methods support our diagnosis that hallucination can arise when the
model relies on memory instead of supplied evidence. They require attention,
hidden components, or trained probes and should be reported as white-box or
supervised baselines.

### 5. Time-series log-probability detectors

HALT uses top-20 token log-probabilities and entropy features with a trained
GRU. It is close to our telemetry contract, but it is supervised and mainly
evaluates response-level hallucination across a broad benchmark. It is a useful
learned ceiling, not yet a direct RAG span baseline.

### 6. Evidence-Contrast U-PCR / EC-DUFS-LIU

Our planned method keeps a published answer fixed and constructs dependent
views from:

- full context;
- no context;
- leave-one-chunk-out context;
- optionally, controlled replacement or citation-specific removal.

For token `t` and chunk `j`, one basic view is

\[
\Delta \operatorname{NLL}_{j,t}
=
\operatorname{NLL}_t(y\mid C\setminus c_j)
-
\operatorname{NLL}_t(y\mid C).
\]

Related views can use entropy, probability margin, tail mass, or distribution
divergence. The global head summarizes the intervention traces for response
detection. The local head preserves their token or window resolution for span
localization. DUFS-LIU then fuses redundant or unreliable intervention views.

This method is planned, not implemented. Its scientific question is whether
fusion adds transferable signal beyond GASP's strongest single or trained
combination of perturbation features.

## Fair comparison categories

| Category | Methods | What must be disclosed |
|---|---|---|
| Training-free evidence perturbation | GASP, planned EC-U-PCR | Number of rescoring conditions, scoring model, threshold rule |
| Unsupervised gray-box fusion | Planned EC-DUFS-LIU | Token statistics, fusion fit, graph choices, no-label freeze |
| Supervised token detector | RAGTruth detector, LettuceDetect | Training data, overlap with evaluation, model size |
| External verifier | RT4CHART, NLI or LLM judges | Verifier model, prompts, claim decomposition, inference cost |
| White-box mechanistic | ReDeEP, Lookback Lens | Required internals, trained probes, transfer model |
| Supervised telemetry model | HALT | Training labels, GRU size, top-k log-probability access |

## Recommended benchmark sequence

### Stage 1: RAGTruth

Use the exact published responses. Rescore them under fixed evidence
conditions. Group all splits and uncertainty by `source_id`. Freeze feature
definitions, unsupervised parameters, and score hashes before opening labels.

Required comparisons:

1. full-context DUFS-LIU without evidence interventions;
2. perplexity and simple full-versus-no-context likelihood drop;
3. a faithful GASP implementation;
4. EC-U-PCR;
5. EC-DUFS-LIU;
6. LettuceDetect or the original RAGTruth detector as a supervised ceiling;
7. RT4CHART as an external-verifier ceiling when its exact data split is
   compatible.

### Stage 2: confirmation and falsification

- **RAGTruth++ or RAGTruth-Enhance:** test sensitivity to denser labels.
- **TofuEval:** test evidence-built long-form transfer.
- **RAGBench:** test the predicted failure on short answers recoverable from
  model memory.
- **TRIVIA+:** test long contexts and robustness to label noise.
- **L-CiteEval:** add only when the output explicitly predicts citation
  correctness or completeness.

## Failure tests that must be measured

1. The answer is factually correct from model memory but unsupported by the
   supplied context.
2. Two chunks contain the same evidence, so removing either one has little
   effect.
3. Removing an irrelevant chunk changes scores through prompt length or
   attention position alone.
4. The detector mainly predicts answer length, chunk count, or context length.
5. A separate scoring model measures its own beliefs rather than the
   generator's evidence use.
6. Retrieval failure and generation grounding failure are merged into one
   label.
7. Several responses from the same source leak across selection and
   evaluation.
8. The method performs well only on the incomplete original RAGTruth labels.

## Decision

The repository already had the correct high-level RAG idea. The updated field
map sharpens it:

> GASP pre-empts evidence removal as the novelty. EC-DUFS-LIU is justified only
> if unsupervised fusion improves over GASP across sources and transfers beyond
> RAGTruth. RAGTruth is the primary span benchmark; TRIVIA+ is the strongest
> newly identified long-context and label-noise confirmation candidate;
> L-CiteEval is needed only for an explicit citation claim.

Do not use the old Phase-10 cache for a publishable grounding claim. It remains
an engineering smoke-test resource.

## Primary sources

- RAGTruth: <https://aclanthology.org/2024.acl-long.585/>
- RAGTruth official data: <https://github.com/ParticleMedia/RAGTruth>
- RAGTruth++ dataset card: <https://huggingface.co/datasets/blue-guardrails/ragtruth-plus-plus>
- GASP: <https://arxiv.org/abs/2607.04223>
- RT4CHART: <https://arxiv.org/abs/2603.27752>
- TofuEval: <https://arxiv.org/abs/2402.13249>
- RAGBench: <https://arxiv.org/abs/2407.11005>
- TRIVIA+: <https://arxiv.org/abs/2605.11330>
- L-CiteEval: <https://arxiv.org/abs/2410.02115>
- FACTS Grounding: <https://arxiv.org/abs/2501.03200>
- LettuceDetect: <https://arxiv.org/abs/2502.17125>
- ReDeEP: <https://arxiv.org/abs/2410.11414>
- SEReDeEP: <https://arxiv.org/abs/2505.07528>
- Lookback Lens: <https://arxiv.org/abs/2407.07071>
- HALT: <https://arxiv.org/abs/2602.02888>

## Earlier repository research retained for audit

- `docs/research_notes/research_phase10_rag/outline.yaml`
- `docs/research_notes/research_phase10_rag/*.json`
- `docs/research_notes/evidence_contrast_upcr_rag_direction.md`
- `papers/digests/ragtruth-a-hallucination-corpus-for-developing-trustworthy-r.md`

The Phase-10 files are the recovered raw research notes. This document is the
current decision-oriented map.
