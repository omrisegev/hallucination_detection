---
slug: gasp-detecting-hallucinations-in-retrieval-augmented-generat
title: "Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation (GASP)"
authors: "Mohamed Aly Bouke, Multimedia University, Malaysia"
arxiv_id: "arXiv:2607.04223v1 [cs.CL]"
venue: "Research Article (arXiv-only, submitted 5 Jul 2026)"
year: 2026
source_pdf: papers/GASP - Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation.pdf
extracted_text: papers/extracted/gasp-detecting-hallucinations-in-retrieval-augmented-generat.md
last_digested: 2026-08-09
---

## Summary

GASP is a span-level RAG hallucination detector that holds a generated answer fixed and
re-scores it under the full retrieved context, no context, and each context chunk removed in
turn (K+2 forward passes, no generation). It computes four "grounding sensitivity" features
from the resulting log-likelihoods and predictive distributions — full-vs-noctx likelihood
gap, full-vs-noctx JSD, max leave-one-out likelihood drop, max leave-one-out JSD (Eqs. 8–11)
— on the theory (cast as a random nonlinear iterated function system) that a grounded span's
likelihood collapses when its supporting evidence is removed while a hallucinated span is
nearly unaffected. The paper's own reported **default detector, "GASP-threshold," is
training-free**: it standardizes the four features and thresholds their negated sum — no
labeled data needed. An optional supervised variant ("GASP-trained", LightGBM) and a combined
variant ("GASP+base", adds perplexity+length) are reported only for comparison.

## Datasets & models used

- **RAGTruth** (primary; ACL 2024) — 400 responses (200 hallucinated / 200 clean,
  class-balanced), Summarization + Data2txt task types **only** ("the two task types whose
  answers are long enough for stable estimation" — QA excluded). 2,586 sentences (Qwen
  tokenizer) / 2,550 (SmolLM2).
- **TofuEval** (MeetingBank portion) — 884 summaries, 2,401 sentences, transfer-domain check.
- **RAGBench** — 797 responses pooled across 6 domains, 3,858 sentences, class-balanced,
  short-answer QA (probes whether the signal transfers to answers not necessarily
  constructed from context).
- **Scorers**: Qwen2.5-0.5B-Instruct, **Qwen2.5-1.5B-Instruct**, SmolLM2-1.7B-Instruct — all
  run forward-pass-only (no generation), on a single 6 GB consumer GPU. Context capped at
  700 tokens, answer at 200 tokens, K=5 chunks (sentence-grouped).

## Methods it compared itself against

Perplexity (mean answer surprisal, full context), answer length, NLI entailment
(`cross-encoder/nli-deberta-v3-{small,large}`, both whole-context and max-over-K=5-chunks),
and a SelfCheckGPT-style self-consistency baseline (N=4 stochastic regenerations, scored by
average contradiction probability under the compact NLI model). LettuceDetect (supervised,
trained on RAGTruth) and the mechanistic white-box detectors (ReDeEP, LUMINA) are discussed
but explicitly **not** run head-to-head — different access regime (trained-on-target-corpus /
white-box) than GASP's black-box training-free setting.

## Experiments — methodology & scores

5-fold cross-validation (response level: stratified; span level: stratified **grouped by
response**, so sentences from one answer never split across train/test — "leakage-clean").
95% bootstrap CIs over 1000 grouped resamples. AUC is the metric throughout (threshold-free,
prior-insensitive).

| Setup | Metric | Score | Notes |
|---|---|---|---|
| RAGTruth, Qwen2.5-1.5B, GASP-threshold | Response AUC | **0.713** | training-free default; our scorer |
| RAGTruth, Qwen2.5-1.5B, GASP-trained (LightGBM) | Response AUC | 0.726 | supervised, reported for comparison only |
| RAGTruth, Qwen2.5-1.5B, perplexity baseline | Response AUC | 0.624 | |
| RAGTruth, Qwen2.5-0.5B, GASP-threshold | Response AUC | 0.745 | |
| RAGTruth, SmolLM2-1.7B, GASP-threshold | Response AUC | 0.741 | |
| RAGTruth, Qwen2.5-1.5B, GASP-threshold | Span AUC | **0.673** | our scorer |
| RAGTruth, Qwen2.5-1.5B, GASP-trained | Span AUC | 0.645 | |
| RAGTruth, Qwen2.5-1.5B, perplexity baseline | Span AUC | 0.565 | |
| RAGTruth, Qwen2.5-0.5B, GASP-threshold | Span AUC | 0.672 | |
| RAGTruth, SmolLM2-1.7B, GASP-threshold | Span AUC | 0.681 | |
| RAGTruth (all 3 scorers), abstract-level average | Response / Span AUC | ~0.73 / ~0.67 | the numbers quoted in our own manifests/preregistration are this cross-scorer average, not the Qwen2.5-1.5B-specific 0.713/0.673 |

Only a chunk-level entailment verifier is span-level-competitive with GASP; GASP-threshold
(no labels) matches GASP-trained (labels) closely enough that the paper reports the
training-free variant as its recommended default throughout.

## Connection to our pipeline

This is the closest direct competitor to `docs/research_notes/ragtruth_ec_preregistration_v1.md`
("Evidence-Contrast U-PCR/DUFS-LIU on RAGTruth", Step 237, 2026-08-09) — same corpus, same
full/no-context/leave-one-chunk-out intervention design, same scorer class (we picked
Qwen2.5-1.5B-Instruct specifically to match one of GASP's own three tested scorers). Our
reproduction (`scripts/rag_ec_v1/gasp.py`) implements GASP-threshold exactly per Eqs. 8–11,
declared at **comparison level 2** (same protocol, different/larger sampling — we score the
full RAGTruth test split across all 3 task types including QA, not the paper's curated
400-response Summarization+Data2txt-only subsample). **0.713 response AUC / 0.673 span AUC
(Qwen2.5-1.5B) are the specific numbers our own reproduction should be checked against once
labels open** — not the rounder abstract-level ~0.73/~0.67 that mixes in the other two
scorers. Our own campaign's entire novelty claim (per `cluster/manifests/ragtruth_ec_v1.json`,
arm 6 "fusion_isolation_ablation") rests on beating a naive-average baseline over the SAME
evidence-contrast features GASP already perturbs — i.e. on showing DUFS-LIU/U-PCR fusion adds
something GASP-threshold's fixed equal-weighted sum doesn't, since GASP already does the
intervention design itself.

## Notes / open questions

- **GASP does NOT test on QA** ("not long enough for stable estimation") — our round-1 scope
  explicitly includes QA (per `PROVENANCE.md`'s finding that QA always has exactly 3 chunks,
  the most reliable chunking of the three task types). This means our full-corpus numbers are
  not directly comparable to GASP's on the QA slice specifically; worth reporting the
  QA/Data2txt/Summary breakdown separately rather than only a pooled number.
  Their K=5 sentence-grouped chunking is a different unit from our per-task-type chunking
  (QA=3 passages, Data2txt=9 JSON fields, Summary=1 document) — a disclosed protocol
  difference, not a bug.
- Their aggregation order for `drop`/`jsdloo` is **mean-over-span-tokens, THEN max-over-chunks**
  (Eq. 10/11) — our `spectral_utils/evidence_contrast.py`'s pre-existing `dnll_loo_max` used
  the opposite order (per-token max, then mean), which is why `scripts/rag_ec_v1/gasp.py`
  recomputes these features from scratch rather than reusing that module's aggregates; see
  the fidelity note in `gasp.py`'s own docstring.
- The paper's finest evaluation unit is a **sentence** (text-split); it has no token-level
  curve. Our local/span head evaluates against RAGTruth's own annotated `span_token_spans`, so
  `scripts/rag_ec_v1/gasp.py:token_gasp_curve` is a disclosed extension beyond what the paper
  itself validates — registered as GASP-inspired, not a reproduction claim, at that
  granularity.
