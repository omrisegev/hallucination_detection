---
slug: spilled-energy-in-large-language-models
title: "SPILLED ENERGY IN LARGE LANGUAGE MODELS"
authors: "Adrian R. Minut, Hazem Dewidar, Iacopo Masi"
arxiv_id: "arXiv:2602.18671v4"
venue: "ICLR 2026"
year: 2026
source_pdf: papers/Spilled Energy in Large Language Models.pdf
extracted_text: papers/extracted/spilled-energy-in-large-language-models.md
last_digested: 2026-07-13
---

## Summary

Reinterprets the final LLM softmax classifier as an Energy-Based Model (EBM), tracking 'spilled energy' (discrepancies between energy values across consecutive generation steps) directly from output logits as a training-free metric for factual errors and hallucinations.

## Datasets & models used

Nine benchmarks across state-of-the-art LLMs (LLaMA, Mistral, Gemma) and synthetic algebraic operations (Qwen3).

## Methods it compared itself against

Trained probe classifiers, activation ablations, and standard softmax confidence.

## Experiments — methodology & scores

Evaluates answer token error localization and hallucination detection across nine benchmarks.

| Setup | Method | Evaluation Property | Notes |
|---|---|---|---|
| 9 Benchmarks (LLaMA/Mistral/Gemma) | Spilled Energy & Marginalized Energy | Completely training-free error correlation | Requires no trained probes or activation ablations |

## Connection to our pipeline

Directly relates to our energy and trace-level uncertainty metrics, providing a training-free logit energy discrepancy formulation.

## Notes / open questions

Published conference paper at ICLR 2026.
