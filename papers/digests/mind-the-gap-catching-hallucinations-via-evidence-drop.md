---
slug: mind-the-gap-catching-hallucinations-via-evidence-drop
title: "Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning Manifold"
authors: "Qunjie Chen et al., Tongji University"
arxiv_id: "OpenReview/ICML 2026"
venue: "ICML 2026"
year: 2026
source_pdf: not downloaded
extracted_text: not extracted
last_digested: 2026-07-15
---

## Summary

This paper models the multi-step reasoning process as a trajectory on a latent "Evidence Manifold," where each reasoning step should be supported by local evidence. Hallucinations are defined as "Evidence Drops"—sudden, localized declines in evidence support. The authors design a training-free, model-agnostic detector that monitors for the worst-case Evidence Drop, enabling both response-level correctness prediction and step-level error localization.

## Datasets & models used

- **Datasets:** GSM8K, MATH, ProcessBench.
- **Models:** LLaMA-3.1-8B, Qwen-2.5-7B-Instruct, etc.

## Methods it compared itself against

- **Baselines:** Sequence-level uncertainty metrics (Semantic Entropy, LN-Entropy, Perplexity, SelfCheckGPT).

## Experiments — methodology & scores

Evaluated on selective accuracy, risk-coverage trade-offs, and AUROC for correctness prediction.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| GSM8K / MATH | AUROC (%) | **Outperforms sequence-level uncertainty** | Beats standard Semantic Entropy |

## Connection to our pipeline

- **Overlap:** Direct competitor targeting reasoning benchmarks (GSM8K, MATH) with a training-free, model-agnostic approach.
- **Difference:** We use spectral features of $H(n)$ traces (sliding-window variance, EPR, CUSUM) to get a sequence-level score via unsupervised L-SML fusion. They look at semantic/contextual evidence drops directly to localize errors at the step level.
- **Competitor:** Yes, direct competitor on GSM8K/MATH.

## Notes / open questions

Their step-level error localization on ProcessBench represents a key capability we should benchmark against.
