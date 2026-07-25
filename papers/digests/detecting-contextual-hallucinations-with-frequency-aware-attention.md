---
slug: detecting-contextual-hallucinations-with-frequency-aware-attention
title: "Detecting Contextual Hallucinations in Large Language Models with Frequency-Aware Attention"
authors: "Siya Qi et al., Harbin Institute of Technology"
arxiv_id: "2604.18647"
venue: "ICML 2026"
year: 2026
source_pdf: not downloaded
extracted_text: not extracted
last_digested: 2026-07-15
---

## Summary

This paper proposes a training-free contextual hallucination detector based on the frequency components of attention distributions. The authors model attention weights across decoding steps as discrete signals. By applying signal processing (FFT), they show that hallucinated tokens exhibit higher "high-frequency attention energy," reflecting fragmented and unstable visual/textual grounding. They use this frequency-aware attention energy to build a lightweight detector.

## Datasets & models used

- **Datasets:** RAGTruth, HalluRAG.
- **Models:** LLaMA-3.1-8B, Qwen-2.5-7B, etc.

## Methods it compared itself against

- **Baselines:** Lookback-Lens, attention variance/entropy, verification-based methods (similarity/LLM-as-a-judge), and internal-representation-based methods.

## Experiments — methodology & scores

The method is evaluated on hallucination detection AUROC across RAG benchmarks. Frequency-aware attention energy consistently outperforms standard attention entropy and Lookback-Lens.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| RAGTruth (Llama-8B) | AUROC (%) | **Significant lift** vs Lookback-Lens | Exact scores not extracted |

## Connection to our pipeline

- **Overlap:** Both extract frequency/spectral features of decoding traces (we use FFT of entropy $H(n)$ traces, they use FFT of attention maps).
- **Difference:** They are **white-box** (require attention maps), while we are **gray-box** (require logits only, $K=1$), making us much more computationally efficient and API-compatible.
- **Competitor:** Yes, on RAGTruth (we score 87.7% on Llama-8B).

## Notes / open questions

Unresolved: whether combining attention energy with our logprob energy features can provide further performance gains.
