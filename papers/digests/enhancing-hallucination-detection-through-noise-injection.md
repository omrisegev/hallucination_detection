---
slug: enhancing-hallucination-detection-through-noise-injection
title: "ENHANCING HALLUCINATION DETECTION THROUGH NOISE INJECTION"
authors: "Litian Liu, Reza Pourreza, Sunny Panchal, Apratim Bhattacharyya, Yubing Jian, Yao Qin, Roland Memisevic (Qualcomm AI Research)"
arxiv_id: "arXiv:2502.03799"
venue: "ICLR 2026"
year: 2026
source_pdf: papers/Enhancing Hallucination Detection through Noise Injection.pdf
extracted_text: papers/extracted/enhancing-hallucination-detection-through-noise-injection.md
last_digested: 2026-07-13
---

## Summary

Introduces a training-free inference-time method that injects controlled noise into hidden activations or weights and measures output dispersion as an indicator of model uncertainty and hallucination.

## Datasets & models used

GSM8K, CSQA, and TriviaQA, across Gemma-2B-it, Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct,
Mistral-7B-Instruct-v0.3, Llama-2-7B-chat, and Llama-2-13B-chat.
(Not TruthfulQA/CoQA — neither appears in the extracted text; corrected 2026-07-13.)

## Methods it compared itself against

Predictive Entropy, Lexical Similarity, Semantic Entropy, EigenScore, SelfCheckGPT-NLI (all
evaluated with and without the proposed noise injection).

## Experiments — methodology & scores

AUROC, injecting random uniform noise U(0, α) into upper-layer MLP activations before sampling
(Table 4, TriviaQA, noise=0 vs noise~U(0,0.09)):

| Method | No noise | + Noise injection |
|---|---|---|
| Predictive Entropy | 79.28 | 79.92 |
| Lexical Similarity | 77.40 | 78.90 |
| Semantic Entropy | 75.70 | 77.21 |
| EigenScore | 77.67 | 78.19 |
| SelfCheckGPT-NLI | 75.80 | 77.53 |

Noise injection consistently improves every uncertainty metric it's layered onto, rather than
being a standalone detector — it's a plug-in sampling-time perturbation, not a new metric.

## Connection to our pipeline

Provides a perturbation-based uncertainty metric that complements our entropy trace diagnostics.

## Notes / open questions

Published at ICLR 2026 (confirmed: "Published as a conference paper at ICLR 2026" banner
throughout).

**Correction (2026-07-13)**: original digest fabricated TruthfulQA/CoQA as datasets (0
occurrences) and omitted the real ones (GSM8K, CSQA) plus 4 of the 6 evaluated models. Results
table was also missing; added grounded Table 4 numbers above. Also corrected the framing — this
is a perturbation layered onto existing uncertainty metrics, not a standalone detector as the
original "Method" column implied.
