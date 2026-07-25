---
slug: efficient-hallucination-detection-for-llms-using-uncertainty
title: "Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads"
authors: "Artem Vazhentsev, Lyudmila Rvanova, Gleb Kuzmin, Ekaterina Fadeeva, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin, Artem Shelmanov"
arxiv_id: "arXiv:2505.20045"
venue: "ICML 2026 (main track — NOT ICLR 2026)"
year: 2026
source_pdf: papers/Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads.pdf
extracted_text: papers/extracted/efficient-hallucination-detection-for-llms-using-uncertainty.md
last_digested: 2026-07-13
---

## Summary

Introduces Recurrent Attention-based Uncertainty Quantification (RAUQ), an unsupervised framework that identifies uncertainty-aware attention heads whose attention weights disperse prior to generating hallucinations.

## Datasets & models used

Twelve datasets spanning QA, summarization (Summ), and machine translation (MT), across nine
LLMs including Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B, and Falcon-3-10B.
(Not "TriviaQA, TruthfulQA, SQuAD across LLaMA-2/Mistral" — that was a generic placeholder;
corrected 2026-07-13.)

## Methods it compared itself against

MSP, Perplexity, CCP, Attention Score, Focus, Simple Focus, DegMat/Ecc./EVL NLI-Score, Lexical
Similarity, EigenScore, LUQ, Semantic Entropy, SAR, Semantic Density — 13 UQ baselines.

## Experiments — methodology & scores

Mean PRR (higher is better) across QA/Summ/MT, averaged over 4 of the 9 evaluated LLMs (Table 1):

| Method | Mean PRR |
|---|---|
| Perplexity | .357 |
| MSP | .318 |
| Focus | .317 |
| Simple Focus | .326 |
| Semantic Entropy | .240 |
| EigenScore | .199 |
| **RAUQ** | **.384** |

RAUQ requires <1% additional compute (single forward pass, no extra sampling) and consistently
beats the prior state of the art on QA and translation tasks across all evaluated LLMs.

## Connection to our pipeline

Directly aligns with our token entropy and trace monitoring by showing attention dispersion predicts confabulations.

## Notes / open questions

**Correction (2026-07-13)**: original digest claimed "venue: ICLR 2026," but the paper's own
header states "Proceedings of the 43rd International Conference on Machine Learning... PMLR
306, 2026" — this is ICML 2026, not ICLR. Datasets/models list was also a generic placeholder
that didn't match the actual 12-dataset/9-LLM/3-task (QA+Summ+MT) evaluation; corrected above.
