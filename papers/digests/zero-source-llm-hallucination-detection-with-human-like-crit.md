---
slug: zero-source-llm-hallucination-detection-with-human-like-crit
title: "Zero-source LLM Hallucination Detection with Human-like Criteria Probing"
authors: "Jiahao Yang, Shuhai Zhang, Hailong Kang, Feng Liu, Qi Chen, Mingkui Tan"
arxiv_id: "arXiv:2606.12900"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Zero-source LLM Hallucination Detection with Human-like Criteria Probing.pdf
extracted_text: papers/extracted/zero-source-llm-hallucination-detection-with-human-like-crit.md
last_digested: 2026-07-13
---

## Summary

Proposes Human-like Criteria Probing for Hallucination Detection (HCPD), an agentic paradigm that emulates multi-perspective human verification criteria to detect hallucinations under strict zero-source constraints (no model internals or external grounding).

## Datasets & models used

TriviaQA, SciQ, NQ Open, and CoQA, evaluated on LLaMA-3.1-8B and Qwen-3-8B.
(Not HaluEval/TruthfulQA/FactScore — none of those three appear anywhere in the paper;
corrected 2026-07-13, see below.)

## Methods it compared itself against

LN-Entropy, Self-evaluation (Kadavath et al.), CCS, SelfCheckGPT, Perplexity, SAPLMA,
Semantic Entropy, Lexical Similarity, EigenScore, HaloScope, TAD, TSV.

## Experiments — methodology & scores

Evaluates AUROC (%) on Llama-3.1-8b / Qwen-3-8b (Table 2):

| Method | TriviaQA | SciQ | NQ Open | CoQA | Avg. (Llama) |
|---|---|---|---|---|---|
| Perplexity | 80.62 | 66.12 | 57.92 | 81.41 | 71.52 |
| SAPLMA (supervised) | 78.51 | 85.63 | 76.23 | 71.58 | 77.99 |
| Semantic Entropy | 78.71 | 77.81 | 61.04 | 75.26 | 73.21 |
| TSV (supervised) | 79.78 | 80.01 | 70.17 | 69.31 | 74.82 |
| **HCPD (Ours)** | **86.25** | **86.04** | **90.38** | **90.07** | **88.19** |

HCPD beats the strongest unsupervised baseline by ~11-14 AUROC points and the strongest
(fully-labeled-trained) supervised baseline by ~10 points on average.

## Connection to our pipeline

Complements our agreement-based unsupervised verifier ensembling by structuring multi-criteria agreement.

## Notes / open questions

Published at ICML 2026 (confirmed: "Proceedings of the 43rd International Conference on
Machine Learning... PMLR 306, 2026" on p.1).

**Correction (2026-07-13)**: original digest fabricated the datasets ("HaluEval, TruthfulQA,
FactScore" — zero occurrences of any of these three terms in the extracted text) and omitted
the real, heavily-used dataset (TriviaQA, 28 occurrences). Results table was also missing;
added grounded Table 2 numbers above.
