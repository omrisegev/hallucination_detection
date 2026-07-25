---
slug: harnessing-reasoning-trajectories-for-hallucination-detectio
title: "Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping"
authors: "Jianxiong Zhang, Bing Guo, Yuming Jiang, Haobo Wang, Bo An, Sean Du"
arxiv_id: "arXiv:2601.17467"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping.pdf
extracted_text: papers/extracted/harnessing-reasoning-trajectories-for-hallucination-detectio.md
last_digested: 2026-07-13
---

## Summary

Proposes Answer-agreement Representation Shaping (ARS), which shapes internal trace representations of Large Reasoning Models (LRMs) so they reflect answer agreement and stability across trajectory steps rather than superficial trace phrasing.

## Datasets & models used

TruthfulQA, TriviaQA (conversational/open-domain QA) and GSM8K, MATH-500 (multi-step math
reasoning), evaluated on Qwen3-8B and DeepSeek-R1-Distill-Llama-8B.
(Not GPQA — GPQA has zero occurrences in the extracted text; corrected 2026-07-13.)

## Methods it compared itself against

Perplexity, Semantic Entropy, Lexical Similarity, SelfCheckGPT, Verbalized Certainty, TSV
(supervised), plus LRM-specific baselines RHD, RACE, G-Detector (supervised).

## Experiments — methodology & scores

AUROC (%) on Qwen3-8B (Table 1), ARS used as features for a CCS or supervised probing detector:

| Method | TruthfulQA | TriviaQA | GSM8K | MATH-500 |
|---|---|---|---|---|
| Semantic Entropy | 65.60 | 58.37 | 72.51 | 56.13 |
| TSV (supervised) | 77.08 | 89.67 | 83.15 | 63.12 |
| G-Detector (supervised, LRM) | 71.86 | 90.52 | 83.78 | 57.67 |
| **ARS (CCS), unsupervised** | **86.64** | 88.54 | **90.37** | **78.66** |
| **ARS (Probing), supervised** | 83.66 | **91.62** | 89.88 | 78.17 |

ARS(CCS) — unsupervised — beats even the supervised TSV/G-Detector baselines on 3 of 4
benchmarks, and by a wide margin on MATH-500 (+15.5pp over G-Detector).

## Connection to our pipeline

Highly relevant to our reasoning trace analysis on GSM8K/MATH500/GPQA, demonstrating agreement representation shaping.

## Notes / open questions

Published at ICML 2026 (confirmed: "Proceedings of the 43rd International Conference on
Machine Learning... PMLR 306, 2026" on p.1).

**Correction (2026-07-13)**: original digest fabricated GPQA as a dataset (0 occurrences in the
extracted text) and omitted the real QA-domain datasets (TruthfulQA, TriviaQA) and the actual
models (Qwen3-8B, DeepSeek-R1-Distill-Llama-8B). Results table was also missing; added grounded
Table 1 numbers above.
