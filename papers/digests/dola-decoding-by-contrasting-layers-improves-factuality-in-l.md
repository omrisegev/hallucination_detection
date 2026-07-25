---
slug: dola-decoding-by-contrasting-layers-improves-factuality-in-l
title: "DOLA: DECODING BY CONTRASTING LAYERS IMPROVES FACTUALITY IN LARGE LANGUAGE MODELS"
authors: "Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, Pengcheng He (MIT / Microsoft)"
arxiv_id: "arXiv:2309.03883"
venue: "ICLR 2024"
year: 2024
source_pdf: papers/DoLa Decoding by Contrasting Layers Improves Factuality in Large Language Models.pdf
extracted_text: papers/extracted/dola-decoding-by-contrasting-layers-improves-factuality-in-l.md
last_digested: 2026-07-13
---

## Summary

Proposes DoLa (Decoding by Contrasting Layers), a simple training-free decoding strategy that contrasts output logit distributions between mature top layers and premature lower layers to amplify factual knowledge and suppress hallucinations.

## Datasets & models used

TruthfulQA, Factor, and StrategyQA evaluated across LLaMA family models.

## Methods it compared itself against

Standard greedy decoding, top-p sampling, and Contrastive Decoding.

## Experiments — methodology & scores

Evaluates factuality score improvements on TruthfulQA and multiple-choice benchmarks.

| Setup | Method | Metric | Result |
|---|---|---|---|
| TruthfulQA Evaluation | DoLa | Truthful+Informative % | +12% to +17% absolute improvement over standard decoding |

## Connection to our pipeline

Directly complements our trace-level structural analysis by demonstrating layerwise logit contrast evolution across transformer depth.

## Notes / open questions

Published at ICLR 2024.
