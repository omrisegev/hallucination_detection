---
slug: uncertainty-under-the-curve-a-sequence-level-entropy-area-me
title: "Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLM"
authors: "Yongfu Zhu, Lin Sun, Guangxiang Zhao, Weihong Lin, Xiangzheng Zhang (Qiyuan Tech)"
arxiv_id: "arXiv:2508.20384v1"
venue: "arXiv:2508.20384v1"
year: 2025
source_pdf: papers/Uncertainty Under the Curve A Sequence-Level Entropy Area Metric for Reasoning LLM.pdf
extracted_text: papers/extracted/uncertainty-under-the-curve-a-sequence-level-entropy-area-me.md
last_digested: 2026-07-13
---

## Summary

Introduces Entropy Area Score (EAS) / Uncertainty Under the Curve, integrating token-level predictive entropy over the answer generation trajectory without external models or repeated sampling.

## Datasets & models used

Mathematical reasoning benchmarks and training data filtering suites.

## Methods it compared itself against

Pass Rate filtering, mean token entropy, and repeated sampling uncertainty metrics.

## Experiments — methodology & scores

Evaluates correlation with answer entropy and student model accuracy gains via training data selection.

| Application Setup | Method | Performance | Notes |
|---|---|---|---|
| Training Data Selection | EAS / UUC Filtering | Outperforms Pass Rate filtering | Improves student model math benchmark accuracy under equal sample budgets |

## Connection to our pipeline

Direct sequence-level entropic area competitor to our token-level EPR and trace dynamic features.

## Notes / open questions

EAS requires only single-pass token predictive entropy.
