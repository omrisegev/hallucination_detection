---
slug: semantic-energy-detecting-llm-hallucination-beyond-entropy
title: "Semantic Energy: Detecting LLM Hallucination Beyond Entropy"
authors: "Huan Ma, Jiadong Pan, Jing Liu, Yan Chen, Joey Tianyi Zhou, Guangyu Wang, Qinghua Hu, Hua Wu, Changqing Zhang, Haifeng Wang"
arxiv_id: "arXiv:2508.14496v3"
venue: "arXiv:2508.14496v3"
year: 2025
source_pdf: papers/Semantic Energy Detecting LLM Hallucination Beyond Entropy.pdf
extracted_text: papers/extracted/semantic-energy-detecting-llm-hallucination-beyond-entropy.md
last_digested: 2026-07-13
---

## Summary

Proposes Semantic Energy, an energy-based formulation that evaluates probability mass sharpness across semantic clusters of generated responses to detect hallucinations beyond standard token or sequence entropy.

## Datasets & models used

Free-form QA and factual recall tasks evaluated across open-weight LLMs.

## Methods it compared itself against

Semantic Entropy (Kuhn et al.), predictive entropy, and lexical similarity clustering.

## Experiments — methodology & scores

Evaluates AUROC for discriminating factual vs hallucinated generations.

| Setup | Method | Metric | Observation |
|---|---|---|---|
| Free-form Factual QA | Semantic Energy | ROC-AUC | Outperforms standard Semantic Entropy by separating clustered probability mass |

## Connection to our pipeline

Key literature baseline comparing energy-based semantic clustering against our trace entropy and spectral recovery methods.

## Notes / open questions

Complements token-level EPR by analyzing energy across semantic clusters.
