---
slug: revisiting-hallucination-detection-with-effective-rank-based
title: "REVISITING HALLUCINATION DETECTION WITH EFFECTIVE RANK-BASED UNCERTAINTY"
authors: "Rui Wang, Zeming Wei, Guanzhang Yue, Meng Sun (Peking University)"
arxiv_id: "arXiv:2510.08389"
venue: "ICLR 2026 Submission"
year: 2026
source_pdf: papers/Revisiting Hallucination Detection with Effective Rank-based Uncertainty.pdf
extracted_text: papers/extracted/revisiting-hallucination-detection-with-effective-rank-based.md
last_digested: 2026-07-13
---

## Summary

Proposes effective rank-based uncertainty quantification for hallucination detection, analyzing the spectral rank of hidden state representations and logit covariance across decoding steps.

## Datasets & models used

Factual QA and reasoning benchmarks across open-weight LLMs.

## Methods it compared itself against

Predictive entropy, semantic entropy, and Mahalanobis distance detectors.

## Experiments — methodology & scores

Evaluates ROC-AUC for discriminating factual vs hallucinated generations.

| Setup | Method | Metric | Observation |
|---|---|---|---|
| Spectral Rank Analysis | Effective Rank Uncertainty | ROC-AUC | Outperforms standard scalar entropy by measuring representation dimensionality collapse |

## Connection to our pipeline

Bridges our spectral matrix analysis with internal representation uncertainty during LLM decoding.

## Notes / open questions

2026 ICLR submission.
