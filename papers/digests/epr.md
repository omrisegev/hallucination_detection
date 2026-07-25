---
slug: epr
title: "Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate"
authors: "Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, Emmanuel Malherbe"
arxiv_id: "arXiv:2509.04492v2"
venue: "arXiv:2509.04492v2"
year: 2025
source_pdf: papers/EPR.pdf
extracted_text: papers/extracted/epr.md
last_digested: 2026-07-13
---

## Summary

Introduces a methodology for robust, one-shot hallucination detection in black-box LLMs that expose only top-K candidate log-probabilities. Derives an Entropy Production Rate (EPR) baseline and a learned Weighted Entropy Production Rate (WEPR) estimator over accessible log-probabilities.

## Datasets & models used

TriviaQA (Wikipedia domain), WebQuestions, and ArGiMi-Ardian Finance 10k dataset across Falcon-3, Llama, and Mistral.

## Methods it compared itself against

EPR baseline, HalluDetect, and standard log-probability uncertainty baselines.

## Experiments — methodology & scores

Evaluates hallucination detection ROC-AUC and human expert agreement.

| Setup | Method | ROC-AUC | Notes |
|---|---|---|---|
| Human Agreement Validation | WEPR | 0.75 | Cohen's Kappa = 0.898 on human annotations |
| Human Agreement Validation | HalluDetect | 0.69 | Baseline comparison |
| Human Agreement Validation | EPR Baseline | 0.61 | Unweighted entropy production rate |

## Connection to our pipeline

Core reference defining the EPR and WEPR metrics benchmarked against our unsupervised hallucination detection methods.

## Notes / open questions

WEPR achieves 0.75 ROC-AUC on human agreement validation using only top-K accessible log-probabilities (K=15).
