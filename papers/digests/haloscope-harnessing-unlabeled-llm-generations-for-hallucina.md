---
slug: haloscope-harnessing-unlabeled-llm-generations-for-hallucina
title: "HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection"
authors: "Xuefeng Du, Chaowei Xiao, Yixuan Li (University of Wisconsin-Madison)"
arxiv_id: "arXiv:2409.17504"
venue: "NeurIPS 2024"
year: 2024
source_pdf: papers/HaloScope Harnessing Unlabeled LLM Generations for Hallucination Detection.pdf
extracted_text: papers/extracted/haloscope-harnessing-unlabeled-llm-generations-for-hallucina.md
last_digested: 2026-07-13
---

## Summary

Introduces HaloScope, an unsupervised learning framework that leverages unlabeled LLM generations in the wild for hallucination detection. Uses positive-unlabeled (PU) and contrastive representation learning over free unlabeled generations to estimate factual correctness without human labels.

## Datasets & models used

TruthfulQA, HaluEval, and TriviaQA across LLaMA-2, Vicuna, and Mistral models.

## Methods it compared itself against

Supervised hallucination classifiers, self-evaluation prompting, and zero-shot predictive entropy.

## Experiments — methodology & scores

Evaluates AUROC and precision-recall across out-of-distribution factual QA tasks.

| Setup | Method | Result | Notes |
|---|---|---|---|
| Unlabeled Generations | HaloScope | Outperforms zero-shot entropy baselines | Learns factual boundary directly from unannotated model outputs |

## Connection to our pipeline

Strongly aligns with our unsupervised ensembling and zero-labeled verification ethos (L-SML), providing a representation-level unsupervised alternative.

## Notes / open questions

Published at NeurIPS 2024.
