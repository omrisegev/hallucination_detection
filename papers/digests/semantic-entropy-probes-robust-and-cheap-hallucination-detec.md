---
slug: semantic-entropy-probes-robust-and-cheap-hallucination-detec
title: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
authors: "Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal (University of Oxford)"
arxiv_id: "arXiv:2406.15927"
venue: "ICML 2024"
year: 2024
source_pdf: papers/Semantic Entropy Probes Robust and Cheap Hallucination Detection in LLMs.pdf
extracted_text: papers/extracted/semantic-entropy-probes-robust-and-cheap-hallucination-detec.md
last_digested: 2026-07-13
---

## Summary

Proposes Semantic Entropy Probes (SEPs), a cheap and reliable method for uncertainty quantification and hallucination detection in LLMs. Approximates Semantic Entropy directly from internal hidden states and log-probabilities at a single generation pass, eliminating the 5x-10x cost of sampling multiple semantic clusters.

## Datasets & models used

TriviaQA, CoQA, SQuAD, and BioASQ datasets evaluated across open-weight models (LLaMA-2, Mistral).

## Methods it compared itself against

Standard Semantic Entropy (Farquhar et al.), token log-probability predictive entropy, and p(True) prompt evaluation.

## Experiments — methodology & scores

Evaluates AUROC for discriminating factual vs hallucinated generations at single-pass inference.

| Setup | Method | Metric | Observation |
|---|---|---|---|
| Single-pass Generation | SEPs | ROC-AUC | Matches standard Semantic Entropy performance at 1/5th to 1/10th the inference cost |

## Connection to our pipeline

Directly relevant to our token log-probability and hidden state trace features, showing that linear probes over internal representations predict semantic uncertainty without multi-sample sampling.

## Notes / open questions

Published at ICML 2024.
