---
slug: a-deep-learning-approach-to-unsupervised-ensemble-learning
title: "A Deep Learning Approach to Unsupervised Ensemble Learning"
authors: "Uri Shaham, Xiuyuan Cheng, Omer Dror, Ariel Jaffe, Boaz Nadler, Joseph Chang, and Yuval Kluger"
arxiv_id: "arXiv:1602.02285v1"
venue: "arXiv:1602.02285v1"
year: 2016
source_pdf: papers/A Deep Learning Approach to Unsupervised Ensemble Learning.pdf
extracted_text: papers/extracted/a-deep-learning-approach-to-unsupervised-ensemble-learning.md
last_digested: 2026-07-13
---

## Summary

Proposes a Restricted Boltzmann Machine (RBM)-based Deep Neural Net (DNN) approach for unsupervised ensemble learning and crowdsourcing. Proves that the Dawid-Skene conditional independence model is equivalent to an RBM with a single hidden node, and extends this via DNNs to handle strong violations of conditional independence among base classifiers.

## Datasets & models used

Synthetic conditionally independent/dependent datasets, DREAM datasets (S1, S2, S3), and Magic gamma telescope datasets (40 subsets of size 500 across 16 Weka classifiers: Random Forests, Logistic Trees, SVMs, Naive Bayes).

## Methods it compared itself against

Dawid and Skene (DS), Spectral Meta-Learner (L-SML), CUBAM, Majority Vote (Vote).

## Experiments — methodology & scores

Evaluated on balanced accuracy across synthetic and real-world ensemble predictions.

| Dataset Setup | Method | Balanced Accuracy (%) | Notes |
|---|---|---|---|
| Synthetic condInd | Vote | 75.93 ± 0.5 | Baseline majority voting |
| Synthetic condInd | DS | 94.78 ± 0.13 | Conditionally independent Dawid-Skene |
| Synthetic Tree15-3-1 | Vote | 93.45 ± 0.19 | Dependent tree ensemble |
| Synthetic Tree15-3-1 | DS | 92.68 ± 0.14 | Performance under dependency |

## Connection to our pipeline

Directly relevant to our unsupervised ensemble learning (L-SML) pipeline. Shows how non-linear RBM/DNN aggregators handle classifier dependencies compared to linear spectral recovery.

## Notes / open questions

Verify whether the single-hidden-node RBM equivalence can be applied to LLM verifier preference scores.
