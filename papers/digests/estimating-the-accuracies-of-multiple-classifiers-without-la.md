---
slug: estimating-the-accuracies-of-multiple-classifiers-without-la
title: "Estimating the Accuracies of Multiple Classifiers Without Labeled Data"
authors: "Ariel Jaffe, Boaz Nadler, and Yuval Kluger"
arxiv_id: "arXiv:1407.7644v2"
venue: "arXiv:1407.7644v2 [stat.ML]"
year: 2014
source_pdf: papers/Estimating the Accuracies of Multiple Classifiers Without Labeled Data.pdf
extracted_text: papers/extracted/estimating-the-accuracies-of-multiple-classifiers-without-la.md
last_digested: 2026-07-13
---

## Summary

Presents simple, computationally efficient algebraic algorithms to estimate individual classifier accuracies and construct an improved unsupervised ensemble classifier without labeled data under conditional independence assumptions.

## Datasets & models used

UCI benchmark repository datasets evaluated across 10 classification methods implemented in Weka.

## Methods it compared itself against

Majority Voting, Expectation-Maximization (Dawid-Skene), and unweighted ensemble averaging.

## Experiments — methodology & scores

Evaluates accuracy estimation error and ensemble classification accuracy across UCI datasets.

| Setup | Method | Observation / Score | Notes |
|---|---|---|---|
| UCI Ensembles (10 Classifiers) | Spectral / Algebraic Estimator | Consistent asymptotic accuracy recovery | Provably recovers individual classifier balanced accuracies |
| Ensemble Classification | Bayes Optimal Weighting | Outperforms unweighted majority vote | Weights base classifiers by estimated precision |

## Connection to our pipeline

Theoretical foundation for our Spectral Meta-Learner (SML) track and unsupervised ensemble weighting.

## Notes / open questions

Establishes spectral algebraic moment equations for unlabeled classifier agreement.
