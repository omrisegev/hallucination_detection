---
slug: unsupervised-ensemble-regression
title: "Unsupervised Ensemble Regression"
authors: "Omer Dror, Boaz Nadler, Erhan Bilal, Yuval Kluger"
arxiv_id: "arXiv:1703.02965v1"
venue: "arXiv:1703.02965v1 [stat.ML]"
year: 2017
source_pdf: papers/Unsupervised Ensemble Regression.pdf
extracted_text: papers/extracted/unsupervised-ensemble-regression.md
last_digested: 2026-07-13
---

## Summary

Proposes a framework for unsupervised ensemble regression that estimates unknown continuous responses and detects least/most accurate experts from unlabeled predictions assuming uncorrelated expert deviations.

## Datasets & models used

Continuous regression expert ensembles across synthetic and real prediction datasets.

## Methods it compared itself against

Unweighted mean averaging, median aggregation, and supervised regression weighting.

## Experiments — methodology & scores

Evaluates Mean Squared Error reduction via covariance moment factorization.

| Setup | Method | Result | Notes |
|---|---|---|---|
| Unlabeled Regression Ensembles | Covariance Spectral Estimator | Provably consistent error variance estimation | Recovers optimal MSE weights without ground truth targets |

## Connection to our pipeline

Core foundational paper defining unsupervised spectral regression weighting from prediction covariance matrices.

## Notes / open questions

Establishes covariance eigenvalue equations for continuous prediction ensembling.
