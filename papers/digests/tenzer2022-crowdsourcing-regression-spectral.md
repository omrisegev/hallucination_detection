---
slug: tenzer2022-crowdsourcing-regression-spectral
title: "Unsupervised Ensemble Regression"
authors: "Omer Dror, Boaz Nadler, Erhan Bilal, Yuval Kluger"
arxiv_id: "arXiv:1703.02965v1"
venue: "arXiv:1703.02965v1 [stat.ML]"
year: 2017
source_pdf: Tenzer2022_Crowdsourcing_Regression_Spectral.pdf
extracted_text: papers/extracted/tenzer2022-crowdsourcing-regression-spectral.md
last_digested: 2026-07-13
---

## Summary

Proposes an unsupervised framework for ensemble regression that estimates unknown target responses and detects least/most accurate experts from unlabeled continuous predictions by analyzing prediction covariance moments under uncorrelated error assumptions.

## Datasets & models used

Unlabeled regression expert ensembles across synthetic and real prediction datasets.

## Methods it compared itself against

Simple mean averaging, median aggregation, and supervised regression weighting.

## Experiments — methodology & scores

Evaluates Mean Squared Error reduction via covariance moment factorization.

| Setup | Method | Result | Notes |
|---|---|---|---|
| Unlabeled Regression Ensembles | Covariance Spectral Estimator | Provably consistent error variance estimation | Recovers optimal MSE weights without ground truth targets |

## Connection to our pipeline

Core foundational paper defining unsupervised spectral regression weighting from prediction covariance matrices.

## Notes / open questions

Note: file is named Tenzer2022 at repo root but contains Dror, Nadler, Bilal, Kluger (2017) Unsupervised Ensemble Regression.
