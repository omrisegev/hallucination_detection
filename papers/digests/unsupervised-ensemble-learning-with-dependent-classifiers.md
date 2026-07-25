---
slug: unsupervised-ensemble-learning-with-dependent-classifiers
title: "Unsupervised Ensemble Learning with Dependent Classifiers"
authors: "Ariel Jaffe, Ethan Fetaya, Boaz Nadler, Tingting Jiang and Yuval Kluger"
arxiv_id: "arXiv:1510.05830v2"
venue: "arXiv:1510.05830v2 [cs.LG]"
year: 2016
source_pdf: papers/Unsupervised Ensemble Learning with Dependent Classifiers.pdf
extracted_text: papers/extracted/unsupervised-ensemble-learning-with-dependent-classifiers.md
last_digested: 2026-07-13
---

## Summary

Introduces an unsupervised statistical model that allows for dependencies between base classifiers, developing novel unsupervised methods to detect strongly dependent classifiers, estimate their accuracies accurately, and construct an improved meta-learner.

## Datasets & models used

Artificial dependent classifier ensembles and real-world benchmark classification datasets.

## Methods it compared itself against

Conditional independence models (Dawid-Skene), independent Spectral Meta-Learner (SML), and Majority Voting.

## Experiments — methodology & scores

Evaluates accuracy estimation recovery and meta-learner classification accuracy under classifier correlation.

| Ensemble Setup | Method | Result | Notes |
|---|---|---|---|
| Dependent Classifiers | Dependent Meta-Learner | Outperforms independent SML & Majority Vote | Explicitly detects and models pairwise/graphical classifier dependencies |

## Connection to our pipeline

Core theoretical reference for handling dependent classifiers in our unsupervised ensemble learning (L-SML) and verifier ensembling track.

## Notes / open questions

Overcomes conditional independence assumptions that degrade standard spectral estimators.
