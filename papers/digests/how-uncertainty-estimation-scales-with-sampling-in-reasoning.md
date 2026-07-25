---
slug: how-uncertainty-estimation-scales-with-sampling-in-reasoning
title: "How Uncertainty Estimation Scales with Sampling in Reasoning Models"
authors: "Maksym Del, Markus Kängsepp, Marharyta Domnich, Ardi Tampuu, Lisa Yankovskaya, Meelis Kull, Mark Fishel"
arxiv_id: "arXiv:2603.19118v1"
venue: "arXiv:2603.19118v1"
year: 2026
source_pdf: papers/How Uncertainty Estimation Scales with Sampling in Reasoning Models.pdf
extracted_text: papers/extracted/how-uncertainty-estimation-scales-with-sampling-in-reasoning.md
last_digested: 2026-07-13
---

## Summary

Studies parallel sampling as a black-box uncertainty estimation approach using verbalized confidence and self-consistency across reasoning models. Characterizes how uncertainty signals scale across sampling budgets.

## Datasets & models used

17 tasks spanning mathematics, STEM, and humanities across three reasoning models.

## Methods it compared itself against

Verbalized confidence alone, self-consistency alone, and hybrid signal combinations.

## Experiments — methodology & scores

Evaluates AUROC improvements under parallel sampling budgets.

| Setup | Method | AUROC Gain | Notes |
|---|---|---|---|
| 2 Parallel Samples (N=2) | Hybrid Estimator (Verbalized + Self-Consistency) | up to +12 AUROC on average | Outperforms either signal alone at minimal sample budget |

## Connection to our pipeline

Key guide for configuring sampling budgets in our pipeline, proving that combining verbalized confidence with self-consistency at N=2 yields up to +12 AUROC gain.

## Notes / open questions

Demonstrates high sample efficiency for black-box uncertainty estimation.
