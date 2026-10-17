---
slug: diverging-flows-2602-13061v2
title: "Native Extrapolation Awareness in Flow-Based Conditional Generation"
authors: "Constantinos Tsakonas; Serena Ivaldi; Jean-Baptiste Mouret"
arxiv_id: "2602.13061v2"
venue: "Preprint"
year: 2026
source_pdf: papers/diverging-flows-2602.13061v2.pdf
extracted_text: papers/extracted/diverging-flows-2602-13061v2.md
last_digested: 2026-09-15
---

## Summary

Flow matching gains contrastive velocity-magnitude and angular margins for
synthetic negative conditions. Generated-path deviation supplies an extrapolation
score (PDF pp.4–6; equations 4–8).

## Datasets & models used

Synthetic manifolds, ERA5 weather, and cross-domain style transfer; not reasoning
correctness (pp.7–10).

## Methods it compared itself against

Standard FM likelihood/DOT and an FM ensemble, among other task-specific controls.

## Experiments — methodology & scores

Synthetic regression, three seeds, Table 1 (p.8):

| Method | AUROC |
|---|---:|
| FM likelihood | .566 ± .02 |
| FM DOT | .594 ± .02 |
| DiFlo | .998 ± .00 |

## Connection to our pipeline

Our adaptation conditions on probability-derived token features. Predictability,
extrapolation and semantic correctness need separate evaluation. Mixed unlabeled
answers do not define a clean-correct manifold.

## Notes / open questions

Algorithm 2 sums spatial mean absolute deviations from the line between initial
noise and **generated** endpoint. No true future observation enters DOT. Compare
DOT/N across integration resolutions. Negative mining must respect feature
identities. GMM/KDE tests are not DiFlo replications. Authors verified on p.1;
the previous Bracha Laufer attribution was incorrect. PDF and HTML section
numbering differ; equations and Algorithm 2 above match the cached PDF.
