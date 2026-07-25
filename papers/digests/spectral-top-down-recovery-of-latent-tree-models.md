---
slug: spectral-top-down-recovery-of-latent-tree-models
title: "Spectral Top-Down Recovery of Latent Tree Models"
authors: "Yariv Aizenbud, Ariel Jaffe, Meng Wang, Amber Hu, Noah Amsel, Boaz Nadler, Joseph T. Chang, and Yuval Kluger"
arxiv_id: "arXiv:2102.13276v2"
venue: "arXiv:2102.13276v2"
year: 2021
source_pdf: papers/Spectral Top-Down Recovery of Latent Tree Models.pdf
extracted_text: papers/extracted/spectral-top-down-recovery-of-latent-tree-models.md
last_digested: 2026-07-13
---

## Summary

Develops Spectral Top-Down Recovery (STDR), a deterministic divide-and-conquer approach to infer large latent tree graphical models from observed terminal node correlations via low-rank tensor/matrix decompositions.

## Datasets & models used

Synthetic latent tree models and structured correlation datasets.

## Methods it compared itself against

Expectation-Maximization (EM), Neighbor-Joining tree reconstruction, and local search heuristics.

## Experiments — methodology & scores

Demonstrates provable polynomial-time recovery and exact tree reconstruction.

| Setup | Method | Theoretical Guarantee | Notes |
|---|---|---|---|
| Latent Tree Structure Recovery | STDR | Provable consistency & polynomial time | Avoids non-convex local optima of EM |

## Connection to our pipeline

Core algorithmic bedrock for our spectral recovery methods applied to hierarchical classifier dependencies and latent error estimation.

## Notes / open questions

Provides theoretical justification for top-down spectral recovery of tree-structured classifier dependencies.
