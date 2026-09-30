---
slug: task-based-graph-signal-compression
title: "Task-Based Graph Signal Compression"
authors: "Pei Li, Nir Shlezinger, Haiyang Zhang, Baoyun Wang, and Yonina C. Eldar"
arxiv_id: "2110.12387v1"
venue: "not found in extract"
year: 2021
source_pdf: papers/Task-Based Graph Signal Compression.pdf
extracted_text: papers/extracted/task-based-graph-signal-compression.md
last_digested: 2026-09-07
---

## Summary

Jointly optimize sampling, quantization and reconstruction. The target is
spectral recovery, assuming known graph/support and a Gaussian signal model
(pp.2–4). Sampling can mix nodes (p.3).

## Datasets & models used

Synthetic sensor graphs, Chinese temperatures, Lena image; no LLM (pp.9–10).

## Methods it compared itself against

Separate sampling/quantization, identical quantizers, infinite-resolution
MMSE, DCT (pp.9–10).

## Experiments — methodology & scores

| Setup | Reported result |
|---|---|
| Synthetic, 100 nodes, bandwidth 20, 40 bits | MSE gap to MMSE <0.02 (p.9) |

## Connection to our pipeline

Our proposed adaptation: preserve existing IU/Joint fused scores when
sampling. This does not learn correctness or replace fusion.

## Notes / open questions

Verified PDF inconsistencies: Algorithm 2 uses argmin despite preceding gain
maximization; equation 15 omits equation A.1's constant-minus structure.
Resolve before porting. Full 13-page read; [source](https://arxiv.org/abs/2110.12387).
