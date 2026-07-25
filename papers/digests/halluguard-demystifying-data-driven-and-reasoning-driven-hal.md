---
slug: halluguard-demystifying-data-driven-and-reasoning-driven-hal
title: "HALLUGUARD: DEMYSTIFYING DATA-DRIVEN AND REASONING-DRIVEN HALLUCINATIONS IN LLMS"
authors: "Xinyue Zeng (Virginia Tech), Junhong Lin (MIT), Yujun Yan (Dartmouth College), Feng Guo (Virginia Tech), Liang Shi (Virginia Tech), Jun Wu (Michigan State University), Dawei Zhou (Virginia Tech)"
arxiv_id: "arXiv:2601.18753"
venue: "ICLR 2026"
year: 2026
source_pdf: papers/HalluGuard Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs.pdf
extracted_text: papers/extracted/halluguard-demystifying-data-driven-and-reasoning-driven-hal.md
last_digested: 2026-07-13
---

## Summary

Introduces the "Hallucination Risk Bound," a theoretical framework built on Neural Tangent
Kernel (NTK) theory that formally decomposes hallucination risk into a data-driven term
(training-time representational mismatch) and a reasoning-driven term (inference-time rollout
instability). HALLUGUARD is the resulting NTK-based score:
`HALLUGUARD(u_h) = det(K) + log(σ_max) − log(κ²)`, where `det(K)` (NTK Gram-matrix determinant)
captures data-driven representational adequacy and `log(σ_max) − log(κ²)` (per-step Jacobian
spectral norm / condition number) captures reasoning-driven amplification and instability.
(NOT primarily "spectral norm analysis of hidden-state Jacobians" as the original digest led
with — that phrase cherry-picks one secondary term of the NTK bound and overstates its overlap
with this project's spectral_utils features; corrected 2026-07-13.)

## Datasets & models used

10 benchmarks across 3 categories (data-grounded QA incl. RAGTruth, NQ-Open, SQuAD; reasoning
incl. MATH-500; and TruthfulQA), 11 competitive baselines, 9 LLM backbones.

## Methods it compared itself against

11 baselines spanning standard RAG verifiers, factual probing, and uncertainty-based detectors
(not fully enumerated in the extracted excerpt read so far).

## Experiments — methodology & scores

Table 1 (correlation between NTK proxies and task family) shows `det(K)` correlates most with
the data-centric SQuAD (r=0.84), while `log(σ_max) − log(κ²)` correlates most with the
reasoning-heavy MATH-500 (r=0.88) — the empirical basis for the risk decomposition. An ablation
shows the reasoning-driven stability term is load-bearing: removing `−log(κ²)` raises MSE from
0.0192 to 0.0381 (R² drops 0.985 → 0.890).

## Connection to our pipeline

Weaker connection than originally claimed: the method is an NTK generalization-theoretic risk
bound (kernel Gram-matrix determinant + Jacobian condition number), not a spectral analysis of
entropy/logprob traces like `spectral_utils`. The data- vs reasoning-driven risk *decomposition*
is conceptually relevant to separating our own failure modes, but the machinery doesn't transfer
directly.

## Notes / open questions

Published at ICLR 2026 (confirmed: "Published as a conference paper at ICLR 2026" banner
throughout).

**Correction (2026-07-13)**: original digest listed only 2 of the paper's 7 authors (dropped
Yujun Yan, Feng Guo, Liang Shi, Jun Wu, Dawei Zhou, and the Dartmouth/Michigan State
affiliations entirely) and mischaracterized the core method as "spectral norm analysis of
hidden-state Jacobians" when the paper's own framing is an NTK-based Hallucination Risk Bound —
"spectral"/Jacobian terms appear only as one derived component of that bound. Datasets/baselines
sections were also generic placeholders; replaced with the actual benchmark categories and
Table 1 correlation numbers.
