---
slug: fuse-ensembling-verifiers-with-zero-labeled-data
title: "FUSE: Ensembling Verifiers with Zero Labeled Data"
authors: "Joonhyuk Lee, Virginia Ma, Sarah Zhao, Yash Nair, Asher Spector, Regev Cohen, Emmanuel Candès"
arxiv_id: "arXiv:2604.18547v1"
venue: "arXiv:2604.18547v1"
year: 2026
source_pdf: papers/FUSE - Ensembling Verifiers with Zero Labeled Data.pdf
extracted_text: papers/extracted/fuse-ensembling-verifiers-with-zero-labeled-data.md
last_digested: 2026-07-13
---

## Summary

Introduces Fully Unsupervised Score Ensembling (FUSE), a method for improving verification quality by ensembling LLM judges and reward models without ground truth correctness labels. Controls conditional dependencies between verifiers to improve spectral ensembling performance.

## Datasets & models used

Academic benchmarks including GPQA Diamond, Humanity's Last Exam (HLE), and IMO Shortlist questions across diverse generator models and verifiers.

## Methods it compared itself against

Single best verifier, unweighted average ensemble, and semi-supervised verifier weighting alternatives.

## Experiments — methodology & scores

Evaluates verification quality and test-time scaling on unsaturated reasoning benchmarks.

| Benchmark | Method | Performance | Notes |
|---|---|---|---|
| GPQA Diamond / HLE / IMO Shortlist | FUSE | Matches or improves upon semi-supervised alternatives | Requires zero ground truth correctness labels |

## Connection to our pipeline

Directly intersects our zero-labeled ensembling research, applying spectral dependency control to LLM verifier scores.

## Notes / open questions

Validates spectral unsupervised ensembling on frontier benchmarks (GPQA Diamond, HLE, IMO Shortlist).
