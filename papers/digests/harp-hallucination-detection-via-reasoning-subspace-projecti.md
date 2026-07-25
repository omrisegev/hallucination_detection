---
slug: harp-hallucination-detection-via-reasoning-subspace-projecti
title: "HARP: HALLUCINATION DETECTION VIA REASONING SUBSPACE PROJECTION"
authors: "Junjie Hu, Gang Tu, ShengYu Cheng, Jinxin Li, Jinting Wang, Rui Chen, Zhilong Zhou, Dongbo Shan (Huazhong University of Science and Technology)"
arxiv_id: "arXiv:2509.11536"
venue: "arXiv preprint (arXiv:2509.11536, Dec 2025) — NO confirmed peer-reviewed venue found in the PDF; not ICLR 2026"
year: 2026
source_pdf: papers/HARP Hallucination Detection via Reasoning Subspace Projection.pdf
extracted_text: papers/extracted/harp-hallucination-detection-via-reasoning-subspace-projecti.md
last_digested: 2026-07-13
---

## Summary

Proposes HARP, which decomposes internal hidden representation spaces into orthogonal semantic vs reasoning subspaces, projecting activations onto the reasoning subspace to detect logical drift and factual errors.

## Datasets & models used

Reasoning and factual QA benchmarks across open-weight LLMs.

## Methods it compared itself against

Full hidden state probing and scalar confidence estimation.

## Experiments — methodology & scores

Evaluates AUROC across reasoning benchmarks. Per the abstract: "it achieves an AUROC of 92.8%
on TriviaQA, outperforming the previous best method by 7.5%." Also reduces feature dimension to
~5% of the original (post-SVD reasoning-subspace projection) while improving robustness.

## Connection to our pipeline

Directly parallels our spectral matrix factorization of internal representations.

## Notes / open questions

**Correction (2026-07-13)**: original digest claimed "venue: ICLR 2026," but the extracted text
has no conference-acceptance banner anywhere (unlike the other 8 papers in this batch, which all
show an explicit "Published as a conference paper at ICLR/ICML 2026" line) — the only "ICLR"
hit in the whole document is a citation to an unrelated workshop paper. Treat as an unreviewed
arXiv preprint until a venue can be confirmed independently. The concrete headline number
(92.8% AUROC on TriviaQA, +7.5pp over prior best) was also missing from the original digest and
has been added above.
