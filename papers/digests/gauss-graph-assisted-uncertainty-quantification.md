---
slug: gauss-graph-assisted-uncertainty-quantification
title: "GAUSS: Graph-Assisted Uncertainty Quantification using Structure and Semantics for Long-Form Generation in LLMs"
authors: "Yinhan He, Yaochen Zhu, Mingjia Shi, Wendy Zheng, Lin Su, Xiaoqing Wang, Qi Guo, Jundong Li (affiliations not verified)"
arxiv_id: "OpenReview rm6rHG7p9n — https://openreview.net/forum?id=rm6rHG7p9n"
venue: "ICML 2026"
year: 2026
source_pdf: not available — OpenReview PDF is bot-gated (HTTP 403) as of 2026-07-28
extracted_text: not extracted
last_digested: 2026-07-28
status_note: "ABSTRACT-ONLY. Everything below the Summary is either sourced from the OpenReview abstract or marked UNVERIFIED."
---

> **Partially corrected 2026-07-28.** The PDF could not be retrieved (OpenReview returns 403 to
> non-browser clients). Author list and method description are now sourced from the OpenReview
> listing, https://openreview.net/forum?id=rm6rHG7p9n. The previous card's author line was wrong.
> Everything not in the abstract remains **unverified**.

## Summary

Sourced from the OpenReview abstract. GAUSS targets uncertainty quantification for long-form LLM
output in high-stakes settings (clinical reporting, legal analysis, policy drafting). Each generated
paragraph is modeled as a **semantic graph**: nodes are atomic facts, edges capture inter-fact
relationships. The hypothesis is that uncertainty shows up as structural and semantic discrepancy
between these graphs across independently sampled paragraphs for the same query. The uncertainty score
is the **expected alignment cost** between an anchor paragraph's graph and those of alternative
reference paragraphs. The stated advance over prior work is accounting for interdependency among
atomic facts within a paragraph, rather than scoring facts independently.

## Datasets & models used

**UNVERIFIED.** The previous card said "Long-form text generation benchmarks" and "LLaMA-2-70B, etc.";
neither is sourced. Do not cite.

## Methods it compared itself against

**UNVERIFIED.** The previous card listed "sentence-level confidence, bipartite entailment graphs
(SelfCheckGPT)". Not sourced. Do not cite.

## Experiments — methodology & scores

**UNVERIFIED.** No numbers available. The previous card's "Outperforms SelfCheckGPT" AUROC row was
invented and has been removed.

## Connection to our pipeline

- **Not a competitor, and the reason is structural.** GAUSS needs **K > 1 samples** per query to build
  and align graphs across paragraphs. Our detector is single-pass (K=1) on the entropy trace of one
  generation. Different cost class entirely, ours by roughly an order of magnitude.
- **Different task:** long-form fact-rich narratives, decomposed into atomic facts. Our in-scope cells
  are short-answer QA and math reasoning, where there is one answer to grade, not a fact graph.
- **No overlap to act on.** Keep as background on the long-form UQ line.

## Notes / open questions

- Re-run `/paper-digest` if the PDF becomes reachable, and delete the UNVERIFIED markers.
