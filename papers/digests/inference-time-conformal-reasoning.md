---
slug: inference-time-conformal-reasoning
title: "Inference-Time Conformal Reasoning with Valid Factuality Control for Large Language Models"
authors: "Ting Wang et al., University of Illinois Urbana-Champaign"
arxiv_id: "2606.08831"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Inference_Time_Conformal_Reasoning.pdf
extracted_text: papers/extracted/inference-time-conformal-reasoning.md
last_digested: 2026-07-15
---

## Summary

This paper introduces the Inference-Time Conformal Reasoning (ITCR) framework, which integrates split conformal prediction directly into the multi-step reasoning graph generation process. In multi-step reasoning, claim dependencies form an implicit directed acyclic graph (DAG). ITCR learns a graph-level factuality uncertainty function that aggregates claim-level uncertainty, designs a non-conformity score based on this uncertainty, and dynamically stop/prunes generation using a calibrated conformal threshold. This guarantees valid factuality control (marginal coverage $1-\alpha$) under "no-miss" (recall-conservative) or "no-false" (precision-conservative) objectives.

## Datasets & models used

- **Datasets:** MATH, GSM8K, and QA benchmarks.
- **Models:** LLaMA-3.1-8B-Instruct, Qwen3-4B-Thinking-2507, DeepSeek-R1-Distill-Qwen-1.5B.

## Methods it compared itself against

- **Baselines:** PostCal (post-hoc conformal pruning), heuristic aggregations (MAX, SUM, AVG of claim uncertainties).

## Experiments — methodology & scores

ITCR is evaluated on empirical coverage (targeting $1-\alpha$) and efficiency (compactness of the generated subgraphs). Downstream self-correction performance is measured via Correction Gain (PCR - NCR, where PCR is Positive Correction Rate and NCR is Negative Correction Rate).

| Setup | Metric | Score (Llama-8B) | Notes |
|---|---|---|---|
| GSM8K (Llama-8B) | Avg. Token Usage | **1919.50** (ITCR) vs 2092.80 (PostCal) | ITCR saves compute |
| GSM8K (Llama-8B) | PCR - NCR (%) | **30.86%** (ITCR) vs 8.02% (PostCal) | Downstream correction gain |
| MATH (Llama-8B) | PCR - NCR (%) | **17.39%** (ITCR) vs 3.51% (PostCal) | Average net correction gain |

## Connection to our pipeline

- **Overlap:** Evaluates on GSM8K and MATH, and utilizes conformal prediction frameworks.
- **Difference:** ITCR is an **inference-time generation control** method (stopping/correcting reasoning subgraphs dynamically), whereas our method is a post-generation detector.
- **Competitor:** Complements us. We can feed our continuous L-SML/U-PCR scores into the ITCR framework to calibrate the non-conformity threshold, replacing their trained MLP uncertainty model with our unsupervised spectral score.

## Notes / open questions

The paper demonstrates that learning a structured uncertainty model (MLP) yields much higher efficiency than static sum/average heuristics.
