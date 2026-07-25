---
slug: the-lessons-of-developing-process-reward-models-in-mathemati
title: "The Lessons of Developing Process Reward Models in Mathematical Reasoning"
authors: "Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba Group)"
arxiv_id: "arXiv:2501.07301v2"
venue: "arXiv:2501.07301v2 [cs.CL]"
year: 2025
source_pdf: papers/The Lessons of Developing Process Reward Models in Mathematical Reasoning.pdf
extracted_text: papers/extracted/the-lessons-of-developing-process-reward-models-in-mathemati.md
last_digested: 2026-07-13
---

## Summary

Empirical study detailing the development, data synthesis, and evaluation of Process Reward Models (PRMs) for mathematical reasoning. Demonstrates that Monte Carlo estimation-based data synthesis yields inferior generalization compared to LLM-as-a-judge and human supervision.

## Datasets & models used

Mathematical reasoning benchmarks evaluated using Qwen2.5-Math-PRM-7B and Qwen2.5-Math-PRM-72B.

## Methods it compared itself against

Monte Carlo (MC) estimation-based PRM synthesis, Outcome Reward Models (ORMs), and LLM-as-a-judge annotation.

## Experiments — methodology & scores

Evaluates step-level verification accuracy and downstream reasoning pass rates.

| Model / Method | Verification Approach | Observation | Notes |
|---|---|---|---|
| Qwen2.5-Math-PRM (7B / 72B) | Step-level Process Supervision | Superior generalization over MC synthesis | Highlights critical data quality requirements for PRMs |

## Connection to our pipeline

Provides critical empirical context on step-level PRM supervision compared against our zero-labeled verifier ensembling (FUSE/L-SML).

## Notes / open questions

Directly cites and releases open Qwen2.5-Math-PRM models.
