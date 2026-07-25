---
slug: tracedet-hallucination-detection-from-the-decoding-trace-of
title: "TRACEDET: HALLUCINATION DETECTION FROM THE DECODING TRACE OF DIFFUSION LARGE LANGUAGE MODELS"
authors: "Shenxu Chang, Junchi Yu, Weixing Wang, Yongqiang Chen, Jialin Yu, Philip Torr, Jindong Gu"
arxiv_id: "arXiv:2510.01274"
venue: "Preprint / arXiv:2510.01274"
year: 2025
source_pdf: papers/TraceDet Hallucination Detection from the Decoding Trace of Diffusion Large Language Models.pdf
extracted_text: papers/extracted/tracedet-hallucination-detection-from-the-decoding-trace-of.md
last_digested: 2026-07-13
---

## Summary

Introduces TraceDet, a hallucination detection framework for Diffusion Large Language Models (D-LLMs) that analyzes the intermediate denoising trajectory ('action trace'). Tracks maximum token entropy across denoising steps to identify unstable generation paths.

## Datasets & models used

QA and summarization benchmarks evaluated on diffusion language models.

## Methods it compared itself against

Final step output entropy and single-step confidence estimators.

## Experiments — methodology & scores

Evaluates AUROC for detecting hallucinatory denoising trajectories.

| Setup | Signal | Observation | Notes |
|---|---|---|---|
| Diffusion Action Trace | Intermediate Denoising Entropy | Strong separation of hallucinations | Intermediate steps reveal structural drift unseen in final output |

## Connection to our pipeline

Extends our trace-level entropy analysis to non-autoregressive diffusion generation traces.

## Notes / open questions

Preprint 2025/2026.
