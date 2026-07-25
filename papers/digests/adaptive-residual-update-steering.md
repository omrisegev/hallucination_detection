---
slug: adaptive-residual-update-steering
title: "Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision-Language Models"
authors: "Zhengtao Zou et al., Aalto University"
arxiv_id: "2511.10292"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Adaptive_Residual_Update_Steering.pdf
extracted_text: papers/extracted/adaptive-residual-update-steering.md
last_digested: 2026-07-15
---

## Summary

This paper proposes RUDDER (Residual-Update Directed DEcoding Regulation), a low-overhead steering framework to mitigate object hallucinations in Large Vision-Language Models (LVLMs). Autoregressive generation in LVLMs suffers from "visual dilution," where the prefix visual information fades, causing the model to over-rely on language priors. RUDDER extracts a robust visual evidence direction (CARD) from the prefill residual updates of the visual prefix. During decoding, it injects the CARD vector into the hidden states, modulated by an adaptive trust mechanism (the Beta Gate). RUDDER achieves high efficiency (96% throughput) with single-pass latency.

## Datasets & models used

- **Datasets:** MSCOCO, POPE, MME.
- **Models:** LLaVA-1.5 (7B/13B), Idefics2, InstructBLIP, Qwen2.5-VL.

## Methods it compared itself against

- **Baselines:** DoLa, VCD, VISTA, PAI.

## Experiments — methodology & scores

Evaluated on CHAIRS, CHAIRI, POPE, and MME. Throughput is measured in tokens/second.

| Setup | Metric | Score (LLaVA-1.5-7B) | Notes |
|---|---|---|---|
| COCO (Greedy) | CHAIRS / CHAIRI (%) | **36.5 / 12.1** (RUDDER-Beta) vs 48.6 / 13.6 (Vanilla) | Comparable to VISTA but 1.5x faster |
| POPE (Greedy) | F1-Score (%) | **86.5%** (RUDDER-Beta) vs 84.9% (Vanilla) | High accuracy |
| Latency | Throughput (tok/s) | **54.9** (RUDDER-Beta) vs 56.7 (Vanilla) | Maintains 96% of vanilla speed |

## Connection to our pipeline

- **Overlap:** Focuses on low-overhead, single-pass inference-time intervention.
- **Difference:** Multimodal VLM steering to mitigate object hallucinations, whereas we detect factual/reasoning errors in text-only models.
- **Competitor:** No.

## Notes / open questions

The Beta Gate dynamically scales the injection strength by mapping the cosine similarity of the current hidden state to the visual CARD vector.
