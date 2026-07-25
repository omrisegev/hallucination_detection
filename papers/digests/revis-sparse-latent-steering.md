---
slug: revis-sparse-latent-steering
title: "REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models"
authors: "Jialin Wu et al., Ant Group"
arxiv_id: "2602.11824"
venue: "ICML 2026"
year: 2026
source_pdf: papers/REVIS_Sparse_Latent_Steering.pdf
extracted_text: papers/extracted/revis-sparse-latent-steering.md
last_digested: 2026-07-15
---

## Summary

REVIS is a training-free framework designed to mitigate object hallucinations in Large Vision-Language Models (LVLMs) by re-activating suppressed visual information in the latent space. The authors show that visual features and language priors become entangled in deep layers, leading to visual suppression. REVIS extracts a "pure visual vector" via orthogonal projection (subtracting the language prior direction) and applies sparse latent steering (intervention) only at the specific layer (e.g. layer 27) where visual suppression peaks. This surgical approach reduces object hallucinations while preserving general reasoning.

## Datasets & models used

- **Datasets:** POPE (Random, Popular, Adversarial splits), CHAIR, MME, MM-Vet, MMMU-Pro.
- **Models:** Qwen2.5-VL-7B-Instruct, LLaVA-NeXT, LLaVA-1.5-7B, Qwen3-VL, InternVL3.

## Methods it compared itself against

- **Baselines:** VTI (Vanilla steering), VCD, AGLA, ONLY, Regular greedy decoding.

## Experiments — methodology & scores

Evaluated on CHAIR (sentence-level CS and instance-level CI, %) and POPE accuracy/F1.

| Setup | Metric | Score (Qwen2.5-VL) | Notes |
|---|---|---|---|
| COCO (Generative) | CHAIRS / CHAIRI (%) | **25.00 / 8.23** (REVIS) vs 31.00 / 8.13 (Regular) | ~19% reduction in sentence-level error |
| MM-Vet | Accuracy (Overall) | **72.16** (REVIS) vs 56.38 (VTI) | Preserves/improves general reasoning |
| MME | Perception Score | **1723.21** (REVIS) vs 1715.73 (Regular) | Standard greedy baseline: 1715.73 |

## Connection to our pipeline

- **Overlap:** Focuses on latent space geometry and training-free intervention.
- **Difference:** Targets multimodal VLM object hallucinations and performs latent space steering (mitigation), whereas we focus on text-only reasoning error detection.
- **Competitor:** No.

## Notes / open questions

REVIS leverages a 5-cluster counterfactual semantic state space constructed using force-decoding on correct and hallucinated captions to identify the steering directions.
