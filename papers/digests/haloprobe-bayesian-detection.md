---
slug: haloprobe-bayesian-detection
title: "HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models"
authors: "Reihaneh Zohrabi et al., Technical University of Darmstadt"
arxiv_id: "2604.06165"
venue: "ICML 2026 (Preprint)"
year: 2026
source_pdf: papers/HaloProbe_Bayesian_Detection.pdf
extracted_text: papers/extracted/haloprobe-bayesian-detection.md
last_digested: 2026-07-15
---

## Summary

This paper presents HaloProbe, a Bayesian framework to detect and mitigate object hallucinations in Large Vision-Language Models (LVLMs). The authors reveal a Simpson's paradox in coarse-grained attention-based hallucination detection: token position and object repetition act as confounders that reverse attention statistics when aggregated. HaloProbe factorizes external description features (repetition, position) and internal decoding signals (fine-grained attention, logits). It uses class-balanced training for the internal estimator and combines it with a learned prior over external features to get the true posterior, which is used as an external scoring signal for non-invasive mitigation during decoding.

## Datasets & models used

- **Datasets:** MS COCO 2014, MME, POPE, Shikra, InternVL.
- **Models:** LLaVA-1.5-7B, Shikra, MiniGPT-4, Qwen3-VL, InternVL3.5.

## Methods it compared itself against

- **Baselines:** IC, UT, EAZY, PAI, VISTA, VCD.

## Experiments — methodology & scores

Evaluated on open-ended image captioning using CHAIRS (sentence-level hallucination rate, %) and CHAIRI (instance-level hallucination rate, %), and object probing accuracy on POPE.

| Setup | Metric | Score (LLaVA-1.5) | Notes |
|---|---|---|---|
| COCO (Greedy) | CHAIRS / CHAIRI (%) | **24.8 / 7.2** (HaloProbe) vs 48.6 / 13.6 (Vanilla) | Significant reduction in hallucination |
| COCO (Greedy) | CHAIRS / CHAIRI (%) | **24.8 / 7.2** (HaloProbe) vs 36.5 / 12.9 (VISTA) | Beats steering/contrastive methods |
| POPE (Greedy) | Accuracy (%) | **85.34%** | Guided decoding preserves general recognition |

## Connection to our pipeline

- **Overlap:** Explores logit and attention signals for hallucination detection.
- **Difference:** Specific to **multimodal VLMs** and **object hallucinations**, and requires training an internal estimator on a balanced dataset. Our method is **text-only reasoning** and **fully unsupervised**.
- **Competitor:** No.

## Notes / open questions

The paper demonstrates that factorized Bayesian learning improves robustness under distribution shifts.
