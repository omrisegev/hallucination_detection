---
slug: entropy-and-attention-dynamics-in-small-language-models-a-tr
title: "Entropy and Attention Dynamics in Small Language Models: A Trace-Level Structural Analysis on the TruthfulQA Benchmark"
authors: "Adeyemi Adeseye, Aisvarya Adeseye, Hannu Tenhunen, and Jouni Isoaho"
arxiv_id: "arXiv:2604.03589v1"
venue: "arXiv:2604.03589v1"
year: 2026
source_pdf: papers/Entropy and Attention Dynamics in Small Language Models A Trace-Level Structural Analysis on the TruthfulQA Benchmark.pdf
extracted_text: papers/extracted/entropy-and-attention-dynamics-in-small-language-models-a-tr.md
last_digested: 2026-07-13
---

## Summary

Conducts a trace-level structural analysis of entropy evolution and attention distribution across small language models (SLMs) on the TruthfulQA benchmark. Analyzes how internal decoding entropy and layerwise attention concentration contribute to confident mispredictions and structural hallucinations.

## Datasets & models used

TruthfulQA benchmark across four small language models (SLMs).

## Methods it compared itself against

Final answer outcome accuracy, static hallucination rate, and aggregate sequence probability.

## Experiments — methodology & scores

Analyzes generation phase entropy profiles and attention distribution across layers.

| Phase / Metric | Observation | Notes |
|---|---|---|
| Generation Phase Entropy | Distinct structural differences across SLMs | Localized entropy peaks correlate with factual drift |
| Layerwise Attention | Attention dispersion across hidden layers | Diffuse attention patterns precede hallucinated claims |

## Connection to our pipeline

Foundational literature reference for our trace-level structural analysis track on small language models.

## Notes / open questions

Provides structural trace features linking attention dispersion to entropy spikes.
