---
slug: halluhard-a-hard-multi-turn-hallucination-benchmark
title: "HALLUHARD: A Hard Multi-Turn Hallucination Benchmark"
authors: "Dongyang Fan, Sebastien Delsad, Nicolas Flammarion, Maksym Andriushchenko"
arxiv_id: "arXiv:2602.01031v1"
venue: "arXiv:2602.01031v1"
year: 2026
source_pdf: papers/HalluHard A Hard Multi-Turn Hallucination Benchmark.pdf
extracted_text: papers/extracted/halluhard-a-hard-multi-turn-hallucination-benchmark.md
last_digested: 2026-07-13
---

## Summary

Introduces HALLUHARD, a hard multi-turn hallucination benchmark with 950 seed questions across legal cases, research questions, medical guidelines, and coding. Requires inline citations and uses an iterative web-search evaluation pipeline to fetch full-text sources (including PDFs) and verify factual support.

## Datasets & models used

HALLUHARD multi-turn dialogue benchmark (950 seed questions across legal, research, medical, and coding domains).

## Methods it compared itself against

Frontier proprietary and open-weight models (Opus-4.5, GPT-4, open-weight LLMs) evaluated with and without web search grounding.

## Experiments — methodology & scores

Evaluates multi-turn factual error and hallucination rates with inline citation verification.

| Model Configuration | Domain Setup | Hallucination Rate | Notes |
|---|---|---|---|
| Opus-4.5 with Web Search | Multi-turn Evaluation | ≈30% | Strongest frontier configuration still exhibits substantial error |

## Connection to our pipeline

Critical multi-turn stress-test benchmark for evaluating whether single-turn hallucination detection generalizes across conversational turns.

## Notes / open questions

Shows that even top frontier models with active web search hallucinate on ≈30% of multi-turn high-stakes queries.
