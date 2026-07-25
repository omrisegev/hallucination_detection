---
slug: geofaith-a-spatio-temporal-dual-view-of-faithful-chain-of-th
title: "GeoFaith: A Spatio-Temporal Dual View of Faithful Chain-of-Thought"
authors: "Weijiang Lv, Wentong Zhao, Jiayu Wang, Yuhao Wu, Jiaheng Wei, Xiaobo Xia"
arxiv_id: "arXiv:2605.26893v1"
venue: "arXiv:2605.26893v1"
year: 2026
source_pdf: papers/GeoFaith A Spatio-Temporal Dual View of Faithful Chain-of-Thought.pdf
extracted_text: papers/extracted/geofaith-a-spatio-temporal-dual-view-of-faithful-chain-of-th.md
last_digested: 2026-07-13
---

## Summary

Proposes GeoFaith, a spatio-temporal framework that leverages latent geometric structure and entropy dynamics to diagnose and enforce faithful Chain-of-Thought (CoT) reasoning against post-hoc rationalization.

## Datasets & models used

Multi-step CoT reasoning and QA benchmarks across LLMs.

## Methods it compared itself against

Outcome-based supervision, post-hoc explanation checks, and perturbation-based CoT evaluation.

## Experiments — methodology & scores

Evaluates CoT faithfulness assessment and geometric separation of faithful vs unfaithful reasoning trajectories.

| Setup | Method | Evaluation Signal | Notes |
|---|---|---|---|
| CoT Reasoning Chains | GeoFaith | Spatio-temporal geometry + entropy dynamics | Identifies unfaithful post-hoc rationalization |

## Connection to our pipeline

Directly complements our trace-level structural analysis by combining layerwise spatial representation geometry with temporal token entropy.

## Notes / open questions

Provides geometric interpretations for trace entropy anomalies during unfaithful CoT steps.
