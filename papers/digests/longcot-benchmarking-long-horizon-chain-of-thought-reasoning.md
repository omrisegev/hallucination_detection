---
slug: longcot-benchmarking-long-horizon-chain-of-thought-reasoning
title: "LongCoT: Benchmarking Long-Horizon Chain-of-Thought Reasoning"
authors: "Sumeet Ramesh Motwani, Daniel Nichols, Charles London, Peggy Li, Fabio Pizzati, Acer Blake, Hasan Hammoud, Tavish McDonald, Akshat Naik, Alesia Ivanova, Vignesh Baskaran, Ivan Laptev, Ruben Glatt, Tal Ben-Nun, Philip Torr, Natasha Jaques, Ameya Prabhu, Brian Bartoldson, Bhavya Kailkhura, Christian Schroeder de Witt"
arxiv_id: "arXiv:2604.14140v1"
venue: "arXiv:2604.14140v1"
year: 2026
source_pdf: papers/LongCoT Benchmarking Long-Horizon Chain-of-Thought Reasoning.pdf
extracted_text: papers/extracted/longcot-benchmarking-long-horizon-chain-of-thought-reasoning.md
last_digested: 2026-07-13
---

## Summary

Benchmarks long-horizon Chain-of-Thought reasoning across extended problem suites requiring extensive generation lengths. Evaluates structural error accumulation across long reasoning trajectories.

## Datasets & models used

Long-horizon algorithmic, mathematical, and multi-step reasoning evaluation suites.

## Methods it compared itself against

Short CoT prompts and standard step-by-step reasoning models.

## Experiments — methodology & scores

Evaluates completion accuracy and intermediate error localization across extended trajectories.

| Setup | Evaluation Signal | Observation | Notes |
|---|---|---|---|
| Long-Horizon CoT (>2k tokens) | Error Propagation | Intermediate step errors compound | Highlights critical pivot points in extended traces |

## Connection to our pipeline

Directly supports our trace-level analysis track on long-horizon reasoning models.

## Notes / open questions

Validates analyzing localized entropy variance across extended reasoning sequences.
