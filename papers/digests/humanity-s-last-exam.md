---
slug: humanity-s-last-exam
title: "Humanity's Last Exam"
authors: "Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Richard Ren, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Summer Yue, Alexandr Wang, Dan Hendrycks"
arxiv_id: "arXiv:2501.14249v10"
venue: "arXiv:2501.14249v10"
year: 2025
source_pdf: papers/Humanity's Last Exam.pdf
extracted_text: papers/extracted/humanity-s-last-exam.md
last_digested: 2026-07-13
---

## Summary

Presents Humanity's Last Exam (HLE), an expert-curated benchmark designed to evaluate frontier AI reasoning limits across graduate-level academic subjects without contamination.

## Datasets & models used

Expert-written multi-modal questions across mathematics, humanities, and natural sciences.

## Methods it compared itself against

Frontier LLMs and reasoning models (o1, DeepSeek-R1, Claude 3.5, GPT-4o).

## Experiments — methodology & scores

Evaluates frontier accuracy on expert-curated academic questions.

| Setup | Method | Performance | Notes |
|---|---|---|---|
| Frontier Reasoning Evaluation | Frontier LLMs | Highly unsaturated (<25% accuracy) | Demonstrates severe headroom in graduate-level reasoning |

## Connection to our pipeline

Provides an unsaturated reference evaluation target for testing whether calibration and unsupervised detectors hold up on frontier-hard tasks.

## Notes / open questions

Essential reference benchmark mentioned by FUSE and other evaluation papers.
