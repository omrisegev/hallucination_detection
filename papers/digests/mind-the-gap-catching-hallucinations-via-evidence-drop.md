---
slug: mind-the-gap-catching-hallucinations-via-evidence-drop
title: "Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning Manifold"
authors: "QunJie Chen, Yufei Chen, Xiaodong Yue, Linye Li (affiliations not listed on the ICML page)"
arxiv_id: "none — not on arXiv as of 2026-07-28"
venue: "ICML 2026 (poster)"
year: 2026
source_pdf: not available — no public PDF found 2026-07-28
extracted_text: not extracted
last_digested: 2026-07-28
status_note: "ABSTRACT-ONLY. Everything below the Summary is either quoted from the official abstract or marked UNVERIFIED."
---

> **Partially corrected 2026-07-28.** No PDF is publicly available yet (no arXiv, OpenReview PDF
> gated, code repo has no paper). Metadata and the Summary are now sourced from the official ICML
> 2026 poster page, https://icml.cc/virtual/2026/poster/62422. Everything the previous card asserted
> beyond the abstract remains **unverified** and is flagged as such.
> Code: https://github.com/QJ0114/evidence-drop

## Summary

Verbatim from the official abstract:

> "Most existing uncertainty-based detectors rely on sequence-level averaging, which ignores the
> step-wise dynamics of reasoning and often misclassifies hard-but-correct or easy-but-wrong samples.
> We propose a dynamic perspective that models reasoning as a trajectory on a latent Evidence
> Manifold, where each step is supported by local evidence. Hallucinations are characterized as
> Evidence Drops, i.e., sudden declines in local evidence support that indicate topological
> deviations from this manifold. Based on this insight, we design a training-free and model-agnostic
> detector that identifies hallucinations via the worst-case Evidence Drop and enables step-level
> error localization. Experiments on GSM8K, MATH, and ProcessBench show consistent improvements over
> sequence-level uncertainty baselines in selective accuracy and risk-coverage trade-offs."

## Datasets & models used

- **Datasets (confirmed):** GSM8K, MATH, ProcessBench.
- **Models: UNVERIFIED.** The previous card said "LLaMA-3.1-8B, Qwen-2.5-7B-Instruct, etc."; the
  abstract names no models. Do not cite.

## Methods it compared itself against

**UNVERIFIED.** The abstract says only "sequence-level uncertainty baselines". The previous card's
specific list (Semantic Entropy, LN-Entropy, Perplexity, SelfCheckGPT) is not sourced. Do not cite.

## Experiments — methodology & scores

Metrics confirmed by the abstract: **selective accuracy** and **risk-coverage trade-offs**. No numeric
score is available. The previous card's AUROC row was invented and has been removed.

## Connection to our pipeline

- **Overlap, and it is real:** same task family (GSM8K, MATH), same claim that sequence-level
  *averaging* discards step-wise dynamics. That is the exact argument behind our own pivot away from
  EPR, which is the DC component of H(n), toward the AC spectrum. Two independent groups reaching the
  same premise is useful positioning support.
- **Difference:** we read the dynamics spectrally from the entropy trace and fuse many views with
  L-SML into one score per answer. They read a local evidence-support drop and can localize to a step.
- **Genuinely training-free**, which ours is at runtime too, so it is a fair cost-class comparison if
  we ever benchmark against it.
- **Blocked:** cannot benchmark without the paper or numbers. Revisit once ICML proceedings are public.

## Notes / open questions

- Step-level localization is a capability we do not have. Worth knowing about, not worth chasing.
- Re-run `/paper-digest` on this once the PDF appears, and delete the UNVERIFIED markers.
