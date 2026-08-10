---
slug: unsupervised-process-reward-models
title: "Unsupervised Process Reward Models"
authors: "Artyom Gadetsky, Maxim Kodryan, Siba Smarak Panigrahi, Hang Guo, Maria Brbic — EPFL"
arxiv_id: "2605.10158"
venue: "Preprint (arXiv-only as of extraction)"
year: 2026
source_pdf: papers/Unsupervised Process Reward Models.pdf
extracted_text: papers/extracted/unsupervised-process-reward-models.md
last_digested: 2026-08-10
---

## Summary

uPRM trains a Process Reward Model with **zero human supervision** — no step-level error
annotations, no ground-truth final-answer labels. The key mechanism: construct a sequence that
interleaves each reasoning step with an explicit correctness marker ("+"/"-"), read the LLM's own
next-token probability of emitting that marker, and use this as a training signal. Crucially,
**scoring multiple trajectories jointly in one context** (concatenating several marked
trajectories together so the LLM sees other trajectories' candidate labels as in-context
examples) gives more reliable judgments than scoring each trajectory independently — this joint
score is then distilled into a dedicated PRM via LoRA + a custom RL objective (Eq. 12).

## Datasets & models used

- **Training data**: PRM800K's reasoning trajectories (`Let's Verify Step by Step`), **text only,
  no correctness labels used**.
- **Scoring/base LLM**: Qwen2.5-14B-Instruct (used both to compute the joint training score AND
  as the LoRA backbone for the resulting uPRM).
- **Evaluation datasets**: ProcessBench (GSM8K/MATH/OlympiadBench/Omni-MATH, §5.1); MATH-500,
  MinervaMath, OlympiadBench for test-time-scaling (§5.2, policy models: Qwen2.5-Instruct
  1.5B/7B/14B, Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct); MATH (level 3-5 subset) for RL
  (§5.3, policy models: Qwen2.5-7B, Qwen2.5-Math-7B, Qwen2.5-Math-1.5B).
- **Training compute**: uPRM's own custom RL took ≈5.5 GPU-hours on **8×H200** (≈44 GPU-hours
  total); the supervised control (sPRM, SFT on the same architecture) took ≈4.25 GPU-hours on the
  same hardware. This is a non-trivial training run, not an inference-only reproduction.

## Methods it compared itself against

- **LLM-as-a-Judge (their own name for the paper's OWN Eq. 6 baseline, not an external
  citation)**: scores each trajectory INDEPENDENTLY (no joint/batched scoring) using the exact
  same marked-sequence next-token-probability mechanism and the exact same base model
  (Qwen2.5-14B-Instruct). This isolates the benefit of joint/in-context scoring — it is the
  single most useful thing for us to reproduce, since it needs no training at all (see below).
- Supervised PRMs on Best-of-8 (§5.2, Table 2): Math-Shepherd-PRM-7B, RLHFlow-PRM-Mistral/Deepseek-8B,
  Skywork-PRM-7B, Qwen2.5-Math-7B-PRM800K, **Qwen2.5-Math-PRM-7B** (the exact checkpoint this
  project is already building a separate ceiling job for — `pb_prm_qwen25math7b_v1.json`),
  Implicit PRM (CE/DPO).
- sPRM: an in-paper supervised control, same architecture/backbone as uPRM but SFT-trained on
  PRM800K's real step labels — isolates "does removing supervision cost accuracy" from "is the
  overall pipeline reasonable."
- Majority voting (reward-model-free) as the TTS baseline.
- VR (verifiable outcome reward) as the RL baseline reward source.

## Experiments — methodology & scores

ProcessBench protocol (§5.1) follows Zheng et al. exactly: F1 = harmonic mean of accuracy on
erroneous trajectories and accuracy on correct trajectories.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| LLM-as-a-Judge, GSM8K | ProcessBench F1 | 49.8 | Qwen2.5-14B-Instruct, independent (non-joint) scoring — Eq. 6 |
| uPRM, GSM8K | ProcessBench F1 | 58.3 | +8.5pp over their own LLM-as-a-Judge control |
| LLM-as-a-Judge, MATH | ProcessBench F1 | 42.8 | |
| uPRM, MATH | ProcessBench F1 | 52.6 | +9.8pp |
| LLM-as-a-Judge, OlympiadBench | ProcessBench F1 | 29.4 | |
| uPRM, OlympiadBench | ProcessBench F1 | 42.7 | +13.3pp (largest gain) |
| LLM-as-a-Judge, Omni-MATH | ProcessBench F1 | 26.6 | |
| uPRM, Omni-MATH | ProcessBench F1 | 39.8 | +13.2pp |
| uPRM vs sPRM, Best-of-8 avg (MATH-500/Minerva/OlympiadBench, Qwen2.5-Math-7B policy) | accuracy | 60.1 (uPRM) vs 60.0 (sPRM) vs 60.6 (Qwen2.5-Math-PRM-7B) | uPRM ≈ every supervised PRM tested despite zero supervision |
| uPRM vs VR, RL policy=Qwen2.5-Math-1.5B | avg accuracy | +4pp for uPRM over VR-only | uPRM also shows LESS reward hacking than sPRM in RL (§5.3) |

## Connection to our pipeline

This is the single most relevant published label-free peer for our reasoning-localization work
(GL-LIU/DUFS-LIU on ProcessBench) — it is the paper the project's own competitor gate
(`cluster/manifests/pb_llama31_8b_external_v1.json`) already named as "quoted, not rerun."
**Important correction after actually reading it**: the number 58.3/52.6/42.7/39.8 (uPRM itself)
is NOT a cheap scoring pass over existing telemetry — it requires **training a new LoRA-tuned
model via a bespoke RL objective** (custom actor-critic-style gradient estimator, described only
in Appendix B; trajectory-packing strategy in Appendix C.2; a degenerate-solution correction term,
Appendix A) at a cost of ≈44 GPU-hours on 8×H200. Faithfully reproducing THAT number is a
project-sized undertaking, not a quick addition to the competitor table.

However, the paper's own **"LLM-as-a-Judge" baseline (Eq. 6, independent per-trajectory scoring,
no training)** is directly and cheaply reproducible with our own infrastructure: for a trajectory
with T steps, construct T+1 candidate-marked sequences (mark steps 1..j-1 as "+", step j as "-",
for each j=1..T, plus the all-"+" j=T+1 case), read the next-token probability of the marker
token after each marked step from a teacher-forced forward pass (exactly the machinery
`run_teacher_forced.py`/`backfill_views.py` already has), sum per Eq. 6, and take argmax over j.
This gives a genuine "unsupervised, next-token-probability, LLM-as-judge" competitor number in
the SAME category uPRM is compared against in its own paper (Table 1) — at inference-only cost,
no training. It is NOT uPRM itself and must be labeled as their own baseline, not their method.

## Notes / open questions

- The paper explicitly cautions (Conclusion) that uPRM "may lag behind state-of-the-art
  supervised PRMs on error localization benchmarks such as ProcessBench" and that raw localization
  accuracy is "an incomplete proxy for downstream utility" — their own strongest claims are about
  TTS/RL utility, not ProcessBench F1 itself, even though ProcessBench F1 is what our competitor
  gate cites them for.
- Base-model choice matters: they explicitly flag (Limitations) that joint scoring requires
  "an LLM with sufficient context length... and sufficient capability," limiting which base
  models are viable — Qwen2.5-14B-Instruct was their choice; a smaller model (e.g. our own
  Qwen3-8B) might not reproduce the joint-scoring benefit even if it reproduces the independent
  LLM-as-a-Judge baseline fine.
- The full ProcessBench breakdown is in their Table D1 (Appendix), not yet read in this pass —
  needed if a per-subset error/correct accuracy split (not just the aggregate F1) is required for
  our own manifest.
- Decision needed from Omri before building anything: (a) reproduce ONLY the cheap "LLM-as-a-
  Judge" baseline now (recommended — tractable, no training, same access category the paper
  itself uses as its ceiling-free control), or (b) commit to a much larger effort reproducing full
  uPRM training (would need Appendices A-C read first, ~44 GPU-hours, real implementation risk
  from a bespoke RL estimator with no public code found).
