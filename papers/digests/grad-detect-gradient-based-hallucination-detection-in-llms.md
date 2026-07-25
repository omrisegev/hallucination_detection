---
slug: grad-detect-gradient-based-hallucination-detection-in-llms
title: "Grad Detect: Gradient-Based Hallucination Detection in LLMs"
authors: "Anand Kamat, Daniel Blake, Brent Werness"
arxiv_id: "arXiv:2606.24790"
venue: "2nd Workshop on Compositional Learning, ICML 2026 (workshop paper, not main track)"
year: 2026
source_pdf: papers/Grad Detect Gradient-Based Hallucination Detection in LLMs.pdf
extracted_text: papers/extracted/grad-detect-gradient-based-hallucination-detection-in-llms.md
last_digested: 2026-07-13
---

## Summary

Presents Grad Detect, a gradient-based hallucination detection framework that analyzes layer-wise gradient norms and directions during a single forward-backward pass to predict factual inaccuracies.

## Datasets & models used

TriviaQA, SciQ, PopQA, and TruthfulQA, evaluated across eleven instruction-tuned models from
four families: Qwen2.5 (1.5B–7B), Falcon3 (1B–10B), Gemma-3 (1B–12B), and SmolLM3 (3B).
(Not LLaMA/Mistral/CoQA — corrected 2026-07-13, see below.)

## Methods it compared itself against

Self-Assessment (prompted self-judgment), Confidence Score (max softmax), Sequence Perplexity,
Self-Consistency (5-gen majority vote), Semantic Entropy (10-gen), and Internal State Probing
(MLP on last-layer hidden-state activations).

## Experiments — methodology & scores

Evaluates AUROC/accuracy for Correctness (Correct vs Incorrect), Response (Answered vs Did-Not-
Answer / abstention), and Full (3-way) classification tasks, using a lightweight transformer
encoder over per-layer gradient-cosine-similarity features (Table 3, TriviaQA, AUC):

| Method | Qwen2.5-1.5B | Qwen2.5-7B | Falcon3-1B | Gemma-3-12B |
|---|---|---|---|---|
| Self-Assessment | .53 | .56 | .52 | .57 |
| Sequence Perplexity | .58 | .61 | .59 | .62 |
| Confidence | .74 | .78 | .73 | .79 |
| Internal State Probing | .71 | .76 | .69 | .77 |
| Semantic Entropy | .76 | .80 | .74 | .81 |
| **Grad-Detect (all layers)** | **.82** | **.86** | **.81** | **.86** |

Grad-Detect beats confidence-based baselines by 3–8 accuracy points and Semantic Entropy by
10–12 points, at ~1/5th the inference cost of 10x-sampling Semantic Entropy. Abstention
(Response task) is near-solved: 94–99% accuracy across all eleven models. Final-5-layer subset
retains 98–99% of full-model accuracy (deployable at 1.5x inference cost vs 2.0x for all-layer).

## Connection to our pipeline

Extends our white-box trace features to include backward-pass gradient sensitivity vectors.

## Notes / open questions

Workshop paper (2nd Workshop on Compositional Learning, co-located with ICML 2026), not a
main-track ICML 2026 paper — correct this if citing for "peer-reviewed at ICML 2026."

**Correction (2026-07-13)**: original digest fabricated the datasets ("TriviaQA, CoQA,
TruthfulQA") and models ("LLaMA and Mistral") — verified against extracted text, actual paper
uses TriviaQA/SciQ/PopQA/TruthfulQA on Qwen2.5/Falcon3/Gemma-3/SmolLM3. Results table above was
also missing entirely in the original digest (replaced vague qualitative claims with a real
grounded snippet of Table 3).
