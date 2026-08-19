---
slug: leash-logit-entropy-adaptive-stopping-heuristic-for-efficien
title: "Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought Reasoning"
authors: "Mohammad Atif Quamar, Mohammad Areeb — Independent / Purdue University"
arxiv_id: "arXiv:2511.04654v1"
venue: "NeurIPS 2025 Workshop: Efficient Reasoning"
year: 2025
source_pdf: "papers/LEASH Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought Reasoning (arXiv 2511.04654v1).pdf"
extracted_text: papers/extracted/leash-logit-entropy-adaptive-stopping-heuristic-for-efficien.md
last_digested: 2026-08-16
---

## Summary

LEASH is a training-free token-level heuristic that stops a rationale when full-vocabulary
entropy slope and top-two log-probability-margin improvement both plateau, subject to a
peak-probability saturation guard, entropy-drop gate, warm-up, and majority vote across recent
non-saturated steps. It then issues a second prompt for a short final answer. The savings are
substantial but accompany about a ten-point accuracy loss, so it is a sensitivity baseline rather
than the strongest target.

## Datasets & models used

- GSM8K: random 300-example test subset with a fixed but undisclosed seed.
- AQuA-RAT: test split.
- Llama-3.1-8B-Instruct, Mistral-7B-v0.1, Phi-3-Mini-128k-Instruct, and
  Qwen2.5-7B-Instruct using native tokenizers and Hugging Face Transformers.

## Methods it compared itself against

Vanilla chain-of-thought followed by a short answer, and No-CoT direct answer. There is no
comparison with REFRAIN, exact DeepConf, or a matched-accuracy fixed-budget frontier.

## Experiments — methodology & scores

Upcast raw logits to fp32, replace non-finite values with zero, clip to `[-B,B]`, compute
full-vocabulary entropy and the top-two log-probability margin, and disable EOS during rationale
generation. Published fixed values are `k=8`, `L=5`, `epsilon_H=0.005`, `delta_M=0.05`,
`m=64`, and `M=320`. Rationale decoding uses temperature 0.7/top-p 0.95; final answer uses
greedy decoding. Count rationale plus answer tokens and exact-match normalized numeric answers.

| Model | GSM8K LEASH / CoT | GSM8K token reduction | AQuA LEASH / CoT | AQuA token reduction |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 62.32 / 74.33 | 30.97% | 54.68 / 63.20 | 28.60% |
| Mistral-7B | 38.67 / 47.20 | 35.12% | 19.25 / 26.38 | 34.20% |
| Phi-3-Mini | 69.87 / 82.67 | 41.50% | 50.24 / 61.67 | 28.30% |
| Qwen2.5-7B | 54.85 / 65.33 | 33.45% | 68.15 / 77.35 | 28.15% |

## Connection to our pipeline

LEASH uses exactly the cheap online entropy/margin family we need as a transparent stopping
control. Implement it only after REFRAIN and our method are frozen, and report the full
accuracy-versus-token frontier. It cannot substantiate a matched-accuracy saving claim because
its published operating point accepts a large accuracy loss.

## Notes / open questions

- Critical constants `B`, peak threshold `tau_p`, warm-up `w`, and entropy-drop `gamma` are not
  numerically specified in the PDF despite the claim that concrete settings are reported.
- The exact task prompts, second-stage closure prompt, GSM8K sample seed, hardware, and code are
  also missing. Therefore any implementation must be labeled `paper-specified-partial` and run a
  preregistered sensitivity grid; it cannot be called an exact reproduction.
