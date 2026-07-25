---
slug: how-do-llms-compute-verbal-confidence
title: "How do LLMs Compute Verbal Confidence?"
authors: "Dharshan Kumaran et al., Google DeepMind"
arxiv_id: "2603.17839"
venue: "ICML 2026"
year: 2026
source_pdf: papers/How_do_LLMs_Compute_Verbal_Confidence.pdf
extracted_text: papers/extracted/how-do-llms-compute-verbal-confidence.md
last_digested: 2026-07-15
---

## Summary

This paper presents a mechanistic interpretability study investigating how Large Language Models generate verbal confidence scores (e.g., stating a confidence number or class like "Almost certain"). Using activation steering, patching, noising, swaps, and attention blocking, the authors show that confidence is not computed "just-in-time" when verbalization is prompted. Instead, it is computed automatically during answer generation, cached at the first post-answer position (PANL), and retrieved during verbalization. Variance partitioning shows that these cached representations explain substantial variance in verbal confidence beyond simple token log-probabilities, indicating a richer self-evaluation of answer quality.

## Datasets & models used

- **Datasets:** TriviaQA, BigMath, MMLU.
- **Models:** Gemma 3 27B, Qwen 2.5 7B, Mistral Small 24B (referred to as Magistral Small 24B).

## Methods it compared itself against

- **Baselines:** Token-level log-probabilities (average, min, max, product log-probabilities).

## Experiments — methodology & scores

The authors perform causal interventions on the residual stream at the first post-answer position (PANL). They evaluate the causal effects of patching and steering on confidence reports. Linear probes on PANL activations achieve high accuracy in predicting correctness and verbalized confidence bins.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| Activation Swap (PANL) | Confidence Recovery (%) | **24.3%** | Swap at PANL alters output, PANL+1 has no effect |
| Linear Probing (PANL) | Explained Variance ($R^2$) | Up to **0.60** | Unique variance explained beyond logprob baselines |

## Connection to our pipeline

- **Overlap:** Conceptual interest in LLM self-evaluation and token log-probabilities.
- **Difference:** They focus on **white-box mechanistic interpretability** (activations, attention edges), whereas we focus on **unsupervised gray-box detection** (logits and entropy traces).
- **Competitor:** No, but their findings support our thesis premise: that models perform automatic, rich self-evaluations during generation, which are encoded in decoding trace dynamics (like $H(n)$ traces).

## Notes / open questions

The paper identifies the newline token or the first token following the answer as the primary caching site for confidence representations.
