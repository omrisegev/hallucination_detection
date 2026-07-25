---
slug: gradients-with-respect-to-semantics-preserving-embeddings
title: "Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models"
authors: "Mingda Li et al., Harbin Institute of Technology"
arxiv_id: "2605.04638"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Gradients_with_Respect_to_Semantics_Preserving_Embeddings.pdf
extracted_text: papers/extracted/gradients-with-respect-to-semantics-preserving-embeddings.md
last_digested: 2026-07-15
---

## Summary

This paper proposes SemGrad and HybridGrad, the first gradient-based Uncertainty Quantification (UQ) methods for free-form generation. The core intuition is that a confident LLM should maintain stable output distributions under semantically equivalent input perturbations. The sensitivity is captured by the gradient of the output log-likelihood with respect to a set of "semantic-preserving embeddings" (identified by a Semantic Preservation Score, SPS) at the input token positions. HybridGrad fuses SemGrad with token-importance-weighted parameter gradients (restricted to the LM head weights, W_head) to handle both high and low-aleatoric uncertainty.

## Datasets & models used

- **Datasets:** SciQ, TriviaQA, TruthfulQA.
- **Models:** Qwen3-Instruct4B, Mistral-Nemo-Instruct12B, Llama3.1-Instruct8B.

## Methods it compared itself against

- **Baselines:** LN-PE (Length-Normalized Predictive Entropy), Semantic Entropy (SE), ExGrad, SAR (Semantic Association Ratio).

## Experiments — methodology & scores

The experiments evaluate uncertainty quantification on predicting generation correctness, measured by AUROC (%). Under high aleatoric uncertainty (TruthfulQA), SemGrad outperforms baselines, while HybridGrad achieves the best average performance across all dataset-model pairs.

| Setup | Metric | Score (Llama-8B) | Notes |
|---|---|---|---|
| TruthfulQA (Llama-8B) | AUROC (%) | **70.21** (HybridGrad) vs 64.78 (LN-PE) | SemGrad alone scores 69.80 |
| SciQ (Llama-8B) | AUROC (%) | **78.21** (HybridGrad) vs 72.51 (LN-PE) | ExGrad parameter-gradient baseline: 75.31 |
| TriviaQA (Llama-8B) | AUROC (%) | **85.06** (HybridGrad) vs 84.02 (LN-PE) | Factual QA baseline |

## Connection to our pipeline

- **Overlap:** Both target unsupervised/training-free uncertainty quantification from internal model distributions.
- **Difference:** SemGrad requires **backward passes** to compute gradients ($
abla_{h_E} \log p(\hat{y}|x)$) in semantic space, whereas our method is strictly **forward-only** ($K=1$ logits), which is much more computationally efficient and works without backward-pass access.
- **Competitor:** Yes, on SciQ, TriviaQA, and TruthfulQA. Our continuous L-SML method is a competitive, gradient-free forward-only alternative.

## Notes / open questions

SemGrad relies on identifying the Semantic Preserving Token ($t^*$), which captures the bulk of the input semantics. The paper shows a strong correlation between the Semantic Preservation Score (SPS) of hidden states and the resulting AUROC.
