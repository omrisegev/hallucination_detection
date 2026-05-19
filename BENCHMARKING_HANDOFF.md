# Handoff: Comprehensive Hallucination Detection Benchmarking

## Context & Rationale
We are validating a novel **Nadler Spectral Fusion** method for LLM hallucination detection. While our internal results are strong (e.g., 90% AUROC on MATH-500, 87.7% on RAG), we must benchmark them against the actual State-of-the-Art (SOTA) from literature.

The goal is to move beyond simple "internal" baselines and compare against a rigorous taxonomy of competitors across four domains: **Math, Science, RAG, and Agentic loops**.

## The Task
Implement the comparisons defined in the `SOTA_Comparison_Benchmarking_Plan.md`. This involves:
1. **Cloning official SOTA repositories** (LapEigvals, LOS-Net).
2. **Implementing canonical versions** of established methods (Official Semantic Entropy with NLI, Self-Consistency).
3. **Piping our generation traces** into these baselines to ensure a fair "Apples-to-Apples" comparison on the exact same dataset, model, and generation.

## Baseline Taxonomy
We are comparing against three "Access Levels" and three "Compute Types":
- **Access**: Black-box (Text only), Gray-box (Logits/Entropy), White-box (Attention/Hidden States).
- **Supervision**: Supervised vs. Unsupervised.
- **Compute**: 1-Pass vs. Sampling ($K=10$ iterations).

## Domain Mapping
- **Math (MATH-500/GSM8K)**: Compare against **LapEigvals** (White-box, SOTA 72% on GSM8K).
- **Science (GPQA Diamond)**: Compare against **Official Semantic Entropy** (Sampling, Nature 2024).
- **RAG (L-CiteEval)**: Compare against **LOS-Net** (Gray-box, Supervised) and **SelfCheckGPT** (Black-box, NLI-based).
- **Agentic**: Compare against **AUQ** (Verbalized Confidence).

## Technical Requirements
- Use **AUROC** as the primary evaluation metric.
- Do NOT use "lite" versions (e.g., Jaccard clustering). Use the official models (DeBERTa for NLI).
- Respect the "Single Forward Pass" constraint of our method; emphasize where competitors require extra compute (sampling).

## Files to Reference
- `plans/SOTA_Comparison_Benchmarking_Plan.md`: The approved research strategy.
- `spectral_utils/`: Existing utility functions for loading data and generation.
- `HISTORY.md`: To see previous experiment runs.
