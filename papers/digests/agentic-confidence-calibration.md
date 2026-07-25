---
slug: agentic-confidence-calibration
title: "Agentic Confidence Calibration"
authors: "Jiaxin Zhang et al., Salesforce AI Research"
arxiv_id: "2601.15778"
venue: "Preprint"
year: 2026
source_pdf: papers/Agentic_Confidence_Calibration.pdf
extracted_text: papers/extracted/agentic-confidence-calibration.md
last_digested: 2026-07-15
---

## Summary

This paper introduces the problem of Agentic Confidence Calibration (ACC): estimating the likelihood that an AI agent's multi-step execution trajectory will succeed. To address compounding errors, tool uncertainty, and data scarcity, the authors propose Holistic Trajectory Calibration (HTC). HTC extracts a compact set of 48 process-level features (cross-step dynamics, intra-step stability, positional indicators, structural attributes) from the agent's logprob trace, and trains a regularized linear model (Ridge/Lasso) to predict trajectory success. A General Agent Calibrator (GAC) is trained to achieve out-of-domain generalization.

## Datasets & models used

- **Datasets:** SimpleQA, GPQA, HLE (Humanity's Last Exam), GAIA, WebArena, AgentBench.
- **Models:** smolagents (CodeAct), OAgents, GPT-4, GPT-OSS, DeepSeek-v3.1, Qwen3-235B.

## Methods it compared itself against

- **Baselines:** Verbalized Confidence, Last-Step TP, Global-Trace TP, Temperature Scaling, LSTM, Transformer, XGBoost, Gaussian Process.

## Experiments — methodology & scores

Evaluated using ECE (Expected Calibration Error, lower is better), Brier Score (BS, lower is better), and AUROC (higher is better) for failure prediction.

| Setup | Metric | Score (SimpleQA / GPT-4) | Notes |
|---|---|---|---|
| SimpleQA | ECE / BS | **0.032 / 0.114** (HTC) vs 0.121 / 0.196 (Verbalized) | Outperforms all baselines |
| GPQA | ECE / BS | **0.084 / 0.201** (HTC) vs 0.454 / 0.523 (Verbalized) | Significant calibration improvement |
| GAIA (OOD) | ECE (GAC) | **0.068** (GAC) vs 0.185 (LastStep-TP) | Strong out-of-domain transfer |

## Connection to our pipeline

- **Overlap:** Extracts statistical features from the autoregressive logprob/entropy trace of the model.
- **Difference:** HTC trains a **supervised linear classifier** (Lasso/Ridge) on trajectory-level features of agent runs, whereas our method (L-SML/U-PCR) is **fully unsupervised** (label-free at training and inference). Additionally, HTC is designed for multi-step agent trajectories (planning/tool-use), whereas we target single-turn reasoning.
- **Competitor:** Yes, on GPQA and Humanity's Last Exam. Our continuous L-SML method represents a competitive unsupervised alternative.

## Notes / open questions

The paper demonstrates that positional features (first/last step confidence) are the most predictive of success on hard reasoning tasks like GPQA, while dynamics and stability are important for search-based tasks like SimpleQA.
