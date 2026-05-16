# Phase 11b: Agentic Hallucination Detection — Extended Research & Baselines

**For**: Claude
**From**: Gemini CLI (Research Phase)
**Date**: 2026-05-15
**Purpose**: This document contains the latest research on SOTA agentic hallucination detection benchmarks and methods (2024–2026). Use this data to plan and execute Phase 11b of the agentic experiments, extending our spectral entropy method beyond RAG.

---

## 1. Top Recommended Benchmarks for Phase 11b

Based on our constraints (A100 Colab, open-source models, local simulation, requirement for published AUROC SOTA), here are the top targets to extend our spectral feature suite:

### A. ALFWorld (Embodied AI / Household Tasks)
*   **Task Type**: Text-based state-machine navigation.
*   **Local Setup**: `pip install alfworld` (Low effort). Requires `xvfb` and `pyvirtualdisplay` for Colab.
*   **SOTA Target**: **AUROC 0.791** ($\Phi_{min}$) / **0.968** ($\Phi_{last}$) from the AUQ framework (Zhang et al. 2026, arXiv:2601.15703).
*   **Narrative Fit**: Tests if spectral entropy can detect when an agent "gets lost" or fabricates non-existent objects in a structured environment.

### B. HumanEval (Code Generation / Execution)
*   **Task Type**: Python code generation and execution.
*   **Local Setup**: `datasets` library (`evalplus/humanevalplus`) + Python `exec()` (Low effort).
*   **SOTA Target**: **AUROC 0.82 – 0.84** (Pass@1 Prediction) using execution-behavior disagreement methods like DSDE (2026).
*   **Narrative Fit**: Captures the "moment of fabrication" when the model writes buggy logic or non-existent API calls. A qualitatively different modality than text retrieval.

### C. AppWorld (Multi-App Tool Use)
*   **Task Type**: Complex API navigation across 9 simulated apps (Amazon, Gmail, etc.).
*   **Local Setup**: GitHub repo (Medium effort).
*   **SOTA Target**: **AUROC 0.89 – 0.96** (Tool Necessity Detection) from Probe&Prefill / WHEN2TOOL.
*   **Narrative Fit**: Evaluates if spectral features distinguish between genuine tool necessity and hallucinated tool calls ("cross-app hallucinations").

### D. AgentHallu (Step Localization)
*   **Task Type**: Hallucination attribution (identifying *which* step failed).
*   **Local Setup**: GitHub repo (arXiv:2601.06818).
*   **SOTA Target**: **41.1%** Step Localization Accuracy (Gemini 2.5 Pro). Open-source models average only 10.9%.
*   **Narrative Fit**: Since our spectral features are per-token, we can potentially outperform the SOTA by pinpointing the exact token/step where the agent's reasoning diverged.

---

## 2. Key SOTA Methods to Compare Against (2024–2026)

When reporting Phase 11b results, we must benchmark our unsupervised spectral Nadler fusion against these recent methods:

1.  **AUQ (Agentic Uncertainty Quantification)**
    *   **Paper**: Zhang et al., Jan 2026 (arXiv:2601.15703)
    *   **Signal**: Verbalized confidence ($\Phi_{last}$, $\Phi_{min}$, $\Phi_{avg}$) aggregated across steps.
    *   **Type**: Training-free, gray-box. This is our primary baseline to beat on ALFWorld and WebShop.

2.  **LOS-Net (Beyond Next Token Probabilities)**
    *   **Paper**: Bar-Siman et al., Mar 2025 (arXiv:2503.14043)
    *   **Signal**: LLM Output Signature (LOS) using top-k probabilities.
    *   **Type**: Fast ($10^{-5}$s), gray-box, but requires some training/supervision. Achieves ~0.85-0.90+ AUC on HotpotQA.

3.  **SAUP (Situation Awareness Uncertainty Propagation)**
    *   **Paper**: Dec 2024 (arXiv:2412.01033)
    *   **Signal**: Cumulative situational weights via Hidden Markov Models.
    *   **Type**: Propagates uncertainty through multi-step reasoning. Shows 20% AUROC improvement over self-consistency.

4.  **SIA (Stepwise Informativeness Assumption)**
    *   **Paper**: Apr 2026 (arXiv:2604.06192)
    *   **Theoretical Basis**: Proves that reasoning correlates with "entropy lock-in." When entropy stays high (fails to decrease), it reliably signals hallucination. Strong theoretical grounding for our entropy-based approach.

---

## 3. Integration Plan for Claude (Phase 11b)

Claude, when designing Phase 11b:
1.  **Select ALFWorld or HumanEval** as the immediate next environment after the Phase 11a ReAct/HotpotQA runs. They offer the lowest Colab setup friction and have clear AUROC SOTA targets.
2.  **Evaluate Step Localization:** Design the experiment so we don't just predict overall trajectory success, but also predict *which step* failed (targeting the AgentHallu 41.1% accuracy SOTA).
3.  **Ensure we compute AUQ baselines** (verbalized confidence min/avg/last) for a direct apple-to-apples comparison against our spectral features on the chosen benchmark.