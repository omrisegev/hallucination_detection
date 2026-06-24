# Research Report: Advanced Features for Hallucination Detection

**Author**: Gemini CLI (Omri Segev's Thesis Assistant)  
**Date**: 2026-05-14  
**Context**: Expanding the spectral feature suite for thesis direction 2 (RAG) and 4 (Agentic).

---

## 1. Advisor-Inspired Directions (Expert Signal Processing)

Based on the publications of Bracha Laufer-Goldshtein, Ofir Lindenbaum, Amir Averbuch, and Nir Shlezinger, we identify two "High-Science" paths:

### Path A: Geometric Manifold Learning (Laufer-Goldshtein & Lindenbaum)
*   **Concept**: Treat the token-entropy trajectory as a sequence of points on a low-dimensional "Hallucination Manifold."
*   **Method**: **Local Conformal Autoencoders (LOCA)** or standard Manifold Learning (Diffusion Maps).
*   **Thesis Application**: Instead of scalar features, we can use the **intrinsic dimensionality** or **manifold curvature** of the entropy trace. Hallucinations might correspond to "escaping" the grounded manifold into a higher-entropy parametric space.

### Path B: Hybrid State Estimation / Regime Switching (Averbuch & Shlezinger)
*   **Concept**: A LLM is a non-stationary system that switches between "Grounded" and "Hallucinated" modes.
*   **Method**: **Interacting Multiple Model (IMM)** or **KalmanNet** (Data-driven Kalman Filtering).
*   **Thesis Application**: Implement a **bank of filters** (one for high-entropy drift, one for low-entropy grounding). The "Hallucination Score" is the probability that the system has switched to the "Hallucinated" model.

---

## 2. Advanced DSP Feature Expansion

Based on deep research into non-stationary signal analysis, we propose the following 4 features for implementation:

### Feature 1: Sliding-Window Permutation Entropy (PE)
*   **Why**: Measures complexity via ordinal patterns. Robust to amplitude noise and drifting baselines (very common in token entropy).
*   **Signal**: A sudden *drop* in PE indicates a transition from grounded reasoning (high complexity/uncertainty) to "auto-pilot" hallucination (deterministic/low complexity).
*   **Implementation**: `permutation_entropy(h, order=3, delay=1)` in a sliding window.

### Feature 2: Hurst Exponent (Long-Range Dependency)
*   **Why**: Distinguishes between a random walk (EPR mean) and a trending process. 
*   **Signal**: `H > 0.5` indicates a persistent trend (the model is "drifting" deeper into a hallucination). `H < 0.5` indicates mean-reverting behavior (the model is trying to stay grounded).
*   **Implementation**: Rescaled Range (R/S) analysis or DFA.

### Feature 3: Singular Spectrum Analysis (SSA) Residuals
*   **Why**: Decomposes a signal into Trend, Oscillation, and Noise.
*   **Signal**: By stripping the "Trend" (EPR), we isolate the "Oscillation." High-frequency oscillations in the residual often precede a "Burst" hallucination.
*   **Implementation**: SVD on the trajectory matrix.

### Feature 4: CUSUM / Likelihood Ratio (Online Change Point)
*   **Why**: The "gold standard" for detecting regime shifts.
*   **Signal**: Detects the exact token where the mean entropy shifted from the prompt-context distribution to the model's internal distribution.
*   **Implementation**: Cumulative sum of log-likelihood ratios.

---

## 3. Recommended Implementation Roadmap (Phase C)

We will implement these in `spectral_utils/feature_utils.py` in the following order of priority:
1.  **Permutation Entropy** (Highest robustness).
2.  **Hurst Exponent** (Captures the "Drift" phenomenon).
3.  **CUSUM** (Exact change-point localization).
4.  **Geometric Features** (Intrinsic dimensionality – requires more complex model-based setup).

---

## 4. Retrospective Evaluation Strategy (Phase D)

We will use the **Phase 10 RAG raw caches** to evaluate these. We expect:
*   **Hurst Exponent** to be the best predictor for "Long-Doc RAG" where models lose the thread over time.
*   **Permutation Entropy** to be the best predictor for "Agentic Loops" (ReAct) where the model enters a repetitive failure cycle.