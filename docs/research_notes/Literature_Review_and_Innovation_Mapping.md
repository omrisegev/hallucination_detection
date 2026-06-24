# Literature Review & Innovation Mapping: Intrinsic Spectral Monitoring

This document maps our "Intrinsic Spectral Monitoring" framework against the state-of-the-art (SOTA) literature from 2025–2026. It identifies key citations, shared theoretical foundations, and our unique innovations for the thesis.

---

## 1. Comparative Literature Matrix

| Paper / Framework | Key Methodology | Shared Theoretical Basis | **Our Innovation (The Gap)** |
| :--- | :--- | :--- | :--- |
| **LapEigvals** (Binkowski et al., 2025) | Spectral analysis of **Attention Maps** (graph eigenvalues). | Spectral Graph Theory; Laplacian fingerprints. | **Signal Source**: They analyze *spatial* attention (token-to-token); we analyze *temporal* entropy (token-over-time). |
| **Spectral Uncertainty** (Walha et al., 2025) | Von Neumann entropy of **Similarity Kernels**. | Kernel-based spectral decomposition; Spectral entropy. | **Computation Type**: They require *multiple* samples to build a kernel; we use a **single forward pass** trajectory. |
| **Zhao et al.** (2026) | Entropy trajectory **Monotonicity** metrics. | Analysis of the entropy trajectory $H(n)$ shape. | **Domain Resolution**: They look for simple *monotonicity*; we perform a full **Fourier Transform** to find rhythmic reasoning patterns. |
| **EDIS** (Zhu et al., 2026) | Trajectory **Instability** (Burst/Rebound) scores. | Direct analysis of the entropy time-series. | **Metric Rigor**: EDIS uses heuristic thresholds; we use **Spectral Power Density**, a principled physical analysis of signal noise. |
| **Sarkar & Das** (2025) | Multimodal **Manifold Gaps** via spectral-graphs. | Multi-view spectral embeddings; Graph Laplacians. | **Task Versatility**: We are the first to use **PCA Fingerprinting** to prove that spectral signatures are *task-dependent* (Math vs. RAG). |

---

## 2. Our Core Innovations

### Innovation A: The "Rhythm of Reasoning" (Frequency Domain)
While prior work (Zhao 2026, EDIS 2026) analyzes the raw *shape* of entropy, our work is the first to move entirely into the **Frequency Domain (H(f))**.
*   **The Discovery**: Correct reasoning has a periodic frequency signature (step-by-step "settling"), while hallucinations look like **"Thermodynamic Noise"** (spectral energy dispersed across high frequencies).
*   **The Tool**: We utilize the **HL-Ratio** (High/Low power ratio), a physically-grounded noise metric derived from FFT.

### Innovation B: Task-Specific Spectral Fingerprinting (PCA)
This is our most significant interpretability contribution.
*   **The Discovery**: We prove that **hallucinations have distinct spectral "fingerprints" depending on the task**.
*   **The Proof**: Our PCA loading analysis shows that **Math** hallucinations are "Procedural" failures (captured by `sw_var_peak`), while **RAG** hallucinations are "State" failures (captured by jitter in the `high_band_power`).

### Innovation C: Unsupervised Nadler Consensus
Connecting to **Ofir Lindenbaum's** work on kernel methods, we use the **Nadler weighted leading eigenvector** as an unsupervised consensus kernel.
*   **The Value**: Unlike *LapEigvals* (which requires a trained supervised probe), our method recovers the hallucination signal from the *statistical consensus* of multiple spectral views without requiring any labels.

---

## 3. Theoretical Synthesis
We share a **"Geometric/Thermodynamic"** view of LLMs with contemporary researchers: we define hallucination as a **phase transition** or a **breakdown in connectivity** that can be measured mathematically.

**Our Unique Identity**: We focus on the **temporal spectral power** of a **single forward pass** and map its **domain-specific importance** across different cognitive task categories (Math, Science MCQ, Factual QA, and Grounded RAG).

---

## 4. Citation Checklist for Thesis
- [ ] **Walha et al. (2025)**: *Spectral Uncertainty for LLMs*. (NeurIPS Reliable ML).
- [ ] **Binkowski et al. (2025)**: *Hallucination Detection via Spectral Features of Attention Maps*. (EMNLP).
- [ ] **Zhao et al. (2026)**: *Entropy trajectory shape predicts LLM reasoning reliability*. (Preprint).
- [ ] **Zhu et al. (2026)**: *EDIS: Diagnosing LLM Reasoning via Entropy Dynamics*. (arXiv).
- [ ] **Sarkar & Das (2025)**: *Grounding the Ungrounded: A Spectral-Graph Framework*. (MLLM).
