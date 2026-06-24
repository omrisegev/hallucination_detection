# Hallucination Detection Research: June 2026 Synthesis & Handoff

> **Last updated**: 2026-06-08 — merged Step 127 findings from Claude sessions.
> Sections marked ⚠️ were corrected after the original Gemini draft.

## 1. Goal
Break the performance plateau of the 5-feature L-SML pipeline (`epr`, `low_band_power`, `sw_var_peak`, `cusum_max`, `spectral_entropy`) on GPQA and RAG datasets by introducing highly orthogonal, completely **unsupervised**, **single-pass** hallucination detection signals.

**⚠️ Known pipeline issues (from Step 127 diagnosis — highest priority):**
- **L-SML fusion delivers -5.7pp vs. best single feature** across 29 cached cells. The fusion mechanism is underperforming, not individual features. The most likely causes are: (a) N≈20-50 per cell is too small for reliable covariance estimation, (b) `spectral_entropy` has inverted sign for math-specialist models (Qwen-Math series), breaking the FEATURE_SIGNS assumption. Fix approaches: pool all three temperatures per (model, dataset) cell to triple N; use per-domain FEATURE_SIGNS rather than a global constant.
- **`trace_length` is right-censored** — when the model hits max_new_tokens, fraction-positive after median binarization drops to <30%, making it a degenerate classifier. It is excluded from GOOD_FEATURES and should not be treated as an orthogonal anchor until the saturation is explicitly flagged (saturated trace → force -1 classifier).
- **`dominant_freq`** appears broken because it is paired with `trace_length` in the L-SML covariance matrix. It may carry independent signal and should be tested in isolation.

---

## 2. Unsupervised, Single-Pass Methods

The project strictly requires **unsupervised, single-pass** methods. Methods that require training, calibration datasets, or multiple generation passes are excluded.

**Out of scope (confirmed):** WEPR, Koopman Residuals (supervised/calibration-dependent), Behavioral Probes / Negation-Persistence / Skeptic probing (require a 2nd pass).

### A. Spilled Energy (Minut et al., ICLR 2026 — arXiv:2602.18671)
- **Status:** Fully unsupervised, single-pass. **In scope.**
- **Exact formula:** ΔE(x_i) = −log p(x_i | x_{<i}) — the negative log-probability of the **specific token that was sampled**, not the Shannon entropy of the distribution. Derivation: E(x_i) = −logit(x_i), E_m(x_{i−1}) = −log Σ_k exp(logit_k) = −log Z, so ΔE = log Z − logit(x_i) = −log softmax(x_i).
- **Why orthogonal to entropy:** Shannon entropy = −Σ_k p_k log p_k (average surprise over all tokens). ΔE = surprise at the *chosen* token only. They decouple when the model is uncertain (high entropy) but happens to sample a high-probability token — entropy high, ΔE low.
- **Key AUROC (Table 1, LLaMA-3-Instruct, 9 benchmarks):** 73.16% mean AUROC with min-pooling, vs 65.16% logit baseline. Better cross-dataset generalisation than probing classifiers.
- **⚠️ Implementation constraint:** Our `generate_full()` stores only Shannon entropies (top-K approximation). The per-token log-prob of the sampled token is discarded. **Existing 29-cell cache cannot yield Spilled Energy.** Requires a one-line addition to `token_entropies_from_scores()` in `model_utils.py` and a new inference run on ≥1 dataset.

### B. ~~Semantic Energy (Ma et al., 2025 — arXiv:2508.14496)~~
- **⚠️ OUT OF SCOPE — violates single-pass constraint.** This method requires K sampled responses per problem + semantic clustering over the K responses. It is multi-pass. Moving to the excluded list.
- For reference: Boltzmann Cluster Energy over semantic clusters achieves +4.5pp AUROC over Semantic Entropy in multi-cluster cases, but is structurally multi-pass.

### C. EDIS (arXiv:2602.01288) — currently running (Phase 13)
- **Status:** Implemented in `feature_utils.py` as `compute_edis()`. Phase 13 notebook (`Spectral_Analysis_MathComp_Phase13.ipynb`) is currently comparing EDIS vs L-SML on Qwen2.5-Math-1.5B / AMC23 / AIME24. Results pending.
- Operates on entropy traces — **no new inference needed**.

---

## 3. Highly Orthogonal Historical Signals

1. **`dominant_freq` (Steps 95–102):** Individual AUC of 73.0% (beats EPR 66.8%), ρ(EPR) = 0.123. Broken in current GOOD_FEATURES set because it co-clusters with right-censored `trace_length`. **Action: test `dominant_freq` standalone after fixing trace_length saturation.**

2. **`trace_length` (Steps 114–127):** ρ(EPR) ≈ 0.15–0.25. Was the anchor of the Phase 5 90.0% AUC result. **⚠️ Currently broken** due to right-censoring at max_new_tokens (fraction-positive < 30%). Fix: explicit saturation flag or reduce max_new_tokens cap. Do not use in current form.

3. **Behavioral Probes — Negation Persistence / Skeptic (Steps 30–45):** ρ(EPR) = 0.20–0.63, +7.8pp absolute lift on WebQ. **Out of scope** (2nd pass required). Archive unless injected into a single-pass inline CoT prompt.

4. **V4 RAG prompt (Step 113):** +18.6pp fusion AUROC on RAG datasets over baseline V0 prompt. This is a prompt-engineering intervention, not a feature change. Infrastructure exists; re-run is low-priority but confirmed effective.

---

## 4. Modern 2026 Benchmarks

### A. Humanity's Last Exam (HLE — arXiv:2501.14249)
- Graduate-level expert reasoning. AUROC 60–75% for token-level uncertainty / response length methods on GPT-4o and Claude 3.5. RMS calibration error >80% for frontier models (confidently wrong).

### B. RAGTruth (ACL 2024)
- RAG passage/span hallucination. F1 ~63.4% zero-shot (GPT-4-Turbo); ~65-68% with fine-tuned Llama-2-13B.

### C. HalluHard (arXiv:2602.01031)
- Multi-turn, citation-grounded, high-stakes domains. Detection F1 67% with GPT-o3-mini judge. Target models (Claude 4.5 Opus, GPT-5.2-thinking) hallucinate at 30–38% even with web search.

---

## 5. 2026 Trends in Math Reasoning Hallucination Detection

**Core shift: ORM → PRM.** Outcome Reward Models evaluate only the final answer and are vulnerable to "fluent failures" (correct answer, wrong intermediate steps). Process Reward Models (PRMs) evaluate every reasoning step — gold standard for 2025/2026.

- **Fine-Grained PRM (FG-PRM):** Classifies specific hallucination types (Logical Gap vs. Calculation Error).
- **Outcome-Conditioned Centering (PROGRS):** Penalises steps that look correct but lead to a dead end.
- **ProcessBench (Qwen2.5-Math-PRM, arXiv:2501.07301):** Step-level hallucination benchmark. Most directly relevant for our thesis — our `cusum_shift_idx` and `sw_var_peak` are natural unsupervised PRMs.
- **Streaming Detection trend:** Monitor CoT in real-time (exactly what `cusum_max`, `sw_var_peak`, and Spilled Energy do). Detect hallucination onset before the wrong final answer is generated.

---

## 6. Papers — All Present in Workspace

| Paper | arXiv | File | Status |
|---|---|---|---|
| Spilled Energy (Minut, ICLR 2026) | 2602.18671 | Spilled Energy in Large Language Models.pdf | ✅ |
| ~~Semantic Energy (Ma, 2025)~~ | 2508.14496 | Semantic Energy Detecting LLM Hallucination Beyond Entropy.pdf | ✅ (out of scope) |
| HLE (Phan, 2025) | 2501.14249 | Humanity's Last Exam.pdf | ✅ |
| RAGTruth (Niu, ACL 2024) | — | RAGTruth A Hallucination Corpus....pdf | ✅ |
| HalluHard (Fan, 2026) | 2602.01031 | HalluHard A Hard Multi-Turn Hallucination Benchmark.pdf | ✅ |
| ProcessBench / PRM (Qwen Team) | 2501.07301 | The Lessons of Developing Process Reward Models....pdf | ✅ |

---

## 7. Immediate Priority Order

1. **Get Phase 13 results** (EDIS vs L-SML on Qwen-1.5B / AMC23+AIME24) — determines whether EDIS alone already closes the gap.
2. **Fix L-SML fusion** — pool temps (3× N), per-domain signs, diagnose cluster quality. This is the thesis method; fix before adding new features.
3. **Fix trace_length saturation** — explicit saturation flag; re-test `dominant_freq` independently.
4. **Add Spilled Energy** — modify `generate_full()` to also return `token_neg_logprobs`; run one small inference cell (MATH-500 / 1.5B, 50 samples) to validate the feature.
5. **New benchmarks** — HLE / RAGTruth / HalluHard after pipeline is stable on math.
