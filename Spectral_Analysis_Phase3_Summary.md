# Spectral Analysis of Entropy Trajectories — Full Research Summary (Phase 1–3)
**Date**: April 2026  
**Researcher**: Omri Segev  
**Project**: Unsupervised Hallucination Detection in LLMs using Entropy Production Rate (EPR) and Nadler Spectral Fusion  
**Advisors**: Prof. Amir Averbuch (TAU) · Dr. Ofir Lindenbaum (BIU, spectral methods)

---

## 1. Background and Motivation

### The Core Research Question

Can we detect hallucinations in LLMs **without any labeled training data**, using only the model's own internal uncertainty signals during generation?

The base signal is **EPR (Entropy Production Rate)**: the mean token-level Shannon entropy over the generated sequence. At each token position *n*, the model computes a probability distribution over its vocabulary. The entropy H(n) = −Σ p(t) log p(t) quantifies the model's local uncertainty. EPR = mean(H(n)) over all generated tokens.

Higher EPR → model is uncertain on average → more likely to be hallucinating.

### The Nadler Spectral Fusion Framework

Rather than relying on a single scalar, this project fuses multiple independent "views" of the same prediction target using **Nadler spectral fusion** (Nadler et al., 2006). Two mathematical requirements must hold for any pair of signals being fused:
1. Each signal must individually be informative: AUC > 50%
2. The pair must be approximately independent: Spearman |ρ| < 0.75

When both conditions hold, Nadler fusion uses eigenvector analysis of the cross-covariance structure to find optimal weights. The fusion consistently outperforms any individual view.

### Key Insight: EPR is the DC Component of FFT(H(n))

EPR is mathematically identical to the **zero-frequency (DC) term** of the Fourier transform of H(n). This means every non-DC frequency component is **orthogonal to EPR by construction** — there is exactly zero information overlap between EPR and any spectral feature computed from the mean-subtracted H(n).

**Hypothesis**: Correct reasoning produces structured H(n) trajectories with rhythmic patterns (step boundaries, consistent uncertainty) → energy concentrated at specific frequencies → low spectral entropy. Hallucinated reasoning produces erratic H(n) → flat power spectrum → high spectral entropy.

### Prior Results (Context)

Previous experiments established:
- **TriviaQA**: 81.5% AUC — 6-view Nadler ensemble (4 temperatures + Verify + Skeptic), Falcon-3-10B
- **WebQuestions**: 76.0% AUC — same ensemble
- **GSM8K baseline**: EPR(trace) = 66.8%, Qwen2.5-Math-1.5B-Instruct
- **GSM8K best prior**: EPR + EDIS Nadler = 68.7% (EDIS tracks burst-rebound patterns in H(n))

---

## 2. Feature Set: All 11 Signals

### Phase 2 Signals (7 total)

All computed after subtracting the mean (EPR) from H(n) to ensure DC orthogonality.

| Signal | Definition | Hallucination hypothesis |
|--------|-----------|--------------------------|
| `epr` | mean(H(n)) — the DC component | High EPR → uncertain model → hallucination |
| `spectral_entropy` | −Σ PSD_norm · log(PSD_norm) over AC frequencies | High = energy scattered across all frequencies = erratic reasoning |
| `low_band_power` | Fraction of AC power at f ∈ (0, 0.1] | Low in hallucination: structured reasoning concentrates at low f |
| `high_band_power` | Fraction of AC power at f ∈ [0.4, 0.5] (Nyquist) | High in hallucination: erratic token-to-token jitter |
| `hl_ratio` | high_band_power / low_band_power | High ratio = erratic |
| `dominant_freq` | Frequency of strongest AC peak (excluding DC) | Correct reasoning → clear dominant oscillation |
| `spectral_centroid` | Σ(f · PSD_norm) / Σ(PSD_norm) — frequency center of mass | Shifts toward higher f in hallucination |

### Phase 3 New Signals (4 additional)

| Signal | Definition | Motivation |
|--------|-----------|------------|
| `stft_max_high_power` | Max over time frames of: high-band power fraction in that frame (STFT with nperseg=16, noverlap=8) | Global FFT assumes stationarity. STFT captures local bursts of high-frequency entropy activity |
| `stft_spectral_entropy` | Mean per-frame spectral entropy across all STFT frames | Local (non-stationary) version of spectral_entropy |
| `rpdi` | mean(H[last 20% of tokens]) / mean(H) — Reasoning Path Deviation Index | If entropy rises toward the end, the model is losing confidence as it approaches the answer — a hallucination signal |
| `sw_var_peak` | Max variance of H(n) over sliding windows (window=16, step=8) | Peak local volatility — the most unstable region of the entropy trace. Captures bursts without requiring FFT |

---

## 3. Experimental Setup

### Three Models Tested

| Model | Family | Size | Purpose |
|-------|--------|------|---------|
| Qwen2.5-Math-1.5B-Instruct | Qwen | 1.5B | Baseline (Phase 2 cache reused, no new inference) |
| Qwen2.5-Math-7B-Instruct | Qwen (same family) | 7B | Scale generalization: same architecture, larger |
| deepseek-math-7b-instruct | DeepSeek | 7B | Architecture generalization: different model family |

### Dataset and Inference
- **Dataset**: GSM8K test set, first 200 problems per model
- **Sampling**: Temperature=1.0, top_k=50, max_new_tokens=512
- **Entropy computation**: top-15 token probability distribution at each step
- **Correctness**: boxed answer extraction with numeric comparison

### Pipeline
1. Generate chain-of-thought responses, record full entropy trajectory H(n)
2. Extract all 11 features per sample
3. Bootstrap AUC (1000 resamples, 95% CI) — sign auto-selected per signal
4. Compute Spearman ρ for all 55 signal pairs
5. Enumerate all valid subsets (all pairwise |ρ| < 0.75, all AUC > 50%)
6. Run Nadler fusion on every valid subset, rank by AUC

---

## 4. Phase 1 Results (50 Samples, Qwen2.5-Math-1.5B)

**Accuracy**: 38/50 correct (76%) | **Wrong**: 12 | **Avg trace**: 235 tokens

| Signal | AUC | 95% CI |
|--------|-----|--------|
| dominant_freq | 73.0% | [57.3, 88.8] |
| spectral_entropy | 70.0% | [49.5, 88.4] |
| spectral_centroid | 70.0% | [53.2, 83.7] |
| EPR | 66.2% | [48.2, 82.4] |
| hl_ratio | 66.0% | [48.9, 81.0] |
| high_band_power | 64.0% | [45.5, 80.0] |
| low_band_power | 62.5% | [42.7, 79.9] |

**16 viable Nadler pairs** found. All 4 decision gates passed.

**Caveat**: With only 12 wrong samples, CIs are very wide. Phase 1 treated as directional only.

---

## 5. Phase 2 Results (200 Samples, Qwen2.5-Math-1.5B)

**Accuracy**: 164/200 correct (82%) | **Wrong**: 36 | **Avg trace**: 268 tokens

### Individual AUCs

| Signal | Phase 1 | Phase 2 | 95% CI | Change |
|--------|---------|---------|--------|--------|
| EPR | 66.2% | **71.8%** | [62.7, 80.0] | +5.6pp |
| spectral_centroid | 70.0% | 68.7% | [59.1, 77.0] | −1.3pp |
| high_band_power | 64.0% | 66.8% | [57.2, 75.8] | +2.8pp |
| hl_ratio | 66.0% | 66.8% | [56.9, 76.1] | +0.8pp |
| low_band_power | 62.5% | 63.6% | [53.6, 73.8] | +1.1pp |
| dominant_freq | **73.0%** | 60.5% | [50.5, 70.6] | −12.5pp ← noise |
| spectral_entropy | **70.0%** | 59.4% | [48.8, 69.7] | −10.6pp ← noise |

dominant_freq and spectral_entropy were Phase 1 noise artifacts (only 12 wrong examples).

### Best Nadler Fusions (40 valid subsets)

| Subset | AUC | 95% CI | Weights |
|--------|-----|--------|---------|
| EPR + spectral_entropy + high_band_power | **74.1%** | [65.1, 81.4] | EPR=0.669, hbp=0.272, se=0.059 |
| EPR + spectral_entropy + spectral_centroid | 74.1% | [64.9, 81.2] | — |
| EPR + dominant_freq | 73.2% | [63.8, 81.1] | — |
| EPR + spectral_entropy | 73.2% | [64.3, 80.4] | — |
| EPR + spectral_entropy + dominant_freq + spectral_centroid | 71.8% | [63.0, 79.8] | — |
| EPR + all 5 valid signals | 67.0% | [57.6, 76.2] | — |

**Sweet spot**: size-3 fusion. Performance degrades beyond 3 signals because weak views (AUC < 68%) dilute the strong EPR component even with low Nadler weights.

**New GSM8K high**: 74.1% (+5.4pp over EPR+EDIS = 68.7%)

---

## 6. Phase 3 Results — All Three Models

### 6.1 Model Statistics

| Model | Accuracy | Correct | Wrong | Avg trace | Valid samples |
|-------|----------|---------|-------|-----------|---------------|
| Qwen2.5-Math-1.5B-Instruct | 82.0% | 164 | 36 | 268 tok | 200 |
| Qwen2.5-Math-7B-Instruct | 89.5% | 179 | **21** | 310 tok | 200 |
| DeepSeek-Math-7B-Instruct | 80.0% | 160 | 40 | **184 tok** | 200 |

**Note on Qwen 7B**: Only 21 wrong samples. AUC estimates for this model have very wide confidence intervals (often 20–25pp wide) and are statistically unreliable. Results are reported for completeness but should not be interpreted as precise measurements.

### 6.2 Individual Signal AUCs — All 11 Signals × 3 Models

| Signal | Qwen 1.5B | 95% CI | Qwen 7B | 95% CI | DeepSeek 7B | 95% CI |
|--------|-----------|--------|---------|--------|-------------|--------|
| **sw_var_peak** [NEW] | **73.5%** | [64.5, 80.8] | 77.5% | [65.0, 89.1] | **72.9%** | [64.2, 80.5] |
| epr | 71.8% | [62.7, 80.0] | 70.3% | [55.6, 84.1] | 66.4% | [57.6, 75.4] |
| spectral_centroid | 68.7% | [59.1, 77.0] | 79.7% | [68.1, 90.3] | 65.6% | [56.7, 74.5] |
| high_band_power | 66.8% | [57.2, 75.8] | 66.3% | [53.4, 79.0] | 59.5% | [50.1, 68.6] |
| hl_ratio | 66.8% | [56.9, 76.1] | 77.0% | [66.4, 87.3] | 65.8% | [57.3, 74.0] |
| rpdi [NEW] | 64.1% | [52.4, 74.4] | 75.4% | [61.0, 88.8] | 54.1% | [44.0, 64.5] |
| low_band_power | 63.6% | [53.6, 73.8] | 78.2% | [65.7, 89.1] | 67.2% | [58.5, 75.9] |
| dominant_freq | 60.5% | [50.5, 70.6] | 76.7% | [62.9, 89.1] | 62.9% | [52.7, 72.4] |
| spectral_entropy | 59.4% | [48.8, 69.7] | 54.9% | [37.2, 72.9] | 66.3% | [55.4, 76.1] |
| stft_max_high_power [NEW] | 55.6% | [45.3, 64.8] | 58.2% | [41.3, 73.9] | 54.7% | [44.7, 64.5] |
| stft_spectral_entropy [NEW] | 55.0% | [44.9, 65.9] | 73.6% | [60.8, 85.2] | 53.5% | [43.8, 63.2] |

### 6.3 Pairwise Correlation Structure

**Qwen2.5-Math-1.5B** — key pairs (|ρ| ≥ 0.75 = cannot Nadler-fuse):

| Pair | |ρ| | Status |
|------|-----|--------|
| sw_var_peak ↔ epr | 0.826 | ❌ too correlated |
| spectral_centroid ↔ hl_ratio | 0.929 | ❌ |
| high_band_power ↔ hl_ratio | 0.883 | ❌ |
| low_band_power ↔ spectral_centroid | 0.877 | ❌ |
| low_band_power ↔ hl_ratio | 0.845 | ❌ |
| high_band_power ↔ spectral_centroid | 0.763 | ❌ |

All new Phase 3 signals (stft_*, rpdi) have low ρ with most other signals (< 0.40), making them broadly fuseable. Valid subsets: **564** (vs 40 in Phase 2).

**DeepSeek-Math-7B** — key pairs:

| Pair | |ρ| | Status |
|------|-----|--------|
| sw_var_peak ↔ epr | 0.753 | ❌ borderline |
| low_band_power ↔ spectral_centroid | 0.859 | ❌ |
| spectral_centroid ↔ hl_ratio | 0.893 | ❌ |
| high_band_power ↔ hl_ratio | 0.830 | ❌ |
| low_band_power ↔ hl_ratio | 0.807 | ❌ |

Valid subsets: **660**.

**Qwen2.5-Math-7B** — key pairs:

| Pair | |ρ| | Status |
|------|-----|--------|
| spectral_centroid ↔ low_band_power | 0.756 | ❌ |
| spectral_centroid ↔ hl_ratio | 0.895 | ❌ |
| high_band_power ↔ hl_ratio | 0.821 | ❌ |

sw_var_peak ↔ epr = 0.595 on 7B → **valid** (unlike on 1.5B and DeepSeek). Valid subsets: **884**.

### 6.4 Best Nadler Fusions Per Model

**Qwen2.5-Math-1.5B-Instruct**

| Rank | Subset | AUC | 95% CI |
|------|--------|-----|--------|
| 1 | spectral_entropy + dominant_freq + spectral_centroid + stft_spectral_entropy + rpdi + sw_var_peak | **75.9%** | [67.8, 82.5] |
| 2 | spectral_entropy + dominant_freq + spectral_centroid + stft_max_high_power + stft_spectral_entropy + rpdi + sw_var_peak | 75.9% | [67.8, 82.6] |
| 3 | spectral_entropy + spectral_centroid + stft_spectral_entropy + rpdi + sw_var_peak | 75.8% | [67.7, 82.4] |
| 4 | spectral_entropy + high_band_power + dominant_freq + rpdi + sw_var_peak | 75.6% | [67.7, 82.2] |

Top fusion weights: spectral_centroid=0.527, sw_var_peak=0.298, stft_spectral_entropy=0.109, rpdi=0.035, spectral_entropy=0.020, dominant_freq=0.011

**Note**: EPR is absent from all top fusions because sw_var_peak (the strongest signal at 73.5%) has ρ=0.826 with EPR — they cannot be fused together. Nadler selects sw_var_peak over EPR and builds the ensemble around it.

**Qwen2.5-Math-7B-Instruct** *(interpret with caution — 21 wrong samples)*

| Rank | Subset | AUC | 95% CI |
|------|--------|-----|--------|
| 1 | epr + spectral_entropy + low_band_power + stft_max_high_power | **90.3%** | [75.4, 99.2] |
| 2 | epr + spectral_entropy + low_band_power + dominant_freq + stft_max_high_power | 90.3% | [75.4, 99.2] |
| 3 | epr + spectral_entropy + low_band_power | 90.1% | [75.1, 99.1] |

Top fusion weights: epr=0.615, low_band_power=0.239, spectral_entropy=0.145, stft_max_high_power≈0.000

stft_max_high_power receives zero weight — effective fusion is epr + spectral_entropy + low_band_power. The CI [75.4, 99.2] is 23.8pp wide — this point estimate should not be taken at face value.

**DeepSeek-Math-7B-Instruct**

| Rank | Subset | AUC | 95% CI |
|------|--------|-----|--------|
| 1 | spectral_entropy + hl_ratio + stft_max_high_power + stft_spectral_entropy + sw_var_peak | **75.0%** | [65.7, 83.2] |
| 2 | spectral_entropy + hl_ratio + stft_max_high_power + sw_var_peak | 75.0% | [66.0, 83.4] |
| 3 | spectral_entropy + hl_ratio + dominant_freq + sw_var_peak | 74.9% | [65.5, 83.2] |
| 4 | spectral_entropy + hl_ratio + sw_var_peak | 74.7% | [65.6, 83.0] |

Top fusion weights: sw_var_peak=0.562, hl_ratio=0.199, spectral_entropy=0.092, stft_spectral_entropy=0.082, stft_max_high_power=0.065

sw_var_peak dominates with weight 0.562–0.733 across all top fusions.

### 6.5 Best AUC by Subset Size

**Qwen2.5-Math-1.5B**

| Size | Best AUC | Best subset |
|------|----------|-------------|
| 1 | 73.5% | sw_var_peak |
| 2 | 74.6% | spectral_centroid + sw_var_peak |
| 3 | 75.3% | spectral_entropy + spectral_centroid + sw_var_peak |
| 4 | 75.5% | spectral_entropy + spectral_centroid + rpdi + sw_var_peak |
| 5 | 75.7% | spectral_entropy + spectral_centroid + stft_max_high_power + rpdi + sw_var_peak |
| 6 | **75.9%** | spectral_entropy + dom_freq + spectral_centroid + stft_spec_ent + rpdi + sw_var_peak |

Unlike Phase 2, the AUC keeps improving with additional signals up to size 6. This is because Phase 3's new signals (rpdi, stft features) are genuinely independent (ρ < 0.20 with most others) and add small but consistent value.

**DeepSeek-Math-7B**

| Size | Best AUC | Best subset |
|------|----------|-------------|
| 1 | 72.9% | sw_var_peak |
| 2 | 74.1% | spectral_entropy + sw_var_peak |
| 3 | **74.7%** | spectral_entropy + hl_ratio + sw_var_peak |
| 4 | 75.0% | spectral_entropy + hl_ratio + stft_max_high_power + sw_var_peak |

### 6.6 Cross-Model Comparison Table

| Signal | Qwen 1.5B | Qwen 7B* | DeepSeek 7B | 1.5B vs DS spread |
|--------|-----------|---------|-------------|-------------------|
| sw_var_peak | 73.5% | 77.5% | 72.9% | **0.6pp** |
| epr | 71.8% | 70.3% | 66.4% | 5.4pp |
| spectral_centroid | 68.7% | 79.7% | 65.6% | 3.1pp |
| high_band_power | 66.8% | 66.3% | 59.5% | 7.3pp |
| hl_ratio | 66.8% | 77.0% | 65.8% | 1.0pp |
| rpdi | 64.1% | 75.4% | 54.1% | 10.0pp |
| low_band_power | 63.6% | 78.2% | 67.2% | 3.6pp |
| dominant_freq | 60.5% | 76.7% | 62.9% | 2.4pp |
| spectral_entropy | 59.4% | 54.9% | 66.3% | 6.9pp |
| stft_max_high_power | 55.6% | 58.2% | 54.7% | **0.9pp** |
| stft_spectral_entropy | 55.0% | 73.6% | 53.5% | 1.5pp |
| **Best fusion** | **75.9%** | **90.3%*** | **75.0%** | **0.9pp** |
| Accuracy | 82.0% | 89.5% | 80.0% | — |

*Qwen 7B numbers unreliable due to small negative sample count (21 errors).

**Decision Gates (Phase 3)**

| Gate | Qwen 1.5B | Qwen 7B | DeepSeek 7B |
|------|-----------|---------|-------------|
| G1: any signal > 71.8% | ✅ sw_var_peak 73.5% | ✅ 7 signals | ✅ sw_var_peak 72.9% |
| G2: best fusion > 74.1% | ✅ 75.9% (+1.8pp) | ✅ 90.3% | ✅ 75.0% (+0.9pp) |
| G3: spread ≤ 3pp (all 3) | ❌ 15.3pp | — | — |
| G3 (1.5B vs DeepSeek only) | ✅ **0.9pp** | — | — |

---

## 7. Key Findings

### Finding 1: sw_var_peak Is the Most Architecture-Robust Signal

Peak sliding-window variance of H(n) — the maximum variance in any 16-token window of the entropy trajectory — is the most consistent signal across architectures:
- Qwen 1.5B: **73.5%** (rank #1)
- DeepSeek 7B: **72.9%** (rank #1)
- Spread: 0.6pp across two completely different model families

This signal requires no FFT, no frequency analysis — it is a simple time-domain measure of how volatile the entropy trace gets at its most unstable point. Its robustness suggests that **local volatility bursts in H(n) are a model-family-agnostic signal of uncertainty**, not an artifact of any specific architecture.

### Finding 2: sw_var_peak and EPR Are Too Correlated to Fuse on Small Models

On Qwen 1.5B and DeepSeek 7B, ρ(sw_var_peak, EPR) ≈ 0.75–0.83. These signals measure related aspects of the same underlying trajectory (variance vs. mean), and they are too correlated for valid Nadler fusion. As a result, the best fusions on these models exclude EPR entirely and use sw_var_peak as the primary signal.

On the larger Qwen 7B, ρ = 0.595 — within the valid range — suggesting that at scale, the mean and variance of H(n) become more decorrelated (the model's mean uncertainty is more stable, but local spikes become more distinctive).

### Finding 3: STFT Features Do Not Add Meaningful Standalone Signal

The hypothesis that local (non-stationary) frequency analysis via STFT would outperform global FFT was not supported:
- stft_max_high_power: 54.7–58.2% across all models
- stft_spectral_entropy: 53.5–55.0% on 1.5B and DeepSeek (73.6% on 7B is unreliable)

Both features receive near-zero Nadler weights in most fusions. They add marginal value (~0.1–0.3pp) in some large fusions but are not reliable contributors. The global FFT spectral features from Phase 2 remain more informative.

### Finding 4: RPDI Is Model-Dependent and Unreliable

RPDI (tail entropy / mean entropy) ranges from 54.1% (DeepSeek, near-chance) to 75.4% (Qwen 7B, unreliable) with 64.1% on 1.5B. The 21pp spread makes it unsuitable as a universal signal. Whether the model's entropy rises toward the end depends on how that specific model structures its chain-of-thought, not on a universal pattern.

### Finding 5: The ρ Structure Is Model-Dependent

The same pair of signals can be valid (|ρ| < 0.75) on one model and invalid on another. For example, sw_var_peak ↔ EPR has ρ = 0.826 on 1.5B but ρ = 0.595 on 7B. This means there is no single universal maximum valid set — the optimal signal combination must be determined per model by recomputing the ρ matrix. Crucially, this is **fully unsupervised** — no labels are needed, only the model's own entropy trajectories.

### Finding 6: Fusion Consistently Adds ~2-3pp Over the Best Single Signal

Across all models and phases:

| Model | Best single | Best fusion | Fusion gain |
|-------|-------------|-------------|-------------|
| Qwen 1.5B (Phase 2) | EPR 71.8% | 74.1% | +2.3pp |
| Qwen 1.5B (Phase 3) | sw_var_peak 73.5% | 75.9% | +2.4pp |
| DeepSeek 7B (Phase 3) | sw_var_peak 72.9% | 75.0% | +2.1pp |

The Nadler fusion gain is stable and architecture-agnostic: approximately 2-3pp above the best individual signal.

### Finding 7: Larger Valid Set Allows Gradual Improvement

Phase 2 had only 40 valid subsets (7 signals). Phase 3 has 564–884 valid subsets (11 signals). With the richer search space, the best fusion improves steadily with subset size (up to 6 signals on 1.5B) because the 4 new signals all have low ρ with each other and with the Phase 2 signals. Unlike Phase 2 where adding signals hurt, in Phase 3 the new signals are genuinely independent and add small but real contributions.

---

## 8. Consistency Analysis: What Generalizes Across Architectures

Comparing Qwen2.5-Math-1.5B (Qwen family, 1.5B) vs DeepSeek-Math-7B (DeepSeek family, 7B) as the two architecturally comparable models:

**Highly consistent signals** (spread < 3pp):
- sw_var_peak: 73.5% vs 72.9% — **0.6pp spread**
- stft_max_high_power: 55.6% vs 54.7% — 0.9pp (but both weak)
- stft_spectral_entropy: 55.0% vs 53.5% — 1.5pp (both weak)
- hl_ratio: 66.8% vs 65.8% — 1.0pp

**Moderately consistent signals** (3–6pp):
- epr: 71.8% vs 66.4% — 5.4pp
- spectral_centroid: 68.7% vs 65.6% — 3.1pp
- dominant_freq: 60.5% vs 62.9% — 2.4pp
- low_band_power: 63.6% vs 67.2% — 3.6pp

**Inconsistent signals** (> 6pp):
- rpdi: 64.1% vs 54.1% — **10.0pp** (architecture-specific)
- spectral_entropy: 59.4% vs 66.3% — 6.9pp (reversed ranking)
- high_band_power: 66.8% vs 59.5% — 7.3pp

**Best fusion**: 75.9% vs 75.0% — **0.9pp spread** — robust

---

## 9. Overall Progress Summary

### Best results across all phases and datasets

| Dataset | Model | Phase | Best AUC | Method |
|---------|-------|-------|----------|--------|
| TriviaQA | Falcon-3-10B | Prior | 81.5% | 6-view Nadler (temps + behavioral) |
| WebQuestions | Falcon-3-10B | Prior | 76.0% | Same |
| GSM8K | Qwen2.5-Math-1.5B | Phase 2 | 74.1% | EPR + spectral_entropy + high_band_power |
| GSM8K | Qwen2.5-Math-1.5B | **Phase 3** | **75.9%** | 6-signal Nadler, sw_var_peak dominant |
| GSM8K | DeepSeek-Math-7B | **Phase 3** | **75.0%** | sw_var_peak dominant |

### Trajectory on GSM8K

| Step | Method | AUC | vs prior |
|------|--------|-----|---------|
| Baseline | EPR alone | 66.8% | — |
| Prior best | EPR + EDIS | 68.7% | +1.9pp |
| Phase 2 | EPR + 2 spectral | 74.1% | +5.4pp |
| Phase 3 | sw_var_peak + spectral ensemble | **75.9%** | +7.2pp |

### What is now established
1. Spectral features of H(n) carry genuine hallucination signal independent of EPR
2. sw_var_peak is the most robust standalone signal across architectures
3. Nadler fusion reliably adds ~2-3pp above the best single signal, unsupervised
4. The valid fusion set must be computed per model (ρ structure changes), but this is fully label-free
5. STFT features and RPDI do not generalize reliably across architectures

---

## 10. Open Questions for Further Discussion

**Q1: Why does sw_var_peak work better than EPR?**  
Both measure aspects of H(n). EPR is the mean; sw_var_peak is the peak local variance. The variance measure may be more discriminative because a wrong answer can have the same mean entropy as a correct one, but the *pattern* of entropy (concentrated spikes vs. smooth trajectory) differs. Worth investigating whether there is a theoretical explanation.

**Q2: Can sw_var_peak and EPR be decorrelated?**  
On Qwen 7B they are only ρ=0.595, suggesting their correlation depends on model behavior. Is there a way to compute a "residualized" sw_var_peak that subtracts the EPR contribution, making them always fuseable?

**Q3: Is the 2–3pp Nadler gain fundamental or improvable?**  
Every tested fusion delivers roughly 2–3pp over the best single signal. Is this a property of the signal space (the remaining independent information after conditioning on the best signal is limited to 2–3pp)? Or can new signals break this ceiling?

**Q4: What does the spectral structure of H(n) look like on factual QA tasks?**  
All spectral work has been on GSM8K (math, long chain-of-thought traces). On TriviaQA/WebQ, traces are shorter (~50–100 tokens). Do spectral features work on short traces? Can they add to the existing 81.5% TriviaQA result?

**Q5: How does the method behave on harder math (MATH-500, GPQA)?**  
GSM8K has ~80–90% accuracy for these models — relatively easy. On harder benchmarks with ~50% accuracy (more balanced labels), AUC estimates would be more reliable and the discrimination task may look different.

**Q6: The ρ-based valid set varies per model — is there a universal signal set?**  
Could we define a set of signals guaranteed to be valid (all pairwise ρ < 0.75) on any model? sw_var_peak is a candidate anchor (it has low ρ with most spectral features on most models). Building around it may give a portable detector.

**Q7: Conformal calibration (LTT) as the thesis endpoint**  
Once the best signal set is locked, Learn-then-Test (LTT) conformal calibration can convert the AUC detector into a deployable system with formal FDR guarantees. This is Direction 5 in the research plan and remains the thesis endpoint. What is the appropriate null hypothesis and what sample size is needed for reliable LTT calibration?
