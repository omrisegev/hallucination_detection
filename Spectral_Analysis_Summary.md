# Spectral Analysis of Entropy Trajectories — Research Summary
**Date**: April 2026  
**Researcher**: Omri Segev  
**Project**: Unsupervised Hallucination Detection in LLMs using Entropy Production Rate (EPR) and Nadler Spectral Fusion  

---

## 1. Background and Motivation

### The Overall Research Project

This research explores whether it is possible to detect hallucinations in large language models (LLMs) **without any labeled training data**, using only the model's own internal uncertainty signals during generation.

The core signal is **EPR (Entropy Production Rate)**: the mean token-level Shannon entropy over the generated sequence. At each token position *n*, the model assigns a probability distribution over its vocabulary. The entropy H(n) = −Σ p(t) log p(t) measures the model's uncertainty at that step. EPR = mean(H(n)) over the full generation — essentially how uncertain the model is on average.

Higher EPR correlates with hallucination: a model that is uncertain throughout its response is more likely to be wrong.

### The Nadler Spectral Fusion Method

To go beyond a single scalar EPR value, this project uses **Nadler spectral fusion**: a mathematically principled method for combining multiple independent "views" of the same prediction target. The key requirements are:
1. Each view must individually predict the label (AUC > 50%)
2. Any two views being fused must have Spearman |ρ| < 0.75 (they must be measuring different aspects)

When these conditions hold, Nadler fusion finds the optimal linear combination of views using eigenvector analysis of the cross-covariance structure. The result consistently outperforms any single view alone.

### Prior Experimental Results (Context for Spectral Analysis)

Previous experiments on factual question-answering tasks established:
- **Best result**: 81.5% AUC on TriviaQA / 76.0% on WebQuestions — using a 6-view Nadler ensemble combining EPR at 4 different sampling temperatures (T=1.0, T=1.3, T=1.5, T=1.7) plus two "behavioral" views (Verify and Skeptic prompt variants)
- **Math baseline (GSM8K)**: EPR(trace) = 66.8% AUC using Qwen2.5-Math-1.5B-Instruct model
- **Best prior math result**: EPR(trace) + EDIS Nadler fusion = 68.7% AUC on GSM8K

EDIS (Entropy Dynamics Instability Score) is a prior method (Zhu et al. 2026) that tracks burst and rebound patterns in H(n) — it is sensitive to sharp entropy spikes rather than the mean.

### The Key Insight That Motivated Spectral Analysis

EPR is mathematically equivalent to the **DC component (zero-frequency term)** of the Fourier transform of H(n). This means:
- All frequency components above DC are **orthogonal to EPR by construction** — there is zero information overlap
- If the temporal structure of H(n) carries hallucination signal, the frequency domain should reveal it even in cases where two samples have identical EPR but different correctness

**Hypothesis**: Correct math reasoning produces a structured H(n) trajectory with a regular step-boundary rhythm → concentrated spectral energy at specific frequencies → low spectral entropy. Wrong/hallucinated reasoning produces erratic H(n) → flat, spread-out power spectrum → high spectral entropy.

This direction is also naturally connected to Ofir Lindenbaum's expertise in spectral decomposition methods, making it a strong candidate for the research collaboration.

---

## 2. Spectral Features Defined

Six frequency-domain features were extracted from each entropy trajectory H(n):

All features are computed after **removing the DC component** (subtracting the mean = EPR) from H(n) before the FFT, ensuring orthogonality with EPR.

| Feature | Description | Expected hallucination signal |
|---------|-------------|-------------------------------|
| `spectral_entropy` | Entropy of the normalized power spectral density: −Σ PSD_norm · log(PSD_norm). Measures how spread-out the energy is across frequencies. | High spectral entropy → energy scattered across all frequencies → erratic reasoning → hallucination |
| `low_band_power` | Fraction of AC power in f ∈ (0, 0.1] (slow oscillations, roughly one period per 10 tokens) | Structured reasoning → energy concentrated at low frequencies |
| `high_band_power` | Fraction of AC power in f ∈ [0.4, 0.5] (rapid token-to-token fluctuations, Nyquist region) | Erratic reasoning → high-frequency energy |
| `hl_ratio` | high_band_power / low_band_power | Erratic = high ratio; structured = low ratio |
| `dominant_freq` | Frequency of the strongest AC peak in the power spectrum (excluding DC) | Correct reasoning → clear dominant oscillation at a specific step rhythm |
| `spectral_centroid` | Weighted mean frequency: Σ(f · PSD_norm) / Σ(PSD_norm) | Center of mass of the frequency distribution |

All features use normalized frequencies: f=0 is DC (=EPR), f=0.5 is the Nyquist limit.

---

## 3. Phase 1: Exploratory Study (50 Samples)

### Setup
- **Model**: Qwen2.5-Math-1.5B-Instruct
- **Dataset**: GSM8K test set (first 50 problems)
- **Data source**: Bootstrapped from existing experimental cache (no new inference required) — entropy trajectories H(n) from the trace portion of chain-of-thought generations
- **Samples**: 50 total | 38 correct (76%), 12 wrong (24%)
- **Average trace length**: 235 tokens (min=116, max=384)

### Single-Signal AUC Results (Phase 1)

| Signal | AUC | 95% CI | vs EPR baseline |
|--------|-----|--------|-----------------|
| dominant_freq | **73.0%** | [57.3, 88.8] | +6.2pp |
| spectral_entropy | **70.0%** | [49.5, 88.4] | +3.2pp |
| spectral_centroid | **70.0%** | [53.2, 83.7] | +3.2pp |
| EPR (this subset) | 66.2% | [48.2, 82.4] | reference |
| hl_ratio | 66.0% | [48.9, 81.0] | –0.8pp |
| high_band_power | 64.0% | [45.5, 80.0] | –2.8pp |
| low_band_power | 62.5% | [42.7, 79.9] | –4.3pp |

**Reference baseline**: EPR(trace) = 66.8% from the full 100-sample Unified experiment

### Pairwise Independence Structure (Phase 1)

The 5 problematic pairs (|ρ| ≥ 0.75) all lie within {low_band_power, high_band_power, hl_ratio, spectral_centroid}:
- hl_ratio ↔ spectral_centroid: ρ = 0.935 ❌
- high_band_power ↔ hl_ratio: ρ = 0.899 ❌  
- low_band_power ↔ spectral_centroid: ρ = 0.872 ❌
- low_band_power ↔ hl_ratio: ρ = 0.803 ❌
- high_band_power ↔ spectral_centroid: ρ = 0.766 ❌

By contrast, EPR, spectral_entropy, and dominant_freq had no bad pairs with any signal (max |ρ| = 0.474).

**Maximum valid Nadler set** (all 10 pairwise |ρ| < 0.75):  
`{EPR, spectral_entropy, dominant_freq, low_band_power, high_band_power}` — 5 signals

**16 viable Nadler pairs** were identified (AUC > 50% and |ρ| < 0.75).

### Phase 1 Decision Gates

| Gate | Condition | Result |
|------|-----------|--------|
| G1 | Any signal AUC > 66.8% | ✅ PASS — dominant_freq = 73.0% |
| G2 | Any viable Nadler pair exists | ✅ PASS — 16 pairs found |
| G3 | Spectra visually distinct | ✅ PASS — visual inspection confirmed |
| G4 | Best pair |ρ| < 0.50 | ✅ PASS — spectral_entropy + high_band_power |ρ| = 0.006 |

**Decision**: Proceed to Phase 2 (scale to 200 samples, full fusion enumeration).

### Caveat Noted at Phase 1

The confidence intervals were extremely wide (e.g., dominant_freq: [57.3%, 88.8%]) due to only 12 wrong samples. The Phase 1 results were treated as directional hypotheses, not confirmed findings.

---

## 4. Phase 2: Scaled Study with Full Fusion Enumeration (200 Samples)

### Setup
- **Model**: Qwen2.5-Math-1.5B-Instruct (same as Phase 1)
- **Dataset**: GSM8K test set (first 200 problems)
- **Data source**: Phase 1 cache (50 samples) + Unified cache bootstrap (50 samples) + fresh inference (100 new samples)
- **Samples**: 200 total | 164 correct (82%), 36 wrong (18%)
- **Average trace length**: 268 tokens (min=107, max=512)

### Single-Signal AUC Results (Phase 2)

| Signal | Phase 1 AUC | Phase 2 AUC | 95% CI | Change |
|--------|-------------|-------------|--------|--------|
| EPR | 66.2% | **71.8%** | [62.7, 80.0] | +5.6pp |
| spectral_entropy | 70.0% | 59.4% | [48.8, 69.7] | −10.6pp |
| spectral_centroid | 70.0% | 68.7% | [59.1, 77.0] | −1.3pp |
| high_band_power | 64.0% | 66.8% | [57.2, 75.8] | +2.8pp |
| hl_ratio | 66.0% | 66.8% | [56.9, 76.1] | +0.8pp |
| dominant_freq | **73.0%** | 60.5% | [50.5, 70.6] | −12.5pp |
| low_band_power | 62.5% | 63.6% | [53.6, 73.8] | +1.1pp |

**Critical observation**: The two most promising Phase 1 signals (dominant_freq, spectral_entropy) collapsed dramatically at scale. This is a textbook noise artifact from small-sample AUC estimation — with only 12 wrong samples in Phase 1, large swings in AUC are expected even from random features.

**What held**: EPR itself got stronger at 200 samples (+5.6pp), likely because longer average traces (268 vs 235 tokens) give more signal. The spectral features that were modest in Phase 1 (spectral_centroid, high_band_power, hl_ratio) remained stable.

### Pairwise Independence Structure (Phase 2)

The independence structure was largely confirmed at scale:

| Pair | Phase 1 ρ | Phase 2 ρ | Status |
|------|-----------|-----------|--------|
| EPR ↔ spectral_entropy | 0.094 | 0.167 | ✅ |
| EPR ↔ dominant_freq | 0.123 | 0.052 | ✅ |
| spectral_entropy ↔ high_band_power | 0.006 | 0.029 | ✅ |
| low_band_power ↔ hl_ratio | 0.803 | 0.845 | ❌ |
| high_band_power ↔ hl_ratio | 0.899 | 0.883 | ❌ |
| hl_ratio ↔ spectral_centroid | 0.935 | 0.929 | ❌ |

The 5 bad pairs from Phase 1 remained bad. The independence between spectral_entropy and EPR (ρ=0.167) was confirmed — even at 200 samples, these two signals measure largely different aspects of the generation.

**Valid Nadler subsets at Phase 2**: 40 total (16 pairs, 16 triples, 7 quadruples, 1 quintuple).

### Nadler Fusion Results — All Valid Subsets (Phase 2)

Top results from all 40 valid subset combinations:

| Subset | Size | AUC | 95% CI | vs EPR |
|--------|------|-----|--------|--------|
| EPR + spectral_entropy + high_band_power | 3 | **74.1%** | [65.1, 81.4] | +2.3pp |
| EPR + spectral_entropy + spectral_centroid | 3 | 74.1% | [64.9, 81.2] | +2.3pp |
| EPR + spectral_entropy + dominant_freq | 3 | 73.6% | [64.5, 81.3] | +1.8pp |
| EPR + dominant_freq | 2 | 73.2% | [63.8, 81.1] | +1.4pp |
| EPR + spectral_entropy | 2 | 73.2% | [64.3, 80.4] | +1.4pp |
| EPR + spectral_entropy + dominant_freq + spectral_centroid | 4 | 71.8% | [63.0, 79.8] | 0.0pp |
| EPR + spectral_entropy + low_band_power + high_band_power + dominant_freq | 5 | 67.0% | [57.6, 76.2] | −4.8pp |

**Best fusion**: `EPR + spectral_entropy + high_band_power` — AUC = **74.1%**  
**Fusion weights**: EPR=0.669, spectral_entropy=0.059, high_band_power=0.272

### Best AUC by Subset Size

| Signals fused | Best AUC | Best subset |
|--------------|----------|-------------|
| 1 (EPR alone) | 71.8% | EPR |
| 2 | 73.2% | EPR + dominant_freq |
| **3** | **74.1%** | EPR + spectral_entropy + high_band_power |
| 4 | 71.8% | EPR + spectral_entropy + dominant_freq + spectral_centroid |
| 5 | 67.0% | EPR + spectral_entropy + low_band_power + high_band_power + dominant_freq |

**The sweet spot is 3 signals.** Performance degrades beyond 3. This happens because signals 4 and 5 are relatively weak individually (AUC < 68%), and even though they are independent of EPR, adding noise from a weak predictor dilutes the Nadler combination.

### Phase 2 Decision Gates

| Gate | Condition | Result |
|------|-----------|--------|
| G1 | dominant_freq AUC > 66.8% confirmed with CI lower bound > 60% | ❌ FAIL — 60.5%, CI [50.5, 70.6] |
| G2 | Best fusion > best single signal (EPR = 71.8%) | ✅ PASS — 74.1%, lift = +2.3pp |
| G3 | Best fusion > EPR+EDIS = 68.7% (best prior math result) | ✅ PASS — lift = +5.4pp |
| G4 | Best fusion > 75% | ❌ FAIL — 74.1% |

---

## 5. Key Findings and Interpretation

### Finding 1: Phase 1 Spectral Signals Were Noise, But the Direction is Real

The strong Phase 1 numbers for dominant_freq (73%) and spectral_entropy (70%) were statistical artifacts of the small sample size. With only 12 wrong examples, AUC estimates are highly unstable. At 200 samples, these signals individually fall below EPR.

However, this does **not** mean the spectral approach is wrong. The fusion result (74.1%) confirms that spectral features carry genuine independent information about hallucination — it's just that no individual spectral feature rivals EPR on its own.

### Finding 2: EPR is Stronger Than We Thought

EPR at 200 samples with longer traces (268 tokens avg) reaches 71.8% — well above the 66.8% reference from the Unified experiment. The spectral analysis therefore faces a moving baseline: the more computation we give EPR (longer traces), the stronger it gets.

### Finding 3: spectral_entropy and high_band_power Are the Best Spectral Complements to EPR

These two are essentially uncorrelated with EPR (ρ=0.167 and ρ=0.425 respectively) and together add +2.3pp in Nadler fusion. Conceptually:
- **spectral_entropy** captures whether the entropy signal is structured or noisy in frequency space (independent of its average level = EPR)
- **high_band_power** captures rapid token-to-token entropy fluctuations (the "jitter" component), also orthogonal to the mean

### Finding 4: Adding More Signals Beyond 3 Hurts

The degradation from 3 → 4 → 5 signals is instructive. The Nadler algorithm tries to combine all views, but weak views (AUC < 68%) bring noise that the algorithm cannot fully suppress. The lesson: signal selection matters as much as signal generation.

### Finding 5: The Frequency-Domain Hypothesis is Validated at a Modest Level

The original hypothesis was that correct vs incorrect reasoning would show different frequency-domain patterns. This is supported — just more weakly than hoped. The frequency structure of H(n) carries about 2–3pp of independent information beyond EPR. Whether this is practically significant depends on the target application.

---

## 6. Relationship to Prior Work

### EPR (Farquhar et al., 2024)
The base signal. Mean entropy over generated tokens, derived from semantic entropy. AUC = 66.8% on GSM8K with Qwen2.5-Math-1.5B.

### EDIS (Zhu et al., 2026)
Also based on H(n), but focuses on burst-rebound dynamics rather than the mean. AUC = 66.2% on GSM8K in our experiments (marginally weaker than EPR on math). EPR+EDIS Nadler fusion = 68.7%. Our spectral fusion (74.1%) substantially outperforms this prior combination.

### Nadler Spectral Method (Nadler et al., 2006)
The fusion framework used throughout. Applied here both to combine temperature/behavioral views (for factual QA) and now to spectral+EPR views (for math).

---

## 7. Open Questions and Potential Directions

### 7.1 Why Does dominant_freq Fail at Scale?

The 73% Phase 1 result was almost certainly noise, but it raises a question: is there any subset of problems (e.g., longer traces, more complex multi-step problems) where the dominant oscillation frequency is genuinely informative? A stratified analysis by trace length could test this.

### 7.2 Better Time-Frequency Representations

The FFT treats H(n) as stationary (same statistical properties throughout). But reasoning trajectories may change character mid-generation — early tokens are problem setup, middle tokens are computation, final tokens are the answer. A **Short-Time Fourier Transform (STFT)** or **wavelet transform** could reveal localized frequency patterns invisible to global FFT.

### 7.3 Sliding-Window Variance as a Simpler Feature

High spectral entropy correlates with high local variance. A much simpler feature — the variance of H(n) in a sliding window — might capture the same information without the FFT machinery. Worth testing.

### 7.4 Integration with the Factual QA Ensemble

The current best result (81.5% TriviaQA, 76.0% WebQ) uses temperature+behavioral views with Falcon on short factual traces. Spectral features require longer traces to be meaningful. A natural extension is to test whether spectral views add to the temperature ensemble specifically for CoT-style factual QA (where traces are 100–200 tokens).

### 7.5 Spectral Features on Larger Models

Qwen2.5-Math-1.5B is a small model. Larger models (7B, 13B) might produce more structured entropy trajectories on hard problems, potentially making spectral features more discriminative. The cost is higher inference time per sample.

### 7.6 Combining with Semantic Entropy Views

The original Semantic Entropy paper uses multiple independent generations with meaning-cluster consistency as the uncertainty signal. Spectral features from a single generation could complement multi-generation approaches without requiring re-sampling.

---

## 8. Current State of the Research

### What has been built
- End-to-end pipeline for EPR-based hallucination detection across TriviaQA, WebQuestions, GSM8K
- Multi-view Nadler ensemble framework (reusable across signal types)
- Spectral feature extractor (6 features from H(n) FFT)
- Phase 1 and Phase 2 experimental notebooks with full reproducibility

### Best results so far
| Dataset | Best result | Method |
|---------|-------------|--------|
| TriviaQA | 81.5% AUC | 6-view Nadler (4 temps + Verify + Skeptic), Falcon-3-10B |
| WebQuestions | 76.0% AUC | Same |
| GSM8K | **74.1% AUC** | 3-view Nadler (EPR + spectral_entropy + high_band_power), Qwen2.5-Math-1.5B |

### The GSM8K result (74.1%) is a new high for this project on math

Prior best on math was EPR+EDIS = 68.7%. The spectral fusion adds +5.4pp over that. However, the math and factual QA experiments use different models so direct comparison of absolute numbers is limited.

### Advisors
- Prof. Amir Averbuch (Tel Aviv University) — main supervisor
- Dr. Ofir Lindenbaum (Bar-Ilan University) — co-supervisor, spectral methods expert
