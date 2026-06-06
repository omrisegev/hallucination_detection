# MV_EPR — Session Progress Handoff

**Date**: 2026-06-06
**Last updated**: Step 122 — LSML_Optimized ablation complete; V2 adopted (4 features, median)

---

## TL;DR — where we are today

**Current official method**: L-SML (Jaffé-Fetaya-Nadler 2016) with **pre-oriented classifiers + feature selection**.
Pipeline: `binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(*binary_filt.values())`.
Fully unsupervised at test time. FEATURE_SIGNS from Step 110; GOOD_FEATURES from Step 122 ablation.

```python
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max']
```

**Step 122 finding**: quantile calibration is a null result (+0.001). Median binarization is final. Feature selection gives +1.7pp mean across 29 cells (domain-dependent: QA/RAG benefit, GPQA regresses ~−3.8pp).

**The old supervised numbers (Step 100) must not be used or referenced going forward.**

---

## Immediate next actions

### Action 1 — Update + run Consolidated Results notebook ← NEXT

**Task**: `Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb` — CPU-only, ~15–30 min.

Add to Cell 2 config:
```python
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max']
```
In the fusion cell, filter before `lsml_fuse`:
```python
binary = binarize_classifiers(fd, FEATURE_SIGNS)
binary_filt = {fn: binary[fn] for fn in GOOD_FEATURES if fn in binary}
fused, meta = lsml_fuse(*binary_filt.values())
```
This gives the official V2 per-domain/per-model AUROCs for the comparison table.
After running: rebuild `Phase12_Comparison_Results.html`.

**Why V2 not V4**: quantile calibration adds +0.001 (noise). Median binarization is final.

### Action 2 — Run Step 120 notebook on Colab (still pending)
**Task**: `Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb` — CPU-only, ~15–30 min.
- Produces the official oriented L-SML v2 AUROC numbers for the comparison table
- After it runs: rebuild `Phase12_Comparison_Results.html`

**FEATURE_SIGNS** (Step 110 consensus):
```python
FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}
```

---

## Running experiments (Colab — GPU needed)

- **Phase 13**: `Spectral_Analysis_MathComp_Phase13.ipynb` — L-SML vs EDIS on GSM8K+MATH500+AMC23+AIME24, Qwen2.5-Math-1.5B, T=0.2/0.6/1.0, K=8. Currently running or waiting to run.
- **Phase 14**: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb` — L-SML vs VC/SC on GPQA Diamond, DeepSeek-R1-0528-Qwen3-8B. Currently running or waiting to run.

---

## Completed phases

| Phase | Notebook | Status | Key result |
|-------|----------|--------|------------|
| Step 100 | Consolidated_Results | ✅ | Old supervised numbers — do not use |
| Step 107 | Consolidated_Results_LSML | ✅ | L-SML with assumption (iii) — superseded by v2 |
| Step 110 | LSML_Diagnostics | ✅ | Consensus FEATURE_SIGNS derived |
| Step 113 | Pilot_RAG_Prompt_Variants | ✅ | V4 prompt wins (+18.6pp) but RAG direction dropped |
| Phase 12 | Phase12_Benchmarking | ✅ | SE/SC/VC/SelfCheckGPT computed — use for competitor rows |
| Step 121–122 | LSML_Optimized | ✅ complete | V2 wins: 4 features, median binarization, +1.7pp mean |

---

## Available competitor numbers (Phase 12 — same model/dataset)

| Domain | Model | Competitor | AUROC | Source |
|--------|-------|------------|-------|--------|
| GSM8K | Llama-3.1-8B | SC K=10 | 78.5% [72.0,84.5] | Computed by us |
| GSM8K | Llama-3.1-8B | SE NLI K=10 | 77.4% [70.9,83.5] | Computed by us |
| MATH-500 | Qwen2.5-Math-7B | SC K=10 | 87.2% [72.1,98.4] | Computed by us |
| MATH-500 | Qwen2.5-Math-7B | SE NLI K=10 | 87.7% [79.7,93.9] | Computed by us |
| GPQA | Qwen2.5-7B | SE NLI K=10 | 70.6% [43.6,93.3] | Computed by us |
| GPQA | Qwen2.5-7B | VC K=1 | 67.9% [49.5,83.3] | Computed by us |
| GPQA | Qwen2.5-7B | SC K=10 | 33.6% [11.0,58.2] | Computed by us — fails on GPQA |
| RAG HotpotQA | Qwen2.5-7B | SelfCheckGPT K=5 | 51.4% [41.5,62.9] | Computed by us |
| RAG NQ | Qwen2.5-7B | SelfCheckGPT K=5 | 57.1% [42.9,70.5] | Computed by us |
| RAG 2Wiki | Qwen2.5-7B | SelfCheckGPT K=5 | 55.3% [35.7,78.3] | Computed by us |
| RAG NarrativeQA | Qwen2.5-7B | SelfCheckGPT K=5 | 52.4% [41.7,65.4] | Computed by us |

---

## Key decisions made this session

1. **No old supervised numbers anywhere** — Step 100 results are historical only.
2. **Correct L-SML pipeline** = `binarize_classifiers(FEATURE_SIGNS)` → `lsml_fuse` (not `sml_unsupervised`).
3. **HTML comparison table**: per domain, per model, same-task/same-model/same-dataset only.
4. **L-SML is from Jaffé, Fetaya, Nadler 2016** (continuation of Parisi 2014). We cite the 2016 paper. The 2014 SML is a subroutine inside L-SML.
5. Pre-orienting features with FEATURE_SIGNS before binarization is valid within the paper's framework and doesn't violate unsupervised property (signs derived cross-dataset, no test-time labels).

---

## Deferred

- Phase 10 RAG re-run with variant=4 prompt — still pending
- Feature direction constants baked into `feature_utils.py` — nice cleanup, not urgent
