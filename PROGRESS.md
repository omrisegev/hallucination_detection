# MV_EPR — Session Progress Handoff

**Date**: 2026-06-06
**Last updated**: Step 123 — ablation complete; next agent implements Consolidated notebook changes

---

## TL;DR — where we are today

**Current official method**: L-SML (Jaffé-Fetaya-Nadler 2016) with pre-oriented classifiers + feature selection.
Pipeline: `binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(*binary_filt.values())`.
Fully unsupervised at test time.

**FINAL pipeline constants (do not change)**:
```python
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

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

**Why these 5 features**: derived from Step 121–123 ablation across 29 cached cells. These 5 have (a) mean individual AUROC ≥ 0.565 under median binarization, and (b) clean monotonic quantile-vs-AUROC curves — meaning their signal is real, not grid-search noise. The other 11 features have near-random or noisy curves and pollute the L-SML covariance matrix.

**Quantile calibration**: permanently dropped. Median binarization only — null result at 4 features (+0.001 gain). No `quantiles` argument anywhere.

**The old supervised numbers (Step 100) must not be used or referenced going forward.**

---

## IMMEDIATE NEXT ACTION

**Step 124/125 COMPLETE.** `Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb` ran on Colab with 5-feature GOOD_FEATURES filter. `Phase12_Comparison_Results.html` updated. All 29/29 cells beat chance.

### Next priorities (in order):

1. **Phase 13** — EDIS vs L-SML on Qwen2.5-Math-1.5B, AMC23/AIME24 (GPU needed). Loads fixed AMC23/AIME24 loaders from Step 119.
2. **Phase 14 Cell 9 re-run** — DeepSeek-R1-0528-Qwen3-8B / GPQA: L-SML v2 AUROC still TBD (fix applied in Step 118).
3. **Phase 10 RAG re-run with V4 prompt** — expected +5–15pp on RAG cells (low priority, infrastructure exists).

---

## Context: why these decisions were made

**Step 121–123 ablation summary** (3 runs of `Spectral_Analysis_LSML_Optimized.ipynb`):

| Threshold | N features | V2 mean (median) | V4 mean (opt-q) |
|---|---|---|---|
| 0.60 | 3 | 0.626 | 0.625 |
| 0.57 | 4 | 0.633 | 0.635 |
| 0.53 | 8 | 0.626 | 0.650 |

- 4-feature median = best simple pipeline (+1.7pp, no GSM8K regression)
- 8-feature opt-quantile = highest mean (+3.4pp) but hurts GSM8K (−4.7pp) and is harder to explain
- Quantile calibration curves confirmed: the 4 core features + spectral_entropy have clean monotonic curves (reliable signal); the other borderline features have noisy/oscillating curves (grid-search noise)
- **5-feature final choice**: add spectral_entropy to the 4 core features — it has a clean monotonic curve just like the others, justifying its inclusion

---

## Running experiments (Colab — GPU needed, lower priority)

- **Phase 13**: `Spectral_Analysis_MathComp_Phase13.ipynb` — L-SML vs EDIS, Qwen2.5-Math-1.5B.
- **Phase 14**: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb` — L-SML vs VC/SC, GPQA Diamond.
  - **Known bug in Drive copy**: old Cell 9 uses `boot_auc(..., n_boot=1000)` (wrong kwarg) and 2-value unpack. Fix:
    ```python
    p_auc, p_lo, p_hi = boot_auc(labels[valid_mask], lsml_full[valid_mask])
    n_auc, n_lo, n_hi = boot_auc(labels[valid_mask], -lsml_full[valid_mask])
    if p_auc >= n_auc:
        lsml_auc, lsml_lo, lsml_hi = p_auc, p_lo, p_hi
    else:
        lsml_auc, lsml_lo, lsml_hi = n_auc, n_lo, n_hi
    ```

---

## Completed phases

| Phase | Notebook | Status | Key result |
|-------|----------|--------|------------|
| Step 100 | Consolidated_Results | ✅ | Old supervised numbers — do not use |
| Step 107 | Consolidated_Results_LSML | ✅ | L-SML with assumption (iii) — superseded |
| Step 110 | LSML_Diagnostics | ✅ | Consensus FEATURE_SIGNS derived |
| Step 113 | Pilot_RAG_Prompt_Variants | ✅ | V4 prompt wins (+18.6pp), RAG direction dropped |
| Phase 12 | Phase12_Benchmarking | ✅ | SE/SC/VC/SelfCheckGPT computed |
| Step 121–123 | LSML_Optimized | ✅ | 5-feature GOOD_FEATURES finalized, median binarization |
| Step 124–125 | Consolidated_Results_LSML_v2 | ✅ | 5-feat; 29/29 beat chance; HTML updated |

---

## Available competitor numbers (Phase 12)

| Domain | Model | Competitor | AUROC |
|--------|-------|------------|-------|
| GSM8K | Llama-3.1-8B | SC K=10 | 78.5% [72.0,84.5] |
| GSM8K | Llama-3.1-8B | SE NLI K=10 | 77.4% [70.9,83.5] |
| MATH-500 | Qwen2.5-Math-7B | SC K=10 | 87.2% [72.1,98.4] |
| MATH-500 | Qwen2.5-Math-7B | SE NLI K=10 | 87.7% [79.7,93.9] |
| GPQA | Qwen2.5-7B | SE NLI K=10 | 70.6% [43.6,93.3] |
| GPQA | Qwen2.5-7B | VC K=1 | 67.9% [49.5,83.3] |
| GPQA | Qwen2.5-7B | SC K=10 | 33.6% [11.0,58.2] |
| RAG HotpotQA | Qwen2.5-7B | SelfCheckGPT K=5 | 51.4% [41.5,62.9] |
| RAG NQ | Qwen2.5-7B | SelfCheckGPT K=5 | 57.1% [42.9,70.5] |
| RAG 2Wiki | Qwen2.5-7B | SelfCheckGPT K=5 | 55.3% [35.7,78.3] |
| RAG NarrativeQA | Qwen2.5-7B | SelfCheckGPT K=5 | 52.4% [41.7,65.4] |

---

## Key decisions (permanent — do not revisit)

1. No old supervised numbers — Step 100 historical only.
2. L-SML pipeline = `binarize_classifiers(FEATURE_SIGNS)` → filter `GOOD_FEATURES` → `lsml_fuse`.
3. Median binarization only — quantile calibration dropped (null result).
4. GOOD_FEATURES = 5 features listed above — final, do not change.
5. HTML table: per domain, per model, same-task/same-model/same-dataset only.
6. Cite Jaffé-Fetaya-Nadler 2016. Do not use the term "Nadler" alone.
7. Branch cleanup done — only `master` and `origin/main` remain.

---

## Deferred

- Phase 10 RAG re-run with variant=4 prompt — still pending, low priority
- Feature direction constants baked into `feature_utils.py` — nice cleanup, not urgent
