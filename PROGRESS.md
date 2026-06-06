# MV_EPR — Session Progress Handoff

**Date**: 2026-06-06
**Last updated**: Step 122 — LSML_Optimized rerun in progress (threshold 0.53 → 8 features)

---

## TL;DR — where we are today

**Current official method**: L-SML (Jaffé-Fetaya-Nadler 2016) with **pre-oriented classifiers + feature selection**.
Pipeline: `binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(*binary_filt.values())`.
Fully unsupervised at test time. FEATURE_SIGNS from Step 110; GOOD_FEATURES from Step 122 ablation.

**Step 122 findings (confirmed)**:
- Quantile calibration is a **null result** (V2→V4 = +0.001). Median binarization is final — do not add quantile calibration to any notebook.
- Feature selection gives **+1.7pp mean** (4-feature run). Domain-dependent: QA/RAG benefit, GPQA regresses ~−3.8pp (explainable: 4 correlated features violate L-SML conditional independence more than 16 diverse ones).
- Currently re-running with **threshold 0.53 → 8 features** to test whether more features reduce GPQA regression while preserving QA/RAG gains.

**Expected GOOD_FEATURES at threshold 0.53** (pending confirmation from run):
```python
GOOD_FEATURES = ['epr', 'sw_var_peak', 'cusum_max', 'low_band_power',
                 'spectral_entropy', 'rpdi', 'pe_mean', 'stft_max_high_power']
```
(8 features with mean individual AUROC ≥ 0.53 under median binarization)

**The old supervised numbers (Step 100) must not be used or referenced going forward.**

---

## Immediate next actions

### Action 1 — Get 8-feature ablation results ← IN PROGRESS ON COLAB

`Spectral_Analysis_LSML_Optimized.ipynb` is running with `MIN_IND_AUC_THRESHOLD = 0.53` and `FORCE_VARIANTS = True`.

**When results arrive, compare vs 4-feature run:**

| Variant | 4-feat mean | 8-feat mean | Goal |
|---|---|---|---|
| V1 all-16 median | 0.616 | (same, reference) | — |
| V2 filtered median | 0.633 | ? | higher AND less GPQA loss |
| V3 all-16 optimized | 0.618 | (same) | — |
| V4 filtered optimized | 0.635 | ? | should ≈ V2 (null result confirmed) |

**Decision gate**: if 8-feature V2 mean ≥ 0.633 AND GPQA regression shrinks → adopt 8 features.
If 8-feature V2 < 4-feature V2 (weak features dilute signal) → keep 4 features.

### Action 2 — Update + run Consolidated Results notebook

**Task**: `Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb` — CPU-only, ~15–30 min.

After picking the winning GOOD_FEATURES from Action 1, add to Cell 2 config:
```python
GOOD_FEATURES = [... winning list ...]
```
In the fusion cell, change to:
```python
binary = binarize_classifiers(fd, FEATURE_SIGNS)
binary_filt = {fn: binary[fn] for fn in GOOD_FEATURES if fn in binary}
fused, meta = lsml_fuse(*binary_filt.values())
```
**Do NOT add quantile calibration** — median binarization is final (Step 122 null result).

After running: rebuild `Phase12_Comparison_Results.html` with new per-domain AUROCs.

### Action 3 — Phase 13 / Phase 14 (GPU, lower priority)

- **Phase 13**: `Spectral_Analysis_MathComp_Phase13.ipynb` — L-SML vs EDIS on GSM8K+MATH500+AMC23+AIME24, Qwen2.5-Math-1.5B, T=0.2/0.6/1.0, K=8.
- **Phase 14**: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb` — L-SML vs VC/SC on GPQA Diamond, DeepSeek-R1-0528-Qwen3-8B. **Known bug fixed**: old Drive copy had `boot_auc(..., n_boot=1000)` (wrong kwarg) and 2-value unpack. Repo version is correct. Use the repo copy or manually fix Cell 9 on Colab:
  ```python
  p_auc, p_lo, p_hi = boot_auc(labels[valid_mask], lsml_full[valid_mask])
  n_auc, n_lo, n_hi = boot_auc(labels[valid_mask], -lsml_full[valid_mask])
  if p_auc >= n_auc:
      lsml_auc, lsml_lo, lsml_hi = p_auc, p_lo, p_hi
  else:
      lsml_auc, lsml_lo, lsml_hi = n_auc, n_lo, n_hi
  ```

---

## FEATURE_SIGNS (Step 110 consensus — do not change)

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

## Completed phases

| Phase | Notebook | Status | Key result |
|-------|----------|--------|------------|
| Step 100 | Consolidated_Results | ✅ | Old supervised numbers — do not use |
| Step 107 | Consolidated_Results_LSML | ✅ | L-SML with assumption (iii) — superseded by v2 |
| Step 110 | LSML_Diagnostics | ✅ | Consensus FEATURE_SIGNS derived |
| Step 113 | Pilot_RAG_Prompt_Variants | ✅ | V4 prompt wins (+18.6pp) but RAG direction dropped |
| Phase 12 | Phase12_Benchmarking | ✅ | SE/SC/VC/SelfCheckGPT computed — use for competitor rows |
| Step 121–122 | LSML_Optimized | ✅ ablation done | 4-feat V2: +1.7pp mean; 8-feat rerun in progress |

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

## Key decisions (permanent)

1. **No old supervised numbers anywhere** — Step 100 results are historical only.
2. **Correct L-SML pipeline** = `binarize_classifiers(FEATURE_SIGNS)` → filter `GOOD_FEATURES` → `lsml_fuse` (not `sml_unsupervised`, not `best_nadler_on`).
3. **Quantile calibration is dropped** — median binarization only (Step 122 null result: +0.001).
4. **HTML comparison table**: per domain, per model, same-task/same-model/same-dataset only.
5. **L-SML is from Jaffé, Fetaya, Nadler 2016** (continuation of Parisi 2014). Cite the 2016 paper.
6. Pre-orienting features with FEATURE_SIGNS is valid and unsupervised at test time (signs derived cross-dataset, no test-time labels).
7. **Branch cleanup done**: only `master` and `origin/main` remain. `gemini/phase10-pilot` archived as tag `archive/gemini-phase10-pilot`.

---

## Deferred

- Phase 10 RAG re-run with variant=4 prompt — still pending
- Feature direction constants baked into `feature_utils.py` — nice cleanup, not urgent
