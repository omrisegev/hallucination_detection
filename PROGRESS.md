# MV_EPR — Session Progress Handoff

**Date**: 2026-06-05
**Last updated**: Step 120 — oriented L-SML rerun planned; handoff to next session

---

## TL;DR — where we are today

**Current official method**: L-SML (Jaffé-Fetaya-Nadler 2016) with **pre-oriented classifiers**.
Pipeline: `binarize_classifiers(feats_dict, FEATURE_SIGNS)` → `lsml_fuse(*binary.values())`.
Fully unsupervised at test time. FEATURE_SIGNS derived from Step 110 cross-dataset consensus.

**The old supervised numbers (Step 100) must not be used or referenced going forward.**

---

## Immediate next action — BUILD THIS NOTEBOOK

**Task**: `Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb` — CPU-only Colab, ~15–30 min.

**What it does**:
1. Load cached features from Drive `consolidated_results/`:
   - `math500_res.pkl`, `gsm8k_res.pkl`, `gpqa_res.pkl`, `rag_feats_all.pkl`, `qa_res.pkl`
2. For each (domain, model, dataset) cell:
   - Apply `binarize_classifiers(feats_dict, FEATURE_SIGNS)` — orients then binarizes
   - Run `lsml_fuse(*binary.values())` — Algorithm 2 from Jaffé et al. 2016
   - Compute AUROC with bootstrap CI using `boot_auc(labels, scores)`
3. Save results to Drive `consolidated_results/lsml_v2_results_all.pkl` + `lsml_v2_summary.csv`

**FEATURE_SIGNS** (Step 110 consensus — baked into Phase 13 Cell 2 config):
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
Convention: +1 = higher value → more likely correct; -1 = higher value → hallucination.

**After the notebook runs**: rebuild `Phase12_Comparison_Results.html` with:
- Our method rows: v2 L-SML numbers only (no Step 100 supervised numbers)
- Competitor rows: same-model, same-dataset, same-task only
  - If from paper: cite it. If we computed it in Phase 12: mark as "computed by us".
- Structure: per domain → per model table

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
