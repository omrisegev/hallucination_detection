# MV_EPR — Session Progress Handoff

**Date**: 2026-05-14
**Last updated**: Step 89 complete — pe_min dropped, Phase 11a pre-run fixes applied, ready to push and run

---

## What this project is

Thesis on hallucination detection in LLMs. The core method: compute spectral features of the per-token entropy trajectory H(n) from a single gray-box forward pass, then fuse them with Nadler spectral fusion (covariance-weighted leading eigenvector). No labels needed at test time.

**Key package**: `spectral_utils` — all feature extraction, model loading, fusion, and data loaders live here. Always clone from `https://github.com/omrisegev/hallucination_detection.git` (branch: `master`).

---

## Best results so far

| Setup | AUROC | Notes |
|-------|-------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **90.0%** | spectral Nadler fusion |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
| GPQA / Qwen2.5-72B-AWQ / T=1.0 | **69.0%** | Phase 8 — +3.6 pp over 7B |
| Phase 10 RAG / qwen7b / 2wikimultihopqa | **80.5%** | best RAG cell |
| Phase 10 RAG / qwen7b / hotpotqa | 79.5% | |
| Phase 10 RAG / qwen72b / hotpotqa | 79.4% | |

---

## Completed phases

- **Phase 8** ✅: GPQA Diamond / Qwen2.5-72B-AWQ — 69.0% AUC (Step 80)
- **Phase 9** ✅: Factual QA (TriviaQA + WebQ) CoT — spectral features don't transfer, clean negative result (Step 82)
- **Phase 10 pilot** ✅: L-CiteEval HotpotQA / Falcon-3-10B — INVALID pre-conditions, but strong signal: Nadler 76.0%, EPR 69.9% (Step 84)
- **Phase 10 Main RAG** ✅: All 16 cells complete — qwen7b + mistral24b + qwen72b + **llama8b** (switched from 70B at Step 87); full 4×4 AUC heatmap available
- **Meta-Analysis** ✅: 7,001 samples across 5 domains; cross-domain feature ranking complete (Step 89); pe_min dropped, cusum_max confirmed as best Phase C feature

---

## Phase 10 Main RAG — final status

**Notebook**: `Spectral_Analysis_Phase10_Main_RAG.ipynb`

All 16 inference cells (4 models × 4 datasets) complete. Llama-3.1-8B replaced Llama-3.3-70B for stability.

### Best Nadler results (16/16 cells)

```
[qwen7b    /hotpotqa            ] AUC=79.5%  spectral_entropy + stft_max_high_power + rpdi
[qwen7b    /natural_questions   ] AUC=75.3%  trace_length + hl_ratio + dominant_freq
[qwen7b    /2wikimultihopqa     ] AUC=80.5%  spectral_entropy + low_band_power + dominant_freq + sw_var_peak_adaptive
[qwen7b    /narrativeqa         ] AUC=70.0%  spectral_centroid + sw_var_peak_adaptive
[mistral24b/hotpotqa            ] AUC=67.3%  spectral_centroid + rpdi
[mistral24b/natural_questions   ] AUC=74.0%  high_band_power + rpdi + sw_var_peak_adaptive
[mistral24b/2wikimultihopqa     ] AUC=74.2%  epr + spectral_centroid + stft_spectral_entropy + rpdi
[mistral24b/narrativeqa         ] AUC=66.1%  epr + spectral_entropy
[qwen72b   /hotpotqa            ] AUC=79.4%  low_band_power + stft_max_high_power + rpdi
[qwen72b   /natural_questions   ] AUC=71.8%  high_band_power + dominant_freq + stft_spectral_entropy + sw_var_peak
[qwen72b   /2wikimultihopqa     ] AUC=73.4%  epr + high_band_power + stft_spectral_entropy + rpdi
[qwen72b   /narrativeqa         ] AUC=72.2%  hl_ratio + stft_max_high_power + rpdi + sw_var_peak
[llama8b   /hotpotqa            ] — inference complete, analysis pending
[llama8b   /natural_questions   ] — inference complete, analysis pending
[llama8b   /2wikimultihopqa     ] — inference complete, analysis pending
[llama8b   /narrativeqa         ] — inference complete, analysis pending
```

Median (12 analyzed cells) ≈ 74%; 7/12 cells ≥ 70%.

---

## Meta-Analysis — final status

**Notebook**: `Spectral_Analysis_Meta_Analysis.ipynb`
**Samples**: 7,001 across Math-500, GSM8K, GPQA Diamond, Factual QA, Phase 10 RAG

### Cross-domain feature ranking (avg rank across 5 domains)

| Rank | Feature | Avg Rank |
|------|---------|----------|
| 1 | cusum_max | 3.0 |
| 1 | sw_var_peak | 3.0 |
| 3 | epr | 5.4 |
| 4 | spectral_entropy | 5.6 |
| 5 | rpdi | 6.2 |
| 8 | pe_mean | 8.6 |
| 15 | hurst_exponent | 10.0 |
| **17** | **pe_min** | **17.0 — DROPPED** |

**Decision**: `pe_min` removed from `FEAT_NAMES` → 16 features total.

---

## Current experiment: Phase 11a — Agentic

**Notebook**: `Spectral_Analysis_Phase11_Agentic_11a.ipynb`
**Status**: Ready to run on Colab A100 — code verified, pre-run fixes applied
**Goal**: Spectral Nadler vs AUQ verbalized confidence on 3-step ReAct multi-hop QA

### Design

| | Qwen2.5-7B | DeepSeek-R1-Distill-Qwen-7B |
|-|------------|----------------------------|
| hotpotqa | N=200 | N=200 |
| 2wikimultihopqa | N=200 | N=200 |

- MAX_STEPS=3, T=1.0, MAX_NEW_PER_STEP=256
- Aggregations: Φ_min, Φ_avg, Φ_last
- Baseline: AUQ verbalized confidence (Zhang et al. 2026, SOTA Φ_min=0.791 on ALFWorld)
- Spectral features: 16 features (pe_min excluded) + sw_var_peak_adaptive override + branching_entropy

### Pre-run fixes applied (Step 89)

1. `spectral_utils/feature_utils.py`: `pe_min` removed from `FEAT_NAMES` (16 features)
2. Phase 11a Cell 2: `sw_var_peak_adaptive` added to imports
3. Phase 11a Cell 11: `f['sw_var_peak'] = sw_var_peak_adaptive(ents)` override (adaptive window for 50-150 token steps)
4. Phase 11a Cell 15: Nadler key filter excludes both `'trace_length'` and `'pe_min'`

### Checkpoints

Raw trajectories: `/content/drive/MyDrive/hallucination_detection/cache/phase11_agentic_v2/raw/`
Features: `.../features/`
Results: `.../results_a/`

---

## Immediate next actions

1. **Commit and push** (3 commits):
   - Commit 1: `spectral_utils/feature_utils.py` + `HISTORY.md` + `PROGRESS.md` — "Step 89: Meta-Analysis results + drop pe_min from FEAT_NAMES"
   - Commit 2: `Spectral_Analysis_Phase11_Agentic_11a.ipynb` — "Phase 11a: sw_var_peak_adaptive override + pe_min filter"
   - Commit 3: `Spectral_Analysis_Phase10_Main_RAG.ipynb` + `Spectral_Analysis_Meta_Analysis.ipynb` — "Commit Phase 10 RAG (Llama-8B complete) and Meta-Analysis with outputs"

2. **Run Phase 11a on Colab A100**:
   - Cell 1–6: setup + data + spot check
   - Cell 9: inference driver (~4–6 hours for 4 cells × 200 trajectories × 3 steps)
   - Cells 11–21: feature extraction, analysis, Nadler search, AUQ baseline, fusion, headline table

3. **After Phase 11a results**:
   - Update `Research_Directions.md` Direction 4 (Agentic) with headline numbers
   - Prepare advisor meeting materials: Phase 10 RAG heatmap + Phase 11a headline table

---

## Key rules / gotchas

- **gptqmodel on Python 3.12**: stub `pcre` with stdlib `re` + install `device-smi tokenicer defuser` with `--no-deps`, `logbar ninja` plainly, then `pip install --no-deps gptqmodel`.
- **70B BNB models**: OOM after any freed smaller model; use a fresh runtime with `expandable_segments:True`.
- **HF cache on Drive**: NEVER rely on standard `HF_HOME` cache — snapshot symlinks break. Use `ensure_flat_dir()` flat-dir approach.
- **Analysis result persistence**: Every analysis cell > 30s MUST persist to `.pkl` with the three-branch reload pattern.
- **AWQ models**: `load_model(model_id, quantize_4bit=False)` — auto-detects AWQ. Requires `autoawq` AND `gptqmodel`.
- **L-CiteEval dataset sizes**: hotpotqa=240, NQ=160, 2wiki=240, narrativeqa=240.
- **FEAT_NAMES**: 16 features (pe_min removed). `compute_permutation_entropy()` still returns pe_min+pe_mean for compatibility.
- **sw_var_peak_adaptive**: NOT called by `extract_all_features()`. Must be applied as post-extraction override in Phase 11a Cell 11 (and any future short-trace experiment).
