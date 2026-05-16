# MV_EPR — Session Progress Handoff

**Date**: 2026-05-16
**Last updated**: Step 90 complete — Phase 11a extended (4 models), Phase 11b pilot notebooks built, all pushed

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
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals 72.0% |
| GPQA / Qwen2.5-72B-AWQ / T=1.0 | **69.0%** | Phase 8 — +3.6 pp over 7B |
| Phase 10 RAG / qwen7b / 2wikimultihopqa | **80.5%** | best RAG cell |
| Phase 10 RAG / qwen7b / hotpotqa | 79.5% | |
| Phase 10 RAG / qwen72b / hotpotqa | 79.4% | |
| Phase 11a (mid-run) / deepseek / 2wiki / Φ_min | **85.0%** | beats AUQ SOTA 0.791 — not yet official |

---

## Completed phases

- **Phase 8** ✅: GPQA Diamond / Qwen2.5-72B-AWQ — 69.0% AUC (Step 80)
- **Phase 9** ✅: Factual QA (TriviaQA + WebQ) CoT — spectral features don't transfer, clean negative result (Step 82)
- **Phase 10 Main RAG** ✅: All 16 cells complete — qwen7b + mistral24b + qwen72b + llama8b; full 4×4 AUC heatmap
- **Meta-Analysis** ✅: 7,001 samples across 5 domains; cross-domain feature ranking complete (Step 89); pe_min dropped, cusum_max #1

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

## Current experiments: Phase 11

### Phase 11a — Agentic ReAct on multi-hop QA

**Notebook**: `Spectral_Analysis_Phase11_Agentic_11a.ipynb`
**Status**: Inference partially done (qwen25_7b + deepseek_r1_7b ran in prior session); mistral24b + qwen72b not yet run
**Goal**: Spectral Nadler vs AUQ verbalized confidence on 3-step ReAct multi-hop QA

#### Design

| | Qwen2.5-7B | DeepSeek-R1-Distill-Qwen-7B | Mistral-Small-24B | Qwen2.5-72B-AWQ |
|-|------------|------------------------------|-------------------|-----------------|
| hotpotqa | N=200 | N=200 | N=200 | N=200 |
| 2wikimultihopqa | N=200 | N=200 | N=200 | N=200 |

- MAX_STEPS=3, T=1.0, MAX_NEW_PER_STEP=256
- Aggregations: Φ_min, Φ_avg, Φ_last
- Baseline: AUQ verbalized confidence (Zhang et al. 2026, SOTA Φ_min=0.791 on ALFWorld)
- Spectral features: 16 features (pe_min excluded) + sw_var_peak_adaptive override + branching_entropy

#### Mid-run signal (not yet official)

| Cell | Nadler Φ_min | Notes |
|------|-------------|-------|
| deepseek / 2wiki | **85.0%** | beats AUQ SOTA 0.791 |
| epr_last (deepseek/hotpotqa) | 83.2% | strong individual feature |
| hurst_last (deepseek/hotpotqa) | 82.8% | |
| pe_mean_last | 80.3% | |

#### Run order

1. **Normal runtime** — `ONLY_MODEL_KEYS = ['qwen25_7b', 'deepseek_r1_7b', 'mistral24b']`
   - qwen25_7b and deepseek_r1_7b will reload from checkpoint; only mistral24b runs fresh
2. **Fresh runtime** — `ONLY_MODEL_KEYS = ['qwen72b']` + run gptqmodel stub cell first
3. **Analysis** (any runtime with Drive access) — Cells 12–22: feature extraction, AUC table, Nadler, AUQ baseline, headline table, plots

#### Checkpoints

Raw trajectories: `/content/drive/MyDrive/hallucination_detection/cache/phase11_agentic_v2/raw/`
Features: `.../features/`
Results: `.../results_a/`

---

### Phase 11b — Extension to new agent modalities (PILOTS)

**Goal**: Test whether spectral entropy features generalize beyond retrieval-based agents. Two pilots before committing to full runs.

#### Pilot A: HumanEval (code execution)

**Notebook**: `Pilot_Phase11b_HumanEval.ipynb`
**Status**: Not yet run
**Design**: N=20, qwen25_7b, 3 attempts per problem, label = any_passed (unit test pass/fail)
**SOTA target**: AUROC 0.82–0.84 (DSDE execution-based disagreement, 2026)

GO/NO-GO gates:
- G0: ≥5 problems solved (both classes present)
- G1: entropy traces non-degenerate (std > 0 on ≥90%)
- G2: feature extraction coverage ≥15/20
- G3: subprocess execution stable (no Colab crash)

#### Pilot B: ALFWorld (embodied navigation)

**Notebook**: `Pilot_Phase11b_ALFWorld.ipynb`
**Status**: Not yet run
**Design**: N=5 tasks, pick_and_place type, MAX_STEPS=20, qwen25_7b, label = task_success
**SOTA target**: AUROC 0.791 (AUQ Φ_min, Zhang et al. 2026) — same paper we compare against in Phase 11a

GO/NO-GO gates (G0+G1 required; G2–G4 informative):
- G0: alfworld imports without corrupting numpy/pyarrow
- G1: env.step returns valid observation
- G2: ≥1 task solved
- G3: entropy traces non-degenerate (≥80% of steps)
- G4: feature extraction coverage (≥70% of steps)

**Note**: Pilots 4+5 are independent and can run in parallel in separate Colab tabs.

---

## Key rules / gotchas

- **gptqmodel on Python 3.12**: stub `pcre` with stdlib `re` + install `device-smi tokenicer defuser` with `--no-deps`, `logbar` plainly, then `pip install --no-deps gptqmodel`. Stub cell is already in Phase 11a notebook.
- **70B BNB models**: OOM after any freed smaller model; use a fresh runtime with `expandable_segments:True`.
- **HF cache on Drive**: NEVER rely on standard `HF_HOME` cache — snapshot symlinks break. Use `ensure_flat_dir()` flat-dir approach.
- **Analysis result persistence**: Every analysis cell > 30s MUST persist to `.pkl` with the three-branch reload pattern.
- **AWQ models**: `load_model(model_id, quantize_4bit=False)` — auto-detects AWQ. Requires `autoawq` AND `gptqmodel`.
- **L-CiteEval dataset sizes**: hotpotqa=240, NQ=160, 2wiki=240, narrativeqa=240.
- **FEAT_NAMES**: 16 features (pe_min removed). `compute_permutation_entropy()` still returns pe_min+pe_mean for compatibility.
- **sw_var_peak_adaptive**: NOT called by `extract_all_features()`. Must be applied as post-extraction override in short-trace experiments (Phase 11a Cell 12, pilot notebooks Cell 8/9).
- **Drive mount**: MUST be in Cell 1 or Cell 3 (before any path that uses `/content/drive/`). Missing mount = ephemeral local path, all data lost on disconnect. This bug bit us in Phase 11a — now fixed.
- **alfworld_utils.py**: Not imported by `__init__.py`. Import directly: `from spectral_utils.alfworld_utils import ...`

---

## Immediate next actions

1. **Run Phase 11a inference** — mistral24b (normal runtime) + qwen72b (fresh runtime with stub cell)
2. **Run Phase 11a analysis** — Cells 12–22 after all 8 raw pkl files exist
3. **Run pilots in parallel**:
   - `Pilot_Phase11b_HumanEval.ipynb` — any runtime
   - `Pilot_Phase11b_ALFWorld.ipynb` — any runtime
4. **After pilots**: if GO → build full Phase 11b notebooks (same structure, N=164 HumanEval / ~100 ALFWorld tasks, multi-model)
5. **Update Research_Directions.md** Direction 4 (Agentic) with Phase 11a headline numbers once analysis is complete
