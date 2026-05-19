# MV_EPR — Session Progress Handoff

**Date**: 2026-05-18
**Last updated**: Step 93 complete — Phase 12 benchmarking environment set up; `baselines.py` extended with official SE/SC/SelfCheck/VC; `Spectral_Analysis_Phase12_Benchmarking.ipynb` ready to run on Colab

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
| Phase 10 RAG / llama8b / hotpotqa | **87.7%** | new overall best RAG cell — beats LOS-Net (72.92%) by +14.8 pp |
| Phase 10 RAG / qwen7b / 2wikimultihopqa | 80.5% | |
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
[llama8b   /hotpotqa            ] AUC=87.7%  ← NEW BEST overall RAG cell
[llama8b   /natural_questions   ] AUC=70.3%
[llama8b   /2wikimultihopqa     ] AUC=64.5%
[llama8b   /narrativeqa         ] AUC=63.2%
```

All 16 cells complete. Median across 16 cells ≈ 72.8%; 12/16 cells ≥ 70%.

**llama8b pattern**: very strong on HotpotQA (87.7%) but weak on 2wiki/NarrativeQA (64–63%). This is the inverse of qwen7b (strong on 2wiki). Dataset–model interaction: HotpotQA factoid retrieval suits Llama-8B's generation style; 2Wiki multi-hop chains favour Qwen-7B's reasoning structure.

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

## Current experiments: Phase 12 — Benchmarking

### Phase 12 — Systematic SOTA Comparison (NEW, Step 93)

**Notebook**: `Spectral_Analysis_Phase12_Benchmarking.ipynb`
**Status**: Ready to run — all code implemented and smoke-tested locally

**Competitors implemented in `spectral_utils/baselines.py`**:

| Method | Access | Compute | Function |
|--------|--------|---------|---------|
| Official Semantic Entropy | Gray-box + sampling | K=10 + NLI | `official_semantic_entropy()` |
| Self-Consistency | Black-box | K=10 | `self_consistency_score()` |
| SelfCheckGPT (NLI) | Black-box | K=5 + NLI | `selfcheck_nli_score()` |
| Verbalized Confidence | Black-box | 1-pass | `parse_verbalized_confidence()` |

**Reference numbers** (paper, not re-run):

| Method | Paper result | Task |
|--------|-------------|------|
| LapEigvals unsup (Phase 7) | 72.0% GSM8K | already re-run in our Phase 7 |
| LapEigvals supervised | 87.2% GSM8K | different supervision level |
| LOS-Net (AAAI 2026) | 72.92% | std HotpotQA (different task) |

**Run order**:
1. **Normal runtime**: Load NLI model, run Math + GPQA sections
2. **Second session or fresh runtime**: Run RAG section (Llama-8B)
3. **Analysis**: Section 5 — master table + save `Research_Phase12_Comparison_Results.md`

---

## Immediate next actions

1. **Run Phase 12 benchmarking notebook** — `Spectral_Analysis_Phase12_Benchmarking.ipynb`
   - Section 2 (Math): loads Phase 7 cache, K=10 sampling on N=200 GSM8K
   - Section 3 (GPQA): fresh Qwen-7B inference + K=10 sampling
   - Section 4 (RAG): loads Phase 10 cache, K=5 SelfCheckGPT
2. **Run Phase 11a inference** — mistral24b (normal runtime) + qwen72b (fresh runtime with stub cell)
3. **Run Phase 11a analysis** — Cells 12–22 after all 8 raw pkl files exist; saves `phase11a_detector_heatmap.png` + `phase11a_rho.png` to Drive
4. **Run pilots in parallel**:
   - `Pilot_Phase11b_HumanEval.ipynb` — any runtime
   - `Pilot_Phase11b_ALFWorld.ipynb` — any runtime
5. **After pilots**: if GO → build full Phase 11b notebooks (same structure, N=164 HumanEval / ~100 ALFWorld tasks, multi-model)
6. **Update Research_Directions.md** Direction 4 (Agentic) with Phase 11a headline numbers once analysis is complete
7. **Add savefig calls to Meta Analysis notebook** — plots exist as Colab outputs but are not persistently saved; add `fig.savefig(os.path.join(PLOT_DIR, '...'))` before each `plt.show()` and rerun

---

## Advisor meeting — May 18, 2026

Presentation ready: `Hallucination_Detection_May18.pptx` (17 slides), `Meeting_May18_Speaker_Notes.md`.
Build script: `build_presentation.py` (reproducible; pulls plots from `presentation_plots/`).

**Step 92 changes (16-point feedback addressed):**
- Slide 2: subtitle clarified; real MATH-500 correct+hallucinated traces visible
- Slide 3: replaced with `psd_comparison.png` (4-panel: MATH + GPQA); x-axis explanation added
- Slide 4: Nadler best-subset box (`trace_length + spectral_centroid + rpdi + sw_var_peak`); dataset metadata (68.7% accuracy, why 7B); `make_formula_panel()` with 4 key formulas
- Slide 5: subtitle updated to "all 16 features from 7,001-sample meta-analysis"
- Slide 7: `make_math_example()` now shows BOTH correct and hallucinating traces side by side with cusum_max + rpdi annotations
- Slide 8: LapEigvals corrected to `White-box (attn.maps)` + note about access advantage
- Slide 9: `B3_gpqa_trajectories.png` (real GPQA Phase 8 traces) added alongside bar chart
- Slide 10: WebQ direct-answer note added (0 correct samples → AUC undefined)
- Slide 12: LOS-Net framed as "different task (std HotpotQA, no citations)"; "novel setting" bullet added
- Slide 15: task summary box (input/action/label); explicit RAG vs Agentic structural comparison; Phi definitions
- Slide 16: Phi_min/avg/last definition box; AUQ white-box framing strengthened

New Drive plots downloaded: `B3_gpqa_trajectories.png`, `psd_comparison.png`, `fig1_individual_traces.png`, `fig5_avg_trajectories.png`
