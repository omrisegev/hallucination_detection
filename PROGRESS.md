# MV_EPR — Session Progress Handoff

**Date**: 2026-06-02
**Last updated**: Step 113 — RAG prompt pilot complete; V4 wins with +18.6pp fusion

---

## TL;DR — where we are today

**Current official method**: paper-aligned L-SML (Parisi-Nadler-Kluger 2014 + Jaffé-Fetaya-Nadler 2016) with **offline consensus feature orientation**. Fully unsupervised at inference time; uses prior empirical knowledge of feature semantics to satisfy Paper 2 assumption (iii).

**Headline numbers (Step 110, run on Consolidated cached features, 29 cells)**:

| Domain | L-SML + consensus | Cells |
|---|---|---|
| Math reasoning (MATH-500, GSM8K) | 65 – 91% | 5 |
| GPQA Diamond (science MCQ) | 41 – 62% | 5 |
| RAG / HotpotQA | 64 – 82% | 4 |
| RAG / 2WikiMultiHop | 41 – 52% | 4 |
| RAG / NaturalQuestions, NarrativeQA | 39 – 62% | 8 |
| Factual QA (TriviaQA, WebQ) | 43 – 78% | 3 |

(Full 29-row table: `consolidated_results/consensus_vs_paper2_summary.csv`. See HISTORY Step 110/111.)

**Active goal**: Phase 10 RAG re-run with `lciteeval_prompt(row, variant=4)` across all 4×4 cells to get final bootstrapped AUROC numbers with the winning prompt.

**RAG prompt pilot (Step 113, COMPLETE)**: V4 — "Consider whether the passages clearly answer the question, then answer" — wins with fusion AUROC 75.6% vs baseline 57.0% (+18.6pp). Interestingly V4 produces shorter traces (57 tokens) than V1/V2/V3 (120-143), confirming the gain is from entropy *shape* change (evaluative preamble peak) not trace length. `hl_ratio` recovered to 62.5% AUROC on V4 (passes G2); `dominant_freq` (58.3%) and `spectral_centroid` (59.4%) narrowly miss G2 but improve. Pilot data saved to `cache/prompt_pilot/` on Drive.

**In flight**: Phase 12 inference re-run on Colab (was blocked by Cell 11 bug; fix is on remote at `3dffa90`, user needs to refresh notebook in Colab session). Adds Mistral/GSM8K, R1-Distill+Qwen3/GPQA, SelfCheckGPT/RAG cells to the comparison table.

**Deferred**:
1. Phase 10 RAG re-run with variant=4 — replace `lciteeval_prompt(row)` with `lciteeval_prompt(row, variant=4)` in all 16 Phase 10 inference cells; re-run N=200 each for final bootstrapped numbers.
2. Feature-level orientation refactor — bake `FEATURE_CANONICAL_SIGNS` into `feature_utils.py` so Paper 2 (iii) holds naturally; eliminates the `feature_signs` argument plumbing. Numerically same as current consensus; cleaner code.
3. `boot_auc` zero-variance NaN root fix — only affects `trace_length` on cells where the model hit the token cap on every sample (gpqa/DeepSeek-R1-Distill).

---

## Phase 12 unblock — what you need to do

Your Colab is running an OLD copy of `Spectral_Analysis_Phase12_Benchmarking.ipynb`. The Cell 11 bug fix (`math_prompt(row)` instead of `math_prompt(row['problem'])`) IS on GitHub at `feature/nadler-paper-alignment` (commit `3dffa90`). To pick it up:

1. In your Colab session, **File → Open notebook → GitHub** → choose `omrisegev/hallucination_detection`, branch `feature/nadler-paper-alignment`, file `Spectral_Analysis_Phase12_Benchmarking.ipynb`. This pulls the corrected notebook.
2. Re-run Cell 1 (clone/imports), Cell 2 (config), Cell 3 (NLI), then jump straight to Cell 11. The `p4_math500_qwen7b_k10.pkl` checkpoint will preserve any prior progress.

If you'd rather keep your current notebook: just edit Cell 11 in place to change `prompt = math_prompt(row['problem'])` → `prompt = math_prompt(row)`. Same fix.

---

## ⚠ Historical: Step 107 result — L-SML AUROCs were 5–30 pp LOWER than Step 100

Step 107 ran the new paper-aligned **L-SML** pipeline (binary inputs, fully unsupervised, no subset selection, Paper 1 dependent-classifier handling) on **the same cached features** that produced the Step 100 Nadler numbers. Every (domain, model) cell dropped:

| Domain | Best L-SML | Old Nadler (Step 100) | Δ |
|---|---|---|---|
| MATH-500 / Qwen-Math-7B    | **91.2%** [86.0, 95.2] | 96.7% | −5.5pp |
| GSM8K / Llama-8B            | **70.4%** [66.9, 74.0] | 75.9% | −5.5pp |
| GPQA / Qwen-72B-AWQ         | **62.4%** [54.6, 70.4] | 67.5% | −5.0pp |
| RAG / Llama-8B / hotpotqa   | **71.1%** [59.9, 81.9] | 88.2% | −17.1pp |
| RAG / Qwen-7B / 2wikimultihopqa | 52.1% [32.7, 69.8] | 81.3% | **−29.3pp** |
| Factual QA / trivia_qa_cot  | 56.9% [49.5, 64.6] | 71.1% | −14.2pp |

(Full table in HISTORY.md Step 107.)

**Why**: the Step 100 numbers used (a) supervised label-based sign orientation, (b) exhaustive subset search picking the winner on the same N used for AUROC reporting (in-sample selection bias), (c) continuous features (violating Lemma 1), (d) M-matrix weight formula. The L-SML drops are the **honest price** of correcting all four issues to match Parisi-Nadler-Kluger 2014 + Jaffé-Fetaya-Nadler 2016.

**Both number sets remain valid measurements** — just under different methodological assumptions. Step 100 = "best supervised in-sample fit" (NOT used in thesis; supervised in-sample selection is methodologically unsound); Step 107 = "fully unsupervised paper-aligned" (intermediate; superseded by Step 110).

**Step 108 (diagnostics) traced the Step 107 collapse**: sign-agreement ≤ 5/16 cells collapsed to AUROC ≈ 1 − AUROC. Diagnosis: Paper 2 assumption (iii) "majority of binary classifiers beat random" is systematically violated by our entropy-heavy feature set, so L-SML's eigenvector-sign rule picked the wrong global sign.

**Step 110 (offline consensus orientation, current official)** resolved the collapse: each feature gets a fixed direction derived once from per-cell supervised signs across past data, then features are pre-oriented before binarization. Result: 13/29 cells gained 5-63pp; the worst regressions were 6 RAG cells with already-marginal signal. Stage 5 now matches Stage 3 (binary + supervised + SML) across the board, meaning the only remaining cost is the binarization step itself.

---

## Step 105–106 — Package now exposes paper-aligned pipeline

On branch **`feature/nadler-paper-alignment`** (commits 9cad474 → aed20c9):

- `binarize_classifiers(feats_dict, signs)` — median-threshold features into ±1 binary classifiers (Lemma 1 input contract)
- `sml_fuse(*classifiers)` — direct Lemma 1 SML: leading eigenvector of off-diagonal R; verified Pearson 0.964 vs theoretical (2α−1) on synthetic data
- `nadler_fuse` retained for backward compatibility (M-matrix variant)
- `best_nadler_on` gained `binarize=False` default; `binarize=True` switches the internal weight estimator to `sml_fuse` (Lemma 1 exact)
- **NEW** L-SML stack — `sml_fuse_signed`, `detect_dependent_groups`, `lsml_fuse`, `sml_unsupervised`, `sml_unsupervised_compare`:
  - Paper 1 Algorithm 1 group detection (score matrix `s_ij = Σ |r_ij·r_kl − r_il·r_kj|` + spectral clustering)
  - Two K-selection methods: `residual` (paper-faithful) vs `eigengap` (fast heuristic)
  - On synthetic Paper-1 model (K=3 true groups): residual K=3 ✓ AUC=0.869; eigengap K=2 AUC=0.814; naive SML K=1 AUC=0.824 → residual is best, eigengap NOT redundant

Two paper-aligned Colab notebooks ready on branch:
- `Spectral_Analysis_Consolidated_Results_LSML.ipynb` — CPU only, reuses cached features
- `Spectral_Analysis_Phase12_Benchmarking.ipynb` — edited in place; uses `sml_unsupervised` for P1b/P2b/P2c/P2d; saves to `*_lsml.pkl`

---

## What this project is

Thesis on hallucination detection in LLMs. The core method: compute spectral features of the per-token entropy trajectory H(n) from a single gray-box forward pass, then fuse them with Nadler spectral fusion (covariance-weighted leading eigenvector). No labels needed at test time.

**Key package**: `spectral_utils` — all feature extraction, model loading, fusion, and data loaders live here. Always clone from `https://github.com/omrisegev/hallucination_detection.git` (branch: `feature/meta-agentic-integration`). Note: `baselines.py` (needed for Phase 12 SC/SE/SelfCheckGPT) only exists on this branch, not on `master`.

---

## Best results — two parallel tables (post Step 107)

### Step 107 — paper-aligned L-SML (new official, fully unsupervised, no subset)

| Setup | L-SML AUROC | CI | K (residual) |
|-------|---|----|---|
| MATH-500 / Qwen-Math-7B / T=1.0 | **91.2%** | [86.0, 95.2] | 5 |
| MATH-500 / Qwen-Math-1.5B / T=1.0 | 82.1% | [76.7, 86.8] | 6 |
| MATH-500 / DeepSeek-R1-Llama-8B / T=1.0 | 78.9% | [73.5, 84.3] | 6 |
| GSM8K / Llama-3.1-8B | **70.4%** | [66.9, 74.0] | 4 |
| GPQA / Qwen-72B-AWQ / T=1.0 | **62.4%** | [54.6, 70.4] | 4 |
| GPQA / Qwen-7B / T=1.0 | 58.5% | [50.5, 66.6] | 4 |
| GPQA / Mistral-7B / T=1.0 | 56.8% | [47.1, 66.4] | 6 |
| RAG / Llama-8B / hotpotqa | **71.1%** | [59.9, 81.9] | 4 |
| RAG / Qwen-72B / hotpotqa | 70.1% | [61.0, 78.7] | 4 |
| Phase 9 Factual QA / trivia_qa_cot | 56.9% | [49.5, 64.6] | 4 |

RAG median across 16 cells dropped from 72.8% (Step 100) to ~55–58% (Step 107).

### Step 100 — old supervised continuous M-matrix Nadler (kept for comparison only)

| Setup | AUROC | CI | Notes |
|-------|-------|----|-------|
| MATH-500 / Qwen-Math-7B / T=1.0 | **96.69%** | [93.90, 98.69] | ← updated from 90.0% (full 16-feat) |
| MATH-500 / Qwen-Math-1.5B / T=1.0 | 87.97% | [83.94, 91.49] | |
| MATH-500 / DeepSeek-R1-Llama-8B / T=1.0 | 86.28% | [81.85, 90.11] | |
| GSM8K / Llama-3.1-8B | **75.92%** | [72.48, 79.39] | vs LapEigvals 72.0% |
| GPQA / Qwen-72B-AWQ / T=1.0 | **67.47%** | [59.71, 74.74] | ← updated from 69.0% |
| GPQA / Mistral-7B / T=1.0 | 65.28% | [56.72, 73.96] | |
| RAG / Llama-8B / hotpotqa | **88.15%** | [80.64, 94.37] | overall best — beats LOS-Net 72.9% by +15.3 pp |
| RAG / Qwen-7B / natural-questions | 82.81% | [70.85, 92.64] | |
| RAG / Qwen-7B / 2wikimultihopqa | 81.34% | [71.42, 89.68] | ← updated from 80.5% |
| RAG / Qwen-7B / hotpotqa | 80.15% | [66.52, 91.40] | ← updated from 79.5% |
| RAG / Qwen-72B / hotpotqa | 79.40% | [70.45, 86.84] | ← updated from 79.4% |
| Phase 11a (mid-run) / deepseek / 2wiki / Φ_min | **85.0%** | — | beats AUQ SOTA 0.791 — not yet official |

RAG summary (16 cells): 13/16 ≥ 70%, median 72.8%.

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

- **Pre-commit hook**: `.git/hooks/pre-commit` validates all staged `.ipynb` files as JSON before every commit. If a notebook is corrupt (e.g. unescaped quotes from a string-replace script), the commit is aborted.
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

### Phase 12 — Systematic SOTA Comparison (Steps 96–104 — READY TO RUN ✅)

**Notebook**: `Spectral_Analysis_Phase12_Benchmarking.ipynb`
**Status**: Step 104 — All comparison issues fixed. Pull from `feature/meta-agentic-integration` and run.

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
| EDIS (arXiv 2602.01288) | 0.804 pooled 4 math datasets | Qwen-Math-1.5B, K=8 ⚠ cross-model |
| Mean entropy (EDIS paper) | 0.673 | same model/paper ⚠ |
| LapEigvals unsup (Phase 7) | 72.0% GSM8K | already re-run in our Phase 7 |
| LapEigvals supervised | 87.2% GSM8K | different supervision level |
| LOS-Net (AAAI 2026) | 72.92% | std HotpotQA (different task) |

**All fixes applied (Steps 96–104)**:
- Step 96: 6 structural bugs (wrong branch, lciteeval kwargs, grounding label, stale-pkl guards, P4 init, Section 5)
- Step 99: dict-keyed incremental saves to all 4 SE/SelfCheckGPT scoring cells
- Step 101: `generate_full` API migration (dict return), `gpqa_prompt_and_answer` missing idx, AUROCs updated to Step-100 official numbers, EDIS comparisons added
- Step 102: `boot_auc` NaN filtering, 5 NaN display guards, JSON corruption repaired, pre-commit hook
- Step 104: Supervision column added to all tables; apples-to-apples comparison cells added:
  - **P1b**: Mistral-7B-Instruct-v0.3 GSM8K inference + pseudo-label Nadler (matches arXiv 2502.03799)
  - **Cell 8b**: Nadler on existing Qwen-7B GPQA entropies from Cell 7 (no model reload)
  - **Cell 8c**: DeepSeek-R1-Distill-Qwen-7B GPQA inference + pseudo-label Nadler (matches arXiv 2603.19118)
  - **Cell 8d**: Qwen3-8B GPQA inference + pseudo-label Nadler (matches arXiv 2603.19118)
  - `best_nadler_pseudo_label` added to `spectral_utils/fusion_utils.py` — fully unsupervised Nadler fusion via seed-feature majority-vote pseudo-labels
  - `nli_load_model` now accepts `cache_dir` param — fixes Drive FUSE OSError [Errno 5] crash blocker

**Known behavior**: `self_consistency_score()` returns `NaN` for samples where answer extraction fails (documented). `boot_auc` now silently drops NaN pairs. After running Cell 8 (GPQA), check `np.isnan(sc_p2).sum()` to verify SC coverage.

**Run order**:
1. `git -C /content/hallucination_detection pull -q` in Colab, then reload
2. **Normal runtime**: Load NLI model, run Math + GPQA sections (Cells 5–8)
3. **Second session or fresh runtime**: Run RAG section (Cell 9, Qwen-7B on 4 datasets)
4. **Section 5** (any runtime with Drive access): loads consolidated pkl + prints master tables + writes MD

---

## Consolidated Results Notebook (Step 100 — COMPLETE ✅)

**Notebook**: `Spectral_Analysis_Consolidated_Results.ipynb`
**Status**: Finished on Colab (CPU runtime). All outputs saved to Drive.

**Outputs on Drive**:
- `consolidated_results/results_all.pkl` — full dict {math500, gsm8k, gpqa, rag, qa} — read by Phase 12 Section 5
- `consolidated_results/results_summary.csv` — 29-row flat table sorted by AUROC
- `consolidated_results/all_results_dict.pkl` — same as above, flat key format
- `consolidated_results/plots/` — ~30 PNGs (heatmaps, AUC bars, trajectories, PSDs)

**Key findings**: See Step 100 in HISTORY.md for full table. Headlines:
- MATH-500/Qwen-Math-7B: **96.69%** (was 90.0% — full 16-feat gains +6.7 pp)
- RAG best: Llama-8B/hotpotqa **88.15%** (beats LOS-Net 72.9% by +15.3 pp)
- 13/16 RAG cells ≥70%

---

## Immediate next actions

1. **Phase 10 RAG re-run with variant=4** ← **HIGHEST PRIORITY**
   - Open `Spectral_Analysis_Phase10_Main_RAG.ipynb` on Colab
   - In every inference cell that calls `lciteeval_prompt(row)`, change to `lciteeval_prompt(row, variant=4)`
   - Re-run all 16 cells (4 models × 4 datasets), N=200 each
   - V4 wins pilot with +18.6pp fusion; expect Phase 10 RAG cells to recover 5–15pp vs current L-SML numbers
   - Pull branch `feature/nadler-paper-alignment` first (`git -C /content/hallucination_detection pull -q`) so `lciteeval_prompt` has the `variant` param
2. **Phase 12 (running/pending on Colab)**
   - Branch is `feature/nadler-paper-alignment`; notebook uses `sml_unsupervised` everywhere
   - Total runtime ~4–5h on A100 (Mistral-7B GSM8K ~45min, DeepSeek-R1-7B GPQA ~1.5h, Qwen3-8B GPQA ~2h, plus NLI baselines)
   - **Critical question Phase 12 answers**: does L-SML still beat SE/SC/VC baselines on the same models?
3. **After Phase 10 re-run + Phase 12**: build the advisor competitor-comparison table and write Step 114 in HISTORY.md
4. **Phase 11a inference** (queued, lower priority) — mistral24b + qwen72b
5. **Phase 11b pilots** (queued) — HumanEval + ALFWorld

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
