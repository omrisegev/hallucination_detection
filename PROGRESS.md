# MV_EPR — Session Progress Handoff

**Date**: 2026-05-07  
**Context at handoff**: ~78% used → starting fresh session

---

## What this project is

Thesis on hallucination detection in LLMs. The core method: compute spectral features of the per-token entropy trajectory H(n) from a single gray-box forward pass, then fuse them with Nadler spectral fusion (covariance-weighted leading eigenvector). No labels needed at test time.

**Key package**: `spectral_utils` — pip-installable from `https://github.com/omrisegev/hallucination_detection.git` (branch: `master`). All feature extraction, model loading, fusion, and data loaders live here.

**In Colab**: always clone with `git clone -b master https://github.com/omrisegev/hallucination_detection.git` + `sys.path.insert(0, REPO_DIR)`. Do NOT use `pip install git+...`.

---

## Current experiment status

### Phase 8 — GPQA Diamond with Qwen2.5-72B (BLOCKED → fixed, not yet run)

**Notebook**: `GPQA_Phase8_Fixed.ipynb`  
**Goal**: Run spectral hallucination detection on GPQA Diamond (198 samples) with a 72B model (~65% accuracy, much better class balance than 7B at ~30%).

**Three OOM bugs fixed (all in the notebook now)**:
1. Async shard loading peak → switched to AWQ model (pre-quantized on disk)
2. `torch_dtype` + `quantization_config` coexistence → removed `torch_dtype`
3. `device_map='auto'` dispatches to CPU before BNB → changed to `device_map={"": 0}`

**Current notebook config** (already fixed in file):
- `MODEL_ID = 'Qwen/Qwen2.5-72B-Instruct-AWQ'`
- `QUANTIZE = False` (AWQ is already quantized)
- `device_map={"": 0}` in `load_model`
- Saves to `epr_spectral_gpqa_72b/Qwen2.5-72B-Instruct-AWQ__gpqa_T1.0/`

**Status**: Notebook is fixed and committed. **Needs to be run on Colab A100 (40 GB).**  
**Expected**: ~65% accuracy, meaningful spectral AUC, compare vs prior Phase 4 best of 65.4% (Mistral-7B).

---

### Phase 9 — QA Transfer (Falcon-3-10B on TriviaQA + WebQ) (PENDING RUN)

**Notebook**: `Spectral_Analysis_Phase9_QA_Validation.ipynb`  
**Goal**: Validate fixed 4-feature subset (`sw_var_peak + trace_length + spectral_centroid + stft_max_high_power`) on factual QA domains.

**Two parts**:
- **Part 1** (direct-answer): Already ran — 83% of TriviaQA traces too short for FFT, discarded. Near-zero signal. Bug fixed (size mismatch in `extract_dataset_features`).
- **Part 2** (CoT): Added to notebook. CoT prompts generate 50–256 token traces. **Not yet run.** Expected: ~100% trace survival, meaningful spectral AUC.

**Status**: Notebook committed. **Needs to be run on Colab (Part 2 CoT inference).**

---

### Phase 10 — Agentic hallucination (PLANNED, not started)

**Concept**: GPQA Diamond questions in a ReAct agent loop (Thought → Action → Observation). Capture per-Thought-step entropy traces. Apply spectral/Nadler fusion to predict step-level hallucinations.

**Model**: Qwen2.5-72B-Instruct-AWQ (same as Phase 8).  
**Benchmark inspiration**: AgentHallu (GPT-4.1 generated, text-only — can't use directly). Will re-run our own trajectories.  
**Key paper**: "The Reasoning Trap" (ICLR 2026) — deeper reasoning amplifies tool hallucination.

**Status**: Research planned in HISTORY.md Step 78. No notebook yet.

---

## Best results so far

| Setup | AUROC | Notes |
|-------|-------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **90.0%** | spectral Nadler fusion |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
| GPQA / Mistral-7B / T=1.0 | 65.4% | Phase 4 best — needs 72B |
| HotpotQA / Mistral-7B | 59.5% | spectral doesn't transfer to multi-hop QA |

**Thesis claim**: Spectral features of H(n) work on reasoning tasks (math, science MCQ). Scope is reasoning-domain-specific, not general-purpose.

---

## Key files

| File | Purpose |
|------|---------|
| `GPQA_Phase8_Fixed.ipynb` | Phase 8 — run this next on Colab |
| `Spectral_Analysis_Phase9_QA_Validation.ipynb` | Phase 9 Part 2 CoT — run after Phase 8 |
| `spectral_utils/` | Core package (feature extraction, fusion, loaders) |
| `HISTORY.md` | Full step-by-step experiment log (Steps 1–79) |
| `Experiments_Report.md` | Clean summary for advisors |
| `Research_Directions.md` | Remaining directions with hypotheses |

---

## Key rules / gotchas

- **Never use `pip install git+...`** in Colab — use `git clone -b master` + `sys.path.insert`
- **AWQ models**: `load_model(model_id, quantize_4bit=False)` — package detects AWQ automatically
- **BNB 4-bit models**: never pass `torch_dtype` alongside `quantization_config`; use `device_map={"": 0}`
- **spectral_utils memory rule**: never inline helpers in notebooks — always import from the package
- **Nadler requires ≥3 views** — 2-view fusion collapses to near-random AUC
- **Z-score normalization** in `best_nadler_on` is on by default — needed for short-trace QA, negligible on long math traces

---

## Immediate next actions

1. **Run Phase 8** (`GPQA_Phase8_Fixed.ipynb`) on Colab A100 — expect ~65% accuracy, spectral AUC > 65.4%
2. **Run Phase 9 Part 2** (`Spectral_Analysis_Phase9_QA_Validation.ipynb`) — CoT inference for TriviaQA + WebQ
3. **Document results** in HISTORY.md (Steps 80+)
4. If Phase 8 succeeds: **create Phase 10 notebook** (ReAct agent loop on GPQA, step-level entropy)
