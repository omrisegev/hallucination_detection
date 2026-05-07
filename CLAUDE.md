# CLAUDE.md — MV_EPR Spectral Hallucination Detection

## Session start
**Always read `PROGRESS.md` before doing anything else.** It has the current experiment status, what's running, what's fixed, and what to do next. Do not rely on git log alone — PROGRESS.md is the handoff document.

---

## Project layout

| Path | Purpose |
|------|---------|
| `spectral_utils/` | Core package — all helpers live here |
| `HISTORY.md` | Step-by-step experiment log (append only, see format below) |
| `PROGRESS.md` | Handoff summary — update at end of each session |
| `Research_Directions.md` | Full thesis roadmap — directions, hypotheses, decision gates, priority order |
| `Experiments_Report.md` | Clean summary for advisors |
| `*.ipynb` | Colab notebooks — run on Colab A100, never locally |
| `*.pdf` | Research papers — read when referenced or newly added |

---

## spectral_utils: the non-negotiable rule

**NEVER inline helpers in notebooks.** Every feature extractor, fusion function, model loader, and grading function lives in `spectral_utils`. If something is missing from the package, add it to the right module and commit *before* using it in a notebook.

| Module | Contents |
|--------|---------|
| `feature_utils.py` | `compute_spectral_features`, `compute_stft_features`, `compute_time_domain`, `extract_all_features`, `sw_var_peak_with_window`, `sw_var_peak_adaptive`, `FEAT_NAMES` |
| `fusion_utils.py` | `zscore`, `boot_auc`, `nadler_fuse`, `simple_average_fusion`, `best_nadler_on` |
| `model_utils.py` | `load_model`, `fmt_prompt`, `generate_full`, `token_entropies_from_scores`, `free_memory` |
| `data_loaders.py` | `load_gsm8k/math500/gpqa/hotpotqa/trivia_qa/webq` + prompt + grading functions |
| `io_utils.py` | `load_cache`, `save_cache` |

---

## Colab setup — standard Cell 1 for every new notebook

```python
import os, sys, shutil

REPO_DIR = '/content/hallucination_detection'

# Remove stale clone if spectral_utils is missing
if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
    shutil.rmtree(REPO_DIR)

if not os.path.exists(REPO_DIR):
    os.system(f'git clone -b master https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
else:
    os.system(f'git -C {REPO_DIR} pull -q')

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.system('pip install -q "transformers>=4.40" accelerate datasets bitsandbytes autoawq gptqmodel scipy')

from spectral_utils import (
    load_model, generate_full, free_memory,
    extract_all_features, sw_var_peak_with_window, sw_var_peak_adaptive,
    FEAT_NAMES, load_cache, save_cache,
    zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on,
)
print('spectral_utils imported OK')
```

**NEVER use `pip install git+https://...`** — it ignores the branch, conflicts with Colab's module cache, and has failed repeatedly in this project.

---

## Model loading rules

- **AWQ / GPTQ models** (ID contains `awq` or `gptq`): `load_model(model_id, quantize_4bit=False)` — package auto-detects, uses `dtype=torch.bfloat16`. `device_map="auto"` is safe because AWQ weights are already quantized on disk (~36 GB for 72B). **Requires both `autoawq` AND `gptqmodel`** — gptqmodel provides the `AwqMarlinLinear` (Marlin fp16) kernel; without it, AWQ will fail or use a slow fallback. Install both in the same cell.
- **BNB 4-bit (70B+)**: `load_model(model_id, quantize_4bit=True)`. Never pass `torch_dtype` alongside `quantization_config` — bitsandbytes owns dtype internally; passing it bypasses BNB and loads full FP16 → OOM.
- **BNB + 72B on A100**: `device_map="auto"` reads the *pre-quantization* FP16 size (~145 GB for Qwen-72B) and dispatches layers to CPU → BNB raises `ValueError: modules dispatched on CPU`. Fix: use `device_map={"": 0}` to force all layers to GPU before BNB quantizes them.

**Known package gap**: `model_utils.py` still uses `device_map="auto"`. Safe for AWQ and small models. For 72B BNB, override in the notebook or patch the package before the run.

---

## Notebook standard cell sequence

1. Clone + pip install + imports (`spectral_utils`)
2. Config (`MODEL_ID`, `TEMP`, `MAX_NEW`, `CACHE_DIR`, `N_SAMPLES`, feature subset)
3. Mount Google Drive + create cache dirs
4. Load model (`load_model`)
5. Inference loop with checkpointing (save every 25 samples to `.pkl`)
6. Unload model (`del mdl, tok; free_memory()`)
7. Feature extraction (`extract_all_features` → aligned labels + valid_results)
8. Window ablation for `sw_var_peak` over `WINDOW_SIZES`
9. Individual feature AUC table
10. Nadler fusion — `best_nadler_on` for new experiments; `apply_fixed_subset` for validation phases
11. Gate/decision cell (explicit pass/fail thresholds, printed summary)
12. Save results dict to `.pkl`
13. Plots (feature AUC bar, results landscape, entropy trajectories)

---

## HISTORY.md format

Append new steps at the bottom of `## Steps`. Number sequentially from the last step.

```
### Step N — <one-line title>

**What**: What was done or discovered.
**Why**: Why this matters / what triggered it.
**Result**: Outcome, numbers, or status.

---
```

For debugging sessions with multiple iterations, group under one step with sub-headers (`#### Sub-step` or `**Attempt N**`). Don't create a new step for every failed attempt — one step per logical investigation.

---

## Research_Directions.md — the thesis roadmap

This is the single source of truth for what to do next and why. It contains:
- 7 directions with hypotheses, proposed experiments, feasibility/novelty/risk ratings, and supervisor connections
- Per-direction status (`Active`, `Completed`, `Not started`, `Pending`)
- Per-experiment status (✅ COMPLETED, ← NEXT PRIORITY, etc.)
- Decision gates with explicit thresholds (e.g. G3: beat LOS-Net 72.92% on HotpotQA)
- Priority order at the bottom, updated as experiments complete

**When to read it**: before planning a new experiment or notebook, when the user asks "what's next", or when a paper changes the roadmap.

**When to update it**: after an experiment completes, update the status field and add results to the relevant Phase/Experiment section. When a new paper shifts priorities, update the priority order section.

Do **not** duplicate information between Research_Directions.md and HISTORY.md. Research_Directions.md holds the plan and aggregated results tables; HISTORY.md holds the step-by-step narrative of what happened and why.

---

## Research papers

When a PDF appears in the working directory or the user references a paper:
1. Read it with the `Read` tool (specify pages for large PDFs — max 20 per call).
2. Extract: (a) core method/finding, (b) connection to our spectral pipeline, (c) usable benchmarks, code, or hyperparameters.
3. Append a HISTORY.md step: `### Step N — [Paper title]: assessed`.
4. If the paper changes the roadmap, update `PROGRESS.md` and `Research_Directions.md`.

---

## Nadler fusion invariants (never violate these)

- Requires **≥ 3 views** — 2-view fusion collapses to near-random AUC.
- Apply **z-score normalization** before fusion (`normalize=True` in `best_nadler_on`) — required for short-trace QA, negligible on long math traces.
- Correlation filter: skip any subset where pairwise Spearman |ρ| ≥ 0.75.
- Feature sign: orient each feature so higher score → more likely correct before fusing.

---

## Best results (reference)

| Setup | Nadler AUC | Notes |
|-------|-----------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **90.0%** | spectral features work on long reasoning |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
| GPQA / Mistral-7B / T=1.0 | 65.4% | Phase 4 best — beaten by 72B (Phase 8) |
| HotpotQA / Mistral-7B | 59.5% | spectral doesn't transfer to multi-hop QA |

**Thesis scope**: Spectral features of H(n) work on reasoning-heavy domains (math, science MCQ). Not general-purpose; short factual QA traces are structurally incompatible.
