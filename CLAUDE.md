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

# Set BEFORE any torch import. Expandable segments let the allocator reclaim
# physical pages from a freed model, which is what makes a 70B BNB load after
# unloading a smaller model possible at all. Without this, fragmentation OOMs.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Persist HuggingFace cache to Drive — saves a 36 GB AWQ re-download (~15 min)
# on every runtime restart. Set BEFORE any HF import.
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'

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

# autoawq is safe here. gptqmodel is NOT — it rewrites numpy/pyarrow .so files
# during install and corrupts them mid-session. Defer gptqmodel to the cell that
# loads the model (see "Model loading rules" below).
os.system('pip install -q "transformers>=4.40" accelerate datasets bitsandbytes autoawq scipy')

from spectral_utils import (
    load_model, generate_full, free_memory,
    extract_all_features, sw_var_peak_with_window, sw_var_peak_adaptive,
    FEAT_NAMES, load_cache, save_cache,
    zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on,
)

# Force-load datasets (and through it pyarrow + pyarrow.parquet) into memory
# BEFORE the later gptqmodel install. C extensions can't be unloaded from a
# running Python process, so freezing them in memory now makes on-disk rewrites
# inert. Without this, the first lazy pyarrow import after gptqmodel install
# fails with `IpcReadOptions size changed`.
import datasets  # noqa: F401 — imported for side-effect

print('spectral_utils imported OK')
```

**NEVER use `pip install git+https://...`** — it ignores the branch, conflicts with Colab's module cache, and has failed repeatedly in this project.

---

## Model loading rules

- **AWQ / GPTQ models** (ID contains `awq` or `gptq`): `load_model(model_id, quantize_4bit=False)` — package auto-detects, uses `dtype=torch.bfloat16`. `device_map="auto"` is safe because AWQ weights are already quantized on disk (~36 GB for 72B). **Requires both `autoawq` AND `gptqmodel`** — gptqmodel provides the `AwqMarlinLinear` (Marlin fp16) kernel; without it, AWQ will fail or use a slow fallback.
  - **CRITICAL — install order + flags**: `autoawq` goes in Cell 1 (safe alone). `gptqmodel` MUST be installed in the model-load cell with `--no-deps`, AFTER `import datasets` has frozen pyarrow in memory. Installing both together in Cell 1 corrupts numpy and pyarrow on disk (`cannot import name '_center'`, `IpcReadOptions size changed`); installing gptqmodel without `--no-deps` ALSO corrupts transformers (`cannot import name 'divide_to_patches' from 'transformers.image_transforms'` — partial upgrade where new `image_processing_backends.py` references symbols missing in the still-old `image_transforms.py`). All three are unrecoverable in-session — only a runtime restart fixes them. The model-load cell should look like:
    ```python
    # gptqmodel's logger does `import pcre`. The PyPI package is pypcre (C extension
    # over libpcre2, no Py3.12 wheel — needs apt libpcre2-dev to build from source).
    # gptqmodel only uses pcre.compile()+.sub() on a trivial ANSI-escape pattern,
    # so stub pcre with stdlib re — bulletproof, no system libs, no C build.
    import re as _re, types as _types
    _pcre = _types.ModuleType('pcre')
    for _fn in ('compile','match','search','findall','sub','split','fullmatch'):
        setattr(_pcre, _fn, getattr(_re, _fn))
    _pcre.error = _re.error
    for _flag in ('IGNORECASE','MULTILINE','DOTALL','VERBOSE','UNICODE','ASCII'):
        setattr(_pcre, _flag, getattr(_re, _flag))
    sys.modules['pcre'] = _pcre

    # gptqmodel runtime deps that --no-deps skips. All pure-Python:
    # - device-smi (zero deps, eagerly imported by gptqmodel/utils/device.py)
    # - tokenicer (would otherwise pull transformers>=5.x and break Colab's 4.x)
    # - defuser (depends on pypcre — our stub covers it at runtime; --no-deps
    #   avoids pip trying to build pypcre from source against libpcre2-dev)
    # - logbar (zero deps; safe with full pip)
    os.system('pip install -q --no-deps device-smi tokenicer defuser')
    os.system('pip install -q logbar')
    os.system('pip install -q --no-deps gptqmodel')
    mdl, tok = load_model(MODEL_ID, quantize_4bit=False)
    ```
    `--no-deps` is safe because gptqmodel's heavy runtime deps (torch, numpy, transformers, accelerate, safetensors, pyarrow, datasets) are already in Colab. The four packages above are the only gptqmodel-specific runtime deps not in stock Colab Py3.12. **Do not** try `pip install pcre` (different obsolete package) or `pip install pypcre` (needs libpcre2-dev, slow source build) — the stdlib `re` stub is the reliable path.
- **BNB 4-bit (70B+)**: `load_model(model_id, quantize_4bit=True)`. Never pass `torch_dtype` alongside `quantization_config` — bitsandbytes owns dtype internally; passing it bypasses BNB and loads full FP16 → OOM.
- **BNB + 72B on A100**: `device_map="auto"` reads the *pre-quantization* FP16 size (~145 GB for Qwen-72B) and dispatches layers to CPU → BNB raises `ValueError: modules dispatched on CPU`. Fix: use `device_map={"": 0}` to force all layers to GPU before BNB quantizes them.
- **70B BNB on A100 (Llama-3.3-70B etc.)**: 4-bit quantization peaks around 80 GB. With `expandable_segments:True` set in Cell 1 it usually fits on a fresh runtime, but after any other model has been loaded and freed the load still OOMs. Gate the load behind a freshness check (`torch.cuda.max_memory_allocated() < 5 GB`); if not fresh, refuse the load and tell the user to restart and run Cells 1–6 + the 70B driver cell only. Checkpoints from completed (model, dataset) pairs persist to Drive and reload automatically, so the user loses no work.

**Known package gap**: `model_utils.py` still uses `device_map="auto"`. Safe for AWQ and small models. For 72B BNB, override in the notebook or patch the package before the run.

**Colab C-extension corruption — recovery**: If you see `cannot import name '_center'` (numpy) or `IpcReadOptions size changed` (pyarrow) mid-session, do NOT try `pip install --force-reinstall` to recover — C extensions can't be unloaded from a running Python process and the `.so` you're trying to replace is held open. The only fix is `Runtime → Restart runtime`. Prevention is the Cell 1 pre-import + deferred gptqmodel install above.

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
