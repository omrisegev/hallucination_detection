---
description: Generate complete Colab notebook boilerplate — Cell 1 (Drive mount + clone + pip install + imports), Cell 2 (config), and optionally the gptqmodel stub cell. Use when creating a new notebook or resetting the setup cells of an existing one. Accepts: MODEL (model ID or family), DATASET, QUANT (none|bnb4|awq).
---

Generate the standard Colab setup cells for this project. Ask for any missing parameters before generating.

## Parameters
- **MODEL**: HuggingFace model ID or short name (e.g., `Qwen/Qwen2.5-7B-Instruct`, `qwen72b`, `llama8b`)
- **DATASET**: which dataset to load (e.g., `gsm8k`, `gpqa`, `hotpotqa`, `math500`, `humaneval`)
- **QUANT**: quantization method — `none` (default), `bnb4` (4-bit BitsAndBytes), `awq` (AWQ/GPTQ)
- **PHASE**: experiment phase number (for cache directory naming)

---

## Cell 1 — Drive mount + clone + pip install + imports

```python
# Cell 1 — Drive mount + setup
import os, sys, shutil

# MUST be before any torch import — lets allocator reclaim pages from freed models
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Persist HF cache to Drive (avoids 15-min re-download on restart)
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'

from google.colab import drive
drive.mount('/content/drive')

REPO_DIR = '/content/hallucination_detection'
BRANCH   = 'feature/meta-agentic-integration'   # baselines.py only exists here

if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
    shutil.rmtree(REPO_DIR)
if not os.path.exists(REPO_DIR):
    os.system(f'git clone -b {BRANCH} https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
else:
    os.system(f'git -C {REPO_DIR} pull -q')
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# autoawq is safe here. gptqmodel is NOT — install it in the model-load cell only.
os.system('pip install -q "transformers>=4.40" accelerate datasets bitsandbytes autoawq scipy')

from spectral_utils import (
    load_model, generate_full, free_memory,
    extract_all_features, sw_var_peak_adaptive, FEAT_NAMES,
    load_cache, save_cache,
    zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on,
    segment_by_citations, lciteeval_grounding_label,
)
# Dataset-specific imports — add as needed:
# from spectral_utils import load_gsm8k, gsm8k_prompt, is_correct_gsm8k
# from spectral_utils import load_math500, math_prompt, is_correct_math
# from spectral_utils import load_gpqa, gpqa_prompt_and_answer, is_correct_gpqa
# from spectral_utils import load_lciteeval, lciteeval_prompt, lciteeval_grounding_label
# from spectral_utils import load_hotpotqa, hotpotqa_prompt, is_correct_hotpotqa

# Force-load pyarrow NOW before any gptqmodel install rewrites it
import datasets  # noqa — side-effect import to freeze C extensions

assert len(FEAT_NAMES) == 16, f"Expected 16 features, got {len(FEAT_NAMES)}: {FEAT_NAMES}"
print(f'spectral_utils OK | branch={BRANCH} | FEAT_NAMES ({len(FEAT_NAMES)}): {FEAT_NAMES}')
```

---

## Cell 2 — Config

```python
# Cell 2 — Config
MODEL_ID   = '<MODEL_ID>'        # e.g. 'Qwen/Qwen2.5-7B-Instruct'
TEMP       = 1.0                 # sampling temperature
MAX_NEW    = 512                 # max new tokens per generation
N_SAMPLES  = 200                 # dataset size
K_SE_SC    = 10                  # K for Semantic Entropy / Self-Consistency (if used)

BASE_DRIVE = '/content/drive/MyDrive'
HALL_DIR   = f'{BASE_DRIVE}/hallucination_detection'
CACHE_DIR  = f'{HALL_DIR}/cache/phase<PHASE>_<DATASET>'
os.makedirs(CACHE_DIR, exist_ok=True)

# Short model display name for plots/tables
MODEL_SHORT = MODEL_ID.split('/')[-1].replace('-Instruct','').replace('-instruct','')

print(f'Config: {MODEL_SHORT} | T={TEMP} | N={N_SAMPLES} | CACHE_DIR={CACHE_DIR}')
```

---

## gptqmodel stub cell (only for AWQ/GPTQ models — insert before model-load cell)

**IMPORTANT**: Only include this cell when `QUANT=awq`. Do NOT put in Cell 1.

```python
# gptqmodel stub — insert directly before model-load cell, NOT in Cell 1
# Stub pcre with stdlib re (gptqmodel uses pcre only for ANSI escape stripping)
import re as _re, types as _types
_pcre = _types.ModuleType('pcre')
for _fn in ('compile','match','search','findall','sub','split','fullmatch'):
    setattr(_pcre, _fn, getattr(_re, _fn))
_pcre.error = _re.error
for _flag in ('IGNORECASE','MULTILINE','DOTALL','VERBOSE','UNICODE','ASCII'):
    setattr(_pcre, _flag, getattr(_re, _flag))
sys.modules['pcre'] = _pcre

# Runtime deps that --no-deps skips (all pure-Python, no system libs needed)
os.system('pip install -q --no-deps device-smi tokenicer defuser')
os.system('pip install -q logbar')
os.system('pip install -q --no-deps gptqmodel')

mdl, tok = load_model(MODEL_ID, quantize_4bit=False)   # AWQ auto-detected from model ID
print(f'Model loaded: {MODEL_ID}')
```

---

## Model-load cell variants (choose based on QUANT)

**QUANT=none** (standard 7B models):
```python
mdl, tok = load_model(MODEL_ID)
print(f'Model loaded: {MODEL_ID}')
```

**QUANT=bnb4** (70B+ models — needs fresh runtime):
```python
# Gate on fresh runtime (BNB 70B OOMs after any prior model)
import torch
if torch.cuda.max_memory_allocated() > 5e9:
    raise RuntimeError('Not a fresh runtime — restart and run only Cells 1-2 + this cell')
mdl, tok = load_model(MODEL_ID, quantize_4bit=True)
print(f'Model loaded (BNB 4-bit): {MODEL_ID}')
```

**QUANT=awq** (72B AWQ):
```python
# Run the gptqmodel stub cell first, then:
mdl, tok = load_model(MODEL_ID, quantize_4bit=False)  # AWQ auto-detected
print(f'Model loaded (AWQ): {MODEL_ID}')
```

---

## Notes
- **Never** use `pip install git+https://...` — use `git clone -b {BRANCH}` only
- **Always** set `PYTORCH_CUDA_ALLOC_CONF` before any torch import (Cell 1)
- **Always** `import datasets` in Cell 1 to freeze pyarrow before gptqmodel installs
- **HF_HOME on Drive breaks symlinks** — if model re-downloads every session, use `ensure_flat_dir()` pattern from CLAUDE.md instead
- **BNB + 72B + A100**: use `device_map={"": 0}` not `device_map="auto"` (auto dispatches to CPU → BNB error)
