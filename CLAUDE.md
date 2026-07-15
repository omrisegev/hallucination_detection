# CLAUDE.md — MV_EPR Spectral Hallucination Detection

## Session start
**Always read `PROGRESS.md` before doing anything else.** It has the current experiment status, what's running, what's fixed, and what to do next. Do not rely on git log alone — PROGRESS.md is the handoff document.

**Review [SUPERVISED_ORACLE_CORRECTION.md](file:///C:/Users/omris/TAU/hallucination_detection/SUPERVISED_ORACLE_CORRECTION.md)** to understand the ML evaluation guidelines (specifically regarding class weight balancing and avoiding the `cross_val_predict` calibration pitfall) established after correcting the Step 142 Logistic Regression baseline.

Shortcut: type `/session-start` to run the full initialization sequence automatically.

---

## Slash commands available

| Command | When to use |
|---------|-------------|
| `/session-start` | **Start of every session** — reads PROGRESS.md, git status, last HISTORY steps, prints priority action |
| `/update-docs` | After completing work — drafts HISTORY.md Step N entry + PROGRESS.md update, then commits |
| `/new-cell` | Adding an analysis/inference cell — generates correct three-branch pkl reload template |
| `/nadler-audit` | Before/after Nadler fusion — validates all 4 invariants (views, z-score, ρ, sign) |
| `/colab-setup` | New notebook — generates Cell 1 + Cell 2 + gptqmodel stub for the requested model |
| `/notebook-audit` | Before committing a notebook — spawns sub-agent to check for 8 common bugs |
| `/aircc-setup` | One-time AIRCC cluster bootstrap (first login is manual; config/dirs/prefetch automated) |
| `/aircc-submit` | Submit an inference job to the AIRCC cluster (sync code + sbatch + job id) |
| `/aircc-status` | Check AIRCC job state — squeue/sacct + log tail + verdict |
| `/aircc-fetch` | Fetch finished cluster results + validate the rich-save pkl schema |
| `/paper-digest` | Read/re-read a paper under `papers/` — checks the cache (`papers/index.md`) first, only extracts + digests if not already cached |

---

## Sub-agent patterns

Spawn sub-agents for these recurring tasks to avoid context pollution:

**Cache Explorer** — Use when user asks "what's in the cache?" or "which phases are done?":
```
Spawn subagent_type=Explore:
"List all .pkl files under /content/drive/MyDrive/hallucination_detection/[path].
For each pkl, report: filename, size, top-level keys, and whether any values are None.
Classify each as VALID / STALE (all-None) / PARTIAL (some None). Report in a table."
```

**Notebook Reviewer** — Use before committing any notebook with new cells (or just use `/notebook-audit`):
```
Spawn subagent_type=Explore with the notebook path + the 8-bug checklist from /notebook-audit.
Returns findings without modifying files. Synthesize before deciding what to fix.
```

**Results Extractor** — Use when user asks "what are our current numbers?" without re-running:
```
Spawn subagent_type=Explore:
"Read consolidated_results/results_all.pkl (or results_summary.csv).
Print a table: domain | model | Nadler AUROC | CI | best_subset.
Sort by Nadler AUROC descending. Flag any None results."
```

**Rule**: Brief the agent with exact file paths + what to return. Synthesize the result yourself before acting on it. Never delegate the decision — only the data gathering.

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

---

## Google Drive cache — don't trust the HF default

Drive's FUSE doesn't support real symlinks. HF's hub cache (`HF_HOME=/content/drive/...`) stores blobs as real files but tries to symlink them into `snapshots/<rev>/<file>` — those symlinks become 0-byte broken stubs on Drive, and HF re-downloads the full model every session despite the blobs sitting there.

**Fix**: bypass the cache. Use `snapshot_download(local_dir=...)` to a flat directory on Drive (real files, no symlinks), then pass the local path to `from_pretrained` instead of the Hub repo ID. Standard helper to put in a notebook setup cell:

```python
from huggingface_hub import snapshot_download
FLAT_CACHE = '/content/drive/MyDrive/hf_cache_flat'
os.makedirs(FLAT_CACHE, exist_ok=True)

def ensure_flat_dir(repo_id, token=None):
    """Download repo to flat dir on Drive (real files, no symlinks). Idempotent."""
    local_dir = os.path.join(FLAT_CACHE, repo_id.replace('/', '__'))
    sentinel = os.path.join(local_dir, 'config.json')
    if os.path.exists(sentinel):
        return local_dir
    kwargs = dict(repo_id=repo_id, local_dir=local_dir, token=token)
    try:
        snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)  # newer hf_hub removed the kwarg; default is copies
    return local_dir
```

`load_model()` still auto-detects AWQ from the path string (looks for "awq"/"gptq"), so passing `/content/drive/MyDrive/hf_cache_flat/Qwen__Qwen2.5-72B-Instruct-AWQ` works the same as the Hub ID.

---

## AIRCC cluster (Slurm GPU allocation)

Second GPU backend besides Colab: national AIRCC cluster, 8× NVIDIA B200, ssh alias `aircc`
(omrisegev1@slurm-login.iucc.ac.il, **TAU VPN required** — a hanging ssh means VPN is down).
Full reference: [cluster/README.md](cluster/README.md). Rules that must never be violated:

- Work only under `/shared/cycle2_tau_averbuch_prj/omrisegev1`, never `$HOME`.
- B200 = sm_100: jobs run inside `nvcr.io/nvidia/pytorch:25.01-py3` (rootless Docker).
  Never pip-upgrade torch inside it; `cluster/requirements.txt` deliberately omits torch/numpy.
- Preemption: SIGTERM → 15 min → SIGKILL → auto-requeue. Every long job must use the
  `cluster/run_inference.py` checkpoint/resume pattern (atomic saves via `save_cache_atomic`,
  SIGTERM trap, idempotent restart).
- Code reaches the cluster via `bash cluster/sync_code.sh` (tar-over-ssh) — never rely on
  Claude pushing to GitHub (credentials are not available in-session).
- Workflow: `/aircc-setup` (once) → `/aircc-submit` → `/aircc-status` → `/aircc-fetch`.
- **All cluster polling / log-tailing goes through `/aircc-status` or the `cluster-ops` sub-agent — never raw `ssh aircc "squeue/sacct/tail"` loops in the main context.** Each raw ssh re-prints the login banner and dumps full logs into context; the sub-agent returns a one-line verdict. (Step-163 retro: inline ssh polling was the single biggest recurring token sink.)
- **A new `cluster/presets.py` preset MUST pass `python scripts/smoke_preset.py <id>` (CPU-only) before it is submitted.** It runs the preset's real prompt/grader/judge helpers on fixtures — catching prompt / grader / judge-parse bugs offline instead of via a GPU round-trip (4 of the 6 Step-163 pilot bugs were this kind). Gate order: **local smoke → N=30 pilot → full N.**

---

## Raw inference data — save everything, derive later

**Rule**: save the richest raw form of each inference result to Drive. Never discard information during the inference loop that could be useful for any future feature or baseline.

Per-sample pkl entry must include:

| Key | Type | What it is |
|-----|------|------------|
| `full_text` | `str` | Generated answer string |
| `token_entropies` | `list[float]` | H(n) — Shannon entropy per token (top-K=15) |
| `token_spilled_energies` | `list[float]` | ΔE(n) = −log p(sampled token) per token |
| `top_k_logprobs` | `{'ids': int32 [T,50], 'logprobs': float32 [T,50]}` | Top-50 log-probs per token as a numpy-array pair (~3.5× smaller than the old `list[list[tuple]]` form; both schemas are valid in old caches) |
| `gen_token_ids` | `list[int]` | Sampled token IDs (needed for ΔE and attention features) |
| `label` | `int/bool` | Correctness label (graded at inference time) |
| `question` | `str` | Original question text |

**Why `top_k_logprobs`**: H(n) and ΔE(n) are derived quantities that fix K=15 and the sampled token at generation time. Saving top-50 logprobs lets you recompute entropy at any K, compute probability mass features, implement token-level confidence, or compute any future feature — without re-running the model. Storage cost: ~200 KB/sample at 500 tokens × 50 top entries (numpy form).

`generate_full()` returns `top_k_logprobs` and `gen_token_ids` since the AIRCC onboarding commit (param `logprob_top_k=50`, set 0 to disable). The standalone extractor is also exported as `spectral_utils.extract_top_k_logprobs(scores, top_k=50)` for recomputation from cached scores.

Old cached pkls that only have `token_entropies` are still valid for H(n)-based features. The spilled energy and logprob features simply won't be available for those runs.

---

## Analysis-result persistence (Colab `background_save` survival)

Long-running analysis cells (Nadler subset search, length-controlled, PCA, SE baseline) MUST persist their output dict to disk. Colab's `background_save: true` lets the cell finish printing after a kernel disconnect, but in-memory variables (`NADLER_RES`, `LEN_RES`, `PCA_RES`) are gone. Downstream cells then `NameError`.

Standard pattern at the top of each analysis cell:

```python
RES_PATH = os.path.join(RES_DIR, 'foo_res.pkl')
FORCE_RECOMPUTE = False

if not FORCE_RECOMPUTE and 'FOO_RES' in globals() and FOO_RES:
    print('already in memory; skipping')
elif not FORCE_RECOMPUTE and os.path.exists(RES_PATH):
    with open(RES_PATH, 'rb') as f: FOO_RES = pickle.load(f)
    print(f'loaded from {RES_PATH}')
else:
    FOO_RES = {}
    # ... compute ...
    with open(RES_PATH, 'wb') as f: pickle.dump(FOO_RES, f)
    print(f'saved to {RES_PATH}')
```

Same pattern as Cell 6's `run_inference_for_cell` checkpoint logic. Apply it to every cell that takes more than ~30 seconds.

**Local offline scoring — token/time economy** (Step-163 retro):
- **Scoring or feature-extracting a cell >100 MB or K≥10 runs in the background** (`run_in_background: true`) with a generous timeout — never a foreground short timeout. The 857 MB losnet cell timed out at `timeout 400` and had to be re-run; K=10 FFT extraction on a big pkl needs minutes.
- **Inspect any cell's schema with `python scripts/inspect_cell.py <pkl|preset_dir>` before scoring** — it prints N/K, label dist, trace lengths, key-presence (base + energy + judge keys), and the extractable feature set + valid-rate. This replaces ad-hoc `python -c` pkl spelunking.

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

Papers live in `papers/`. **Check `papers/index.md` first** — if a paper is already
`digested`, read `papers/digests/<slug>.md` (and `papers/extracted/<slug>.md` for exact
quotes) instead of re-reading the PDF from scratch.

If it's not cached yet: follow `skills/paper-digest/SKILL.md` (extract → digest → index),
or just run `/paper-digest`. Same procedure whether you're Claude Code or antigravity/Gemini
— the skill is mirrored to `.gemini/skills/paper-digest/` for exactly that reason.

If the paper changes the roadmap, still update `PROGRESS.md` and `Research_Directions.md`.
A substantive new read is worth a HISTORY.md step (title + pointer to the digest file); a
cache hit is not.

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
