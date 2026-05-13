# MV_EPR — Session Progress Handoff

**Date**: 2026-05-13
**Last updated**: NADLER_RES / LEN_RES / PCA_RES persistence pattern applied to Phase 10 Main RAG notebook (Cells 14–16); ready to re-run on Colab

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
| GPQA / Qwen2.5-72B-AWQ / T=1.0 | **69.0%** | Phase 8 — +3.6 pp over 7B; acc only 40.4% |
| Phase 10 RAG / qwen7b / 2wikimultihopqa | **80.5%** | Cell 14 output, this session |
| Phase 10 RAG / qwen7b / hotpotqa | 79.5% | Cell 14 output, this session |
| Phase 10 RAG / qwen72b / hotpotqa | 79.4% | Cell 14 output, this session |

---

## Completed phases

- **Phase 8** ✅: GPQA Diamond / Qwen2.5-72B-AWQ — 69.0% AUC (Step 80)
- **Phase 9** ✅: Factual QA (TriviaQA + WebQ) CoT — spectral features don't transfer, clean negative result (Step 82)
- **Phase 10 pilot** ✅: L-CiteEval HotpotQA / Falcon-3-10B — INVALID pre-conditions, but strong signal: Nadler 76.0%, EPR 69.9% (Step 84)
- **Phase 10 Main RAG — 3/4 models** ✅: qwen7b + mistral24b + qwen72b inference all complete; analysis cells partially done (NADLER_RES needs persistence patch)

---

## Current experiment: Phase 10 Main RAG

**Notebook**: `Spectral_Analysis_Phase10_Main_RAG.ipynb`
**Goal**: 4 models × 4 L-CiteEval datasets = 16 (model, dataset) cells. Headline: 4×4 AUC heatmap showing spectral features generalise across tasks.
**Datasets**: hotpotqa, natural_questions, 2wikimultihopqa, narrativeqa
**Models**: Qwen2.5-7B, Mistral-Small-24B, Qwen2.5-72B-AWQ, Llama-3.3-70B

### Inference status

| Model | hotpotqa | natural_questions | 2wikimultihopqa | narrativeqa | Status |
|-------|----------|-------------------|-----------------|-------------|--------|
| qwen7b     | 240/240 ✅ | 160/160 ✅ | 240/240 ✅ | 240/240 ✅ | Complete |
| mistral24b | 240/240 ✅ | 160/160 ✅ | 240/240 ✅ | 240/240 ✅ | Complete |
| qwen72b    | 240/240 ✅ | 160/160 ✅ | 240/240 ✅ | 240/240 ✅ | Complete (this session) |
| llama70b   | 0          | 0          | 0          | 0          | Pending — needs fresh runtime |

Checkpoints on Drive: `/content/drive/MyDrive/hallucination_detection/cache/phase10_main/raw/`
HF cache on Drive (flat-dir, no symlinks): `/content/drive/MyDrive/hf_cache_flat/`

### Cell 14 best Nadler results (12 of 16 cells)

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
```

Median ≈ 74%; 7/12 cells ≥ 70% (G1 threshold). Spectral features generalise across models AND tasks.

### Where it stopped (and the fix)

Cell 14 ran successfully and printed all 12 results, but had `"background_save": true` in its metadata. The kernel disconnected before formally finalising the cell, so `NADLER_RES` was wiped from memory. Cells 16/17/18 then errored with `NameError: NADLER_RES not defined`.

**Fix applied** (this session): Cells 14, 15, 16 in `Spectral_Analysis_Phase10_Main_RAG.ipynb` now persist `NADLER_RES`/`LEN_RES`/`PCA_RES` to disk as `.pkl` files in `RES_DIR`, with the standard in-memory → on-disk → recompute pattern and a `FORCE_RECOMPUTE_*` flag for explicit refresh. Same pattern as Cell 6's inference checkpoints. Future kernel restarts reload in milliseconds. Spec preserved in `FIX_NADLER_RES.md` for reference.

---

## Resolved blockers (this session and prior)

### 1. Qwen-72B-AWQ — pcre / gptqmodel (Python 3.12) ✅

**Root cause**: gptqmodel's logger AND its `cpp.py` AND its `defuser` dep all import `pcre`. The PyPI package is `pypcre` (C ext over libpcre2, no Py3.12 wheel — and the previously-tried `libpcre3-dev` is the wrong libpcre version).

**Fix**: Stub `pcre` against stdlib `re` before gptqmodel import. The stub exposes:
- functions: `compile/match/search/findall/sub/split/fullmatch`
- classes: `Pattern`, `Match` (for type annotations in `defuser/utils/common.py`)
- flags: both re-style (`IGNORECASE`, `VERBOSE`) AND PCRE-style (`CASELESS`, `EXTENDED`, `UTF8`, `UCP`, etc.) — gptqmodel mixes both
- `Flag` namespace (used as `pcre.Flag.CASELESS` in `gptqmodel/models/writer.py`)

PCRE-specific flags with no `re` equivalent (`ANCHORED`, `UNGREEDY`, ...) map to `0` (no-op).

### 2. Llama-3.3-70B — OOM after Mistral-24B ✅

**Root cause**: 70B BNB 4-bit peaks ~80 GB during quantization. After any smaller model has been loaded/freed, the allocator carries fragmentation and the peak doesn't fit.

**Fix** (two-pronged):
1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in Cell 1 before any torch import.
2. Cell 10 guards on `torch.cuda.max_memory_allocated()` > 5 GB. If the runtime isn't fresh, it refuses the load and prints the recovery procedure.

### 3. gptqmodel `--no-deps` skips real runtime deps ✅

**Discovered**: After the pcre stub worked, gptqmodel failed at import on `device_smi`, then `tokenicer`, then `defuser`. `--no-deps` is needed to avoid transformers .py rewrites, but it also skips genuine pure-Python runtime deps. Install them explicitly:
```
pip install --no-deps device-smi tokenicer defuser
pip install logbar ninja
pip install --no-deps gptqmodel
```
`ninja` is required at model-load time to JIT-build the Marlin fp16 CUDA kernel.

### 4. Google Drive symlink bug → HF re-downloads every session ✅

**Discovered**: HF hub cache stores blobs as real files + `snapshots/<rev>/` as symlinks. Google Drive's FUSE doesn't support real symlinks, so snapshot symlinks come out as 0-byte broken stubs. The 17.8 GB AWQ download succeeded EVERY session (blobs saved), but HF couldn't resolve the snapshot dir and re-downloaded.

Confirmed by Cell 3b diagnostic: 431 GB of blobs sitting on Drive, all snapshot symlinks `islink=True size=0`.

**Fix**: Cell 3c `ensure_flat_dir(repo_id)` uses `snapshot_download(local_dir=...)` to flat-dir on Drive (no symlinks). Cells 9 and 10 call `ensure_flat_dir(MODEL_ID)` before `load_model`. After the one-time re-download, the model loads from Drive in seconds. Pattern documented in CLAUDE.md.

### 5. `lciteeval_grounding_label` list-of-list answers ✅

NQ/NarrativeQA return `answers` as `list[list[str]]`, not `list[str]`. Flatten before substring matching. Fixed in `spectral_utils/data_loaders.py` (commit `8aa3587`).

### 6. `best_nadler_on` 4-tuple vs 5-tuple ✅

Notebook expected `auc, lo, hi, subset, weights` but package returned 4 values. Updated `fusion_utils.py` to also return the leading-eigenvector weights for the best subset (commit `b3c45a4`). Needed for Cell 18's spectral fingerprint heatmap.

### 7. NADLER_RES disappears on Colab disconnect ✅

Cell 14's `background_save: true` lets the cell keep printing after kernel disconnect, but the variable is gone. Cells 14/15/16 of `Spectral_Analysis_Phase10_Main_RAG.ipynb` now persist `NADLER_RES`/`LEN_RES`/`PCA_RES` to `RES_DIR/{nadler,len,pca}_res.pkl` and reload from disk on subsequent runs (`FORCE_RECOMPUTE_*` to override). Spec retained in `FIX_NADLER_RES.md` for reference.

---

## Key rules / gotchas

- **gptqmodel on Python 3.12**: stub `pcre` with stdlib `re` (including `Pattern`/`Match` classes and the full `Flag` namespace including PCRE-style names like `CASELESS`/`EXTENDED`). Plus install `device-smi tokenicer defuser` with `--no-deps`, and `logbar ninja` plainly, before `pip install --no-deps gptqmodel`.
- **70B BNB models**: with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in Cell 1, loading after a freed smaller model is possible but unreliable. Cell 10 guards via `max_memory_allocated()`; for safety, run Llama as the only model in a fresh runtime.
- **HF cache on Drive**: NEVER rely on the standard `HF_HOME=/content/drive/...` cache — snapshot symlinks break. Use `snapshot_download(local_dir=...)` to a flat dir, then pass that path to `from_pretrained`. Helper `ensure_flat_dir()` in Cell 3c.
- **Analysis result persistence**: Long-running analysis cells (Nadler search, length-controlled, PCA) MUST persist results to disk (`pickle.dump` to `RES_DIR`). Colab's `background_save: true` lets cells finish printing after kernel disconnect, but in-memory variables are lost. Same checkpoint pattern as `Cell 6`'s inference loop.
- **AWQ models**: `load_model(model_id, quantize_4bit=False)` — auto-detects AWQ. Requires both `autoawq` AND `gptqmodel`.
- **L-CiteEval dataset sizes**: hotpotqa=240, NQ=160, 2wiki=240, narrativeqa=240. `N_SAMPLES=500` is capped automatically.
- **Never use `pip install git+...`** — use `git clone -b master` + `sys.path.insert`.

---

## Immediate next actions

1. **Re-run Cells 11 → 25 on Colab** with the patched notebook. Cells 14/15/16 will now save `NADLER_RES`/`LEN_RES`/`PCA_RES` to `RES_DIR` as `.pkl`, so downstream cells (16/17/18/20/21/24) won't `NameError` even if the kernel disconnects. After this you have 12-cell results saved to Drive.

2. **Run Llama-70B in a separate Colab session** (fresh runtime):
   - Cells 1–6 → SKIP 7, 8, 9 → Cell 10
   - Cell 10's guard ensures fresh-runtime requirement
   - Cell 10 will use `ensure_flat_dir(MODEL_ID, token=hf_token)` for the one-time flat-dir download

3. **Re-run Cells 11–25** after Llama-70B is done. Cells 11/14/15/16 will detect new raw files; set `FORCE_RECOMPUTE_*` to refresh the persisted dicts instead of loading the 12-cell snapshots. Final outputs: 4×4 AUC heatmap, 16-row Nadler weight fingerprint heatmap, fusion distributions, length-controlled bars, gates.

4. **After 16/16 cells**:
   - Append a follow-up step to HISTORY.md with headline numbers for the full 16-cell run
   - Update `Research_Directions.md` Direction 2 (RAG) status
   - Update advisor draft

5. **Agent experiment** (Direction 4, separate notebook `Spectral_Analysis_Phase10_Agentic.ipynb`): not started.

---

## Email to advisors

Draft completed in prior session. Ready to send once Llama-70B finishes and we have the full 4×4 heatmap.
