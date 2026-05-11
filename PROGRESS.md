# MV_EPR — Session Progress Handoff

**Date**: 2026-05-11  
**Last updated**: Cell 9 (Qwen-72B-AWQ) and Cell 10 (Llama-70B) blockers fixed; ready to run

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
| HotpotQA pilot / Falcon-3-10B | 76.0% (INVALID) | Pre-conditions failed by thin margin |
| TriviaQA / Falcon-3-10B CoT | 53.6% | Spectral features don't transfer to factual QA |

---

## Completed phases

- **Phase 8** ✅: GPQA Diamond / Qwen2.5-72B-AWQ — 69.0% AUC (Step 80)
- **Phase 9** ✅: Factual QA (TriviaQA + WebQ) CoT — spectral features don't transfer, clean negative result (Step 82)
- **Phase 10 pilot** ✅: L-CiteEval HotpotQA / Falcon-3-10B — INVALID pre-conditions, but strong signal: Nadler 76.0%, EPR 69.9% (Step 84)

---

## Current experiment: Phase 10 Main RAG

**Notebook**: `Spectral_Analysis_Phase10_Main_RAG.ipynb`  
**Goal**: 4 models × 4 L-CiteEval datasets = 16 (model, dataset) cells. Headline: 4×4 AUC heatmap showing spectral features generalise across tasks.  
**Datasets**: hotpotqa, natural_questions, 2wikimultihopqa, narrativeqa  
**Models**: Qwen2.5-7B, Mistral-Small-24B, Qwen2.5-72B-AWQ, Llama-3.3-70B

### Inference status (as of session end)

| Model | hotpotqa | natural_questions | 2wikimultihopqa | narrativeqa | Status |
|-------|----------|------------------|-----------------|-------------|--------|
| qwen7b | 240/240 ✅ | 160/160 ✅ | 240/240 ✅ | 240/240 ✅ | Complete |
| mistral24b | 240/240 ✅ | 160/160 ✅ | 240/240 ✅ | 240/240 ✅ | Complete |
| qwen72b | 0 | 0 | 0 | 0 | Ready to run (Cell 9 fix landed) |
| llama70b | 0 | 0 | 0 | 0 | Ready on fresh runtime (Cell 10 fix landed) |

Checkpoints saved to Drive: `/content/drive/MyDrive/hallucination_detection/cache/phase10_main/raw/`

### Where it stopped

Cell 11 (feature extraction) fails with `AttributeError: 'list' object has no attribute 'lower'`.

**This bug is FIXED and PUSHED** (commit `8aa3587`). The fix flattens list-of-list answers in `lciteeval_grounding_label` — NaturalQuestions and NarrativeQA return `answers` as `list[list[str]]`, not `list[str]`.

**To resume**: restart runtime, run all cells 1–11 (fresh runtime, Cell 1 pulls the fix), proceed to 12–25. Cell 9 and Cell 10 are marked as skipped (print message + return). Analysis cells (11–25) handle missing cells automatically.

---

## Resolved blockers

### 1. Qwen-72B-AWQ — pcre / gptqmodel (Python 3.12) ✅

**Root cause**: gptqmodel's logger does `import pcre`. The PyPI package is `pypcre` (C extension over **libpcre2**, not libpcre3 — the prior `apt-get install libpcre3-dev` was the wrong system lib), with no Python 3.12 wheel.

**Fix**: Stub `pcre` against stdlib `re` before gptqmodel install. gptqmodel only uses `pcre.compile(r"\x1b\[[0-9;]*m")` + `.sub()` for ANSI escape stripping (verified in `gptqmodel/utils/logger.py`) — stdlib `re` handles it identically. The stub is bulletproof: no C extension, no system libs, no build.

**Cell 9 now does**:
1. Stub `pcre` with stdlib `re` (handlers for `compile/match/search/findall/sub/split/fullmatch` + flags)
2. `pip install -q --no-deps gptqmodel` (no transformers .py rewrite)
3. `pip install -q logbar` (pure-Python, safe mid-session)
4. `load_model(MODEL_ID, **KW)` — runs inference across 4 datasets

### 2. Llama-3.3-70B — OOM after Mistral-24B ✅

**Root cause**: 70B BNB 4-bit quantization peaks ~80 GB on A100. After any smaller model has been loaded and freed, the allocator carries historical fragmentation and the peak doesn't fit.

**Fix** (two-pronged):
1. Cell 1 now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before any torch import. The allocator can now reclaim physical pages across model boundaries.
2. Cell 10 still guards against the case where the runtime isn't fresh — checks `torch.cuda.max_memory_allocated()`. If > 5 GB, it refuses the load and prints the recovery procedure (restart runtime → run Cells 1–6 → skip 7/8/9 → run Cell 10). Drive checkpoints from already-completed cells reload automatically.

---

## Bugs fixed this session

| Bug | File | Status |
|-----|------|--------|
| `answers` list-of-list AttributeError in `lciteeval_grounding_label` | `spectral_utils/data_loaders.py` | ✅ Pushed (`8aa3587`) |
| `run_inference_for_cell` ran 500 iterations on 240-item dataset | Cell 6 hard-cap | ✅ Fixed in notebook |
| `pcre` missing from gptqmodel install | Cell 9 stdlib-re stub | ✅ Fixed in notebook |
| Llama-70B BNB OOM after Mistral-24B | Cell 1 `expandable_segments` + Cell 10 freshness guard | ✅ Fixed in notebook |

---

## Key rules / gotchas (updated)

- **gptqmodel on Python 3.12**: `import pcre` resolves to the PyPI `pypcre` package (C ext over libpcre2). Stub with stdlib `re` instead — gptqmodel only uses `pcre.compile()` + `.sub()` on an ANSI-escape pattern, and stdlib `re` handles it identically. The "use `types.ModuleType('pcre')` empty stub" path doesn't work because gptqmodel does `pcre.compile(...)` and gets `AttributeError` on empty modules.
- **70B BNB models**: with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in Cell 1, they can sometimes load after another model. To be safe, Cell 10 guards on `torch.cuda.max_memory_allocated()` and refuses the load if any prior model has used the GPU in this runtime. The user gets a clear recovery procedure printed.
- **AWQ models**: `load_model(model_id, quantize_4bit=False)` — auto-detects AWQ. Requires both `autoawq` AND `gptqmodel`.
- **L-CiteEval dataset sizes**: hotpotqa=240, NQ=160, 2wiki=240, narrativeqa=240. `N_SAMPLES=500` exceeds all of them — inference caps automatically at real size.
- **Never use `pip install git+...`** — use `git clone -b master` + `sys.path.insert`.
- **gptqmodel install order**: always `--no-deps`, always after `import datasets` has frozen pyarrow, never in Cell 1.

---

## Immediate next actions

1. **Run Phase 10 Main RAG end-to-end** — single fresh Colab session, in this order:
   - Restart runtime, then run Cells 1–8 (already-complete checkpoints reload from Drive; Cells 7/8 short-circuit fast since qwen7b + mistral24b are done)
   - Run Cell 9 → Qwen-72B-AWQ (pcre stub + gptqmodel install + load + inference on 4 datasets)
   - Then Cells 11–25 to produce results for 12/16 cells (qwen7b + mistral24b + qwen72b)

2. **Run Llama-70B in a second Colab session**:
   - Fresh runtime → Cells 1–6 (no model loads) → SKIP Cells 7/8/9 → Cell 10
   - Cell 10's guard ensures the load only attempts on a fresh runtime
   - Checkpoints persist to Drive; after this completes you have 16/16 cells

3. **Run Cells 11–25 once more** (either session, after both 70B models done) to produce the full 16-cell analysis: AUC heatmap, Nadler weight fingerprint matrix, fusion distributions, length-controlled bars, gates.

4. **After 16/16 cells complete**: update HISTORY.md Step 85 with headline numbers, update Research_Directions.md Direction 2 (RAG) status.

5. **Agent experiment** (Direction 4): separate notebook `Spectral_Analysis_Phase10_Agentic.ipynb` — ReAct loop on GPQA Diamond questions, spectral features per Thought step. Benchmark: AgentHallu (motivating benchmark); GPQA Diamond (actual dataset). Model: Qwen2.5-72B-AWQ (same as Phase 8). Not started yet.

---

## Email to advisors

Draft completed this session — ready to send. Key points:
1. Normalization added (−0.1 pp on GSM8K, negligible on long traces)
2. Nadler beats simple average by +1.7 pp; averaging beats best single feature by +0.3 pp
3. GPQA Diamond with Qwen-72B: accuracy 40% (expected 70%), AUC improved 65.4% → 69.0%
4. Factual QA negative result validates prior suspicion
5. Two experiments running: RAG (L-CiteEval) + Agent (GPQA Diamond ReAct loop)
6. Repo: https://github.com/omrisegev/hallucination_detection
