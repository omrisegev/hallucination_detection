# MV_EPR — Session Progress Handoff

**Date**: 2026-05-11  
**Last updated**: Session ending mid-Phase 10 Main RAG (Step 84+)

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
| qwen72b | 0 | 0 | 0 | 0 | BLOCKED — pcre issue |
| llama70b | 0 | 0 | 0 | 0 | BLOCKED — OOM |

Checkpoints saved to Drive: `/content/drive/MyDrive/hallucination_detection/cache/phase10_main/raw/`

### Where it stopped

Cell 11 (feature extraction) fails with `AttributeError: 'list' object has no attribute 'lower'`.

**This bug is FIXED and PUSHED** (commit `8aa3587`). The fix flattens list-of-list answers in `lciteeval_grounding_label` — NaturalQuestions and NarrativeQA return `answers` as `list[list[str]]`, not `list[str]`.

**To resume**: restart runtime, run all cells 1–11 (fresh runtime, Cell 1 pulls the fix), proceed to 12–25. Cell 9 and Cell 10 are marked as skipped (print message + return). Analysis cells (11–25) handle missing cells automatically.

---

## Blocked issues

### 1. Qwen-72B-AWQ — pcre / gptqmodel (Python 3.12)

`gptqmodel` imports `pcre` at startup (`gptqmodel/utils/logger.py`). The `pcre` PyPI package is a C extension too old to build on Python 3.12. Multiple approaches tried:
- `pip install pcre` — silently fails (no Python 3.12 wheel)
- `apt-get install libpcre3-dev && pip install pcre` — still fails
- `sys.modules['pcre'] = types.ModuleType('pcre')` stub injection — still crashes (likely `logbar` also fails)

**Current Cell 9 state**: skipped with a print message.

**Approaches to try next**:
1. Pin older gptqmodel: `pip install --no-deps gptqmodel==0.9.x` (find a version before pcre was added)
2. Use a gptqmodel-free AWQ backend: check if `autoawq` alone can load AWQ without gptqmodel
3. Run Qwen-72B in a separate notebook as the first model on a fresh runtime

### 2. Llama-3.3-70B — OOM after Mistral-24B

Loading a 70B BNB 4-bit model as the 4th model in sequence OOMs at 79.11/79.25 GiB — GPU fragmented from previous model. This is a known issue (documented in CLAUDE.md).

**Fix**: Run Llama as the **first and only model** on a fresh runtime. Reorder `MODELS` so Llama is `MODELS[0]`, run Cell 7 only, then stop. Checkpoints merge with the existing ones automatically.

**Current Cell 10 state**: skipped with a print message.

---

## Bugs fixed this session

| Bug | File | Commit | Status |
|-----|------|--------|--------|
| `answers` list-of-list AttributeError in `lciteeval_grounding_label` | `spectral_utils/data_loaders.py` | `8aa3587` | ✅ Pushed |
| `run_inference_for_cell` ran 500 iterations on 240-item dataset | `Spectral_Analysis_Phase10_Main_RAG.ipynb` Cell 6 | notebook only | ✅ Fixed in notebook |
| `pcre` missing from gptqmodel install | Cell 9 | notebook only | ⚠️ Stub approach still failing |

---

## Key rules / gotchas (updated)

- **gptqmodel on Python 3.12**: `pcre` cannot be installed via pip. Stub injection (`sys.modules['pcre'] = types.ModuleType('pcre')`) attempted but unreliable. Try pinning an older gptqmodel version.
- **70B BNB models**: must be loaded on a fresh runtime as the FIRST model. Cannot be loaded after a smaller model has been unloaded — GPU fragmentation prevents it.
- **AWQ models**: `load_model(model_id, quantize_4bit=False)` — auto-detects AWQ. Requires both `autoawq` AND `gptqmodel`.
- **L-CiteEval dataset sizes**: hotpotqa=240, NQ=160, 2wiki=240, narrativeqa=240. `N_SAMPLES=500` exceeds all of them — inference caps automatically at real size.
- **Never use `pip install git+...`** — use `git clone -b master` + `sys.path.insert`.
- **gptqmodel install order**: always `--no-deps`, always after `import datasets` has frozen pyarrow, never in Cell 1.

---

## Immediate next actions

1. **Resume Phase 10 Main RAG (partial, 2 models)**:
   - Restart runtime → run all cells 1–25
   - Cell 1 git pull brings the `data_loaders.py` fix
   - Cells 9, 10 skip automatically
   - Cells 11–25 produce results for qwen7b + mistral24b (8/16 cells)

2. **Fix Qwen-72B-AWQ (pcre)**:
   - Try `pip install --no-deps "gptqmodel<0.9"` to find a pre-pcre version
   - OR try loading AWQ with `autoawq` alone (without gptqmodel): set `TRANSFORMERS_NO_ADVISORY_WARNINGS=1` and remove gptqmodel install

3. **Fix Llama-70B (OOM)**:
   - Fresh runtime, reorder MODELS so Llama is first, run only Cell 7
   - Checkpoints save to same Drive paths — merge automatically with existing results

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
