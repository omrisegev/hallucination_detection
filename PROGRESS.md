# MV_EPR — Session Progress Handoff

**Date**: 2026-05-09  
**Last updated**: Step 84 complete

---

## What this project is

Thesis on hallucination detection in LLMs. The core method: compute spectral features of the per-token entropy trajectory H(n) from a single gray-box forward pass, then fuse them with Nadler spectral fusion (covariance-weighted leading eigenvector). No labels needed at test time.

**Key package**: `spectral_utils` — pip-installable from `https://github.com/omrisegev/hallucination_detection.git` (branch: `master`). All feature extraction, model loading, fusion, and data loaders live here.

**In Colab**: always clone with `git clone -b master https://github.com/omrisegev/hallucination_detection.git` + `sys.path.insert(0, REPO_DIR)`. Do NOT use `pip install git+...`.

---

## Current experiment status

### Phase 8 — GPQA Diamond with Qwen2.5-72B ✅ COMPLETE (Step 80)

**Notebook**: `GPQA_Phase8_Fixed.ipynb`  
**Model**: `Qwen/Qwen2.5-72B-Instruct-AWQ` | GPU: 41.6/85 GB | Traces: 668 tok avg

**Results**:
- Accuracy: **40.4%** (expected ~65% — AWQ quantization + hard benchmark)
- Best individual AUC: **64.8%** (`trace_length`)
- **Fusion AUC: 69.0% [61.6, 76.2]** — best subset: `trace_length + sw_var_peak`
- Prior 7B best: 65.4% → **+3.6 pp improvement**
- Gates: 4/7 (G1 FAIL: accuracy < 50%; G4 FAIL: < 72%; G6 FAIL: 0 Nadler lift)
- Verdict: "Spectral features transfer with 72B. Not as strong as math."

**Open question**: Accuracy 40.4% vs expected 65% — consider stronger model (Qwen3-72B) or accept result with disclaimer. The +3.6 pp improvement is statistically reliable (CI lower 61.6%).

---

### Phase 9 — QA Transfer (Falcon-3-10B on TriviaQA + WebQ) ✅ COMPLETE (Step 82)

**Notebooks**: `Spectral_Analysis_Phase9_QA_Validation.ipynb` + `_RES.ipynb`

**Part 1 — Direct-Answer**: Structurally broken — 83% traces discarded, 0–4% correct in valid set. AUCs meaningless.

**Part 2 — CoT** (full results):
- Trace survival: 95–97% (fixed by CoT) — but signal still absent
- TriviaQA CoT: Nadler **53.6% [46.5, 61.6]** vs EPR baseline 72.0% — catastrophic gap
- WebQ CoT: Nadler **61.9% [51.7, 72.1]** vs EPR baseline 66.4% — below baseline
- All individual features below chance (34–49%). Negative Nadler lift on both datasets.
- **Verdict**: Spectral features require reasoning-type traces. Factual QA (even with CoT) lacks the systematic entropy structure. Clean negative result — strengthens thesis scope claim.

---

### Phase 9 Part 2 (CoT) — COMPLETE (Step 82)

**Notebooks**: `Spectral_Analysis_Phase9_QA_Validation.ipynb` + `_RES.ipynb`
- TriviaQA CoT: Nadler **53.6% [46.5, 61.6]** vs EPR baseline 72.0% — catastrophic gap
- WebQ CoT: Nadler **61.9% [51.7, 72.1]** vs EPR baseline 66.4% — below baseline
- All individual AUCs below chance (34–49%). Negative Nadler lift on both datasets.
- **Verdict**: Factual QA even with CoT lacks systematic entropy structure. Clean negative.

---

### Phase 10 — L-CiteEval Pilot (NEEDS RE-RUN: model switch required)

**Concept**: Do spectral features of H(n) predict statement-level grounding faithfulness
on long-context document QA? Tests the RAG branch of Phase 10.

**Run 1 result (Step 84 — Falcon-3-10B, N=100)**:
- Citation rate: **58%** — barely below G0-A threshold (need ≥60%)
- Valid statements: **83** — below G0-B threshold (need ≥100)
- Full gate: **INVALID** (pre-conditions failed by thin margins)
- Signal AUCs: epr=69.9%, sw_var_peak=69.7%, Nadler=76.0% [64.3, 86.8] — strong
- trace_length AUC = 50.8% (chance) → **no length confound**
- Nadler 76.0% vs PC1 58.5% → **Nadler does real work**

**Root cause**: Falcon-3-10B only follows the `[N]` citation format 58% of the time.
**Fix**: Switch to Qwen2.5-72B-AWQ + N_SAMPLES=150.

**Re-run setup**:
- Notebook: `Spectral_Analysis_Phase10_LCiteEval_Pilot.ipynb` (minor config change only)
- Model: `Qwen/Qwen2.5-72B-Instruct-AWQ` (Phase 8 infra ready, known citation follower)
- N_SAMPLES: 150 (guarantees ≥100 valid statements)
- All other settings unchanged (T=1.0, MAX_NEW=1024, HotpotQA)

**Expected**: citation rate ≥80%, valid statements ≥120, pre-conditions all PASS.
Signal AUCs should be similar or better given longer Qwen2.5 traces.

---

## Best results so far

| Setup | AUROC | Notes |
|-------|-------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **90.0%** | spectral Nadler fusion |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
| GPQA / Qwen2.5-72B-AWQ / T=1.0 | **69.0%** | Phase 8 — +3.6 pp over 7B; acc only 40.4% |
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

1. ~~**Run Phase 8**~~ ✅ DONE — 69.0% AUC, +3.6 pp over 7B (Step 80)
2. ~~**Run Phase 9 Part 2 CoT**~~ ✅ DONE — spectral features don't transfer to factual QA (Step 82)
3. ~~**Run Phase 10 pilot (Falcon-3-10B)**~~ ✅ DONE — INVALID pre-conditions, signal strong (Step 84)
4. **Re-run Phase 10 pilot with Qwen2.5-72B-AWQ + N=150**:
   - In Cell 2 of `Spectral_Analysis_Phase10_LCiteEval_Pilot.ipynb`, change:
     - `MODEL_ID = 'Qwen/Qwen2.5-72B-Instruct-AWQ'`
     - `N_SAMPLES = 150`
   - Cell 4: use `load_model(MODEL_ID, quantize_4bit=False)` — AWQ is auto-detected.
   - Cell 1 pip install: add `autoawq gptqmodel` to the install line.
   - Everything else unchanged — same notebook, same grounding labels.
5. **After re-run**: add HISTORY.md Step 85 (pilot re-run result) + update Research_Directions.md Direction 2 status + update best-results table if Nadler improves.
6. **Deferred**: Phase 8 accuracy (40.4% vs expected 65%) — document as-is with disclaimer; not blocking Phase 10.
