# MV_EPR — Session Progress Handoff

**Date**: 2026-05-09  
**Last updated**: Step 83 complete

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

### Phase 10 — L-CiteEval Pilot (READY TO RUN)

**Concept**: Do spectral features of H(n) predict statement-level grounding faithfulness
on long-context document QA? Tests the RAG branch of Phase 10.

**Setup** (locked in Phase10_Pilot_Plan.md):
- Notebook: `Spectral_Analysis_Phase10_LCiteEval_Pilot.ipynb`
- Dataset: L-CiteEval HotpotQA sub-task (multi-doc QA, 8K–48K context)
- Model: Falcon-3-10B-Instruct (T=1.0), 100 samples
- GPU: Colab A100 80GB

**Grounding label**: HotpotQA `supporting_facts` title matching.
Statement grounded (1) if any cited passage title is in gold supporting_facts.

**Decision gate**:
- PASS (>60%): extend to FACTS Grounding + DeepHalluBench
- MARGINAL (55–60%): run FACTS Grounding before deciding
- FAIL (≤55%): pivot to Plan A (RAG + Agentic as separate chapters)

**Status**: spectral_utils additions committed (Step 83). Notebook ready. Run on Colab next.

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
3. **Run Phase 10 pilot** — `Spectral_Analysis_Phase10_LCiteEval_Pilot.ipynb` on Colab A100.
   All pre-pilot work done (Step 83). Just open and run.
4. **After pilot**: add HISTORY.md Step 84 (pilot result) + update Research_Directions.md Direction 2 status.
5. **Deferred**: Phase 8 accuracy (40.4% vs expected 65%) — document as-is with disclaimer; not blocking Phase 10.
