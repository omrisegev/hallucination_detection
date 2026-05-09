# Phase 10 — Pilot Plan: Spectral Features for Long-Form Grounded Generation

**Date**: 2026-05-08
**Status**: LOCKED. Ready to implement.
**Supersedes**: `Phase10_Long_Doc_RAG_Design.md` (pre-research design discussion).
**Reads with**: `research_phase10_rag/` (14 deep-research JSONs), `Research_Directions.md`, `CLAUDE.md`.

This is the handoff doc for the agent implementing the pilot. The Big Plan is the framing context; the Micro Plan is the exact next experiment.

---

## Big Plan — Spectral features for grounded generation

### Anchor result (the spectral chapter, math/reasoning)
Spectral features of the token-entropy trajectory `H(n)` reach **76–96.6% AUC** on long structured reasoning traces (MATH-500, GSM8K, GPQA). They **fail** on short factual QA traces (TriviaQA 53.6%, WebQ 61.9%, HotpotQA 59.5%). The mechanism is: methodical reasoning produces stable entropy "islands" with characteristic uncertainty spikes at decision points. `sw_var_peak` catches the local burst pattern.

**Scope claim (locked)**: spectral features of `H(n)` need long structured traces with step-level entropy modulation. Phase 10 extends this by showing that **grounded↔parametric transitions in long-doc generation produce the same kind of local entropy bursts**.

### Why this is novel (pre-emption verdict)
14 papers deep-researched in `research_phase10_rag/`. The closest concurrent works:
- **Energy Mountain** (arXiv:2510.19117) — spectral features on **depth axis** (per-layer attention-graph Laplacian). We're on **generation-time axis**. Different signal, different graph, different axis. Cite as concurrent work.
- **LapEigvals** (arXiv:2502.17598) — Laplacian eigenvalues of **static** attention adjacency. No time axis. Cite as the spectral-on-attention competitor.
- **SIA** (arXiv:2604.06192) — explicitly names entropy-spike-as-failure-signal as a corollary of its theoretical framework. **Strengthens** our framing rather than preempting it.
- **Streaming Prefix-Level** (arXiv:2601.02170) — aggregates entropy as a scalar; no FFT/Fourier/eigenvalues. Different signal.
- **TOHA** (arXiv:2504.10063) — topological (persistent homology) on attention graphs. Strong RAG result (HotpotQA 0.71–0.80) — must-beat baseline.

**Open niche**: gray-box, generation-time-axis, frequency-domain features of entropy trajectory, applied to long-doc grounded generation.

### Two domains, one chapter

| Setting | Trajectory | Output | Label | Phase |
|---|---|---|---|---|
| Long-doc RAG (cited) | 1 step | multi-statement+citations | per-statement faithfulness | **Pilot (L-CiteEval)** |
| Long-doc RAG (free-form) | 1 step | long answer | per-claim atomicity | Confirmation (FACTS Grounding) |
| Long-doc RAG (span) | 1 step | free-form | character span | Ablation (RAGTruth/++) |
| Multi-step agentic | N steps | per-step Thought + answer | per-step hallucination | Phase 10 main (DeepHalluBench) |

#### Domain 1 — RAG, defined for this project
**What it is**: long-context document QA where the model produces a multi-statement output, each statement either grounded in retrieved context or fabricated/inferred from parametric memory.

**How it generates long reasoning**: long input context (8K–48K tokens) forces the model to extract, synthesize, cite — producing 200–2000-token outputs. Each output statement is a mini reasoning step.

**Spectral hypothesis (RAG)**: per-statement spectral signature of `H(n)` discriminates grounded from fabricated statements. `sw_var_peak` peaks at grounded↔parametric transitions; `spectral_centroid` shifts higher for fabrication.

#### Domain 2 — Agentic, defined for this project
**What it is**: multi-step trajectory where the model alternates Thought → Action → Observation. Each Thought is itself a long reasoning trace; the cumulative trajectory is the concatenation. Hallucination can occur at any step.

**How it generates long reasoning**: ReAct or DRA loop (read → search → reason → search → reason → answer). Each Thought is a math-like reasoning trace (where our method already works).

**Spectral hypothesis (Agentic)**: spectral features per Thought step predict step-level hallucination. Cumulative trajectory features (running `sw_var_peak` across steps) detect cascading failure ("Spiral of Hallucination").

### Comparison targets

**RAG (for the pilot and Phase 10 RAG sub-chapter)**:

| Method | Their result | Our target |
|---|---|---|
| TOHA | HotpotQA 0.71–0.80 | beat by ≥3pp on L-CiteEval HotpotQA |
| Semantic Entropy | ~0.71 long-form | beat by ≥5pp (sanity baseline) |
| Real-Time Entity Probes | LongFact 0.90 | thesis comparison (not pilot scope) |
| LapEigvals | TriviaQA 0.83–0.89 | "different axis" framing in writeup |

**Agentic (for Phase 10 main)**:

| Method | Their result | Our target |
|---|---|---|
| AUQ (Salesforce) | ALFWorld Φmin 0.791 | beat at agentic Φmin |
| AgentHallu SOTA (Gemini 2.5 Pro) | 41.1% step localization | beat at step-level |
| SAUP | +20% AUROC over baselines | beat as trajectory-aware comparison |
| Streaming Prefix-Level | Qwen-7B MATH 0.81 | "different signal" framing (scalar entropy vs spectrum) |

---

## Micro Plan — L-CiteEval Pilot

### Goal
Determine in **1 day on Colab A100** whether spectral features of `H(n)` predict statement-level grounding faithfulness on long-context document QA. Result decides whether Phase 10 proceeds (Plan B unified) or pivots (Plan A separate chapters).

### Setup (locked)
- **Notebook**: `Spectral_Analysis_Phase10_LCiteEval_Pilot.ipynb` (new)
- **Dataset**: L-CiteEval, **HotpotQA sub-task only** (multi-doc QA, 8K–48K context)
- **Model**: **Falcon-3-10B-Instruct** (already integrated; T=1.0)
- **Samples**: **100** (gives ±5% SE on individual-feature AUC)
- **GPU**: single Colab A100 80GB

### Pre-pilot work (~1 hour, local before Colab)
1. **Add L-CiteEval data loader** to `spectral_utils/data_loaders.py`.
   - Function: `load_lciteeval(task='hotpotqa', n=100)` → list of `{question, contexts, gold_answer, gold_citations}`
   - Source: `https://huggingface.co/datasets/Jonaszky123/L-CiteEval` ; code mirror `https://github.com/LCM-Lab/L-CITEEVAL`
2. **Add citation-grounded prompt template** to `spectral_utils/data_loaders.py`.
   - Format: "Read the following passages. Answer the question. Format your answer as a sequence of statements, each followed by a citation [n] to the passage that supports it."
   - One in-context example to anchor the format.
3. **Add statement-segmentation helper** to `spectral_utils/feature_utils.py`:
   - Function: `segment_by_citations(text, token_offsets) -> List[(statement_text, token_start, token_end, citation_index)]`
   - Regex on `\[(\d+)\]` markers + token-position tracking from the tokenizer's offset mapping.
4. **Commit and push** to repo so Colab clones it.

### Pipeline (notebook)

Standard cell sequence per `CLAUDE.md`:

1. **Cell 1 — Setup**: standard Colab clone + pip install + `spectral_utils` import (per CLAUDE.md template). Add `pip install datasets` for L-CiteEval HF loader.
2. **Cell 2 — Config**: `MODEL_ID='tiiuae/Falcon3-10B-Instruct'`, `T=1.0`, `MAX_NEW=1024`, `N_SAMPLES=100`, `TASK='hotpotqa'`, cache dir on Drive.
3. **Cell 3 — Mount Drive** + create cache dirs.
4. **Cell 4 — Load model** via `load_model(MODEL_ID, quantize_4bit=False)`.
5. **Cell 5 — Load data** via new `load_lciteeval('hotpotqa', n=100)`.
6. **Cell 6 — Inference loop with checkpointing**: for each sample, build prompt with passages + question, call `generate_full(model, tok, prompt, T=1.0, max_new=1024)`. Capture tokens + per-token entropies. Save every 25 samples.
7. **Cell 7 — Unload model**: `del mdl, tok; free_memory()`.
8. **Cell 8 — Statement segmentation**: for each generation, segment by citations using new helper. Get `(statement_text, token_start, token_end, citation_index)` per statement.
9. **Cell 9 — Per-statement features**: for each statement, slice `H(n)` to `[token_start:token_end]` and call `extract_all_features()`. Compute `sw_var_peak`, `sw_var_peak_adaptive`, `spectral_centroid`, `spectral_entropy`, `epr`, `trace_length`, `dominant_freq`, `stft_max_high_power`. Filter statements with trace length < 10 tokens (FFT minimum).
10. **Cell 10 — Ground truth per statement**: use L-CiteEval's automated scorer (CR — Citation Recall) to label each statement as grounded (1) or ungrounded (0). Vendor the scorer logic into the notebook if needed.
11. **Cell 11 — Statement-level AUC table** (individual features): bootstrap 95% CIs.
12. **Cell 12 — Nadler fusion**: `best_nadler_on(features_dict, labels, max_size=4, min_size=3)` with z-score normalization. Apply correlation filter |ρ| ≥ 0.75.
13. **Cell 13 — Baseline**: Semantic Entropy via k=10 samples per statement (Farquhar 2024 protocol).
14. **Cell 14 — Visual diagnostic**: plot `H(n)` overlaid for 5 grounded vs 5 ungrounded statements (random sample). Eyeball whether shapes are characteristically different.
15. **Cell 15 — Spearman heatmap** across all features (Nadler decorrelation check).
16. **Cell 16 — Decision gate**: print pass/fail per below.
17. **Cell 17 — Save**: `pilot_results.pkl` to Drive cache.
18. **Cell 18 — Plots** saved to disk: feature AUC bar, H(n) traces by class, Spearman heatmap.

### Decision gate

| Outcome | Threshold | Next action |
|---|---|---|
| **Pass** | any individual feature AUC > 60% **AND** beats Semantic Entropy by ≥ 3pp | build Phase 10 main: extend to FACTS Grounding + DeepHalluBench |
| **Marginal** | best AUC 55–60% | run on FACTS Grounding (100 samples) before deciding |
| **Fail** | best AUC ≤ 55% | confidence masking dominates → pivot to Plan A: RAG and Agentic as separate chapters, no unified Phase 10 |

### Risks + contingencies
- **Statement boundaries are fuzzy** (model produces statements without citation markers): fallback to fixed-length 50-token windows. Add cell that runs both segmentations and compares.
- **Falcon-3-10B doesn't follow citation format** (<60% of outputs have parseable citations): switch to Qwen2.5-72B-AWQ (Phase 8 infra is ready). This is cheap to try if Falcon fails.
- **L-CiteEval scorer disagrees with intuition**: spot-check 10 statements manually before trusting headline AUC.
- **Per-statement traces too short for FFT** (<10 tokens): `sw_var_peak_adaptive` should handle this; report fraction filtered.

### Outputs
- `Spectral_Analysis_Phase10_LCiteEval_Pilot.ipynb` (notebook with results)
- `pilot_results.pkl` in Drive cache
- 3 plots saved to repo: feature AUC bar, H(n) traces by class, Spearman heatmap
- HISTORY.md step (post-pilot) — append per `CLAUDE.md` format
- Verdict in PROGRESS.md (post-pilot)

---

## Documentation owed (to be done after pilot)
- HISTORY.md Step 83: Phase 6 HotpotQA results (logged retroactively — `Spectral_Analysis_Phase6.ipynb` ran but never got a HISTORY entry)
- PROGRESS.md: add Phase 6 to current experiment status
- HISTORY.md Step 84: Phase 10 pilot result
- PROGRESS.md: update with Phase 10 status + decision verdict
- Research_Directions.md: update Direction 7 (Phase 6 done), Direction 2 (folded into Phase 10 if pilot passes), Direction 4 (Phase 10 design locked)

---

## Pointers for the implementing agent

- **Per-paper findings**: `research_phase10_rag/*.json` (14 files). Most relevant: `lapeigvals.json`, `toha.json`, `realtime_hallucinated_entity_probes.json`, `streaming_prefix_level_hallucination_detection.json` for baseline comparisons.
- **Existing infra**:
  - `spectral_utils/feature_utils.py` — `extract_all_features`, `sw_var_peak_with_window`, `sw_var_peak_adaptive`
  - `spectral_utils/fusion_utils.py` — `best_nadler_on`, `zscore`, `boot_auc`
  - `spectral_utils/model_utils.py` — `load_model`, `generate_full`, `token_entropies_from_scores`, `free_memory`
  - `spectral_utils/data_loaders.py` — needs new `load_lciteeval`
  - `spectral_utils/io_utils.py` — `load_cache`, `save_cache`
- **Project rules** (from CLAUDE.md):
  - Never inline helpers in notebooks. Add to `spectral_utils` and commit *before* using.
  - Never use `pip install git+...` — clone via `git clone -b master`.
  - Falcon-3-10B is in BNB territory? No — it's smaller and loads via standard transformers. Use default `load_model(MODEL_ID, quantize_4bit=False)`.
  - Nadler requires ≥3 views; z-score on; ρ-filter at 0.75; orient features so higher score = more likely correct.
- **Validator note**: `validate_json.py` has a fields.yaml format incompatibility; not blocking, just don't trust validation output until fixed.

---

## Decision history (compact summary for context)

- Phase 6 (HotpotQA / Mistral-7B / single-shot) ran 2026-05-07: 59.5% AUC, 6/7 gates failed. Multi-hop QA confirmed outside spectral scope at single-shot level. Untested: long-context multi-hop with citations (which is the L-CiteEval pilot).
- Phase 9 (TriviaQA + WebQ CoT / Falcon-3-10B) ran: 53.6% / 61.9%. Confirmed factual QA out of scope.
- Pre-emption check (this session, 14 papers deep-researched): closest concurrent works operate on different axes; spectral-trajectory-on-generation-time framing is open.
- Pilot decision: L-CiteEval HotpotQA, 100 samples, Falcon-3-10B, single experiment. Decision gate dictates Plan B (unified Phase 10) vs Plan A (separate chapters).
