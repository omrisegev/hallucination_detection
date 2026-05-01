# Research Directions — Thesis Roadmap
*Omri Segev | Supervised by Bracha Laufer-Goldshtein & Ofir Lindenbaum*

---

## How to Read This Document

Each direction is assessed on four axes:
- **Feasibility** — how much new infrastructure is needed (Low / Medium / High effort)
- **Novelty** — how much this departs from existing literature
- **Supervisor alignment** — how directly this connects to Bracha's or Ofir's core work
- **Risk** — probability that the main hypothesis fails

The directions are ordered from most to least connected to the current experimental thread.

---

## Direction 1 — LLM Hallucination Detection: Extending the Current Framework

**Status**: Active. Best ensemble: 81.5% TriviaQA / 76.0% WebQ (6-view temp+behavioral, Step 29). CoT signals characterized (Step 45). Next: merge CoT into 6-view ensemble (Experiment 1E).
**Results notebook**: `Unified_EPR_Ensemble_res.ipynb` (Step 45) — CoT signals fully characterized across TriviaQA, WebQ, GSM8K.

### Core Hypothesis
The 6-view Nadler ensemble (T=0.3/1.0/1.5/2.0 + Verify + Skeptic) can be further improved by adding process-level signals extracted from Chain-of-Thought reasoning traces. EPR on the reasoning trace is decorrelated from EPR on the final answer because they measure different computational phases: *stochastic instability of reasoning* vs *surprisal of the final token state*.

### Connection to Supervisors
- **Bracha**: LTT / conformal prediction can calibrate the Nadler fusion score into a threshold with a distribution-free false-negative rate guarantee — a direct application of her COIN framework and LTT work
- **Ofir**: VSDE's density-stability principle maps onto EPR: both ask not "what is the probability?" but "how stable is the prediction?" The VSDE ensemble-of-permuted-models idea parallels the temperature-variation approach

### What We Already Have
- Working 6-view ensemble: all-6 (4 temps + Verify + Skeptic) = +2.4% TriviaQA, +4.2% WebQ
- Nadler two-condition diagnostic fully understood
- Checkpoints saved for Falcon-3-10B on TriviaQA + WebQ

### Proposed Experiments (in order)

**Experiment 1A — CoT trace EPR as a new Nadler view** ✅ COMPLETED (Step 45, `Unified_EPR_Ensemble_res.ipynb`)

*History: invalid run (Step 43) → rebuilt clean notebook (Step 44) → clean run (Step 45)*

**Results from `Unified_EPR_Ensemble_res.ipynb`:**

| Signal | TriviaQA | WebQ | GSM8K |
|--------|---------|------|-------|
| EPR(direct_fresh) — reference | **72.0%** | **66.4%** | 57.8% |
| EPR(trace) | 70.2% | 65.7% | **66.8%** |
| EPR(answer) | 63.9% | 63.8% | 59.5% |
| EDIS | 61.2% | 57.5% | 66.2% |
| **Best Nadler fusion** | **70.7%** (trace+ans) | **67.0%** (trace+ans) | **68.7%** (trace+EDIS) |
| vs EPR(direct) | −1.3% | **+0.7%** | **+11.0%** |

**Spearman ρ diagnostics:**
- ρ(trace, EDIS): 0.695 (TriviaQA), 0.700 (WebQ), 0.799 (GSM8K — above threshold, reduces benefit)
- ρ(trace, answer): 0.28–0.39 range (well below threshold — both should co-fuse)
- ρ(direct, trace): ~0.374 (decorrelated → trace adds new information over temperature views)

**Five key findings:**
1. **Confidence masking** (factual QA): EPR(trace) < EPR(direct) — CoT smooths entropy, direct generation is stronger single signal on factual QA
2. **Math inversion**: EPR(trace) >> EPR(direct) on GSM8K (66.8% vs 57.8%) — the trace IS the only usable signal when direct=2% accuracy
3. **EPR(answer) viable**: hybrid split yields std=0.467, AUC 63–64% on factual QA — wrong answers have 59% higher answer-token entropy than correct
4. **ρ(trace, EDIS) borderline**: 0.695–0.700 on factual QA (just below 0.75), 0.799 on GSM8K (above — limits Nadler on math)
5. **EDIS domain-dependence confirmed**: gap vs EPR(trace) is −8.9%/−8.2% on factual QA, only −0.7% on math — EDIS competitive only when trajectories have deep reasoning structure

**Key takeaway**: CoT signals alone don't beat the 6-view temperature+behavioral ensemble (81.5%/76.0%). The open question is whether **EPR(trace) adds orthogonal lift on top of those 6 views** — since ρ(trace, EPR_direct) ≈ 0.374 and trace measures a different computational phase. That is Experiment 1E (next).

**Marker compliance note**: Falcon-3-10B outputs "Answer:" in 0–2% of samples. Hybrid fallback (last 25% of tokens) is always the active path. Design is validated — EPR(answer) is non-constant and discriminative despite 100% fallback usage.

**Experiment 1B — EDIS (Entropy Dynamics Instability Score) as a view** ✅ INCLUDED IN 1A
- **Source**: **Zhu et al. (2026), "EDIS: Diagnosing LLM Reasoning via Entropy Dynamics"** (arXiv:2602.01288). Real paper, verified. Authors: Chenghua Zhu, Siyan Wu et al., South China Normal University / Sun Yat-sen University.
- **Correct formula** (Eq. 7): `EDIS(H) = S(H) × (1 + Var(H))` where `S(H) = ½(S_burst + S_rebound)`
  - `S_burst = Σ I(H_{t+w} − H_t > τ_b)` — windows with cumulative growth above threshold
  - `S_rebound = Σ I(H_t − min_{s<t} H_s > τ_r)` — positions exceeding running historical minimum
  - `Var(H)` — full trajectory variance (multiplicative amplifier)
- **Our original implementation was wrong** — used additive formula with arbitrary coefficients (0.05, 0.02) and approximated both components incorrectly. **Fixed in Step 35** to use the exact paper formula with `tau_b=tau_r=0.5` as default.
- **Critical caveats for our thesis**:
  - Paper validates exclusively on **math reasoning** (GSM8K, MATH, AMC23, AIME24) — never on factual QA. Transfer to TriviaQA/WebQ is an open question and a potential thesis contribution.
  - Paper's use case is **Best-of-N selection** (rank N candidates). Our use case is **single-sample binary detection**. Different task — need to validate EDIS works here.
  - Paper reports EDIS AUC = **0.804** vs mean entropy AUC = 0.673 on math. On factual QA this gap may be smaller.
  - Spearman ρ(EDIS, mean entropy) = **0.66** on math — borderline for Nadler co-inclusion (our threshold ~0.75). Must verify on our data.
  - **τ_b and τ_r require calibration** per model family — the paper explicitly warns these vary. Paper Appendix E gives τ_b=1.36, τ_r=1.33 (validated on Qwen2.5-Math). Our `EDIS_Replication.ipynb` will confirm these, then Cell 9 grid search finds optimal values for Falcon-3-10B.
- EDIS and EPR(trace) are theoretically orthogonal: EPR measures mean surprisal, EDIS measures trajectory dynamics (specifically the pathological burst/rebound patterns that EPR averages away)
- Will be evaluated as a standalone view and as a Nadler fusion member in the same notebook

**Experiment 1B-pre — EDIS replication on math** ✅ COMPLETED (partial replication, Steps 41–42)

*Run 1 (Step 41) — broken grading:*
- `EDIS_Replication_res.ipynb` (first run): accuracy 3–5% due to `extract_gsm8k_answer` missing `\boxed{}` format used by Qwen Instruct. All AUC results invalid. Only the spike ratio (3.34×, within paper's 1.7–3.6×) was valid — confirms formula correctness.

*Run 2 (Step 42) — grading fixed:*
- Fixed `extract_gsm8k_answer` (added `\boxed{}` as first pattern); re-graded cached answers via new Cell 4b (no re-inference). 658/800 labels changed.
- New results:

| Metric | Paper | Fixed run | Status |
|--------|-------|-----------|--------|
| Accuracy T=0.6 | ~60–70% | **84.5%** | Over-high |
| EDIS AUC pooled | 0.804 | **0.601** | ✗ FAIL |
| Mean-H AUC pooled | 0.673 | **0.604** | close |
| EDIS gap over Mean-H | +13.1 pp | **−0.3 pp** | ✗ FAIL |
| Spike ratio | 1.7–3.6× | **4.02×** | ✓ PASS |

- **Root cause of remaining gap**: model accuracy too high (85%). Paper tested at ~60–70% accuracy. At 85%, only ~120 wrong answers in 800 samples → too few negatives for EDIS to demonstrate its +13 pp advantage.
- **Thesis interpretation**: EDIS advantage over mean-H is **accuracy-regime dependent** — manifests at moderate accuracy (~60–70%), not near-ceiling (85%). On our factual QA (TriviaQA acc=51%, WebQ acc=38.5%), EDIS achieves 65.3% / 61.5% with correct labels — this is the regime we care about.
- **τ values to use**: keep τ_b=1.36, τ_r=1.33 (paper Appendix E). Grid search best τ_b=0.1, τ_r=2.0 is noise-optimized at 85% accuracy — do not use.
- **EDIS formula is validated** by spike ratio (4.02×). Formula is correct. Use in Unified_EPR_Ensemble.ipynb.

**Post-run optional checks (add ONLY after clean CoT run confirms ρ diagnostics)**

The following signals were suggested by NotebookLM and confirmed as real papers, but all come from reasoning models doing math with 2,000–10,000 token traces. Their transfer to Falcon-3-10B doing factual QA (50–200 token CoT) is uncertain. Do not implement before seeing the Spearman ρ results from the clean run.

- **RPDI**: `mean(H[-W:]) / mean(H)` — 1 line. Check ρ(RPDI, EPR_trace) after run. Include if ρ < 0.5, skip if ρ > 0.8.
- **Trace Length**: `len(trace_tokens)` — 1 line. Check variance first; likely low on short factual QA traces. The SELFDOUBT paper itself warns it only works on intermediate-difficulty benchmarks.
- **HVR**: requires unsupervised per-model marker discovery (90 unlabeled traces + embedding pipeline). NOT a simple fixed regex. Check 10–20 Falcon CoT traces for hedging language first. If Falcon doesn't hedge, skip entirely.

**Citations (theory only, no new code)**

- **Detection-Extraction Gap (Wang & Zhu 2026, arXiv:2604.06613)**: 52–88% of reasoning-model tokens are post-commitment. Cite as motivation for our trace/answer EPR split. Not applicable as a Nadler view on short 50–200 token factual QA traces.
- **DiffAdapt (Liu et al., ICLR 2026, arXiv:2510.19669)**: U-shaped entropy vs. difficulty. Means high EPR ≠ hallucinating on easy questions — mean EPR is non-monotone with correctness. Cite as motivation for trajectory-sensitive signals (EDIS, RPDI) over mean EPR. Does NOT directly validate EDIS (different phenomenon).

**Experiment 1E — Add EPR(trace) + EPR(answer) to the 6-view temperature+behavioral ensemble** ← NEXT PRIORITY

*Motivation*: The 6-view ensemble (T=0.3/1.0/1.5/2.0 + Verify + Skeptic) achieves 81.5% TriviaQA / 76.0% WebQ (Step 29). EPR(trace) is decorrelated from any single-temperature EPR (ρ≈0.374) because it measures a genuinely different computational phase: stochastic instability of reasoning vs surprisal at the generation temperature. Adding it as view 7 should satisfy both Nadler conditions.

*Design*:
- Load T=1.5 inference from `epr_unified_experiment/triviaqa/` and `epr_unified_experiment/webq/` (already on Drive from Step 44–45)
- Load 6-view consolidated data from `epr_temp_varied/{dataset}/Falcon-3-10B/` (Steps 27–29)
- Merge on sample index → 8-view array: [T0.3, T1.0, T1.5, T2.0, Verify, Skeptic, trace, answer]
- Run Nadler fusion over all subsets ≥3 that include at least one CoT view
- Key Nadler condition check: verify ρ(trace, each temperature view) < 0.75 and ρ(answer, each view) < 0.75
- Exclude EPR(direct_fresh) from fusion (different answer format → different correctness labels, common-target violation)

*Expected outcome*: If ρ(trace, T=1.5) is low (different forward pass, reasoning expansion vs direct answer), expect +1–3% lift over the current 81.5% ceiling on TriviaQA.

*Implementation note*: No new inference needed. All data already on Drive. Only need a new consolidation + fusion notebook cell.

**Experiment 1C-original — Step-boundary branching entropy**
- Identify first token after each reasoning step delimiter (\n\n, "Therefore,", "Step k:")
- Extract entropy of predictive distribution at each branch point
- Mean/max over steps as a scalar view
- Very cheap: single forward pass, no segmentation model needed

**Experiment 1D — Conformal threshold calibration (Bracha's LTT)**
- Take the Nadler fusion score from the best ensemble
- Apply Learn-then-Test to find a threshold τ such that: P(false negative rate ≤ α) ≥ 1-δ
- Evaluate on held-out split with distribution-free guarantee
- This transforms the ensemble from an AUC metric to a deployable detector with formal guarantees

### Expected Outcome
**Completed so far**: 1B-pre ✅, 1A ✅. Key numbers: best 6-view=81.5%/76.0% (temp+behavioral), best CoT-only=70.7%/67.0% (trace+answer), best cross-domain=+11% on GSM8K math.

**Next**: 1E is highest priority — merge CoT signals into the 6-view ensemble. No new inference needed, only a consolidation cell. If ρ(trace, T_views) < 0.75 (expected given different forward pass), lifting WebQ above +5% becomes realistic. 1D is the natural theory chapter once we have a stable best ensemble.

### Feasibility: Low effort | Novelty: Medium | Risk: Low (trace-EPR computation is straightforward)

---

## Direction 2 — RAG Hallucination Detection

**Status**: Not started. Promoted to second priority after re-evaluation (Step 39). TriviaQA already has Wikipedia passages — lowest infrastructure cost of any unexplored direction.

### Core Hypothesis
In RAG systems, hallucination has two distinct failure modes: (a) *intrinsic* — the model ignores or contradicts the retrieved context; (b) *extrinsic* — the model answers from parametric memory instead of grounding in the retrieved passage. EPR on grounded tokens (those supported by the retrieved context) should be decorrelated from EPR on the parametric tokens, providing two genuinely independent Nadler views that predict the same factual correctness label.

A secondary hypothesis: EPR is *lower* on grounded tokens (model is more certain when reading from context) and *higher* on parametric tokens (model is uncertain when relying on memory). The contrast between these two is itself a hallucination signal.

### Connection to Supervisors
- **Bracha**: Her conformal prediction work extends naturally to RAG attribution — setting risk bounds on whether a specific claim is grounded in the retrieved document. Her semi-supervised PPI framework (Prediction-Powered Inference) could use the retrieved document as a proxy label source
- **Ofir**: VSDE's anomaly principle maps perfectly: a grounded answer is "in-distribution" (model stays on the document manifold); a hallucinated one is "out-of-distribution" (model drifts to parametric memory). PRAE's robust reconstruction objective could identify tokens that deviate from the retrieved context's latent space

### Proposed Experiments

**Experiment 2A — Grounded vs ungrounded EPR contrast**
- Dataset: TriviaQA with retrieved Wikipedia passages (already part of the dataset)
- For each question: compute EPR with context (model reads the passage) vs without context (parametric generation)
- Measure: Spearman ρ between EPR(with context) and EPR(without context)
- Hypothesis: the contrast score EPR(no context) − EPR(with context) is itself a hallucination predictor — larger contrast = model is less certain without support = more likely to hallucinate

**Experiment 2B — Faithfulness signal via context-conditional Verify**
- Modify the Verify prompt: "Given the following passage: [context]. Is the answer correct?"
- Compare P(Yes | context) vs P(Yes | no context) — the gap is a faithfulness signal
- Test as a new Nadler view alongside temperature-varied EPR

**Experiment 2C — Nadler fusion of EPR(context) + EPR(no-context) + faithfulness gap**
- Three views, all predicting the same correctness label
- Both EPR views share the common target; the faithfulness gap is a behavioral view
- Full fusion evaluation vs single-view EPR baseline

**Experiment 2D — RAG-specific conformal calibration**
- Risk-control with retrieval-based pseudo-labels: if the answer string appears verbatim in the retrieved passage, label it as grounded (Bracha's PPI approach)
- This enables semi-supervised risk control without any judge model

### Key Open Question
Does the Nadler framework require re-validation when the context changes between views (i.e., EPR(with context) and EPR(without context) are computed with different inputs)? The common-target condition must hold — the correctness label is the same regardless of whether context was used.

### Feasibility: Low-Medium effort | Novelty: High | Risk: Medium
The main risk is that EPR(with context) is uniformly lower regardless of correctness — if the model always becomes more confident when given a passage, the signal may be uninformative.

---

## Direction 3 — Visual Language Model (VLM) Hallucination Detection

**Status**: Not started. Conceptually clean extension of the current framework to multimodal models.

### Core Hypothesis
In VLMs (e.g., LLaVA, InternVL, Qwen-VL), hallucination has a modality-specific structure: the model may be uncertain about the *visual* grounding (what objects are present) while being confident about the *textual* narration (how to describe them fluently). EPR computed on visual-description tokens should be decorrelated from EPR on factual-claim tokens, because they tap different attention pathways.

Additionally, the Verify/Skeptic behavioral view can be adapted to VLMs: "Is the above description of the image accurate? Answer Yes or No."

### Connection to Supervisors
- **Ofir**: His multi-view kernel consensus and COPER (multi-view clustering) frameworks were designed for multi-modal data. STG (Stochastic Gates) could select which visual features/tokens contribute most to uncertainty. His diffusion maps work with heterogeneous feature types is directly applicable to fusing visual and textual uncertainty signals
- **Bracha**: Her conformal prediction work has been extended to multi-dimensional prediction sets — relevant for VLM outputs where the answer may involve spatial coordinates, object counts, or free-form descriptions

### Proposed Experiments

**Experiment 3A — EPR on visual description tokens vs factual claim tokens**
- Dataset: POPE (hallucination benchmark for VLMs) or HallusionBench
- For a VLM answer like "The red car is parked in front of a *building*": split tokens into visual-descriptive ("red car", "parked") and factual-claim ("building", location claims)
- Compute EPR separately for each segment
- Measure Spearman ρ between visual-EPR and factual-EPR
- Expected: lower correlation than prompt-template variations, higher correlation than temperature variations

**Experiment 3B — Multimodal Verify view**
- Present the image + "Does the generated description contain visual errors?" prompt
- P(Yes) from first token as a behavioral view — measures visual consistency
- Test as a Nadler view alongside token-level EPR views

**Experiment 3C — Temperature-varied EPR on VLMs**
- Replicate the temperature variation experiment (T=0.3/1.0/1.5/2.0) on a VLM
- Hypothesis: the decorrelation pattern (ρ=0.38-0.75) observed for text-only LLMs should hold for VLMs, since the temperature scaling applies to the same generation mechanism

**Experiment 3D — Full Nadler fusion on VLMs**
- Combine visual-EPR + factual-EPR + temperature-varied EPR + visual Verify
- Benchmark against single-view EPR on POPE/HallusionBench

### Key Challenge
VLMs require more GPU memory (a 7B VLM needs more RAM than Falcon-3-10B text-only). Qwen2.5-VL-7B or LLaVA-7B would fit on a single A100 with float16.

### Feasibility: Medium effort | Novelty: High | Risk: Medium-High
The risk is that visual and textual EPR are actually highly correlated because the VLM processes them in the same transformer stack.

---

## Direction 4 — Agentic Flow Hallucination Detection

**Status**: Research plan complete. Not yet implemented. Prerequisites: CoT run complete, Qwen3-7B access confirmed.

### Prior Art — AUQ Framework (Zhang et al. 2026, arXiv:2601.15703, Salesforce AI)

AUQ is the complete prior-art baseline for this direction. It must be understood before planning experiments.

- **System 1 (UAM)**: at every step, model outputs `action + confidence c_hat ∈ [0,1] + explanation e_hat`. Stored in memory; propagates uncertainty forward via attention.
- **System 2 (UAR)**: triggered when `c_hat < τ`. Best-of-N reflection using `e_hat` as diagnostic cue. Memory expansion if reflection fails.
- Training-free. Results: ALFWorld +10.7% SR, WebShop +13.6% SR over ReAct. SOTA on DeepResearch Bench.
- **Trajectory metrics**: Φmin (weakest link) AUROC = 0.791 on ALFWorld.
- **Limitation**: verbalized confidence degrades in models < 7B parameters.
- **What AUQ does NOT do**: no token-level EPR, no Nadler fusion, no formal calibration guarantee (τ set empirically).

### Core Hypothesis (Thesis Contribution)

AUQ uses only verbalized confidence — one modality. EPR is logit-based — a different modality. These two signals measure orthogonal aspects of step uncertainty (cognitive self-assessment vs. token-level surprisal). If Spearman ρ(EPR_step, verbalized_conf) < 0.5, Nadler fusion of the two will outperform AUQ alone. Additionally, AUQ's empirical τ threshold can be replaced with a LTT-calibrated threshold giving a formal guarantee on undetected trajectory failure rate.

### Model and Domain

**Model**: Qwen3-7B (or DeepSeek-R1-Distill-Qwen-7B) — above AUQ's 7B verbalized-confidence threshold; reasoning model so RPDI, HVR, EDIS all apply naturally on its long traces.

**Benchmark**: Multi-hop factual QA (HotpotQA or MuSiQue) rather than ALFWorld/WebShop.
- Same domain as current experiments, gold string matching works, no external environment simulator
- 3-step agent chain: retrieve fact 1 → reason → retrieve fact 2 → reason → synthesize answer
- Naturally produces a trajectory where step-1 errors propagate to the final answer
- Available on HuggingFace, no special access needed

### Connection to Supervisors
- **Bracha**: LTT conformal calibration of the trajectory score → formal guarantee on undetected failure rate. Direct application of her core work. AUQ uses no such guarantee; this is the thesis-specific addition.
- **Ofir**: Spiral of Hallucination = uncertainty propagation in sequential decision-making. Aligns with his VSDE stability principle and UProp framework.

### Proposed Experiments

**Experiment 4A — AUQ baseline on HotpotQA/MuSiQue**
- Model: Qwen3-7B | Dataset: HotpotQA 200 samples | 3-step agent chain
- Implement AUQ System 1 (UAM): append confidence + concern to each step prompt
- Evaluate: Φlast, Φavg, Φmin AUROC (same metrics as AUQ paper), task success rate
- This confirms verbalized confidence works on our model and domain before adding EPR

**Experiment 4B — EPR as a parallel per-step signal**
- Extract token entropy at each step (same `generate_with_entropies()` function, called per step)
- Compute EPR(step), RPDI(step) from the step's trace entropy array
- Check Spearman ρ(EPR_step, verbalized_conf) across all steps and samples
- **Decision gate**: if ρ < 0.5, fusion is viable; if ρ > 0.7, EPR adds no diversity

**Experiment 4C — Nadler fusion of EPR + verbalized confidence**
- Fuse step-level EPR and verbalized confidence vectors via Nadler (same `jaffa_nadler_estimation`)
- Compare: EPR-only trajectory, verbalized-only (AUQ), Nadler-fused
- Primary metric: Φmin AUROC — target > 0.791 (AUQ paper's best)
- Trajectory aggregations: Φmin, Φavg, Φlast all reported

**Experiment 4D — Spiral of Hallucination detection**
- Inject deliberate error at step 1 (replace correct retrieval with plausible wrong fact)
- Measure how EPR(step 2,3) and verbalized confidence(step 2,3) respond
- Test: does Nadler score spike earlier than verbalized confidence alone?
- Directly validates the "fused signal detects cascade propagation" claim

**Experiment 4E — LTT conformal calibration of trajectory score (Bracha)**
- Take the best Nadler trajectory score from 4C
- Apply LTT on held-out 100 samples: find τ with formal guarantee P(undetected failure ≤ α) ≥ 1−δ
- Compare vs AUQ's empirical τ ∈ [0.8, 0.95]
- This is the Bracha chapter: replaces AUQ's ad-hoc threshold with a distribution-free guarantee

### Infrastructure (~300 lines new code)
- 3-step ReAct loop over HotpotQA
- AUQ System 1 prompt modification (one appended sentence per step)
- `generate_with_entropies()` called per step (already exists)
- Trajectory aggregation Φmin/Φavg/Φlast (10 lines)
- Nadler fusion on step-vector pairs (already exists)

### Feasibility: Medium effort | Novelty: High | Risk: Medium
The main risk is that verbalized confidence and EPR are correlated on our domain (both track overall question difficulty). The Spearman ρ check in 4B is the critical diagnostic.

### Key reference numbers (AUQ paper targets)
- ReAct baseline Φmin AUROC: 0.667 (ALFWorld), 0.608 (WebShop)
- AUQ Φmin AUROC: **0.791** (ALFWorld), **0.755** (WebShop)
- Our target: Nadler-fused Φmin AUROC > 0.791

---

## Direction 5 — Statistical Guarantees: Conformal Calibration of the Fusion Detector

**Status**: Not started, but all prerequisites are in place. Pure theory/analysis direction.

### Core Hypothesis
The Nadler fusion score, despite being unsupervised, can be calibrated using a small held-out set (without ground-truth labels, using Bracha's PPI framework) to produce a threshold τ that guarantees: P(model hallucinated | score > τ) ≥ 1-δ. This is the rigorous deployment version of the thesis — not just "does the ensemble achieve higher AUC?" but "can we deploy it with a formal safety guarantee?"

### Connection to Supervisors
- **Bracha**: This is the most direct application of her core contributions. LTT, COIN (uncertainty-guarding selection), and Pareto testing are exactly the tools needed. Particularly her recent PPI work (Semi-Supervised Risk Control via Prediction-Powered Inference) enables calibration using model-generated pseudo-labels rather than human annotations — perfectly suited to our unsupervised setting
- **Ofir**: Less central here, but his VSDE stability criterion could inform which samples are "calibration-reliable" vs noisy

### Proposed Experiments

**Experiment 5A — LTT calibration of the Nadler score**
- Split TriviaQA + WebQ into calibration (100) + test (100)
- Calibrate τ using LTT on the calibration split with gold labels
- Report: P(false negative rate ≤ 10%) ≥ 90% — a formal guarantee statement
- Compare: uncalibrated AUC vs calibrated precision/recall at the guaranteed operating point

**Experiment 5B — Label-free calibration via PPI**
- Use model-generated pseudo-labels (Verify score > 0.9 → pseudo-correct) as the calibration signal
- Apply PPI to correct for pseudo-label noise
- Test whether distribution-free guarantees hold with zero human labeling

**Experiment 5C — Multi-objective Pareto testing**
- Define two risks simultaneously: false negative rate (missed hallucinations) and false positive rate (rejected correct answers)
- Use Pareto testing to find the τ that simultaneously satisfies both bounds
- This enables a safety-accuracy tradeoff curve with formal guarantees

### Why This Matters
Current hallucination detectors only report AUC — a ranking metric with no deployment semantics. A conformal guarantee turns the detector into a deployable tool: "this system will flag at least 90% of hallucinations at this confidence level." That's a publishable contribution even if the AUC improvement is modest.

### Feasibility: Low-Medium effort | Novelty: Medium (novel application, methods exist) | Risk: Low

---

## Direction 6 — Ofir's VSDE / PRAE Applied to Hallucination Detection

**Status**: Not started. Requires white-box or semi-white-box access (hidden states).

### Core Hypothesis
Ofir's VSDE framework proposes that anomalies are found not in low-density regions but in *high-variance* regions of the density function. Applied to LLMs: a hallucinated generation should produce hidden states with *higher variance* across samples than a grounded generation — because the model's internal representation is unstable when it doesn't know the answer. This predicts that variance of hidden states (across temperature-varied generations) is a more stable anomaly signal than the mean EPR.

More specifically: train a density model (or autoencoder, PRAE-style) on hidden states from correct answers, then measure how far a new generation's hidden states deviate from this "correct answer manifold."

### Connection to Supervisors
- **Ofir**: VSDE and PRAE are directly his contributions. This experiment is effectively a domain transfer of his methods to hallucination detection, which is a natural extension he would likely be enthusiastic about
- **Bracha**: eMOSAIC (Mahalanobis OOD detection in embedding space) is exactly the same idea applied to drug discovery. Applying eMOSAIC-style detection to LLM hidden states for hallucination is the NLP analogue of her drug discovery work

### Constraint
This direction requires **white-box or at least semi-white-box access** — specifically, the ability to extract hidden states from intermediate layers. This violates the current gray-box constraint (API-only). However, since we're running Falcon-3-10B locally on an A100 via HuggingFace, we CAN add forward hooks to extract hidden states — it's technically possible in our Colab setup, just not through a public API.

### Proposed Experiments

**Experiment 6A — Hidden state variance as a hallucination signal**
- Register a forward hook on Falcon-3-10B at layer 16 (mid-network)
- For each question at temperature T=1.0, T=1.5: extract the hidden state at the final answer token
- Compute variance of hidden state across K=5 temperature-varied generations
- Use as a new Nadler view alongside EPR views
- Hypothesis: variance(hidden state) is decorrelated from EPR (measures *stability* not *surprisal*)

**Experiment 6B — VSDE-style density estimation on correct-answer hidden states**
- Collect hidden states from questions where Falcon answered correctly (gold labels)
- Train a simple density estimator (Gaussian KDE or autoencoder) on this "correct manifold"
- For test questions: measure Mahalanobis distance from the correct manifold as an anomaly score
- Compare as a standalone view vs Nadler-fused with EPR views

**Experiment 6C — PRAE for contaminated training sets**
- In practice, we don't know which training examples are correct
- Apply PRAE's robust reconstruction objective: fit an autoencoder to all hidden states, but penalize L0 norm of included samples (force the model to find the clean subset)
- Use reconstruction error as hallucination score

### Feasibility: Medium effort | Novelty: High | Risk: Medium
The main uncertainty is whether hidden-state variance is decorrelated enough from EPR to add Nadler value, or whether they both track the same parametric uncertainty.

---

## Comparison Table

| Direction | Core Signal | Setting | Effort | Novelty | Bracha Link | Ofir Link |
|-----------|------------|---------|--------|---------|-------------|-----------|
| 1. LLM (CoT extension) | Trace-EPR, EDIS | Current setup | Low | Medium | LTT calibration | VSDE analogy |
| 2. RAG | Context-gap EPR, faithfulness | Text QA + retrieval | Low-Med | High | PPI pseudo-labels | PRAE grounding |
| 3. VLM | Visual vs textual EPR | Multimodal | Medium | High | Conformal for VLMs | Multi-view kernels, STG |
| 4. Agentic | Per-step EPR, cascade | Tool-use agent | High | Very High | Pareto testing, LTD | UProp, RADAR |
| 5. Conformal guarantees | Nadler score calibration | Current setup | Low-Med | Medium | Core Bracha work | Density stability |
| 6. Hidden states (VSDE) | Hidden state variance | Semi-white-box | Medium | High | eMOSAIC | Core Ofir work |

---

## Recommended Priority Order
*(Revised after Steps 39, 42–44)*

**Completed**:
- `EDIS_Replication.ipynb` ✅ — Formula validated (spike ratio 4.02×). AUC replication partial (accuracy-regime mismatch). EDIS usable in thesis.
- `CoT_EPR_Ensemble_res.ipynb` ❌ — Run was from old buggy version. Results discarded. Replaced by Unified notebook.

**Now — clean CoT + math run**:
1. **`Unified_EPR_Ensemble.ipynb`** — run into `epr_unified_experiment/`. This is the definitive Direction 1A experiment. Key diagnostics:
   - Is EPR(answer) non-constant? (hybrid split should guarantee yes)
   - Is ρ(EPR_trace, EDIS) < 0.75? (was 0.752 in invalid run — critical to re-confirm)
   - Does EDIS show larger gap over EPR(trace) on GSM8K vs factual QA? (domain-dependence test)
   - Does Nadler{trace+answer+EDIS} produce positive lift on any dataset?

**Decision gate after Unified run**:
- If ρ(trace, EDIS) < 0.75 AND Nadler lift > 0 on any dataset → Direction 1 viable, consider adding more CoT signals
- If ρ(trace, EDIS) > 0.75 OR Nadler lift ≤ 0 on all datasets → EPR(trace) alone is the CoT contribution; pivot to Direction 2

**After Unified run (regardless of outcome)**:
2. **Direction 2 (RAG)** — EPR(with context) vs EPR(no context) on TriviaQA Wikipedia passages. Same model, same infrastructure, orthogonal signal. Lowest-risk unexplored direction.

**Once best ensemble confirmed**:
3. **Direction 5 (Conformal)** — thesis endpoint, not optional. LTT calibration of best ensemble score. ~50 lines. Turns AUC into formal guarantee. This is the Bracha chapter.

**After Directions 1–2–5 are complete**:
4. **Direction 4 (Agentic)** — Qwen3-7B, HotpotQA multi-hop, Nadler{EPR + AUQ verbalized confidence}.
5. **Direction 6A (Hidden states)** — one forward hook on Falcon. Low effort, high Ofir alignment.

**Lower priority**:
6. **Direction 3 (VLM)** — only if committee wants multimodal chapter.

---

## Thesis Narrative Thread

## Direction 4 — Spectral Analysis of Entropy Trajectory for Hallucination Detection

**Status**: Not started. Novel hypothesis, no prior work found in reviewed literature.

### Core Hypothesis
EPR collapses the entropy trajectory H(n) to a single scalar — its mean (the DC component). This discards all temporal structure. The hypothesis is that **the frequency domain of H(n) contains discriminative information that EPR throws away**: correct reasoning traces have a more regular, periodic structure (entropy rises at step boundaries and settles within steps), while hallucinating traces show erratic oscillations with energy spread across higher frequencies. Applying FFT to H(n) and extracting spectral features should yield hallucination signals that are orthogonal to EPR by construction.

### Connection to Supervisors
- **Ofir**: Spectral decomposition of signals is his core methodology — diffusion maps, multi-view kernel methods, spectral graph theory. Framing hallucination detection as spectral signal analysis of the entropy time series maps directly onto his language and expertise. This is the direction most likely to generate genuine interest from him.
- **Bracha**: Spectral features extracted per sample can be fed into a conformal prediction framework — bounding the false-negative rate of a spectral-feature-based detector with distribution-free guarantees.

### Expected Patterns in H(f)

Hypotheses for what the FFT of H(n) might reveal:

1. **Step-boundary periodicity**: For structured math reasoning, steps are roughly L tokens long. This creates a quasi-periodic entropy signal → peaks in H(f) around f = 1/L and harmonics. Correct answers should have cleaner, more pronounced peaks at the step frequency.

2. **Spectral entropy as hallucination signal**: Wrong answers have erratic trajectories → energy spread across many frequencies → high spectral entropy (entropy of the power spectral density). This is nearly uncorrelated with EPR (which is the DC component) by construction.

3. **High-frequency power ratio**: The EDIS burst/rebound patterns are a special case of high-frequency energy. A principled high-band power ratio captures this and more, without requiring hand-crafted thresholds.

4. **Dominant frequency clarity**: Correct answers may have a single dominant frequency (coherent reasoning rhythm). Wrong answers lack a dominant frequency — energy is spread, no clear rhythm.

5. **Phase of first entropy peak**: Where in the trace does entropy first spike? An early spike before the model has established its reasoning path may indicate the hallucination began immediately.

### Research Plan

**Phase 1 — Exploratory analysis (no new inference)**
Load per-token entropies from `epr_unified_experiment/` (GSM8K traces already saved). Split by correctness label. For 10–20 correct and 10–20 wrong samples:
- Plot H(n) side by side — do wrong traces look more erratic visually?
- Apply FFT, plot average power spectrum per class
- Look for: different dominant frequencies, different spectral spread, class separation

**Phase 2 — Feature extraction and AUC evaluation**
Extract scalar features per sample from H(f):
- Spectral entropy: `H_spec = -Σ p(k) log p(k)` where `p(k) = |H(f_k)|² / Σ|H(f)|²`
- Low-band power: `Σ |H(f)|²` for f below threshold
- High-band power: `Σ |H(f)|²` for f above threshold
- High/low power ratio
- Dominant frequency: `argmax |H(f)|²`
- Spectral centroid: `Σ f·|H(f)|² / Σ|H(f)|²`

Evaluate AUC of each feature individually. Compare to EPR(trace)=66.8% and EDIS=66.2% on GSM8K.

**Phase 3 — Nadler integration** ✅ COMPLETED (Steps 46–51, three models)
Measure Spearman ρ between spectral features and EPR/EDIS. Features with ρ < 0.75 become additional Nadler views. Spectral entropy is the strongest candidate — uncorrelated with EPR (DC) by construction.

**Phase 3 results summary:**
- 11 signals tested (7 Phase 2 + 4 new: stft_max_high_power, stft_spectral_entropy, rpdi, sw_var_peak)
- Three models: Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, DeepSeek-Math-7B on GSM8K (200 samples, T=1.0)
- **sw_var_peak** (peak sliding-window variance) is the most robust individual signal: 73.5% (Qwen 1.5B) / 72.9% (DeepSeek 7B) — 0.6pp spread across architectures
- Best fusion: 75.9% (Qwen 1.5B), 75.0% (DeepSeek 7B) — 0.9pp spread, new GSM8K project high
- STFT features weak standalone (55–58%), RPDI architecture-dependent (54–75%)
- sw_var_peak and EPR are too correlated (ρ=0.75–0.83) to Nadler-fuse on small models; best fusions drop EPR and use sw_var_peak as the primary signal

**Phase 4 — Multi-dataset, multi-model generalization** ✅ COMPLETED (Step 54, April 22 2026)

Design: T=1.5, pipeline notebook, 12 signals (Phase 3 + trace_length), 8 configs total.

**MATH-500 results (A1–A4):**
| Config | Model | Best fusion | AUC |
|--------|-------|-------------|-----|
| A1 | Qwen2.5-Math-1.5B | epr+dominant_freq+rpdi | 88.3% [84.4, 91.8] |
| A2 | Qwen2.5-Math-7B | epr+high_band_power+rpdi | **96.6% [93.8, 98.7]** |
| A3 | DeepSeek-Math-7B | epr+trace_length+stft+rpdi | 75.2% [67.1, 82.0] |
| A4 | DeepSeek-R1-Distill | epr+spectral_entropy+dominant_freq+centroid+rpdi | 85.6% [80.8, 89.7] |

EPR dominates on MATH-500 (70–97% range). Longer traces (478–1151 tok) give full spectral resolution. Nadler fusion adds consistent lift (G2 passes 4/4).

**GPQA Diamond results (B1–B4):**
| Config | Model | Best fusion | AUC |
|--------|-------|-------------|-----|
| B1 | Mistral-7B | spectral_entropy+dominant_freq+stft | 65.0% [56.6, 74.0] |
| B2 | Qwen2.5-7B | epr+high_band_power+dominant_freq+rpdi | 60.1% [51.4, 68.3] |
| B3 | DeepSeek-R1-Distill | spectral_entropy+dominant_freq+rpdi | 59.1% [50.1, 68.4] |
| B4 | Llama-3.1-8B | stft_max_high_power+stft_spectral_entropy | 58.2% [48.8, 67.6] |

GPQA is near-random for all models — EPR collapses to 51–56%, CIs mostly touch 50%. Multiple-choice science MCQ does not produce discriminative entropy patterns with these models.

**Key finding: EPR signal is domain-specific.** On math reasoning, mean entropy directly encodes correctness. On science MCQ, it doesn't — spectral/structural features marginally lead but all signals are weak.

**Phase 5 — Temperature ablation (T=1.0) + cross-temperature fusion** ✅ COMPLETED (Steps 56–58, April 2026)

Design: T=1.0, same 12 signals, 4 configs (A1/A2 MATH-500, B1/B2 GPQA), cross-temperature Nadler fusion.

**MATH-500 T=1.0 results:**
| Config | Model | Best fusion | AUC | vs T=1.5 |
|--------|-------|-------------|-----|---------|
| A1 | Qwen2.5-Math-1.5B | 7-signal fusion | 81.7% [76.2, 86.6] | −6.6pp |
| A2 | Qwen2.5-Math-7B | trace_length+spectral_centroid+rpdi+sw_var_peak | **90.0% [85.5, 94.2]** | +3.4pp |

**GPQA Diamond T=1.0 results:**
| Config | Model | Best fusion | AUC |
|--------|-------|-------------|-----|
| B1 | Mistral-7B | dominant_freq+stft_max_high_power | 65.4% [57.3, 73.4] |
| B2 | Qwen2.5-7B | spectral_entropy+spectral_centroid+stft+rpdi+sw_var_peak | 57.4% [49.3, 66.3] |

**Cross-temperature fusion (T=1.0 ⊕ T=1.5, max_size=3):**
- A1: T=1.0=81.5%, T=1.5=74.1%, Combined=82.3% (+0.9pp) — marginal gain
- A2: T=1.0=89.4%, T=1.5=67.0%, Combined=still running when saved

**Phase 5 key findings:**
1. T=1.0 is the better operating point for MATH-500 for capable models (A2: 90.0% — new project best)
2. GPQA is domain-limited, not temperature-limited — T=1.0 does not help
3. **sw_var_peak is the only temperature-stable signal** (+1.1pp change A1; becomes #1 signal at T=1.0)
4. stft_spectral_entropy catastrophically collapses at T=1.0 (−29.6pp) — not robust
5. T=1.5 features are much more correlated with each other (200/286 subsets ρ-filtered for A2 vs only 60/286 at T=1.0) — T=1.0 produces more independent spectral features

**Core feature set (stable across temperatures, models, datasets):**
`sw_var_peak`, `spectral_centroid`, `stft_max_high_power`, `trace_length`
- EPR remains important for MATH-500 but is temperature/domain-sensitive
- These 4 features are the backbone for cross-domain claims

**Phase 5B — HotpotQA generalization** ← NEXT (see Direction 7)

### Key Technical Challenge
Variable trace lengths mean FFT bins correspond to different physical frequencies across samples. Solutions:
- Normalize frequency axis by trace length (use relative frequency f × N)
- Interpolate all traces to a fixed length before FFT
- Use Welch's method or STFT for more robust spectral estimation on short sequences

### Why Math First
GSM8K with Qwen2.5-Math produces long structured traces (hundreds of tokens, explicit step structure). Falcon on factual QA produces short traces (50–200 tokens) with no step structure — too short for meaningful frequency resolution. Validate on math first, then test whether any signal transfers to factual QA.

### Feasibility: Low effort (Phase 1–2 use existing data) | Novelty: High (not found in reviewed literature) | Risk: Medium (short traces may limit resolution; patterns may not generalize across datasets)

---

---

## Direction 7 — Baseline Comparison: LOS-Net, RENT, LapEigvals + HotpotQA Generalization

**Status**: Active. `Spectral_Analysis_Phase6.ipynb` built (Step 63). Ready to run on Colab.

### Motivation

Three papers with partial dataset/model overlap were identified as direct comparison targets:

| Paper | arXiv | Datasets | Models | Our overlap |
|-------|-------|---------|--------|-------------|
| **LOS-Net** | 2503.14043 | HotpotQA, IMDB, Movies (hallucination); WikiMIA, BookMIA (contamination) | Mistral-7B, LLaMA-3-8B | None yet — need HotpotQA run |
| **RENT** | 2505.22660 | GSM8K, MATH-500, AMC, AIME, GPQA | Qwen2.5-Math-1.5B/7B, Mistral-7B, Llama-3.1-8B | MATH-500, GPQA — exact same models as Phase 4/5 |
| **LapEigvals** | 2502.17598 | GSM8K + 6 QA benchmarks | Llama-3.1-8B, Phi-3.5, Mistral | GSM8K — Phase 1–3 results overlap |

LapEigvals is structurally the most direct parallel: spectral features of attention maps (theirs) vs spectral features of entropy trajectories (ours). RENT uses the same models and datasets as Phase 4/5 with near-perfect overlap. LOS-Net is the HotpotQA benchmark.

### Core Hypothesis

Our unsupervised spectral fusion over entropy trajectories is competitive with (a) supervised learnable classifiers (LOS-Net) and (b) spectral analysis of attention maps (LapEigvals) on overlapping benchmarks. On math reasoning, our method benefits from domain-natural long structured traces that give full spectral resolution — an advantage that attention-map spectral methods may not have.

For HotpotQA specifically: multi-hop reasoning (find fact A → reason → find fact B → synthesize) produces inherent step-level entropy structure, similar to math. HotpotQA is the best factual QA candidate for spectral features because it is structurally closer to math than single-hop TriviaQA.

Evidence from Step 45: CoT on TriviaQA *hurts* EPR (trace < direct), suggesting no useful spectral structure. HotpotQA's multi-hop nature is different — each hop is a genuine reasoning sub-task.

### Notebook: `Spectral_Analysis_Phase6.ipynb`

**Design decisions** (Step 62):
1. **No trace/answer split** — full response only. The "Answer:" marker appeared 0–2% of the time in prior experiments; the fallback (last 25%) is arbitrary. The full 50–200 token response trajectory IS the signal.
2. **Window size ablation** — sw_var_peak tested with w ∈ {3, 5, 7, 9, 16}, sw_step=1. LSC paper confirms w=2–3 optimal for short factual QA. Phase 6 validates this on HotpotQA.
3. **Phase naming** — this is Phase 6, not a standalone comparison notebook. The comparison is embedded within the HotpotQA experiment.

**Part 1 — Comparison table (no new inference, loads Phase 4/5 pkl files)**

| Dataset | Our AUC | Method | Competitor | Their AUC | Notes |
|---------|---------|--------|-----------|---------|-------|
| MATH-500 (Qwen-7B, T=1.0) | **90.0%** | Spectral Nadler | RENT | TBD | Same model, same dataset |
| MATH-500 (Qwen-1.5B, T=1.5) | **88.3%** | Spectral Nadler | RENT | TBD | Same model |
| MATH-500 (Qwen-7B, T=1.5) | **96.6%** | Spectral Nadler | RENT | TBD | Same model |
| GPQA (Mistral-7B, T=1.0) | **65.4%** | Spectral Nadler | RENT | TBD | Same model |
| GSM8K (Qwen-1.5B) | **75.9%** | Spectral Nadler | LapEigvals | TBD | Method-level comparison (different models) |
| HotpotQA (Mistral-7B) | **Phase 6 result** | Spectral Nadler | LOS-Net | **72.92%** | Exact same model+dataset |

**Part 2 — HotpotQA new inference**
- Model: Mistral-7B-Instruct-v0.2 (same as LOS-Net)
- Dataset: HotpotQA `fullwiki` validation, 200 samples, gold string matching
- Prompt: multi-hop step-by-step CoT (explicitly asks to find intermediate fact, then synthesize)
- Pipeline: identical to Phase 5 — same `generate_full()`, same 12 spectral features, same Nadler fusion
- New: window ablation cell (w ∈ {3, 5, 7, 9, 16}) post-inference
- Primary comparison: best fusion AUC vs LOS-Net's 72.92% on HotpotQA/Mistral

**Decision Gates** (7 gates, automatic pass/fail in notebook):

| Gate | Condition | Pass | Fail |
|------|-----------|------|------|
| G0 | len(labels) ≥ 150 | Sufficient samples | Run more |
| G1 | Any signal AUC > 55% | Spectral structure exists | No transfer |
| G2 | Best fusion > 65% | Method works on multi-hop QA | Math-specific only |
| **G3** | **Best fusion > 72.92%** | **Beat LOS-Net on HotpotQA** | LOS-Net leads factual QA |
| G4 | sw_var_peak > 60% | Core feature transfers | sw_var_peak is math-specific |
| G5 | Optimal w* ≤ 9 | Window ablation confirms LSC | Window size doesn't matter |
| G6 | CI lower bound > 55% | Statistically reliable | Too few samples |

**Expectations from prior experiments:**

| Expectation | Basis | Confidence |
|-------------|-------|-----------|
| sw_var_peak will be strongest individual | Temperature-stable in Phases 4/5; w=3 catches entity spans | Medium |
| EPR weaker than on math | Step 45: confidence masking on factual QA; trace < direct | High |
| AUC range: 60–75% (below MATH-500) | Shorter traces, no explicit step structure, domain mismatch | High |
| w=3 or w=5 beats w=16 | LSC: w=2–3 optimal on NQ/TriviaQA/SQuAD | Medium |
| stft_max_high_power may be weak | Traces < 50 tokens may hit STFT min_len=32 floor | Medium |
| Best fusion: trace_length + sw_var_peak (EPR not dominant) | Phase 5 A2 core set; same pattern expected | Medium |

### Connection to Supervisors
- **Bracha**: Comparison against published baselines is required for the thesis. This provides the empirical grounding for the scope claim.
- **Ofir**: LapEigvals comparison is directly relevant — two different spectral approaches to the same forward pass. Positions our work within spectral methods.

### Feasibility: Low-Medium effort | Novelty: Low (comparison study) | Risk: Low
The comparison table needs no new inference. HotpotQA is the only new run — same infrastructure as Phase 5, new dataset and prompt.

---

## Comparison Table (updated)

| Direction | Core Signal | Setting | Effort | Novelty | Bracha Link | Ofir Link |
|-----------|------------|---------|--------|---------|-------------|-----------|
| 1. LLM (CoT extension) | Trace-EPR, EDIS | TriviaQA/WebQ | Low | Medium | LTT calibration | VSDE analogy |
| 2. RAG | Context-gap EPR, faithfulness | Text QA + retrieval | Low-Med | High | PPI pseudo-labels | PRAE grounding |
| 3. VLM | Visual vs textual EPR | Multimodal | Medium | High | Conformal for VLMs | Multi-view kernels |
| 4. Agentic | Per-step EPR, cascade | Tool-use agent | High | Very High | Pareto testing | UProp, RADAR |
| 5. Conformal guarantees | Nadler score calibration | Current setup | Low-Med | Medium | Core Bracha work | Density stability |
| 6. Hidden states (VSDE) | Hidden state variance | Semi-white-box | Medium | High | eMOSAIC | Core Ofir work |
| **7. Baseline comparison** | **Spectral features** | **MATH/GSM8K/HotpotQA** | **Low-Med** | **Low** | **Scope validation** | **LapEigvals positioning** |

---

## Recommended Priority Order (updated April 2026)

**Completed**:
- Spectral Analysis Phases 1–5 ✅
- Unified EPR / CoT ensemble ✅
- Temperature-varied EPR + behavioral views ✅

**Next — Direction 7 (Comparison + HotpotQA)**:
1. ✅ `Spectral_Analysis_Phase6.ipynb` built (Step 63) — ready to run on Colab
2. Run Phase 6: Part 1 (static comparison table loads automatically) + Part 2 (HotpotQA/Mistral-7B inference)
3. Window ablation cell runs post-inference: sw_var_peak with w ∈ {3, 5, 7, 9, 16}
4. Read decision gates (G0–G6) to determine thesis scope claim

**After comparison**:
4. Direction 5 (Conformal) — thesis endpoint, formal guarantee chapter, ~50 lines
5. Direction 4 (Agentic, HotpotQA already partially explored) — if committee wants it
6. Direction 1E (Add trace-EPR to 6-view ensemble) — if TriviaQA/WebQ scope still in thesis

---

## Thesis Narrative Thread

Across all directions, the unifying claim is:

> *Single-view uncertainty metrics (EPR, Semantic Entropy) leave signal on the table. Nadler spectral fusion recovers that signal — but only when views satisfy two conditions: conditional independence AND a shared prediction target. Each direction in this roadmap is a new arena where this principle is tested: different modalities (VLM), different contexts (RAG), different computation levels (CoT trace vs terminal answer), different access levels (gray-box vs semi-white-box), and different deployment requirements (point-estimate AUC vs formal risk bounds).*

This thread gives the thesis a single identity regardless of which combination of directions is ultimately pursued.
