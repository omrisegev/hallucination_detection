# Phase 11 — Agentic Hallucination Detection: Plan

**Notebook**: `Spectral_Analysis_Phase11_Agentic.ipynb`
**Direction**: Research_Directions.md → Direction 4 (Agentic Flow Hallucination Detection)
**Goal**: Apply the spectral pipeline + Nadler fusion to ReAct agent trajectories on HotpotQA, with AUQ verbalized confidence as the prior-art baseline. Deliver insights for the advisor presentation (Ofir + Bracha): orthogonality of signals, where in the trajectory spectral helps most, comparison vs SOTA.

---

## What this experiment measures

A 3-step ReAct agent on HotpotQA fullwiki:

```
Step k:
  Thought_k  ← model generates reasoning
  Action_k   ← model emits search(query) or finish(answer)
  Confidence_k ← model self-reports c_hat ∈ [0,1]   ← AUQ System 1 (UAM)
  Observation_k ← simulate_retrieve_tool(query, context)
```

Two questions are scored per question:
- **step_correct_k**: did this step retrieve a supporting-fact paragraph?
- **traj_correct**: did the final answer EM-match the gold?

For each step we record per-token entropies. From those we extract the standard 13-feature spectral vector (EPR, spectral_entropy, low/high band power, dominant_freq, spectral_centroid, hl_ratio, stft_max_high_power, stft_spectral_entropy, rpdi, sw_var_peak, sw_var_peak_adaptive, trace_length). We also record verbalized confidence c_hat per step.

Per-trajectory aggregations follow AUQ:
- **Φ_min**: weakest-link (min over steps)
- **Φ_avg**: mean over steps
- **Φ_last**: final step only

---

## Lessons from Phase 10 carried into Phase 11

1. **Persistence-by-default**: every analysis cell that runs > 30 s persists its output dict to `RES_DIR/*.pkl` with the three-branch pattern (in-memory → on-disk → recompute, with `FORCE_RECOMPUTE_*` override). No more `background_save: true` losing variables.
2. **Standard Cell 1**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `HF_HOME` on Drive, `import datasets` to freeze pyarrow.
3. **Drive symlink workaround**: `ensure_flat_dir()` for models even though Qwen-7B / Mistral-24B are non-AWQ — keeps the loader uniform.
4. **All helpers in `spectral_utils`**: ReAct loop + tool simulator live in a new `agent_utils.py` module, never inline in the notebook.
5. **`sw_var_peak_adaptive`**: short per-step traces (50–150 tokens) require adaptive window (used in Phase 9/10).

---

## Notebook structure

| Cell | Purpose | Persists to |
|------|---------|-------------|
| 1 | Setup: clone + install + imports + datasets preload | — |
| 2 | Master config (`MODELS`, paths, N=200, MAX_STEPS=3) | — |
| 3 | Mount Drive + create dirs | — |
| 3c | `ensure_flat_dir` helper | — |
| 4 | Load HotpotQA fullwiki validation | — |
| 5 | Spot-check one trajectory | — |
| 6 | `run_agentic_inference()` with .pkl checkpoint every 25 | `raw/{model}__hotpotqa.pkl` |
| 7 | Qwen-7B driver | (writes raw) |
| 8 | Mistral-24B driver | (writes raw) |
| 9 | Per-step + per-trajectory feature extraction → `ALL_CELLS` | `features/{model}__hotpotqa.pkl` |
| 10 | Pre-conditions G0 | — |
| 11 | Individual feature AUC table (per-step × aggregation) | — |
| 12 | Best Nadler subset + weights per (model, aggregation) | `nadler_res.pkl` |
| 13 | **AUQ baseline** — verbalized confidence Φ_min / Φ_avg / Φ_last AUC | `auq_res.pkl` |
| 14 | **Spectral + AUQ Nadler fusion** | `fusion_res.pkl` |
| 15 | Length-controlled analysis | `len_res.pkl` |
| 16 | PCA diagnostic | `pca_res.pkl` |
| 17 | ρ heatmap: ρ(EPR_step, verb_conf_step) — **G1** | — |
| 18 | Step-position analysis: which step's signal is most informative | — |
| 19 | Failure-mode breakdown (planning / tool / synthesis) | — |
| 20 | Lite Semantic Entropy baseline (per-trajectory) | `se_res.pkl` |
| 21 | Headline detector × aggregation AUC table + SOTA refs | — |
| 22 | Gate evaluation (G0–G5) | — |
| 23 | Save comprehensive results pickle | `phase11_results.pkl` |
| 24 | Plots (heatmap, ρ scatter, by-step bars, by-failure-mode bars) | `plots/*.png` |

---

## Gates

| Gate | Threshold | Why |
|------|-----------|-----|
| **G0** | ≥150 trajectories per model, both classes ≥10 each | sample-size sanity |
| **G1** | Spearman ρ(EPR_step, verb_conf_step) < 0.5 on ≥1 model | Direction 4 critical gate — fusion only works if signals are orthogonal |
| **G2** | Best spectral Nadler Φ_min AUROC ≥ 0.70 | spectral features work on agent trajectories at all |
| **G3** | Spectral + AUQ Nadler beats AUQ-alone by ≥ 3pp on ≥1 model | EPR adds real signal over verbalized confidence |
| **G4** | Spectral-only Φ_min beats trace_length-alone Φ_min by ≥ 3pp | no length confound — same control as Phase 10 G3 |
| **G5** | Spectral + AUQ Φ_min ≥ 0.791 (AUQ paper SOTA on ALFWorld) | beat prior art on at least one model |

---

## SOTA comparison

| Source | Benchmark | Metric | Score | Notes |
|--------|-----------|--------|-------|-------|
| **AUQ paper** (Zhang et al. 2026, arXiv:2601.15703) | ALFWorld | Φ_min AUROC | **0.791** | 7B+ models, verbalized confidence + best-of-N reflection |
| AUQ paper | WebShop | Φ_min AUROC | 0.755 | |
| AUQ paper | ReAct baseline / ALFWorld | Φ_min AUROC | 0.667 | what we want to beat with EPR-only |
| **LOS-Net** (arXiv:2503.14043) | HotpotQA / Mistral-7B | AUC | 72.92% | supervised learnable classifier; our spectral target on HotpotQA |
| Phase 10 RAG / qwen7b / hotpotqa | L-CiteEval HotpotQA | Best Nadler AUC | 79.5% | reference: spectral works on statement-level RAG grounding |

**Honest caveat**: AUQ scores are on ALFWorld/WebShop (different domain). Our HotpotQA is closer to LOS-Net's domain. We report both, frame our number as: "Φ_min AUROC on multi-hop QA agent trajectories — first benchmark of its kind for the (model, dataset) cells."

---

## Advisor-facing insights this notebook will produce

For **Ofir** (spectral / multiview):
- ρ(EPR_step, verb_conf_step) — empirical evidence that token-level surprisal and meta-cognitive self-assessment are orthogonal views, satisfying Nadler condition 1
- Nadler weight matrix (2 cells × 3 aggregations × 13 features) — which spectral features fuse productively in the agent setting vs the RAG setting
- Step-position bar chart — does the spectral signal load most on early-step (planning) errors, late-step (synthesis) errors, or step-3 (commitment) errors?

For **Bracha** (calibration / baselines):
- AUQ vs Spectral vs Fused AUROC table with bootstrap CIs — comparison vs the established prior-art baseline
- Failure-mode AUC breakdown — does spectral fusion improve recall specifically on tool-fabrication or planning failures?
- Gates with explicit thresholds and pass/fail — sets up Direction 5 (LTT calibration) as the natural next step

---

## v1.1 additions (after agentic-benchmark gap audit)

After comparing the v1 design against the AgentHallu / GAIA2 / CoT-survey research docs, three additions landed:

1. **Branching entropy (SBUT) as the 14th per-step feature** — Cell 9 now stores `branching_entropy = mean(token_entropies[:3])` alongside the 13 spectral features. This is exactly the Step-Boundary Uncertainty Trajectory signal that Cheng et al. 2025 and the CoT survey identify as the most promising new view. It enters the Cell 12 Nadler subset search automatically.
2. **Cell 17b — Step Localization Accuracy (Φ_LSA)** — AgentHallu's canonical metric. For each incorrect trajectory: does the detector's argmin-score step match the first actually-incorrect step? Reports Φ_LSA for EPR / AUQ / Spectral Nadler / Fused, plus a random baseline and the AgentHallu SOTA reference (41.1% Gemini 2.5 Pro / 10.9% open-source avg).
3. **Cell 20c — Spiral-of-Hallucination injection (Direction 4D)** — replay SPIRAL_SUBSET=30 originally-correct trajectories with step-1 observation replaced by a distractor. Reports per-step EPR / verb_conf / branching_entropy response curves and a cascade-onset analysis. Gated behind an uncomment (needs each model reloaded, ~5–10 min per model).

The v1.1 results pickle (Cell 23) includes `LSA_RES` and `SPIRAL_RES`. Plots 5–6 in Cell 24 visualize them.

## What is explicitly **out of scope** for Phase 11

- Llama-3.3-70B / Qwen-72B-AWQ (BNB / gptqmodel headaches; rerun later if needed)
- Multi-dataset (2WikiMHQA, MuSiQue) — focus on HotpotQA for a clean SOTA hook
- LTT conformal calibration (Direction 4E / Direction 5) — separate notebook
- Real tool calls — we simulate retrieval from the HotpotQA gold context, which is the same trick LOS-Net used; lets us focus on uncertainty signals, not retrieval IR quality
- AgentHallu's full dataset / 14-sub-category taxonomy — reproducing requires the LangGraph/AutoGen agent stack; our 4-way `categorize_failure_mode` is the simplified version
- AUQ System 2 (UAR reflection / best-of-N) — paper's 79.1% number is System 1 alone, so the baseline comparison is already fair
