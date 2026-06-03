# Phase 12 — SOTA Comparison Tables

**Status**: Complete. Phase 12 notebook run finished 2026-06-02. All `[COMPUTE]` cells filled.
New competitors added from `research_phase10_rag/` deep-research: TOHA (ACL 2026), Streaming Prefix-Level Detection (arXiv 2601.02170), LapEigvals published range.

---

## Domain 1: Math — GSM8K

**Our best result**: Llama-3.1-8B / T=1.0 → **75.92% AUROC** [72.48, 79.39] (Nadler, Step 100, N=1,319, uses val labels for sign orientation)

| Method | Model | AUROC | 95% CI | Access | Compute | Supervision | Source |
|--------|-------|-------|--------|--------|---------|-------------|--------|
| **Nadler Spectral Fusion (ours)** | Llama-3.1-8B | **75.9%** | [72.5, 79.4] | Gray-box | 1-pass | Val labels | Step 100 (official) |
| **L-SML paper-aligned (ours)** | Mistral-7B | 55.6% | [47.2, 63.1] | Gray-box | 1-pass | None | Phase 12 P1b |
| Self-Consistency K=10 | Llama-3.1-8B | **78.5%** | [72.0, 84.5] | Black-box | K=10 | None | Phase 12 computed |
| Semantic Entropy NLI K=10 | Llama-3.1-8B | **77.4%** | [70.9, 83.5] | Black-box | K=10 | None | Phase 12 computed |
| Streaming Prefix-Level (arXiv 2601.02170) | LLaMA-3.1-8B | 87.8% (step) / 72.7% (prefix) | n/a | White-box | 1-pass | Supervised | Lu et al. 2026 |
| LapEigvals unsupervised | Llama-3.1-8B | 72.0% | n/a | White-box | 1-pass | None | Phase 7 re-run (arXiv 2502.17598) |
| LapEigvals supervised | Llama-3.1-8B | 87.2% | n/a | White-box | 1-pass | 80% labeled | Phase 7 re-run (paper range: 87.0–92.5%) |
| Semantic Entropy NLI K=10 | Mistral-7B | 75.8% | n/a | Black-box | K=10 | None | arXiv 2502.03799 ⚠ diff. model |
| Semantic Entropy NLI K=10 | Llama-3.2-3B | 76.5% | n/a | Black-box | K=10 | None | arXiv 2502.03799 ⚠ diff. model |
| Self-Consistency K=8 | Reasoning models avg. | 70.7% | n/a | Black-box | K=8 | None | arXiv 2603.19118 ⚠ diff. models |
| EDIS (entropy dynamics) | Qwen2.5-Math-1.5B | 80.4% (agg.) | n/a | Gray-box | 1-pass | None | arXiv 2602.01288 ⚠⚠ |
| Mean entropy baseline | Qwen2.5-Math-1.5B | 67.3% (agg.) | n/a | Gray-box | 1-pass | None | arXiv 2602.01288 ⚠⚠ |

⚠ Different model from ours (same size class, not exact match)
⚠⚠ Aggregate AUROC over {GSM8K + MATH + AMC23 + AIME24}, not GSM8K alone

**Key narrative**: Our Nadler (75.9%, 1-pass gray-box, val labels) is competitive with SE NLI (77.4%) and SC (78.5%), both of which need K=10 re-generations and no model internals. LapEigvals supervised (87.2%) and Streaming Prefix-Level (87.8%) beat us but require full white-box access + labeled training data — different access tier. Our paper-aligned unsupervised L-SML (55.6%) trails: honest cost of removing supervision.

---

## Domain 2: Math — MATH-500

**Our best result**: Qwen2.5-Math-7B-Instruct / T=1.0 → **96.69% AUROC** [93.90, 98.69] (Nadler, Step 100 official)

| Method | Model | AUROC | 95% CI | Access | Compute | Supervision | Source |
|--------|-------|-------|--------|--------|---------|-------------|--------|
| **Nadler Spectral Fusion (ours)** | Qwen2.5-Math-7B | **96.7%** | [93.9, 98.7] | Gray-box | 1-pass | Val labels | Step 100 (official) |
| **Nadler Spectral Fusion (ours)** | Qwen-1.5B | **88.0%** | n/a | Gray-box | 1-pass | Val labels | Step 100 (official) |
| Self-Consistency K=10 | Qwen2.5-Math-7B | 87.2% | [72.1, 98.4] | Black-box | K=10 | None | Phase 12 computed |
| Semantic Entropy NLI K=10 | Qwen2.5-Math-7B | 87.7% | [79.7, 93.9] | Black-box | K=10 | None | Phase 12 computed |
| Streaming Prefix-Level (arXiv 2601.02170) | Qwen2.5-7B | 86.7% (step) / 81.1% (prefix) | n/a | White-box | 1-pass | Supervised | Lu et al. 2026 ⚠ |
| EDIS (entropy dynamics) | Qwen2.5-Math-1.5B | 80.4% (agg.) | n/a | Gray-box | 1-pass | None | arXiv 2602.01288 ⚠⚠ |

⚠ Qwen2.5-7B (general), not Qwen2.5-Math-7B (math-finetuned) — different model family
⚠⚠ Aggregate AUROC over 4 math datasets on a 1.5B math-specific model

**Key narrative**: Our Nadler (96.7%) beats SE (87.7%) and SC (87.2%) by ~9pp with 1 forward pass. Also beats Streaming Prefix-Level (87% even with supervision and white-box access). This is likely the strongest published AUROC on MATH-500 with any method on Qwen-Math-7B. EDIS 80.4% uses a weaker 1.5B model on 4 pooled datasets — not a direct competitor.

---

## Domain 3: Science — GPQA Diamond

**Our best results**:
- Nadler: Qwen2.5-72B-AWQ → **67.47%** [59.71, 74.74] (val labels)
- L-SML paper-aligned (unsupervised): Qwen2.5-7B → **71.3%** [50.4, 89.0]

| Method | Model | AUROC | 95% CI | Access | Compute | Supervision | Source |
|--------|-------|-------|--------|--------|---------|-------------|--------|
| **L-SML paper-aligned (ours)** | Qwen2.5-7B | **71.3%** | [50.4, 89.0] | Gray-box | 1-pass | None | Phase 12 P2b |
| **Nadler Spectral Fusion (ours)** | Qwen2.5-72B-AWQ | **67.5%** | [59.7, 74.7] | Gray-box | 1-pass | Val labels | Step 100 (official) |
| **Nadler Spectral Fusion (ours)** | Mistral-7B | **65.3%** | [56.7, 74.0] | Gray-box | 1-pass | Val labels | Step 100 (official) |
| **L-SML paper-aligned (ours)** | DeepSeek-R1-Distill-7B | 55.3% | [44.2, 65.2] | Gray-box | 1-pass | None | Phase 12 P2c |
| **L-SML paper-aligned (ours)** | Qwen3-8B | 53.2% | [43.0, 63.6] | Gray-box | 1-pass | None | Phase 12 P2d |
| Semantic Entropy NLI K=10 | Qwen2.5-7B | 70.6% | [43.6, 93.3] | Black-box | K=10 | None | Phase 12 computed |
| Verbalized Confidence K=1 | Qwen2.5-7B | 67.9% | [49.5, 83.3] | Black-box | 1-pass | None | Phase 12 computed |
| Self-Consistency K=10 | Qwen2.5-7B | 33.6% | [11.0, 58.2] | Black-box | K=10 | None | Phase 12 computed ⚠⚠⚠ |
| Verbalized Confidence K=1 | Reasoning models avg. | 74.6% | n/a | Black-box | 1-pass | None | arXiv 2603.19118 ⚠ |
| Self-Consistency K=8 | Reasoning models avg. | 75.4% | n/a | Black-box | K=8 | None | arXiv 2603.19118 ⚠ |
| SC + VC K=8 | Reasoning models avg. | 82.1% | n/a | Black-box | K=8 | None | arXiv 2603.19118 ⚠ |

⚠ Published baselines use reasoning models (gpt-oss-20b, Qwen3-30B-A3B, DeepSeek-R1-8B) — 4–10× stronger than Qwen-7B/Mistral-7B
⚠⚠⚠ SC catastrophically fails on GPQA (33.6%): consistent wrong answers fool consistency — models confidently repeat the same incorrect choice

**Key narrative**: GPQA is hard for all methods on non-reasoning 7B models. Our L-SML (71.3%) and SE NLI (70.6%) are within noise of each other at n=51 (wide CIs). The noteworthy finding is SC's complete failure (33.6%) — GPQA knowledge errors are systematic, so consistency ≠ correctness here. Published SC/VC numbers (74-82%) come from models 4-10× stronger; direct comparison is cross-model-class.

---

## Domain 4: RAG — L-CiteEval (citation-grounded QA)

**Task**: Model generates response with `[1][2]` citation markers; we score whether each cited statement is grounded. **No published competitor targets this exact AUROC framing on citation grounding.**

### HotpotQA (L-CiteEval, N=240)

| Method | Model | AUROC | 95% CI | Access | Compute | Supervision | Source |
|--------|-------|-------|--------|--------|---------|-------------|--------|
| **Nadler Spectral Fusion (ours)** | Llama-3.1-8B | **88.1%** | [80.6, 94.4] | Gray-box | 1-pass | Val labels | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Qwen2.5-7B | **80.2%** | [66.5, 91.4] | Gray-box | 1-pass | Val labels | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Qwen2.5-72B-AWQ | **79.4%** | [70.5, 86.8] | Gray-box | 1-pass | Val labels | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Mistral-Small-24B | **77.2%** | [62.2, 90.3] | Gray-box | 1-pass | Val labels | Phase 10 |
| SelfCheckGPT NLI K=5 | Qwen2.5-7B | 51.4% | [41.5, 62.9] | Black-box | K=5 | None | Phase 12 P3 |
| TOHA (arXiv 2504.10063, ACL 2026) | Mistral-7B / LLaMA-3.1-8B | 71–80% | n/a | White-box | 1-pass | 50 val samples | ⚠ std HotpotQA, no citations |
| LOS-Net supervised | Mistral-7B | 72.9% | n/a | Gray-box | supervised | Full sup. | arXiv 2503.14043 ⚠ std HotpotQA |

⚠ TOHA and LOS-Net use **standard** HotpotQA (no citation markers, different scoring). Not directly comparable.

### 2WikiMultiHopQA (L-CiteEval, N=26 — small sample)

| Method | Model | AUROC | 95% CI | Access | Compute | Source |
|--------|-------|-------|--------|--------|---------|--------|
| **Nadler Spectral Fusion (ours)** | Qwen2.5-7B | **81.3%** | [71.4, 89.7] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Qwen2.5-72B-AWQ | **76.2%** | [65.2, 85.9] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Mistral-Small-24B | **74.0%** | [56.9, 87.9] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Llama-3.1-8B | **71.0%** | [58.7, 81.6] | Gray-box | 1-pass | Phase 10 |
| SelfCheckGPT NLI K=5 | Qwen2.5-7B | 55.3% | [35.7, 78.3] | Black-box | K=5 | Phase 12 P3 |
| Any published competitor | — | No published result | — | — | Novel task |

### Natural Questions (L-CiteEval, N=160)

| Method | Model | AUROC | 95% CI | Access | Compute | Source |
|--------|-------|-------|--------|--------|---------|--------|
| **Nadler Spectral Fusion (ours)** | Qwen2.5-7B | **82.8%** | [70.8, 92.6] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Mistral-Small-24B | **74.0%** | [61.3, 91.5] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Qwen2.5-72B-AWQ | **72.5%** | [61.7, 82.6] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Llama-3.1-8B | **70.3%** | [45.6, 86.2] | Gray-box | 1-pass | Phase 10 |
| SelfCheckGPT NLI K=5 | Qwen2.5-7B | 57.1% | [42.9, 70.5] | Black-box | K=5 | Phase 12 P3 |
| Any published competitor | — | No published result | — | — | Novel task |

### NarrativeQA (L-CiteEval, N=240)

| Method | Model | AUROC | 95% CI | Access | Compute | Source |
|--------|-------|-------|--------|--------|---------|--------|
| **Nadler Spectral Fusion (ours)** | Qwen2.5-72B-AWQ | **73.1%** | [63.8, 81.2] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Qwen2.5-7B | **70.0%** | [58.3, 80.8] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Mistral-Small-24B | **67.0%** | [56.2, 77.3] | Gray-box | 1-pass | Phase 10 |
| **Nadler Spectral Fusion (ours)** | Llama-3.1-8B | **63.7%** | [56.2, 70.7] | Gray-box | 1-pass | Phase 10 |
| SelfCheckGPT NLI K=5 | Qwen2.5-7B | 52.4% | [41.7, 65.4] | Black-box | K=5 | Phase 12 P3 |
| Any published competitor | — | No published result | — | — | Novel task |

**Key narrative (RAG)**: Strongest domain. Our Nadler (+20–37pp over SelfCheckGPT across all 4 datasets) with 1 pass vs K=5 re-generations. TOHA (71–80% on standard HotpotQA, white-box) and LOS-Net (72.9%, supervised) are the closest competitors — both use a different task formulation (no citation markers). On the citation-grounding AUROC framing, we are first.

---

## Supervision legend

| Label | Meaning |
|-------|---------|
| **Val labels** | Ground-truth correctness labels used for feature-sign orientation + subset selection (best_nadler_on). Supervised at selection time; no labels needed at inference. |
| **None** | Fully unsupervised — no labels at any stage (L-SML + offline consensus orientation). Real labels used only for external AUROC evaluation. |
| **80% labeled** | 80% training set used to fit logistic probe (LapEigvals supervised). |
| **Full sup.** | Fully supervised end-to-end (LOS-Net). |
| **Supervised** | Probe or model trained on labeled hallucination examples (Streaming Prefix-Level, LapEigvals). |
| **50 val samples** | Small labeled validation set used only for head selection (TOHA). |

---

## Methods not added (rationale)

| Method | Why excluded |
|--------|-------------|
| Energy Mountain / GSP (arXiv 2510.19117) | Reports **accuracy** (88.75%), not AUROC; tested on short factual-statement classification, not our benchmarks |
| RHD / Reasoning Score (arXiv 2505.12886, ICLR 2026) | AUROC numbers uncertain; tested on reasoning LRMs (DeepSeek-R1 family) not our instruction-tuned models |
| Real-Time Hallucinated Entity Probes (arXiv 2509.03531) | Token-level entity detection on LongFact/HealthBench — different task and granularity; supervised (8K training samples) |
| RES — Reasoning-Explanation Symmetry (ICLR 2026) | Paper not publicly found; reported 0.996 AUROC is unverified; model/benchmark unspecified |
