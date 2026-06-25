# Research Directions — Thesis Roadmap
*Omri Segev | Supervised by Bracha Laufer-Goldshtein & Ofir Lindenbaum*

---

## The Thesis

**Claim**: Spectral features of the per-token entropy trajectory H(n) — fused via the Spectral Meta-Learner (L-SML; Jaffé–Fetaya–Nadler 2016) — detect LLM hallucinations at state-of-the-art AUROC in a single forward pass, with no ground-truth labels at inference time.

**Operating regime**: The method works on reasoning-heavy generation (MATH-500, GSM8K, multi-hop QA) where the entropy trace is long enough (≥100 tokens) to carry discriminative spectral structure. Performance is reduced on short factual QA traces (<60 tokens) and MCQ formats where entropy dynamics are structurally suppressed.

**Why it's novel**:
1. Spectral features of H(n) — not hidden states, not verbalized confidence, not sampling ensembles
2. L-SML fusion is unsupervised: no labels used at inference time; feature directions calibrated once offline from aggregated empirical evidence
3. Single-pass (K=1): cost is fixed and independent of question difficulty

---

## Supervisor Connections

| Supervisor | Core connection |
|-----------|----------------|
| **Ofir Lindenbaum** | Spectral decomposition of uncertainty signals maps onto his core methodology (diffusion maps, multi-view spectral methods, VSDE). L-SML is a spectral fusion method applied to entropy signals. |
| **Bracha Laufer-Goldshtein** | The L-SML score is a continuous input to LTT/COIN calibration, turning AUROC (a ranking result) into a deployable detector with a formal false-negative-rate guarantee — the conformal chapter. |

---

## Core Method (what is settled as of Jun 2026)

### Step 1 — Feature extraction

From a single greedy forward pass, extract per-token entropy H(n). From H(n), compute spectral and time-domain features:

| Feature set | Features |
|-------------|----------|
| **GOOD_5** (primary candidate) | `epr`, `low_band_power`, `sw_var_peak`, `cusum_max`, `spectral_entropy` |
| **STABLE_H9** | GOOD_5 + `spectral_centroid`, `dominant_freq`, `rpdi`, `cusum_mean` |
| **All-16** | Full `FEAT_NAMES` list in `spectral_utils/feature_utils.py` |

All 16 features are implemented in `spectral_utils/feature_utils.py`. Feature count is **open** — the logistic oracle (Item 2) will bound the headroom from more features.

### Step 2 — Fusion via L-SML (CONT configuration)

`lsml_continuous_pipeline(feats, subset, FEATURE_SIGNS)`:
1. Pre-orient each feature: `x_i_oriented = sign_i · x_i` (offline consensus direction)
2. Z-score normalize
3. Binarize → L-SML rank-1 eigenvector → continuous cross-cluster score
4. Returns one real number per sample (higher = more likely correct)

**FEATURE_SIGNS** = orientation vector derived offline from majority vote across 29 cells (AUROC-weighted). Unsupervised at inference time — no per-sample labels used.

**Why CONT over binary**: continuous encoding beats the old `np.sign()` pipeline by **+4.9pp macro** (65.2→70.1%) and +7.2pp on the reasoning regime. The binarization was the largest single source of lost signal.

### What is NOT settled (pending meeting experiments)

- **Final feature set** (5 / 9 / 16) — logistic oracle (Item 2) will bound the headroom
- **Whether sampling fusion adds lift** — Item 5 tests SE K=10 + spectral
- **Whether temperature diversity matters** — Item 6 ablates same-T vs mixed-T multi-pass
- **Scope on factual QA** — Item 3 extends to NQ, SQuAD v2, AmbigQA, PopQA

The final method choice has not been made. CONT + GOOD_5 is the current strongest result, not a decided thesis configuration.

---

## Current Best Results

*(Step 135, honest numbers — do not cite old Step-117 supervised numbers: 96.7/71.3/88.1)*

| Domain | Model | CONT AUROC | Competitor (K=10 sampling) |
|--------|-------|-----------|---------------------------|
| MATH-500 | Qwen2.5-Math-7B | **94.4%** | SE NLI 87.7%, SC 87.2% |
| GSM8K | Llama-3.1-8B | **75.6%** | SC 78.5%, SE 77.4% |
| Macro avg (29 cells) | multiple | **70.1%** | simple avg5 68.1%, oracle best-single 68.3% |
| Reasoning regime only (5 cells) | multiple | **78.3%** | — |

GPQA Diamond (MCQ science) is structurally out-of-regime: entropy dynamics are suppressed by the fixed-choice format. Phase 14 will compare against K=2 VC/SC/SCVC baselines (arXiv:2603.19118) on DeepSeek-R1-0528-Qwen3-8B.

---

## Completed Experiments

| Phase / Step | Description | Key result |
|-------------|-------------|-----------|
| Phase 1–3 (Steps 46–51) | Spectral features on GSM8K, 3 models | Best fusion 75.9% (Qwen-1.5B); sw_var_peak most robust |
| Phase 4 (Step 54) | MATH-500 + GPQA, 8 configs, T=1.5 | Honest best: 88.3% (Qwen-1.5B), 90.0% Qwen-7B T=1.0 |
| Phase 5 (Steps 56–58) | Temperature ablation T=1.0 vs T=1.5 | T=1.0 better for capable models; sw_var_peak temperature-stable |
| Phase 8 (Step 80) | GPQA / Qwen2.5-72B-AWQ | ~65% GPQA accuracy; spectral AUC modest — MCQ structurally limited |
| Phase 10 (Steps 85–91) | RAG — 4 models × 4 datasets (16 cells) | llama8b/hotpotqa **87.7%**, beats LOS-Net 72.92% unsupervised |
| Meta-analysis (Steps 89–91) | Cross-domain feature stability, 29-cell diagnostics | sw_var_peak most stable; epr dominant on math; rpdi/spectral_entropy on RAG |
| Steps 105–111 | Paper alignment: correct L-SML vs Step-100 leakage | Honest 65–91% math, 41–62% GPQA, 64–82% RAG (29 cells) |
| Steps 133–134 | 12-variant grid (CONT/PROD × encoding variants) | CONT best overall; encoding is the dominant lever; cross-weights always K=2 equal |
| Step 135 | Benchmarking vs SE/SC competitors | MATH-500: 94.4% vs 87.7%; GSM8K: 75.6% vs 78.5% |
| Step 137 | Theorem validation (branch `analysis/theorem-validation`) | HTML report + flowchart generated; **pending commit** |

### What we explored and moved on from

**Early EPR ensemble (Steps 27–45)**: 6-view Nadler ensemble (T=0.3/1.0/1.5/2.0 + Verify + Skeptic) on TriviaQA/WebQ with Falcon-3-10B — reached 81.5%/76.0%. This was the initial approach. We pivoted to spectral features because: (a) it requires 6 forward passes vs our 1, (b) the spectral framing is a cleaner contribution that connects to Ofir's methods.

**Supervised Step-100 numbers (96.6% MATH-500)**: had four methodological errors — (a) label-based sign orientation, (b) in-sample subset selection bias, (c) continuous features violating Lemma 1's binary contract, (d) M-matrix instead of rank-1 eigenvector. All corrected by Steps 105–110. The honest best is 90.0% (Qwen-7B, T=1.0).

**EDIS (Zhu et al. 2026, arXiv:2602.01288)**: Formula validated (Steps 41–42, spike ratio 4.02×). Not adopted as a core L-SML view — ρ(EDIS, spectral) too high on some cells. Useful as a comparison baseline only.

**Phase 13 (AMC23/AIME24)**: has `\boxed{}` grading bug — results invalid. Do not cite until fixed.

---

## Meeting Action Items — Jun 17, 2026 (Ofir, Bracha, Amir)

*Confirmed by email (Omri → Ofir/Bracha/Amir Jun 17 2026). These 6 items are the current experimental priority.*

| # | Action | Status | GPU? |
|---|--------|--------|------|
| 1 | L-SML follow-up literature search (Nadler post-2016) | Not started | No |
| 2 | Logistic regression oracle (5/9/16 features, 5-fold CV) | Not started | No |
| 3 | Extend QA evaluation (NQ, SQuAD v2, AmbigQA, PopQA) | Not started | Yes |
| 4 | Benchmarking completion (MATH-500 done; QA + Phase 14 remaining) | In progress | Partial |
| 5 | Experiment 1 — sampling fusion: SE (K=10) + spectral features | Not started | Yes |
| 6 | Experiment 2 — temperature variation: T effect + diversity ablation | Not started | Yes |

---

### Item 1 — L-SML Literature Search ✅ COMPLETED (Step 139)

**Result**: U-PCR (Tenzer, Dror, Nadler, Bilal, Kluger; AISTATS 2022 / arXiv:1703.02965) is Nadler's own continuous-input extension of L-SML. Under uncorrelated-error assumption, covariance off-diagonal C_ij = ρ_i + ρ_j − g² — recovers expert-response covariances ρ̂ without labels. Our CONT pipeline ≈ U-PCR; offline orientation = U-PCR's ρ̂_i exclusion criterion. **Cite Tenzer et al. (2022) in the thesis instead of "workaround for Lemma 1" language.**

Also found and deeply read (Step 141):

**FUSE** (Lee et al., arXiv:2604.18547, 2026) — applies Jaffe-Nadler moment structure to LLM verifiers for Best-of-N response selection with zero labels. Same theoretical base as our work (Jaffe et al. 2015). Different task: multi-response selection vs single-generation hallucination detection. Strong related-work citation. Critical finding for us: **our closed-form eigenvector weights (`w = (v₁ᵀρ̂/λ₁)·v₁`, then `score = w@F`) underperform naive equal-weight averaging in 7/10 FUSE benchmark settings** (Figure 3). FUSE's fix: pseudo-label logistic regression trained on MoM-estimated triplet posteriors `p̂(r_i)` — fully unsupervised (`p̂` never uses true labels). This is the single biggest available architectural upgrade to our pipeline. **Next experiment**: implement FUSE-style pseudo-label LR as replacement for `w@F` in `lsml_continuous_pipeline`.

**Deep L-SML** (Shaham et al., ICML 2016, arXiv:1602.02285) — Lemma 4.1 proves our L-SML IS already an RBM: Dawid-Skene model = single-hidden-node RBM (bijective parameter map). Our covariance+eigenvector step = closed-form MoM training of that RBM. Stacked RBM (Deep L-SML) handles correlated features without exclusion — each hidden layer decorrelates the representation. Relevant if 16-feature expansion triggers heavy ρ > 0.75 filter exclusions (band-power pairs ρ 0.77–0.88). Still fully unsupervised (objective = `log P(features)`, no labels).

**STDR** (Aizenbud et al., arXiv:2102.13276, 2021) — hierarchical tree-structured dependency recovery via Fiedler vector, O(m² log m). Not relevant at 5–16 features; revisit if feature set expands to 50+.

**Empirical confirmation (Step 140)**: U-PCR ≈ L-SML continuous on 5/9 features (low correlation regime, assumption holds); L-SML wins on 16 features (band-power block violates U-PCR's uncorrelated-error assumption; clustering compensates).

Implementation: `upcr_fuse()` + `upcr_pipeline()` added to `spectral_utils/fusion_utils.py`. Comparison script: `scripts/run_upcr_comparison.py`.

---

### Item 2 — Logistic Regression Oracle

Fit supervised logistic regression on our feature sets to upper-bound what any fusion method can extract from the same features.

**Setup**: 29 cached feature cells (`consolidated_results/features_all.pkl`); `sklearn.LogisticRegression` with 5-fold stratified CV; report macro AUROC for GOOD_5, STABLE_H9, all-16. Compare against CONT 70.1% (unsupervised), simple avg5 68.1%, oracle best-single 68.3%.

**Interpretation**:
- Gap < 5pp → L-SML is near-optimal; feature engineering is the bottleneck
- Gap > 15pp → significant headroom from supervision or better features

**Script**: `scripts/logistic_oracle.py`. No GPU needed.

---

### Item 3 — Extend QA Evaluation

Characterise method scope on factual QA by adding four datasets (priority order):
1. NaturalQuestions (CoT prompt, same setup as TriviaQA/WebQ)
2. SQuAD v2 (includes unanswerable questions — tests specificity)
3. AmbigQA (ambiguous queries test calibration near decision boundary)
4. PopQA (entity popularity confound)

**Setup**: same model (Qwen2.5-Math-7B or Falcon-3-10B), same spectral pipeline, N=200 samples, CONT fusion.

**Decision gate**: ≥3 of 4 datasets show CONT AUROC ≥ 65% → method extends credibly to factual QA domain.

---

### Item 4 — Benchmarking Completion

**Done (Step 135)**:
- MATH-500 / Qwen-Math-7B: CONT **94.4%** vs SE NLI 87.7% / SC 87.2% (K=10) ✅
- GSM8K / Llama-8B: CONT **75.6%** vs SC 78.5% / SE 77.4% (K=10) ✅

**Still needed**:
- **QA datasets**: SelfCheckGPT / Semantic Entropy comparison on same model + dataset (WebQ, TriviaQA)
- **GPQA Phase 14**: re-run Cell 9 with `DeepSeek-R1-0528-Qwen3-8B` at T=0.6. **Fix `boot_auc(n_boot=1000)` kwarg bug first.** Compare L-SML@K=1 vs VC/SC/SCVC@K=2 from arXiv:2603.19118:

| Method | K=2 AUROC |
|--------|----------|
| VC | 77.0 ± 2.0 |
| SC | 64.8 ± 3.0 |
| SCVC | 80.3 ± 1.5 |

Notebook: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb`.

---

### Item 5 — Experiment 1: Sampling Fusion

Fuse Semantic Entropy (K=10 generations) with single-pass spectral features.

**Primary question**: does SE K=10 (10× compute) add meaningful lift on top of CONT K=1?
**Secondary question**: does spectral (K=1) + SE (K=10) beat SE alone? — tests whether single-pass spectral adds orthogonal signal beyond the sampling budget.

**Dataset**: MATH-500 / Qwen2.5-Math-7B (T=1.0 cache exists) or GSM8K / Llama-8B.

**Decision gate**: ρ(SE score, CONT score) < 0.75 AND fused AUROC > max(CONT, SE) + 1pp → complementary signals; claim: "single-pass spectral provides cheap orthogonal signal to sampling-based methods."

---

### Item 6 — Experiment 2: Temperature Variation

**Questions from the meeting**:
1. Does higher temperature improve detectability? (Plot CONT AUROC vs T)
2. Does multi-temperature fusion gain from diversity or just from more passes?
   - **Condition A**: K=5 at T=1.0 (same T, more passes)
   - **Condition B**: K=5 at T∈{0.3, 0.6, 1.0, 1.5, 2.0} (different T, same K)
   - If B >> A: temperature diversity is the source of lift
   - If A ≈ B: multiple passes alone explain the gain; T doesn't matter

**Setup**: Qwen2.5-Math-7B / MATH-500. Existing caches: T=1.0 and T=1.5. New inference needed: T=0.3, 0.6, 2.0 + 4 additional T=1.0 runs (for Condition A ablation).

---

## Future Extensions

Not the current priority. Ordered by proximity to the main thesis.

### Extension A — Conformal Calibration (Bracha chapter)

Convert the AUROC result into a deployable detector with formal guarantees.

**A1 — Frozen-weights scorer + detection metrics under class imbalance** (engineering prerequisite for A2/A3)

Our cells are heavily imbalanced (GSM8K 79% majority, RAG/hotpotqa 91% majority) — raw accuracy is meaningless. Build:
- `fit_lsml(calibration_batch)` → freeze cluster assignment, group weights, cross-weights, per-feature μ/σ/sign. Unsupervised, fit once on a representative batch.
- `score_one(features)` → true single-sample inference (current experiments are transductive: fit+evaluate same batch, valid for AUROC but not streaming deployment).
- `decision_report(scores, labels, τ)`: recall (detection rate / TPR), precision, F1, balanced accuracy, TPR@FPR(1/5/10%), AUPRC.

**A2 — LTT calibration**: split calibration (100) + test (100); find threshold τ with P(FNR ≤ α) ≥ 1−δ.

**A3 — Label-free calibration via PPI**: use model-generated pseudo-labels (Verify > 0.9 → pseudo-correct) + PPI correction for pseudo-label noise.

### Extension B — Agentic Flow (Ofir alignment)

3-step HotpotQA agent chain; fuse per-step EPR with AUQ verbalized confidence (Zhang et al. 2026, arXiv:2601.15703).

Key check: ρ(EPR_step, verbalized_conf) < 0.5 → fusion is viable.
Target: Φmin AUROC > 0.791 (AUQ paper best on ALFWorld).
Model: Qwen3-7B. No new infrastructure for spectral features — same `generate_full()` per step.

### Extension C — Hidden State Variance (VSDE connection, Ofir alignment)

Register a forward hook on a transformer layer; compute variance of hidden states across K=5 temperature-varied generations as an additional L-SML view alongside spectral features.
- Low effort: one hook, existing fusion infrastructure
- Direct connection to Ofir's VSDE (high-variance regions ≈ hallucination) and PRAE

### Extension D — VLM Hallucination Detection

Apply spectral features to visual language models; split visual-description tokens vs factual-claim tokens. Not started. Only if committee wants a multimodal chapter.

---

## Recommended Priority Order

*(Single authoritative list — updated 2026-06-23, post Jun 17 advisor meeting)*

**Now — no GPU needed**
1. ~~L-SML literature search (Item 1)~~ ✅ done (Step 139)
2. Logistic regression oracle `scripts/logistic_oracle.py` (Item 2)

**Next Colab session**
3. Benchmarking: fix `boot_auc` kwarg → Phase 14 Cell 9 re-run (Item 4)
4. Sampling fusion: SE K=10 + CONT spectral (Item 5)
5. Temperature variation: T=0.3/0.6/2.0 inference + A/B ablation (Item 6)

**Subsequent Colab sessions**
6. Extend QA evaluation: NQ, SQuAD v2, AmbigQA, PopQA (Item 3)
7. Extension A (Conformal): A1 frozen scorer + imbalance metrics first, then A2 LTT

**Later**
8. Extension B (Agentic): Qwen3-7B, HotpotQA multi-hop
9. Extension C (Hidden states): one forward hook on Falcon
10. Extension D (VLM): only if committee wants multimodal chapter

**De-prioritized (valid but not blocking)**
- Step 132: MATH-500 SpilledEnergy GPU run — run opportunistically when Colab is free
- Merge decision (continuous L-SML → master): contingent on Step 132
- Phase 13 (AMC23/AIME24): fix `\boxed{}` grading bug before any re-run

---

## Thesis Narrative Thread

> *The per-token entropy trajectory H(n) is a signal, not a scalar. Collapsing it to its mean (EPR) discards temporal structure that predicts hallucination. Spectral features of H(n) recover that structure. L-SML fuses those features without labels, in a single forward pass. This gives a detector that is cheap (K=1), interpretable (spectral signal processing on an information-theoretic signal), and formally calibratable (the L-SML score is a continuous input to LTT). The thesis validates this on math reasoning, extends it to RAG and QA, and closes with a conformal chapter that turns the AUROC result into a deployment-ready detector with a formal false-negative-rate guarantee.*
