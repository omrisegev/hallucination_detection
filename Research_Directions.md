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
- **Scope on factual QA** — Item 3 extends to CoQA, SQuAD v2, TruthfulQA (priority corrected Step 155 — published SE/SC baselines exist)

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
| 1 | L-SML follow-up literature search (Nadler post-2016) | ✅ Completed (Steps 139–141) | No |
| 2 | Logistic regression oracle (5/9/16 features, 5-fold CV) | ✅ Completed (Steps 142–143, 147) | No |
| 3 | Extend QA evaluation (CoQA > SQuAD v2 > TruthfulQA — priority corrected Step 155) | In progress — actively running (Steps 160–169) | Yes (AIRCC) |
| 4 | Benchmarking completion (Phase 12 Corrected run done — Step 152; 4 open issues before citable; QA + Phase 14 remaining) | In progress — actively running (Steps 160–169) | Partial |
| 5 | Experiment 1 — sampling fusion: SE (K=10) + spectral features | ✅ Completed (Step 152) — gate NOT passed | No |
| 6 | Experiment 2 — temperature variation: T effect + diversity ablation | ✅ Completed (Step 158) — diversity hurts, same-T sampling helps | Yes |

---

### Item 1 — L-SML Literature Search ✅ COMPLETED (Step 139)

**Result**: U-PCR (Tenzer, Dror, Nadler, Bilal, Kluger; AISTATS 2022 / arXiv:1703.02965) is Nadler's own continuous-input extension of L-SML. Under uncorrelated-error assumption, covariance off-diagonal C_ij = ρ_i + ρ_j − g² — recovers expert-response covariances ρ̂ without labels. Our CONT pipeline ≈ U-PCR; offline orientation = U-PCR's ρ̂_i exclusion criterion. **Cite Tenzer et al. (2022) in the thesis instead of "workaround for Lemma 1" language.**

Also found and deeply read (Step 141):

**FUSE** (Lee et al., arXiv:2604.18547, 2026) — applies Jaffe-Nadler moment structure to LLM verifiers for Best-of-N response selection with zero labels. Same theoretical base as our work (Jaffe et al. 2015). Different task: multi-response selection vs single-generation hallucination detection. Strong related-work citation. Critical finding for us: **our closed-form eigenvector weights (`w = (v₁ᵀρ̂/λ₁)·v₁`, then `score = w@F`) underperform naive equal-weight averaging in 7/10 FUSE benchmark settings** (Figure 3). FUSE's fix: pseudo-label logistic regression trained on MoM-estimated triplet posteriors `p̂(r_i)` — fully unsupervised (`p̂` never uses true labels). This is the single biggest available architectural upgrade to our pipeline. **Next experiment**: implement FUSE-style pseudo-label LR as replacement for `w@F` in `lsml_continuous_pipeline`.

**Positioning against FUSE (Step 147, both Ofir and Bracha flagged it).** Three concrete differentiators, in decreasing order of importance: (1) **Signal** — we fuse spectral views of *one* model's own entropy/probability trace (internal, single-pass, no extra compute); FUSE fuses scores from *many external verifier models*. (2) **Task** — per-answer hallucination detection (absolute, across queries) vs within-query Best-of-N selection. (3) **Dependence handling** — FUSE detects dependent verifier pairs (triplet-conditional-independence violation) and *transforms* the scores so a single spectral fusion is well-conditioned; ours runs *K-group spectral clustering* then hierarchical within-/across-group fusion. Net: FUSE innovates on the fusion; our contribution is the **signal**, so the two are complementary. The thesis must foreground the entropy-trace signal, not "unsupervised spectral fusion," to avoid overlap. (Memory: `project-fuse-positioning`.)

**Deep L-SML** (Shaham et al., ICML 2016, arXiv:1602.02285) — Lemma 4.1 proves our L-SML IS already an RBM: Dawid-Skene model = single-hidden-node RBM (bijective parameter map). Our covariance+eigenvector step = closed-form MoM training of that RBM. Stacked RBM (Deep L-SML) handles correlated features without exclusion — each hidden layer decorrelates the representation. Relevant if 16-feature expansion triggers heavy ρ > 0.75 filter exclusions (band-power pairs ρ 0.77–0.88). Still fully unsupervised (objective = `log P(features)`, no labels).

**STDR** (Aizenbud et al., arXiv:2102.13276, 2021) — hierarchical tree-structured dependency recovery via Fiedler vector, O(m² log m). Not relevant at 5–16 features; revisit if feature set expands to 50+.

**Empirical confirmation (Step 140)**: U-PCR ≈ L-SML continuous on 5/9 features (low correlation regime, assumption holds); L-SML wins on 16 features (band-power block violates U-PCR's uncorrelated-error assumption; clustering compensates).

Implementation: `upcr_fuse()` + `upcr_pipeline()` added to `spectral_utils/fusion_utils.py`. Comparison script: `scripts/run_upcr_comparison.py`.

---

### Item 2 — Logistic Regression Oracle ✅ COMPLETED (Steps 142–143, 147)

Fit supervised logistic regression on our feature sets to upper-bound what any fusion method can extract from the same features.

**Setup**: 28 common LR-valid cells; `sklearn.LogisticRegression(class_weight='balanced')`, 5-fold stratified CV with **per-fold AUROC averaging** (not concatenated OOF — see `SUPERVISED_ORACLE_CORRECTION.md`).

**Result (Step 147, common-cell macro AUROC)** — supervised LR beats unsupervised L-SML everywhere once corrected:

| Feat set | L-SML (CONT) | LR bal-CV | gap | in-sample ceiling |
| :-- | :-: | :-: | :-: | :-: |
| GOOD_5 | 64.2% | 68.9% | +4.7pp | 70.5% |
| STABLE_H9 | 62.9% | 66.8% | +3.8pp | 73.7% |
| ALL_H16 | 64.1% | 67.8% | +3.6pp | 79.3% |

- **Per-domain**: gap ~0 on reasoning (both near the ~84% ceiling), +4.9pp GPQA (ceiling 60.9%), +5.8pp RAG+QA (ceiling 69.5%). The gap is largest exactly where the feature ceiling itself is low → **features are the bottleneck, not the fusion**. This lands in the "< 5pp on reasoning / moderate elsewhere" interpretation band: L-SML is near-optimal where the signal exists.
- **"5 features best" explained** (`scripts/lr_convergence.py`): the named sets are non-nested (STABLE_H9 drops `spectral_entropy`, a top-3 feature), and in a proper nested ranked sweep the CV is flat from k=5 to k=16 (~68–69.5%) while the in-sample ceiling climbs to 79.3% — the extra features overfit rather than generalise. The same 9-feat dip appears in the unsupervised L-SML, so it is a feature-composition effect, not a supervision artifact.
- **LR vs L-SML weights** (`scripts/lr_weight_analysis.py`, answers Bracha Q4): correlate only weakly (Spearman ≈ 0.1–0.2, ~0.32 on GPQA). Both lean on epr/spectral_entropy/cusum_max but weight them differently — the features are correlated/redundant, so the weighting is underdetermined and both reach similar AUROC through different routes.

**Scripts**: `scripts/logistic_oracle.py` (oracle + `logistic_oracle.png`), `scripts/oracle_report.py` (common-cell tables + `oracle_feature_count.png`), `scripts/lr_convergence.py` (`lr_convergence.png`), `scripts/lr_weight_analysis.py` (`lr_weight_agreement.png`). No GPU needed.

---

### Item 3 — Extend QA Evaluation

**Priority corrected (Step 155)** — pick datasets with published SE/SC baselines so results are directly comparable (AmbigQA/PopQA have none):
1. CoQA (SE-ICLR primary dataset — 8K dev, published SE numbers; INSIDE 80.4 EigenScore reference)
2. SQuAD v2 (includes unanswerable questions — tests specificity; INSIDE reference 81.5)
3. TruthfulQA (hallucination-specific benchmark; LapEigvals + HSAD references)

**Setup**: folded into the Step-155 replication grid — AIRCC inference-only presets per competitor protocol (K, T, prompt, labeling locked per paper), all scoring local CPU. Loaders (CoQA, SQuAD v2, NQ-Open, TruthfulQA, SciQ) are the implementation follow-up.

**Decision gate**: ≥3 of 4 datasets show CONT AUROC ≥ 65% → method extends credibly to factual QA domain.

---

### Item 4 — Benchmarking Completion

**Done (Step 135, old caches)**:
- MATH-500 / Qwen-Math-7B: CONT **94.4%** vs SE NLI 87.7% / SC 87.2% (K=10) ✅
- GSM8K / Llama-8B: CONT **75.6%** vs SC 78.5% / SE 77.4% (K=10) ✅

**Done (Step 152, Phase 12 Corrected — fresh shared caches, paper-accurate baselines)**:
- GSM8K / Llama-8B: **L-SML 1-pass 0.754 beats every multi-pass baseline** (SCGPT-official 0.701; D-SE/LW-SE/SC K=10 all ≈0.61). Third run at 75.4–76.0.
- MATH-500 / Qwen-Math-7B: L-SML 0.230 = global sign flip (no `anchor_orient`; flipped ≡ 0.770 — still far below the 94.4 old-cache number, unresolved). SC K=10 wins at 0.863.
- GPQA / Qwen2.5-7B: all sampling baselines at chance (0.50); VC 0.428; L-SML 0.553 best.
- RAG×4: SelfCheckGPT below chance everywhere (official 0.24–0.44 < hard 0.32–0.48) — orientation/grading investigation needed.
- ⚠ Fresh-cache SE/SC baselines collapse vs old Phase 12 (GSM8K SC 78.5→60.8, SE 77.4→61.4; GPQA SE 70.6→50.1; MATH SE 87.7→63.0 with SC stable). NLI truncation on long traces is the prime suspect. **Neither table is citable until reconciled** — see PROGRESS.md Priority 1.

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

**✅ COMPLETED (Step 152) — gate NOT passed.** Fusion = L-SML GOOD_5 + LW-SE as 6th view in `lsml_continuous_pipeline`, run inside Phase 12 Corrected:

| Cell | ρ(L-SML, LW-SE) | L-SML alone | LW-SE alone | Fused | Gain vs max |
|------|-----------------|-------------|-------------|-------|-------------|
| GSM8K / Llama-8B | 0.263 | 0.754 | 0.613 | 0.758 | +0.4pp — FAIL |
| MATH-500 / Qwen-Math-7B | −0.251 | 0.230 (sign flip) | 0.625 | 0.232 | invalid (flip) |
| GPQA / Qwen2.5-7B | −0.188 | 0.553 | 0.501 | 0.573 | +2.0pp — passes numerically, but LW-SE is at chance |

- **Primary answer**: SE K=10 (10× compute) adds ≈nothing on top of 1-pass spectral on reasoning.
- **Secondary answer**: the orthogonality runs the other way — spectral adds **+14.5pp** on top of LW-SE (GSM8K). Supports the "cheap single-pass signal" framing, but as spectral rescuing SE rather than SE lifting spectral.
- MATH-500 fusion must be re-run with `anchor_orient` before the row is usable (PROGRESS.md Priority 1).

---

### Item 6 — Experiment 2: Temperature Variation

**Questions from the meeting**:
1. Does higher temperature improve detectability? (Plot CONT AUROC vs T)
2. Does multi-temperature fusion gain from diversity or just from more passes?
   - **Condition A**: K=5 at T=1.0 (same T, more passes)
   - **Condition B**: K=5 at T∈{0.3, 0.6, 1.0, 1.5, 2.0} (different T, same K)
   - If B >> A: temperature diversity is the source of lift
   - If A ≈ B: multiple passes alone explain the gain; T doesn't matter

**Setup**: Qwen2.5-Math-7B / MATH-500. ~~Existing caches: T=1.0 and T=1.5~~ — **claim corrected (Step 157)**: no reusable raw cache exists for this cell. The T=1.5 88.3% cell is Qwen-**1.5B**; Step 148 established MATH-500/Qwen-7B has no raw entropy-trace cache anywhere; Phase 12 Corrected `p2` predates the Step-149/150 grading fixes and has no top-k logprobs. → **All 9 runs fresh** (5 temps + 4 extra T=1.0), each saving the full raw-data schema. T=1.0 run0 doubles as the canonical raw-trace cache for this cell, repaying the Extension E data debt.

**Status (Step 158) — ✅ RAN on Colab A100. Both pre-registered gates FAIL; the negative result is clean and interpretable.**

Results (9 runs × 200 MATH-500 / Qwen2.5-Math-7B; full narrative in HISTORY Step 158; consolidated `cache/phase15_temperature/results/phase15_results.pkl`):

- **Q1 — AUROC vs T (single-pass L-SML-continuous, GOOD_5)**: inverted-U — 0.545 / 0.644 / 0.851 / 0.878 / 0.629 at T = 0.3 / 0.6 / 1.0 / 1.5 / 2.0 — **confounded by accuracy collapsing 80% → 4%** across the curve, so the "peak" partly reflects the shifting class mix, not detectability alone. T=2.0 is underpowered (8 correct). **G-T1 FAIL** (T=1.5's higher 0.878 has overlapping CIs and sits at 27.5% acc).
- **Q2 (primary) — diversity vs more passes**, paired on the 200 common samples (labels = T=1.0 run0):
  - **AUC(A: K=5 same-T=1.0) = 0.912**, **AUC(B: K=5 multi-T) = 0.859**, single-pass base 0.851.
  - paired **AUC(B) − AUC(A) = −0.053 [−0.103, −0.011]** → **G-T2 FAIL, sign negative** — temperature diversity *hurts*.
  - paired **AUC(A) − AUC(base) = +0.061 [+0.004, +0.128]** → more same-T passes *help*.
  - Mechanism: A off-diagonal Spearman ρ +0.45 (same signal + independent noise → averaging denoises); B off-diagonal ρ +0.01, but that decorrelation is the off-temperature passes being *near-random* (T=0.3/0.6 weak, T=2.0 degenerate), not independent true signal.
  - **Answer to the meeting question**: A ≈ B is refuted in the *unfavourable* direction — the multi-pass lift is **variance reduction from repeated sampling at a single good temperature (T≈1.0)**, and mixing temperatures dilutes it. Temperature is not the lever; repeated sampling is.
- **Two method flags surfaced (not fatal, → follow-up)**: (1) `spectral_entropy` sign is temperature-dependent — AUROC 0.261 @ T=1.0 / 0.140 @ T=1.5 with the fixed −1 sign (i.e. informative if flipped); (2) the label-free L-SML fusion **underperforms the best single feature at every T** (fused 0.851 vs `cusum_max` 0.927 @ T=1.0; fused 0.545 vs `cusum_max` 0.811 @ T=0.3) because the `epr` anchor is weak at low T (0.681 @ T=0.3) → fragile global-sign orientation. The low-T "poor detectability" in Q1 is plausibly a fusion/anchor artifact, not a signal property.

**Data-debt repaid**: T=1.0 run0 is now the **canonical MATH-500/Qwen-7B raw-trace cache** (entropies + spilled energies + top-50 logprobs + token ids, N=200, 70.5% acc) — closes the Extension E gap.

**Follow-up experiments on this data — all CPU once the 9 caches are downloaded** (prioritised):

1. **Self-consistency / semantic-entropy baseline** (highest value; also closes Item 5). We have 5 T=1.0 full-text generations per question → extract final answers, compute answer-agreement / cluster (semantic) entropy = the standard sampling-based confidence. Answers the reviewer-mandatory *"does spectral add anything over just sampling 5× and voting?"* and whether SC ⊕ spectral is complementary (ρ < 0.75, fused > max + 1pp).
2. **K-sweep for Condition A**: AUROC(A) at K = 1..5 — does the same-T lift saturate at K=3? A practical cost/benefit curve from data already in hand.
3. **Anchor / sign robustness across T**: re-fuse with (a) a stronger, more T-stable anchor (`cusum_max`), (b) per-feature label-free sign via each feature's own anchor, (c) leave-`spectral_entropy`-out — quantify how much of the low-T gap and the fusion-vs-best-single gap is recoverable. Directly tests whether Q1's low-T dip is real.
4. **New feature families from saved-but-unused data**: (a) run the spectral suite on the **ΔE spilled-energy** trace (saved for all 9 runs, never used) and test orthogonality to H(n); (b) **top-50 logprob** features — top1−top2 margin, varentropy, Rényi entropy at several orders, tail mass; recompute entropy at any K.
5. **Fairer diversity set**: re-run B dropping the degenerate T=2.0 (and maybe T=0.3), e.g. B′ = {0.6, 1.0, 1.5} — confirms the negative Q2 is robust and not an artifact of including the useless hot pass.
6. **Cross-temperature probing**: does a hot pass's entropy trace predict the *cold* (T=1.0) answer's correctness? (Sample hot to probe uncertainty, evaluate the cold answer.) Uses the index-aligned runs.
7. **Length-controlled AUROC per T**: hot traces are longer/degenerate — partial out trace length to confirm the spectral signal isn't just length.
8. **Streaming earliest-prefix replication (Extension E)** — now unblocked by the fresh raw cache; run absolute-budget prefixes on the T=1.0 run0 traces.

A couple (K-sweep beyond K=5, more temperatures for the pooling curve) would need a small extra GPU run; everything else is local CPU.

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

### Extension E — Streaming / Online Detection (pivot candidate — pilot ✅ COMPLETED, Step 148)

**Status**: Pilot run 2026-07-02, local CPU, pre-registered gates. **Verdict: pivot NOT supported in its original framing (G2 FAIL); one significant surviving thread.** Full narrative: HISTORY.md Step 148; explainer: `results/Streaming_Pilot_Explainer.html`.

**Hypothesis**: the spectral suite computed on growing prefixes of H(n) detects a failing CoT *while it is generated* — unsupervised, logprob-only — and beats a naive windowed statistic in that streaming regime.

**Competitor** (closest prior work): *Streaming Hallucination Detection in Long CoT Reasoning*, arXiv:2601.02170 (BUPT/NTU/SWJTU/RUC, **arXiv preprint Jan 2026**, no venue as of Jul 2026). SUPERVISED probes over intermediate **hidden states** (anchor + synchronization losses), step labels annotated by Claude-4.5; custom MuSiQue-derived long-CoT set (10k+ trajectories / 200k+ steps). Prefix-level AUC: LLaMA-3.1-8B 72.69 / Qwen2.5-7B 81.05 / R1-Distill-8B 92.18. Their own limitations: "not directly applicable to black-box or API-only settings" — exactly our setting. **Reproducible baseline**: DeepConf (arXiv:2508.15260, Meta, Aug 2025) lowest-group-confidence — black-box, computable on our cached traces, hence the primary bar (G2).

**Pilot results** (2 clean cells: GSM8K/Llama-8B n=200, MATH-500/Qwen-1.5B n=400 non-canonical; 2 R1/GPQA cells excluded — 99–100% truncated at 1024-token cap):
- **G1 PASS** — AUROC@50%-of-trace ≥ 95% of full-trace on both clean cells; 32 tokens ≈ 91% of full signal on GSM8K. Early signal is real.
- **G2 FAIL** — fused L-SML does not clear +2pp over the best DeepConf window at ≥2 absolute budgets on ≥2 clean cells. Over most of the trace, the fusion ≈ windowed entropy mean.
- **Surviving thread** — the only *significant* spectral edge (paired bootstrap) is in the **earliest 10% of the trace, on both clean cells**: +9.8pp GSM8K, +4.6pp MATH-500. Fusion helps exactly where windows starve.
- **G3 context** — our unsupervised GSM8K/Llama-8B 75.4 (L-SML-5) vs their supervised hidden-state 72.69 on the same model family (different benchmark + label protocol; context only).
- **E3/E4** — best causal monitor flags 38% of wrong GSM8K traces @10% FA, saving 28% of wasted tokens.

**Data debt exposed**: MATH-500/Qwen-7B (our ~90% cell) has NO raw-trace cache anywhere (Phase-12 K10 files are texts-only); no clean R1 cell exists (all capped at 1024 mid-`<think>`).

**Next steps (in order)**:
1. Colab re-inference: MATH-500/Qwen-7B + one R1 cell with ≥4096-token cap, saving `token_entropies` + top-50 logprobs (raw-data rule).
2. Replicate the earliest-prefix edge there — absolute budgets only (fractions need oracle length), n large enough for the paired test.
3. If replicated → reframe as **hybrid early-warning monitor** (spectral early / windowed late), not "fusion wins everywhere" (G2 refutes that).
4. Method: per-budget refusion is sign-unstable (anchor_orient mitigates; 16-feat still erratic) → fit fusion weights once at a reference budget offline, reuse across budgets.
5. Advisor decision: pursue hybrid framing vs fold streaming in as a thesis section.

### Extension F — Step-Level Error Localization (ProcessBench / MR-GSM8K) — DEFERRED (2026-07-10)

The July-2026 SOTA survey recommends a process-level benchmark as a secondary evaluation for
reasoning-focused detectors: **ProcessBench** (arXiv 2412.06559 — 3,400 expert-annotated cases
across GSM8K/MATH/OlympiadBench/Omni-MATH with first-error-step labels, F1 metric) or
**MR-GSM8K** (arXiv 2312.17080). This is a different task from our sequence-level AUROC
detection — it asks *where* the reasoning breaks, not *whether* the answer is wrong.

**Why it fits us structurally**: our sliding-window features (`sw_var_peak_with_window` keeps the
window index) and CUSUM drift (`cusum_shift_idx` is literally a change-point location) are
naturally step-localizable — a per-step L-SML score is a modest extension, not a redesign.

**Why deferred (Omri, 2026-07-10)**: keeps the current benchmarking pass focused on AUROC
head-to-heads; step-level would need a new grading harness (their provided solutions, not our
generations), a step-alignment layer (token index → solution step), and an F1 protocol. Revisit
after the reasoning replication grid completes, if a reviewer or committee member asks for
error localization.

---

## Recommended Priority Order

*(Single authoritative list — updated 2026-07-02, post streaming pilot Step 148)*

**Now — no GPU needed**
1. ~~L-SML literature search (Item 1)~~ ✅ done (Step 139)
2. ~~Logistic regression oracle `scripts/logistic_oracle.py` (Item 2)~~ ✅ done (Steps 142–143, 147)
3. ~~Streaming pivot pilot (Extension E)~~ ✅ done (Step 148 — G1 PASS / G2 FAIL; earliest-prefix edge is the surviving thread)
4. Present streaming pilot verdict to advisors → decide hybrid framing vs thesis section (Extension E step 5)

**Next Colab session**
5. Benchmarking: fix `boot_auc` kwarg → Phase 14 Cell 9 re-run (Item 4)
6. **Raw-trace regeneration** (Extension E step 1): MATH-500/Qwen-7B + one R1 cell with ≥4096-token cap, saving `token_entropies` + top-50 logprobs — unblocks the earliest-prefix replication AND repays the raw-data debt
7. Sampling fusion: SE K=10 + CONT spectral (Item 5)
8. Temperature variation: T=0.3/0.6/2.0 inference + A/B ablation (Item 6)

**Subsequent Colab sessions**
9. Streaming earliest-prefix replication on the regenerated cells (Extension E steps 2–3; local CPU once traces exist)
10. Extend QA evaluation: CoQA > SQuAD v2 > TruthfulQA (Item 3, priority corrected Step 155 — runs on AIRCC as part of the replication grid)
11. Extension A (Conformal): A1 frozen scorer + imbalance metrics first, then A2 LTT

**Later**
12. Extension B (Agentic): Qwen3-7B, HotpotQA multi-hop
13. Extension C (Hidden states): one forward hook on Falcon
14. Extension D (VLM): only if committee wants multimodal chapter

**De-prioritized (valid but not blocking)**
- Step 132: MATH-500 SpilledEnergy GPU run — run opportunistically when Colab is free
- Merge decision (continuous L-SML → master): contingent on Step 132
- Phase 13 (AMC23/AIME24): fix `\boxed{}` grading bug before any re-run

---

## Thesis Narrative Thread

> *The per-token entropy trajectory H(n) is a signal, not a scalar. Collapsing it to its mean (EPR) discards temporal structure that predicts hallucination. Spectral features of H(n) recover that structure. L-SML fuses those features without labels, in a single forward pass. This gives a detector that is cheap (K=1), interpretable (spectral signal processing on an information-theoretic signal), and formally calibratable (the L-SML score is a continuous input to LTT). The thesis validates this on math reasoning, extends it to RAG and QA, and closes with a conformal chapter that turns the AUROC result into a deployment-ready detector with a formal false-negative-rate guarantee.*
