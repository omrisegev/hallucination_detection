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
| 3 | Extend QA evaluation (CoQA > SQuAD v2 > TruthfulQA — priority corrected Step 155) | Nearly complete (Steps 160–171): CoQA/INSIDE full-N scored 68.4 vs 80.4 (floor caveat, Step 171); SQuAD v2 / NQ-Open / TruthfulQA / SciQ scored | Yes (AIRCC) |
| 4 | Benchmarking completion (Phase 12 Corrected run done — Step 152; 4 open issues before citable; QA + Phase 14 remaining) | In progress (Steps 160–171): one cell left in flight (ars_math500_qwen3 wall 3/4); A2 qwen3-GSM8K documented REJECT (ceiling + truncation leakage); report now carries CSV-driven figures (report_figs.py) + LOS-Net Table-1 baseline family incl. p(True) | Partial |
| 5 | Experiment 1 — sampling fusion: SE (K=10) + spectral features | ✅ Completed — verdict REVISED (Step 174): answer-agreement SC K=5 fuses with 1-pass L-SML to 95.2 [91.8, 98.0] (ρ +0.23) — gate PASSES; the Step-152 FAIL was the NLI-based arm | No |
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

1. **Self-consistency / semantic-entropy baseline** (highest value; also closes Item 5). ✅ **DONE (Step 174)** — answer-agreement SC K=5 fused with 1-pass L-SML → 95.2 [91.8,98.0], gate PASS (ρ +0.23, +10.1pp over best single arm).
2. **K-sweep for Condition A**: AUROC(A) at K = 1..5 — does the same-T lift saturate at K=3? ✅ **DONE (Step 181)** — 0.851/0.869/0.863/0.905/0.912 for K=1..5; no early saturation, a dip at K=3, most of the lift arrives at K=4-5. Repeated sampling needs close to the full K=5 budget on this cell.
3. **Anchor / sign robustness across T**: re-fuse with (a) a stronger, more T-stable anchor (`cusum_max`), (b) per-feature label-free sign via each feature's own anchor, (c) leave-`spectral_entropy`-out. **BLOCKED (Step 181)** — needs raw per-sample GOOD_5 feature values at T≠1.0; `phase15_results.pkl` only stores scalar per-(feature,temp) AUROCs. Needs `math500_qwen7b_T{0.3,0.6,1.5,2.0}_run0.pkl` copied from Drive to `local_cache/`.
4. **New feature families from saved-but-unused data**: (a) spilled-energy suite; (b) top-50 logprob features (margin, varentropy, Rényi entropy, tail mass). ✅ **DONE (Step 181)** — `cusum_max_spilled` (AUROC 0.909) fused as a 6th view clears the Item-5-style gate (+1.13pp over GOOD_5, CI excludes 0) — a second genuine complementary signal. New `topk_tail_mass`/`varentropy`/`renyi_entropy_2` logprob features added (`repgrid_scoring.logprob_features_extended`); `topk_tail_mass` (AUROC 0.902) fusion gain +0.72pp is CI-significant but below the 1pp gate (near-miss).
5. **Fairer diversity set**: re-run B dropping the degenerate T=2.0 (and maybe T=0.3), e.g. B′ = {0.6, 1.0, 1.5}. ✅ **DONE (Step 181)** — B′ simple-avg 0.881 / L-SML 0.856, statistically indistinguishable from a matched K=3 same-T arm (0.863; CI spans 0). Confirms the negative Q2 result isn't an artifact of the degenerate passes.
6. **Cross-temperature probing**: does a hot pass's entropy trace predict the *cold* (T=1.0) answer's correctness? ✅ **DONE (Step 181)** — every hot T predicts its OWN label far better than the COLD label (e.g. T=1.5 own 0.878 vs cold 0.626; T=0.3 own 0.545 vs cold 0.388, anti-predictive) — mechanistic reason mixing temperatures hurts fusion: each pass's signal is entangled with its own generation, not a stable per-question difficulty read.
7. **Length-controlled AUROC per T**: hot traces are longer/degenerate — partial out trace length to confirm the spectral signal isn't just length. **BLOCKED (Step 181)** — same data gap as #3 (needs per-sample trace length at T≠1.0).
8. **Streaming earliest-prefix replication (Extension E)** — now unblocked by the fresh raw cache; run absolute-budget prefixes on the T=1.0 run0 traces. Still open — not attempted.

Items 2/4/5/6 results: `results/repgrid/phase15_followups.json` (Step 181). A couple (K-sweep beyond K=5, more temperatures for the pooling curve) would need a small extra GPU run; everything else is local CPU.

---

## Meeting Action Items — Jul 2026 (Ofir, Bracha)

*Ofir and Bracha were pleased with the results shown; FUSE is not considered blocking. The one
concrete action item from this meeting: add a new contribution to the algorithm, and the chosen
candidate is a principled, label-free, in-pipeline **feature-subset selection step** — see Extension G
below. Bracha also raised conformal calibration; **explicitly parked** (still Extension A, unchanged
priority).*

| # | Action | Status |
|---|--------|--------|
| 1 | Feature-subset selection step: literature survey (Lindenbaum FS line, Nadler portfolio, tabular foundation-model frontier, assumption diagnostics) + assumptions audit + candidate designs | ✅ Research memo complete (Step 185) — `docs/research_notes/feature_subset_selection_landscape.md`; see Extension G |

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

### Extension G — Automatic Feature-Subset Selection (meeting priority, Jul 2026)

**Status**: Memo (Step 185) → **full multi-algorithm bench EXECUTED (Step 186, 2026-07-17/18)** →
**punch-list follow-ups + split-half honest oracle (Step 189, 2026-07-18) — see below, the motivating
+7.6pp prize itself is now known to be mostly winner's-curse** —
six label-free selector families implemented + benched on both pools (H16 51 cells, 46-view 19
repgrid cells) through one select→same-L-SML→AUROC harness with labels structurally unreachable
during selection. All results in `results/selector_bench/comparison.csv` + the dashboard
(`results/selector_bench/dashboard.html`); research note
`docs/research_notes/selector_bench_results.md`; no pass/fail gatekeeping — the researcher reads
the full leaderboard.

**UPDATE (Steps 194-195, 2026-07-22) — two supersessions of the Step-186 verdict below:**
1. **Best learned selector is now `a6.pl_dufs`** (pseudo-label-supervised gates, Omri's idea):
   macro 0.7524 on the 25 in-scope cells, significantly better than `a2.dufs` (+0.22pp,
   p=0.0273) and the FIRST label-free selector to nominally edge GOOD_5 (0.7519, p=0.17 n.s.).
   Both pre-registered gates FAILED (mechanism rho 0.207 vs 0.30 bar; effect below +1.0pp), so
   the claim stays "GOOD_5 parity, GOOD_6 gap not closed" — but it is the selector of record.
2. **A fixed subset that honestly beats GOOD_6 exists.** The sizes-3-5 exhaustive sweep over
   the 30-view pool (Step 194/195, `results/subset_sweep_c46/`) yields a LOCO consensus stable
   in 22/25 folds: `{cusum_max, logprob_margin, min_energy, spectral_entropy, topk_tail_mass}`
   — LOCO-honest +1.59pp vs GOOD_5 (19W/2L); vs GOOD_6 on the same 24 cells 0.7705 vs 0.7632
   (+0.73pp, p=0.029), sign label-free. Coverage 24/25 (`inside_coqa_llama7b` lacks the energy
   views). This REVERSES the Step-154 "LOCO cannot beat GOOD_5" verdict — the enlarged pool
   changed the answer. Pruning stays negative (LOCO drop list empty in all folds).

**UPDATE (Step 198, 2026-07-24) — the selection line is now measured out, and the bottleneck is renamed:**

3. **`GOOD_6` is unbeaten by every label-free selector, and it is a local optimum.** Post-fix
   seven-arm bench on 25 in-scope cells (`results/advisor_inscope/seven_arm_summary.csv`, one run,
   canonical `eval_subset_flex`): GOOD_6 0.7594 > D1_D2 0.7580 > D2 (PL-mRMR) 0.7573 >
   `a6.pruned_dufs` 0.7537 > `a6.pl_dufs` 0.7527 > GOOD_5 0.7519 > D1 0.7506. D2 beats GOOD_5
   significantly (p=0.037) and beats every prior DUFS variant, but under LOCO-CV budget selection
   lands 0.7572, below GOOD_6, and its math edge is p=0.2114 (9W/6L). The best D2 configuration is
   GOOD_6 **plus one** selected feature at 0.7590, i.e. adding any selected feature to GOOD_6 hurts
   macro at every budget K=7..20 even with the budget chosen on test data.
4. **Adaptive-K (D1) is refuted.** Five label-free size rules tested against oracle K
   (`results/advisor_inscope/adaptive_k_validation_rules.csv`): best rule r_s = +0.007, p = 0.975.
   The residual correlating with AUROC (r=0.65) does not transfer to predicting the optimal size.
   `D1_alone` is the worst of seven arms. Closed.
5. **The one real win of the step is the pseudo-label seed rule**: `ANCHOR_PRIORITY`x4 -> `GOOD_6`
   (`A6_SEED_RULE` env, default `good6`) takes the pseudo-label from 0.7249/0.6821 QA to
   0.7594/0.7274 QA and removes 2 sign-inverted cells. A weak consensus target points the gates at
   the wrong features, it does not merely add noise.
6. **The bottleneck is estimation, not model capacity, and the QA deficit is two cells.** The
   per-cell supervised oracle is logistic regression, i.e. a stationary global linear model with
   fixed per-feature signs, and it reaches 0.7810 macro / 0.7524 QA on the same 30 features
   (`lr_oracle_audit.csv`, `fset=30`). So the linear class already contains a solution above us.
   The QA gap concentrates in `inside_coqa_llama7b` (0.667 vs oracle 0.826, INSIDE publishes 0.804:
   **estimation failure**) and `seiclr_triviaqa_opt30b` (0.588 vs oracle 0.720 while SE publishes
   0.830: **feature-coverage failure that no fusion change can fix**). The other 8 QA cells average
   ~0 gap to the supervised oracle. This kills the "stationary sign bottleneck" framing and the
   three methods proposed on top of it (regime-conditional signs, SNF, GMM density ratio).
7. **Next**: `SPEC_gap_ladder.md` (repo root) specifies a 7-rung gap-decomposition ladder at two
   feature sets with pre-registered kill-gates: `R3->R4` (supervised nonlinear vs supervised linear)
   kills the nonlinear directions if flat, `R3->R5` (oracle regime signs) kills the non-stationary
   sign direction. Both run with labels, so a negative is conclusive. Gemini implements
   `scripts/gap_ladder.py`, Claude reviews and analyses. Candidate follow-ons if sign recovery
   dominates: Z2 synchronisation on the pairwise-sign matrix (`sign(cov_ij)` estimates `s_i*s_j`)
   and a robust (Spearman / Tyler) covariance in place of Pearson.

**Step-186 outcome (headline numbers, superseded as above)**:
- **No learned selector beats the curated subsets.** c46/repgrid-19 macro: GOOD_6 0.7440 >
  top_macro_5 0.7364 > GOOD_5 0.7328; best learned = **GroupFS `a2.select` 0.7323 — a
  label-free TIE with GOOD_5** (first learned selector to reach it); everything else trails by
  1-6pp. On H16/51-cell every learned family lands 0.56-0.63 vs GOOD_5 0.671.
- **Pre-registered admissibility (A1.0)**: no label-free objective is globally admissible as a
  selection criterion; the relative Eq-14 residual is weakly admissible on repgrid/qa only
  (median Spearman −0.109/−0.17); the lsml-vs-upcr structural-residual router is NOT-USEFUL in
  every domain (worse than best-constant by 3-6pp). The ρ-filter refutation (Step 153)
  replicates as a family-wide pattern.
- **Clustering swap** (theorem-validation follow-up): GroupFS's discovered groups replacing
  L-SML's spectral clustering ≈ tie on GOOD_5 (0.717-0.728 vs 0.733) — clustering is not the
  bottleneck on the repgrid pool.
- The **+7.6pp RAG/GPQA oracle prize remains uncaptured** by every label-free method tried.

**Step-189 correction — the prize itself was mostly winner's-curse.** A split-half honest oracle
(`scripts/selector_splithalf_oracle.py`: bounded greedy search on held-out half A, refit + scored on
half B, R=10 splits × 51 cells) found the 0.7472-macro exhaustive-sweep oracle **collapses to 0.668
macro when fully honest — a statistical TIE with GOOD_5 (0.6692) on the identical splits.**
Per-domain, this lands exactly on the two domains the "+7.6pp prize" framing above was built on:
**RAG's claimed +14.1pp shrinks to ~+1.6pp honestly; GPQA's claimed +10.2pp shrinks to ~+1.6pp
honestly.** This retroactively explains the uniform Step-186–189 negative results (six selector
families, A1–A5, all failing to beat GOOD_5): the 65,536-subset exhaustive search (Step 153)
guarantees a large multiple-comparisons overfit at n≈100–500 per cell, so "no selector captures the
prize" was never really a selector-design failure. **Also this session**: an autopsy of `a2.select`'s
one catastrophic miss (`inside_coqa_llama7b`, −14pp) found GroupFS's gates saturate open (selects
100% of a 23-feature pool containing 7 anti-oriented features) on a severely imbalanced, small-n-style
cell — connects directly to the still-open Step-187 feature-sign-fix item. An mRMR hybrid (A5,
`spectral_utils/selectors/a5_mrmr.py`) salvages part of A4's "picks epr's clones" pathology on the
46-view pool (+0.57pp over bare epr) but not on H16, and still doesn't clear GOOD_5. Full detail:
HISTORY Step 189; `docs/research_notes/selector_bench_results.md` (split-half section).

**Next steps (revised)**: the selection direction's motivating premise needs re-scoping with
Ofir/Bracha — realistic honest headroom looks like ~1–2pp, not ~7–8pp, which changes whether further
selector-design investment is worthwhile at all. Before any further design work: (0) the Step-187
feature-sign fix (13/30 anti-oriented features) is the one still-open, concrete, likely-cheap win,
independent of the selection-prize question — do this first, it may already close some of the small
residual gap on its own. If selection work continues: (i) GroupFS on the 46-view pool remains the
best label-free tie-with-GOOD_5 result and needs no further justification to ship as a deployable
default; (ii) the D5-(ii) cross-cell signature router is the one design from the original memo never
attempted — lower priority now given (2) above suggests little headroom exists to route toward.

**Motivation**: 46 registered fusion features (`CANONICAL_POOL`); no fixed macro wins consistently —
GOOD_5, the documented main configuration, wins only 3/40 per-cell picks in the repgrid headline
comparison. In-cell oracle subset selection is worth **+7.6pp macro AUROC** over fixed GOOD_5 (0.747
vs 0.671, 51-cell sweep), concentrated in RAG (+14.1pp) and GPQA (+10.2pp) — but LOCO (leave-one-cell-
out) subset transfer is flat (0.664 vs 0.674), so **the prize is only reachable by an in-cell,
label-free selection mechanism**, not a domain lookup table.

**Approach**: follow the FUSE precedent (Candès et al., arXiv:2604.18547) — turn a label-free
assumption-violation statistic into a selection objective, the same move FUSE makes for verifier
binarization thresholds. Full assumptions audit (SML/L-SML/U-PCR/FUSE), 4-thread literature survey,
and 5 candidate pipeline-step designs (D1 assumption-violation-minimizing subset search — lowest
risk/highest priority; D2 unsupervised gated FS pre-fusion step; D3 rank/eigengap-guided grouping; D4
FUSE-style transformation search; D5 Omri's dual-use data-signature router, two access-tier flavors)
are in the memo. Key finding: **U-PCR and continuous L-SML are not the same algorithm** — different
structural covariance models (multiplicative rank-1 vs. additive) — and which one fits a given cell
better is itself a candidate label-free diagnostic (domain-dependent: L-SML dominant on GSM8K 90% win
rate, near coin-flip with U-PCR on GPQA/RAG 53%).

**Full memo**: `docs/research_notes/feature_subset_selection_landscape.md` — problem statement +
evidence, per-method assumptions audit with primary-source quotes, annotated bibliographies (Thread A:
Lindenbaum's Gated-Laplacian trace criterion identified as the "sub-matrix trace" method; Thread B:
Nadler portfolio incl. Parisi 2014 PNAS lineage-root citation gap closed, Kritchman-Nadler rank
estimation; Thread C: tabular foundation-model concepts, Concrete Autoencoders flagged as the most
directly adoptable primitive; Thread D: FUSE's Ŝ statistic, vanishing-tetrad tests, MetaOD as the
closest per-instance-router precedent), candidate designs, open questions for Ofir/Bracha.

**Next steps**: resolve the open questions in the memo (§5) with Ofir/Bracha, then pilot D1 (lowest
implementation risk, reuses existing L-SML/U-PCR residual code) on the 19-cell replication grid.

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

### Extension H — Prior-Free L-SML: derive orientation, size, and selection from structure alone (NEW top priority, Step 199, 2026-07-25)

**Omri's decision (2026-07-25).** Stop optimizing the prior-dependent selector. Every piece of the
current pipeline is bootstrapped from hand-picked prior knowledge, and Step 199 proved that caps it:

- **Seeds = GOOD_6** (a hand-picked subset) build the pseudo-label. The GOOD_6-seeded pseudo-label
  is **byte-identical to the GOOD_6 fused score on 25/25 cells** (`pseudolabel_quality_audit.csv`),
  so the selector is guided by GOOD_6 and mathematically cannot beat it.
- **Anchor = `epr` / `logprob_margin`** (a hand-picked feature) sets the orientation sign.
- **K = 15** is a fixed hyperparameter.

A full week of variants over this scaffolding moved macro AUROC ~1pp and stayed 0.2pp under GOOD_6.
The goal now: a selector with **zero hand-picked features or subsets**, deriving all three decisions
from the data's own structure. Three sub-problems.

**H1 — Orientation without an anchor feature.** *Current*: `anchor_orient` against `epr`.
*Target*: recover the fused score's sign from structure alone. *Candidate*: Z2 synchronization —
`sign(cov_ij)` is a noisy observation of `s_i * s_j`, so recover the relative sign vector
`s in {+/-1}^p` spectrally / by SDP from the pairwise-sign matrix (no anchor). The single remaining
global +/-1 ambiguity is broken by a **distributional** prior, not a feature: e.g. the class-imbalance
mode (hallucination is the minority) or the skew of the consensus score. *Honest caveat (Step 199)*:
at small subsets orientation costs ~0 (R2-R0 = +0.0002 at GOOD_6) but at the full pool it costs
~2pp — so prior-free orientation matters most precisely in the large-pool regime a prior-free
selector operates in.

**H2 — Feature-set size without a fixed K.** *Current*: fixed K=15; the residual-elbow rule already
**FAILED** (Step 198, r_s=+0.007 vs oracle-K). *Target*: a label-free size from the covariance
spectrum. *Candidates*: effective rank / participation ratio of `cov(V)`; count of eigenvalues above
a Marchenko-Pastur noise floor (the "signal dimension" under L-SML's low-rank model); bootstrap
stability selection. *Must* be validated against oracle-K the same honest way D1 was — correlating
with AUROC is not predicting the optimal size.

**H3 — Feature selection without seed priors.** *Current*: mRMR against a GOOD_6-seeded pseudo-label
(proven ≡ GOOD_6). *Target*: a seed-free consensus. *Candidate*: build the pseudo-label as the
**L-SML consensus over ALL features** (Nadler/Jaffe: the ensemble's self-consistent agreement, which
down-weights uninformative views through the covariance structure), then select features by
agreement with that consensus and iterate. *Risk*: garbage features polluting the consensus — bound
it with the never-run **R6** (perfect-target ceiling, `SPEC_gap_ladder.md`) and a low-signal-cell
guard. This is the natural escape from the GOOD_6 cap because the full-pool consensus is not tied to
any hand-picked subset.

**Grounding**: all three sit in the project's spectral-meta-learning lineage (Nadler 2012, Jaffe
2014, Parisi) — the right place to look for structure-only estimators of orientation (Z2 sync),
dimension (spectrum), and consensus (L-SML over all views).

**Decision gate**: a prior-free pipeline is worth adopting only if it reaches **GOOD_6 (0.7594)**
with zero hand-picked input. Matching GOOD_6 prior-free is itself a real contribution (it removes the
hand-tuning); beating it is the headline. First concrete step (CPU): H3's full-pool L-SML consensus
pseudo-label + H1's Z2-sync orientation, benched vs GOOD_6 on the 25 cells, before touching H2.

> **STATUS: ✅ CLOSED AS BOUNDED — gate NOT met (Steps 200–202, 2026-07-25).**
> Built (Step 200), audited (Step 201 — 9 defects, the "GroupFS sweep" never ran GroupFS), fixed and
> re-measured on a canonical scoring path with GOOD_6 = 0.7594 asserted on every bench (Step 202).
> All four components are bounded and **GOOD_6 remains unbeaten**:
>
> | Component | Verdict | Evidence |
> |---|---|---|
> | **H1** orientation | no headroom | L-SML is **gauge-invariant** to feature signs — 1150/1150 sign vectors bit-identical, so sign is worth exactly **0.0pp**; the prior-free skew tiebreaker costs **−10.7pp** and its premise (hallucination = minority) is false here (9/25 cells have pos_rate > 0.5) |
> | **H2** label-free K | **REFUTED** | no rule met the pre-registered oracle-K bar; `eff_rank` Spearman **−0.0995** (p=0.64); **fixed K=15 beats every adaptive rule**. Oracle-K median is **14** while the spectrum rules predict 4–6 — they count independent directions (~4.5), but L-SML exploits **correlated** views, so effective rank is the wrong quantity |
> | **H3** selection | bounded | **R6 = 0.7676 DEAD** — +0.82pp, under the +1.0pp gate. Note the corrected contrast gives **22W/3L, p = 0.00014**: a perfect label-derived target buys a *reliable but sub-1pp* gain, not nothing. The fixed `a7.iter_consensus` lands −2.16pp (8W/17L, p=0.011) |
> | **Phase 4b** GroupFS grouping | bounded | now genuinely swept (λ1 guard PASS: 71/700 configs vs 0/350 for the stand-in); best 0.7508, and its **label-peeking ceiling 0.7585 only ties GOOD_6** (−0.09pp, p=0.33) — bounded, not mis-tuned |
>
> **Durable gain (subtractive)**: `ALL_SIGNS` — 42 hand-derived per-feature polarities — is provably a
> no-op in the fusion path and can be deleted at zero cost. The only orientation prior that remains is
> a **single ±1 bit**, and the `epr` anchor already spends it optimally (an oracle bit ties it).
>
> **Where the gap actually is**: the ladder's clean rungs put the deploy-point gap in **weight
> estimation**, not sign, target quality, or K. Any successor direction should attack that.

---

### Extension I — Inverted-fit selection: the criterion is right, the sign is wrong (Step 203, 2026-07-26)

**Status**: ❌ **CLOSED AS REFUTED (Step 204, 2026-07-27). Do not build I1.** The whole premise — that
the fit criterion's sign is inverted — was an artifact of the L-SML loading scale.
`_estimate_von_voff` returned the unit-norm eigenvector where Lemma 1 requires the loadings to
reproduce the covariance, so misfit was inflated by group size × coupling strength — largest exactly
where the clustering succeeded. With a masked rank-one completion estimator:

| loading scale | Spearman(misfit, AUROC) | positive cells | **re-measured Step 205** |
|---|---|---|---|
| `unit` (what Step 203 measured) | **+0.223** | 24/25 | **+0.222**, 23/25 |
| `eigen` (the SPEC's literal fix) | +0.183 | 25/25 | +0.188, 25/25 |
| `complete` (exact on the unit checks) | **−0.006** | 12/25 | **−0.022**, 10/25 |

Shift **−0.228, Wilcoxon p = 0.0015**; the `unit` arm reproduces Step 203 exactly, so the harness is
sound. **Re-run on Step-205's fixed code** (this study's size grid starts at 3, so it was the one
most exposed to the small-m degeneracy): shift **−0.243, p = 0.0006** — the conclusion holds and
strengthens slightly. **The criterion never needed inverting — it needed scaling.** I1 (sign-flip the selectors)
would have been curing a symptom. I2–I4 and theorems T1–T4 rest on the same artifact and are void as
stated; T1 in particular ("redundancy inflates misfit monotonically") is *true of the broken
estimator* and is precisely the bug, not a property of the data.
Evidence: `results/pruning_study/06_scale_vs_criterion/`, `results/residual_scaling/`, HISTORY Step 204 §B.

<details><summary>Original Step 203 framing, kept for the record</summary>

**The finding.** The L-SML residual — the quantity every trimming rule in this project steers by — is
*informative about subset quality but anti-correlated with the direction we optimise*:

| Evidence | Number | Consistency |
|---|---|---|
| Within-size Spearman(residual, AUROC), live 30-view pool | **+0.223** mean / +0.185 median | **24/25 cells positive** |
| Repair worst-fitting group vs repair **random** group | **−2.22pp** | W/L 7/18, p = 0.032 |

Residual is *misfit* (lower = better rank-one fit), so a positive correlation means **worse-fitting
subsets score higher**. Minimising misfit is descending the wrong gradient.

**Why (mechanism, measured).** The worst-fitting group is reliably the near-duplicate confidence
cluster — `epr`, `epr_spilled`, `epr_energy`, `mean_top1_logprob`, `logprob_margin` — i.e. the
*strongest* individual views. They break the rank-one model **because** they are several readings of
one underlying quantity, and that duplication is precisely the extra shared structure a single-factor
model cannot absorb. In this data **redundancy and informativeness travel together**, so poor fit
marks where the signal is concentrated.

**This reframes three earlier closed results rather than contradicting them.** Extension G/H closed
because every label-free selector *minimising* something rank-one-flavoured landed at ≈0.75. If the
sign is inverted, they were all optimising away from the answer, and "bounded" may be "bounded in the
direction tested".

> **⚠ RUN `SPEC_residual_scaling_fix.md` FIRST (raised 2026-07-26, after this block was written).**
> Omri asked why the clustering — which is *supposed* to group dependent features together — ends up
> flagging those groups as the worst fit. Answer: `_estimate_von_voff` returns the **unit-norm**
> eigenvector, but Lemma 1 requires `v_i·v_j = r_ij`, i.e. `a = √λ₁·v`. A perfect `m`-duplicate block
> is therefore scored with misfit/pair rising 0.25 → 0.83 as `m` goes 2 → 11, so misfit is inflated by
> **group size × coupling strength** — largest exactly where the clustering *succeeded*. That makes
> "repair the worst group" mean "dismantle the biggest tight cluster", i.e. **the selection step
> optimises against the clustering step**. It also sits in the deployed detector, since K is chosen by
> minimising this residual (15/25 cells pinned at K ≥ 7). **If the scaling fix flips the sign, I1 below
> is the wrong remedy** — the criterion needed scaling, not inverting. Treat I1 as the fallback.

#### Proposed experiments (in dependency order)

- **I0 — Fix the eigenvalue scale first.** See `SPEC_residual_scaling_fix.md` (checks U1-U2, predictions P1-P3, anchors R1-R3). Gates everything below.
- **I1 — Sign-flip the existing selectors (only if I0 does not flip the sign).** Re-run `a1.residual`,
  `a6.pl_dufs`, and the Step-203 cluster-localized arm with the criterion **maximised** instead of
  minimised. Pre-register: label-free macro ≥ 0.7524 (`a6.pl_dufs`, the automatic-picker bar) with
  W/L and Wilcoxon reported; report effect sizes, do not gate on <1pp.
- **I2 — Redundancy-preserving trimming.** Trim to *increase* misfit, i.e. keep at least one member
  of each near-duplicate family and cut from the best-fitting (most "explained") groups. This is
  Yen's Q3 read backwards and is the natural constructive form of the finding.
- **I3 — Is the inversion a property of the data or of the estimator?** Recompute the correlation
  after removing one member of every |ρ|>0.9 pair. If the correlation collapses toward 0, the
  inversion is *caused by* duplication (and disappears in a de-duplicated pool); if it survives, it is
  a property of the fusion. This decides whether I2 is a fix or a workaround.
- **I4 — Per-cell sign check.** The 1/25 cell with negative correlation is a natural held-out probe:
  is it structurally different (fewer duplicate pairs? lower second-factor share?), or noise?

#### Candidate theorems / analytical targets

These are conjectures suggested by the measurements, stated so they can be proved or refuted rather
than assumed:

- **T1 (redundancy inflates misfit monotonically).** For `x_j = a_j y + ε_j`, adding a duplicate view
  `x_{m+1} = x_k + δ` with `Var(δ) → 0` increases the Eq.(14) rank-one residual by a quantity
  monotone in `a_k²`. *If true, misfit is partly a measure of how much signal a subset carries, which
  explains the observed positive correlation directly and predicts the effect scales with the
  duplicated view's loading.*
- **T2 (the localizer is a signal detector).** Under T1, `argmax_g` (within-group misfit per pair)
  selects the group maximising `Σ_{i,j∈g} a_i²a_j²` rather than the group containing the least
  informative view. *This would make "repair the worst group" provably a signal-removal operator, and
  the −2.22pp vs random a predicted consequence rather than an empirical accident.*
- **T3 (no interior optimum in expectation).** For subsets drawn uniformly at size k from a pool with
  bounded loadings, `E[AUROC(k)]` is non-decreasing in k. *Consistent with 25/25 cells here and with
  D1/H2's refutations; if provable it closes "find the right K by following a curve" analytically
  rather than one grid at a time.*
- **T4 (gauge invariance extends to the diagonal).** Step 201 proved L-SML is invariant to input
  feature signs (`X→XD`). Does the same hold for the precision-weighted variant `w_j ∝ a_j/(R_jj −
  a_j²)`, which *reads* the diagonal? Step 203 measured precision weighting at −0.13pp; a proof that
  it is also gauge-invariant would explain why it cannot move.

---

</details>

### Extension J — Weight estimation (the R3 − R2 = +1.45pp term) — first evidence in, still open

**Status**: Open, and **narrowed by Step 204, narrowed further by Step 205**.

> **Step 205 closes the last U-PCR lead and corrects how one of its numbers should be read.**
> The `lambda2_threshold` hypothesis — that we are pinned at 2 eigenvectors and the redundant second
> factor is what hurts — is **refuted**. `lambda2_frac` is tightly clustered just above the hardcoded
> 0.1 (median 0.1435, range 0.0942–0.2328), so the threshold flips 24/25 cells at once; sweeping it
> to remove the second component everywhere buys **+0.43pp (9W/15L, p = 0.16)** (`exp07`).
> **And Step 204's "−3.67pp for the 2-eigenvector rule" is a factorial MAIN EFFECT**, averaged over
> the 32 combinations of the other five factors. At the deployed configuration the same switch is
> **−0.43pp mean / +0.07pp median, 15W/9L, p = 0.16** — a wash. The sign reversal of Step 142 stands;
> the magnitude does not transfer, and should not be quoted as a deployed cost.
> Separately, Step 204's B1 ("the g2 range never binds") holds only for the **pre-exclusion** fit: the
> g2 the pipeline returns comes from the post-exclusion refit and is at the grid ceiling in 24/25
> cells. Un-pinning it is −0.28pp (12W/13L), so the conclusion survives and the mechanism does not.
> **Nothing in the g2 / component-count apparatus is actionable. Do not re-open it.**

The U-PCR line is closed as inert: every paper-faithful
flag hurts or does nothing, one-component U-PCR is *exactly* PC1 of the surviving features (cosine
deviation 7e-12) so its ρ/g²/Eq.-20 machinery enters only through the exclusion mask, and no
configuration of 64 reaches GOOD_6. The dependent-features extension (fit the additive system on
cross-cluster pairs only) is **refuted**: it fails both pre-registered mechanism gates and loses
−4.46pp (9W/16L, p = 0.030), and its premise was a confound — fit error is essentially pair
correlation (Spearman 0.870), the raw 2.03× same-vs-cross gap collapses to 0.97–1.00 when matched on
|C_ij| decile, and magnitude-only clustering separates it *better* than L-SML.
**What is left of J**: the disagreement diagnostic itself (rank agreement +0.186, top-5 overlap 1/5)
is unexplained, and the one lever that moved anything this session was **orientation**, not weighting.

Step 203 supplies the first systematic measurement; no repair found.

**What was tested (Step 203, Exp 5).** A 2×4×2 factorial over the three slots the five proposed
paradigms actually occupy — conditioning (none / RMT eigenvalue clipping) × loading estimator
(eigenvector / triplets / low-rank+sparse / robust-IRLS) × weighting (signal / precision). **Span:
0.7434–0.7555.** Main effects: triplets +0.21pp over the eigenvector, low-rank+sparse +0.11pp, robust
IRLS −0.33pp, RMT cleaning +0.14pp, precision weighting **−0.13pp**. Nothing separates from noise on
25 cells.

**Diagnostic (Exp 4), which should aim any next attempt.** Second factor at median **0.312** of the
first (one factor explains ~81% of `R_off`'s squared spectral mass) — the rank-one premise is
approximate, not exact. Guessed vs supervised trust levels: rank agreement **+0.186**, sign agreement
**0.55** (after resolving the single global flip), **top-5 overlap 1/5**. The label-free estimator and
the supervised model largely disagree about *which* views matter — this is not a calibration problem.

**Ruled out analytically or empirically**: Ledoit–Wolf linear shrinkage (provably inert — scales
`R_off` by a positive constant, identical eigenvector); non-linear shrinkage *as usually specified*
(modifies eigenvalues, keeps eigenvectors — inert unless the diagonal is re-zeroed and re-decomposed,
which is the form tested here); bagging (identical to 4 d.p.).

**Next**: given Extension I, the most promising route is not a better rank-one estimator but an
explicitly **rank-2** model, or de-duplicating the pool before estimation (I3). A repair aimed at the
one-factor premise is aimed at a premise the data only approximately satisfies.

---

## Recommended Priority Order

*(Single authoritative list — updated 2026-07-02, post streaming pilot Step 148)*

**Now — no GPU needed**
0. ~~**Extension H — Prior-Free L-SML** (Step 199 pivot): strip every hand-picked prior (`epr`
   anchor, `GOOD_6` seeds, fixed K); derive orientation (H1), size (H2), selection (H3) from
   structure alone~~ ✅ **CLOSED AS BOUNDED (Steps 200–202) — decision gate NOT met.** H1 has no
   headroom (sign is a gauge, worth 0.0pp), H2 is refuted (fixed K=15 beats every label-free rule;
   `eff_rank` r_s = −0.0995), H3 is capped (R6 perfect-target = DEAD at +0.82pp), and GroupFS
   grouping's own label-peeking ceiling only ties GOOD_6. **GOOD_6 (0.7594) still unbeaten.** The one
   durable gain is subtractive: `ALL_SIGNS` (42 priors) is a provable no-op and can be deleted.
   See the Extension H status block above.

0-NEXT. ~~**Extension I — inverted-fit selection (Step 203)**~~ ❌ **CLOSED AS REFUTED
   (Step 204), re-verified on fixed code (Step 205).** The +0.223 correlation was an artifact of
   the L-SML loading scale. Re-measured after the Step-205 small-m fix: unit **+0.222** (23/25
   positive) → complete **−0.022** (10/25), shift **−0.243, p = 0.0006** (Step 204 published
   −0.228, p = 0.0015). **P2 holds, slightly strengthened. Do not build I1.** See the Extension I
   block above.

0-NOW. **← Orientation without hand-picked signs (Step 204, Phase E).** The one lever that moved
   anything: deriving per-feature polarity from `sign(rho)` beats the 42 hand signs by **+1.46pp
   (20W/5L, p < 0.001)**, and **15 of 30 pool features carry the wrong hand sign**. Two caveats
   that shape the experiment: correcting the signs is a **0.0000pp no-op on the L-SML path**
   (sign-gauge invariance), so the value is only for sign-SENSITIVE consumers; and the global
   ±1 is **provably not recoverable** from the covariance (a global flip leaves rho
   bit-identical), so the `epr` anchor bit cannot be removed this way. Pre-register against the
   0.7524 automatic-picker bar, not 0.7594.

0-ALSO. **DECIDE WITH OMRI: attack weight estimation (Extension J).** Both the gap-ladder's clean rungs and the
   Extension H post-mortem point at the same place — at the deploy point the gap to the supervised
   linear oracle is **weight estimation**, not sign (0.0pp), not target quality (R6 DEAD), and not
   subset size (fixed K wins). Extensions G and H are both closed as bounded, so this is the open
   question. No experiment is pre-registered for it yet.
0b. ~~Feature-subset selection: memo (Step 185), full bench (Step 186), a6 pseudo-label gates +
   30-view LOCO sweep (Steps 194-195), D1/D2 build + honest refutation (Steps 197-198)~~ ✅ **CLOSED
   as bounded** — no label-free selector beats GOOD_6; D1 (adaptive-K) refuted, D2 (PL-mRMR) beats
   GOOD_5 (p=0.037) but not GOOD_6. The one durable win is the seed rule (→GOOD_6, +3.5pp macro).
   `LOCO_5` (sweep consensus, 77.1% on 24 cells) is the strongest fixed subset found and still
   warrants naming + `REFERENCE_SUBSETS` entry independent of Extension H.
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
