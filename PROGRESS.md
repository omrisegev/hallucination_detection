# Spectral Hallucination Detection — Session Progress Handoff

**Date**: 2026-06-25
**Last updated**: Step 143 — LR oracle evaluation corrected (two bugs: cross_val_predict concatenation + missing class_weight='balanced'). Corrected result: supervised LR (67.1–67.6% balanced averaged-fold CV) beats L-SML by 2–5pp; in-sample ceiling 71–73%. See SUPERVISED_ORACLE_CORRECTION.md. Items 1 and 2 from advisor meeting both complete.

---

## TL;DR — where we are today

**Recommended method** (established by the Step-134 method comparison, 12 variants × 29 cells):
`lsml_continuous_pipeline(feats_dict, GOOD_FEATURES, FEATURE_SIGNS)` — **L-SML continuous** (previously called "CONT" — that term is retired).
- Macro AUROC **70.1%** vs the old binary PROD pipeline 65.2% (**+4.9pp**); **78.3%** on the reasoning regime {MATH-500, GSM8K, QA}.
- On reasoning it beats a simple average (+2.2pp) and even the per-cell oracle best-single-feature (+0.7pp).

**Old production method** (binary, Steps 100–131 — now superseded as the recommendation):
`binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(...)` — the `np.sign()` binarization was the single biggest source of lost signal.

**Key conclusions from Step 134** (independently co-signed by Gemini, `LSML_IMPLEMENTATION_REPORT.md` §13–17):
- **Encoding is the dominant lever**, not features or signs. Continuous beats binary by +4.9pp macro / +7.2pp reasoning.
- **Feature selection is a minor tweak**: continuous L-SML on *all 16* features (`lsml16c`) = 69.2%, within 0.9pp of the selected 5-feature CONT. It helps on reasoning, hurts GPQA. (Answers Bracha Q1.)
- **FEATURE_SIGNS = one global orientation bit**, not a learned dictionary (all 5 GOOD_5 signs equal → a single global flip). Required for deployment orientation; adds zero separability. The paper's internal sign algorithm fails on our error-predicting features (~14% concordance).
- **Robustness (R4) hypothesis rejected**: grouping does *not* insulate against volatile features — avg5 is the most cross-domain-stable (8.9pp std), CONT the least (10.9pp). Fusion's justification is in-regime peak accuracy, not robustness.
- **Operating regime**: spectral L-SML is a reasoning-trace method. GPQA (forced-choice MCQ) and RAG (retrieval-grounded) lack the temporal structure; there a simple average is as good or better.
- **Deliverables**: `Bracha_Reply_Jun2026.md` (answers her 3 Jun-8 questions), updated `results/method_comparison_report.html` (§13–16: lsml16c, R4 robustness, reasoning-only, per-cluster AUC).

**Step 135 — grid completion + benchmarking + narrative report**:
- Full design grid done (5/9/16 × binary/continuous × flat/L-SML + avg). **Continuous beats binary in every cell.** L-SML clustering helps only with many features (5 feat: ties flat; 16 feat: +6.1). Flat-SML-continuous collapses 70→63 as features added; L-SML holds 68–70.
- **Benchmarking (model-matched, CONT, 1-pass)**: MATH 94.4 (win vs SE 87.7/SC 87.2), GSM8K 75.6 (competitive; beats LapEigvals-unsup 72.0), GPQA 52.3 (loss vs SE 70.6), RAG beats SelfCheckGPT 3/4.
- ⚠ **Do NOT reuse Step-117 "ours" numbers** (96.7/71.3/88.1 — leaked supervised). ⚠ **EDIS Phase-13 invalid** (7.7% acc = `\boxed{}` grading bug); fix before citing.
- New: **`results/Spectral_LSML_Report.html`** — story-driven advisor report (this is the one to attach, not method_comparison_report.html).

**Step 136 — cross-cluster weights + full correlation + report v2 (report sent to advisors)**:
- **Across-group fusion weight now stored** per cluster (`cross_weight` col in table2/JSON). Mechanism: it is the leading eigenvector of the clusters' off-diagonal covariance = each cluster's estimated reliability, **not** an average.
  - **K=2 → always 0.50/0.50** (structural — 2×2 zero-diag covariance). So a 2-cluster even split is NOT evidence of adaptive weighting.
  - **K≥3 → weights separate**; a weak isolated cluster gets ≈0 (e.g. pe_mean 0.02 on 16-feat MATH-500). A true average would give it 0.25.
- **pe_mean is domain-dependent — do NOT hard-delete it**: isolated + weight 0.02–0.05 where weak (MATH-500, both QA-CoT cells), but joins a useful `epr,pe_mean` cluster (67.7%, weight 0.24) on GSM8K. L-SML's weighting suppresses it adaptively, only where it should.
- **Full 16×16 dependence matrix** → `results/feature_correlation_16.csv` (new `scripts/feature_correlation_full.py`): band-power block ρ 0.77–0.88, median pair 0.25, pe_mean near-independent. This is the structure L-SML exploits / flat SML ignores.
- **No feature is both strong and stable**: strong features (epr/cusum_max/sw_var_peak) swing ~30pp across domains; stable features (pe_mean range 8.5) are weak everywhere.
- Report v2: removed exec summary; added terminology + aggregation note + 9-feature data + 3 graphs (dependence heatmap, stability scatter, per-domain ranking heatmap). Self-contained except Chart.js CDN.
- **Open**: fix EDIS grading + re-run; complete Phase 14 (GPQA/DeepSeek-R1-8B).

**Step 142 — U-PCR algorithm correction + re-run**:
- Fixed two bugs: (1) weight formula `w_k = (v1@rho/lam1)*v1` hardcoded to v1 even for n_components=2 — corrected to `Σ_c (vc@rho/lamc)*vc`; (2) no λ₂ auto-threshold — added `auto_components=True, lambda2_threshold=0.1`.
- Re-run (29 cells, 3 feat sets): U-PCR-auto gets +0.5pp over old U-PCR-1 on 16-feat (63.0% vs 62.5%), still below L-SML continuous (65.1%). On 5/9 feat the correction slightly hurts (−0.6pp, −0.8pp) because v₂ captures structured noise for low-correlation feature sets.
- λ₂/Trace = 9–34% across cells (28/29 exceed 10% threshold). The paper's 10% threshold is too permissive — 15–20% would be more appropriate for our curated feature sets.
- **v₂ as soft clustering**: (v₁[i], v₂[i]) are continuous cluster coordinates for each feature; U-PCR uses them directly instead of L-SML's hard group assignment. Same structural idea, different tradeoff.
- Updated `results/upcr_comparison.pkl` + new `results/upcr_comparison.png` (3-panel visualization).

**Step 141 — Deep literature review: FUSE, Deep L-SML, STDR, U-PCR**:
- **FUSE finding**: our closed-form eigenvector weights (`w@F`) underperform naive averaging in 7/10 FUSE benchmark settings (Figure 3). Fix: replace with pseudo-label logistic regression on MoM-estimated triplet posteriors `p̂(r_i)` — still fully unsupervised. **Highest-priority next experiment.**
- **RBM = L-SML equivalence** (Lemma 4.1, Shaham et al. 2016): our covariance+eigenvector step IS a single-hidden-node RBM trained by MoM. Stacked RBM (Deep L-SML) handles correlated features without exclusion; relevant for 16-feat expansion where band-power pairs (ρ 0.77–0.88) trigger heavy filtering.
- STDR (Fiedler vector tree recovery): not relevant at current feature counts.
- Step 140 numbers now explained: U-PCR ≈ L-SML on 5/9 features (low-corr regime matches assumption); L-SML wins on 16 because clustering handles band-power block violation.

**Steps 139–140 — U-PCR literature + implementation + empirical comparison**:
- U-PCR (`upcr_fuse`, `upcr_pipeline`) implemented in `spectral_utils/fusion_utils.py` (Tenzer et al. 2022).
- Comparison run across 29 cells, 5/9/16 feature sets. Results (macro AUROC):

| Feature set | L-SML continuous | U-PCR | Delta |
|-------------|-----------------|-------|-------|
| 5-feat | 65.3% | 65.7% | +0.4pp |
| 9-feat | 63.9% | 65.0% | +1.1pp |
| 16-feat | 65.1% | 62.5% | −2.5pp |

- **Conclusion**: U-PCR ≈ L-SML continuous on low-correlation feature sets (5, 9 feat — the assumption E[h_i h_j]=0 approximately holds). L-SML continuous wins on 16 features where correlated features (band-power block ρ 0.77–0.88) violate U-PCR's assumption; clustering handles this, plain eigenvector weighting doesn't.
- Provides the theoretical citation for Step 134: L-SML continuous ↔ U-PCR's ρ̂-proportional weighting. Cite Tenzer et al. (2022) instead of "workaround for Lemma 1".
- Advisor meeting Item 1 (lit search) ✅ complete.

**Prior session (Step 131)**:
- GSM8K cross-dataset verification: spilled energy transfers well (cusum_max_spilled = 0.725 best individual)
- Verbalized confidence: **null result on 1.5B**; adding VC hurts L-SML (−1.77pp)
- Structural finding: within_H/cross ratio = 0.04 (MATH-500) vs 0.99 (GSM8K) — H features are near-independent views on long traces but redundant on short traces
- All changes on branch `experiment/lsml-variants` (commit `f4bc5e8`)

---

## MEETING ACTION ITEMS — Jun 17, 2026 (Ofir, Bracha, Amir)

*Email thread: Omri → Ofir/Bracha/Amir, Jun 17 2026, confirmed by Ofir same day.*

These 6 items are the current priority order. They supersede the old Step 132 GPU-first priority (Step 132 is still pending but de-prioritized until these are underway).

| # | Action | Status |
|---|--------|--------|
| 1 | **L-SML literature search** — find Nadler post-2016 follow-up work extending or improving L-SML | ✅ Complete (Step 141) |
| 2 | **Logistic regression oracle** — supervised LR on 5/9/16 feature sets → upper bound on fusion AUROC (5-fold CV, no in-sample leakage) | ✅ Complete (Steps 142–143: evaluation corrected) |
| 3 | **Extend QA evaluation** — run more QA datasets (NQ, SQuAD v2, AmbigQA, PopQA) to characterise CoT factual QA performance | Not started |
| 4 | **Benchmarking completion** — model-matched comparisons for MATH-500, GSM8K, QA vs SE/SC/SelfCheckGPT | In progress (Step 135 partial) |
| 5 | **Experiment 1 — Sampling fusion** — fuse SE (K=10) with single-pass spectral features; measure AUROC gain vs each alone | Not started |
| 6 | **Experiment 2 — Temperature variation** — run same model at T∈{0.3,0.6,1.0,1.5,2.0}; does higher T improve detectability? Ablate: T-diversity vs just more passes | Not started |

See `Research_Directions.md` § "Meeting Action Items — Jun 17, 2026" for full experimental designs.

---

## Current best-candidate pipeline constants (not finalised — pending meeting experiments and merge decision)

```python
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
    # Spilled energy signs — validated on GSM8K; confirm on MATH-500 in Step 132
    'epr_spilled': -1, 'sw_var_peak_spilled': -1,
    'cusum_max_spilled': -1, 'min_spilled': -1,
    # Verbalized confidence — null on 1.5B; may work on 7B+
    'verb_conf': +1, 'verb_conf_1p': +1,
}
```

Note: `min_spilled` sign updated from initial `+1` estimate to `-1` — validated on GSM8K Cell 12 sign check.

---

## IMMEDIATE NEXT ACTIONS

*Priority order from the Jun 17 meeting. See table in "MEETING ACTION ITEMS" section above for status.*

### Priority 1 — Benchmarking completion (Item 4, partial)

**Done** (Step 135): MATH-500 94.4% vs SE 87.7/SC 87.2 ✅ | GSM8K 75.6% vs SC 78.5/SE 77.4 ✅

**Still needed**:
- QA datasets: SelfCheckGPT / SE comparison on same model+dataset
- Phase 14 Cell 9 re-run: DeepSeek-R1-0528-Qwen3-8B / GPQA (L-SML v2 AUROC still TBD; Cell 9 `n_boot` kwarg bug needs fix — see Running Experiments below)

### Priority 2 — Experiment 1: Sampling fusion (Item 5, Colab GPU)

Fuse SE (K=10) with CONT spectral features on one cell (MATH-500 or GSM8K). Key check: Spearman ρ(SE, spectral_score) < 0.75 before fusing.

### Priority 3 — Temperature variation experiment (Item 6, Colab GPU)

Run Qwen2.5-Math-7B on MATH-500 at T∈{0.3, 0.6, 1.0, 1.5, 2.0} + 4 extra runs at T=1.0 for the ablation. (T=1.0 and T=1.5 caches already exist.)

### Priority 4 — Extend QA evaluation (Item 3, Colab GPU)

Additional QA datasets with CoT prompt: NaturalQuestions, SQuAD v2, AmbigQA (in priority order). Use Qwen2.5-Math-7B or Falcon-3-10B.

### De-prioritized (was Priority 1 before meeting)

- **Step 132** (MATH-500 SpilledEnergy GPU run) — still pending, still valid, but not the current focus. Run when a Colab session is available between other GPU tasks.
- **Merge decision** (continuous L-SML + spilled energy → master) — contingent on Step 132.
- **Phase 13** (EDIS vs L-SML on AMC23/AIME24) — EDIS grading bug must be fixed first.
- **Verbalized confidence on 7B** — low priority relative to meeting items.

---

## Research directions and open questions

### What we know works
- **Spectral features of H(n)** work on reasoning-heavy domains (MATH-500, GPQA). GOOD_5, continuous L-SML: best published unsupervised single-pass numbers on these domains.
- **Spilled energy ΔE(n)** cross-dataset validated: competitive individual AUROCs on both MATH-500 and GSM8K, corr(H,ΔE) = 0.984–0.989.
- **Not general-purpose**: short factual QA traces (TriviaQA, WebQ) are structurally incompatible.
- **Continuous L-SML** (+3.53pp over binarized, 25/29 cells) — merge to master pending Step 132.
- **within_H/cross ratio** is a dataset-level diagnostic for L-SML benefit: long reasoning = 0.04 (near-independent, L-SML gains a lot); short traces = 0.99 (redundant, gains less).

### Verbalized confidence — model-size gated
- 1.5B: null result confirmed (Step 131). Model doesn't follow "Confidence: X" instruction.
- `parse_verbalized_confidence` is now correct — ready to test on 7B+.
- **Do not include verb_conf in GOOD_FEATURES for 1.5B runs.**

### Open: M=9 orthogonal feature set design
If Step 132 confirms Pearson corr(epr_H, epr_ΔE) < 0.6 on MATH-500, proceed with:
```
Group A — H(n): epr, cusum_max, sw_var_peak
Group B — ΔE(n): epr_spilled, cusum_max_spilled, min_spilled
Group C — structural: rpdi, dominant_freq, stft_max_high_power
```

### What was ruled out
- **Verbalized confidence on 1.5B**: null result (Step 131). Model-size gated.
- **Hedging count**: not formalized, domain-dependent, weaker than spectral. Do not implement.
- **NLI/semantic entropy methods**: require additional model inference. Out of scope for zero-extra-compute.
- **Quantile calibration**: null result. Median binarization only.

---

## Branch situation

| Branch | Status | Contents |
|--------|--------|----------|
| `master` | Production | Steps 1–125; 16-feature binarized L-SML |
| `experiment/lsml-variants` | **Active, not merged** | Continuous L-SML + spilled energy + GSM8K verify + parser fix (Steps 128–131) |

**To merge** (after Step 132 validation):
```bash
git checkout master
git merge experiment/lsml-variants
git push origin master
```

---

## Running experiments (Colab — GPU needed)

- **Step 132** (priority): `Spectral_Analysis_SpilledEnergy_Verify.ipynb` on `experiment/lsml-variants`
- **Phase 13**: `Spectral_Analysis_MathComp_Phase13.ipynb` — L-SML vs EDIS, Qwen2.5-Math-1.5B
- **Phase 14**: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb` — L-SML vs VC/SC, GPQA Diamond
  - **Known bug in Drive copy**: old Cell 9 uses `boot_auc(..., n_boot=1000)` (wrong kwarg). Fix:
    ```python
    p_auc, p_lo, p_hi = boot_auc(labels[valid_mask], lsml_full[valid_mask])
    n_auc, n_lo, n_hi = boot_auc(labels[valid_mask], -lsml_full[valid_mask])
    if p_auc >= n_auc:
        lsml_auc, lsml_lo, lsml_hi = p_auc, p_lo, p_hi
    else:
        lsml_auc, lsml_lo, lsml_hi = n_auc, n_lo, n_hi
    ```

---

## Completed phases

| Phase | Notebook / Script | Status | Key result |
|-------|------------------|--------|------------|
| Step 100 | Consolidated_Results | ✅ | Old supervised numbers — do not use |
| Step 107 | Consolidated_Results_LSML | ✅ | L-SML with assumption (iii) — superseded |
| Step 110 | LSML_Diagnostics | ✅ | Consensus FEATURE_SIGNS derived |
| Step 113 | Pilot_RAG_Prompt_Variants | ✅ | V4 prompt wins (+18.6pp), RAG direction dropped |
| Phase 12 | Phase12_Benchmarking | ✅ | SE/SC/VC/SelfCheckGPT baselines computed |
| Step 121–123 | LSML_Optimized | ✅ | 5-feature GOOD_FEATURES finalized, median binarization |
| Step 124–125 | Consolidated_Results_LSML_v2 | ✅ | 5-feat; 29/29 beat chance; HTML updated |
| Step 126 | run_lsml_local.py | ✅ | −5.7pp fusion lift; feature sign instability diagnosed |
| Step 127 | analyze_features.py | ✅ | Cluster structure mapped; trace_length suppression confirmed |
| Step 128 | verify_lsml_paper.py | ✅ | Implementation correct; K_range bug confirmed + fixed |
| Step 129 | experiment/lsml-variants | ✅ | Continuous L-SML: +3.53pp, 25/29 wins — pending merge |
| Step 130 | model_utils + feature_utils | ✅ | Spilled Energy implemented; verification notebook created |
| Step 131 | GSM8K_SpilledEnergy_Verify | ✅ | Spilled energy cross-dataset confirmed; VC null on 1.5B |
| Step 132 | SpilledEnergy_Verify.ipynb | ⏳ | **NEXT — MATH-500 run, needs Colab GPU** |

---

## Best results (reference, do not use Step 100 supervised numbers)

| Setup | L-SML AUC | Notes |
|-------|-----------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **88.2%** | PROD (binary L-SML). CONT (continuous L-SML) achieves **94.4%**. |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
| GSM8K / Qwen2.5-Math-1.5B | 70.8% | L-SML GOOD_5; best individual 72.5% (cusum_max_spilled) |
| GPQA / Mistral-7B / T=1.0 | 65.4% | Phase 4 best — beaten by 72B (Phase 8) |
| HotpotQA / Mistral-7B | 59.5% | spectral doesn't transfer to multi-hop QA |

---

## Available competitor numbers (Phase 12)

| Domain | Model | Competitor | AUROC |
|--------|-------|------------|-------|
| GSM8K | Llama-3.1-8B | SC K=10 | 78.5% [72.0,84.5] |
| GSM8K | Llama-3.1-8B | SE NLI K=10 | 77.4% [70.9,83.5] |
| MATH-500 | Qwen2.5-Math-7B | SC K=10 | 87.2% [72.1,98.4] |
| MATH-500 | Qwen2.5-Math-7B | SE NLI K=10 | 87.7% [79.7,93.9] |
| GPQA | Qwen2.5-7B | SE NLI K=10 | 70.6% [43.6,93.3] |
| GPQA | Qwen2.5-7B | VC K=1 | 67.9% [49.5,83.3] |
| GPQA | Qwen2.5-7B | SC K=10 | 33.6% [11.0,58.2] |
| RAG HotpotQA | Qwen2.5-7B | SelfCheckGPT K=5 | 51.4% [41.5,62.9] |
| RAG NQ | Qwen2.5-7B | SelfCheckGPT K=5 | 57.1% [42.9,70.5] |
| RAG 2Wiki | Qwen2.5-7B | SelfCheckGPT K=5 | 55.3% [35.7,78.3] |
| RAG NarrativeQA | Qwen2.5-7B | SelfCheckGPT K=5 | 52.4% [41.7,65.4] |

---

## Key decisions (permanent — do not revisit)

1. No old supervised numbers — Step 100 historical only.
2. GOOD_FEATURES = `['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']` — final 5.
3. Median binarization only — quantile calibration dropped (null result).
4. HTML table: per domain, per model, same-task/same-model/same-dataset only.
5. Cite Jaffé-Fetaya-Nadler 2016. Never say "Nadler" alone. Method name = L-SML.
6. Never say "MV_EPR" — the method is spectral/L-SML.
7. Branch cleanup done — only `master`, `origin/main`, and `experiment/lsml-variants` remain.
8. Hedging count: ruled out — not formalized, domain-dependent, weaker than spectral.
9. Continuous L-SML (`lsml_continuous_pipeline`) is the candidate replacement — pending Step 132 validation before merge.
10. Verbalized confidence on 1.5B: null result (Step 131). Do not include in GOOD_FEATURES for 1.5B runs.
11. `min_spilled` sign = −1. Validated GSM8K Cell 12.
12. "CONT" is retired. Say "L-SML continuous" (with feature count when relevant: "L-SML continuous 5").
13. LR oracle corrected conclusion: supervised LR (67.1–67.6% averaged-fold balanced CV) beats L-SML by 2–5pp macro; in-sample ceiling 71–73%. See `SUPERVISED_ORACLE_CORRECTION.md` for evaluation rules.

---

## Deferred

- Verbalized confidence on 7B+ — parser is ready; needs one inference run with Qwen2.5-Math-7B on GSM8K
- Phase 10 RAG re-run with variant=4 prompt — low priority
- LapEigvals integration into spectral_utils — potential Group D feature for M=12, low priority
- M=9 orthogonal feature set experiment — contingent on Step 132 confirming ΔE group independence on MATH-500
