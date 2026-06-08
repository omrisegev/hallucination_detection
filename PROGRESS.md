# MV_EPR — Session Progress Handoff

**Date**: 2026-06-08
**Last updated**: Step 130 — Spilled Energy implemented; verification notebook ready to run in Colab

---

## TL;DR — where we are today

**Current official method** (production, master branch):
`binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(*binary_filt.values())`

**Candidate upgrade** (branch `experiment/lsml-variants`, pending merge):
`lsml_continuous_pipeline(feats_dict, GOOD_FEATURES, FEATURE_SIGNS)` → +3.53pp mean, 25/29 wins

**New this session (Step 130)**:
- Spilled Energy ΔE(n) = −log p(sampled token) extracted alongside H(n) in `generate_full()`
- 4 new features: `epr_spilled`, `sw_var_peak_spilled`, `cusum_max_spilled`, `min_spilled`
- `FEAT_NAMES` now 20 features (was 16)
- Verification notebook created: `Spectral_Analysis_SpilledEnergy_Verify.ipynb`
- All changes on branch `experiment/lsml-variants` (commit `78c82b9`)
- **Not yet run**: notebook requires new Colab GPU inference

---

## FINAL pipeline constants (do not change until merge decision)

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
    # Spilled energy signs (initial estimates — validate in verification notebook Cell 14)
    'epr_spilled': -1, 'sw_var_peak_spilled': -1,
    'cusum_max_spilled': -1, 'min_spilled': +1,
}
```

---

## IMMEDIATE NEXT ACTION

### Step 131 — Run verification notebook in Colab

Open `Spectral_Analysis_SpilledEnergy_Verify.ipynb` on branch `experiment/lsml-variants`.
Cell 1 is fixed to clone the right branch. Run sequentially:

**Cells 1–6** (setup + inference): ~30–60 min for 100 samples on Qwen2.5-Math-1.5B / MATH-500.

**After inference, read these outputs carefully:**

| Cell | What to check | Decision gate |
|------|--------------|---------------|
| Cell 7 | `Saturated: X%` — should be <5% with max_new=2048 | If >20%, increase further |
| Cell 7 | `Spilled energies stored: True` — must be True | If False, inference used old code |
| Cell 9 | Bar chart: are spilled features competitive individually? | Need ≥0.55 AUROC to be useful |
| Cell 14 | Sign mismatches for `min_spilled` — is +1 right? | Update FEATURE_SIGNS if wrong |
| Cell 10 | `within/cross ratio` for H(n) and ΔE(n) groups | Need ratio >1.5 for L-SML separation |
| Cell 12 | Group assignment: do spilled features land in their own group? | If mixed with H(n), they're not orthogonal |
| Cell 15 | Lift of GOOD_5+spilled vs GOOD_5 | If >+1pp, merge the spilled features |
| Cell 16 | Pearson corr(epr_H, epr_ΔE) | If >0.8: not orthogonal; if <0.5: truly independent |

### Next priorities (in order after Step 131):

2. **[DECISION] Merge continuous L-SML + spilled energy to master** — based on Step 131 results. Swap `binarize_classifiers + lsml_fuse` for `lsml_continuous_pipeline` in production notebook. GOOD_FEATURES unchanged. Add spilled features only if Step 131 shows within/cross improvement.
3. **Phase 13** — EDIS vs L-SML on Qwen2.5-Math-1.5B, AMC23/AIME24 (GPU needed). Run *after* merge decision so notebook reflects the final pipeline.
4. **Phase 14 Cell 9 re-run** — DeepSeek-R1-0528-Qwen3-8B / GPQA: L-SML v2 AUROC still TBD.
5. **Orthogonal feature set design** — If Step 131 confirms ΔE(n) is independent of H(n), design a 9-feature M=9 L-SML experiment: 3 groups × 3 features (H(n) group, ΔE(n) group, structural group). See "Research Directions" section below.

---

## Research directions and open questions

### What we know works
- **Spectral features of H(n)** work on reasoning-heavy domains (MATH-500, GPQA). 5 features, median binarization, L-SML continuous: best published unsupervised single-pass numbers on these domains.
- **Not general-purpose**: short factual QA traces (TriviaQA, WebQ) have structurally incompatible traces.
- **Continuous L-SML** (+3.53pp over binarized, 25/29 cells) should be merged to master.

### Open research question: can we build a truly orthogonal M=9 feature set?

**Why this matters**: The covariance audit (Steps 128–130) showed that all 5 GOOD_FEATURES are functions of the same H(n) time series. Within-group R correlations are 0.35–0.88. The L-SML paper (Jaffé-Fetaya-Nadler 2016) needs features from genuinely different information sources to form distinct groups. With M=5 from one source, L-SML collapses to near-naive-SML.

**Proposed 9-feature design** (to test if Step 131 confirms ΔE orthogonality):
```
Group A — H(n) entropy dynamics (existing, confirmed good):
    epr,  cusum_max,  sw_var_peak

Group B — ΔE(n) spilled energy dynamics (new — validate in Step 131):
    epr_spilled,  cusum_max_spilled,  min_spilled

Group C — structural / local spectral (orthogonal by construction):
    rpdi,  dominant_freq,  stft_max_high_power
    (or: trace_length_sat_flag, dominant_freq, stft_spectral_entropy)
```

**Decision gate for Group B**: If Step 131 shows Pearson corr(epr_H, epr_ΔE) < 0.6 AND within/cross ratio for ΔE group > 1.5, proceed with M=9 design.

**Decision gate for Group C**: Step 127 showed rpdi, dominant_freq, stft_* have near-random AUROCs due to right-censoring / trace saturation at max_new_tokens=512. With max_new_tokens=2048 (Step 131), re-check their individual AUROCs. If they recover to ≥0.55, include in Group C.

### What was ruled out

- **Hedging count** (regex on output text): Not formalized as a standalone hallucination detection paper. Domain-dependent (math models hedge very little), model-dependent. Weaker than spectral features. **Do not implement.**
- **Semantic Energy** (RESEARCH_PROPOSAL Section 2B): Multi-pass — requires re-running the model. Out of scope for 1-pass unsupervised.
- **EDIS** (Section 2C): Already implemented in `feature_utils.py` as `compute_edis()`. Phase 13 compares EDIS vs L-SML.
- **Quantile calibration**: Permanently dropped — null result (+0.001 on 4 features). Median binarization only.
- **Supervised numbers (Step 100)**: Must not be used or referenced.

### LapEigvals (EMNLP 2025, arXiv:2502.17598)
Uses Laplacian diagonal of cross-layer **attention maps** — a completely different circuit from entropy/spilled. Potentially the ideal Group D feature if we want M=12. Code exists in `baselines/lapeigvals/hallucinations/features/laplacian.py`. Currently not integrated into spectral_utils. Low priority until M=9 design is validated.

### Spilled Energy (Minut et al., ICLR 2026, arXiv:2602.18671)
- ΔE(n) = −log p(sampled token) — different from H(n) = −Σ p log p
- Decouples from H when model is uncertain but picks a common token (hedging) or confident but picks a rare token (committed hallucination)
- Paper reports 73.16% mean AUROC with min-pooling across 9 benchmarks
- Now implemented: `token_entropies_and_spilled()` in model_utils, `compute_spilled_energy_features()` in feature_utils
- **Needs validation** in Step 131

---

## Branch situation

| Branch | Status | Contents |
|--------|--------|----------|
| `master` | Production | Steps 1–125; 16-feature binarized L-SML |
| `experiment/lsml-variants` | **Active, not merged** | K_range fix + continuous L-SML + spilled energy (Steps 128–130) |

**To merge** (after Step 131 validation):
```bash
git checkout master
git merge experiment/lsml-variants
git push origin master
```

---

## Running experiments (Colab — GPU needed)

- **Step 131** (priority): `Spectral_Analysis_SpilledEnergy_Verify.ipynb` on `experiment/lsml-variants`
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
| Step 131 | SpilledEnergy_Verify.ipynb | ⏳ | **NEXT — needs Colab GPU run** |

---

## Best results (reference, do not use Step 100 supervised numbers)

| Setup | L-SML AUC | Notes |
|-------|-----------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **90.0%** | spectral features work on long reasoning |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
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
8. Hedging count: ruled out — not formalized, domain-dependent, weaker than spectral. Do not implement.
9. Continuous L-SML (`lsml_continuous_pipeline`) is the candidate replacement for binarized pipeline — pending Step 131 validation before merge.
10. Spilled Energy signs are **initial estimates** — `min_spilled=+1` may need correction after Step 131 Cell 14.

---

## Deferred

- Phase 10 RAG re-run with variant=4 prompt — still pending, low priority
- LapEigvals integration into spectral_utils — potential Group D feature for M=12 design, low priority
- M=9 orthogonal feature set experiment — contingent on Step 131 confirming ΔE orthogonality
- Local cluster diagnostic: `python scripts/analyze_features.py --features all` — CPU-only, reads `local_cache/`
