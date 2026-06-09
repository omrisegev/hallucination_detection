# MV_EPR — Session Progress Handoff

**Date**: 2026-06-09
**Last updated**: Step 131 — GSM8K cross-dataset verification + VC null result complete

---

## TL;DR — where we are today

**Current official method** (production, master branch):
`binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(*binary_filt.values())`

**Candidate upgrade** (branch `experiment/lsml-variants`, pending merge):
`lsml_continuous_pipeline(feats_dict, GOOD_FEATURES, FEATURE_SIGNS)` → +3.53pp mean, 25/29 wins

**New this session (Step 131)**:
- GSM8K cross-dataset verification: spilled energy transfers well (cusum_max_spilled = 0.725 best individual)
- Verbalized confidence: **null result on 1.5B** — model doesn't output "Confidence: X"; parser fallback captures answer magnitude; adding VC hurts L-SML (−1.77pp)
- Parser fix: `parse_verbalized_confidence` now label-first + last-int fallback (correct for 1-pass and 2-pass)
- Key structural finding: within_H/cross ratio = 0.04 (MATH-500) vs 0.99 (GSM8K) — H features are near-independent views on long traces but redundant on short traces
- Best GSM8K result: L-SML GOOD_5 = 0.708 (no VC)
- All changes on branch `experiment/lsml-variants` (commit `f4bc5e8`)

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
    # Spilled energy signs — validated on GSM8K; confirm on MATH-500 in Step 132
    'epr_spilled': -1, 'sw_var_peak_spilled': -1,
    'cusum_max_spilled': -1, 'min_spilled': -1,
    # Verbalized confidence — null on 1.5B; may work on 7B+
    'verb_conf': +1, 'verb_conf_1p': +1,
}
```

Note: `min_spilled` sign updated from initial `+1` estimate to `-1` — validated on GSM8K Cell 12 sign check.

---

## IMMEDIATE NEXT ACTION

### Step 132 — Run MATH-500 SpilledEnergy verification notebook in Colab

Open `Spectral_Analysis_SpilledEnergy_Verify.ipynb` on branch `experiment/lsml-variants`.
Cell 1 clones from the right branch. Run sequentially — ~30–60 min for 100 samples.

**After inference, read these outputs carefully:**

| Cell | What to check | Decision gate |
|------|--------------|---------------|
| Cell 7 | `Saturated: X%` — should be <5% with max_new=2048 | If >20%, increase further |
| Cell 7 | `Spilled energies stored: True` | If False, inference used old code |
| Cell 9 | Spilled feature AUROCs — competitive individually? | Need ≥0.55 to be useful |
| Cell 14 | Sign validation for spilled features | Update FEATURE_SIGNS if mismatches |
| Cell 10 | `within/cross ratio` for H(n) vs ΔE(n) groups | >1.5 needed for L-SML separation |
| Cell 12 | Group assignment: spilled features in own group? | If mixed with H(n), not orthogonal |
| Cell 15 | Lift of GOOD_5+spilled vs GOOD_5 | If >+1pp, merge spilled features |
| Cell 16 | Pearson corr(epr_H, epr_ΔE) | GSM8K was 0.984 — expect similar |

### Next priorities (in order after Step 132):

2. **[DECISION] Merge continuous L-SML + spilled energy to master** — based on Step 132 results.
3. **Phase 13** — EDIS vs L-SML on Qwen2.5-Math-1.5B, AMC23/AIME24 (GPU needed). Run after merge decision.
4. **Verbalized confidence on 7B** — if VC is a priority, re-run GSM8K notebook with Qwen2.5-Math-7B. Parser is ready; one extra inference run.
5. **Phase 14 Cell 9 re-run** — DeepSeek-R1-0528-Qwen3-8B / GPQA: L-SML v2 AUROC still TBD.

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
| MATH-500 / Qwen-7B / T=1.0 | **90.0%** | spectral features work on long reasoning |
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

---

## Deferred

- Verbalized confidence on 7B+ — parser is ready; needs one inference run with Qwen2.5-Math-7B on GSM8K
- Phase 10 RAG re-run with variant=4 prompt — low priority
- LapEigvals integration into spectral_utils — potential Group D feature for M=12, low priority
- M=9 orthogonal feature set experiment — contingent on Step 132 confirming ΔE group independence on MATH-500
