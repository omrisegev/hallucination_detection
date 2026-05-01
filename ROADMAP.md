# Roadmap — Hallucination Detection Spectral Pipeline

Last updated: May 2026

---

## Current status

- Phase 7 (GSM8K / Llama-3.1-8B, T=1.0) is **running on Colab** — await results.
- `spectral_utils` package created — shared implementation for all future notebooks.
- Normalization bug identified and fixed in `fusion_utils.py` (z-score before Nadler).

---

## Sequencing rationale

The table below lists all open work items, their dependencies, and why this order
is preferred.

### Why refactoring before re-running

If we fixed the normalization bug directly in Phases 5/6/7 notebooks, we would patch
3 separate files with the same fix. Any future bug would again need 3 patches. By
moving the canonical implementation into `spectral_utils`, **every notebook
automatically gets the correct behaviour** from `pip install git+...`. The refactoring
cost is paid once; the correctness dividend is permanent.

### Why normalization before simple-average ablation

The simple-average fusion baseline (`AUC_mean`) is only meaningful when features are
on the same scale. Without z-score normalization, the "simple average" would be
dominated by `trace_length` (~300) and the comparison would be invalid. Both changes
are already in `fusion_utils.py`.

---

## Work items

### Tier 1 — Immediate (unblocked)

| # | Item | Effort | Notes |
|---|---|---|---|
| 1.1 | **Await Phase 7 results** (GSM8K without normalization) | — | Capture baseline before re-running with fix |
| 1.2 | **Update notebooks to import from spectral_utils** | Low | Replace repeated helper cells with `!pip install git+...` + imports |
| 1.3 | **Push repo to GitHub** | Low | Change remote, initial commit, push |

### Tier 2 — After Phase 7 completes

| # | Item | Effort | Notes |
|---|---|---|---|
| 2.1 | **Re-run Phase 7 (GSM8K) with z-score fix** | Medium | Re-run with `fusion_utils.py` from package; compare vs un-normalized baseline |
| 2.2 | **Nadler Lift ablation in Phase 7** | Low | `compare_mean=True` is now default in `best_nadler_on` — already done |
| 2.3 | **Re-run Phase 5 (MATH-500 + GPQA) with z-score fix** | Medium | Quantify how much the normalization changes the AUC numbers |

### Tier 3 — New experiments

| # | Item | Effort | Notes |
|---|---|---|---|
| 3.1 | **GPQA Diamond with Qwen2.5-72B-Instruct** | Medium | Replace 7B models (30% acc) with 72B (~65% acc); use `quantize_4bit=True` |
| 3.2 | **Cross-temperature fusion revisit** | Medium | Re-run Phase 5 cross-T analysis with normalized features; Nadler Lift may change |
| 3.3 | **GSM8K at T=1.5** | Low | Already have infrastructure; adds one data point for temperature ablation |

### Tier 4 — Thesis framing

| # | Item | Effort | Notes |
|---|---|---|---|
| 4.1 | **Literature search: temperature variation theory** | Medium | Query: "hallucination entropy temperature mode fragility"; check SIA (arXiv:2604.06192) |
| 4.2 | **Write theoretical framing for cross-T fusion** | High | Mode fragility / fluctuation-dissipation analogy; candidate thesis section |
| 4.3 | **Final results table across all phases** | Low | Update `Experiments_Report.md` with normalized numbers once re-runs complete |

---

## Decision points

**After Phase 7 results arrive:**
- If normalized AUC > 87.2% → write up the GSM8K result as the main contribution.
- If normalized AUC < 87.2% but > 72.0% → frame as "competitive unsupervised baseline; supervised Nadler could close the gap."
- If normalized AUC < 72.0% → the normalization fix may have hurt (scale bias was accidentally helpful); investigate feature-by-feature.

**After GPQA 72B run:**
- If AUC > 75% → spectral features generalize to science MCQ when the model is competent.
- If AUC < 65% → spectral features are math-specific (step-by-step symbolic reasoning signal doesn't transfer).

---

## Server migration checklist

When moving from Colab to a dedicated server:

- [ ] Install: `pip install -e ".[inference]"` from the cloned repo
- [ ] Replace `from google.colab import drive, userdata` with local path config
- [ ] Replace `drive.mount('/content/drive')` with direct filesystem paths in CFG
- [ ] HF token: set `HF_TOKEN` environment variable instead of `userdata.get`
- [ ] For large models: ensure `bitsandbytes` is installed for 4-bit quantization
- [ ] Inference loop: same code, just different `BASE_DIR` paths
- [ ] Results will save as `.pkl` locally instead of to Drive

The notebooks themselves need no other changes. The `spectral_utils` package is
environment-agnostic — it works identically on Colab and on a server.
