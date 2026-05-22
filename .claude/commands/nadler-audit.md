---
description: Validate a Nadler fusion setup against all 4 invariants. Use before running best_nadler_on() on a new dataset, after getting unexpectedly low AUC, or when proposing a new feature subset. Accepts a code snippet, a description, or just the feature list.
---

Check the described Nadler fusion setup against the 4 mandatory invariants. For each, report PASS or FAIL with a concrete fix.

## The 4 Nadler Invariants

### Invariant 1 — Minimum 3 views (HARD REQUIREMENT)
**Check**: Does the proposed subset contain at least 3 features?

- PASS: subset size ≥ 3
- FAIL: "2-view Nadler collapses to near-random AUC. Add a third independent feature. Run individual AUC table and pick the highest-AUC feature with |ρ| < 0.75 against the existing two."

### Invariant 2 — Z-score normalization before fusion (REQUIRED)
**Check**: Is `zscore()` applied to each view before passing to `nadler_fuse()` or `best_nadler_on()`?

Common mistakes:
- Passing raw feature arrays directly to `nadler_fuse()`
- Calling `best_nadler_on(..., normalize=True)` — `normalize` is NOT a valid parameter; `best_nadler_on` applies z-score internally, but raw calls to `nadler_fuse` do not
- Forgetting z-score on short-trace QA data (most critical there; negligible on long math traces)

- PASS: `best_nadler_on()` is used (z-score is internal) OR manual `zscore()` calls appear before `nadler_fuse()`
- FAIL: "Apply `zscore()` from `spectral_utils.fusion_utils` to each view: `views = [zscore(feats[fn]) for fn in subset]`, then pass to `nadler_fuse(*views, labels)`"

### Invariant 3 — Correlation filter |ρ| < 0.75 (REQUIRED)
**Check**: Is every pairwise Spearman correlation between proposed features below 0.75?

- If using `best_nadler_on()`: automatically enforced — it skips any subset where any pair has |ρ| ≥ 0.75
- If manually selecting a subset: must check manually

To check manually:
```python
from scipy.stats import spearmanr
for i, fi in enumerate(subset):
    for j, fj in enumerate(subset):
        if i >= j: continue
        rho, _ = spearmanr(feats[fi], feats[fj])
        status = 'OK' if abs(rho) < 0.75 else 'FAIL — too correlated'
        print(f'{fi} × {fj}: ρ={rho:.3f} {status}')
```

- PASS: all pairs |ρ| < 0.75
- FAIL: "Remove one of the correlated features. Candidates to drop: prefer the one with lower individual AUC. Alternatively, use `best_nadler_on()` which enforces this automatically."

### Invariant 4 — Feature sign orientation (REQUIRED)
**Check**: Is each feature oriented so that **higher value = model more likely to be correct**?

Standard sign conventions in this project:
| Feature | Sign | Reason |
|---------|------|--------|
| `epr` | Higher = correct | High entropy production = confident generation |
| `trace_length` | Higher = correct | Longer reasoning = more deliberate |
| `cusum_max` | Higher = correct | Large CUSUM shift correlates with commitment |
| `hurst_exponent` | Higher = correct | Long-range dependence = structured reasoning |
| `pe_mean` | Higher = correct | Higher permutation entropy on correct outputs |
| `sw_var_peak` | Higher = correct | Peak sliding-window variance = local confidence burst |
| `high_band_power` | Context-dependent | Check per-dataset with individual AUC |
| `rpdi` | Higher = hallucinated → flip | RPDI high = reasoning deviation = wrong → negate before fusing |

`best_nadler_on()` handles sign orientation automatically via AUC > 0.5 check. If using `nadler_fuse()` directly, orient manually:
```python
view = zscore(feats[fn])
if roc_auc_score(labels, view) < 0.5:
    view = -view  # flip sign
```

- PASS: `best_nadler_on()` used (auto-handles sign) OR manual flip applied
- FAIL: "Flip the sign of features where individual AUC < 0.5 before passing to `nadler_fuse()`"

---

## Common Failure Mode Summary

| Symptom | Most likely cause | Invariant |
|---------|-------------------|-----------|
| AUC ≈ 50% despite good individual features | 2-view fusion | #1 |
| AUC ≈ 50% with 3+ features | Missing z-score | #2 |
| Negative lift (Nadler worse than simple avg) | Correlated features OR wrong sign | #3 or #4 |
| `TypeError: best_nadler_on() got unexpected keyword argument 'normalize'` | Invalid kwarg | #2 (remove it) |
| AUC varies wildly across runs | Too few samples (need ≥ 200 for stable ranking) | Not an invariant, but flag it |

---

After reporting the audit, recommend one concrete next step based on the first failing invariant found (fix in invariant order: 1 → 2 → 3 → 4).
