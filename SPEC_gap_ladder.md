# SPEC — Gap Decomposition Ladder (Step 198)

**Author of spec:** Claude (design + review + analysis)
**Implementer:** Gemini
**Status:** ready to implement. Do not deviate from the output schemas in §6; the analysis
is written against those exact column names.

---

## 1. Why this exists (read this, it changes what you may conclude)

A proposal was drafted to attack a "stationary sign bottleneck" with three new methods
(regime-conditional signs, Similarity Network Fusion, GMM density-ratio calibration).
I audited its empirical premises against the result files. They do not hold:

| Draft claim | Verified value | Source |
|---|---|---|
| coqa fusion "collapsed to 0.4483" | 0.4483 is `auc_anchor4`, the seed rule we **already rejected and replaced** in Step 1 of this work. Live pipeline: GOOD_5 = 0.6841, GOOD_6 = 0.6674 | `results/advisor_inscope/pseudolabel_quality_audit.csv` |
| best single feature 0.6408 beats the fusion | 0.6408 is `auc_best_seed`. GOOD_5 fusion = 0.6841, so fusion **beats** the best single feature by +4.3pp on that cell | same |
| supervised oracle ~0.80 macro, gap +4.74pp | 0.8000 is the **math-only** LR mean. True macro = **0.7810**. Real gaps vs GOOD_6: macro 2.16pp, QA 2.50pp, math 1.93pp | `results/advisor_inscope/lr_oracle_audit.csv`, `fset=30` |

And the structural problem with the proposal: **the supervised oracle is logistic regression,
which is itself a stationary global linear model with fixed per-feature signs.** It reaches
QA 0.7524 / macro 0.7810 on the same features. So the model class already contains a solution
above where we are. The binding constraint is *estimation without labels*, not model capacity.
Adding capacity attacks a term we have no evidence is limiting.

Also, the QA deficit is not diffuse. It is two cells:

| QA cell | GOOD_6 | LR(30) oracle | gap |
|---|---|---|---|
| inside_coqa_llama7b | 0.6674 | 0.8257 | **+15.8pp** |
| seiclr_triviaqa_opt30b | 0.5884 | 0.7202 | **+13.2pp** |
| (other 8 cells) | | | −7.3 to +5.0, mean ≈ 0 |

On both, a single fixed-sign fixed-weight linear model recovers 0.83 / 0.72 with labels.

**Therefore: do not build any of the three proposed methods yet.** Build the measurement that
says which of them, if any, has headroom. That is this spec. It is CPU-only and reuses
existing code. It is the experiment that would have saved the adaptive-K effort, where we
built the whole thing and then measured r_s = +0.007.

---

## 2. What you are building

One new script: **`scripts/gap_ladder.py`**. It evaluates a ladder of 8 rungs on all 25
in-scope cells, at 2 feature sets, and writes 4 CSVs + 1 HTML + 1 JSON.

Each rung answers "how well could we do if we were handed one more thing for free?" The
**differences between adjacent rungs** decompose the 2.16pp macro gap into named causes.

| Rung id | Name | Labels? | Definition |
|---|---|---|---|
| `R0` | `lf_lsml` | No | **Status quo.** `eval_subset_flex(ctx, cols, fusion='lsml')` — L-SML continuous fusion, label-free `anchor_orient`, raw AUROC. |
| `R0b` | `lf_lsml_rank` | No | Same as R0 but each column normal-score transformed first (see §4.6). Exploratory label-free candidate. |
| `R1` | `oracle_single` | Yes | Best single feature with oracle polarity: `max_j max(auc_j, 1 - auc_j)`. |
| `R2` | `oracle_sign_eq` | Yes | **Oracle signs, equal weights.** Flip each column by its oracle sign, then fuse by unweighted mean of z-scored columns. |
| `R3` | `oracle_lin` | Yes | **Supervised linear oracle.** 5-fold CV logistic regression (§4.4). |
| `R4` | `oracle_nonlin` | Yes | **Supervised nonlinear oracle.** Same folds, `HistGradientBoostingClassifier` (§4.5). |
| `R5` | `oracle_regime_sign` | Yes | **Non-stationary signs, best case.** Per-feature signs fit *within* anchor tertiles (§4.7). |
| `R6` | `oracle_target_select` | Yes | **Perfect consensus target.** Run PL-mRMR selection with `y_hat` replaced by the true labels, then score the selected subset through the normal label-free fusion (§4.9). `fset=FULL` only. |

### What each gap means

```
R0  -> R2   sign recovery + fusion loss   (can we even orient features correctly, label-free?)
R2  -> R3   weight estimation loss        (how much is uniform weighting costing us?)
R3  -> R4   NONLINEARITY headroom         (is there anything above a linear model at all?)
R3  -> R5   NON-STATIONARY-SIGN headroom  (do per-sample signs buy anything, even with labels?)
```

`R3 -> R4` is the kill-test for the GMM density-ratio direction and for most of the SNF
direction. `R3 -> R5` is the kill-test for the regime-conditional-sign direction. Both are
tested **with labels**, i.e. under the most generous possible conditions. If a direction has
no headroom with labels, it has none without them, and we abandon it.

---

## 3. Feature sets — run the whole ladder twice

Every rung runs at **both** of these, because otherwise "feature set" and "estimation quality"
are confounded (GOOD_6 has p=6; the LR oracle number we are chasing used p=30):

| `fset` value | Columns |
|---|---|
| `GOOD_6` | `spectral_utils.subset_sweep.GOOD_6` names mapped to `u.pool` indices |
| `FULL` | all of `u.pool` |

Record the actual `p_used` per cell (it varies; e.g. `inside_coqa_llama7b` has p=27).
If a cell has fewer than 3 usable GOOD_6 columns, skip that (cell, fset) pair and write the
reason in `notes`. Do not silently drop it.

---

## 4. Exact protocol

### 4.1 Loading

```python
from compare_anchor_quality import load_all_inscope_cells   # scripts/
from inscope_cells import GROUP                             # scripts/
from spectral_utils.selector_bench import eval_subset_flex
from spectral_utils.subset_sweep import GOOD_6

cells = load_all_inscope_cells()   # {cell_key: {'unlabeled': UnlabeledCell, 'labels': ..., ...}}
```

`UnlabeledCell` has no `.labels` by design. `eval_subset_flex` needs one. Reuse the exact
`Ctx` shim already written at `scripts/bench_seven_arms.py:65-72` — import it or copy it
verbatim. **Do not re-derive `V`, `anchor`, or `rho`.** Use them as `prepare_cell` produced
them, except where a rung explicitly defines a transformation.

Note: `u.V` is already z-scored and **label-free sign-oriented**. That label-free orientation
is exactly what rung R2 tests against, so R2's oracle signs are computed relative to `u.V`
as given.

### 4.2 R1 — oracle single feature

For each column `j`: `a_j = roc_auc_score(labels, V[:, j])`. Report
`R1 = max_j max(a_j, 1 - a_j)`. Also record `argmax` feature name in `notes`.

### 4.3 R2 — oracle signs, equal weights

```
s_j    = +1 if a_j >= 0.5 else -1          # oracle sign, uses labels
score  = mean_j ( s_j * zscore(V[:, j]) )
R2     = roc_auc_score(labels, score)
```
Report **two variants**:
- `R2` (in-sample signs) — this is an upper bound, label it as such.
- `R2_cv` (signs fit on the 4 training folds of the §4.4 split, applied to the held-out fold;
  per-fold AUROC then averaged).

Report both. Do not report only the favorable one.

### 4.4 R3 — supervised linear oracle

Match the existing convention exactly so the result is comparable to
`results/advisor_inscope/lr_oracle_audit.csv` (see `scripts/logistic_oracle.py:240-247`):

```python
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(C=1.0, class_weight='balanced',
                                        max_iter=1000, solver='lbfgs'))
```

Per-fold AUROC, **floored at 0.5** (direction-free convention for a supervised oracle),
then mean across folds. Also record the **unfloored** mean in `auroc_nofloor`.

**This same `skf` object / same split indices must be reused for R2_cv, R4 and R5_cv.**
Generate the splits once per (cell, fset) and pass them down. Paired comparisons across
rungs are meaningless otherwise.

### 4.5 R4 — supervised nonlinear oracle

Same folds, same flooring, same scaling-irrelevant input:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.15,
    random_state=42)
```

Handle class imbalance with `sample_weight` = balanced weights computed on the training fold
(`sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)`), to match R3's
`class_weight='balanced'`. **These hyperparameters are fixed a priori. Do not tune them.**
If you tune them against the outcome, the R3->R4 gate is invalid and the whole run is wasted.

### 4.6 R0b — normal-score (rank) transform, label-free

```python
from scipy.stats import rankdata, norm
r  = rankdata(V[:, j])                  # per column, per cell
Vt[:, j] = norm.ppf((r - 0.5) / len(r))
```
Then build a `Ctx` with `V = Vt` (keep the same `anchor`, also normal-score transformed) and
call `eval_subset_flex` identically to R0. This is a legitimate label-free candidate, not an
oracle: it tests whether monotone-invariant encoding helps on our heavy-tailed features, which
is the only genuinely useful idea inside the SNF proposal, at O(N log N) instead of O(N²p).

### 4.7 R5 — regime-conditional (non-stationary) signs

The generous-case test of the regime direction.

```
regimes: R = 3, tertiles of `u.anchor` (the label-free anchor already on the cell)
for each regime r, for each column j:
    a_{j,r} = roc_auc_score(labels[r], V[r, j])       # oracle, uses labels
    s_{j,r} = +1 if a_{j,r} >= 0.5 else -1
score_n = mean_j ( s_{j, regime(n)} * zscore(V[:, j])_n )
R5 = roc_auc_score(labels, score_n)
```

`R = 3` is fixed a priori. Do not sweep it.
Skip a (regime, feature) pair and fall back to the global oracle sign if that regime has
fewer than 30 samples or is single-class; count these in `notes`.

Report **both**:
- `R5` (in-sample signs) — the absolute ceiling the direction could ever reach.
- `R5_cv` (regime signs fit on training folds, applied to the held-out fold).

The in-sample `R5` is the one the gate uses, deliberately: we want to kill the direction under
the most generous possible conditions, so that a negative result is conclusive.

### 4.9 R6 — perfect consensus target (the missing kill-test for the whole selection line)

Added 2026-07-24 after establishing that the pseudo-label under `A6_SEED_RULE=good6` is **exactly
the GOOD_6 fused score on 25/25 cells** (`pseudolabel_quality_audit.csv`: `auc_pl` == `auc_good6`
elementwise). Every label-free selector we have is therefore ranking candidate features by agreement
with GOOD_6's own score, which biases selection toward features redundant with GOOD_6 and against
the features that would correct it. The family may be capped at GOOD_6 by construction.

R6 tests that directly by removing the cap:

```
y_hat  <- true labels                      # the only change
agree  <- _corr_with(V[:, sel_cols], y_hat)
order  <- _plmrmr_order(V[:, sel_cols], agree, alpha=MRMR_ALPHA)
cols   <- seed_cols + [sel_cols[j] for j in order[:K-len(seed_cols)]]
R6     <- eval_subset_flex(ctx, cols)      # normal label-free fusion + anchor orientation
```

Reuse `_seed_cols`, `_corr_with`, `_plmrmr_order` and `MRMR_ALPHA` from
`spectral_utils/selectors/a6_pseudolabel_gates.py` unchanged. Only the target is swapped. Report
R6 at **K = 15** (the fixed size used by the `D2_alone` arm in `seven_arm_summary.csv`), so it is
directly comparable to that row's 0.7573.

Note the asymmetry that makes this a clean test: the labels enter **only the selection step**. The
fusion and the orientation stay label-free, so R6 is "what our real pipeline would score if someone
handed us a perfect consensus target".

### 4.8 Confidence intervals

`spectral_utils.fusion_utils.boot_auc(labels, scores)` — note the argument order is
**(labels, scores)**. Use it for the non-CV rungs (R0, R0b, R1, R2, R5). For the CV rungs
(R2_cv, R3, R4, R5_cv) use the fold-bootstrap in `scripts/logistic_oracle.py:88-122`
(`cv_avg_auc_with_ci`) so the CIs are comparable to the existing oracle table.

---

## 5. Mechanism diagnostics (run alongside; these matter as much as the AUROCs)

These test the proposal's premises **directly**, independent of any fusion. Computed per cell
at `fset=FULL` only.

**5.1 Label-free orientation error rate.** For each column, does `u.V`'s label-free sign
agree with the oracle sign? Report the count and fraction wrong. *This is the direct
measurement of "is sign recovery our problem".*

**5.2 Regime sign disagreement.** Fraction of (feature, regime) pairs whose oracle sign
differs from that feature's **global** oracle sign. *This is the direct test of whether
non-stationary signs exist at all.* If this is near 0, the regime direction is dead before
any AUROC is computed. Weight each feature equally.

**5.3 Non-monotonicity gain.** Per column: bin the feature into 4 quantile bins, map each bin
to its empirical positive rate on the training data, score by that mapping, take AUROC
(`auc_binned`, 5-fold CV'd with the same folds to avoid trivial in-sample inflation). Report
`nonmono_gain_j = auc_binned_j - max(a_j, 1 - a_j)`. *This is the direct test of the
U-shaped-feature premise behind the GMM direction.* Report mean, max and 90th percentile
across features per cell.

**5.4** Also record `lsml_K`, `lsml_residual` (both from `eval_subset_flex`'s return dict at
R0/FULL), `anchor_name`, and `anchor_auc = max(auc(anchor), 1 - auc(anchor))`.

---

## 6. Output schemas — exact, do not rename columns

All under `results/advisor_inscope/`.

### 6.1 `ladder_percell.csv` — one row per (cell, fset, rung)

```
cell, group, fset, p_used, n, pos_rate, rung, auroc, auroc_nofloor,
ci_lo, ci_hi, uses_labels, cv, notes
```
- `group` ∈ {`QA`, `math`} from `inscope_cells.GROUP`
- `rung` ∈ {`R0`,`R0b`,`R1`,`R2`,`R2_cv`,`R3`,`R4`,`R5`,`R5_cv`,`R6`} — 10 rows per cell at
  `fset=FULL`, 9 at `fset=GOOD_6` (R6 is FULL only, since it performs selection)
- `uses_labels` ∈ {0,1}; `cv` ∈ {0,1}
- `auroc_nofloor` = `auroc` for non-CV rungs
- `notes` = free text (argmax feature for R1, fallback counts for R5, skip reasons)

### 6.2 `ladder_summary.csv` — one row per (fset, rung)

```
fset, rung, n_cells, macro_all, macro_qa, macro_math,
d_R0_all, d_R0_qa, d_R0_math,
d_R3_all, d_R3_qa, d_R3_math,
w_p_vs_R0_all, w_p_vs_R0_qa, w_wins_vs_R0_all, w_losses_vs_R0_all,
w_p_vs_R3_all, w_p_vs_R3_qa, w_wins_vs_R3_all, w_losses_vs_R3_all
```
- `macro_all = (10 * macro_qa + 15 * macro_math) / 25`. Macro within a group is the
  **unweighted mean over cells**, not sample-weighted.
- `d_*` are deltas in **absolute AUROC** (not pp), signed, rung minus reference.
- `w_p_*` = paired Wilcoxon signed-rank p-value across cells, two-sided
  (`scipy.stats.wilcoxon`). Report wins/losses counts alongside every p.

### 6.3 `ladder_signdiag.csv` — one row per cell

```
cell, group, n, pos_rate, p_used,
n_labelfree_sign_wrong, frac_labelfree_sign_wrong,
regime_sign_disagree_frac,
nonmono_gain_mean, nonmono_gain_p90, nonmono_gain_max,
lsml_K, lsml_residual, anchor_name, anchor_auc
```

### 6.4 `ladder_featdiag.csv` — one row per (cell, feature), `fset=FULL`

```
cell, group, feature, auc_raw, auc_oriented, labelfree_sign, oracle_sign,
sign_wrong, auc_binned, nonmono_gain, regime_signs, regime_sign_disagree
```
`regime_signs` = a 3-char string like `"++-"` (low, mid, high anchor tertile).
This file is what I use to diagnose `inside_coqa_llama7b` and `seiclr_triviaqa_opt30b`
specifically, so it must be complete for at least those two cells.

### 6.5 `ladder_gates.json` — machine-readable verdicts

```json
{
  "validity": {
    "R3_FULL_macro_qa": 0.0, "R3_FULL_macro_math": 0.0,
    "ref_lr_oracle_qa": 0.7524, "ref_lr_oracle_math": 0.8000,
    "R3_reproduces_lr_oracle": true,
    "R0_GOOD_6_macro_all": 0.0, "ref_good6_macro_all": 0.7594,
    "R0_reproduces_good6": true
  },
  "gates": {
    "nonlinearity":      {"delta_all": 0.0, "delta_qa": 0.0, "p_all": 0.0, "verdict": "DEAD|ALIVE"},
    "nonstationary_sign":{"delta_all": 0.0, "delta_qa": 0.0, "p_all": 0.0, "verdict": "DEAD|ALIVE"},
    "sign_recovery_loss":{"delta_all": 0.0, "delta_qa": 0.0},
    "weight_estimation_loss": {"delta_all": 0.0, "delta_qa": 0.0},
    "target_quality":    {"R6_macro_all": 0.0, "R6_macro_qa": 0.0,
                          "delta_vs_good6": 0.0, "delta_vs_D2_alone": 0.0,
                          "p_vs_good6": 0.0, "verdict": "DEAD|ALIVE"},
    "dominant_term": "sign_recovery|weight_estimation|nonlinearity|nonstationary_sign|target_quality"
  }
}
```

### 6.6 `ladder.html`

Per the standing phase-control directive (visual per step). Follow the existing pattern in
`scripts/compare_anchor_quality.py:build_anchor_dashboard`. Must contain, at minimum: the
R6 target-quality result stated against GOOD_6 and D2_alone, the
ladder bar chart (macro_all / macro_qa / macro_math per rung, both fsets), the four gate
verdicts stated in plain words on the page, the per-cell matrix, and a scatter of
`frac_labelfree_sign_wrong` vs `(R2 - R0)` per cell. Self-contained, no external assets
beyond the Chart.js CDN already used by the existing dashboards.

---

## 7. Pre-registered decision rules — write these into the script before you see results

Computed at `fset=FULL`, and separately for QA-only.

| Gate | Rule | Consequence |
|---|---|---|
| **Nonlinearity** | `macro(R4) - macro(R3) < +0.010` **or** Wilcoxon p > 0.05 | GMM density-ratio (D3) and the capacity argument for SNF (D2) are **DEAD**. Do not build them. |
| **Non-stationary sign** | `macro(R5_insample) - macro(R3) < +0.010` **or** p > 0.05 | Regime-conditional signs (D1) are **DEAD**, refuted with labels. Do not build them. |
| **Sign recovery** | `macro(R2) - macro(R0)` | If this is the largest positive gap, the next build is **better label-free sign estimation**, not more capacity. |
| **Weight estimation** | `macro(R3) - macro(R2)` | If this is the largest, the next build is **better label-free weight estimation**. |
| **Rank transform** | `macro(R0b) - macro(R0)`, Wilcoxon | If ≥ +0.010 and p < 0.05, adopt the normal-score transform. Free win, ship it. |
| **Target quality** | `macro(R6) - 0.7594` (GOOD_6), and `macro(R6) - macro(D2_alone=0.7573)` | If R6 does **not** clear GOOD_6 by ≥ +0.010, then even a perfect consensus target does not rescue selection, and the entire label-free feature-selection line is capped by something other than target quality. Report that plainly; it closes the direction. If R6 clears GOOD_6 substantially, target construction is the lever and it is where the next work goes. |

**Power note.** 25 cells with a paired Wilcoxon resolves roughly ≥1pp consistent effects.
The entire macro gap to the supervised oracle is 2.16pp. That is why the threshold is 1.0pp
and not 0.25pp. Do not report a sub-1pp delta as a finding.

---

## 8. Validity checks — the script must print these and they must pass

1. **R3 at `fset=FULL` must reproduce the existing LR oracle** within ±0.005 macro:
   expected QA 0.7524, math 0.8000, macro 0.7810
   (`results/advisor_inscope/lr_oracle_audit.csv`, rows with `fset=30`, column `floored`).
2. **R0 at `fset=GOOD_6` must reproduce** macro 0.7594 / QA 0.7274 / math 0.7807 within ±0.002
   (`results/advisor_inscope/seven_arm_summary.csv`, row `ref.GOOD_6`).
3. **Ordering sanity:** `R1 <= R2` should hold on most cells and `R2 <= R3` should hold on
   most cells at `fset=FULL`. Where it does not, that is interesting, not a bug: log it.
4. `n_cells == 25` for `fset=FULL`. Print the count.

If check 1 or 2 fails, **stop and report**. It means the data being loaded is not the data the
existing numbers came from, and every downstream conclusion would be wrong.

---

## 9. Rules of engagement

1. **New file only.** Create `scripts/gap_ladder.py`. Do not modify `spectral_utils/`,
   existing scripts, or existing CSVs. Do not run `bench-refresh`.
2. **No outcome-driven choices.** `R=3`, the GBM hyperparameters, the CV seed, the 4 quantile
   bins, and the 1.0pp threshold are all fixed by this spec. If you change any of them, the
   corresponding gate is invalid. If you think one is wrong, say so in your report and run it
   as specified anyway.
3. **Never report only the in-sample variant** where a CV variant is specified. Report both.
4. **Every claim needs a p-value and a win/loss count.** "X beats Y" without
   `wilcoxon p` + `wins/losses` will be rejected in review.
5. **Every number in your write-up must map to a column in §6.** If it does not appear in a
   CSV, do not state it. The last two rounds each contained a number that could not be traced
   to its claimed source; this rule exists because of that.
6. **Do not draw conclusions beyond the gates.** If the nonlinearity gate says DEAD, say DEAD.
   Do not soften it, and do not propose a variant that "might still work" in the same report.
7. Runtime should be well under 30 min on CPU. If it is not, reduce `n_boot`, not the cells.

---

## 10. What to hand back

1. The 4 CSVs, the JSON, and the HTML at the paths in §6.
2. A short report (in chat, not a new md file) containing:
   - the §8 validity check results, PASS/FAIL each
   - the `ladder_summary.csv` table for `fset=FULL`, all 9 rungs
   - the five gate verdicts (nonlinearity, non-stationary sign, sign recovery, weight estimation, target quality) with delta, p, wins/losses
   - the two mechanism numbers I care most about: median `frac_labelfree_sign_wrong` and
     median `regime_sign_disagree_frac`, QA vs math
   - the `inside_coqa_llama7b` and `seiclr_triviaqa_opt30b` rows from `ladder_percell.csv`
   - anything that surprised you, stated plainly
3. Do **not** propose the next method. That is the analysis step, and I do it after reviewing
   this. Hand back measurements.
