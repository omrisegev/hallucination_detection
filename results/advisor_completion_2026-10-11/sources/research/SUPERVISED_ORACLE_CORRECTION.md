# SUPERVISED ORACLE CORRECTION — ML evaluation guidelines

This document serves as a permanent reference for future coding agents (such as Claude/Gemini) working on the Hallucination Detection repository. It details a critical methodological mistake made in **Step 142** during the implementation of the supervised Logistic Regression (LR) oracle baseline, how it was corrected, the updated results, and coding rules for machine learning evaluation in this project.

---

## 1. The Mistakes in the Step 142 Implementation

The initial implementation of the supervised baseline (`scripts/logistic_oracle.py`) reported that the unsupervised method (L-SML continuous) outperformed the supervised baseline. This mathematically anomalous finding was due to two major evaluation bugs:

### Bug A: The `cross_val_predict` Calibration Pitfall (Methodological Bug)
* **What happened:** The code used `cross_val_predict(pipe, X, y, cv=skf, method='predict_proba')[:, 1]` to generate out-of-fold (OOF) probabilities, concatenated them across all 5 folds into a single array, and computed a single global AUROC.
* **Why it is wrong:** The probability estimates returned by `cross_val_predict` for different folds are computed by models trained on different subsets of data. These models have slightly different probability scales (calibration shifts in bias/intercepts and coefficients). Sorting them globally mixes up these uncalibrated values, artificially dragging down the concatenated OOF AUROC by **~1.0pp** across all cells.
* **The Rule:** Never evaluate ranking metrics (like AUROC, Precision-Recall AUC) on concatenated probabilities from `cross_val_predict`. Instead, evaluate the metric on each test fold individually and **average the fold scores**.

### Bug B: Neglecting Prevalence Skew / Class Imbalance (Optimization Bug)
* **What happened:** The model was trained using default parameters: `LogisticRegression(C=1.0)`.
* **Why it is wrong:** Many of the datasets in this project have extreme target imbalance (e.g., 90% correct responses / 10% hallucinations). Logistic regression minimizes binary cross-entropy (log-loss), which does not directly maximize AUROC (a rank-based metric). Under extreme skew, minimizing cross-entropy focuses on optimizing the calibration of the majority class, sacrificing the relative ordering of the minority class, which degrades the AUROC score.
* **The Rule:** Always use `class_weight='balanced'` when fitting classifiers on skewed datasets if the target evaluation metric is AUROC.

---

## 2. Corrected Code Architecture

The corrected implementation in [logistic_oracle.py](file:///C:/Users/omris/TAU/hallucination_detection/scripts/logistic_oracle.py) uses three main components to ensure mathematically correct and fast evaluation:

1. **`safe_auc_raw`:** Computes raw AUROC using `roc_auc_score` and takes `max(auc, 1 - auc)` to handle arbitrary sign orientations.
2. **`cv_avg_auc_with_ci`:** Performs Stratified 5-Fold CV, fits on train folds, predicts on test folds, and computes the individual fold AUROCs. It averages the fold scores to eliminate fold-to-fold calibration shifts. Confidence intervals are calculated via a fast bootstrap over test predictions per fold (which avoids refitting the estimator 1000 times).
3. **`lr_oracle_auc_variants`:** Compares five variants of the supervised baseline:
   * `std_cv`: Standard Averaged Fold CV (no class weights).
   * `bal_cv`: Balanced Averaged Fold CV (`class_weight='balanced'`).
   * `legacy_cv`: Buggy concatenated OOF (for reference).
   * `std_in`: Standard In-Sample ceiling.
   * `bal_in`: Balanced In-Sample ceiling.

---

## 3. Corrected Results (macro AUROC, common-cell basis)

**Snapshot note (updated 2026-07-01).** This table is a point-in-time snapshot. The figures originally here were the Step-143 numbers (2026-06-25); the local feature caches were then recomputed in **Steps 144–146** (paper-accurate spectral-entropy, SelfCheckGPT baseline, Phase 12 / GPQA fixes), which shifted the numbers (e.g. the 16-feat balanced ceiling moved 72.8% → 79.3%). Sections 1, 2, 4, and 5 (methodology and rules) are unchanged and remain authoritative.

**Common-cell convention.** Macros average only over cells where BOTH the unsupervised CONT score and the supervised LR score exist for that feature set (28 cells). The earlier "29-cell" macro counted a CONT-only cell (`trivia_qa_traces`, where LR is N/A) toward CONT alone, inflating the CONT macro and understating the supervised gap by ~1pp. Reproduce with `python scripts/oracle_report.py`.

| Feature Set | Unsupervised CONT | Supervised LR CV (Balanced Avg) | Supervised gap | In-Sample Ceiling (Balanced) |
| :--- | :---: | :---: | :---: | :---: |
| **5-Feat (GOOD_5)** | 64.2% | **68.9%** | +4.7pp | 70.5% |
| **9-Feat (STABLE_H9)** | 62.9% | **66.8%** | +3.8pp | 73.7% |
| **16-Feat (ALL_H16)** | 64.1% | **67.8%** | +3.6pp | 79.3% |

*The supervised baseline consistently beats the unsupervised CONT pipeline once corrected. The gap is largest on GPQA/RAG (+4–6pp) and negligible on reasoning (+0.3–0.6pp), where both methods already sit near the feature ceiling. See `scripts/oracle_report.py` for the per-domain breakdown and `scripts/lr_convergence.py` for the feature-count convergence view.*

---

## 4. How to Recreate the Results

To regenerate the corrected results, pickle files, and figures, run the following command from the repository root:

```bash
python scripts/logistic_oracle.py
```

This will read the cached features from `local_cache/` and write the outputs to:
* **Data file:** `results/logistic_oracle.pkl`
* **Visualization:** `results/logistic_oracle.png`

---

## 5. Lessons for Future Coding in this Repository

When adding models, baselines, or evaluations, follow these rules:

1. **Verify your baselines mathematically:** If an unsupervised method appears to beat a supervised oracle, double-check your code for evaluation leaks, class weights, and calibration errors before accepting the result.
2. **Handle target skew explicitly:** Always check target class prevalence. For AUROC metrics on imbalanced datasets, use `class_weight='balanced'`.
3. **Avoid the `cross_val_predict` pitfall:** Keep folds isolated during evaluation. Average the scores of individual folds rather than evaluating concatenated test outputs.
4. **Distinguish loss surrogate mismatch:** Remember that minimizing cross-entropy is not identical to maximizing AUROC. In-sample direct optimization of AUROC (e.g., Nelder-Mead search for linear weights) represents the true mathematical ceiling of a linear combination of features.
