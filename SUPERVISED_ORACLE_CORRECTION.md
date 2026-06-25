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

## 3. Corrected Results (Macro-Averaged AUROC across 29 cells)

| Feature Set | Unsupervised CONT | Supervised CV (Old Concatenated) | Supervised CV (Corrected Balanced Avg) | Supervised In-Sample (Balanced Ceiling) |
| :--- | :---: | :---: | :---: | :---: |
| **5-Feat (GOOD_5)** | 65.3% | 63.7% | **67.5%** (+2.2pp) | 71.1% (+5.8pp) |
| **9-Feat (STABLE_H9)** | 63.9% | 62.4% | **67.1%** (+3.2pp) | 71.1% (+7.2pp) |
| **16-Feat (ALL_H16)** | 63.0% | 62.6% | **67.6%** (+4.6pp) | 72.8% (+9.8pp) |

*The supervised baseline consistently beats the unsupervised CONT pipeline once corrected.*

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
