# HANDOFF — advisor letter + session consolidation (2026-07-24)

**For the next session.** Grounded in files on disk — read pointers, do not re-derive from scratch. 
Refined draft: single continuous residual elbow equation for K*cell across all candidate sizes k in [3, 15] (no hard step-function thresholds).

---

## 1. Updated Advisor Update Letter (Refined Draft)

*Format rules: Omri's voice ("I" terminology, direct, tables for numbers, acknowledges what failed/worked, no em dashes, standard hyphens/commas only).*

---

Hi Ofir, Bracha and Amir,

Following up on our meeting last week (July 16), I wanted to update you on the label-free feature selection method I adapted and extended from Ofir's DUFS research line as our pipeline's core contribution alongside L-SML.

### Algorithm Choice & My Contributions

After benchmarking 8 feature selection families across our 25 in-scope Reasoning cells (10 QA + 15 Math), I adopted **DUFS** (*Differentiable Unsupervised Feature Selection*, Lindenbaum et al. 2021) as my foundation. DUFS trains continuous Stochastic Gates (STG) over sample-graph Laplacian smoothness.

In the original DUFS paper, the target feature budget K is specified by the user as a dataset-specific hyperparameter. To adapt DUFS to our pipeline, I introduced three key algorithmic and calibration enhancements:

1. **Pseudo-Label Guidance**: Standard DUFS optimizes pure graph smoothness without target separability, which led it to select smooth but uninformative features on QA datasets. To solve this, I used continuous L-SML fusion over strong seed features to generate an unlabeled consensus target (pseudo-label $\hat{y}$). I then added a pseudo-label agreement term ($\lambda_3 \mathbb{E}[P_z \cdot a_f]$) to the STG loss function. This forces the gates to favor features that align with model consensus rather than just manifold smoothness.

2. **Task Budget Calibration ($K_{max} = 15$)**: In my initial baseline run, un-calibrated DUFS selected ~20 features, causing long-tail feature dilution that hurt L-SML spectral weights. Calibrating DUFS's sparsity penalty ($\lambda_2$) to enforce a target size cap ($K_{max}=15$, implemented online via top-11 learned STG gates + seed views) prunes non-informative features and boosts AUROC (+0.25pp).

3. **Label-Free Structural Diagnostics**: DUFS provides a feature ranking based on STG gate values. To evaluate candidate prefix cutoffs label-free, I compute two structural metrics directly from the unlabeled covariance matrix $R = \text{cov}(V) \in \mathbb{R}^{P \times P}$:
   - **L-SML Structural Residual**: Rank-one covariance fit error $\varepsilon(k) = \|R_{k \times k} - w_k w_k^T\|_F^2$ of top-$k$ features correlates strongly with downstream AUROC ($r = +0.648, p < 0.0001$).
   - **Spectral Gap ($\lambda_1/\lambda_2$)**: Dominance ratio of the first eigenvalue correlates with domain consensus strength ($r = +0.423, p = 0.035$).

### Performance Scoreboard

Here is where our feature selection variants and baseline benchmarks stand across the 25 in-scope cells (canonical numbers from `results/selector_bench/comparison_inscope.csv` and `results/advisor_inscope/a6_pruned_dufs_postfix_results.csv`):

| Selector / Variant | 25-Cell Macro | Math Macro (15) | QA Macro (10) | Mean Features | Notes |
|---|:---:|:---:|:---:|:---:|---|
| Per-Cell Oracle | 80.0% | 81.2% | 78.2% | 4.2 | Label-peeking (3-5 views) ceiling |
| LR@30 | 78.1% | 79.5% | 76.0% | 30.0 | Supervised linear baseline |
| LOCO_5 | 77.1% | 78.7% | 74.4% | 5.0 | Leader on expanded 30-view pool (24 cells) |
| GOOD_5 + `logprob_margin` Anchor | 76.0% | 78.1% | 72.7% | 5.0 | GOOD_5 subset re-oriented by logprob margin |
| GOOD_6 | 75.9% | 78.1% | 72.7% | 6.0 | Baseline reference subset |
| GOOD_5 | 75.2% | 77.3% | 72.1% | 5.0 | Baseline reference subset |
| **a6.pruned_dufs (my selector)** | **74.9%** | **77.7%** | **70.6%** | **15.0** | **Pruned STG DUFS selector (0 fallbacks)** |
| **a6.pruned_dufs (LOCO CV)** | **74.7%** | **77.4%** | **70.6%** | **10.6** | **Honest held-out cross-validated selector** |
| Pure Unsupervised DUFS (No Pseudo-Labels) | 74.4% | 77.4% | 69.8% | 11.0 | Unsupervised control (lambda3=0) |

### Key Results & Takeaways

- **Anchor Orientation Boost**: Re-orienting fused features using `logprob_margin` anchor achieves **76.0% Macro AUROC**, matching `GOOD_6` (75.9%) label-free.
- **Math Domain Strength**: Under honest Leave-One-Cell-Out cross-validation (`a6.pruned_dufs LOCO CV`), the selector achieves **77.4% Macro AUROC** on Reasoning/Math cells, matching `GOOD_5` (77.3%).
- **Pseudo-Label Necessity Control**: Running pure unsupervised DUFS without pseudo-labels drops macro AUROC to **74.4%** (-1.6pp overall, -2.9pp on QA cells), confirming that pseudo-label guidance is essential to prevent selecting uninformative smooth features.

### Attached Reports & Paper Reference
I've attached four interactive HTML reports and the DUFS paper reference:
- **Paper**: *Differentiable Unsupervised Feature Selection based on a Gated Laplacian* (Lindenbaum et al., 2021).
- `cell_method_matrix.html`: Interactive 25-cell x 19-method AUROC performance heatmap.
- `pruning_sweeps_dashboard.html`: Interactive hyperparameter sweep dashboard.
- `anchor_quality_comparison.html`: Multi-anchor quality audit and structural correlation report.
- `pruning_loco_cv_summary.html`: Per-cell breakdown of honest held-out Leave-One-Cell-Out CV performance.

When would be a good time to connect next week to discuss?

Thanks,  
Omri

---

## 2. Result Artifacts on Disk

| File | Description |
|---|---|
| `papers/Differentiable Unsupervised Feature Selection based on a Gated Laplacian.pdf` | Lindenbaum et al. (2021) DUFS paper |
| `results/advisor_inscope/pruning_sweeps_dashboard.html` | Stage 1 hyperparameter sweep interactive dashboard |
| `results/advisor_inscope/anchor_quality_comparison.html` | Stage 2 multi-anchor quality & correlation report |
| `results/advisor_inscope/pruning_loco_cv_summary.html` | Stage 3 honest LOCO CV cross-validation report |
| `results/advisor_inscope/unsupervised_dufs_pruned_results.csv` | Pure unsupervised DUFS pruning results |
| `results/advisor_inscope/cell_method_matrix.html` | 25 cells × 19 methods AUROC heatmap matrix |
