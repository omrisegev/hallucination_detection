# HANDOFF - advisor update letter (2026-07-24, Gmail-Optimized Draft)

Hi Ofir, Bracha and Amir,

Following up on our meeting last week (July 16), I wanted to give an update on the feature selection work and the algorithms I've been developing to create our own unique label-free contribution to the pipeline alongside L-SML.

### TL;DR / Key Takeaways
* **Expanded Feature Pool**: Re-ran cluster inference to save logprob distributions and energy series (Z_n), expanding our pool from 16 to 30 views across all 25 core reasoning cells.
* **DUFS + L-SML Pipeline**: Combined Differentiable Unsupervised Feature Selection (DUFS) with L-SML pseudo-labeling, orientation anchoring, and feature cap/pruning into a single label-free pipeline. This reaches **75.7% macro AUROC** (statistically beating GOOD_5 at 75.2%, p = 0.037, and closing within 0.2pp of GOOD_6 at 75.9%).
* **Current Focus**: I am now actively optimizing this joint DUFS + L-SML algorithm across both Reasoning (Math) and QA benchmarks using different gating and weighting strategies to close the remaining 2.4pp gap to the supervised linear oracle (78.1%).
* **Meeting Request**: I would love to meet sometime this week to discuss these results, review the joint optimization strategy, and sanity-check the next steps.

---

### 1. Expanding the Feature Pool (Cluster Re-runs)
In earlier runs, saving only the token entropy series H(n) prevented us from extracting features requiring full logit distributions or top-K probabilities. Inspired by literature on LLM uncertainty (e.g., Kadavath et al. 2022 for varentropy, and energy-based detection literature), I re-ran inference across all cells on the AIRCC cluster to save top-K logprobs and energy series (Z_n).

This expanded our pool from 16 to 30 views per cell (adding Z_n energy, logprob margin, varentropy, and top-K tail mass).
* **RAG & GPQA**: The expanded pool did not revive GPQA (features remain at chance, 0.51 to 0.55) or RAG (only improves on HotpotQA where signal already existed). Both stay out of scope for now, so all evaluations below focus on the 25 core Reasoning (QA + Math) cells.
* **QA & Math Impact**: On our 25 core cells, 7 of the top 10 most informative features now come from these newly added energy and logprob views.

### 2. Feature Selection Benchmark & DUFS Algorithm Optimization
I benchmarked 8 algorithm families targeting different components of our detection pipeline:
1. **GroupFS & DUFS** (Lifshitz et al. 2026, Lindenbaum et al. 2021): Stochastic feature gates trained on sample-graph Laplacian smoothness.
2. **Classical Spectral FS**: Laplacian Score (He et al. 2005), SPEC (Zhao & Liu 2007), and MCFS (Cai et al. 2010).
3. **Concrete Autoencoders** (Balin et al. 2019): Gumbel-softmax relaxation minimizing linear reconstruction error.
4. **mRMR** (Peng et al. 2005): Greedy selection balancing anchor relevance against feature redundancy.
5. **Structural Residual Search** (Jaffe et al. 2014): Subsets minimizing rank-one covariance fit residual.
6. **Column Subset Selection**: Greedy linear reconstruction of the full feature pool.
7. **Anchor Correlation**: Ranking features by correlation with the anchor view (epr).
8. **Statistical Floors**: Baseline floors (random draw, kurtosis, median absolute deviation).

**DUFS Optimization & Variant Testing**:
GroupFS and DUFS performed best among pure unsupervised selectors (~75.0% macro AUROC). However, when running standard DUFS as-is, two key bottlenecks emerged:
* **Objective Mismatch**: Graph Laplacian smoothness did not track downstream label separability (Spearman correlation between learned gates and feature AUROC was only +0.15).
* **Feature Over-selection**: Standard DUFS selected 15 to 20 features per cell, introducing a long tail of weak/noisy features that diluted L-SML weights.

To address these limitations and optimize for higher AUROC, I developed and tested several algorithmic enhancements combining DUFS with L-SML:
* **Pseudo-Label Gating**: Instead of relying purely on graph smoothness, I used L-SML fusion over seed features to generate a continuous pseudo-label, training the stochastic gates to maximize agreement with this pseudo-label. This keeps the method 100% label-free while directing gates toward discriminative features.
* **Seed Strategy Variations**: I swept multiple pseudo-label seed configurations:
  * Using 4 stable core seeds (epr, low_band_power, spectral_entropy, cusum_max).
  * Using alternative seed sets (such as LOCO_5 consensus features or diverse-centrality features).
  * Fusing all 30 views in L-SML to generate a full-pool pseudo-label without holding out seeds.
* **Orientation Anchor**: Added a label-free anchor view to lock the orientation/sign of the fused score across cells.
* **Feature Budgeting & Tail Pruning**: Implemented explicit feature caps/pruning to trim the noisy long tail. I evaluated cross-validated fixed-budget sweeps as well as an adaptive per-cell selection criterion based on L-SML covariance residual elbows.

**Impact of DUFS Optimizations**:
Together, pseudo-label gating, orientation anchoring, and feature pruning raised our fully label-free selector to **75.7% macro AUROC** (71.9% QA). This yields a statistically significant improvement over our baseline GOOD_5 (75.2%, paired Wilcoxon p = 0.037) and brings us within 0.2pp of our hand-picked GOOD_6 benchmark (75.9%).

### 3. Consensus Subset & Scoreboard
An exhaustive leave-one-cell-out (LOCO) sweep over size 3 to 5 subsets of the 30-view pool converged in 22 of 25 folds on the same 5-view subset: {cusum_max, logprob_margin, min_energy, spectral_entropy, topk_tail_mass}.

Named **LOCO_5**, it reaches **77.1% macro AUROC** (beating GOOD_6 by +0.73pp out-of-sample across 24 cells), with 3 of its 5 views coming from our new cluster features.

**Scoreboard Summary Across 25 Core Cells:**
* **Per-Cell Oracle** (Label-peeking, 3-5 views): **80.0%** Macro AUROC | N/A QA (25 cells) - Theoretical per-cell ceiling
* **Supervised LR @ 30** (Supervised, all 30 views): **78.1%** Macro AUROC | **75.2%** QA AUROC (25 cells) - Supervised linear baseline
* **LOCO_5** (LOCO consensus sweep): **77.1%** Macro AUROC | N/A QA (24 cells) - Leader on expanded pool
* **GOOD_6** (Fixed 6-view subset): **75.9%** Macro AUROC | **72.7%** QA AUROC (25 cells) - Previous hand-picked detector
* **My Selector (Optimized DUFS)** (Label-free algorithm): **75.7%** Macro AUROC | **71.9%** QA AUROC (25 cells) - Best selector (beats GOOD_5, p = 0.037)
* **GOOD_5** (Fixed 5-view subset): **75.2%** Macro AUROC | **72.1%** QA AUROC (25 cells) - Reference baseline

*(Note: Raw Markdown Table provided below if preferred)*
| Method | Selection | Macro AUROC | QA AUROC | Cells | Notes |
|---|---|:---:|:---:|:---:|---|
| Per-Cell Oracle | Label-peeking (3-5 views) | 80.0% | N/A | 25 | Theoretical per-cell ceiling |
| Supervised LR @ 30 | Supervised (all 30 views) | 78.1% | 75.2% | 25 | Supervised linear baseline |
| LOCO_5 | LOCO consensus sweep | 77.1% | N/A | 24 | Sweep leader on expanded pool |
| GOOD_6 | Fixed 6-view subset | 75.9% | 72.7% | 25 | Previous hand-picked detector |
| My Selector (Optimized DUFS) | Label-free algorithm | 75.7% | 71.9% | 25 | Best selector (beats GOOD_5, p = 0.037) |
| GOOD_5 | Fixed 5-view subset | 75.2% | 72.1% | 25 | Reference baseline |

### 4. What I'm Investigating Next
The scoreboard highlights two key reference points: the supervised linear baseline (78.1%) and the per-cell oracle ceiling (80.0%).

The gap between our label-free selector (75.7%) and the supervised linear oracle (78.1%) is 2.4pp overall (2.5pp on QA). Because the supervised oracle is a standard linear model with one fixed sign per feature - the exact same model family that our L-SML fusion operates in - the bottleneck is not model capacity, but rather accurately recovering feature weights and signs without true labels.

My primary focus now is analyzing the combined DUFS + L-SML algorithm, testing different fusion and weighting methods to optimize performance specifically across both Reasoning (Math) and QA subsets.

### Attached Interactive Reports
I've attached two interactive HTML reports from this week's analysis for you to explore:
* cell_method_matrix.html: Interactive 25-cell x 18-method AUROC performance heatmap.
* cell_oracle_vs_chosen.html: Per-cell breakdown comparing the oracle ceiling against our selector's actual picks and feature overlap.

When would be a good time to connect this week to discuss these findings and sanity-check the next steps?

Thanks,  
Omri
