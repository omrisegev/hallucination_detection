Subject: Feature selection update + meeting next week?
Hi Ofir, Bracha and Amir,
Following up on our meeting last week (July 16), I wanted to give an update on the feature selection track we discussed. I've focused on building and benchmarking label-free feature selection algorithms to move beyond hand-curated subsets, as well as expanding our feature extraction pipeline.
Here is where things stand:
#### 1. Expanding the Feature Pool (Cluster Re-runs)
In our earlier Colab runs, we only saved the token entropy series H(n). That prevented us from extracting features requiring full logit distributions or top-K probabilities. Inspired by earlier exploratory work and literature on LLM uncertainty (e.g. Kadavath et al. 2022 [LINK] for varentropy, and energy-based detection literature), I re-ran inference across all cells on the AIRCC cluster to save the top-K logprobs and energy series (Z_n).
This expanded our pool from 16 to 30 views (adding Z_n energy, logprob margin, varentropy, and tail mass).
- **RAG & GPQA**: The expanded pool did not revive GPQA (features stay at chance, 0.51 to 0.55) or RAG (only improves on HotpotQA where it already had signal). Both stay out of scope for now, so all evaluations below focus on the 25 QA + Math cells.
- **QA & Math impact**: On our 25 core cells, 7 of the top 10 most informative views now come from these new energy and logprob features.
#### 2. Feature Selection Benchmark & Pseudo-Label Gates
Following the meeting action item, I built a unified benchmark on the 25 cells to test label-free selectors on the 30-view pool, feeding into our L-SML fusion model. I implemented and evaluated 8 algorithm families:
1. **GroupFS & DUFS** (Lifshitz et al. 2026 [LINK], Lindenbaum et al. 2021 [LINK]): Stochastic feature gates trained on sample-graph Laplacian smoothness (from Ofir's research line).
2. **Classical Spectral FS**: Laplacian Score (He et al. 2005 [LINK]), SPEC (Zhao & Liu 2007 [LINK]), and MCFS (Cai et al. 2010 [LINK]), ranking features by manifold smoothness or spectral embedding regression.
3. **Concrete Autoencoders** (Balin et al. 2019 [LINK]): Gumbel-softmax relaxation selecting features to minimize linear reconstruction error of the pool.
4. **mRMR** (Peng et al. 2005 [LINK]): Greedy selection balancing anchor relevance against feature redundancy.
5. **Structural Residual Search** (Jaffe et al. 2014 [LINK]): Subsets minimizing rank-one covariance fit residual.
6. **Column Subset Selection**: Greedy linear reconstruction of the full feature pool.
7. **Anchor Correlation**: Ranking features by correlation with the anchor view (epr).
8. **Statistical Floors**: Baseline floors (random draw, kurtosis, median absolute deviation).
GroupFS/DUFS came out best among the 8 families (~75.0% macro AUROC), essentially tying our hand-picked GOOD_5 baseline (75.2%).
**Why DUFS ties rather than wins**: DUFS optimizes for sample-graph smoothness, an unsupervised objective that never sees label separability (correlation between learned gates and feature AUROC was only +0.15). To address this, I used L-SML fusion to generate a continuous pseudo-label, adding an agreement term to guide gate training.
I evaluated several variations of this pseudo-label gating mechanism:
- Using 4 stable core seeds (`epr`, `low_band_power`, `spectral_entropy`, `cusum_max`) to generate the pseudo-label.
- Using alternative seed sets (such as LOCO_5 features or diverse-centrality features).
- Using all 30 views in L-SML fusion to generate a full-pool pseudo-label without holding out seeds.
All variations landed in the same 75.08% to 75.24% range. To be transparent, a +0.2pp gain over standard DUFS is modest and does not close the gap to GOOD_6 (75.9%). Its main value is being our first fully label-free algorithm to reach parity with hand-curated GOOD_5 without human intervention, but seed choice is not the bottleneck.
#### 3. Consensus Subset & Current Scoreboard
An exhaustive leave-one-cell-out (LOCO) consensus sweep over size 3 to 5 subsets of the 30-view pool converged in 22 of 25 folds on the same 5-view subset: `{cusum_max, logprob_margin, min_energy, spectral_entropy, topk_tail_mass}`.
Named **LOCO_5**, it reaches **77.1% macro AUROC** (beating GOOD_6 by +0.73pp out-of-sample), with 3 of its 5 views coming from the new cluster features.
|
 Method 
|
 Selection 
|
 Macro AUROC 
|
 Cells 
|
 Notes 
|
|
---
|
---
|
---
|
---
|
---
|
|
 Per-Cell Oracle 
|
 Label-peeking (best 3-5 per cell) 
|
 80.0% 
|
 25 
|
 Theoretical per-cell L-SML ceiling 
|
|
 LR@30 
|
 Supervised (all 30 views) 
|
 78.1% 
|
 25 
|
 Supervised linear baseline 
|
|
 LOCO_5 
|
 LOCO sweep 
|
 77.1% 
|
 24 
|
 Overall leader (uses new pool) 
|
|
 GOOD_6 
|
 Fixed subset 
|
 75.9% 
|
 25 
|
 Previous detector (+varentropy) 
|
|
 Pseudo-label gates 
|
 Label-free algo 
|
 75.2% 
|
 25 
|
 Our best selector (reaches GOOD_5) 
|
|
 GOOD_5 
|
 Fixed subset 
|
 75.2% 
|
 25 
|
 Baseline fixed subset 
|
#### 4. What I'm Investigating Next
The scoreboard above highlights two key reference points: the supervised LR baseline (78.1%) and the per-cell oracle ceiling (80.0%).
The per-cell oracle beats our label-free selector (75.2%) by +4.7pp, and shares only 17% feature overlap with our selector's picks. This shows there is meaningful headroom left on the table.
My primary focus now is analyzing the LR model and per-cell optimal subsets in comparison to our algorithm, to understand what features the optimal subsets rely on per cell and why our label-free algorithm fails to choose them.
(I also ran an exhaustive Leave-One-View-Out test on all 30 features, which confirmed that no view is safely droppable out-of-sample, so the full 30-view pool remains necessary.)
#### Attached Interactive Reports
I've attached two interactive HTML reports from this week's analysis for you to explore:
1. **`cell_method_matrix.html`**: Interactive 25-cell x 18-method AUROC performance heatmap.
2. **`cell_oracle_vs_chosen.html`**: Per-cell breakdown comparing the oracle ceiling against our selector's actual picks and feature overlap.
When would be a good time to connect next week to discuss?
Thanks,
Omri