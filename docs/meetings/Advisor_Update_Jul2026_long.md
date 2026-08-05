Subject: Feature selection results + the conformal chapter

Hi Ofir, Bracha and Amir,

Following up on our July 16 meeting. The action items I took away were:

1. Add a contribution to the current algorithm, best candidate being the feature selection step.
2. Get inspiration from recent feature engineering work on tabular data.
3. Go through the feature selection algorithms from your earlier research, Ofir.
4. Deep dive into the assumptions behind the L-SML and U-PCR line, and check they hold in practice, in the spirit of what FUSE did with verifier dependence.
5. Bracha's suggestion to look at conformal calibration.

The first four are done. Here is what came out, including the parts that did not work.

### First step: re-running every cell on the cluster

Before any selection work there was a data problem. The early Colab runs only saved the token entropy series H(n), so any feature needing the full logit distribution or the top-K probabilities could not be computed at all. I re-ran inference across every cell on the AIRCC cluster, this time also saving the top-50 logprobs and the energy series Z_n.

That took the pool from 16 views to 30, adding energy, logprob margin, varentropy and tail mass, and 7 of the 10 most informative views are now new ones. The honest version: the gap between the old 16-view pool and the new 30 (0.683 to 0.750 for the same selector) is not a size effect. The old 16 was a badly chosen 16 that excluded the strongest views, and a well-chosen 16 scores 0.752. Composition matters, size does not.

The bigger pool did not rescue GPQA, where features sit at chance, or RAG, which only has signal on HotpotQA. Both stay out of scope, so everything below is the 25 QA and math cells.

The method in one paragraph: one forward pass gives H(n), the top-50 logprobs and the energy series. EPR, the mean token entropy, is exactly the DC component of the FFT of H(n), so everything above DC is orthogonal to it and free once the pass is done. We remove DC, extract spectral, STFT, sliding-window variance, time-domain, energy and logprob features to get the 30 views, orient and z-score each, then fuse with continuous L-SML, which clusters the dependent views, fuses within, then across. U-PCR (Dror et al. 2017, arXiv:1703.02965 [LINK]) is the continuous-input member of the same family and is the second fusion path we now run.

### Items 2 and 3: the selection benchmark

The survey produced 8 algorithm families, all now benchmarked on the same 25 cells: DUFS and GroupFS from your line, Ofir [LINK], classical spectral FS (Laplacian Score, SPEC, MCFS), concrete autoencoders, mRMR, structural residual search, column subset selection, and statistical floors.

I have attached `comparison.html`, with all 196 variants scored on those cells, sortable and filterable. It carries a provenance column, since I replayed every published number through current code: of 169 rows, 0 are unexplained. The interesting axis is not the score, it is how much hand-tuning each method needs.

| Method | Prior it needs | Macro | QA | Math |
|---|---|---|---|---|
| LOCO_5 | label-chosen 5-view subset | 77.1% (24 cells) | 74.4% | 78.7% |
| GOOD_6 | hand-picked 6-view subset | 75.9% | 72.7% | 78.1% |
| **U-PCR + sign(rho)** | **anchor bit only** | **75.5%** | 71.3% | 78.3% |
| GOOD_5 | hand-picked 5-view subset | 75.2% | 72.1% | 77.3% |
| **DUFS parameter-free + L-SML** | **anchor bit only** | **75.1%** | 70.9% | 77.9% |
| **DUFS + L-SML** | **anchor bit only** | **75.0%** | 70.9% | 77.8% |
| L-SML, all 30 views, no selection | anchor bit only | 74.6% | | |
| Supervised LR on 30 views | labels | 78.1% | 75.2% | |

The bolded families need nothing hand-picked. The DUFS arms came out of this benchmark, and the parameter-free variant uses DUFS's own Eq. 7 loss, so there is no lambda to tune at all. U-PCR with sign(rho) came out of item 4 below.

The honest reading: these sit within 0.4 to 0.9pp of the hand-picked subsets and none of the gaps is significant, GOOD_6 over U-PCR is +0.43pp at p=0.615. The claim is not that the label-free selector wins, it is that it reaches the hand-curated bar without ever seeing an answer key.

The one prior we cannot remove is the global anchor bit, the single sign saying which direction means correct. Dropping it costs 0.51 AUROC and inverts all 25 cells, and that is provable rather than empirical: flipping every feature leaves the covariance bit-identical, so no covariance-based rule can recover it.

### Item 4: auditing the assumptions

This was the most useful item, and not for the reason I expected. Going through our U-PCR against the paper turned up seven real deviations: squared instead of absolute loss, a hardcoded variance that capped the search to the bottom quarter of its range, no difficulty gate, no simple-average fallback when few experts survive, no recompute after exclusion, the residual projected onto 2 components instead of 1, and exclusion that could not be switched off.

Surprisingly, fixing all seven did not help. Fully paper-faithful scores 69.1% against 73.9% for the old path, and none of the 64 flag combinations beats GOOD_6. The reason is structural: one-component U-PCR is exactly PC1 of the surviving views, so the whole rho and g2 apparatus only ever enters through the exclusion mask. The estimation machinery is inert on our data.

I also tested the dependence assumption directly, which is the FUSE-style question. Clustering the dependent views before fusing, rather than assuming independence, is refuted here: it loses 4.5pp, and the structure it keys on turns out to be ordinary pair correlation rather than a real assumption violation.

What did work came out of the same audit but was not one of the seven. Deriving each view's polarity from sign(rho) instead of my 42 hand-assigned signs gives +1.5pp, 20 wins 5 losses, p < 0.001. The covariance recovers the empirically correct direction on 91.8% of views, and 15 of my 30 hand signs turned out to be wrong. The value was in orientation, not estimation, and that is now the strongest arm we have.

Pool composition closed out too, in both directions. Removing views hurts U-PCR (-0.50pp, p=0.0096), and all six pre-registered variants that add the strongest unused views land below GOOD_6.

### Against published numbers

In our own cost class, unsupervised and one forward pass, we are +8.7pp on the 11 cells with a published comparison, p=0.042. Against unsupervised methods that use 10 samples we are level, and against the best method in each paper, usually a supervised probe, we are behind by roughly 8pp.

### Item 5: the conformal chapter

Bracha, I would like to start this now that selection is done.

Two papers are worth reading first. ITCR (arXiv:2606.08831, ICML 2026) [LINK] puts split conformal prediction inside reasoning-graph generation and calibrates when to stop. It is not a drop-in for us since it intervenes during generation and we are a post-generation detector, so Mohri and Hashimoto's conformal factuality work (ICML 2024) [LINK] is the closer fit.

The question I want to put to you comes from ITCR's own limitations section. Split conformal needs exchangeability between calibration and test, and they flag that it breaks when task domain, prompt or model backbone changes. Our 25 cells vary exactly those three axes. So does one calibration transfer across cells, or do we need localized per-cell calibration? That seems like the real question rather than just applying a threshold.

The prerequisite on my side is a frozen-weights scorer plus imbalance-aware metrics (recall, TPR at fixed FPR, AUPRC), since we currently fit and evaluate on the same batch. That is local CPU work on data we already have.

When would be a good time to meet?

Thanks,
Omri
