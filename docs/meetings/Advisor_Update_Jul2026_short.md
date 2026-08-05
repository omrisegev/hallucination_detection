Subject: Feature selection results + the conformal chapter

Hi Ofir, Bracha and Amir,

Following up on our July 16 meeting. My action items were: add a contribution to the algorithm, best candidate being the feature selection step; get inspiration from recent tabular feature engineering; go through the selection algorithms from your earlier research, Ofir; deep dive into the assumptions behind the L-SML and U-PCR line and check they hold in practice, in the spirit of what FUSE did with verifier dependence; and Bracha's conformal calibration suggestion. The first four are done.

**First step was a re-run.** The early Colab runs only saved the token entropy series H(n), so any feature needing full logits or top-K probabilities could not be computed at all. I re-ran inference across every cell on the AIRCC cluster, also saving the top-50 logprobs and the energy series Z_n. That took the pool from 16 views to 30, and 7 of the 10 most informative views are now new ones. The honest version: the gap from the old 16 to the new 30 (0.683 to 0.750) is not a size effect, the old 16 was a badly chosen 16, and a well-chosen 16 scores 0.752. Composition matters, size does not. The bigger pool did not rescue GPQA or RAG, so both stay out of scope and everything below is the 25 QA and math cells.

**Selection.** The survey produced 8 families, all benchmarked on the same cells: DUFS and GroupFS from your line, Ofir [LINK], classical spectral FS, concrete autoencoders, mRMR, structural residual search, column subset selection, and floors. I have attached `comparison.html` with all 196 variants scored on those cells, plus a provenance column: I replayed every published number through current code and of 169 rows, 0 are unexplained. The interesting axis is not the score, it is how much hand-tuning each method needs.

| Method | Prior it needs | Macro | QA | Math |
|---|---|---|---|---|
| LOCO_5 | label-chosen 5-view subset | 77.1% (24 cells) | 74.4% | 78.7% |
| GOOD_6 | hand-picked 6-view subset | 75.9% | 72.7% | 78.1% |
| **U-PCR + sign(rho)** | **anchor bit only** | **75.5%** | 71.3% | 78.3% |
| GOOD_5 | hand-picked 5-view subset | 75.2% | 72.1% | 77.3% |
| **DUFS + L-SML** | **anchor bit only** | **75.0%** | 70.9% | 77.8% |
| Supervised LR on 30 views | labels | 78.1% | 75.2% | |

The two bolded ones need nothing hand-picked. They sit within 0.4 to 0.9pp of the hand-curated subsets and none of the gaps is significant, GOOD_6 over U-PCR is +0.43pp at p=0.615. So the claim is not that the label-free selector wins, it is that it reaches the hand-curated bar without ever seeing an answer key. The one prior we cannot remove is the global anchor bit, the single sign saying which direction means correct. Dropping it costs 0.51 AUROC and inverts all 25 cells, and that is provable: flipping every feature leaves the covariance bit-identical, so no covariance-based rule can recover it.

**Assumptions.** I found seven real deviations between our U-PCR and the paper (squared instead of absolute loss, a capped variance search range, three missing steps from Algorithm 1, and the wrong projection rank). Surprisingly, fixing them did not help: fully paper-faithful scores 69.1% against 73.9% for the old path, and none of 64 combinations beats GOOD_6. One-component U-PCR turns out to be exactly PC1 of the surviving views, so the whole rho and g2 apparatus only enters through the exclusion mask. Testing the dependence assumption directly, clustering dependent views before fusing, also fails, losing 4.5pp. What did work was orientation: deriving each view's polarity from sign(rho) instead of my 42 hand-assigned signs is worth +1.5pp, 20 wins 5 losses, p < 0.001, and it showed 15 of my 30 hand signs were wrong. The value was in orientation, not estimation.

Against published numbers in our own cost class, unsupervised and one forward pass, we are +8.7pp on the 11 cells that have a comparison, p=0.042. Against methods using 10 samples we are level, and against supervised probes we are behind by about 8pp.

**Conformal.** Bracha, I would like to start this now. One paper is worth reading first, ITCR (arXiv:2606.08831, ICML 2026) [LINK], which puts split conformal prediction inside reasoning-graph generation. It is not a drop-in for us since it intervenes during generation and we are a post-generation detector, but its limitations section raises the question I want to ask you: split conformal needs exchangeability between calibration and test, and it breaks when task domain, prompt or model backbone changes. Our 25 cells vary exactly those axes. So does one calibration transfer across cells, or do we need localized per-cell calibration?

When would be a good time to meet?

Thanks,
Omri
