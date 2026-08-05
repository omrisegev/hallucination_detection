---
slug: deep-unsupervised-feature-selection-by-discarding-nuisance-a
title: "Deep Unsupervised Feature Selection by Discarding Nuisance and Correlated Features"
authors: "Uri Shaham*, Ofir Lindenbaum*, Jonathan Svirsky, Yuval Kluger. *equal contribution"
arxiv_id: "not found in extract (dated October 2021; code at github.com/jsvir/lscae)"
venue: "not found in extract — preprint header only"
year: 2021
source_pdf: papers/Deep Unsupervised Feature Selection by Discarding Nuisance and Correlated Features.pdf
extracted_text: papers/extracted/deep-unsupervised-feature-selection-by-discarding-nuisance-a.md
last_digested: 2026-08-05
---

## Summary

**LS-CAE** (Laplacian-Score-regularized Concrete Autoencoder) is a **fully unsupervised**
feature selector that discards *nuisance* and *correlated* features simultaneously. Its central
claim is diagnostic before it is algorithmic: **the Laplacian score must be computed on the
SELECTED SUBSET, not on the complete feature set** — with many nuisance features the graph
Laplacian is corrupted and the score becomes meaningless. The concrete layer supplies the
handle: compute the Laplacian at the concrete layer's *output*, so as training sharpens the
selection the Laplacian is progressively cleaned, which sharpens the selection further.

## The objective (Eq. 6, verbatim)

```
L(X) = ||X - X_hat||^2 / SG(||X - X_hat||^2)  -  Trace[C^T L_diff(C) C] / SG(Trace[C^T L_diff(C) C])
```

- `X` is an `m × d` minibatch, `X_hat` the autoencoder output, `C = C(X)` the **concrete-layer
  output** (`m × k`), `L_diff(C) = D⁻¹W` the **diffusion** Laplacian computed on `C`.
- `SG` = stop-gradient (identity forward, zero derivative).
- The **minus** is the paper's: reconstruction minimised, Laplacian trace **maximised**. Correct
  for the diffusion Laplacian, whose *largest* eigenvalues carry the main structure (§2.1).
- The SG denominators are the paper's **balancing mechanism** (§4.2): each term inversely
  weighted by its own magnitude, which *"removes the need to use a tunable hyperparameter"*.
  So there is **no λ** — the loss value is identically 0 and only the gradients matter.
- `k` is **architectural** (the concrete layer's size). The paper gives no rule for choosing it.

Preliminaries (§2): features are assumed **centred with ‖fᵢ‖²₂ = 1**. Kernel bandwidth σ is
*"the maximal Euclidean distance from any point to its nearest neighbor"* (§2.1 fn. 2, offered
as common practice, not pinned).

Technical details (§5.3): decoder = two hidden layers of 128 LeakyReLU units; **300 epochs**;
lr **1.0** for the concrete layer, **0.01** for the decoder.

## Analytical result (§3)

With two clusters a distance `r` apart, the number of nuisance dimensions needed to break the
cluster structure carried by the leading nontrivial eigenvector ψ₂ scales as
`d = O( r⁴ / −log(1 − (2n² − 1)√(1−ε)) )` — i.e. the closer the clusters, the fewer nuisance
features suffice to destroy the Laplacian. Verified empirically in Fig. 2.

## Datasets & models used

- **Synthetic ablation**: augmented two-moons, `n = 1200`, `2d + 4` features — 2 true moons
  coordinates, a noisy copy of them, and two copies of `d` nuisance features drawn from a
  Gaussian with `C_ij = (−0.25)^|i−j|`. `d = 3, 6, 12, 15`, 10 repetitions, 2 concrete units.
- **Noisy MNIST**: a single fixed noise pattern applied through a Bernoulli(0.2) mask, so the
  data contains *correlated* high-frequency features. Select 5/10/15/20/25 features, then
  k-means (k=10) on 60,000 train, accuracy on 10,000 test, 3 repetitions.
- **Nine/ten real-world FS benchmarks**: RCV1, GISETTE, PIX10, COIL20, Yale, TOX-171, ALLAML,
  PROSTATE, FAN, POLLEN. Sample sizes 56 → 21,332, features often exceeding sample size.

## Methods it compared itself against

LS (Laplacian Score alone), **MCFS** (Cai, Zhang, He 2010), **NDFS**, **LLCFS**, **SRCFS**,
**CAE** (Balın, Abid, Zou 2019). The two ablations that matter are **LS** (Laplacian term only)
and **CAE** (reconstruction term only) — the paper's contribution is that you need *both*.

## Experiments — methodology & scores

Each method tuned to select 50/100/150/200/250/300 features; k-means applied 20×; the best
average clustering accuracy reported with the feature count in parentheses.

| Dataset | LS | MCFS | NDFS | LLCFS | SRCFS | CAE | **LS-CAE** |
|---|---|---|---|---|---|---|---|
| RCV1 | 54.9 (300) | 50.1 (150) | 55.1 (150) | 55.0 (300) | 53.7 (300) | 54.9 (300) | **83.7 (300)** |
| GISETTE | 75.8 (50) | 56.5 (50) | 69.3 (250) | 72.5 (50) | 68.5 (50) | 77.3 (250) | **80.7 (50)** |
| PIX10 | 76.6 (150) | 75.9 (200) | 76.7 (200) | 69.1 (300) | 75.9 (100) | 94.1 (250) | **94.5 (250)** |
| COIL20 | 60.0 (300) | 59.7 (250) | 60.1 (300) | 48.1 (300) | 59.9 (300) | **65.6 (200)** | 61.8 (300) |
| Yale | 42.7 (300) | 41.7 (300) | 42.5 (300) | 42.6 (300) | 46.3 (250) | 45.4 (250) | **48.0 (200)** |
| TOX-171 | 47.5 (200) | 42.5 (100) | 46.1 (100) | 46.7 (250) | 45.8 (150) | 47.7 (100) | **48.3 (100)** |
| ALLAML | 73.2 (150) | 72.9 (250) | 72.2 (100) | **77.8 (50)** | 67.7 (250) | 73.5 (250) | 76.5 (150) |
| PROSTATE | 58.6 (300) | 57.3 (300) | 58.3 (100) | 57.8 (50) | 60.6 (50) | 56.9 (250) | **71.4 (50)** |
| FAN | 42.9 (150) | 45.5 (150) | 48.8 (100) | 29.0 (50) | 29.0 (100) | 35.2 (300) | **51.7 (100)** |
| POLLEN | 46.9 (150) | **66.5 (300)** | 48.9 (50) | 35.0 (100) | 34.9 (300) | 58.0 (250) | 65.8 (100) |
| **Mean rank** | 4.0 | 6.0 | 5.0 | 5.0 | 6.0 | 2.0 | **1.0** |

Best on 7 of 10, second on the remaining 3. RCV1 is a ~50% relative jump over the next best.

## Connection to our pipeline

**Directly relevant, and the reason it was implemented as `spectral_utils/selectors/a8_lscae.py`
(Step 224).** Every label-free condition priced in the U-PCR feature-selection channel so far
attacks one failure mode only: `a2.dufs` / `lapscore_adapt` chase the Laplacian score (nuisance),
`a3.cae` chases reconstruction (correlation). We held **both halves separately and had never run
the combination** — which is exactly the comparison this paper's contribution (iii) is about.
`a2_groupfs.py` already cites the LSCAE repo as a building block.

Also note it is the **direct successor** to DUFS: *"This work builds on an earlier work of ours
Lindenbaum et al. [12], which contains only the Laplacian score objective and utilizes stochastic
gates for the selection mechanism."* And §4.2 records why the concrete layer replaced the gates —
with stochastic gates *"the Laplacian score term alone encourages the selection of all features,
[while] the regularization term encourages the opposite goal,"* causing instability. That is a
published account of the same gate-saturation failure our GroupFS reimplementation hit
(`a2_groupfs.py` deviation 8).

Our reimplementation reproduces the paper's planted claim in `a8_lscae.smoke()`: on a world of
2 informative directions × 2 correlated copies + 8 nuisance columns, `lscae.k4` selects one
representative from each correlated block and **zero** nuisance columns.

## Notes / open questions

- Venue and arXiv id are genuinely absent from the extracted text — preprint header only,
  "October 2021".
- The real-world table is **clustering accuracy**, not AUROC, and every benchmark is `N ≪ D`
  (56–21,332 samples, features often exceeding sample count). Our regime is the reverse
  (`n ≈ 100–600`, `p ≈ 28`), so the nuisance-swamping motivation is much weaker for us: we do
  not have hundreds of nuisance columns corrupting the Laplacian. Worth stating beside any
  result we get.
- The paper never selects fewer than 50 features. We run it at k = 3–8. Far outside its tested
  regime.
