---
slug: l0-based-sparse-canonical-correlation-analysis
title: "ℓ0-based Sparse Canonical Correlation Analysis"
authors: "Ofir Lindenbaum (Yale), Moshe Salhov (Tel Aviv University), Amir Averbuch (Tel Aviv University), Yuval Kluger (Yale, corresponding)"
arxiv_id: "arXiv:2010.05620v2 [cs.LG]"
venue: "arXiv-only (v2 dated 8 Jun 2021; paper header dated June 9, 2021)"
year: 2021
source_pdf: papers/l0-based Sparse Canonical Correlation Analysis.pdf
extracted_text: papers/extracted/l0-based-sparse-canonical-correlation-analysis.md
last_digested: 2026-08-04
---

## Summary

ℓ0-CCA learns maximally correlated representations of two modalities from a **sparse subset of
input variables**, by multiplying the inputs by stochastic gates whose parameters are trained
jointly with the CCA weights under an ℓ0-regularized correlation loss (Eq. 3, relaxed via Eq. 4).
The stated motivation is two-fold: CCA breaks when the number of variables exceeds the number of
samples, and **"often a significant fraction of the variables measures modality-specific
information, and thus removing them is beneficial"** (abstract). ℓ0-DCCA extends the same gating
to two neural sub-nets trained in tandem on a shared total-correlation loss (Eq. 5). The paper
also proposes a closed-form **gate initialization** from the thresholded cross-covariance matrix
(§3.2), which is a standalone label-free cross-modality score.

## Datasets & models used

| Data | Shape | Role |
|---|---|---|
| Synthetic Gaussian, three covariance models (Identity / Toeplitz / sparse-inverse) | (N, Dx, Dy) = (400,800,800), (500,600,600), (700,1200,1200); φ, η sparse with 5 nonzeros, ρ₀ = 0.9 | canonical-vector recovery in N ≪ D |
| Spinning puppets, two cameras (Lederman & Talmon) | 400 images/camera, 240×320 = 76,800 px | N < D, shared latent = bulldog rotation |
| Noisy MNIST (two noise variants) | 62,000 samples; 40k train / 12k test / 10k val | ℓ0-DCCA, d = 10 |
| Seismic events, E and N channels | 537 explosions, 3 quarries, sonograms z ∈ R^1157 (89 time × 13 freq bins) | ℓ0-DCCA, d = 3 |
| METABRIC breast cancer (RNA expression + CNA) | 1,112 patients, 10 subtypes; Dx = 15,709, Dy = 47,127 | ℓ0-DCCA, d = 10 |

No language models — the paper is a general multi-view feature-selection method.

## Methods it compared itself against

Sparse-CCA line: PMA (Witten et al.), IP-SCCA (Mai & Zhang), SCCA-I (Hardoon &
Shawe-Taylor), SCCA-II (Gao et al.), mod-SCCA (Suo et al.). Non-linear / fusion line: CCA, PCA,
KCCA, grad-KCCA, NCCA, SCCA-HSIC, multiview-ICA, DCCA, DCCAE, raw data.

The differentiator claimed: prior ℓ0 work ([13, 14]) is **greedy** and can be suboptimal, and ℓ1
relaxations suffer **coefficient shrinkage**; the stochastic-gate relaxation is differentiable and
trains gates *jointly* with the model.

## Experiments — methodology & scores

Synthetic: 100 realizations per covariance model; error `e_φ = 2(1 − |φᵀφ̂|)`, reported as the
pair (e_φ, e_η). Real data: embeddings scored by k-means (20 random inits, best SSE) → clustering
accuracy KM and mutual information MI, plus a linear SVM accuracy on held-out test. **All
hyperparameters, λ included, are tuned on a validation set by maximizing total correlation — no
labels.**

| Setup | Metric | ℓ0-CCA / ℓ0-DCCA | Best baseline | Notes |
|---|---|---|---|---|
| Synthetic Model I, (400,800,800) | (e_φ, e_η) ↓ | **(0.003, 0.009)** | mod-SCCA (0.056, 0.062) | ~20× lower error |
| Synthetic Model II Toeplitz, (400,800,800) | (e_φ, e_η) ↓ | **(0.101, 0.079)** | mod-SCCA (0.173, 0.218) | |
| Synthetic Model III, (400,800,800) | (e_φ, e_η) ↓ | (0.108, 0.103) | SCCA-II (0.129, 0.190) | IP-SCCA better on e_φ at (500,600,600) |
| Noisy MNIST | MI / KM% / SVM% | **2.05 / 95.4 / 95.5** | DCCA 1.97 / 93.2 / 93.2 | 277 and 258 pixels selected |
| Seismic | MI / KM% / SVM% | **0.97 / 98.1 / 97.2** | DCCAE 0.92 / 97.0 / 97.0 | 17 and 16 features from E / N |
| METABRIC | MI / KM% / SVM% | **0.88 / 50.3 / 74.1** | DCCA 0.79 / 45.2 / 72.1 | |
| Spinning puppets | total correlation | 1.99 (of d = 2) | — | 372 / 403 active pixels, λ = 50 |

Method details that matter for reimplementation (§3.4, Algorithm 1, Appendix A.1):
gate `z_i = max(0, min(1, µ_i + ε_i))`, ε ~ N(0, σ²); `E‖z‖₀ = Σ_i ½(1 − erf(−µ_i/(√2 σ)))`;
total correlation = `trace(Ĉ_y^{-1/2} Ĉ_yx Ĉ_x^{-1} Ĉ_xy Ĉ_y^{-1/2})` with ridge `γI` on the
within-view covariances; µ initialized at 0.5, or from the thresholded `C̄_xy` as
`µ_x = ū + 0.5` where `ū` is the thresholded |leading left singular vector| (§3.2); one Monte
Carlo sample per gradient step; **post-training keep rule is `z_i = max(0,min(1,µ_i)) > 0`**, and
Algorithm 1's stated return is "**s features with largest µ_i**". Synthetic linear run: lr 0.005,
10,000 epochs, λx = λy = 30, **σ = 0.25** (the paper explicitly departs from STG's σ = 0.5, saying
smaller σ converged better for ℓ0-CCA).

Fig. 2 (left) is the λ-robustness claim: a wide range of λ recovers the correct 10 active
coefficients and ρ̂ = 0.9; small λ selects many variables, attains higher correlation, and
**overfits**.

## Connection to our pipeline

This is the method behind **Step 223** of the U-PCR feature-selection item, and it is an advisors'
paper (Averbuch, Salhov). Two properties make it survive the Step 221/222 closures:

- Its criterion is **shared structure across two measurement channels**, not correlation with
  correctness. Everything closed so far bounds what `rho_hat` can buy; this is a different
  quantity, so that floor does not apply to it.
- Its objective is **set-level** — gates enter the total-correlation loss jointly with the
  canonical weights, so a feature's gate depends on the others. A marginal per-feature statistic
  cannot express that, and Step 222 closed the marginal family.

The gate itself is **already implemented in this repo**: `_train_dufs` in
[a2_groupfs.py:303](spectral_utils/selectors/a2_groupfs.py#L303) uses the same STG (Eq. 4) and
the same `P(Z ≥ 0)` erf penalty as the DUFS paper `Differentiable Unsupervised Feature Selection
based on a Gated Laplacian.pdf`, reference [33] here. Only the loss differs — gated Laplacian
there, total correlation here. Pre-registered channel split for our pool: X = the 16 entropy-trace
spectral views, Y = the 14 spilled-energy + token-logprob views.

Note the regime inversion: the paper's motivating case is **N ≪ D** (400 samples, 800 variables);
ours is N ≫ D (thousands of rows, ~30 features). The degeneracy argument does not transfer to us —
only the **nuisance-removal** argument does.

## Notes / open questions

- **Algorithm 1's readout is still a per-feature ranking** ("return the s features with largest
  µ_i"). Step 222 closed the *shape* "score each feature, keep top-k"; ℓ0-CCA's escape is in how
  the statistic is **computed**, not how it is read out. It must be scored on both our tests
  (performance vs the matched floor, overlap vs the composition-matched null) like everything else.
- **λ tuning is label-free by the paper's own procedure** — maximize total correlation on held-out
  rows. Fig. 2 claims robustness across λ; if *our* support is unstable across λ, that is a finding
  about our data and must be reported, not absorbed by picking a λ.
- σ = 0.25 here vs σ = 0.5 in STG/DUFS. Our `STG_SIGMA = 0.5` follows the DUFS side; the CCA loss
  may want the paper's smaller value.
- Stated limitations (Appendix E): λ needs cross-validation, and the method "**lacks guarantees
  when trained on small batches**". Several of our test sets are small.
- The §3.2 initialization is a complete, training-free, label-free cross-modality score in its own
  right (threshold `C_xy` at the r-th percentile → leading singular vectors → threshold |u|).
  Cheap to run as a probe before committing to trained gates — but a weak result from it is
  **evidence, not a bound**, since it is only an initializer in the paper.
- The generalized (>2 view) form, Appendix C Eq. 7, is explicitly left unanalyzed by the authors:
  "the initialization of G and U^k plus the analysis ... are left for further research."
