---
slug: crowdsourcing-regression-a-spectral-approach
title: "Crowdsourcing Regression: A Spectral Approach"
authors: "Yaniv Tenzer, Omer Dror, Boaz Nadler, Erhan Bilal, Yuval Kluger"
venue: "AISTATS 2022, PMLR 151:5225-5242"
year: 2022
source_pdf: https://proceedings.mlr.press/v151/tenzer22a/tenzer22a.pdf
last_digested: 2026-08-05
---

## Critical repository correction

The root file `Tenzer2022_Crowdsourcing_Regression_Spectral.pdf` is not this paper.  It contains
Dror et al. 2017 *Unsupervised Ensemble Regression*.  The older digest records the mismatch.  Use
the PMLR source above for any claim about Tenzer et al. 2022.

## Summary

U-PCR combines continuous expert predictions without labels by estimating the unobserved vector
`rho_i = Cov(f_i,Y)` from the observed expert covariance.  Writing
`f_i(x)=g(x)+h_i(x)` gives

```text
C = L + S,
L = g²11ᵀ + a1ᵀ + 1aᵀ,       rank(L) <= 2,
S_ij = E[h_i h_j].
```

Given the off-diagonal entries of `L` and a candidate `g²`, `rho` follows from an overdetermined
additive system.  A spectral projection residual selects `g²`, and the final U-PCR weights use the
first two principal components of the observed covariance.

## Two variants

- **IU-PCR:** all off-diagonal entries of `S` are zero (uncorrelated deviations).
- **SU-PCR:** `S` is sparse.  The paper decomposes the off-diagonal covariance into low-rank and
  sparse pieces using the projected-gradient robust matrix-completion method of Cherapanamjeri et
  al. before solving for `rho`.

Theorem 2 says the exact l0 formulation has a unique solution iff
`||vec(S)||_0 < (m-1)/2` for `m>=5`.  This is extremely strict: at `m=28`, fewer than 14 correlated
pairs.  Real data are acknowledged to be only approximately sparse, so the experiment must report
support diagnostics rather than assume the theorem applies.

## Experiments

SU-PCR achieves lower MSE than IU-PCR on 15 of 17 manually constructed regression tasks and is at
least as good as mean/median on 12 of 17.  The objective is regression MSE, not AUROC, and inputs are
commensurate real-valued regressors.

## Connection to this project

Sparse correlated errors are already published and must be implemented as a baseline.  The open
contribution is to tailor the idea to heterogeneous rank-like hallucination scores and test whether
the recovered dependency structure should enter the final regularized weights, rather than only
the reliability estimate.

## Sources

- Paper and supplement: https://proceedings.mlr.press/v151/tenzer22a/tenzer22a.pdf
- Proceedings page: https://proceedings.mlr.press/v151/tenzer22a.html
- Robust matrix completion: https://arxiv.org/abs/1606.07315
