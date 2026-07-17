---
slug: concrete-autoencoders-differentiable-feature-selection
title: "Concrete Autoencoders: Differentiable Feature Selection and Reconstruction"
authors: "Muhammed Fatih Balin, Abubakar Abid, James Zou"
arxiv_id: "arXiv:1901.09346"
venue: "ICML 2019 (PMLR v97)"
year: 2019
license: "arXiv non-exclusive"
source_pdf: "n/a (fetched arXiv text, 2026-07-17)"
extracted_text: papers/extracted/concrete-autoencoders-differentiable-feature-selection.md
last_digested: 2026-07-17
---

## Summary

The Concrete Autoencoder (CAE) is an end-to-end differentiable, fully **unsupervised**
k-of-p feature selector: an encoder made of a single "concrete selector layer" (k
stochastic nodes, each a Concrete/Gumbel-Softmax distribution over the p input
features) feeds a standard decoder that must reconstruct **all** p inputs. Training
minimizes plain reconstruction MSE; as the Concrete temperature anneals to ~0 each
selector node collapses onto one input feature, so at eval the layer *is* a hard
subset of k features. Selection quality is therefore exactly "which k features best
linearly/nonlinearly reconstruct the rest" — a global, label-free criterion.

## Method (grounded in the extraction)

- **Concrete selector node** (extraction ~l.263-330): each of the k nodes holds
  logits α_i over the p features; a sample is
  `m = softmax((log α + Gumbel noise)/T)` and the node outputs `x·m`. As `T → 0`
  the sample approaches a one-hot at `argmax α`; at eval the layer uses
  `arg max α_i` directly. α initialized to small positive values.
- **Annealing schedule** (§3.2, extraction l.335-357, verbatim formula):
  `T(b) = T0 (TB/T0)^(b/B)` — first-order exponential decay from initial `T0`
  to final `TB` over the B training epochs. Fixed-high T never concentrates;
  fixed-low T freezes the initial choice; annealing "stochastically explores
  combinations of features in the initial phases" then converges.
- **Defaults** (extraction l.363-365): Adam, learning rate 1e-3; `T0 = 10`,
  `TB = 0.01`.
- **Decoder**: standard reconstruction network; the paper notes MSE
  reconstruction is the chosen criterion (~l.227). A linear decoder gives the
  "linear reconstruction" variant (what we use at p ≤ 46).

## Role in this project (Step 186 — A3 selector)

The tabular-research pick for the **pre-fusion feature-selection stage**: per
cell, CAE picks k of p trace features label-free; the same continuous L-SML
then fuses the subset. k swept {3..8} plus a label-free adaptive-k rule
(validation-MSE elbow). Known risk carried explicitly: reconstruction-good ≠
label-relevant (project Step-151 lesson) — with unit-variance (z-scored)
columns, a pure-noise feature can only be reconstructed by selecting itself, so
reconstruction value comes solely from predicting *other* features; whether
that aligns with hallucination-detection AUROC is exactly what the bench
measures.

## Implementation deviations (documented in spectral_utils/selectors/a3_concrete_ae.py)

1. torch CPU reimplementation (canonical repo github.com/mfbalin/Concrete-Autoencoders
   is Keras/TF); linear decoder.
2. Selector-logit learning rate raised above the paper's 1e-3 (mode-collapse fix at
   our tiny p: with all-equal small logits and few epochs, multiple nodes collapse
   onto the same feature; a faster logit lr with the paper's decoder lr separates
   them — found empirically on the planted smoke world, 2026-07-17).
3. Duplicate-argmax repair + cross-seed majority rule (3 seeds) since k ≪ p and
   duplicates would silently shrink the subset below k.

## Notes / open questions

- The paper targets large-p regimes (images, genomics); our p ≤ 16/46 is far
  smaller — the method is well-defined but the interesting question is whether
  reconstruction value tracks detection value at all (Step-151 says often not).
