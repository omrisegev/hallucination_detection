---
slug: loca-local-conformal-autoencoder
title: "LOCA: LOcal Conformal Autoencoder for standardized data coordinates"
authors: "Erez Peterfreund; Oﬁr Lindenbaum; Felix Dietrich; Tom Bertalan; Matan Gavish; Ioannis G. Kevrekidis; Ronald R. Coifman"
arxiv_id: "2004.07234v2"
venue: "not found in extract"
year: 2021
source_pdf: papers/LOCA - LOcal Conformal Autoencoder.pdf
extracted_text: papers/extracted/loca-local-conformal-autoencoder.md
last_digested: 2026-09-07
---

## Summary

LOCA learns coordinates that undo a smooth, injective measurement deformation
using repeated local measurements, or bursts, around each latent point. An
encoder makes each burst covariance approximately isotropic; a decoder's
reconstruction loss discourages loss of geometric information. This is a
geometry-recovery method, not a feature-count eigen-ratio rule or a
hallucination detector. Metadata: extract lines 12–23 and 55; mechanism:
lines 122–151, 256–309 and Algorithm 1, lines 315–357.

## Datasets & models used

- Synthetic nonlinear deformation of two-dimensional uniform latent points:
  2,000 anchors, 200 Gaussian samples per burst, sigma .01; lines 486–508.
- Out-of-sample frame/interior/exterior experiment: 2,000 training anchors
  and 20,000 additional test samples; lines 528–547.
- Stereographic projection of a sphere patch: 491 training anchors and 55
  withheld points; 400 samples per burst; lines 603–637 and 658–667.
- Photographed deformed printed grid: 2,500 points; text says 50 burst
  samples but Appendix C says 60; lines 681–698 and 1074–1080.
- **Simulated** Wi-Fi localization on a simplified MIT Stata Center floor
  plan: 17 transmitters, 4,000 anchors and six receivers in each local circle;
  lines 701–720. This is not a measured radio benchmark or LLM experiment.

Networks use fully connected encoder/decoder layers with tanh or leaky-ReLU
activations; Appendix C gives per-experiment architectures. Training uses
90% of clouds and 10% for validation, alternating losses with Adam and
validation early stopping; lines 1038–1102. No correctness labels are used
to train an LLM localizer because no such localizer is studied.

## Methods it compared itself against

Diffusion Maps (DM) and Anisotropic Diffusion Maps (A-DM), lines 454–463.
A-DM uses burst covariance in a local Mahalanobis distance; ordinary DM
does not use bursts. Comparisons optimize scaling for the distance-stress
evaluation, lines 503–523. Do not list generic AE/PRAE, IU-PCR or KalmanNet
as experimentally compared baselines in this paper.

## Experiments — methodology & scores

Values below are literal numerical results from the extracted results text.
Stress measures latent-vs-embedded pairwise distance distortion; lower is
better. It is unrelated to AUROC or ProcessBench F1.

| Setup | Metric | LOCA | DM | A-DM | Extract |
|---|---|---:|---:|---:|---|
| Synthetic nonlinear deformation | Stress | 1.5 × 10^-5 | 0.03 | 0.002 | lines 509–523 |
| Frame/interpolation/extrapolation | Stress | approximately 10^-4 in each region | not reported there | not reported there | lines 545–547 |
| Stereographic training region | Stress | 10^-3 | 0.18 | 6 × 10^-3 | lines 658–662 |
| Stereographic withheld region | Stress | 10^-4 | not reported there | not reported there | lines 663–665 |
| Decoder interpolation | MSE / standard deviation | 2.3 × 10^-4 / 2.4 × 10^-4 | not reported there | not reported there | lines 590–594 |

No literal localization-error number was found for Wi-Fi; its evidence is a
calibrated embedding plot. The displayed rigid/scale alignment uses ground
truth geometry: four corners for the printed grid, all training locations
for the Wi-Fi illustration (lines 692–698 and 734–739). That alignment is
separate from training the unsupervised geometry representation.

## Connection to our pipeline

**Proposed supporting role only:** improve the geometry or measurement
reliability feeding IU-PCR / Joint L-SML. Fusion remains the central method.
The attractive idea is to estimate local deformation/noise from repeated
measurements before deciding feature groups, observation sampling or graph
penalties. A neural standalone anomaly detector is outside this direction.

The paper assumes small isotropic perturbations in the latent space around
each anchor, with smooth local deformation (lines 122–151). Consecutive tokens
are changing, correlated contexts, so a time window does not automatically
supply those bursts. Bootstrap copies of a window also do not create new
independent measurements or establish latent isotropy.

A possible **our-own adaptation**, not LOCA reproduction, is to use
within-window block resampling to estimate uncertainty of our feature
measurements. Test whether that uncertainty helps Joint's diagonal
regularization, grouping graph or sampling stability. Preserve fusion
with/without this addition, diagonal-only and permuted controls, and compare
IU/equal on identical IDs. Simply multiplying a feature by a reliability
weight before z-scoring can cancel the weighting; the intervention must
actually change the fusion calculation. This proposal has not been run.

## Notes / open questions

- The recovered coordinates are ambiguous up to rotation/shift (and scale
  when burst scale is unknown); latent geometry alone does not identify the
  correctness direction. See lines 156–165 and 444–452.
- The sphere experiment distinguishes intrinsic dimension two from required
  Euclidean embedding dimension three. Neither is the number K of Joint
  feature groups. Figure 11 discusses dimensions 1–5, not a direct Joint-K
  selector; lines 1131–1135.
- The paper allows other measurement strategies that estimate the local
  Jacobian (lines 787–792). This keeps adaptations open but does not prove
  a token-neighborhood construction meets the assumptions.
- Text/appendix inconsistencies: printed-grid M=50 versus M=60; Wi-Fi main
  text says two output coordinates whereas the listed encoder ends in three
  (lines 734–739 and 1081–1087). Do not silently resolve these as facts.
- All 25 pages have extracted text. Mathematical glyph extraction is imperfect;
  check the PDF image before reproducing an equation verbatim. The card's
  algorithm description and numerical results are grounded in readable text.
