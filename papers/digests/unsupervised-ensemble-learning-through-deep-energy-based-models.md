---
slug: unsupervised-ensemble-learning-through-deep-energy-based-models
title: "Unsupervised Ensemble Learning Through Deep Energy-based Models"
authors: "Ariel Maymon, Yanir Buznah, Uri Shaham"
arxiv_id: "arXiv:2601.20556v1"
venue: "AISTATS 2026"
year: 2026
source_pdf: ../Unsupervised Ensemble Learning Through Deep Energy-based Models.pdf
last_digested: 2026-08-05
---

## Summary

DEEM is an unsupervised ensemble classifier built from learned multinomial preprocessing layers
and a final inverse Restricted Boltzmann Machine (iRBM).  The paper proves a bijection between the
single-hidden-unit multinomial iRBM and the Dawid–Skene conditional-independence model.  The deep
layers are intended to transform correlated learner predictions into a representation closer to
conditional independence, after which the iRBM estimates the latent class.

## What is proved versus empirical

- Identifiability/consistency is established for the iRBM under conditional independence and an
  attained maximum-likelihood solution.
- The full deep model has no corresponding identifiability or consistency theorem under dependent
  learners.  Its ability to handle complex dependence is empirical.
- Maximum likelihood for the energy model is intractable and non-concave in practice; training uses
  approximate discrete Langevin/Gibbs-with-gradients negative sampling.

## Inputs, objective, and alignment

The paper is centered on hard multinomial learner predictions and accuracy.  The released 0.2.0
package also accepts soft `(N,K,D)` inputs.  The latent class permutation is aligned to majority
vote using a Hungarian assignment, not to true labels.  This assumes the ensemble's average
direction is useful.  The API's `return_probs=True` probabilities are unaligned; callers must apply
the learned class map before interpreting a class-specific probability.

## Dependency evidence and limitations

On MnistE the paper reports decreasing class-conditional mutual-information summaries through the
network, but this analysis conditions on true labels and is post-hoc rather than a deployable
selection criterion.  The paper also reports sensitivity to learning rate, training divergence and
dead units, and lacks a reliable fine-grained convergence criterion.  The one-hot architecture and
sampler become expensive as the number of classes grows.

## Connection to this project

DEEM attacks the same broken-independence premise as sparse-error U-PCR, but by a different route:
it learns a nonlinear latent representation rather than explicitly estimating a sparse error
covariance.  It is therefore a strong nonlinear baseline for the dependency-fusion experiment, not
the mathematical equivalent of SDSF.  Our registered adapter evaluates median-binarized inputs and
rank pseudo-probabilities, uses five seeds, and never chooses a seed using AUROC.

## Sources

- Paper: https://arxiv.org/abs/2601.20556
- AISTATS page: https://openreview.net/forum?id=YF1ObZwFnk
- Code/API: https://github.com/shaham-lab/deem
