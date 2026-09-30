---
slug: differentiable-unsupervised-feature-selection-based-on-a-gat
title: "Differentiable Unsupervised Feature Selection based on a Gated Laplacian"
authors: "Ofir Lindenbaum; Uri Shaham; Jonathan Svirsky; Erez Peterfreund; Yuval Kluger"
arxiv_id: "2007.04728v3"
venue: "not found in extract"
year: 2020
source_pdf: papers/Differentiable Unsupervised Feature Selection based on a Gated Laplacian.pdf
extracted_text: papers/extracted/differentiable-unsupervised-feature-selection-based-on-a-gat.md
last_digested: 2026-09-07
---

## Summary

DUFS learns stochastic feature gates and recomputes the graph on gated data,
so nuisance coordinates need not define the geometry used to select them.
The loss rewards smooth features under that learned random-walk operator
while penalizing gate survival. This attached version has an additive penalty
(equation 6) and a ratio objective without that penalty coefficient (equation
7); the experiments use a two-step random walk. Source: extract pages 1–3,
8–9, especially lines 390–458.

The attached PDF is the five-author arXiv v3 dated 9 November 2020, with
Yale, Technion and Hebrew University affiliations (page 1, lines 12–51).
Do not replace its author list, date or equation numbers with those of the
later six-author NeurIPS 2021 publication. The conference venue is not printed
in the attached extract; metadata above follows the attached version.

## Datasets & models used

Synthetic two moons: 100 observations, two informative coordinates with
Gaussian variance 0.1 and additional independent Gaussian nuisance features
(page 10). Noisy MNIST uses digits 3 and 8; appendix specifies 1,000 examples.
Noisy PIX10 adds uniform [0,0.3] pixel noise. Real-data table: GISETTE,
PIX10, COIL20, Yale, TOX-171, ALLAML, PROSTATE, RCV1 and ISOLET. Appendix
also discusses COIL100, 7,200 images of 100 objects, and mentions GLIOMA in
the parameter settings. No LLM or hallucination dataset was evaluated.
Source: pages 10–11, 16–17 (lines 460–654 and 794–895).

## Methods it compared itself against

Laplacian Score (LS), Multi-Cluster Feature Selection (MCFS), Nonnegative
Discriminative Feature Selection (NDFS), Local Learning based Clustering
(LLCFS), Multi-Subspace Randomization and Collaboration (SRCFS), Concrete
Auto-encoders (CAE), and clustering with all features. Source: page 11,
lines 503–654. The key difference from ordinary LS is optimizing the feature
gates jointly with the graph construction rather than trusting a fixed graph
contaminated by all nuisance features.

## Experiments — methodology & scores

Real-data evaluation uses k-means on leading 50, 100, 150, 200, 250 or 300
selected features, averaging 20 runs. The number of clusters equals the true
class count. The best average accuracy is recorded together with the number
of selected features. These are paper-reported best configurations, not a
frozen label-free hyperparameter-selection benchmark (page 9, lines 450–458).
The RCV1 subset uses the first two classes, removes multilabel examples and
balances classes by downsampling (page 17). Those label-dependent data and
evaluation choices must not silently enter our answer-only protocol.

Literal Table 1 clustering accuracies (%), page 11:

| Dataset | DUFS (selected features) | LS | CAE | All features |
|---|---:|---:|---:|---:|
| GISETTE | 99.5 (50) | 75.8 | 77.3 | 74.4 |
| PIX10 | 88.4 (50) | 76.6 | 94.1 | 74.3 |
| COIL20 | 65.8 (250) | 55.2 | 65.6 | 53.6 |
| Yale | 47.9 (200) | 42.7 | 45.4 | 38.3 |
| TOX-171 | 49.1 (50) | 47.5 | 44.4 | 41.5 |
| ALLAML | 74.5 (100) | 73.2 | 72.2 | 67.3 |
| PROSTATE | 64.7 (150) | 57.5 | 56.9 | 58.1 |
| RCV1 | 60.2 (300) | 54.9 | 54.9 | 50.0 |
| ISOLET | 63.6 (150) | 56.4 | 63.8 | 60.8 |

The paper reports first place on six datasets and second on three; mean rank
1.3 and median rank 1. ALLAML's best comparator is LLCFS at 77.8, not the
CAE column displayed above. Table dimensions and sample counts are preserved
in the extract; this card does not silently correct unusual counts there.

Appendix training uses SGD with thousands of epochs and dataset-dependent
learning rates, penalties and bandwidth settings. Two moons uses learning
rate 1 and 5,000 epochs; ALLAML uses 0.3 and 20,000 epochs; COIL20 uses 0.3
and 26,000 epochs. This is materially different from the repository's fixed
120-epoch Adam adaptation (source: page 17, lines 851–895).

## Connection to our pipeline

The repository's `adapted_dufs_soft_gates` already supports our Joint graph
regularizer; the new sampling pilot reuses it to help fit the same fusion.
An N-window by P-measurement matrix can be interpreted in two distinct ways:

- Gate P features and graph N windows: the existing feature-axis use.
- Transpose the learning interpretation, gate N windows and graph P feature
  coordinates: the new fitting-observation selector. The helper's interface
  expects gated-features by observations, so the latter passes N-by-P directly.

The second interpretation does not create a graph between windows. The
separate window-diffusion representative selector does that explicitly; it
is our geometric adaptation, not DUFS reproduction. Both provide fitting
rows to IU-PCR / Joint L-SML, retaining full-grid, uniform, raw-risk, equal
aggregation and permutation controls. Neither establishes correctness from
geometry alone. Details: `docs/experiments/FUSION_WINDOW_SAMPLING_PILOT_V1.md`.

## Notes / open questions

- Features in the paper are centered and unit L2 norm (page 2, lines 102–109).
  Transposition changes which vectors receive this normalization. Explicitly
  specify it; reusing the same array without its axis meaning is insufficient.
- The paper calls D^-1 K a random-walk Laplacian; it is a transition operator,
  whereas D-K or I-D^-1/2 K D^-1/2 are different matrices (page 3).
- Appendix bandwidth wording/equations mix an L1 phrase with Euclidean
  notation and define a global maximum local scale. Our k=7 product-of-local
  scales graph is an adaptation, not an exact implementation of that appendix.
- Parameter-free equation 7 removes one penalty coefficient; it does not
  eliminate choices of optimization, graph bandwidth, epochs or selected count.
- Smooth geometric structure need not preserve a brief erroneous reasoning
  step. Short-event retention and downstream fusion benefit need direct tests.
- This read supports a scoped adaptation; it does not establish a currently
  complete literature survey or a positive localization result.
