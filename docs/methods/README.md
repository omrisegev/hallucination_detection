# Spectral fusion methods

## Current leading ProcessBench method

**GL-LIU v1 (Global-Local Laplacian IU-PCR)** is the current leading method
for end-to-end error detection and first-error localization. It combines a
global mixed-contract DUFS-LIU detector with a continuous moving-window
Laplacian IU-PCR token locator. Read its complete definition, claim boundary,
and results in [GL-LIU v1](gl_liu_v1.md).

The documents below define the component methods and the earlier frozen
24-cell answer-correctness benchmark. GL-LIU does not erase those results: it
uses a different ProcessBench task and explicitly separates whole-trace
detection from continuous-token localization.

The post-freeze factorial follow-up keeps GL-LIU v1 as the reproducible
reference and carries **unified DUFS-LIU with five local views** as the simplest
candidate for external confirmation. It scores 31.72% ProcessBench F1 versus
31.36% for v1, but the gain is mixed across cells. Expanding the local pool to
28 curves lowers F1 to 29.03%. Read the executed design in
[GL-LIU factorial v2](../experiments/GL_LIU_FACTORIAL_V2.md).

## Earlier methods in the frozen 24-cell fusion benchmark

This directory explains every headline method in plain English and then gives
its mathematical definition. Read this page before the individual method
documents.

## Common terms

- A **cell** is one dataset/model pair. The benchmark has 24 cells: 9
  question-answering cells and 15 mathematics cells.
- A **sample** is one generated answer.
- A **feature** is one continuous signal computed from the model trace. U-PCR
  treats features as regressors or “experts.”
- The **target** is binary correctness. It is hidden while every fusion method
  is fitted and is opened only for evaluation.
- A **sample graph** connects answers that look similar in a chosen feature or
  representation space.
- A graph **Laplacian** measures how rapidly a score changes between connected
  samples. A small Laplacian energy means the score is smooth on the graph.
- **AUROC** measures ranking: it is the probability that a random correct
  answer receives a higher score than a random incorrect answer. Random ranking
  has AUROC 0.5.
- **AUPRC** summarizes precision and recall. Its random reference equals the
  positive rate, so it is important for highly imbalanced cells.
- A **cell-macro average** gives every cell equal weight. A **family-macro
  average** first combines repeated cells from the same dataset family.

## Headline comparison

| benchmark name | paper source | what is paper-based | what is ours |
|---|---|---|---|
| Deployed U-PCR | Dror et al. (2017), with later U-PCR details from Tenzer et al. (2022) | unlabeled covariance identity and PCR fusion | fixed confidence directions, stable feature pool, deployed exclusion/fallback policy |
| IU-PCR | Tenzer et al. (2022) | uncorrelated-error U-PCR model | fixed two-component, full-pool realization for a common LIU anchor |
| DUFS-LIU | Lindenbaum et al. (2021) + Tenzer et al. (2022) | DUFS stochastic gates; IU-PCR reliability estimate | use soft gates as a sample metric and add a Laplacian penalty to the final IU-PCR solve |
| Adapted SpecRaGE-Y-LIU | Yacobi et al. (2024 manuscript) + Tenzer et al. (2022) | multi-view neural spectral embedding and dynamic fusion | three registered view schemas, numerical changes, graph on learned embedding, and LIU solve |
| CA-SpecRaGE-alpha-LIU | inspired by Yacobi et al. | SpecRaGE encoders, fusion weights, and spectral loss | cross-view agreement target, view-mass normalization, edge-mass loss, alpha-weighted graph, and LIU solve |

The last three methods are combinations or extensions. Their full names must be
used in a paper. They must not be presented as algorithms evaluated in the
source papers.

The SpecRaGE factorial compares three view schemas. `manual` uses semantic
provenance families, `atomic` uses one feature per view with duplicate-balanced
prior mass, and `micro` uses leave-one-cell-out groups learned from stable
projected IU-PCR roughness. View construction is therefore an experimental
factor, not a fixed assumption hidden inside the method.

## Shared data contract

All headline methods receive exactly the same `fixed_stable_v1` features. Each
feature has a frozen direction: larger means “more likely correct.” Four raw
features with repeatedly unstable or non-monotone behavior are excluded. No
method may estimate a sign from the labels of the cell being evaluated, and no
score may be flipped after AUROC is observed.

The feature contract was developed using information from these 24 cells. It
therefore prevents per-cell leakage during this run, but it does not make the
24 cells a previously unseen confirmation set.

## Primary sources

- Omer Dror, Boaz Nadler, Erhan Bilal, and Yuval Kluger,
  [Unsupervised Ensemble Regression](https://arxiv.org/abs/1703.02965), 2017.
- Yaniv Tenzer, Omer Dror, Boaz Nadler, Erhan Bilal, and Yuval Kluger,
  [Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html),
  AISTATS 2022.
- Ofir Lindenbaum, Uri Shaham, Erez Peterfreund, Jonathan Svirsky, Nicolas
  Casey, and Yuval Kluger,
  [Differentiable Unsupervised Feature Selection based on a Gated Laplacian](https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html),
  NeurIPS 2021.
- Amitai Yacobi, Ofir Lindenbaum, and Uri Shaham,
  [Generalizable and Robust Spectral Method for Multi-view Representation Learning](https://arxiv.org/abs/2411.02138),
  arXiv:2411.02138. This is cited as a manuscript, not as a peer-reviewed result.

## Documents

- [Deployed U-PCR](deployed_upcr.md)
- [IU-PCR](iu_pcr.md)
- [DUFS-LIU](dufs_liu.md)
- [Adapted SpecRaGE-Y-LIU](adapted_specrage_y_liu.md)
- [CA-SpecRaGE-alpha-LIU](ca_specrage_alpha_liu.md)
- [Frozen experiment protocol](../experiments/FROZEN_24_CELL_BENCHMARK.md)
