# GL-LIU factorial v2: unified DUFS graph and broad token views

Date: 2026-08-08

Status: executed on the eight cached ProcessBench model/dataset cells.

## Research question

GL-LIU v1 used two Laplacian constructions:

- global head: a DUFS-gated feature-space kNN graph;
- local head: a temporal-chain graph over adjacent tokens.

The previous results suggested that local DUFS-LIU transferred slightly better
than temporal LIU outside the two development cells. The first question is
whether one DUFS-LIU construction can be used in both heads.

The second question is whether the local head improves when it receives
token-resolved counterparts of the broad global feature pool instead of only
five native entropy and spilled-energy dynamics.

## Hypotheses

### H1: unified DUFS-LIU

When the local feature pool is fixed, a DUFS-gated local graph will transfer
more consistently than a fixed temporal graph because it connects tokens with
similar telemetry states rather than assuming that useful risk is temporally
smooth.

Failure condition: the feature geometry may describe confidence level, token
position, or generation style instead of first-error relevance.

### H2: broad token views

The broad pool may improve localization if the additional energy and top-k
distribution curves contain local error information that the five native
dynamics miss. DUFS gates may reduce the effect of irrelevant curves.

Failure condition: DUFS optimizes neighbourhood preservation, not correctness.
It can prefer coherent nuisance features and move the risk argmax away from the
first error.

## Controlled design

Two 2x2 matrices are reported. This prevents graph construction and feature
pool size from changing in the same comparison.

### Matrix A: graph construction

| | local temporal LIU, core 5 | local DUFS-LIU, core 5 |
|---|---|---|
| global IU-PCR | A1 | A2 |
| global DUFS-LIU | A3, frozen GL-LIU v1 | A4, unified candidate |

### Matrix B: local feature pool

| | local DUFS-LIU, core 5 | local DUFS-LIU, broad 28 |
|---|---|---|
| global IU-PCR | B1 | B2 |
| global DUFS-LIU | B3, unified core | B4, unified broad |

No cell is selected after evaluation. Every matrix cell is reported.

## Frozen parameters

| setting | value |
|---|---:|
| global Laplacian strength | 0.1 |
| local Laplacian strength | 0.3 |
| graph neighbours | 7 |
| DUFS seeds | 11, 23, 37 |
| DUFS epochs | 80 |
| maximum graph-fit tokens | 60,000 |
| IU-PCR subspace | 2 components |
| threshold evaluation | 100 repeated half calibration/evaluation splits |
| split seed | 0 |

These values come from the frozen GL-LIU v1 experiment. They were not tuned for
the broad local pool.

## Local feature contracts

### Core 5

1. token entropy;
2. entropy sliding-window variance;
3. absolute entropy CUSUM;
4. spilled-energy sliding-window variance;
5. absolute spilled-energy CUSUM.

### Broad 28

The global registry has 30 feature names. In these caches, 29 scalar features
survived the frozen global preprocessing. A local counterpart must vary within
one trace:

- `trace_length` is excluded because it is constant within a trace;
- `cusum_max` and `cusum_shift_idx` use one shared absolute-CUSUM curve rather
  than two deterministic duplicates;
- the rolling `min_spilled` curve is included even though its global scalar was
  saturated, because the local curve varies.

The result is 28 unique token curves. They include raw entropy, spilled energy,
log-partition energy, top-k distribution statistics, and rolling spectral,
variance, CUSUM, permutation-entropy, tail-ratio, Hurst-proxy, and minimum
curves. The exact mapping is in
`spectral_utils/token_feature_views.py::TOKEN_TO_GLOBAL_FEATURES`.

The rolling spectral window is 32 tokens and the local window is 16 tokens.
These are local proxies for full-trace reductions. They are not exact copies of
the global scalar features.

## Label boundary

Score constructors do not accept or read correctness labels or ProcessBench
step spans. Detector scores, locator predictions, and continuous token curves
are hashed before labels are read.

Labels are then used for:

1. global AUROC and local component evaluation;
2. calibration-half threshold selection;
3. final evaluation on the untouched half.

This is calibrated unsupervised scoring, not a fully label-free decision rule.

## Required diagnostics

- exact hash equality with frozen GL-LIU v1 for both global heads and both
  five-view local heads;
- global AUROC and AUPRC;
- local exact and tolerance-one localization before thresholding;
- ProcessBench F1, erroneous-step accuracy, and clean accuracy;
- results for every model/dataset cell;
- DUFS gate survival probabilities and effective feature count;
- local feature effective rank;
- within-trace rank displacement caused by changing graph or feature pool;
- development and six-cell non-selection summaries.

## Decision rules

- A unified DUFS system is a leading candidate only if its gain is not confined
  to the two development cells and it does not create a large cell-level tail
  loss.
- The broad pool is useful only if it improves local exact localization and
  end-to-end F1 without relying on one dataset/model cell.
- A negative broad-pool result does not prove that every possible local feature
  expansion fails. It rejects this direct 28-curve construction under the
  frozen windows and DUFS settings.

## Executed result

- Frozen GL-LIU v1: 31.36% ProcessBench F1.
- Unified DUFS-LIU with core 5: 31.72%, a +0.37-point descriptive gain.
- Unified DUFS-LIU with broad 28: 29.03%, a -2.70-point loss against core 5.
- Unified core wins five of eight cells against GL-LIU v1.
- Broad 28 loses seven of eight cells against unified core.
- On the six non-selection cells, unified core is 31.41% versus 30.76% for
  GL-LIU v1.

The full interpretation is in
`results/gl_liu_factorial_v2/REPORT.md`. The concise presentation is in
`results/gl_liu_factorial_v2/ADVISOR_BRIEF.md`.

## Literature basis

- Yaniv Tenzer et al., *Crowdsourcing Regression: A Spectral Approach*, AISTATS
  2022. IU-PCR supplies the two-component unsupervised ensemble-regression
  estimate.
- Ofir Lindenbaum et al., *Differentiable Unsupervised Feature Selection based
  on a Gated Laplacian*, NeurIPS 2021. DUFS supplies the unlabeled gated feature
  metric used to construct the sample graph.

DUFS-LIU is our combination. Neither cited paper evaluates this two-head
hallucination-localization system.
