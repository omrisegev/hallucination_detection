# Repeated-measurement reliability U-PCR: development report

Date: 2026-08-08

## Short conclusion

The experiment found a stable covariance component created by resampling one
saved token trace. It did **not** improve hallucination detection in a meaningful
way. The best corrected method, Wiener-filtered DUFS-LIU, changed AUROC by only
`+0.0006` on Qwen3-4B/GSM8K and `+0.0013` on Qwen3-4B/MATH. Both paired
confidence intervals include zero. The method is therefore **not promoted** and
was not run on the six held ProcessBench cells.

This is still a useful result. It shows which part of the idea is valid, which
implementation is mathematically incompatible with U-PCR, and why stability
alone does not identify target-relevant information.

## Terms used in this report

- A **cell** is one dataset/model combination.
- A **repeated measurement** is an offline moving-block bootstrap of one saved
  token-telemetry trace. It is not another LLM generation or forward pass.
- **AUROC** measures how well a score ranks correct answers above incorrect
  answers. `0.5` is random ranking and `1.0` is perfect ranking.
- A **paired confidence interval** resamples the same answers for both methods.
  It measures uncertainty in their AUROC difference.
- (S_{\text{total}}) is covariance across the original answer-level features.
- (S_{\text{within}}) is covariance across bootstrap versions of the same
  answer. It measures sensitivity to this resampling procedure.
- (S_{\text{signal}}=S_{\text{total}}-S_{\text{within}}) is the proposed
  target-preserving covariance. This interpretation is allowed only when the
  bootstrap validity checks pass.

## Why this experiment was run

U-PCR estimates an unobserved target-covariance vector from covariance between
multiple regressors. Its cleanest model assumes independent regressor errors.
Earlier diagnostics in this repository showed that the feature errors are
strongly dependent and contain stable low-rank factors. However, removing those
factors did not help because the same factor can contain both target signal and
nuisance.

The new hypothesis was that repeated measurements could separate the two:

\[
S_{\text{signal}} \approx S_{\text{total}}-S_{\text{within}}.
\]

Reliable directions are then found from

\[
S_{\text{signal}}v=\lambda S_{\text{within}}v.
\]

A large generalized eigenvalue means that a direction varies more across
answers than it varies across bootstrap versions of the same answer.

This extension is motivated by the covariance view in Dror et al.,
*Unsupervised Ensemble Regression* (2017, arXiv:1703.02965), and Tenzer et al.,
*Crowdsourcing Regression: A Spectral Approach* (AISTATS 2022). DUFS-LIU uses
the Gated-Laplacian idea from Lindenbaum et al., *Let the Data Choose Its
Features: Differentiable Unsupervised Feature Selection* (arXiv:2007.04728).
The repeated-measurement covariance separation itself is our experimental
extension; it is not claimed as an algorithm from those papers.

## Protocol

The experiment used the existing Qwen3-4B ProcessBench caches. Every answer
still came from one LLM pass. The same bootstrap token indices were applied to
entropy, spilled energy, log-partition, and top-k log-probability channels.

The development sequence was:

1. Test block fractions `0.05`, `0.10`, and `0.20` on GSM8K without reading
   labels.
2. Freeze block fraction `0.20`.
3. Exclude a feature from the reliability estimator when its bootstrap mean is
   more than `0.5` original standard deviations from its original value, or its
   within variance exceeds its total variance. This rule was discovered on
   GSM8K and frozen before MATH was inspected.
4. Confirm the bootstrap assumptions on a separate MATH sample without labels.
5. Fit every fusion score and save its SHA-256 hash.
6. Only then read `final_answer_correct` and calculate AUROC.

Trace length was not used in (S_{\text{within}}), because a length-preserving
bootstrap gives it exactly zero within variance. It remained in the ordinary
`mixed-v2` baseline.

## Phase 0: does the repeated-measurement construction make sense?

Using all 28 varying features failed. The bootstrap changed several order-
sensitive statistics too strongly. On GSM8K at block fraction `0.20`, the full
pool had within/total trace ratio `1.00` and negative signal eigenmass `49.9%`.
The strongest failures included `sw_var_peak_spilled`, `rpdi`,
`cusum_shift_idx`, `dominant_freq`, and `pe_mean`.

The frozen procedure-compatibility rule retained 17 features on GSM8K and 18
on MATH. On the full benchmark rows, the retained pools had:

| Diagnostic | GSM8K | MATH |
|---|---:|---:|
| Answers | 400 | 998 |
| Compatible features | 17 | 18 |
| Within-covariance split correlation | 0.993 | 0.999 |
| Top-3 generalized-subspace overlap | 0.989 | 0.990 |
| Negative signal eigenmass | 4.12% | 0.76% |
| Within/total trace ratio | 0.349 | 0.240 |
| Directions with estimated signal/noise above 1 | 5 | 7 |

These checks passed. They establish that the restricted estimator is stable and
approximately compatible with the covariance subtraction. They do **not**
establish that the removed variation is hallucination-irrelevant noise.

## Fusion variants

The same `mixed-v2` feature contract was used throughout.

- **IU-PCR mixed-v2**: ordinary IU-PCR on the full feature pool.
- **DUFS-LIU mixed-v2**: current baseline; DUFS gates build a sample graph and
  the graph Laplacian regularizes the final IU-PCR system.
- **RM latent**: use the retained generalized eigenvectors directly as U-PCR
  regressors. This is the first implementation and a negative control.
- **RM projected**: keep the reliable generalized subspace, project into it,
  then return to the original feature axes before U-PCR.
- **RM Wiener**: softly filter the original feature axes using

  \[
  H=(S_{\text{signal}}+S_{\text{within}}+\epsilon I)^{-1}
  S_{\text{signal}}.
  \]

  This keeps all compatible feature axes while attenuating directions estimated
  to have low reliability.

All score signs were fixed without labels by positive correlation with the
confidence-oriented feature consensus.

## Detection results

| Method | GSM8K AUROC | MATH AUROC |
|---|---:|---:|
| IU-PCR mixed-v2 | 0.7656 | 0.7178 |
| DUFS-LIU mixed-v2 | **0.7673** | **0.7188** |
| RM latent IU-PCR | 0.7605 | 0.4707 |
| RM latent DUFS-LIU | 0.7607 | 0.4689 |
| RM projected IU-PCR | 0.7613 | 0.7101 |
| RM projected DUFS-LIU | 0.7617 | 0.7108 |
| RM Wiener IU-PCR | 0.7675 | 0.7199 |
| RM Wiener DUFS-LIU | **0.7679** | **0.7202** |

Paired differences for RM Wiener DUFS-LIU minus ordinary DUFS-LIU:

| Cell | AUROC difference | 95% paired interval | Bootstrap probability difference > 0 |
|---|---:|---:|---:|
| GSM8K | +0.0006 | [-0.0124, +0.0138] | 0.520 |
| MATH | +0.0013 | [-0.0068, +0.0095] | 0.638 |

The intervals include zero by a wide margin. The RM Wiener and baseline scores
have Pearson correlations `0.975` and `0.977`. The filter mostly reproduces the
same ranking.

![AUROC comparison](/Users/osegev/Desktop/hallucination_detection/results/repeated_measurement_reliability/benchmark/auroc_by_cell.png)

## What failed and why

The direct latent-basis version is mathematically unsuitable for U-PCR. The
generalized eigenbasis is designed to diagonalize signal and noise covariance.
U-PCR, however, estimates its missing target relation from off-diagonal
covariance between regressors. The off-diagonal covariance fraction fell from
`0.893` to `0.028` on GSM8K and from `0.897` to `0.067` on MATH. On MATH the
latent DUFS-LIU score variance collapsed to about `0.000025`.

Returning to the original feature axes fixed this mechanical failure. The hard
projection still lost information. The soft Wiener filter preserved the
baseline, but did not add a useful target-aligned ranking.

The main scientific conclusion is therefore:

> Moving-block bootstrap stability measures sensitivity to the resampling
> procedure. It is not evidence that a direction is irrelevant to answer
> correctness.

This is consistent with earlier repository results: stable graphs, factors,
and feature groups were often reproducible but did not predict which operation
improves AUROC.

## Decision and next step

- Keep **DUFS-LIU mixed-v2** as the current answer-level baseline.
- Do not promote any repeated-measurement variant.
- Do not run this candidate on the six held ProcessBench cells; the development
  effect is too small to justify spending confirmation data.
- Keep the Phase-0 diagnostics. They are useful for checking any future
  repeated-measurement construction.
- Reopen the direction only when the replicate-generation process has a clearer
  target-preserving interpretation than token block resampling. A useful next
  design should vary a known nuisance while holding the generated answer and
  its semantic content fixed.

## Reproduction

Run from the repository root:

```bash
.venv/bin/python scripts/test_repeated_measurement_reliability.py

MPLCONFIGDIR=/tmp/mpl-rm .venv/bin/python \
  scripts/repeated_measurement_reliability_pilot.py \
  cache/localization/processbench/pb_qwen3_4b/processbench_gsm8k.pkl \
  cache/localization/processbench/pb_qwen3_4b/processbench_math.pkl \
  --out-dir results/repeated_measurement_reliability/validity_confirmation \
  --max-rows 200 --repeats 12 --fractions 0.20

MPLCONFIGDIR=/tmp/mpl-rm LOKY_MAX_CPU_COUNT=8 .venv/bin/python \
  scripts/repeated_measurement_reliability_benchmark.py \
  cache/localization/processbench/pb_qwen3_4b/processbench_gsm8k.pkl \
  cache/localization/processbench/pb_qwen3_4b/processbench_math.pkl \
  --out-dir results/repeated_measurement_reliability/benchmark
```

The benchmark writes score hashes before it accesses the evaluation label. Two
independent 120-row reproduction runs produced identical hashes for all eight
methods.
