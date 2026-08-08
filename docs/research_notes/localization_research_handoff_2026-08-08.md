# Localization research handoff

Date: 2026-08-08

Status: stop and discuss before developing another variant.

## Executive summary

The localization work produced one promising end-to-end result, but too many
names now hide a much simpler structure. Almost every method in this study uses
the same two-component IU-PCR solve. The variants differ mainly in:

1. the unit being scored: a complete answer or one token;
2. the input features: full-trace summaries or token-resolved curves;
3. the graph used by the Laplacian: none, feature kNN, DUFS-gated kNN, or a
   temporal chain.

The frozen system is GL-LIU v1. It uses global DUFS-LIU to decide whether an
error exists and temporal Laplacian IU-PCR to propose the token location. It
beats the reproduced Mind the Gap control under the shared ProcessBench
calibration protocol. However, the temporal locator did not transfer as a
universal improvement. The global DUFS-LIU detector is the reliable part of
the result.

No new localization method is approved after this handoff. The next action is
to discuss the evidence and decide whether scientific simplicity requires the
same DUFS-LIU graph construction in both heads.

## One-page method map

### Solver family

| name | mathematical role | graph |
|---|---|---|
| U-PCR | estimates a continuous target from unlabeled regressor covariance | none |
| IU-PCR | the common two-component realization used as the protected baseline | none |
| Laplacian IU-PCR, or LIU | adds score roughness `R = F L F^T / n` inside the IU-PCR subspace | supplied by the method |
| DUFS-LIU | LIU where DUFS gates define the feature metric and the sample kNN graph | DUFS-gated kNN |
| temporal LIU | LIU where adjacent tokens in the same trace are connected | temporal chain |
| uniform LIU | LIU with an ungated feature-space kNN graph | uniform kNN |

DUFS-LIU and temporal LIU are not different regression solvers. They are two
ways to construct `L` for the same final LIU equation:

```text
w = U [U^T (C + lambda R_bar) U]^-1 U^T rho_hat
```

### Resolution family

| head | one column/sample represents | input |
|---|---|---|
| global | one complete answer | scalar summaries of the full trace |
| local | one token | continuous token-resolved curves |

The global and local heads use the same LLM generation and raw token telemetry.
There is no second LLM, no second prompt, and no second generation. They are
two post-processing views of one trace.

## Data and evaluation boundary

The study uses ProcessBench outputs for Qwen3-4B and Qwen3-8B on GSM8K, MATH,
OlympiadBench, and OmniMath.

Component-selection cells:

- Qwen3-4B / GSM8K;
- Qwen3-4B / MATH.

Non-selection and model-transfer cells:

- Qwen3-4B / OlympiadBench and OmniMath;
- all four Qwen3-8B cells.

The 4B and 8B cells reuse the same underlying ProcessBench examples. There are
four independent dataset families, not eight. Only OlympiadBench and OmniMath
are new dataset families relative to component selection.

Score construction uses no correctness label and no reasoning-step boundary.
Labels are used for development selection, split-local threshold calibration,
and final evaluation. The method is calibrated unsupervised scoring, not a
fully label-free decision system.

## Experiments completed

### 1. Full-trace detector comparison

Candidates:

- deployed U-PCR;
- ordinary IU-PCR;
- uniform LIU;
- DUFS-LIU;
- stable-only and mixed feature contracts;
- maximum and top-5% aggregation of local token scores.

Result:

- mixed DUFS-LIU has development AUROC 0.7812;
- stable DUFS-LIU has development AUROC 0.7800;
- mixed ordinary IU-PCR has development AUROC 0.7791;
- local-token score aggregations are around 0.72 AUROC or below.

Conclusion:

- error presence should be estimated from full-trace features;
- mixed DUFS-LIU beats mixed ordinary IU-PCR in all eight cells, by about
  +0.22 AUROC percentage points on average;
- the global Laplacian effect is small but consistent;
- the mixed feature contract is not the main source of the result.

### 2. Native continuous-token localization

The method did not recompute features inside reasoning steps. It preserved the
token grid and constructed positional curves over the complete trace.

Five-feature core:

- entropy;
- entropy sliding-window variance;
- absolute entropy CUSUM;
- spilled-energy sliding-window variance;
- absolute spilled-energy CUSUM.

Seven-feature full positional pool:

- the five core views;
- rolling permutation entropy;
- STFT high-frequency power.

Candidates:

- token U-PCR with five and seven views;
- ordinary token IU-PCR;
- uniform feature-graph LIU;
- DUFS feature-graph LIU;
- temporal-chain LIU;
- `lambda` in `{0.03, 0.1, 0.3}`.

Development exact localization:

| locator | exact localization |
|---|---:|
| temporal LIU, lambda 0.3 | 30.22% |
| token U-PCR, five views | 29.60% |
| token U-PCR, seven views | 29.60% |
| DUFS feature-graph LIU, lambda 0.3 | 29.45% |
| ordinary token IU-PCR | 29.21% |

Six non-selection cells:

| locator | exact localization |
|---|---:|
| DUFS feature-graph LIU, lambda 0.3 | 25.78% |
| ordinary token IU-PCR | 25.75% |
| token U-PCR, five views | 25.27% |
| temporal LIU, lambda 0.3 | 25.14% |
| token U-PCR, seven views | 24.86% |

Conclusion:

- native moving-window curves contain localization information;
- adding PE and STFT to U-PCR did not help;
- temporal LIU won development because of GSM8K, but did not transfer;
- local DUFS-LIU and ordinary token IU-PCR are more stable controls;
- no experiment used all 29 global features as token-resolved localization
  features.

### 3. End-to-end GL-LIU v1

Frozen components:

- global detector: mixed-contract DUFS-LIU, `k=7`, `lambda=0.1`;
- local locator: temporal LIU, `lambda=0.3`;
- threshold: calibration-half ProcessBench-F1 optimum;
- evaluation: 100 repeated calibration/evaluation splits.

All eight cells:

| system | PB-F1 | exact | within one step | clean accuracy |
|---|---:|---:|---:|---:|
| Mind the Gap control | 25.71% | 17.84% | 39.35% | 48.63% |
| Mind the Gap detector + GL-LIU locator | 29.68% | 21.40% | 45.33% | 51.03% |
| GL-LIU v1 | **31.36%** | **21.79%** | **46.76%** | **57.99%** |

Six cells excluded from component selection:

| system | PB-F1 | exact | within one step | clean accuracy |
|---|---:|---:|---:|---:|
| Mind the Gap control | 24.74% | 16.98% | 38.21% | 47.81% |
| GL-LIU v1 | **30.76%** | **21.30%** | **46.62%** | **57.10%** |

GL-LIU F1 is higher in all eight cells. Exact localization is higher in seven
of eight cells.

## What succeeded

1. Full-trace fusion is the correct level for error-presence detection.
2. Our global DUFS-LIU score can replace the reproduced Mind the Gap detector.
3. Continuous token curves can localize without step-based feature
   construction.
4. Separating error presence from error placement improves diagnosis and final
   F1.
5. The same LLM output is sufficient; no second generation or model is needed.

## What failed or remains fragile

1. Maximum or top-5% local-risk aggregation is a weak global detector.
2. The temporal Laplacian did not transfer as a universal locator improvement.
3. Adding PE and STFT positional curves did not improve token U-PCR.
4. The mixed global feature contract adds only a small amount over stable
   DUFS-LIU.
5. The current threshold requires calibration labels.
6. Mind the Gap is the only external published method measured in this exact
   run. The result is not yet a complete state-of-the-art benchmark.

## What has not been tested

The following must not be described as results:

1. global DUFS-LIU plus local DUFS-LIU evaluated end to end under the full
   repeated threshold protocol;
2. local DUFS-LIU using the seven-view positional pool;
3. local DUFS-LIU using token-resolved counterparts of the broader 29-feature
   global pool;
4. a fixed external threshold with no ProcessBench calibration labels;
5. a completely new dataset and model family;
6. comparison with more external published localization methods.

The 29 scalar global features cannot be copied directly to every token. Such a
feature is constant inside one trace and cannot directly change the within-
trace argmax. A valid broad local pool must return to raw telemetry and expose
the token-resolved curve before each global reduction. Some global features do
not have a natural local counterpart.

## Decision requested from the advisors

We should not optimize another variant before answering these questions:

1. Should scientific simplicity be prioritized by using DUFS-LIU for both the
   global and local heads?
2. Is the small but consistent global DUFS-LIU gain enough to support a method
   contribution, or is GL-LIU primarily a task decomposition contribution?
3. Should the next validation use a new ProcessBench-compatible model family,
   a new localization dataset, or both?
4. Is calibrated unsupervised scoring an acceptable claim, or must the final
   method use a completely label-free threshold?
5. Which additional external localization baselines are required before the
   result can be presented as a competitive benchmark?
6. Should we first test the simpler unified system—global DUFS-LIU plus local
   DUFS-LIU with five frozen views—before expanding the local feature pool?

## Recommended options for discussion

### Option A: freeze GL-LIU v1 and validate externally

This preserves the registered method. It is the cleanest confirmatory action,
but keeps the fragile temporal locator.

### Option B: simplify to DUFS-LIU in both heads

Use global DUFS-LIU on full-trace summaries and local DUFS-LIU on the five
token-resolved core views. This is algorithmically cleaner and the local DUFS
variant transfers slightly better descriptively. It requires a new frozen
end-to-end evaluation and cannot inherit the GL-LIU v1 headline.

### Option C: expand token-resolved features before choosing the local graph

Construct positional counterparts from raw top-k probabilities, energy, margin,
varentropy, Rényi entropy, and tail mass. Compare ordinary IU-PCR, DUFS-LIU,
and temporal LIU under one frozen feature contract. This may add information,
but risks reopening a large development search.

The current recommendation is to present Options A--C to the advisors and not
choose between them inside the current ProcessBench development data.

## Canonical artifacts

- Method definition: `docs/methods/gl_liu_v1.md`.
- Advisor report: `results/ours_only_localization_v1/REPORT.html`.
- Scientific report: `results/ours_only_localization_v1/REPORT.md`.
- Frozen run definition: `results/ours_only_localization_v1/RUN_DEFINITION.json`.
- Exact reproduction runner: `scripts/gl_liu_v1/run.py`.
- Result tables: `results/ours_only_localization_v1/*.csv`.
- Score diagnostics and hashes: `results/ours_only_localization_v1/diagnostics/`.
