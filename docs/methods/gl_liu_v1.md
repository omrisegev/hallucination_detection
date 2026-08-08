# GL-LIU v1: Global-Local Laplacian IU-PCR

**Status:** leading ProcessBench method, frozen on 2026-08-08.

**Full name:** Global-Local Laplacian IU-PCR (GL-LIU).

GL-LIU is our two-stage method for deciding whether a reasoning trace contains
an error and, if it does, locating the error. It uses only statistics collected
from one model generation. It does not use Mind the Gap scores and it does not
divide the trace into reasoning steps when it constructs features.

The method is a **calibrated unsupervised scoring method**. Correctness labels
are not used to construct its feature scores, DUFS gates, graphs, or IU-PCR
weights. Declared development labels are used to choose one detector and one
locator. Calibration labels are then used to convert the continuous detector
score into an error/no-error decision. This distinction is part of the method,
not a reporting detail.

## 1. Problem and output

For a model-generated trace `i`, GL-LIU returns:

1. a whole-trace error risk `q_i`;
2. a continuous token-risk curve `r_it`;
3. either **no error**, or the token index with maximum local risk.

Let `e_i=1` mean that the trace contains an error. Let `a_i` be the token index
of the first annotated error. Neither variable is visible while the two score
functions are fitted.

## 2. Stage A: global error detector

### 2.1 Full-trace features

Each complete trace is represented by a feature vector

\[
g_i \in \mathbb{R}^{m}.
\]

The ProcessBench run had 29 available features in every cell. They summarize
the complete entropy and token-probability trajectories. Put the vectors into

\[
F_g=[g_1,\ldots,g_n]\in\mathbb{R}^{m\times n},
\qquad C_g=\frac{1}{n}F_gF_g^\top.
\]

The feature contract is `dufs-liu-mixed-v2-development-2026-08-07`. Four
non-monotone features receive frozen, label-free transformations:

| feature | operation | purpose |
|---|---|---|
| `pe_mean` | negative squared standardized value, `-z^2` | reward proximity to the central confidence regime |
| `stft_spectral_entropy` | negative percentile-rank distance from its unlabeled KDE mode percentile | represent a non-monotone optimum without correctness labels |
| `cusum_shift_idx` | raw value with fixed confidence orientation | retain ordered shift timing |
| `rpdi` | raw value with fixed confidence orientation | retain its stable direction |

Each transformed feature replaces its original column. It is not added as a
duplicate.

### 2.2 DUFS sample geometry

DUFS is based on Lindenbaum et al. (2021). It learns a soft gate `gamma_j` for
each input feature without correctness labels. We use the gates as a sample
metric, rather than using DUFS only to delete features:

\[
d_g(i,j)^2=\sum_{r=1}^{m}\gamma_r^2(g_{ri}-g_{rj})^2.
\]

A symmetric self-tuning `k`-nearest-neighbour graph `W_g` is built from these
distances. Its normalized Laplacian is

\[
L_g=I-D_g^{-1/2}W_gD_g^{-1/2}.
\]

The corresponding feature-space roughness matrix is

\[
R_g=\frac{1}{n}F_gL_gF_g^\top.
\]

This is our DUFS-LIU construction. It is not an algorithm evaluated in either
the DUFS paper or the IU-PCR paper.

### 2.3 Laplacian IU-PCR solve

IU-PCR is based on the uncorrelated-error spectral estimator in Tenzer et al.
(2022). Let `rho_hat` be its estimated feature-target covariance vector and let
`U_g` contain the leading two eigenvectors of `C_g`. We trace-match the
projected roughness to the projected covariance, producing `R_bar_g`, and
solve only inside this protected two-dimensional subspace:

\[
w_g=
U_g\left[U_g^\top(C_g+\lambda_g\bar R_g)U_g\right]^{-1}
U_g^\top\widehat\rho.
\]

The confidence score is `w_g^T g_i`. The error-risk convention used by GL-LIU
is

\[
q_i=-w_g^\top g_i,
\]

so a larger value means that an error is more likely.

## 3. Stage B: continuous token locator

The locator treats the generated answer as one continuous token sequence. For
token `t`, it constructs

\[
x_{it}=\left[
H_{it},\;\operatorname{SWVar}_{16}(H)_{it},\;
|\operatorname{CUSUM}(H)_{it}|,\;
\operatorname{SWVar}_{16}(S)_{it},\;
|\operatorname{CUSUM}(S)_{it}|
\right],
\]

where `H` is token entropy and `S` is spilled energy. These are our native
moving-window signals. No reasoning-step boundary is given to the locator.

The frozen v1 locator connects adjacent tokens from the same trace. If `W_l`
is this temporal-chain graph, then

\[
L_l=I-D_l^{-1/2}W_lD_l^{-1/2},
\qquad
R_l=\frac{1}{N}F_lL_lF_l^\top.
\]

The same two-component Laplacian IU-PCR equation produces `w_l`, using
`lambda_l=0.3`. The risk curve is

\[
r_{it}=w_l^\top x_{it},
\qquad
\widehat t_i=\operatorname*{arg\,max}_t r_{it}.
\]

Its direction is fixed against the entropy-risk anchor without correctness
labels. At most 60,000 unlabeled tokens per cell are used to fit the locator.

The temporal locator is the formally frozen v1 choice because it won on the
declared development cells. Its gain did not transfer consistently. It is
therefore a development candidate inside the leading end-to-end system, not a
confirmed independent contribution. Ordinary token IU-PCR and DUFS feature-
graph IU-PCR remain required robustness controls in the next run.

## 4. Decision rule and evaluation mapping

A threshold `tau` is fitted on a calibration split:

\[
\widehat y_i=
\begin{cases}
\text{no error}, & q_i\leq\tau,\\
\widehat t_i, & q_i>\tau.
\end{cases}
\]

In the current benchmark, `tau` maximizes ProcessBench F1 on the calibration
half and is evaluated on the untouched half. This process is repeated 100
times with seed 0.

ProcessBench supplies step spans. GL-LIU does not use them to compute any
feature or score. Only after the token prediction is frozen, the benchmark maps
`t_hat_i` to a step in order to calculate localization accuracy.

## 5. Frozen v1 configuration

| component | setting |
|---|---|
| global detector | mixed-contract DUFS-LIU |
| DUFS seeds | 11, 23, 37 |
| DUFS epochs | 80 |
| global graph | gated feature-space kNN, `k=7` |
| global Laplacian strength | `lambda_g=0.1` |
| global feature count | 29 where all registered features are available |
| local views | entropy, entropy SWVar/CUSUM, spilled-energy SWVar/CUSUM |
| local graph | within-trace temporal chain |
| local Laplacian strength | `lambda_l=0.3` |
| maximum local fit data | 60,000 unlabeled tokens per cell |
| IU-PCR subspace | two components |
| threshold | calibration-half ProcessBench-F1 optimum |
| evaluation | 100 repeated calibration/evaluation splits |

## 6. What the method relies on

The global stage assumes that traces close in the DUFS-gated feature geometry
should receive similar confidence scores. The local stage assumes that the
first-error signal has useful temporal continuity. Both assumptions can fail
when a stable graph describes length, style, difficulty, or another nuisance
rather than correctness.

Required diagnostics are:

- global detection AUROC before threshold calibration;
- exact equality to ordinary IU-PCR at `lambda=0`;
- graph connectivity, degree, edge mass, and spectral gap;
- DUFS gate stability across seeds;
- score-rank displacement and projected roughness;
- exact and tolerance-one localization before the global decision threshold;
- clean-trace abstention accuracy and ProcessBench F1 after calibration;
- results by dataset family and model, not only an eight-cell macro average.

## 7. Evidence as of 2026-08-08

The method was selected on Qwen3-4B/GSM8K and Qwen3-4B/MATH. The other six
model/dataset cells were not used for component selection. However, Qwen3-4B
and Qwen3-8B use the same underlying ProcessBench examples, so there are four
independent dataset families, not eight. Only OlympiadBench and OmniMath are
new dataset-family confirmation sets.

Across all eight cells:

| system | ProcessBench F1 | exact localization | within one step | clean accuracy |
|---|---:|---:|---:|---:|
| Mind the Gap detector and locator | 25.71% | 17.84% | 39.35% | 48.63% |
| Mind the Gap detector + GL-LIU locator | 29.68% | 21.40% | 45.33% | 51.03% |
| **GL-LIU v1** | **31.36%** | **21.79%** | **46.76%** | **57.99%** |

On the six cells not used for component selection, GL-LIU v1 scored 30.76%
ProcessBench F1 versus 24.74% for the Mind the Gap control. GL-LIU F1 was
higher in every cell. Exact localization was higher in seven of eight cells.

The global mixed DUFS-LIU detector achieved development AUROC 0.7812. It beat
mixed ordinary IU-PCR in all eight cells by about 0.22 AUROC percentage points
on average. This is a small but consistent global Laplacian effect.

The local temporal Laplacian achieved 30.22% exact localization on development,
versus 29.60% for token U-PCR and 29.21% for ordinary token IU-PCR. On the six
non-selection cells, however, it averaged about 25.14%, below the DUFS feature-
graph locator at about 25.78%. A universal temporal-Laplacian improvement is
therefore not supported.

## 8. Competitor boundary

Mind the Gap is the immediate same-data competitor. The control in our report
uses its reproduced Shannon-Drop error score and token locator. We apply the
same split-local F1 threshold calibration to both systems. This is a fair
comparison of the scores and locators under one ProcessBench protocol, but it
is not the paper's original Neyman-Pearson decision operating point.

The hybrid control keeps the Mind the Gap detector and replaces only its
locator. It is a mechanism ablation, not a proposed final method.

Deployed U-PCR, ordinary IU-PCR, uniform-LIU, and stable DUFS-LIU are component
controls. They establish whether an observed gain comes from the transformed
feature contract, the graph, the DUFS metric, or only the two-component IU-PCR
solve.

Mind the Gap is currently the only external published method measured in this
exact GL-LIU run. The internal spectral controls are necessary mechanism
comparisons, but they are not additional external competitors. The present
result should therefore not be described as a complete state-of-the-art
benchmark.

## 9. Claim that can be presented

On the reproduced ProcessBench outputs, GL-LIU v1 is a one-generation,
token-statistics-based system that constructs its scores without correctness
labels or reasoning-step boundaries. Under the same repeated calibration
protocol, it improves ProcessBench F1 over the reproduced Mind the Gap control
from 25.71% to 31.36%, and from 24.74% to 30.76% on the six cells excluded from
component selection. The global DUFS-LIU detector is the confirmed part of the
method. The temporal locator remains provisional and requires confirmation on
a new dataset family and preferably a new model family.

Do not claim that GL-LIU is fully label-free, that its temporal Laplacian is
universally better, or that eight cells are eight independent datasets.

## 10. Sources and artifacts

- Yaniv Tenzer et al., [Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html), AISTATS 2022.
- Ofir Lindenbaum et al., [Differentiable Unsupervised Feature Selection based on a Gated Laplacian](https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html), NeurIPS 2021.
- Reproduced Mind the Gap artifacts and ProcessBench caches in this repository.
- Full GL-LIU report: `results/ours_only_localization_v1/REPORT.md`.
- Frozen configuration: `results/ours_only_localization_v1/RUN_DEFINITION.json`.
- Component selection: `results/ours_only_localization_v1/selection.json`.
- Reproducibility runner: `scripts/gl_liu_v1/run.py`.
