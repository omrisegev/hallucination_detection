# GL-LIU v1: Global-Local Laplacian IU-PCR

Date: 2026-08-08

## Result in one paragraph

GL-LIU v1, an end-to-end system using only our spectral methods, beat the reproduced Mind
the Gap control on ProcessBench. The system uses global DUFS-LIU to decide
whether a trace contains an error and continuous moving-window IU-PCR to place
the error. It does not construct features separately inside reasoning steps.
Across eight model/dataset cells, ProcessBench F1 increased from 25.71% to
31.36%. On the six confirmation/model-transfer cells, it increased from 24.74%
to 30.76%. The global DUFS-LIU detector transferred well. The temporal
Laplacian locator chosen on development was less stable: it helped strongly on
development GSM8K but did not beat ordinary or feature-graph IU consistently on
confirmation data. The system result is promising; the temporal locator should
remain a development candidate, not a settled contribution.

## What the metrics mean

- **Detection AUROC** measures whether one whole-trace score ranks erroneous
  traces above fully correct traces. It does not require a threshold.
- **Exact localization** is the fraction of erroneous traces for which the
  predicted token falls in the annotated first erroneous step.
- **SLA within one step** also accepts a prediction one step before or after the
  annotation.
- **ProcessBench F1** is the harmonic mean of exact localization on erroneous
  traces and correct abstention on fully correct traces. A system must answer
  both "is there an error?" and "where is it?".

## Research question

Can our global fusion and moving-window features solve ProcessBench end to end,
without using Mind the Gap as the error-presence detector?

The intended decomposition was diagnostic, not architectural dependence:

1. Choose a global detector using only our full-trace features.
2. Choose a continuous token locator using only our native moving-window
   features.
3. Combine the two after both choices are frozen.

## Data and split

The experiment used ProcessBench caches for Qwen3-4B and Qwen3-8B on GSM8K,
MATH, OlympiadBench, and OmniMath.

Development cells:

- Qwen3-4B / GSM8K
- Qwen3-4B / MATH

Confirmation and model-transfer cells:

- Qwen3-4B / OlympiadBench
- Qwen3-4B / OmniMath
- all four Qwen3-8B cells

The two model sizes use the same underlying ProcessBench examples. Therefore,
the experiment has four independent dataset families, not eight independent
datasets. Only OlympiadBench and OmniMath are new dataset-family confirmation
sets relative to the selection rule.

## No step-based feature construction

The method sees each output as one continuous token sequence. Its moving-window
statistics are computed over the complete trace. Step spans are not supplied to
U-PCR, IU-PCR, DUFS, the graph, or the detector.

The method outputs a token index. ProcessBench step spans are used only after
all scores have frozen, to map that token index to the benchmark's step label.
This mapping is evaluation, not feature construction.

## Detector candidates

The detector candidates used full-trace features. They included deployed U-PCR,
IU-PCR, uniform Laplacian IU-PCR, and DUFS-LIU under both the stable-only feature
contract and the frozen mixed feature contract.

For DUFS-LIU, let `F` be the feature-by-trace matrix, `C = FF^T/n`, and `L` the
normalized Laplacian of the DUFS-gated trace graph. The roughness matrix is

```text
R = F L F^T / n.
```

Inside the two-dimensional IU-PCR subspace `U`, the final weights are

```text
w_lambda = U [U^T (C + lambda R_bar) U]^-1 U^T rho_hat.
```

The detector risk is `-w_lambda^T F`: larger values mean a higher probability
that the trace contains an error. The frozen detector Laplacian strength was
`lambda=0.1`, with graph `k=7` and DUFS seeds 11, 23, and 37.

The mixed contract used 29 available full-trace features in every ProcessBench
cell. Its four registered decisions were:

- `pe_mean`: `-z^2`
- `stft_spectral_entropy`: negative distance from its label-free KDE mode
- `cusum_shift_idx`: raw with fixed orientation
- `rpdi`: raw with fixed orientation

The best development detector was `answer_dufs_liu_mixed`, with macro AUROC
0.7812. The stable DUFS-LIU detector scored 0.7800. Thus the mixed feature
contract helped only slightly. More importantly, mixed DUFS-LIU beat mixed
ordinary IU-PCR in all eight evaluated cells, by about 0.22 AUROC percentage
points on average. This is small but consistent evidence for the Laplacian at
the global detection stage.

![Development detector ranking](figures/development_detector_ranking.png)

## Continuous token locator candidates

The locator used five native full-trace token series:

- entropy;
- sliding-window entropy variance (`sw_var`);
- absolute entropy CUSUM;
- sliding-window spilled-energy variance;
- absolute spilled-energy CUSUM.

Every series is evaluated on the original token grid. No step-specific feature
extraction or text chunking is performed.

The study compared:

- deployed-style token U-PCR;
- ordinary token IU-PCR;
- IU-PCR with a uniform feature-space kNN Laplacian;
- IU-PCR with a DUFS-gated feature-space kNN Laplacian;
- IU-PCR with a temporal Laplacian that connects adjacent tokens within each
  trace.

The graph-based variants tested `lambda` in `{0.03, 0.1, 0.3}`. Every graph was
fit on at most 60,000 unlabeled tokens per cell. The predicted error token was
the maximum of the resulting continuous risk curve.

The development selector chose temporal Laplacian IU-PCR with `lambda=0.3`:

- development exact localization: 30.22%;
- deployed-style token U-PCR: 29.60%;
- ordinary token IU-PCR: 29.21%.

However, this advantage was driven by GSM8K. On MATH, the selected temporal
locator was worse than ordinary IU. Across the six confirmation/model-transfer
cells, the temporal locator averaged about 25.14% exact localization, while the
DUFS feature-graph locator at `lambda=0.3` averaged about 25.78%.

Therefore the temporal Laplacian result did not confirm as a universal
improvement. A robust next candidate should prefer ordinary or DUFS feature-
graph IU unless new data independently confirms the temporal graph.

![Development locator ranking](figures/development_locator_ranking.png)

![Laplacian transfer](figures/laplacian_lambda_transfer.png)

## End-to-end results

All systems below received the same repeated calibration/evaluation splits.
The threshold was selected on each calibration half to maximize ProcessBench
F1, then evaluated on the untouched half.

| system | PB-F1 | exact SLA | SLA within one step | clean accuracy |
|---|---:|---:|---:|---:|
| Mind the Gap detector + locator | 25.71% | 17.84% | 39.35% | 48.63% |
| Mind the Gap detector + GL-LIU locator | 29.68% | 21.40% | 45.33% | 51.03% |
| GL-LIU v1 | **31.36%** | **21.79%** | **46.76%** | **57.99%** |

GL-LIU v1 improved over Mind the Gap by:

- **+5.65 percentage points** ProcessBench F1;
- **+3.95 points** exact SLA;
- **+7.40 points** SLA within one step;
- **+9.36 points** clean-trace accuracy.

On the six non-selection cells only:

| system | PB-F1 | exact SLA | SLA within one step | clean accuracy |
|---|---:|---:|---:|---:|
| Mind the Gap detector + locator | 24.74% | 16.98% | 38.21% | 47.81% |
| Mind the Gap detector + GL-LIU locator | 29.08% | 20.77% | 45.08% | 50.66% |
| GL-LIU v1 | **30.76%** | **21.30%** | **46.62%** | **57.10%** |

The ours-only ProcessBench F1 was higher in all eight cells. Exact SLA was
higher in seven of eight; it was slightly lower on Qwen3-8B / OmniMath. The
global detector was also better than the Mind the Gap detector when both used
the same selected token locator. This means our previous dependence on Shannon
Drop was not necessary.

![Final F1 by cell](figures/final_f1_per_cell.png)

## What worked and what did not

### Confirmed useful

1. **Global and local evidence should be modeled separately.** Full-trace
   fusion was much better at deciding whether any error existed than taking the
   maximum or top 5% of a token-risk curve.
2. **Our global DUFS-LIU detector is viable.** It replaced Shannon Drop and
   improved final F1 on every cell.
3. **Native moving-window curves contain localization information.** They do
   not need to be recomputed inside steps.
4. **The global Laplacian effect is small but consistent.** Mixed DUFS-LIU beat
   mixed ordinary IU on detection AUROC in all eight cells.

### Not confirmed

1. **Temporal Laplacian IU is not yet a general locator improvement.** Its
   development gain came mainly from one dataset and did not transfer.
2. **The mixed feature contract is not the main cause of the result.** It
   improved the development detector by only about 0.12 AUROC percentage points
   over stable DUFS-LIU and was mixed across confirmation cells.
3. **A larger Laplacian strength is not universally better.** The lambda curves
   are nearly flat for feature-space graphs and decline for the temporal graph
   on confirmation data.

## Label use and claim boundary

All feature construction, DUFS gates, graphs, U-PCR/IU estimates, score curves,
and score hashes were produced without correctness labels.

Labels were used for:

- selecting the detector and locator on the two declared development cells;
- calibrating the final decision threshold inside each repeated split;
- calculating evaluation metrics.

Therefore this is a calibrated unsupervised scoring method, not a fully
label-free decision policy. A fixed external threshold or a label-free
abstention rule remains necessary for completely label-free deployment.

The Mind the Gap control in this report uses the same F1-optimized split-local
threshold as our system. It is a fair ProcessBench comparison of scores and
locators, but it is not the paper's original Neyman-Pearson operating point.

## Decision

The strongest result to carry forward is **GL-LIU v1**, the ours-only
two-stage system:

- global detector: frozen mixed-contract DUFS-LIU;
- continuous token locator: moving-window IU family;
- no Mind the Gap score in either component;
- step spans used only for evaluation mapping.

For a future external run, keep the selected temporal locator as the formally
frozen v1 candidate, but also pre-register ordinary IU and DUFS feature-graph IU
as robustness controls. Do not claim that temporal Laplacian localization is
better until it wins on a new dataset family that was not inspected here.

## Files

- `selection.json`: frozen v1 selection and hyperparameters.
- `development_detector_ranking.csv`: development detector selection.
- `development_locator_ranking.csv`: development locator selection.
- `component_metrics_per_cell.csv`: all component diagnostics.
- `final_systems_per_cell.csv`: final repeated-split system results.
- `diagnostics/`: score-generation diagnostics and score hashes.
- `figures/`: generated plots.
- `../../scripts/plot_ours_only_localization_v1.py`: plot reproduction.
