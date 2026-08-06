# Feature selection × PCR solver comparison

Version: `feature-selection-pcr-comparison-v1-2026-08-06`; feature schema: `confidence-orientation-v1`; contract: `remove_unstable`.

This reruns feature selection on the corrected, orientation-free pool. No old L-SML subset is reused. There is no per-cell `sign(rho)` and no post-hoc score flip; labels enter only after selection and fusion are frozen.

**Result:** no tested selector improves any of the three PCR solvers over the full corrected pool. Some selectors beat random subsets of the same size, which shows they recover structure, but every reduction still discards complementary hallucination signal.

![Selector and solver interactions](comparison.png)

## Where selection is inserted

| solver | insertion and guardrail | why |
|---|---|---|
| deployed U-PCR | selector → U-PCR's maintained exclusion → recompute | external selection must add value beyond the solver's existing weak-view exclusion |
| IU-PCR | selector → IU-PCR; report residual dependence/effective rank | DPP and decorrelation target the uncorrelated-error assumption, but may select independent noise |
| SU-PCR | selector → SU-PCR; report decomposition convergence and sparse-support theorem | selection changes the covariance graph, so theorem support is a validity diagnostic, not an objective to game by deletion |

## Selector provenance and tested concept

| family | source/concept tested here |
|---|---|
| DUFS | Lindenbaum et al. (NeurIPS 2021), parameter-free gated-Laplacian objective |
| GroupFS | Lifshitz et al. (AAAI 2026), joint feature-group discovery and group gates |
| Laplacian Score / SPEC / MCFS | sample-manifold spectral ranking (He et al. 2005; Zhao & Liu 2007; Cai et al. 2010) |
| Concrete AE | Balin, Abid & Zou (ICML 2019), reconstruction-preserving subset |
| LS-CAE | Shaham, Lindenbaum, Svirsky & Kluger (2021), reconstruction plus a Laplacian computed on the selected representation |
| DPP / decorrelation | covariance-volume and minimum-redundancy conditions; these directly probe IU-PCR's uncorrelated-error assumption |
| U-PCR residual | method-specific projection-residual minimization |

## Baselines

| method | full corrected pool AUROC | mean input / final kept |
|---|---:|---:|
| `deployed_upcr` | 0.7735 | 24.75 / 20.42 |
| `iu_pcr` | 0.7741 | 24.75 / 24.75 |
| `su_pcr` | 0.7737 | 24.75 / 24.75 |

## Native stopping rules

`Δfull` is the paired cell-macro change from the full corrected pool. `Δrandom` compares each chosen subset with 32 random subsets of the same size in the same cell and solver. A method must improve `Δfull` and beat `Δrandom` to establish useful selection rather than a generic small-subset effect.

| selector | mean k / no-op cells | deployed U-PCR Δfull / Δrandom | IU-PCR Δfull / Δrandom | SU-PCR Δfull / Δrandom |
|---|---:|---:|---:|---:|
| `dufs_native` | 16.42 / 0 | -1.27pp / -0.59pp | -0.86pp / -0.43pp | -0.80pp / +0.55pp |
| `groupfs_native` | 21.71 / 19 | -0.36pp / +0.07pp | -0.45pp / -0.12pp | -0.45pp / +0.41pp |
| `lapscore_native` | 6.17 / 0 | -2.86pp / -0.23pp | -3.44pp / -0.53pp | -5.76pp / -1.55pp |
| `spec_native` | 6.17 / 0 | -2.86pp / -0.23pp | -3.44pp / -0.53pp | -5.76pp / -1.55pp |
| `mcfs_native` | 4.62 / 0 | -2.64pp / +0.35pp | -6.36pp / -2.21pp | -5.60pp / +1.91pp |
| `cae_native` | 4.92 / 0 | -2.78pp / +0.10pp | -2.50pp / +1.37pp | -4.55pp / +0.04pp |
| `lscae_native` | 6.50 / 0 | -2.22pp / +0.12pp | -2.08pp / +0.30pp | -2.89pp / +0.89pp |
| `dpp_native` | 18.96 / 0 | -0.53pp / -0.24pp | -0.67pp / -0.48pp | -1.17pp / -0.22pp |
| `dpp_ridge_native` | 19.75 / 0 | -0.37pp / -0.13pp | -0.55pp / -0.36pp | -1.13pp / -0.35pp |

## Equal-budget ranking test (k=6)

| selector | deployed U-PCR Δfull / Δrandom | IU-PCR Δfull / Δrandom | SU-PCR Δfull / Δrandom |
|---|---:|---:|---:|
| `dufs_k6` | -2.10pp / +0.38pp | -3.80pp / -1.38pp | -3.50pp / +0.34pp |
| `lapscore_k6` | -2.43pp / +0.05pp | -3.11pp / -0.68pp | -2.45pp / +1.38pp |
| `spec_k6` | -2.43pp / +0.05pp | -3.11pp / -0.68pp | -2.45pp / +1.38pp |
| `mcfs_k6` | -1.19pp / +1.29pp | -1.45pp / +0.97pp | -5.12pp / -1.28pp |
| `cae_k6` | -2.08pp / +0.40pp | -1.65pp / +0.78pp | -4.85pp / -1.02pp |
| `lscae_k6` | -1.64pp / +0.84pp | -1.68pp / +0.75pp | -1.87pp / +1.97pp |
| `dpp_k6` | -5.12pp / -2.64pp | -4.77pp / -2.34pp | -4.99pp / -1.16pp |
| `decorr_k6` | -5.62pp / -3.14pp | -4.91pp / -2.49pp | -8.15pp / -4.31pp |

## Solver-specific control

The U-PCR-residual greedy selector retained 3.79 views on average and changed deployed U-PCR by -4.53pp; its advantage over matched random was -1.08pp. It is not applied to IU-PCR or SU-PCR because its objective is U-PCR's own projection residual.

## SU-PCR validity audit

Numerical SU-PCR outputs below five views are retained for diagnosis, but are outside the paper's minimum-size theorem condition. The table therefore reports both the unconditional support rate and the rate among size-eligible cells.

| arm | size ≥5 | theorem support: all / size-eligible | decomposition convergence |
|---|---:|---:|---:|
| `full` | 100.0% | 100.0% / 100.0% | 100.0% |
| `dufs_native` | 100.0% | 91.7% / 91.7% | 100.0% |
| `groupfs_native` | 100.0% | 100.0% / 100.0% | 100.0% |
| `lapscore_native` | 83.3% | 70.8% / 85.0% | 100.0% |
| `spec_native` | 83.3% | 70.8% / 85.0% | 100.0% |
| `mcfs_native` | 41.7% | 12.5% / 30.0% | 100.0% |
| `cae_native` | 62.5% | 12.5% / 20.0% | 100.0% |
| `lscae_native` | 87.5% | 66.7% / 76.2% | 100.0% |
| `dpp_native` | 100.0% | 87.5% / 87.5% | 100.0% |
| `dpp_ridge_native` | 100.0% | 87.5% / 87.5% | 100.0% |
| `dufs_k6` | 100.0% | 95.8% / 95.8% | 100.0% |
| `lapscore_k6` | 100.0% | 79.2% / 79.2% | 100.0% |
| `spec_k6` | 100.0% | 79.2% / 79.2% | 100.0% |
| `mcfs_k6` | 100.0% | 33.3% / 33.3% | 100.0% |
| `cae_k6` | 100.0% | 25.0% / 25.0% | 100.0% |
| `lscae_k6` | 100.0% | 87.5% / 87.5% | 100.0% |
| `dpp_k6` | 100.0% | 16.7% / 16.7% | 100.0% |
| `decorr_k6` | 100.0% | 20.8% / 20.8% | 100.0% |

## What the experiment establishes

- **Deployed U-PCR:** its internal exclusion is the correct selection location for now. The best external arm, GroupFS, is still negative; its apparent closeness comes from making no selection in 19/24 cells. The U-PCR-residual selector also fails, so a lower equation residual is not a proxy for hallucination relevance.
- **IU-PCR:** enforcing diversity does not rescue the independence model. At k=6, DPP changes IU-PCR by -4.77pp and decorrelation by -4.91pp; both also lose to matched random. Independence without a relevance term preferentially keeps orthogonal noise.
- **SU-PCR:** the full stable pool already has 100% sparse-support validity and 100% decomposition convergence. Selection cannot improve that condition; aggressive small subsets often make the theorem inapplicable and can create raw-score inversions. LS-CAE at k=6 is meaningfully better than random six-view subsets, but remains below full SU-PCR.
- **Structure is real but insufficient:** MCFS/CAE/LS-CAE sometimes beat matched random. That is evidence that their rankings are not arbitrary, not evidence that feature removal improves the detector. The useful next concept is relevance-aware shrinkage or soft weighting—not another diversity-only hard selector.
- Laplacian Score and SPEC selected exactly the same subsets in all 24 cells under this standardized construction; their identical rows are one empirical condition, not two independent confirmations.

## Decision rule

A selector is considered a credible improvement only if its mean `Δfull` is positive, its 95% paired bootstrap interval excludes zero, it beats matched random on average, and it does not introduce orientation failures. SU-PCR additionally requires acceptable decomposition convergence and theorem-support behavior. Retrospective success is still a hypothesis for a new dataset/model family, not prospective proof.

Best observed non-method-specific arm by raw mean change:

- `deployed_upcr`: `groupfs_native` at -0.36pp versus full and +0.07pp versus matched random.
- `iu_pcr`: `groupfs_native` at -0.45pp versus full and -0.12pp versus matched random.
- `su_pcr`: `groupfs_native` at -0.45pp versus full and +0.41pp versus matched random.

## Validation audit

- Full-pool solver scores reproduce the preceding feature-contract experiment with maximum absolute AUROC difference `0`.
- Selector fallbacks: `0`; fixed-k arms with a non-six subset: `0`.
- Laplacian Score/SPEC subset identity: `24`/24 native and `24`/24 at k=6.
- Raw-score orientation failures: `7` across `1320` evaluations; all occurred in aggressive SU-PCR subset arms, not in any full-pool baseline.

## Files and reproduction

- `per_cell.csv`: every selector × solver result and matched-random floor.
- `selections.csv`: selected feature names and label-free diagnostics.
- `summary.csv`: macro effects, paired intervals, structural diagnostics.
- `feature_frequency.csv`: which corrected views each selector repeatedly chose.

```bash
python scripts/feature_selection_pcr_comparison.py
```

Runtime: 174.1s; cells: 24; matched-random repeats: 32.
