# Fixed feature-contract × solver comparison

Version: `feature-contract-solver-comparison-v1-2026-08-06`. Feature orientation schema: `confidence-orientation-v1`.

This is the apples-to-apples replay: every solver receives the same matrix for a given feature contract. There is **no per-cell `sign(rho)` orientation and no global anchor flip**. Larger input values are defined to mean greater confidence, and the raw fused score is evaluated directly. Labels enter only after all scores are frozen.

![Common feature contracts by solver](comparison.png)

## Solver provenance

| method | source | implementation scope |
|---|---|---|
| `deployed_upcr` | [Dror, Nadler, Bilal & Kluger (2017), Unsupervised Ensemble Regression](https://arxiv.org/abs/1703.02965) | maintained deployment configuration with exclusion and recomputation |
| `iu_pcr` | [Tenzer, Dror, Nadler, Bilal & Kluger (2022), Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html) | independent/uncorrelated-error variant, two-component PCR |
| `su_pcr` | [Tenzer, Dror, Nadler, Bilal & Kluger (2022), Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html) | sparse correlated-error variant, two-component PCR |

## Common-contract results

Cell-macro AUROC. `inv` is the number of cells whose unflipped score has AUROC below 0.5, which is an orientation-assumption failure.

| feature contract | deployed U-PCR | IU-PCR | SU-PCR | inv (U/I/S) |
|---|---:|---:|---:|---:|
| `fixed_all` | 0.7748 | 0.7754 | 0.7648 | 0/0/0 |
| `remove_unstable` | 0.7735 | 0.7741 | 0.7737 | 0/0/0 |
| `replace_squared` | 0.7730 | 0.7750 | 0.7580 | 0/0/0 |
| `replace_mode` | 0.7731 | 0.7756 | 0.7623 | 0/0/0 |

### Primary common baseline: remove unstable views

| method | overall | QA | math | mean input / retained views |
|---|---:|---:|---:|---:|
| `deployed_upcr` | 0.7735 | 0.7587 | 0.7824 | 24.8 / 20.4 |
| `iu_pcr` | 0.7741 | 0.7592 | 0.7830 | 24.8 / 24.8 |
| `su_pcr` | 0.7737 | 0.7592 | 0.7823 | 24.8 / 24.8 |

Contracts:

- `fixed_all`: all raw views with frozen higher-means-correct directions.
- `remove_unstable`: removes `pe_mean`, `stft_spectral_entropy`,   `cusum_shift_idx`, and `rpdi`.
- `replace_squared`: replaces those four columns with `-z²`; higher means closer to   the mean and therefore greater confidence under the declared central-confidence assumption.
- `replace_mode`: replaces them with `-|rank(x)-mode_rank|`, using a label-free KDE mode.

## Feature-contract effects

Paired cell deltas against the removal baseline:

| method | candidate | mean [95% CI] | W/L/T |
|---|---|---:|---:|
| `deployed_upcr` | `fixed_all` | +0.13pp [+0.06, +0.21] | 15/2/7 |
| `deployed_upcr` | `replace_squared` | -0.06pp [-0.14, +0.01] | 5/7/12 |
| `deployed_upcr` | `replace_mode` | -0.04pp [-0.13, +0.03] | 7/6/11 |
| `iu_pcr` | `fixed_all` | +0.13pp [-0.07, +0.40] | 13/11/0 |
| `iu_pcr` | `replace_squared` | +0.10pp [-0.09, +0.38] | 11/13/0 |
| `iu_pcr` | `replace_mode` | +0.15pp [-0.06, +0.48] | 13/11/0 |
| `su_pcr` | `fixed_all` | -0.88pp [-2.30, +0.24] | 14/10/0 |
| `su_pcr` | `replace_squared` | -1.57pp [-3.20, -0.19] | 11/13/0 |
| `su_pcr` | `replace_mode` | -1.13pp [-2.48, -0.05] | 13/11/0 |

## Solver effects within each identical contract

| contract | candidate vs deployed U-PCR | mean [95% CI] | W/L/T |
|---|---|---:|---:|
| `fixed_all` | `iu_pcr` | +0.06pp [-0.24, +0.36] | 13/11/0 |
| `fixed_all` | `su_pcr` | -1.00pp [-2.48, +0.14] | 13/11/0 |
| `remove_unstable` | `iu_pcr` | +0.05pp [-0.16, +0.29] | 13/11/0 |
| `remove_unstable` | `su_pcr` | +0.01pp [-0.26, +0.27] | 13/11/0 |
| `replace_squared` | `iu_pcr` | +0.21pp [-0.05, +0.50] | 13/11/0 |
| `replace_squared` | `su_pcr` | -1.50pp [-3.15, -0.11] | 12/12/0 |
| `replace_mode` | `iu_pcr` | +0.25pp [-0.04, +0.56] | 14/10/0 |
| `replace_mode` | `su_pcr` | -1.08pp [-2.54, +0.11] | 13/11/0 |

## Leave-one-family-out contract diagnostic

This diagnostic asks whether each solver should use a different whole feature contract. For each held-out dataset family, the contract is selected only on the other families. It is not the primary solver comparison because the methods may receive different inputs.

| method | LOFO-selected macro | contract choices by held-out family |
|---|---:|---|
| `deployed_upcr` | 0.7748 | `fixed_all`: 8 |
| `iu_pcr` | 0.7752 | `fixed_all`: 1; `replace_mode`: 7 |
| `su_pcr` | 0.7737 | `remove_unstable`: 8 |

## Interpretation boundary

The highest average common contract on this retrospective artifact is `remove_unstable`. This identifies the clean baseline for subsequent solver work; it is not prospective validation. Method-specific contract choices are reported only through the leave-one-family-out diagnostic, and no per-cell label-selected transformation is allowed.

The fixed directions themselves were frozen after examining earlier cells, so a new dataset/model family remains necessary to validate the complete orientation-free pipeline.

## Reproduction

```bash
python scripts/feature_contract_solver_comparison.py
```

Runtime: 1.8s; cells: 24.
