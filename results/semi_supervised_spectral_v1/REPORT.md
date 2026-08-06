# Semi-supervised spectral fusion v1

Decision: **STOP_AND_REVISE**.

This is a retrospective label-budget replay on the existing 24-cell feature bundle plus a disjoint synthetic mechanism study. A pass would require a prospective dataset/model-family confirmation.

## Registered gates

| gate | observed | rule | result |
|---|---:|---:|:---:|
| `platt_ranking_invariant` | 0.000000 | <= 0.000000 | **PASS** |
| `real_mean_vs_upcr` | -0.003553 | >= 0.010000 | **FAIL** |
| `real_ci_low_vs_upcr` | -0.006399 | > 0.000000 | **FAIL** |
| `real_mean_vs_gold_ridge` | 0.072886 | >= 0.000000 | **PASS** |
| `real_qa_vs_upcr` | -0.006147 | >= -0.005000 | **FAIL** |
| `real_math_vs_upcr` | -0.001996 | >= -0.005000 | **PASS** |
| `real_catastrophic_losses` | 0.000000 | <= 2.000000 | **PASS** |
| `synthetic_grouped_vs_upcr` | 0.000709 | > 0.000000 | **PASS** |
| `synthetic_weak_block_vs_upcr` | 0.307116 | > 0.000000 | **PASS** |
| `synthetic_worlds_beating_ridge` | 3.000000 | >= 3.000000 | **PASS** |
| `synthetic_independent_not_harmful` | -0.003277 | >= -0.005000 | **PASS** |

## Real-cell learning curve

Cell-macro held-out AUROC after averaging registered repetitions.

| labels | U-PCR | ridge-all | anchored-2 | anchored-6 | pseudo+gold-6 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7749 | — | 0.7749 | 0.7749 | 0.7749 |
| 5 | 0.7749 | 0.6721 | 0.7748 | 0.7728 | 0.7401 |
| 10 | 0.7749 | 0.6916 | 0.7742 | 0.7716 | 0.7329 |
| 20 | 0.7749 | 0.6984 | 0.7734 | 0.7713 | 0.7316 |
| 40 | 0.7749 | 0.7152 | 0.7738 | 0.7724 | 0.7414 |
| 80 | 0.7749 | 0.7382 | 0.7746 | 0.7739 | 0.7562 |

## Primary 20-label contrasts

| source/group | reference -> candidate | mean [95% CI] | W/L | <= -5pp |
|---|---|---:|---:|---:|
| `real/all` | `upcr` -> `anchored_pcr6` | -0.36pp [-0.64, -0.05] | 8/16 | 0 |
| `real/QA` | `upcr` -> `anchored_pcr6` | -0.61pp [-0.92, -0.29] | 1/8 | 0 |
| `real/math` | `upcr` -> `anchored_pcr6` | -0.20pp [-0.60, +0.20] | 7/8 | 0 |
| `synthetic/independent` | `upcr` -> `anchored_pcr6` | -0.33pp [-0.40, -0.26] | 1/39 | 0 |
| `synthetic/grouped` | `upcr` -> `anchored_pcr6` | +0.07pp [-0.26, +0.38] | 22/18 | 0 |
| `synthetic/sparse_pairs` | `upcr` -> `anchored_pcr6` | +0.79pp [+0.58, +1.01] | 35/4 | 0 |
| `synthetic/correlated_weak_block` | `upcr` -> `anchored_pcr6` | +30.71pp [+29.53, +31.90] | 40/0 | 0 |
| `real/all` | `gold_ridge_all` -> `anchored_pcr6` | +7.29pp [+6.23, +8.38] | 24/0 | 0 |
| `real/QA` | `gold_ridge_all` -> `anchored_pcr6` | +7.86pp [+6.16, +9.57] | 9/0 | 0 |
| `real/math` | `gold_ridge_all` -> `anchored_pcr6` | +6.94pp [+5.63, +8.29] | 15/0 | 0 |
| `synthetic/independent` | `gold_ridge_all` -> `anchored_pcr6` | +2.58pp [+2.17, +3.02] | 40/0 | 0 |
| `synthetic/grouped` | `gold_ridge_all` -> `anchored_pcr6` | +4.49pp [+3.38, +5.76] | 38/2 | 0 |
| `synthetic/sparse_pairs` | `gold_ridge_all` -> `anchored_pcr6` | +4.32pp [+3.37, +5.33] | 39/1 | 0 |
| `synthetic/correlated_weak_block` | `gold_ridge_all` -> `anchored_pcr6` | -14.09pp [-15.36, -12.87] | 0/40 | 39 |

## Protocol notes

- Feature schema: `confidence-orientation-v1`.
- Real bundle validity macro: `0.773527902891`.
- Repetitions: 30 real, 40 synthetic.
- Acquisition is controlled stratification, approximately preserving cell prevalence and forcing both classes. It is optimistic and is not an active label-acquisition result.
- Test labels are read only after every score is frozen.
