# Fixed confidence-orientation validation

Feature schema: `confidence-orientation-v1`.

The fixed arms do not estimate per-cell feature polarity. The LOFO arms are a cross-family calibration diagnostic; the frozen-v1 arms are retrospective and must be confirmed on a new dataset/model family.

## Method scores

| method | arm | cell macro | equal-family macro | mean views |
|---|---|---:|---:|---:|
| `upcr` | `signrho` | 0.7741 | 0.7419 | 28.4 |
| `upcr` | `fixed_all_v1` | 0.7748 | 0.7429 | 28.4 |
| `upcr` | `fixed_stable_v1` | 0.7735 | 0.7414 | 24.8 |
| `upcr` | `lofo_all_diagnostic` | 0.7746 | 0.7422 | 28.4 |
| `upcr` | `lofo_stable_diagnostic` | 0.7735 | 0.7414 | 24.8 |
| `su_pcr` | `signrho` | 0.7668 | 0.7255 | 28.4 |
| `su_pcr` | `fixed_all_v1` | 0.7648 | 0.7242 | 28.4 |
| `su_pcr` | `fixed_stable_v1` | 0.7737 | 0.7414 | 24.8 |
| `su_pcr` | `lofo_all_diagnostic` | 0.7647 | 0.7234 | 28.4 |
| `su_pcr` | `lofo_stable_diagnostic` | 0.7737 | 0.7414 | 24.8 |
| `sdsf` | `signrho` | 0.7104 | 0.6877 | 28.4 |
| `sdsf` | `fixed_all_v1` | 0.7155 | 0.6875 | 28.4 |
| `sdsf` | `fixed_stable_v1` | 0.7266 | 0.7113 | 24.8 |
| `sdsf` | `lofo_all_diagnostic` | 0.7155 | 0.6712 | 28.4 |
| `sdsf` | `lofo_stable_diagnostic` | 0.7266 | 0.7113 | 24.8 |

## Contrasts against per-cell sign(rho)

| method | candidate | cell delta | family delta | W/L/T | worst cell |
|---|---|---:|---:|---:|---:|
| `upcr` | `fixed_all_v1` | +0.07pp | +0.10pp | 10/2/12 | -0.40pp |
| `upcr` | `fixed_stable_v1` | -0.06pp | -0.04pp | 6/10/8 | -0.71pp |
| `upcr` | `lofo_all_diagnostic` | +0.05pp | +0.03pp | 10/3/11 | -0.40pp |
| `upcr` | `lofo_stable_diagnostic` | -0.06pp | -0.04pp | 6/10/8 | -0.71pp |
| `su_pcr` | `fixed_all_v1` | -0.20pp | -0.12pp | 12/10/2 | -3.25pp |
| `su_pcr` | `fixed_stable_v1` | +0.68pp | +1.59pp | 9/15/0 | -4.08pp |
| `su_pcr` | `lofo_all_diagnostic` | -0.22pp | -0.20pp | 10/11/3 | -3.01pp |
| `su_pcr` | `lofo_stable_diagnostic` | +0.68pp | +1.59pp | 9/15/0 | -4.08pp |
| `sdsf` | `fixed_all_v1` | +0.52pp | -0.02pp | 9/13/2 | -18.20pp |
| `sdsf` | `fixed_stable_v1` | +1.62pp | +2.36pp | 9/15/0 | -13.07pp |
| `sdsf` | `lofo_all_diagnostic` | +0.52pp | -1.66pp | 9/12/3 | -18.75pp |
| `sdsf` | `lofo_stable_diagnostic` | +1.62pp | +2.36pp | 9/15/0 | -13.07pp |

## Global-sign check

For fixed-schema scores, the consensus anchor and historical `epr` anchor selected the same sign in **288/288** method/cell/arm comparisons.
