# Leverage-Balanced Contribution-Subspace IU v1

**Status:** retrospective mechanism audit; label-free fit, but the formula was discovered after these development labels had been inspected.

## Main result

The frozen leverage-balanced score changed cell-macro AUROC by **+0.569pp** and equal-family AUROC by **+0.633pp** ([+0.314, +0.951]pp), with 19W/4L and worst-cell -0.606pp.

This is evidence that the fusion-internal leverage mechanism is worth an external-family confirmation. It is not prospective confirmation and must not be reported as one.

Against the independently frozen mixed-v2 DUFS-LIU incumbent on the same eligible cells, the descriptive equal-family contrast is +0.565pp ([+0.246, +0.863]pp).

## Methods

| method | cell AUROC | cell delta | equal-family delta [95%] | W/L | worst |
|---|---:|---:|---:|---:|---:|
| `leverage_balanced` | 0.7748 | +0.569pp | +0.633 [+0.314, +0.951] | 19/4 | -0.606pp |
| `dufs_liu` | 0.7695 | +0.047pp | +0.069 [-0.034, +0.185] | 17/6 | -0.247pp |
| `uniform` | 0.7727 | +0.367pp | +0.543 [+0.242, +0.831] | 15/8 | -0.803pp |
| `cardinality` | 0.7737 | +0.467pp | +0.442 [+0.204, +0.679] | 17/6 | -0.595pp |
| `reverse` | 0.7553 | -1.379pp | -1.222 [-1.462, -0.959] | 0/23 | -3.594pp |
| `permuted_mean` | 0.7648 | -0.432pp | -0.340 [-0.455, -0.232] | 0/23 | -1.615pp |

## Mechanism contrasts

| contrast | equal-family delta [95%] |
|---|---:|
| `leverage_balanced - dufs_liu` | +0.565 [+0.246, +0.863]pp |
| `leverage_balanced - uniform` | +0.090 [-0.143, +0.329]pp |
| `leverage_balanced - cardinality` | +0.191 [-0.041, +0.434]pp |
| `leverage_balanced - reverse` | +1.855 [+1.310, +2.362]pp |
| `leverage_balanced - permuted_mean` | +0.974 [+0.660, +1.293]pp |

## Continuation gates

- **PASS — positive equal-family interval:** low=+0.314pp
- **PASS — cell wins:** 19/23 wins
- **PASS — tail safety:** worst=-0.606pp
- **PASS — teacher recovery:** 87.9% of frozen teacher gain
- **PASS — specificity beyond simple balancing:** vs uniform +0.090pp; vs cardinality +0.191pp
- **PASS — orientation falsifier:** primary-reverse=+1.855pp
- **PASS — family correspondence:** primary-permuted=+0.974pp
- **PASS — numerical invariants:** scale error=2.78e-17; |cov|=5.74e-16; weight error=1.78e-15; IU mismatch=0.00e+00

## Audit boundary

Cells excluded by the pre-existing positive-count rule: `spilled_triviaqa_llama8b`.
Maximum reconstruction error / correction-scale error / absolute baseline-correction covariance / effective-weight reconstruction error / frozen-IU AUROC mismatch: 8.882e-16 / 2.776e-17 / 5.744e-16 / 1.776e-15 / 0.000e+00.

The leverage-specific advantages over uniform and cardinality balancing have intervals that cross zero. The positive primary therefore supports contribution-family balancing more strongly than it uniquely identifies L1 leverage as the only mechanism.

The next admissible claim requires an unchanged run on a new intrinsic-detection dataset or model family whose labels were not used during discovery.
