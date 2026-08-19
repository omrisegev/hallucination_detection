# Aggregate existing-cache early/online result

CPU-only retrospective screen over **11** materialized
dataset×model/generator cells. Cells are weighted equally. No inference, GPU
job, Drive mutation, or raw-data mutation was performed.

## Score convergence

Completed-trace ranking first:

| method | cells | macro final AUROC |
|---|---:|---:|
| deepconf_entropy_w32 | 11 | 0.784 |
| deepconf_entropy_w64 | 11 | 0.777 |
| iu28_no_length | 11 | 0.764 |
| iu29_elapsed_length | 11 | 0.766 |
| max_entropy | 11 | 0.755 |
| mean_entropy | 11 | 0.710 |

| budget | method | cells | cells with ≥20 at risk | macro AUROC | macro Spearman vs final | decision agreement |
|---:|---|---:|---:|---:|---:|---:|
| 16 | iu28_no_length | 11 | 10 | 0.505 | 0.097 | 0.535 |
| 16 | iu29_elapsed_length | 11 | 10 | 0.506 | 0.101 | 0.515 |
| 16 | deepconf_entropy_w64 | 11 | 10 | 0.528 | 0.138 | 0.549 |
| 32 | iu28_no_length | 11 | 10 | 0.580 | 0.172 | 0.564 |
| 32 | iu29_elapsed_length | 11 | 10 | 0.577 | 0.169 | 0.540 |
| 32 | deepconf_entropy_w64 | 11 | 10 | 0.584 | 0.335 | 0.575 |
| 64 | iu28_no_length | 11 | 10 | 0.648 | 0.417 | 0.640 |
| 64 | iu29_elapsed_length | 11 | 10 | 0.648 | 0.418 | 0.614 |
| 64 | deepconf_entropy_w64 | 11 | 10 | 0.616 | 0.490 | 0.600 |
| 128 | iu28_no_length | 11 | 10 | 0.694 | 0.659 | 0.739 |
| 128 | iu29_elapsed_length | 11 | 10 | 0.695 | 0.656 | 0.727 |
| 128 | deepconf_entropy_w64 | 11 | 10 | 0.671 | 0.680 | 0.769 |
| 256 | iu28_no_length | 11 | 7 | 0.604 | 0.773 | 0.840 |
| 256 | iu29_elapsed_length | 11 | 7 | 0.604 | 0.767 | 0.835 |
| 256 | deepconf_entropy_w64 | 11 | 7 | 0.643 | 0.773 | 0.861 |
| 512 | iu28_no_length | 7 | 7 | 0.644 | 0.817 | 0.880 |
| 512 | iu29_elapsed_length | 7 | 7 | 0.642 | 0.812 | 0.878 |
| 512 | deepconf_entropy_w64 | 7 | 7 | 0.692 | 0.836 | 0.878 |

The score does converge with generation: correlation with the completed score
and final-decision agreement generally rise by 64–128 tokens. But convergence
is not the same as superiority over a simple same-access control.

## IU28 versus DeepConf entropy proxy

| budget | cells | dataset families | cell-macro delta | equal-family delta | family-bootstrap 95% interval | family W/T/L |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 11 | 5 | -0.023 | -0.030 | [-0.100, 0.030] | 2/0/3 |
| 32 | 11 | 5 | -0.004 | -0.025 | [-0.093, 0.021] | 2/0/3 |
| 64 | 11 | 5 | 0.032 | 0.024 | [-0.005, 0.056] | 3/0/2 |
| 128 | 11 | 5 | 0.023 | 0.014 | [-0.031, 0.058] | 3/0/2 |
| 256 | 11 | 5 | -0.039 | -0.043 | [-0.090, 0.010] | 1/0/4 |
| 512 | 7 | 4 | -0.048 | -0.056 | [-0.102, -0.010] | 0/0/4 |

## Frozen early-declaration transfer

| method | macro coverage | macro ever-wrong | cells ≤10% ever-wrong | selective error |
|---|---:|---:|---:|---:|
| deepconf_entropy_w32 | 0.432 | 0.154 | 6/11 | 0.345 |
| deepconf_entropy_w64 | 0.435 | 0.160 | 4/11 | 0.345 |
| iu28_no_length | 0.366 | 0.137 | 5/11 | 0.364 |
| iu29_elapsed_length | 0.343 | 0.119 | 5/11 | 0.349 |
| max_entropy | 0.329 | 0.129 | 5/11 | 0.386 |
| mean_entropy | 0.424 | 0.168 | 5/11 | 0.365 |

The 10% constraint was imposed only on calibration questions. Failure on a
held-out half is therefore a real transfer failure, not a threshold that may be
retuned after seeing evaluation labels.

## Decision

**FAIL_EXISTING_DATA_PROMOTION_GATE.** IU28 has no budget whose equal-dataset-family paired advantage over the same-access DeepConf entropy proxy has a 95% interval above zero; the calibration-constrained declaration policy also fails the 10% held-out ever-wrong target in multiple cells.

The evidence supports the scientific question — causal scores become more like
their final values over time — but does not currently support promoting the
frozen 28/29-stream maximum-risk adapter as a better online detector. Exact
native-paper reproductions or new GPU inference are therefore not authorized
by this gate. The next CPU-only diagnosis should separate the weak early score
into aggregation (maximum token risk), population fit, and feature-family
components before considering new data collection.
