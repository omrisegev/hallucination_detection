# Aggregate causal localization-model online result

CPU-only retrospective replay over **11** cells and
**5** equal-weight dataset families. No inference,
GPU job, Drive mutation, or raw-data mutation was performed.

## Completed-trace performance

| method | cells | macro final AUROC |
|---|---:|---:|
| global_gl_liu_no_length | 11 | 0.788 |
| global_gl_liu_elapsed_length | 11 | 0.792 |
| local_temporal_gl_liu_max | 11 | 0.728 |
| local_dufs_gl_liu_top5 | 11 | 0.738 |
| fused_gl_liu | 11 | 0.782 |
| cusum_max | 11 | 0.766 |
| sw_var_peak | 11 | 0.784 |
| cusum_swvar_equal | 11 | 0.798 |
| iu28_no_length | 11 | 0.764 |
| deepconf_entropy_w32 | 11 | 0.784 |
| deepconf_entropy_w64 | 11 | 0.777 |

## Fixed-budget performance

| budget | method | cells | cells with ≥20 at risk | macro AUROC | Spearman vs final |
|---:|---|---:|---:|---:|---:|
| 16 | global_gl_liu_no_length | 11 | 10 | 0.500 | 0.006 |
| 16 | local_temporal_gl_liu_max | 11 | 10 | 0.513 | 0.141 |
| 16 | local_dufs_gl_liu_top5 | 11 | 10 | 0.518 | 0.171 |
| 16 | fused_gl_liu | 11 | 10 | 0.513 | 0.087 |
| 16 | cusum_max | 11 | 10 | 0.505 | 0.083 |
| 16 | sw_var_peak | 11 | 10 | 0.516 | 0.121 |
| 16 | cusum_swvar_equal | 11 | 10 | 0.516 | 0.098 |
| 16 | iu28_no_length | 11 | 10 | 0.505 | 0.097 |
| 16 | deepconf_entropy_w64 | 11 | 10 | 0.528 | 0.138 |
| 32 | global_gl_liu_no_length | 11 | 10 | 0.578 | 0.260 |
| 32 | local_temporal_gl_liu_max | 11 | 10 | 0.551 | 0.206 |
| 32 | local_dufs_gl_liu_top5 | 11 | 10 | 0.551 | 0.287 |
| 32 | fused_gl_liu | 11 | 10 | 0.578 | 0.295 |
| 32 | cusum_max | 11 | 10 | 0.547 | 0.171 |
| 32 | sw_var_peak | 11 | 10 | 0.573 | 0.242 |
| 32 | cusum_swvar_equal | 11 | 10 | 0.575 | 0.245 |
| 32 | iu28_no_length | 11 | 10 | 0.580 | 0.172 |
| 32 | deepconf_entropy_w64 | 11 | 10 | 0.584 | 0.335 |
| 64 | global_gl_liu_no_length | 11 | 10 | 0.638 | 0.488 |
| 64 | local_temporal_gl_liu_max | 11 | 10 | 0.574 | 0.312 |
| 64 | local_dufs_gl_liu_top5 | 11 | 10 | 0.579 | 0.410 |
| 64 | fused_gl_liu | 11 | 10 | 0.635 | 0.510 |
| 64 | cusum_max | 11 | 10 | 0.582 | 0.275 |
| 64 | sw_var_peak | 11 | 10 | 0.643 | 0.434 |
| 64 | cusum_swvar_equal | 11 | 10 | 0.635 | 0.423 |
| 64 | iu28_no_length | 11 | 10 | 0.648 | 0.417 |
| 64 | deepconf_entropy_w64 | 11 | 10 | 0.616 | 0.490 |
| 128 | global_gl_liu_no_length | 11 | 10 | 0.679 | 0.699 |
| 128 | local_temporal_gl_liu_max | 11 | 10 | 0.618 | 0.423 |
| 128 | local_dufs_gl_liu_top5 | 11 | 10 | 0.638 | 0.545 |
| 128 | fused_gl_liu | 11 | 10 | 0.678 | 0.695 |
| 128 | cusum_max | 11 | 10 | 0.643 | 0.382 |
| 128 | sw_var_peak | 11 | 10 | 0.679 | 0.637 |
| 128 | cusum_swvar_equal | 11 | 10 | 0.681 | 0.609 |
| 128 | iu28_no_length | 11 | 10 | 0.694 | 0.659 |
| 128 | deepconf_entropy_w64 | 11 | 10 | 0.671 | 0.680 |
| 256 | global_gl_liu_no_length | 11 | 7 | 0.643 | 0.793 |
| 256 | local_temporal_gl_liu_max | 11 | 7 | 0.587 | 0.491 |
| 256 | local_dufs_gl_liu_top5 | 11 | 7 | 0.583 | 0.565 |
| 256 | fused_gl_liu | 11 | 7 | 0.637 | 0.765 |
| 256 | cusum_max | 11 | 7 | 0.590 | 0.463 |
| 256 | sw_var_peak | 11 | 7 | 0.663 | 0.762 |
| 256 | cusum_swvar_equal | 11 | 7 | 0.648 | 0.664 |
| 256 | iu28_no_length | 11 | 7 | 0.604 | 0.773 |
| 256 | deepconf_entropy_w64 | 11 | 7 | 0.643 | 0.773 |
| 512 | global_gl_liu_no_length | 7 | 7 | 0.695 | 0.834 |
| 512 | local_temporal_gl_liu_max | 7 | 7 | 0.608 | 0.577 |
| 512 | local_dufs_gl_liu_top5 | 7 | 7 | 0.664 | 0.659 |
| 512 | fused_gl_liu | 7 | 7 | 0.707 | 0.803 |
| 512 | cusum_max | 7 | 7 | 0.655 | 0.524 |
| 512 | sw_var_peak | 7 | 7 | 0.671 | 0.803 |
| 512 | cusum_swvar_equal | 7 | 7 | 0.693 | 0.701 |
| 512 | iu28_no_length | 7 | 7 | 0.644 | 0.817 |
| 512 | deepconf_entropy_w64 | 7 | 7 | 0.692 | 0.836 |

## Equal-family delta versus DeepConf-w64

| endpoint | method | delta AUROC | family-bootstrap 95% interval | family W/T/L |
|---:|---|---:|---:|---:|
| 64 | global_gl_liu_no_length | 0.017 | [-0.022, 0.053] | 4/0/1 |
| 64 | local_temporal_gl_liu_max | -0.019 | [-0.095, 0.066] | 2/0/3 |
| 64 | local_dufs_gl_liu_top5 | -0.016 | [-0.075, 0.043] | 2/0/3 |
| 64 | fused_gl_liu | 0.025 | [-0.017, 0.068] | 4/0/1 |
| 64 | sw_var_peak | 0.022 | [-0.005, 0.052] | 3/0/2 |
| 64 | cusum_swvar_equal | 0.019 | [-0.006, 0.055] | 3/0/2 |
| 128 | global_gl_liu_no_length | 0.003 | [-0.034, 0.040] | 3/0/2 |
| 128 | local_temporal_gl_liu_max | -0.034 | [-0.093, 0.037] | 1/0/4 |
| 128 | local_dufs_gl_liu_top5 | -0.020 | [-0.060, 0.020] | 1/0/4 |
| 128 | fused_gl_liu | 0.012 | [-0.017, 0.041] | 3/0/2 |
| 128 | sw_var_peak | 0.010 | [-0.016, 0.043] | 2/0/3 |
| 128 | cusum_swvar_equal | 0.013 | [-0.011, 0.044] | 2/1/2 |
| 512 | global_gl_liu_no_length | -0.011 | [-0.077, 0.029] | 3/0/1 |
| 512 | local_temporal_gl_liu_max | -0.079 | [-0.103, -0.058] | 0/0/4 |
| 512 | local_dufs_gl_liu_top5 | -0.027 | [-0.055, -0.005] | 1/0/3 |
| 512 | fused_gl_liu | 0.010 | [-0.013, 0.033] | 2/0/2 |
| 512 | sw_var_peak | -0.028 | [-0.094, 0.044] | 1/0/3 |
| 512 | cusum_swvar_equal | -0.009 | [-0.079, 0.061] | 2/0/2 |
| final | global_gl_liu_no_length | 0.005 | [-0.014, 0.020] | 3/0/2 |
| final | local_temporal_gl_liu_max | -0.043 | [-0.071, -0.012] | 1/0/4 |
| final | local_dufs_gl_liu_top5 | -0.034 | [-0.053, -0.014] | 0/0/5 |
| final | fused_gl_liu | 0.003 | [0.000, 0.007] | 4/0/1 |
| final | sw_var_peak | -0.001 | [-0.032, 0.031] | 2/0/3 |
| final | cusum_swvar_equal | 0.012 | [-0.018, 0.041] | 2/0/3 |

## Equal-family delta versus IU28

| endpoint | method | delta AUROC | family-bootstrap 95% interval |
|---:|---|---:|---:|
| 64 | global_gl_liu_no_length | -0.008 | [-0.041, 0.021] |
| 64 | fused_gl_liu | 0.000 | [-0.038, 0.038] |
| 64 | sw_var_peak | -0.002 | [-0.013, 0.008] |
| 64 | cusum_swvar_equal | -0.005 | [-0.027, 0.014] |
| 128 | global_gl_liu_no_length | -0.011 | [-0.021, 0.005] |
| 128 | fused_gl_liu | -0.002 | [-0.028, 0.043] |
| 128 | sw_var_peak | -0.004 | [-0.037, 0.033] |
| 128 | cusum_swvar_equal | -0.001 | [-0.030, 0.036] |
| 512 | global_gl_liu_no_length | 0.045 | [0.012, 0.084] |
| 512 | fused_gl_liu | 0.066 | [0.043, 0.089] |
| 512 | sw_var_peak | 0.028 | [-0.008, 0.072] |
| 512 | cusum_swvar_equal | 0.047 | [0.017, 0.089] |
| final | global_gl_liu_no_length | 0.029 | [0.002, 0.058] |
| final | fused_gl_liu | 0.028 | [0.003, 0.052] |
| final | sw_var_peak | 0.024 | [0.008, 0.039] |
| final | cusum_swvar_equal | 0.036 | [0.019, 0.049] |

## Held-out early declaration

| method | coverage | ever wrong | cells ≤10% ever wrong | selective error |
|---|---:|---:|---:|---:|
| global_gl_liu_no_length | 0.382 | 0.134 | 5/11 | 0.355 |
| local_temporal_gl_liu_max | 0.370 | 0.149 | 4/11 | 0.397 |
| local_dufs_gl_liu_top5 | 0.431 | 0.169 | 5/11 | 0.377 |
| fused_gl_liu | 0.393 | 0.138 | 4/11 | 0.354 |
| cusum_max | 0.378 | 0.148 | 5/11 | 0.380 |
| sw_var_peak | 0.348 | 0.121 | 6/11 | 0.336 |
| cusum_swvar_equal | 0.382 | 0.122 | 5/11 | 0.326 |
| iu28_no_length | 0.366 | 0.137 | 5/11 | 0.364 |
| deepconf_entropy_w64 | 0.435 | 0.160 | 4/11 | 0.345 |

## Interpretation

**Promising parity remains, but the localization head does not produce the
hoped-for early jump.** The causal global GL-LIU detector is competitive with
DeepConf and improves completed-trace scoring. The local temporal/DUFS heads
are substantially weaker as answer-level detectors, and equal-weight fusion
does not materially improve the 64–128 token result over the global head or
IU28. `sw_var_peak` and the fixed CUSUM/sw-var combination are the strongest
mechanism-level findings, including the best completed-trace macro AUROC.

No early equal-family 95% interval is wholly above zero, and the 10% held-out
declaration constraint still transfers inconsistently. This does not close the
comparison; it narrows the next step to a better causal global aggregation or
calibrated dynamic model rather than reusing the localization locator as-is.
