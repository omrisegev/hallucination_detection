# SUB5 on every benchmark (development data; see README for what is in-sample)

### prmbench / standalone / PRMScore_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 68.01 | 69.49 | 66.24 | 59.39 |
| TELE_N16 | 68.26 | 67.80 | 67.63 | 68.20 |
| MEAN_N16 | 67.14 | 67.39 | 67.06 | 67.45 |
| CHAIN_N16 | 69.10 | 69.95 | 69.02 | 68.31 |
| SUB5 alone | 67.20 | 65.59 | 66.78 | 66.82 |

### prmbench / standalone / PRMScore_sameTPR87.3

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 68.26 | 69.22 | 66.41 | 55.26 |
| TELE_N16 | 66.21 | 65.77 | 65.54 | 66.26 |
| MEAN_N16 | 65.52 | 65.40 | 65.20 | 65.65 |
| CHAIN_N16 | 67.07 | 67.71 | 66.95 | 66.22 |
| SUB5 alone | 65.19 | 63.96 | 64.98 | 64.68 |

### prmbench / standalone / PRMScore_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 65.49 | 68.18 | 65.31 | 54.43 |

### prmbench / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 0.801 | 0.795 | 0.763 | 0.662 |
| TELE_N16 | 0.793 | 0.781 | 0.770 | 0.788 |
| MEAN_N16 | 0.782 | 0.780 | 0.769 | 0.783 |
| CHAIN_N16 | 0.810 | 0.817 | 0.803 | 0.793 |
| SUB5 alone | 0.775 | 0.756 | 0.763 | 0.770 |

### prmbench / standalone / crop_kept_a05

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM (1-reward) | 4.83 | 20.37 | 13.73 | 8.35 |
| TELE_N16 | 18.21 | 16.74 | 16.42 | 18.21 |
| MEAN_N16 | 17.87 | 17.15 | 16.08 | 18.11 |
| CHAIN_N16 | 19.95 | 19.49 | 19.39 | 17.76 |
| SUB5 alone | 17.98 | 17.07 | 17.64 | 18.42 |

### prmbench / standalone / crop_auroc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM (1-reward) | 0.800 | 0.784 | 0.781 | 0.631 |
| TELE_N16 | 0.744 | 0.732 | 0.717 | 0.734 |
| MEAN_N16 | 0.730 | 0.726 | 0.713 | 0.727 |
| CHAIN_N16 | 0.762 | 0.771 | 0.750 | 0.739 |
| SUB5 alone | 0.726 | 0.701 | 0.707 | 0.717 |

### socratic / standalone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 70.09 | 71.72 | 70.58 | 50.38 |
| TELE_N16 | 63.29 | 65.31 | 63.81 | 62.45 |
| MEAN_N16 | 62.84 | 64.25 | 63.34 | 61.40 |
| CHAIN_N16 | 66.88 | 67.39 | 64.80 | 63.26 |
| SUB5 alone | 69.12 | 64.02 | 64.32 | 62.59 |

### socratic / standalone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 68.09 | 71.96 | 63.71 | 48.12 |

### socratic / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.824 | 0.814 | 0.796 | 0.549 |
| PRM | 0.824 | 0.814 | 0.796 | 0.549 |
| TELE_N16 | 0.696 | 0.695 | 0.663 | 0.662 |
| MEAN_N16 | 0.681 | 0.680 | 0.658 | 0.653 |
| CHAIN_N16 | 0.779 | 0.753 | 0.695 | 0.669 |
| SUB5 alone | 0.749 | 0.694 | 0.669 | 0.656 |

### hard2verify / standalone / balanced_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 45.82 | 35.52 | 50.76 | 5.11 |
| TELE_N16 | 47.22 | 43.47 | 39.33 | 40.24 |
| MEAN_N16 | 46.05 | 43.77 | 37.56 | 38.72 |
| CHAIN_N16 | 47.38 | 43.31 | 40.72 | 37.37 |
| SUB5 alone | 47.69 | 46.87 | 44.97 | 43.90 |

### hard2verify / standalone / balanced_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 16.33 | 31.14 | 52.59 | 5.93 |

### hard2verify / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.638 | 0.513 | 0.763 | 0.185 |
| PRM | 0.638 | 0.513 | 0.763 | 0.185 |
| TELE_N16 | 0.540 | 0.560 | 0.489 | 0.528 |
| MEAN_N16 | 0.533 | 0.552 | 0.478 | 0.530 |
| CHAIN_N16 | 0.565 | 0.548 | 0.521 | 0.474 |
| SUB5 alone | 0.564 | 0.555 | 0.524 | 0.542 |

### processbench / backbone / avg_f1_NC

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 72.94 | 78.20 | 52.58 | 29.16 |
| TELE_N16 | 61.89 | 64.63 | 38.46 | 33.92 |
| MEAN_N16 | 61.23 | 59.84 | 40.96 | 33.66 |
| CHAIN_N16 | 67.23 | 68.83 | 42.07 | 33.04 |
| SUB5 alone | 58.86 | 61.51 | 39.66 | 34.62 |
| SUB5 +PRM | 68.50 | 74.65 | 42.43 | 31.21 |

### processbench / backbone / avg_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 72.94 | 78.20 | 52.58 | 29.16 |

### processbench / backbone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.939 | 0.951 | 0.925 | 0.329 |
| PRM | 0.939 | 0.951 | 0.925 | 0.329 |
| TELE_N16 | 0.876 | 0.893 | 0.688 | 0.670 |
| MEAN_N16 | 0.859 | 0.860 | 0.717 | 0.670 |
| CHAIN_N16 | 0.910 | 0.920 | 0.728 | 0.558 |
| SUB5 alone | 0.823 | 0.861 | 0.696 | 0.695 |
| SUB5 +PRM | 0.908 | 0.940 | 0.747 | 0.513 |

### processbench / standalone / avg_f1_NC

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 72.94 | 78.20 | 52.58 | 29.16 |
| TELE_N16 | 54.12 | 48.26 | 36.84 | 34.47 |
| MEAN_N16 | 54.23 | 48.90 | 38.74 | 33.80 |
| CHAIN_N16 | 63.23 | 57.93 | 40.03 | 33.88 |
| SUB5 alone | 50.49 | 45.33 | 35.98 | 32.05 |

### processbench / standalone / avg_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 72.94 | 78.20 | 52.58 | 29.16 |

### processbench / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.939 | 0.951 | 0.925 | 0.329 |
| PRM | 0.939 | 0.951 | 0.925 | 0.329 |
| TELE_N16 | 0.773 | 0.746 | 0.681 | 0.698 |
| MEAN_N16 | 0.771 | 0.754 | 0.704 | 0.706 |
| CHAIN_N16 | 0.851 | 0.840 | 0.725 | 0.609 |
| SUB5 alone | 0.724 | 0.695 | 0.660 | 0.667 |

### crv_arith / backbone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 52.32 | 56.10 | 48.39 | 45.11 |
| TELE_N16 | 49.80 | 48.39 | 40.46 | 45.56 |
| MEAN_N16 | 49.02 | 47.17 | 43.87 | 43.22 |
| CHAIN_N16 | 50.66 | 49.51 | 42.96 | 40.93 |
| SUB5 alone | 50.43 | 47.83 | 44.32 | 42.97 |
| SUB5 +PRM | 51.92 | - | 40.21 | 41.88 |

### crv_arith / backbone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 77.27 | 83.82 | 67.96 | 67.86 |

### crv_arith / backbone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.902 | 0.961 | 0.870 | 0.686 |
| PRM | 0.902 | 0.961 | 0.870 | 0.686 |
| TELE_N16 | 0.740 | 0.744 | 0.277 | 0.345 |
| MEAN_N16 | 0.651 | 0.673 | 0.366 | 0.307 |
| CHAIN_N16 | 0.768 | 0.773 | 0.394 | 0.448 |
| SUB5 alone | 0.730 | 0.746 | 0.339 | 0.396 |
| SUB5 +PRM | 0.855 | - | 0.320 | 0.575 |

### crv_arith / backbone / crop_kept_a05

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 96.87 | 97.02 | 90.32 | 95.08 |
| TELE_N16 | 22.30 | 32.75 | 43.59 | 52.48 |
| MEAN_N16 | 23.49 | 40.45 | 49.51 | 52.50 |
| CHAIN_N16 | 23.96 | 35.60 | 42.64 | 31.20 |
| SUB5 alone | 21.29 | 37.48 | 45.99 | 42.90 |
| SUB5 +PRM | 35.18 | - | 44.85 | 21.67 |

### crv_arith / backbone / crop_auroc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 0.943 | 0.978 | 0.877 | 0.860 |
| TELE_N16 | 0.680 | 0.679 | 0.392 | 0.414 |
| MEAN_N16 | 0.612 | 0.603 | 0.431 | 0.401 |
| CHAIN_N16 | 0.690 | 0.691 | 0.461 | 0.461 |
| SUB5 alone | 0.673 | 0.658 | 0.425 | 0.444 |
| SUB5 +PRM | 0.690 | - | 0.418 | 0.531 |

### crv_arith / standalone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 52.32 | 56.10 | 48.39 | 45.11 |
| TELE_N16 | 47.63 | 46.42 | 42.27 | 46.20 |
| MEAN_N16 | 47.15 | 46.87 | 43.51 | 45.85 |
| CHAIN_N16 | 48.80 | 47.54 | 42.40 | 46.92 |
| SUB5 alone | - | 46.25 | 42.61 | 45.81 |

### crv_arith / standalone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 77.27 | 83.82 | 67.96 | 67.86 |

### crv_arith / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.902 | 0.961 | 0.870 | 0.686 |
| PRM | 0.902 | 0.961 | 0.870 | 0.686 |
| TELE_N16 | 0.683 | 0.538 | 0.289 | 0.479 |
| MEAN_N16 | 0.583 | 0.517 | 0.316 | 0.452 |
| CHAIN_N16 | 0.728 | 0.567 | 0.353 | 0.488 |
| SUB5 alone | - | 0.428 | 0.314 | 0.538 |

### crv_arith / standalone / crop_kept_a05

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 96.87 | 97.02 | 90.32 | 95.08 |
| TELE_N16 | 43.42 | 36.43 | 49.92 | 52.53 |
| MEAN_N16 | 44.75 | 41.66 | 50.44 | 52.18 |
| CHAIN_N16 | 40.15 | 27.57 | 44.72 | 47.65 |
| SUB5 alone | - | 32.03 | 50.21 | 56.86 |

### crv_arith / standalone / crop_auroc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 0.943 | 0.978 | 0.877 | 0.860 |
| TELE_N16 | 0.614 | 0.530 | 0.410 | 0.485 |
| MEAN_N16 | 0.548 | 0.503 | 0.413 | 0.472 |
| CHAIN_N16 | 0.632 | 0.543 | 0.454 | 0.497 |
| SUB5 alone | - | 0.465 | 0.418 | 0.506 |

### crv_gsm8k / backbone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 55.61 | 59.36 | 55.00 | 49.92 |
| TELE_N16 | 49.67 | 49.16 | 47.46 | 48.79 |
| MEAN_N16 | 49.43 | 48.85 | 47.49 | 47.99 |
| CHAIN_N16 | 49.93 | 49.53 | 48.26 | 47.55 |
| SUB5 alone | 48.58 | 48.79 | 48.14 | 47.94 |
| SUB5 +PRM | 49.94 | 51.01 | 49.07 | 50.18 |

### crv_gsm8k / backbone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 72.51 | 77.36 | 70.90 | 62.69 |

### crv_gsm8k / backbone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.743 | 0.760 | 0.804 | 0.648 |
| PRM | 0.743 | 0.760 | 0.804 | 0.648 |
| TELE_N16 | 0.621 | 0.629 | 0.531 | 0.526 |
| MEAN_N16 | 0.606 | 0.622 | 0.537 | 0.549 |
| CHAIN_N16 | 0.646 | 0.665 | 0.572 | 0.607 |
| SUB5 alone | 0.612 | 0.642 | 0.548 | 0.558 |
| SUB5 +PRM | 0.640 | 0.687 | 0.611 | 0.644 |

### crv_gsm8k / backbone / crop_kept_a05

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 80.87 | 81.40 | 63.04 | 62.29 |
| TELE_N16 | 27.77 | 28.18 | 30.52 | 33.24 |
| MEAN_N16 | 29.90 | 30.94 | 31.98 | 36.67 |
| CHAIN_N16 | 32.51 | 30.90 | 33.33 | 25.41 |
| SUB5 alone | 29.99 | 28.43 | 32.26 | 41.07 |
| SUB5 +PRM | 32.87 | 29.90 | 33.60 | 19.98 |

### crv_gsm8k / backbone / crop_auroc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 0.849 | 0.879 | 0.847 | 0.797 |
| TELE_N16 | 0.582 | 0.593 | 0.515 | 0.517 |
| MEAN_N16 | 0.570 | 0.589 | 0.516 | 0.525 |
| CHAIN_N16 | 0.600 | 0.615 | 0.542 | 0.555 |
| SUB5 alone | 0.568 | 0.598 | 0.519 | 0.535 |
| SUB5 +PRM | 0.593 | 0.638 | 0.569 | 0.578 |

### crv_gsm8k / standalone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 55.61 | 59.36 | 55.00 | 49.92 |
| TELE_N16 | 48.41 | 47.30 | 47.46 | 47.74 |
| MEAN_N16 | 48.40 | 47.65 | 47.45 | 48.10 |
| CHAIN_N16 | 49.15 | 48.06 | 47.99 | 47.79 |
| SUB5 alone | 48.66 | 47.58 | 46.99 | 47.70 |

### crv_gsm8k / standalone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 72.51 | 77.36 | 70.90 | 62.69 |

### crv_gsm8k / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.743 | 0.760 | 0.804 | 0.648 |
| PRM | 0.743 | 0.760 | 0.804 | 0.648 |
| TELE_N16 | 0.578 | 0.555 | 0.515 | 0.550 |
| MEAN_N16 | 0.581 | 0.547 | 0.518 | 0.549 |
| CHAIN_N16 | 0.607 | 0.569 | 0.596 | 0.555 |
| SUB5 alone | 0.575 | 0.548 | 0.531 | 0.567 |

### crv_gsm8k / standalone / crop_kept_a05

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 80.87 | 81.40 | 63.04 | 62.29 |
| TELE_N16 | 27.63 | 33.19 | 32.20 | 33.28 |
| MEAN_N16 | 28.81 | 31.82 | 32.42 | 32.10 |
| CHAIN_N16 | 29.94 | 34.61 | 36.17 | 36.40 |
| SUB5 alone | 31.82 | 35.49 | 32.35 | 37.42 |

### crv_gsm8k / standalone / crop_auroc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 0.849 | 0.879 | 0.847 | 0.797 |
| TELE_N16 | 0.554 | 0.534 | 0.505 | 0.520 |
| MEAN_N16 | 0.554 | 0.533 | 0.507 | 0.521 |
| CHAIN_N16 | 0.575 | 0.541 | 0.553 | 0.522 |
| SUB5 alone | 0.548 | 0.531 | 0.513 | 0.530 |

### prm800k / backbone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 65.30 | 66.73 | 64.12 | 48.46 |
| TELE_N16 | 56.53 | 54.57 | 55.77 | 52.84 |
| MEAN_N16 | 55.40 | 55.30 | 53.67 | 51.86 |
| CHAIN_N16 | 55.92 | 56.51 | 60.78 | 49.67 |
| SUB5 alone | 53.65 | 53.00 | 51.28 | 50.39 |
| SUB5 +PRM | 64.11 | 63.05 | 66.78 | 48.57 |

### prm800k / backbone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 75.18 | 80.44 | 61.99 | 54.06 |

### prm800k / backbone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.929 | 0.926 | 0.942 | 0.382 |
| PRM | 0.929 | 0.926 | 0.942 | 0.382 |
| TELE_N16 | 0.719 | 0.838 | 0.623 | 0.644 |
| MEAN_N16 | 0.794 | 0.820 | 0.703 | 0.640 |
| CHAIN_N16 | 0.835 | 0.885 | 0.760 | 0.490 |
| SUB5 alone | 0.757 | 0.768 | 0.615 | 0.623 |
| SUB5 +PRM | 0.920 | 0.906 | 0.932 | 0.391 |

### prm800k / backbone / crop_kept_a05

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 69.11 | 75.42 | 61.77 | 37.30 |
| TELE_N16 | 29.14 | 34.53 | 20.25 | 13.47 |
| MEAN_N16 | 30.64 | 32.13 | 22.02 | 18.49 |
| CHAIN_N16 | 34.97 | 39.35 | 30.79 | 7.75 |
| SUB5 alone | 33.00 | 31.70 | 20.02 | 18.70 |
| SUB5 +PRM | 51.89 | 41.41 | 70.68 | 7.83 |

### prm800k / backbone / crop_auroc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 0.930 | 0.950 | 0.878 | 0.748 |
| TELE_N16 | 0.680 | 0.790 | 0.640 | 0.558 |
| MEAN_N16 | 0.754 | 0.772 | 0.675 | 0.600 |
| CHAIN_N16 | 0.783 | 0.835 | 0.738 | 0.481 |
| SUB5 alone | 0.729 | 0.732 | 0.610 | 0.596 |
| SUB5 +PRM | 0.865 | 0.852 | 0.859 | 0.424 |

### prm800k / standalone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 65.30 | 66.73 | 64.12 | 48.46 |
| TELE_N16 | 50.37 | 50.99 | 51.47 | 53.38 |
| MEAN_N16 | 52.24 | 51.56 | 51.38 | 53.21 |
| CHAIN_N16 | 48.15 | 52.02 | 55.74 | 53.24 |
| SUB5 alone | 49.14 | 48.73 | 48.79 | 51.52 |

### prm800k / standalone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 75.18 | 80.44 | 61.99 | 54.06 |

### prm800k / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.929 | 0.926 | 0.942 | 0.382 |
| PRM | 0.929 | 0.926 | 0.942 | 0.382 |
| TELE_N16 | 0.461 | 0.433 | 0.446 | 0.503 |
| MEAN_N16 | 0.679 | 0.649 | 0.625 | 0.684 |
| CHAIN_N16 | 0.611 | 0.525 | 0.587 | 0.517 |
| SUB5 alone | 0.567 | 0.532 | 0.541 | 0.663 |

### prm800k / standalone / crop_kept_a05

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 69.11 | 75.42 | 61.77 | 37.30 |
| TELE_N16 | 14.71 | 17.94 | 17.29 | 22.81 |
| MEAN_N16 | 20.10 | 19.89 | 17.06 | 24.22 |
| CHAIN_N16 | 32.91 | 18.41 | 22.26 | 19.24 |
| SUB5 alone | 18.29 | 16.94 | 14.87 | 24.29 |

### prm800k / standalone / crop_auroc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 0.930 | 0.950 | 0.878 | 0.748 |
| TELE_N16 | 0.500 | 0.497 | 0.525 | 0.553 |
| MEAN_N16 | 0.654 | 0.634 | 0.614 | 0.661 |
| CHAIN_N16 | 0.560 | 0.564 | 0.654 | 0.543 |
| SUB5 alone | 0.565 | 0.538 | 0.549 | 0.640 |

### deltabench / backbone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 58.60 | 57.78 | 57.74 | 50.94 |
| TELE_N16 | 56.94 | 57.12 | 54.58 | 54.75 |
| MEAN_N16 | 56.84 | 56.81 | 55.21 | 54.15 |
| CHAIN_N16 | 58.22 | 58.92 | 48.71 | 53.83 |
| SUB5 alone | 55.92 | 54.64 | 55.34 | 55.70 |
| SUB5 +PRM | 57.12 | 56.12 | 50.72 | 51.23 |

### deltabench / backbone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 56.25 | 57.05 | 53.87 | 49.88 |

### deltabench / backbone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.656 | 0.630 | 0.648 | 0.536 |
| PRM | 0.656 | 0.630 | 0.648 | 0.536 |
| TELE_N16 | 0.655 | 0.650 | 0.591 | 0.584 |
| MEAN_N16 | 0.658 | 0.637 | 0.608 | 0.589 |
| CHAIN_N16 | 0.671 | 0.665 | 0.395 | 0.582 |
| SUB5 alone | 0.626 | 0.591 | 0.600 | 0.602 |
| SUB5 +PRM | 0.650 | 0.612 | 0.438 | 0.537 |

### deltabench / standalone / macro_f1_GAP

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM | 58.60 | 57.78 | 57.74 | 50.94 |
| TELE_N16 | 57.91 | 55.51 | 54.13 | 54.07 |
| MEAN_N16 | 57.31 | 55.56 | 54.90 | 54.76 |
| CHAIN_N16 | 59.59 | 58.22 | 54.84 | 53.63 |
| SUB5 alone | 53.85 | 54.79 | 54.70 | 54.75 |

### deltabench / standalone / macro_f1_native

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 56.25 | 57.05 | 53.87 | 49.88 |

### deltabench / standalone / within_auc

| arm | qwen_prm_7b | qwen_prm_72b | pure_prm | rlhflow_prm |
|---|---|---|---|---|
| PRM native | 0.656 | 0.630 | 0.648 | 0.536 |
| PRM | 0.656 | 0.630 | 0.648 | 0.536 |
| TELE_N16 | 0.666 | 0.619 | 0.590 | 0.601 |
| MEAN_N16 | 0.665 | 0.620 | 0.598 | 0.609 |
| CHAIN_N16 | 0.674 | 0.637 | 0.600 | 0.597 |
| SUB5 alone | 0.587 | 0.587 | 0.602 | 0.599 |
