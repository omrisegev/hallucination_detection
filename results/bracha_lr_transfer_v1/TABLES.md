# Bracha LR transfer: one table per benchmark

Main metric under the stated rule for each PRM, then within-answer AUC for the 4 PRMs. Below each table: pooled step AUROC and CROP kept %, and the paired bootstrap of LR (overlap removed) minus AUTO. Summary in SUMMARY.md, protocol in PROTOCOL.json.

## PRMBench

| arm | rule | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B | within-AUC (4 PRMs) |
|---|---|---|---|---|---|---|
| PRM, native threshold | native | 65.5 | 68.2 | 65.3 | 54.4 | — / — / — / — |
| PRM, GAP rule | GAP | 68.0 | 69.5 | 66.2 | 59.4 | 0.801 / 0.795 / 0.763 / 0.662 |
| LR on the PRM only (supervised, PRM800K) | GAP | 66.4 | 64.5 | 64.7 | 45.6 | 0.801 / 0.795 / 0.763 / 0.338 |
| LR on the PRM only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 64.0 | 65.1 | 62.8 | 39.7 |  |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | GAP | 69.9 | 70.2 | 69.1 | 65.9 | 0.824 / 0.828 / 0.799 / 0.765 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | GAP | 70.4 | 71.1 | 69.5 | 66.3 | 0.838 / 0.843 / 0.816 / 0.775 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | GAP | 67.1 | 68.0 | 64.9 | 56.4 | 0.791 / 0.806 / 0.757 / 0.615 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed), probability > 0.5 | THR05 (label-trained) | 64.7 | 66.7 | 63.4 | 54.9 |  |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | GAP | 67.1 | 68.2 | 65.4 | 57.4 | 0.792 / 0.807 / 0.763 / 0.631 |
| LR: PRM + backbone telemetry (supervised, all PRM800K), probability > 0.5 | THR05 (label-trained) | 64.9 | 66.8 | 63.7 | 56.1 |  |
| TELE16 standalone, L-SML (label-free) | GAP | 68.3 | 67.8 | 67.6 | 68.2 | 0.793 / 0.781 / 0.770 / 0.788 |
| TELE16 standalone, plain average | GAP | 67.1 | 67.4 | 67.1 | 67.5 | 0.782 / 0.780 / 0.769 / 0.783 |
| LR: standalone telemetry only (supervised, PRM800K) | GAP | 62.7 | 62.6 | 64.8 | 65.3 | 0.713 / 0.720 / 0.741 / 0.757 |
| LR: standalone telemetry only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 59.5 | 59.9 | 62.2 | 61.9 |  |

| arm | metric | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|---|
| PRM, GAP rule | pooled step AUROC | 0.798 | 0.785 | 0.778 | 0.626 |
| PRM, GAP rule | CROP kept % (α 0.05) | 4.8 | 20.4 | 13.7 | 8.3 |
| LR on the PRM only (supervised, PRM800K) | pooled step AUROC | 0.759 | 0.759 | 0.731 | 0.368 |
| LR on the PRM only (supervised, PRM800K) | CROP kept % (α 0.05) | 20.0 | 17.5 | 18.3 | 7.8 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.789 | 0.789 | 0.760 | 0.721 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 21.6 | 20.5 | 18.9 | 15.1 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.801 | 0.803 | 0.775 | 0.726 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 23.3 | 21.9 | 20.5 | 16.2 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | pooled step AUROC | 0.777 | 0.793 | 0.728 | 0.612 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | CROP kept % (α 0.05) | 20.2 | 21.4 | 19.4 | 14.4 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | pooled step AUROC | 0.777 | 0.794 | 0.734 | 0.626 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | CROP kept % (α 0.05) | 20.4 | 21.7 | 19.4 | 14.9 |
| TELE16 standalone, L-SML (label-free) | pooled step AUROC | 0.756 | 0.744 | 0.730 | 0.748 |
| TELE16 standalone, L-SML (label-free) | CROP kept % (α 0.05) | 18.2 | 16.7 | 16.4 | 18.2 |
| TELE16 standalone, plain average | pooled step AUROC | 0.742 | 0.738 | 0.726 | 0.741 |
| TELE16 standalone, plain average | CROP kept % (α 0.05) | 17.9 | 17.2 | 16.1 | 18.1 |
| LR: standalone telemetry only (supervised, PRM800K) | pooled step AUROC | 0.692 | 0.697 | 0.712 | 0.723 |
| LR: standalone telemetry only (supervised, PRM800K) | CROP kept % (α 0.05) | 14.6 | 15.3 | 16.3 | 17.4 |

LR (PRM + telemetry, overlap removed) minus AUTO, main metric, paired bootstrap 95% CI: Qwen-PRM-7B -2.74 [-3.28, -2.17]; Qwen-PRM-72B -2.17 [-2.76, -1.57]; PURE-PRM-7B -4.18 [-4.73, -3.56]; RLHFlow-PRM-8B -9.57 [-10.28, -8.82] (resampling source groups).

## Socratic-PRMBench

| arm | rule | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B | within-AUC (4 PRMs) |
|---|---|---|---|---|---|---|
| PRM, native threshold | native | 68.1 | 72.0 | 63.7 | 48.1 | 0.824 / 0.814 / 0.796 / 0.549 |
| PRM, GAP rule | GAP | 70.1 | 71.7 | 70.6 | 50.4 | 0.824 / 0.814 / 0.796 / 0.549 |
| LR on the PRM only (supervised, PRM800K) | GAP | 69.5 | 68.2 | 70.0 | 46.7 | 0.824 / 0.814 / 0.796 / 0.451 |
| LR on the PRM only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 71.0 | 70.1 | 69.8 | 46.6 |  |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | GAP | 65.5 | 60.4 | 63.5 | 57.7 | 0.722 / 0.661 / 0.660 / 0.595 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | GAP | 66.9 | 57.4 | 65.0 | 57.1 | 0.776 / 0.613 / 0.698 / 0.590 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | GAP | 71.1 | 67.4 | 66.4 | 54.1 | 0.827 / 0.811 / 0.769 / 0.618 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed), probability > 0.5 | THR05 (label-trained) | 71.2 | 66.8 | 63.0 | 52.9 |  |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | GAP | 71.0 | 67.4 | 66.8 | 55.1 | 0.828 / 0.811 / 0.774 / 0.633 |
| LR: PRM + backbone telemetry (supervised, all PRM800K), probability > 0.5 | THR05 (label-trained) | 70.9 | 66.5 | 63.7 | 54.5 |  |
| TELE16 standalone, L-SML (label-free) | GAP | 63.3 | 65.3 | 63.8 | 62.5 | 0.696 / 0.695 / 0.663 / 0.662 |
| TELE16 standalone, plain average | GAP | 62.8 | 64.2 | 63.3 | 61.4 | 0.681 / 0.680 / 0.658 / 0.653 |
| LR: standalone telemetry only (supervised, PRM800K) | GAP | 57.7 | 57.5 | 53.2 | 58.5 | 0.588 / 0.624 / 0.554 / 0.626 |
| LR: standalone telemetry only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 57.7 | 58.1 | 53.5 | 59.6 |  |

| arm | metric | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|---|
| PRM, GAP rule | pooled step AUROC | 0.820 | 0.822 | 0.738 | 0.579 |
| LR on the PRM only (supervised, PRM800K) | pooled step AUROC | 0.789 | 0.773 | 0.791 | 0.462 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.703 | 0.645 | 0.640 | 0.580 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.748 | 0.606 | 0.678 | 0.578 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | pooled step AUROC | 0.814 | 0.785 | 0.751 | 0.606 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | pooled step AUROC | 0.814 | 0.787 | 0.757 | 0.620 |
| TELE16 standalone, L-SML (label-free) | pooled step AUROC | 0.679 | 0.678 | 0.643 | 0.642 |
| TELE16 standalone, plain average | pooled step AUROC | 0.663 | 0.663 | 0.638 | 0.632 |
| LR: standalone telemetry only (supervised, PRM800K) | pooled step AUROC | 0.569 | 0.613 | 0.547 | 0.620 |

LR (PRM + telemetry, overlap removed) minus AUTO, main metric, paired bootstrap 95% CI: Qwen-PRM-7B +5.63 [+4.93, +6.40]; Qwen-PRM-72B +6.97 [+6.28, +7.74]; PURE-PRM-7B +2.99 [+2.11, +3.85]; RLHFlow-PRM-8B -3.60 [-4.45, -2.76] (resampling answers).

## Hard2Verify (177 answers)

| arm | rule | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B | within-AUC (4 PRMs) |
|---|---|---|---|---|---|---|
| PRM, native threshold | native | 16.3 | 31.1 | 52.6 | 5.9 | 0.638 / 0.513 / 0.763 / 0.185 |
| PRM, GAP rule | GAP | 45.8 | 35.5 | 50.8 | 5.1 | 0.638 / 0.513 / 0.763 / 0.185 |
| LR on the PRM only (supervised, PRM800K) | GAP | 45.8 | 35.5 | 50.8 | 47.7 | 0.638 / 0.513 / 0.763 / 0.815 |
| LR on the PRM only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 48.1 | 42.0 | 55.2 | 64.5 |  |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | GAP | 45.6 | 46.0 | 38.7 | 35.5 | 0.570 / 0.563 / 0.471 / 0.486 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | GAP | 44.3 | 43.4 | 40.0 | 33.1 | 0.584 / 0.548 / 0.486 / 0.450 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | GAP | 49.9 | 40.1 | 48.3 | 42.3 | 0.736 / 0.584 / 0.749 / 0.620 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed), probability > 0.5 | THR05 (label-trained) | 48.9 | 42.0 | 48.1 | 40.5 |  |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | GAP | 48.7 | 40.4 | 47.2 | 40.4 | 0.733 / 0.584 / 0.759 / 0.591 |
| LR: PRM + backbone telemetry (supervised, all PRM800K), probability > 0.5 | THR05 (label-trained) | 46.8 | 42.3 | 47.0 | 37.8 |  |
| TELE16 standalone, L-SML (label-free) | GAP | 47.2 | 43.5 | 39.3 | 40.2 | 0.540 / 0.560 / 0.489 / 0.528 |
| TELE16 standalone, plain average | GAP | 46.0 | 43.8 | 37.6 | 38.7 | 0.533 / 0.552 / 0.478 / 0.530 |
| LR: standalone telemetry only (supervised, PRM800K) | GAP | 38.1 | 39.0 | 36.7 | 38.4 | 0.617 / 0.577 / 0.615 / 0.584 |
| LR: standalone telemetry only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 43.5 | 41.3 | 44.1 | 46.0 |  |

| arm | metric | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|---|
| PRM, GAP rule | pooled step AUROC | 0.634 | 0.581 | 0.682 | 0.347 |
| LR on the PRM only (supervised, PRM800K) | pooled step AUROC | 0.601 | 0.524 | 0.683 | 0.703 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.550 | 0.545 | 0.485 | 0.493 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.557 | 0.536 | 0.497 | 0.471 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | pooled step AUROC | 0.667 | 0.568 | 0.660 | 0.559 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | pooled step AUROC | 0.665 | 0.567 | 0.663 | 0.535 |
| TELE16 standalone, L-SML (label-free) | pooled step AUROC | 0.543 | 0.548 | 0.499 | 0.524 |
| TELE16 standalone, plain average | pooled step AUROC | 0.537 | 0.546 | 0.492 | 0.525 |
| LR: standalone telemetry only (supervised, PRM800K) | pooled step AUROC | 0.554 | 0.548 | 0.550 | 0.548 |

LR (PRM + telemetry, overlap removed) minus AUTO, main metric, paired bootstrap 95% CI: Qwen-PRM-7B +4.28 [-0.97, +9.83]; Qwen-PRM-72B -5.95 [-10.08, -1.81]; PURE-PRM-7B +9.60 [+3.29, +15.86]; RLHFlow-PRM-8B +6.81 [+1.15, +12.22] (resampling answers).

## ProcessBench

| arm | rule | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B | within-AUC (4 PRMs) |
|---|---|---|---|---|---|---|
| PRM, native threshold | native | 72.9 | 78.2 | 52.6 | 29.2 | 0.939 / 0.951 / 0.925 / 0.329 |
| PRM, GAP rule | NC | 72.9 | 78.2 | 52.6 | 29.2 | 0.939 / 0.951 / 0.925 / 0.329 |
| LR on the PRM only (supervised, PRM800K) | NC | 72.9 | 78.2 | 52.6 | 10.5 | 0.939 / 0.951 / 0.925 / 0.671 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | NC | 63.2 | 67.3 | 37.7 | 33.2 | 0.889 / 0.914 / 0.673 / 0.594 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | NC | 67.2 | 68.8 | 42.1 | 33.0 | 0.910 / 0.920 / 0.728 / 0.558 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | NC | 60.4 | 73.1 | 51.3 | 30.8 | 0.970 / 0.969 / 0.933 / 0.879 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | NC | 60.8 | 72.7 | 51.4 | 31.9 | 0.969 / 0.969 / 0.934 / 0.885 |
| TELE16 standalone, L-SML (label-free) | NC | 54.1 | 48.3 | 36.8 | 34.5 | 0.773 / 0.746 / 0.681 / 0.698 |
| TELE16 standalone, plain average | NC | 54.2 | 48.9 | 38.7 | 33.8 | 0.771 / 0.754 / 0.704 / 0.706 |
| LR: standalone telemetry only (supervised, PRM800K) | NC | 50.5 | 50.0 | 31.2 | 28.9 | 0.823 / 0.812 / 0.682 / 0.694 |

| arm | metric | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|---|
| PRM, GAP rule | pooled step AUROC | 0.954 | 0.970 | 0.879 | 0.691 |
| PRM, GAP rule | CROP kept % (α 0.05) | 44.2 | 49.6 | 28.1 | 20.8 |
| LR on the PRM only (supervised, PRM800K) | pooled step AUROC | 0.851 | 0.877 | 0.761 | 0.431 |
| LR on the PRM only (supervised, PRM800K) | CROP kept % (α 0.05) | 22.7 | 23.3 | 16.4 | 8.8 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.788 | 0.805 | 0.713 | 0.681 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 13.8 | 14.4 | 11.2 | 9.5 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.818 | 0.831 | 0.743 | 0.664 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 15.1 | 15.7 | 12.6 | 10.2 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | pooled step AUROC | 0.781 | 0.839 | 0.753 | 0.685 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | CROP kept % (α 0.05) | 17.1 | 18.4 | 16.9 | 12.5 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | pooled step AUROC | 0.783 | 0.838 | 0.758 | 0.694 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | CROP kept % (α 0.05) | 17.0 | 18.2 | 17.0 | 12.6 |
| TELE16 standalone, L-SML (label-free) | pooled step AUROC | 0.733 | 0.739 | 0.713 | 0.715 |
| TELE16 standalone, L-SML (label-free) | CROP kept % (α 0.05) | 10.8 | 11.1 | 10.5 | 10.8 |
| TELE16 standalone, plain average | pooled step AUROC | 0.734 | 0.744 | 0.724 | 0.715 |
| TELE16 standalone, plain average | CROP kept % (α 0.05) | 11.1 | 11.8 | 11.1 | 10.8 |
| LR: standalone telemetry only (supervised, PRM800K) | pooled step AUROC | 0.683 | 0.680 | 0.576 | 0.613 |
| LR: standalone telemetry only (supervised, PRM800K) | CROP kept % (α 0.05) | 10.3 | 11.7 | 9.7 | 10.2 |

## CRV-Arithmetic

| arm | rule | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B | within-AUC (4 PRMs) |
|---|---|---|---|---|---|---|
| PRM, native threshold | native | 77.3 | 83.8 | 68.0 | 67.9 | 0.902 / 0.961 / 0.870 / 0.686 |
| PRM, GAP rule | GAP | 52.3 | 56.1 | 48.4 | 45.1 | 0.902 / 0.961 / 0.870 / 0.686 |
| LR on the PRM only (supervised, PRM800K) | GAP | 44.0 | 45.1 | 45.3 | 39.1 | 0.902 / 0.961 / 0.870 / 0.314 |
| LR on the PRM only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 44.4 | 44.7 | 47.3 | 36.3 |  |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | GAP | 50.2 | 49.8 | 40.2 | 45.2 | 0.756 / 0.769 / 0.270 / 0.350 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | GAP | 50.7 | 49.5 | 43.0 | 40.9 | 0.768 / 0.773 / 0.394 / 0.448 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | GAP | 50.3 | 49.6 | 44.6 | 43.9 | 0.951 / 0.960 / 0.891 / 0.713 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed), probability > 0.5 | THR05 (label-trained) | 53.5 | 54.5 | 49.2 | 45.7 |  |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | GAP | 50.2 | 49.6 | 44.5 | 44.1 | 0.951 / 0.962 / 0.898 / 0.759 |
| LR: PRM + backbone telemetry (supervised, all PRM800K), probability > 0.5 | THR05 (label-trained) | 53.6 | 54.6 | 49.1 | 45.7 |  |
| TELE16 standalone, L-SML (label-free) | GAP | 47.6 | 46.4 | 42.3 | 46.2 | 0.683 / 0.538 / 0.289 / 0.479 |
| TELE16 standalone, plain average | GAP | 47.1 | 46.9 | 43.5 | 45.8 | 0.583 / 0.517 / 0.316 / 0.452 |
| LR: standalone telemetry only (supervised, PRM800K) | GAP | 49.1 | 47.4 | 47.2 | 45.4 | 0.878 / 0.873 / 0.830 / 0.530 |
| LR: standalone telemetry only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 47.4 | 46.2 | 47.1 | 44.4 |  |

| arm | metric | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|---|
| PRM, GAP rule | pooled step AUROC | 0.942 | 0.979 | 0.864 | 0.857 |
| PRM, GAP rule | CROP kept % (α 0.05) | 96.9 | 97.0 | 90.3 | 95.1 |
| LR on the PRM only (supervised, PRM800K) | pooled step AUROC | 0.704 | 0.747 | 0.709 | 0.402 |
| LR on the PRM only (supervised, PRM800K) | CROP kept % (α 0.05) | 57.2 | 76.3 | 66.3 | 86.2 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.686 | 0.700 | 0.381 | 0.423 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 22.8 | 32.0 | 44.3 | 50.6 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.690 | 0.693 | 0.459 | 0.461 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 24.0 | 35.6 | 42.6 | 31.2 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | pooled step AUROC | 0.826 | 0.881 | 0.737 | 0.654 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | CROP kept % (α 0.05) | 52.7 | 87.0 | 83.2 | 84.2 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | pooled step AUROC | 0.829 | 0.882 | 0.741 | 0.692 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | CROP kept % (α 0.05) | 51.7 | 86.5 | 81.6 | 85.1 |
| TELE16 standalone, L-SML (label-free) | pooled step AUROC | 0.609 | 0.531 | 0.407 | 0.480 |
| TELE16 standalone, L-SML (label-free) | CROP kept % (α 0.05) | 43.4 | 36.4 | 49.9 | 52.5 |
| TELE16 standalone, plain average | pooled step AUROC | 0.541 | 0.502 | 0.407 | 0.466 |
| TELE16 standalone, plain average | CROP kept % (α 0.05) | 44.8 | 41.7 | 50.4 | 52.2 |
| LR: standalone telemetry only (supervised, PRM800K) | pooled step AUROC | 0.741 | 0.707 | 0.697 | 0.514 |
| LR: standalone telemetry only (supervised, PRM800K) | CROP kept % (α 0.05) | 59.3 | 73.1 | 63.8 | 56.9 |

LR (PRM + telemetry, overlap removed) minus AUTO, main metric, paired bootstrap 95% CI: Qwen-PRM-7B +0.04 [-0.96, +1.00]; Qwen-PRM-72B -0.16 [-1.23, +0.94]; PURE-PRM-7B +4.40 [+3.33, +5.48]; RLHFlow-PRM-8B -1.32 [-2.07, -0.54] (resampling answers).

## CRV-GSM8K

| arm | rule | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B | within-AUC (4 PRMs) |
|---|---|---|---|---|---|---|
| PRM, native threshold | native | 72.5 | 77.4 | 70.9 | 62.7 | 0.743 / 0.760 / 0.804 / 0.648 |
| PRM, GAP rule | GAP | 55.6 | 59.4 | 55.0 | 49.9 | 0.743 / 0.760 / 0.804 / 0.648 |
| LR on the PRM only (supervised, PRM800K) | GAP | 51.2 | 53.6 | 52.0 | 43.0 | 0.743 / 0.760 / 0.804 / 0.352 |
| LR on the PRM only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 52.3 | 54.6 | 53.2 | 38.5 |  |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | GAP | 50.0 | 49.1 | 47.5 | 48.0 | 0.621 / 0.621 / 0.548 / 0.532 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | GAP | 49.9 | 49.5 | 48.3 | 47.6 | 0.646 / 0.665 / 0.572 / 0.607 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | GAP | 51.6 | 54.2 | 51.2 | 46.7 | 0.742 / 0.785 / 0.800 / 0.622 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed), probability > 0.5 | THR05 (label-trained) | 57.1 | 60.8 | 54.0 | 46.2 |  |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | GAP | 52.3 | 53.9 | 51.2 | 47.2 | 0.745 / 0.786 / 0.802 / 0.648 |
| LR: PRM + backbone telemetry (supervised, all PRM800K), probability > 0.5 | THR05 (label-trained) | 57.7 | 60.7 | 54.3 | 46.8 |  |
| TELE16 standalone, L-SML (label-free) | GAP | 48.4 | 47.3 | 47.5 | 47.7 | 0.578 / 0.555 / 0.515 / 0.550 |
| TELE16 standalone, plain average | GAP | 48.4 | 47.7 | 47.4 | 48.1 | 0.581 / 0.547 / 0.518 / 0.549 |
| LR: standalone telemetry only (supervised, PRM800K) | GAP | 51.0 | 47.9 | 48.6 | 46.6 | 0.698 / 0.654 / 0.603 / 0.585 |
| LR: standalone telemetry only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 50.4 | 48.3 | 49.4 | 47.3 |  |

| arm | metric | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|---|
| PRM, GAP rule | pooled step AUROC | 0.862 | 0.893 | 0.840 | 0.808 |
| PRM, GAP rule | CROP kept % (α 0.05) | 80.9 | 81.4 | 63.0 | 62.3 |
| LR on the PRM only (supervised, PRM800K) | pooled step AUROC | 0.674 | 0.698 | 0.708 | 0.411 |
| LR on the PRM only (supervised, PRM800K) | CROP kept % (α 0.05) | 34.1 | 25.3 | 43.6 | 29.7 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.587 | 0.590 | 0.532 | 0.522 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 29.2 | 28.7 | 31.1 | 35.1 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.604 | 0.617 | 0.546 | 0.559 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | CROP kept % (α 0.05) | 32.5 | 30.9 | 33.3 | 25.4 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | pooled step AUROC | 0.704 | 0.750 | 0.712 | 0.552 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | CROP kept % (α 0.05) | 30.0 | 41.2 | 46.3 | 32.0 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | pooled step AUROC | 0.709 | 0.750 | 0.713 | 0.574 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | CROP kept % (α 0.05) | 30.5 | 41.3 | 47.0 | 31.2 |
| TELE16 standalone, L-SML (label-free) | pooled step AUROC | 0.557 | 0.536 | 0.508 | 0.519 |
| TELE16 standalone, L-SML (label-free) | CROP kept % (α 0.05) | 27.6 | 33.2 | 32.2 | 33.3 |
| TELE16 standalone, plain average | pooled step AUROC | 0.556 | 0.535 | 0.510 | 0.520 |
| TELE16 standalone, plain average | CROP kept % (α 0.05) | 28.8 | 31.8 | 32.4 | 32.1 |
| LR: standalone telemetry only (supervised, PRM800K) | pooled step AUROC | 0.642 | 0.601 | 0.574 | 0.550 |
| LR: standalone telemetry only (supervised, PRM800K) | CROP kept % (α 0.05) | 31.6 | 34.7 | 32.0 | 32.7 |

LR (PRM + telemetry, overlap removed) minus AUTO, main metric, paired bootstrap 95% CI: Qwen-PRM-7B +1.62 [+0.47, +2.71]; Qwen-PRM-72B +5.13 [+3.83, +6.26]; PURE-PRM-7B +3.68 [+2.35, +5.11]; RLHFlow-PRM-8B -1.31 [-2.42, -0.24] (resampling answers).

## DeltaBench

| arm | rule | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B | within-AUC (4 PRMs) |
|---|---|---|---|---|---|---|
| PRM, native threshold | native | 56.3 | 57.0 | 53.9 | 49.9 | 0.656 / 0.630 / 0.648 / 0.536 |
| PRM, GAP rule | GAP | 58.6 | 57.8 | 57.7 | 50.9 | 0.656 / 0.630 / 0.648 / 0.536 |
| LR on the PRM only (supervised, PRM800K) | GAP | 58.0 | 57.8 | 56.6 | 47.3 | 0.656 / 0.630 / 0.648 / 0.464 |
| LR on the PRM only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 56.7 | 55.8 | 56.2 | 41.4 |  |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | GAP | 57.2 | 57.9 | 49.5 | 54.2 | 0.660 / 0.654 / 0.416 / 0.584 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | GAP | 58.2 | 58.9 | 48.7 | 53.8 | 0.671 / 0.665 / 0.395 / 0.582 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | GAP | 57.2 | 59.9 | 58.5 | 53.7 | 0.651 / 0.661 / 0.679 / 0.581 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed), probability > 0.5 | THR05 (label-trained) | 56.5 | 59.2 | 58.6 | 52.2 |  |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | GAP | 57.7 | 59.9 | 57.9 | 54.3 | 0.656 / 0.661 / 0.680 / 0.605 |
| LR: PRM + backbone telemetry (supervised, all PRM800K), probability > 0.5 | THR05 (label-trained) | 56.9 | 59.2 | 58.6 | 53.4 |  |
| TELE16 standalone, L-SML (label-free) | GAP | 57.9 | 55.5 | 54.1 | 54.1 | 0.666 / 0.619 / 0.590 / 0.601 |
| TELE16 standalone, plain average | GAP | 57.3 | 55.6 | 54.9 | 54.8 | 0.665 / 0.620 / 0.598 / 0.609 |
| LR: standalone telemetry only (supervised, PRM800K) | GAP | 57.9 | 56.9 | 56.3 | 55.2 | 0.639 / 0.640 / 0.614 / 0.593 |
| LR: standalone telemetry only (supervised, PRM800K), probability > 0.5 | THR05 (label-trained) | 55.7 | 55.9 | 54.4 | 54.0 |  |

| arm | metric | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|---|
| PRM, GAP rule | pooled step AUROC | 0.631 | 0.608 | 0.623 | 0.551 |
| LR on the PRM only (supervised, PRM800K) | pooled step AUROC | 0.651 | 0.627 | 0.636 | 0.463 |
| AUTO: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.661 | 0.650 | 0.403 | 0.594 |
| CHAIN: PRM + backbone telemetry, L-SML (label-free) | pooled step AUROC | 0.675 | 0.666 | 0.393 | 0.588 |
| LR: PRM + backbone telemetry (supervised, PRM800K, overlap removed) | pooled step AUROC | 0.647 | 0.665 | 0.661 | 0.583 |
| LR: PRM + backbone telemetry (supervised, all PRM800K) | pooled step AUROC | 0.653 | 0.665 | 0.661 | 0.593 |
| TELE16 standalone, L-SML (label-free) | pooled step AUROC | 0.662 | 0.623 | 0.595 | 0.602 |
| TELE16 standalone, plain average | pooled step AUROC | 0.661 | 0.626 | 0.602 | 0.610 |
| LR: standalone telemetry only (supervised, PRM800K) | pooled step AUROC | 0.635 | 0.626 | 0.599 | 0.591 |

LR (PRM + telemetry, overlap removed) minus AUTO, main metric, paired bootstrap 95% CI: Qwen-PRM-7B +0.02 [-1.94, +1.93]; Qwen-PRM-72B +1.97 [+0.33, +3.80]; PURE-PRM-7B +8.98 [+6.69, +11.17]; RLHFlow-PRM-8B -0.56 [-2.52, +1.84] (resampling answers).
