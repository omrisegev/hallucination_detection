# Unified Causal subset scorer-model validation v1

Frozen transfer from Qwen3-8B-scored development telemetry to the complete Llama-3.1-8B-scored ProcessBench panel. This is robustness, not untouched confirmation.

- Source development run: `results/unified_causal_subset_search_base7_dufs80_v1`
- Development groups used for the frozen fit: 128
- Validation rows: 3400
- Validation groups: 3400
- Control: `base7_full28`

| candidate | Global | ΔG | Localization | ΔL | Early | ΔE |
|---|---:|---:|---:|---:|---:|---:|
| base7_full28 | 0.6629 | +0.0000 | 0.2880 | +0.0000 | 0.5587 | +0.0000 |
| base7_full28__dufs_l0p1 | 0.6613 | -0.0016 | 0.2895 | +0.0015 | 0.5573 | -0.0014 |
| base7_full28__dufs_l0p3 | 0.6590 | -0.0039 | 0.2824 | -0.0056 | 0.5556 | -0.0032 |
| raw9_full36 | 0.6645 | +0.0016 | 0.2705 | -0.0175 | 0.5616 | +0.0029 |
| base7_full28__dufs_l1 | 0.6550 | -0.0079 | 0.2580 | -0.0301 | 0.5526 | -0.0062 |
| base7_full28__rw_a0p5 | 0.6503 | -0.0126 | 0.2561 | -0.0319 | 0.5497 | -0.0090 |
| base7_full28__dufs_l3 | 0.6512 | -0.0117 | 0.2352 | -0.0528 | 0.5503 | -0.0084 |

No validation label entered feature selection, sign estimation, reference fitting, IU/DUFS fitting, alpha/lambda selection, or threshold calibration. The labels are used only to score this report.

The panel is nevertheless not untouched: these ProcessBench questions and the Llama scorer cache have appeared in earlier repository analyses.
