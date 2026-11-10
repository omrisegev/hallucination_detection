# C4U with telemetry only, no PRM input (2026-11-11)

Omri asked for the frozen C4U (C4U-parisi-within-jaffe-cross-v1) run on telemetry only, with no PRM score and no PRM
anchor, in two modes: telemetry from the standalone base model (`sa`) and telemetry from the PRM network's backbone
without its reward output (`prm`). PRMBench and ProcessBench, 4 PRMs. CPU only, stored raw banks. Development data.

`run.py` docstring has the exact adaptation (3 changes: inputs = the 16 R1 telemetry columns only; L-SML sign anchor =
the mean of the 16 columns; spectral anchor group = the group most correlated with the fused score). Fixed before any
quality number, no tuned parameter. Full C4U and PRM rows are recomputed in the same cells and gated against
`c4u_tele16_all_benchmarks_v1/RESULTS_{prm,sa}.csv`: 48/48 parity checks pass (`PARITY.json`).

| file | what |
|---|---|
| `run.py` | runs C4U-T, full C4U and the PRM on PRMBench + ProcessBench, both modes -> `RESULTS.csv`, `FIT.json`, `PARITY.json` |
| `figures.py` | `fig_prmbench.png`, `fig_processbench.png` |

## PRMBench PRMScore (own flags; PRM at native threshold)

| PRM | PRM alone | C4U full sa | C4U-T sa | C4U full bb | C4U-T bb | TELE16 bb |
|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 65.49 | 70.09 | 67.32 | 71.44 | 69.24 | 69.26 |
| Qwen-PRM-72B | 68.18 | 68.51 | 65.43 | 72.58 | 68.73 | 69.68 |
| PURE | 65.31 | 69.25 | 67.29 | 69.94 | 67.95 | 68.71 |
| RLHFlow | 54.43 | 68.63 | 68.28 | 67.88 | 67.32 | 66.17 |

## PRMBench within-answer AUC

| PRM | PRM alone | C4U full sa | C4U-T sa | C4U full bb | C4U-T bb |
|---|---|---|---|---|---|
| Qwen-PRM-7B | .801 | .823 | .782 | .840 | .811 |
| Qwen-PRM-72B | .795 | .810 | .766 | .853 | .822 |
| PURE | .763 | .805 | .768 | .818 | .787 |
| RLHFlow | .662 | .794 | .788 | .790 | .779 |

## ProcessBench (no within-AUC: invalid there). Macro SLA = first flagged step = first error, exact

| PRM | PRM alone | C4U full sa | C4U-T sa | C4U full bb | C4U-T bb |
|---|---|---|---|---|---|
| Qwen-PRM-7B | 61.87 | 39.95 | 31.64 | 56.65 | 44.85 |
| Qwen-PRM-72B | 71.22 | 38.08 | 26.75 | 61.97 | 32.80 |
| PURE | 38.98 | 31.11 | 28.03 | 34.39 | 31.70 |
| RLHFlow | 19.09 | 27.70 | 30.85 | 17.32 | 33.59 |

PRISM T6 average F1: PRM 72.9 / 78.2 / 52.6 / 29.2; both C4U versions ~0 (C4U-T bb Qwen-72B 29.7), because the own
count flags at least one step in almost every answer (correct answers then score 0). C4U-T bb on Qwen-PRM-72B is the
one cell with K=3 groups and prevalence .11; it leaves 30% of answers unflagged, hence its F1 and its low SLA.
