# Fixed-bank sampling replication - Step317

Review PASS. No consistent IU/Joint improvement on both tasks. Same110 development answers, v3 labels/v2 groups.

| Selector | IU AUC | IU within | IU PB | Joint graph AUC | Joint graph PB | Equal perm AUC | Equal perm PB | Native Joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All windows | 0.68131 | 0.76881 | 30.16% | 0.65545 | 30.22% | 0.69226 | 31.32% | 107/110 |
| Uniform | 0.67643 | 0.77697 | 28.34% | 0.64566 | 26.87% | 0.69597 | 31.25% | 101/110 |
| Entropy risk | 0.75835 | 0.77892 | 28.35% | 0.74133 | 28.20% | 0.76842 | 33.92% | 90/110 |
| Transposed DUFS | 0.65957 | 0.76354 | 29.69% | 0.62168 | 28.07% | 0.64758 | 29.54% | 84/110 |
| Permuted DUFS | 0.67391 | 0.77081 | 26.27% | 0.64178 | 30.02% | 0.66712 | 24.63% | 92/110 |
| Window diffusion | 0.68498 | 0.76999 | 26.26% | 0.64446 | 29.64% | 0.69825 | 27.61% | 99/110 |

149 entries include107 anchors and seven full aliases. All42 final outputs cover110; native Joint counts above include the unchanged short-answer fits.
Risk-IU pooled AUC improves strongly, within-answer change is small/uncertain, PB falls. Equal-permuted-graph risk sampling has higher points on both tasks, but PB improvement CI includes zero.
Post-evaluation affine diagnostic, exact PB transitions, eligible-only metrics, short-error fit support and all93 comparisons are in REPORT.html / DIAGNOSTICS.json.
Scoring 393.16s; contrasts 45.93s; review 153.31s. Dense features/scoring retained; no inference saving claimed.
No candidate promotion or full-goal completion. Preserve anchors and inspect calibration/peak/gate attribution before another sweep.
