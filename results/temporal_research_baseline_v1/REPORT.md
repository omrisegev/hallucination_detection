# Baseline replay and feature-bank development results

Independent raw-data baseline replay: PASS.

All 13,769 answers; 145,597 steps. Frozen tail15 Top10 gate q=.33.

| Method | PB % | Within AUC | PRMScore |
|---|---:|---:|---:|
| mean__H0lim | 36.3335 | 0.743972 | 0.633422 |
| mean__VE0 | 36.5270 | 0.753406 | 0.635477 |
| mean__VE075 | 37.8892 | 0.732302 | 0.618679 |
| mean__VE1 | 36.9021 | 0.737786 | 0.625781 |
| mean__H0lim_VE0 | 36.5725 | 0.750576 | 0.634832 |
| mean__H0lim_VE075 | 37.3217 | 0.750331 | 0.633370 |
| mean__H0lim_VE1 | 37.0785 | 0.746191 | 0.633476 |
| mean__VE0_VE075 | 37.3415 | 0.755948 | 0.635477 |
| mean__VE0_VE1 | 37.0289 | 0.753846 | 0.635649 |
| mean__VE075_VE1 | 37.9962 | 0.739048 | 0.624732 |
| mean__H0lim_VE0_VE075 | 37.0949 | 0.753620 | 0.634685 |
| mean__H0lim_VE0_VE1 | 36.8834 | 0.751743 | 0.634750 |
| mean__H0lim_VE075_VE1 | 37.6070 | 0.749877 | 0.633096 |
| mean__VE0_VE075_VE1 | 37.5097 | 0.755207 | 0.634654 |
| mean__H0lim_VE0_VE075_VE1 | 37.4749 | 0.753436 | 0.634412 |
| entropy15 | 36.3476 | 0.730111 | 0.625426 |
| append_innovation__H0lim | 39.8314 | 0.760293 | 0.638830 |
| append_innovation__VE075 | 37.7799 | 0.755932 | 0.635784 |

These are development results. See CONTRASTS.json for paired group intervals.
Historical PB repair is partial; missing learned-method archives are explicitly pending.
