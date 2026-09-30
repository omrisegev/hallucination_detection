
## Track A — anomaly scorers vs L-SML continuous (markdown)

### feature set 16 — L-SML continuous macro = 0.651 (29 common cells)

| method | raw | **anchored** | oracle | delta vs L-SML (anch) | reasoning | gpqa | rag | flips |
|---|---|---|---|---|---|---|---|---|
| maha | 0.453 | **0.513** | 0.604 | -13.8pp | 0.607 | 0.463 | 0.481 | 17/29 |
| gmm2 | 0.491 | **0.553** | 0.600 | -9.8pp | 0.701 | 0.453 | 0.510 | 14/29 |
| kde | 0.484 | **0.542** | 0.592 | -10.9pp | 0.644 | 0.450 | 0.520 | 15/29 |
| iforest | 0.448 | **0.504** | 0.605 | -14.6pp | 0.587 | 0.463 | 0.476 | 19/29 |
| ae | 0.502 | **0.535** | 0.591 | -11.6pp | 0.627 | 0.432 | 0.524 | 12/25 |
| prae | 0.479 | **0.523** | 0.590 | -12.8pp | 0.636 | 0.429 | 0.498 | 16/25 |

### feature set 5 — L-SML continuous macro = 0.653 (29 common cells)

| method | raw | **anchored** | oracle | delta vs L-SML (anch) | reasoning | gpqa | rag | flips |
|---|---|---|---|---|---|---|---|---|
| maha | 0.500 | **0.504** | 0.575 | -14.9pp | 0.572 | 0.496 | 0.472 | 5/29 |
| gmm2 | 0.561 | **0.555** | 0.589 | -9.8pp | 0.650 | 0.521 | 0.518 | 3/29 |
| kde | 0.544 | **0.531** | 0.586 | -12.2pp | 0.614 | 0.474 | 0.507 | 4/29 |
| iforest | 0.484 | **0.500** | 0.585 | -15.3pp | 0.597 | 0.484 | 0.456 | 7/29 |
| ae | 0.559 | **0.548** | 0.584 | -10.5pp | 0.580 | 0.483 | 0.556 | 4/25 |
| prae | 0.543 | **0.528** | 0.581 | -12.5pp | 0.593 | 0.472 | 0.514 | 4/25 |


## Track B — temporal candidates (markdown)

### gsm8k/Llama-3.1-8B (PRIMARY, n=200, frac_correct=0.80)

Baselines: lsml5=0.754 lsml16=0.710 epr=0.731 best DeepConf = deepconf_w32 0.735

| candidate | raw | **anchored** | oracle |
|---|---|---|---|
| hmm_occ | 0.719 | **0.719** | 0.719 |
| hmm_tail | 0.662 | **0.662** | 0.662 |
| hmm_switch | 0.508 | **0.508** | 0.508 |
| bocpd_ecp | 0.685 | **0.685** | 0.685 |
| bocpd_meanp0 | 0.598 | **0.402** | 0.598 |
| bocpd_map | 0.604 | **0.396** | 0.604 |
| bocpd_ecp_l50 | 0.703 | **0.703** | 0.703 |
| bocpd_meanp0_l50 | 0.630 | **0.630** | 0.630 |
| bocpd_map_l50 | 0.647 | **0.353** | 0.647 |
| bocpd_ecp_l200 | 0.660 | **0.660** | 0.660 |
| bocpd_meanp0_l200 | 0.584 | **0.416** | 0.584 |
| bocpd_map_l200 | 0.558 | **0.442** | 0.558 |
| ar2_mse | 0.717 | **0.717** | 0.717 |
| ar2_ratio | 0.344 | **0.656** | 0.656 |
| kalman_mse | 0.699 | **0.699** | 0.699 |
| kalman_nis | 0.703 | **0.703** | 0.703 |

### math500/Qwen2.5-Math-1.5B_T1.0 (SECONDARY / NON-CANONICAL, n=400, frac_correct=0.21)

Baselines: lsml5=0.656 lsml16=0.669 epr=0.671 best DeepConf = deepconf_w128 0.672

| candidate | raw | **anchored** | oracle |
|---|---|---|---|
| hmm_occ | 0.675 | **0.675** | 0.675 |
| hmm_tail | 0.710 | **0.710** | 0.710 |
| hmm_switch | 0.361 | **0.639** | 0.639 |
| bocpd_ecp | 0.603 | **0.603** | 0.603 |
| bocpd_meanp0 | 0.438 | **0.562** | 0.562 |
| bocpd_map | 0.493 | **0.507** | 0.507 |
| bocpd_ecp_l50 | 0.630 | **0.630** | 0.630 |
| bocpd_meanp0_l50 | 0.465 | **0.535** | 0.535 |
| bocpd_map_l50 | 0.513 | **0.487** | 0.513 |
| bocpd_ecp_l200 | 0.569 | **0.431** | 0.569 |
| bocpd_meanp0_l200 | 0.424 | **0.576** | 0.576 |
| bocpd_map_l200 | 0.484 | **0.516** | 0.516 |
| ar2_mse | 0.657 | **0.657** | 0.657 |
| ar2_ratio | 0.334 | **0.666** | 0.666 |
| kalman_mse | 0.650 | **0.650** | 0.650 |
| kalman_nis | 0.651 | **0.651** | 0.651 |


figures written to C:\Users\omris\TAU\hallucination_detection\results\figs
