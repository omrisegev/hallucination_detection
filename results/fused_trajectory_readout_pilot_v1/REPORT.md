# Fused trajectory readout pilot v1

2026-09-07. Completed adaptive development pilot; fusion remains the core. No winner promoted.

A peak locator repairs part of the previous first-crossing failure. None of the tested chronological additions improves the strongest simple readout on both tasks. Learned fusion still has no demonstrated advantage over its simple controls on this small cohort.

## Scope and attribution

The exact same 58 answers (12 PRMB, 46 PB), feature fits, signs, weights, graphs and binary error gates are retained from the representation pilot. Six cores × seven readouts were frozen before this version was evaluated. The parent labels were already known: this is adaptive development, not fresh confirmation. The error gate stays fixed; this stage tests where to localize an accepted error, not a better clean/error classifier. All full-pipeline PB scores count failed fits/readouts as misses.

IU-PCR and Joint L-SML generate every proposed fused input. Entropy and equal averaging are controls. IMM, HMM and BOCPD process those saved scores; they are supporting components, not replacement detectors.

## ProcessBench macro F1 (%) — same 46-answer population

| Fusion core | First crossing | Raw peak | Hold peak | HMM entry | Kalman level | IMM level | BOCPD rise |
|---|---:|---:|---:|---:|---:|---:|---:|
| equal | 0.00 | 17.76 | 17.76 | 0.00 | 0.00 | 14.79 | 5.36 |
| iu | 0.00 | 17.71 | 17.71 | 0.00 | 5.36 | 14.79 | 13.69 |
| joint_lambda0 | 0.00 | 12.50 | 12.50 | 0.00 | 12.50 | 12.50 | 8.33 |
| joint_graph010 | 0.00 | 8.33 | 8.33 | 0.00 | 4.17 | 8.33 | 4.17 |
| joint_graph_permuted | 0.00 | 4.17 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 |
| entropy_mean_w8 | 8.33 | 19.85 | 19.85 | 0.00 | 12.50 | 19.85 | 17.76 |

IU rises from 0 to 17.71% with the raw peak locator; equal fusion is 17.76% and entropy is 19.85%. The same IU gate and same PRMB ranking are retained. This is a readout repair, not evidence that IU learned superior fusion weights. IMM gives IU 14.79%, BOCPD 13.69%, and HMM 0%. Joint lambda-zero raw peak gives 12.50%, meaningful graph 8.33%, permutation 4.17%; their small-sample differences need the paired uncertainty below.

## PRMB step AUROC — availability is different

Each entry is AUROC (number of valid answers). Do not rank entries with different availability as a common-population leaderboard. HMM uses high-state occupancy for PRMB ranking and entry probability for PB onset; this distinction was declared before scoring.

| Core | Raw peak | Hold peak | HMM | Kalman | IMM | BOCPD |
|---|---:|---:|---:|---:|---:|---:|
| equal | 0.61717 (12) | 0.59728 (12) | 0.61967 (10) | 0.57946 (12) | 0.60326 (12) | 0.59728 (12) |
| iu | 0.62261 (12) | 0.60272 (12) | 0.54822 (10) | 0.58946 (12) | 0.61163 (12) | 0.60902 (12) |
| joint_lambda0 | 0.66171 (7) | 0.65918 (7) | 0.58224 (5) | 0.58113 (7) | 0.64346 (7) | 0.62605 (7) |
| joint_graph010 | 0.66255 (7) | 0.65722 (7) | 0.55162 (5) | 0.58113 (7) | 0.62661 (7) | 0.59882 (7) |
| joint_graph_permuted | 0.66115 (7) | 0.65918 (7) | 0.57699 (5) | 0.58226 (7) | 0.64458 (7) | 0.62774 (7) |
| entropy_mean_w8 | 0.62587 (12) | 0.62185 (12) | 0.63213 (10) | 0.57141 (12) | 0.60946 (12) | 0.60674 (12) |

Raw peak leaves original PRMB rankings untouched. The hold control drops the extra overlapping end window, so temporal additions must be compared with hold_peak as well. This mapping change alone reduces IU PRMB AUROC from 0.62261 to 0.60272, illustrating why implementation-level anchors matter.

## Paired contrasts

Exploratory, unadjusted, 1,000 source-group bootstrap draws. PRMB uses common valid IDs; PB uses all 46 answers with failed-readout penalties. Tiny or degenerate intervals cannot establish population equivalence. All 58 registered contrasts are retained in CONTRASTS.json.

| Left minus right | PRMB common N | PRMB delta [95% CI] | PB delta in percentage points [95% CI] |
|---|---:|---|---|
| iu@@parent_peak minus iu@@parent_first | 12 | +0.00000 [+0.0000, +0.0000] | +17.71 [+0.0000, +27.7778] |
| iu@@hmm_entry minus iu@@hold_peak | 10 | -0.07812 [-0.2547, +0.0131] | -17.71 [-26.1364, +0.0000] |
| iu@@imm_level minus iu@@hold_peak | 12 | +0.00891 [-0.0557, +0.0779] | -2.92 [-16.4860, +6.6667] |
| iu@@bocpd_rise minus iu@@hold_peak | 12 | +0.00630 [-0.1025, +0.1386] | -4.02 [-12.5000, +0.0000] |
| iu@@imm_level minus iu@@kalman_level | 12 | +0.02217 [-0.0725, +0.1195] | +9.43 [-8.3333, +22.4844] |
| joint_graph010@@parent_peak minus iu@@parent_peak | 7 | +0.01291 [-0.0302, +0.0486] | -9.37 [-23.5831, +3.0642] |
| joint_graph010@@parent_peak minus joint_lambda0@@parent_peak | 7 | +0.00084 [-0.0230, +0.0170] | -4.17 [-15.3846, +7.5000] |
| iu@@parent_peak minus equal@@parent_peak | 12 | +0.00543 [-0.0249, +0.0429] | -0.05 [-9.6280, +9.3750] |
| joint_graph010@@parent_peak minus joint_graph_permuted@@parent_peak | 7 | +0.00140 [-0.0297, +0.0200] | +4.17 [+0.0000, +9.5238] |

## Gate limitation and availability

The following ceiling assumes a perfect locator but preserves every core’s frozen clean/error decision. It is an oracle diagnostic, not an achieved score or a supervised candidate. HMM failures can further reduce coverage below this ceiling.

| Core | Parent valid / 58 | HMM valid / 58 | PB F1 ceiling with perfect locator (%) |
|---|---:|---:|---:|
| equal | 58 | 52 | 54.13 |
| iu | 58 | 47 | 55.18 |
| joint_lambda0 | 43 | 35 | 23.33 |
| joint_graph010 | 43 | 37 | 28.24 |
| joint_graph_permuted | 43 | 36 | 20.83 |
| entropy_mean_w8 | 58 | 49 | 37.56 |

On this cohort, even a perfect locator cannot take the present Joint graph gate above 28.24% PB F1, whereas the IU gate ceiling is 55.18%. Both fit availability and clean/error decisions constrain the result. Improving the locator alone cannot solve every failure. These ceilings depend on the pilot labels and are diagnostics only.

## Review and historical continuity

Seven scientific-contract tests passed, including exact HMM state-path probabilities, exhaustive Gaussian partition evidence, IMM mixing covariance and the single-Kalman limit. Independent review verifies 303 exact parent replays, 2074 unchanged error gates, 2936 span mappings, 909 filter probability/covariance checks, all 42 endpoints and 58 direct source-label joins. Source and score hashes match. Scoring took about 34 seconds on three CPU workers; bootstrap time is additional.

HISTORY Step 246 already tested pooled token-level IU-HMM and found it worse than ordinary IU. This new single-answer window version also fails to improve PB in the tested form. It does not close all chronological models or Joint/graph feature development. The historical 30-long-answer IU 0.70070 and Claude’s pooled/calibrated PB figures belong to different cohorts/access contracts; parent_first replays the compatible current baseline exactly.

The old temporal_models BOCPD code mixes reset-before-observation prediction with an unupdated reset branch. Its claim that constant P(r=0) always indicates a bug is also too broad: the original paper uses an after-observation boundary convention. This pilot uses a consistent reset-before-observation Gaussian adaptation, verified against all short-sequence partitions. Historical numerical outputs remain intact.

Sources: [Adams and MacKay](https://arxiv.org/html/0710.3742v1), [FilterPy IMM](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/IMM.html). The localization readouts, fixed parameters and boundary heuristic are our adaptations, not published hallucination detectors.

## Next bounded direction

Retain raw peak as the current simple localization control. Do not expand these losing temporal configurations into a full sweep. Next investigate the representation and reliability inputs to fusion, including label-free Joint regularization/stability and the user’s token/window sampling idea, with IU/equal and meaningful/permuted graph anchors. Separately address the within-answer clean/error gate; a different gate needs a new frozen version. KalmanNet, LOCA and flow-derived views remain supporting hypotheses. Full matched comparator coverage, untouched confirmation on both tasks and frozen-candidate 24-cell transfer remain open.
