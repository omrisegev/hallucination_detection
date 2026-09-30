# Stability regularization supports our fusion

Completed development pilot, 2026-09-07. The core remains IU-PCR / Joint L-SML. This is not untouched confirmation.

The penalties reduce sensitivity to block perturbations but establish no consistent improvement on both localization tasks. Larger Joint graph lambdas change the scores substantially; they do not yield a clear winner. The next priority is the interface between fused scores and the no-error decision.

## What this experiment tests

Keep all fitting windows and the same moment features, groups and score orientation. Resample two-token blocks inside each eight-token window. Sixteen perturbations estimate sensitivity penalties; eight separate perturbations choose lambda from 0, 0.1, 1 and 10. No error label enters the choice. Each candidate has unit original-score standard deviation; choose the smallest lambda within 1% of the baseline instability of the best validation loss.

Joint uses its native fitted covariance and global loading in the regularized inverse solve. IU and equal aggregation retain their fitted weights and receive an explicitly different identity-head correction. Each component is compared with all three cores. These corrections support fusion; they do not replace it with a new detector.

The five penalties are isotropic conditioning, existing DUFS graph roughness, diagonal block sensitivity, permuted diagonal sensitivity and the full sensitivity matrix. The matrix includes perturbation bias; it is not established measurement-noise covariance or a LOCA burst model. The full trace and the declared negative-entropy sign anchor are retained. The method is not anchor-free.

## ProcessBench macro F1 (%) - same 46 answers

| Core | Parent peak | Isotropic | DUFS graph | Block diagonal | Permuted diagonal | Full block matrix |
|---|---:|---:|---:|---:|---:|---:|
| equal | 17.76 | 17.76 | 11.32 | 11.13 | 17.76 | 5.56 |
| iu | 17.71 | 17.71 | 14.45 | 11.42 | 13.69 | 5.77 |
| joint | 12.50 | 10.05 | 5.88 | 10.26 | 8.33 | 9.62 |

## PRMBench AUROC (valid answers) - coverage matters

| Core | Parent peak | Isotropic | DUFS graph | Block diagonal | Permuted diagonal | Full block matrix |
|---|---:|---:|---:|---:|---:|---:|
| equal | 0.61717 (12) | 0.61717 (12) | 0.60011 (12) | 0.62207 (12) | 0.61783 (12) | 0.60185 (12) |
| iu | 0.62261 (12) | 0.62261 (12) | 0.61141 (12) | 0.61761 (12) | 0.62674 (12) | 0.60880 (12) |
| joint | 0.66171 (7) | 0.65637 (7) | 0.66760 (7) | 0.64795 (7) | 0.65665 (7) | 0.60331 (7) |

Do not compare AUROCs with different available answers as a matched leaderboard. Paired PRMB comparisons use the common valid IDs; PB always penalizes failure on the full population.

## Direct answer to the larger-lambda suggestion

| Joint graph recipe | PRMB AUROC | Valid PRMB | PB F1 (%) | Mean correlation with lambda-zero score |
|---|---:|---:|---:|---:|
| joint_parent | 0.66171 | 7 | 12.50 | parent replay |
| joint_graph010_parent | 0.66255 | 7 | 8.33 | parent replay |
| joint_graph_fixed1 | 0.61426 | 7 | 8.33 | 0.9625609697570842 |
| joint_graph_fixed10 | 0.51713 | 7 | 9.17 | 0.9233186349954786 |
| joint__dufs_graph | 0.66760 | 7 | 5.88 | 0.9741996948432565 |

## What the label-free selection chose

| Core / penalty | Lambda counts | Selection sensitivity / baseline | Fresh-perturbation sensitivity / baseline | Mean correlation with parent |
|---|---|---:|---:|---:|
| equal__isotropic | {'0.0': 58} | 1.000 | 1.000 | 1.000 |
| equal__dufs_graph | {'0.1': 9, '1.0': 16, '0.0': 14, '10.0': 19} | 0.816 | 0.825 | 0.966 |
| equal__block_diag | {'10.0': 57, '1.0': 1} | 0.572 | 0.563 | 0.959 |
| equal__block_diag_permuted | {'0.0': 28, '10.0': 23, '1.0': 5, '0.1': 2} | 0.938 | 0.942 | 0.996 |
| equal__block_full | {'10.0': 40, '0.1': 2, '1.0': 15, '0.0': 1} | 0.515 | 0.524 | 0.899 |
| iu__isotropic | {'0.0': 58} | 1.000 | 1.000 | 1.000 |
| iu__dufs_graph | {'0.1': 8, '10.0': 26, '0.0': 10, '1.0': 14} | 0.833 | 0.842 | 0.969 |
| iu__block_diag | {'10.0': 56, '1.0': 1, '0.0': 1} | 0.695 | 0.686 | 0.977 |
| iu__block_diag_permuted | {'0.0': 33, '10.0': 18, '1.0': 6, '0.1': 1} | 0.963 | 0.963 | 0.998 |
| iu__block_full | {'10.0': 58} | 0.472 | 0.463 | 0.937 |
| joint__isotropic | {'10.0': 15, '1.0': 21, '0.1': 5, '0.0': 2} | 0.880 | 0.883 | 0.991 |
| joint__dufs_graph | {'1.0': 12, '0.1': 12, '10.0': 10, '0.0': 9} | 0.893 | 0.923 | 0.974 |
| joint__block_diag | {'10.0': 39, '0.1': 1, '1.0': 3} | 0.810 | 0.810 | 0.983 |
| joint__block_diag_permuted | {'10.0': 20, '1.0': 12, '0.1': 7, '0.0': 4} | 0.900 | 0.909 | 0.990 |
| joint__block_full | {'10.0': 43} | 0.543 | 0.542 | 0.940 |

Lower sensitivity on the selection set is partly built into the selection rule. The independent review adds eight unused perturbations (replicates 24-31) after score freezing, retaining every chosen lambda and prediction. This is a post-freeze unlabeled robustness diagnostic, not a new correctness test. Robustness to this block perturbation does not establish better error ranking, factual correctness or robustness to every plausible trace change. Isotropic correction of IU/equal is an invariance control: its scalar scaling cancels after score normalization.

## Paired comparisons

Exploratory, unadjusted 95% source-group bootstrap intervals, 1,000 draws. Some draws omit a class in a PB subset and remain undefined; valid-draw counts are retained in CONTRASTS.json. All 38 registered comparisons are saved.

| Left minus right | Common PRMB N | PRMB delta [95% CI] | PB delta in percentage points [95% CI] |
|---|---:|---|---|
| equal__dufs_graph minus equal_parent | 12 | -0.0171 [-0.0375, +0.0141] | -6.4384 [-20.4009, +9.7633] |
| equal__block_diag minus equal_parent | 12 | +0.0049 [-0.0574, +0.0530] | -6.6368 [-15.7240, +2.3447] |
| equal__isotropic minus equal_parent | 12 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| equal__block_full minus equal_parent | 12 | -0.0153 [-0.0608, +0.0258] | -12.2076 [-24.9089, +3.2410] |
| equal__block_diag_permuted minus equal_parent | 12 | +0.0007 [-0.0153, +0.0155] | +0.0000 [+0.0000, +0.0000] |
| iu__isotropic minus iu_parent | 12 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| iu__dufs_graph minus iu_parent | 12 | -0.0112 [-0.0593, +0.0495] | -3.2603 [-20.6510, +13.6342] |
| iu__block_diag minus iu_parent | 12 | -0.0050 [-0.0525, +0.0226] | -6.2906 [-17.7413, +5.2632] |
| iu__block_diag_permuted minus iu_parent | 12 | +0.0041 [-0.0026, +0.0152] | -4.0179 [-12.1212, +0.0000] |
| iu__block_full minus iu_parent | 12 | -0.0138 [-0.0524, +0.0261] | -11.9391 [-23.3710, +1.2809] |
| joint__isotropic minus joint_parent | 7 | -0.0053 [-0.0399, +0.0084] | -2.4510 [-15.0000, +10.3448] |
| joint__dufs_graph minus joint_parent | 7 | +0.0059 [-0.0407, +0.0413] | -6.6176 [-19.7874, +9.5238] |
| joint__block_diag minus joint_parent | 7 | -0.0138 [-0.0603, +0.0003] | -2.2368 [-15.0000, +12.5000] |
| joint__block_diag_permuted minus joint_parent | 7 | -0.0051 [-0.0333, +0.0073] | -4.1667 [-15.3846, +7.5000] |
| joint__block_full minus joint_parent | 7 | -0.0584 [-0.0958, -0.0081] | -2.8846 [-8.3333, +7.1057] |
| joint__block_full minus iu__block_full | 7 | -0.0171 [-0.0367, +0.0034] | +3.8462 [-11.9048, +17.1798] |
| joint__dufs_graph minus joint_graph010_parent | 7 | +0.0051 [-0.0341, +0.0450] | -2.4510 [-8.3333, +7.0188] |
| iu__block_full minus equal__block_full | 12 | +0.0070 [-0.0331, +0.0489] | +0.2137 [-10.4310, +11.9048] |
| joint_graph_fixed1 minus joint_graph010_parent | 7 | -0.0483 [-0.0925, +0.0178] | +0.0000 [-12.3264, +10.3718] |
| joint_graph_fixed10 minus joint_graph010_parent | 7 | -0.1454 [-0.3660, +0.0189] | +0.8333 [-12.5000, +14.3973] |

## Isolating the error gate

All candidates use the same GMM rule on all original fused windows, but regularization changes the inputs and can change the gate. The diagnostic below instead fixes each core to its lambda-zero parent decision; it cannot credit unavailable fits.

| Candidate | PB F1 (%) | PB with fixed parent gate (%) | Valid answers / 58 |
|---|---:|---:|---:|
| iu__isotropic | 17.71 | 17.71 | 58 |
| iu__dufs_graph | 14.45 | 24.64 | 58 |
| iu__block_diag | 11.42 | 14.64 | 58 |
| iu__block_diag_permuted | 13.69 | 17.71 | 58 |
| iu__block_full | 5.77 | 13.69 | 58 |
| joint__isotropic | 10.05 | 12.50 | 43 |
| joint__dufs_graph | 5.88 | 8.33 | 43 |
| joint__block_diag | 10.26 | 12.50 | 43 |
| joint__block_diag_permuted | 8.33 | 12.50 | 43 |
| joint__block_full | 9.62 | 8.33 | 43 |

## Review and historical continuity

Six scientific tests passed. The independent review reconstructs 1392 perturbation matrices directly from raw token arrays, verifies 3180 weight solves, 795 lambda-zero replays, 303 exact parent score replays, all 23 endpoints and 58 direct label joins. Selection rules, PSD, normalization, spans and hashes pass. Numerical discrepancies are recorded in REVIEW.json.

Scoring took 45.6 seconds on three CPU workers, including perturbations and grid evaluation. Bootstrap time is additional. Six parent endpoints exactly reproduce the previous peak report. Historical 30-long-answer IU 0.70070 and Claude pooled-fit values use different populations/access contracts; they remain context, not matched gains. The 58 answers have already been used for development. No candidate is a publication winner from this pilot alone.

## Findings and next action

No recipe establishes a consistent improvement on both benchmarks. Keep the
full-grid IU/equal references and the Joint controls. Do not extend this
particular sensitivity objective into a larger search or promote the most
favorable point estimate from the pilot.

The larger-lambda suggestion was tested directly. Joint graph lambda 0.1
gives PRMB 0.66255 and PB 8.33%; lambda 1 gives 0.61426 and 8.33%; lambda 10
gives 0.51713 and 9.17%. The graph changes the score: its mean correlation
with the lambda-zero trajectory is 0.963 at lambda 1 and 0.923 at lambda 10.
It is not an inactive implementation. Larger influence does not provide a
consistent gain. Lambda-10 minus lambda-0.1 PRMB is -0.14542, CI
[-0.36596,+0.01892]; the small pilot does not establish a precise population
effect. PB difference is +0.83 points [-12.50,+14.40].

The automatic Joint graph rule gives PRMB 0.66760 versus lambda-zero 0.66171
on the same seven answers, difference +0.00590 [-0.04071,+0.04134]. PB falls
from 12.50% to 5.88%, difference -6.62 points [-19.79,+9.52]. Choosing lambda
without labels is feasible, but this objective has not chosen a winning
localizer. Joint feature-bank and grouping development remain open; neither
changed in this experiment.

Sensitivity did decrease on the eight additional, unused perturbations:
the full-matrix correction has 0.463 times the parent's sensitivity for IU,
and 0.542 for Joint. Yet IU PB falls from 17.71% to 5.77%, and Joint PRMB
falls from 0.66171 to 0.60331. The latter exploratory paired difference is
-0.05839 [-0.09582,-0.00805]. This separates stability under the imposed
perturbation from usefulness for error detection. Those perturbations may
disturb useful reasoning changes; the data does not establish them as noise.
The extra perturbation audit was added after freezing, did not use labels
or change any selection, and is not a new-answer confirmation test.

The error gate remains an important unresolved interface. IU with the selected
graph correction scores 14.45% PB under the unchanged GMM rule, but 24.64%
if we preserve its parent's binary error decisions and change only location.
This is a diagnostic, not a registered new winner; PRMB is still lower than
the parent (0.61141 versus 0.62261). It shows why a single F1 number should
not merge fusion ranking and clean/error gating into one explanation.

Next audit the answer-only normalization and no-error interface. Establish
which absolute uncertainty information is removed by per-answer centering,
and whether the current mixture states have evidence of a correctness
interpretation. Then freeze one bounded gate experiment with the fusion
scores and equal/entropy controls held fixed. If a pooled unlabeled gate is
needed, report it as a separate hybrid fit scope; do not call it answer-only
or calibrate it with test correctness labels. Preserve the feature-bank,
grouping and task-aware sampling tracks as open supporting-fusion work.

The complete comparator replay, untouched two-benchmark confirmation and
historical 24-cell transfer still remain. These are measured steps in
developing our fusion method, not completion of the full research objective.