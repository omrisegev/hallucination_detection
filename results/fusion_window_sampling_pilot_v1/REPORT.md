# Window sampling to support our fusion

Completed 2026-09-07. Adaptive development; no confirmed winner.

IU-PCR and Joint L-SML remain the core. We test which windows should supply the data used to fit their weights. The selector has no access to correctness labels. Every selector is shared with equal aggregation and both graph controls.

## What changed

One answer becomes N nonoverlapping 8-token windows by 27 moment measurements. The selected fitting budget is min(N, max(32, ceil(N/2))). Short answers use all rows with exact parent replay. The same fitted fusion still scores every window and official step. This is a fitting-data experiment; feature extraction is not reduced.

Full, uniform and raw-entropy top-risk controls accompany two graph ideas. Transposed DUFS gates windows but graphs feature coordinates; window diffusion graphs windows and chooses geometric representatives. Its geometry is not a learned measure of correctness. A shuffled-DUFS control tests whether learned placement matters.

Every arm uses the same mixture error-gate rule on all original nonoverlapping fused scores and the same peak locator. Refitting may change both risk and the gate. A separate fixed-parent-gate diagnostic below isolates the location change.

## ProcessBench macro F1 (%) — all 46 fixed answers

| Core | Full | Uniform | Top risk | DUFS windows | Shuffled DUFS | Window diffusion |
|---|---:|---:|---:|---:|---:|---:|
| equal | 17.76 | 17.71 | 11.13 | 13.69 | 15.14 | 5.26 |
| iu | 17.71 | 15.70 | 19.05 | 13.69 | 21.88 | 15.03 |
| joint_lambda0 | 12.50 | 11.31 | 13.26 | 13.26 | 17.42 | 11.31 |
| joint_graph010 | 8.33 | 4.17 | 4.17 | 4.17 | 4.17 | 4.17 |
| joint_graph_permuted | 4.17 | 4.17 | 4.17 | 4.17 | 4.17 | 4.17 |

## PRMBench pooled step AUROC — availability shown

Each entry is AUROC (valid answers). Cells with different coverage are not a common-population leaderboard. Use the paired contrasts to judge method differences.

| Core | Full | Uniform | Top risk | DUFS windows | Shuffled DUFS | Window diffusion |
|---|---:|---:|---:|---:|---:|---:|
| equal | 0.61717 (12) | 0.64109 (12) | 0.65554 (12) | 0.65370 (12) | 0.64880 (12) | 0.63196 (12) |
| iu | 0.62261 (12) | 0.64272 (12) | 0.64217 (12) | 0.63663 (12) | 0.63761 (12) | 0.62793 (12) |
| joint_lambda0 | 0.66171 (7) | 0.61430 (8) | 0.37719 (3) | 0.58333 (4) | 0.64964 (7) | 0.61346 (6) |
| joint_graph010 | 0.66255 (7) | 0.60759 (8) | 0.37719 (3) | 0.58333 (4) | 0.66086 (7) | 0.60038 (6) |
| joint_graph_permuted | 0.66115 (7) | 0.61374 (8) | 0.37719 (3) | 0.57667 (4) | 0.65300 (7) | 0.61192 (6) |

## Paired evidence

Exploratory, unadjusted 95% intervals from 1,000 source-group bootstrap draws. PRMB compares common valid IDs; PB includes all fixed IDs with failure penalties. Some resamples lack one class within a PB subset and have undefined macro F1; only defined draws form the interval, and their count is logged per comparison. All 57 registered comparisons are saved, including comparisons not displayed here.

| Left minus right | Common PRMB N | PRMB difference [95% CI] | PB difference in points [95% CI] |
|---|---:|---|---|
| iu@@uniform minus iu@@full | 12 | +0.0201 [-0.0206, +0.0691] | -2.0064 [-10.9740, +7.5000] |
| iu@@risk_top minus iu@@full | 12 | +0.0196 [-0.0253, +0.0814] | +1.3441 [+0.0000, +9.7579] |
| iu@@dufs_transposed minus iu@@full | 12 | +0.0140 [-0.0461, +0.0777] | -4.0179 [-12.1212, +0.0000] |
| iu@@dufs_permuted minus iu@@full | 12 | +0.0150 [-0.0199, +0.0521] | +4.1667 [-8.5071, +17.7264] |
| iu@@window_diffusion minus iu@@full | 12 | +0.0053 [-0.0167, +0.0378] | -2.6738 [-11.1111, +7.2135] |
| joint_graph010@@risk_top minus joint_graph010@@full | 3 | +0.0000 [+0.0000, +0.0000] | -4.1667 [-10.3448, +0.0000] |
| joint_graph010@@dufs_transposed minus joint_graph010@@full | 4 | -0.0233 [-0.0833, +0.0625] | -4.1667 [-9.8548, +0.0000] |
| joint_graph010@@uniform minus joint_graph010@@full | 7 | -0.0298 [-0.0513, +0.0045] | -4.1667 [-9.5238, +0.0000] |
| joint_graph010@@window_diffusion minus joint_graph010@@full | 5 | -0.0111 [-0.0317, +0.0350] | -4.1667 [-10.0000, +0.0000] |
| joint_graph010@@dufs_permuted minus joint_graph010@@full | 7 | -0.0017 [-0.0427, +0.0773] | -4.1667 [-9.5238, +0.0000] |
| joint_graph010@@dufs_transposed minus iu@@dufs_transposed | 4 | +0.0100 [-0.0655, +0.1875] | -9.5238 [-20.3899, +0.0000] |
| joint_graph010@@dufs_transposed minus joint_lambda0@@dufs_transposed | 4 | +0.0000 [-0.0268, +0.0250] | -9.0909 [-17.8571, +0.0000] |
| joint_graph010@@dufs_transposed minus joint_graph_permuted@@dufs_transposed | 4 | +0.0067 [-0.0268, +0.0250] | +0.0000 [+0.0000, +0.0000] |
| iu@@dufs_transposed minus equal@@dufs_transposed | 12 | -0.0171 [-0.0794, +0.0211] | +0.0000 [-15.0486, +14.3076] |

## Sampling stability and potential short-error support

Sampling reduces rows in 37/58 answers, including 8 PRMB answers. Stability below is measured only on these eligible answers. Short-event support means any selected fitting window overlaps the first erroneous official step; it is not sparse-detector recall.

| Selector | Mean selected fraction | Block-perturbation Jaccard | DUFS seed Jaccard | Selection seconds | First-error support | <=32-token error support |
|---|---:|---:|---:|---:|---|---|
| full | 1.000 | 1.000 | - | 0.000 | 19/19 | 0/0 |
| uniform | 0.572 | 1.000 | - | 0.000 | 19/19 | 0/0 |
| risk_top | 0.572 | 0.815 | - | 0.000 | 19/19 | 0/0 |
| dufs_transposed | 0.572 | 0.646 | 0.859 | 1.216 | 19/19 | 0/0 |
| dufs_permuted | 0.572 | 0.646 | - | 0.000 | 19/19 | 0/0 |
| window_diffusion | 0.572 | 0.474 | - | 0.008 | 19/19 | 0/0 |

There are zero <=32-token first-error steps in the sampling-eligible PB group. The 0/0 entries therefore mean no evidence, not perfect retention. Selection seconds exclude fusion fitting and perturbation repeats; shuffled DUFS shares the DUFS training cost.

Two-token block bootstrap stays within the original eight-token window. It tests measurement sensitivity, adds no independent observations and does not reproduce LOCA. The DUFS selector uses fixed 120-epoch optimization; a negative result does not close longer optimization or different task-aware sampling objectives.

## Fit coverage and gate diagnostic

| Core / selector | Valid answers / 58 | PB with refitted-score gate (%) | PB with fixed parent gate (%) |
|---|---:|---:|---:|
| iu@@full | 58 | 17.71 | 17.71 |
| iu@@uniform | 58 | 15.70 | 19.72 |
| iu@@risk_top | 58 | 19.05 | 17.71 |
| iu@@dufs_transposed | 58 | 13.69 | 17.71 |
| iu@@dufs_permuted | 58 | 21.88 | 22.84 |
| iu@@window_diffusion | 58 | 15.03 | 17.71 |
| joint_graph010@@full | 43 | 8.33 | 8.33 |
| joint_graph010@@uniform | 46 | 4.17 | 4.17 |
| joint_graph010@@risk_top | 35 | 4.17 | 4.17 |
| joint_graph010@@dufs_transposed | 35 | 4.17 | 4.17 |
| joint_graph010@@dufs_permuted | 43 | 4.17 | 4.17 |
| joint_graph010@@window_diffusion | 42 | 4.17 | 4.17 |

Fixed-parent-gate values are diagnostics; unavailable parent gates or current fits earn no credit. The full cohort remains primary. Sampling-eligible point estimates, within-answer PRMB AUROC and every subset accuracy are retained in REVIEW.json and EVALUATION.json.

## Review, cost and historical bridge

Six contract tests passed. Independent review validates hashes, selected-only normalization and weight reconstruction, exact parent replays, official span mapping, valid Joint fit requirements, direct label-ID joins and every endpoint. Detailed counts are in REVIEW.json. Replay metadata uses PARENT_READOUT_INVALID for an unavailable parent path, including a parent fit that never produced a readout; the original failure causes are separately recovered in REVIEW.json. The independent span sum needed a 1e-12 floating-point tolerance instead of bitwise equality; frozen scores were unchanged.

Scoring, selection and two perturbation repetitions took 178.4 seconds on three CPU workers. Bootstrap time is additional. This does not establish a runtime saving: most selectors add work, and every window feature is still computed.

The five full arms exactly match the previous readout pilot peak endpoints. The same-cohort entropy peak control remains PB 19.85% and PRMB 0.62587; it is a diagnostic control, not a replacement for our method. Earlier 30-long-answer IU 0.70070 and Claude pooled-fit figures use different populations and fit contracts. They remain historical context and cannot be used as matched improvement claims.

Source: local DUFS arXiv:2007.04728v3 (2020), equations 6–7 and the full paper digest. The later NeurIPS 2021 paper has a different author list and equation numbering. Both selectors here are explicitly adaptations for our fusion architecture.

## What we learned and how to continue

Keep full-grid IU as the working feature-fusion reference. This is a continuity
choice, not a claim that it beats every comparator. No tested sampling rule
establishes a consistent two-task advantage. Equal fusion remains essential:
under transposed DUFS it gives the same PB F1 as IU and a higher PRMB point
estimate; the learned-fusion advantage remains unresolved.

For IU, transposed DUFS changes PRMB by +0.01402 with CI [-0.04608,+0.07772]
and PB by -4.02 points with CI [-12.12,0]. Its shuffled control has the
largest observed IU PB score, 21.88%, versus full 17.71%, but the improvement
interval is [-8.51,+17.73] points. This is not evidence that the learned
window placement works. Do not select the shuffled control as a winner based
on this already inspected, very small development set.

IU top-risk sampling raises pooled PRMB from 0.62261 to 0.64217 and PB from
17.71% to 19.05%. Both paired intervals include zero. With the parent gate
fixed, PB stays at 17.71%; its observed improvement comes from changed
clean/error gating, not a higher aggregate score for choosing the error step.
Its mean within-answer PRMB AUROC falls from 0.67567 to 0.66647. Pooled rank
improvement therefore does not by itself prove improved within-answer
localization; fitting-subset normalization also changes cross-answer scores.

For Joint graph fusion, every reduced selector gives PB 4.17% versus 8.33%
for full fitting. Transposed DUFS lowers coverage from 43/58 to 35/58, with
only four valid PRMB answers. Uniform sampling raises coverage to 46/58 but
reduces PRMB AUROC by 0.02976 on the seven common valid answers, with CI
[-0.05126,+0.00454]. More admissible fits and better localization are different
outcomes. A near-zero or degenerate interval on a tiny selected subgroup
does not establish equivalence or population reliability.

The graph selectors are also sensitive to within-window block perturbation:
mean selected-set Jaccard is 0.646 for transposed DUFS and 0.474 for window
diffusion. DUFS seed agreement is higher, 0.859, so optimization-seed agreement
alone misses measurement sensitivity. No short-error preservation conclusion
is possible because the eligible PB group has no <=32-token first-error steps.

The next bounded direction is to keep every fitting row and study feature
reliability and label-free Joint regularization/stability. Retain lambda-zero,
meaningful graph, permuted graph, IU and equal references. Any reliability
penalty must affect the fusion fit or weight solve; scaling columns only to
undo it by z-scoring is not an intervention. Local block sensitivity is an
observable diagnostic, not verified measurement-noise variance or a LOCA
burst model. Separate changes in fusion ranking from the clean/error gate.

Sparse end-to-end scoring, task-aware sampling, the full comparator registry,
untouched confirmation on both benchmarks and the frozen-candidate 24-cell
transfer remain open. This stage develops and audits our fusion method; it
does not replace it with a standalone geometric detector or finish the wider
research objective.