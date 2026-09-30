# Context features change Joint coverage and localization

Completed development pilot, 2026-09-07. Same 58 cached answers, 17 arms, 31 paired contrasts. IU-PCR / Joint L-SML are still the fusion cores. All six parent controls reproduce their original scores and 12 task endpoints.

The context bank raises the observed ProcessBench result for Joint lambda-zero from 12.50% to 27.43%, and valid fits from 43/58 to 50/58. It also loses some previously valid fits, performs worse on the four common valid PRMB answers, and does not establish a learned-fusion advantage over its simple control. There is no consistent two-task winner.

## What changed

| Factor | Parent | New comparison |
|---|---|---|
| Feature definitions | Nine primitive streams x {window mean, SD, slope} | Same nine streams x {window mean, mean EMA8, mean EMA32} |
| Nominal P / fitting windows | 27 features / original non-overlapping eight-token windows | 27 features / exactly the same fitting and scoring windows |
| Grouping search | K in {3,4,6,8} | Also test all K from 3 to floor(active P / 3) |
| Fusion controls | Equal, IU, Joint native inverse at lambda 0, graph 0.1, permuted graph | Same cores and numerical validity rules |
| Primary decision | Answer-only GMM gate, then peak official step | Same rule fitted to the new score |
| Gate diagnostic | One frozen parent IU binary decision per answer | Use that identical gate for every bank and core |

EMA is an existing project idea. We reuse its linear update from the causal DSP work, initialize it with the first observed value, and import none of that historical pipeline's label-selected rosters, signs or fitted references. All nine streams remain. The feature definitions are fixed engineering choices, not a learned optimum. The declared negative-entropy sign anchor remains.

The new features bring trajectory context into the feature matrix before fusion. They do not add independent observations or constitute a separately learned second trajectory-fusion stage. Full-answer fitting and step decisions remain offline, using the existing trace with no new inference or other-answer fit.

## Primary results and the gate diagnostic

PB macro-F1 (%) always uses all 46 PB answers, including failed fits. PRMB AUROCs use available valid answers; see the common-ID comparisons below before comparing different coverage. The common-IU-gate column is diagnostic, not a newly selected candidate.

| Arm | Valid fits / 58 | PRMB valid answers | PRMB pooled AUC | PRMB mean within-answer AUC | PB native gate % | PB common IU gate % |
|---|---|---|---|---|---|---|
| moment__equal | 58 | 12 | 0.61717 | 0.66742 | 17.76 | 20.83 |
| moment__iu | 58 | 12 | 0.62261 | 0.67567 | 17.71 | 17.71 |
| moment__joint0 | 43 | 7 | 0.66171 | 0.69200 | 12.50 | 8.33 |
| moment__graph010 | 43 | 7 | 0.66255 | 0.68745 | 8.33 | 8.33 |
| moment__graph_perm | 43 | 7 | 0.66115 | 0.69131 | 4.17 | 8.33 |
| context__equal | 58 | 12 | 0.66087 | 0.66182 | 27.67 | 18.80 |
| context__iu | 58 | 12 | 0.65652 | 0.66217 | 16.97 | 20.19 |
| context__joint0 | 50 | 9 | 0.63826 | 0.66430 | 27.43 | 14.93 |
| context__graph010 | 50 | 9 | 0.64297 | 0.67794 | 20.19 | 14.93 |
| context__graph_perm | 50 | 9 | 0.64101 | 0.66579 | 19.10 | 14.93 |
| moment_allk__joint0 | 43 | 7 | 0.66171 | 0.69200 | 12.50 | 8.33 |
| moment_allk__graph010 | 43 | 7 | 0.66255 | 0.68745 | 8.33 | 8.33 |
| moment_allk__graph_perm | 43 | 7 | 0.66115 | 0.69131 | 4.17 | 8.33 |
| context_allk__joint0 | 50 | 9 | 0.64808 | 0.66579 | 27.43 | 14.93 |
| context_allk__graph010 | 50 | 9 | 0.65161 | 0.67794 | 20.19 | 14.93 |
| context_allk__graph_perm | 50 | 9 | 0.64650 | 0.66579 | 19.10 | 14.93 |
| entropy_parent | 58 | 12 | 0.62587 | 0.61303 | 19.85 | 18.80 |

Context equal fusion reaches PB 27.67%, context IU 16.97%, context Joint 27.43%, and context Joint graph 20.19%. A larger PB score for a Joint variant does not by itself demonstrate that learning its weights helped.

Descriptive matched simple controls within each bank: all methods below use that bank's same Joint-valid PRMB answers. Different rows of this table still have different answer populations. This panel is an attribution diagnostic; the 31 registered intervals are saved separately.

| Bank | Common PRMB answers | Equal | IU | Joint lambda 0 | Graph lambda 0.1 | Permuted graph |
|---|---|---|---|---|---|---|
| moment | 7 | 0.65525 | 0.64964 | 0.66171 | 0.66255 | 0.66115 |
| context | 9 | 0.65436 | 0.64611 | 0.63826 | 0.64297 | 0.64101 |

## Coverage is not a simple increase

| Population | Valid in both banks | Rescued by context | Lost with context | Invalid in both |
|---|---|---|---|---|
| prm | 4 | 5 | 3 | 0 |
| pb_ | 33 | 8 | 3 | 2 |
| all | 37 | 13 | 6 | 2 |

Joint gains 13 valid answers and loses six, leaving 50 instead of 43. On PRMB, its old seven and new nine valid answers overlap in only four. The common-ID Joint comparison is 0.66823 (context) versus 0.72613 (moment), not the unmatched 0.63826 versus 0.66171. The exploratory difference interval is negative, but it rests on four development answers and is not a population-level confirmation.

Peak-only counts among the same 25 erroneous PB answers; these ignore the binary gate, and invalid fits still count as misses. They help separate a locator change from a change in clean/error decisions.

| Arm | Exact peak hits / 25 |
|---|---|
| moment__equal | 8 |
| moment__iu | 7 |
| moment__joint0 | 9 |
| moment__graph010 | 8 |
| moment__graph_perm | 8 |
| context__equal | 7 |
| context__iu | 8 |
| context__joint0 | 7 |
| context__graph010 | 7 |
| context__graph_perm | 7 |
| moment_allk__joint0 | 9 |
| moment_allk__graph010 | 8 |
| moment_allk__graph_perm | 8 |
| context_allk__joint0 | 7 |
| context_allk__graph010 | 7 |
| context_allk__graph_perm | 7 |
| entropy_parent | 5 |

## What the broader K search found

| Bank / roster | Selected K counts, plus blocked cases |
|---|---|
| moment_allk | {'4': 13, '3': 29, 'BLOCKED_NO_ADMISSIBLE_PARTITION': 15, '5': 1} |
| context | {'BLOCKED_NO_ADMISSIBLE_PARTITION': 8, '3': 37, '4': 13} |
| context_allk | {'BLOCKED_NO_ADMISSIBLE_PARTITION': 8, '3': 34, '4': 13, '5': 3} |

The expanded search selects K=5 in one moment-bank answer and three context-bank answers. K counts groups, not features: a three-group solution can contain many features per group. Every accepted group still has at least three coordinates. Larger K is an option, not a target or a guarantee of better localization.

The expanded moment roster leaves all reported endpoints unchanged. In the context bank it changes pooled Joint AUC from 0.63826 to 0.64808 and graph AUC from 0.64297 to 0.65161, with PB unchanged. These are small development differences; no K setting is selected by labels.

The context bank has more concentrated covariance. The mean participation rank of the normalized matrix is 3.64 for moments and 2.29 for context. This measures concentration of covariance eigenvalues; it is not a literal count of independent features or correctness information. The nominal feature count remains 27.

Both banks retain 27 active coordinates for every answer; fitting-row counts range from 13 to 176. Valid-model geometry below is descriptive and uses each family's available fits. Its off-diagonal residuals are independently recomputed from the empirical and saved model covariances; Jacobian conditions were checked in the main review. Reused identical partitions are counted once per reported family.

| Family | Valid models | Smallest group: min / median | Jacobian condition: median / max | Off-diagonal relative misfit: median [Q25, Q75] |
|---|---|---|---|---|
| moment_allk | 43 | 3 / 3.0 | 2.1 / 6.5 | 0.1783 [np.float64(0.1509), np.float64(0.2047)] |
| context | 50 | 3 / 3.0 | 2.1 / 2.8 | 0.1177 [np.float64(0.0969), np.float64(0.1672)] |
| context_allk | 50 | 3 / 3.0 | 2.1 / 2.8 | 0.1177 [np.float64(0.1019), np.float64(0.1672)] |

## Paired evidence

Exploratory, unadjusted 95% source-group intervals, 1,000 draws. PRMB uses common valid IDs; PB uses the fixed full population. Some bootstrap draws omit a class and remain undefined, with counts retained. Do not promote an isolated positive interval from this development grid.

| Left minus right | Common PRMB N | PRMB difference [CI] | Within-answer difference [CI] | PB difference, percentage points [CI] |
|---|---|---|---|---|
| context__equal minus moment__equal | 12 | 0.04370 [-0.0321, +0.1125] | -0.00560 [-0.1289, +0.0886] | 9.90 [-10.3846, +26.6009] |
| context__iu minus moment__iu | 12 | 0.03391 [-0.0411, +0.1070] | -0.01350 [-0.0995, +0.0649] | -0.74 [-15.7979, +11.4989] |
| context__joint0 minus moment__joint0 | 4 | -0.05790 [-0.2857, -0.0117] | -0.14749 [-0.2857, -0.0139] | 14.93 [-1.2416, +28.4747] |
| context__graph010 minus moment__graph010 | 4 | -0.05164 [-0.2857, +0.0088] | -0.14286 [-0.2857, +0.0000] | 11.86 [+0.0000, +24.5864] |
| context__joint0 minus context__iu | 9 | -0.00786 [-0.0431, +0.0197] | -0.01569 [-0.0536, +0.0058] | 10.46 [-6.6667, +22.2363] |
| context__graph010 minus context__joint0 | 9 | 0.00471 [-0.0088, +0.0198] | 0.01364 [+0.0013, +0.0347] | -7.24 [-16.2121, +5.7815] |
| context__graph010 minus context__graph_perm | 9 | 0.00196 [-0.0070, +0.0150] | 0.01215 [+0.0000, +0.0347] | 1.10 [+0.0000, +8.3333] |
| context_allk__joint0 minus context__joint0 | 9 | 0.00982 [-0.0026, +0.0297] | 0.00149 [+0.0000, +0.0045] | 0.00 [+0.0000, +0.0000] |

The context graph has a small positive within-answer interval versus context Joint lambda-zero, but its native PB score is lower, and its within-answer comparison to the permuted graph includes zero. This does not establish a learned-graph advantage on both tasks.

## The next short experiment

Test an explicit fallback policy using the existing fusion fits: retain moment-bank Joint when it is valid; try context-bank Joint when the first fit fails; use answer-only IU if both fail. Keep pure Joint rows with their failures visible. Compare against moment Joint with the same IU fallback and against plain IU. Include equal aggregation under the same selected-bank routing to measure whether learned fusion adds value.

This is a proposed, not yet implemented policy. Freeze eligibility, routing, gate handling and comparator budgets before evaluating it. It is motivated by 13 rescued fits and six regressions; it must not become a label-chosen winner per answer. All fitting remains within the current answer. It could make the comparison cover every answer, but improved accuracy is unproven.

Keep the two banks as separate evidence rather than replacing the old one. Feature information beyond repeated telemetry transforms, remaining temporal/geometry/sampling support, the full comparator replay, untouched two-task confirmation and historical 24-cell transfer remain open.

## Review and execution

Six scientific tests pass, including a cached Joint/permuted-graph replay. The pre-freeze smoke test caught an omitted release prefix in the graph permutation identity; it was fixed before scoring and is now guarded by an exact parent-seed assertion. A floating-point reduction-order difference was also fixed before freezing so the unchanged level columns replay bit for bit. No parent artifact was changed.

| Independent check | Count |
|---|---|
| direct_label_joins | 58 |
| permutation_identity_checks | 58 |
| raw_context_reconstructions | 58 |
| exact_level_column_replays | 58 |
| normalization_reconstructions | 116 |
| group_candidate_contracts | 1044 |
| joint_validity_checks | 143 |
| exact_parent_replays | 303 |
| span_maps | 848 |
| independent_gmm_gates | 848 |
| independent_weight_reconstructions | 545 |
| same_partition_parent_replays | 126 |
| arm_endpoint_checks | 17 |

The review independently reconstructs EMA features with a linear filter, normalization, grouping admissibility/ARI selection, native inverse weights, span mappings, GMM decisions, labels and endpoints. It reuses the unchanged source-bound nearest-neighbor graph builder and IU kernel; the Laplacian, inverse and metrics are separately reconstructed. Numerical checks establish implementation consistency, not scientific success.

Scoring: 104.5 seconds with three CPU workers. All 31 contrasts: 11.4 seconds using cached pair-count statistics on one process. The review additionally replayed three representative 1,000-draw contrasts with the old explicit implementation; all original pooled-PRMB/PB intervals and valid-draw counts match to 1e-12. That reference check took 58.0 seconds. These different workloads are not a direct speed ratio.

Maximum raw-feature, weight and span discrepancies: 2.13e-14, 9.74e-14, 2.22e-15. The separate geometry audit verifies 143 covariance residuals. All bound source and result hashes match. Scores froze before this stage decoded labels; these answers were already exposed in development. All scoring, contrasts and reviews are complete.

Historical context: the earlier 30-long-answer IU 0.70070 and Claude pooled-fit results use different populations and fitting contracts. The six current parents provide the exact bridge here. A new bank is not a new benchmark release, and this pilot is not publication confirmation.
