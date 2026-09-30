# Joint L-SML with an explicit IU fallback

The fallback gives a score for all 58 development answers. Original Joint with IU fallback improves both headline point estimates over original IU. It does not establish a winner: context equal fusion has higher pooled PRMB and PB points, and the paired intervals against IU include zero.

## Routing

Single: moment Joint -> moment IU. Dual: moment Joint -> context Joint -> moment IU. Use fit validity only; a no-error prediction or readout failure does not trigger a new route.

## Matched results: all 12 PRMB and all 46 PB answers

These rows all have full fit and decision coverage. PRMB pooled AUC and average within-answer AUC answer different questions. Only 10 of the 12 PRMB answers contain both classes and contribute to the within-answer average.

The always-context equal control is essential. Its higher headline points prevent a claim that the current learned fusion is best. Joint has a higher within-answer mean, so the comparison is not uniform across metrics.

| Method | PRMB pooled AUC | Within-answer AUC | PB macro F1 |
| --- | --- | --- | --- |
| Original IU | 0.62261 | 0.67567 | 17.71% |
| Original equal fusion | 0.61717 | 0.66742 | 17.76% |
| Context equal fusion | 0.66087 | 0.66182 | 27.67% |
| Original Joint -> IU | 0.63663 | 0.68798 | 27.22% |
| Original graph 0.1 -> IU | 0.63478 | 0.68571 | 14.94% |
| Permuted graph -> IU | 0.63707 | 0.68763 | 15.82% |
| Original Joint -> context Joint -> IU | 0.62859 | 0.69614 | 27.22% |

## What the second bank actually changes

Dual routes seven PRMB answers through original Joint and five through context Joint. For PB the counts are 36 original Joint, eight context Joint and two IU.

Compared with the single policy, each dual Joint variant changes four PB predictions and three peak locations. Every answer retains the same exact-success status: changed predictions are still wrong. This is why the PB metric and its paired difference interval are unchanged.

Dual Joint without a graph has a lower pooled PRMB AUC than single Joint, but a higher within-answer mean. Neither difference establishes a reliable advantage on this small, repeatedly inspected sample.

## The PB improvement has a tradeoff

PB is the average across four subsets of the harmonic mean of clean-answer accuracy and exact first-error accuracy. It is not the fraction of all answers predicted correctly.

Original Joint -> IU finds two more exact errors than IU, but recognizes four fewer clean answers. Its total exact successes fall from 16 to 14 of 46, while the registered PB macro F1 rises. Report both facts; this is not improvement on every answer type. Peak hits ignore the no-error gate and are diagnostic only.

| Method | Clean hits / 21 | Exact error hits / 25 | Error peak hits / 25 |
| --- | --- | --- | --- |
| Original IU | 12 | 4 | 7 |
| Original Joint -> IU | 8 | 6 | 9 |
| Original graph 0.1 -> IU | 9 | 3 | 8 |

## PB subset results

Joint -> IU improves GSM8K and OlympiadBench, ties MATH and worsens Omni-MATH relative to IU.

| Subset | IU | Joint -> IU | Graph -> IU |
| --- | --- | --- | --- |
| pb_gsm8k_q8 | 0.00% | 33.33% | 0.00% |
| pb_math_q8 | 33.33% | 33.33% | 23.08% |
| pb_olympiadbench_q8 | 0.00% | 22.22% | 16.67% |
| pb_omnimath_q8 | 37.50% | 20.00% | 20.00% |

## Paired uncertainty: registered comparisons

These are 95% percentile intervals for left minus right, using 1,000 source-group draws stratified by cell. They are exploratory and unadjusted for multiple comparisons and repeated development. The displayed full-coverage pairs have 1,000 defined PRMB draws and 982 defined PB draws.

Original Joint -> IU versus IU changes PRMB AUC by +0.01402 and PB by +9.51 percentage points; both intervals include zero. The graph still has no established advantage over lambda zero or the permuted graph.

| Pair | PRMB AUC difference CI | Within-answer difference CI | PB difference CI |
| --- | --- | --- | --- |
| single__joint0 minus moment__iu | [-0.0077, +0.0270] | [-0.0089, +0.0510] | [-9.3750, +20.5011] pp |
| single__joint0 minus moment__equal | [-0.0184, +0.0569] | [-0.0321, +0.0717] | [-10.9395, +21.5000] pp |
| dual__joint0 minus single__joint0 | [-0.0602, +0.0381] | [-0.0292, +0.0522] | [+0.0000, +0.0000] pp |
| dual__joint0 minus dual__equal | [-0.0245, +0.0222] | [-0.0452, +0.0477] | [-10.9395, +21.5000] pp |
| dual__graph010 minus dual__joint0 | [-0.0071, +0.0183] | [-0.0056, +0.0274] | [-23.5655, +3.1486] pp |
| dual__graph010 minus dual__graph_perm | [-0.0118, +0.0165] | [-0.0064, +0.0262] | [-7.8018, +5.2632] pp |

## Review finding: include the stronger simple incumbent

The original 30-pair roster contained routed equal controls but omitted a paired comparison with always-context equal. Review added these two comparisons in a separate, explicitly post-evaluation artifact. The original protocol and 30 contrasts are unchanged.

These additional intervals also include zero for the two headline endpoints. No candidate is promoted from them.

| Post-evaluation comparison | PRMB AUC difference CI | PB difference CI |
| --- | --- | --- |
| single__joint0 minus context__equal | [-0.0874, +0.0344] | [-16.8262, +11.7157] pp |
| dual__joint0 minus context__equal | [-0.0791, +0.0239] | [-16.8262, +11.7157] pp |

## All 25 arms, including all 17 previous anchors

Pure Joint rows keep their fit failures. Their PRMB AUC uses fewer answers and cannot be compared directly with a full-coverage row as an algorithmic gain. PB includes all 46 answers and counts invalid decisions as failures. Expanded-K arms are unchanged references, not additional routing candidates.

The last column holds the same answer-fitted parent-IU binary gate fixed for every method. It is a diagnostic, separate from the primary native-gate decision.

| Arm | Fit coverage | PRMB answers | Pooled AUC | Within AUC | PB native | PB decision coverage | PB common-IU gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| moment__equal | 58/58 | 12 | 0.61717 | 0.66742 | 17.76% | 46/46 | 20.83% |
| moment__iu | 58/58 | 12 | 0.62261 | 0.67567 | 17.71% | 46/46 | 17.71% |
| moment__joint0 | 43/58 | 7 | 0.66171 | 0.69200 | 12.50% | 36/46 | 8.33% |
| moment__graph010 | 43/58 | 7 | 0.66255 | 0.68745 | 8.33% | 36/46 | 8.33% |
| moment__graph_perm | 43/58 | 7 | 0.66115 | 0.69131 | 4.17% | 36/46 | 8.33% |
| context__equal | 58/58 | 12 | 0.66087 | 0.66182 | 27.67% | 46/46 | 18.80% |
| context__iu | 58/58 | 12 | 0.65652 | 0.66217 | 16.97% | 46/46 | 20.19% |
| context__joint0 | 50/58 | 9 | 0.63826 | 0.66430 | 27.43% | 41/46 | 14.93% |
| context__graph010 | 50/58 | 9 | 0.64297 | 0.67794 | 20.19% | 41/46 | 14.93% |
| context__graph_perm | 50/58 | 9 | 0.64101 | 0.66579 | 19.10% | 41/46 | 14.93% |
| moment_allk__joint0 | 43/58 | 7 | 0.66171 | 0.69200 | 12.50% | 36/46 | 8.33% |
| moment_allk__graph010 | 43/58 | 7 | 0.66255 | 0.68745 | 8.33% | 36/46 | 8.33% |
| moment_allk__graph_perm | 43/58 | 7 | 0.66115 | 0.69131 | 4.17% | 36/46 | 8.33% |
| context_allk__joint0 | 50/58 | 9 | 0.64808 | 0.66579 | 27.43% | 41/46 | 14.93% |
| context_allk__graph010 | 50/58 | 9 | 0.65161 | 0.67794 | 20.19% | 41/46 | 14.93% |
| context_allk__graph_perm | 50/58 | 9 | 0.64650 | 0.66579 | 19.10% | 41/46 | 14.93% |
| entropy_parent | 58/58 | 12 | 0.62587 | 0.61303 | 19.85% | 46/46 | 18.80% |
| single__joint0 | 58/58 | 12 | 0.63663 | 0.68798 | 27.22% | 46/46 | 21.88% |
| single__graph010 | 58/58 | 12 | 0.63478 | 0.68571 | 14.94% | 46/46 | 18.80% |
| single__graph_perm | 58/58 | 12 | 0.63707 | 0.68763 | 15.82% | 46/46 | 18.80% |
| dual__joint0 | 58/58 | 12 | 0.62859 | 0.69614 | 27.22% | 46/46 | 21.88% |
| dual__graph010 | 58/58 | 12 | 0.63217 | 0.70339 | 14.94% | 46/46 | 18.80% |
| dual__graph_perm | 58/58 | 12 | 0.63163 | 0.69698 | 15.82% | 46/46 | 18.80% |
| dual__equal | 58/58 | 12 | 0.62435 | 0.69777 | 17.76% | 46/46 | 20.83% |
| dual__iu | 58/58 | 12 | 0.62587 | 0.68383 | 17.71% | 46/46 | 17.71% |

## What was verified

Five scientific routing tests pass: all eligibility cases, clean/no-error behavior, failed readouts, exhausted fallbacks, exact source copying and invalid-source rejection.

Independent review reconstructs the routing truth table and metrics without importing the composite implementation or evaluator. It verifies the provenance of previously audited parent fits, without refitting them. Three registered 1,000-draw comparisons are independently reconstructed by explicitly resampling rows, including all four endpoint intervals and defined-draw counts.

A Windows file-access error interrupted the first contrast checkpoint write. The process exited, and the identical frozen runner resumed its completed checkpoints; all 30 contrasts now exist and pass review. No scientific code or frozen parent artifact was changed.

| Check | Count |
| --- | --- |
| direct_label_joins | 58 |
| independent_route_pairs | 58 |
| exact_source_metadata | 1450 |
| exact_score_array_replays | 2624 |
| peak_and_common_gate_checks | 1312 |
| independent_metric_bundles | 25 |
| unchanged_parent_bundles | 17 |
| paired_point_bundles | 30 |

## Cost and fitting scope

Composing the cached score bundles took 2.84 seconds on one CPU process. This excludes the earlier feature, grouping, graph and fusion fits, and is not an end-to-end speed claim. The resumed contrast invocation took 10.02 seconds; that excludes work in the interrupted invocation.

Every fitted quantity comes from the same answer under the previously declared rules and negative-entropy anchor. There is one model generation pass and offline processing of saved gray-box telemetry. The routed simple controls still use Joint eligibility and therefore do not avoid its fitting cost.

## The next useful step

Keep Original Joint -> IU as a viable fusion candidate, retain IU and both equal-fusion banks, and retain lambda-zero and permuted-graph controls. The dual bank improves Joint coverage but does not justify replacing the simpler route.

Before more tuning on these 58 answers, audit exposure and source-group overlap, then freeze a small disjoint-group replication with the same recipes and evaluator. The existing release explicitly says development_previously_evaluated_by_v2. A different subset from it is development replication, not an untouched publication test.

Joint-specific features/grouping, further graph hypotheses, supporting temporal/geometry/sampling methods, the wider comparator registry, untouched two-task confirmation and the requested 24-cell transfer remain open. This stage does not complete the research goal.
