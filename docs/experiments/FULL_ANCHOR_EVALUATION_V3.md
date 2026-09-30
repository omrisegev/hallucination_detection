# Full anchor evaluation adapter

2026-09-07. Operational completion of pass 1 in the frozen
LOCALIZATION_FULL_BENCHMARK_V3.md; written before computing full results.
No scorer, candidate, normalization, decision or label contract changes.

Evaluate all 19 frozen arms and all 13,769 registered rows. Check score-file
hashes, exact release joins and the existing 110-answer numerical bridge.
Invalid PB decisions stay failures; invalid PRMB scores have explicit coverage.
Report PRMB pooled and within-answer AUC and eight PB cells, Q4/Q8 macros,
and the eight-cell macro. Keep native scores and routed fallbacks distinguishable.

Use 1,000 ordinary cluster-bootstrap draws with seed 2026090707. One draw
resamples all unique canonical source groups together. All answers from a
group, repeated scoring models and cross-task occurrences share that weight.
The roster of contrasts is every other arm minus dual__iu, plus graph010
minus joint0 and graph_perm within moment/context/single/dual. This is a
fixed mechanical reference/control roster, not selection from full results.
PRMB paired comparisons use each pair's common-valid answers, including its
own point estimates and mixed-answer count. PB includes all rows. Percentile
intervals are exploratory, not adjusted for multiple comparisons.

Report metric validation and source checks separately from intervals. Reuse
frozen score arrays without training. Numerically test weighted AUC with ties
against explicit replicated observations. Independent rank/decision arithmetic
and pilot replays check the aggregation. No winner or completed historical
benchmark claim: full shortlist evaluation and historical refits remain pending.
