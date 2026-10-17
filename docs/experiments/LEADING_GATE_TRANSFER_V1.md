# Leading simple gates as localization gates v1

Status before execution: **PROTOCOL FROZEN / NOT YET RUN**
Frozen on: 2026-09-14

## Question

The preceding complexity review selected entropy Top10, which changes the
readout but remains in the original entropy family. Do other leading, distinct
math-panel signals work better when they are used as gates for the same frozen
localizer?

## Frozen roster

Use the five highest-ranked conceptually distinct simple candidates from the
15-cell math single-feature screen:

1. `q15_VE1__token_top10`;
2. `entropy_native__token_top10` (also represents its numerical H1 duplicate);
3. `q15_Hinf__token_top10` (do not duplicate it with raw top-1 surprisal);
4. `q15_raw4_mean__token_top10`, the audited frozen q15 static token fusion;
5. `tail15_mass__token_top10`.

For every candidate, freeze the threshold selected on the complete math panel.
Do not tune its feature, readout, fusion, or threshold on ProcessBench. Apply a
label-free within-cell mid-rank transform and use the same frozen
`selected_q15_raw_per_view_top10` locator in all eight ProcessBench cells.

The comparison reports two different objectives:

- total-answer clean/error family-macro F1, AUROC, and AUPRC;
- official ProcessBench exact-localization macro-F1 after gating.

The existing q15 locator plus mean-entropy q=.3 gate is the operational
baseline. Each candidate-versus-baseline localization contrast uses 10,000
paired whole-source-group bootstrap draws and a family-wise 99% interval over
the five candidate comparisons.

This is explicit ProcessBench development comparison. Ranking the methods here
is permitted, but any promoted choice still requires new-data/new-model
confirmation.
