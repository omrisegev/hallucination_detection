# DUFS31 experiment contract

Frozen before selector training, 2026-09-15. Parent program:
TEMPORAL_RESEARCH_PROGRAM_20260915.md. Requires complete 47-record PB repair PASS.

Use the existing 31 q15 definitions (`renyi_alpha_sweep.VIEW_NAMES`), no alpha
search. Orient each fusion stream to answer VE1; standardize per answer for
fitting, retain natural units for per-stream Top10 then equal-mean scoring.
All 13,769 answers enter evaluation. From each eligible training answer retain
up to 32 evenly spaced token rows for a bounded stochastic fit. Draw 8,192
observations with equal source-group probability, then equal answer probability
within group, then equal retained-row probability. These are training samples,
not an evaluation subset.

Reuse `adapted_dufs_soft_gates`: self-tuning graph, 120 optimization iterations,
seeds 0/1/2. Rank averaged survival probabilities, stable original-index ties.
Score the selected 2, 3 or 4 streams. This is the repository's DUFS adaptation,
not a claim to reproduce the original paper's optimizer.

Fit separately by cell and source fold, excluding the entire held fold.
PRMScore additionally excludes both target and calibration folds (nested fits).
Gate remains transductive tail15 Top10 midrank >= .33. Interfaces reject label
metadata. Save groups, exclusions, gates, seed variability and selected streams.

Compare all three policies to the frozen four-feature baseline with 10,000
paired source-group bootstrap draws and 98.333% primary intervals. Also retain
all fixed subsets at matching sizes and the already observed H0lim innovation
bank. Rank using the declared Pareto/tie rules. This remains reused development
data; no result is untouched confirmation.
