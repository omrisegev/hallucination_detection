# Frozen-24 convenience leaderboards

These CSVs are deterministic exports of the validated DuckDB views behind the
full frozen-24 visual report. They are presentation conveniences; the report's
paired contrasts remain the inferential source of truth.

- `cell_leaderboard.csv`: exact dataset–model cells.
- `dataset_leaderboard.csv`: equal-cell dataset aggregates.
- `slice_leaderboard.csv`: registered domain and model-family slices.
- `task_leaderboard.csv`: the one registered final-answer-detection task.
- `release_leaderboard.csv`: the frozen 24-cell release macro.

The historical v2 bridge stored the single task macro at release level, so
`task_leaderboard.csv` is an exact byte-for-byte alias of
`release_leaderboard.csv`; no value or ranking was transformed. The other four
files are direct exports from the current validated views rebuilt from the
same registry and tidy source tables.

Rows are rankable only inside their exact `comparison_group_id`.
`point_leader` means the largest point estimate in that group, while
`uncertainty_tie` is only marginal-interval overlap. Use the registered paired
contrast rows in the full report before making an inferential claim.
