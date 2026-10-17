# Advisor page v11 and v12 (2026-10-12)

Published page: https://claude.ai/artifact/NQNTYpFqVyx6xRnRS4NENV

| File | What it is |
|---|---|
| `published_v11_2026-10-12.html` | Page as published on 2026-10-12 06:22 UTC (version 11) |
| `published_v12_2026-10-12.html` | Page as published on 2026-10-12 (version 12) |
| `V11_REVIEW_2026-10-12.md` | Claude review of the Codex v11 package (no number mismatches, ~400 cells) |

Build chain (input: `advisor_review_v11_github.html` from `codex/advisor-v11-completion` @ 07cb7a4,
`results/advisor_completion_2026-10-11/`):

1. `add_v11_takeaways.py <in> <out>`: 2-3 takeaway bullets per figure.
2. `add_model_size_section.py <data_dir> <in> <out>`: Qwen3-4B vs 8B section.
   `<data_dir>` = `results/qwen3_4b_vs_8b_2026-10-12/`.
3. `finalize_v11_for_artifact.py <in> <out>`: notice, footer, evidence links to GitHub. Output = v11.
4. `add_open_direction.py <v11> <v12>`: adds the section "Open direction: how many steps to flag per answer"
   (task list `PAGE_REVIEW_TASKS_2026-10-11.md` @ cbb498e, Omri's decision of 2026-10-12: existing results
   only, no new experiments). Numbers come from `count_structure_check_v1` (17aa1e4) and
   `codex/label-free-error-count-v1` (6a0b870). The MoM quota section moves after "Next steps".

A Codex rebuild of the base page must rerun steps 1-4, or the Claude additions are lost.
Rule: before you update the artifact, check that the live page equals the last `published_*.html` copy; merge, do not overwrite.
