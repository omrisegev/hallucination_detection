---
description: Adversarial replication of one experimental claim before it enters a report — three independent agents (recompute from raw by a different path; full-population coverage audit; shuffled-label null + math check), then a CLAIM | VERDICT | EVIDENCE table. Formalizes the Claude/Codex cross-review that caught nine real errors.
---

Argument: `$ARGUMENTS` = the claim, quoted, plus the results directory or file that
supports it. Example: `"L-SML beats equal weighting, 35.92 vs 32.59 PB" results/token_lsml_v1`.

Launch three `general-purpose` agents IN PARALLEL (one message, three tool calls). Each
gets the claim and the directory, and is told it must NOT read summary files, REPORT.html,
HISTORY.md, or the other agents' output — only raw per-response artifacts (pkl/npz/parquet)
and the code.

**Agent A — independent recomputation.** Recompute the headline number from the raw
per-response data by a different code path than the original script (a fresh 40-line
script, not an import of the original function). Report the number, the delta from the
claim, the file hashes it read, and the exact command.

**Agent B — population coverage audit.** Find every place the claim rests on a sample:
a pilot cohort, a handful of cells/selectors/seeds, a subset of folds. Re-run over the
FULL registered population. Report `n_checked / n_total` for every sub-claim, and flag
any conclusion drawn from fewer than all available units.

**Agent C — null and math check.** Re-run the same pipeline with shuffled labels and
with one permuted feature; report whether the effect survives and the null's spread.
Then check the math: is any identity used additively that is multiplicative, rank-1
where the code assumes full rank, a sign gauge that makes a "repair" a no-op, or a
`max(a, 1−a)` that can only inflate? Derive symbolically and compare to the code.

When all three return, write:

```
| CLAIM | VERDICT (confirmed / weakened / refuted) | EVIDENCE (agent, number, n_checked/n_total, file) |
```

Rules:
- Be adversarial, not agreeable. If the number is wrong, say so in the first line.
- A verdict of "confirmed" requires all three agents to agree; otherwise "weakened"
  with the disagreement spelled out.
- Append the table to `results/<dir>/RED_TEAM.md` and, if the verdict is not
  "confirmed", add a LESSONS.md entry describing what the original run missed.
