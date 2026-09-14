# Simple gate choice v1 — report

Status: **COMPLETE / REVIEW PASS**

## Math development choice

| Candidate | Signals | q | Family macro-F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Native H1 entropy, answer Top10 | 1 | 0.45 | 0.633271 | 0.791377 | 0.775099 |
| Frozen q15 static token fusion, answer Top10 | 4 | 0.45 | 0.632755 | 0.787228 | 0.759507 |

Decision: select `entropy_native__token_top10` at q=0.45. It is both
simpler and slightly better on every reported math-panel metric. The earlier
three-feature equal-mean result is retained as an ablation only and is not the
promoted gate.

## Exact reuse audit

`q15_raw4_mean` reconstructs the frozen
`original_static_fusion_before_top10` localization definition after the
registered per-step Top10 readout with maximum absolute discrepancy
0. The answer candidate
changes only the final readout to one Top10 over the complete response. This is
not the current per-view-Top10 finalist, for which Top10 occurs before fusion.

## Frozen ProcessBench transfer

| Gate on frozen q15 locator | q source | Answer macro-F1 | AUROC | PB macro-F1 | Clean accuracy | Error exact |
|---|---|---:|---:|---:|---:|---:|
| Entropy Top10 | Math q=0.45 | 0.693567 | 0.792620 | 0.355339 | 0.732824 | 0.219721 |
| Existing entropy mean | PB q=.30 baseline | 0.649999 | 0.742301 | 0.366201 | 0.500848 | 0.276677 |

Localization delta is -1.086 percentage
points; the conservative 98.75% paired
whole-source-group interval is [-2.946, +0.769] points.

Relative to the old mean-entropy q=.3 gate, the transferred gate removes
698 clean false alarms
and adds 151, but newly
closes 797 erroneous answers
while reopening 328. It gains
85 exact error
localizations and loses 338.

No feature, method, fusion, or q was selected on ProcessBench. The within-cell
mid-rank transform is label-free. This is development transfer evidence and
still requires confirmation on new data or a new model.
