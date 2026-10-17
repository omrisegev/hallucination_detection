# Reasoning Localization 0.3662 — H3 Llama Transfer Results V1

Status: `COMPLETE / PROMISING_UNCONFIRMED / NO_PROMOTION`; scorer-family
transfer, not fresh-question confirmation.

All score-firewall gates passed. The exact Qwen H0-combined, H2 and H3 scores
were reproduced with maximum absolute error `0`; H2 and H3 made zero changes
to H0 abstention decisions; every source group remained inside one fold.

| Arm | Macro F1 | Exact error | Within one | Clean abstention |
|---|---:|---:|---:|---:|
| H0 family6/top-ten | 0.348909 | 0.253385 | 0.453610 | 0.596646 |
| H2 cleanup plus C7 | **0.355583** | **0.261875** | 0.463071 | 0.596646 |
| H3 equal plus C8 | 0.353281 | 0.259234 | **0.467414** | 0.596646 |

The frozen simultaneous macro-F1 contrasts are:

- H2−H0: `+0.006674 [-0.007091,+0.020943]`, 2/0/2 family W/T/L,
  worst family `-0.010019`.
- H3−H0: `+0.004372 [-0.009677,+0.018452]`, 2/0/2,
  worst family `-0.021595`.
- H3−H2: `-0.002303 [-0.011662,+0.007001]`, 2/0/2,
  worst family `-0.012481`.

Both candidate-versus-H0 point estimates are positive, but both intervals
cross zero. They are `PROMISING_UNCONFIRMED`, not rejected. H2 is the raw-best
Llama arm, and H3 does not show incremental macro-F1 value over H2. H3 does
raise the secondary within-one metric versus H0 by `+0.013804
[+0.000811,+0.027030]` with 4/0/0 family W/T/L, but this unadjusted secondary
diagnostic cannot reverse the primary verdict.

The H3 worst-family delta also misses the `-0.020` robustness boundary by
`0.001595`, while remaining above the `-0.030` hard-stop boundary. The result
therefore narrows any future fresh-question study to the frozen H0→H2→H3
ladder: H2 must be retained as a separate parent, and C8 must prove incremental
value over H2 rather than inherit the Qwen development gain.

Canonical artifacts are under
`results/reasoning_localization_03662_v1/phase_2/transfer/h3_llama4/`. The
standalone result plot is `evaluation/H3_LLAMA_TRANSFER_RESULTS.svg`; the same
three contrasts are rendered in the living report.
