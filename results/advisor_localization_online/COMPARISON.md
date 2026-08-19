# Where our method stands: localization and early detection

Generated 2026-08-17T15:12:46+00:00 from the frozen Stage-4 scorer-transfer audit.
Protocol `c921b0d446eebd46…`. Preregistered verdict of that audit: **`REGRESSES_DIRECT_COMPETITOR`**.

### Early (prefix) detection — AUROC at 64 and 128 observed tokens

Same access throughout: one generated trace, log-probabilities only, one model pass. Deltas are paired by source question across the Qwen3-8B and Llama-3.1-8B scorer copies.

Every paired delta in this grid is against **28-stream causal prefix (ours)**, the lane's direct reference — that is the comparison the audit actually computed. Our method here is **28-stream causal prefix (ours)**, which *is* the reference.

| Metric (direction) | **28-stream causal prefix (ours)** | Two-head trajectory (Step 272) | Mean entropy | DeepConf lowest-group confidence, window 64 — our proxy | Maximum entropy | Global detector + step-top-5 locator (new architecture) | DeepConf lowest-group confidence, window 32 — our proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUROC@64/128 — higher is better | **0.6104** ✅ | 0.6082 | 0.5926 | 0.5922 | 0.5921 | 0.5882 | 0.5853 |
| Paired delta vs the reference — positive is better than the reference | reference | -0.0022 | -0.0178 | -0.0182 | -0.0183 | -0.0222 | -0.0251 |
| Grouped 95% interval on that delta | — | [-0.0180, +0.0132] | [-0.0496, +0.0144] | [-0.0508, +0.0138] | [-0.0333, -0.0027] | [-0.0502, +0.0042] | [-0.0575, +0.0063] |
| Families won / lost vs the reference (of 4) | — | 2/2 | 1/3 | 1/3 | 0/4 | 1/3 | 1/3 |
| Verdict vs the reference | **reference** | parity | parity | parity | loses to reference | parity | parity |

### Error localization — ProcessBench, macro F1 over four families

Same access throughout: one generated trace, log-probabilities only, one model pass. Deltas are paired by source question across the Qwen3-8B and Llama-3.1-8B scorer copies.

Every paired delta in this grid is against **Maximum entropy + step-top-5 locator**, the lane's direct reference — that is the comparison the audit actually computed. Our method here is **Global detector + step-top-5 locator (new architecture)**, which is one of the rows judged against it.

| Metric (direction) | **Global detector + step-top-5 locator (new architecture)** | Maximum entropy + step-top-5 locator | GL-LIU replay | Maximum entropy + persistent-q90 locator | Maximum entropy + peak locator | Two-head trajectory (Step 272) | Mind the Gap / Evidence-Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| ProcessBench macro F1 — higher is better | **0.3662** ✅ | 0.3614 | 0.3364 | 0.3280 | 0.3111 | 0.3078 | 0.2646 |
| Paired delta vs the reference — positive is better than the reference | +0.0048 | reference | -0.0250 | -0.0335 | -0.0503 | -0.0536 | -0.0968 |
| Grouped 95% interval on that delta | [-0.0264, +0.0375] | — | [-0.0669, +0.0155] | [-0.0721, +0.0064] | [-0.0783, -0.0202] | [-0.0899, -0.0189] | [-0.1333, -0.0582] |
| Families won / lost vs the reference (of 4) | 3/1 | — | 1/3 | 0/4 | 1/3 | 1/3 | 0/4 |
| Verdict vs the reference | parity | **reference** | parity | parity | loses to reference | loses to reference | loses to reference |

#### Higher-access panel — not comparable to the grid above

These see more than we do: step-level supervision, or eight sampled passes of a 72B model. They are compute ceilings on the same questions, never same-access deltas.

| Method | What it sees | ProcessBench macro F1 |
|---|---|---:|
| Qwen2.5-Math-PRM-7B (step-level supervision) | step-level PRM800K supervision, 1 pass | 0.7280 |
| Qwen2.5-72B critic (8-sample vote) | no labels, 8 sampled passes of a 72B model | 0.5895 |
| Qwen3-8B judge control | no labels, 1 pass, judge prompt | 0.0913 |

## Caveats that travel with these numbers

- The DeepConf columns are OUR approximate proxies from saved log-probabilities, not the published method. The pinned official confidence is being acquired (M2) and is preregistered null N7 of the prefix-lane claim registry.
- Tier-B rows are same-question compute ceilings, not same-access deltas.
- Potential tokens remaining are not realized savings.
- The localization result is a PARITY: the point estimate favours the new architecture by +0.0048 macro F1 and its grouped interval crosses zero.

## Sources (hashed)

- `results\local_online_comprehensive_v1\STAGE_4_AGGREGATE.csv` — `c7d57e9649c00e0e…`
- `results\local_online_comprehensive_v1\STAGE_4_INTERVALS.csv` — `18dd40d71dd3ca6f…`
- `results\local_online_comprehensive_v1\STAGE_4_DECISION.json` — `e74ab8815989e03f…`
