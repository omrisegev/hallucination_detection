# L-SML gate/locator research v1

Development-only; no untouched confirmation.

## Mind-the-Gap-style raw SLA

Raw SLA evaluates exact first-error localization only on erroneous ProcessBench
traces, before the answer-level gate. It is not the same quantity as gated
exact-error accuracy or end-to-end ProcessBench macro F1.

| Locator | Raw SLA pooled | Raw SLA equal-cell all-8 | Qwen-4B macro | Qwen-8B macro | Gated exact, pooled |
|---|---:|---:|---:|---:|---:|
| digit025 | 38.1810% | 40.9643% | 41.1784% | 40.7501% | 31.0221% |
| L08 Continuous L-SML | 38.9689% | 41.4562% | 41.9567% | 40.9557% | 31.8325% |
| Mind the Gap Shannon Drop, published | n/a | n/a | 39.1375% | 39.3925% | n/a |

L08 is +2.8192pp over the published Mind the Gap Qwen-4B equal-cell macro and
+1.5632pp on Qwen-8B. This is published context only: exact trace/generation
identity has not yet been verified, so it is not a matched-replay superiority
claim. Per-cell values and counts are in `SLA_METRICS.json`.

Reporting decision: future localization results report all three lanes together:
raw SLA, gated exact-error accuracy, and ProcessBench macro F1 including clean
trace abstention.

## Locator cross-fitted candidates

| Candidate | PB | Within | Effective rank |
|---|---:|---:|---:|
| L08_atlas_diverse_continuous | 43.7402% | 0.778143 | 2.27 |
| L07_bank6_plus_tcn1_continuous | 42.4688% | 0.768976 | 2.06 |
| L10_bank6_plus_tcn4_continuous | 42.0340% | 0.755053 | 2.32 |
| L10_incumbent_plus_continuous | 41.9421% | 0.766449 | 2.11 |
| L06_bank6_continuous | 40.9854% | 0.764108 | 1.95 |
| L14_valid_fixed_groups_continuous | 39.7865% | 0.754955 | 2.22 |
| L14_valid_joint | 39.2785% | 0.751434 | 2.22 |
| LALL24_continuous | 38.9074% | 0.750769 | 1.88 |

## Gate cross-fitted candidates

| Candidate | PB with frozen locator | Gate AUC | False open | False close |
|---|---:|---:|---:|---:|
| G13_joint_virtuals_plus_digit | 43.0586% | 0.8143 | 0.3779 | 0.1749 |
| G06_continuous | 42.4790% | 0.8141 | 0.3791 | 0.1756 |
| G12_valid_joint_no_digit | 42.0588% | 0.7931 | 0.3961 | 0.1846 |
| GALL22_continuous | 42.0063% | 0.7960 | 0.3957 | 0.1844 |
| G10_continuous | 41.8908% | 0.7998 | 0.3914 | 0.1821 |

## Frozen 2x2 interaction

| Locator | Gate | PB | Within |
|---|---|---:|---:|
| digit025 | current_tail15_digit_rate | 43.2546% | 0.776036 |
| digit025 | G13_joint_virtuals_plus_digit | 43.1431% | 0.776036 |
| L08_atlas_diverse_continuous | current_tail15_digit_rate | 43.7402% | 0.778143 |
| L08_atlas_diverse_continuous | G13_joint_virtuals_plus_digit | 43.0586% | 0.778143 |

Interaction delta (PB): -0.5700 pp.

## Stability

Locator selections: `['L08_atlas_diverse_continuous', 'L08_atlas_diverse_continuous', 'L08_atlas_diverse_continuous', 'L08_atlas_diverse_continuous', 'L08_atlas_diverse_continuous']`.
Gate selections: `['G13_joint_virtuals_plus_digit', 'G13_joint_virtuals_plus_digit', 'G13_joint_virtuals_plus_digit', 'G13_joint_virtuals_plus_digit', 'G13_joint_virtuals_plus_digit']`.

## Finalist diagnostics

Nested leave-one selection: `['L08_minus_5_top8', 'L08_minus_3_top10', 'L08_minus_3_top10', 'L08_minus_3_top10', 'L08_minus_5_top8']`; frozen diagnostic finalist: `nested_unstable_stitched`.

Stable L08 paired source bootstrap: PB delta +0.4856 pp [-0.5679, +1.5266]; within delta +0.002107 [-0.000555, +0.004851].

Paired source bootstrap (10,000 draws): PB delta +0.5632 pp [-0.4746, +1.6031]; within delta +0.001446 [-0.001238, +0.004160].

| Ablation | PB | Within |
|---|---:|---:|
| L08_minus_3_top10 | 44.0238% | 0.778341 |
| L08_minus_4_top10 | 43.9959% | 0.779746 |
| L08_minus_5_top8 | 43.9392% | 0.776875 |
| L08_minus_6_top10 | 43.8041% | 0.776880 |
| L08_minus_1_top1 | 43.3381% | 0.777046 |
| L08_minus_7_top10 | 43.1384% | 0.774509 |
| L08_minus_0_top2 | 42.8828% | 0.776793 |
| L08_minus_2_top10 | 42.5910% | 0.772545 |
| L08_equal | 42.5287% | 0.774339 |

## Simplification result

The five source folds recovered the same L08 grouping: digit `[0,1]`,
VE0.75-innovation plus mass-above `[2,7]`, and the probability/Renyi/tail group
`[3,4,5,6]`. Replacing learned weights and group discovery with fixed
family-equal weights `[1/6,1/6,1/6,1/12,1/12,1/12,1/12,1/6]` produced
**43.7745% PB / 0.778222 within**. This is a post-hoc simplification candidate,
not a confirmed successor; no further selection run was opened.

Joint gate promotion is blocked independently of quality: its multistart audit
was `BLOCKED` in all five folds. The incumbent gate therefore remains frozen.
