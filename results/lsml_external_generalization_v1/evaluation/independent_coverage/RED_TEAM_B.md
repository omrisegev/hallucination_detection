# Independent full-population coverage audit — 2026-09-24

**PASS: all 6,190 registered answer/backbone records were checked directly.**
This is an execution, provenance and coverage verdict. No external METRICS,
CONTRASTS, reports or other agents' result summaries were read, and no quality
metric was computed. The audit started only after explicit parent GO and the
three-cell prediction seal. It used one CPU and took 284.4 seconds.

| Cell | Answers checked / total | Included steps | Nonempty scored steps | Native local fits | Local fallbacks | Matched-native included steps |
|---|---:|---:|---:|---:|---:|---:|
| Hard2Verify / Qwen3-8B | 200 / 200 | 1,860 | 1,860 | 200 | 0 | 1,860 |
| Socratic / Qwen3-8B | 2,995 / 2,995 | 26,055 | 26,052 | 2,984 | 11 | 25,967 |
| Socratic / QwQ-32B | 2,995 / 2,995 | 26,055 | 26,052 | 2,984 | 11 | 25,967 |

All seven arms have identical answer and step coverage. All 53,970 step positions
and 5,399,900 answer tokens were accounted for. The 6,190 model-answer records
represent 3,195 unique answers and 1,844 source groups; the two Socratic backbones
reuse the same question identities.

The six empty model-step instances (three Socratic steps scored by two backbones)
remain in the primary complete-policy population. Every arm has a null score and
prediction 0 at these positions. They are excluded only from nonempty-score
diagnostics. For each of the eighteen registered contrasts, the full, disjoint and
matched-native comparison populations coincide across the two compared arms.

All local fallbacks report insufficient or invalid fitting observations and use
chosen surprisal for the three local arms. Fallback scores are exactly identical
across those arms. Arm-specific source-calibrated thresholds create one different
fallback-step decision for local L-SML versus local partition-equal on
Socratic/Qwen3, and zero for the other local/equal fallback comparisons. Local
L-SML versus CT7 differs on 15 fallback steps in each Socratic cell; CT7 is a
different locator. These differences are not evidence of a successful local fit.

The matched-native diagnostic uses the same local-fit/nonempty mask for every
arm, including frozen controls and CT7. It contains 79 source groups for
Hard2Verify and 1,763 for each Socratic backbone. It is conditional evidence,
not a replacement for the full-population panel.

The pre-evaluation population artifacts were rehashed against their earlier
independent hashes. Observed-disjoint populations remain 200 answers / 79 groups
for Hard2Verify and 2,553 answers / 1,514 groups / 22,179 included steps per
Socratic cell. The excluded Socratic population is 442 source-linked answers,
including 31 variants that direct text matching alone would miss.

| Claim | Verdict | Evidence |
|---|---|---|
| Every registered record has all seven prediction arms | Confirmed for coverage | 6,190 / 6,190 UID sets, complete shard and seal agreement; `AUDIT.json` |
| Predictions use the recorded frozen package, bundle and analysis | Confirmed for recorded identities | All package/source hashes; five analysis/prose hashes; source-input hashes; three seals and all-cell analysis binding |
| Scores came from the recorded collected telemetry | Confirmed for local records | SHA256, UID, collection identity, spans and token counts checked for every raw record; per-cell `*_HASHES.json` |
| Empty steps and local failures are silently dropped | Refuted | Full inclusion and explicit null/0 policy; 11 matched fallback answers per Socratic cell |
| All external underlying problems are unseen | Not established | Exact-hash/source-component audit only; PRMB original/modified question coverage and semantic overlap remain limited |
| This coverage audit proves a quality advantage or supervised-model superiority | Not established | No quality metrics or supervised comparator inference were evaluated by this audit |

Primary machine-readable evidence: `AUDIT.json`, `ANSWER_COVERAGE.json`, three
`*_HASHES.json` ledgers and `POPULATION_OUTPUT_CHECK.json`. The registered package
was not modified. Exact executed command:

```text
python results/lsml_external_generalization_v1/evaluation/independent_coverage/audit_coverage.py --authorized-go
```

`AUDIT.json` SHA256:
`bbf99b128269b4a53559e3ffdf0258e817b969a7d272d6a71335b2bdbb9627dc`.

Broader scientific requirements remain separate: independent metric/fidelity/null
checks, assessment of ranking versus threshold transfer, any supervised-comparator
claim, and historical final-answer transfer. This audit provides full registered
external coverage rather than a feasibility-only sample; it does not establish
those other claims.
