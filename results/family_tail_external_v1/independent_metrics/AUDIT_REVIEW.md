# Independent family-tail external metric audit

PASS for numerical recomputation and artifact integrity. This is the Agent A component, not a claim that the research candidate improved.

Checked 6190/6190 answer/backbone records, 53970 included steps, 30 method/cell rows, 18 registered contrast point deltas, and 35 frozen code/input hashes.

Only raw sharded predictions, their seals/freeze/lock metadata, evaluator-only annotations, and the pinned author evaluator source were used. No root METRICS, CONTRASTS, REPORT, or other reviewer output was read. No model inference or fitting was run.

Hard2Verify uses the harmonic mean of pooled correct-step and error-step recalls. Socratic uses the arithmetic mean of pooled correct-step and error-step F1. Positive means valid/correct. The author sentinel convention is preserved. Six Socratic precision/recall/F1 components were replayed for the full set and all 20 categories on every method and both backbones: maximum error 0. Hard2Verify author rounded percentages matched for all 10 methods.

The three Socratic empty steps per backbone remain included and are predicted invalid. No steps were excluded. Out-of-range annotations remain inert: 160 answer/backbone records, 204 index/backbone occurrences. These are repeated-backbone counts, not unique-source counts.

Critical script review added fail-closed integrity checks after predictions were sealed. AUDIT_AMENDMENT.json records the original/revised SHA256, exact diff and timing; metric functions are AST-identical and the primary evaluator hashes are unchanged. Do not describe the revised audit script as frozen before scoring.

| Method | Hard2Verify Qwen3-8B | Socratic Qwen3-8B | Socratic QwQ-32B |
|---|---:|---:|---:|
| B11_lsml | 0.43669543773119607 | 0.6322130421917849 | 0.6423822661085058 |
| F15_tailtie_lsml | 0.4101956789337408 | 0.59920509189056 | 0.6268971427505688 |
| F15_cov_lsml | 0.4202105658958451 | 0.6002284644826249 | 0.6115382746058667 |
| K28_cov_lsml | 0.41867847914359546 | 0.6056513909775987 | 0.6148649967558695 |
| K28_equal | 0.4208081057704189 | 0.6125041216489616 | 0.6300294199815453 |
| F15_equal | 0.42380063468359164 | 0.6112386407672286 | 0.6293652710276612 |
| B11_equal | 0.4088223552894211 | 0.6079124374368405 | 0.6150337659095937 |
| B11_partition_equal | 0.3975761101415552 | 0.6126966571557679 | 0.6210976027506199 |
| A48o_equal | 0.39697085159077916 | 0.5950371597618382 | 0.616008629521447 |
| ct7 | 0.3775100401606426 | 0.5875289335890428 | 0.6016558146303826 |

The tail-weighted family candidate has lower point estimates than B11 L-SML and its exact F15 equal control in all three cells. This audit does not estimate confidence intervals or evaluate statistical significance.

| Registered contrast (left minus right) | Hard2Verify | Socratic Qwen3-8B | Socratic QwQ-32B |
|---|---:|---:|---:|
| F15_tailtie_lsml - B11_lsml | -0.026499758797455253 | -0.03300795030122483 | -0.015485123357936947 |
| F15_tailtie_lsml - F15_equal | -0.013604955749850822 | -0.01203354887666852 | -0.0024681282770923074 |
| F15_tailtie_lsml - F15_cov_lsml | -0.010014886962104286 | -0.001023372592064864 | 0.01535886814470211 |
| F15_cov_lsml - K28_cov_lsml | 0.0015320867522496395 | -0.005422926494973779 | -0.0033267221500027366 |
| F15_equal - K28_equal | 0.002992528913172754 | -0.001265480881733061 | -0.0006641489538841139 |
| K28_equal - A48o_equal | 0.023837254179639722 | 0.017466961887123378 | 0.014020790460098298 |

Full-precision class components, category breakdowns and confusion counts are in INDEPENDENT_METRICS.json; per-answer counts are in ANSWER_COUNTS.csv; every raw prediction file SHA256 is in RAW_HASHES.json.

Source lock SHA256: `65b35336fcc2f66b7843ec040d3bdafbbaa03bb44ae5f61f1d00335abfaea5cf`.
All-cell seal SHA256: `328601645f41ad5e18fdae3ed45a0da965a9d42f6c50423d73bd20c299285fa5`.
