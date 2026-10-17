# Joint L-SML existing-data evaluation amendment R1

Registered status: `REGISTERED_POSTFREEZE_EVALUATOR_ADAPTER_AMENDMENT_R1`

## Scope

The registered v0 evaluator stopped before computing any metric because it
incorrectly required equality between the full PRMBench score roster and the
official step-label roster. This amendment changes only that join contract.
It does not refit fusion, change a score, choose a method, alter a reducer, or
open ProcessBench labels.

The failure occurred after the independent score-freeze audit had passed and
after the PRMBench label file was opened, but before any evaluation artifact
or metric was produced. The original registered evaluator remains unchanged
and hash-bound as the failed v0 implementation.

## Canonical PRMBench join

The score freeze intentionally covers all 6,966 admitted PRMBench responses
and 94,112 official spans. The canonical evaluator exposes only the 6,208
non-`correct` responses and their 83,280 labeled steps. Therefore the valid
post-freeze join is:

- both ID rosters are unique;
- the 6,208 label IDs are an exact subset of the 6,966 score IDs;
- the label order is the score-roster subsequence order;
- exactly 758 score-only responses remain;
- every selected response has the same number of score spans and label steps;
- selected step counts sum to exactly 83,280.

This is the maintained repository contract: `configs/reconstruction_benchmark_v1/localization.json`
sets `excluded_class=correct`, `expected_error_responses=6208`, and
`expected_steps=83280`; `localization_postfreeze._load_prmbench_panel` selects
`classification != correct`; and the frozen Phase-1/H3 evaluators use the same
opaque-ID subset join.

No separate target-free membership sidecar exists. The inclusion roster is
revealed only by the already authorized post-freeze PRMBench evaluator file.

## Versioning and gates

The new evaluator is `evaluate_existing_v1_r1.py`. Before metrics it writes a
versioned registry binding:

- the original execution registry, structural ledger, score manifest, and
  independent score-freeze audit;
- the original failed evaluator SHA and the R1 evaluator/test SHAs;
- the canonical PRMBench label SHA and join-contract source SHAs;
- the exact 6,966 / 6,208 / 758 / 94,112 / 83,280 counts;
- the fact that labels were opened but metrics were not computed in v0.

An independent audit must pass on this registry before R1 evaluation. Output
remains `RETROSPECTIVE_OPENED_DEVELOPMENT`, PRMBench-only. ProcessBench remains
`STRUCTURAL_NO_SCORE`; no PB label source may be opened by R1.
