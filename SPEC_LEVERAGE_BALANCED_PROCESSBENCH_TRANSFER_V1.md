# Specification: frozen leverage-balanced IU ProcessBench transfer v1

**Date:** 2026-08-12

**Status:** formula frozen before this transfer evaluation.

## Purpose

Test whether the unchanged label-free Leverage-Balanced Contribution-Subspace
IU rule transfers from answer correctness on the 24-cell QA/math development
bundle to ProcessBench reasoning-error presence.

This is an external-task transfer test, not a pristine prospective benchmark:
ProcessBench labels have been used by earlier project studies.  The leverage
formula, family registry, and `1/G` scale were not developed on ProcessBench,
and no ProcessBench label may select or alter them here.

## Cells and split

Eight frozen full-N cells:

- models: Qwen3-4B and Qwen3-8B;
- subsets: GSM8K, MATH, OlympiadBench, and OmniMath.

The primary transfer subset is the six cells previously designated as
confirmation/model-transfer cells by GL-LIU v1:

- Qwen3-4B OlympiadBench and OmniMath;
- all four Qwen3-8B cells.

Qwen3-4B GSM8K and MATH are reported separately as historical development
cells.  The two model sizes share underlying ProcessBench examples, so
uncertainty must aggregate complete dataset subsets rather than treating all
eight cells as independent.

## Frozen inputs and method

For each cell, reconstruct the exact global mixed-v2 feature contract used by
GL-LIU:

1. extract the already-registered one-pass full-trace features;
2. apply the fixed availability, constant, and saturation rules;
3. apply `dufs-liu-mixed-v2-development-2026-08-07`;
4. fit ordinary two-component IU-PCR;
5. apply the unchanged leverage-balanced contribution rule from
   `SPEC_LEVERAGE_BALANCED_CS_IU_V1.md`.

No extra model call, feature, white-box signal, step boundary, or correctness
label enters score construction.  The risk score is the negative of the
correctness-oriented IU/LB score.

Mechanism controls use the same frozen `1/G` trust scale:

- uniform residual direction;
- provenance-family cardinality balancing;
- reversed leverage direction.

The existing hashed `global_mixed_v2_dufs` score from
`results/processbench_latent_state_v1/` is the DUFS-LIU incumbent.  Its score
file and row IDs must verify before evaluation.

## Fit/report separation

`fit` may inspect trace telemetry and alignment diagnostics but must not access
`row["label"]` or `row["final_answer_correct"]`.  It writes per-cell scores,
effective weights, and hashes.

`report` verifies source, cache, score, incumbent, and row-alignment hashes
before opening labels.  Primary target:

```text
reasoning_error_present = row["label"] != -1
```

Final-answer incorrect is a secondary target only.

## Transfer gates

On the six confirmation cells and the primary reasoning-error target:

1. leverage-balanced IU improves ordinary IU in cell macro;
2. equal-subset bootstrap interval for LB minus IU has lower bound above zero;
3. at least four of six cells improve;
4. worst cell is no lower than -1.0pp;
5. LB exceeds the frozen DUFS-LIU incumbent in cell macro;
6. effective-weight reconstruction and IU-score reproduction errors are below
   `1e-10` in every cell.

Uniform/cardinality controls determine attribution but are not tuned or used
for candidate selection.  A failure is retained as a transfer boundary; no
ProcessBench-specific scale or family rule may be fitted afterward.
