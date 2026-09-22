# Reasoning Localization 0.3662 — Frozen H3-Equal Llama Transfer V1

Status: `FROZEN_BEFORE_RUN`; scorer-family transfer, not fresh-question
confirmation.

## Population audit

The local sealed release contains 3,400 ProcessBench source-question groups.
Qwen3-4B, Qwen3-8B and Llama-3.1-8B each contain exactly those same 3,400
groups: pairwise intersection is 3,400 with zero model-unique groups. The
Llama labels were already opened in Phase 1. No genuinely fresh ProcessBench
question population is available locally.

Therefore this stage may test scorer-family transfer only. It must never be
reported as fresh confirmation, even though its scores are frozen before this
run imports labels.

## Frozen arms

- `P2E_H0_FAMILY6_TOP10_LLAMA4`: exact five-family H0 curve with top-ten
  reducer and one grouped Llama cross-fitted detector threshold.
- `P2E_H2_CLEAN_C7_LLAMA4`: remove sampled-token energy, remove partition
  `energy_series`, and insert frozen C7 inside entropy dynamics. It may rerank
  only H0 non-abstentions.
- `P2E_H3_EQUAL_C8_RERANK_LLAMA4`: equal within-response rank fusion of the
  frozen H2 and C8 top-ten step curves, again only on H0 non-abstentions.

The implementation must first reproduce the imported eight-Qwen H2 and
H3-equal frozen scores with maximum absolute error at most `1e-12`. No
reliability weighting, alpha search, threshold refit, new view, or task-label
selection is allowed.

## Evaluation

- Four Llama scorer cells on the same GSM8K, Math, OlympiadBench and OmniMath
  questions.
- 20,000 paired whole-source-question bootstrap draws.
- Three simultaneous macro-F1 contrasts: H2−H0, H3−H0, and H3−H2.
- Required diagnostics: exact error, within-one, clean abstention, overall
  accuracy, W/T/L and worst cell.
- Hard gates: exact Qwen score alias, H0 reconstruction, one fold per source
  group, and zero abstention mismatches for H2/H3.

This transfer can strengthen or weaken the H3 mechanism premise but cannot
earn promotion. Fresh questions remain mandatory before Phase-3 or PRMBench
promotion claims.

