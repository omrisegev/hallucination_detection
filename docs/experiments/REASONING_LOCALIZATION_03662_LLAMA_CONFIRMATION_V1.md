# Reasoning Localization 0.3662 — C7/C8 Llama Transfer Confirmation V1

## Purpose and evidence boundary

This bounded stage follows the closed C1--C8 Qwen development family. It asks
whether the two positive-but-unconfirmed arms, C7 standardized EDIS onset and
C8 IU29 plus self innovation, retain value on the four frozen Llama-3.1 8B
ProcessBench scorer cells. These cells use the same GSM8K, Math,
OlympiadBench, and OmniMath source questions as the Qwen panel. They are a
scorer-family transfer, not unseen-question fresh confirmation, and all claims
must use `TRANSFER` evidence language.

## Frozen roster

- `P2C_C7_EDIS_LLAMA4`: exact C7 standardized onset scorer from Phase 2.
- `P2C_C8_INNOV_LLAMA4`: exact C8 29-stream self-lag residual augmentation
  and ordinary two-component IU-PCR from Phase 2.
- Required references: entropy top-ten and entropy top-five under the same
  Llama-only threshold/evaluator contract.
- Mechanism comparators: frozen Phase-1 family6/top-five and, for C8, the
  original-only IU29/top-ten matched parent.

No source, threshold, reducer, residual definition, EDIS threshold, fusion
weight, or orientation may be selected using Llama labels. Every arm receives
its own deterministic grouped five-fold threshold only after all step scores
are frozen. The response detector remains `equal_feature_mean` and all
comparisons use identical rows, source groups, spans, folds, and bootstrap
draws.

## Inference and promotion

The primary family contains four macro-F1 contrasts: C7 and C8 versus both
entropy top-ten and entropy top-five. It uses 20,000 paired whole-source-
question draws and Bonferroni simultaneous percentile intervals. Promotion to
PRMBench/Phase-3 eligibility requires, against both required references:

1. point delta at least `+0.005`;
2. simultaneous interval lower bound strictly above `+0.005`;
3. at least three of four scorer cells nonnegative;
4. worst-cell delta at least `-0.020`;
5. exact-error and clean-abstention deltas each at least `-0.010`;
6. no provenance, alias, suffix-invariance, leakage, or single-class failure.

An interval crossing zero is `PROMISING_UNCONFIRMED` or `INCONCLUSIVE`, not a
generic rejection. A supported material loss or hard technical/robustness
failure rejects the transfer. Family6 and IU29-parent contrasts are paired
diagnostics and cannot replace the two required references.

## Complementarity audit and future fusion boundary

After both transfer runs close, report deterministic exact-decision overlap
for C7 vs family6, C8 vs family6, and C7 vs C8: both correct, left-only,
right-only, and neither. Also report score-rank correlation and a clearly
labeled oracle-union ceiling. These are mechanism diagnostics, not accessible
routers and not permission to tune a fusion on Llama labels.

Only a transfer survivor may enter the previously registered Phase-3
survivor-only ladder. If a survivor exists, the first future candidate is a
fixed equal-family fusion with family6; ordinary IU follows only after that,
and at most one conditional mechanism may be opened. All inner/outer fits are
calibration-only. If neither arm survives, no fusion or PRMBench transfer is
opened automatically; the complementarity audit may motivate a separately
registered fresh-population experiment.
