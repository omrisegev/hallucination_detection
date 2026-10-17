# Direct probability fusion v2: selected token and residual tail

**Frozen before outcomes:** 2026-09-10
**Scope:** cached gray-box output probabilities only. No hidden states, attention,
labels in fusion fitting, or new model inference.

## One question

The v1 experiment fused the sorted Top-15 distribution but omitted two parts of
the observable output signature. Does adding the probability assigned to the
token that actually appears and the mass outside Top-15 improve the existing
fusion method?

## The only representation change

Each scored token produces 17 risk-oriented columns, in this exact order:

1. the frozen v1 columns `[1-p1, p2, ..., p15]`;
2. `selected-token surprisal = -log p(selected token)`, read directly from
   `token_spilled_energies`;
3. `residual tail mass = clip(1-sum(exp(top15 log-probabilities)), 0, 1)`.

The selected-token field is the ATP component emphasized by LOS-Net. The exact
full-vocabulary rank of that token is not cached and will not be inferred or
claimed. Residual tail mass is our compact summary of the probability mass not
represented by Top-15; it is not an LOS-Net architecture component.

K remains 15. There is no K sweep. Top-10 remains a separate aggregation over
token positions. There is no token window: every scored token is one matrix row.
All 17 columns are standardized on the same fitting population used by v1 before
the fusion weights are estimated.

## Frozen arms

1. **Token Entropy** — unchanged reference.
2. **Selected + Tail Probability Fusion - Equal Weights** — mechanism control.
3. **Selected + Tail Probability Fusion - IU-PCR** — primary candidate.
4. **Selected + Tail Probability Fusion - Joint Shrinkage** — supporting candidate.

The corresponding v1 Direct Probability Fusion arms are frozen comparators, not
refitted variants. No graph, lambda, window, readout, gate, detector, feature, K,
or DEEM search is allowed.

Selected-token surprisal and Top-K tail summaries already appeared among the
Step334 primitive token streams and as engineered features in the historical
mixed-v2 pools. The new question is narrower: whether appending their raw
token-level coordinates to the direct Top-15 matrix improves the same fusion
estimator.

## Track A: one-answer localization

Use the same 13,769-answer v3 roster: all 6,800 ProcessBench rows in eight cells
and all 6,969 PRMBench rows. For each answer independently:

`T scored answer tokens x 17 inputs -> label-free fusion -> token risk -> top-10 token mean per reasoning step`.

Use the same corrected annotations, source groups, folds, exclusions, and frozen
foldwise mean-entropy q=0.3 no-error gate as v1. A failed fusion fit remains an
explicit entropy fallback and counts in coverage. Report ProcessBench all-eight,
Q4, Q8, every cell, exact/early/late localization, within-one, clean-answer
accuracy and coverage. Report PRMBench within-answer AUC, pooled AUC and the same
cross-fold q=0.8 PRMScore.

Predeclared localization comparisons, all paired by canonical source group with
10,000 bootstrap draws and 97.5% intervals:

1. Selected + Tail IU-PCR minus v1 Direct Probability IU-PCR — isolates the two
   new inputs.
2. Selected + Tail IU-PCR minus Token Entropy — promotion comparison.
3. Selected + Tail Equal Weights minus v1 Direct Equal Weights — mechanism check.
4. Selected + Tail Joint Shrinkage minus v1 Direct Joint Shrinkage — mechanism check.

The first two are the primary comparisons. Pooled PRMBench AUC alone cannot
promote the method.

## Track B: complete-answer hallucination detection

Use the exact complete-case rows, labels, order and 24 cells of the historical
`mixed_v2 / full / iu_pcr` contract. Within each answer, aggregate each of the 17
token sequences by its top-10 token mean. Within each cell:

`N answers x 17 summaries -> label-free cell-local fusion -> answer risk`.

The main publication comparator is **Historical IU-PCR**, replayed from its
frozen feature bundle and required to reproduce exactly. The v1 Direct
Probability Fusion score is also loaded unchanged to isolate the representation
change. Report every cell, QA9, math15, all24, coverage, fallback counts, learned
weights and runtime.

Predeclared complete-answer comparisons:

1. Selected + Tail IU-PCR minus Historical IU-PCR — publication comparison.
2. Selected + Tail IU-PCR minus v1 Direct Probability IU-PCR — representation
   change.
3. The corresponding equal and Joint v2-minus-v1 contrasts — mechanism checks.

The main IU comparison uses grouped resampling of canonical problems inside each
cell and paired resampling of cells for the macro, with 10,000 draws and a 97.5%
interval.

## Required preflight

Before scientific scoring:

- verify all probability matrices are finite, descending and aligned with
  `token_entropies`, `token_spilled_energies` and `gen_token_ids`;
- verify raw Top-15 mass never exceeds one beyond numerical tolerance and compute
  residual tail without Top-K renormalization; allow at most 5e-7 mass excess
  for saved float32 rounding (the frozen-corpus maximum observed before scoring
  was 3.35e-7), then clip the residual to `[0,1]`;
- when the selected token is present among the saved Top-50 IDs, verify its saved
  log-probability reproduces `-token_spilled_energies`; report Top-1, Top-15,
  Top-50 membership rather than treating tokens outside Top-50 as errors;
- distinguish teacher-forced ProcessBench/PRMBench scored tokens from sampled
  historical outputs in the audit wording;
- hash all raw inputs, frozen benchmark artifacts, v1 score artifacts, protocol
  and implementation;
- obtain a read-only PASS from a clean-context reviewer before running scores.

Any mismatch stops the run. An original result is preserved if a later correction
is needed.

## Interpretation rule

No automatic promotion threshold is applied in this exploratory experiment. The
report must show point differences and frozen paired intervals separately for
ProcessBench, PRMBench and the historical 24 cells. We will decide after seeing
the complete evidence whether the representation has enough signal to justify a
new focused development experiment. A gain over v1 alone explains the value of
ATP/tail in this matrix but does not establish a leading method. All results
remain development evidence until frozen confirmation on untouched data.
