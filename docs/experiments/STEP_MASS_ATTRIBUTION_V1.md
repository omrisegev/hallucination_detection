# Step-mass attribution along numeral dependencies (v1) — frozen protocol

User requested this on 2026-09-11 immediately after Step 354, which found that
moving individual digit tokens cannot move a top-10 argmax over ~100-token
steps. This stage moves STEP-level surprise instead. Base commit b6b10043
(`claude/readout-provenance-v1`). Same worktree/branch; same frozen inputs,
gate, tie rule, labels, folds and evaluator as Step 354. No fitting, no labels
in any score, no new inference.

## Hypothesis

Under teacher-forced scoring of a provided solution, surprise appears where an
inconsistency becomes visible (late), while the annotation marks where the
error was committed. If step `k` consumes a value computed at step `j < k`,
then `k`'s surprise is evidence about `j`. Transferring part of `k`'s surprise
to its numeric parents should move predictions earlier AND to the right step;
a shuffled-parent control and a dependency-free uniform-earlier control
separate "earlier" from "right".

## Dependency graph (per answer, label-free, cache only)

Numerals, given literals and origins exactly as in Step 354
(`spectral_utils.provenance_readout.build_numerals`). For steps `j < k`,
`n_kj` = number of inherited, non-given numerals in step `k` whose origin is
`j`; `n_k = Σ_j n_kj`. Step `k` "has parents" iff `n_k > 0`. The attribution
weight is `A(k→j) = n_kj / n_k` (row-stochastic over parents). One hop only.

## Step scores

Per stream (entropy, varentropy): `s` = frozen top-10 token mean per step;
`z` = within-answer standardisation of `s` (`z = 0` when the SD is zero). The
standardisation makes unsurprising children (`z ≈ 0`) contribute nothing and
below-average children subtract, so a parent is not rewarded merely for
having many dependents. Then, with transfer gain `α`:

```
score'_j = (1 − α·[step j has parents]) · z_j  +  α · Σ_{k>j} A(k→j) · z_k
```

Mass is conserved: a step with parents sends `α` of its own standardised
surprise to its parents in proportion `A`. The evaluator's first-max tie rule
is unchanged.

Arms (per stream) — all frozen before any result:

| arm | definition | role |
|---|---|---|
| `top10` | `s` | frozen reference |
| `attr_a05` | formula above, `α = 0.5` | primary candidate |
| `attr_a10` | `α = 1.0` (full transfer) | secondary, declared |
| `attr_shuffled_a05` | as `attr_a05`, but every parent `j` of `k` is replaced by a uniformly random earlier step (seed 2026091103, per answer) | control: earlier but uninformative target |
| `attr_uniform_a05` | as `attr_a05`, but each step with parents sends `α` uniformly to ALL earlier steps | control: structure-free earliness |
| `zscore_only` | `z` | shows that standardisation alone changes nothing at the argmax (PRMB ranks identical) |

`α` is not tuned: two declared values, both reported. `length` and
`random_step` (same seeds as Step 354) are reproduced in the table.

## Evaluation and inference

Identical to Step 354: `evaluate_arrays` / `paired_bootstrap` imported
unchanged, 10,000 paired canonical-source draws. Primary (97.5%): per stream,
`attr_a05 − top10` and `attr_a05 − attr_shuffled_a05` on PB all-8 macro F1
and PRMB within-answer AUC. Exploratory (95%): `attr_a10 − top10`,
`attr_uniform_a05 − top10`, `attr_a05 − attr_uniform_a05`. Reproduction
gates for the two references as in Step 354. Stratified (true-step-longest)
and early/exact/late panels for every arm; plus a dependency panel: fraction
of answers / steps with parents, mean parents per step, fraction of erroneous
answers where the true step is a parent of the frozen argmax step (the
mechanism's ceiling; diagnostic, label-using, reported separately).

Decision question: does `attr_a05` beat BOTH the reference and the shuffled
control on both benchmarks while lowering late/early? No promotion threshold;
development evidence only.

## Preflight

Unit tests: dependency counts, row-stochastic weights, conservation, shuffled
targets earlier than the sender, uniform control, `zscore_only` argmax
identity. 27-answer smoke with dependency counts. Tokenizer round-trip rule
as amended in Step 354.

## Outputs

`results/step_mass_attribution_v1/`: `METRICS.json`, `SCORES.npz` (ignored),
`SUMMARY.csv`, `MANIFEST.json`, `RUN_STATE.json`, `SMOKE.json`. Chat-first.
