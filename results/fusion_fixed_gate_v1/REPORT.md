# Fixed-constant no-error gate on raw answer telemetry (fusion_fixed_gate_v1)

**Date:** 2026-09-08. **Author:** Claude (main tree). **Inputs:** frozen outputs of
`results/localization_full_benchmark_v3` (19 answer-only anchor arms, all 13,769 records,
v3 labels, canonical source groups, `FOLDS_V2.json`) and the raw token telemetry memmaps in
its `inputs/`. **No new fusion fit, no changed peak, no changed PRMB ranking.** Only the
final no-error decision is replaced. Code: `spectral_utils/fixed_gate_readout.py`,
`scripts/evaluate_fixed_gate_v1.py`. Numbers: `METRICS.json`, detectors: `DETECTORS.npz`.

## Question

The answer-only pipeline decides "no error / error at the risk peak" with a per-answer
GMM/BIC mixture on that answer's own fused step risks. Can a single fixed constant on a raw
answer-level telemetry summary do the job instead, and how much of the ProcessBench gap to
the pooled historical comparators does that close?

## Why a threshold on the fused scores cannot work

Every feature is z-scored inside the answer before fusion, so the fused step risks have no
cross-answer scale. On all 6,800 ProcessBench rows the answer-level max, mean and
max-minus-median of `dual__iu` step risks separate erroneous from clean answers at
AUC 0.43-0.52 (chance in every cell). Raw, un-normalized summaries of the same answer do:

| Detector (raw stream, summary) | mean AUC error-vs-clean over 8 cells | range |
|---|---|---|
| mean token entropy | 0.742 | 0.717-0.788 |
| max 8-token-window mean entropy | 0.770 | 0.740-0.807 |
| mean top-k tail mass | 0.784 | 0.760-0.825 |
| max 8-token-window top-k varentropy | 0.784 | 0.752-0.818 |

## Gate rules tested (all use the frozen argmax step when the gate opens)

- `gmm_bic_saved`: the saved answer-only decisions (baseline).
- `<detector>|nested_labels`: one constant per outer source-group fold, chosen by maximizing
  eight-cell macro F1 on the other four folds (labels used, training folds only).
- `<detector>|quantile_q`: constant = q-quantile of the detector over the unlabeled training
  folds, q in {0.2, 0.3, 0.4, 0.5}. **No labels anywhere.**
- `<detector>|transfer_*`: constant chosen with labels on one dataset family (gsm8k+math or
  olympiadbench+omnimath) and applied to the other.
- `oracle_gate`: perfect error/no-error knowledge with the same peaks (label-using ceiling).
- Historical pooled comparators from `historical_fusion_refit_v3` are quoted for context;
  they fit other answers and calibrate their threshold with nested labels.

## Result: eight-cell macro F1 (ProcessBench, all 6,800 rows)

| Arm (frozen peaks) | GMM/BIC saved | mean-entropy constant, nested labels | mean-entropy constant, label-free q=0.3 | tail-mass constant, nested labels | oracle gate |
|---|---|---|---|---|---|
| dual__iu (primary anchor) | 20.00 | **31.31** | 31.16 | 31.49 | 43.38 |
| dual__equal | 19.60 | 31.53 | 31.37 | 31.30 | 43.68 |
| dual__graph_perm | 21.18 | 32.73 | 32.74 | 33.06 | 45.65 |
| context__iu | 17.43 | 32.32 | 32.16 | 32.46 | 44.98 |
| moment__iu | 20.38 | 31.15 | 31.01 | 31.48 | 43.10 |
| entropy_parent (single-stream control) | 19.22 | 32.18 | 32.10 | 32.46 | 44.72 |
| *historical IU, pooled fit + nested label threshold* | *34.18* | | | | |
| *historical fixed-family CONT* | *34.60* | | | | |

Paired source-group bootstrap (1,000 draws, seed 2026090707), eight-cell macro F1 minus
the saved GMM gate, `dual__iu`:

| Gate | difference | 95% CI |
|---|---|---|
| mean-entropy constant, nested labels | +11.36 pp | [+9.52, +13.23] |
| max-window-entropy constant, nested labels | +8.64 pp | [+6.81, +10.42] |
| max-window-entropy constant, label-free q=0.3 | +8.45 pp | [+6.60, +10.30] |
| tail-mass constant, nested labels | +11.53 pp | [+9.59, +13.36] |

Q4 / Q8 scorer panels behave the same (see `METRICS.json`); e.g. dual__iu Q8: 20.38 ->
31.82 (mean entropy, nested) / 32.63 (tail mass, nested); historical IU Q8 34.29.

## Where the gain comes from (dual__iu, Q8 cells)

| Cell | erroneous | raw peak exact (unchanged) | final exact GMM -> constant | correct peak suppressed GMM -> constant | erroneous called clean GMM -> constant | clean correct GMM -> constant |
|---|---|---|---|---|---|---|
| gsm8k | 207 | 79 | 23 -> 56 | 56 -> 23 | 146 -> 75 | 129/193 -> 142/193 |
| math | 594 | 163 | 92 -> 148 | 71 -> 15 | 244 -> 55 | 169/406 -> 165/406 |
| olympiadbench | 661 | 158 | 111 -> 152 | 47 -> 6 | 204 -> 31 | 75/339 -> 73/339 |
| omnimath | 759 | 191 | 112 -> 188 | 79 -> 3 | 325 -> 16 | 85/241 -> 68/241 |

The constant gate almost never suppresses a correct peak and rarely calls an erroneous
answer clean; clean accuracy is roughly unchanged. The remaining loss is location: the peak
is exact in only 25-38% of erroneous answers, identical for every gate and nearly identical
to the pooled historical comparator (81 vs 79 on gsm8k, 209 vs 191 on omnimath).

## Stability and transfer of the constant

- Nested thresholds vary little across folds: mean entropy 0.202-0.212 (label-chosen),
  max-window entropy 0.856-0.869 (label-chosen) vs 0.936-0.947 (label-free q=0.3).
- Quantile sensitivity (dual__iu, mean entropy): q=0.2 28.5, q=0.3 31.2, q=0.4 30.3,
  q=0.5 27.0. Every q in 0.2-0.4 beats the GMM gate by more than 8 pp.
- Cross-family transfer of a label-chosen constant (mean entropy, dual__iu): easy->hard
  29.0 on the hard cells, hard->easy 32.9 on the easy cells, compared with 31.3 in-fold.
  Max-window entropy transfers worse (23.3 / 23.1). Mean entropy is the more robust choice.

## Reading

1. The answer-only pipeline's ProcessBench deficit versus the pooled historical comparators
   is mostly the gate. One constant on raw mean token entropy closes about 80% of the
   14-point gap (20.0 -> 31.3 vs 34.2) without touching fusion weights, peaks or any
   other answer's fit. The label-free quantile version is within 0.2 pp of the
   label-chosen one.
2. Location is the shared bottleneck of every arm, including the historical ones: the
   oracle gate ceiling with these peaks is 43-46%.
3. With a working gate, the fusion arms and the single-stream entropy control are within
   about 1.5 pp of each other on ProcessBench F1 (31.2-33.1). Fusion does not yet buy
   exact-step localization over raw entropy on this benchmark; this must be said plainly
   in any advisor material.
4. These 6,800 rows are exposed development data. The nested folds estimate within-
   development generalization only; the choice of q and detector was made after seeing
   these tables. A frozen candidate (detector, q) must be fixed before any untouched
   confirmation.

## Not done here

PRMBench dense ranking is unchanged by construction. The historical comparators were not
re-run with this gate. Mind the Gap is still not on the v3 benchmark. No fusion-weight or
peak change was attempted.
