# RBM depth suite — amendment of 2026-09-12 (Claude, Stage 1 of the completion mandate)

Parent protocol: `docs/experiments/RBM_LITERATURE_COMPLETION_V1.md`, suite D (depth).
Original driver: `scripts/run_rbm_literature_completion.py` (unchanged, not edited).
Amended driver: `scripts/run_rbm_depth_amended.py`; reviewer: `scripts/review_rbm_depth_amended.py`.
Output: `results/rbm_literature_completion_v1/depth_amended/`. The original failed smoke artifacts
under `results/rbm_literature_completion_v1/depth/` are preserved untouched.

## 1. What failed and why (diagnosis, read-only)

The original depth smoke (27 answers) failed on 6 answers / 14 model records with
`ValueError: fewer than three varying hidden views`. Diagnosis in `depth/SMOKE_DIAGNOSIS.json`
(from `scripts/analyze_rbm_capacity_convergence.py`, saved exact-H4 states only):

| Failed answer (bank) | Surviving views | Unit logit std | Unit posterior std | Condition |
|---|---:|---|---|---|
| pb_gsm8k_q8__fe224022c6d88ee8 (12) | 2 | 55.0, 12.6, 15.7, 15.8 | 4.6e-2, 1.9e-13, 3.6e-1, 6.1e-43 | (a) saturation ×2 |
| pb_math_q4__b452cbcb105c9207 (6) | 2 | 10.2, 21.6, 8.5, 10.1 | 2.0e-13, 1.3e-1, 7.1e-13, 2.3e-1 | (a) saturation ×2 |
| pb_math_q8__52a98518c0f3a7e4 (6) | 2 | 6.6, 20.8, 8.3, 4.2 | 3.0e-18, 1.3e-1, 3.1e-1, 7.8e-17 | (a) saturation ×2 |
| pb_olympiadbench_q4__3f2662fde8fdf887 (6, 12) | 2, 2 | 6–33 | two units ≤ 1e-21 each bank | (a) saturation ×2 |
| pb_olympiadbench_q8__6e677f1a8dc370e7 (12) | 2 | 7.0, 10.0, 37.5, 28.4 | 3.3e-31, 2.9e-21, 7.8e-2, 1.9e-1 | (a) saturation ×2 |
| pb_omnimath_q4__15666a79f5937b83 (12) | 2 | 9.4, 24.6, 11.5, 44.9 | 1.5e-32, 2.1e-1, 1.3e-37, 9.8e-2 | (a) saturation ×2 |

Three conditions were distinguished for every exact-H4 unit on all 13,769 answers:
(a) **numerical sigmoid saturation** — the oriented logit varies (std > 1e-8) but the posterior is
constant to machine precision (std ≤ 1e-10); (b) **duplicate units** — |corr| ≥ 0.999 with an
earlier live unit; (c) **no varying signal** — logit std ≤ 1e-8. Every smoke failure is condition
(a). Population-wide (bank6 / bank12): mean saturated units per fit 0.12 / 0.22; duplicate 0.004 /
0.004; dead 0 / 0. Answers with only two surviving views: **140 / 13,769 (bank6, 1.02%)** and
**353 / 13,769 (bank12, 2.56%)**; the rest have three (1,367 / 2,281) or four (12,262 / 11,135).

Interpretation: the collapse is a property of the saved first layer (units driven into a region
where the posterior is 0 for every token of the answer, with large weight norms 8–24), not an
implementation error in the second layer. Whether the saturated units' logits carry useful
signal is unknown; the logit-input variant below measures it and does not assume it.

## 2. Amendment, part 1 — full measurement of the original implementation with declared failures

- Variants `layer2_exact` and `layer2_cd` keep their exact original definition: oriented hidden
  posteriors `expit((x@W+b)·sign)` → `zscore_columns` (scale > 1e-10) → one-hidden-unit Gaussian RBM
  (exact L-BFGS-B maxiter 100, or CD-10 with the original seed purpose `bank{b}:layer2cd`) → oriented
  unit → logit/posterior readouts → top-10 token mean → earliest argmax → fixed entropy q=0.3 gate.
- An answer with fewer than three varying views is a **declared per-answer failure** named
  `COLLAPSED_HIDDEN_VIEWS` (NaN step scores). It is counted in coverage and in the full-population
  metrics as a missed decision. It is not dropped, not substituted by another arm, and not a fix.
- Smoke acceptance: `PASS` (no failures), `PASS_WITH_DECLARED_FAILURES` (every failure carries the
  named condition), otherwise `FAIL`. `--verify-original-smoke` asserts that all non-failing
  original smoke records replay bit-for-bit and that each original failure maps to the named one.
- This yields a valid measurement of the original second layer's **performance and coverage**.
  It does not claim that the second layer works on every answer.

## 3. Amendment, part 2 — registered logit-input variants (enabled by the diagnosis)

- `layer2_logit_exact` / `layer2_logit_cd`: identical pipeline, but the second layer's visible input
  is the oriented hidden **logit** `(x@W+b)·sign` instead of its posterior. CD seed purpose
  `bank{b}:layer2logitcd` (distinct from the original). Same failure rule applies.
- Expected effect: removes condition-(a) collapses (logits vary where posteriors do not). It is
  **not** expected to repair (b) or (c), and it changes the representation seen by the second layer
  for every answer, so it is a different scoring configuration, compared beside the original.

## 4. Evaluation and reporting

Same full benchmark, labels (v3), source groups, folds, gate, readout and 10,000-draw paired
source-group bootstrap (97.5% for the pre-registered primaries `b6_layer2_exact_posterior −
b6_exact4_posterior` and `b12_layer2_exact_logit − b12_exact4_logit`; 95% otherwise). Reported for
every arm: **full-population** PB all-8 / Q4 / Q8 (failures counted as missed decisions), PRMB
within-answer AUC (over covered answers, with n), pooled AUC, PRMScore (conditional whenever
coverage < 100%; the full PRMScore is then unavailable by the frozen rule), **coverage**, and a
**conditional** PB panel on covered answers with a paired bootstrap on the common covered answers.
Conditional numbers are not comparable to full-coverage rows without the coverage column.

## 5. What this amendment is not

Not a change to the first layer (the capacity suite is reused as saved), not a new optimizer
budget (H4 fits remain the registered maxiter-100 states; see `capacity/CAPACITY_INTERPRETATION.md`),
not a candidate promotion, and not untouched confirmation.
