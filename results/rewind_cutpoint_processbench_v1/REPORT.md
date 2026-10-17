# REWIND cut-point baseline on ProcessBench Localization (2026-10-16)

**Script:** `scripts/run_rewind_cutpoint_processbench_v1.py`
**Locator code:** `spectral_utils/rewind_cutpoint.py`
**Source:** REWIND (Ding, Wang, Zhou, Yang; arXiv 2610.19561).

CPU only. Reuses the Step-279 fair Localization lane unchanged:

- the 3,400 official Llama-3.1-8B rows;
- the five localization folds;
- the threshold-only cross-fit;
- 2,000 paired grouped bootstrap draws, seed 20260818, with the threshold refit in every draw.

Before anything new is scored, the script checks that Unified-28 (0.284832) and the dedicated
incumbent (0.326141) reproduce exactly.

## Method

**Cut-point rule (as in the paper).**

1. Find the token with the highest entropy (the peak).
2. Within the 5 tokens before the peak, find the token with the largest one-step entropy increase.
3. Look for the last delimiter (`.`, `,` or newline) in the preceding 10 tokens. The cut is the token right after it.

**Our adaptation.** The token where REWIND would cut is mapped to the step that contains it, and that step is read as the predicted first error.

**Detector.** Mean token entropy, which REWIND reports as its best response-level signal (AUROC 0.746).

**No labels** are used in any score or locator.

**Deviations from the paper:**

- Cached top-K entropies instead of full-vocabulary entropies.
- Delimiters are matched as substrings of byte-level BPE tokens.

## Results — ProcessBench equal-subset macro-F1 [95% CI]

| Method | F1 |
|---|---|
| mean entropy + step-top5 locator (2×2 control, **post hoc**) | **0.338** [0.318, 0.358] |
| dedicated incumbent (family6 level, step-top5) | 0.326 [0.305, 0.345] |
| GL-LIU v1 | 0.320 [0.299, 0.340] |
| **REWIND cut point** | **0.317** [0.298, 0.337] |
| REWIND peak only | 0.313 |
| Unified-28 | 0.285 [0.265, 0.306] |
| max entropy + step-top5 | 0.269 |
| max entropy + REWIND cut (2×2) | 0.269 |
| Mind the Gap replay | 0.202 [0.178, 0.219] |

## Paired deltas

| Contrast | Δ F1 [95% CI] |
|---|---|
| REWIND − Mind the Gap | +0.114 [+0.093, +0.146] |
| REWIND − Unified-28 | **+0.032 [+0.009, +0.055]** |
| dedicated − REWIND | +0.009 [−0.014, +0.031] |
| GL-LIU − REWIND | +0.003 [−0.020, +0.026] |
| REWIND − REWIND peak-only | +0.004 [−0.002, +0.010] |
| **Detector swap:** REWIND − (max-entropy detector, same REWIND locator) | **+0.048 [+0.026, +0.070]** |
| **Locator swap:** (step-top5 locator, same mean-entropy detector) − REWIND | **+0.022 [+0.002, +0.038]** |

## Distance diagnostic (2,221 erroneous traces; predicted step − gold first-error step)

| Locator | exact | within ±1 | early | late |
|---|---|---|---|---|
| family6 step-top5 (dedicated) | 0.301 | 0.575 | 0.275 | 0.424 |
| max-entropy step-top5 | 0.294 | 0.559 | 0.257 | 0.449 |
| REWIND cut | 0.267 | 0.543 | 0.269 | 0.464 |
| REWIND peak only | 0.260 | 0.540 | 0.263 | 0.477 |
| Mind the Gap | 0.233 | 0.509 | 0.252 | 0.515 |
| Unified-28 | 0.222 | 0.499 | 0.430 | 0.348 |
| *uniform random step* | 0.155 | | | |
| *always step 0* | 0.122 | | | |

**How much each REWIND stage moves the predicted step:**

- Backtracking moves it in 4.1% of traces.
- Delimiter alignment finds a delimiter in 30.6% of traces, but **never changes the step** (0%).

## Reading

1. **REWIND's gain comes from its detector, not its cut point.**
   - Mean entropy beats max entropy by +0.048 F1 with the locator held fixed.
   - At step granularity the cut-point rule is about the raw entropy peak (+0.004, not significant).
   - It is worse than the step-top5 locator (−0.022, significant).
2. **Entropy-peak locators err late.** About 46% of their predictions are late and about 27% early. The entropy peak comes *after* the gold error, and by more than REWIND's 5+10-token backtrack can recover. This is consistent with REWIND's "instability before the peak" observation, but the lag is measured in steps, not tokens.
3. **This confirms Step 331, and is not new.** The transparent mean-entropy + step-top5 row
   reaches 0.338, the highest point F1 in this lane. Step 331 (2026-09-08) already showed that a
   raw mean-entropy constant gate takes the answer-only fusion from 20.00 to 31.31 F1, and that
   with a working gate every fusion arm and the single-stream entropy control lie within 1.5 pp.
   The REWIND result is the same finding seen from a published method.
   - The row was added after the smoke run, so it is exploratory.
   - Unified-28's locator is the weakest of the entropy family (exact 0.222, 43% early).
4. **Scope.** ProcessBench was de-prioritized at the 2026-10-08 advisor meeting (Step 451). This
   is a closing record, not a new direction.
5. **Mind the Gap row.** Per Step 333, the public Mind the Gap code does not implement the
   paper's step aggregation, so the Mind the Gap replay is our reconstruction.
6. **Not re-run on this branch.** The run used the Step-279 lane on the stale lineage. The lane
   code and inputs are byte-identical on this branch except the tokenizer path, which now points
   to a tracked file with the identical SHA-256. The ProcessBench cells stay local-only in
   `dataset_cache/`.
