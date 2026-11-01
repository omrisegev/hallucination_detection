# Raw-null gate diagnosis, stages D1-D3 (2026-11-01)

**Plan:** `docs/experiments/RAW_NULL_GATE_DIAGNOSIS_PLAN_2026-10-31.md`.

**Inputs:**
- Code: the frozen v1b `fix` code (`fix_mono` beside it where it differs).
- Cells: all 68 ok cells (the 4 PRMs, 5 arm/mode combinations, 4 benchmarks; 12 PRMBench backbone cells are missing as
  before).
- Machine: the Windows PC, CPU only.

**Labels are used only to diagnose.** Every column ending in `_DIAG` uses labels: it is an oracle or diagnostic number,
never a method.

**Checks:**
- **Before the run:** an independent reviewer checked the script. There is no label leakage. Its should-fix items were
  applied: strict reproduction checks, the correct labelled set for PRMBench in D2, and each channel's sign in the fused
  score.
- **During the run:** in every cell, the recomputed label-free arrays (raw_fused, z_fused, G1 and G2 per-step P(err))
  reproduce the saved v1b `fix` arrays to 1e-9.

**Files:**
- `D1.csv`, `D2.csv`, `D3.csv`: the per-stage tables, one row per cell and statistic.
- `diag_d1_d3.py`: the script.
- `bundle/`: the data needed to re-analyse without the raw data. See `HANDOFF.md`.

The numbers below are medians over cells.

## D1: is there a clean-answer signal? (answer level)

Each entry is median AUC / oracle closed_clean at kept_error = .85, both **DIAG**. The number of cells is in
parentheses.

| Statistic | ProcessBench | CRV arith | CRV GSM8K | PRMBench |
|---|---|---|---|---|
| raw fused, answer max | .770 / .50 (20) | .813 / .61 (20) | .694 / .34 (20) | .557 / .12 (8) |
| raw fused at the worst step (G1's value) | .760 / .50 | .693 / .39 | .693 / .34 | .570 / .12 |
| best single group, answer max (per cell) | .795 / .57 | .827 / .62 | .739 / .39 | .685 / .30 |
| G1 P(err) of the worst step | **.436** / .00 | .688 / .39 | .661 / .21 | .570 / .12 |
| G1 P(err), monotone (F3) | .674 / .39 | .691 / .39 | .663 / .22 | .570 / .12 |
| G1b 1 − p | .731 / .44 | .697 / .38 | .655 / .26 | .588 / .14 |
| G2 P(err) of the worst step | .615 / .15 | .804 / .58 | .577 / .22 | .654 / .21 |
| G2 P(err), monotone | .664 / .29 | .826 / .61 | .653 / .29 | .653 / .13 |
| G3 posterior | .530 / .01 (6) | .746 / .48 (16) | .688 / .33 (10) | .603 / .03 (4) |

The true share of answers with an error is .653 (ProcessBench), .083 (CRV arith), .210 (CRV GSM8K) and .972 (PRMBench,
evaluated answers only).

- **The signal is there.** With an oracle threshold, the raw scores reach closed_clean of at least .30 at
  kept_error .85 on ProcessBench, CRV arith and CRV GSM8K. PRMBench is the exception: about 97% of evaluated answers
  have an error, and clean-answer AUC is only .56-.69.
- **The P(err) transform loses the signal.** On ProcessBench, the raw value has AUC .76, but G1's P(err) of the same
  step has AUC .44: the ranking is inverted, because P(err) falls in the right tail. The monotone correction recovers
  .67. G2 and G3 rank below the raw maximum on ProcessBench and CRV GSM8K. On CRV arith they are about equal (G2 monotone
.826 against .813). On PRMBench, G2 ranks higher (.65 against .56).
- **The worst-step choice matters on CRV arith** (D7 preview): the raw answer maximum (.813) beats the value at the
  answer-z argmax (.693).

## D2: does the fitted null match the correct steps?

**What "true" means here:** the true values come from labelled correct steps in the fit pool. A labelled step is one
with keep_eval: c.keep, or keep_gold on ProcessBench. The table shows the fused score, with medians over 20 or 8 cells.

| Benchmark | True share of correct steps | π₀_raw | σ₀ / true sd | μ₀ − true mean (true-sd units) | Error share: all labelled / central band | True skew / excess kurtosis | Share of the pool labelled |
|---|---|---|---|---|---|---|---|
| ProcessBench | .866 | 1.010 | 1.11 | +0.36 | .134 / .101 | −0.39 / 1.34 | .644 |
| CRV arith | .966 | .984 | 0.82 | −0.38 | .034 / .022 | 0.80 / 1.29 | 1.000 |
| CRV GSM8K | .927 | 1.003 | 1.00 | −0.04 | .073 / .060 | 0.43 / 0.24 | 1.000 |
| PRMBench | .842 | .958 | 0.99 | −0.00 | .158 / .100 | 0.11 / 0.38 | .885 |

The group scores are in `D2.csv`. Their nulls fail more often: 76% of group fits are ok on ProcessBench, against
93-98% on the other benchmarks.

- **ProcessBench: contamination.**
  - Error steps are almost as common in the central band (10%) as overall (13%), so the errors do not form a tail.
  - The fitted null is wider than the correct steps (×1.11) and is shifted towards the errors (+0.36 sd).
  - π₀_raw is 1.01, against a true .87.
  - Also, 36% of the pool has no label: these are the steps after the first error, which the null still absorbs.
- **CRV arith: shape.** The correct steps are right-skewed (skew 0.8). The central Gaussian is too narrow (σ₀ = 0.82 of
  the true sd) and sits low, so normal steps in the right tail look abnormal.
- **CRV GSM8K and PRMBench:** the fitted null is close to the correct steps. Even so, π₀_raw (1.00 / .96) is above the
  true share (.93 / .84), because the error steps overlap the centre.

## D3: do error steps separate from correct steps? (step level)

Each entry is the median step-level AUC / Cohen's d of error versus correct steps, **DIAG**.

| Score | ProcessBench | CRV arith | CRV GSM8K | PRMBench |
|---|---|---|---|---|
| Fused (raw) | .814 / 1.15 | .743 / 1.15 | .691 / 0.79 | .736 / 0.91 |
| Group scores | .709 / 0.75 | .712 / 0.81 | .641 / 0.47 | .646 / 0.49 |

The v1 selftest (synthetic), where the gates worked, used d = 2.3-3.0.

- **Separation is only about one sd.** At the step level, errors sit about 1 sd above correct steps. That is roughly
  half to one third of the synthetic separation at which the two-groups model works. At d ≈ 1 the error distribution
  lies inside the null, which explains D2.
- **The strongest channels** (TELE16, oriented as in the fused score, median d): q15_H1 and q15_VE1 (0.6-1.4),
  digit_spread (0.45-1.56), logprob_margin and chosen_surprisal (0.6-1.1).
- **Channels with no signal:** dominant_freq16, top50_js, energy_innovation and realized_z (except PRMBench) have
  d ≈ 0 to 0.2. bocpd_p0 is weak. These are the discrete or spiky channels that Review B flagged.

## Which assumption breaks, per benchmark

| Benchmark | Signal (D1) | Null (D2) | Separation (D3) | Main break |
|---|---|---|---|---|
| ProcessBench | yes (.77, oracle .50) | contaminated: wide, shifted, unlabelled post-error steps in the pool | d ≈ 1.15 | **The zero assumption fails** (errors are not a tail), and the P(err) transform inverts the ranking |
| CRV arith | yes (.81, oracle .61) | too narrow: right-skewed correct steps | d ≈ 1.15 | **The Gaussian-null shape fails**, so normal tail steps are over-called |
| CRV GSM8K | moderate (.69, oracle .34) | close to the correct steps | d ≈ 0.8 | **Weak separation**, so there is little room above the .30 bar |
| PRMBench | weak (.56-.69, oracle .12-.30) | close | d ≈ 0.9 | **Almost no clean answers** (3%), so a clean-answer gate has little to win |

**What this suggests (not started):**
- The ranking is usable, but "fit a Gaussian null to the centre, then cut at P(err) = .5" is not.
- Candidates: a null with a skewed shape (or a rank-based conformal p), or a null fitted only on steps before the
  first flagged step (to reduce contamination). The worst step should be the raw maximum, and P(err) must be monotone.
- The remaining stages are D4 (answer nuisance) and D6 (group independence).
