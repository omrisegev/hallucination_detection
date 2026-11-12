# GSP ablation baseline on standalone base-model telemetry (2026-11-12)

## Purpose

This document shows where GSP is before the experiments that the advisors asked for on 2026-11-12. Each row adds or
removes one change. All numbers come from `TABLES.md`, which `make_tables.py` writes from `BOOT.csv` and `RESULTS*.csv`.

## Setup

| Item | Value |
|---|---|
| GSP | Frozen recipe C4U-parisi-within-jaffe-cross-v1 |
| Telemetry | TELE16 from the standalone base model (no PRM in the telemetry) |
| PRM pairings | Qwen-PRM-7B with Qwen2.5-Math-7B-Instruct; Qwen-PRM-72B with Qwen2.5-Math-72B-Instruct; PURE with Qwen2.5-Math-7B; RLHFlow with Llama-3.1-8B-Instruct |
| Benchmarks | PRMBench (6,969 answers), Socratic-PRMBench (2,995), Hard2Verify (181-200), ProcessBench (3,400) |
| Metrics | PRMBench and Socratic: official PRMScore. Hard2Verify: official balanced F1. ProcessBench: official average F1. Within-answer AUC on the first three only (it is not valid on ProcessBench). |
| Uncertainty | 95% bootstrap CI, 2,000 draws. Source-question groups on PRMBench, Socratic and Hard2Verify. Answers within each subset on ProcessBench. |

## Arms

| Arm | Change | Uses PRM |
|---|---|---|
| A | GSP-T (telemetry only), own count k = round(sum of posteriors) | No |
| B | A with the GAP count (largest gap in the sorted step scores) | No |
| D | A or B with the step-length readout fix: for the 11 token channels, a step with fewer than X tokens uses the median of its token values, other steps keep the top-10 mean. One fixed X for all benchmarks: X = 10 (main), X = 30 (sensitivity). | No |
| E | ProcessBench only: change-point first-error rule (3-state chain, MAP). PRM-free variants: free EM, and fixed prior P(no error) = 1/2. | Free: No. Anchored: Yes |
| C | References: GSP full (PRM + telemetry) with own count or GAP; PRM alone with its own threshold or GAP; change-point anchored on the PRM | Yes |

## Data checks

- Parity: A, B and C on PRMBench, Socratic and Hard2Verify match the earlier numbers of the threads "Evaluate C4U on telemetry only" and the advisor brief.
- Readout extraction: the top-10 step values that we rebuild from the raw standalone telemetry match the stored bank in all 16 cells (largest difference 9.1e-14). Branch `claude/sa-readouts-2026-11-12`.
- Hybrid gate: the top-10 column rebuilt from the readout file equals the stored R1 column in all 16 cells (difference 0.0).

## Result 1: GAP compared with own count (B minus A)

| Benchmark | 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|
| PRMBench | -0.1 [-0.5, 0.2] | +0.9 [0.5, 1.3] | -0.1 [-0.5, 0.2] | -0.5 [-0.9, -0.1] |
| Socratic | +38.0 | +0.8 [0.3, 1.2] | +0.9 [0.5, 1.4] | +0.1 [-0.4, 0.5] |
| Hard2Verify | +11.0 [7.2, 14.6] | +5.1 [1.8, 8.6] | +4.9 [0.8, 8.9] | +0.7 [-2.8, 4.1] |

- On PRMBench, GAP and own count are almost equal.
- On Socratic 7B, the own count flags 98.9% of the steps (the known count flip, PRMScore 27.1). GAP removes this failure (65.1).
- On Hard2Verify, GAP is better in 3 of 4 pairings with a CI above 0.

## Result 2: effect of the PRM (GSP full minus GSP-T, own count)

| Benchmark | 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|
| PRMBench | +2.8 | +3.1 | +2.0 | +0.4 |
| Socratic | +45.7 | +1.8 | +3.7 | -31.4 |
| Hard2Verify | +3.3 | +0.1 | +1.3 | -0.4 |

- The PRM adds 0.4 to 3.1 points on PRMBench.
- On Socratic RLHFlow, GSP full with own count fails (31.0, flag rate 74.5%). The own count is not stable with the PRM either.
- GSP-T (no PRM) is better than the PRM alone with its own threshold on PRMBench in 3 of 4 pairings (7B +1.8, PURE +2.0, RLHFlow +13.9; 72B -2.8).

## Result 3: step-length readout fix (D minus the same count without the fix)

GAP count, X = 10 and X = 30:

| Benchmark | X | 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|---|
| PRMBench | 10 | -0.9 [-1.3, -0.5] | -1.2 [-1.5, -0.8] | -3.2 [-3.5, -2.8] | -1.5 [-1.8, -1.2] |
| PRMBench | 30 | -3.2 [-3.7, -2.7] | -5.5 [-6.1, -4.9] | -9.8 [-10.4, -9.2] | -4.5 [-4.9, -4.0] |
| Socratic | 10 | +2.7 [1.9, 3.4] | +0.4 [-0.1, 1.0] | -0.2 [-0.7, 0.3] | -1.9 [-2.6, -1.3] |
| Socratic | 30 | +2.3 [1.5, 3.1] | 0.0 [-0.7, 0.7] | -1.1 [-1.8, -0.4] | -0.1 [-0.8, 0.6] |
| Hard2Verify | 10 | +0.4 [-0.3, 1.5] | +0.3 [0.0, 0.7] | 0.0 [0.0, 0.0] | +0.3 [0.0, 1.0] |
| Hard2Verify | 30 | -0.1 [-1.8, 1.6] | +0.2 [-0.6, 0.8] | +1.3 [0.0, 3.2] | +0.7 [-0.3, 1.9] |

Share of steps with fewer than X tokens (Qwen-PRM-7B cell): PRMBench 6.6% (X = 10), 66.5% (X = 30); Socratic 7.1%, 13.7%; Hard2Verify 0.1%, 1.9%; ProcessBench 0.4%, 12.6%.

- With the own count, X = 10 removes the Socratic 7B count flip: 27.1 becomes 69.0.
- Socratic within-AUC goes up with X = 10 in 3 of 4 pairings (0.730 to 0.772, 0.720 to 0.740, 0.664 to 0.678; RLHFlow 0.674 to 0.673).
- PRMBench gets worse in all 4 pairings at both cutoffs. Within-AUC also goes down (for example 7B 0.782 to 0.772 at X = 10).
- Hard2Verify has almost no short steps, so the fix changes almost nothing.
- Conclusion: one fixed cutoff does not help on all benchmarks. The gain on Socratic is smaller than the loss on PRMBench.

## Result 4: ProcessBench (official average F1)

| Arm | 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|
| A, B, D (no PRM, own count or GAP, with or without the fix) | 0.0 | 0.0 | 0.0 | 0.0 |
| C GSP full, own count or GAP | 0.4 / 0.0 | 0.0 | 0.0 | 0.0 |
| C PRM alone, own threshold | 72.9 [71.2, 74.6] | 78.2 [76.7, 79.6] | 52.6 [50.4, 54.6] | 29.2 [26.9, 31.2] |
| C PRM alone, GAP | 0.0 | 0.0 | 0.0 | 0.0 |
| E change-point, PRM anchored | 72.8 | 78.7 | 52.2 | 29.2 |
| E change-point, PRM + GSP full rank, anchored | 72.1 | 77.5 | 54.8 | 29.6 |
| E change-point, GSP-T, free EM (no PRM) | 0.0 | 0.0 | 0.0 | 0.0 |
| E change-point, GSP-T, P(no error) = 1/2 (no PRM) | 25.7 | 24.0 | 22.8 | 20.4 |
| E as above, with the fix X = 10 | 25.5 | 22.9 | 22.5 | 20.4 |

- F1 = 0 is a correct value, not a bug. GAP and the own count flag at least one step in every answer. Thus the accuracy on clean answers is 0, and the harmonic mean is 0.
- With the PRM as anchor, the change-point rule on PRM + GSP full is better than the PRM only on PURE (+2.2 [0.8, 3.6]). On the other 3 pairings the difference is -0.9 to +0.4.
- Without the PRM, free EM goes to P(no error) = 0 and flags an error in every answer. With the prior fixed at 1/2, F1 is 20-26.
- The fix does not change the ProcessBench result.

## What this means for the next experiments

- The main missing part is an answer-level decision (no error / error) that does not use the PRM. ProcessBench F1 is 0 without it, and the free change-point EM cannot learn it. This is the answer gate that the advisors asked for (plan step 2).
- GAP is the safer count: equal on PRMBench, better on Socratic and Hard2Verify.
- The step-length fix with one cutoff for all benchmarks is not a free gain. Plan step 3 assumes one fixed cutoff; this result shows that X = 10 costs 0.9 to 3.2 PRMScore points on PRMBench.

## Files

| File | Content |
|---|---|
| `ablation.py` | Arms A, B, C; writes `cells/<bench>__<prm>.npz` (not in git, 48 MB) and `RESULTS_<bench>.csv` |
| `hybrid.py` | Arm D; gate in `HYBRID_GATE_*.json`; `RESULTS_hybrid_*.csv` |
| `pb_cp.py` | Arm E; `RESULTS_pb_cp*.csv` |
| `boot.py` | `BOOT.csv` |
| `make_tables.py`, `figures.py` | `TABLES.md`, `charts/<bench>.png` |

## Addendum (2026-11-12): ProcessBench in the Mind the Gap metric (macro SLA)

Macro SLA = share of erroneous answers where the predicted first-error step is exact, mean over the 4 subsets
(`pb_sla.py`, `PB_SLA.csv`, 95% CI from 2,000 stratified draws). Mind the Gap reports 39.39 with Qwen3-8B.

| Arm (no PRM unless stated) | 7B base | 72B base | PURE base | RLHFlow base |
|---|---|---|---|---|
| GSP-T own count | 31.6 | 26.7 | 28.0 | 30.8 |
| GSP-T + GAP | 34.8 [32.7, 37.1] | 29.0 | 29.5 | 32.9 |
| GSP-T + GAP, fix X = 10 | 34.3 | 26.9 | 29.3 | 33.1 |
| Change-point on GSP-T, free EM | 36.8 [34.6, 39.3] | 32.4 | 33.2 | 35.7 |
| PRM alone, own threshold (uses PRM) | 61.9 | 71.2 | 39.0 | 19.1 |

Same arms on the telemetry of Mind the Gap's own scorer, Qwen3-8B (thinkfix bank, 3,400 answers; `qwen3_8b_pb.py`,
`PB_QWEN3_8B.csv`):

| Arm | macro SLA [95% CI] |
|---|---|
| GSP-T own count | 18.8 [16.9, 20.6] |
| GSP-T + GAP | 22.1 [20.1, 24.0] |
| Change-point on GSP-T, free EM | 21.8 [19.8, 23.9] |
| Control: equal mean of the same 16 columns, argmax | 40.1 [38.0, 42.3] |
| Control: equal mean of the same 16 columns, GAP | 37.0 [34.7, 39.1] |
| Mind the Gap (paper) | 39.39 |

- On Qwen3-8B the GSP weighting reduces SLA: the GSP rank with argmax gives 27.5, the equal mean gives 40.1.
- All 16 single columns with argmax give 16.4-35.1 (best: digit_spread 35.1); none reaches the equal mean.
