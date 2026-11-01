# Stage 2: how many in-domain labels does supervised weighting need? (2026-11-01)

Protocol frozen first: `PROTOCOL.json`. Code: `budget.py` (reuses stage 1's `anchor.py`). CPU, 726 s.
Parity: AUTO reproduces stage 8 in all 8 cells. 20 repetitions per budget; each one splits the source groups
(Socratic: the source problem, from the answer id) 50/50 into a labelled pool and a test half. The LR is fitted on k
labelled answers and scored on the test half. C = 1 fixed. No repetition failed.

**k\*** = the smallest budget from which the supervised arm is ahead on mean PRMScore (GAP) **and** in ≥16/20
repetitions, at that budget and every larger one.

| | Qwen-7B | Qwen-72B | PURE | RLHFlow |
|---|---|---|---|---|
| PRMBench: full LR beats AUTO from | 50 | 50 | 100 | 50 |
| PRMBench: full LR beats the PRM from | 20 | 5 | 20 | 5 |
| Socratic: full LR beats AUTO from | 5 | 5 | 10 | 20 |
| Socratic: full LR beats the PRM from | 50 | 50 | never (tie: +0.2 with all labels) | 5 |
| Socratic: ratio-only (ANCH) beats the PRM from | never | never | never | 50 |

Gain of the full LR over AUTO (mean over repetitions, PRMScore):

| labelled answers | 20 | 50 | 100 | 200 | all of the pool (~3,100 PRMBench / ~1,500 Socratic) |
|---|---|---|---|---|---|
| PRMBench Qwen-7B | −0.87 | +0.40 | +0.94 | +1.25 | +1.58 |
| PRMBench Qwen-72B | −0.18 | +1.06 | +1.76 | +1.99 | +2.11 |
| PRMBench PURE | −1.29 | +0.08 | +0.77 | +1.03 | +1.32 |
| PRMBench RLHFlow | −0.42 | +0.93 | +1.63 | +2.09 | +2.56 |
| Socratic Qwen-7B (vs PRM) | +0.04 | +1.58 | +2.23 | +2.38 | +2.69 |
| Socratic Qwen-72B (vs PRM) | −0.07 | +1.48 | +2.01 | +2.12 | +2.91 |
| Socratic PURE (vs PRM) | −1.70 | −0.61 | −0.30 | −0.09 | +0.21 |

## Predeclared readings
- **B1** (PRMBench, full LR beats AUTO with ≤200 labelled answers in ≥3/4 PRMs): **met, 4/4** (50 to 100 answers).
- **B2** (Socratic, full LR beats a strong PRM at some budget in ≥2/3): **met, 2/3** (Qwen-7B and Qwen-72B from 50
  answers; PURE only ties).

## Reading
1. **About 50 labelled answers from the same benchmark are enough** for the full supervised weighting to pass
   label-free AUTO on PRMBench, and to pass the strong Qwen PRMs on Socratic. Most of the gain comes by 100–200
   answers.
2. **Below about 20 labels, label-free AUTO is better on PRMBench.** With few labels, the full LR overfits its 17
   weights.
3. **Learning only the PRM/telemetry ratio saturates early and stays lower.** On PRMBench, ANCH beats AUTO from 10–50
   answers, but it stops at +0.3 to +1.5. On Socratic it never beats a strong PRM.
4. **Stage 1's Socratic caveat is resolved.** The cross-fit with problem groups instead of per-answer groups gives the
   same numbers (`RECHECK.json`; LR17 72.48 / 72.17 / 70.26 / 63.32 against stage 1's 72.55 / 72.22 / 70.21 / 63.28).

## Caveats to check before an advisor claim
- **Socratic's within-answer gain is very large** (LR − AUTO within-AUC +0.11 to +0.17). Step position alone ranks
  Socratic steps well (within-AUC .731, Step 481), so the LR may be learning position through the channels, which is
  an artefact of the benchmark's design. Next check: an LR on [PRM, position] at the same budgets.
- The spread over repetitions mixes the label draw and the test split. It is not a CI for one fixed test set.
- AUTO sees the test half without labels (transductive), as in every earlier stage.
- Development data that we have already inspected.

Files: `PROTOCOL.json`, `budget.py`, `BUDGET.csv` (one row per repetition × budget), `SUMMARY.json` (k\*, means,
percentiles, win counts, parity), `RECHECK.json`, `run.log`.
