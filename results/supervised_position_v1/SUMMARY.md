# Position control for stage 2 (2026-11-01)

Protocol frozen first: `PROTOCOL.json`. Code: `position.py` (imports stage 2's `budget.py` unchanged). CPU, 665 s.
Same 20 splits and label draws as stage 2. Parity: LR17 reproduces stage 2's `BUDGET.csv` exactly in all 1,440
repetition × budget × cell checks. Position = relative position of the step in its answer, one linear term.

New arms: **POS2** = LR on [PRM, position]; **LR18** = LR17 + position.

## With all labels of the pool (mean PRMScore under GAP over 20 repetitions)

| | PRM | AUTO | POS2 | LR17 | LR18 | LR17 − POS2 (positive reps) |
|---|---|---|---|---|---|---|
| PRMBench Qwen-7B | 66.31 | 69.86 | 65.30 | 71.44 | 71.25 | +6.13 (20/20) |
| PRMBench Qwen-72B | 64.50 | 70.18 | 66.41 | 72.30 | 72.27 | +5.89 (20/20) |
| PRMBench PURE | 64.71 | 69.06 | 64.42 | 70.38 | 70.32 | +5.96 (20/20) |
| PRMBench RLHFlow | 59.36 | 65.87 | 63.12 | 68.43 | 69.48 | +5.31 (20/20) |
| Socratic Qwen-7B | 69.51 | 65.41 | 71.81 | 72.20 | 72.66 | +0.39 (17/20) |
| Socratic Qwen-72B | 68.21 | 60.46 | 71.09 | 71.13 | 72.80 | +0.03 (12/20) |
| Socratic PURE | 69.83 | 63.42 | 69.91 | 70.04 | 69.73 | +0.13 (13/20) |
| Socratic RLHFlow | 50.37 | 57.71 | 65.97 | 63.01 | 68.36 | −2.96 (0/20) |

On Socratic, POS2 already beats the PRM from k = 5 labelled answers for Qwen-7B, Qwen-72B and RLHFlow (k\* = 5).

## Predeclared readings
- **P1** (Socratic: LR17 beats POS2 in ≥2/3 strong PRMs): **not met**, 1/3 (Qwen-7B only, +0.39).
- **P2** (Socratic: adding position to LR17 adds ≤ +0.5 in ≥2/3 strong PRMs): **met**, 2/3 (Qwen-7B +0.46, PURE −0.31;
  Qwen-72B +1.67).
- **P3** (PRMBench: LR17 beats POS2 in ≥3/4 PRMs): **met**, 4/4 (+5.3 to +6.1, 20/20 each, from k = 10).

Post-hoc, not predeclared: LR18 − POS2 with all labels, Socratic: Qwen-7B +0.85 (20/20), Qwen-72B +1.70 (20/20),
PURE −0.19 (5/20), RLHFlow +2.39 (20/20). PRMBench +5.9 to +6.4 (20/20).

## Reading
1. **Stage 2's Socratic result was mostly position.** Labels teaching the PRM plus step position (2 weights) already
   give most of what LR17 gives over a strong PRM on Socratic. That is a property of how Socratic was built, not a
   better fusion. The claim "LR17 beats strong PRMs on Socratic" must not go to the advisors as a telemetry result.
2. **The telemetry does add something on Socratic, beyond position, for the Qwen PRMs**, when position is given
   explicitly (LR18 over POS2: +0.85 and +1.70, all repetitions positive; post-hoc). Not for PURE.
3. **PRMBench is unaffected.** Position gives nothing there (POS2 ≈ PRM), and the full LR's gain over AUTO
   (stage 2) and over POS2 is telemetry re-weighting. Stage 2's PRMBench result stands.

## Caveats
- Linear relative position only; a non-linear position shape could explain more of Socratic.
- Variation over repetitions is not a CI for one fixed test set. Development data, already inspected.
