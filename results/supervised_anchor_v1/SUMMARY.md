# Stage 1: supervised mixing ratio over label-free telemetry (2026-11-01)

**Question** (Bracha, emails 2026-10-23 and 2026-10-29). Keep the telemetry direction label-free (TELE_N16, L-SML over the
16 backbone channels) and let labels learn only the PRM/telemetry ratio (ANCH, 2 coefficients). Does this beat
label-free AUTO, the PRM, and a full 17-input supervised LR (LR17) trained on the same labels?
Protocol frozen first: `PROTOCOL.json` (amendment A1: the SUB5 arm was dropped before any new-arm score, because its
score is not stored outside PRMBench).

**Run.** `anchor.py`, CPU, 98 s. 24 parity checks pass: AUTO reproduces stage 8 exactly (GAP main metric and
within-AUC) in all 12 cells. 4 PRMs × {in-domain cross-fit on PRMBench and on Socratic (GroupKFold 5), transfer
PRMBench↔Socratic and to Hard2Verify}. GAP rule everywhere. Paired bootstrap, 2,000 draws.

## Main metric under GAP (PRMBench/Socratic PRMScore, Hard2Verify balanced F1)

In-domain (cross-fit; the labels come from the same benchmark, from other source groups):

| | PRM | AUTO | TELE16 | ANCH | LR17 |
|---|---|---|---|---|---|
| PRMBench Qwen-7B | 66.37 | 69.87 | 69.26 | 70.26 | **71.46** |
| PRMBench Qwen-72B | 64.47 | 70.18 | 69.68 | 71.69 | **72.38** |
| PRMBench PURE | 64.67 | 69.06 | 68.71 | 69.46 | **70.29** |
| PRMBench RLHFlow | 59.39 | 65.92 | 66.17 | 66.15 | **68.39** |
| Socratic Qwen-7B | 69.49 | 65.48 | 64.47 | 69.31 | **72.55** |
| Socratic Qwen-72B | 68.16 | 60.41 | 58.65 | 67.06 | **72.22** |
| Socratic PURE | 69.99 | 63.45 | 63.92 | 70.08 | **70.21** |
| Socratic RLHFlow | 50.38 | 57.66 | 57.53 | 54.26 | **63.28** |

LR17 − AUTO, CI above 0 in 8/8 cells (PRMBench +1.2 to +2.5; Socratic +5.6 to +11.8).
LR17 − PRM on Socratic: +3.1, +4.1, +0.2 (tie), +12.9.

Transfer (trained on all of one benchmark, applied unchanged): no arm wins consistently.
- PRMBench → Socratic: LR17 ties the PRM for Qwen-7B/72B and loses to it for PURE (−5.3); ANCH is below the PRM for the
  3 strong PRMs (−1.0 to −2.8).
- Socratic → PRMBench: label-free AUTO is best for all 4 PRMs (ANCH −0.5 to −4.2, LR17 −1.1 to −3.1).
- Hard2Verify (177 answers): wide CIs, mixed.

## Predeclared readings
- **H1** (PRMBench in-domain, ANCH > AUTO in ≥3/4): **not met**, 2/4 (Qwen-7B +0.39, Qwen-72B +1.51; PURE and RLHFlow ties).
- **H2** (Socratic in-domain, ANCH > PRM in ≥3/4): **not met**, 1/4 (RLHFlow only). ANCH does raise the PRM's share
  (0.43 on PRMBench, 0.81–0.84 on Socratic for the Qwen PRMs), but this only matches the PRM, it does not beat it.
- **H3** (PRMBench → Socratic, ANCH > LR17 in ≥3/4): **not met**, 1/4 (PURE +4.2).

## Reading
1. **Keeping the telemetry direction label-free and learning only the ratio is not enough.** It fixes AUTO's
   under-weighting of the PRM (it reaches the PRM on Socratic), but it loses to the full LR in-domain in 7/8 cells.
   The full LR's extra gain comes from re-weighting the telemetry channels themselves.
2. **With in-domain labels, supervised weighting beats both AUTO and the PRM**, including the strong PRMs on Socratic,
   where no label-free arm did. This is the first arm in the project that beats a strong PRM on Socratic.
3. **Labels from a different benchmark do not carry over.** This agrees with `bracha_lr_transfer_v1` (PRM800K): the
   weighting problem is a distribution-shift problem. The open question for stage 2 is how few in-domain labels are
   enough.

## Caveats
- **Socratic's groups are per answer** (2,995 groups for 2,995 answers), not per problem. Answers to the same problem
  can sit in different folds, so the Socratic in-domain numbers may be optimistic. PRMBench uses 707 source groups.
- The PRM column here is the answer-z of the PRM risk under GAP (as stage 8). It is below the PRM row in earlier
  tables, which used the probability scale (PRMBench Qwen-7B 66.37 here vs 68.01; Socratic 69.49 vs 70.10). LR17 is
  above both versions in the cells quoted.
- All three benchmarks are development data that we have already inspected.
- LR17's coefficient "PRM share" (0.4–25%) is not comparable with ANCH's, because LR17 spreads its weight over 17
  inputs.

Files: `PROTOCOL.json`, `anchor.py`, `RESULTS.json` (points, contrasts with CIs, shares and coefficients), `ROWS.csv`,
`PARITY.json`, `run.log`.
