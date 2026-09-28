# decision_rule_v1: how many steps to flag in each answer

Omri 2026-09-29: "try it, then explain". Step 454 showed that our answer-z q80 rule flags about 20% of every answer's steps
(about 99% of clean answers get a flag) and that most of the matched-rule gap to Qwen PRM is how many steps are flagged per
answer. This stage changes ONLY the flag decision; the step scores (S_equal: DS filter then plain average, stage B) and their
within-answer ranking are frozen.

- Protocol `PROTOCOL.json` frozen at d3e3cebac (before any rule was scored); runner `scripts/experiments/decision_rule_run.py`
  with pre-run review fixes (a952021d6); run `run_20260929` (4435169f6); post-hoc amendment A1 (f22ba09c8, `run_20260929_a1`).
- Gates: R0 reproduces the frozen PRMScore 0.6565188557935739 exactly; raw channels rebuild S_equal to 8.9e-16; DS closed-form
  posterior = EM model; official scorer = counts. Red team `RED_TEAM.md` (numbers confirmed; C1 confirmed; C2-C5 readings weakened).
- Development evidence on the already-evaluated population; not untouched confirmation.

## Rules and results (PRMBench official PRMScore, 6,211 answers; Holm-level intervals vs R0)

| Rule | What decides the number of flags per answer | PRMScore | vs R0 |
|---|---|---:|---|
| R0 (frozen) | answer-z + one q80 threshold: about 20% of every answer | 0.6565 | reference |
| R1 | a globally standardized score (raw channel levels, not normalized within the answer), q80 threshold; also places the flags | 0.6647 | +0.0082 [0.0018, 0.0147] |
| **R2 (primary)** | count from R1, placement by the frozen ranking | 0.6635 | **+0.0070 [0.0012, 0.0132]** |
| R3 | Dawid-Skene posterior > 0.5 on global marks | 0.6423 (A1: 0.6388) | loss |
| R4 | DS threshold maximizing the expected PRMScore | 0.6421 (A1: 0.6386) | loss |
| R5 | DS expected error count per answer, placement by the frozen ranking | 0.6511 (A1: 0.6437) | n.s. (A1: loss) |

- R2's gain is answer-level information: not a flag-rate effect (R0 at the same rate loses 0.0027; R2 beats it by 0.0097),
  not length (length-only counts +0.0004; R2 minus length-only +0.0067; shuffling counts among same-length answers costs
  0.036-0.045). It comes from removing flags from answers whose raw levels are low; concentrated in circular and
  missing_condition; the share of erroneous answers with an error step flagged falls 81.2% -> 77.0%; against a label-tuned
  global rate the gain includes zero.
- Clean answers: controls with >=1 flag 99.1% -> 57.4%, but modified-but-correct answers stay flagged (93%): the level signal
  separates original from modified texts, not clean from erroneous.
- The DS threshold route fails because the estimates are biased by dependent channels: prevalence 0.22 vs 0.14, sensitivity
  of the level channels about 0.8 vs 0.47, specificity about 0.97 vs 0.84 (A1, PRMBench-only fit), so posteriors are overconfident.
- ProcessBench official F1 (secondary): 0.062 -> 0.163 (R2), DS 0.25-0.29, but only by leaving correct answers unflagged;
  first-error accuracy falls; the PRMBench-calibrated threshold flags 49.5% of ProcessBench steps.
