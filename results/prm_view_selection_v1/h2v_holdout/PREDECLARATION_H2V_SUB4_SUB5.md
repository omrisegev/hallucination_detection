# Pre-declaration: SUB4 and SUB5 on Hard2Verify (2026-10-26, before the first run)

**Request (Omri, 2026-10-26):** run SUB4 and SUB5 on Hard2Verify to see how they do there.

**Status of the data.** Hard2Verify (177 answers, 5 PRMs) was not used to select SUB4 or SUB5: they were chosen on
PRMBench and Socratic labels (stages 1–2). Hard2Verify results of other arms (CHAIN_N16, TELE_N16, AUTO_N16) were
already inspected, so this is a held-out test of the *selection*, not an untouched test of the method. Development
evidence; small n.

**Frozen sets** (no change, no re-selection):
- SUB4 = q15_VE1 + chosen_surprisal + energy_innovation + top15_turnover + realized_z + digit_alternative +
  digit_alternative_innovation.
- SUB5 = q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread.

**Methods** (code unchanged, refitted from scratch per cell, labels only score):
- "+PRM": stage-1 fusion (`stage1/stage1.py` `CellData.evaluate`): PRM-anchored orientation, GAP votes, L-SML groups
  (K ≥ 3), SML within, DS-EM between with the PRM as anchor, weights on continuous answer-z values.
- "alone": stage-2 fusion (`stage2/stage2.py` `TeleCell.evaluate`): stored orientation, no PRM, majority label switch.
- If K < 3 or all between-group weights clip, the cell is reported as invalid (no fallback).

**Cells.** Hard2Verify × qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm (the 4-PRM line); math_shepherd reported
separately (dropped from the line on 2026-10-26).

**Endpoints.**
- Primary: the official Hard2Verify metric under GAP (exact evaluator path `E.official`), as in STATUS_MAIN.
- Secondary: within-answer AUC (`E.per_answer_auc`, keep mask).
- Comparators: PRM same rule (STATUS_MAIN "PRM same rule (better scale under GAP)"), the same-set plain mean (same
  code path as the fusion), CHAIN_N16, TELE_N16 (STATUS_MAIN), AUTO_N16 (math-rigor stage 8 JSON).
- Uncertainty: paired bootstrap over answers, 2,000 draws, seed 20261026, for fused − PRM and fused − own mean.

**What counts.** Descriptive only; no pass/fail gate, nothing tuned on this output. Report per cell: the two goals of
stages 1–2 (with PRM: beats max(PRM, own mean); alone: beats own mean), plus the point differences against CHAIN_N16
and TELE_N16 and the bootstrap intervals.

**Parity before reading results.** The PRM-alone exact metric from `CellData` must equal STATUS_MAIN's
"PRM same rule" GAP value in each cell (tolerance 1e-6); otherwise stop and report.
