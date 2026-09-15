# Step 396 (Claude): digit disagreement as answer-gate evidence, frozen before results

Bounded follow-up to the first "future question" in `docs/reviews/fusion_insertion_map_2026-09-16.md`:
does digit disagreement add clean/error discrimination conditional on tail mass, and does that improve
end-to-end PB with the locator frozen? Development data (13,769 answers, fully exposed). No labels enter
any detector; labels are used only in evaluation.

## Fixed pieces

- **Locators frozen.** Primary locator: innovation5 (frozen step scores). Secondary locator: innovation5 + digit .25
  (Step 393/`claude_real_checks_v1`). Peaks are never recomputed.
- **Operating rule unchanged.** For every detector, gate opens when the answer's within-cell midrank of the detector
  score is >= .33 (the current transductive rule). Higher detector = more likely erroneous. The rule fixes the
  opened fraction per cell, so detectors are compared at a matched operating point.
- **Answer-level detectors (label-free):**
  1. `tail15` = current gate detector (frozen `gate_raw`: answer Top10 mean of tail-15 mass).
  2. `digit_count` = number of tokens where the provided token is a digit and the scorer's top-1 is a different digit.
  3. `digit_rate` = digit_count / max(number of provided digit tokens, 1).
  4. `digit_presence` = number of provided digit tokens (confound control: digits alone, no disagreement).
  5. `equal_rank(tail15, digit_rate)` = mean of the two within-cell midranks (two genuinely different sources;
     count/rate are nested and are not split into separate experts; no IU with fewer than three views).
  6. `equal_rank(tail15, digit_count)`.
- **Endpoints.** PB macro F1 (all-8, q4, q8), clean accuracy, error exact accuracy, correct peaks suppressed by the gate,
  gate-open count. within-AUC is unaffected by the gate and is not reported as a contrast.
- **Uncertainty.** 10,000 paired source-group bootstrap draws. Two primary pairs on the primary locator:
  `equal_rank(tail15,digit_rate)` vs `tail15`, and `digit_rate` vs `tail15`; 97.5% intervals. The same pairs on the
  secondary locator and all other detectors are descriptive 95%.
- **Not done here.** No q sweep, no learned gate, no change to the locator, no PRMScore recalibration.

## Acceptance

Replay of the frozen tail15 gate must reproduce innovation5 39.8314% and the digit025 41.3300% exactly.
Digit counts must replay the Step 393/`claude_real_checks_v1` predicate (ids 15..24, scorer top-1 digit differs).
Report gains/losses of final PB hits and of clean successes per detector versus the current gate.
