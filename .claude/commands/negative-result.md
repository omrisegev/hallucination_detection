---
description: Write a negative result as a first-class deliverable — results/<dir>/NEGATIVE_RESULT.md in the fixed format (hypothesis, arms, win/loss, effect with CI, three reasons, direction-vs-implementation verdict), then link it from the HISTORY step. Use whenever an experiment did not beat its reference.
---

A negative result is most of this thesis's argument (retrospective: ~130 negative vs
~50 positive steps). It is only useful if it can be found again without re-deriving it.

Argument: `$ARGUMENTS` = the results directory (e.g. `results/fusion_token_gap_v1`).
Read its REPORT.html / summary CSVs and the matching `docs/experiments/*.md` first.

Write `results/<dir>/NEGATIVE_RESULT.md` with exactly these sections:

```
# Negative result — <experiment name>

**Hypothesis (one sentence):** ...
**Date / step:** YYYY-MM-DD, Step N
**Benchmark / population:** name, label release (e.g. v3), N rows, folds

## Arms compared
| arm | what it is | PRMB within-AUC | PB macro-F1 | vs reference |
| reference (incumbent) | ... | ... | ... | — |
| simple control (equal / permuted / entropy-only) | ... | ... | ... | ... |
| candidate | ... | ... | ... | ... |

**Win/loss record:** candidate vs reference: W-L over cells; paired CI on the primary endpoint: [lo, hi] (includes 0: yes/no)

## Three plausible reasons it failed
1. ...
2. ...
3. ...

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [ ] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open
State which box and why in one sentence.

## What would reopen it
One concrete condition (a new view, a corrected label, a different population).

**Source files:** REPORT.html, the CSVs, the command that produced them, `n_checked / n_total`.
```

Rules:
- Every number must come from a file in that directory; name the file next to it.
- Use the required non-conclusive wording for interim stages; never "does not matter".
- Then add one line to the HISTORY.md step: `Negative result recorded: results/<dir>/NEGATIVE_RESULT.md`.
- Do not commit; Omri commits, or run `/update-docs`.
