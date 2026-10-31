# Review A (independent reviewer on the Windows PC), 2026-10-31

Read-only review of design, code, run and results of raw_null_gate_v1 (commit 42f83d01).
The reviewer did not rerun the experiment. Numbers below come from FITS.json, RESULTS.csv and
the saved labelfree arrays on the PC. A second, independent review is in REVIEW_B_CLOUD_2026-10-31.md.

## Verdict

"All 20 tests fail" is correct under the pre-registered rule. The failure comes mainly from an
empirical null that is not identified on these scores, and from the fixed P(err) < .5 cut.
It does NOT show that raw step scores carry no clean-answer signal.

Suggested wording: "The lfdr gate is not identified on these scores, so its decisions collapse to
all-clean or all-error." Not: "There is no label-free signal."

## Findings (most severe first)

| # | Severity | Finding | Evidence |
|---|---|---|---|
| 1 | Critical | The null is usually not identified, but the fit is accepted. Code and prereg reject only b2 >= 0. | G1: raw pi0 >= 1 in 31/68 cells, sigma0 >= 1 in 17. G2 groups: 140/244, max raw pi0 1765. |
| 1b | Critical | Decisions collapse. | 22/68 G1 cells and 17 G2 cells call all answers clean or all erroneous. ProcessBench G1 calls a median 0.1% of answers erroneous (truth 65%); kept_error at most .17 (need .85). Main reason for 0/4 on ProcessBench. |
| 2 | Severe | Signal exists; the operating point does not. | G1 statistic answer-level AUC .73-.78 on 4 ProcessBench cells checked. A label-chosen threshold would close 45-54% of clean answers at kept_error .85 (passes) on those 4, and on 3/3 CRV-Arith cells checked; 26% on the one CRV-GSM8K cell checked (fails). Label-chosen threshold = diagnostic only, not a method. |
| 3 | Severe | G2 P(err) is not monotone in the score. | With a too-wide null, extreme steps get LR = 1e-3 (evidence of correctness). LR also uses the shared median pi0, not the group's own. Inverted AUC .38-.44 in 2/3 ProcessBench cells checked. |
| 4 | Severe | G3 votes are often constant. | In 29/55 cells fewer than 3 groups have votes that vary, yet the cells pass the ">= 3 votes" check. When all groups vote "clean", DS returns prevalence .95 (label switch). SUB5 L-SML often gives only 2 groups, so G3 is invalid in 13 cells. |
| 5 | Severe | Worst-step choice is poor. | Worst step = argmax of answer-z fused score; the max answer-z itself has AUC .30-.49. On CRV-Arith the raw answer max gives AUC .74-.76 vs .65 for the chosen step. |
| 6 | Minor | ece_step tie hypothesis confirmed. | The two ECE functions are identical on the same input. P(err) differences of 4.7e-13 move 17 steps across a bin edge; the reviewer reproduced both runner and verifier values exactly. No effect on the verdict. |
| 7 | Minor | No label leakage found. | PRMBench backbone "0/0" = no such cells (as pre-registered). GAP_ONLY ProcessBench F1 = 0 by construction. |

## Not checked by this reviewer

Literature note and unit tests; the votes_e1b loader; PRMBench cells in depth; the label-chosen
threshold beyond 7 cells.

## Suggestions for a follow-up (not started; needs Omri's OK)

- Identifiability checks before using a null: raw pi0 <= 1 and sigma0 < sd of the score.
- Require votes that actually vary (per group, not only a count of groups).
- Require P(err) to increase with the score (monotone LR).
