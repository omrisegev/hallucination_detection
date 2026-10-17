# Red team: PRMScore decomposition (prmscore_decomposition_v1), 2026-09-27

Three independent agents, launched in parallel, read only raw per-answer artefacts and code (no output tables):
A recomputed from raw inputs by a fresh code path, B audited population coverage, C ran label nulls, a permuted
feature and a math check. Their scripts and outputs are in `red_team/agent_{A,B,C}/`; the pre-run code review's
formula tests are in `red_team/prerun_review/`.

**First line: no number is wrong.** Every point estimate recomputes to 4 decimals (A, C) on the full population
(B). Three of six claims are weakened in their READING, not in their numbers.

| CLAIM | VERDICT | EVIDENCE (agent, number, n_checked/n_total, file) |
|---|---|---|
| C1 filter-then-average (S_equal) 0.6565 vs realized_drv alone 0.6590, difference -0.0024 [-0.0066, +0.0018]: the fused row does not beat the single channel on PRMScore | **weakened** | A: 0.6565 / 0.6590, -0.0024, own bootstrap [-0.0067, +0.0015] over six seeds within Monte Carlo spread, 6,211/6,211 answers (agent_A/run_output.txt). B: point -0.0024385, 707/707 groups, 4 of 5 folds negative (agent_B/audit_log.txt). C: against a whole-answer label swap between answers of equal step count, realized_drv carries MORE answer-specific signal: residual -0.0056 [-0.0093, -0.0020] (agent_C/rt_c_part2.json). The apparent tie is the fused row's positional structure compensating; do not describe the two as equivalent localizers. realized_drv remains post hoc (best of 22). |
| C2 gap to Qwen2.5-Math-PRM-7B: 0.0241 to raw-scale q80 (0.6807), 0.0168 under exactly our answer-z q80 rule (0.6733); we lead on redundency/circular/domain_inconsistency, trail on confidence/counterfactual/deception/missing_condition | **weakened** | A: all six PRMScores and seven class differences exact (agent_A). B: 8/8 classes, 750-758 answers and 701-707 groups each; redundency and circular are scored through the validity fallback (no redundancy head on either side), so the two largest leads carry that caveat (agent_B). C: of the matched-rule gap -0.0168, a within-answer label permutation already yields -0.0140; residual -0.0028 [-0.0080, +0.0027] includes 0. The PRM's lead under our rule comes from how many steps it flags per answer (flag share vs error share r = 0.39 vs 0.19), not from better placement inside an answer; against the swap null the residual is +0.0058 [+0.0004, +0.0111] (agent_C/rt_c_part2.json). |
| C3 answer-z rows flag at least one step in 98-100% of the 758 clean controls (~20% of steps); PRM raw q80 55.5% (9.4%), native 19.0% (1.9%) | **weakened: numbers confirmed, reading as a comparison between rows refuted** | A: S_equal 99.08% / 20.37%, PRM_z 99.08% / 18.98%, PRM raw 55.54% / 9.37%, native 19.00% / 1.90% (agent_A). B: random scores under the same rule give 98.5-99.2% / 0.198 over five draws (agent_B/audit_log2.txt). C: i.i.d. noise gives 99.2-99.8%; a within-answer shuffle leaves it unchanged; mean flagged share 20.9% in controls vs 20.6% in erroneous answers; bound k <= n/(1+tau^2) (agent_C/rt_c_part2.json). It is a structural property of per-answer standardization plus a global threshold; only the raw-scale PRM rows differ. |
| C4 within-answer pairs are 0.02% of error/correct pairs, so pooled AUC = cross-answer AUC | **confirmed** (A did not test) | B: 148,384 of 923,349,078 pairs (agent_B). C: exact share 0.0161% (report "0.016%", not "0.02%"); pooled minus cross <= 3.1e-6 for all arms; identity exact; brute force on 400 answers exact; group-weighted draw algorithm exact against replicated data (agent_C/rt_c_part2.json). The identity uses the pair-weighted within-AUC (S_equal 0.7284), not the answer-mean 0.7802. |
| C5 filter contribution S_equal - B13_equal +0.0038 [0.0018, 0.0058], positive in 5 of 8 error classes, negative in missing_condition; secondary: concentrated in late first errors | **weakened** | A: 0.6565 - 0.6527 exact. B: direction mixed (also -0.0007 domain_inconsistency, -0.0005 deception); only circular and counterfactual have the same sign in 5/5 folds; the position gradient mirrors class mix (first-step bin 159 answers, 51% missing_condition; late bin 30% confidence), so it is not a position effect (agent_B). C: beats the within-answer null (z 3.1) but the whole-answer swap null reproduces +0.0027 of +0.0038 (z 1.5); answer-specific residual +0.0011 [-0.0008, +0.0031] includes 0 (agent_C). The filter gain is mostly error-position structure, consistent with the stage-B red team. |
| C6 argmax-hit complementarity on 6,035 erroneous answers (S_equal/realized_drv 2,990/657/779/1,609; S_equal/PRM 2,343/1,304/1,144/1,244) | **confirmed** (C did not test) | A: counts exact. B: 6,035/6,035; five all-error answers count as a hit for every row; PRM tied maxima (107 answers) move the counts by at most 9. |

Additional notes carried into the report:
- Bonferroni family is the 9 classifications per contrast and endpoint only; no correction across contrasts x endpoints (C).
- The 16 inert-annotation answers change no sign in 42/42 checked contrasts; in the confidence class the order of
  B_sml__merge and realized_drv flips when they are excluded, so no confidence-class ranking of fused rows against
  realized_drv is made (B).
- realized_drv's thresholds are rebuilt by the stated rule (0.8605, 0.8610, 0.8688, 0.8495, 0.8596), not saved by a
  stage runner (A). A within-answer permutation of realized_drv drops it to 0.5028 +- 0.0015 (chance plus flag-count
  allocation), so its PRMScore reflects signal (C).
- PRM_native uses the manufacturer 0.5 cut, uncalibrated; the fair threshold comparator for the PRM is raw q80 (A).
- PRM reward ties at the raw-q80 threshold (336 steps): "<" vs "<=" moves PRMScore 0.6807 -> 0.6810 (A).
