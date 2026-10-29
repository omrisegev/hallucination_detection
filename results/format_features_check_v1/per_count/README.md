# Check F, per count (Omri, 2026-10-29)

Each of the 10 format counts judged alone. Question: is it above chance, and are its errors conditionally independent of
our telemetry groups and the PRM? `python per_count.py` (CPU, ~15 min). Labels only split classes and score.

Cleaning: position removed with the Step 481 procedure (`step481_clean`; parity with probe.py's own `residual_cell`
asserted, max diff 7e-15), then log char length removed from the 8 non-length counts. Entries constant inside an answer
stay 0 and are left out of the fits. **The first run did not do this** and is kept in `first_run_position_leak/`:
a count that is 0 in a whole answer became a pure position score after the cubic was subtracted, so rare counts ('<>',
'()', ':') showed AUC .38-.65 that was position only. Orientation: sign of the correlation with the cleaned
16-channel average (label-free).

Pass rule (predeclared in per_count.py, per cell): AUC 95% source-group bootstrap CI above 0.5 AND max conditional |r|
(error steps and correct steps separately; vs every TELE16 group score and the PRM) below the reference = the max
conditional |r| among the existing groups and the PRM themselves, AND the same for the conditional covariance of GAP votes.

## Per count (mean over 4 PRMs; passes = cells out of 4)

| count | bench | AUC raw -> after position -> final [min CI low] | max cond r (ref) | vote cov (ref) | vs PRM | passes |
|---|---|---|---|---|---|---|
| char_len | PRMBench | .633 -> .656 -> .656 [.648] | .640 (.636) | .382 (.429) | .155 | 2 |
| char_len | ProcessBench | .601 -> .688 -> .688 [.671] | .680 (.501) | .445 (.249) | .283 | 0 |
| n_words | PRMBench | .592 -> .623 -> .623 [.613] | .553 (.636) | .313 (.429) | .134 | 3 |
| n_words | ProcessBench | .609 -> .686 -> .686 [.670] | .663 (.501) | .415 (.249) | .271 | 0 |
| n_digits | PRMBench | .601 -> .566 -> .533 [.524] | .513 (.636) | .236 (.429) | .062 | 3 |
| n_digits | ProcessBench | .655 -> .597 -> .537 [.519] | .441 (.501) | .271 (.249) | .074 | 1 |
| n_digits | CRV-Arith | .590 -> .660 -> .656 [.618] | .590 (.618) | .412 (.394) | .253 | 1 |
| n_colons | PRMBench | .558 -> .515 -> .509 [.502] | .165 (.636) | .095 (.429) | .031 | 4 |
| n_parens | ProcessBench | .606 -> .660 -> .557 [.537] | .228 (.501) | .109 (.249) | .069 | 4 |
| n_letters | CRV-Arith | .779 -> .498 -> .558 [.515] (sign -) | .425 (.618) | .224 (.394) | .102 | 4 |
| n_ops | CRV-Arith | .508 -> .606 -> .617 [.565] | .523 (.618) | .326 (.394) | .204 | 3 |
| n_angle, n_newlines, n_eq | all | about .47-.52 | | | | 0 |

Full table: P1_PER_COUNT.csv (per cell, also own-cluster flags and Lemma-4 positivity), P2_COUNT_PAIRS.csv,
P3_PRMBENCH_TYPES.csv. CRV-GSM8K: no count passes in any cell.

## Reading

- No count passes on all four benchmarks. The passing counts change from benchmark to benchmark.
- The two clearly informative counts, char_len and n_words (AUC .62-.69 on PRMBench/ProcessBench), are about as
  dependent on one telemetry group as our groups are on each other (cond r .55-.68); on ProcessBench above the reference.
  Against the PRM alone they are much less dependent (.13-.28).
- The counts with low dependence (n_colons on PRMBench, n_parens on ProcessBench: cond r .17-.23, vote cov ~.1) have
  weak signal (AUC .51 and .56). n_colons on PRMBench comes from one perturbation type: redundancy .65, others .47-.54.
- The reference level is high (.50-.64) because the existing telemetry groups are themselves strongly dependent; passing
  it does not mean independent.
- Adding the cell's passing counts to TELE16 L-SML (P4_FUSION_ROWS.csv) does not help: PRMBench -0.000 to -0.018
  within-AUC, ProcessBench -0.003 to +0.008, CRV-Arith -0.013 to +0.036 (the +0.036 is in a cell where TELE16 is .28,
  i.e. inverted). The E1b row with counts was not run: no stable set of 2-3 counts passes, and E1b fails for structural
  reasons (see results/ds_gate_e1b_fuse).
