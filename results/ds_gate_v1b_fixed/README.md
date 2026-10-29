# ds_gate_v1b_fixed: E1 rerun with the fixes. FAILED on all three arms (stopped as frozen)

The rerun fixed four things relative to ds_gate_v1, all recorded in PROTOCOL.json (B1-B8) before the real run:
- stored channel orientation;
- the PRM also cleaned of position, so position enters only through the prior;
- a 10-bin position prior;
- three arms: AUTO (PRM + N16 backbone), TELE16 standalone, and SUB5 standalone.

Run with `python run_e1.py`; the log is in e1_run.log.

**Model checks** (`test_synthetic.py`):
- brute-force enumeration of P(has-error), q and the log-likelihood: maximum error 7e-15;
- synthetic recovery: three simulations pass;
- all valid real fits: 0 likelihood drops, 0 label switches, 0 degenerate fits.

**Not done:** the independent reviewer subagent requested by the coordinator (this worker cannot spawn agents; see B8).

## Verdict

| arm | D1 prevalence | D3 ECE | D4 gate | D5 count | invalid (K<3) |
|---|---|---|---|---|---|
| AUTO | fail | fail | fail | fail | 1 (PRMBench Qwen-7B) |
| TELE16 | fail | fail | fail | fail | 0 |
| SUB5 | fail | fail | fail | fail | 5 |

## Results by benchmark (range over the 4 PRMs)

| arm | benchmark | rho_hat | truth | ECE | clean closed at rho_hat | erroneous kept | metric: GAP -> gated (rho / cost / 0.5) |
|---|---|---|---|---|---|---|---|
| AUTO | PRMBench | .85-.94 | .866 | .07-.14 | .13-.22 | .80-.91 | PRMScore 65.9-70.2 -> within -0.5..+0.7 (cost: -3.1 on Qwen-72B) |
| AUTO | ProcessBench | .96-.99 | .653 | .32-.34 | .06-.10 | .86-.93 | F1 0 -> 9.6-17.1 / 2.6-25.8 / 1.2-3.0 |
| AUTO | CRV-Arith | .75-.97 | .083 | .66-.89 | .17-.32 | .40-.92 | macro F1 40.2-50.2 -> -2.4..+2.3 |
| AUTO | CRV-GSM8K | .98-1.0 | .21 | .77-.79 | .04-.06 | .86-.92 | macro F1 47.5-50.0 -> within +0.7 |
| TELE16 | PRMBench | .92-.94 | .866 | .07-.09 | .11-.23 | .88-.90 | PRMScore 67.6-68.3 -> +0.0..+0.7 (all rules) |
| TELE16 | ProcessBench | .98-.99 | .653 | .33-.34 | .06 | .89-.94 | F1 0 -> 9.7-10.7 / 0.3-4.0 / 0.2-2.0 |
| TELE16 | CRV-Arith | .71-.97 | .083 | .65-.89 | .17-.30 | .66-.93 | macro F1 42.3-47.6 -> +1.2..+1.8 |
| TELE16 | CRV-GSM8K | ~1.0 | .21 | .79 | .02-.04 | .89-.94 | unchanged (+-0.1) |
| SUB5 | PRMBench | .48-.90 | .866 | .11-.52 | .18-.33 | .48-.86 | PRMScore 65.4-66.9 -> -8.6 (RLHFlow) .. +0.4 |
| SUB5 | ProcessBench (3 valid) | .92-.94 | .653 | .29-.30 | .20-.24 | .74-.76 | F1 0 -> 22.5-24.7 / 7.6-11.4 / 3.5-4.6 |
| SUB5 | CRV (4 valid of 8) | .68-1.0 | .08/.21 | .62-.90 | .02-.34 | .59-.93 | macro F1 +0.1..+2.5 |

ProcessBench: plain GAP always flags at least one step, so its first-error F1 is 0. That makes the gated F1
comparable only against 0. For reference, the NC rule in results/prm_sub5_all_benchmarks_v1 reaches far higher F1.

## Reading

The fixes do not change the picture from E1. The estimated share of answers with an error stays at 0.85-1.0
everywhere, against a truth of 0.08-0.87.

PRMBench is the one place where the prevalence estimate is near the truth: TELE16 gives .92-.94 vs .866 and
ECE .07-.09. That is because almost all PRMBench answers contain an error, so "almost all" is right there by
accident; it does not show that the model can tell clean answers from erroneous ones.

The gate is not usable as a k=0 decision on any benchmark: it closes at most a third of clean answers and in
places also closes erroneous answers. The small PRMScore gains (up to +0.7) are within the range a rule that
moves a few answers would give; no bootstrap was run.

SUB5 gives K < 3 in 5 of 16 cells. Where it is valid, it closes more clean ProcessBench answers (20-24%), but it
also loses about a quarter of the erroneous ones and fails D1 and D3.

Cause, unchanged from E1: GAP votes are relative within each answer, so every answer, clean or not, produces the
same kind of vote pattern.
