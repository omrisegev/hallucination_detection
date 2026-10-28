# Bracha's question: supervised LR instead of label-free L-SML (2026-10-28)

**Question** (Bracha Laufer, 2026-10-23). Would a logistic regression trained on labelled data of the kind PRMs are
fine-tuned on weight the PRM and the telemetry better than label-free L-SML, and give a strong PRM the weight it deserves?

**What was run.** The protocol was frozen in `PROTOCOL.json` before any result.
- **Training data:** one LR per PRM, trained on PRM800K's human step labels (CROP's 8,000 answers).
- **Inputs:** exactly AUTO's 17: the answer-z of the PRM risk plus the 16 backbone telemetry channels.
- **Model:** L2 penalty, balanced class weights, C chosen by 5-fold CV grouped by problem.
- **Leakage control:** problems that also appear in a test set were removed from training. That covers 823 of the
  5,049 PRM800K problems (2,559 answers): 436 shared with PRMBench, 447 with ProcessBench (its MATH subset) and 14
  with DeltaBench. Socratic and Hard2Verify could not be checked, because their problem texts are not available
  offline. A model trained on all of PRM800K moves the main metric by at most 1.9 points, in both directions.
- **Testing:** the LR was applied unchanged to 7 benchmarks under the same GAP rule (ProcessBench: the NC rule),
  plus a label-trained threshold of 0.5. CPU only, stored scores.
- **Controls:** an LR on the PRM alone, and an LR on standalone telemetry alone.
- **Parity:** 92 checks, 0 failed. AUTO and the PRM reproduce stage 8, AUTO_NEW.csv and SUB5_ALL.csv exactly. The
  bootstrap's point estimate equals the reported metric in every cell.

## Answer

1. **The PRM does not get more weight from a supervised model.** PRM share of the LR's coefficients on standardized
   inputs, against AUTO's least-squares share computed with the same function on PRMBench:

   | | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
   |---|---|---|---|---|
   | LR (PRM800K) | 4.6% | 3.1% | 29.8% | 0.4% |
   | AUTO on PRMBench | 5.8% | 0.5% | 4.7% | 7.0% |
   | CHAIN on PRMBench | 23.7% | 15.0% | 20.6% | 23.7% |

   Even with labels, the Qwen PRMs get 3–5%; only PURE gets more. The coefficients are shares in a correlated input
   set, so read them as indicative.

2. **Neither weighting wins everywhere.** LR minus AUTO on the main metric, paired bootstrap 95% CI, 24 cells
   (6 benchmarks × 4 PRMs; ProcessBench is a point estimate only). LR is significantly better in 11 cells and AUTO in
   8; 5 are ties.
   - **PRMBench:** AUTO is better for all 4 PRMs, by 2.2 to 9.6 points. The LR trained on PRM800K is even below the
     PRM alone under GAP on all 4 (67.1 / 68.0 / 64.9 / 56.4 vs 68.0 / 69.5 / 66.2 / 59.4).
   - **Socratic, CRV-GSM8K, DeltaBench, where the telemetry is weak:** the LR is better for the strong PRMs (Socratic
     +3.0 to +7.0; CRV-GSM8K +1.6 to +5.1; DeltaBench PURE +9.0, Qwen-72B +2.0) and worse for RLHFlow.
   - **Hard2Verify (177 answers):** mixed: +9.6 PURE, +6.8 RLHFlow, −5.9 Qwen-72B, Qwen-7B a tie.
3. **Out of domain, no fusion reliably beats a strong PRM alone.** On Socratic the LR beats the PRM under GAP in 2
   of 4 PRMs, AUTO in 1 of 4. On ProcessBench the PRM's native rule stays ahead for Qwen-7B, Qwen-72B and PURE
   (LR 60.4 / 73.1 / 51.3, AUTO 63.2 / 67.3 / 37.7, PRM 72.9 / 78.2 / 52.6). On CRV the PRM's native threshold
   (63–84) beats every GAP arm, because GAP always flags at least one step.
4. **Telemetry only:** the PRM800K-trained LR on standalone telemetry is below label-free TELE16 on PRMBench (62.6–65.3
   vs 67.6–68.3) and Socratic. It ranks steps better than TELE16 within answers on CRV.
5. **CROP kept % on PRMBench:** LR 14.4–21.4, AUTO 15.1–21.6, CHAIN 16.2–23.3. The in-benchmark supervised LR ceiling
   is 25.1–25.4 (`prm_boost_collection_v1`, quoted, not recomputed).

**Reading.** Labels from another dataset do not solve the strong-PRM problem. A PRM800K-trained LR transfers
poorly to PRMBench, where label-free AUTO is better. It helps where the telemetry carries little signal and a
strong PRM should dominate, but even there it rarely beats the PRM alone. The weighting problem is a
distribution-shift problem for both methods, not only a consequence of being label-free.

## Caveats

- PRM800K is the closest public step-labelled set, but it is not the Qwen PRMs' own training data. PRISM was trained
  on it; Qwen's data is not public.
- The bootstrap resamples PRMBench by source group and other benchmarks by answer, the only grouping available
  offline.
- AUTO is fitted transductively and label-free on each benchmark. The LR is fitted once on PRM800K. On Hard2Verify,
  AUTO is fitted on each PRM's full readable set and scored on the 177 common answers, as in stage 8.
- RLHFlow's PRM score points the opposite way on PRM800K: the PRM-only LR gives it a negative weight, and that weight
  inverts it on the test sets (PRMBench within-AUC 0.338).

Files: `PROTOCOL.json`, `lr_transfer.py`, `RESULTS.csv` (1,680 rows), `WEIGHTS.csv`, `BOOTSTRAP.csv`, `OVERLAP.json`,
`PARITY.json`, `TABLES.md` (one table per benchmark), `tables.py`, `run.log`.
