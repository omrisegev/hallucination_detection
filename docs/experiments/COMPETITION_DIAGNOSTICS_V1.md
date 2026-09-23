# Competition diagnostics on the frozen locators — 2026-09-22 (Step 430)

Question: on long chains the first-error step outscores error-free steps yet loses the argmax
(Step 428). Three diagnostics on frozen predictions, labels used for evaluation only, nothing
fitted, nothing selected:

- **A1 generation drift.** Do the channels' statistics drift along the answer on a label-free
  reference population (answers the frozen CT7 gate closes)? If they do, the competing peak
  should sit where the drift is highest, and the late bias should grow with depth. This is the
  hypothesis the next method stage (Step 432) rests on; it is stated and measured here as a
  hypothesis, with what would falsify it.
- **A2 error complementarity.** Step 428 measured separation per readout (KL), not independence
  of errors. Do the weak readouts and channels miss *different* answers than the top-k readouts
  and the fused locators? Hit vectors, miss-correlation matrices, conditional hit rates given a
  reference miss, union ceilings (Step 359 rule: ceilings, never results), and the same on the
  within-answer token shuffle as the null.
- **A3 non-max decision rules.** With the fused mass frozen and the gate frozen, do
  parameter-free rules that prefer the earlier of two competing peaks change the outcome?

Population: the cumulative-vote-v2 / readout-family population, 13,769 answers (6,800
ProcessBench, 4,442 erroneous; 6,969 PRMBench), v3 labels, source folds v2, frozen CT7 gate.
Inputs: `results/readout_family_v1/profiles_full.npy` and `shuffled_full.npy` (145,597 x 11 x 17),
`OOF_STEP_SCORES.npz` (top5 soft equal, top5 pmf equal), the CT7 and token L-SML references, the
frozen token matrices. No new inference, no refit.

## A1 — declared before running

- Series: every channel x readout (187), the four fused locators (CT7, token L-SML, top5 soft
  equal, top5 pmf equal; answer-standardised), the eleven raw token channels at token level.
- Populations (ProcessBench): `gate_closed` = the frozen gate says no error (label-free
  reference); `label_clean` = no-error label (diagnostic); `pre_error` = steps before the
  first error of erroneous answers.
- Relative position in ten bins, absolute step index 0..19 and 20+; strata all / 2-5 / 6-10 /
  11+ steps. Slope of the bin means on the bin centres with a 1,000-draw source-group bootstrap.
- Rank drift: probability that the argmax of a series falls in the last fifth of the answer on
  gate-closed answers, beside the uniform expectation.
- Miss geometry (fused locators, erroneous answers): relative position of prediction and target
  for late and early misses, and the gate-closed drift curve evaluated at the predicted position
  minus at the target position (paired, bootstrap CI). Positive = the competing peak sits where
  the null is higher.
- Falsifiers: no drift on gate-closed answers; misses not concentrated where the drift is; late
  fraction not growing with the drift across strata.

## A2 — declared before running

- Single locators: earliest-tie argmax of every channel x readout column, real and shuffled;
  fused: CT7, token L-SML, top5 soft equal, top5 pmf equal.
- Hit = exact first-error step (tolerance-one as secondary). Tied-argmax rate per locator; a
  locator tied on more than 90 % of erroneous answers is `degenerate` and excluded from unions.
- Readout families, fixed: peak = max, q90, top10, top30, log_top5, boxcar8_max; spread = std,
  iqr, frac_above_z; trend = slope, jump; sequential = cusum_top5, page_wmax, onset80;
  first_token; mean.
- Conditional hit rates P(locator hits | reference misses) for reference = same-channel top5,
  fused top5 soft equal, CT7; by stratum and by miss type (any / late / early); the shuffled
  locator against the same real reference misses as the null; paired bootstrap of real minus
  shuffled (2,000 draws) on all answers and on the 11+ stratum.
- Phi correlation of exact hits: readouts within a channel; channels at top5 and at top30.
- Union ceilings: top5 U r, top5 U family, all 17 (non-degenerate) per channel; all channels at
  top5; everything; fused top5 U each family across channels; CT7 U everything. Real and shuffled.
- What would carry Stage B (fusion before the readout, readouts as voters): a family whose
  conditional hit rate given a top5 miss exceeds its shuffled null with an interval excluding
  zero, concentrated on late misses of the 11+ stratum.

## A3 — declared before running

- Masses: top5 soft equal, top5 pmf equal, CT7, token L-SML (OOF / frozen).
- Rules: `argmax` (earliest tie, the frozen rule), `earliest_of_top2`, `latest_of_top2`,
  `random_of_top2` (seed 20260922), `earliest_within_delta` on the answer-standardised score
  with delta = 0.5 SD primary and 0.25 / 1.0 as sensitivity, **never selected**.
- Gate frozen; endpoints gate-free SLA macro8, macro-F1 under the frozen gate, tolerance-one,
  early / late, MAE, per-stratum SLA; paired source-group bootstrap (10,000 draws) of SLA and F1
  against the argmax of the same mass.

Out of scope: any fit, any threshold chosen on labels, any promotion. Outputs in
`results/competition_diagnostics_v1/` with `MANIFEST.json` (input shas).
